// The runtime's shader lookup (src/shader_library.h) over a translated corpus from the player's own copy.
//
//   LEONIDA_SHADERS=<work>/shaders/spirv      the translator's output (index.json and the .spv modules)
//   LEONIDA_SHADER_DXBC=<work>/shaders/dxbc   optional; the carved DXBC containers, default <spirv>/../dxbc
//
// For every container the test does what d3d12_vulkan_device.cpp does with a PSO's bytecode: inflate the FSSM program
// from the DXIL/XDAT/XBCH entry, hash it and look it up under the container's stage. The stage comes from the DXIL
// part's program header, not from index.json. Every container has to resolve to its own entry (the one named after
// the file, or one of its `~N` duplicates) with a readable SPIR-V module, and every RPF index entry has to be reached
// by a container. The entries from <shaders>/fspx/index.json and <shaders>/bvh/index.json are counted apart: their
// programs come from the FidelityFX DLL and from XboxBvhBuilder.dll, not from the carved DXBC containers, so they are
// not orphans. Containers with more than one 0x0D payload are
// counted, and their first payload must not stand in for the last: that is the rule fssm_program follows, so a
// regression to the first payload fails here.
//
// A container with no index entry under its name at all is reported apart, as "not translated": the translator left
// it out (for example a module over SPIR-V's id bound), so the lookup has nothing to find. That still fails the test,
// but it points at the shaders stage instead of looking like a lookup regression. Only a container whose own entry
// exists and is not reached by its program hash is a lookup failure.
//
// Without LEONIDA_SHADERS it prints SKIP and exits 77 (SKIP_RETURN_CODE). It reads the work folder in place, writes
// nothing, and prints counts; on a failure it names at most a few shaders, which stay on your machine.
#include "../src/shader_library.h"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

std::string env(const char* name) {
    const char* value = std::getenv(name);
    return value ? value : "";
}

std::vector<std::uint8_t> read_file(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}

std::uint32_t u32(std::span<const std::uint8_t> data, std::size_t at) {
    if (at + 4 > data.size()) return 0;
    std::uint32_t v;
    std::memcpy(&v, data.data() + at, 4);
    return v;
}

// The stage the DXIL program header names (DxilProgramHeader::ProgramVersion, kind in bits 31:16; D3D12's
// D3D12_SHADER_VERSION_TYPE), in the translator's spelling. Empty for a kind the corpus has no translation for.
std::string dxil_stage(std::span<const std::uint8_t> container) {
    const auto dxil = leonida::dxbc_part(container, "DXIL");
    if (dxil.size() < 4) return {};
    switch (u32(dxil, 0) >> 16) {
    case 0: return "ps";
    case 1: return "vs";
    case 2: return "gs";
    case 5: return "cs";
    case 13: return "ms";
    default: return {};
    }
}

// Every 0x0D (XBCH) payload of the container's XDAT table, in table order, inflated. fssm_program keeps the last.
std::vector<std::vector<std::uint8_t>> all_programs(std::span<const std::uint8_t> container) {
    std::vector<std::vector<std::uint8_t>> out;
    const auto dxil = leonida::dxbc_part(container, "DXIL");
    std::size_t xdat = 0;
    while (xdat + 4 <= dxil.size() && std::memcmp(dxil.data() + xdat, "XDAT", 4) != 0) ++xdat;
    const std::uint32_t count = u32(dxil, xdat + 4);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::size_t entry = xdat + 12 + 12 * std::size_t(i);
        if (entry + 12 > dxil.size()) break;
        const std::uint32_t tag = u32(dxil, entry), offset = u32(dxil, entry + 4), size = u32(dxil, entry + 8);
        if (tag != 0x0d || size < 24 || std::size_t(offset) + size > dxil.size()) continue;
        const auto blob = dxil.subspan(offset, size);
        std::vector<std::uint8_t> program;
        if (std::memcmp(blob.data(), "XBCH", 4) == 0 && leonida::zlib_inflate(blob.subspan(24), program, u32(blob, 16)))
            out.push_back(std::move(program));
    }
    return out;
}

bool own_entry(const leonida::ShaderEntry& entry, const std::string& stem) {
    return entry.name == stem || entry.name.rfind(stem + "~", 0) == 0;
}

// The index.json names ShaderLibrary::load counts: entries with a string stage and fssm_sha256. Read here rather than
// through the library, which keys its entries by (stage, hash) and has no lookup by name.
bool index_names(const std::string& spirv_dir, std::set<std::string>& names) {
    std::ifstream file(fs::path(spirv_dir) / "index.json", std::ios::binary);
    if (!file) return false;
    const std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    leonida::Json root;
    if (!leonida::JsonReader(text).parse(root) || root.kind != leonida::Json::Kind::object) return false;
    for (const auto& [name, meta] : root.object) {
        const leonida::Json* hash = meta.find("fssm_sha256");
        const leonida::Json* stage = meta.find("stage");
        if (hash && hash->kind == leonida::Json::Kind::string && stage && stage->kind == leonida::Json::Kind::string)
            names.insert(name);
    }
    return true;
}

// Whether the index has an entry named after the container: the stem itself or one of its `~N` duplicates.
bool indexed(const std::set<std::string>& names, const std::string& stem) {
    for (auto it = names.lower_bound(stem); it != names.end() && it->rfind(stem, 0) == 0; ++it)
        if (it->size() == stem.size() || (*it)[stem.size()] == '~') return true;
    return false;
}

}  // namespace

int main() {
    const std::string spirv_dir = env("LEONIDA_SHADERS");
    if (spirv_dir.empty()) {
        std::printf("SKIP (no translated corpus: set LEONIDA_SHADERS to <work>/shaders/spirv)\n");
        return 77;
    }
    std::string dxbc_dir = env("LEONIDA_SHADER_DXBC");
    if (dxbc_dir.empty()) dxbc_dir = (fs::path(spirv_dir) / ".." / "dxbc").lexically_normal().string();

    int failures = 0, named = 0;
    const auto fail = [&](const std::string& what, const std::string& shader) {
        ++failures;
        if (named++ < 8) std::fprintf(stderr, "FAIL %s: %s\n", what.c_str(), shader.c_str());
    };

    leonida::ShaderLibrary library;
    std::string error;
    if (!library.load(spirv_dir, &error)) {
        std::fprintf(stderr, "FAIL loading %s: %s\n", spirv_dir.c_str(), error.c_str());
        return 1;
    }
    std::set<std::string> names;
    if (!index_names(spirv_dir, names)) {
        std::fprintf(stderr, "FAIL reading the entry names of %s/index.json\n", spirv_dir.c_str());
        return 1;
    }
    std::error_code ec;
    if (!fs::is_directory(dxbc_dir, ec)) {
        std::fprintf(stderr, "FAIL no DXBC folder %s (set LEONIDA_SHADER_DXBC)\n", dxbc_dir.c_str());
        return 1;
    }

    std::size_t containers = 0, multi = 0, untranslated = 0;
    std::set<std::string> reached;
    for (const auto& item : fs::directory_iterator(dxbc_dir)) {
        if (!item.is_regular_file() || item.path().extension() != ".dxbc") continue;
        ++containers;
        const std::string stem = item.path().stem().string();
        const auto bytes = read_file(item.path());
        const std::string stage = dxil_stage(bytes);
        if (stage.empty()) {
            fail("no shader stage in the DXIL program header", stem);
            continue;
        }
        const auto program = leonida::fssm_program(bytes);
        if (program.size() < 4 || std::memcmp(program.data(), "FSSM", 4) != 0) {
            fail("no FSSM program", stem);
            continue;
        }
        bool found = false;
        for (const leonida::ShaderEntry* entry : library.candidates(stage, leonida::sha256_hex(program))) {
            if (!own_entry(*entry, stem)) continue;
            found = true;
            reached.insert(entry->name);
            if (library.spirv(*entry).empty()) fail("no SPIR-V module for", entry->name);
        }
        if (!found && !indexed(names, stem)) {
            if (untranslated++ < 8) std::fprintf(stderr, "NOT TRANSLATED (no index entry): %s\n", stem.c_str());
        } else if (!found) {
            fail("not found under its " + stage + " stage", stem);
        }

        // The 0x0D payloads that differ from the last: none of them may be what the index has for this container.
        const auto programs = all_programs(bytes);
        if (programs.empty() || programs.back() != program) fail("fssm_program is not the last 0x0D payload", stem);
        bool differs = false;
        for (std::size_t i = 0; i + 1 < programs.size(); ++i) {
            if (programs[i] == program) continue;
            differs = true;
            for (const leonida::ShaderEntry* entry : library.candidates(stage, leonida::sha256_hex(programs[i])))
                if (own_entry(*entry, stem)) fail("an earlier 0x0D payload is the indexed program", stem);
        }
        multi += differs;
    }

    if (containers == 0) fail("no .dxbc containers in", dxbc_dir);
    const std::size_t rpf_entries = library.size() - library.fspx_size() - library.bvh_size();
    if (reached.size() != rpf_entries)
        fail(std::to_string(reached.size() > rpf_entries ? reached.size() - rpf_entries : rpf_entries - reached.size()) +
                 " RPF index entries not matched one to one",
             spirv_dir);
    std::printf("%zu containers, %zu RPF index entries, %zu reached, %zu FSPX index entries, %zu BVH builder index "
                "entries; %zu containers with more than one program; "
                "%zu not translated; %d failures\n",
                containers, rpf_entries, reached.size(), library.fspx_size(), library.bvh_size(), multi, untranslated,
                failures);
    if (untranslated)
        std::fprintf(stderr, "%zu container(s) have no index entry: the translator left them out, so the lookup had "
                             "nothing to find. Run the shaders stage again with a current translator.\n",
                     untranslated);
    return failures || untranslated ? 1 : 0;
}
