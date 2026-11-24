// Optional Windows test: map the package's Bink2Scarlett.dll through rt::Loader, verify that its xmem imports bind
// to the runtime's shim table, and decode a .bk2 through the loader's export map.
//
//   LEONIDA_GAME_DIR=<package root>   (contains Bink2Scarlett.dll)
//   LEONIDA_MOVIE=<file.bk2>          (a carved Bink 2 stream)
//   LEONIDA_GOTO=<frame>              (optional: also BinkGoto that frame)
//   LEONIDA_RUN_DLLMAIN=1             (optional: call the image entry point as DllMain before BinkOpen)
//
// With either required variable unset the test prints "SKIP (no package/movie)" and exits 77 (CTest skip).
// Output frames are written under %TEMP%\leonida_bink and are never committed.
#include <windows.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../shims/shims.h"
#include "../src/pe_loader.h"

namespace {

int g_failures = 0;

void check(bool condition, const std::string& what) {
    std::printf("%s %s\n", condition ? "PASS" : "FAIL", what.c_str());
    if (!condition) ++g_failures;
}

[[noreturn]] void fatal(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::exit(1);
}

std::string env(const char* name) {
    const char* v = std::getenv(name);
    return v ? v : "";
}

uint32_t read_u32(const uint8_t* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

struct MovieHeader {
    uint32_t stored = 0;
    uint32_t frames = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t fps_num = 0;
    uint32_t fps_den = 0;
};

MovieHeader read_movie_header(const std::filesystem::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) fatal("cannot open movie " + path.string());
    uint8_t h[0x2C]{};
    f.read(reinterpret_cast<char*>(h), sizeof(h));
    if (f.gcount() != (std::streamsize)sizeof(h)) fatal("movie is shorter than the 44-byte Bink 2 header");
    if (h[0] != 'K' || h[1] != 'B' || h[2] != '2' || h[3] < 'a' || h[3] > 'z')
        fatal("movie does not start with KB2[a-z]");
    MovieHeader out;
    out.stored = read_u32(h + 0x04) + 8;
    out.frames = read_u32(h + 0x08);
    out.width = read_u32(h + 0x14);
    out.height = read_u32(h + 0x18);
    out.fps_num = read_u32(h + 0x1C);
    out.fps_den = read_u32(h + 0x20);
    std::error_code ec;
    auto size = std::filesystem::file_size(path, ec);
    if (ec || out.stored > size) fatal("movie header's stored size is past EOF");
    if (!out.frames || !out.width || !out.height || !out.fps_num || !out.fps_den)
        fatal("movie header has a zero frame count, size or fps");
    return out;
}

// Address of the IAT slot for dll!name, so the test can compare it with the shim table, not just trust a label.
void* iat_slot(rt::Module& m, const std::string& want_dll, const std::string& want_name) {
    auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(m.base);
    auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(m.base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    auto* desc = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(m.base + dir.VirtualAddress);
    for (; desc->Name; ++desc) {
        const char* dll = reinterpret_cast<const char*>(m.base + desc->Name);
        if (rt::lower(dll) != rt::lower(want_dll)) continue;
        auto* lookup = reinterpret_cast<IMAGE_THUNK_DATA64*>(
            m.base + (desc->OriginalFirstThunk ? desc->OriginalFirstThunk : desc->FirstThunk));
        auto* iat = reinterpret_cast<IMAGE_THUNK_DATA64*>(m.base + desc->FirstThunk);
        for (; lookup->u1.AddressOfData; ++lookup, ++iat) {
            if (IMAGE_SNAP_BY_ORDINAL64(lookup->u1.Ordinal)) continue;
            auto* name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(m.base + (uint32_t)lookup->u1.AddressOfData);
            if (want_name == name->Name) return &iat->u1.Function;
        }
    }
    return nullptr;
}

uint64_t fnv1a(const std::vector<uint8_t>& bytes) {
    uint64_t h = 1469598103934665603ull;
    for (uint8_t b : bytes) {
        h ^= b;
        h *= 1099511628211ull;
    }
    return h;
}

size_t distinct_bytes(const std::vector<uint8_t>& bytes) {
    bool seen[256]{};
    size_t n = 0;
    for (uint8_t b : bytes)
        if (!seen[b]) {
            seen[b] = true;
            ++n;
        }
    return n;
}

void write_raw(const std::filesystem::path& path, const std::vector<uint8_t>& bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) fatal("cannot write " + path.string());
    f.write(reinterpret_cast<const char*>(bytes.data()), (std::streamsize)bytes.size());
}

void write_ppm(const std::filesystem::path& path, const std::vector<uint8_t>& bgra, uint32_t w, uint32_t h) {
    std::ofstream f(path, std::ios::binary);
    if (!f) fatal("cannot write " + path.string());
    f << "P6\n" << w << ' ' << h << "\n255\n";
    for (size_t i = 0; i + 3 < bgra.size(); i += 4) {
        const char rgb[3] = {(char)bgra[i + 2], (char)bgra[i + 1], (char)bgra[i]};
        f.write(rgb, 3);
    }
}

struct BinkApi {
    using Open = void* (*)(const char*, uint32_t);
    using Wait = uint32_t (*)(void*);
    using DoFrame = uint32_t (*)(void*);
    using CopyToBuffer = uint32_t (*)(void*, void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);
    using NextFrame = void (*)(void*);
    using Goto = void (*)(void*, uint32_t, uint32_t);
    using Close = void (*)(void*);
    using GetError = const char* (*)();

    Open open = nullptr;
    Wait wait = nullptr;
    DoFrame do_frame = nullptr;
    CopyToBuffer copy = nullptr;
    NextFrame next = nullptr;
    Goto go_to = nullptr;
    Close close = nullptr;
    GetError get_error = nullptr;
};

void* symbol(rt::Module& m, const char* name) {
    auto it = m.exports.find(name);
    if (it == m.exports.end()) fatal(std::string("Bink2Scarlett.dll does not export ") + name);
    return it->second;
}

}  // namespace

int main() {
    const std::string game_dir_arg = env("LEONIDA_GAME_DIR");
    const std::string movie_arg = env("LEONIDA_MOVIE");
    if (game_dir_arg.empty() || movie_arg.empty()) {
        std::printf("SKIP (no package/movie)\n");
        return 77;
    }

    const std::filesystem::path game_dir = game_dir_arg;
    const std::filesystem::path movie = movie_arg;
    std::error_code ec;
    if (!std::filesystem::is_directory(game_dir, ec)) fatal("LEONIDA_GAME_DIR is not a directory");
    if (!std::filesystem::is_regular_file(game_dir / "Bink2Scarlett.dll", ec))
        fatal("LEONIDA_GAME_DIR does not contain Bink2Scarlett.dll");
    if (!std::filesystem::is_regular_file(movie, ec)) fatal("LEONIDA_MOVIE is not a file");

    const MovieHeader header = read_movie_header(movie);
    std::printf("movie: %s  %ux%u  %u frames  %u/%u fps  %u bytes\n", movie.string().c_str(), header.width,
                header.height, header.frames, header.fps_num, header.fps_den, header.stored);

    try {
        rt::Loader loader(game_dir);
        rt::Module& bink = loader.load(game_dir / "Bink2Scarlett.dll");
        std::printf("mapped %s at %p, size %u, entry +%#x%s\n", bink.name.c_str(), bink.base, bink.size,
                    bink.entry_rva, bink.relocated ? " (relocated)" : "");

        int shim = 0, host = 0, module = 0, unimplemented = 0;
        bool xmem_alloc = false, xmem_free = false, non_audio_unimplemented = false;
        for (auto& r : bink.imports) {
            switch (r.binding) {
                case rt::Binding::Shim: ++shim; break;
                case rt::Binding::Host: ++host; break;
                case rt::Binding::Module: ++module; break;
                default: ++unimplemented; break;
            }
            if (rt::lower(r.dll) == "xmem.dll") {
                if (r.name == "XMemAlloc") xmem_alloc = true;
                if (r.name == "XMemFree") xmem_free = true;
                check(r.binding == rt::Binding::Shim, "xmem.dll!" + r.name + " binds to the shim");
                void* expected = rt::shims::find("xmem.dll", r.name);
                void* slot = iat_slot(bink, r.dll, r.name);
                check(expected && slot && *static_cast<void**>(slot) == expected,
                      "xmem.dll!" + r.name + " IAT points at rt::shims::find");
            }
            if (r.binding == rt::Binding::Unimplemented) {
                std::printf("  UNIMPLEMENTED %s!%s\n", r.dll.c_str(), r.name.c_str());
                // Audio is not exercised by this test; everything else must resolve.
                if (rt::lower(r.dll).find("xaudio") == std::string::npos) non_audio_unimplemented = true;
            }
        }
        std::printf("imports: shim=%d host=%d module=%d unimplemented=%d\n", shim, host, module, unimplemented);
        check(xmem_alloc && xmem_free, "both XMemAlloc and XMemFree imports are present");
        check(!non_audio_unimplemented, "no non-audio unimplemented imports");

        BinkApi api;
        api.open = reinterpret_cast<BinkApi::Open>(symbol(bink, "BinkOpen"));
        api.wait = reinterpret_cast<BinkApi::Wait>(symbol(bink, "BinkWait"));
        api.do_frame = reinterpret_cast<BinkApi::DoFrame>(symbol(bink, "BinkDoFrame"));
        api.copy = reinterpret_cast<BinkApi::CopyToBuffer>(symbol(bink, "BinkCopyToBuffer"));
        api.next = reinterpret_cast<BinkApi::NextFrame>(symbol(bink, "BinkNextFrame"));
        api.go_to = reinterpret_cast<BinkApi::Goto>(symbol(bink, "BinkGoto"));
        api.close = reinterpret_cast<BinkApi::Close>(symbol(bink, "BinkClose"));
        api.get_error = reinterpret_cast<BinkApi::GetError>(symbol(bink, "BinkGetError"));

        if (env("LEONIDA_RUN_DLLMAIN") == "1") {
            check(bink.entry_rva != 0, "image has an entry point to call as DllMain");
            auto dll_main = reinterpret_cast<BOOL(WINAPI*)(HMODULE, DWORD, LPVOID)>(bink.base + bink.entry_rva);
            check(dll_main(reinterpret_cast<HMODULE>(bink.base), DLL_PROCESS_ATTACH, nullptr) != FALSE,
                  "DllMain(DLL_PROCESS_ATTACH) returned TRUE");
        }

        void* handle = api.open(movie.string().c_str(), 0);
        if (!handle) {
            const char* error = api.get_error ? api.get_error() : "";
            std::printf("FAIL BinkOpen returned null: %s\n", error && *error ? error : "(no BinkGetError text)");
            return 1;
        }
        std::printf("PASS BinkOpen returned %p\n", handle);

        const uint32_t pitch = header.width * 4;
        std::vector<uint8_t> frame((size_t)pitch * header.height);
        const std::filesystem::path out_dir = std::filesystem::temp_directory_path() / "leonida_bink";
        std::filesystem::create_directories(out_dir, ec);
        const std::string stem = movie.stem().string();

        auto decode = [&](uint32_t index, const char* label) {
            const uint32_t wait = api.wait(handle);
            const uint32_t do_rc = api.do_frame(handle);
            // This GDK build returns 0 from BinkDoFrame/BinkCopyToBuffer even when it writes a valid frame; the
            // sentinel replacement and the hash below are the proof, not the return values.
            std::fill(frame.begin(), frame.end(), (uint8_t)0xCD);
            const uint32_t copy_rc = api.copy(handle, frame.data(), pitch, header.height, 0, 0, 3);
            const size_t untouched = (size_t)std::count(frame.begin(), frame.end(), 0xCD);
            const size_t unique = distinct_bytes(frame);
            const uint64_t hash = fnv1a(frame);
            std::printf("%s frame %u: wait=%u BinkDoFrame=%u BinkCopyToBuffer=%u untouched=%zu unique=%zu "
                        "hash=%016llx\n",
                        label, index, wait, do_rc, copy_rc, untouched, unique, (unsigned long long)hash);
            check(untouched < frame.size(), std::string(label) + " BinkCopyToBuffer replaced the sentinel buffer");
            check(unique > 1, std::string(label) + " wrote more than one byte value");
            const std::filesystem::path raw = out_dir / (stem + "_f" + std::to_string(index) + ".bgra");
            const std::filesystem::path ppm = out_dir / (stem + "_f" + std::to_string(index) + ".ppm");
            write_raw(raw, frame);
            write_ppm(ppm, frame, header.width, header.height);
            std::printf("  wrote %s and %s\n", raw.string().c_str(), ppm.string().c_str());
            return hash;
        };

        const uint64_t hash0 = decode(0, "first");
        api.next(handle);
        const uint64_t hash1 = decode(1, "next");
        std::printf("frame 0 vs frame 1: %s\n", hash0 == hash1 ? "identical" : "different");

        if (!env("LEONIDA_GOTO").empty()) {
            char* end = nullptr;
            unsigned long target = std::strtoul(env("LEONIDA_GOTO").c_str(), &end, 10);
            if (!end || *end || target >= header.frames) fatal("LEONIDA_GOTO is not a frame in this movie");
            api.go_to(handle, (uint32_t)target, 0);
            const uint64_t hash_goto = decode((uint32_t)target, "goto");
            std::printf("frame 0 vs goto %lu: %s\n", target, hash0 == hash_goto ? "identical" : "different");
        }

        api.close(handle);
        std::printf("%s\n", g_failures ? "RESULT FAIL" : "RESULT PASS");
        return g_failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
