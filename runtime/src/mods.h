// Mods: folders under mods/ with a mod.json that the runtime reads at start (docs/modding.md).
//
// A mod never ships game data. It names things the game already has, by the names (or joaat hashes) the game uses,
// and says how to swap them. The first kind is an archetype swap: whenever the game asks for the model archetype
// `from`, it gets `to` instead (e.g. Lucia's player model player_story000_00 -> cs_mandy). Applying a swap needs one
// hook in the game: its archetype lookup by hash, whose address is per executable version and is filled in once the
// executable is analysed. Until then the swaps are loaded, checked and logged, and `remap_archetype` is ready for it.
//
// mod.json:
//   { "name": "Mandy in the prison intro", "enabled": true, "game_version": "1.0.2.1",
//     "archetype_swaps": [ { "from": "player_story000_00", "to": "cs_mandy", "when": "always" } ] }
// `from`/`to`: a model name (hashed with joaat, lower case, as the game does) or a 32-bit hash written "0x...".
// `when`: "always" (default). Narrower conditions (one mission, one cutscene) need the game's mission state and are
// rejected until a hook provides it, so a mod never silently does more than it says.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "shader_library.h"  // leonida::Json / JsonReader

namespace leonida::mods {

// RAGE's joaat (Jenkins one-at-a-time over the lower-cased name): joaat("player_story000_00") = 0xE472B687,
// joaat("cs_mandy") = 0xD7C0CDFF, the model ids docs/character_skeleton.md measured in the packs tables.
constexpr uint32_t joaat(std::string_view name) {
    uint32_t h = 0;
    for (char c : name) {
        const auto u = static_cast<unsigned char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
        h += u;
        h += h << 10;
        h ^= h >> 6;
    }
    h += h << 3;
    h ^= h >> 11;
    h += h << 15;
    return h;
}

struct ArchetypeSwap {
    uint32_t from = 0, to = 0;
    std::string from_name, to_name;  // as written in mod.json
    std::string when = "always";
};

// "camera": the view mode the game's view-mode getter answers (docs/modding.md, "First-person view"). The getter
// takes a context (on foot, in a vehicle, ...) and returns an enum; `first_person_value` is that enum's first-person
// value, 4 in GTA V's camViewModeSettings, to be confirmed in the executable. `contexts`: the getter's first-argument
// values to override; empty = every context.
struct CameraView {
    bool set = false;           // the mod has a "camera" section
    bool first_person = false;  // "view_mode": "first_person" (else "default": leave the game's choice)
    uint32_t first_person_value = 4;
    std::vector<uint32_t> contexts;
};

struct Mod {
    std::string folder;  // the folder name under mods/
    std::string name;
    std::string game_version;
    bool enabled = true;
    std::vector<ArchetypeSwap> archetype_swaps;
    CameraView camera;
};

struct LoadResult {
    std::vector<Mod> mods;          // every mod.json that parsed, enabled or not
    std::vector<std::string> errors;  // "<folder>: what was wrong" (that mod is skipped)
};

// "0x..." -> that value; anything else -> joaat(name). Returns false for an empty name or a malformed hex value.
inline bool parse_model_id(const std::string& text, uint32_t& out) {
    if (text.empty()) return false;
    if (text.size() > 2 && text[0] == '0' && (text[1] == 'x' || text[1] == 'X')) {
        if (text.size() > 10) return false;
        uint32_t v = 0;
        for (std::size_t i = 2; i < text.size(); ++i) {
            const char c = text[i];
            const int d = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10
                        : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
            if (d < 0) return false;
            v = v << 4 | static_cast<uint32_t>(d);
        }
        out = v;
        return true;
    }
    out = joaat(text);
    return true;
}

// A JSON number that is a whole value in [0, max]. Checked before any integer cast: converting a double outside the
// target type's range is undefined, so `v != static_cast<int>(v)` cannot be the test for a value past INT_MAX.
inline bool whole_number(double v, double max) { return v >= 0 && v <= max && v == std::floor(v); }

// Parses one mod.json's text. On failure returns false with `error` set; `mod` is then not to be used.
inline bool parse_mod(std::string_view text, const std::string& folder, Mod& mod, std::string& error) {
    // Windows editors and shells mark the encoding. Notepad's "UTF-8 with BOM" and Windows PowerShell 5.1's
    // `-Encoding utf8` put EF BB BF in front of the same JSON, so it is skipped. PowerShell 5.1's `>` writes UTF-16,
    // which is not readable as bytes: say so, instead of blaming the JSON. A NUL in the first two bytes is UTF-16
    // without a mark; no UTF-8 JSON text starts that way.
    if (text.substr(0, 3) == std::string_view("\xEF\xBB\xBF", 3)) {
        text.remove_prefix(3);
    } else if (text.size() >= 2 && (text.substr(0, 2) == std::string_view("\xFF\xFE", 2) ||
                                    text.substr(0, 2) == std::string_view("\xFE\xFF", 2) || text[0] == '\0' ||
                                    text[1] == '\0')) {
        error = "mod.json is UTF-16; save it as UTF-8";
        return false;
    }
    Json root;
    if (!JsonReader(text).parse(root) || root.kind != Json::Kind::object) {
        error = "mod.json is not a JSON object";
        return false;
    }
    mod = Mod{};
    mod.folder = folder;
    const Json* name = root.find("name");
    mod.name = name && name->kind == Json::Kind::string ? name->string : folder;
    if (const Json* v = root.find("game_version"); v && v->kind == Json::Kind::string) mod.game_version = v->string;
    if (const Json* e = root.find("enabled")) {
        if (e->kind != Json::Kind::boolean) {
            error = "\"enabled\" must be true or false";
            return false;
        }
        mod.enabled = e->boolean;
    }
    if (const Json* swaps = root.find("archetype_swaps")) {
        if (swaps->kind != Json::Kind::array) {
            error = "\"archetype_swaps\" must be a list";
            return false;
        }
        for (const Json& s : swaps->array) {
            const Json* from = s.find("from");
            const Json* to = s.find("to");
            if (s.kind != Json::Kind::object || !from || !to || from->kind != Json::Kind::string ||
                to->kind != Json::Kind::string) {
                error = "each archetype swap needs \"from\" and \"to\" model names";
                return false;
            }
            ArchetypeSwap swap;
            swap.from_name = from->string;
            swap.to_name = to->string;
            if (!parse_model_id(swap.from_name, swap.from) || !parse_model_id(swap.to_name, swap.to)) {
                error = "archetype swap " + swap.from_name + " -> " + swap.to_name + ": not a model name or 0x hash";
                return false;
            }
            if (const Json* when = s.find("when")) {
                if (when->kind != Json::Kind::string || when->string != "always") {
                    error = "archetype swap " + swap.from_name + ": \"when\" can only be \"always\" until a game hook "
                            "provides mission state (docs/modding.md)";
                    return false;
                }
            }
            if (swap.from == swap.to) {
                error = "archetype swap " + swap.from_name + " -> " + swap.to_name + " swaps a model with itself";
                return false;
            }
            mod.archetype_swaps.push_back(std::move(swap));
        }
    }
    if (const Json* cam = root.find("camera")) {
        if (cam->kind != Json::Kind::object) {
            error = "\"camera\" must be an object";
            return false;
        }
        mod.camera.set = true;
        const Json* view = cam->find("view_mode");
        if (!view || view->kind != Json::Kind::string ||
            (view->string != "first_person" && view->string != "default")) {
            error = "\"camera\": \"view_mode\" must be \"first_person\" or \"default\"";
            return false;
        }
        mod.camera.first_person = view->string == "first_person";
        if (const Json* v = cam->find("first_person_value")) {
            if (v->kind != Json::Kind::number || !whole_number(v->number, 255)) {
                error = "\"camera\": \"first_person_value\" must be a whole number 0-255";
                return false;
            }
            mod.camera.first_person_value = static_cast<uint32_t>(v->number);
        }
        if (const Json* c = cam->find("contexts")) {
            if (c->kind != Json::Kind::array) {
                error = "\"camera\": \"contexts\" must be a list of numbers";
                return false;
            }
            for (const Json& x : c->array) {
                if (x.kind != Json::Kind::number || !whole_number(x.number, 4294967295.0)) {
                    error = "\"camera\": \"contexts\" must be a list of whole numbers from 0 to 4294967295";
                    return false;
                }
                mod.camera.contexts.push_back(static_cast<uint32_t>(x.number));
            }
        }
    }
    return true;
}

// Reads every <dir>/<folder>/mod.json, in folder-name order. A missing dir is no mods, not an error.
inline LoadResult load_mods(const std::filesystem::path& dir) {
    LoadResult result;
    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec)) return result;
    std::vector<std::filesystem::path> folders;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec))
        if (entry.is_directory(ec)) folders.push_back(entry.path());
    std::sort(folders.begin(), folders.end());
    for (const auto& folder : folders) {
        const std::string name = folder.filename().string();
        std::ifstream f(folder / "mod.json", std::ios::binary);
        if (!f) continue;  // a folder without a manifest is not a mod (notes, assets of another tool)
        const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        Mod mod;
        std::string error;
        if (parse_mod(text, name, mod, error))
            result.mods.push_back(std::move(mod));
        else
            result.errors.push_back(name + ": " + error);
    }
    return result;
}

// The swap table the archetype hook consults: enabled mods only, later folders win for the same `from`.
class ArchetypeSwaps {
public:
    explicit ArchetypeSwaps(const std::vector<Mod>& mods) {
        for (const Mod& m : mods) {
            if (!m.enabled) continue;
            for (const ArchetypeSwap& s : m.archetype_swaps) {
                bool replaced = false;
                for (auto& [from, to] : table_)
                    if (from == s.from) to = s.to, replaced = true;
                if (!replaced) table_.emplace_back(s.from, s.to);
            }
        }
    }

    // The archetype the game should get when it asks for `hash` (itself when no mod swaps it). One step only: a swap
    // to a model that is itself swapped is not followed, so two mods cannot loop.
    uint32_t remap(uint32_t hash) const {
        for (const auto& [from, to] : table_)
            if (from == hash) return to;
        return hash;
    }

    std::size_t size() const { return table_.size(); }

private:
    std::vector<std::pair<uint32_t, uint32_t>> table_;
};

// The process's swaps: set once at start (main.cpp), before any game code runs; read by the archetype hook.
inline ArchetypeSwaps& active_slot() {
    static ArchetypeSwaps swaps{std::vector<Mod>{}};
    return swaps;
}
inline void set_active(ArchetypeSwaps swaps) { active_slot() = std::move(swaps); }
inline const ArchetypeSwaps& active() { return active_slot(); }

// The camera override of the enabled mods: the last one (in folder order) with a "camera" section decides.
inline CameraView camera_of(const std::vector<Mod>& mods) {
    CameraView view;
    for (const Mod& m : mods)
        if (m.enabled && m.camera.set) view = m.camera;
    return view;
}

inline CameraView& camera_slot() {
    static CameraView view;
    return view;
}
inline void set_active_camera(CameraView view) { camera_slot() = std::move(view); }

// What the hooked view-mode getter returns: the mod's first-person value for an overridden context, else the game's.
inline uint32_t override_view_mode(uint32_t result, uint32_t context) {
    const CameraView& v = camera_slot();
    if (!v.first_person) return result;
    if (!v.contexts.empty()) {
        bool listed = false;
        for (uint32_t c : v.contexts) listed = listed || c == context;
        if (!listed) return result;
    }
    return v.first_person_value;
}

}  // namespace leonida::mods
