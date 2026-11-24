// mods.h: mod.json manifests and archetype swaps (docs/modding.md). No game files.
#include "../src/mods.h"
#include "test_temp_dir.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

namespace {

int g_checks = 0, g_failures = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (!ok) {
        ++g_failures;
        std::printf("FAIL %s\n", what);
    }
}

void write(const std::filesystem::path& p, const std::string& text) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p, std::ios::binary) << text;
}

}  // namespace

int main() {
    using namespace leonida::mods;

    // The model ids docs/character_skeleton.md measured in the packs tables are the joaat of their names.
    static_assert(joaat("player_story000_00") == 0xE472B687u, "Lucia's player model id");
    static_assert(joaat("cs_mandy") == 0xD7C0CDFFu, "cs_mandy's model id");
    static_assert(joaat("mandy") == 0x03A885E6u, "Mandy's PERSCHARS name hash");
    static_assert(joaat("CS_Mandy") == joaat("cs_mandy"), "joaat lower-cases");

    uint32_t id = 0;
    check(parse_model_id("0xD7C0CDFF", id) && id == 0xD7C0CDFFu, "a 0x hash is taken as it is");
    check(parse_model_id("cs_monalisa", id) && id == 0x9CB9BCC6u, "a name is hashed");
    check(!parse_model_id("0x123456789", id), "a hash wider than 32 bits is refused");
    check(!parse_model_id("0xZZ", id), "a malformed hash is refused");
    check(!parse_model_id("", id), "an empty name is refused");

    Mod mod;
    std::string error;
    check(parse_mod(R"({"name":"Mandy","archetype_swaps":[{"from":"player_story000_00","to":"cs_mandy"}]})", "m",
                    mod, error) && mod.enabled && mod.archetype_swaps.size() == 1 &&
              mod.archetype_swaps[0].from == 0xE472B687u && mod.archetype_swaps[0].to == 0xD7C0CDFFu,
          "a swap parses into the two model ids");
    check(!parse_mod(R"({"archetype_swaps":[{"from":"a","to":"b","when":"mission:intro"}]})", "m", mod, error) &&
              error.find("\"when\"") != std::string::npos,
          "a condition the runtime cannot honour yet is refused, not ignored");
    check(!parse_mod(R"({"archetype_swaps":[{"from":"cs_mandy","to":"0xD7C0CDFF"}]})", "m", mod, error),
          "a swap of a model with itself is refused");
    check(!parse_mod(R"({"enabled":"yes"})", "m", mod, error), "\"enabled\" must be a boolean");
    check(!parse_mod("[1,2]", "m", mod, error), "a manifest that is not an object is refused");
    check(parse_mod("{}", "folder_name", mod, error) && mod.name == "folder_name", "the name defaults to the folder");

    // The encodings Windows editors and shells write. A UTF-8 byte-order mark is skipped; UTF-16 is named.
    const std::string manifest =
        R"({"name":"Mandy","archetype_swaps":[{"from":"player_story000_00","to":"cs_mandy"}]})";
    const std::string bom = "\xEF\xBB\xBF";
    check(parse_mod(bom + manifest, "m", mod, error) && mod.name == "Mandy" && mod.archetype_swaps.size() == 1 &&
              mod.archetype_swaps[0].to == 0xD7C0CDFFu,
          "a manifest saved as UTF-8 with a byte-order mark parses");
    check(parse_mod(bom + "{\r\n  \"name\": \"crlf\"\r\n}\r\n", "m", mod, error) && mod.name == "crlf",
          "a byte-order mark before CRLF text parses");
    check(!parse_mod(bom + "[1,2]", "m", mod, error) && error == "mod.json is not a JSON object",
          "a byte-order mark does not make a list a manifest");
    check(!parse_mod(bom, "m", mod, error) && !parse_mod(bom + bom + manifest, "m", mod, error),
          "a byte-order mark alone, or two of them, is still refused");
    const auto utf16 = [&manifest](bool little_endian, bool mark) {
        std::string out;
        if (mark) out = little_endian ? std::string("\xFF\xFE", 2) : std::string("\xFE\xFF", 2);
        for (const char c : manifest) {
            if (!little_endian) out.push_back('\0');
            out.push_back(c);
            if (little_endian) out.push_back('\0');
        }
        return out;
    };
    for (const std::string& text : {utf16(true, true), utf16(false, true), utf16(true, false), utf16(false, false)})
        check(!parse_mod(text, "m", mod, error) && error == "mod.json is UTF-16; save it as UTF-8",
              "a UTF-16 manifest is refused by name, with or without a byte-order mark");
    check(!parse_mod("", "m", mod, error) && !parse_mod("{", "m", mod, error) &&
              error == "mod.json is not a JSON object",
          "an empty or one-byte manifest is not mistaken for UTF-16");

    // "camera": first-person view (docs/modding.md, "First-person view")
    check(parse_mod(R"({"camera":{"view_mode":"first_person","first_person_value":4,"contexts":[0]}})", "fp", mod,
                    error) && mod.camera.set && mod.camera.first_person && mod.camera.first_person_value == 4 &&
              mod.camera.contexts.size() == 1 && mod.camera.contexts[0] == 0,
          "a first-person camera section parses");
    check(parse_mod(R"({"camera":{"view_mode":"default"}})", "d", mod, error) && mod.camera.set &&
              !mod.camera.first_person,
          "\"default\" leaves the game's view mode");
    check(!parse_mod(R"({"camera":{"view_mode":"top_down"}})", "x", mod, error), "an unknown view mode is refused");
    check(!parse_mod(R"({"camera":{"view_mode":"first_person","first_person_value":4.5}})", "x", mod, error),
          "a fractional enum value is refused");
    check(!parse_mod(R"({"camera":{"view_mode":"first_person","contexts":"all"}})", "x", mod, error),
          "contexts must be a list");
    // A context is a uint32: every whole value up to 4294967295 is taken, and nothing is cast before it is bounded.
    check(parse_mod(R"({"camera":{"view_mode":"first_person","contexts":[2147483647,2147483648,4294967295]}})", "x",
                    mod, error) && mod.camera.contexts.size() == 3 && mod.camera.contexts[0] == 2147483647u &&
              mod.camera.contexts[1] == 2147483648u && mod.camera.contexts[2] == 4294967295u,
          "contexts past INT_MAX are whole numbers too");
    for (const char* value : {"4294967296", "1e300", "-1", "1.5", "-0.5"}) {
        const std::string text = std::string(R"({"camera":{"view_mode":"first_person","contexts":[)") + value + "]}}";
        check(!parse_mod(text, "x", mod, error) &&
                  error.find("whole numbers from 0 to 4294967295") != std::string::npos,
              "a context that is not a whole uint32 is refused");
    }
    for (const char* value : {"256", "-1", "1e300", "4.5"}) {
        const std::string text =
            std::string(R"({"camera":{"view_mode":"first_person","first_person_value":)") + value + "}}";
        check(!parse_mod(text, "x", mod, error), "a first_person_value outside 0-255 is refused");
    }
    check(parse_mod(R"({"camera":{"view_mode":"first_person","first_person_value":255}})", "x", mod, error) &&
              mod.camera.first_person_value == 255,
          "255 is the largest first_person_value");

    Mod fp_on_foot, fp_everywhere, off;
    check(parse_mod(R"({"camera":{"view_mode":"first_person","contexts":[0]}})", "a", fp_on_foot, error) &&
              parse_mod(R"({"camera":{"view_mode":"first_person","first_person_value":7}})", "b", fp_everywhere,
                        error) &&
              parse_mod(R"({"enabled":false,"camera":{"view_mode":"default"}})", "c", off, error),
          "three camera mods parse");
    set_active_camera(camera_of({fp_on_foot}));
    check(override_view_mode(2, 0) == 4, "on foot (context 0) the view becomes first person");
    check(override_view_mode(2, 1) == 2, "a context the mod does not list keeps the game's view");
    set_active_camera(camera_of({fp_on_foot, fp_everywhere, off}));
    check(override_view_mode(1, 5) == 7, "the last enabled camera mod decides (a disabled one is skipped)");
    set_active_camera(camera_of({}));
    check(override_view_mode(3, 0) == 3, "with no camera mod the game's view mode passes through");

    const testfiles::ScratchDir scratch("leonida_mods_test");
    const auto& dir = scratch.path();
    write(dir / "a_mandy" / "mod.json",
          R"({"name":"Mandy","archetype_swaps":[{"from":"player_story000_00","to":"cs_mandy","when":"always"}]})");
    write(dir / "b_off" / "mod.json",
          R"({"name":"Off","enabled":false,"archetype_swaps":[{"from":"cs_mandy","to":"cs_monalisa"}]})");
    write(dir / "c_broken" / "mod.json", "{not json");
    write(dir / "d_notes" / "readme.txt", "not a mod");
    write(dir / "e_later" / "mod.json", R"({"archetype_swaps":[{"from":"player_story000_00","to":"cs_monalisa"}]})");

    const LoadResult loaded = load_mods(dir);
    check(loaded.mods.size() == 3, "three manifests load (the broken one and the folder without one do not)");
    check(loaded.errors.size() == 1 && loaded.errors[0].rfind("c_broken:", 0) == 0, "the broken one is reported");

    const ArchetypeSwaps swaps(loaded.mods);
    check(swaps.remap(0xE472B687u) == joaat("cs_monalisa"), "a later folder wins for the same model");
    check(swaps.remap(0xD7C0CDFFu) == 0xD7C0CDFFu, "a disabled mod swaps nothing");
    check(swaps.remap(0x12345678u) == 0x12345678u, "a model no mod names is left alone");

    const ArchetypeSwaps mandy_only({loaded.mods[0]});
    check(mandy_only.remap(0xE472B687u) == 0xD7C0CDFFu, "the Mandy mod swaps Lucia's player model for cs_mandy");

    check(load_mods(dir / "missing").mods.empty(), "no mods folder is no mods");

    std::printf("mods_test: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
