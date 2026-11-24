#include "manifest.h"

#include <cstdio>
#include <fstream>
#include <ostream>

#include "diagnostics.h"

namespace {

// Escapes the bytes JSON requires escaped. UTF-8 passes through unchanged, which JSON
// allows and keeps the adapter name readable.
void WriteJsonString(std::ostream& output, const char* text) {
    output << '"';
    for (const char* cursor = text; *cursor != '\0'; ++cursor) {
        const unsigned char byte = static_cast<unsigned char>(*cursor);
        switch (byte) {
            case '"':
                output << "\\\"";
                break;
            case '\\':
                output << "\\\\";
                break;
            case '\n':
                output << "\\n";
                break;
            case '\r':
                output << "\\r";
                break;
            case '\t':
                output << "\\t";
                break;
            default:
                if (byte < 0x20) {
                    char escape[8];
                    std::snprintf(escape, sizeof escape, "\\u%04X", byte);
                    output << escape;
                } else {
                    output << static_cast<char>(byte);
                }
                break;
        }
    }
    output << '"';
}

void WriteCheck(std::ostream& output, const conform::PixelCheck& check) {
    output << "{\"x\": " << check.x << ", \"y\": " << check.y
           << ", \"r\": " << static_cast<int>(check.r) << ", \"g\": " << static_cast<int>(check.g)
           << ", \"b\": " << static_cast<int>(check.b) << ", \"a\": " << static_cast<int>(check.a)
           << ", \"why\": ";
    WriteJsonString(output, check.why);
    output << '}';
}

void WriteScene(std::ostream& output, const conform::Scene& scene, bool is_last_scene) {
    const std::string png_name = std::string(scene.name) + ".png";
    output << "    {\n      \"name\": ";
    WriteJsonString(output, scene.name);
    output << ",\n      \"png\": ";
    WriteJsonString(output, png_name.c_str());
    output << ",\n      \"checks\": [\n";
    for (int index = 0; index < scene.checkCount; ++index) {
        output << "        ";
        WriteCheck(output, scene.checks[index]);
        output << (index + 1 < scene.checkCount ? ",\n" : "\n");
    }
    output << "      ]\n    }" << (is_last_scene ? "\n" : ",\n");
}

}  // namespace

bool WriteManifest(const std::filesystem::path& path, const std::string& adapter_description,
                   const conform::Scene* scenes, int scene_count) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        ReportFailure(__FILE__, __LINE__, "std::ofstream::open", "could not create " + path.string(),
                      "the output directory is writable and not on a read-only volume");
        return false;
    }

    output << "{\n  \"renderer\": \"native\",\n  \"adapter\": ";
    WriteJsonString(output, adapter_description.c_str());
    output << ",\n  \"scenes\": [\n";
    for (int index = 0; index < scene_count; ++index) {
        WriteScene(output, scenes[index], index + 1 == scene_count);
    }
    output << "  ]\n}\n";
    output.close();
    if (output.fail()) {
        ReportFailure(__FILE__, __LINE__, "std::ofstream::close", "could not write " + path.string(),
                      "the disk has free space; a full volume fails on close, not on open");
        return false;
    }
    return true;
}
