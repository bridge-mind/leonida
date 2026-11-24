// Exercise the fixture's output preparation without initializing Vulkan.
#include "xdxr_build_output.h"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

namespace fs = std::filesystem;

#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

int main() {
    const auto suffix = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto root = fs::temp_directory_path() / ("leonida-xdxr-output-" + std::to_string(suffix));
    CHECK(fs::create_directory(root));
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); }
    } cleanup{root};

    const auto output = root / "relocated" / "xdxr_build_cases";
    CHECK(!prepare_xdxr_build_output(output, true));
    CHECK(!fs::exists(output.parent_path()));

    CHECK(!prepare_xdxr_build_output(output, false));
    CHECK(fs::is_directory(output));
    CHECK(bool(std::ofstream(output / "old.json") << "{}"));
    CHECK(bool(std::ofstream(output / "old.bin") << "case"));
    CHECK(bool(std::ofstream(output / "keep.txt") << "keep"));

    CHECK(!prepare_xdxr_build_output(output, true));
    CHECK(fs::is_regular_file(output / "old.json"));
    CHECK(fs::is_regular_file(output / "old.bin"));

    CHECK(!prepare_xdxr_build_output(output, false));
    CHECK(!fs::exists(output / "old.json"));
    CHECK(!fs::exists(output / "old.bin"));
    CHECK(fs::is_regular_file(output / "keep.txt"));
    CHECK(prepare_xdxr_build_output(output / "keep.txt", false));
}
