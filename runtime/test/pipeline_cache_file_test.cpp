// Exercise checkpoint replacement and real filesystem errors without a Vulkan device.
#include "pipeline_cache_file.h"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

#ifdef _WIN32
#include <windows.h>
#else
#include <csignal>
#include <sys/resource.h>
#endif

namespace fs = std::filesystem;

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

static std::string contents(const fs::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), {}};
}

static int test_checkpoint(const fs::path& dir) {
    const auto cache = dir / "cache.bin";
    const auto partial = dir / "cache.bin.partial";
    std::ofstream(partial, std::ios::binary) << "unrelated";
    const std::string saved("new\0data\r\n", 10);
    CHECK(!write_pipeline_cache_file(cache, "older checkpoint", 16));
    CHECK(contents(cache) == "older checkpoint");
    CHECK(!write_pipeline_cache_file(cache, saved.data(), saved.size()));
    CHECK(contents(cache) == saved);

    const auto directory = dir / "empty";
    CHECK(fs::create_directory(directory));
    CHECK(write_pipeline_cache_file(directory, "next", 4));
    CHECK(fs::is_directory(directory));

#ifdef _WIN32
    // A reader that did not share DELETE access prevents replacement.
    HANDLE held = CreateFileW(cache.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                              OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    CHECK(held != INVALID_HANDLE_VALUE);
    const auto failed = write_pipeline_cache_file(cache, "next", 4);
    CloseHandle(held);
    CHECK(failed);
#else
    // A buffered write may first report the size limit when it flushes or closes.
    struct rlimit original;
    CHECK(getrlimit(RLIMIT_FSIZE, &original) == 0);
    struct rlimit limited = original;
    limited.rlim_cur = 2;
    CHECK(std::signal(SIGXFSZ, SIG_IGN) != SIG_ERR);
    CHECK(setrlimit(RLIMIT_FSIZE, &limited) == 0);
    const auto failed = write_pipeline_cache_file(cache, "next", 4);
    const auto absent = dir / "absent";
    const auto failed_create = write_pipeline_cache_file(absent, "next", 4);
    CHECK(setrlimit(RLIMIT_FSIZE, &original) == 0);
    CHECK(failed && failed_create);
    CHECK(!fs::exists(absent));
#endif
    CHECK(contents(cache) == saved);
    CHECK(contents(partial) == "unrelated");
    CHECK(std::distance(fs::directory_iterator(dir), fs::directory_iterator{}) == 3);
    return 0;
}

int main() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto dir = fs::temp_directory_path() / ("leonida_pipeline_cache_" + std::to_string(stamp));
    CHECK(fs::create_directory(dir));
    const int result = test_checkpoint(dir);
    fs::remove_all(dir);
    return result;
}
