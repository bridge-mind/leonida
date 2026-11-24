#pragma once

#include <filesystem>

inline std::error_code prepare_xdxr_build_output(const std::filesystem::path& dir, bool timing_only) {
    if (timing_only) return {};

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return ec;

    // A failed run must not leave the previous run's cases for the checker to pass.
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        const auto ext = entry.path().extension();
        if (ext == ".bin" || ext == ".json") std::filesystem::remove(entry.path(), ec);
    }
    return {};
}
