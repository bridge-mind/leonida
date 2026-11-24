#pragma once

#include <cstddef>
#include <cstdint>

inline bool xbox_present_readback_size_matches(std::size_t bytes, std::uint32_t width,
                                              std::uint32_t height, std::uint32_t expected_width,
                                              std::uint32_t expected_height) {
    const std::size_t pixels = bytes / 4;
    if (width == 0 || height == 0 || bytes % 4 != 0 ||
        pixels / width != height || pixels % width != 0)
        return false;

    // A resize may still produce a complete frame at the previous extent.
    return expected_width == 0 || (width == expected_width && height == expected_height);
}
