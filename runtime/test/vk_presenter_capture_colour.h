#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>

inline bool presenter_capture_matches_rgb(std::span<const std::uint8_t> pixels, std::uint32_t width,
                                          std::uint32_t height, bool bgra,
                                          const std::array<int, 3>& rgb, int tolerance) {
    const std::size_t pixel_count = pixels.size() / 4;
    if (width == 0 || height == 0 || pixels.size() % 4 != 0 ||
        pixel_count / width != height || pixel_count % width != 0)
        return false;

    for (std::size_t i = 0; i < pixels.size(); i += 4) {
        const int r = pixels[i + (bgra ? 2 : 0)];
        const int g = pixels[i + 1];
        const int b = pixels[i + (bgra ? 0 : 2)];
        if (std::abs(r - rgb[0]) > tolerance || std::abs(g - rgb[1]) > tolerance ||
            std::abs(b - rgb[2]) > tolerance)
            return false;
    }
    return true;
}

// The capture in step 2 of vk_presenter_sync_test should contain one uniform clear colour.
inline bool presenter_capture_matches_clear(std::span<const std::uint8_t> pixels, std::uint32_t width,
                                            std::uint32_t height, bool bgra) {
    return presenter_capture_matches_rgb(pixels, width, height, bgra, {51, 153, 230}, 2);
}
