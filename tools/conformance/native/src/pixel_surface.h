#pragma once

#include <cstdint>
#include <vector>

// Pixels in the form both the readback path and the PNG decode path produce: four
// bytes per pixel in R,G,B,A order, top-left origin, tightly packed. Keeping one shape
// for both means a single comparison function covers rendering and file output.
struct PixelSurface {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgba;

    // Points at the RGBA bytes of one pixel. The caller owns the bounds check; sizing
    // is validated once when the surface is filled.
    const uint8_t* At(int x, int y) const {
        return rgba.data() + (static_cast<size_t>(y) * static_cast<size_t>(width) +
                              static_cast<size_t>(x)) * 4;
    }
};
