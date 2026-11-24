#pragma once

#include <filesystem>

#include "pixel_surface.h"

// Writes the surface as an 8-bit RGBA PNG. The WIC encoder is handed BGRA, the byte
// order it stores natively, so the file's R, G and B channels match the surface exactly.
bool WritePngFile(const std::filesystem::path& path, const PixelSurface& surface);

// Decodes the first frame of a PNG back into R,G,B,A bytes. Used to prove the file on
// disk, not just the readback buffer, matches the contract.
bool ReadPngFile(const std::filesystem::path& path, PixelSurface& surface_out);
