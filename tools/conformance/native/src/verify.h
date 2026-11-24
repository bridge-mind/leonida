#pragma once

#include "pixel_surface.h"
#include "scenes.h"

// How many of a scene's contract checks matched one pixel surface. Kept per source so
// the summary can say which path failed: rendering or PNG encoding.
struct CheckTally {
    int passed = 0;
    int failed = 0;
};

// Compares every conform::PixelCheck of one scene against the surface and prints one
// PASS/FAIL line per check with the scene, coordinate, expected and actual pixel. Does
// not stop at the first failure, and does not modify the surface.
CheckTally VerifyChecks(const conform::Scene& scene, const PixelSurface& surface,
                        const char* source_label);
