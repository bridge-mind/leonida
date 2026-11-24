#include "verify.h"

#include <cstdio>
#include <iterator>

namespace {

// A scene with zero checks proves nothing, and a count above the array size reads past
// the end. Both are contract errors, so they are build errors rather than runtime ones.
constexpr bool CheckCountsAreSane() {
    for (const conform::Scene& scene : conform::kScenes) {
        if (scene.checkCount < 1 || scene.checkCount > static_cast<int>(std::size(scene.checks))) {
            return false;
        }
    }
    return true;
}
static_assert(CheckCountsAreSane(), "every scene needs 1..N pixel checks; see conform::Scene");

}  // namespace

CheckTally VerifyChecks(const conform::Scene& scene, const PixelSurface& surface,
                        const char* source_label) {
    CheckTally tally;
    for (int index = 0; index < scene.checkCount; ++index) {
        const conform::PixelCheck& check = scene.checks[index];
        if (check.x < 0 || check.y < 0 || check.x >= surface.width || check.y >= surface.height) {
            std::printf("FAIL %s scene=%s (%d,%d) lies outside the %dx%d image  %s\n", source_label,
                        scene.name, check.x, check.y, surface.width, surface.height, check.why);
            ++tally.failed;
            continue;
        }

        const uint8_t* actual = surface.At(check.x, check.y);
        const bool matches = actual[0] == check.r && actual[1] == check.g &&
                             actual[2] == check.b && actual[3] == check.a;
        std::printf("%s %s scene=%s (%d,%d) expected %u,%u,%u,%u actual %u,%u,%u,%u  %s\n",
                    matches ? "PASS" : "FAIL", source_label, scene.name, check.x, check.y,
                    static_cast<unsigned>(check.r), static_cast<unsigned>(check.g),
                    static_cast<unsigned>(check.b), static_cast<unsigned>(check.a),
                    static_cast<unsigned>(actual[0]), static_cast<unsigned>(actual[1]),
                    static_cast<unsigned>(actual[2]), static_cast<unsigned>(actual[3]), check.why);
        if (matches) {
            ++tally.passed;
        } else {
            ++tally.failed;
        }
    }
    return tally;
}
