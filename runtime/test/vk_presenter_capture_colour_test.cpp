#include "vk_presenter_capture_colour.h"

#include <array>
#include <cstdio>
#include <initializer_list>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

int main() {
    constexpr std::uint32_t width = 5, height = 3;
    for (const bool bgra : {false, true}) {
        const std::array<std::uint8_t, 4> clear = bgra ? std::array<std::uint8_t, 4>{230, 153, 51, 255}
                                                      : std::array<std::uint8_t, 4>{51, 153, 230, 255};
        std::vector<std::uint8_t> good;
        for (std::uint32_t i = 0; i < width * height; ++i)
            good.insert(good.end(), clear.begin(), clear.end());
        EXPECT(presenter_capture_matches_clear(good, width, height, bgra));

        auto stale_edge = good;
        for (std::uint32_t y = 0; y < height; ++y) {
            const auto offset = (y * width + width - 1) * 4;
            stale_edge[offset] = stale_edge[offset + 1] = stale_edge[offset + 2] = 0;
        }
        EXPECT(!presenter_capture_matches_clear(stale_edge, width, height, bgra));

        auto drift = good;
        drift[1] += 2;
        EXPECT(presenter_capture_matches_clear(drift, width, height, bgra));
        ++drift[1];
        EXPECT(!presenter_capture_matches_clear(drift, width, height, bgra));

        EXPECT(!presenter_capture_matches_clear({}, width, height, bgra));
        EXPECT(!presenter_capture_matches_clear({}, 0, 0, bgra));
        good.pop_back();
        EXPECT(!presenter_capture_matches_clear(good, width, height, bgra));
        good.resize(good.size() - 3);
        EXPECT(!presenter_capture_matches_clear(good, width, height, bgra));

        const std::array<std::uint8_t, 4> green = bgra ? std::array<std::uint8_t, 4>{102, 204, 51, 255}
                                                      : std::array<std::uint8_t, 4>{51, 204, 102, 255};
        std::vector<std::uint8_t> composed;
        for (std::uint32_t i = 0; i < width * height; ++i)
            composed.insert(composed.end(), green.begin(), green.end());
        constexpr std::array<int, 3> expected_green{51, 204, 102};
        const auto edge_green = composed.size() - 3;
        composed[edge_green] -= 12;
        EXPECT(presenter_capture_matches_rgb(composed, width, height, bgra, expected_green, 12));
        composed[edge_green] = clear[1];
        EXPECT(!presenter_capture_matches_rgb(composed, width, height, bgra, expected_green, 12));
        composed.resize(composed.size() - 4);
        EXPECT(!presenter_capture_matches_rgb(composed, width, height, bgra, expected_green, 12));
    }
    return 0;
}
