#include "xbox_present_readback_size.h"

#include <array>
#include <cstdio>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

int main() {
    constexpr std::array<std::uint8_t, 8> red_frame{255, 0, 0, 255, 255, 0, 0, 255};
    EXPECT(xbox_present_readback_size_matches(red_frame.size(), 2, 1, 2, 1));
    // While the resize settles, even the previous extent is valid if the frame is complete.
    EXPECT(xbox_present_readback_size_matches(red_frame.size(), 2, 1, 0, 0));
    EXPECT(!xbox_present_readback_size_matches(red_frame.size(), 2, 1, 3, 2));

    EXPECT(!xbox_present_readback_size_matches(0, 0, 0, 0, 0));
    EXPECT(!xbox_present_readback_size_matches(0, 2, 0, 0, 0));
    EXPECT(!xbox_present_readback_size_matches(0, 0, 1, 0, 0));
    EXPECT(!xbox_present_readback_size_matches(0, 2, 1, 0, 0));
    EXPECT(!xbox_present_readback_size_matches(4, 2, 1, 0, 0));
    EXPECT(!xbox_present_readback_size_matches(7, 2, 1, 2, 1));
    EXPECT(!xbox_present_readback_size_matches(12, 2, 1, 2, 1));
    return 0;
}
