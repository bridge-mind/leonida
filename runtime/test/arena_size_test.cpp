// The GPU address arena's size (src/arena_size.h) without a GPU: LEONIDA_ARENA_MB's text against synthetic
// maxBufferSize and maxMemoryAllocationSize values, including the RX 6800's (AMD 26.11.2): 2 GiB for both.
// No game data.
#include "../src/arena_size.h"

#include <cstdio>
#include <string_view>

#define CHECK(condition) do {                                    \
    if (!(condition)) {                                          \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                \
    }                                                            \
} while (0)

using namespace leonida;

namespace {

constexpr VkDeviceSize MiB = kArenaMiB;
constexpr VkDeviceSize GiB = 1024 * MiB;

int request_without_limits() {
    const ArenaLimits none{};
    // Unset (or empty): the default; otherwise clamped to 16..4095 MiB; text that is no number reads as 0, so 16.
    CHECK(decide_arena_size(nullptr, none).bytes == 256 * MiB);
    CHECK(decide_arena_size("", none).bytes == 256 * MiB);
    CHECK(decide_arena_size("1024", none).bytes == 1024 * MiB);
    CHECK(decide_arena_size("1", none).bytes == 16 * MiB);
    CHECK(decide_arena_size("0", none).bytes == 16 * MiB);
    CHECK(decide_arena_size("lots", none).bytes == 16 * MiB);
    CHECK(decide_arena_size("99999", none).bytes == 4095 * MiB);
    const ArenaSize s = decide_arena_size("3000", none);
    CHECK(s.bytes == 3000 * MiB && s.requested == 3000 * MiB && s.capped == arena_cap_none);
    return 0;
}

int each_limit() {
    // RX 6800: both limits 2 GiB. Under them nothing changes; over them both are named.
    const ArenaLimits rdna2{2 * GiB, 2 * GiB};
    ArenaSize s = decide_arena_size("2000", rdna2);
    CHECK(s.bytes == 2000 * MiB && s.capped == arena_cap_none);
    s = decide_arena_size("2048", rdna2);
    CHECK(s.bytes == 2 * GiB && s.capped == arena_cap_none);
    s = decide_arena_size("3000", rdna2);
    CHECK(s.bytes == 2 * GiB && s.requested == 3000 * MiB && s.capped == (arena_cap_buffer | arena_cap_allocation));
    CHECK(std::string_view(arena_cap_name(s.capped)) == "maxBufferSize and maxMemoryAllocationSize");

    // One limit below the other: that one applies.
    s = decide_arena_size("4095", ArenaLimits{1 * GiB, 4 * GiB});
    CHECK(s.bytes == 1 * GiB && s.capped == arena_cap_buffer);
    CHECK(std::string_view(arena_cap_name(s.capped)) == "maxBufferSize");
    s = decide_arena_size("4095", ArenaLimits{4 * GiB, 3 * GiB});
    CHECK(s.bytes == 3 * GiB && s.capped == arena_cap_allocation);
    CHECK(std::string_view(arena_cap_name(s.capped)) == "maxMemoryAllocationSize");

    // A limit that is not a whole number of MiB rounds the arena down to one.
    s = decide_arena_size("4095", ArenaLimits{2 * GiB + 12345, 0});
    CHECK(s.bytes == 2 * GiB && s.capped == arena_cap_buffer);
    return 0;
}

int unknown_limits() {
    // 0 is "not reported": it caps nothing, and the other limit still does.
    ArenaSize s = decide_arena_size("3000", ArenaLimits{0, 0});
    CHECK(s.bytes == 3000 * MiB && s.capped == arena_cap_none);
    s = decide_arena_size("3000", ArenaLimits{0, 2 * GiB});
    CHECK(s.bytes == 2 * GiB && s.capped == arena_cap_allocation);
    s = decide_arena_size("3000", ArenaLimits{2 * GiB, 0});
    CHECK(s.bytes == 2 * GiB && s.capped == arena_cap_buffer);
    return 0;
}

int allocation_overhead() {
    // The allocation is the buffer's memory requirement, which can be larger than the buffer: the allocation limit
    // is checked against buffer + overhead, the buffer limit against the buffer alone.
    const ArenaLimits rdna2{2 * GiB, 2 * GiB};
    ArenaSize s = decide_arena_size("3000", rdna2, 64 * 1024);
    CHECK(s.bytes == 2047 * MiB && s.capped == arena_cap_allocation);
    CHECK(s.bytes + 64 * 1024 <= rdna2.max_allocation_size);
    s = decide_arena_size("3000", rdna2, 1 * MiB);
    CHECK(s.bytes == 2047 * MiB && s.capped == arena_cap_allocation);
    s = decide_arena_size("1024", rdna2, 1 * MiB);  // still far under: unchanged
    CHECK(s.bytes == 1024 * MiB && s.capped == arena_cap_none);
    // Overhead at or past the limit leaves nothing.
    s = decide_arena_size("3000", ArenaLimits{0, 4 * MiB}, 4 * MiB);
    CHECK(s.bytes == 0 && s.capped == arena_cap_allocation);
    return 0;
}

int floor_and_limits() {
    // The 16 MiB floor applies to the request; a device limit wins over it (no conformant device reports under
    // 1 GiB, so this is a simulated device).
    ArenaSize s = decide_arena_size("1", ArenaLimits{1 * GiB, 1 * GiB});
    CHECK(s.bytes == 16 * MiB && s.capped == arena_cap_none);
    s = decide_arena_size("1", ArenaLimits{8 * MiB, 0});
    CHECK(s.bytes == 8 * MiB && s.requested == 16 * MiB && s.capped == arena_cap_buffer);
    s = decide_arena_size("64", ArenaLimits{0, 32 * MiB});
    CHECK(s.bytes == 32 * MiB && s.capped == arena_cap_allocation);
    // Under 1 MiB: nothing to create.
    s = decide_arena_size(nullptr, ArenaLimits{512 * 1024, 0});
    CHECK(s.bytes == 0 && s.capped == arena_cap_buffer);
    return 0;
}

}  // namespace

int main() {
    for (const auto test : {request_without_limits, each_limit, unknown_limits, allocation_overhead, floor_and_limits})
        if (test()) return 1;
    std::printf("arena_size: request clamp, each device limit, unknown limits, allocation overhead and the floor\n");
    return 0;
}
