#pragma once

// The size of the GPU address arena (docs/d3d12_vulkan.md section 3.3): one VkBuffer bound to one VkDeviceMemory
// allocation. LEONIDA_ARENA_MB asks for a size (default 256 MiB, 16..4095 MiB: every console VA must fit 32 bits);
// the device then caps it, since the buffer may be no larger than maxBufferSize (Vulkan 1.3 / VK_KHR_maintenance4)
// and the allocation, which can be larger than the buffer, no larger than maxMemoryAllocationSize (Vulkan 1.1 /
// VK_KHR_maintenance3). Vulkan guarantees at least 1 GiB for both; the RX 6800 (AMD 26.11.2) reports 2 GiB for both.
//
// The decision is a pure function of the variable's text and the two limits, so it is tested without a GPU
// (arena_size_test). It depends only on the Vulkan headers.

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>

namespace leonida {

constexpr VkDeviceSize kArenaMiB = VkDeviceSize(1) << 20;
constexpr VkDeviceSize kArenaDefaultMiB = 256;
constexpr VkDeviceSize kArenaMinMiB = 16;
constexpr VkDeviceSize kArenaMaxMiB = 4095;

// The device's limits; 0 is "not known", which caps nothing.
struct ArenaLimits {
    VkDeviceSize max_buffer_size = 0;      // VkPhysicalDeviceMaintenance4Properties::maxBufferSize
    VkDeviceSize max_allocation_size = 0;  // VkPhysicalDeviceMaintenance3Properties::maxMemoryAllocationSize
};

// Reads both limits with vkGetPhysicalDeviceProperties2 (the device is Vulkan 1.3, which has both structures).
template <class GetProperties2>
ArenaLimits query_arena_limits(GetProperties2 get_properties2, VkPhysicalDevice physical) {
    VkPhysicalDeviceMaintenance4Properties m4{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_4_PROPERTIES};
    VkPhysicalDeviceMaintenance3Properties m3{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES, &m4};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &m3};
    get_properties2(physical, &properties);
    return {m4.maxBufferSize, m3.maxMemoryAllocationSize};
}

// Which limits made the arena smaller than asked (both bits when they cap it to the same size).
enum ArenaCap : unsigned {
    arena_cap_none = 0,
    arena_cap_buffer = 1,      // maxBufferSize
    arena_cap_allocation = 2,  // maxMemoryAllocationSize
};

struct ArenaSize {
    VkDeviceSize requested = 0;  // LEONIDA_ARENA_MB (or the default) after the 16..4095 MiB clamp, in bytes
    VkDeviceSize bytes = 0;      // the buffer size to create; 0 when a limit leaves less than 1 MiB
    unsigned capped = arena_cap_none;
};

// `env` is LEONIDA_ARENA_MB's text, or null when it is unset or empty. `allocation_overhead` is how much larger the
// buffer's memory requirement is than the buffer (alignment padding): 0 for a first try, then requirements.size -
// buffer size if that allocation would pass maxMemoryAllocationSize.
//   - The request is clamped to 16..4095 MiB first (text that is not a number reads as 0, so 16).
//   - Each known limit then caps it, rounded down to whole MiB: the buffer to maxBufferSize, the buffer plus the
//     overhead to maxMemoryAllocationSize. A limit wins over the 16 MiB floor: an arena over a device limit is
//     invalid Vulkan, while one under 16 MiB is only small (and no conformant device reports under 1 GiB).
inline ArenaSize decide_arena_size(const char* env, const ArenaLimits& limits, VkDeviceSize allocation_overhead = 0) {
    VkDeviceSize megabytes = kArenaDefaultMiB;
    if (env && *env) megabytes = std::clamp<VkDeviceSize>(std::strtoull(env, nullptr, 10), kArenaMinMiB, kArenaMaxMiB);
    ArenaSize size;
    size.requested = size.bytes = megabytes * kArenaMiB;
    VkDeviceSize by_buffer = size.requested, by_allocation = size.requested;
    if (limits.max_buffer_size) by_buffer = limits.max_buffer_size / kArenaMiB * kArenaMiB;
    if (limits.max_allocation_size)
        by_allocation = limits.max_allocation_size > allocation_overhead
                            ? (limits.max_allocation_size - allocation_overhead) / kArenaMiB * kArenaMiB
                            : 0;
    const VkDeviceSize cap = std::min(by_buffer, by_allocation);
    if (cap >= size.requested) return size;
    size.bytes = cap;
    if (by_buffer == cap) size.capped |= arena_cap_buffer;
    if (by_allocation == cap) size.capped |= arena_cap_allocation;
    return size;
}

// The Vulkan names of the limits in `capped`, for the log line.
inline const char* arena_cap_name(unsigned capped) {
    switch (capped) {
        case arena_cap_buffer: return "maxBufferSize";
        case arena_cap_allocation: return "maxMemoryAllocationSize";
        case arena_cap_buffer | arena_cap_allocation: return "maxBufferSize and maxMemoryAllocationSize";
        default: return "no limit";
    }
}

}  // namespace leonida
