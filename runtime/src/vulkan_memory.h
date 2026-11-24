#pragma once

// Memory-type policy for the D3D12 layer that depends only on VkPhysicalDeviceMemoryProperties, so it can be checked
// without a GPU (d3d12_state_map_test) on the heap layouts real cards report.

#ifndef VK_NO_PROTOTYPES
#define VK_NO_PROTOTYPES
#endif
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace leonida {

// Without Resizable BAR the CPU sees at most a 256 MiB window of VRAM: the DEVICE_LOCAL | HOST_VISIBLE type sits in a
// heap of that size or smaller (256 MiB on AMD and on RADV, 214 MiB on an RTX 3090 with driver 616.56). Anything
// larger is Resizable BAR (or Smart Access Memory), or a GPU whose memory is all device-local and host-visible.
inline constexpr VkDeviceSize kBarWindowBytes = VkDeviceSize(256) << 20;

// True when the device has DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT memory in a device-local heap larger than
// the 256 MiB BAR window (D3D-96): then CPU-written memory that the GPU reads, such as an upload heap, can live in
// VRAM without exhausting a small window that every such allocation would share.
inline bool resizable_bar(const VkPhysicalDeviceMemoryProperties& memory) {
    constexpr VkMemoryPropertyFlags wanted = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (std::uint32_t i = 0; i < memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
        const VkMemoryType& type = memory.memoryTypes[i];
        if ((type.propertyFlags & wanted) != wanted || type.heapIndex >= memory.memoryHeapCount) continue;
        const VkMemoryHeap& heap = memory.memoryHeaps[type.heapIndex];
        if ((heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) && heap.size > kBarWindowBytes) return true;
    }
    return false;
}

// The memory types to try for an allocation of `size` bytes, best first: those with `required | preferred`, then
// those with only `required`. A type whose heap is smaller than the allocation is left out, as the allocation would
// break VUID-vkAllocateMemory-pAllocateInfo-01713 (some drivers still return VK_SUCCESS). Without Resizable BAR the
// heap of DEVICE_LOCAL | HOST_VISIBLE memory is typically 256 MiB or less, so a larger host-visible allocation falls
// through to system memory. A heap that is big enough but already partly used is left to the driver: a failed
// vkAllocateMemory moves on to the next type.
inline std::vector<std::uint32_t> memory_type_order(const VkPhysicalDeviceMemoryProperties& memory,
                                                    std::uint32_t type_bits, VkMemoryPropertyFlags required,
                                                    VkMemoryPropertyFlags preferred, VkDeviceSize size) {
    std::vector<std::uint32_t> order;
    for (const VkMemoryPropertyFlags flags : {required | preferred, required}) {
        for (std::uint32_t i = 0; i < memory.memoryTypeCount && i < VK_MAX_MEMORY_TYPES; ++i) {
            const VkMemoryType& type = memory.memoryTypes[i];
            if (!(type_bits & (1u << i)) || (type.propertyFlags & flags) != flags) continue;
            if (type.heapIndex >= memory.memoryHeapCount || memory.memoryHeaps[type.heapIndex].size < size) continue;
            if (std::find(order.begin(), order.end(), i) == order.end()) order.push_back(i);
        }
    }
    return order;
}

// D3D-299: true when an allocation that preferred DEVICE_LOCAL without needing host access (a texture, a default-heap
// buffer outside the arena) got a type without it, so the GPU now reads it from system memory. A host-visible
// allocation (the arena, an upload heap) in system memory is the memory type policy without Resizable BAR, not this.
inline bool spilled_to_host(VkMemoryPropertyFlags required, VkMemoryPropertyFlags preferred,
                            VkMemoryPropertyFlags chosen) {
    return (preferred & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(required & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
           !(chosen & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

}  // namespace leonida
