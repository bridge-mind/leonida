// Chooses among enumerated Vulkan physical devices for the test harnesses.
//
// Windows can expose Mesa's Dozen (a Vulkan-on-D3D12 translation layer installed by the "OpenCL, OpenGL and Vulkan
// Compatibility Pack") next to the real GPU. Dozen reports VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, so the old
// "the last discrete device wins" loop picked the wrapper over the native driver and produced failures that looked
// like translator bugs (#915). Prefer a native ICD, then discrete > integrated > virtual > CPU > other, and keep the
// first device on a tie.
#pragma once

#include <vulkan/vulkan.h>

#include "../src/vulkan_device_filter.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace leonida::vk_test {

// Microsoft is the vendor of the D3D12 translation layers Windows adds (Dozen and the Basic Render Driver); no
// native GPU driver uses it. Checking the vendor avoids depending on VkPhysicalDeviceDriverProperties.
inline constexpr std::uint32_t microsoft_vendor_id = leonida::microsoft_vulkan_vendor_id;

static_assert(VK_PHYSICAL_DEVICE_TYPE_OTHER == 0 && VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU == 1 &&
                  VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU == 2 && VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU == 3 &&
                  VK_PHYSICAL_DEVICE_TYPE_CPU == 4,
              "vulkan_device_rank takes VkPhysicalDeviceType by value");

// The runtime's rank (runtime/src/vulkan_device_filter.h), so the harnesses and the runtime take the same GPU.
inline int device_rank(VkPhysicalDeviceType type, std::uint32_t vendor_id) {
    return leonida::vulkan_device_rank(static_cast<int>(type), vendor_id);
}

// Index of the best candidate, or candidates.size() when there are none. Ties keep the earlier device.
inline std::size_t pick_device(const std::vector<VkPhysicalDeviceProperties>& candidates) {
    std::size_t best = candidates.size();
    int best_rank = -1;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (!leonida::vulkan_device_wanted(candidates[i].deviceName)) continue;
        const int rank = device_rank(candidates[i].deviceType, candidates[i].vendorID);
        if (rank > best_rank) {
            best = i;
            best_rank = rank;
        }
    }
    return best;
}

// Why no device was chosen, naming LEONIDA_VK_DEVICE when its filter excluded every device (otherwise the list was
// empty). The tools call this instead of dereferencing a null device.
inline void report_no_device() {
    const std::string filter = leonida::vulkan_device_filter();
    if (!filter.empty()) {
        std::fprintf(stderr, "LEONIDA_VK_DEVICE=%s: no Vulkan physical device matches\n", filter.c_str());
    } else {
        std::fprintf(stderr, "no Vulkan physical device found\n");
    }
}

}  // namespace leonida::vk_test
