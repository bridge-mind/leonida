#pragma once

// LEONIDA_VK_DEVICE: which Vulkan GPU the runtime and its tests take on a machine with more than one. The value is
// part of the device name, compared without case ("5070", "radeon"); unset or empty takes every device, and the
// caller then picks by vulkan_device_rank (below) among the devices it can use. Several GPUs of one vendor share
// one driver, so the loader's own driver filters cannot tell them apart. A value that matches no device must fail
// the caller, not fall back, so a typo cannot silently test another GPU (as LEONIDA_HOST_ADAPTER does).

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>

namespace leonida {

// The filter, lower-cased; empty when unset.
inline std::string vulkan_device_filter() {
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4996)  // getenv: read at device selection only
#endif
    const char* value = std::getenv("LEONIDA_VK_DEVICE");
#ifdef _MSC_VER
#pragma warning(pop)
#endif
    std::string out = value ? value : "";
    for (char& c : out) c = char(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// True when `device_name` (VkPhysicalDeviceProperties::deviceName) is wanted.
inline bool vulkan_device_wanted(const char* device_name) {
    const std::string filter = vulkan_device_filter();
    if (filter.empty()) return true;
    std::string name = device_name ? device_name : "";
    for (char& c : name) c = char(std::tolower(static_cast<unsigned char>(c)));
    return name.find(filter) != std::string::npos;
}

// Microsoft is the vendor of the D3D12 translation layers Windows adds next to the real GPU (Mesa's Dozen from the
// "OpenCL, OpenGL and Vulkan Compatibility Pack", and the Basic Render Driver); no native GPU driver uses it. Dozen
// reports a discrete GPU, so "the last discrete device wins" took the wrapper over the native driver (#915).
inline constexpr std::uint32_t microsoft_vulkan_vendor_id = 0x1414;

// How much a device is wanted, higher first: a native driver over Microsoft's layers, then discrete > integrated >
// virtual > CPU > other. `device_type` is VkPhysicalDeviceType (OTHER 0, INTEGRATED 1, DISCRETE 2, VIRTUAL 3, CPU 4;
// as an int so this header needs no Vulkan headers). The runtime and every test harness pick the first device with
// the highest rank among those they can use, so on a PC with two usable GPUs they take the same one.
inline int vulkan_device_rank(int device_type, std::uint32_t vendor_id) {
    int type_rank = 0;
    switch (device_type) {
        case 2: type_rank = 4; break;  // VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        case 1: type_rank = 3; break;  // VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU
        case 3: type_rank = 2; break;  // VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU
        case 4: type_rank = 1; break;  // VK_PHYSICAL_DEVICE_TYPE_CPU
        default: type_rank = 0; break;
    }
    return (vendor_id == microsoft_vulkan_vendor_id ? 0 : 10) + type_rank;
}

} // namespace leonida
