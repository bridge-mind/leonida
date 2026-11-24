// Unit tests for LEONIDA_VK_DEVICE name matching and the device rank (runtime/src/vulkan_device_filter.h). No Vulkan
// instance and no GPU: the filter is a pure function of the environment and a device name, the rank of type and vendor.
#include "vulkan_device_filter.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

int failures = 0;

void expect(bool ok, const char* what) {
    if (ok) return;
    std::fprintf(stderr, "FAIL %s\n", what);
    ++failures;
}

void set_filter(const char* value) {
#ifdef _WIN32
    _putenv_s("LEONIDA_VK_DEVICE", value);
#else
    if (value[0] == '\0') {
        unsetenv("LEONIDA_VK_DEVICE");
    } else {
        setenv("LEONIDA_VK_DEVICE", value, 1);
    }
#endif
}

}  // namespace

int main() {
    using leonida::vulkan_device_filter;
    using leonida::vulkan_device_wanted;

    const char* saved = std::getenv("LEONIDA_VK_DEVICE");
    const std::string saved_value = saved ? saved : "";

    // Unset (or empty) takes every device, including a null name.
    set_filter("");
    expect(vulkan_device_filter().empty(), "empty filter");
    expect(vulkan_device_wanted("AMD Radeon RX 7900 XTX"), "unset matches a real name");
    expect(vulkan_device_wanted(""), "unset matches an empty name");
    expect(vulkan_device_wanted(nullptr), "unset matches a null name");

    // A filter is a case-insensitive substring of the device name.
    set_filter("RADEON");
    expect(vulkan_device_filter() == "radeon", "filter is lower-cased");
    expect(vulkan_device_wanted("AMD Radeon RX 7900 XTX"), "case-insensitive match");
    expect(vulkan_device_wanted("amd radeon graphics"), "lower-case name matches");
    expect(!vulkan_device_wanted("NVIDIA GeForce RTX 5070 Ti"), "different vendor does not match");

    set_filter("rtx 5070");
    expect(vulkan_device_wanted("NVIDIA GeForce RTX 5070 Ti"), "substring with a space matches");
    expect(!vulkan_device_wanted("NVIDIA GeForce RTX 4070 Ti"), "other model does not match");

    set_filter("7900");
    expect(vulkan_device_wanted("AMD Radeon RX 7900 XTX"), "model number matches");
    expect(!vulkan_device_wanted(nullptr), "a filter never matches a null name");
    expect(!vulkan_device_wanted(""), "a filter never matches an empty name");

    // The rank the runtime and every harness pick by (VkPhysicalDeviceType values: OTHER 0, INTEGRATED 1, DISCRETE 2,
    // VIRTUAL 3, CPU 4). A native driver beats Microsoft's layers of any type, so Dozen ("Microsoft Direct3D12 (...)",
    // reported as discrete) never wins over the GPU it wraps, nor over a native integrated GPU (#915).
    using leonida::vulkan_device_rank;
    constexpr std::uint32_t amd = 0x1002, nvidia = 0x10de, microsoft = leonida::microsoft_vulkan_vendor_id;
    expect(microsoft == 0x1414, "Microsoft's PCI vendor id");
    expect(vulkan_device_rank(2, amd) > vulkan_device_rank(1, amd), "discrete over integrated");
    expect(vulkan_device_rank(1, amd) > vulkan_device_rank(3, amd), "integrated over virtual");
    expect(vulkan_device_rank(3, amd) > vulkan_device_rank(4, amd), "virtual over CPU");
    expect(vulkan_device_rank(4, amd) > vulkan_device_rank(0, amd), "CPU over other");
    expect(vulkan_device_rank(0, amd) > vulkan_device_rank(2, microsoft), "any native device over Dozen");
    expect(vulkan_device_rank(2, microsoft) > vulkan_device_rank(4, microsoft), "among Microsoft's, discrete first");
    expect(vulkan_device_rank(2, amd) == vulkan_device_rank(2, nvidia), "vendors other than Microsoft tie");
    expect(vulkan_device_rank(99, amd) == vulkan_device_rank(0, amd), "an unknown type ranks as other");

    // Do not leak the test's value into anything that runs later in this process.
#ifdef _WIN32
    _putenv_s("LEONIDA_VK_DEVICE", saved_value.c_str());
#else
    if (saved_value.empty()) {
        unsetenv("LEONIDA_VK_DEVICE");
    } else {
        setenv("LEONIDA_VK_DEVICE", saved_value.c_str(), 1);
    }
#endif

    if (failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("vulkan_device_filter: all checks passed\n");
    return 0;
}
