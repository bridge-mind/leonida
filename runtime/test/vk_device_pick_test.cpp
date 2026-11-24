// CPU-only tests for the physical-device ranking the Vulkan harnesses use (#915). No Vulkan instance or GPU is
// needed: the rank is a pure function of the device type and vendor.
#include "vk_device_pick.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int failures = 0;

VkPhysicalDeviceProperties device(VkPhysicalDeviceType type, std::uint32_t vendor, const char* name) {
    VkPhysicalDeviceProperties properties{};
    properties.deviceType = type;
    properties.vendorID = vendor;
    std::snprintf(properties.deviceName, sizeof properties.deviceName, "%s", name);
    return properties;
}

void expect_index(const char* what, std::size_t got, std::size_t want) {
    if (got == want) return;
    std::fprintf(stderr, "FAIL %s: picked %zu, expected %zu\n", what, got, want);
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

constexpr std::uint32_t nvidia = 0x10de;
constexpr std::uint32_t amd = 0x1002;
constexpr std::uint32_t intel = 0x8086;
constexpr std::uint32_t microsoft = leonida::vk_test::microsoft_vendor_id;

}  // namespace

int main() {
    using leonida::vk_test::pick_device;

    const char* saved = std::getenv("LEONIDA_VK_DEVICE");
    const std::string saved_value = saved ? saved : "";
    set_filter("");

    // The enumeration from #915: native NVIDIA first, Dozen's wrapper second, Dozen's Basic Render Driver third. The
    // old loop overwrote on every discrete device, so it selected index 1.
    std::vector<VkPhysicalDeviceProperties> devices{
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, nvidia, "NVIDIA GeForce RTX 4060 Ti"),
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, microsoft, "Microsoft Direct3D12 (NVIDIA GeForce RTX 4060 Ti)"),
        device(VK_PHYSICAL_DEVICE_TYPE_CPU, microsoft, "Microsoft Direct3D12 (Microsoft Basic Render Driver)"),
    };
    expect_index("native discrete before Dozen", pick_device(devices), 0);

    // Order must not matter: Dozen first, native GPU second.
    std::swap(devices[0], devices[1]);
    expect_index("native discrete after Dozen", pick_device(devices), 1);

    // A native integrated GPU is still a real driver and is preferred over a Dozen wrapper on a discrete card.
    devices = {
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, microsoft, "Microsoft Direct3D12 (NVIDIA GeForce RTX 4060 Ti)"),
        device(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, amd, "AMD Radeon 780M"),
    };
    expect_index("native integrated beats Dozen discrete", pick_device(devices), 1);

    // Among native devices, discrete wins over integrated and virtual.
    devices = {
        device(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, intel, "Intel UHD 770"),
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, amd, "AMD Radeon RX 7900 XTX"),
        device(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU, amd, "virtual"),
    };
    expect_index("native discrete wins", pick_device(devices), 1);

    // When only translator devices are present, classification still breaks the tie, and the first device wins an
    // exact tie.
    devices = {
        device(VK_PHYSICAL_DEVICE_TYPE_CPU, microsoft, "Microsoft Direct3D12 (Microsoft Basic Render Driver)"),
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, microsoft, "Microsoft Direct3D12 (other)"),
    };
    expect_index("Dozen discrete beats Dozen CPU", pick_device(devices), 1);

    devices = {
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, amd, "first"),
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, amd, "second"),
    };
    expect_index("tie keeps the first", pick_device(devices), 0);

    devices.clear();
    expect_index("empty list", pick_device(devices), devices.size());

    // LEONIDA_VK_DEVICE narrows the candidates before ranking, so a match wins even when its type or vendor would
    // otherwise lose, and a filter that matches nothing selects no device instead of falling back.
    devices = {
        device(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, microsoft, "Microsoft Direct3D12 (NVIDIA GeForce RTX 4060 Ti)"),
        device(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, amd, "AMD Radeon 780M"),
    };
    set_filter("780m");
    expect_index("filter picks the integrated AMD", pick_device(devices), 1);
    set_filter("microsoft");
    expect_index("filter picks the Dozen wrapper despite its rank", pick_device(devices), 0);
    set_filter("no-such-gpu");
    expect_index("filter matches nothing", pick_device(devices), devices.size());
    set_filter("");

    if (failures) {
        std::fprintf(stderr, "%d device-pick test(s) failed\n", failures);
        return 1;
    }
    set_filter(saved_value.c_str());

    std::printf("device pick: all cases passed\n");
    return 0;
}
