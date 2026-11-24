// Which adapter a test's host D3D12 reference devices run on, for the tests that create those devices themselves and
// compare the layer with a hardware adapter and with WARP. copycmp::create_host (d3d12_copy_compare.h) answers the
// same question for the tests that go through it. It reads LEONIDA_HOST_ADAPTER and matches the name through wanted()
// and named() below, so the two cannot come to read the variable differently. The rule:
//   - LEONIDA_HOST_ADAPTER names the hardware side by part of an adapter's name, without case. A name that matches no
//     adapter, or selects one without a D3D12 device, ends the run with an error: a typo must not turn the comparisons
//     into a silent pass. "warp" leaves the hardware side out, since these tests always compare with WARP as well.
//   - Unset, the hardware side is the adapter behind the layer's Vulkan device (VkPhysicalDeviceIDProperties::
//     deviceLUID). On a PC with two GPUs D3D12's default adapter can be the other one, and the comparison is then
//     between the layer on one GPU and the host driver of the other. Measured with the layer on a Ryzen 7 9700X iGPU
//     (AMD 32.0.21042.62) while an RTX 3070 (NVIDIA 610.74) was the default adapter: d3d12_depth_resolve set the four
//     "D24S8 stencil as X24_G8" resolves aside as host quirks, which are NVIDIA's, and compared the four D24S8 depth
//     MIN/MAX resolves, which the Radeon's own host driver does not resolve to the clears either.
//   - The default adapter, as before, when the Vulkan device has no LUID or that adapter gives no D3D12 device.
#pragma once

#include "vk_test_device.h"

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_4.h>

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace hostadapter {

inline std::string description(IDXGIAdapter* adapter) {
    DXGI_ADAPTER_DESC desc{};
    std::string text;
    if (adapter && SUCCEEDED(adapter->GetDesc(&desc)))
        for (const wchar_t* c = desc.Description; *c; ++c) text += char(*c < 128 ? *c : '?');
    return text;
}

// LEONIDA_HOST_ADAPTER in lower case; empty when it is unset or empty.
inline std::string wanted() {
    std::string needle(128, '\0');
    for (;;) {
        const DWORD length = GetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", needle.data(),
                                                      static_cast<DWORD>(needle.size()));
        if (length < needle.size()) {
            needle.resize(length);
            break;
        }
        // A short buffer's contents are undefined; retry with the required size (which includes the terminator).
        // A long explicit name must not become automatic selection. Retry again if the value grows meanwhile.
        needle.resize(length);
    }
    for (char& c : needle) c = char(std::tolower(static_cast<unsigned char>(c)));
    return needle;
}

// The first adapter whose name contains `needle` (lower case), with a reference the caller releases; null when none
// does.
inline IDXGIAdapter1* named(IDXGIFactory1* factory, const std::string& needle) {
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* candidate = nullptr;
        if (FAILED(factory->EnumAdapters1(i, &candidate))) return nullptr;
        std::string lower = description(candidate);
        for (char& c : lower) c = char(std::tolower(static_cast<unsigned char>(c)));
        if (lower.find(needle) != std::string::npos) return candidate;
        candidate->Release();
    }
}

// The adapter for one pass of a test's `for (const bool warp : {false, true})` loop, with a reference the caller
// releases (null: the default adapter), and the name the test prints for it. False when the pass has no adapter: no
// WARP on this PC, or LEONIDA_HOST_ADAPTER=warp on the hardware pass. `luid` is the layer's adapter (null: unknown),
// for a test whose layer device is the standalone one (ID3D12Device::GetAdapterLuid) rather than a vktest::Device.
inline bool reference(IDXGIFactory4* factory, const LUID* luid, bool warp, IDXGIAdapter** adapter, std::string& name) {
    *adapter = nullptr;
    if (warp) {
        name = "WARP";
        return SUCCEEDED(factory->EnumWarpAdapter(IID_PPV_ARGS(adapter)));
    }
    const std::string needle = wanted();
    if (needle == "warp") return false;
    if (!needle.empty()) {
        if (IDXGIAdapter1* found = named(factory, needle)) {
            if (SUCCEEDED(D3D12CreateDevice(found, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
                name = description(found);
                *adapter = found;
                return true;
            }
            found->Release();
        }
        std::fprintf(stderr, "LEONIDA_HOST_ADAPTER=%s: no such adapter, or no D3D12 device on it\n", needle.c_str());
        std::exit(1);
    }
    // With a null device pointer D3D12CreateDevice only says whether the adapter would give one.
    if (luid && SUCCEEDED(factory->EnumAdapterByLuid(*luid, IID_PPV_ARGS(adapter))) &&
        SUCCEEDED(D3D12CreateDevice(*adapter, D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), nullptr))) {
        name = description(*adapter);
        return true;
    }
    if (*adapter) (*adapter)->Release();
    *adapter = nullptr;
    name = "the default adapter";
    return true;
}

// The same, with the layer's adapter read from its Vulkan device (VkPhysicalDeviceIDProperties::deviceLUID).
inline bool reference(IDXGIFactory4* factory, const vktest::Device& vk, bool warp, IDXGIAdapter** adapter,
                      std::string& name) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    if (vk.physical)
        vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &properties);
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(id.deviceLUID));
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    return reference(factory, id.deviceLUIDValid ? &luid : nullptr, warp, adapter, name);
}

}  // namespace hostadapter
