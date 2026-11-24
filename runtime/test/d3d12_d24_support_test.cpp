// CheckFeatureSupport for the D24 family must describe what CreateCommittedResource does.
//
// A GPU without Vulkan D24_UNORM_S8_UINT (AMD) gets D32_SFLOAT_S8_UINT behind a D24 texture (#737). The layer's
// FORMAT_SUPPORT asked the D24 format itself and reported Support1 = 0 for D24_UNORM_S8_UINT and R24G8_TYPELESS
// on an RX 9070 XT, and MULTISAMPLE_QUALITY_LEVELS reported 0 levels for R24G8_TYPELESS at every count, while
// creating the depth-stencil resources succeeded (found by @zneel, #1380). This checks the layer against itself on
// whatever GPU runs it, native D24 or not:
//   * a 1x ALLOW_DEPTH_STENCIL texture is creatable exactly when FORMAT_SUPPORT reports DEPTH_STENCIL (and then
//     TEXTURE2D);
//   * at 2x, 4x and 8x a quality level is reported exactly when a multisampled depth-stencil texture is creatable.
//
// Synthetic resources only; no game files.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"

#include <cstdio>

namespace {

int g_failures = 0;

void check(bool ok, const char* what, DXGI_FORMAT format, UINT samples = 1) {
    std::printf("%s %s (format %d, %ux)\n", ok ? "PASS" : "FAIL", what, static_cast<int>(format), samples);
    if (!ok) ++g_failures;
}

bool creatable(ID3D12Device* device, DXGI_FORMAT format, UINT samples) {
    D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = 64;
    desc.Height = 64;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = samples;
    desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
    D3D12_CLEAR_VALUE clear{};
    clear.Format = format == DXGI_FORMAT_R24G8_TYPELESS ? DXGI_FORMAT_D24_UNORM_S8_UINT : format;
    clear.DepthStencil.Depth = 1.0f;
    ID3D12Resource* resource = nullptr;
    const HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_DEPTH_WRITE,
                                                       &clear, IID_PPV_ARGS(&resource));
    if (resource) resource->Release();
    return SUCCEEDED(hr);
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status != vktest::DeviceStatus::ready) return 1;

    VkFormatProperties d24{};
    d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
        d.physical, VK_FORMAT_D24_UNORM_S8_UINT, &d24);
    const bool native_d24 = (d24.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0;
    std::printf("Vulkan D24_UNORM_S8_UINT depth attachment: %s\n",
                native_d24 ? "native" : "no (D24 is stored as D32_SFLOAT_S8_UINT)");

    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = d.instance_proc;
    info.instance = d.instance;
    info.physical = d.physical;
    info.device = d.device;
    info.queue = d.queue;
    info.queue_family = d.family;
    info.bindless = d.bindless;
    info.bindless_partially_bound = d.partially_bound;
    info.image_view_min_lod = d.image_view_min_lod;
    info.host_image_copy = d.host_image_copy;
    ID3D12Device* device = nullptr;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device))) || !device) {
        std::printf("FAIL Vulkan-backed D3D12 device created\n");
        return 1;
    }

    for (const DXGI_FORMAT format : {DXGI_FORMAT_D24_UNORM_S8_UINT, DXGI_FORMAT_R24G8_TYPELESS}) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{format};
        const HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs));
        const bool reported_ds = SUCCEEDED(hr) && (fs.Support1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL);
        std::printf("format %d: FORMAT_SUPPORT hr=0x%08lx Support1=0x%08x Support2=0x%08x\n", static_cast<int>(format),
                    static_cast<unsigned long>(hr), fs.Support1, fs.Support2);
        // A TYPELESS member can be created with ALLOW_DEPTH_STENCIL, but D3D12 reports DEPTH_STENCIL only for its
        // typed depth format: on an RTX 4060 Ti and WARP, R24G8_TYPELESS gives Support1 0x001010b0 and
        // D24_UNORM_S8_UINT 0x003110b0 (the DEPTH_STENCIL bit is 0x00010000).
        const bool typeless = format == DXGI_FORMAT_R24G8_TYPELESS;
        const bool made = creatable(device, format, 1);
        check(made, "a 1x ALLOW_DEPTH_STENCIL texture is created", format);
        check(reported_ds == (made && !typeless),
              "FORMAT_SUPPORT reports DEPTH_STENCIL exactly when creation succeeds, for the typed format only", format);
        if (made) check((fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) != 0, "FORMAT_SUPPORT reports TEXTURE2D", format);

        for (const UINT samples : {2u, 4u, 8u}) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels{format, samples};
            device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels, sizeof(levels));
            const bool ms_made = creatable(device, format, samples);
            std::printf("format %d %ux: quality levels %u, created %s\n", static_cast<int>(format), samples,
                        levels.NumQualityLevels, ms_made ? "yes" : "no");
            check((levels.NumQualityLevels > 0) == ms_made,
                  "a quality level is reported exactly when the multisampled depth-stencil texture is created",
                  format, samples);
        }
    }
    device->Release();

    if (g_failures) {
        std::printf("d3d12_d24_support_test: %d checks failed\n", g_failures);
        return 1;
    }
    std::printf("d3d12_d24_support_test: OK\n");
    return 0;
}
