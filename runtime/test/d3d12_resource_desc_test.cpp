// Compare ID3D12Resource::GetDesc on the Vulkan-backed device with Windows D3D12 (the layer's own adapter and WARP):
// what a description that leaves fields at their defaults (Alignment 0, MipLevels 0) reads back as.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};

struct Case {
    const char* name;
    D3D12_RESOURCE_DESC desc;
    D3D12_HEAP_TYPE heap;  // of a committed resource
    D3D12_RESOURCE_STATES state;
    bool reserved;
};

D3D12_RESOURCE_DESC buffer(UINT64 bytes, UINT64 alignment = 0, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Alignment = alignment;
    desc.Width = bytes;
    desc.Height = desc.SampleDesc.Count = 1;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC texture(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth_or_layers,
                            UINT16 mips, DXGI_FORMAT format, UINT samples = 1,
                            D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE, UINT64 alignment = 0) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dimension;
    desc.Alignment = alignment;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth_or_layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = samples;
    desc.Flags = flags;
    return desc;
}

HRESULT create(ID3D12Device* device, const Case& c, ID3D12Resource** out) {
    if (c.reserved) return device->CreateReservedResource(&c.desc, c.state, nullptr, IID_PPV_ARGS(out));
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = c.heap;
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &c.desc, c.state, nullptr, IID_PPV_ARGS(out));
}

bool same(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b) {
    return a.Dimension == b.Dimension && a.Alignment == b.Alignment && a.Width == b.Width && a.Height == b.Height &&
           a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels && a.Format == b.Format &&
           a.SampleDesc.Count == b.SampleDesc.Count && a.SampleDesc.Quality == b.SampleDesc.Quality &&
           a.Layout == b.Layout && a.Flags == b.Flags;
}

void print(const char* who, const D3D12_RESOURCE_DESC& d) {
    std::fprintf(stderr,
                 "  %-6s dimension %d alignment %llu %llux%ux%u mips %u format %d samples %u/%u layout %d flags 0x%x\n",
                 who, int(d.Dimension), static_cast<unsigned long long>(d.Alignment),
                 static_cast<unsigned long long>(d.Width), d.Height, unsigned(d.DepthOrArraySize), unsigned(d.MipLevels),
                 int(d.Format), d.SampleDesc.Count, d.SampleDesc.Quality, int(d.Layout), unsigned(d.Flags));
}

struct Totals {
    unsigned matched = 0, mismatched = 0, host_refused = 0, ours_refused = 0;
};

// Every case the reference creates must read back the same from the layer. A case the reference refuses says
// nothing about GetDesc and is skipped; one only the layer refuses is counted (dimensions it does not create yet).
Totals compare(ID3D12Device* ours, ID3D12Device* host, const char* host_name, const std::vector<Case>& cases) {
    Totals totals;
    for (const Case& c : cases) {
        Ref<ID3D12Resource> expected, actual;
        if (FAILED(create(host, c, &expected.p)) || !expected.p) {
            ++totals.host_refused;
            continue;
        }
        const HRESULT hr = create(ours, c, &actual.p);
        if (FAILED(hr) || !actual.p) {
            std::fprintf(stderr, "%s: not created by the layer (0x%08lx), not compared with %s\n", c.name,
                         static_cast<unsigned long>(hr), host_name);
            ++totals.ours_refused;
            continue;
        }
        const D3D12_RESOURCE_DESC a = actual->GetDesc(), e = expected->GetDesc();
        if (same(a, e)) {
            ++totals.matched;
            continue;
        }
        ++totals.mismatched;
        std::fprintf(stderr, "%s: GetDesc differs from %s\n", c.name, host_name);
        print("asked", c.desc);
        print("ours", a);
        print("host", e);
    }
    return totals;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;   // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned matched = 0, mismatched = 0, references = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;

        constexpr auto t1 = D3D12_RESOURCE_DIMENSION_TEXTURE1D, t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                       t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        constexpr auto rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
        constexpr auto none = D3D12_RESOURCE_FLAG_NONE, target = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                       depth = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
        constexpr auto common = D3D12_RESOURCE_STATE_COMMON, render = D3D12_RESOURCE_STATE_RENDER_TARGET;
        constexpr auto default_heap = D3D12_HEAP_TYPE_DEFAULT;
        constexpr UINT64 k4 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT, k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT,
                         m4 = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
        D3D12_RESOURCE_DESC tiled = texture(t2, 1024, 1024, 1, 0, rgba8);
        tiled.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        const std::vector<Case> cases = {
            // Buffers: Alignment 0 reads back as 64 KiB on every heap type.
            {"buffer, default heap", buffer(100), default_heap, common, false},
            {"buffer, upload heap", buffer(100), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ, false},
            {"buffer, readback heap", buffer(100), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST, false},
            {"buffer, 64 KiB asked", buffer(100, k64), default_heap, common, false},
            {"buffer, unordered access", buffer(70000, 0, uav), default_heap, common, false},
            // Single-sampled textures: 64 KiB unless the description asks for 4 KiB; MipLevels 0 is the full chain.
            {"2D 64x64", texture(t2, 64, 64, 1, 1, rgba8), default_heap, common, false},
            {"2D 64x64, all mips", texture(t2, 64, 64, 1, 0, rgba8), default_heap, common, false},
            {"2D 100x60, all mips", texture(t2, 100, 60, 1, 0, rgba8), default_heap, common, false},
            {"2D 64x64, 4 KiB asked", texture(t2, 64, 64, 1, 1, rgba8, 1, none, k4), default_heap, common, false},
            {"2D 64x64, 64 KiB asked", texture(t2, 64, 64, 1, 1, rgba8, 1, none, k64), default_heap, common, false},
            {"2D 1024x1024, all mips", texture(t2, 1024, 1024, 1, 0, rgba8), default_heap, common, false},
            {"2D array of 6, all mips", texture(t2, 256, 256, 6, 0, rgba8), default_heap, common, false},
            {"2D typeless", texture(t2, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_TYPELESS), default_heap, common, false},
            {"2D render target", texture(t2, 256, 256, 1, 1, rgba8, 1, target), default_heap, render, false},
            {"2D render target, all mips", texture(t2, 256, 256, 1, 0, rgba8, 1, target), default_heap, render, false},
            {"2D depth", texture(t2, 256, 256, 1, 1, DXGI_FORMAT_D32_FLOAT, 1, depth), default_heap,
             D3D12_RESOURCE_STATE_DEPTH_WRITE, false},
            {"2D unordered access", texture(t2, 256, 256, 1, 1, rgba8, 1, uav), default_heap, common, false},
            {"3D 32x32x32, all mips", texture(t3, 32, 32, 32, 0, rgba8), default_heap, common, false},
            {"3D 64x16x4, all mips", texture(t3, 64, 16, 4, 0, rgba8), default_heap, common, false},
            {"1D 128, all mips", texture(t1, 128, 1, 1, 0, rgba8), default_heap, common, false},
            // Multisampled textures: 4 MiB unless the description asks for 64 KiB.
            {"2D 4x render target", texture(t2, 256, 256, 1, 1, rgba8, 4, target), default_heap, render, false},
            {"2D 4x render target, 64 KiB asked", texture(t2, 64, 64, 1, 1, rgba8, 4, target, k64), default_heap,
             render, false},
            {"2D 4x render target, 4 MiB asked", texture(t2, 64, 64, 1, 1, rgba8, 4, target, m4), default_heap, render,
             false},
            {"2D 4x depth", texture(t2, 256, 256, 1, 1, DXGI_FORMAT_D32_FLOAT, 4, depth), default_heap,
             D3D12_RESOURCE_STATE_DEPTH_WRITE, false},
            // Reserved resources report their own layout and the same alignment.
            {"reserved 2D 1024x1024, all mips", tiled, default_heap, common, true},
            {"reserved buffer", buffer(1 << 20), default_heap, common, true},
        };

        if (hostwine::host_is_wine()) return 77;  // every case is a comparison with the host
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        for (const bool warp : {false, true}) {
            Ref<IDXGIAdapter> adapter;
            Ref<ID3D12Device> host;
            std::string adapter_name;
            if (!hostadapter::reference(factory.p, vk, warp, &adapter.p, adapter_name)) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
            const char* name = adapter_name.c_str();
            const Totals totals = compare(ours.p, host.p, name, cases);
            std::printf("resource desc against %s: %u of %zu descriptions match, %u differ, %u refused by the host, "
                        "%u not created by the layer\n",
                        name, totals.matched, cases.size(), totals.mismatched, totals.host_refused, totals.ours_refused);
            matched += totals.matched;
            mismatched += totals.mismatched;
            ++references;
        }
    }
    if (!references) {
        std::fprintf(stderr, "no D3D12 reference device (hardware adapter or WARP)\n");
        return 1;
    }
    if (mismatched || !matched) return 1;
    if (vk.validation_errors()) return 1;
    return 0;
}
