// ArraySize = -1 (UINT_MAX) in a render-target or depth-stencil view of a texture array: every slice from
// FirstArraySlice on, as D3D12 documents and the host measures. Each case seeds every subresource of a four-slice
// texture with zeros through one-slice views, clears to one through a single view with ArraySize -1, reads every
// subresource back and names the ones the clear reached: TEXTURE2DARRAY views of a mip chain (typed, typeless and
// depth) and TEXTURE2DMSARRAY views of 4x textures (read through a resolve). The layer must reach exactly the
// slices from the first one on, in the view's mip only, and so must the host's D3D12 on the layer's adapter and on
// WARP (skipped under Wine; WARP is not given the depth resolve, which removes its device, d3d12_depth_resolve_test).
// The 4x depth array is also cleared through views that count their slices (slice 0 alone, slice 1 alone, slices
// 1-3): its readback resolves each slice on its own, which is where D3D-417 showed (the layer's depth resolve of a
// slice above 0 read slice 0 on the AMD driver, whatever the view had cleared).
// A range the host's debug layer refuses stays refused by the layer: a first slice past the array (with -1 too), no
// slices, more slices than are left, and UINT_MAX - 1. Those are not run on the host.
// No game data or shaders. Needs a Vulkan GPU; LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "d3d12_copy_compare.h"

namespace {
using copycmp::Ptr;

constexpr UINT kSlices = 4, kSize = 8;

struct Case {
    const char* name;
    bool depth;                  // a depth-stencil view of a depth texture, else a render-target view
    DXGI_FORMAT resource, view;
    UINT samples;                // 4: TEXTURE2DMSARRAY views of a texture with one mip
};

// Seeds, clears through {mip, first, count} and reads back. `reached` gets one bit per subresource (mip + slice *
// mips) the clear wrote. Returns an error text when the list was refused or the readback is not all zeros or ones.
std::string clear_through(copycmp::Run& run, const Case& c, UINT mip, UINT first, UINT count, std::uint32_t& reached) {
    ID3D12Device* device = run.device;
    const bool msaa = c.samples > 1;
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = desc.Height = kSize;
    desc.DepthOrArraySize = kSlices;
    desc.MipLevels = msaa ? 1 : 3;
    desc.Format = c.resource;
    desc.SampleDesc.Count = c.samples;
    desc.Flags = c.depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const auto written = c.depth ? D3D12_RESOURCE_STATE_DEPTH_WRITE : D3D12_RESOURCE_STATE_RENDER_TARGET;
    D3D12_RESOURCE_DESC plain = desc;  // what a multisampled texture is resolved into, slice by slice
    plain.Format = c.view;
    plain.SampleDesc.Count = 1;
    if (!c.depth) plain.Flags = D3D12_RESOURCE_FLAG_NONE;
    const UINT n = UINT(desc.MipLevels) * kSlices;
    std::vector<copycmp::Placed> placed(n);
    std::vector<UINT> rows(n);
    std::vector<UINT64> row_bytes(n);
    UINT64 total = 0;
    device->GetCopyableFootprints(msaa ? &plain : &desc, 0, n, 0, placed.data(), rows.data(), row_bytes.data(), &total);

    Ptr<ID3D12Resource> texture, resolved, readback;
    Ptr<ID3D12DescriptorHeap> heap;
    Ptr<ID3D12GraphicsCommandList1> list1;
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{
        c.depth ? D3D12_DESCRIPTOR_HEAP_TYPE_DSV : D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1,
        D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    if (!run.texture(desc, written, texture) ||
        (msaa && !run.texture(plain, D3D12_RESOURCE_STATE_RESOLVE_DEST, resolved)) ||
        !run.buffer(D3D12_HEAP_TYPE_READBACK, total, D3D12_RESOURCE_STATE_COPY_DEST, readback) ||
        FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))) ||
        (msaa && c.depth && FAILED(run.list->QueryInterface(IID_PPV_ARGS(&list1.p)))))
        return "resource creation failed";
    const auto slot = heap->GetCPUDescriptorHandleForHeapStart();
    // A clear takes the descriptor as it is when the clear is recorded, so one slot serves every view.
    const auto clear = [&](UINT view_mip, UINT view_first, UINT view_count, FLOAT value) {
        if (c.depth) {
            D3D12_DEPTH_STENCIL_VIEW_DESC view{};
            view.Format = c.view;
            view.ViewDimension = msaa ? D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY : D3D12_DSV_DIMENSION_TEXTURE2DARRAY;
            if (msaa) view.Texture2DMSArray = {view_first, view_count};
            else view.Texture2DArray = {view_mip, view_first, view_count};
            device->CreateDepthStencilView(texture.p, &view, slot);
            run.list->ClearDepthStencilView(slot, D3D12_CLEAR_FLAG_DEPTH, value, 0, 0, nullptr);
        } else {
            D3D12_RENDER_TARGET_VIEW_DESC view{};
            view.Format = c.view;
            view.ViewDimension = msaa ? D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY : D3D12_RTV_DIMENSION_TEXTURE2DARRAY;
            if (msaa) view.Texture2DMSArray = {view_first, view_count};
            else view.Texture2DArray = {view_mip, view_first, view_count, 0};
            device->CreateRenderTargetView(texture.p, &view, slot);
            const FLOAT colour[4] = {value, value, value, value};
            run.list->ClearRenderTargetView(slot, colour, 0, nullptr);
        }
    };
    for (UINT slice = 0; slice < kSlices; ++slice)
        for (UINT level = 0; level < desc.MipLevels; ++level) clear(level, slice, 1, 0.0f);
    clear(mip, first, count, 1.0f);

    if (msaa) {
        run.barrier(texture.p, written, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
        for (UINT slice = 0; slice < kSlices; ++slice) {
            if (c.depth)
                list1->ResolveSubresourceRegion(resolved.p, slice, 0, 0, texture.p, slice, nullptr, c.view,
                                                D3D12_RESOLVE_MODE_MIN);
            else
                run.list->ResolveSubresource(resolved.p, slice, texture.p, slice, c.view);
        }
        run.barrier(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    } else {
        run.barrier(texture.p, written, D3D12_RESOURCE_STATE_COPY_SOURCE);
    }
    for (UINT i = 0; i < n; ++i) {
        const auto dst = copycmp::foot_loc(readback.p, placed[i]);
        const auto src = copycmp::sub_loc(msaa ? resolved.p : texture.p, i);
        run.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (std::string error = run.submit(); !error.empty()) return error;

    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped)) || !mapped) return "Map failed";
    const auto* bytes = static_cast<const std::uint8_t*>(mapped);
    reached = 0;
    std::string error;
    for (UINT i = 0; i < n && error.empty(); ++i) {
        // One, in every format here, has a byte that is not zero in each texel; zero has none.
        const size_t texel = size_t(row_bytes[i]) / placed[i].Footprint.Width;
        size_t ones = 0, texels = 0;
        for (UINT row = 0; row < rows[i]; ++row)
            for (UINT x = 0; x < placed[i].Footprint.Width; ++x, ++texels) {
                const std::uint8_t* at =
                    bytes + placed[i].Offset + size_t(row) * placed[i].Footprint.RowPitch + x * texel;
                ones += std::any_of(at, at + texel, [](std::uint8_t b) { return b != 0; });
            }
        if (ones == texels) reached |= 1u << i;
        else if (ones) error = "subresource " + std::to_string(i) + " is partly cleared";
    }
    readback->Unmap(0, nullptr);
    return error;
}

std::string slices_of(std::uint32_t reached, UINT mips) {
    std::string text;
    for (UINT i = 0; i < mips * kSlices; ++i)
        if (reached & (1u << i)) text += " " + std::to_string(i / mips) + "." + std::to_string(i % mips);
    return text.empty() ? " none" : text;  // slice.mip
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto status = vk.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (status != vktest::DeviceStatus::ready) return 1;
    Ptr<ID3D12Device> ours;
    rt::d3d12::VulkanDeviceInfo info{};
    info.get_instance_proc = vk.instance_proc;
    info.instance = vk.instance;
    info.physical = vk.physical;
    info.device = vk.device;
    info.queue = vk.queue;
    info.queue_family = vk.family;
    if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;

    // The references: the host's D3D12 on the layer's adapter (by LUID), and WARP.
    struct Host {
        std::string name;
        Ptr<ID3D12Device> device;
        bool warp = false;
    };
    Host hosts[2];
    if (!copycmp::host_is_wine()) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &properties);
        LUID luid{};
        std::memcpy(&luid, id.deviceLUID, sizeof(luid));
        if (!id.deviceLUIDValid || !copycmp::try_create_host(hosts[0].device, hosts[0].name, &luid))
            std::printf("no host D3D12 device on the layer's adapter: compared with WARP only\n");
        Ptr<IDXGIFactory4> factory;
        Ptr<IDXGIAdapter> adapter;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
            FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hosts[1].device.p)))) {
            std::fprintf(stderr, "WARP reference unavailable\n");
            return 1;
        }
        hosts[1].name = "WARP";
        hosts[1].warp = true;
    }

    constexpr DXGI_FORMAT rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM, depth32 = DXGI_FORMAT_D32_FLOAT;
    const Case cases[] = {
        {"RTV R8G8B8A8_UNORM", false, rgba8, rgba8, 1},
        {"RTV R8G8B8A8_UNORM of TYPELESS", false, DXGI_FORMAT_R8G8B8A8_TYPELESS, rgba8, 1},
        {"RTV R16G16B16A16_FLOAT", false, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_FLOAT, 1},
        {"DSV D32_FLOAT", true, depth32, depth32, 1},
        {"RTV R8G8B8A8_UNORM 4x", false, rgba8, rgba8, 4},
        {"RTV R8G8B8A8_UNORM of TYPELESS 4x", false, DXGI_FORMAT_R8G8B8A8_TYPELESS, rgba8, 4},
        {"DSV D32_FLOAT 4x", true, depth32, depth32, 4},
    };
    unsigned failures = 0, cleared = 0, compared = 0, refused = 0;
    struct Range { UINT first, count; };
    for (const Case& c : cases) {
        const UINT mips = c.samples > 1 ? 1 : 3;
        // Every slice from the first on, in the view's mip; for the 4x depth array, counted slices as well.
        std::vector<Range> ranges{{0, UINT_MAX}, {1, UINT_MAX}, {3, UINT_MAX}};
        if (c.depth && c.samples > 1) ranges.insert(ranges.end(), {{0, 1}, {1, 1}, {1, 3}});
        for (const Range range : ranges) {
            const UINT first = range.first, mip = c.samples > 1 ? 0 : std::min(first, 2u);
            const UINT end = range.count == UINT_MAX ? kSlices : first + range.count;
            std::uint32_t expected = 0;
            for (UINT slice = first; slice < end; ++slice) expected |= 1u << (mip + slice * mips);
            const auto check = [&](ID3D12Device* device, bool is_ours, const char* who) {
                copycmp::Run run;
                std::uint32_t reached = 0;
                std::string error = run.init(device, is_ours) ? "" : "no queue or list";
                if (error.empty()) {
                    run.begin();
                    error = clear_through(run, c, mip, first, range.count, reached);
                }
                if (error.empty() && reached == expected) return;
                ++failures;
                std::fprintf(stderr, "%s, mip %u, first slice %u, ArraySize %d, on %s: %s%s; expected slice.mip%s\n",
                             c.name, mip, first, int(range.count), who,
                             error.empty() ? "cleared slice.mip" : error.c_str(),
                             error.empty() ? slices_of(reached, mips).c_str() : "", slices_of(expected, mips).c_str());
            };
            check(ours.p, true, "the layer");
            ++cleared;
            for (const Host& host : hosts) {
                if (!host.device.p || (host.warp && c.depth && c.samples > 1)) continue;
                check(host.device.p, false, host.name.c_str());
                ++compared;
            }
        }
        // What the host's debug layer refuses, the layer refuses: the list fails at Close or when it runs.
        for (const Range range : {Range{kSlices, UINT_MAX}, Range{UINT_MAX, UINT_MAX}, Range{0, 0}, Range{3, 2},
                                  Range{0, kSlices + 1}, Range{1, UINT_MAX - 1}}) {
            copycmp::Run run;
            std::uint32_t reached = 0;
            if (!run.init(ours.p, true)) return 1;
            run.begin();
            const std::string error = clear_through(run, c, 0, range.first, range.count, reached);
            if (run.refused) {
                ++refused;
                continue;
            }
            ++failures;
            std::fprintf(stderr, "%s, first slice %d, ArraySize %d: the layer %s\n", c.name, int(range.first),
                         int(range.count), error.empty() ? "ran the clear" : error.c_str());
        }
    }
    if (hosts[0].device.p) std::printf("host reference: %s, and WARP\n", hosts[0].name.c_str());
    std::printf("array views: %u clears on the layer, %u on the host; %u out-of-range views refused; %u failures\n",
                cleared, compared, refused, failures);
    return failures || vk.validation_errors() ? 1 : 0;
}
