// 1D render targets and depth buffers: TEXTURE1D resources with ALLOW_RENDER_TARGET or ALLOW_DEPTH_STENCIL, compared
// with Windows D3D12. The host adapter and WARP both create them; the layer must too wherever the Vulkan device can
// make the 1D attachment image. Clears through TEXTURE1D and TEXTURE1DARRAY RTVs/DSVs, whole and with rectangles,
// and a rectangle clear through a 1D UAV are read back and compared with WARP byte for byte.
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "host_adapter.h"
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <cstring>
#include <string>
#include <vector>
#define CHECK(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { if (p) p->Release(); }
    T* operator->() const { return p; }
};
struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

D3D12_RESOURCE_DESC line(DXGI_FORMAT format, UINT64 width, UINT16 layers, UINT16 mips, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    desc.Width = width;
    desc.Height = 1;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

HRESULT create(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, ID3D12Resource** out) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_STATES state = (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)
                                            ? D3D12_RESOURCE_STATE_DEPTH_WRITE
                                            : (desc.Flags & D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)
                                                  ? D3D12_RESOURCE_STATE_RENDER_TARGET
                                                  : D3D12_RESOURCE_STATE_COMMON;
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(out));
}

// What the clears below do; each reads back every subresource of one texture.
enum class Scenario { color, depth, uav };

bool clear_and_read(ID3D12Device* device, Scenario scenario, std::vector<unsigned char>& bytes) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    constexpr UINT width = 5;  // odd, so a rectangle clear has texels on both sides
    D3D12_RESOURCE_DESC desc{};
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    switch (scenario) {
    case Scenario::color:
        desc = line(DXGI_FORMAT_R8G8B8A8_UNORM, width, 3, 2, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        state = D3D12_RESOURCE_STATE_RENDER_TARGET;
        break;
    case Scenario::depth:
        desc = line(DXGI_FORMAT_D32_FLOAT, width, 3, 2, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
        break;
    case Scenario::uav:
        desc = line(DXGI_FORMAT_R8G8B8A8_UNORM, width, 1, 1, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        break;
    }
    Ref<ID3D12Resource> image, readback;
    const D3D12_HEAP_PROPERTIES local{D3D12_HEAP_TYPE_DEFAULT};
    CHECK(SUCCEEDED(device->CreateCommittedResource(&local, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                    IID_PPV_ARGS(&image.p))));
    const UINT count = desc.MipLevels * desc.DepthOrArraySize;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> placed(count);
    std::vector<UINT64> row_bytes(count);
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, count, 0, placed.data(), nullptr, row_bytes.data(), &total);
    CHECK(total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = buffer.SampleDesc.Count = 1;
    buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                    IID_PPV_ARGS(&readback.p))));

    Ref<ID3D12DescriptorHeap> targets, visible, cpu_only;
    const D3D12_DESCRIPTOR_HEAP_TYPE target_type =
        scenario == Scenario::depth ? D3D12_DESCRIPTOR_HEAP_TYPE_DSV : D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
    const D3D12_DESCRIPTOR_HEAP_DESC target_desc{target_type, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&target_desc, IID_PPV_ARGS(&targets.p))));
    const UINT step = device->GetDescriptorHandleIncrementSize(target_type);
    const auto slot = [&](UINT i) {
        return D3D12_CPU_DESCRIPTOR_HANDLE{targets->GetCPUDescriptorHandleForHeapStart().ptr + i * step};
    };

    if (scenario == Scenario::color) {
        D3D12_RENDER_TARGET_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;
        // A distinct value in every subresource first, through single-slice array views.
        for (UINT layer = 0; layer < desc.DepthOrArraySize; ++layer)
            for (UINT mip = 0; mip < desc.MipLevels; ++mip) {
                view.Texture1DArray = {mip, layer, 1};
                device->CreateRenderTargetView(image.p, &view, slot(0));
                const FLOAT seed[4] = {0.2f * FLOAT(layer + 1), 0.2f * FLOAT(mip + 2), 0.6f, 1};
                list->ClearRenderTargetView(slot(0), seed, 0, nullptr);
            }
        view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1D;  // slice 0 of the array, mip 1
        view.Texture1D = {1};
        device->CreateRenderTargetView(image.p, &view, slot(1));
        const FLOAT whole[4] = {0.75f, 0.25f, 0.125f, 0.8f};
        list->ClearRenderTargetView(slot(1), whole, 0, nullptr);
        view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE1DARRAY;  // slices 1-2 of mip 0, texels 1-3
        view.Texture1DArray = {0, 1, 2};
        device->CreateRenderTargetView(image.p, &view, slot(2));
        const FLOAT part[4] = {0.0f, 1.0f, 0.25f, 0.75f};
        const D3D12_RECT rect{1, 0, 4, 1};
        list->ClearRenderTargetView(slot(2), part, 1, &rect);
        view.Texture1DArray = {1, 2, UINT(-1)};  // slice 2 onwards of mip 1, texel 0
        device->CreateRenderTargetView(image.p, &view, slot(3));
        const FLOAT corner[4] = {1.0f, 0.0f, 1.0f, 0.0f};
        const D3D12_RECT first{0, 0, 1, 1};
        list->ClearRenderTargetView(slot(3), corner, 1, &first);
    } else if (scenario == Scenario::depth) {
        D3D12_DEPTH_STENCIL_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_D32_FLOAT;
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;
        for (UINT mip = 0; mip < desc.MipLevels; ++mip) {
            view.Texture1DArray = {mip, 0, desc.DepthOrArraySize};
            device->CreateDepthStencilView(image.p, &view, slot(0));
            list->ClearDepthStencilView(slot(0), D3D12_CLEAR_FLAG_DEPTH, 0.125f + 0.25f * mip, 0, 0, nullptr);
        }
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1D;  // slice 0, mip 1
        view.Texture1D = {1};
        device->CreateDepthStencilView(image.p, &view, slot(1));
        list->ClearDepthStencilView(slot(1), D3D12_CLEAR_FLAG_DEPTH, 1.0f, 0, 0, nullptr);
        view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE1DARRAY;  // slice 1 of mip 0, texels 1-2
        view.Texture1DArray = {0, 1, 1};
        device->CreateDepthStencilView(image.p, &view, slot(2));
        const D3D12_RECT rect{1, 0, 3, 1};
        list->ClearDepthStencilView(slot(2), D3D12_CLEAR_FLAG_DEPTH, 0.75f, 0, 1, &rect);
    } else {
        const D3D12_DESCRIPTOR_HEAP_DESC gpu_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                  D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
        const D3D12_DESCRIPTOR_HEAP_DESC cpu_desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1,
                                                  D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&gpu_desc, IID_PPV_ARGS(&visible.p))));
        CHECK(SUCCEEDED(device->CreateDescriptorHeap(&cpu_desc, IID_PPV_ARGS(&cpu_only.p))));
        D3D12_UNORDERED_ACCESS_VIEW_DESC view{};
        view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        view.ViewDimension = D3D12_UAV_DIMENSION_TEXTURE1D;
        view.Texture1D = {0};
        device->CreateUnorderedAccessView(image.p, nullptr, &view, visible->GetCPUDescriptorHandleForHeapStart());
        device->CreateUnorderedAccessView(image.p, nullptr, &view, cpu_only->GetCPUDescriptorHandleForHeapStart());
        ID3D12DescriptorHeap* heaps[] = {visible.p};
        list->SetDescriptorHeaps(1, heaps);
        const FLOAT whole[4] = {0.25f, 0.8f, 0.75f, 1.0f}, part[4] = {1.0f, 0.0f, 0.2f, 0.25f};
        list->ClearUnorderedAccessViewFloat(visible->GetGPUDescriptorHandleForHeapStart(),
                                            cpu_only->GetCPUDescriptorHandleForHeapStart(), image.p, whole, 0,
                                            nullptr);
        D3D12_RESOURCE_BARRIER uav{};
        uav.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
        uav.UAV.pResource = image.p;
        list->ResourceBarrier(1, &uav);
        const D3D12_RECT rect{2, 0, 4, 1};
        list->ClearUnorderedAccessViewFloat(visible->GetGPUDescriptorHandleForHeapStart(),
                                            cpu_only->GetCPUDescriptorHandleForHeapStart(), image.p, part, 1, &rect);
    }

    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = image.p;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = state;
    barrier.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
    list->ResourceBarrier(1, &barrier);
    for (UINT subresource = 0; subresource < count; ++subresource) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = image.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = subresource;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = placed[subresource];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event event;
    CHECK(event.handle);
    CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);
    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped);
    bytes.clear();
    for (UINT i = 0; i < count; ++i) {  // one row per subresource: the row's bytes without the pitch padding
        const auto* row = static_cast<const unsigned char*>(mapped) + placed[i].Offset;
        bytes.insert(bytes.end(), row, row + row_bytes[i]);
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

struct Format {
    DXGI_FORMAT dxgi;
    VkFormat vk;  // the image the layer makes (typeless depth families are their depth format)
    D3D12_RESOURCE_FLAGS flag;
};
}  // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned failures = 0, accepted = 0, device_lacks = 0, references = 0, compared = 0;
    {
        Ref<ID3D12Device> ours;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        info.conditional_rendering = vk.conditional_rendering;
        info.storage_buffer_8bit = vk.storage_buffer_8bit;
        info.buffer_device_address = vk.buffer_device_address;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        if (hostwine::host_is_wine()) return 77;  // every check is a comparison with Windows D3D12
        Ref<IDXGIFactory4> factory;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
            std::fprintf(stderr, "no DXGI factory\n");
            return 1;
        }
        Ref<ID3D12Device> hosts[2];  // the layer's own adapter, then WARP
        std::string names[2];
        for (int warp = 0; warp < 2; ++warp) {
            Ref<IDXGIAdapter> adapter;
            if (!hostadapter::reference(factory.p, vk, warp == 1, &adapter.p, names[warp])) continue;
            if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hosts[warp].p))))
                hosts[warp].p = nullptr;
            else
                ++references;
        }
        if (!hosts[1].p) {
            std::fprintf(stderr, "WARP reference unavailable\n");
            return 1;
        }

        // Creation: where every reference creates the texture and the Vulkan device can make the 1D image with
        // that attachment usage, the layer must create it, and GetResourceAllocationInfo must answer a size.
        const auto image_format_properties = vk.instance_fn<PFN_vkGetPhysicalDeviceImageFormatProperties>(
            "vkGetPhysicalDeviceImageFormatProperties");
        constexpr auto target = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, depth = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
        const Format formats[] = {
            {DXGI_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, target},
            {DXGI_FORMAT_R8G8B8A8_TYPELESS, VK_FORMAT_R8G8B8A8_UNORM, target},
            {DXGI_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_B8G8R8A8_UNORM, target},
            {DXGI_FORMAT_R16G16B16A16_FLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, target},
            {DXGI_FORMAT_R32G32B32A32_FLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, target},
            {DXGI_FORMAT_R10G10B10A2_UNORM, VK_FORMAT_A2B10G10R10_UNORM_PACK32, target},
            {DXGI_FORMAT_R11G11B10_FLOAT, VK_FORMAT_B10G11R11_UFLOAT_PACK32, target},
            {DXGI_FORMAT_R32_FLOAT, VK_FORMAT_R32_SFLOAT, target},
            {DXGI_FORMAT_R16_UINT, VK_FORMAT_R16_UINT, target},
            {DXGI_FORMAT_R8_UNORM, VK_FORMAT_R8_UNORM, target},
            {DXGI_FORMAT_D32_FLOAT, VK_FORMAT_D32_SFLOAT, depth},
            {DXGI_FORMAT_R32_TYPELESS, VK_FORMAT_D32_SFLOAT, depth},
            {DXGI_FORMAT_D16_UNORM, VK_FORMAT_D16_UNORM, depth},
            {DXGI_FORMAT_D32_FLOAT_S8X24_UINT, VK_FORMAT_D32_SFLOAT_S8_UINT, depth},
        };
        for (const Format& format : formats) {
            const VkImageUsageFlags usage =
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                (format.flag == depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                                      : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
            const bool typeless = format.dxgi == DXGI_FORMAT_R8G8B8A8_TYPELESS;
            VkImageFormatProperties limits{};
            const bool device_can =
                image_format_properties(vk.physical, format.vk, VK_IMAGE_TYPE_1D, VK_IMAGE_TILING_OPTIMAL, usage,
                                        typeless ? VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT : 0, &limits) == VK_SUCCESS &&
                limits.maxArrayLayers >= 6;
            for (const auto& desc : {line(format.dxgi, 64, 1, 1, format.flag), line(format.dxgi, 100, 6, 0, format.flag)}) {
                bool every_reference = true;
                for (const auto& host : hosts) {
                    Ref<ID3D12Resource> resource;
                    if (host.p && FAILED(create(host.p, desc, &resource.p))) every_reference = false;
                }
                if (!every_reference) continue;  // nothing to match
                Ref<ID3D12Resource> resource;
                const HRESULT hr = create(ours.p, desc, &resource.p);
                const D3D12_RESOURCE_ALLOCATION_INFO size = ours->GetResourceAllocationInfo(0, 1, &desc);
                if (SUCCEEDED(hr) && size.SizeInBytes != UINT64_MAX) {
                    ++accepted;
                    continue;
                }
                if (!device_can) {
                    ++device_lacks;
                    std::fprintf(stderr, "format %d flags 0x%x: the Vulkan device has no such 1D image, not checked\n",
                                 int(format.dxgi), unsigned(format.flag));
                    continue;
                }
                ++failures;
                std::fprintf(stderr, "format %d flags 0x%x, %u layers: the layer refuses (0x%08lx, size %llu) what "
                             "every reference creates\n", int(format.dxgi), unsigned(format.flag),
                             unsigned(desc.DepthOrArraySize), static_cast<unsigned long>(hr),
                             static_cast<unsigned long long>(size.SizeInBytes));
            }
        }

        // Clears through 1D views, read back against WARP. The depth clears need a 1D D32 depth attachment, which
        // Vulkan does not require of a device.
        VkImageFormatProperties limits{};
        const bool depth_line =
            image_format_properties(vk.physical, VK_FORMAT_D32_SFLOAT, VK_IMAGE_TYPE_1D, VK_IMAGE_TILING_OPTIMAL,
                                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                        VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                                    0, &limits) == VK_SUCCESS &&
            limits.maxArrayLayers >= 3;
        for (const auto [scenario, name] : {std::pair{Scenario::color, "1D render target"},
                                            std::pair{Scenario::depth, "1D depth buffer"},
                                            std::pair{Scenario::uav, "1D unordered access"}}) {
            if (scenario == Scenario::depth && !depth_line) {
                ++device_lacks;
                std::fprintf(stderr, "%s: the Vulkan device has no 1D D32 depth attachment, not checked\n", name);
                continue;
            }
            std::vector<unsigned char> expected, actual;
            if (!clear_and_read(hosts[1].p, scenario, expected)) {
                std::fprintf(stderr, "%s: WARP could not run the clears\n", name);
                ++failures;
                continue;
            }
            if (!clear_and_read(ours.p, scenario, actual)) {
                std::fprintf(stderr, "%s: the layer could not run the clears\n", name);
                ++failures;
                continue;
            }
            ++compared;
            if (actual != expected) {
                ++failures;
                std::fprintf(stderr, "%s: readback differs from WARP\n", name);
                for (size_t i = 0; i < expected.size() && i < actual.size(); ++i)
                    if (expected[i] != actual[i])
                        std::fprintf(stderr, "  byte %zu: ours %u, WARP %u\n", i, actual[i], expected[i]);
                if (expected.size() != actual.size())
                    std::fprintf(stderr, "  sizes: ours %zu, WARP %zu\n", actual.size(), expected.size());
            }
        }
    }
    std::printf("1D targets: %u descriptions created as the references do, %u not checked (Vulkan device), "
                "%u clear readbacks compared with WARP, %u failures (%u references)\n",
                accepted, device_lacks, compared, failures, references);
    if (failures || !accepted) return 1;
    if (vk.validation_errors()) return 1;
    return 0;
}
