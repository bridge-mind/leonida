// Typeless depth-stencil textures without ALLOW_DEPTH_STENCIL (R24G8_TYPELESS, R32G8X24_TYPELESS), as an application
// makes to keep a copy of its depth buffer for sampling: clear a depth buffer of the family, CopyResource it into
// the plain texture, read both planes back and compare with Windows D3D12/WARP. No game data or shaders.
// LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <algorithm>
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

struct Family {
    const char* name;
    DXGI_FORMAT typeless;
    DXGI_FORMAT depth_view;
};

constexpr Family families[] = {
    {"R24G8_TYPELESS", DXGI_FORMAT_R24G8_TYPELESS, DXGI_FORMAT_D24_UNORM_S8_UINT},
    {"R32G8X24_TYPELESS", DXGI_FORMAT_R32G8X24_TYPELESS, DXGI_FORMAT_D32_FLOAT_S8X24_UINT},
};

constexpr UINT width = 6, height = 4;

D3D12_RESOURCE_DESC texture(DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

bool create(ID3D12Device* device, D3D12_HEAP_TYPE type, const D3D12_RESOURCE_DESC& desc,
            D3D12_RESOURCE_STATES state, ID3D12Resource** out) {
    D3D12_HEAP_PROPERTIES heap{};
    heap.Type = type;
    return SUCCEEDED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(out)));
}

void transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    list->ResourceBarrier(1, &barrier);
}

// Two depth/stencil values, the second in a rectangle, so a copy that drops texels or swaps planes shows.
void clear(ID3D12Device* device, ID3D12GraphicsCommandList* list, ID3D12Resource* depth, DXGI_FORMAT view_format,
           ID3D12DescriptorHeap* heap) {
    D3D12_DEPTH_STENCIL_VIEW_DESC view{};
    view.Format = view_format;
    view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2D;
    const auto handle = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateDepthStencilView(depth, &view, handle);
    const auto both = static_cast<D3D12_CLEAR_FLAGS>(D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL);
    list->ClearDepthStencilView(handle, both, 0.25f, 0x5a, 0, nullptr);
    const D3D12_RECT rect{1, 1, 5, 3};
    list->ClearDepthStencilView(handle, both, 0.75f, 0x3c, 1, &rect);
}

// Clears a depth buffer of the family, copies it into a texture of the same format without ALLOW_DEPTH_STENCIL and
// returns that copy's depth plane then stencil plane, row by row without padding.
bool copy_and_read(ID3D12Device* device, const Family& family, std::vector<unsigned char>& out) {
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    Ref<ID3D12DescriptorHeap> heap;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                             IID_PPV_ARGS(&list.p))));
    CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));

    Ref<ID3D12Resource> depth, plain, readback;
    const D3D12_RESOURCE_DESC plain_desc = texture(family.typeless, D3D12_RESOURCE_FLAG_NONE);
    CHECK(create(device, D3D12_HEAP_TYPE_DEFAULT, texture(family.typeless, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                 D3D12_RESOURCE_STATE_DEPTH_WRITE, &depth.p));
    CHECK(create(device, D3D12_HEAP_TYPE_DEFAULT, plain_desc, D3D12_RESOURCE_STATE_COPY_DEST, &plain.p));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed[2]{};
    UINT64 row_bytes[2]{}, total = 0;
    device->GetCopyableFootprints(&plain_desc, 0, 2, 0, placed, nullptr, row_bytes, &total);
    D3D12_RESOURCE_DESC buffer{};
    buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    buffer.Width = total;
    buffer.Height = buffer.DepthOrArraySize = buffer.MipLevels = 1;
    buffer.SampleDesc.Count = 1;
    buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    CHECK(create(device, D3D12_HEAP_TYPE_READBACK, buffer, D3D12_RESOURCE_STATE_COPY_DEST, &readback.p));

    clear(device, list.p, depth.p, family.depth_view, heap.p);
    transition(list.p, depth.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyResource(plain.p, depth.p);
    transition(list.p, plain.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT plane = 0; plane < 2; ++plane) {
        D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
        src.pResource = plain.p;
        src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
        src.SubresourceIndex = plane;
        dst.pResource = readback.p;
        dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
        dst.PlacedFootprint = placed[plane];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(SUCCEEDED(list->Close()));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
    Event event;
    CHECK(event.handle && SUCCEEDED(fence->SetEventOnCompletion(1, event.handle)));
    CHECK(WaitForSingleObject(event.handle, 5000) == WAIT_OBJECT_0);

    void* mapped = nullptr;
    const D3D12_RANGE range{0, SIZE_T(total)};
    CHECK(SUCCEEDED(readback->Map(0, &range, &mapped)) && mapped);
    const auto* base = static_cast<const unsigned char*>(mapped);
    out.clear();
    for (UINT plane = 0; plane < 2; ++plane) {
        const auto& p = placed[plane];
        for (UINT y = 0; y < p.Footprint.Height; ++y) {
            const unsigned char* line = base + p.Offset + UINT64(y) * p.Footprint.RowPitch;
            out.insert(out.end(), line, line + row_bytes[plane]);
        }
    }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// The first byte where the two readbacks differ, or -1.
long first_difference(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    if (a.size() != b.size()) return long(std::min(a.size(), b.size()));
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) return long(i);
    return -1;
}
}  // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned failures = 0;
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

        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> adapter;
        Ref<ID3D12Device> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p))) ||
            FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&warp.p)))) {
            std::fprintf(stderr, "no WARP device, skipped\n");
            return 77;
        }
        for (const Family& family : families) {
            std::vector<unsigned char> expected, actual;
            if (!copy_and_read(warp.p, family, expected)) {
                std::fprintf(stderr, "%s: WARP could not run the copy\n", family.name);
                ++failures;
                continue;
            }
            if (!copy_and_read(ours.p, family, actual)) {
                std::fprintf(stderr, "%s: the layer could not run the copy\n", family.name);
                ++failures;
                continue;
            }
            const long at = first_difference(actual, expected);
            if (at >= 0) {
                std::fprintf(stderr, "%s: readback differs from WARP at byte %ld of %zu\n", family.name, at,
                             expected.size());
                ++failures;
                continue;
            }
            std::printf("%s without ALLOW_DEPTH_STENCIL: copy of a cleared depth buffer matches WARP (%zu bytes)\n",
                        family.name, expected.size());
        }
    }
    if (vk.validation_errors()) return 1;
    return failures ? 1 : 0;
}
