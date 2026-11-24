// Placed 1D and 3D textures (CreatePlacedResource) on the Vulkan-backed device:
//   - compared with Windows D3D12 on the default adapter and on WARP: the answer for each description in a heap sized
//     for both sides' GetResourceAllocationInfo, at offset 0 and at 64 KiB, and the description a placed resource
//     reads back. A dimension outside the enum is refused alike;
//   - on the layer alone: bytes copied through a buffer into every subresource of a placed 3D texture and a placed 1D
//     array, and back out, come back unchanged.
// No game data or shaders. LEONIDA_VK_VALIDATION=1 adds VK_LAYER_KHRONOS_validation to the layer's device.
#include "host_wine.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    T* operator->() const { return p; }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
};

constexpr UINT64 k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
constexpr auto t1 = D3D12_RESOURCE_DIMENSION_TEXTURE1D, t2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
               t3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;

D3D12_RESOURCE_DESC texture(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth_or_layers,
                            UINT16 mips, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = dimension;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = depth_or_layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC buffer(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

UINT64 round_up(UINT64 value, UINT64 alignment) { return (value + alignment - 1) / alignment * alignment; }

// A default heap of `bytes` that takes `desc`'s class of texture on any resource-heap tier.
HRESULT make_heap(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, UINT64 bytes, ID3D12Heap** heap) {
    D3D12_HEAP_DESC heap_desc{};
    heap_desc.SizeInBytes = bytes;
    heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_desc.Alignment = k64;
    const bool rt_ds =
        (desc.Flags & (D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)) != 0;
    heap_desc.Flags = rt_ds ? D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES : D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
    return device->CreateHeap(&heap_desc, IID_PPV_ARGS(heap));
}

bool same(const D3D12_RESOURCE_DESC& a, const D3D12_RESOURCE_DESC& b) {
    return a.Dimension == b.Dimension && a.Alignment == b.Alignment && a.Width == b.Width && a.Height == b.Height &&
           a.DepthOrArraySize == b.DepthOrArraySize && a.MipLevels == b.MipLevels && a.Format == b.Format &&
           a.SampleDesc.Count == b.SampleDesc.Count && a.SampleDesc.Quality == b.SampleDesc.Quality &&
           a.Layout == b.Layout && a.Flags == b.Flags;
}

struct Case {
    const char* name;
    D3D12_RESOURCE_DESC desc;
};

struct Placed {
    Ref<ID3D12Heap> heap;  // declared first, so the resource is released before its heap
    Ref<ID3D12Resource> resource;
    HRESULT hr = E_FAIL;
};

void place(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, UINT64 heap_bytes, UINT64 offset, Placed& out) {
    out.hr = make_heap(device, desc, heap_bytes, &out.heap.p);
    if (FAILED(out.hr)) {
        std::fprintf(stderr, "CreateHeap of %llu bytes failed (0x%08lx)\n", static_cast<unsigned long long>(heap_bytes),
                     static_cast<unsigned long>(out.hr));
        return;
    }
    out.hr = device->CreatePlacedResource(out.heap.p, offset, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                          IID_PPV_ARGS(&out.resource.p));
}

// Every answer and every description read back must be the host's. Returns the number that differ.
unsigned compare(ID3D12Device* ours, ID3D12Device* host, const char* host_name, const std::vector<Case>& cases,
                 unsigned& compared) {
    unsigned differ = 0;
    for (const Case& c : cases) {
        // The heap is sized for the larger of the two answers: each side checks the room in the heap against its own
        // size (the layer's is Vulkan's, which can be larger than a host driver's), and what is compared here is
        // whether a texture of that dimension is placed at all. A description with no size (an invalid dimension)
        // still gets a heap, to see the refusal.
        const D3D12_RESOURCE_ALLOCATION_INFO info = host->GetResourceAllocationInfo(0, 1, &c.desc);
        const D3D12_RESOURCE_ALLOCATION_INFO mine = ours->GetResourceAllocationInfo(0, 1, &c.desc);
        UINT64 bytes = 1024 * 1024;
        if (info.SizeInBytes != UINT64_MAX) {
            bytes = info.SizeInBytes;
            if (mine.SizeInBytes != UINT64_MAX && mine.SizeInBytes != info.SizeInBytes) {
                std::printf("%s: %s's size %llu, the layer's %llu; the heap takes the larger\n", c.name, host_name,
                            static_cast<unsigned long long>(info.SizeInBytes),
                            static_cast<unsigned long long>(mine.SizeInBytes));
                if (mine.SizeInBytes > bytes) bytes = mine.SizeInBytes;
            }
            bytes = round_up(bytes, k64);
        }
        for (const UINT64 offset : {UINT64(0), k64}) {
            Placed expected, actual;
            place(host, c.desc, bytes + offset, offset, expected);
            place(ours, c.desc, bytes + offset, offset, actual);
            ++compared;
            if (expected.hr != actual.hr) {
                ++differ;
                std::fprintf(stderr, "%s at %llu: ours 0x%08lx, %s 0x%08lx\n", c.name,
                             static_cast<unsigned long long>(offset), static_cast<unsigned long>(actual.hr), host_name,
                             static_cast<unsigned long>(expected.hr));
                continue;
            }
            if (FAILED(expected.hr)) continue;
            ++compared;
            if (!same(actual.resource->GetDesc(), expected.resource->GetDesc())) {
                ++differ;
                std::fprintf(stderr, "%s at %llu: GetDesc differs from %s\n", c.name,
                             static_cast<unsigned long long>(offset), host_name);
            }
            ++compared;
            if (actual.resource->GetGPUVirtualAddress() != 0) {  // only buffers are addressable
                ++differ;
                std::fprintf(stderr, "%s at %llu: a placed texture has a GPU VA\n", c.name,
                             static_cast<unsigned long long>(offset));
            }
        }
    }
    return differ;
}

std::uint8_t pattern(UINT subresource, UINT z, UINT y, UINT64 x) {
    return static_cast<std::uint8_t>(subresource * 61 + z * 29 + y * 7 + x * 3 + 1);
}

// Writes a pattern into every subresource of a placed texture through an upload buffer, copies it back out to a
// readback buffer and compares. True when every byte came back.
bool round_trip(ID3D12Device* device, ID3D12CommandQueue* queue, ID3D12Fence* fence, UINT64& fence_value,
                const char* name, const D3D12_RESOURCE_DESC& desc) {
    const D3D12_RESOURCE_ALLOCATION_INFO info = device->GetResourceAllocationInfo(0, 1, &desc);
    if (info.SizeInBytes == UINT64_MAX) {
        std::fprintf(stderr, "%s: no allocation size\n", name);
        return false;
    }
    Ref<ID3D12Heap> heap;
    Ref<ID3D12Resource> placed;
    if (FAILED(make_heap(device, desc, round_up(info.SizeInBytes, k64), &heap.p)) ||
        FAILED(device->CreatePlacedResource(heap.p, 0, &desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                            IID_PPV_ARGS(&placed.p)))) {
        std::fprintf(stderr, "%s: not placed\n", name);
        return false;
    }
    const UINT layers = desc.Dimension == t3 ? 1u : desc.DepthOrArraySize;
    const UINT count = desc.MipLevels * layers;
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts(count);
    std::vector<UINT> rows(count);
    std::vector<UINT64> row_bytes(count);
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, count, 0, layouts.data(), rows.data(), row_bytes.data(), &total);

    D3D12_HEAP_PROPERTIES upload_heap{}, readback_heap{};
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    const D3D12_RESOURCE_DESC staging = buffer(total);
    Ref<ID3D12Resource> upload, readback;
    if (FAILED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &staging,
                                               D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))) ||
        FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &staging,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p)))) {
        std::fprintf(stderr, "%s: no staging buffers\n", name);
        return false;
    }
    // Where row y of slice z of subresource s starts in either staging buffer.
    const auto row_at = [&](UINT s, UINT z, UINT y) {
        const UINT64 pitch = layouts[s].Footprint.RowPitch;
        return layouts[s].Offset + (UINT64(z) * rows[s] + y) * pitch;
    };
    void* mapped = nullptr;
    if (FAILED(upload->Map(0, nullptr, &mapped))) return false;
    for (UINT s = 0; s < count; ++s)
        for (UINT z = 0; z < layouts[s].Footprint.Depth; ++z)
            for (UINT y = 0; y < rows[s]; ++y) {
                auto* row = static_cast<std::uint8_t*>(mapped) + row_at(s, z, y);
                for (UINT64 x = 0; x < row_bytes[s]; ++x) row[x] = pattern(s, z, y, x);
            }
    upload->Unmap(0, nullptr);

    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))))
        return false;
    for (UINT s = 0; s < count; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dst{placed.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        dst.SubresourceIndex = s;
        D3D12_TEXTURE_COPY_LOCATION src{upload.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        src.PlacedFootprint = layouts[s];
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {placed.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, D3D12_RESOURCE_STATE_COPY_DEST,
                          D3D12_RESOURCE_STATE_COPY_SOURCE};
    list->ResourceBarrier(1, &barrier);
    for (UINT s = 0; s < count; ++s) {
        D3D12_TEXTURE_COPY_LOCATION dst{readback.p, D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT, {}};
        dst.PlacedFootprint = layouts[s];
        D3D12_TEXTURE_COPY_LOCATION src{placed.p, D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX, {}};
        src.SubresourceIndex = s;
        list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    if (FAILED(list->Close())) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    ++fence_value;
    if (FAILED(queue->Signal(fence, fence_value)) || FAILED(fence->SetEventOnCompletion(fence_value, nullptr))) {
        std::fprintf(stderr, "%s: the copies did not finish\n", name);
        return false;
    }

    if (FAILED(readback->Map(0, nullptr, &mapped))) return false;
    UINT64 wrong = 0, checked = 0;
    for (UINT s = 0; s < count; ++s)
        for (UINT z = 0; z < layouts[s].Footprint.Depth; ++z)
            for (UINT y = 0; y < rows[s]; ++y) {
                const auto* row = static_cast<const std::uint8_t*>(mapped) + row_at(s, z, y);
                for (UINT64 x = 0; x < row_bytes[s]; ++x, ++checked) {
                    if (row[x] == pattern(s, z, y, x)) continue;
                    if (!wrong++)
                        std::fprintf(stderr, "%s: subresource %u slice %u row %u byte %llu reads 0x%02x, want 0x%02x\n",
                                     name, s, z, y, static_cast<unsigned long long>(x), row[x], pattern(s, z, y, x));
                }
            }
    readback->Unmap(0, nullptr);
    if (wrong) {
        std::fprintf(stderr, "%s: %llu of %llu bytes differ\n", name, static_cast<unsigned long long>(wrong),
                     static_cast<unsigned long long>(checked));
        return false;
    }
    std::printf("%s: %u subresources, %llu bytes round-tripped\n", name, count,
                static_cast<unsigned long long>(checked));
    return true;
}
}  // namespace

int main() {
    Ref<ID3D12Device> ours;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&ours.p))) || !ours.p) {
        std::printf("SKIP: no Vulkan device\n");
        return 77;
    }
    unsigned failures = 0;

    {
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        if (FAILED(ours->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
            FAILED(ours->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p)))) {
            std::fprintf(stderr, "no command queue or fence\n");
            return 1;
        }
        UINT64 fence_value = 0;
        const Case trips[] = {
            {"placed 3D R8G8B8A8_UNORM 8x8x4, 2 mips", texture(t3, 8, 8, 4, 2, DXGI_FORMAT_R8G8B8A8_UNORM)},
            {"placed 3D R16G16B16A16_FLOAT 4x4x4", texture(t3, 4, 4, 4, 1, DXGI_FORMAT_R16G16B16A16_FLOAT)},
            {"placed 1D array R8G8B8A8_UNORM 64 x3, 3 mips", texture(t1, 64, 1, 3, 3, DXGI_FORMAT_R8G8B8A8_UNORM)},
        };
        for (const Case& c : trips)
            if (!round_trip(ours.p, queue.p, fence.p, fence_value, c.name, c.desc)) ++failures;
    }

    // A desc with a dimension outside the enum, which the host refuses.
    D3D12_RESOURCE_DESC unknown = texture(t2, 64, 64, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
    unknown.Dimension = D3D12_RESOURCE_DIMENSION_UNKNOWN;
    D3D12_RESOURCE_DESC past_3d = unknown;
    past_3d.Dimension = D3D12_RESOURCE_DIMENSION(5);
    constexpr auto rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
    constexpr auto uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, target = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    const std::vector<Case> cases = {
        {"1D 4096", texture(t1, 4096, 1, 1, 1, rgba8)},
        {"1D 256, all mips", texture(t1, 256, 1, 1, 0, rgba8)},
        {"1D array of 8, 13 mips", texture(t1, 4096, 1, 8, 13, rgba8)},
        {"1D R32_FLOAT unordered access", texture(t1, 1024, 1, 1, 1, DXGI_FORMAT_R32_FLOAT, uav)},
        {"1D render target", texture(t1, 1024, 1, 1, 1, rgba8, target)},
        {"3D 64x64x64", texture(t3, 64, 64, 64, 1, rgba8)},
        {"3D 32x32x32, all mips", texture(t3, 32, 32, 32, 0, rgba8)},
        {"3D 64x16x4, all mips", texture(t3, 64, 16, 4, 0, rgba8)},
        {"3D BC1 128x128x16", texture(t3, 128, 128, 16, 1, DXGI_FORMAT_BC1_UNORM)},
        {"3D R16G16B16A16_FLOAT 32x32x32", texture(t3, 32, 32, 32, 1, DXGI_FORMAT_R16G16B16A16_FLOAT)},
        {"3D R32_FLOAT unordered access", texture(t3, 64, 64, 64, 1, DXGI_FORMAT_R32_FLOAT, uav)},
        {"3D render target", texture(t3, 64, 64, 64, 1, rgba8, target)},
        {"2D 64x64 (control)", texture(t2, 64, 64, 1, 1, rgba8)},
        {"dimension UNKNOWN", unknown},
        {"dimension 5", past_3d},
    };

    if (hostwine::host_is_wine()) {
        std::printf("placed dimensions: the host comparison is skipped under Wine\n");
        return failures ? 1 : 0;
    }
    Ref<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p)))) {
        std::fprintf(stderr, "no DXGI factory\n");
        return 1;
    }
    unsigned references = 0;
    for (const bool warp : {false, true}) {
        Ref<IDXGIAdapter> adapter;
        Ref<ID3D12Device> host;
        if (warp && FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter.p)))) continue;
        if (FAILED(D3D12CreateDevice(adapter.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) continue;
        const char* name = warp ? "WARP" : "the default adapter";
        unsigned compared = 0;
        const unsigned differ = compare(ours.p, host.p, name, cases, compared);
        std::printf("placed dimensions against %s: %u answers compared, %u differ\n", name, compared, differ);
        failures += differ;
        ++references;
    }
    if (!references) {
        std::fprintf(stderr, "no D3D12 reference device (default adapter or WARP)\n");
        return 1;
    }
    return failures ? 1 : 0;
}
