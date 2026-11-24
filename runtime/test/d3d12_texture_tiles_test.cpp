// D3D-285: compare synthetic sparse RGBA8 texture tiles on the Vulkan shim and native D3D12.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {

using Microsoft::WRL::ComPtr;
constexpr UINT tile_width = 128, tile_height = 128, tiles = 4;
constexpr UINT64 tile_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
constexpr UINT64 image_bytes = tiles * tile_bytes;
constexpr UINT row_pitch = tile_width * sizeof(std::uint32_t);
constexpr UINT full_width = 2 * tile_width;
constexpr UINT snapshots = 3;
constexpr UINT64 row_major_offset = snapshots * image_bytes + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
constexpr auto to_tiled = D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE;
constexpr auto to_linear = D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER;

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

struct Gate {
    ID3D12Fence* fence;
    ~Gate() { if (fence) fence->Signal(1); }
};

D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

bool linear_buffer(ID3D12Device& device, UINT64 bytes, D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state,
                   ComPtr<ID3D12Resource>& resource) {
    const D3D12_HEAP_PROPERTIES heap{type, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1};
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return SUCCEEDED(device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                     IID_PPV_ARGS(&resource)));
}

std::uint32_t pixel(UINT tile, UINT x, UINT y) {
    return 0xcc000000u | ((tile + 1) << 16) | (y << 8) | x;
}

void transition(ID3D12GraphicsCommandList& list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    list.ResourceBarrier(1, &barrier);
}

bool compare_pixel(std::vector<std::uint32_t>& result, std::uint32_t actual, std::uint32_t expected,
                   const char* read, UINT tile, UINT x, UINT y) {
    if (actual != expected) {
        std::fprintf(stderr, "%s tile %u at (%u,%u): got %#x, expected %#x\n", read, tile, x, y, actual, expected);
        return false;
    }
    result.push_back(actual);
    return true;
}

bool run(ID3D12Device& device, UINT width, UINT height, UINT64 tile_copy_offset,
         std::vector<std::uint32_t>& result, bool layer, bool separate_copy_queue, bool& sparse) {
    ComPtr<ID3D12Resource> a, b, upload, readback;
    const auto desc = texture_desc(width, height);
    EXPECT(SUCCEEDED(device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a))));
    sparse = !layer || rt::d3d12::is_sparse_image(a.Get());
    if (!sparse) return true;
    EXPECT(SUCCEEDED(device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b))));
    D3D12_TILE_SHAPE shape{};
    D3D12_PACKED_MIP_INFO mips{};
    UINT tile_count = 0;
    device.GetResourceTiling(a.Get(), &tile_count, &mips, &shape, nullptr, 0, nullptr);
    EXPECT(tile_count == tiles && !mips.NumPackedMips && shape.WidthInTexels == tile_width &&
           shape.HeightInTexels == tile_height && shape.DepthInTexels == 1);

    const D3D12_HEAP_DESC heap_desc{
        image_bytes, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 1, 1},
        0, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES};
    ComPtr<ID3D12Heap> heap;
    EXPECT(SUCCEEDED(device.CreateHeap(&heap_desc, IID_PPV_ARGS(&heap))));
    EXPECT(linear_buffer(device, tile_copy_offset + image_bytes, D3D12_HEAP_TYPE_UPLOAD,
                         D3D12_RESOURCE_STATE_GENERIC_READ, upload));
    EXPECT(linear_buffer(device, row_major_offset + image_bytes, D3D12_HEAP_TYPE_READBACK,
                         D3D12_RESOURCE_STATE_COPY_DEST, readback));
    const D3D12_RANGE empty{0, 0};
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, &empty, &mapped)));
    auto* data = static_cast<std::uint8_t*>(mapped) + tile_copy_offset;
    for (UINT tile = 0; tile < tiles; ++tile)
        for (UINT y = 0; y < tile_height; ++y)
            for (UINT x = 0; x < tile_width; ++x) {
                const auto value = pixel(tile, x, y);
                std::memcpy(data + tile * tile_bytes + y * row_pitch + x * sizeof(value), &value, sizeof(value));
            }
    upload->Unmap(0, nullptr);

    ComPtr<ID3D12Fence> gate, done;
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))));
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))));
    std::array<ComPtr<ID3D12CommandAllocator>, snapshots> allocators;
    std::array<ComPtr<ID3D12GraphicsCommandList>, snapshots> lists;
    for (UINT i = 0; i < snapshots; ++i) {
        const auto type = separate_copy_queue && i == 1 ? D3D12_COMMAND_LIST_TYPE_COPY : D3D12_COMMAND_LIST_TYPE_DIRECT;
        EXPECT(SUCCEEDED(device.CreateCommandAllocator(type, IID_PPV_ARGS(&allocators[i]))));
        EXPECT(SUCCEEDED(device.CreateCommandList(0, type, allocators[i].Get(), nullptr, IID_PPV_ARGS(&lists[i]))));
    }
    ComPtr<ID3D12CommandQueue> queue, copy_queue;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    EXPECT(SUCCEEDED(device.CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))));
    ComPtr<ID3D12Fence> handoff;
    if (separate_copy_queue) {
        const D3D12_COMMAND_QUEUE_DESC copy_desc{D3D12_COMMAND_LIST_TYPE_COPY};
        EXPECT(SUCCEEDED(device.CreateCommandQueue(&copy_desc, IID_PPV_ARGS(&copy_queue))));
        EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&handoff))));
    }
    const Event event;
    EXPECT(event.handle);
    const auto submit = [&](UINT i) {
        if (FAILED(lists[i]->Close())) return false;
        ID3D12CommandList* batch[] = {lists[i].Get()};
        auto* target = separate_copy_queue && i == 1 ? copy_queue.Get() : queue.Get();
        target->ExecuteCommandLists(1, batch);
        return SUCCEEDED(device.GetDeviceRemovedReason());
    };
    const auto wait_done = [&](UINT64 value) {
        return SUCCEEDED(queue->Signal(done.Get(), value)) &&
               SUCCEEDED(done->SetEventOnCompletion(value, event.handle)) &&
               WaitForSingleObject(event.handle, 15000) == WAIT_OBJECT_0 &&
               SUCCEEDED(device.GetDeviceRemovedReason());
    };

    const D3D12_TILED_RESOURCE_COORDINATE origin{}, right{1, 0, 0, 0}, bottom{0, 1, 0, 0};
    const D3D12_TILE_REGION_SIZE all{tiles, TRUE, 2, 2, 1}, column{2, TRUE, 1, 2, 1}, row{2, TRUE, 2, 1, 1};
    lists[0]->CopyTiles(a.Get(), &origin, &all, readback.Get(), tile_copy_offset, to_linear);
    EXPECT(submit(0));
    EXPECT(wait_done(1));

    // Box regions traverse a column with a gap between rows. Swap the two columns in the alias.
    const D3D12_TILED_RESOURCE_COORDINATE starts[] = {origin, right};
    const D3D12_TILE_REGION_SIZE column_regions[] = {column, column};
    const UINT heap_offset = 0;
    queue->UpdateTileMappings(a.Get(), 2, starts, column_regions, heap.Get(), 1, nullptr, &heap_offset, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    queue->CopyTileMappings(b.Get(), &right, a.Get(), &origin, &column, D3D12_TILE_MAPPING_FLAG_NONE);
    queue->CopyTileMappings(b.Get(), &origin, a.Get(), &right, &column, D3D12_TILE_MAPPING_FLAG_NONE);
    transition(*lists[1].Get(), a.Get(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    lists[1]->CopyTiles(a.Get(), &origin, &all, upload.Get(), tile_copy_offset, to_tiled);
    transition(*lists[1].Get(), a.Get(), D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON);
    D3D12_RESOURCE_BARRIER alias{};
    alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    alias.Aliasing = {a.Get(), b.Get()};
    lists[1]->ResourceBarrier(1, &alias);
    lists[1]->CopyTiles(b.Get(), &origin, &all, readback.Get(), tile_copy_offset + image_bytes, to_linear);
    // An ordinary row-major copy also checks where the constrained CopyTiles path wrote each texel.
    D3D12_TEXTURE_COPY_LOCATION src{}, dst{};
    src.pResource = b.Get();
    src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    dst.pResource = readback.Get();
    dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    dst.PlacedFootprint = {row_major_offset, {desc.Format, width, height, 1, 2 * row_pitch}};
    lists[1]->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    if (separate_copy_queue) {
        // The copy must see the direct queue's mappings. The later direct read must see the copy's writes.
        EXPECT(SUCCEEDED(queue->Signal(handoff.Get(), 1)));
        EXPECT(SUCCEEDED(copy_queue->Wait(handoff.Get(), 1)));
    }
    EXPECT(submit(1));
    if (separate_copy_queue) {
        EXPECT(SUCCEEDED(copy_queue->Signal(handoff.Get(), 2)));
        EXPECT(SUCCEEDED(queue->Wait(handoff.Get(), 2)));
    }
    EXPECT(wait_done(2));

    Gate open_on_exit{gate.Get()};
    EXPECT(SUCCEEDED(queue->Wait(gate.Get(), 1)));
    const auto reuse = D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE;
    queue->UpdateTileMappings(b.Get(), 1, &origin, &row, heap.Get(), 1, &reuse, &heap_offset, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    const auto unmap = D3D12_TILE_RANGE_FLAG_NULL;
    queue->UpdateTileMappings(b.Get(), 1, &bottom, &row, nullptr, 1, &unmap, nullptr, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    // In-flight mappings retain their heap memory after the caller drops the heap object.
    if (layer) heap.Reset();
    lists[2]->CopyTiles(b.Get(), &origin, &all, readback.Get(), tile_copy_offset + 2 * image_bytes, to_linear);
    EXPECT(submit(2));
    EXPECT(SUCCEEDED(queue->Signal(done.Get(), 3)));
    EXPECT(done->GetCompletedValue() == 2);
    EXPECT(SUCCEEDED(gate->Signal(1)));
    open_on_exit.fence = nullptr;
    EXPECT(SUCCEEDED(done->SetEventOnCompletion(3, event.handle)));
    EXPECT(WaitForSingleObject(event.handle, 15000) == WAIT_OBJECT_0);

    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto read_pixel = [mapped](UINT64 offset) {
        std::uint32_t value = 0;
        std::memcpy(&value, static_cast<const std::uint8_t*>(mapped) + offset, sizeof(value));
        return value;
    };
    for (UINT tile = 0; tile < tiles; ++tile) {
        const UINT first_y = (tile / 2) * tile_height, first_x = (tile % 2) * tile_width;
        const UINT rows = std::min(tile_height, height - first_y), columns = std::min(tile_width, width - first_x);
        for (UINT y = 0; y < rows; ++y) {
            for (UINT x = 0; x < columns; ++x) {
                const auto tiled = tile_copy_offset + tile * tile_bytes + y * row_pitch + x * sizeof(std::uint32_t);
                EXPECT(compare_pixel(result, read_pixel(tiled), 0, "initially unbound", tile, x, y));
                const UINT source_x = ((tile ^ 1) % 2) * tile_width;
                // The source never wrote the texels beyond its own clipped edge.
                if (source_x + x < width) {
                    const auto aliased = pixel(tile ^ 1, x, y);
                    const auto row_major = row_major_offset +
                                           ((first_y + y) * full_width + first_x + x) * sizeof(std::uint32_t);
                    EXPECT(compare_pixel(result, read_pixel(image_bytes + tiled), aliased, "alias", tile, x, y));
                    EXPECT(compare_pixel(result, read_pixel(row_major), aliased, "row-major alias", tile, x, y));
                }
                const auto remapped = tile < 2 ? pixel(0, x, y) : 0;
                EXPECT(compare_pixel(result, read_pixel(2 * image_bytes + tiled), remapped, "remap", tile, x, y));
            }
        }
    }
    readback->Unmap(0, &empty);
    return true;
}

ComPtr<ID3D12Device> native_device() {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (factory->EnumAdapters1(i, &adapter) != S_OK) return {};
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        ComPtr<ID3D12Device> device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) continue;
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
            options.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2) continue;
        std::printf("Native D3D12: %ls, tiled resources tier %u\n", desc.Description, unsigned(options.TiledResourcesTier));
        return device;
    }
}

}  // namespace

int main() {
    ComPtr<ID3D12Device> vulkan;
    if (FAILED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&vulkan)))) {
        std::puts("Vulkan standalone device unavailable");
        return 77;
    }
    auto multiple_mips = texture_desc(256, 256);
    multiple_mips.MipLevels = 2;
    ComPtr<ID3D12Resource> fallback;
    if (FAILED(vulkan->CreateReservedResource(&multiple_mips, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                               IID_PPV_ARGS(&fallback))) ||
        rt::d3d12::is_sparse_image(fallback.Get())) {
        std::fputs("A texture with multiple mips did not use the committed fallback\n", stderr);
        return 1;
    }
    auto native = native_device();
    const auto* info = rt::d3d12::vulkan_device_info(vulkan.Get());
    const bool dedicated_copy = info && !info->copy_queues.empty() && info->copy_family != info->queue_family;
    struct ReplayCase {
        UINT width, height, linear_offset;
        bool separate_copy_queue;
    };
    constexpr std::array cases{ReplayCase{256, 256, 4, false}, ReplayCase{256, 240, 1, false},
                               ReplayCase{240, 240, 3, false}, ReplayCase{256, 256, 4, true}};
    if (!dedicated_copy) std::puts("No separate Vulkan copy family for the cross-family case");
    for (const auto& [width, height, linear_offset, separate_copy_queue] : cases) {
        if (separate_copy_queue && !dedicated_copy) continue;
        bool sparse = false;
        std::vector<std::uint32_t> observed, expected;
        if (!run(*vulkan.Get(), width, height, linear_offset, observed, true, separate_copy_queue, sparse)) return 1;
        if (!sparse) {
            std::printf("Vulkan device does not support compatible sparse RGBA8 tiling for %u x %u\n", width, height);
            if (height == 256) return 77;
            continue;
        }
        std::printf("Vulkan replay: %u x %u, linear offset %u, initially unbound, aliased, row-major, gated remap%s\n",
                    width, height, linear_offset,
                    separate_copy_queue ? ", COPY-to-DIRECT handoff" : "");
        if (native) {
            if (!run(*native.Get(), width, height, linear_offset, expected, false, separate_copy_queue, sparse) ||
                observed != expected) return 1;
            std::printf("Native D3D12 matched all %zu verified pixels\n", observed.size());
        }
    }
    return 0;
}
