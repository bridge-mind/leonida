// Synthetic reserved buffers and real D3D12 heap mappings. Run the same cases on the Vulkan layer and the host.
#include "../shims/d3d12_com.h"

#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
} while (0)

namespace {

using Microsoft::WRL::ComPtr;
constexpr UINT64 tile_bytes = D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES;
constexpr UINT snapshots = 3, tiles_per_buffer = 4;
constexpr UINT64 buffer_bytes = tiles_per_buffer * tile_bytes;
constexpr std::array<std::uint32_t, tiles_per_buffer> patterns{0x23456789, 0xf1c0a761, 0x9a879651, 0xe721cc71};
using Snapshot = std::array<std::uint32_t, tiles_per_buffer>;
using Results = std::array<Snapshot, snapshots>;
constexpr Results expected{{
    {patterns[0], patterns[1], patterns[2], 0},
    {patterns[0], 0, patterns[2], patterns[2]},
    {0, patterns[1], 0, 0},
}};

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = desc.DepthOrArraySize = desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

struct Event {
    HANDLE handle = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ~Event() { if (handle) CloseHandle(handle); }
};

struct Gate {
    ID3D12Fence* fence;
    ~Gate() { if (fence) fence->Signal(1); }
};

bool run(ID3D12Device& device, Results& result, bool layer = false) {
    ComPtr<ID3D12Resource> a, b, upload, readback;
    const auto reserved = buffer_desc(buffer_bytes);
    EXPECT(SUCCEEDED(device.CreateReservedResource(&reserved, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a))));
    EXPECT(SUCCEEDED(device.CreateReservedResource(&reserved, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b))));
    EXPECT(a->GetGPUVirtualAddress() && b->GetGPUVirtualAddress() && a->GetGPUVirtualAddress() != b->GetGPUVirtualAddress());
    UINT tile_count = 0;
    device.GetResourceTiling(a.Get(), &tile_count, nullptr, nullptr, nullptr, 0, nullptr);
    EXPECT(tile_count == tiles_per_buffer);

    D3D12_HEAP_DESC heap_desc{};
    heap_desc.SizeInBytes = 3 * tile_bytes;
    heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    heap_desc.Properties.CreationNodeMask = heap_desc.Properties.VisibleNodeMask = 1;
    heap_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ComPtr<ID3D12Heap> heap;
    EXPECT(SUCCEEDED(device.CreateHeap(&heap_desc, IID_PPV_ARGS(&heap))));
    D3D12_HEAP_PROPERTIES upload_heap = heap_desc.Properties, readback_heap = heap_desc.Properties;
    upload_heap.Type = D3D12_HEAP_TYPE_UPLOAD;
    readback_heap.Type = D3D12_HEAP_TYPE_READBACK;
    const auto source_desc = buffer_desc(tile_bytes + buffer_bytes);
    const auto target_desc = buffer_desc(snapshots * buffer_bytes);
    EXPECT(SUCCEEDED(device.CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &source_desc,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload))));
    EXPECT(SUCCEEDED(device.CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &target_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback))));
    constexpr auto words_per_tile = std::size_t(tile_bytes / sizeof(std::uint32_t));
    const D3D12_RANGE empty{0, 0};
    void* mapped = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, &empty, &mapped)));
    auto* words = static_cast<std::uint32_t*>(mapped);
    for (std::size_t i = 0; i < patterns.size(); ++i)
        std::fill_n(words + (i + 1) * words_per_tile, words_per_tile, patterns[i]);
    upload->Unmap(0, nullptr);

    ComPtr<ID3D12Fence> gate, done;
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate))));
    EXPECT(SUCCEEDED(device.CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done))));
    std::array<ComPtr<ID3D12CommandAllocator>, snapshots> allocators;
    std::array<ComPtr<ID3D12GraphicsCommandList>, snapshots> lists;
    for (UINT i = 0; i < snapshots; ++i) {
        EXPECT(SUCCEEDED(device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocators[i]))));
        EXPECT(SUCCEEDED(device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocators[i].Get(), nullptr,
                                                 IID_PPV_ARGS(&lists[i]))));
    }
    ComPtr<ID3D12CommandQueue> queue;
    const D3D12_COMMAND_QUEUE_DESC queue_desc{};
    EXPECT(SUCCEEDED(device.CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue))));
    Event event;
    EXPECT(event.handle);
    Gate open_on_exit{gate.Get()};
    EXPECT(SUCCEEDED(queue->Wait(gate.Get(), 1)));

    // Map disjoint tiles in a different order, using the default one-tile resource regions.
    const D3D12_TILED_RESOURCE_COORDINATE starts[] = {{2, 0, 0, 0}, {0, 0, 0, 0}, {1, 0, 0, 0}};
    const UINT heap_offsets[] = {2, 0, 1}, heap_counts[] = {1, 1, 1};
    queue->UpdateTileMappings(a.Get(), 3, starts, nullptr, heap.Get(), 3, nullptr, heap_offsets, heap_counts,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    const D3D12_TILED_RESOURCE_COORDINATE origin{};
    const D3D12_TILE_REGION_SIZE all{tiles_per_buffer, FALSE, 0, 0, 0};
    queue->CopyTileMappings(b.Get(), &origin, a.Get(), &origin, &all, D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    lists[0]->CopyTiles(a.Get(), &origin, &all, upload.Get(), tile_bytes,
                        D3D12_TILE_COPY_FLAG_LINEAR_BUFFER_TO_SWIZZLED_TILED_RESOURCE);
    D3D12_RESOURCE_BARRIER alias{};
    alias.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    alias.Aliasing = {a.Get(), b.Get()};
    lists[0]->ResourceBarrier(1, &alias);
    lists[0]->CopyTiles(b.Get(), &origin, &all, readback.Get(), 0,
                        D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER);
    const auto submit = [&](UINT i) {
        if (FAILED(lists[i]->Close())) return false;
        ID3D12CommandList* batch[] = {lists[i].Get()};
        queue->ExecuteCommandLists(1, batch);
        return true;
    };
    EXPECT(submit(0));
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));

    // Preserve B[0], detach B[1] and make B[2:4] share the same physical tile. One range crosses a region boundary.
    const D3D12_TILED_RESOURCE_COORDINATE regions[] = {{0, 0, 0, 0}, {3, 0, 0, 0}};
    const D3D12_TILE_REGION_SIZE sizes[] = {{3, FALSE, 0, 0, 0}, {1, FALSE, 0, 0, 0}};
    const D3D12_TILE_RANGE_FLAGS flags[] = {D3D12_TILE_RANGE_FLAG_SKIP, D3D12_TILE_RANGE_FLAG_NULL,
                                           D3D12_TILE_RANGE_FLAG_REUSE_SINGLE_TILE};
    const UINT offsets[] = {0, 0, 2}, counts[] = {1, 1, 2};
    queue->UpdateTileMappings(b.Get(), 2, regions, sizes, heap.Get(), 3, flags, offsets, counts,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    EXPECT(SUCCEEDED(device.GetDeviceRemovedReason()));
    if (layer) {
        // One valid tile followed by an out-of-bounds heap index. The entire update is rejected.
        const D3D12_TILE_RANGE_FLAGS invalid_flags[] = {D3D12_TILE_RANGE_FLAG_NULL, D3D12_TILE_RANGE_FLAG_NONE};
        const UINT invalid_offsets[] = {0, 3}, invalid_counts[] = {1, 1};
        const D3D12_TILE_REGION_SIZE first_two{2, FALSE, 0, 0, 0};
        queue->UpdateTileMappings(b.Get(), 1, &origin, &first_two, heap.Get(), 2, invalid_flags, invalid_offsets,
                                  invalid_counts, D3D12_TILE_MAPPING_FLAG_NONE);
    }
    lists[1]->CopyBufferRegion(readback.Get(), buffer_bytes, b.Get(), 0, buffer_bytes);
    EXPECT(submit(1));

    const auto unmap = D3D12_TILE_RANGE_FLAG_NULL;
    queue->UpdateTileMappings(b.Get(), 1, nullptr, nullptr, nullptr, 1, &unmap, nullptr, nullptr,
                              D3D12_TILE_MAPPING_FLAG_NONE);
    const D3D12_TILE_REGION_SIZE one{1, FALSE, 0, 0, 0};
    queue->CopyTileMappings(b.Get(), &starts[2], a.Get(), &starts[2], &one, D3D12_TILE_MAPPING_FLAG_NONE);
    lists[2]->CopyTiles(b.Get(), &origin, &all, readback.Get(), 2 * buffer_bytes,
                        D3D12_TILE_COPY_FLAG_SWIZZLED_TILED_RESOURCE_TO_LINEAR_BUFFER);
    EXPECT(submit(2));
    EXPECT(SUCCEEDED(queue->Signal(done.Get(), 1)));
    const auto before_open = done->GetCompletedValue();
    if (before_open != 0) {
        std::fprintf(stderr, "fence before opening the gate: %llu, device status %#lx\n",
                     static_cast<unsigned long long>(before_open), static_cast<unsigned long>(device.GetDeviceRemovedReason()));
        return false;
    }
    EXPECT(SUCCEEDED(gate->Signal(1)));
    open_on_exit.fence = nullptr;
    EXPECT(SUCCEEDED(done->SetEventOnCompletion(1, event.handle)));
    EXPECT(WaitForSingleObject(event.handle, 15000) == WAIT_OBJECT_0);

    EXPECT(SUCCEEDED(readback->Map(0, nullptr, &mapped)));
    const auto* data = static_cast<const std::uint32_t*>(mapped);
    // Every word of every tile, with the first wrong word of each tile printed, so a failure says which side,
    // snapshot and tile differ and what was read there.
    UINT wrong_tiles = 0;
    for (UINT i = 0; i < snapshots; ++i) {
        for (UINT tile = 0; tile < tiles_per_buffer; ++tile) {
            const auto* first = data + (i * tiles_per_buffer + tile) * words_per_tile;
            result[i][tile] = *first;
            const auto* bad = std::find_if(first, first + words_per_tile, [&](auto word) { return word != expected[i][tile]; });
            if (bad == first + words_per_tile) continue;
            ++wrong_tiles;
            const auto wrong_words = std::count_if(bad, first + words_per_tile,
                                                   [&](auto word) { return word != expected[i][tile]; });
            std::fprintf(stderr, "%s snapshot %u tile %u: word %u reads %#010x, want %#010x (%lld of %llu words wrong)\n",
                         layer ? "Vulkan layer" : "native D3D12", i, tile, unsigned(bad - first), unsigned(*bad),
                         unsigned(expected[i][tile]), static_cast<long long>(wrong_words),
                         static_cast<unsigned long long>(words_per_tile));
        }
    }
    readback->Unmap(0, &empty);
    EXPECT(wrong_tiles == 0);
    return true;
}

// D3D12 on the adapter `luid` names, when it has tiled resources tier 2 or more; null otherwise.
ComPtr<ID3D12Device> tiled_device(IDXGIFactory4& factory, LUID luid, bool print = true) {
    ComPtr<IDXGIAdapter1> adapter;
    if (!luid.LowPart && !luid.HighPart) return {};
    if (FAILED(factory.EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter)))) return {};
    DXGI_ADAPTER_DESC1 desc{};
    ComPtr<ID3D12Device> device;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(adapter->GetDesc1(&desc)) || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) ||
        FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device))) ||
        FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2)
        return {};
    if (print)
        std::printf("Native D3D12: %ls (the layer's adapter), tiled resources tier %u\n", desc.Description,
                    unsigned(options.TiledResourcesTier));
    return device;
}

// The host adapter for the native half. LEONIDA_HOST_ADAPTER picks it as in d3d12_copy_compare.h: "warp", or part of
// an adapter name (case-insensitive); a value that names no adapter with tiled resources tier 2 skips the native half
// rather than testing another GPU. Unset: the adapter behind the layer's Vulkan device (`layer_luid`, as
// host_adapter.h picks it), so a PC with two GPUs compares the layer with its own GPU's driver; when that adapter
// has no tier 2, the first hardware adapter with tier 2 or more.
ComPtr<ID3D12Device> native_device(LUID layer_luid) {
    ComPtr<IDXGIFactory4> factory;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return {};
    char want[128] = {};
    GetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", want, sizeof(want));
    std::string needle(want);
    std::transform(needle.begin(), needle.end(), needle.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (needle.empty())
        if (auto device = tiled_device(*factory.Get(), layer_luid)) return device;
    for (UINT i = 0;; ++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if (needle == "warp") {
            if (i > 0 || FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)))) break;
        } else if (factory->EnumAdapters1(i, &adapter) != S_OK) {
            break;
        }
        DXGI_ADAPTER_DESC1 desc{};
        if (FAILED(adapter->GetDesc1(&desc))) continue;
        if (needle.empty()) {
            if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
        } else if (needle != "warp") {
            std::string name;
            for (const wchar_t* c = desc.Description; *c; ++c) name += char(std::tolower(*c < 128 ? *c : '?'));
            if (name.find(needle) == std::string::npos) continue;
        }
        ComPtr<ID3D12Device> device;
        if (FAILED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)))) continue;
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
            options.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_2) continue;
        std::printf("Native D3D12: %ls, tiled resources tier %u\n", desc.Description, unsigned(options.TiledResourcesTier));
        return device;
    }
    if (!needle.empty())
        std::printf("Native D3D12: no adapter matching LEONIDA_HOST_ADAPTER=%s with tiled resources tier 2 or more; "
                    "native half skipped\n", want);
    return {};
}

}  // namespace

int main() {
    ComPtr<ID3D12Device> vulkan;
    const auto init = rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&vulkan));
    if (FAILED(init)) {
        std::puts("Vulkan standalone device unavailable");
        return 77;
    }
    if (!rt::d3d12::device_sparse_buffer_tiles(vulkan.Get())) {
        std::puts("Vulkan device lacks sparse buffer residency with aliasing and strict unbound reads");
        return 77;
    }
    Results observed{}, native{};
    if (!run(*vulkan.Get(), observed, true)) return 1;
    std::puts("Vulkan replay: 12 full tiles matched across aliased, skipped, unmapped, reused and remapped ranges");
    std::puts("A rejected update left all tile mappings intact");
    const LUID layer_luid = vulkan->GetAdapterLuid();
    if (auto device = native_device(layer_luid)) {
        // Unset LEONIDA_HOST_ADAPTER means the layer's own GPU whenever its driver can run the cases: comparing with
        // another GPU's driver measures that driver (an Intel tier-4 iGPU read snapshot 1 as zeros), not the layer.
        char want[2] = {};
        const LUID host_luid = device->GetAdapterLuid();
        ComPtr<IDXGIFactory4> factory;
        if (!GetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", want, sizeof(want)) &&
            SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))) && tiled_device(*factory.Get(), layer_luid, false) &&
            (host_luid.LowPart != layer_luid.LowPart || host_luid.HighPart != layer_luid.HighPart)) {
            std::fprintf(stderr, "Native D3D12 runs on another adapter than the layer's, which has tiled resources "
                                 "tier 2 or more\n");
            return 1;
        }
        if (!run(*device.Get(), native) || native != observed) return 1;
        std::puts("Native D3D12 matched all three GPU readbacks");
    }
    return 0;
}
