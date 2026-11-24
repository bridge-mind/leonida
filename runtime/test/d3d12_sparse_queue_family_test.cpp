// Synthetic D3D12 reserved-buffer mapping on DIRECT/COPY/COMPUTE queues, with GPU readback.
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include "../shims/d3d12_com.h"
#include <wrl/client.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

using Microsoft::WRL::ComPtr;
constexpr UINT64 PAGE_BYTES = 65536, BUFFER_BYTES = 2 * PAGE_BYTES;

static void ok(HRESULT hr, const char* where) {
    if (FAILED(hr)) {
        std::printf("FAIL %s hr=0x%08lx\n", where, static_cast<unsigned long>(hr));
        throw std::runtime_error(where);
    }
}
static unsigned char pattern(size_t n) {
    return static_cast<unsigned char>((n * 131u + (n >> 7u) * 17u + 29u) & 255u);
}
static D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Alignment = 0;
    d.Width = bytes;
    d.Height = 1;
    d.DepthOrArraySize = 1;
    d.MipLevels = 1;
    d.Format = DXGI_FORMAT_UNKNOWN;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}
static ComPtr<ID3D12Resource> committed(ID3D12Device* dev, UINT64 bytes,
                                        D3D12_HEAP_TYPE type, D3D12_RESOURCE_STATES state) {
    ComPtr<ID3D12Resource> res;
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = type;
    auto desc = buffer_desc(bytes);
    ok(dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                    IID_PPV_ARGS(&res)), "CreateCommittedResource");
    return res;
}
static ComPtr<ID3D12CommandQueue> make_queue(ID3D12Device* dev, D3D12_COMMAND_LIST_TYPE type) {
    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = type;
    ComPtr<ID3D12CommandQueue> q;
    ok(dev->CreateCommandQueue(&desc, IID_PPV_ARGS(&q)), "CreateCommandQueue");
    return q;
}
struct CopyList {
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> list;
};
static CopyList copy_list(ID3D12Device* dev, D3D12_COMMAND_LIST_TYPE type,
                          ID3D12Resource* dst, ID3D12Resource* src, UINT64 bytes) {
    CopyList cl;
    ok(dev->CreateCommandAllocator(type, IID_PPV_ARGS(&cl.alloc)), "CreateCommandAllocator");
    ok(dev->CreateCommandList(0, type, cl.alloc.Get(), nullptr, IID_PPV_ARGS(&cl.list)),
       "CreateCommandList");
    cl.list->CopyBufferRegion(dst, 0, src, 0, bytes);
    ok(cl.list->Close(), "Close command list");
    return cl;
}

static bool run_case(ID3D12Device* dev, D3D12_COMMAND_LIST_TYPE map_type,
                     ID3D12CommandQueue* direct, ID3D12CommandQueue* copy,
                     ID3D12CommandQueue* compute, const char* label) {
    ID3D12CommandQueue* mapper = map_type == D3D12_COMMAND_LIST_TYPE_DIRECT ? direct :
                                  map_type == D3D12_COMMAND_LIST_TYPE_COMPUTE ? compute : copy;

    D3D12_HEAP_DESC hdesc{};
    hdesc.SizeInBytes = PAGE_BYTES;
    hdesc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    hdesc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
    hdesc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
    ComPtr<ID3D12Heap> heap;
    ok(dev->CreateHeap(&hdesc, IID_PPV_ARGS(&heap)), "CreateHeap(one tile)");
    ComPtr<ID3D12Resource> sparse;
    auto desc = buffer_desc(BUFFER_BYTES);
    ok(dev->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                   IID_PPV_ARGS(&sparse)), "CreateReservedResource(buffer)");
    UINT total_tiles = 0;
    dev->GetResourceTiling(sparse.Get(), &total_tiles, nullptr, nullptr, nullptr, 0, nullptr);
    std::printf("case=%s reserved_tiles=%u mapping_queue=%d\n", label, total_tiles, int(map_type));
    if (total_tiles != 2) throw std::runtime_error("unexpected reserved-buffer tile count");

    auto upload = committed(dev, PAGE_BYTES, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
    auto readback = committed(dev, BUFFER_BYTES, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
    unsigned char* host = nullptr;
    D3D12_RANGE empty{0, 0};
    ok(upload->Map(0, &empty, reinterpret_cast<void**>(&host)), "Upload Map");
    for (size_t i = 0; i < PAGE_BYTES; ++i) host[i] = pattern(i);
    D3D12_RANGE written{0, size_t(PAGE_BYTES)};
    upload->Unmap(0, &written);

    ComPtr<ID3D12Fence> fence;
    ok(dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)), "CreateFence");
    D3D12_TILED_RESOURCE_COORDINATE coord{};
    D3D12_TILE_REGION_SIZE region{};
    region.NumTiles = 1;
    region.UseBox = FALSE;
    D3D12_TILE_RANGE_FLAGS range_flags = D3D12_TILE_RANGE_FLAG_NONE;
    UINT heap_start = 0, range_count = 1;
    mapper->UpdateTileMappings(sparse.Get(), 1, &coord, &region, heap.Get(),
                               1, &range_flags, &heap_start, &range_count, D3D12_TILE_MAPPING_FLAG_NONE);
    ok(mapper->Signal(fence.Get(), 1), "mapper Signal 1");

    auto write = copy_list(dev, D3D12_COMMAND_LIST_TYPE_COPY, sparse.Get(), upload.Get(), PAGE_BYTES);
    ok(copy->Wait(fence.Get(), 1), "copy Wait 1");
    ID3D12CommandList* write_lists[] = {write.list.Get()};
    copy->ExecuteCommandLists(1, write_lists);
    ok(copy->Signal(fence.Get(), 2), "copy Signal 2");

    auto read = copy_list(dev, D3D12_COMMAND_LIST_TYPE_COMPUTE, readback.Get(), sparse.Get(), BUFFER_BYTES);
    ok(compute->Wait(fence.Get(), 2), "compute Wait 2");
    ID3D12CommandList* read_lists[] = {read.list.Get()};
    compute->ExecuteCommandLists(1, read_lists);
    ok(compute->Signal(fence.Get(), 3), "compute Signal 3");

    HANDLE done = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!done) throw std::runtime_error("CreateEventW failed");
    ok(fence->SetEventOnCompletion(3, done), "SetEventOnCompletion");
    DWORD wait = WaitForSingleObject(done, 15000);
    CloseHandle(done);
    if (wait != WAIT_OBJECT_0) throw std::runtime_error("GPU fence timeout");

    D3D12_RANGE read_range{0, size_t(BUFFER_BYTES)};
    unsigned char* data = nullptr;
    ok(readback->Map(0, &read_range, reinterpret_cast<void**>(&data)), "Readback Map");
    size_t wrong_mapped = 0, wrong_unmapped = 0;
    for (size_t i = 0; i < PAGE_BYTES; ++i) wrong_mapped += data[i] != pattern(i);
    for (size_t i = PAGE_BYTES; i < BUFFER_BYTES; ++i) wrong_unmapped += data[i] != 0;
    readback->Unmap(0, &empty);
    const bool passed = wrong_mapped == 0 && wrong_unmapped == 0;
    std::printf("RESULT %s %s mapped_mismatch=%zu/65536 unmapped_nonzero=%zu/65536\n",
                label, passed ? "PASS" : "FAIL", wrong_mapped, wrong_unmapped);
    return passed;
}

int main() {
    // Unbuffered: the results must reach check_sparse_queue_family.py even if a layer crashes at vkDestroyDevice.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        ComPtr<ID3D12Device> device;
        ok(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&device)),
           "Leonida create_vulkan_device_standalone");
        if (!rt::d3d12::device_sparse_buffer_tiles(device.Get())) {
            std::puts("SKIP Leonida sparse_buffer_tiles not supported");
            return 77;
        }
        std::puts("LEONIDA Vulkan-layer test; independent D3D12 API surface, direct->copy->compute queues");
        auto direct = make_queue(device.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT);
        auto copy = make_queue(device.Get(), D3D12_COMMAND_LIST_TYPE_COPY);
        auto compute = make_queue(device.Get(), D3D12_COMMAND_LIST_TYPE_COMPUTE);
        bool all_pass = true;
        all_pass &= run_case(device.Get(), D3D12_COMMAND_LIST_TYPE_DIRECT,
                             direct.Get(), copy.Get(), compute.Get(), "LAYER_MAP_DIRECT_COPY_COMPUTE");
        all_pass &= run_case(device.Get(), D3D12_COMMAND_LIST_TYPE_COPY,
                             direct.Get(), copy.Get(), compute.Get(), "LAYER_MAP_COPY_COPY_COMPUTE");
        all_pass &= run_case(device.Get(), D3D12_COMMAND_LIST_TYPE_COMPUTE,
                             direct.Get(), copy.Get(), compute.Get(), "LAYER_MAP_COMPUTE_COPY_COMPUTE");
        const HRESULT removed = device->GetDeviceRemovedReason();
        std::printf("DEVICE_REMOVED_HRESULT=0x%08lx\n", static_cast<unsigned long>(removed));
        all_pass &= SUCCEEDED(removed);
        std::printf("LAYER_FINAL %s\n", all_pass ? "PASS" : "FAIL");
        return all_pass ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "LAYER_FATAL: %s\n", e.what());
        return 2;
    }
}
