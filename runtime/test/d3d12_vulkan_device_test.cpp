// The Vulkan-backed ID3D12Device behind the Xbox-ABI device proxy, end to end, on a GPU. Everything goes through
// the proxy the way the game will call it: device methods (forwarded slots), descriptor heaps from slot 14 with
// start handles read in the Xbox shape (RAX), a texture from slot 76 with the Xbox layout word, committed
// resources, views, command lists, a queue, a fence and query heaps. Results are read back through D3D12 itself
// (CopyTextureRegion into a readback heap, then Map). GetCopyableFootprints, GetResourceAllocationInfo,
// GetResourceTiling, copies (d3d12_copy_compare.h), CheckFeatureSupport's error paths, fence semantics and the
// HRESULTs of invalid arguments to resource, heap, queue and fence creation and Map are compared with the host PC
// runtime's
// (LEONIDA_D3D49_VERBOSE=1 prints every argument case). WriteToSubresource/ReadFromSubresource run on
// devices of their own, once per copy path. No shaders or game data. Set LEONIDA_VK_VALIDATION=1 to run
// under VK_LAYER_KHRONOS_validation; any validation error fails the test.
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include "../shims/d3d12_feature_compat.h"
#include "../shims/d3d12_xs_device.h"
#include "d3d12_copy_compare.h"  // D3D-46: copies compared with the host runtime
#include "d3d12_depth24_compare.h"  // D24 clears compared with the host runtime
#include "../src/vulkan_memory.h"  // D3D-96: resizable_bar

#include <dxgi1_4.h>

#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#define EXPECT(condition) do { \
    if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return 1; } \
} while (0)

namespace {

// The Xbox call shapes measured in the prior art (docs/d3d12_vulkan.md section 7).
using XboxHeapHandleFn = std::uint64_t(WINAPI*)(void* heap);  // heap slots 9/10: 8-byte handle in RAX
using XboxCreateResourceFn = HRESULT(WINAPI*)(void* device, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
                                              const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
                                              const D3D12_CLEAR_VALUE*, const IID*, void**);  // device slot 76

template<class T>
struct Ref {
    T* p = nullptr;
    ~Ref() { reset(); }
    void reset() {
        if (p) p->Release();
        p = nullptr;
    }
    T* operator->() const { return p; }
};

void* slot(void* object, unsigned index) { return (*reinterpret_cast<void***>(object))[index]; }

D3D12_RESOURCE_DESC texture_desc(UINT width, UINT height, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags,
                                 UINT16 mips = 1, UINT16 layers = 1) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    desc.Width = width;
    desc.Height = height;
    desc.DepthOrArraySize = layers;
    desc.MipLevels = mips;
    desc.Format = format;
    desc.SampleDesc.Count = 1;
    desc.Flags = flags;
    return desc;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 bytes) {
    D3D12_RESOURCE_DESC desc{};
    desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    desc.Width = bytes;
    desc.Height = 1;
    desc.DepthOrArraySize = 1;
    desc.MipLevels = 1;
    desc.SampleDesc.Count = 1;
    desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return desc;
}

D3D12_RESOURCE_DESC volume_desc(UINT16 mips) {
    auto desc = texture_desc(8, 6, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, mips, 16);
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    return desc;
}

D3D12_RESOURCE_DESC line_desc(UINT width, UINT16 mips, UINT16 layers, DXGI_FORMAT format) {
    auto desc = texture_desc(width, 1, format, D3D12_RESOURCE_FLAG_NONE, mips, layers);
    desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
    return desc;
}

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* resource, UINT index) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = index;
    return location;
}

D3D12_TEXTURE_COPY_LOCATION footprint(ID3D12Resource* buffer, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& placed) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = buffer;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint = placed;
    return location;
}

// Places one subresource of `resource` in a buffer at the next aligned offset after `cursor`.
D3D12_PLACED_SUBRESOURCE_FOOTPRINT place(ID3D12Device* device, ID3D12Resource* resource, UINT index,
                                         UINT64& cursor) {
    const D3D12_RESOURCE_DESC desc = resource->GetDesc();
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
    UINT64 total = 0;
    const UINT64 base = (cursor + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT - 1) / D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT *
                        D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
    device->GetCopyableFootprints(&desc, index, 1, base, &layout, nullptr, nullptr, &total);
    cursor = base + total;
    return layout;
}

std::array<std::uint8_t, 4> pattern(UINT x, UINT y) {
    return {std::uint8_t(x * 16 + y), std::uint8_t(255 - y * 16), std::uint8_t(x * y), 255};
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER barrier{};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    barrier.Transition.pResource = resource;
    barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    barrier.Transition.StateBefore = before;
    barrier.Transition.StateAfter = after;
    return barrier;
}

std::uint8_t texel_byte(UINT x, UINT y, UINT z, UINT i) { return std::uint8_t(x * 31 + y * 7 + z * 61 + i * 3 + 1); }

// WriteToSubresource/ReadFromSubresource on `device`. Textures in a CPU-visible custom heap copy on the host when the
// device has VK_EXT_host_image_copy, through a staging submit when it doesn't; main runs this once per path. Covers
// zero-filled first reads, boxes with pitches unlike the texture's, a mip, a volume, BC1 blocks, a default-heap
// texture (always staged), the rejected calls, and both directions between the CPU copies and GPU copies.
int cpu_copies(ID3D12Device* device, bool bc, const char* path) {
    const D3D12_HEAP_PROPERTIES cpu_heap{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0};
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    const auto create = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                            D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
        return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr, IID_PPV_ARGS(&out.p));
    };
    unsigned bad = 0;
    const auto expect_bytes = [&](const char* what, UINT x, UINT y, UINT z, const std::uint8_t* got,
                                  const std::array<std::uint8_t, 4>& want) {
        if (std::memcmp(got, want.data(), 4) == 0) return;
        if (bad++ < 8)
            std::fprintf(stderr, "%s, %s (%u,%u,%u): got %02x %02x %02x %02x, want %02x %02x %02x %02x\n", path, what,
                         x, y, z, got[0], got[1], got[2], got[3], want[0], want[1], want[2], want[3]);
    };
    const auto texel = [](UINT x, UINT y, UINT z) {
        return std::array<std::uint8_t, 4>{texel_byte(x, y, z, 0), texel_byte(x, y, z, 1), texel_byte(x, y, z, 2),
                                           texel_byte(x, y, z, 3)};
    };
    constexpr std::array<std::uint8_t, 4> zero{};

    Ref<ID3D12Resource> tex, vol, bc1, gpu_only, buffer, upload, readback;
    EXPECT(SUCCEEDED(create(cpu_heap, texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 2),
                            D3D12_RESOURCE_STATE_COMMON, tex)));
    D3D12_RESOURCE_DESC volume = texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 4);
    volume.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    EXPECT(SUCCEEDED(create(cpu_heap, volume, D3D12_RESOURCE_STATE_COMMON, vol)));
    if (bc)
        EXPECT(SUCCEEDED(create(cpu_heap, texture_desc(8, 8, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COMMON, bc1)));
    EXPECT(SUCCEEDED(create(default_heap, texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                            D3D12_RESOURCE_STATE_COPY_SOURCE, gpu_only)));
    EXPECT(SUCCEEDED(create(cpu_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_COMMON, buffer)));
    EXPECT(SUCCEEDED(create(upload_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_GENERIC_READ, upload)));
    EXPECT(SUCCEEDED(create(readback_heap, buffer_desc(4096), D3D12_RESOURCE_STATE_COPY_DEST, readback)));

    // A texture in a CPU-visible heap maps only without a pointer; a default-heap texture not at all.
    void* pointer = nullptr;
    EXPECT(tex->Map(0, nullptr, nullptr) == S_OK);
    EXPECT(tex->Map(1, nullptr, nullptr) == S_OK);
    EXPECT(tex->Map(0, nullptr, &pointer) == E_INVALIDARG);
    EXPECT(tex->Map(2, nullptr, nullptr) == E_INVALIDARG);
    EXPECT(gpu_only->Map(0, nullptr, nullptr) == E_INVALIDARG);
    tex->Unmap(1, nullptr);

    // A new texture reads back zero-filled.
    std::vector<std::uint8_t> got(1024, 0xcd);
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 32, 0, 0, nullptr)));
    for (UINT y = 0; y < 8; ++y)
        for (UINT x = 0; x < 8; ++x) expect_bytes("new texture", x, y, 0, &got[y * 32 + x * 4], zero);

    // A 5x4 box from rows 100 bytes apart, read back whole at a 40-byte pitch.
    std::vector<std::uint8_t> src(1024, 0);
    const D3D12_BOX box{2, 1, 0, 7, 5, 1};
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 5; ++x) std::memcpy(&src[y * 100 + x * 4], texel(x + 2, y + 1, 0).data(), 4);
    EXPECT(SUCCEEDED(tex->WriteToSubresource(0, &box, src.data(), 100, 0)));
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 40, 0, 0, nullptr)));
    for (UINT y = 0; y < 8; ++y)
        for (UINT x = 0; x < 8; ++x) {
            const bool inside = x >= 2 && x < 7 && y >= 1 && y < 5;
            expect_bytes("boxed write", x, y, 0, &got[y * 40 + x * 4], inside ? texel(x, y, 0) : zero);
        }

    // Mip 1 (4x4) written whole, read back as a box.
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(&src[y * 16 + x * 4], texel(x, y, 1).data(), 4);
    EXPECT(SUCCEEDED(tex->WriteToSubresource(1, nullptr, src.data(), 16, 0)));
    const D3D12_BOX mip_box{1, 1, 0, 3, 4, 1};
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 8, 0, 1, &mip_box)));
    for (UINT y = 0; y < 3; ++y)
        for (UINT x = 0; x < 2; ++x) expect_bytes("mip 1", x + 1, y + 1, 1, &got[y * 8 + x * 4], texel(x + 1, y + 1, 1));

    // A volume: three slices of a 3x4 box, rows 12 and slices 64 bytes apart; read back whole at other pitches.
    std::fill(src.begin(), src.end(), std::uint8_t(0));
    const D3D12_BOX volume_box{1, 0, 1, 4, 4, 4};
    for (UINT z = 0; z < 3; ++z)
        for (UINT y = 0; y < 4; ++y)
            for (UINT x = 0; x < 3; ++x) std::memcpy(&src[z * 64 + y * 12 + x * 4], texel(x + 1, y, z + 1).data(), 4);
    EXPECT(SUCCEEDED(vol->WriteToSubresource(0, &volume_box, src.data(), 12, 64)));
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(vol->ReadFromSubresource(got.data(), 20, 96, 0, nullptr)));
    for (UINT z = 0; z < 4; ++z)
        for (UINT y = 0; y < 4; ++y)
            for (UINT x = 0; x < 4; ++x)
                expect_bytes("volume", x, y, z, &got[z * 96 + y * 20 + x * 4], x >= 1 && z >= 1 ? texel(x, y, z) : zero);

    // BC1: 2x2 blocks of 8 bytes written whole, then one block replaced through a box.
    if (bc1.p) {
        std::uint8_t blocks[32];
        for (UINT i = 0; i < 32; ++i) blocks[i] = std::uint8_t(0x40 + i);
        EXPECT(SUCCEEDED(bc1->WriteToSubresource(0, nullptr, blocks, 16, 0)));
        const D3D12_BOX corner{4, 4, 0, 8, 8, 1};
        const std::uint8_t replaced[8] = {0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7};
        EXPECT(SUCCEEDED(bc1->WriteToSubresource(0, &corner, replaced, 8, 0)));
        std::memcpy(blocks + 24, replaced, 8);
        std::uint8_t back[64] = {};
        EXPECT(SUCCEEDED(bc1->ReadFromSubresource(back, 32, 0, 0, nullptr)));
        for (UINT row = 0; row < 2; ++row)
            if (std::memcmp(back + row * 32, blocks + row * 16, 16) != 0 && bad++ < 8)
                std::fprintf(stderr, "%s, BC1 block row %u differs\n", path, row);
        const D3D12_BOX cut{2, 0, 0, 8, 4, 1};  // starts inside a block
        EXPECT(bc1->WriteToSubresource(0, &cut, blocks, 16, 0) == E_INVALIDARG);
    }

    // Calls D3D12 rejects, and two it doesn't: an empty box copies nothing, and pitches aren't checked (the host's
    // answers, d3d12_subresource_copy_test).
    const D3D12_BOX outside{0, 0, 0, 9, 1, 1}, empty{2, 2, 0, 2, 3, 1};
    EXPECT(tex->WriteToSubresource(0, &outside, src.data(), 100, 0) == E_INVALIDARG);
    EXPECT(tex->WriteToSubresource(0, &empty, src.data(), 100, 0) == S_OK);
    EXPECT(tex->WriteToSubresource(2, nullptr, src.data(), 100, 0) == E_INVALIDARG);  // 2 mips, 1 layer
    EXPECT(tex->WriteToSubresource(0, nullptr, nullptr, 32, 0) == E_INVALIDARG);
    EXPECT(tex->ReadFromSubresource(nullptr, 32, 0, 0, nullptr) == E_INVALIDARG);
    EXPECT(vol->WriteToSubresource(0, nullptr, src.data(), 16, 32) == S_OK);  // 64-byte slices 32 apart
    EXPECT(buffer->WriteToSubresource(0, nullptr, src.data(), 16, 0) == E_INVALIDARG);

    // A default-heap texture: D3D12 wants a CPU-visible heap, but the copy still works (and is logged).
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(&src[y * 16 + x * 4], texel(x, y, 7).data(), 4);
    EXPECT(SUCCEEDED(gpu_only->WriteToSubresource(0, nullptr, src.data(), 16, 0)));

    // GPU copies see the CPU's writes (mip 1, the default-heap texture), and the CPU sees a GPU copy's (a 4x4
    // upload into the bottom left of mip 0, which is in COMMON and promoted).
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    std::uint8_t* up = nullptr;
    std::uint8_t* down = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))));
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT square{0, {DXGI_FORMAT_R8G8B8A8_UNORM, 4, 4, 1, 256}};
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) std::memcpy(up + y * 256 + x * 4, texel(x, y + 4, 0).data(), 4);
    std::memset(down, 0xcd, 4096);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT second = square;
    second.Offset = 2048;
    {
        const auto dst = subresource(tex.p, 0);
        const auto in = footprint(upload.p, square);
        list->CopyTextureRegion(&dst, 0, 4, 0, &in, nullptr);
        const auto out = footprint(readback.p, square);
        const auto mip1 = subresource(tex.p, 1);
        list->CopyTextureRegion(&out, 0, 0, 0, &mip1, nullptr);
        const auto out2 = footprint(readback.p, second);
        const auto staged = subresource(gpu_only.p, 0);
        list->CopyTextureRegion(&out2, 0, 0, 0, &staged, nullptr);
    }
    EXPECT(list->Close() == S_OK);
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    EXPECT(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) {
            expect_bytes("GPU copy of mip 1", x, y, 1, down + y * 256 + x * 4, texel(x, y, 1));
            expect_bytes("GPU copy of the default-heap texture", x, y, 7, down + 2048 + y * 256 + x * 4, texel(x, y, 7));
        }
    const D3D12_BOX bottom_left{0, 4, 0, 4, 8, 1};
    std::fill(got.begin(), got.end(), std::uint8_t(0xcd));
    EXPECT(SUCCEEDED(tex->ReadFromSubresource(got.data(), 16, 0, 0, &bottom_left)));
    for (UINT y = 0; y < 4; ++y)
        for (UINT x = 0; x < 4; ++x) expect_bytes("CPU read of a GPU upload", x, y + 4, 0, &got[y * 16 + x * 4], texel(x, y + 4, 0));
    upload->Unmap(0, nullptr);
    readback->Unmap(0, nullptr);
    EXPECT(!bad);
    std::printf("WriteToSubresource/ReadFromSubresource (%s): boxes, pitches, a mip, a volume%s, rejected calls and "
                "GPU copies both ways matched\n", path, bc1.p ? ", BC1" : "");
    return 0;
}

// The host PC runtime's D3D12 device to compare answers with: the layer's own adapter (set in main), else the default
// one, or the one LEONIDA_HOST_ADAPTER names ("warp", or part of an adapter's name); false if there is none.
// Under Wine there is none (copycmp::create_host).
bool create_host_device(Ref<ID3D12Device>& host, std::string* adapter_name = nullptr) {
    copycmp::Ptr<ID3D12Device> device;
    std::string name;
    if (!copycmp::create_host(device, name)) return false;
    if (adapter_name) *adapter_name = name;
    host.p = device.p;
    device.p = nullptr;
    return true;
}

bool block_compressed(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_BC1_UNORM:
    case DXGI_FORMAT_BC1_UNORM_SRGB:
    case DXGI_FORMAT_BC2_UNORM:
    case DXGI_FORMAT_BC2_UNORM_SRGB:
    case DXGI_FORMAT_BC3_UNORM:
    case DXGI_FORMAT_BC3_UNORM_SRGB:
    case DXGI_FORMAT_BC4_UNORM:
    case DXGI_FORMAT_BC4_SNORM:
    case DXGI_FORMAT_BC5_UNORM:
    case DXGI_FORMAT_BC5_SNORM:
    case DXGI_FORMAT_BC6H_UF16:
    case DXGI_FORMAT_BC6H_SF16:
    case DXGI_FORMAT_BC7_UNORM:
    case DXGI_FORMAT_BC7_UNORM_SRGB: return true;
    default: return false;
    }
}

// A block-compressed texture whose top mip is not a whole number of 4x4 blocks. The PC runtime only answers
// GetCopyableFootprints for these when D3D12_OPTIONS8.UnalignedBlockTexturesSupported is TRUE (Windows 10 does
// not); it returns -1 for every field otherwise.
bool uses_unaligned_blocks(const D3D12_RESOURCE_DESC& desc) {
    return block_compressed(desc.Format) && (desc.Width % 4 != 0 || desc.Height % 4 != 0);
}

bool host_supports_unaligned_block_textures(ID3D12Device* host) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS8 options{};
    return SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &options, sizeof(options))) &&
           options.UnalignedBlockTexturesSupported;
}

// ResolveSubresourceRegion is on ID3D12GraphicsCommandList7, which the Windows 10 runtime does not have.
bool host_has_graphics_command_list7(ID3D12Device* host) {
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList7> list7;
    if (FAILED(host->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))))
        return false;
    if (FAILED(host->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                       IID_PPV_ARGS(&list.p))))
        return false;
    if (FAILED(list->Close())) return false;
    return SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p)));
}

int compare_format_info(ID3D12Device* ours, const vktest::Device& vk) {
    VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
    vk.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(vk.physical, &properties);
    LUID luid{};
    static_assert(sizeof(luid) == sizeof(id.deviceLUID));
    std::memcpy(&luid, id.deviceLUID, sizeof(luid));
    copycmp::Ptr<ID3D12Device> host;
    std::string adapter_name;
    if (!id.deviceLUIDValid || !copycmp::try_create_host(host, adapter_name, &luid)) {
        std::printf("FORMAT_INFO: no native D3D12 device on the Vulkan adapter, skipped comparison\n");
        return 0;
    }

    unsigned compared = 0, host_unknown = 0;
    int mismatches = 0;
    const auto compare = [&](DXGI_FORMAT format, UINT size) {
        D3D12_FEATURE_DATA_FORMAT_INFO expected{format, 0xff}, actual = expected;
        HRESULT expected_hr = host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &expected, size);
        // Some Windows runtimes predate these single-plane formats. Still require the layer to recognize them.
        if (expected_hr == E_INVALIDARG && size == sizeof(expected) &&
            (format == DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE ||
             format == DXGI_FORMAT_SAMPLER_FEEDBACK_MIP_REGION_USED_OPAQUE || format == DXGI_FORMAT_A4B4G4R4_UNORM)) {
            expected_hr = S_OK;
            expected.PlaneCount = 1;
            ++host_unknown;
            std::printf("FORMAT_INFO format %u: host returned E_INVALIDARG, checking for one plane\n", UINT(format));
        }
        const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, &actual, size);
        ++compared;
        if (our_hr != expected_hr || actual.PlaneCount != expected.PlaneCount || actual.Format != format) {
            std::fprintf(stderr, "FORMAT_INFO format %u size %u: HRESULT/planes %08lx/%u, expected %08lx/%u\n",
                         UINT(format), size, static_cast<unsigned long>(our_hr), UINT(actual.PlaneCount),
                         static_cast<unsigned long>(expected_hr), UINT(expected.PlaneCount));
            ++mismatches;
        }
    };
    constexpr UINT size = sizeof(D3D12_FEATURE_DATA_FORMAT_INFO);
    // Include UNKNOWN, holes between format families, and the value after the last named format.
    for (UINT format = 0; format <= UINT(DXGI_FORMAT_A4B4G4R4_UNORM) + 1; ++format)
        compare(static_cast<DXGI_FORMAT>(format), size);
    compare(DXGI_FORMAT_FORCE_UINT, size);
    compare(DXGI_FORMAT_D24_UNORM_S8_UINT, size - 1);
    std::printf("FORMAT_INFO on %s: %u queries, %u newer formats unknown to host, %d mismatches\n",
                adapter_name.c_str(), compared, host_unknown, mismatches);
    return mismatches;
}

// D3D-122: descriptions GetCopyableFootprints must refuse (false) or answer (true), as the PC runtime does. All were
// measured on WARP. Also on an NVIDIA GPU: texture Width/Height 0, Count 0, Count 4 without a render-target or depth
// flag, Count 3; buffer Width 0, Height 2, Format, MipLevels 0, Layout, Count 0; and the buffer render-target, extra
// mip, ROW_MAJOR and Quality 1 controls. compare_footprints checks them against the host, the main test on their own.
std::vector<std::pair<D3D12_RESOURCE_DESC, bool>> footprint_validity_cases() {
    const auto rgba = DXGI_FORMAT_R8G8B8A8_UNORM;
    const auto rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    std::vector<std::pair<D3D12_RESOURCE_DESC, bool>> cases;
    const auto add = [&](D3D12_RESOURCE_DESC d, bool valid, auto&& change) {
        change(d);
        cases.push_back({d, valid});
    };
    const auto buffer = buffer_desc(256);
    add(buffer, false, [](auto& d) { d.Width = 0; });
    add(buffer, false, [](auto& d) { d.Height = 2; });
    add(buffer, false, [](auto& d) { d.DepthOrArraySize = 2; });
    add(buffer, false, [](auto& d) { d.MipLevels = 0; });
    add(buffer, false, [](auto& d) { d.MipLevels = 2; });
    add(buffer, false, [&](auto& d) { d.Format = rgba; });
    add(buffer, false, [](auto& d) { d.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN; });
    add(buffer, false, [](auto& d) { d.SampleDesc.Count = 0; });
    add(buffer, false, [](auto& d) { d.SampleDesc.Count = 2; });
    add(buffer, false, [](auto& d) { d.SampleDesc.Quality = 1; });
    add(buffer, false, [](auto& d) { d.Alignment = 4096; });
    add(buffer, true, [&](auto& d) { d.Flags = rt; });
    add(buffer, true, [](auto& d) { d.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT; });

    const auto texture = texture_desc(16, 16, rgba, D3D12_RESOURCE_FLAG_NONE);
    add(texture, false, [](auto& d) { d.Width = 0; });
    add(texture, false, [](auto& d) { d.Height = 0; });
    add(texture, false, [](auto& d) { d.DepthOrArraySize = 0; });
    add(texture, false, [](auto& d) { d.SampleDesc.Count = 0; });
    add(texture, false, [](auto& d) { d.SampleDesc.Count = 4; });  // no render-target or depth flag
    add(texture, false, [](auto& d) { d.SampleDesc.Count = 4, d.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS; });
    add(texture, false, [&](auto& d) { d.SampleDesc.Count = 3, d.Flags = rt; });
    add(texture, false, [&](auto& d) { d.SampleDesc.Count = 4, d.Flags = rt, d.MipLevels = 2; });
    add(texture, false, [&](auto& d) { d.SampleDesc.Count = 4, d.Flags = rt, d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; });
    add(texture, true, [&](auto& d) { d.SampleDesc.Count = 4, d.Flags = rt; });
    add(texture, true, [](auto& d) { d.MipLevels = 9; });  // past the full chain
    add(texture, true, [](auto& d) { d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; });
    add(texture, true, [](auto& d) { d.SampleDesc.Quality = 1; });
    add(texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), true,
        [](auto& d) { d.SampleDesc.Count = 4; });

    add(line_desc(16, 1, 1, rgba), false, [](auto& d) { d.Width = 0; });
    add(line_desc(16, 1, 1, rgba), false, [&](auto& d) { d.SampleDesc.Count = 2, d.Flags = rt; });
    add(volume_desc(1), false, [](auto& d) { d.DepthOrArraySize = 0; });
    add(volume_desc(1), false, [](auto& d) { d.Height = 0; });
    add(volume_desc(1), false, [&](auto& d) { d.SampleDesc.Count = 2, d.Flags = rt; });
    return cases;
}

// GetCopyableFootprints must match the PC runtime's exactly, since applications size and fill upload buffers from
// it. Unaligned block-compressed cases are skipped when the host runtime cannot answer them, so the rest still
// run. Returns the number of mismatches, or -1 when the host has no D3D12 device to compare with.
int compare_footprints(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const bool unaligned_blocks = host_supports_unaligned_block_textures(host.p);
    unsigned skipped = 0;
    struct Case {
        D3D12_RESOURCE_DESC desc;
        UINT subresources;
    };
    std::vector<Case> cases = {
        {volume_desc(5), 5},
        {line_desc(13, 4, 1, DXGI_FORMAT_R8G8B8A8_UNORM), 4},
        {line_desc(5, 3, 2, DXGI_FORMAT_R8G8B8A8_UNORM), 6},
        {buffer_desc(300), 1},
        {texture_desc(8, 8, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_R9G9B9E5_SHAREDEXP, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_B5G5R5A1_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_R16G16B16A16_FLOAT, D3D12_RESOURCE_FLAG_NONE, 4), 4},
        {texture_desc(13, 7, DXGI_FORMAT_B8G8R8X8_UNORM, D3D12_RESOURCE_FLAG_NONE, 4, 2), 8},
        {texture_desc(13, 7, DXGI_FORMAT_B8G8R8X8_UNORM_SRGB, D3D12_RESOURCE_FLAG_NONE, 4), 4},
        {texture_desc(16, 16, DXGI_FORMAT_B8G8R8X8_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(10, 6, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE, 4), 4},
        {texture_desc(4, 4, DXGI_FORMAT_BC7_UNORM_SRGB, D3D12_RESOURCE_FLAG_NONE, 1, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_BC5_UNORM, D3D12_RESOURCE_FLAG_NONE, 5), 5},
        {texture_desc(8, 8, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 2), 4},
        {texture_desc(8, 8, DXGI_FORMAT_D24_UNORM_S8_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 2, 2), 8},
        {texture_desc(300, 5, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), 2},
        {texture_desc(8, 8, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL), 1},
        {texture_desc(8, 8, DXGI_FORMAT_D16_UNORM, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 4), 4},
        {texture_desc(6, 6, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_NONE), 1},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R32G32_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R10G10B10A2_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, D3D12_RESOURCE_FLAG_NONE, 1), 1},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R16G16_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_TYPELESS, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_UINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        {texture_desc(16, 16, DXGI_FORMAT_R8G8_SINT, D3D12_RESOURCE_FLAG_NONE, 3), 3},
        // Mip counts past 32 (and past 64 for the width): D3D12 accepts any count, every mip past the chain is 1x1.
        {texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 40), 40},
        {line_desc(16, 70, 1, DXGI_FORMAT_R8G8B8A8_UNORM), 70},
        {volume_desc(40), 40},
    };
    for (const auto& c : footprint_validity_cases()) cases.push_back({c.first, 1});
    int mismatches = 0;
    for (const Case& c : cases) {
        if (!unaligned_blocks && uses_unaligned_blocks(c.desc)) {
            ++skipped;
            continue;
        }
        for (const UINT64 base : {UINT64(0), UINT64(1024)}) {
            std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> a(c.subresources), b(c.subresources);
            std::vector<UINT> rows_a(c.subresources), rows_b(c.subresources);
            std::vector<UINT64> size_a(c.subresources), size_b(c.subresources);
            UINT64 total_a = 0, total_b = 0;
            ours->GetCopyableFootprints(&c.desc, 0, c.subresources, base, a.data(), rows_a.data(), size_a.data(),
                                        &total_a);
            host->GetCopyableFootprints(&c.desc, 0, c.subresources, base, b.data(), rows_b.data(), size_b.data(),
                                        &total_b);
            for (UINT i = 0; i < c.subresources; ++i) {
                const auto &x = a[i], &y = b[i];
                if (x.Offset == y.Offset && x.Footprint.Format == y.Footprint.Format &&
                    x.Footprint.Width == y.Footprint.Width && x.Footprint.Height == y.Footprint.Height &&
                    x.Footprint.Depth == y.Footprint.Depth && x.Footprint.RowPitch == y.Footprint.RowPitch &&
                    rows_a[i] == rows_b[i] && size_a[i] == size_b[i])
                    continue;
                if (mismatches++ < 24)
                    std::fprintf(stderr,
                                 "footprint format %d base %llu subresource %u: ours offset %llu fmt %d %ux%ux%u pitch %u "
                                 "rows %u row %llu; host offset %llu fmt %d %ux%ux%u pitch %u rows %u row %llu\n",
                                 int(c.desc.Format), (unsigned long long)base, i, (unsigned long long)x.Offset,
                                 int(x.Footprint.Format), x.Footprint.Width, x.Footprint.Height, x.Footprint.Depth,
                                 x.Footprint.RowPitch, rows_a[i], (unsigned long long)size_a[i],
                                 (unsigned long long)y.Offset, int(y.Footprint.Format), y.Footprint.Width,
                                 y.Footprint.Height, y.Footprint.Depth, y.Footprint.RowPitch, rows_b[i],
                                 (unsigned long long)size_b[i]);
            }
            if (total_a != total_b && mismatches++ < 8)
                std::fprintf(stderr, "footprint format %d base %llu: total ours %llu host %llu\n", int(c.desc.Format),
                             (unsigned long long)base, (unsigned long long)total_a, (unsigned long long)total_b);
        }
    }
    if (skipped)
        std::printf("GetCopyableFootprints: %u unaligned block-compressed cases skipped (the host runtime has no "
                    "D3D12_OPTIONS8.UnalignedBlockTexturesSupported)\n", skipped);
    return mismatches;
}

// GetResourceAllocationInfo against the PC runtime. Buffer sizes, every alignment and which descriptions are invalid
// must match it exactly. A texture's size is the driver's choice on both sides, so ours is only checked to be a
// whole number of alignments and at least the texels of its top mip; both sizes are printed. Also checks the
// Device4/Device8 variants' per-resource layout. Returns the number of mismatches, or -1 without a host device.
int compare_allocation_info(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const auto with = [](D3D12_RESOURCE_DESC desc, UINT64 alignment, UINT samples = 1) {
        desc.Alignment = alignment;
        desc.SampleDesc.Count = samples;
        return desc;
    };
    constexpr UINT64 k4 = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT, k64 = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
    constexpr D3D12_RESOURCE_FLAGS none = D3D12_RESOURCE_FLAG_NONE, rt = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    constexpr DXGI_FORMAT rgba8 = DXGI_FORMAT_R8G8B8A8_UNORM;
    struct Case {
        const char* name;
        D3D12_RESOURCE_DESC desc;
        UINT64 top_mip_bytes;  // 0: the size must match the host's exactly (buffers, invalid descriptions)
    };
    D3D12_RESOURCE_DESC bad_buffer = buffer_desc(256);
    bad_buffer.Height = 2;
    D3D12_RESOURCE_DESC typed_buffer = buffer_desc(256);
    typed_buffer.Format = DXGI_FORMAT_R8_UNORM;
    D3D12_RESOURCE_DESC big_volume = volume_desc(1);
    big_volume.Width = big_volume.Height = 32;
    big_volume.DepthOrArraySize = 8;
    D3D12_RESOURCE_DESC bad_line = line_desc(8, 1, 1, rgba8);
    bad_line.Height = 2;
    D3D12_RESOURCE_DESC line_rt = line_desc(8, 1, 1, rgba8);
    line_rt.Flags = rt;
    const auto small_volume = [&](UINT width, UINT height, UINT16 depth, DXGI_FORMAT format) {
        D3D12_RESOURCE_DESC desc = with(texture_desc(width, height, format, none, 1, depth), k4);
        desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        return desc;
    };
    const Case cases[] = {
        {"buffer 300", buffer_desc(300), 0},
        {"buffer 64 KiB", buffer_desc(65536), 0},
        {"buffer 64 KiB + 1", buffer_desc(65537), 0},
        {"buffer aligned 64 KiB", with(buffer_desc(1000), k64), 0},
        {"buffer width 0", buffer_desc(0), 0},
        {"buffer height 2", bad_buffer, 0},
        {"buffer with a format", typed_buffer, 0},
        // Resource flag handling (D3D-467).
        {"buffer render target", [&] {
             auto d = buffer_desc(256);
             d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
             return d;
         }(), 0},
        {"buffer render target UAV", [&] {
             auto d = buffer_desc(256);
             d.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return d;
         }(), 0},
        {"buffer cross adapter", [&] {
             auto d = buffer_desc(256);
             d.Flags = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER;
             return d;
         }(), 0},
        {"buffer depth stencil (invalid)", [&] {
             auto d = buffer_desc(256);
             d.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
             return d;
         }(), 0},
        {"buffer simultaneous access (invalid)", [&] {
             auto d = buffer_desc(256);
             d.Flags = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS;
             return d;
         }(), 0},
        {"buffer video decode reference only (invalid)", [&] {
             auto d = buffer_desc(256);
             d.Flags = D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY;
             return d;
         }(), 0},
        {"buffer flag 0x1000 (invalid)", [&] {
             auto d = buffer_desc(256);
             d.Flags = D3D12_RESOURCE_FLAGS(0x1000);
             return d;
         }(), 0},
        {"tex2d deny shader resource alone (invalid)", [&] {
             auto d = texture_desc(16, 16, rgba8, D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
             return d;
         }(), 0},
        {"tex2d single sample quality 1", [&] {
             auto d = texture_desc(16, 16, rgba8, none);
             d.SampleDesc.Quality = 1;
             return d;
         }(), 16 * 16 * 4},
        {"tex2d single sample quality center pattern", [&] {
             auto d = texture_desc(16, 16, rgba8, none);
             d.SampleDesc.Quality = 0xffffffff;
             return d;
         }(), 16 * 16 * 4},
        {"256x256 full chain", texture_desc(256, 256, rgba8, none, 0), 256 * 256 * 4},
        {"16x16 small", with(texture_desc(16, 16, rgba8, none), k4), 16 * 16 * 4},
        {"16x16 small render target", with(texture_desc(16, 16, rgba8, rt), k4), 16 * 16 * 4},
        {"1024x1024 small (too big)", with(texture_desc(1024, 1024, rgba8, none), k4), 1024 * 1024 * 4},
        {"64x64 BC1 small", with(texture_desc(64, 64, DXGI_FORMAT_BC1_UNORM, none, 3), k4), 16 * 16 * 8},
        // 4 KiB alignment counts the 4 KiB tiles of one slice's top mip (at most 16), not the whole resource.
        {"128x128 full chain small", with(texture_desc(128, 128, rgba8, none, 0), k4), 128 * 128 * 4},
        {"64x64 array of 8 small", with(texture_desc(64, 64, rgba8, none, 1, 8), k4), 64 * 64 * 4 * 8},
        {"16x16 array of 256 small", with(texture_desc(16, 16, rgba8, none, 1, 256), k4), 16 * 16 * 4 * 256},
        {"256x64 full chain small", with(texture_desc(256, 64, rgba8, none, 0), k4), 256 * 64 * 4},
        {"1D 16384 small", with(line_desc(16384, 1, 1, rgba8), k4), 16384 * 4},
        {"R8 32x32x64 volume small", small_volume(32, 32, 64, DXGI_FORMAT_R8_UNORM), 32 * 32 * 64},
        {"R16 16x16x128 volume small", small_volume(16, 16, 128, DXGI_FORMAT_R16_FLOAT), 16 * 16 * 128 * 2},
        {"33x480 small (30 tiles)", with(texture_desc(33, 480, rgba8, none), k4), 33 * 480 * 4},
        {"128x129 small (20 tiles)", with(texture_desc(128, 129, rgba8, none), k4), 128 * 129 * 4},
        {"R8 65x1000 small (32 tiles)", with(texture_desc(65, 1000, DXGI_FORMAT_R8_UNORM, none), k4), 65 * 1000},
        {"100x100 volume small (49 tiles)", small_volume(100, 100, 1, rgba8), 100 * 100 * 4},
        {"R16 16x8x256 volume small (32 tiles)", small_volume(16, 8, 256, DXGI_FORMAT_R16_FLOAT), 16 * 8 * 256 * 2},
        // One shape per element size, each exactly 16 tiles along one axis, then one texel more.
        {"R8 16x256x16 volume small", small_volume(16, 256, 16, DXGI_FORMAT_R8_UNORM), 16 * 256 * 16},
        {"R8 16x257x16 volume small (17 tiles)", small_volume(16, 257, 16, DXGI_FORMAT_R8_UNORM), 16 * 257 * 16},
        {"R16 256x16x8 volume small", small_volume(256, 16, 8, DXGI_FORMAT_R16_FLOAT), 256 * 16 * 8 * 2},
        {"R16 257x16x8 volume small (17 tiles)", small_volume(257, 16, 8, DXGI_FORMAT_R16_FLOAT), 257 * 16 * 8 * 2},
        {"rgba8 256x8x8 volume small", small_volume(256, 8, 8, rgba8), 256 * 8 * 8 * 4},
        {"rgba8 257x8x8 volume small (17 tiles)", small_volume(257, 8, 8, rgba8), 257 * 8 * 8 * 4},
        {"rgba16 8x128x8 volume small", small_volume(8, 128, 8, DXGI_FORMAT_R16G16B16A16_FLOAT), 8 * 128 * 8 * 8},
        {"rgba16 8x129x8 volume small (17 tiles)", small_volume(8, 129, 8, DXGI_FORMAT_R16G16B16A16_FLOAT),
         8 * 129 * 8 * 8},
        {"rgba32 128x8x4 volume small", small_volume(128, 8, 4, DXGI_FORMAT_R32G32B32A32_FLOAT), 128 * 8 * 4 * 16},
        {"rgba32 129x8x4 volume small (17 tiles)", small_volume(129, 8, 4, DXGI_FORMAT_R32G32B32A32_FLOAT),
         129 * 8 * 4 * 16},
        {"BC1 32x512x8 volume small", small_volume(32, 512, 8, DXGI_FORMAT_BC1_UNORM), 8 * 128 * 8 * 8},
        {"BC1 32x516x8 volume small (17 tiles)", small_volume(32, 516, 8, DXGI_FORMAT_BC1_UNORM), 8 * 129 * 8 * 8},
        // Simultaneous access is for single-sampled color only; a volume cannot be a depth buffer.
        {"depth with simultaneous access",
         texture_desc(64, 64, DXGI_FORMAT_D32_FLOAT,
                      D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS), 0},
        {"4x MSAA with simultaneous access",
         with(texture_desc(64, 64, rgba8, rt | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS), 0, 4), 0},
        {"depth-stencil volume", [&] {
             D3D12_RESOURCE_DESC desc = small_volume(8, 8, 4, DXGI_FORMAT_D32_FLOAT);
             desc.Alignment = 0;
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
             return desc;
         }(), 0},
        {"64x64 depth", texture_desc(64, 64, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
         64 * 64 * 4},
        {"8x8 array of 6", texture_desc(8, 8, rgba8, none, 4, 6), 8 * 8 * 4 * 6},
        {"32x32x8 volume", big_volume, 32 * 32 * 8 * 4},
        {"1D 13 full chain", line_desc(13, 4, 1, rgba8), 13 * 4},
        {"1D array 5x3", line_desc(5, 3, 2, rgba8), 5 * 4 * 2},
        {"1D height 2", bad_line, 0},
        {"1D render target", line_rt, 8 * 4},
        {"64x64 4x MSAA", with(texture_desc(64, 64, rgba8, rt), 0, 4), 64 * 64 * 4 * 4},
        {"64x64 32x MSAA", with(texture_desc(64, 64, rgba8, rt), 0, 32), 64 * 64 * 4 * 32},
        {"64x64 64x MSAA (invalid)", with(texture_desc(64, 64, rgba8, rt), 0, 64), 0},
        {"64x64 zero sample count (invalid)", with(texture_desc(64, 64, rgba8, rt), 0, 0), 0},
        {"64x64 4x MSAA small", with(texture_desc(64, 64, rgba8, rt), k64, 4), 64 * 64 * 4 * 4},
        {"1024x1024 4x MSAA small", with(texture_desc(1024, 1024, rgba8, rt), k64, 4), 1024 * 1024 * 4 * 4},
        {"zero width", texture_desc(0, 16, rgba8, none), 0},
        {"alignment 12345", with(texture_desc(16, 16, rgba8, none), 12345), 0},
        {"4x MSAA with 2 mips", with(texture_desc(64, 64, rgba8, rt, 2), 0, 4), 0},
        {"16x16 R32G32_UINT", texture_desc(16, 16, DXGI_FORMAT_R32G32_UINT, none, 1), 16 * 16 * 8},
        {"16x16 R32G32_SINT", texture_desc(16, 16, DXGI_FORMAT_R32G32_SINT, none, 1), 16 * 16 * 8},
        {"16x16 R32G32_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R32G32_TYPELESS, none, 1), 16 * 16 * 8},
        {"16x16 R16G16B16A16_UINT", texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_UINT, none, 1), 16 * 16 * 8},
        {"16x16 R16G16B16A16_SINT", texture_desc(16, 16, DXGI_FORMAT_R16G16B16A16_SINT, none, 1), 16 * 16 * 8},
        {"16x16 R10G10B10A2_UINT", texture_desc(16, 16, DXGI_FORMAT_R10G10B10A2_UINT, none, 1), 16 * 16 * 4},
        {"16x16 R10G10B10_XR_BIAS_A2_UNORM",
         texture_desc(16, 16, DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R16G16_TYPELESS, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_UINT", texture_desc(16, 16, DXGI_FORMAT_R16G16_UINT, none, 1), 16 * 16 * 4},
        {"16x16 R16G16_SINT", texture_desc(16, 16, DXGI_FORMAT_R16G16_SINT, none, 1), 16 * 16 * 4},
        {"16x16 R8G8_TYPELESS", texture_desc(16, 16, DXGI_FORMAT_R8G8_TYPELESS, none, 1), 16 * 16 * 2},
        {"16x16 R8G8_UINT", texture_desc(16, 16, DXGI_FORMAT_R8G8_UINT, none, 1), 16 * 16 * 2},
        {"16x16 R8G8_SINT", texture_desc(16, 16, DXGI_FORMAT_R8G8_SINT, none, 1), 16 * 16 * 2},
        // Tiled (64KB_UNDEFINED_SWIZZLE) textures: 64 KiB alignment and tile-multiple size; 1D, buffers, 4 KiB
        // alignment, untileable 96-bit and two-plane depth-stencil formats are invalid.
        {"tiled 2D 256x256", [&] {
             D3D12_RESOURCE_DESC d = texture_desc(256, 256, rgba8, none, 1);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 256 * 256 * 4},
        {"tiled 2D D32F 256x256", [&] {
             D3D12_RESOURCE_DESC d = texture_desc(256, 256, DXGI_FORMAT_D32_FLOAT,
                                                  D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 256 * 256 * 4},
        {"tiled 3D 64x64x64", [&] {
             D3D12_RESOURCE_DESC d = small_volume(64, 64, 64, rgba8);
             d.Alignment = 0;
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 64 * 64 * 64 * 4},
        {"tiled buffer (invalid)", [&] {
             D3D12_RESOURCE_DESC d = buffer_desc(4096);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 1D (invalid)", [&] {
             D3D12_RESOURCE_DESC d = line_desc(256, 1, 1, rgba8);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D 4K alignment (invalid)", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(16, 16, rgba8, none, 1), k4);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D RGB32F (invalid)", [&] {
             D3D12_RESOURCE_DESC d = texture_desc(64, 64, DXGI_FORMAT_R32G32B32_FLOAT, none, 1);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D D24S8 (invalid)", [&] {
             D3D12_RESOURCE_DESC d = texture_desc(64, 64, DXGI_FORMAT_D24_UNORM_S8_UINT,
                                                  D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D 4x MSAA RGBA8", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, rgba8, rt, 1), 0, 4);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 256 * 256 * 4 * 4},
        {"tiled 2D 4x MSAA RGBA32F", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, DXGI_FORMAT_R32G32B32A32_FLOAT, rt, 1), 0, 4);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 256 * 256 * 16 * 4},
        {"tiled 2D 4x MSAA D32F", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, DXGI_FORMAT_D32_FLOAT,
                                                       D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1), 0, 4);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 256 * 256 * 4 * 4},
        {"tiled 2D 2x MSAA (invalid)", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, rgba8, rt, 1), 0, 2);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D 8x MSAA (invalid)", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, rgba8, rt, 1), 0, 8);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D 4x MSAA NONE (invalid)", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, rgba8, none, 1), 0, 4);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D 4x MSAA RT+DS (invalid)", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, rgba8,
                                                       rt | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1), 0, 4);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"tiled 2D 4x MSAA 4M alignment (invalid)", [&] {
             D3D12_RESOURCE_DESC d = with(texture_desc(256, 256, rgba8, rt, 1),
                                          D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT, 4);
             d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
             return d;
         }(), 0},
        {"UNKNOWN 4x MSAA NONE (invalid)", with(texture_desc(256, 256, rgba8, none, 1), 0, 4), 0},
        {"UNKNOWN 4x MSAA RT+DS (invalid)",
         with(texture_desc(256, 256, rgba8, rt | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1), 0, 4), 0},
    };
    int mismatches = 0;
    const auto mismatch = [&](const char* name, const D3D12_RESOURCE_ALLOCATION_INFO& a,
                              const D3D12_RESOURCE_ALLOCATION_INFO& b) {
        if (mismatches++ < 24)
            std::fprintf(stderr, "allocation info %s: ours size %llu alignment %llu, host size %llu alignment %llu\n",
                         name, (unsigned long long)a.SizeInBytes, (unsigned long long)a.Alignment,
                         (unsigned long long)b.SizeInBytes, (unsigned long long)b.Alignment);
    };
    for (const Case& c : cases) {
        const D3D12_RESOURCE_ALLOCATION_INFO a = ours->GetResourceAllocationInfo(0, 1, &c.desc);
        const D3D12_RESOURCE_ALLOCATION_INFO b = host->GetResourceAllocationInfo(0, 1, &c.desc);
        if (c.top_mip_bytes && b.SizeInBytes == UINT64_MAX && c.desc.SampleDesc.Count > 1 &&
            c.desc.Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS ms{
                c.desc.Format, c.desc.SampleDesc.Count, D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE, 0};
            if (SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &ms, sizeof(ms))) &&
                ms.NumQualityLevels == 0) {
                if (a.SizeInBytes == UINT64_MAX || a.Alignment != D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT)
                    mismatch(c.name, a, b);
                std::printf("allocation info %s: host unsupported (NumQualityLevels=0), skipped\n", c.name);
                continue;
            }
        }
        if (b.SizeInBytes == UINT64_MAX || !c.top_mip_bytes) {
            if (a.SizeInBytes != b.SizeInBytes || (b.SizeInBytes != UINT64_MAX && a.Alignment != b.Alignment))
                mismatch(c.name, a, b);
            continue;
        }
        if (a.Alignment != b.Alignment || a.SizeInBytes == UINT64_MAX || a.SizeInBytes % a.Alignment ||
            a.SizeInBytes < c.top_mip_bytes)
            mismatch(c.name, a, b);
        std::printf("allocation info %s: alignment %llu, size ours %llu host %llu\n", c.name,
                    (unsigned long long)a.Alignment, (unsigned long long)a.SizeInBytes,
                    (unsigned long long)b.SizeInBytes);
    }

    // Several resources at once: each at its own alignment after the one before, the largest alignment overall.
    // Buffers only for the host comparison (their sizes are fixed); the texture checks our own layout.
    const D3D12_RESOURCE_DESC buffers[] = {buffer_desc(300), buffer_desc(70000), buffer_desc(1)};
    const D3D12_RESOURCE_ALLOCATION_INFO a = ours->GetResourceAllocationInfo(0, 3, buffers);
    const D3D12_RESOURCE_ALLOCATION_INFO b = host->GetResourceAllocationInfo(0, 3, buffers);
    if (a.SizeInBytes != b.SizeInBytes || a.Alignment != b.Alignment) mismatch("three buffers", a, b);
    // The total is rounded up to the largest alignment, saturates rather than wraps, and is aligned 1 when empty.
    // The textures here are smaller than their alignment, so their sizes are fixed too.
    const D3D12_RESOURCE_DESC to_total[][2] = {
        {buffer_desc(300), with(texture_desc(16, 16, rgba8, none), k4)},
        {with(texture_desc(64, 64, rgba8, rt), 0, 4), buffer_desc(300)},
        {buffer_desc(1ull << 63), buffer_desc(1ull << 63)},
        {buffer_desc(UINT64_MAX - k64), buffer_desc(1)},
    };
    const char* total_names[] = {"buffer + 4 KiB texture", "4x MSAA + buffer", "two 2^63 buffers",
                                 "largest buffer + buffer"};
    // The last two overflow 2^64. Hosts disagree there: WARP and the RTX 5070/4070 SUPER hosts of #1204 saturate at
    // 0xFFFFFFFFFFFF0000, while the Intel Raptor Lake-S (101.7092) and RTX 5060 Laptop (621.24) hosts on Windows
    // 26300.9974 (D3D12Core 10.0.26100.9818) wrap, to 0 and 65536, and WARP on that same PC saturates. Ours is pinned
    // to the saturated answer, which can't make a caller allocate less than one of its resources; a host whose total
    // is below its largest resource has wrapped, and is reported rather than compared.
    constexpr UINT64 saturated = UINT64_MAX / k64 * k64;
    for (int i = 0; i < 4; ++i) {
        const D3D12_RESOURCE_ALLOCATION_INFO x = ours->GetResourceAllocationInfo(0, 2, to_total[i]);
        const D3D12_RESOURCE_ALLOCATION_INFO y = host->GetResourceAllocationInfo(0, 2, to_total[i]);
        if (i >= 2) {
            if (x.SizeInBytes != saturated || x.Alignment != k64) mismatch(total_names[i], x, y);
            else if (y.SizeInBytes < to_total[i][0].Width && y.Alignment == k64)
                std::printf("allocation info %s: host wraps to %llu, ours saturates\n", total_names[i],
                            (unsigned long long)y.SizeInBytes);
            else if (y.SizeInBytes != x.SizeInBytes || y.Alignment != x.Alignment)
                mismatch(total_names[i], x, y);
            continue;
        }
        if (x.SizeInBytes != y.SizeInBytes || x.Alignment != y.Alignment) mismatch(total_names[i], x, y);
    }
    const D3D12_RESOURCE_ALLOCATION_INFO empty_a = ours->GetResourceAllocationInfo(0, 0, nullptr);
    const D3D12_RESOURCE_ALLOCATION_INFO empty_b = host->GetResourceAllocationInfo(0, 0, nullptr);
    if (empty_a.SizeInBytes != empty_b.SizeInBytes || empty_a.Alignment != empty_b.Alignment)
        mismatch("no resources", empty_a, empty_b);

    // A visible mask above 1 names a node a single-node adapter does not have: the host answers the invalid
    // allocation (size UINT64_MAX, 64 KiB alignment), for one resource and for none, through GetResourceAllocationInfo
    // and its 1 and 2 variants, and leaves the per-resource array as it was.
    {
        Ref<ID3D12Device4> ours4, host4;
        Ref<ID3D12Device8> ours8_mask, host8_mask;
        ours->QueryInterface(IID_PPV_ARGS(&ours4.p));
        host->QueryInterface(IID_PPV_ARGS(&host4.p));
        ours->QueryInterface(IID_PPV_ARGS(&ours8_mask.p));
        host->QueryInterface(IID_PPV_ARGS(&host8_mask.p));
        const D3D12_RESOURCE_DESC buffer = buffer_desc(256);
        D3D12_RESOURCE_DESC1 buffer1{};
        buffer1.Dimension = buffer.Dimension;
        buffer1.Width = buffer.Width;
        buffer1.Height = buffer1.DepthOrArraySize = buffer1.MipLevels = 1;
        buffer1.SampleDesc.Count = 1;
        buffer1.Layout = buffer.Layout;
        const auto same_info = [](const D3D12_RESOURCE_ALLOCATION_INFO& x, const D3D12_RESOURCE_ALLOCATION_INFO& y) {
            return x.SizeInBytes == y.SizeInBytes && x.Alignment == y.Alignment;
        };
        for (const UINT mask : {0u, 1u, 2u, 3u, 0x80000000u}) {
            char name[64];
            std::snprintf(name, sizeof(name), "visible mask 0x%x", mask);
            for (const UINT count : {0u, 1u}) {
                const D3D12_RESOURCE_DESC* descs = count ? &buffer : nullptr;
                const auto ia = ours->GetResourceAllocationInfo(mask, count, descs);
                const auto ib = host->GetResourceAllocationInfo(mask, count, descs);
                if (!same_info(ia, ib)) mismatch(name, ia, ib);
            }
            if (ours4.p && host4.p) {
                const D3D12_RESOURCE_ALLOCATION_INFO1 sentinel{0xdead, 0xbeef, 0xcafe};
                D3D12_RESOURCE_ALLOCATION_INFO1 each_a = sentinel, each_b = sentinel;
                const auto ia = ours4->GetResourceAllocationInfo1(mask, 1, &buffer, &each_a);
                const auto ib = host4->GetResourceAllocationInfo1(mask, 1, &buffer, &each_b);
                if (!same_info(ia, ib) || each_a.Offset != each_b.Offset || each_a.SizeInBytes != each_b.SizeInBytes ||
                    each_a.Alignment != each_b.Alignment)
                    mismatch("visible mask, GetResourceAllocationInfo1", ia, ib);
            }
            if (ours8_mask.p && host8_mask.p) {
                const D3D12_RESOURCE_ALLOCATION_INFO1 sentinel{0xdead, 0xbeef, 0xcafe};
                D3D12_RESOURCE_ALLOCATION_INFO1 each_a = sentinel, each_b = sentinel;
                const auto ia = ours8_mask->GetResourceAllocationInfo2(mask, 1, &buffer1, &each_a);
                const auto ib = host8_mask->GetResourceAllocationInfo2(mask, 1, &buffer1, &each_b);
                if (!same_info(ia, ib) || each_a.Offset != each_b.Offset || each_a.SizeInBytes != each_b.SizeInBytes ||
                    each_a.Alignment != each_b.Alignment)
                    mismatch("visible mask, GetResourceAllocationInfo2", ia, ib);
            }
        }
    }

    Ref<ID3D12Device8> ours8;
    // MULTISAMPLE_QUALITY_LEVELS with TILED_RESOURCE: the same answers as the host, except 128-bit formats at 4x,
    // which the AMD drivers refuse and WARP and NVIDIA report (and GetResourceAllocationInfo allocates here).
    for (const DXGI_FORMAT format : {rgba8, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32_FLOAT,
                                     DXGI_FORMAT_R32G32B32A32_FLOAT, DXGI_FORMAT_D32_FLOAT,
                                     DXGI_FORMAT_R32G32B32_FLOAT}) {
        for (const UINT samples : {1u, 2u, 4u, 8u, 16u}) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS x{format, samples,
                                                            D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_TILED_RESOURCE};
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS y = x;
            const HRESULT hx = ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &x, sizeof(x));
            const HRESULT hy = host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &y, sizeof(y));
            const bool wide = format == DXGI_FORMAT_R32G32B32A32_FLOAT;
            if (hx == hy && (x.NumQualityLevels == y.NumQualityLevels ||
                             (wide && samples == 4 && x.NumQualityLevels == 1 && y.NumQualityLevels == 0)))
                continue;
            if (mismatches++ < 24)
                std::fprintf(stderr, "tiled quality levels format %d %ux: ours 0x%08lx %u, host 0x%08lx %u\n",
                             int(format), samples, static_cast<unsigned long>(hx), x.NumQualityLevels,
                             static_cast<unsigned long>(hy), y.NumQualityLevels);
        }
    }

    if (FAILED(ours->QueryInterface(IID_PPV_ARGS(&ours8.p)))) return mismatches + 1;
    const D3D12_RESOURCE_DESC mixed[] = {with(texture_desc(16, 16, rgba8, none), k4), buffer_desc(300),
                                         with(texture_desc(64, 64, rgba8, rt), 0, 4)};
    D3D12_RESOURCE_DESC1 mixed1[3]{};
    for (int i = 0; i < 3; ++i) std::memcpy(&mixed1[i], &mixed[i], sizeof(mixed[i]));
    D3D12_RESOURCE_ALLOCATION_INFO1 each[3]{}, each2[3]{};
    const D3D12_RESOURCE_ALLOCATION_INFO total = ours8->GetResourceAllocationInfo1(0, 3, mixed, each);
    const D3D12_RESOURCE_ALLOCATION_INFO total2 = ours8->GetResourceAllocationInfo2(0, 3, mixed1, each2);
    const auto aligned = [](UINT64 v, UINT64 to) { return (v + to - 1) / to * to; };
    bool layout_ok = total.SizeInBytes == total2.SizeInBytes && total.Alignment == total2.Alignment &&
                     each[0].Offset == 0 && each[0].Alignment == k4 && each[1].Alignment == k64 &&
                     each[2].Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT &&
                     total.Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT &&
                     total.SizeInBytes == each[2].Offset + each[2].SizeInBytes;
    for (int i = 0; i < 3; ++i) {
        const D3D12_RESOURCE_ALLOCATION_INFO single = ours->GetResourceAllocationInfo(0, 1, &mixed[i]);
        layout_ok = layout_ok && each[i].SizeInBytes == single.SizeInBytes &&
                    each[i].Alignment == single.Alignment && each[i].Offset == each2[i].Offset &&
                    each[i].SizeInBytes == each2[i].SizeInBytes && each[i].Alignment == each2[i].Alignment &&
                    (i == 0 || each[i].Offset == aligned(each[i - 1].Offset + each[i - 1].SizeInBytes,
                                                         each[i].Alignment));
    }
    // An invalid description in the list makes the whole answer invalid.
    const D3D12_RESOURCE_DESC with_bad[] = {buffer_desc(300), buffer_desc(0)};
    layout_ok = layout_ok && ours8->GetResourceAllocationInfo1(0, 2, with_bad, each).SizeInBytes == UINT64_MAX &&
                host->GetResourceAllocationInfo(0, 2, with_bad).SizeInBytes == UINT64_MAX;
    if (!layout_ok) {
        ++mismatches;
        std::fprintf(stderr, "allocation info: Device4/Device8 per-resource layout wrong (total %llu/%llu)\n",
                     (unsigned long long)total.SizeInBytes, (unsigned long long)total.Alignment);
    }
    return mismatches;
}

struct Rule {
    const char* name;
    HRESULT result;
};

constexpr const char* kOpenListExecuted = "device after executing an open list";

// Command-list and allocator misuse (D3D-51), in one fixed order on one device; each entry is what that call
// returned. Run on the PC runtime and on ours, the two lists must agree. The queue waits on a fence the CPU signals
// later, so "in flight" really is: the GPU cannot have finished the list yet.
std::vector<Rule> command_list_rules(ID3D12Device* device) {
    std::vector<Rule> out;
    const auto add = [&](const char* name, HRESULT hr) { out.push_back({name, hr}); };
    constexpr D3D12_COMMAND_LIST_TYPE direct = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandAllocator> alloc, other, compute;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12Fence> gate, done;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = direct;
    if (FAILED(device->CreateCommandAllocator(direct, IID_PPV_ARGS(&alloc.p))) ||
        FAILED(device->CreateCommandAllocator(direct, IID_PPV_ARGS(&other.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&compute.p))) ||
        FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&gate.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&done.p)))) {
        add("setup", E_FAIL);
        return out;
    }
    Ref<ID3D12GraphicsCommandList> list, second, third;
    add("CreateCommandList", device->CreateCommandList(0, direct, alloc.p, nullptr, IID_PPV_ARGS(&list.p)));
    if (!list.p) return out;
    add("Reset while recording", list->Reset(alloc.p, nullptr));
    add("allocator Reset while a list records into it", alloc->Reset());
    add("CreateCommandList on an allocator a list records into",
        device->CreateCommandList(0, direct, alloc.p, nullptr, IID_PPV_ARGS(&second.p)));
    add("Close", list->Close());
    add("Close twice", list->Close());
    // Not compared: Reset with no allocator. The PC runtime dereferences the null pointer (an access violation);
    // ours returns E_INVALIDARG.
    add("Reset with a compute allocator", list->Reset(compute.p, nullptr));
    add("Reset", list->Reset(alloc.p, nullptr));
    add("CreateCommandList on another allocator",
        device->CreateCommandList(0, direct, other.p, nullptr, IID_PPV_ARGS(&third.p)));
    if (third.p) {
        add("Close the other list", third->Close());
        add("Reset onto an allocator another list records into", third->Reset(alloc.p, nullptr));
    }
    add("Close", list->Close());
    add("allocator Reset with its lists closed", alloc->Reset());

    // In flight: the queue first waits for `gate`, which only the CPU signals, after the checks below.
    add("Reset", list->Reset(alloc.p, nullptr));
    add("Close", list->Close());
    add("queue Wait", queue->Wait(gate.p, 1));
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    add("allocator Reset while its list is in flight", alloc->Reset());
    add("Reset the in-flight list onto another allocator", list->Reset(other.p, nullptr));
    add("Close", list->Close());
    add("fence Signal from the CPU", gate->Signal(1));
    add("queue Signal", queue->Signal(done.p, 1));
    add("SetEventOnCompletion (blocking)", done->SetEventOnCompletion(1, nullptr));
    add("allocator Reset after the GPU finished", alloc->Reset());

    // Last, since the PC runtime removes the device for it: executing a list that is still open.
    add("Reset", list->Reset(alloc.p, nullptr));
    queue->ExecuteCommandLists(1, lists);
    add(kOpenListExecuted, device->GetDeviceRemovedReason());
    return out;
}

// Returns the number of rules where ours differs from the PC runtime, or -1 without a host device. One difference
// is known and allowed: the PC runtime removes the device (DXGI_ERROR_INVALID_CALL) for executing an open list,
// while this layer submits nothing and keeps the device (docs/d3d12_vulkan.md).
int compare_command_list_rules(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const std::vector<Rule> a = command_list_rules(ours), b = command_list_rules(host.p);
    int mismatches = a.size() == b.size() ? 0 : 1;
    for (size_t i = 0; i < a.size() || i < b.size(); ++i) {
        const Rule* x = i < a.size() ? &a[i] : nullptr;
        const Rule* y = i < b.size() ? &b[i] : nullptr;
        const bool same = x && y && x->result == y->result;
        const bool known = x && y && std::strcmp(x->name, kOpenListExecuted) == 0 && x->result == S_OK &&
                           y->result == DXGI_ERROR_INVALID_CALL;
        if (!same && !known) ++mismatches;
        std::printf("command-list rule %-55s ours 0x%08lx host 0x%08lx%s\n", (x ? x : y)->name,
                    x ? static_cast<unsigned long>(x->result) : 0ul, y ? static_cast<unsigned long>(y->result) : 0ul,
                    same ? "" : known ? "  (known: an invalid call does not remove our device)" : "  <-- differs");
    }
    return mismatches;
}

D3D12_RESOURCE_DESC reserved_desc(D3D12_RESOURCE_DIMENSION dimension, UINT64 width, UINT height, UINT16 depth,
                                  UINT16 mips, DXGI_FORMAT format) {
    D3D12_RESOURCE_DESC desc = dimension == D3D12_RESOURCE_DIMENSION_BUFFER
                                   ? buffer_desc(width)
                                   : texture_desc(UINT(width), height, format, D3D12_RESOURCE_FLAG_NONE, mips, depth);
    desc.Dimension = dimension;
    if (dimension != D3D12_RESOURCE_DIMENSION_BUFFER) desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

// GetResourceTiling against the PC runtime, for the same reserved resources. What D3D12 defines must match: the
// tile shape, every standard mip's tile counts and its start within its array slice, and that no mip filling a
// whole tile is packed. Which smaller mips are packed (D3D11.3 spec 5.9.2.8.5: "may or may not be packed,
// depending on the IHV") and how many tiles they take is the driver's choice (src/resource_tiling.h), and with it
// the slice size and the total: compared only against a host that splits at the same mip. A layout the host
// refuses is skipped (drivers differ in what they accept); one the host accepts and ours refuses is a mismatch.
// Returns the number of mismatches, or -1 without a host D3D12 device with tiled resources or with every layout
// skipped.
int compare_tiling(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))) ||
        options.TiledResourcesTier == D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED)
        return -1;
    const bool volumes = options.TiledResourcesTier >= D3D12_TILED_RESOURCES_TIER_3;
    const auto B = D3D12_RESOURCE_DIMENSION_BUFFER, T2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
               T3 = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
    const D3D12_RESOURCE_DESC cases[] = {
        reserved_desc(B, 200000, 1, 1, 1, DXGI_FORMAT_UNKNOWN),
        reserved_desc(T2, 1024, 1024, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 1000, 300, 3, 0, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R8_UNORM),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R16_FLOAT),
        reserved_desc(T2, 512, 512, 1, 0, DXGI_FORMAT_R16G16B16A16_FLOAT),
        reserved_desc(T2, 512, 512, 2, 0, DXGI_FORMAT_R32G32B32A32_FLOAT),
        reserved_desc(T2, 2048, 2048, 1, 0, DXGI_FORMAT_BC1_UNORM),
        reserved_desc(T2, 2048, 1024, 2, 0, DXGI_FORMAT_BC7_UNORM),
        reserved_desc(T2, 1000, 600, 1, 0, DXGI_FORMAT_BC3_UNORM),
        reserved_desc(T2, 128, 128, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 100, 100, 1, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T2, 4096, 64, 1, 0, DXGI_FORMAT_B8G8R8A8_UNORM),
        reserved_desc(T2, 1024, 1024, 4, 3, DXGI_FORMAT_R32_FLOAT),
        reserved_desc(T3, 256, 256, 64, 0, DXGI_FORMAT_R8G8B8A8_UNORM),
        reserved_desc(T3, 128, 128, 128, 0, DXGI_FORMAT_R8_UNORM),
        reserved_desc(T3, 64, 64, 64, 0, DXGI_FORMAT_R16G16B16A16_FLOAT),
        reserved_desc(T3, 64, 64, 64, 0, DXGI_FORMAT_R32G32B32A32_FLOAT),
    };
    struct Tiling {
        UINT total = 0, count = 64;
        D3D12_PACKED_MIP_INFO packed{};
        D3D12_TILE_SHAPE shape{};
        D3D12_SUBRESOURCE_TILING subresources[64]{};
    };
    int mismatches = 0, compared = 0;
    const auto report = [&](int index, const char* what, UINT a, UINT b) {
        if (mismatches++ < 24) std::fprintf(stderr, "tiling case %d %s: ours %u host %u\n", index, what, a, b);
    };
    for (int index = 0; index < int(std::size(cases)); ++index) {
        const D3D12_RESOURCE_DESC& desc = cases[index];
        if (desc.Dimension == T3 && !volumes) continue;
        Ref<ID3D12Resource> a, b;
        const HRESULT ours_hr =
            ours->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&a.p));
        const HRESULT host_hr =
            host->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&b.p));
        if (FAILED(host_hr)) {
            // Which reserved resources a PC driver accepts varies: an RTX 3090 on driver 616.56 refuses texture
            // arrays whose mips reach the packed tail (cases 2, 6 and 8, at tier 3), an RTX 5070 on 621.01 accepts
            // them. The console is the reference for what the layer accepts, not the host, so there is nothing to
            // compare here.
            std::printf("GetResourceTiling case %d: the host runtime refuses it (0x%08lx), skipped\n", index,
                        static_cast<unsigned long>(host_hr));
            continue;
        }
        if (FAILED(ours_hr)) {
            report(index, "creation refused by ours, accepted by the host (HRESULT low word)", ours_hr & 0xffff, 0);
            continue;
        }
        ++compared;
        Tiling x, y;
        ours->GetResourceTiling(a.p, &x.total, &x.packed, &x.shape, &x.count, 0, x.subresources);
        host->GetResourceTiling(b.p, &y.total, &y.packed, &y.shape, &y.count, 0, y.subresources);
        if (x.shape.WidthInTexels != y.shape.WidthInTexels) report(index, "tile width", x.shape.WidthInTexels, y.shape.WidthInTexels);
        if (x.shape.HeightInTexels != y.shape.HeightInTexels) report(index, "tile height", x.shape.HeightInTexels, y.shape.HeightInTexels);
        if (x.shape.DepthInTexels != y.shape.DepthInTexels) report(index, "tile depth", x.shape.DepthInTexels, y.shape.DepthInTexels);
        if (x.count != y.count) {
            report(index, "subresource count", x.count, y.count);
            continue;
        }
        // Which small mips are packed is the driver's choice (D3D11.3 functional spec 5.9.2.8.5, D3D12_PACKED_MIP_INFO):
        // only a mip that fills at least one standard tile in every dimension must stay standard; a coarser mip with
        // any dimension below the tile "may or may not be packed, depending on the IHV", and once a mip is packed all
        // coarser ones are. Hosts differ: an RTX 5070 packs from the first mip below the tile (like ours), a Radeon
        // 880M keeps one or two more standard, WARP packs nothing. So the layer must keep every mip that fills a
        // tile standard and be self-consistent; the host's boundary is informational, and tile counts are compared
        // only for mips both treat as standard (and the packed totals only when both split at the same mip).
        const UINT layers = desc.Dimension == T2 ? desc.DepthOrArraySize : 1;
        const UINT mips = x.count / layers;
        const UINT slice_x = x.total / layers, slice_y = y.total / layers;
        UINT shared = 0;
        if (desc.Dimension != B) {
            UINT required = 0;  // leading mips that fill a whole tile in every dimension
            for (UINT m = 0; m < mips; ++m) {
                const UINT64 w = std::max<UINT64>(1, desc.Width >> m);
                const UINT h = std::max<UINT>(1, desc.Height >> m);
                const UINT d = desc.Dimension == T3 ? std::max<UINT>(1, desc.DepthOrArraySize >> m) : 1;
                if (w < x.shape.WidthInTexels || h < x.shape.HeightInTexels || d < x.shape.DepthInTexels) break;
                ++required;
            }
            if (x.packed.NumStandardMips < required || x.packed.NumStandardMips + x.packed.NumPackedMips != int(mips))
                report(index, "standard mips (a mip filling a tile must not be packed)", x.packed.NumStandardMips, required);
            if (y.packed.NumStandardMips < required)
                report(index, "host standard mips below the D3D12 minimum", y.packed.NumStandardMips, required);
            if (bool(x.packed.NumPackedMips) != bool(x.packed.NumTilesForPackedMips))
                report(index, "packed tiles", x.packed.NumPackedMips, x.packed.NumTilesForPackedMips);
            if (x.packed.NumStandardMips != y.packed.NumStandardMips)
                std::printf("GetResourceTiling case %d: standard mips ours %u, host %u (driver's choice, not compared)\n",
                            index, x.packed.NumStandardMips, y.packed.NumStandardMips);
            shared = std::min(x.packed.NumStandardMips, y.packed.NumStandardMips);
            if (x.packed.NumStandardMips == y.packed.NumStandardMips) {
                if (x.packed.NumPackedMips != y.packed.NumPackedMips) report(index, "packed mips", x.packed.NumPackedMips, y.packed.NumPackedMips);
                if (x.packed.StartTileIndexInOverallResource != y.packed.StartTileIndexInOverallResource)
                    report(index, "packed start", x.packed.StartTileIndexInOverallResource, y.packed.StartTileIndexInOverallResource);
                if (slice_x - x.packed.NumTilesForPackedMips != slice_y - y.packed.NumTilesForPackedMips)
                    report(index, "standard tiles per slice", slice_x - x.packed.NumTilesForPackedMips,
                           slice_y - y.packed.NumTilesForPackedMips);
            }
        } else if (x.total != y.total) {
            report(index, "buffer tiles", x.total, y.total);
        }
        for (UINT i = 0; i < x.count; ++i) {
            const auto &s = x.subresources[i], &t = y.subresources[i];
            const UINT layer = i / mips;
            if (desc.Dimension != B && i % mips >= x.packed.NumStandardMips) {
                // Packed: no tiles of its own. (On slices after the first the PC runtime writes a tile index here,
                // not the D3D12_PACKED_TILE its documentation gives; ours follows the documentation.)
                if (s.WidthInTiles || s.HeightInTiles || s.DepthInTiles || s.StartTileIndexInOverallResource != D3D12_PACKED_TILE)
                    report(index, "packed subresource", i, 0);
                continue;
            }
            if (desc.Dimension != B && i % mips >= shared) continue;  // standard in ours only: the host packs it
            if (s.WidthInTiles != t.WidthInTiles || s.HeightInTiles != t.HeightInTiles || s.DepthInTiles != t.DepthInTiles ||
                s.StartTileIndexInOverallResource - layer * slice_x != t.StartTileIndexInOverallResource - layer * slice_y)
                report(index, "standard subresource", i, 0);
        }
    }
    return compared ? mismatches : -1;
}


// whole_rectangle is a rectangle over the whole source: by the documentation the same as a null rectangle.
enum class ResolveScenario { full, rectangle, null_rectangle, whole_rectangle };

const char* resolve_scenario_name(ResolveScenario scenario) {
    switch (scenario) {
    case ResolveScenario::full: return "ResolveSubresource";
    case ResolveScenario::rectangle: return "ResolveSubresourceRegion(rect)";
    case ResolveScenario::null_rectangle: return "ResolveSubresourceRegion(null rect)";
    case ResolveScenario::whole_rectangle: return "ResolveSubresourceRegion(whole-source rect)";
    }
    return "unknown";
}

UINT resolve_pixel_size(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R9G9B9E5_SHAREDEXP:
    case DXGI_FORMAT_R10G10B10A2_UNORM: return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    default: return 0;
    }
}

// Runs one deterministic resolve on either the host D3D12 device or the Vulkan-backed proxy and returns only
// packed texel bytes (row-pitch padding is discarded). The source/destination are cleared to solid colours so
// every format and sample count exercises the resolve path without requiring a shader or game data.
bool capture_resolve(ID3D12Device* device, DXGI_FORMAT format, UINT samples, ResolveScenario scenario,
                     std::vector<std::uint8_t>& pixels) {
#define RESOLVE_CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "resolve capture failed at line %d: %s (format %d, samples %u, %s)\n", \
                     __LINE__, #condition, int(format), samples, resolve_scenario_name(scenario)); \
        return false; \
    } \
} while (false)

    const bool proxy = rt::d3d12_xs::is_device_proxy(device);
    const UINT src_width = 8, src_height = 8;
    const UINT dst_width = scenario == ResolveScenario::full ? src_width : 12;
    const UINT dst_height = scenario == ResolveScenario::full ? src_height : 10;
    const UINT pixel_size = resolve_pixel_size(format);
    RESOLVE_CHECK(pixel_size != 0);

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
    D3D12_RESOURCE_DESC src_desc =
        texture_desc(src_width, src_height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    src_desc.SampleDesc.Count = samples;
    D3D12_RESOURCE_DESC dst_desc =
        texture_desc(dst_width, dst_height, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);

    Ref<ID3D12Resource> src, dst, readback;
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &src_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&src.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &dst_desc,
                                                             D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                             IID_PPV_ARGS(&dst.p))));

    D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
    Ref<ID3D12DescriptorHeap> rtv_heap;
    RESOLVE_CHECK(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
    D3D12_CPU_DESCRIPTOR_HANDLE src_rtv{}, dst_rtv{};
    if (proxy) {
        const std::uint64_t start =
            reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p);
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        RESOLVE_CHECK(start != 0 && increment >= sizeof(leonida::TargetView));
        src_rtv = {SIZE_T(start)};
        dst_rtv = {SIZE_T(start + increment)};
    } else {
        src_rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        dst_rtv = {src_rtv.ptr + device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV)};
    }
    D3D12_RENDER_TARGET_VIEW_DESC msaa_view{};
    msaa_view.Format = format;
    msaa_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
    device->CreateRenderTargetView(src.p, &msaa_view, src_rtv);
    device->CreateRenderTargetView(dst.p, nullptr, dst_rtv);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12GraphicsCommandList7> list7;
    Ref<ID3D12Fence> fence;
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                       IID_PPV_ARGS(&list.p))));
    // ResolveSubresourceRegion is on ID3D12GraphicsCommandList7; the full ResolveSubresource path does not need
    // it, so a host without list7 (Windows 10) can still be compared on the full form.
    if (scenario != ResolveScenario::full)
        RESOLVE_CHECK(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p))));
    RESOLVE_CHECK(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));

    UINT64 cursor = 0;
    const auto readback_layout = place(device, dst.p, 0, cursor);
    RESOLVE_CHECK(cursor != 0 && readback_layout.Footprint.RowPitch >= dst_width * pixel_size);
    const D3D12_RESOURCE_DESC readback_desc = buffer_desc(cursor);
    RESOLVE_CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE,
                                                             &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                             nullptr, IID_PPV_ARGS(&readback.p))));

    const FLOAT green[4] = {0, 1, 0, 1};
    const FLOAT red[4] = {1, 0, 0, 1};
    // Use a rectangle even for the full source: multisampled images are cleared as rendering attachments,
    // not with vkCmdClearColorImage (which is for single-sample transfer clears).
    const D3D12_RECT full_source{0, 0, LONG(src_width), LONG(src_height)};
    list->ClearRenderTargetView(src_rtv, green, 1, &full_source);
    list->ClearRenderTargetView(dst_rtv, red, 0, nullptr);
    const D3D12_RESOURCE_BARRIER before_resolve[] = {
        transition(src.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
        transition(dst.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST)};
    list->ResourceBarrier(2, before_resolve);
    if (scenario == ResolveScenario::full) {
        list->ResolveSubresource(dst.p, 0, src.p, 0, format);
    } else {
        D3D12_RECT rectangle{2, 1, 6, 5};
        if (scenario == ResolveScenario::whole_rectangle) rectangle = full_source;
        list7->ResolveSubresourceRegion(dst.p, 0, 3, 2, src.p, 0,
                                        scenario == ResolveScenario::null_rectangle ? nullptr : &rectangle,
                                        format, D3D12_RESOLVE_MODE_AVERAGE);
    }
    const auto to_copy = transition(dst.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    const auto readback_location = footprint(readback.p, readback_layout);
    const auto resolved_location = subresource(dst.p, 0);
    list->CopyTextureRegion(&readback_location, 0, 0, 0, &resolved_location, nullptr);
    RESOLVE_CHECK(list->Close() == S_OK);

    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (proxy)
        RESOLVE_CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
    RESOLVE_CHECK(SUCCEEDED(queue->Signal(fence.p, 1)) && SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));

    D3D12_RANGE read_range{0, SIZE_T(cursor)};
    std::uint8_t* mapped = nullptr;
    RESOLVE_CHECK(SUCCEEDED(readback->Map(0, &read_range, reinterpret_cast<void**>(&mapped))) && mapped);
    pixels.resize(size_t(dst_width) * dst_height * pixel_size);
    for (UINT row = 0; row < dst_height; ++row)
        std::memcpy(pixels.data() + size_t(row) * dst_width * pixel_size,
                    mapped + readback_layout.Offset + size_t(row) * readback_layout.Footprint.RowPitch,
                    size_t(dst_width) * pixel_size);
    const D3D12_RANGE written{0, 0};
    readback->Unmap(0, &written);
#undef RESOLVE_CHECK
    return true;
}

// Compare RGB9E5 capability queries with the sample counts of the actual Vulkan attachment, then
// resolve green over a red single-sample destination and check its packed shared-exponent texels.
bool check_rgb9e5_resolves(ID3D12Device* device, const vktest::Device& vk) {
    constexpr auto format = DXGI_FORMAT_R9G9B9E5_SHAREDEXP;
    constexpr VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageFormatProperties properties{};
    const auto query = vk.instance_fn<PFN_vkGetPhysicalDeviceImageFormatProperties>("vkGetPhysicalDeviceImageFormatProperties");
    const auto result = query(vk.physical, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, VK_IMAGE_TYPE_2D,
                              VK_IMAGE_TILING_OPTIMAL, usage, 0, &properties);
    if (result != VK_SUCCESS && result != VK_ERROR_FORMAT_NOT_SUPPORTED) return false;
    const VkSampleCountFlags counts = result == VK_SUCCESS ? properties.sampleCounts : 0;
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
    if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support)))) return false;
    constexpr UINT bits = D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE;
    const bool multisampled = (counts & (2u | 4u | 8u | 16u | 32u)) != 0;
    if ((UINT(support.Support1) & bits) != (multisampled ? bits : 0u)) {
        std::fprintf(stderr, "RGB9E5 Support1=0x%x, Vulkan sample counts=0x%x\n", UINT(support.Support1), counts);
        return false;
    }
    for (UINT samples : {2u, 4u, 8u, 16u, 32u}) {
        const bool supported = (counts & samples) != 0;
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{format, samples};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality, sizeof(quality))) ||
            quality.NumQualityLevels != UINT(supported)) return false;
        if (!supported) continue;
        std::vector<std::uint8_t> pixels;
        if (!capture_resolve(device, format, samples, ResolveScenario::full, pixels) || pixels.size() != 8 * 8 * 4)
            return false;
        for (size_t i = 0; i < pixels.size(); i += 4) {
            std::uint32_t packed;
            std::memcpy(&packed, pixels.data() + i, sizeof(packed));
            if (packed != 0x80020000u) {  // green mantissa 256, shared exponent 16
                std::fprintf(stderr, "RGB9E5 %ux resolve: pixel %zu got 0x%08x, expected 0x80020000\n",
                             samples, i / 4, packed);
                return false;
            }
        }
    }
    return true;
}

// Compare D3D12 multisample quality queries for depth/stencil formats with the host runtime. The Vulkan
// backend has one standard sample pattern, so every supported count reports one quality level (index 0).
// A count the Vulkan GPU cannot do for depth and stencil attachments (depth_sample_counts, from its limits) is not
// something the layer can offer: a software host such as WARP advertises 16x, an RTX or Radeon does not.
int compare_depth_multisample_quality_levels(ID3D12Device* ours, VkSampleCountFlags depth_sample_counts) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;

    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_D32_FLOAT,
        DXGI_FORMAT_D16_UNORM,
        DXGI_FORMAT_D24_UNORM_S8_UINT,
        DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
        DXGI_FORMAT_R24G8_TYPELESS,
        DXGI_FORMAT_R32G8X24_TYPELESS,
    };
    constexpr UINT sample_counts[] = {2, 4, 8, 16, 32};
    unsigned compared = 0, skipped = 0;
    int mismatches = 0;
    for (DXGI_FORMAT format : formats) {
        for (UINT samples : sample_counts) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS host_levels{};
            host_levels.Format = format;
            host_levels.SampleCount = samples;
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS our_levels = host_levels;
            const HRESULT host_hr = host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                               &host_levels, sizeof(host_levels));
            if (FAILED(host_hr) || host_levels.NumQualityLevels == 0 || !(depth_sample_counts & samples)) {
                ++skipped;
                continue;
            }
            ++compared;
            const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                             &our_levels, sizeof(our_levels));
            if (FAILED(our_hr) || our_levels.NumQualityLevels != 1) {
                std::fprintf(stderr,
                             "depth MSAA quality mismatch: format %d samples %u (host %u quality levels, Vulkan "
                             "HRESULT 0x%08lx / %u levels)\n",
                             int(format), samples, host_levels.NumQualityLevels,
                             static_cast<unsigned long>(our_hr), our_levels.NumQualityLevels);
                ++mismatches;
            }
        }
    }
    std::printf("Depth/stencil MSAA quality queries: %u compared, %u host-unsupported skipped, %d mismatches\n",
                compared, skipped, mismatches);
    return mismatches;
}

bool check_plain_resolve_formats(ID3D12Device* device) {
    struct Case { DXGI_FORMAT format; bool valid; bool depth = false; DXGI_FORMAT requested = DXGI_FORMAT_UNKNOWN; };
    constexpr Case cases[] = {
        {DXGI_FORMAT_R8G8B8A8_UINT, false},
        {DXGI_FORMAT_R8G8B8A8_SINT, false},
        {DXGI_FORMAT_R32_UINT, false},
        {DXGI_FORMAT_R32_SINT, false},
        {DXGI_FORMAT_R32G32B32A32_SINT, false},
        {DXGI_FORMAT_R8G8B8A8_UINT, false, false, DXGI_FORMAT_R8G8B8A8_UNORM},
        {DXGI_FORMAT_R8G8B8A8_UNORM, true},
        {DXGI_FORMAT_R16G16B16A16_FLOAT, true},
        {DXGI_FORMAT_D32_FLOAT, true, true},
    };
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p)))) return false;
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    bool ok = true;
    unsigned tested = 0, skipped = 0;
    for (const auto& c : cases) {
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{c.format, 4};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality, sizeof(quality))))
            return false;
        if (!quality.NumQualityLevels) {
            std::printf("plain resolve format %d: 4x MSAA unsupported, skipped\n", int(c.format));
            ++skipped;
            continue;
        }
        auto desc = texture_desc(8, 8, c.format, c.depth ? D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL
                                                         : D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        desc.SampleDesc.Count = 4;
        Ref<ID3D12Resource> src, dst;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_RESOLVE_SOURCE, nullptr,
                                                    IID_PPV_ARGS(&src.p)))) return false;
        desc.SampleDesc.Count = 1;
        if (FAILED(device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr,
                                                    IID_PPV_ARGS(&dst.p)))) return false;
        const DXGI_FORMAT requested = c.requested == DXGI_FORMAT_UNKNOWN ? c.format : c.requested;
        list->ResolveSubresource(dst.p, 0, src.p, 0, requested);
        const HRESULT result = list->Close();
        if (result != (c.valid ? S_OK : E_INVALIDARG)) {
            std::fprintf(stderr, "plain resolve format %d: Close returned 0x%08lx, expected 0x%08lx\n",
                         int(c.format), static_cast<unsigned long>(result),
                         static_cast<unsigned long>(c.valid ? S_OK : E_INVALIDARG));
            ok = false;
        }
        ++tested;
        if (FAILED(list->Reset(allocator.p, nullptr))) return false;
    }
    if (FAILED(list->Close())) return false;
    std::printf("plain resolve format validation: %u tested, %u unsupported cases skipped\n", tested, skipped);
    return ok;
}

// Compare the Vulkan-backed resolve output byte-for-byte with the host runtime for every common supported format,
// sample count and region form. A host-unsupported combination is a deliberate skip; a mismatch in advertised
// support or in the resulting pixels is a failure. The region forms need ID3D12GraphicsCommandList7 on the host;
// without it (Windows 10) only the full ResolveSubresource form is compared, so the later device checks still run.
int compare_resolves(ID3D12Device* ours, VkSampleCountFlags color_sample_counts) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    const bool host_regions = host_has_graphics_command_list7(host.p);
    if (!host_regions)
        std::printf("Resolve comparison: the host runtime has no ID3D12GraphicsCommandList7; "
                    "ResolveSubresourceRegion cases are skipped\n");

    constexpr DXGI_FORMAT formats[] = {
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R32G32B32A32_FLOAT,
    };
    constexpr UINT sample_counts[] = {2, 4, 8, 16, 32};
    constexpr UINT required =
        UINT(D3D12_FORMAT_SUPPORT1_RENDER_TARGET) | UINT(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET) |
        UINT(D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE);
    constexpr ResolveScenario scenarios[] = {
        ResolveScenario::full, ResolveScenario::rectangle, ResolveScenario::null_rectangle,
        ResolveScenario::whole_rectangle};
    int mismatches = 0;
    unsigned compared = 0, matched = 0, skipped = 0;

    for (DXGI_FORMAT format : formats) {
        D3D12_FEATURE_DATA_FORMAT_SUPPORT host_format{format}, our_format{format};
        const HRESULT host_format_hr =
            host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &host_format, sizeof(host_format));
        const HRESULT our_format_hr =
            ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &our_format, sizeof(our_format));
        if (FAILED(host_format_hr) || (UINT(host_format.Support1) & required) != required) {
            std::printf("resolve format %d: host does not advertise multisample colour resolve, skipped\n", int(format));
            ++skipped;
            continue;
        }
        if (FAILED(our_format_hr) || (UINT(our_format.Support1) & required) != required) {
            std::fprintf(stderr, "resolve format %d: Vulkan device does not match host multisample format support\n",
                         int(format));
            ++mismatches;
            continue;
        }

        for (UINT samples : sample_counts) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS host_levels{};
            host_levels.Format = format;
            host_levels.SampleCount = samples;
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS our_levels = host_levels;
            const HRESULT host_hr = host->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                               &host_levels, sizeof(host_levels));
            const HRESULT our_hr = ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS,
                                                              &our_levels, sizeof(our_levels));
            // Not a count the Vulkan GPU can do for colour attachments (a software host like WARP offers 16x).
            if (FAILED(host_hr) || host_levels.NumQualityLevels == 0 || !(color_sample_counts & samples)) {
                ++skipped;
                continue;
            }
            if (FAILED(our_hr) || our_levels.NumQualityLevels == 0) {
                std::fprintf(stderr, "resolve format %d samples %u: Vulkan reports unsupported, host supports it\n",
                             int(format), samples);
                ++mismatches;
                continue;
            }

            for (ResolveScenario scenario : scenarios) {
                if (!host_regions && scenario != ResolveScenario::full) {
                    ++skipped;
                    continue;
                }
                std::vector<std::uint8_t> host_pixels, our_pixels;
                if (!capture_resolve(host.p, format, samples, scenario, host_pixels) ||
                    !capture_resolve(ours, format, samples, scenario, our_pixels)) {
                    ++mismatches;
                    continue;
                }
                ++compared;
                if (host_pixels != our_pixels && scenario == ResolveScenario::null_rectangle) {
                    // A null rectangle is the whole source (ID3D12GraphicsCommandList1::ResolveSubresourceRegion).
                    // Some hosts do not honour that (the Radeon 880M on driver 26.11.2 and WARP resolve to the
                    // destination's origin and leave the rest unwritten; an RTX 5070 follows the documentation), so
                    // where the host disagrees with its own whole-source rectangle, the layer is held to the
                    // documented result: that rectangle's.
                    std::vector<std::uint8_t> host_whole;
                    if (capture_resolve(host.p, format, samples, ResolveScenario::whole_rectangle, host_whole) &&
                        host_whole != host_pixels && host_whole == our_pixels) {
                        std::printf("resolve format %d samples %u: the host's null-rectangle resolve differs from "
                                    "its whole-source rectangle (host quirk); ours follows the documentation\n",
                                    int(format), samples);
                        ++matched;
                        continue;
                    }
                }
                if (host_pixels != our_pixels) {
                    std::fprintf(stderr, "resolve mismatch: format %d samples %u %s (host %zu bytes, Vulkan %zu)\n",
                                 int(format), samples, resolve_scenario_name(scenario),
                                 host_pixels.size(), our_pixels.size());
                    ++mismatches;
                } else {
                    ++matched;
                }
            }
        }
    }
    std::printf("Resolve comparison: %u compared, %u matched, %u unsupported or host-unavailable cases skipped, "
                "%d mismatches\n", compared, matched, skipped, mismatches);
    if (compared == 0) {
        std::fprintf(stderr, "Resolve comparison: no common supported format/sample combination was exercised\n");
        ++mismatches;
    }
    return mismatches;
}

// D3D12_FEATURE_SHADER_MODEL and D3D12_FEATURE_ROOT_SIGNATURE must give the host PC runtime's HRESULT, with the
// highest supported value clamped to what the layer can honour (DXIL 1.0, root signature 1.1). Returns the number
// of mismatches, or -1 when the host has no D3D12 device.
int compare_feature_support(ID3D12Device* ours) {
    if (copycmp::host_is_wine()) return -1;
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                              : nullptr;
    Ref<ID3D12Device> host;
    if (!create || FAILED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) return -1;
    int mismatches = 0;
    const auto report = [&](const char* what, UINT request, HRESULT got_hr, UINT got, HRESULT want_hr, UINT want) {
        if (got_hr == want_hr && (FAILED(got_hr) || got == want)) return;
        if (mismatches++ < 8)
            std::fprintf(stderr, "%s request 0x%02x: ours hr=0x%08lX value 0x%02x; host hr=0x%08lX value 0x%02x\n",
                         what, request, static_cast<unsigned long>(got_hr), got,
                         static_cast<unsigned long>(want_hr), want);
    };
    // A host runtime older than a request refuses it with E_INVALIDARG where a newer one clamps it: Windows 10's
    // d3d12.dll answers shader models up to 6.6 and root signatures up to 1.1. Such a request can't be compared with
    // the host, so the layer's answer is checked against its own clamp instead. The requests ascend, so `answered`
    // is the highest one the host has accepted so far.
    unsigned predated = 0;
    const auto host_predates = [&](const char* what, UINT request, UINT answered, HRESULT host_hr, HRESULT got_hr,
                                   UINT got, UINT clamp) {
        if (host_hr != E_INVALIDARG || !answered || request <= answered) return false;
        ++predated;
        if ((got_hr != S_OK || got != clamp) && mismatches++ < 8)
            std::fprintf(stderr, "%s request 0x%02x: ours hr=0x%08lX value 0x%02x; the host runtime predates it, "
                                 "the layer's clamp is S_OK 0x%02x\n",
                         what, request, static_cast<unsigned long>(got_hr), got, clamp);
        return true;
    };

    // 5.1 upwards: D3D12's own baseline, which every SDK names, then 6.0-6.9 by number (only 6.0 is guaranteed to be
    // named by every SDK that has the query at all). And only values the runtime defines go to the host: it crashed
    // the process on an undefined one (0x5f, and once on 0), so the layer's refusal of an undefined value is checked
    // in main.
    const D3D_SHADER_MODEL models[] = {
        D3D_SHADER_MODEL_5_1, D3D_SHADER_MODEL_6_0, static_cast<D3D_SHADER_MODEL>(0x61),
        static_cast<D3D_SHADER_MODEL>(0x62),
        static_cast<D3D_SHADER_MODEL>(0x63), static_cast<D3D_SHADER_MODEL>(0x64),
        static_cast<D3D_SHADER_MODEL>(0x65), static_cast<D3D_SHADER_MODEL>(0x66),
        static_cast<D3D_SHADER_MODEL>(0x67), static_cast<D3D_SHADER_MODEL>(0x68),
        static_cast<D3D_SHADER_MODEL>(0x69)};
    UINT host_model = 0;
    for (const D3D_SHADER_MODEL model : models) {
        D3D12_FEATURE_DATA_SHADER_MODEL ours_data{model}, host_data{model};
        const HRESULT ours_hr =
            ours->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &ours_data, sizeof(ours_data));
        const HRESULT host_hr =
            host->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &host_data, sizeof(host_data));
        if (SUCCEEDED(host_hr)) host_model = static_cast<UINT>(model);
        if (host_predates("shader model", static_cast<UINT>(model), host_model, host_hr, ours_hr,
                          static_cast<UINT>(ours_data.HighestShaderModel),
                          std::min<UINT>(static_cast<UINT>(model), static_cast<UINT>(D3D_SHADER_MODEL_6_0))))
            continue;
        const UINT want = std::min<UINT>(static_cast<UINT>(host_data.HighestShaderModel),
                                         static_cast<UINT>(D3D_SHADER_MODEL_6_0));
        report("shader model", static_cast<UINT>(model), ours_hr,
               static_cast<UINT>(ours_data.HighestShaderModel), host_hr, want);
    }

    // 0x3 is 1.2 by number: SDKs without the 1.2 headers do not name it (runtime/CMakeLists.txt probes for that).
    const D3D_ROOT_SIGNATURE_VERSION versions[] = {D3D_ROOT_SIGNATURE_VERSION_1_0, D3D_ROOT_SIGNATURE_VERSION_1_1,
                                                   static_cast<D3D_ROOT_SIGNATURE_VERSION>(0x3)};
    UINT host_version = 0;
    for (const D3D_ROOT_SIGNATURE_VERSION version : versions) {
        D3D12_FEATURE_DATA_ROOT_SIGNATURE ours_data{version}, host_data{version};
        const HRESULT ours_hr =
            ours->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &ours_data, sizeof(ours_data));
        const HRESULT host_hr =
            host->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &host_data, sizeof(host_data));
        if (SUCCEEDED(host_hr)) host_version = static_cast<UINT>(version);
        if (host_predates("root signature", static_cast<UINT>(version), host_version, host_hr, ours_hr,
                          static_cast<UINT>(ours_data.HighestVersion),
                          std::min<UINT>(static_cast<UINT>(version),
                                         static_cast<UINT>(D3D_ROOT_SIGNATURE_VERSION_1_1))))
            continue;
        const UINT want = std::min<UINT>(static_cast<UINT>(host_data.HighestVersion),
                                         static_cast<UINT>(D3D_ROOT_SIGNATURE_VERSION_1_1));
        report("root signature", static_cast<UINT>(version), ours_hr,
               static_cast<UINT>(ours_data.HighestVersion), host_hr, want);
    }

    if (predated)
        std::printf("CheckFeatureSupport: %u requests the host runtime predates (e.g. Windows 10) checked against the "
                    "layer's clamp instead of the host\n", predated);

    // A wrong DataSize is refused by both, in both queries.
    {
        D3D12_FEATURE_DATA_SHADER_MODEL ours_data{D3D_SHADER_MODEL_6_0};
        D3D12_FEATURE_DATA_ROOT_SIGNATURE ours_root{D3D_ROOT_SIGNATURE_VERSION_1_1};
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &ours_data, sizeof(ours_data) - 1)))
            ++mismatches;
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &ours_root, sizeof(ours_root) - 1)))
            ++mismatches;
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, nullptr, sizeof(ours_data))))
            ++mismatches;
        if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, nullptr, sizeof(ours_root))))
            ++mismatches;
    }
    return mismatches;
}

// Same-adapter host D3D12 device matched by VkPhysicalDeviceIDProperties (LUID first, then PCI IDs), falling back
// to the default adapter. False if the host has no D3D12 device.
// D3D11.3 spec 3.2.3.8's ideal FLOAT -> SRGB encoding of c in [0, 1], before scaling to the format's maximum.
double ideal_srgb(double c) { return c <= 0.0031308 ? c * 12.92 : 1.055 * std::pow(c, 1.0 / 2.4) - 0.055; }

// Both bytes within 0.6 ULP (on the integer side) of the ideal value: the tolerance spec 3.2.3.6 (FLOAT -> UNORM)
// and 3.2.3.8 (FLOAT -> SRGB) permit, so two drivers may give the codes either side of a near-tie.
bool within_06_ulp(std::uint8_t a, std::uint8_t b, double ideal) {
    return std::abs(double(a) - ideal) <= 0.6 && std::abs(double(b) - ideal) <= 0.6;
}

// The code nearest to `ideal`, a tie going to the even one: what the layer writes for a normalised clear (#521's
// quantize_unorm), whichever way the GPU's own driver rounds the tie.
double nearest_even_code(double ideal) {
    const double r = std::floor(ideal + 0.5);
    return r - ideal == 0.5 && std::fmod(r, 2.0) != 0.0 ? r - 1.0 : r;
}

bool create_host_device_for_physical(VkInstance instance, VkPhysicalDevice physical,
                                     PFN_vkGetInstanceProcAddr instance_proc, Ref<ID3D12Device>& host,
                                     std::string& adapter_name) {
    if (copycmp::host_is_wine()) return false;
    const HMODULE d3d12 = LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    const auto create = d3d12 ? reinterpret_cast<PFN_D3D12_CREATE_DEVICE>(
                                    reinterpret_cast<void*>(GetProcAddress(d3d12, "D3D12CreateDevice")))
                              : nullptr;
    if (!create) return false;

    VkPhysicalDeviceIDProperties id_props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id_props};
    if (instance && physical && instance_proc) {
        if (const auto get_props2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
                instance_proc(instance, "vkGetPhysicalDeviceProperties2"))) {
            get_props2(physical, &props2);
        }
    }

    const HMODULE dxgi = LoadLibraryExA("dxgi.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    using CreateFactoryFn = HRESULT(WINAPI*)(REFIID, void**);
    const auto create_factory = dxgi ? reinterpret_cast<CreateFactoryFn>(
                                           reinterpret_cast<void*>(GetProcAddress(dxgi, "CreateDXGIFactory1")))
                                     : nullptr;
    Ref<IDXGIAdapter1> luid_match, pci_match;
    std::string luid_name, pci_name;
    if (create_factory) {
        Ref<IDXGIFactory4> factory;
        if (SUCCEEDED(create_factory(IID_PPV_ARGS(&factory.p)))) {
            Ref<IDXGIAdapter1> candidate;
            for (UINT i = 0; factory->EnumAdapters1(i, &candidate.p) != DXGI_ERROR_NOT_FOUND; ++i) {
                DXGI_ADAPTER_DESC1 desc{};
                if (SUCCEEDED(candidate->GetDesc1(&desc))) {
                    char utf8[256]{};
                    WideCharToMultiByte(CP_UTF8, 0, desc.Description, -1, utf8, sizeof(utf8), nullptr, nullptr);
                    if (!luid_match.p && id_props.deviceLUIDValid &&
                        std::memcmp(&desc.AdapterLuid, id_props.deviceLUID, sizeof(desc.AdapterLuid)) == 0) {
                        luid_match.p = candidate.p;
                        candidate.p = nullptr;
                        luid_name = utf8;
                        break;
                    }
                    if (!pci_match.p && props2.properties.vendorID != 0 && desc.VendorId == props2.properties.vendorID &&
                        desc.DeviceId == props2.properties.deviceID) {
                        pci_match.p = candidate.p;
                        candidate.p = nullptr;
                        pci_name = utf8;
                    }
                }
                candidate.reset();
            }
        }
    }
    if (luid_match.p && SUCCEEDED(create(luid_match.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        adapter_name = luid_name;
        return true;
    }
    if (pci_match.p && SUCCEEDED(create(pci_match.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        adapter_name = pci_name;
        return true;
    }
    if (SUCCEEDED(create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
        adapter_name = "default adapter";
        return true;
    }
    return false;
}

bool integer_format(DXGI_FORMAT format) {
    switch (format) {
    case DXGI_FORMAT_R32G32B32A32_UINT: case DXGI_FORMAT_R32G32B32A32_SINT: case DXGI_FORMAT_R32G32B32_UINT:
    case DXGI_FORMAT_R32G32B32_SINT: case DXGI_FORMAT_R16G16B16A16_UINT: case DXGI_FORMAT_R16G16B16A16_SINT:
    case DXGI_FORMAT_R32G32_UINT: case DXGI_FORMAT_R32G32_SINT: case DXGI_FORMAT_R10G10B10A2_UINT:
    case DXGI_FORMAT_R8G8B8A8_UINT: case DXGI_FORMAT_R8G8B8A8_SINT: case DXGI_FORMAT_R16G16_UINT:
    case DXGI_FORMAT_R16G16_SINT: case DXGI_FORMAT_R32_UINT: case DXGI_FORMAT_R32_SINT: case DXGI_FORMAT_R8G8_UINT:
    case DXGI_FORMAT_R8G8_SINT: case DXGI_FORMAT_R16_UINT: case DXGI_FORMAT_R16_SINT: case DXGI_FORMAT_R8_UINT:
    case DXGI_FORMAT_R8_SINT: return true;
    default: return false;
    }
}

// FORMAT_SUPPORT bits that D3D12 defines per format rather than per GPU, compared with the host PC runtime on the
// same adapter for every format both support. Two kinds of check:
// - against the host: no 1D, 3D, depth-stencil, MIP or atomics the host lacks; storage-only (TYPELESS) formats
//   stay storage-only; depth formats are neither shader resources nor UAVs; MIP wherever the host has it;
// - rules and consistency within the layer: no sample, gather or resolve of integer formats; atomics only on
//   R32_UINT/R32_SINT, and R32_UINT typed UAVs carry them; gather with sample; typed load with typed store;
//   MULTISAMPLE_RENDERTARGET only with quality levels, and on depth formats whenever there are quality levels.
// Hardware-dependent bits (render targets, filtering, typed UAV formats, which float formats resolve) are left to
// the Vulkan driver. Formats the layer reports nothing for (unsupported formats, D24 without Vulkan D24) are listed,
// not checked.
int compare_format_support(ID3D12Device* ours, const rt::d3d12::VulkanDeviceInfo& info) {
    Ref<ID3D12Device> host;
    std::string adapter;
    if (!create_host_device_for_physical(info.instance, info.physical, info.get_instance_proc, host, adapter)) {
        std::printf("FORMAT_SUPPORT comparison: no host D3D12 device, skipped\n");
        return 0;
    }
    if (adapter == "default adapter") {  // neither LUID nor PCI IDs matched: another GPU's answers
        std::printf("FORMAT_SUPPORT comparison: no host device on the same adapter, skipped\n");
        return 0;
    }
    constexpr UINT dims = D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_TEXTURE2D |
                          D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_TEXTURECUBE;
    constexpr UINT storage_only = dims | D3D12_FORMAT_SUPPORT1_MIP | D3D12_FORMAT_SUPPORT1_CAST_WITHIN_BIT_LAYOUT;
    constexpr UINT format_rules = D3D12_FORMAT_SUPPORT1_TEXTURE1D | D3D12_FORMAT_SUPPORT1_TEXTURE3D |
                                  D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL | D3D12_FORMAT_SUPPORT1_MIP;
    constexpr UINT shader_reads = D3D12_FORMAT_SUPPORT1_SHADER_LOAD | D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE |
                                  D3D12_FORMAT_SUPPORT1_SHADER_GATHER;
    constexpr UINT filtered = D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE | D3D12_FORMAT_SUPPORT1_SHADER_GATHER |
                              D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE;
    constexpr UINT atomics = D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_ADD | D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_BITWISE_OPS |
                             D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_COMPARE_STORE_OR_COMPARE_EXCHANGE |
                             D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_SIGNED_MIN_OR_MAX |
                             D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_UNSIGNED_MIN_OR_MAX;
    constexpr UINT any_atomic = atomics | D3D12_FORMAT_SUPPORT2_UAV_ATOMIC_EXCHANGE;
    int mismatches = 0;
    unsigned compared = 0;
    std::string unreported;
    const auto fail = [&](int format, const char* what, UINT o1, UINT o2, UINT h1, UINT h2) {
        if (mismatches++ < 32)
            std::fprintf(stderr, "FORMAT_SUPPORT format %d: %s (layer %08x/%08x, host %08x/%08x)\n", format, what,
                         o1, o2, h1, h2);
    };
    const auto multisampled = [&](DXGI_FORMAT format) {
        for (UINT samples : {2u, 4u, 8u, 16u}) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS levels{format, samples};
            if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &levels,
                                                    sizeof(levels))) &&
                levels.NumQualityLevels)
                return true;
        }
        return false;
    };
    for (int f = 1; f <= DXGI_FORMAT_A4B4G4R4_UNORM; ++f) {
        if (f > DXGI_FORMAT_V408 && f < DXGI_FORMAT_SAMPLER_FEEDBACK_MIN_MIP_OPAQUE) continue;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT o{DXGI_FORMAT(f)}, h{DXGI_FORMAT(f)};
        if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &o, sizeof(o))) ||
            FAILED(host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &h, sizeof(h))))
            continue;
        const UINT o1 = UINT(o.Support1), o2 = UINT(o.Support2), h1 = UINT(h.Support1), h2 = UINT(h.Support2);
        if (!h1) continue;
        if (!o1) {  // the layer answers S_OK with no uses: nothing to compare, but say which
            unreported += (unreported.empty() ? "" : ", ") + std::to_string(f);
            continue;
        }
        ++compared;
        if ((f == DXGI_FORMAT_R8G8B8A8_UNORM || f == DXGI_FORMAT_B8G8R8A8_UNORM ||
             f == DXGI_FORMAT_B8G8R8X8_UNORM || f == DXGI_FORMAT_R8G8B8A8_UNORM_SRGB ||
             f == DXGI_FORMAT_B8G8R8A8_UNORM_SRGB || f == DXGI_FORMAT_B8G8R8X8_UNORM_SRGB) &&
            ((o1 ^ h1) & D3D12_FORMAT_SUPPORT1_BUFFER))
            fail(f, "buffer support differs from host", o1, o2, h1, h2);
        if (o1 & format_rules & ~h1) fail(f, "1D, 3D, depth-stencil or MIP the host doesn't have", o1, o2, h1, h2);
        if (o2 & any_atomic & ~h2) fail(f, "UAV atomics the host doesn't have", o1, o2, h1, h2);
        if (integer_format(DXGI_FORMAT(f)) && (o1 & filtered))
            fail(f, "sample, gather or resolve of an integer format", o1, o2, h1, h2);
        if ((o2 & any_atomic) && f != DXGI_FORMAT_R32_UINT && f != DXGI_FORMAT_R32_SINT)
            fail(f, "UAV atomics on a format other than R32_UINT/R32_SINT", o1, o2, h1, h2);
        // A host answer of nothing but dimensions, mips and casts is a TYPELESS (storage-only) format.
        if ((h1 & dims) && !(h1 & ~storage_only) && !(h2 & ~UINT(D3D12_FORMAT_SUPPORT2_TILED)) &&
            ((o1 & ~storage_only) || o2))
            fail(f, "a storage-only (TYPELESS) format reported with uses of its own", o1, o2, h1, h2);
        if ((h1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) &&
            ((o1 & (shader_reads | D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) & ~h1) || (o2 & ~h2)))
            fail(f, "a depth format reported as a shader resource or UAV", o1, o2, h1, h2);
        if ((h1 & D3D12_FORMAT_SUPPORT1_MIP) && (o1 & (dims | D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)) &&
            !(o1 & D3D12_FORMAT_SUPPORT1_MIP))
            fail(f, "no MIP where the host has mips", o1, o2, h1, h2);
        if (bool(o1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE) != bool(o1 & D3D12_FORMAT_SUPPORT1_SHADER_GATHER))
            fail(f, "gather and sample disagree", o1, o2, h1, h2);
        if (bool(o2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE) != bool(o2 & D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD))
            fail(f, "typed UAV load and store disagree", o1, o2, h1, h2);
        if (f == DXGI_FORMAT_R32_UINT && (o1 & D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW) &&
            (o2 & atomics) != atomics)
            fail(f, "R32_UINT typed UAV without its atomics", o1, o2, h1, h2);
        const bool ms_target = (o1 & D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET) != 0;
        if (ms_target || (o1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL)) {
            const bool levels = multisampled(DXGI_FORMAT(f));
            if (ms_target && !levels) fail(f, "MULTISAMPLE_RENDERTARGET without quality levels", o1, o2, h1, h2);
            if ((o1 & D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL) && levels && !ms_target)
                fail(f, "a multisampled depth format without MULTISAMPLE_RENDERTARGET", o1, o2, h1, h2);
        }
    }
    if (!unreported.empty())
        std::printf("FORMAT_SUPPORT: the layer reports no uses for host formats %s (not compared)\n",
                    unreported.c_str());
    // A layer that answers nothing passes every check above; a working one shares far more than this with any host.
    constexpr unsigned minimum_compared = 40;
    if (compared < minimum_compared) {
        std::fprintf(stderr, "FORMAT_SUPPORT: only %u formats compared (expected at least %u)\n", compared,
                     minimum_compared);
        ++mismatches;
    }
    std::printf("FORMAT_SUPPORT against the host on %s: %u formats compared, %d mismatches\n", adapter.c_str(),
                compared, mismatches);
    return mismatches;
}

// Clears a 1x1 render target of `format` on `device` (the layer's PC-ABI device or the host's) to `colour` and reads
// its texel back into `texel` (8 bytes, zero beyond the format's size). False if any step fails.
bool clear_and_read_texel(ID3D12Device* device, DXGI_FORMAT format, const float colour[4],
                          std::array<std::uint8_t, 8>& texel) {
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC desc = texture_desc(1, 1, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
    const D3D12_RESOURCE_DESC buffer = buffer_desc(256);
    const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12Resource> target, readback;
    Ref<ID3D12DescriptorHeap> heap;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                               D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, IID_PPV_ARGS(&target.p))) ||
        FAILED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                               D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))) ||
        FAILED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))) ||
        FAILED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))) ||
        FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                         IID_PPV_ARGS(&list.p))) ||
        FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))))
        return false;
    const D3D12_CPU_DESCRIPTOR_HANDLE rtv = heap->GetCPUDescriptorHandleForHeapStart();
    device->CreateRenderTargetView(target.p, nullptr, rtv);
    list->ClearRenderTargetView(rtv, colour, 0, nullptr);
    const D3D12_RESOURCE_BARRIER to_copy =
        transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &to_copy);
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{0, {format, 1, 1, 1, 256}};
    const auto src = subresource(target.p, 0);
    const auto out = footprint(readback.p, placed);
    list->CopyTextureRegion(&out, 0, 0, 0, &src, nullptr);
    if (list->Close() != S_OK) return false;
    ID3D12CommandList* lists[] = {list.p};
    queue->ExecuteCommandLists(1, lists);
    if (FAILED(queue->Signal(fence.p, 1)) || FAILED(fence->SetEventOnCompletion(1, nullptr))) return false;
    void* mapped = nullptr;
    if (FAILED(readback->Map(0, nullptr, &mapped)) || !mapped) return false;
    std::memcpy(texel.data(), mapped, texel.size());
    readback->Unmap(0, nullptr);
    return true;
}

// Item 1 of the FORMAT_SUPPORT finding in #1380: formats the host creates that vulkan_format() didn't map. Each one
// is created as a texture (as on the host, when there is one), reported as a 2D texture, and a 4x4 upload copied in
// and back out reads back bit for bit. Every one both the layer and the host render to is cleared to the same colour
// on both, and the bytes must match: a mapping with the right size but the wrong numeric type (SNORM as UNORM, SINT as
// UINT) clears differently. A4B4G4R4_UNORM is also cleared to pure R, G, B and A where the layer reports a render
// target: the host keeps red in bits 12-15 and alpha in bits 0-3 (measured on AMD and WARP), Vulkan's R4G4B4A4_PACK16.
int new_format_round_trips(ID3D12Device* device, const rt::d3d12::VulkanDeviceInfo& info, bool r4g4b4a4) {
    struct NewFormat {
        DXGI_FORMAT format;
        UINT bytes;
    };
    constexpr NewFormat formats[] = {
        {DXGI_FORMAT_R16_UINT, 2},           {DXGI_FORMAT_R16_SINT, 2},       {DXGI_FORMAT_R16_SNORM, 2},
        {DXGI_FORMAT_R8_UINT, 1},            {DXGI_FORMAT_R8_SINT, 1},        {DXGI_FORMAT_R8_SNORM, 1},
        {DXGI_FORMAT_R8_TYPELESS, 1},        {DXGI_FORMAT_R16G16_UNORM, 4},   {DXGI_FORMAT_R16G16_SNORM, 4},
        {DXGI_FORMAT_R16G16B16A16_SNORM, 8}, {DXGI_FORMAT_A4B4G4R4_UNORM, 2},
    };
    constexpr UINT size = 4, pitch = 256, slot = 1024;
    int failures = 0;
    unsigned host_compared = 0, host_lacking = 0;
    const auto fail = [&](DXGI_FORMAT format, const char* what) {
        if (failures++ < 16) std::fprintf(stderr, "new format %d: %s\n", int(format), what);
    };
    // The host on the Vulkan device's own adapter: on a two-GPU PC the default adapter is another GPU's driver.
    Ref<ID3D12Device> host;
    std::string adapter;
    if (create_host_device_for_physical(info.instance, info.physical, info.get_instance_proc, host, adapter) &&
        adapter == "default adapter")
        host.reset();
    if (!host.p) std::printf("new formats: no host device on the same adapter, host comparisons skipped\n");
    // Vulkan doesn't require R4G4B4A4_UNORM_PACK16 (A4B4G4R4): where the driver can't sample it, it is skipped.
    // The skip follows the Vulkan driver, not the layer's answers, so a dropped mapping still fails.
    const auto skipped = [&](DXGI_FORMAT format) { return format == DXGI_FORMAT_A4B4G4R4_UNORM && !r4g4b4a4; };
    if (!r4g4b4a4) std::printf("new formats: the Vulkan driver can't sample R4G4B4A4_UNORM_PACK16, A4B4G4R4 skipped\n");
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
        readback_heap{D3D12_HEAP_TYPE_READBACK};
    const UINT64 buffer_bytes = UINT64(slot) * std::size(formats);
    const D3D12_RESOURCE_DESC buffer = buffer_desc(buffer_bytes);
    Ref<ID3D12Resource> upload, readback;
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                     D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                     IID_PPV_ARGS(&upload.p))));
    EXPECT(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &buffer,
                                                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                     IID_PPV_ARGS(&readback.p))));
    if (!upload.p || !readback.p) return 1;
    std::uint8_t *up = nullptr, *down = nullptr;
    EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))));
    EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))));
    if (!up || !down) return 1;
    std::memset(down, 0xcd, size_t(buffer_bytes));
    for (UINT64 i = 0; i < buffer_bytes; ++i) up[i] = std::uint8_t(i * 37 + (i >> 8) * 11 + 5);

    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
    EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
    EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                               IID_PPV_ARGS(&list.p))));
    EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
    if (!queue.p || !list.p || !fence.p) return 1;
    UINT64 fence_value = 0;
    const auto submit = [&] {
        const bool closed = list->Close() == S_OK;
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        const bool replayed = rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success;
        ++fence_value;
        const bool waited = SUCCEEDED(queue->Signal(fence.p, fence_value)) &&
                            SUCCEEDED(fence->SetEventOnCompletion(fence_value, nullptr));
        if (!closed || !replayed || !waited) fail(DXGI_FORMAT_UNKNOWN, "a command list didn't close, replay or finish");
    };

    std::array<Ref<ID3D12Resource>, std::size(formats)> textures;
    for (size_t i = 0; i < std::size(formats); ++i) {
        const DXGI_FORMAT format = formats[i].format;
        if (skipped(format)) continue;
        D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
        if (FAILED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))) ||
            !(support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D))
            fail(format, "FORMAT_SUPPORT reports no 2D texture");
        const D3D12_RESOURCE_DESC desc = texture_desc(size, size, format, D3D12_RESOURCE_FLAG_NONE);
        if (FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                   D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                   IID_PPV_ARGS(&textures[i].p)))) {
            fail(format, "CreateCommittedResource failed");
            continue;
        }
        if (host.p) {
            // A host runtime older than the format has nothing to match: A4B4G4R4_UNORM is not in Windows 10's D3D12,
            // which answers the query with an error (E_FAIL on Windows 10 build 19045, E_INVALIDARG on another 19041
            // host) or reports no 2D texture, and refuses the texture. Compare only where the host itself says it has
            // the format, so a host that reports it and still refuses the texture fails as before. The answer is
            // printed, so an unexpected code stays visible.
            D3D12_FEATURE_DATA_FORMAT_SUPPORT host_support{format};
            const HRESULT query = host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &host_support,
                                                            sizeof(host_support));
            const bool host_has_format = SUCCEEDED(query) && (host_support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
            if (!host_has_format) {
                ++host_lacking;
                std::printf("new format %d: the host runtime has no such 2D texture (FORMAT_SUPPORT answered %08lx, "
                            "e.g. Windows 10), host comparison skipped\n",
                            int(format), static_cast<unsigned long>(query));
            } else {
                ++host_compared;
                Ref<ID3D12Resource> native;
                if (FAILED(host->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                         D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                         IID_PPV_ARGS(&native.p))))
                    fail(format, "the host refuses a texture the layer creates");
            }
        }
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
        UINT rows = 0;
        UINT64 row_bytes = 0;
        device->GetCopyableFootprints(&desc, 0, 1, UINT64(slot) * i, &placed, &rows, &row_bytes, nullptr);
        if (placed.Footprint.Format != format || placed.Footprint.RowPitch != pitch || rows != size ||
            row_bytes != UINT64(size) * formats[i].bytes) {
            fail(format, "unexpected copyable footprint");
            textures[i].reset();
            continue;
        }
        const auto dst = subresource(textures[i].p, 0);
        const auto in = footprint(upload.p, placed);
        list->CopyTextureRegion(&dst, 0, 0, 0, &in, nullptr);
        const D3D12_RESOURCE_BARRIER readable =
            transition(textures[i].p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &readable);
        const auto out = footprint(readback.p, placed);
        list->CopyTextureRegion(&out, 0, 0, 0, &dst, nullptr);
    }
    submit();
    unsigned round_trips = 0;
    for (size_t i = 0; i < std::size(formats); ++i) {
        if (!textures[i].p) continue;
        const size_t base = size_t(slot) * i;
        bool same = true;
        for (UINT y = 0; y < size && same; ++y)
            same = std::memcmp(down + base + y * pitch, up + base + y * pitch, size_t(size) * formats[i].bytes) == 0;
        if (same)
            ++round_trips;
        else
            fail(formats[i].format, "a texel read back differs from the one uploaded");
    }

    // The numeric type, against the host. Red is -3 because single-channel formats see only red: SNORM clamps it to
    // -1, SINT keeps -3, UNORM and UINT give 0. -0.4, 0.25 and 7 cover negatives, fractions and clamping in G, B, A;
    // none is an exact half step at 4, 8 or 16 bits (-0.5 is one for 16-bit SNORM, where drivers round differently).
    unsigned clears_matched = 0;
    if (host.p) {
        constexpr float colour[4] = {-3.0f, -0.4f, 0.25f, 7.0f};
        for (const NewFormat& f : formats) {
            if (skipped(f.format)) continue;
            D3D12_FEATURE_DATA_FORMAT_SUPPORT ours_support{f.format}, host_support{f.format};
            const bool both_render =
                SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &ours_support,
                                                      sizeof(ours_support))) &&
                (ours_support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET) &&
                SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &host_support,
                                                    sizeof(host_support))) &&
                (host_support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
            if (!both_render) continue;
            std::array<std::uint8_t, 8> ours{}, theirs{};
            if (!clear_and_read_texel(device, f.format, colour, ours) ||
                !clear_and_read_texel(host.p, f.format, colour, theirs)) {
                fail(f.format, "a render-target clear failed");
                continue;
            }
            if (std::memcmp(ours.data(), theirs.data(), f.bytes) == 0) {
                ++clears_matched;
            } else if (failures++ < 16) {
                std::fprintf(stderr, "new format %d: clear to (-3, -0.4, 0.25, 7) differs from the host:", int(f.format));
                for (UINT b = 0; b < f.bytes; ++b) std::fprintf(stderr, " %02x/%02x", ours[b], theirs[b]);
                std::fprintf(stderr, " (layer/host)\n");
            }
        }
    }

    // A4B4G4R4 cleared to pure red, green, blue and alpha in turn.
    D3D12_FEATURE_DATA_FORMAT_SUPPORT a4_support{DXGI_FORMAT_A4B4G4R4_UNORM};
    const bool a4_target = !skipped(DXGI_FORMAT_A4B4G4R4_UNORM) &&
        SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &a4_support, sizeof(a4_support))) &&
        (a4_support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
    unsigned a4_cleared = 0;
    if (a4_target) {
        const D3D12_RESOURCE_DESC desc =
            texture_desc(1, 1, DXGI_FORMAT_A4B4G4R4_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        Ref<ID3D12Resource> target;
        Ref<ID3D12DescriptorHeap> heap;
        const D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 1};
        EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                         D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr,
                                                         IID_PPV_ARGS(&target.p))));
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        if (target.p && heap.p) {
            const D3D12_CPU_DESCRIPTOR_HANDLE rtv = heap->GetCPUDescriptorHandleForHeapStart();
            device->CreateRenderTargetView(target.p, nullptr, rtv);
            constexpr float colours[4][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
            constexpr std::uint16_t words[4] = {0xf000, 0x0f00, 0x00f0, 0x000f};
            const D3D12_PLACED_SUBRESOURCE_FOOTPRINT texel{0, {DXGI_FORMAT_A4B4G4R4_UNORM, 1, 1, 1, pitch}};
            for (int c = 0; c < 4; ++c) {
                EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
                list->ClearRenderTargetView(rtv, colours[c], 0, nullptr);
                const D3D12_RESOURCE_BARRIER to_copy =
                    transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &to_copy);
                const auto src = subresource(target.p, 0);
                const auto out = footprint(readback.p, texel);
                list->CopyTextureRegion(&out, 0, 0, 0, &src, nullptr);
                const D3D12_RESOURCE_BARRIER back =
                    transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
                list->ResourceBarrier(1, &back);
                submit();
                std::uint16_t got = 0;
                std::memcpy(&got, down, 2);
                if (got == words[c]) {
                    ++a4_cleared;
                } else if (failures++ < 16) {
                    std::fprintf(stderr, "A4B4G4R4 clear to channel %d: got %04x, want %04x\n", c, got, words[c]);
                }
            }
        }
    }
    upload->Unmap(0, nullptr);
    readback->Unmap(0, nullptr);
    char compared[64] = "";
    if (host.p) std::snprintf(compared, sizeof(compared), " (host compared on %u, host lacks %u)", host_compared,
                              host_lacking);
    std::printf("new formats: %u of %zu created and copied both ways bit for bit%s, %u render-target clears "
                "matched the host, A4B4G4R4 render target %s, %d failures\n",
                round_trips, std::size(formats), compared, clears_matched,
                a4_target ? (a4_cleared == 4 ? "cleared to R, G, B and A" : "cleared wrong") : "not reported, skipped",
                failures);
    return failures;
}

// D3D-86: validate CheckFeatureSupport(D3D12_OPTIONS2..21 except 6/8/10/12/17), check size/null refusals and local
// invariants, and verify conservative compatibility against the host PC D3D12 runtime on the same adapter.
int compare_feature_options(ID3D12Device* ours, const rt::d3d12::VulkanDeviceInfo& info) {
    int mismatches = 0;
    const auto fail = [&](const char* what) {
        if (mismatches++ < 32) std::fprintf(stderr, "CheckFeatureSupport options: %s\n", what);
    };
    const auto query_ours_raw = [&](const char* name, D3D12_FEATURE feature, const void* init, void* out,
                                      size_t sz) -> bool {
        if (ours->CheckFeatureSupport(feature, nullptr, UINT(sz)) != E_INVALIDARG) {
            std::fprintf(stderr, "%s: expected E_INVALIDARG for nullptr\n", name);
            ++mismatches;
            return false;
        }
        std::vector<std::uint8_t> scratch(sz + 1, 0xAA);
        if (init) std::memcpy(scratch.data(), init, sz);
        for (UINT bad_size : {0u, UINT(sz - 1), UINT(sz + 1)}) {
            if (ours->CheckFeatureSupport(feature, scratch.data(), bad_size) != E_INVALIDARG) {
                std::fprintf(stderr, "%s: expected E_INVALIDARG for size %u\n", name, bad_size);
                ++mismatches;
                return false;
            }
        }
        if (init)
            std::memcpy(out, init, sz);
        else
            std::memset(out, 0xAA, sz);
        if (FAILED(ours->CheckFeatureSupport(feature, out, UINT(sz)))) {
            std::fprintf(stderr, "%s: CheckFeatureSupport failed on Vulkan device\n", name);
            ++mismatches;
            return false;
        }
        return true;
    };
    const auto query_ours = [&](auto tag_feature, auto& out) -> bool {
        using T = std::remove_reference_t<decltype(out)>;
        return query_ours_raw(tag_feature.first, tag_feature.second, nullptr, &out, sizeof(T));
    };
    const auto query_ours_input = [&](auto tag_feature, const auto& init, auto& out) -> bool {
        using T = std::remove_reference_t<decltype(out)>;
        return query_ours_raw(tag_feature.first, tag_feature.second, &init, &out, sizeof(T));
    };

    D3D12_FEATURE_DATA_D3D12_OPTIONS2 o2{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS3 o3{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS4 o4{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS7 o7{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS9 o9{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS11 o11{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS13 o13{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS14 o14{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS15 o15{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS16 o16{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS18 o18{};
    D3D12_FEATURE_DATA_D3D12_OPTIONS19 o19{};
    rt::d3d12::compat::FeatureDataOptions20 o20{};
    rt::d3d12::compat::FeatureDataOptions21 o21{};

    query_ours(std::pair{"OPTIONS2", D3D12_FEATURE_D3D12_OPTIONS2}, o2);
    query_ours(std::pair{"OPTIONS3", D3D12_FEATURE_D3D12_OPTIONS3}, o3);
    query_ours(std::pair{"OPTIONS4", D3D12_FEATURE_D3D12_OPTIONS4}, o4);
    query_ours(std::pair{"OPTIONS5", D3D12_FEATURE_D3D12_OPTIONS5}, o5);
    query_ours(std::pair{"OPTIONS7", D3D12_FEATURE_D3D12_OPTIONS7}, o7);
    query_ours(std::pair{"OPTIONS9", D3D12_FEATURE_D3D12_OPTIONS9}, o9);
    query_ours(std::pair{"OPTIONS11", D3D12_FEATURE_D3D12_OPTIONS11}, o11);
    query_ours(std::pair{"OPTIONS13", D3D12_FEATURE_D3D12_OPTIONS13}, o13);
    query_ours(std::pair{"OPTIONS14", D3D12_FEATURE_D3D12_OPTIONS14}, o14);
    query_ours(std::pair{"OPTIONS15", D3D12_FEATURE_D3D12_OPTIONS15}, o15);
    query_ours(std::pair{"OPTIONS16", D3D12_FEATURE_D3D12_OPTIONS16}, o16);
    query_ours(std::pair{"OPTIONS18", D3D12_FEATURE_D3D12_OPTIONS18}, o18);
    query_ours(std::pair{"OPTIONS19", D3D12_FEATURE_D3D12_OPTIONS19}, o19);
    query_ours(std::pair{"OPTIONS20", rt::d3d12::compat::kFeatureOptions20}, o20);
    query_ours(std::pair{"OPTIONS21", rt::d3d12::compat::kFeatureOptions21}, o21);

    D3D12_FEATURE_DATA_SHADER_CACHE shader_cache{};
    D3D12_FEATURE_DATA_EXISTING_HEAPS existing_heaps{};
    D3D12_FEATURE_DATA_CROSS_NODE cross_node{};
    D3D12_FEATURE_DATA_DISPLAYABLE displayable{};
    rt::d3d12::compat::FeatureSupportFlag predication{};
    rt::d3d12::compat::FeatureSupportFlag driver_state{};
    rt::d3d12::compat::FeatureSupportFlag bypass_hash{};
    D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT prot_support{};
    D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT prot_count{};
    D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES prot_types{};
    D3D12_FEATURE_DATA_SERIALIZATION serialization{};
    D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY queue_priority{};

    query_ours(std::pair{"SHADER_CACHE", D3D12_FEATURE_SHADER_CACHE}, shader_cache);
    query_ours(std::pair{"EXISTING_HEAPS", D3D12_FEATURE_EXISTING_HEAPS}, existing_heaps);
    query_ours(std::pair{"CROSS_NODE", D3D12_FEATURE_CROSS_NODE}, cross_node);
    query_ours(std::pair{"DISPLAYABLE", D3D12_FEATURE_DISPLAYABLE}, displayable);
    query_ours(std::pair{"PREDICATION", rt::d3d12::compat::kFeaturePredication}, predication);
    query_ours(std::pair{"APPLICATION_SPECIFIC_DRIVER_STATE",
                         rt::d3d12::compat::kFeatureApplicationSpecificDriverState},
               driver_state);
    query_ours(std::pair{"BYTECODE_BYPASS_HASH_SUPPORTED",
                         rt::d3d12::compat::kFeatureBytecodeBypassHashSupported},
               bypass_hash);
    query_ours_input(std::pair{"PROTECTED_RESOURCE_SESSION_SUPPORT", D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT},
                     D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT{0, {}}, prot_support);
    query_ours_input(
        std::pair{"PROTECTED_RESOURCE_SESSION_TYPE_COUNT", D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT},
        D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT{0, 0}, prot_count);
    query_ours_input(std::pair{"PROTECTED_RESOURCE_SESSION_TYPES", D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPES},
                     D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES{0, 0, nullptr}, prot_types);
    query_ours_input(std::pair{"SERIALIZATION", D3D12_FEATURE_SERIALIZATION}, D3D12_FEATURE_DATA_SERIALIZATION{0, {}},
                     serialization);
    query_ours_input(std::pair{"COMMAND_QUEUE_PRIORITY", D3D12_FEATURE_COMMAND_QUEUE_PRIORITY},
                     D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY{D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                               D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, FALSE},
                     queue_priority);

    std::uint32_t unknown_feature = 0;
    if (ours->CheckFeatureSupport(static_cast<D3D12_FEATURE>(9999), &unknown_feature, sizeof(unknown_feature)) !=
        E_INVALIDARG)
        fail("unknown feature unexpectedly answered");

    // Local invariants for what this layer actually implements.
    if (o2.DepthBoundsTestSupported ||
        o2.ProgrammableSamplePositionsTier != D3D12_PROGRAMMABLE_SAMPLE_POSITIONS_TIER_NOT_SUPPORTED)
        fail("OPTIONS2 unexpected support");
    if (o3.CopyQueueTimestampQueriesSupported || o3.CastingFullyTypedFormatSupported ||
        o3.WriteBufferImmediateSupportFlags != D3D12_COMMAND_LIST_SUPPORT_FLAG_NONE ||
        o3.ViewInstancingTier != D3D12_VIEW_INSTANCING_TIER_NOT_SUPPORTED || o3.BarycentricsSupported)
        fail("OPTIONS3 unexpected support");
    if (!o4.MSAA64KBAlignedTextureSupported ||
        o4.SharedResourceCompatibilityTier != D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER_0 ||
        o4.Native16BitShaderOpsSupported)
        fail("OPTIONS4 mismatch");
    if (o5.SRVOnlyTiledResourceTier3 || o5.RenderPassesTier != D3D12_RENDER_PASS_TIER_0 ||
        o5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED)
        fail("OPTIONS5 unexpected support");
    if (o7.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED ||
        o7.SamplerFeedbackTier != D3D12_SAMPLER_FEEDBACK_TIER_NOT_SUPPORTED)
        fail("OPTIONS7 unexpected support");
    if (o9.MeshShaderPipelineStatsSupported || o9.MeshShaderSupportsFullRangeRenderTargetArrayIndex ||
        o9.AtomicInt64OnTypedResourceSupported || o9.AtomicInt64OnGroupSharedSupported ||
        o9.DerivativesInMeshAndAmplificationShadersSupported ||
        o9.WaveMMATier != D3D12_WAVE_MMA_TIER_NOT_SUPPORTED)
        fail("OPTIONS9 unexpected support");
    if (o11.AtomicInt64OnDescriptorHeapResourceSupported) fail("OPTIONS11 unexpected support");
    if (o13.UnrestrictedBufferTextureCopyPitchSupported || o13.UnrestrictedVertexElementAlignmentSupported ||
        o13.InvertedViewportHeightFlipsYSupported || o13.InvertedViewportDepthFlipsZSupported ||
        o13.TextureCopyBetweenDimensionsSupported || o13.AlphaBlendFactorSupported)
        fail("OPTIONS13 unexpected support");
    if (o14.AdvancedTextureOpsSupported || o14.WriteableMSAATexturesSupported ||
        o14.IndependentFrontAndBackStencilRefMaskSupported)
        fail("OPTIONS14 unexpected support");
    if (o15.TriangleFanSupported || o15.DynamicIndexBufferStripCutSupported) fail("OPTIONS15 unexpected support");
    if (o16.DynamicDepthBiasSupported || o16.GPUUploadHeapSupported) fail("OPTIONS16 unexpected support");
    if (o18.RenderPassesValid) fail("OPTIONS18 unexpected support");
    if (o19.MismatchingOutputDimensionsSupported || (o19.SupportedSampleCountsWithNoOutputs & ~0x3Fu) != 0 ||
        (o19.SupportedSampleCountsWithNoOutputs & 1u) == 0 || o19.PointSamplingAddressesNeverRoundUp ||
        o19.RasterizerDesc2Supported || o19.NarrowQuadrilateralLinesSupported ||
        o19.AnisoFilterWithPointMipSupported != (info.sampler_anisotropy ? TRUE : FALSE) ||
        o19.MaxSamplerDescriptorHeapSize != D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE ||
        o19.MaxSamplerDescriptorHeapSizeWithStaticSamplers != D3D12_MAX_SHADER_VISIBLE_SAMPLER_HEAP_SIZE ||
        o19.MaxViewDescriptorHeapSize != D3D12_MAX_SHADER_VISIBLE_DESCRIPTOR_HEAP_SIZE_TIER_2 ||
        o19.ComputeOnlyCustomHeapSupported)
        fail("OPTIONS19 mismatch");
    if (o20.ComputeOnlyWriteWatchSupported || o20.RecreateAtTier != rt::d3d12::compat::kRecreateAtTierNotSupported)
        fail("OPTIONS20 unexpected support");
    if (o21.WorkGraphsTier != rt::d3d12::compat::kWorkGraphsTierNotSupported ||
        o21.ExecuteIndirectTier != rt::d3d12::compat::kExecuteIndirectTier1_0 ||
        o21.SampleCmpGradientAndBiasSupported || o21.ExtendedCommandInfoSupported)
        fail("OPTIONS21 unexpected support");

    // Verify OPTIONS19.AnisoFilterWithPointMipSupported follows VulkanDeviceInfo::sampler_anisotropy.
    {
        rt::d3d12::VulkanDeviceInfo toggled_info = info;
        toggled_info.sampler_anisotropy = !info.sampler_anisotropy;
        Ref<ID3D12Device> toggled;
        if (FAILED(rt::d3d12::create_vulkan_device(toggled_info, IID_PPV_ARGS(&toggled.p)))) {
            fail("failed to create toggled sampler_anisotropy device");
        } else {
            D3D12_FEATURE_DATA_D3D12_OPTIONS19 toggled19{};
            if (FAILED(toggled->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS19, &toggled19, sizeof(toggled19))) ||
                toggled19.AnisoFilterWithPointMipSupported != (toggled_info.sampler_anisotropy ? TRUE : FALSE))
                fail("OPTIONS19.AnisoFilterWithPointMipSupported did not follow sampler_anisotropy");
        }
    }

    if (shader_cache.SupportFlags != D3D12_SHADER_CACHE_SUPPORT_NONE) fail("SHADER_CACHE unexpected support");
    if (existing_heaps.Supported) fail("EXISTING_HEAPS unexpected support");
    if (cross_node.SharingTier != D3D12_CROSS_NODE_SHARING_TIER_NOT_SUPPORTED || cross_node.AtomicShaderInstructions)
        fail("CROSS_NODE unexpected support");
    if (displayable.DisplayableTexture ||
        displayable.SharedResourceCompatibilityTier != D3D12_SHARED_RESOURCE_COMPATIBILITY_TIER_0)
        fail("DISPLAYABLE unexpected support");
    if (predication.Supported != (info.conditional_rendering ? TRUE : FALSE))
        fail("PREDICATION did not match conditional_rendering");
    if (driver_state.Supported) fail("APPLICATION_SPECIFIC_DRIVER_STATE unexpected support");
    if (bypass_hash.Supported) fail("BYTECODE_BYPASS_HASH_SUPPORTED unexpected support");
    if (prot_support.Support != D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAG_NONE || prot_count.Count != 0)
        fail("PROTECTED_RESOURCE_SESSION unexpected support");
    if (serialization.HeapSerializationTier != D3D12_HEAP_SERIALIZATION_TIER_0)
        fail("SERIALIZATION unexpected tier");

    // NodeIndex > 0, non-zero protected type count, and zeroed meta-command queries must return E_INVALIDARG.
    {
        D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT ps1{
            1, D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAG_SUPPORTED};
        D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT ptc1{1, 99};
        GUID dummy_guid{};
        D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES pt_n1{1, 0, nullptr};
        D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPES pt_c1{0, 1, &dummy_guid};
        D3D12_FEATURE_DATA_SERIALIZATION ser1{1, D3D12_HEAP_SERIALIZATION_TIER_0};
        D3D12_FEATURE_DATA_QUERY_META_COMMAND meta{};
        if (ours->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT, &ps1, sizeof(ps1)) !=
                E_INVALIDARG ||
            ps1.Support != D3D12_PROTECTED_RESOURCE_SESSION_SUPPORT_FLAG_NONE)
            fail("PROTECTED_RESOURCE_SESSION_SUPPORT NodeIndex=1 not rejected");
        if (ours->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT, &ptc1, sizeof(ptc1)) !=
            E_INVALIDARG)
            fail("PROTECTED_RESOURCE_SESSION_TYPE_COUNT NodeIndex=1 not rejected");
        if (ours->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPES, &pt_n1, sizeof(pt_n1)) !=
                E_INVALIDARG ||
            ours->CheckFeatureSupport(D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPES, &pt_c1, sizeof(pt_c1)) !=
                E_INVALIDARG)
            fail("PROTECTED_RESOURCE_SESSION_TYPES invalid node/count not rejected");
        if (ours->CheckFeatureSupport(D3D12_FEATURE_SERIALIZATION, &ser1, sizeof(ser1)) != E_INVALIDARG)
            fail("SERIALIZATION NodeIndex=1 not rejected");
        if (ours->CheckFeatureSupport(D3D12_FEATURE_QUERY_META_COMMAND, &meta, sizeof(meta)) != E_INVALIDARG ||
            ours->CheckFeatureSupport(D3D12_FEATURE_QUERY_META_COMMAND, nullptr, sizeof(meta)) != E_INVALIDARG ||
            ours->CheckFeatureSupport(D3D12_FEATURE_QUERY_META_COMMAND, &meta, sizeof(meta) - 1) != E_INVALIDARG)
            fail("QUERY_META_COMMAND not rejected");
    }

    // COMMAND_QUEUE_PRIORITY matrix + invalid type/priority validation.
    for (D3D12_COMMAND_LIST_TYPE type :
         {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_BUNDLE, D3D12_COMMAND_LIST_TYPE_COMPUTE,
          D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE, D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
          D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE}) {
        for (UINT prio :
             {UINT(D3D12_COMMAND_QUEUE_PRIORITY_NORMAL), UINT(D3D12_COMMAND_QUEUE_PRIORITY_HIGH),
              UINT(D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME)}) {
            D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY q{type, prio, FALSE};
            const bool expect_ok = (type == D3D12_COMMAND_LIST_TYPE_DIRECT ||
                                    type == D3D12_COMMAND_LIST_TYPE_COMPUTE ||
                                    type == D3D12_COMMAND_LIST_TYPE_COPY) &&
                                   prio == D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
            if (ours->CheckFeatureSupport(D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, &q, sizeof(q)) != S_OK ||
                q.PriorityForTypeIsSupported != (expect_ok ? TRUE : FALSE))
                fail("COMMAND_QUEUE_PRIORITY answer mismatch");
        }
    }
    {
        D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY bad_type{static_cast<D3D12_COMMAND_LIST_TYPE>(99),
                                                           D3D12_COMMAND_QUEUE_PRIORITY_NORMAL, TRUE};
        D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY bad_prio{D3D12_COMMAND_LIST_TYPE_DIRECT, 50u, TRUE};
        if (ours->CheckFeatureSupport(D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, &bad_type, sizeof(bad_type)) !=
                E_INVALIDARG ||
            bad_type.PriorityForTypeIsSupported != FALSE ||
            ours->CheckFeatureSupport(D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, &bad_prio, sizeof(bad_prio)) !=
                E_INVALIDARG ||
            bad_prio.PriorityForTypeIsSupported != FALSE)
            fail("COMMAND_QUEUE_PRIORITY invalid type/priority not rejected");
    }

    // Verify D3D12_FEATURE_PREDICATION follows conditional_rendering when disabled on a device without SetPredication.
    if (info.conditional_rendering) {
        rt::d3d12::VulkanDeviceInfo no_pred_info = info;
        no_pred_info.conditional_rendering = false;
        Ref<ID3D12Device> no_pred;
        if (FAILED(rt::d3d12::create_vulkan_device(no_pred_info, IID_PPV_ARGS(&no_pred.p)))) {
            fail("failed to create conditional_rendering=false device");
        } else {
            rt::d3d12::compat::FeatureSupportFlag pred_off{TRUE};
            if (FAILED(no_pred->CheckFeatureSupport(rt::d3d12::compat::kFeaturePredication, &pred_off,
                                                    sizeof(pred_off))) ||
                pred_off.Supported != FALSE)
                fail("PREDICATION did not become FALSE when conditional_rendering is off");
        }
    }

    Ref<ID3D12Device> host;
    std::string adapter_name;
    if (!create_host_device_for_physical(info.instance, info.physical, info.get_instance_proc, host, adapter_name)) {
        std::printf("CheckFeatureSupport options: no host D3D12 device to compare with, skipped\n");
        return mismatches;
    }

    UINT host_compared = 0, host_skipped = 0;
    const auto check_bool = [&](const char* feature, const char* field, BOOL a, BOOL b) {
        if (!a || b) return;
        if (mismatches++ < 32)
            std::fprintf(stderr, "%s.%s: ours TRUE exceeds host FALSE\n", feature, field);
    };
    const auto check_tier = [&](const char* feature, const char* field, int a, int b) {
        if (a <= b) return;
        if (mismatches++ < 32)
            std::fprintf(stderr, "%s.%s: ours tier %d exceeds host %d\n", feature, field, a, b);
    };
    const auto compare_host_input = [&](const char* name, D3D12_FEATURE feature, const auto& ours_val,
                                        const auto& host_init, auto&& verify) {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(ours_val)>>;
        T host_val = host_init;
        if (FAILED(host->CheckFeatureSupport(feature, &host_val, sizeof(host_val)))) {
            ++host_skipped;
            return;
        }
        ++host_compared;
        verify(name, ours_val, host_val);
    };
    const auto compare_host = [&](const char* name, D3D12_FEATURE feature, const auto& ours_val, auto&& verify) {
        using T = std::remove_cv_t<std::remove_reference_t<decltype(ours_val)>>;
        compare_host_input(name, feature, ours_val, T{}, verify);
    };

    compare_host("OPTIONS2", D3D12_FEATURE_D3D12_OPTIONS2, o2,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "DepthBoundsTestSupported", a.DepthBoundsTestSupported, b.DepthBoundsTestSupported);
                     check_tier(n, "ProgrammableSamplePositionsTier", int(a.ProgrammableSamplePositionsTier),
                                int(b.ProgrammableSamplePositionsTier));
                 });
    compare_host("OPTIONS3", D3D12_FEATURE_D3D12_OPTIONS3, o3,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "CopyQueueTimestampQueriesSupported", a.CopyQueueTimestampQueriesSupported,
                                b.CopyQueueTimestampQueriesSupported);
                     check_bool(n, "CastingFullyTypedFormatSupported", a.CastingFullyTypedFormatSupported,
                                b.CastingFullyTypedFormatSupported);
                     if ((UINT(a.WriteBufferImmediateSupportFlags) & ~UINT(b.WriteBufferImmediateSupportFlags)) != 0 &&
                         mismatches++ < 32)
                         std::fprintf(stderr, "%s.WriteBufferImmediateSupportFlags: ours 0x%x not subset of host 0x%x\n",
                                      n, UINT(a.WriteBufferImmediateSupportFlags),
                                      UINT(b.WriteBufferImmediateSupportFlags));
                     check_tier(n, "ViewInstancingTier", int(a.ViewInstancingTier), int(b.ViewInstancingTier));
                     check_bool(n, "BarycentricsSupported", a.BarycentricsSupported, b.BarycentricsSupported);
                 });
    compare_host("OPTIONS4", D3D12_FEATURE_D3D12_OPTIONS4, o4,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "MSAA64KBAlignedTextureSupported", a.MSAA64KBAlignedTextureSupported,
                                b.MSAA64KBAlignedTextureSupported);
                     check_tier(n, "SharedResourceCompatibilityTier", int(a.SharedResourceCompatibilityTier),
                                int(b.SharedResourceCompatibilityTier));
                     check_bool(n, "Native16BitShaderOpsSupported", a.Native16BitShaderOpsSupported,
                                b.Native16BitShaderOpsSupported);
                 });
    compare_host("OPTIONS5", D3D12_FEATURE_D3D12_OPTIONS5, o5,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "SRVOnlyTiledResourceTier3", a.SRVOnlyTiledResourceTier3, b.SRVOnlyTiledResourceTier3);
                     check_tier(n, "RenderPassesTier", int(a.RenderPassesTier), int(b.RenderPassesTier));
                     check_tier(n, "RaytracingTier", int(a.RaytracingTier), int(b.RaytracingTier));
                 });
    compare_host("OPTIONS7", D3D12_FEATURE_D3D12_OPTIONS7, o7,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_tier(n, "MeshShaderTier", int(a.MeshShaderTier), int(b.MeshShaderTier));
                     check_tier(n, "SamplerFeedbackTier", int(a.SamplerFeedbackTier), int(b.SamplerFeedbackTier));
                 });
    compare_host("OPTIONS9", D3D12_FEATURE_D3D12_OPTIONS9, o9,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "MeshShaderPipelineStatsSupported", a.MeshShaderPipelineStatsSupported,
                                b.MeshShaderPipelineStatsSupported);
                     check_bool(n, "MeshShaderSupportsFullRangeRenderTargetArrayIndex",
                                a.MeshShaderSupportsFullRangeRenderTargetArrayIndex,
                                b.MeshShaderSupportsFullRangeRenderTargetArrayIndex);
                     check_bool(n, "AtomicInt64OnTypedResourceSupported", a.AtomicInt64OnTypedResourceSupported,
                                b.AtomicInt64OnTypedResourceSupported);
                     check_bool(n, "AtomicInt64OnGroupSharedSupported", a.AtomicInt64OnGroupSharedSupported,
                                b.AtomicInt64OnGroupSharedSupported);
                     check_bool(n, "DerivativesInMeshAndAmplificationShadersSupported",
                                a.DerivativesInMeshAndAmplificationShadersSupported,
                                b.DerivativesInMeshAndAmplificationShadersSupported);
                     check_tier(n, "WaveMMATier", int(a.WaveMMATier), int(b.WaveMMATier));
                 });
    compare_host("OPTIONS11", D3D12_FEATURE_D3D12_OPTIONS11, o11,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "AtomicInt64OnDescriptorHeapResourceSupported",
                                a.AtomicInt64OnDescriptorHeapResourceSupported,
                                b.AtomicInt64OnDescriptorHeapResourceSupported);
                 });
    compare_host("OPTIONS13", D3D12_FEATURE_D3D12_OPTIONS13, o13,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "UnrestrictedBufferTextureCopyPitchSupported",
                                a.UnrestrictedBufferTextureCopyPitchSupported,
                                b.UnrestrictedBufferTextureCopyPitchSupported);
                     check_bool(n, "UnrestrictedVertexElementAlignmentSupported",
                                a.UnrestrictedVertexElementAlignmentSupported,
                                b.UnrestrictedVertexElementAlignmentSupported);
                     check_bool(n, "InvertedViewportHeightFlipsYSupported", a.InvertedViewportHeightFlipsYSupported,
                                b.InvertedViewportHeightFlipsYSupported);
                     check_bool(n, "InvertedViewportDepthFlipsZSupported", a.InvertedViewportDepthFlipsZSupported,
                                b.InvertedViewportDepthFlipsZSupported);
                     check_bool(n, "TextureCopyBetweenDimensionsSupported", a.TextureCopyBetweenDimensionsSupported,
                                b.TextureCopyBetweenDimensionsSupported);
                     check_bool(n, "AlphaBlendFactorSupported", a.AlphaBlendFactorSupported,
                                b.AlphaBlendFactorSupported);
                 });
    compare_host("OPTIONS14", D3D12_FEATURE_D3D12_OPTIONS14, o14,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "AdvancedTextureOpsSupported", a.AdvancedTextureOpsSupported,
                                b.AdvancedTextureOpsSupported);
                     check_bool(n, "WriteableMSAATexturesSupported", a.WriteableMSAATexturesSupported,
                                b.WriteableMSAATexturesSupported);
                     check_bool(n, "IndependentFrontAndBackStencilRefMaskSupported",
                                a.IndependentFrontAndBackStencilRefMaskSupported,
                                b.IndependentFrontAndBackStencilRefMaskSupported);
                 });
    compare_host("OPTIONS15", D3D12_FEATURE_D3D12_OPTIONS15, o15,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "TriangleFanSupported", a.TriangleFanSupported, b.TriangleFanSupported);
                     check_bool(n, "DynamicIndexBufferStripCutSupported", a.DynamicIndexBufferStripCutSupported,
                                b.DynamicIndexBufferStripCutSupported);
                 });
    compare_host("OPTIONS16", D3D12_FEATURE_D3D12_OPTIONS16, o16,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "DynamicDepthBiasSupported", a.DynamicDepthBiasSupported,
                                b.DynamicDepthBiasSupported);
                     check_bool(n, "GPUUploadHeapSupported", a.GPUUploadHeapSupported, b.GPUUploadHeapSupported);
                 });
    compare_host("OPTIONS18", D3D12_FEATURE_D3D12_OPTIONS18, o18,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "RenderPassesValid", a.RenderPassesValid, b.RenderPassesValid);
                 });
    compare_host("OPTIONS19", D3D12_FEATURE_D3D12_OPTIONS19, o19,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "MismatchingOutputDimensionsSupported", a.MismatchingOutputDimensionsSupported,
                                b.MismatchingOutputDimensionsSupported);
                     if (b.SupportedSampleCountsWithNoOutputs != 0 &&
                         (a.SupportedSampleCountsWithNoOutputs & ~b.SupportedSampleCountsWithNoOutputs) != 0 &&
                         mismatches++ < 32)
                         std::fprintf(stderr, "%s.SupportedSampleCountsWithNoOutputs: ours 0x%x not subset of host 0x%x\n",
                                      n, a.SupportedSampleCountsWithNoOutputs, b.SupportedSampleCountsWithNoOutputs);
                     check_bool(n, "PointSamplingAddressesNeverRoundUp", a.PointSamplingAddressesNeverRoundUp,
                                b.PointSamplingAddressesNeverRoundUp);
                     check_bool(n, "RasterizerDesc2Supported", a.RasterizerDesc2Supported, b.RasterizerDesc2Supported);
                     check_bool(n, "NarrowQuadrilateralLinesSupported", a.NarrowQuadrilateralLinesSupported,
                                b.NarrowQuadrilateralLinesSupported);
                     check_bool(n, "AnisoFilterWithPointMipSupported", a.AnisoFilterWithPointMipSupported,
                                b.AnisoFilterWithPointMipSupported);
                     if (b.MaxSamplerDescriptorHeapSize > 0 &&
                         a.MaxSamplerDescriptorHeapSize > b.MaxSamplerDescriptorHeapSize && mismatches++ < 32)
                         std::fprintf(stderr, "%s.MaxSamplerDescriptorHeapSize: ours %u > host %u\n", n,
                                      a.MaxSamplerDescriptorHeapSize, b.MaxSamplerDescriptorHeapSize);
                     if (b.MaxSamplerDescriptorHeapSizeWithStaticSamplers > 0 &&
                         a.MaxSamplerDescriptorHeapSizeWithStaticSamplers >
                             b.MaxSamplerDescriptorHeapSizeWithStaticSamplers &&
                         mismatches++ < 32)
                         std::fprintf(stderr,
                                      "%s.MaxSamplerDescriptorHeapSizeWithStaticSamplers: ours %u > host %u\n", n,
                                      a.MaxSamplerDescriptorHeapSizeWithStaticSamplers,
                                      b.MaxSamplerDescriptorHeapSizeWithStaticSamplers);
                     if (b.MaxViewDescriptorHeapSize > 0 &&
                         a.MaxViewDescriptorHeapSize > b.MaxViewDescriptorHeapSize && mismatches++ < 32)
                         std::fprintf(stderr, "%s.MaxViewDescriptorHeapSize: ours %u > host %u\n", n,
                                      a.MaxViewDescriptorHeapSize, b.MaxViewDescriptorHeapSize);
                     check_bool(n, "ComputeOnlyCustomHeapSupported", a.ComputeOnlyCustomHeapSupported,
                                b.ComputeOnlyCustomHeapSupported);
                 });
    compare_host("OPTIONS20", rt::d3d12::compat::kFeatureOptions20, o20,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "ComputeOnlyWriteWatchSupported", a.ComputeOnlyWriteWatchSupported,
                                b.ComputeOnlyWriteWatchSupported);
                     check_tier(n, "RecreateAtTier", int(a.RecreateAtTier), int(b.RecreateAtTier));
                 });
    compare_host("OPTIONS21", rt::d3d12::compat::kFeatureOptions21, o21,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_tier(n, "WorkGraphsTier", int(a.WorkGraphsTier), int(b.WorkGraphsTier));
                     check_tier(n, "ExecuteIndirectTier", int(a.ExecuteIndirectTier), int(b.ExecuteIndirectTier));
                     check_bool(n, "SampleCmpGradientAndBiasSupported", a.SampleCmpGradientAndBiasSupported,
                                b.SampleCmpGradientAndBiasSupported);
                     check_bool(n, "ExtendedCommandInfoSupported", a.ExtendedCommandInfoSupported,
                                b.ExtendedCommandInfoSupported);
                 });

    compare_host("SHADER_CACHE", D3D12_FEATURE_SHADER_CACHE, shader_cache,
                 [&](const char* n, const auto& a, const auto& b) {
                     if ((UINT(a.SupportFlags) & ~UINT(b.SupportFlags)) != 0 && mismatches++ < 32)
                         std::fprintf(stderr, "%s.SupportFlags: ours 0x%x not subset of host 0x%x\n", n,
                                      UINT(a.SupportFlags), UINT(b.SupportFlags));
                 });
    compare_host("EXISTING_HEAPS", D3D12_FEATURE_EXISTING_HEAPS, existing_heaps,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "Supported", a.Supported, b.Supported);
                 });
    compare_host("CROSS_NODE", D3D12_FEATURE_CROSS_NODE, cross_node,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_tier(n, "SharingTier", int(a.SharingTier), int(b.SharingTier));
                     check_bool(n, "AtomicShaderInstructions", a.AtomicShaderInstructions, b.AtomicShaderInstructions);
                 });
    compare_host("DISPLAYABLE", D3D12_FEATURE_DISPLAYABLE, displayable,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "DisplayableTexture", a.DisplayableTexture, b.DisplayableTexture);
                     check_tier(n, "SharedResourceCompatibilityTier", int(a.SharedResourceCompatibilityTier),
                                int(b.SharedResourceCompatibilityTier));
                 });
    compare_host("PREDICATION", rt::d3d12::compat::kFeaturePredication, predication,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "Supported", a.Supported, b.Supported);
                 });
    compare_host("APPLICATION_SPECIFIC_DRIVER_STATE", rt::d3d12::compat::kFeatureApplicationSpecificDriverState,
                 driver_state,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "Supported", a.Supported, b.Supported);
                 });
    compare_host("BYTECODE_BYPASS_HASH_SUPPORTED", rt::d3d12::compat::kFeatureBytecodeBypassHashSupported,
                 bypass_hash,
                 [&](const char* n, const auto& a, const auto& b) {
                     check_bool(n, "Supported", a.Supported, b.Supported);
                 });
    compare_host_input("PROTECTED_RESOURCE_SESSION_SUPPORT", D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_SUPPORT,
                       prot_support, D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_SUPPORT{0, {}},
                       [&](const char* n, const auto& a, const auto& b) {
                           if ((UINT(a.Support) & ~UINT(b.Support)) != 0 && mismatches++ < 32)
                               std::fprintf(stderr, "%s.Support: ours 0x%x not subset of host 0x%x\n", n, UINT(a.Support),
                                            UINT(b.Support));
                       });
    compare_host_input("PROTECTED_RESOURCE_SESSION_TYPE_COUNT", D3D12_FEATURE_PROTECTED_RESOURCE_SESSION_TYPE_COUNT,
                       prot_count, D3D12_FEATURE_DATA_PROTECTED_RESOURCE_SESSION_TYPE_COUNT{0, 0},
                       [&](const char* n, const auto& a, const auto& b) {
                           if (a.Count > b.Count && mismatches++ < 32)
                               std::fprintf(stderr, "%s.Count: ours %u > host %u\n", n, a.Count, b.Count);
                       });
    compare_host_input("SERIALIZATION", D3D12_FEATURE_SERIALIZATION, serialization, D3D12_FEATURE_DATA_SERIALIZATION{0, {}},
                       [&](const char* n, const auto& a, const auto& b) {
                           check_tier(n, "HeapSerializationTier", int(a.HeapSerializationTier),
                                      int(b.HeapSerializationTier));
                       });
    for (D3D12_COMMAND_LIST_TYPE type :
         {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_LIST_TYPE_BUNDLE, D3D12_COMMAND_LIST_TYPE_COMPUTE,
          D3D12_COMMAND_LIST_TYPE_COPY, D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE, D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
          D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE}) {
        for (UINT prio :
             {UINT(D3D12_COMMAND_QUEUE_PRIORITY_NORMAL), UINT(D3D12_COMMAND_QUEUE_PRIORITY_HIGH),
              UINT(D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME)}) {
            D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY ours_q{type, prio, FALSE};
            if (SUCCEEDED(ours->CheckFeatureSupport(D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, &ours_q, sizeof(ours_q)))) {
                compare_host_input("COMMAND_QUEUE_PRIORITY", D3D12_FEATURE_COMMAND_QUEUE_PRIORITY, ours_q,
                                   D3D12_FEATURE_DATA_COMMAND_QUEUE_PRIORITY{type, prio, FALSE},
                                   [&](const char* n, const auto& a, const auto& b) {
                                       check_bool(n, "PriorityForTypeIsSupported", a.PriorityForTypeIsSupported,
                                                  b.PriorityForTypeIsSupported);
                                   });
            }
        }
    }

    std::printf("CheckFeatureSupport options (%s): %u feature queries compared (%u host-unsupported skipped), "
                "MSAA64KB=%d, NoOutputsSamples=0x%x, MaxViewHeap=%u, %d mismatches\n",
                adapter_name.empty() ? "host D3D12" : adapter_name.c_str(), host_compared, host_skipped,
                int(o4.MSAA64KBAlignedTextureSupported), o19.SupportedSampleCountsWithNoOutputs,
                o19.MaxViewDescriptorHeapSize, mismatches);
    return mismatches;
}

// A software Vulkan device (llvmpipe on CI) and the host's WARP device answer about different machines, so the
// OPTIONS1 comparison is skipped there; on a GPU both runtimes see the same hardware.
bool software_vulkan_device(const vktest::Device& vk) {
    const auto get_properties = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties>(
        vk.instance_proc(vk.instance, "vkGetPhysicalDeviceProperties"));
    if (!get_properties) return true;
    VkPhysicalDeviceProperties properties{};
    get_properties(vk.physical, &properties);
    return properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU;
}

// D3D12_OPTIONS1 (D3D-86): wave ops and the lane range come from the Vulkan device's subgroup properties, so the
// host PC runtime on the same GPU is the oracle. Returns the number of mismatches, -2 for a software Vulkan device,
// or -1 when the host has no D3D12 device.
int compare_options1(ID3D12Device* ours, const vktest::Device& vk) {
    if (software_vulkan_device(vk)) return -2;
    // The host device must be the same GPU (matched by LUID, then PCI ids), not the default adapter: on a PC with
    // an AMD iGPU and an NVIDIA card the default is the NVIDIA, whose 32..32 lanes don't match the AMD's 32..64.
    Ref<ID3D12Device> host;
    std::string adapter_name;
    if (!create_host_device_for_physical(vk.instance, vk.physical, vk.instance_proc, host, adapter_name)) return -1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 theirs{};
    if (FAILED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &theirs, sizeof(theirs)))) return -1;
    D3D12_FEATURE_DATA_D3D12_OPTIONS1 mine{};
    if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &mine, sizeof(mine)))) return 1;
    int mismatches = 0;
    const auto same = [&](const char* field, UINT got, UINT want) {
        if (got == want) return;
        if (mismatches++ < 4)
            std::fprintf(stderr, "D3D12_OPTIONS1 %s: ours %u, host %u\n", field, got, want);
    };
    same("WaveOps", mine.WaveOps, theirs.WaveOps);
    same("WaveLaneCountMin", mine.WaveLaneCountMin, theirs.WaveLaneCountMin);
    same("WaveLaneCountMax", mine.WaveLaneCountMax, theirs.WaveLaneCountMax);
    same("ExpandedComputeResourceStates", mine.ExpandedComputeResourceStates,
         theirs.ExpandedComputeResourceStates);
    // TotalLaneCount is deliberately not compared: the host counts every lane of the device (6144 on the RTX 5070)
    // and the layer answers one full wave (docs section 4.1).
    std::printf("D3D12_OPTIONS1: lanes %u..%u, total %u (host total %u)\n", mine.WaveLaneCountMin,
                mine.WaveLaneCountMax, mine.TotalLaneCount, theirs.TotalLaneCount);
    return mismatches;
}

// D3D-49: the HRESULTs of invalid (and a few valid edge-case) calls, compared with the host PC runtime. Games and
// middleware branch on these, so a wrong success hides a bug and a wrong failure takes a path the console never
// takes. Each case is a stateless call on one device that releases whatever it created.
// Releases what a creation call wrote to `object`. Takes the address, not the pointer: the call must run first.
HRESULT released(HRESULT hr, void* const* object) {
    if (*object) static_cast<IUnknown*>(*object)->Release();
    return hr;
}

HRESULT committed(ID3D12Device* device, D3D12_HEAP_PROPERTIES heap, D3D12_RESOURCE_DESC desc,
                  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON,
                  D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE, const D3D12_CLEAR_VALUE* clear = nullptr) {
    void* out = nullptr;
    const HRESULT hr =
        device->CreateCommittedResource(&heap, flags, &desc, state, clear, __uuidof(ID3D12Resource), &out);
    return released(hr, &out);
}

D3D12_RESOURCE_DESC rgba8(D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    return texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, flags);
}

HRESULT with_heap_flags(ID3D12Device* device, const D3D12_RESOURCE_DESC& desc, UINT heap_flags) {
    return committed(device, {D3D12_HEAP_TYPE_DEFAULT}, desc, D3D12_RESOURCE_STATE_COMMON,
                     D3D12_HEAP_FLAGS(heap_flags));
}

HRESULT descriptor_heap(ID3D12Device* device, D3D12_DESCRIPTOR_HEAP_DESC desc) {
    void* out = nullptr;
    return released(device->CreateDescriptorHeap(&desc, __uuidof(ID3D12DescriptorHeap), &out), &out);
}

HRESULT command_queue(ID3D12Device* device, D3D12_COMMAND_QUEUE_DESC desc) {
    void* out = nullptr;
    return released(device->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), &out), &out);
}

HRESULT query_heap(ID3D12Device* device, D3D12_QUERY_HEAP_TYPE type, UINT count, UINT node_mask = 0,
                   bool output = true) {
    const D3D12_QUERY_HEAP_DESC desc{type, count, node_mask};
    void* out = nullptr;
    return released(device->CreateQueryHeap(&desc, __uuidof(ID3D12QueryHeap), output ? &out : nullptr), &out);
}

// A root signature blob from the host's own serializer (version 1.0): empty, or one root CBV at b0, or one table of
// unbounded SRVs from t0. Empty when d3d12.dll has no serializer, and both devices then get the same refusal.
std::vector<std::uint8_t> root_signature_blob(int parameters) {
    using Serialize = HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*, D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**,
                                       ID3DBlob**);
    static const auto serialize = reinterpret_cast<Serialize>(reinterpret_cast<void*>(GetProcAddress(
        LoadLibraryExA("d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32), "D3D12SerializeRootSignature")));
    D3D12_ROOT_PARAMETER parameter{};
    parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
    const D3D12_DESCRIPTOR_RANGE range{D3D12_DESCRIPTOR_RANGE_TYPE_SRV, UINT_MAX, 0, 0,
                                       D3D12_DESCRIPTOR_RANGE_OFFSET_APPEND};
    if (parameters == 1) {
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_CBV;
    } else if (parameters == 2) {
        parameter.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
        parameter.DescriptorTable = {1, &range};
    }
    const D3D12_ROOT_SIGNATURE_DESC desc{parameters ? 1u : 0u, parameters ? &parameter : nullptr, 0, nullptr,
                                         D3D12_ROOT_SIGNATURE_FLAG_NONE};
    ID3DBlob* blob = nullptr;
    ID3DBlob* error = nullptr;
    std::vector<std::uint8_t> bytes;
    if (serialize && SUCCEEDED(serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, &error)) && blob) {
        const auto* first = static_cast<const std::uint8_t*>(blob->GetBufferPointer());
        bytes.assign(first, first + blob->GetBufferSize());
    }
    if (blob) blob->Release();
    if (error) error->Release();
    return bytes;
}

// CreateRootSignature with `blob`, passed as `size` bytes (the blob's own size when 0; the vector is padded with
// zeros when `size` is larger).
HRESULT root_signature(ID3D12Device* device, std::vector<std::uint8_t> blob, SIZE_T size = 0, UINT node_mask = 0,
                       bool output = true, REFIID riid = __uuidof(ID3D12RootSignature)) {
    if (!size) size = blob.size();
    if (blob.size() < size) blob.resize(size);
    void* out = nullptr;
    return released(device->CreateRootSignature(node_mask, blob.empty() ? nullptr : blob.data(), size, riid,
                                                output ? &out : nullptr),
                    &out);
}

// CreateCommandSignature without a root signature. Each argument is only its type; a CONSTANT one sets one value.
HRESULT command_signature(ID3D12Device* device, UINT stride, std::initializer_list<D3D12_INDIRECT_ARGUMENT_TYPE> types,
                          UINT node_mask = 0) {
    std::vector<D3D12_INDIRECT_ARGUMENT_DESC> args;
    for (const D3D12_INDIRECT_ARGUMENT_TYPE type : types) {
        D3D12_INDIRECT_ARGUMENT_DESC arg{};
        arg.Type = type;
        if (type == D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT) arg.Constant.Num32BitValuesToSet = 1;
        args.push_back(arg);
    }
    const D3D12_COMMAND_SIGNATURE_DESC desc{stride, UINT(args.size()), args.data(), node_mask};
    void* out = nullptr;
    return released(device->CreateCommandSignature(&desc, nullptr, __uuidof(ID3D12CommandSignature), &out), &out);
}

// CheckFeatureSupport(FORMAT_SUPPORT) for `value`: the HRESULT, or Support1 when it succeeds (Support2 holds what the
// GPU can do, such as TILED, which the layer doesn't offer, so it isn't compared).
HRESULT format_support1(ID3D12Device* device, UINT value) {
    D3D12_FEATURE_DATA_FORMAT_SUPPORT support{static_cast<DXGI_FORMAT>(value)};
    const HRESULT hr = device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support));
    return SUCCEEDED(hr) ? static_cast<HRESULT>(support.Support1) : hr;
}

HRESULT fence(ID3D12Device* device, D3D12_FENCE_FLAGS flags, REFIID riid = __uuidof(ID3D12Fence)) {
    void* out = nullptr;
    return released(device->CreateFence(0, flags, riid, &out), &out);
}

HRESULT command_allocator(ID3D12Device* device, D3D12_COMMAND_LIST_TYPE type, bool output = true,
                          REFIID riid = __uuidof(ID3D12CommandAllocator)) {
    void* out = nullptr;
    return released(device->CreateCommandAllocator(type, riid, output ? &out : nullptr), &out);
}

// CreateCommandList with a fresh allocator of `allocator_type` (or none when it is -2).
HRESULT command_list(ID3D12Device* device, UINT node_mask, D3D12_COMMAND_LIST_TYPE type,
                     int allocator_type, bool output = true) {
    void* allocator = nullptr;
    if (allocator_type != -2)
        device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE(allocator_type), __uuidof(ID3D12CommandAllocator),
                                       &allocator);
    void* out = nullptr;
    const HRESULT hr = device->CreateCommandList(node_mask, type, static_cast<ID3D12CommandAllocator*>(allocator),
                                                 nullptr, __uuidof(ID3D12GraphicsCommandList), output ? &out : nullptr);
    released(S_OK, &allocator);
    return released(hr, &out);
}

HRESULT command_list1(ID3D12Device* device, UINT node_mask, D3D12_COMMAND_LIST_TYPE type, bool output = true,
                      D3D12_COMMAND_LIST_FLAGS flags = D3D12_COMMAND_LIST_FLAG_NONE) {
    Ref<ID3D12Device4> device4;
    if (FAILED(device->QueryInterface(IID_PPV_ARGS(&device4.p)))) return E_NOINTERFACE;
    void* out = nullptr;
    return released(device4->CreateCommandList1(node_mask, type, flags,
                                                __uuidof(ID3D12GraphicsCommandList), output ? &out : nullptr),
                    &out);
}

// A 64 KiB heap description for the CreateHeap cases (D3D-238).
D3D12_HEAP_DESC heap_desc_of(UINT64 alignment, D3D12_HEAP_PROPERTIES properties = {D3D12_HEAP_TYPE_DEFAULT},
                             D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE, UINT64 size = 64 * 1024) {
    D3D12_HEAP_DESC desc{};
    desc.SizeInBytes = size;
    desc.Properties = properties;
    desc.Alignment = alignment;
    desc.Flags = flags;
    return desc;
}

HRESULT heap_create(ID3D12Device* device, D3D12_HEAP_DESC desc) {
    void* out = nullptr;
    return released(device->CreateHeap(&desc, __uuidof(ID3D12Heap), &out), &out);
}

// CreatePlacedResource in a fresh heap of `heap` made on the same device. A heap the device refuses makes the case
// itself wrong, so it answers E_UNEXPECTED on either side.
struct PlacedArgs {
    D3D12_HEAP_DESC heap = heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_NONE, 1024 * 1024);
    UINT64 offset = 0;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    const D3D12_CLEAR_VALUE* clear = nullptr;
    bool output = true;
    const IID* riid = &__uuidof(ID3D12Resource);
};

HRESULT placed(ID3D12Device* device, const D3D12_RESOURCE_DESC* desc, const PlacedArgs& args = {}) {
    ID3D12Heap* heap = nullptr;
    if (FAILED(device->CreateHeap(&args.heap, IID_PPV_ARGS(&heap)))) return E_UNEXPECTED;
    void* out = nullptr;
    const HRESULT hr = device->CreatePlacedResource(heap, args.offset, desc, args.state, args.clear, *args.riid,
                                                    args.output ? &out : nullptr);
    released(hr, &out);
    heap->Release();
    return hr;
}

HRESULT placed(ID3D12Device* device, D3D12_RESOURCE_DESC desc, const PlacedArgs& args = {}) {
    return placed(device, &desc, args);
}

HRESULT reserved(ID3D12Device* device, const D3D12_RESOURCE_DESC* desc,
                 D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON, const D3D12_CLEAR_VALUE* clear = nullptr,
                 bool output = true, REFIID riid = __uuidof(ID3D12Resource)) {
    void* out = nullptr;
    return released(device->CreateReservedResource(desc, state, clear, riid, output ? &out : nullptr), &out);
}

HRESULT reserved(ID3D12Device* device, D3D12_RESOURCE_DESC desc,
                 D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON, const D3D12_CLEAR_VALUE* clear = nullptr) {
    return reserved(device, &desc, state, clear);
}

D3D12_RESOURCE_DESC tiled(D3D12_RESOURCE_DESC desc) {
    desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
    return desc;
}

// Map on a fresh committed resource; `range` is the read range, or null.
HRESULT map(ID3D12Device* device, D3D12_HEAP_PROPERTIES heap, D3D12_RESOURCE_DESC desc, D3D12_RESOURCE_STATES state,
            UINT subresource, const D3D12_RANGE* range, bool want_pointer = true) {
    ID3D12Resource* resource = nullptr;
    HRESULT hr = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                 IID_PPV_ARGS(&resource));
    if (FAILED(hr)) return E_UNEXPECTED;  // the case itself is wrong: the resource must be creatable on both
    void* data = nullptr;
    hr = resource->Map(subresource, range, want_pointer ? &data : nullptr);
    if (SUCCEEDED(hr)) resource->Unmap(subresource, nullptr);
    return released(hr, reinterpret_cast<void* const*>(&resource));
}

using InvalidCase = HRESULT (*)(ID3D12Device*);

// A call that faults is reported as this, not a crash of the whole test (the host may not validate everything).
constexpr HRESULT kFaulted = HRESULT(0xC0000005);

HRESULT call_guarded(InvalidCase call, ID3D12Device* device) {
    __try {
        return call(device);
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return kFaulted;
    }
}

// The host accepts a UAV buffer in RAYTRACING_ACCELERATION_STRUCTURE state only where OPTIONS5 reports ray tracing
// (an RTX does, an integrated Radeon without DXR refuses it: @aguedou-34 on #2462). The layer accepts it where it
// has acceleration structures: the standalone device (XDXR or KHR), not the device main builds without them.
bool host_ray_tracing(ID3D12Device* host) {
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 o5{};
    return SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5, &o5, sizeof(o5))) &&
           o5.RaytracingTier != D3D12_RAYTRACING_TIER_NOT_SUPPORTED;
}

// Windows 10's runtime (no ID3D12Device10) still checks a buffer's initial state and refuses a HARDWARE_PROTECTED
// heap, where newer runtimes and the layer accept them (measured on Windows 10 22H2 build 19045.5737). Windows 11
// 23H2's runtime (D3D12Core 10.0.22621.5415: ID3D12Device10 but no ID3D12Device11) still refuses the
// HARDWARE_PROTECTED heap. For a case marked windows10_refuses only that pair is excused: the layer's S_OK against
// the host's E_INVALIDARG, on a host without ID3D12Device11. Any other difference, and any difference on a newer
// host, is still a mismatch.
bool windows10_refusal(const char* name, ID3D12Device* host, HRESULT ours, HRESULT theirs) {
    if (ours != S_OK || theirs != E_INVALIDARG) return false;
    Ref<ID3D12Device11> newer;
    if (SUCCEEDED(host->QueryInterface(IID_PPV_ARGS(&newer.p)))) return false;
    std::printf("invalid arguments %s: an older host runtime (Windows 10, Windows 11 23H2) refuses it (ours 0x%08lx, "
                "host 0x%08lx)\n", name, ours, theirs);
    return true;
}

int compare_invalid_arguments(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    using H = D3D12_HEAP_PROPERTIES;
    struct Case {
        const char* name;
        InvalidCase call;
        bool (*comparable)(ID3D12Device*) = nullptr;
        bool windows10_refuses = false;
    };
    const auto device10 = [](ID3D12Device* d) {
        Ref<ID3D12Device10> newer;
        return SUCCEEDED(d->QueryInterface(IID_PPV_ARGS(&newer.p)));
    };
    // Windows 11 23H2's in-box runtime (D3D12Core 10.0.22621.5415) has ID3D12Device10 but not ID3D12Device11, and
    // still answers five of the device10 cases as Windows 10 does (HARDWARE_PROTECTED and flag 0x400 refused, 2049
    // shader-visible samplers refused, a type -1 list created). Those five are compared only on a host with
    // ID3D12Device11, which newer runtimes have (d3d12_sampler2_test compares against one).
    const auto device11 = [](ID3D12Device* d) {
        Ref<ID3D12Device11> newer;
        return SUCCEEDED(d->QueryInterface(IID_PPV_ARGS(&newer.p)));
    };
    static const Case cases[] = {
        // CreateCommittedResource: arguments.
        // A null D3D12_HEAP_PROPERTIES faults in the host runtime, so it is not compared.
        {"committed: null description", [](ID3D12Device* d) {
             const H heap{D3D12_HEAP_TYPE_DEFAULT};
             void* out = nullptr;
             return released(d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, nullptr,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        __uuidof(ID3D12Resource), &out), &out);
         }},
        {"committed: null output (capability check)", [](ID3D12Device* d) {
             const H heap{D3D12_HEAP_TYPE_DEFAULT};
             const D3D12_RESOURCE_DESC desc = buffer_desc(256);
             return d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, __uuidof(ID3D12Resource), nullptr);
         }},
        {"committed: wrong interface", [](ID3D12Device* d) {
             const H heap{D3D12_HEAP_TYPE_DEFAULT};
             const D3D12_RESOURCE_DESC desc = buffer_desc(256);
             void* out = nullptr;
             return released(d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                        D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                        __uuidof(ID3D12Fence), &out), &out);
         }},
        // CreateCommittedResource: the initial state. A buffer takes any bits but the acceleration-structure state
        // without UAV access; a texture takes GENERIC_READ, one write state alone (RENDER_TARGET, UNORDERED_ACCESS,
        // DEPTH_WRITE with their flag), or a mix of read states (DEPTH_READ with ALLOW_DEPTH_STENCIL, the shader
        // resource states without DENY_SHADER_RESOURCE), and never a buffer-only or undefined bit.
        {"committed: plain texture in RENDER_TARGET state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATE_RENDER_TARGET);
         }},
        {"committed: render target in RENDER_TARGET state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_RENDER_TARGET);
         }},
        {"committed: plain texture in UNORDERED_ACCESS state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
         }},
        {"committed: UAV texture in UNORDERED_ACCESS state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                              D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
         }},
        {"committed: plain texture in DEPTH_WRITE state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATE_DEPTH_WRITE);
         }},
        {"committed: plain texture in DEPTH_READ state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATE_DEPTH_READ);
         }},
        {"committed: depth texture in DEPTH_READ | PIXEL_SHADER_RESOURCE state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                              D3D12_RESOURCE_STATE_DEPTH_READ | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
         }},
        {"committed: depth texture in DEPTH_WRITE | DEPTH_READ state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                              D3D12_RESOURCE_STATE_DEPTH_WRITE | D3D12_RESOURCE_STATE_DEPTH_READ);
         }},
        {"committed: depth texture without shader resource in PIXEL_SHADER_RESOURCE state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT,
                                           D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                               D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE),
                              D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
         }},
        {"committed: depth texture without shader resource in GENERIC_READ state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT,
                                           D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                               D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE),
                              D3D12_RESOURCE_STATE_GENERIC_READ);
         }},
        {"committed: texture in GENERIC_READ state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATE_GENERIC_READ);
         }},
        {"committed: texture in GENERIC_READ | COPY_DEST state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(),
                              D3D12_RESOURCE_STATE_GENERIC_READ | D3D12_RESOURCE_STATE_COPY_DEST);
         }},
        {"committed: texture in COPY_DEST | PIXEL_SHADER_RESOURCE state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(),
                              D3D12_RESOURCE_STATE_COPY_DEST | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
         }},
        {"committed: texture in the three shader-read and copy-source states", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(),
                              D3D12_RESOURCE_STATE_ALL_SHADER_RESOURCE | D3D12_RESOURCE_STATE_COPY_SOURCE);
         }},
        {"committed: render target in RENDER_TARGET | PIXEL_SHADER_RESOURCE state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
         }},
        {"committed: texture in VERTEX_AND_CONSTANT_BUFFER state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER);
         }},
        {"committed: texture in INDEX_BUFFER | PIXEL_SHADER_RESOURCE state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(),
                              D3D12_RESOURCE_STATE_INDEX_BUFFER | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
         }},
        {"committed: texture in state 0x80000000", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATES(0x80000000u));
         }},
        {"committed: texture in RESOLVE_DEST state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(), D3D12_RESOURCE_STATE_RESOLVE_DEST);
         }},
        // A buffer's initial state: Windows 10's runtime refuses these two, newer ones accept them, as the layer does.
        {"committed: buffer in state 0x80000000", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256), D3D12_RESOURCE_STATES(0x80000000u));
         }, nullptr, true},
        {"committed: buffer in RENDER_TARGET | UNORDERED_ACCESS state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256),
                              D3D12_RESOURCE_STATE_RENDER_TARGET | D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
         }, nullptr, true},
        {"committed: buffer in RAYTRACING_ACCELERATION_STRUCTURE state", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256),
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
         }},
        // This device has no acceleration structures and refuses it, as a host without ray tracing does.
        {"committed: UAV buffer in RAYTRACING_ACCELERATION_STRUCTURE state", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
         }, [](ID3D12Device* host) { return !host_ray_tracing(host); }},
        // CreateCommittedResource: buffer descriptions.
        {"committed: buffer width 0", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(0));
         }},
        {"committed: buffer height 2", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Height = 2;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer with 2 mips", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.MipLevels = 2;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer with a format", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Format = DXGI_FORMAT_R8_UNORM;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer not row-major", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer 2 samples", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.SampleDesc.Count = 2;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer alignment 12345", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Alignment = 12345;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer as render target", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer as depth stencil", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer with a clear value", [](ID3D12Device* d) {
             D3D12_CLEAR_VALUE clear{};
             clear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON,
                              D3D12_HEAP_FLAG_NONE, &clear);
         }},
        {"committed: upload buffer in COMMON", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_UPLOAD}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON);
         }, +device10},
        {"committed: upload buffer in COPY_DEST", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_UPLOAD}, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST);
         }},
        {"committed: readback buffer in GENERIC_READ", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_READBACK}, buffer_desc(256), D3D12_RESOURCE_STATE_GENERIC_READ);
         }},
        {"committed: upload buffer with UAV", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return committed(d, {D3D12_HEAP_TYPE_UPLOAD}, desc, D3D12_RESOURCE_STATE_GENERIC_READ);
         }},
        // CreateCommittedResource: heap properties and flags.
        {"committed: heap type 99", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE(99)}, buffer_desc(256));
         }},
        {"committed: default heap with a CPU page property", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK}, buffer_desc(256));
         }},
        {"committed: default heap with a memory pool", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_L0},
                              buffer_desc(256));
         }},
        {"committed: custom heap, unknown page property", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_L0},
                              buffer_desc(256));
         }},
        {"committed: custom heap, unknown pool", [](ID3D12Device* d) {
             return committed(d,
                              {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_UNKNOWN},
                              buffer_desc(256));
         }},
        {"committed: node mask 2", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                  D3D12_MEMORY_POOL_UNKNOWN, 2, 2}, buffer_desc(256));
         }},
        {"committed: heap flag DENY_BUFFERS on a buffer", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON,
                              D3D12_HEAP_FLAG_DENY_BUFFERS);
         }},
        {"committed: heap flag 0x80000000", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON,
                              D3D12_HEAP_FLAGS(0x80000000));
         }},
        // CreateCommittedResource: texture descriptions.
        {"committed: dimension UNKNOWN", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Dimension = D3D12_RESOURCE_DIMENSION_UNKNOWN;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: dimension 9", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Dimension = D3D12_RESOURCE_DIMENSION(9);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: texture width 0", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(0, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"committed: texture height 0", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 0, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"committed: texture array size 0", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 0));
         }},
        {"committed: texture array size 2049", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 2049));
         }},
        {"committed: volume depth 2049", [](ID3D12Device* d) {
             auto desc = volume_desc(1);
             desc.DepthOrArraySize = 2049;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: texture format UNKNOWN", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"committed: texture format 250", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT(250), D3D12_RESOURCE_FLAG_NONE));
         }},
        {"committed: texture 0 samples", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc.Count = 0;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: texture 3 samples", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc.Count = 3;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA with 2 mips", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 2);
             desc.SampleDesc.Count = 4;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA quality 999", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc.Count = 4;
             desc.SampleDesc.Quality = 999;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 1x quality 1", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.SampleDesc.Quality = 1;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA without RT or DS", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.SampleDesc.Count = 4;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: colour format as depth stencil", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              rgba8(D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
         }},
        {"committed: BC1 as render target", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
         }},
        {"committed: BC1 6x6 top mip", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(6, 6, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE));
         }, host_supports_unaligned_block_textures},
        {"committed: render target and depth stencil", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                               D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL));
         }},
        {"committed: depth stencil with UAV", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT,
                                           D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
         }},
        {"committed: DENY_SHADER_RESOURCE without depth stencil", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              rgba8(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE));
         }},
        {"committed: resource flag 0x8000", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAGS(0x8000)));
         }},
        {"committed: texture on an upload heap", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_UPLOAD},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                              D3D12_RESOURCE_STATE_GENERIC_READ);
         }},
        {"committed: texture on a readback heap", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_READBACK},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                              D3D12_RESOURCE_STATE_COPY_DEST);
         }},
        {"committed: row-major 2D texture", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: texture alignment 12345", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Alignment = 12345;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: render target clear value format UNKNOWN", [](ID3D12Device* d) {
             D3D12_CLEAR_VALUE clear{};
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_NONE, &clear);
         }},
        {"committed: clear value on a plain texture", [](ID3D12Device* d) {
             D3D12_CLEAR_VALUE clear{};
             clear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_NONE, &clear);
         }},
        // CreateCommittedResource: valid neighbours of the cases above, so a check does not reject too much.
        {"committed: buffer aligned 64 KiB", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer aligned 4 KiB", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: buffer with UAV", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: readback buffer in COMMON", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_READBACK}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON);
         }, +device10},
        {"committed: readback buffer with UAV", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return committed(d, {D3D12_HEAP_TYPE_READBACK}, desc, D3D12_RESOURCE_STATE_COPY_DEST);
         }},
        {"committed: node masks 1", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                  D3D12_MEMORY_POOL_UNKNOWN, 1, 1}, buffer_desc(256));
         }},
        {"committed: custom write-combine L0 buffer", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0},
                              buffer_desc(256));
         }},
        {"committed: heap flag CREATE_NOT_ZEROED", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON,
                              D3D12_HEAP_FLAG_CREATE_NOT_ZEROED);
         }},
        {"committed: heap flag DENY_BUFFERS on a texture", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_DENY_BUFFERS);
         }},
        {"committed: heap flag DENY_RT_DS_TEXTURES on a render target", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES);
         }},
        {"committed: heap flag DENY_NON_RT_DS_TEXTURES on a plain texture", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES);
         }},
        {"committed: 16x16 with 5 mips", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 5));
         }},
        {"committed: texture width 16384", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16384, 4, DXGI_FORMAT_R8_UNORM, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"committed: texture array size 2048", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 2048));
         }},
        {"committed: volume 2049 wide", [](ID3D12Device* d) {
             auto desc = volume_desc(1);
             desc.Width = 2049;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: small texture aligned 4 KiB", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: large texture aligned 4 KiB", [](ID3D12Device* d) {
             auto desc = texture_desc(1024, 1024, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: render target aligned 4 KiB", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: texture aligned 4 MiB", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Alignment = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA aligned 4 MiB", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc.Count = 4;
             desc.Alignment = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA aligned 64 KiB", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc.Count = 4;
             desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA depth", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
             desc.SampleDesc.Count = 4;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc, D3D12_RESOURCE_STATE_DEPTH_WRITE);
         }},
        {"committed: render target with UAV", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
         }},
        {"committed: depth stencil, DENY_SHADER_RESOURCE", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT,
                                           D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                               D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE),
                              D3D12_RESOURCE_STATE_DEPTH_WRITE);
         }},
        {"committed: typeless depth as depth stencil", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R32_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                              D3D12_RESOURCE_STATE_DEPTH_WRITE);
         }},
        {"committed: BC1 render target, typeless", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_BC1_TYPELESS, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
         }},
        {"committed: render target clear value", [](ID3D12Device* d) {
             D3D12_CLEAR_VALUE clear{};
             clear.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_NONE, &clear);
         }},
        {"committed: render target clear value of another format", [](ID3D12Device* d) {
             D3D12_CLEAR_VALUE clear{};
             clear.Format = DXGI_FORMAT_R16G16B16A16_FLOAT;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_NONE, &clear);
         }},
        {"committed: typeless render target clear value", [](ID3D12Device* d) {
             D3D12_CLEAR_VALUE clear{};
             clear.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_TYPELESS,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                              D3D12_RESOURCE_STATE_COMMON, D3D12_HEAP_FLAG_NONE, &clear);
         }},
        {"committed: depth clear value", [](ID3D12Device* d) {
             D3D12_CLEAR_VALUE clear{};
             clear.Format = DXGI_FORMAT_D32_FLOAT;
             clear.DepthStencil.Depth = 1.0f;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT},
                              texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                              D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_HEAP_FLAG_NONE, &clear);
         }},
        // Heap and resource flags one by one (measured the same on an RX 9070 XT, an integrated Radeon and WARP).
        {"committed: heap flag SHARED on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), D3D12_HEAP_FLAG_SHARED);
         }},
        {"committed: heap flag 0x2 on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), 0x2);
         }},
        {"committed: heap flag ALLOW_DISPLAY on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), D3D12_HEAP_FLAG_ALLOW_DISPLAY);
         }},
        {"committed: heap flag ALLOW_DISPLAY on a texture", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(), D3D12_HEAP_FLAG_ALLOW_DISPLAY);
         }},
        {"committed: heap flag 0x10 on a texture", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(), 0x10);
         }},
        {"committed: heap flag SHARED_CROSS_ADAPTER on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER);
         }},
        {"committed: heap flag HARDWARE_PROTECTED on a texture", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(), D3D12_HEAP_FLAG_HARDWARE_PROTECTED);
         }, +device11},
        {"committed: heap flag ALLOW_WRITE_WATCH on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH);
         }},
        {"committed: heap flag ALLOW_SHADER_ATOMICS on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS);
         }},
        {"committed: heap flag ALLOW_SHADER_ATOMICS on a UAV buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return with_heap_flags(d, desc, D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS);
         }},
        {"committed: heap flag ALLOW_SHADER_ATOMICS on a UAV texture", [](ID3D12Device* d) {
             return with_heap_flags(d, texture_desc(16, 16, DXGI_FORMAT_R32_FLOAT,
                                                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                                    D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS);
         }},
        {"committed: heap flag ALLOW_SHADER_ATOMICS on a render-target UAV", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET |
                                             D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS),
                                    D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS);
         }},
        {"committed: heap flag ALLOW_SHADER_ATOMICS on a texture without UAV", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(), D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS);
         }},
        {"committed: heap flag ALLOW_SHADER_ATOMICS on a render target without UAV", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                    D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS);
         }},
        {"committed: heap flag ALLOW_SHADER_ATOMICS on an upload buffer", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_UPLOAD}, buffer_desc(256), D3D12_RESOURCE_STATE_GENERIC_READ,
                              D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS);
         }},
        {"committed: heap flag CREATE_NOT_RESIDENT on a texture", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(), D3D12_HEAP_FLAG_CREATE_NOT_RESIDENT);
         }},
        {"committed: heap flag TOOLS_USE_MANUAL_WRITE_TRACKING on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), D3D12_HEAP_FLAG_TOOLS_USE_MANUAL_WRITE_TRACKING);
         }},
        {"committed: heap flag DENY_RT_DS_TEXTURES on a texture", [](ID3D12Device* d) {
             return with_heap_flags(d, rgba8(), D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES);
         }},
        {"committed: heap flag DENY_NON_RT_DS_TEXTURES on a buffer", [](ID3D12Device* d) {
             return with_heap_flags(d, buffer_desc(256), D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES);
         }},
        {"committed: resource flag DENY_SHADER_RESOURCE on a buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag ALLOW_CROSS_ADAPTER on a buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag ALLOW_CROSS_ADAPTER on a texture", [](ID3D12Device* d) {
             auto desc = rgba8(D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER));
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag ALLOW_SIMULTANEOUS_ACCESS on a buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag ALLOW_SIMULTANEOUS_ACCESS on a texture", [](ID3D12Device* d) {
             auto desc = rgba8(D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS));
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag VIDEO_DECODE_REFERENCE_ONLY on a texture", [](ID3D12Device* d) {
             auto desc = rgba8(D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY));
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag VIDEO_ENCODE_REFERENCE_ONLY on a buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag RAYTRACING_ACCELERATION_STRUCTURE on a buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAGS(D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }, +device10},
        {"committed: resource flag 0x200 on a buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAGS(0x200);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag 0x800 on a texture", [](ID3D12Device* d) {
             auto desc = rgba8(D3D12_RESOURCE_FLAGS(0x800));
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA quality 1", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc = {4, 1};
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA quality standard pattern", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc = {4, DXGI_STANDARD_MULTISAMPLE_QUALITY_PATTERN};  // older SDKs lack the D3D12_ name
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: 4x MSAA quality center pattern", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc = {4, DXGI_CENTER_MULTISAMPLE_QUALITY_PATTERN};
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }},
        {"committed: resource flag RAYTRACING_ACCELERATION_STRUCTURE on a texture", [](ID3D12Device* d) {
             auto desc = rgba8(D3D12_RESOURCE_FLAG_RAYTRACING_ACCELERATION_STRUCTURE);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }, +device10},
        {"committed: resource flag 0x400 on a buffer", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAGS(0x400);
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc);
         }, +device11},
        {"committed: resource flag 0x400 on a texture", [](ID3D12Device* d) {
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, rgba8(D3D12_RESOURCE_FLAGS(0x400)));
         }, +device11},
        // A null output is a capability check; with invalid arguments too, the host's answer says which comes first.
        {"committed: heap type 99, null output", [](ID3D12Device* d) {
             const H heap{D3D12_HEAP_TYPE(99)};
             const D3D12_RESOURCE_DESC desc = buffer_desc(256);
             return d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, __uuidof(ID3D12Resource), nullptr);
         }},
        {"committed: buffer height 2, null output", [](ID3D12Device* d) {
             const H heap{D3D12_HEAP_TYPE_DEFAULT};
             D3D12_RESOURCE_DESC desc = buffer_desc(256);
             desc.Height = 2;
             return d->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                               nullptr, __uuidof(ID3D12Resource), nullptr);
         }},
        // CreateHeap (D3D-238): the four descriptions the host refused on an RX 5700 and WARP, and the heap-property
        // rules CreateCommittedResource shares. A refusal needs no arena; the valid neighbours do, so
        // compare_heap_neighbours compares them on the standalone device.
        {"heap: alignment 4096", [](ID3D12Device* d) { return heap_create(d, heap_desc_of(4096)); }},
        {"heap: custom heap, unknown page property", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                                    D3D12_MEMORY_POOL_L0}));
         }},
        {"heap: all three DENY flags", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT},
                                                D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES |
                                                    D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES));
         }},
        {"heap: type 0", [](ID3D12Device* d) { return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE(0)})); }},
        // ALLOW_DISPLAY is for committed resources on a default heap only (D3D12_HEAP_FLAGS documentation); the host
        // refused it on an RTX 4070 SUPER and a UHD 770 (@CowBoYdu76).
        {"heap: ALLOW_DISPLAY", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_ALLOW_DISPLAY));
         }},
        {"heap: custom heap, unknown pool", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                                                    D3D12_MEMORY_POOL_UNKNOWN}));
         }},
        {"heap: default heap with a CPU page property", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK}));
         }},
        {"heap: default heap with a memory pool", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
                                                    D3D12_MEMORY_POOL_L0}));
         }},
        // CreateDescriptorHeap. A null description faults in the host runtime, so it is not compared.
        {"descriptor heap: null output", [](ID3D12Device* d) {
             const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4};
             return d->CreateDescriptorHeap(&desc, __uuidof(ID3D12DescriptorHeap), nullptr);
         }},
        // The host answers a null output before the flag, type and node rules (measured by @quackottu and here).
        {"descriptor heap: shader-visible RTV, null output", [](ID3D12Device* d) {
             const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE};
             return d->CreateDescriptorHeap(&desc, __uuidof(ID3D12DescriptorHeap), nullptr);
         }},
        {"descriptor heap: flag 0x80, null output", [](ID3D12Device* d) {
             const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                   D3D12_DESCRIPTOR_HEAP_FLAGS(0x80)};
             return d->CreateDescriptorHeap(&desc, __uuidof(ID3D12DescriptorHeap), nullptr);
         }},
        {"descriptor heap: node mask 2, null output", [](ID3D12Device* d) {
             const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                                   D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 2};
             return d->CreateDescriptorHeap(&desc, __uuidof(ID3D12DescriptorHeap), nullptr);
         }},
        {"descriptor heap: wrong interface", [](ID3D12Device* d) {
             const D3D12_DESCRIPTOR_HEAP_DESC desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4};
             void* out = nullptr;
             return released(d->CreateDescriptorHeap(&desc, __uuidof(ID3D12Fence), &out), &out);
         }},
        {"descriptor heap: 0 descriptors", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 0});
         }},
        {"descriptor heap: shader-visible RTV", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE});
         }},
        {"descriptor heap: shader-visible DSV", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE});
         }},
        {"descriptor heap: shader-visible 2049 samplers", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2049,
                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE});
         }, +device11},
        {"descriptor heap: shader-visible 2048 samplers", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 2048,
                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE});
         }},
        {"descriptor heap: flag 0x80", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAGS(0x80)});
         }},
        {"descriptor heap: node mask 2", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 2});
         }},
        {"descriptor heap: node mask 1", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 1});
         }},
        {"descriptor heap: shader-visible CBV/SRV/UAV", [](ID3D12Device* d) {
             return descriptor_heap(d, {D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                        D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE});
         }},
        // CreateCommandQueue.
        {"command queue: null description", [](ID3D12Device* d) {
             void* out = nullptr;
             return released(d->CreateCommandQueue(nullptr, __uuidof(ID3D12CommandQueue), &out), &out);
         }},
        {"command queue: null output", [](ID3D12Device* d) {
             const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
             return d->CreateCommandQueue(&desc, __uuidof(ID3D12CommandQueue), nullptr);
         }},
        {"command queue: wrong interface", [](ID3D12Device* d) {
             const D3D12_COMMAND_QUEUE_DESC desc{D3D12_COMMAND_LIST_TYPE_DIRECT};
             void* out = nullptr;
             return released(d->CreateCommandQueue(&desc, __uuidof(ID3D12Fence), &out), &out);
         }},
        {"command queue: bundle", [](ID3D12Device* d) { return command_queue(d, {D3D12_COMMAND_LIST_TYPE_BUNDLE}); }},
        {"command queue: type 99", [](ID3D12Device* d) { return command_queue(d, {D3D12_COMMAND_LIST_TYPE(99)}); }},
        {"command queue: priority 5", [](ID3D12Device* d) {
             return command_queue(d, {D3D12_COMMAND_LIST_TYPE_DIRECT, 5});
         }},
        {"command queue: flag 0x80", [](ID3D12Device* d) {
             return command_queue(d, {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAGS(0x80)});
         }},
        {"command queue: node mask 2", [](ID3D12Device* d) {
             return command_queue(d, {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 2});
         }},
        {"command queue: global realtime priority", [](ID3D12Device* d) {
             return command_queue(d, {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME});
         }},
        {"command queue: compute", [](ID3D12Device* d) { return command_queue(d, {D3D12_COMMAND_LIST_TYPE_COMPUTE}); }},
        {"command queue: copy", [](ID3D12Device* d) { return command_queue(d, {D3D12_COMMAND_LIST_TYPE_COPY}); }},
        // CreateQueryHeap: the description is checked first (types past PIPELINE_STATISTICS1 are refused), then a
        // null output is E_UNEXPECTED. Known types this layer has no path for (stream output, video decode, mesh
        // statistics) are E_NOTIMPL here and not compared.
        {"query heap: occlusion", [](ID3D12Device* d) { return query_heap(d, D3D12_QUERY_HEAP_TYPE_OCCLUSION, 4); }},
        {"query heap: timestamps", [](ID3D12Device* d) { return query_heap(d, D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 4); }},
        {"query heap: no queries", [](ID3D12Device* d) { return query_heap(d, D3D12_QUERY_HEAP_TYPE_OCCLUSION, 0); }},
        {"query heap: type 8", [](ID3D12Device* d) { return query_heap(d, D3D12_QUERY_HEAP_TYPE(8), 1); }},
        {"query heap: type 99", [](ID3D12Device* d) { return query_heap(d, D3D12_QUERY_HEAP_TYPE(99), 1); }},
        {"query heap: type -1", [](ID3D12Device* d) { return query_heap(d, D3D12_QUERY_HEAP_TYPE(-1), 1); }},
        {"query heap: node mask 1", [](ID3D12Device* d) {
             return query_heap(d, D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 1);
         }},
        {"query heap: node mask 2", [](ID3D12Device* d) {
             return query_heap(d, D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 2);
         }},
        {"query heap: null output", [](ID3D12Device* d) {
             return query_heap(d, D3D12_QUERY_HEAP_TYPE_OCCLUSION, 4, 0, false);
         }},
        {"query heap: null output, no queries", [](ID3D12Device* d) {
             return query_heap(d, D3D12_QUERY_HEAP_TYPE_OCCLUSION, 0, 0, false);
         }},
        {"query heap: null output, type 99", [](ID3D12Device* d) {
             return query_heap(d, D3D12_QUERY_HEAP_TYPE(99), 1, 0, false);
         }},
        {"query heap: null output, node mask 2", [](ID3D12Device* d) {
             return query_heap(d, D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 2, false);
         }},
        {"query heap: wrong interface", [](ID3D12Device* d) {
             const D3D12_QUERY_HEAP_DESC desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 0};
             void* out = nullptr;
             return released(d->CreateQueryHeap(&desc, __uuidof(ID3D12Fence), &out), &out);
         }},
        // CreateRootSignature: a null output is S_FALSE whatever the blob; otherwise node 0 only, and the blob's
        // DXBC size must be the size passed.
        {"root signature: empty", [](ID3D12Device* d) { return root_signature(d, root_signature_blob(0)); }},
        {"root signature: root CBV", [](ID3D12Device* d) { return root_signature(d, root_signature_blob(1)); }},
        {"root signature: unbounded SRV table", [](ID3D12Device* d) {
             return root_signature(d, root_signature_blob(2));
         }},
        {"root signature: size 0", [](ID3D12Device* d) {
             void* out = nullptr;
             const auto blob = root_signature_blob(0);
             return released(d->CreateRootSignature(0, blob.data(), 0, __uuidof(ID3D12RootSignature), &out), &out);
         }},
        {"root signature: one byte short", [](ID3D12Device* d) {
             const auto blob = root_signature_blob(0);
             return root_signature(d, blob, blob.size() - 1);
         }},
        {"root signature: 4 bytes over", [](ID3D12Device* d) {
             const auto blob = root_signature_blob(1);
             return root_signature(d, blob, blob.size() + 4);
         }},
        {"root signature: 16 bytes over", [](ID3D12Device* d) {
             const auto blob = root_signature_blob(0);
             return root_signature(d, blob, blob.size() + 16);
         }},
        {"root signature: 64 zero bytes", [](ID3D12Device* d) {
             return root_signature(d, std::vector<std::uint8_t>(64, 0));
         }},
        {"root signature: null blob", [](ID3D12Device* d) {
             void* out = nullptr;
             return released(d->CreateRootSignature(0, nullptr, 64, __uuidof(ID3D12RootSignature), &out), &out);
         }},
        {"root signature: node mask 1", [](ID3D12Device* d) {
             return root_signature(d, root_signature_blob(0), 0, 1);
         }},
        {"root signature: node mask 2", [](ID3D12Device* d) {
             return root_signature(d, root_signature_blob(0), 0, 2);
         }},
        {"root signature: node mask 0x80000000", [](ID3D12Device* d) {
             return root_signature(d, root_signature_blob(0), 0, 0x80000000u);
         }},
        {"root signature: wrong interface", [](ID3D12Device* d) {
             return root_signature(d, root_signature_blob(0), 0, 0, true, __uuidof(ID3D12Fence));
         }},
        {"root signature: null output", [](ID3D12Device* d) {
             return root_signature(d, root_signature_blob(1), 0, 0, false);
         }},
        {"root signature: null output, 64 zero bytes", [](ID3D12Device* d) {
             return root_signature(d, std::vector<std::uint8_t>(64, 0), 0, 0, false);
         }},
        {"root signature: null output, node mask 2", [](ID3D12Device* d) {
             return root_signature(d, root_signature_blob(0), 0, 2, false);
         }},
        {"root signature: null output, 16 bytes over", [](ID3D12Device* d) {
             const auto blob = root_signature_blob(0);
             return root_signature(d, blob, blob.size() + 16, 0, false);
         }},
        {"command queue: high priority", [](ID3D12Device* d) {
             return command_queue(d, {D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_PRIORITY_HIGH});
         }},
        // Command allocators and lists (video list types are only created here, not compared: the host hands out
        // a video list, which is not an ID3D12GraphicsCommandList). Types past VIDEO_ENCODE and nodes other than 0
        // are refused first; a null output is then E_UNEXPECTED.
        {"command allocator: type direct", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(0));
         }},
        {"command allocator: type bundle", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(1));
         }},
        {"command allocator: type compute", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(2));
         }},
        {"command allocator: type copy", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(3));
         }},
        {"command allocator: type video encode", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(6));
         }},
        {"command allocator: type 7", [](ID3D12Device* d) { return command_allocator(d, D3D12_COMMAND_LIST_TYPE(7)); }},
        {"command allocator: type 99", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(99));
         }},
        {"command allocator: type -1", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(-1));
         }},
        {"command allocator: null output", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE_DIRECT, false);
         }},
        {"command allocator: null output, type 99", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE(99), false);
         }},
        {"command allocator: null output, wrong interface", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE_DIRECT, false, __uuidof(ID3D12Fence));
         }},
        {"command allocator: wrong interface", [](ID3D12Device* d) {
             return command_allocator(d, D3D12_COMMAND_LIST_TYPE_DIRECT, true, __uuidof(ID3D12Fence));
         }},
        {"command list: direct", [](ID3D12Device* d) { return command_list(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, 0); }},
        {"command list: bundle", [](ID3D12Device* d) { return command_list(d, 0, D3D12_COMMAND_LIST_TYPE_BUNDLE, 1); }},
        {"command list: node mask 1", [](ID3D12Device* d) {
             return command_list(d, 1, D3D12_COMMAND_LIST_TYPE_DIRECT, 0);
         }},
        {"command list: node mask 2", [](ID3D12Device* d) {
             return command_list(d, 2, D3D12_COMMAND_LIST_TYPE_DIRECT, 0);
         }},
        {"command list: node mask 3", [](ID3D12Device* d) {
             return command_list(d, 3, D3D12_COMMAND_LIST_TYPE_DIRECT, 0);
         }},
        {"command list: compute allocator", [](ID3D12Device* d) {
             return command_list(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, 2);
         }},
        {"command list: no allocator", [](ID3D12Device* d) {
             return command_list(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, -2);
         }},
        {"command list: type 99", [](ID3D12Device* d) { return command_list(d, 0, D3D12_COMMAND_LIST_TYPE(99), 0); }},
        {"command list: null output", [](ID3D12Device* d) {
             return command_list(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, 0, false);
         }},
        {"command list: null output, node mask 2", [](ID3D12Device* d) {
             return command_list(d, 2, D3D12_COMMAND_LIST_TYPE_DIRECT, 0, false);
         }},
        {"command list: null output, no allocator", [](ID3D12Device* d) {
             return command_list(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, -2, false);
         }},
        {"command list1: type direct", [](ID3D12Device* d) { return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(0)); }},
        {"command list1: type bundle", [](ID3D12Device* d) { return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(1)); }},
        {"command list1: type compute", [](ID3D12Device* d) {
             return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(2));
         }},
        {"command list1: type copy", [](ID3D12Device* d) { return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(3)); }},
        {"command list1: type 7", [](ID3D12Device* d) { return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(7)); }},
        {"command list1: type 99", [](ID3D12Device* d) { return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(99)); }},
        // NONE (-1): Windows 10's runtime (19045, no ID3D12Device10) creates the list and newer Windows 11 runtimes
        // refuse it (measured on #2297); Windows 11 23H2 (22621.5415, ID3D12Device10 but no ID3D12Device11) still
        // creates it, so it is compared only on a host with ID3D12Device11.
        {"command list1: type -1", [](ID3D12Device* d) {
             return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(-1));
         }, +device11},
        {"command list1: node mask 1", [](ID3D12Device* d) {
             return command_list1(d, 1, D3D12_COMMAND_LIST_TYPE_DIRECT);
         }},
        {"command list1: node mask 2", [](ID3D12Device* d) {
             return command_list1(d, 2, D3D12_COMMAND_LIST_TYPE_DIRECT);
         }},
        {"command list1: null output", [](ID3D12Device* d) {
             return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, false);
         }},
        {"command list1: null output, node mask 2", [](ID3D12Device* d) {
             return command_list1(d, 2, D3D12_COMMAND_LIST_TYPE_DIRECT, false);
         }},
        {"command list1: null output, type 99", [](ID3D12Device* d) {
             return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE(99), false);
         }},
        {"command list1: flags 1", [](ID3D12Device* d) {
             return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, true, D3D12_COMMAND_LIST_FLAGS(1));
         }},
        {"command list1: flags 0xffffffff", [](ID3D12Device* d) {
             return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, true, D3D12_COMMAND_LIST_FLAGS(0xffffffffu));
         }},
        {"command list1: null output, flags 1", [](ID3D12Device* d) {
             return command_list1(d, 0, D3D12_COMMAND_LIST_TYPE_DIRECT, false, D3D12_COMMAND_LIST_FLAGS(1));
         }},
        // CreateCommandSignature (no root signature). A null description faults in the host runtime, so it is not
        // compared.
        {"command signature: null output", [](ID3D12Device* d) {
             const D3D12_INDIRECT_ARGUMENT_DESC arg{D3D12_INDIRECT_ARGUMENT_TYPE_DRAW};
             const D3D12_COMMAND_SIGNATURE_DESC desc{16, 1, &arg, 0};
             return d->CreateCommandSignature(&desc, nullptr, __uuidof(ID3D12CommandSignature), nullptr);
         }},
        {"command signature: wrong interface", [](ID3D12Device* d) {
             const D3D12_INDIRECT_ARGUMENT_DESC arg{D3D12_INDIRECT_ARGUMENT_TYPE_DRAW};
             const D3D12_COMMAND_SIGNATURE_DESC desc{16, 1, &arg, 0};
             void* out = nullptr;
             return released(d->CreateCommandSignature(&desc, nullptr, __uuidof(ID3D12Fence), &out), &out);
         }},
        {"command signature: draw", [](ID3D12Device* d) {
             return command_signature(d, 16, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW});
         }},
        {"command signature: draw indexed", [](ID3D12Device* d) {
             return command_signature(d, 20, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED});
         }},
        {"command signature: dispatch", [](ID3D12Device* d) {
             return command_signature(d, 12, {D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH});
         }},
        {"command signature: dispatch, stride 64", [](ID3D12Device* d) {
             return command_signature(d, 64, {D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH});
         }},
        {"command signature: no arguments", [](ID3D12Device* d) { return command_signature(d, 16, {}); }},
        {"command signature: stride 0", [](ID3D12Device* d) {
             return command_signature(d, 0, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW});
         }},
        {"command signature: stride 18", [](ID3D12Device* d) {
             return command_signature(d, 18, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW});
         }},
        {"command signature: draw stride 12", [](ID3D12Device* d) {
             return command_signature(d, 12, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW});
         }},
        {"command signature: draw indexed stride 16", [](ID3D12Device* d) {
             return command_signature(d, 16, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW_INDEXED});
         }},
        {"command signature: two draws", [](ID3D12Device* d) {
             return command_signature(d, 32, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW});
         }},
        {"command signature: draw then dispatch", [](ID3D12Device* d) {
             return command_signature(d, 28,
                                      {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH});
         }},
        {"command signature: vertex buffer view only", [](ID3D12Device* d) {
             return command_signature(d, 16, {D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW});
         }},
        {"command signature: draw before a vertex buffer view", [](ID3D12Device* d) {
             return command_signature(d, 32, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW,
                                              D3D12_INDIRECT_ARGUMENT_TYPE_VERTEX_BUFFER_VIEW});
         }},
        {"command signature: constant without a root signature", [](ID3D12Device* d) {
             return command_signature(d, 20,
                                      {D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW});
         }},
        {"command signature: UAV without a root signature", [](ID3D12Device* d) {
             return command_signature(d, 20, {D3D12_INDIRECT_ARGUMENT_TYPE_UNORDERED_ACCESS_VIEW,
                                              D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH});
         }},
        {"command signature: type 99", [](ID3D12Device* d) {
             return command_signature(d, 16, {D3D12_INDIRECT_ARGUMENT_TYPE(99)});
         }},
        {"command signature: type 99 before a draw", [](ID3D12Device* d) {
             return command_signature(d, 32, {D3D12_INDIRECT_ARGUMENT_TYPE(99), D3D12_INDIRECT_ARGUMENT_TYPE_DRAW});
         }},
        {"command signature: node mask 1", [](ID3D12Device* d) {
             return command_signature(d, 16, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW}, 1);
         }},
        {"command signature: node mask 2", [](ID3D12Device* d) {
             return command_signature(d, 16, {D3D12_INDIRECT_ARGUMENT_TYPE_DRAW}, 2);
         }},
        {"command queue: disable GPU timeout", [](ID3D12Device* d) {
             return command_queue(d, {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT});
         }},
        {"command queue: node mask 1", [](ID3D12Device* d) {
             return command_queue(d, {D3D12_COMMAND_LIST_TYPE_DIRECT, 0, D3D12_COMMAND_QUEUE_FLAG_NONE, 1});
         }},
        // CreateFence.
        // FORMAT_SUPPORT for values the runtime's format table has no entry for (E_FAIL), and for UNKNOWN (a buffer
        // format: Support1 is compared, not Support2).
        {"format support: UNKNOWN", [](ID3D12Device* d) { return format_support1(d, DXGI_FORMAT_UNKNOWN); }},
        {"format support: 116, the first unused value", [](ID3D12Device* d) { return format_support1(d, 0x74u); }},
        {"format support: 129", [](ID3D12Device* d) { return format_support1(d, 0x81u); }},
        {"format support: 130, P208", [](ID3D12Device* d) { return format_support1(d, 0x82u); }},
        {"format support: 131, V208", [](ID3D12Device* d) { return format_support1(d, 0x83u); }},
        {"format support: 132, V408", [](ID3D12Device* d) { return format_support1(d, 0x84u); }},
        {"format support: 133", [](ID3D12Device* d) { return format_support1(d, 0x85u); }},
        {"format support: 188", [](ID3D12Device* d) { return format_support1(d, 0xbcu); }},
        {"format support: 192, past A4B4G4R4", [](ID3D12Device* d) { return format_support1(d, 0xc0u); }},
        {"format support: 1000", [](ID3D12Device* d) { return format_support1(d, 0x3e8u); }},
        {"format support: -1", [](ID3D12Device* d) { return format_support1(d, 0xffffffffu); }},
        {"fence: null output", [](ID3D12Device* d) {
             return d->CreateFence(0, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Fence), nullptr);
         }},
        {"fence: flag 0x80, null output", [](ID3D12Device* d) {
             return d->CreateFence(0, D3D12_FENCE_FLAGS(0x80), __uuidof(ID3D12Fence), nullptr);
         }},
        {"fence: non-monitored, null output", [](ID3D12Device* d) {
             return d->CreateFence(0, D3D12_FENCE_FLAG_NON_MONITORED, __uuidof(ID3D12Fence), nullptr);
         }},
        {"fence: wrong interface", [](ID3D12Device* d) {
             return fence(d, D3D12_FENCE_FLAG_NONE, __uuidof(ID3D12Resource));
         }},
        {"fence: flag 0x80", [](ID3D12Device* d) { return fence(d, D3D12_FENCE_FLAGS(0x80)); }},
        {"fence: cross-adapter without shared", [](ID3D12Device* d) {
             return fence(d, D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER);
         }},
        {"fence: non-monitored", [](ID3D12Device* d) { return fence(d, D3D12_FENCE_FLAG_NON_MONITORED); }},
        {"fence: shared", [](ID3D12Device* d) { return fence(d, D3D12_FENCE_FLAG_SHARED); }},
        {"fence: shared cross-adapter", [](ID3D12Device* d) {
             return fence(d, D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER);
         }},
        {"fence: shared non-monitored", [](ID3D12Device* d) {
             return fence(d, D3D12_FENCE_FLAG_SHARED | D3D12_FENCE_FLAG_NON_MONITORED);
         }},
        // Map.
        {"map: upload buffer", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_UPLOAD}, buffer_desc(256), D3D12_RESOURCE_STATE_GENERIC_READ, 0, nullptr);
         }},
        {"map: upload buffer, null pointer", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_UPLOAD}, buffer_desc(256), D3D12_RESOURCE_STATE_GENERIC_READ, 0, nullptr,
                        false);
         }},
        {"map: upload buffer, subresource 1", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_UPLOAD}, buffer_desc(256), D3D12_RESOURCE_STATE_GENERIC_READ, 1, nullptr);
         }},
        {"map: readback buffer, read range past the end", [](ID3D12Device* d) {
             const D3D12_RANGE range{0, 4096};
             return map(d, {D3D12_HEAP_TYPE_READBACK}, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST, 0, &range);
         }},
        {"map: readback buffer, reversed read range", [](ID3D12Device* d) {
             const D3D12_RANGE range{128, 64};
             return map(d, {D3D12_HEAP_TYPE_READBACK}, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST, 0, &range);
         }},
        {"map: readback buffer, empty read range", [](ID3D12Device* d) {
             const D3D12_RANGE range{0, 0};
             return map(d, {D3D12_HEAP_TYPE_READBACK}, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST, 0, &range);
         }},
        {"map: default buffer", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON, 0, nullptr);
         }},
        {"map: default buffer, null pointer", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_DEFAULT}, buffer_desc(256), D3D12_RESOURCE_STATE_COMMON, 0, nullptr, false);
         }},
        {"map: default texture", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_DEFAULT},
                        texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                        D3D12_RESOURCE_STATE_COMMON, 0, nullptr);
         }},
        {"map: custom write-back buffer", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0},
                        buffer_desc(256), D3D12_RESOURCE_STATE_COMMON, 0, nullptr);
         }},
        {"map: custom NOT_AVAILABLE buffer", [](ID3D12Device* d) {
             return map(d, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, D3D12_MEMORY_POOL_L0},
                        buffer_desc(256), D3D12_RESOURCE_STATE_COMMON, 0, nullptr);
         }},
    };
    int mismatches = 0;
    const bool verbose = GetEnvironmentVariableA("LEONIDA_D3D49_VERBOSE", nullptr, 0) > 0;
    for (const Case& c : cases) {
        if (verbose) {  // flushed first: a host runtime that dies on a case leaves its name behind
            std::printf("invalid arguments %s:", c.name);
            std::fflush(stdout);
        }
        const HRESULT a = call_guarded(c.call, ours), b = call_guarded(c.call, host.p);
        if (verbose) std::printf(" ours 0x%08lx host 0x%08lx\n", a, b);
        if (a == b) continue;
        if (c.windows10_refuses && windows10_refusal(c.name, host.p, a, b)) continue;
        if (c.comparable && !c.comparable(host.p)) {
            std::printf("invalid arguments %s: not comparable on this host (ours 0x%08lx, host 0x%08lx)\n", c.name,
                        a, b);
            continue;
        }
        ++mismatches;
        std::fprintf(stderr, "invalid arguments %s: ours 0x%08lx, host 0x%08lx\n", c.name, a, b);
    }

    // A buffer on a GPU upload heap. The layer reports OPTIONS16.GPUUploadHeapSupported FALSE and refuses the heap
    // type; a host runtime refuses it too when it has no such heap (Resizable BAR off), and accepts it otherwise, so
    // the answers are compared only when the host also reports no support.
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS16 host16{};
        const bool host_gpu_upload =
            SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, &host16, sizeof(host16))) &&
            host16.GPUUploadHeapSupported;
        const auto gpu_upload = [](ID3D12Device* d) {
            return committed(d, {D3D12_HEAP_TYPE_GPU_UPLOAD}, buffer_desc(256));
        };
        const HRESULT a = call_guarded(gpu_upload, ours), b = call_guarded(gpu_upload, host.p);
        if (host_gpu_upload) {
            if (verbose) std::printf("invalid arguments GPU upload heap buffer: the host has GPU upload heaps\n");
        } else if (a != b) {
            ++mismatches;
            std::fprintf(stderr, "invalid arguments GPU upload heap buffer: ours 0x%08lx, host 0x%08lx\n", a, b);
        } else if (verbose) {
            std::printf("invalid arguments GPU upload heap buffer: ours 0x%08lx host 0x%08lx\n", a, b);
        }
    }

    // ALLOW_RENDER_TARGET and ALLOW_DEPTH_STENCIL on every DXGI format both devices create as a plain texture. The
    // host's answers follow the format family (the same on the three adapters measured), except R32G32B32, which is
    // the driver's. Whether the layer can make an attachment of a format is its Vulkan driver's answer, so only the
    // host refusing what the layer accepts is a mismatch; the other way round (D32_FLOAT and D16_UNORM as render
    // targets, the R32/R16 colour formats as depth buffers on the drivers measured) is printed.
    for (int f = 1; f <= 99; ++f) {
        const DXGI_FORMAT format = DXGI_FORMAT(f);
        const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
        const auto create = [&](ID3D12Device* device, D3D12_RESOURCE_FLAGS flags) {
            return committed(device, heap, texture_desc(16, 16, format, flags));
        };
        if (FAILED(create(ours, D3D12_RESOURCE_FLAG_NONE)) || FAILED(create(host.p, D3D12_RESOURCE_FLAG_NONE)))
            continue;
        for (const D3D12_RESOURCE_FLAGS flags : {D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                                 D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL}) {
            const char* what = flags == D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET ? "RT" : "DS";
            const HRESULT a = create(ours, flags), b = create(host.p, flags);
            if (verbose)
                std::printf("invalid arguments format %d %s: ours 0x%08lx host 0x%08lx\n", f, what, a, b);
            if (a == b) continue;
            if (FAILED(a) && SUCCEEDED(b)) {
                std::printf("invalid arguments format %d %s: the host accepts it, this Vulkan driver can't\n", f, what);
                continue;
            }
            ++mismatches;
            std::fprintf(stderr, "invalid arguments format %d %s: ours 0x%08lx, host 0x%08lx\n", f, what, a, b);
        }
    }
    return mismatches;
}

// D3D-238: the valid neighbours of the CreateHeap refusals in compare_invalid_arguments, compared
// with the host on a device that has the GPU address arena (without it the layer answers every heap E_NOTIMPL).
int compare_heap_neighbours(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    struct Case {
        const char* name;
        InvalidCase call;
        bool windows10_refuses = false;
    };
    static const Case cases[] = {
        {"heap: alignment 0", [](ID3D12Device* d) { return heap_create(d, heap_desc_of(0)); }},
        {"heap: alignment 64 KiB", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT));
         }},
        {"heap: alignment 4 MiB", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT,
                                                {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_NONE,
                                                D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT));
         }},
        {"heap: size 1000", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_NONE, 1000));
         }},
        {"heap: upload", [](ID3D12Device* d) { return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_UPLOAD})); }},
        {"heap: readback", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_READBACK}));
         }},
        {"heap: custom write-combine L0", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                                                    D3D12_MEMORY_POOL_L0}));
         }},
        {"heap: ALLOW_ONLY_BUFFERS (two DENY flags)", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS));
         }},
        // The D3D12_HEAP_FLAGS documentation calls it unsupported, but the host accepts it (RTX 4070 SUPER and
        // UHD 770, @CowBoYdu76). Windows 10's and Windows 11 23H2's runtimes refuse it, as they refuse the committed
        // case above.
        {"heap: HARDWARE_PROTECTED", [](ID3D12Device* d) {
             return heap_create(d, heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_HARDWARE_PROTECTED));
         }, true},
    };
    int mismatches = 0;
    for (const Case& c : cases) {
        const HRESULT a = call_guarded(c.call, ours), b = call_guarded(c.call, host.p);
        if (a == b) continue;
        if (c.windows10_refuses && windows10_refusal(c.name, host.p, a, b)) continue;
        ++mismatches;
        std::fprintf(stderr, "invalid arguments %s: ours 0x%08lx, host 0x%08lx\n", c.name, a, b);
    }
    return mismatches;
}

// ID3D12Heap::GetDesc reads back as the host's for heaps of each type with each node-mask pair, on a device with the
// GPU address arena: a node mask of 0 reads back as the one node (1), as the layer's resources, descriptor heaps and
// queues already answer (d3d12_node_mask_test). Returns the number of mismatches, or -1 without a host.
int compare_heap_descs(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    struct HeapCase {
        D3D12_HEAP_PROPERTIES properties;
        UINT64 alignment;  // 0: the default
    };
    const HeapCase heaps[] = {
        {{D3D12_HEAP_TYPE_DEFAULT}, 0},
        {{D3D12_HEAP_TYPE_DEFAULT}, D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT},
        {{D3D12_HEAP_TYPE_UPLOAD}, 0},
        {{D3D12_HEAP_TYPE_READBACK}, 0},
        {{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0}, 0},
    };
    const UINT masks[][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    int mismatches = 0;
    for (const HeapCase& heap : heaps) {
        for (const auto& mask : masks) {
            D3D12_HEAP_PROPERTIES props = heap.properties;
            props.CreationNodeMask = mask[0];
            props.VisibleNodeMask = mask[1];
            const D3D12_HEAP_DESC desc = heap_desc_of(heap.alignment, props, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
                                                      heap.alignment ? heap.alignment : 64 * 1024);
            Ref<ID3D12Heap> a, e;
            const HRESULT ha = ours->CreateHeap(&desc, IID_PPV_ARGS(&a.p));
            const HRESULT he = host->CreateHeap(&desc, IID_PPV_ARGS(&e.p));
            if (FAILED(he) || !e.p) continue;  // nothing to compare
            if (FAILED(ha) || !a.p) {
                ++mismatches;
                std::fprintf(stderr, "heap GetDesc type %d align %llu masks %u/%u: the layer refused it (0x%08lx)\n",
                             int(props.Type), heap.alignment, mask[0], mask[1], ha);
                continue;
            }
            const D3D12_HEAP_DESC x = a->GetDesc(), y = e->GetDesc();
            if (x.SizeInBytes == y.SizeInBytes && x.Alignment == y.Alignment && x.Flags == y.Flags &&
                x.Properties.Type == y.Properties.Type &&
                x.Properties.CPUPageProperty == y.Properties.CPUPageProperty &&
                x.Properties.MemoryPoolPreference == y.Properties.MemoryPoolPreference &&
                x.Properties.CreationNodeMask == y.Properties.CreationNodeMask &&
                x.Properties.VisibleNodeMask == y.Properties.VisibleNodeMask)
                continue;
            ++mismatches;
            std::fprintf(stderr,
                         "heap GetDesc type %d align %llu masks %u/%u: ours size %llu align %llu flags 0x%x cpu %d "
                         "pool %d masks %u/%u, host size %llu align %llu flags 0x%x cpu %d pool %d masks %u/%u\n",
                         int(props.Type), heap.alignment, mask[0], mask[1], x.SizeInBytes, x.Alignment,
                         unsigned(x.Flags), int(x.Properties.CPUPageProperty),
                         int(x.Properties.MemoryPoolPreference), x.Properties.CreationNodeMask,
                         x.Properties.VisibleNodeMask, y.SizeInBytes, y.Alignment, unsigned(y.Flags),
                         int(y.Properties.CPUPageProperty), int(y.Properties.MemoryPoolPreference),
                         y.Properties.CreationNodeMask, y.Properties.VisibleNodeMask);
        }
    }
    std::printf("ID3D12Heap::GetDesc against the host: %zu heaps, %d differ\n", std::size(heaps) * std::size(masks),
                mismatches);
    return mismatches;
}

// CreatePlacedResource and CreateReservedResource answer their arguments as the host runtime does: the placement
// (offset, alignment, room in the heap), the heap's type and DENY flags against the resource, the description and
// initial state rules CreateCommittedResource shares, and the output and interface. Run on a device with the GPU
// address arena (without it every heap is E_NOTIMPL). Returns the number of mismatches, or -1 without a host.
int compare_placed_reserved(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    struct Case {
        const char* name;
        InvalidCase call;
        bool (*comparable)(ID3D12Device*) = nullptr;
        bool windows10_refuses = false;
    };
    static const Case cases[] = {
        // The standalone device has acceleration structures (XDXR, or KHR with LEONIDA_RT_AS=khr) and accepts a UAV
        // buffer in their state, as a host with ray tracing does.
        {"committed (standalone): UAV buffer in RAYTRACING_ACCELERATION_STRUCTURE state", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return committed(d, {D3D12_HEAP_TYPE_DEFAULT}, desc,
                              D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
         }, host_ray_tracing},
        {"placed: UAV buffer in RAYTRACING_ACCELERATION_STRUCTURE state", [](ID3D12Device* d) {
             PlacedArgs a;
             a.state = D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE;
             auto desc = buffer_desc(64 * 1024);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             return placed(d, desc, a);
         }, host_ray_tracing},
        // CreatePlacedResource: arguments.
        {"placed: buffer at 0", [](ID3D12Device* d) { return placed(d, buffer_desc(64 * 1024)); }},
        {"placed: null description", [](ID3D12Device* d) { return placed(d, nullptr); }},
        {"placed: null heap", [](ID3D12Device* d) {
             const D3D12_RESOURCE_DESC desc = buffer_desc(64 * 1024);
             void* out = nullptr;
             return released(d->CreatePlacedResource(nullptr, 0, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                     __uuidof(ID3D12Resource), &out), &out);
         }},
        {"placed: null output (capability check)", [](ID3D12Device* d) {
             PlacedArgs a;
             a.output = false;
             return placed(d, buffer_desc(64 * 1024), a);
         }},
        {"placed: null output, offset 4096", [](ID3D12Device* d) {
             PlacedArgs a;
             a.output = false;
             a.offset = 4096;
             return placed(d, buffer_desc(64 * 1024), a);
         }},
        {"placed: null output, buffer height 2", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Height = 2;
             PlacedArgs a;
             a.output = false;
             return placed(d, desc, a);
         }},
        {"placed: null output, texture past the heap's end", [](ID3D12Device* d) {
             PlacedArgs a;
             a.output = false;
             a.offset = 1024 * 1024 - 64 * 1024;
             return placed(d, texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: wrong interface", [](ID3D12Device* d) {
             PlacedArgs a;
             a.riid = &__uuidof(ID3D12Fence);
             return placed(d, buffer_desc(64 * 1024), a);
         }},
        // CreatePlacedResource: where the resource goes.
        {"placed: buffer at 64 KiB", [](ID3D12Device* d) {
             PlacedArgs a;
             a.offset = 64 * 1024;
             return placed(d, buffer_desc(64 * 1024), a);
         }},
        {"placed: buffer at 4096", [](ID3D12Device* d) {
             PlacedArgs a;
             a.offset = 4096;
             return placed(d, buffer_desc(256), a);
         }},
        {"placed: buffer filling the heap", [](ID3D12Device* d) { return placed(d, buffer_desc(1024 * 1024)); }},
        {"placed: buffer one byte larger than the heap", [](ID3D12Device* d) {
             return placed(d, buffer_desc(1024 * 1024 + 1));
         }},
        {"placed: buffer past the heap's end", [](ID3D12Device* d) {
             PlacedArgs a;
             a.offset = 1024 * 1024 - 64 * 1024;
             return placed(d, buffer_desc(128 * 1024), a);
         }},
        {"placed: buffer at the heap's end", [](ID3D12Device* d) {
             PlacedArgs a;
             a.offset = 1024 * 1024;
             return placed(d, buffer_desc(256), a);
         }},
        {"placed: small texture at 4096 (4 KiB alignment)", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
             PlacedArgs a;
             a.offset = 4096;
             return placed(d, desc, a);
         }},
        {"placed: texture at 4096 (default alignment)", [](ID3D12Device* d) {
             PlacedArgs a;
             a.offset = 4096;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: texture at 64 KiB", [](ID3D12Device* d) {
             PlacedArgs a;
             a.offset = 64 * 1024;
             return placed(d, texture_desc(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: texture larger than the heap", [](ID3D12Device* d) {
             return placed(d, texture_desc(1024, 1024, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"placed: texture past the heap's end", [](ID3D12Device* d) {
             PlacedArgs a;
             a.offset = 1024 * 1024 - 64 * 1024;
             return placed(d, texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: MSAA texture at 64 KiB (4 MiB alignment)", [](ID3D12Device* d) {
             auto desc = texture_desc(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc.Count = 4;
             PlacedArgs a;
             a.heap = heap_desc_of(D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT, {D3D12_HEAP_TYPE_DEFAULT},
                                   D3D12_HEAP_FLAG_NONE, 8 * 1024 * 1024);
             a.offset = 64 * 1024;
             return placed(d, desc, a);
         }},
        {"placed: MSAA texture in a 64 KiB-aligned heap", [](ID3D12Device* d) {
             auto desc = texture_desc(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
             desc.SampleDesc.Count = 4;
             return placed(d, desc);
         }},
        // CreatePlacedResource: the heap's type and DENY flags against the resource.
        {"placed: buffer in an ALLOW_ONLY_NON_RT_DS_TEXTURES heap", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
             return placed(d, buffer_desc(256), a);
         }},
        {"placed: texture in an ALLOW_ONLY_BUFFERS heap", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: render target in an ALLOW_ONLY_NON_RT_DS_TEXTURES heap", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), a);
         }},
        {"placed: plain texture in an ALLOW_ONLY_RT_DS_TEXTURES heap", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: render target in an ALLOW_ONLY_RT_DS_TEXTURES heap", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), a);
         }},
        {"placed: texture in an upload heap", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
             a.state = D3D12_RESOURCE_STATE_GENERIC_READ;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        // CreatePlacedResource: the description rules CreateCommittedResource has.
        {"placed: buffer width 0", [](ID3D12Device* d) { return placed(d, buffer_desc(0)); }},
        {"placed: buffer height 2", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Height = 2;
             return placed(d, desc);
         }},
        {"placed: buffer with 2 mips", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.MipLevels = 2;
             return placed(d, desc);
         }},
        {"placed: buffer with a format", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
             return placed(d, desc);
         }},
        {"placed: buffer with layout UNKNOWN", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
             return placed(d, desc);
         }},
        {"placed: buffer with 4 KiB alignment", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
             return placed(d, desc);
         }},
        {"placed: buffer with ALLOW_DEPTH_STENCIL", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL;
             return placed(d, desc);
         }},
        {"placed: texture width 0", [](ID3D12Device* d) {
             return placed(d, texture_desc(0, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"placed: texture with format UNKNOWN", [](ID3D12Device* d) {
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"placed: texture with layout ROW_MAJOR", [](ID3D12Device* d) {
             auto desc = texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
             return placed(d, desc);
         }},
        {"placed: depth texture with ALLOW_UNORDERED_ACCESS", [](ID3D12Device* d) {
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT,
                                           D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL |
                                               D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS));
         }},
        // CreatePlacedResource: the initial state and the clear value.
        {"placed: upload buffer in GENERIC_READ", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
             a.state = D3D12_RESOURCE_STATE_GENERIC_READ;
             return placed(d, buffer_desc(256), a);
         }},
        {"placed: upload buffer in COPY_DEST", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
             a.state = D3D12_RESOURCE_STATE_COPY_DEST;
             return placed(d, buffer_desc(256), a);
         }, nullptr, true},
        {"placed: upload buffer with ALLOW_UNORDERED_ACCESS", [](ID3D12Device* d) {
             auto desc = buffer_desc(256);
             desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
             PlacedArgs a;
             a.heap.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
             a.state = D3D12_RESOURCE_STATE_GENERIC_READ;
             return placed(d, desc, a);
         }},
        {"placed: readback buffer in COPY_DEST", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Properties.Type = D3D12_HEAP_TYPE_READBACK;
             a.state = D3D12_RESOURCE_STATE_COPY_DEST;
             return placed(d, buffer_desc(256), a);
         }},
        {"placed: readback buffer in GENERIC_READ", [](ID3D12Device* d) {
             PlacedArgs a;
             a.heap.Properties.Type = D3D12_HEAP_TYPE_READBACK;
             a.state = D3D12_RESOURCE_STATE_GENERIC_READ;
             return placed(d, buffer_desc(256), a);
         }, nullptr, true},
        {"placed: default buffer in state 0x80000000", [](ID3D12Device* d) {
             PlacedArgs a;
             a.state = D3D12_RESOURCE_STATES(0x80000000u);
             return placed(d, buffer_desc(256), a);
         }, nullptr, true},
        {"placed: depth texture in DEPTH_WRITE", [](ID3D12Device* d) {
             PlacedArgs a;
             a.state = D3D12_RESOURCE_STATE_DEPTH_WRITE;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                           a);
         }},
        {"placed: plain texture in UNORDERED_ACCESS", [](ID3D12Device* d) {
             PlacedArgs a;
             a.state = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: render target in RENDER_TARGET", [](ID3D12Device* d) {
             PlacedArgs a;
             a.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), a);
         }},
        {"placed: plain texture in RENDER_TARGET", [](ID3D12Device* d) {
             PlacedArgs a;
             a.state = D3D12_RESOURCE_STATE_RENDER_TARGET;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: render target with a clear value", [](ID3D12Device* d) {
             static const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.f, 0.f, 0.f, 1.f}};
             PlacedArgs a;
             a.clear = &clear;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), a);
         }},
        {"placed: plain texture with a clear value", [](ID3D12Device* d) {
             static const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.f, 0.f, 0.f, 1.f}};
             PlacedArgs a;
             a.clear = &clear;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE), a);
         }},
        {"placed: buffer with a clear value", [](ID3D12Device* d) {
             static const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.f, 0.f, 0.f, 1.f}};
             PlacedArgs a;
             a.clear = &clear;
             return placed(d, buffer_desc(256), a);
         }},
        {"placed: render target, clear value of another format", [](ID3D12Device* d) {
             static const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R16G16B16A16_FLOAT, {0.f, 0.f, 0.f, 1.f}};
             PlacedArgs a;
             a.clear = &clear;
             return placed(d, texture_desc(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET), a);
         }},
        // CreateReservedResource.
        {"reserved: buffer", [](ID3D12Device* d) { return reserved(d, buffer_desc(64 * 1024)); }},
        {"reserved: null description", [](ID3D12Device* d) { return reserved(d, nullptr); }},
        {"reserved: null output (capability check)", [](ID3D12Device* d) {
             const auto desc = buffer_desc(64 * 1024);
             return reserved(d, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, false);
         }},
        {"reserved: null output, buffer height 2", [](ID3D12Device* d) {
             auto desc = buffer_desc(64 * 1024);
             desc.Height = 2;
             return reserved(d, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, false);
         }},
        {"reserved: null output, plain texture in RENDER_TARGET", [](ID3D12Device* d) {
             const auto desc =
                 tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
             return reserved(d, &desc, D3D12_RESOURCE_STATE_RENDER_TARGET, nullptr, false);
         }},
        {"reserved: wrong interface", [](ID3D12Device* d) {
             const auto desc = buffer_desc(64 * 1024);
             return reserved(d, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr, true, __uuidof(ID3D12Fence));
         }},
        {"reserved: buffer with layout UNKNOWN", [](ID3D12Device* d) {
             auto desc = buffer_desc(64 * 1024);
             desc.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
             return reserved(d, desc);
         }},
        {"reserved: buffer width 0", [](ID3D12Device* d) { return reserved(d, buffer_desc(0)); }},
        {"reserved: buffer height 2", [](ID3D12Device* d) {
             auto desc = buffer_desc(64 * 1024);
             desc.Height = 2;
             return reserved(d, desc);
         }},
        {"reserved: buffer with a format", [](ID3D12Device* d) {
             auto desc = buffer_desc(64 * 1024);
             desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
             return reserved(d, desc);
         }},
        {"reserved: buffer with 4 KiB alignment", [](ID3D12Device* d) {
             auto desc = buffer_desc(64 * 1024);
             desc.Alignment = D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT;
             return reserved(d, desc);
         }},
        {"reserved: buffer in COPY_DEST", [](ID3D12Device* d) {
             return reserved(d, buffer_desc(64 * 1024), D3D12_RESOURCE_STATE_COPY_DEST);
         }},
        {"reserved: buffer in state 0x80000000", [](ID3D12Device* d) {
             return reserved(d, buffer_desc(64 * 1024), D3D12_RESOURCE_STATES(0x80000000u));
         }, nullptr, true},
        {"reserved: texture", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)));
         }},
        {"reserved: texture with layout UNKNOWN", [](ID3D12Device* d) {
             return reserved(d, texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
         }},
        {"reserved: texture with layout ROW_MAJOR", [](ID3D12Device* d) {
             auto desc = texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
             return reserved(d, desc);
         }},
        {"reserved: texture with 64KB_STANDARD_SWIZZLE", [](ID3D12Device* d) {
             auto desc = texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
             desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE;
             return reserved(d, desc);
         }},
        {"reserved: texture width 0", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(0, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)));
         }},
        {"reserved: full mip chain", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE,
                                                   0)));
         }},
        {"reserved: BC1 texture", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE)));
         }},
        {"reserved: 1D texture", [](ID3D12Device* d) {
             auto desc = tiled(texture_desc(256, 1, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE));
             desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE1D;
             return reserved(d, desc);
         }},
        {"reserved: D24S8 depth texture", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_D24_UNORM_S8_UINT,
                                                   D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)));
         }},
        {"reserved: D32 depth texture", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_D32_FLOAT,
                                                   D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)));
         }},
        {"reserved: R32G32B32 texture", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_R32G32B32_FLOAT, D3D12_RESOURCE_FLAG_NONE)));
         }},
        {"reserved: texture with ALLOW_SIMULTANEOUS_ACCESS", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                   D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS)));
         }},
        {"reserved: render target with a clear value", [](ID3D12Device* d) {
             static const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.f, 0.f, 0.f, 1.f}};
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                   D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)),
                             D3D12_RESOURCE_STATE_COMMON, &clear);
         }},
        {"reserved: plain texture with a clear value", [](ID3D12Device* d) {
             static const D3D12_CLEAR_VALUE clear{DXGI_FORMAT_R8G8B8A8_UNORM, {0.f, 0.f, 0.f, 1.f}};
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)),
                             D3D12_RESOURCE_STATE_COMMON, &clear);
         }},
        {"reserved: plain texture in RENDER_TARGET", [](ID3D12Device* d) {
             return reserved(d, tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)),
                             D3D12_RESOURCE_STATE_RENDER_TARGET);
         }},
        {"reserved: 3D texture", [](ID3D12Device* d) {
             auto desc = tiled(texture_desc(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1, 16));
             desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
             return reserved(d, desc);
         }},
        {"reserved: 4x MSAA render target", [](ID3D12Device* d) {
             auto desc = tiled(texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM,
                                            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET));
             desc.SampleDesc.Count = 4;
             return reserved(d, desc);
         }},
    };
    const bool verbose = GetEnvironmentVariableA("LEONIDA_D3D49_VERBOSE", nullptr, 0) > 0;
    int mismatches = 0;
    for (const Case& c : cases) {
        if (verbose) {
            std::printf("invalid arguments %s:", c.name);
            std::fflush(stdout);
        }
        const HRESULT a = call_guarded(c.call, ours), b = call_guarded(c.call, host.p);
        if (verbose) std::printf(" ours 0x%08lx host 0x%08lx\n", a, b);
        if (a == b) continue;
        if (c.windows10_refuses && windows10_refusal(c.name, host.p, a, b)) continue;
        if (c.comparable && !c.comparable(host.p)) {
            std::printf("invalid arguments %s: not comparable on this host (ours 0x%08lx, host 0x%08lx)\n", c.name,
                        a, b);
            continue;
        }
        ++mismatches;
        std::fprintf(stderr, "invalid arguments %s: ours 0x%08lx, host 0x%08lx\n", c.name, a, b);
    }
    return mismatches;
}

// CheckFeatureSupport's error paths must match the host runtime's exactly: games and middleware branch on these
// HRESULTs at startup, and a wrong success or a wrong error sends them down the wrong path. A null data pointer, a
// structure smaller than the feature requires, or a feature value no runtime defines must all return the same HRESULT
// on both sides. Valid queries are not compared here: their answers are the GPU's choice. Returns the number of
// mismatches, or -1 when the host has no D3D12 device to compare with.
int compare_check_feature_support(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    // Features the shim answers; their error paths must match the host's.
    const D3D12_FEATURE answered[] = {
        D3D12_FEATURE_FEATURE_LEVELS, D3D12_FEATURE_ARCHITECTURE, D3D12_FEATURE_ARCHITECTURE1,
        D3D12_FEATURE_D3D12_OPTIONS, D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, D3D12_FEATURE_FORMAT_SUPPORT,
    };
    // Feature values no runtime defines (well beyond D3D12_FEATURE's range); both sides must reject them the same way.
    const D3D12_FEATURE undefined[] = {D3D12_FEATURE(0x4000), D3D12_FEATURE(0x7fffffff)};
    int mismatches = 0;
    const auto check = [&](const char* what, D3D12_FEATURE feature, void* data, UINT size) {
        const HRESULT a = ours->CheckFeatureSupport(feature, data, size);
        const HRESULT b = host->CheckFeatureSupport(feature, data, size);
        if (a == b) return;
        if (mismatches++ < 24)
            std::fprintf(stderr, "CheckFeatureSupport %s feature %d size %u: ours 0x%08x host 0x%08x\n", what,
                         int(feature), size, static_cast<unsigned>(a), static_cast<unsigned>(b));
    };
    for (const D3D12_FEATURE f : answered) {
        std::uint8_t buf[256] = {};
        check("null data", f, nullptr, sizeof(buf));
        check("zero size", f, buf, 0);
        check("one byte", f, buf, 1);
    }
    // A structure one byte short, for every answered feature's data type.
    {
        D3D12_FEATURE_DATA_FEATURE_LEVELS fl{};
        check("feature levels short", D3D12_FEATURE_FEATURE_LEVELS, &fl, sizeof(fl) - 1);
        D3D12_FEATURE_DATA_ARCHITECTURE ar{};
        check("architecture short", D3D12_FEATURE_ARCHITECTURE, &ar, sizeof(ar) - 1);
        D3D12_FEATURE_DATA_ARCHITECTURE1 a1{};
        check("architecture1 short", D3D12_FEATURE_ARCHITECTURE1, &a1, sizeof(a1) - 1);
        D3D12_FEATURE_DATA_D3D12_OPTIONS op{};
        check("options short", D3D12_FEATURE_D3D12_OPTIONS, &op, sizeof(op) - 1);
        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va{};
        check("gpu va short", D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va, sizeof(va) - 1);
        D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{};
        check("format short", D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs) - 1);
    }
    for (const D3D12_FEATURE f : undefined) {
        std::uint8_t buf[256] = {};
        check("undefined feature", f, buf, sizeof(buf));
    }
    return mismatches;
}

// Fences and queues must follow the host runtime's semantics: Signal raises the completed value and a null event set
// for a reached value blocks until then. Runs the same sequence on ours and the host and compares the completed value
// and the HRESULT at each step. A signal of an already-passed (lower) value is a documented deviation: the shim drops
// it and stays monotonic because a Vulkan timeline cannot go back, so that case is checked against the documented
// behaviour rather than the host. Returns the number of mismatches, or -1 when the host has no D3D12 device.
int compare_fences(ID3D12Device* ours) {
    Ref<ID3D12Device> host;
    if (!create_host_device(host)) return -1;
    D3D12_COMMAND_QUEUE_DESC queue_desc{};
    queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    Ref<ID3D12CommandQueue> oq, hq;
    Ref<ID3D12Fence> of, hf;
    if (FAILED(ours->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&oq.p))) ||
        FAILED(host->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&hq.p))) ||
        FAILED(ours->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&of.p))) ||
        FAILED(host->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&hf.p))))
        return -1;
    int mismatches = 0;
    const auto check = [&](const char* what, UINT64 a, UINT64 b) {
        if (a != b && mismatches++ < 24)
            std::fprintf(stderr, "fence %s: ours %llu host %llu\n", what, static_cast<unsigned long long>(a),
                         static_cast<unsigned long long>(b));
    };
    const auto check_hr = [&](const char* what, HRESULT a, HRESULT b) {
        if (a != b && mismatches++ < 24)
            std::fprintf(stderr, "fence %s: ours 0x%08x host 0x%08x\n", what, static_cast<unsigned>(a),
                         static_cast<unsigned>(b));
    };
    // A signal raises the completed value; a null event blocks until it is reached.
    check_hr("signal 5", oq->Signal(of.p, 5), hq->Signal(hf.p, 5));
    check_hr("block to 5", of->SetEventOnCompletion(5, nullptr), hf->SetEventOnCompletion(5, nullptr));
    check("completed at 5", of->GetCompletedValue(), hf->GetCompletedValue());
    // A higher signal raises it again.
    check_hr("signal 9", oq->Signal(of.p, 9), hq->Signal(hf.p, 9));
    check_hr("block to 9", of->SetEventOnCompletion(9, nullptr), hf->SetEventOnCompletion(9, nullptr));
    check("completed at 9", of->GetCompletedValue(), hf->GetCompletedValue());
    // An event for an already-reached value returns at once.
    check_hr("event at reached 9", of->SetEventOnCompletion(9, nullptr), hf->SetEventOnCompletion(9, nullptr));
    // A signal of an already-passed (lower) value: the host is free to lower the fence on some drivers, but a Vulkan
    // timeline only increases. The shim deliberately drops it and keeps the higher value (see Fence::claim in
    // d3d12_com.cpp), so this is checked against the documented behaviour, not the host.
    check_hr("signal 3 (lower)", oq->Signal(of.p, 3), S_OK);
    if (of->GetCompletedValue() != 9 && mismatches++ < 24)
        std::fprintf(stderr, "fence lower signal: ours %llu, want 9\n",
                     static_cast<unsigned long long>(of->GetCompletedValue()));
    return mismatches;
}

}  // namespace

int main() {
    vktest::Device d;
    const auto status = d.initialize();
    if (status == vktest::DeviceStatus::unavailable) return 77;
    if (status == vktest::DeviceStatus::failed) return 1;
    {
        // The host comparisons (create_host_device, the copies) run on the layer's own adapter, not D3D12's default
        // one, which on a PC with two GPUs can be the other GPU (copycmp::layer_adapter).
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
        d.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(d.physical, &properties);
        if (id.deviceLUIDValid) {
            LUID luid{};
            std::memcpy(&luid, id.deviceLUID, sizeof(luid));
            copycmp::layer_adapter() = luid;
        }
    }
    EXPECT(d.depth_format_usable(VK_FORMAT_D32_SFLOAT_S8_UINT));
    VkSampleCountFlags depth_sample_counts = 0, color_sample_counts = 0;
    {
        VkPhysicalDeviceProperties properties{};
        d.instance_fn<PFN_vkGetPhysicalDeviceProperties>("vkGetPhysicalDeviceProperties")(d.physical, &properties);
        depth_sample_counts =
            properties.limits.framebufferDepthSampleCounts & properties.limits.framebufferStencilSampleCounts;
        color_sample_counts = properties.limits.framebufferColorSampleCounts;
    }
    constexpr UINT size = 8, small_size = 4;
    unsigned mismatches = 0;
    const auto report = [&](const char* what, UINT index, const std::uint8_t* got, const std::uint8_t* want, size_t n) {
        if (std::memcmp(got, want, n) == 0) return;
        if (mismatches++ < 8) {
            std::fprintf(stderr, "%s %u: got", what, index);
            for (size_t i = 0; i < n; ++i) std::fprintf(stderr, " %02x", got[i]);
            std::fprintf(stderr, ", want");
            for (size_t i = 0; i < n; ++i) std::fprintf(stderr, " %02x", want[i]);
            std::fprintf(stderr, "\n");
        }
    };

    {
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.conditional_rendering = d.conditional_rendering;  // SetPredication (the D24 tests check copies under it)
        info.texture_compression_bc = d.bc;
        info.occlusion_query_precise = d.occlusion_query_precise;
        info.pipeline_statistics_query = d.pipeline_statistics_query;
        info.conditional_rendering = d.conditional_rendering;
        ID3D12Device* inner = nullptr;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&inner))));
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(inner, __uuidof(ID3D12Device),
                                                           reinterpret_cast<void**>(&device.p))));
        rt::d3d12::set_device_outer(inner, device.p);  // the proxy now owns `inner`
        EXPECT(rt::d3d12_xs::is_device_proxy(device.p));
        EXPECT(rt::d3d12_xs::device_proxy_slot_count(device.p) == 80);  // ID3D12Device11
        EXPECT(check_plain_resolve_formats(device.p));

        {
            Ref<ID3D12Device11> device11;
            Ref<IUnknown> device12;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device11.p))) && device11.p == device.p);
            EXPECT(device->QueryInterface(__uuidof(ID3D12Device12), reinterpret_cast<void**>(&device12.p)) ==
                   E_NOINTERFACE);
        }
        EXPECT(device->GetNodeCount() == 1);

        // _COM_Outptr_: a failed slot 14 leaves NULL in the caller's output, as the inner device does.
        {
            // An unknown flag (zero descriptors are valid, as on the host runtime).
            D3D12_DESCRIPTOR_HEAP_DESC bad{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 4,
                                           D3D12_DESCRIPTOR_HEAP_FLAGS(0x80)};
            void* inner_out = reinterpret_cast<void*>(0x1111);
            EXPECT(inner->CreateDescriptorHeap(&bad, __uuidof(ID3D12DescriptorHeap), &inner_out) == E_INVALIDARG &&
                   inner_out == nullptr);
            void* proxy_out = reinterpret_cast<void*>(0x1111);
            EXPECT(device->CreateDescriptorHeap(&bad, __uuidof(ID3D12DescriptorHeap), &proxy_out) ==
                   E_INVALIDARG);
            EXPECT(proxy_out == nullptr);
        }

        // BC1 can be sampled, but it cannot be an attachment or a storage image.
        if (d.bc) {
            const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
            Ref<ID3D12Device> host;
            create_host_device(host);
            for (const auto flags : {D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                     D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS}) {
                const auto desc = texture_desc(8, 8, DXGI_FORMAT_BC1_UNORM, flags);
                Ref<ID3D12Resource> image;
                const auto create = [&](ID3D12Device* dev, Ref<ID3D12Resource>& out) {
                    return dev->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                        nullptr, IID_PPV_ARGS(&out.p));
                };
                const HRESULT hr = create(device.p, image);
                const bool supported = flags == D3D12_RESOURCE_FLAG_NONE;
                EXPECT(hr == (supported ? S_OK : E_INVALIDARG));
                EXPECT((image.p != nullptr) == supported);
                if (host.p) {
                    Ref<ID3D12Resource> native;
                    EXPECT(create(host.p, native) == hr);
                    EXPECT((native.p != nullptr) == supported);
                }
                const auto allocation = device->GetResourceAllocationInfo(0, 1, &desc);
                EXPECT((allocation.SizeInBytes != UINT64_MAX) == supported);
            }
        }
        const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_2};
        D3D12_FEATURE_DATA_FEATURE_LEVELS feature_levels{3, levels, D3D_FEATURE_LEVEL(0)};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &feature_levels,
                                                     sizeof(feature_levels))));
        EXPECT(feature_levels.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL_12_1);
        // An empty or missing list is E_INVALIDARG on the host (NVIDIA and WARP), not DXGI_ERROR_UNSUPPORTED or a
        // read through null; a list the device can't meet answers DXGI_ERROR_UNSUPPORTED and writes level 0.
        D3D12_FEATURE_DATA_FEATURE_LEVELS no_levels{0, levels, D3D_FEATURE_LEVEL_11_0};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &no_levels, sizeof(no_levels)) == E_INVALIDARG);
        D3D12_FEATURE_DATA_FEATURE_LEVELS null_levels{1, nullptr, D3D_FEATURE_LEVEL_11_0};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &null_levels, sizeof(null_levels)) ==
               E_INVALIDARG);
        D3D12_FEATURE_DATA_FEATURE_LEVELS too_high{1, &levels[2], D3D_FEATURE_LEVEL_11_0};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FEATURE_LEVELS, &too_high, sizeof(too_high)) ==
               DXGI_ERROR_UNSUPPORTED);
        EXPECT(too_high.MaxSupportedFeatureLevel == D3D_FEATURE_LEVEL(0));
        D3D12_FEATURE_DATA_ARCHITECTURE architecture{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &architecture, sizeof(architecture))));
        D3D12_FEATURE_DATA_D3D12_OPTIONS options{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options))));
        EXPECT(options.ResourceBindingTier == D3D12_RESOURCE_BINDING_TIER_3);
        EXPECT(options.MaxGPUVirtualAddressBitsPerResource == 40);

        D3D12_FEATURE_DATA_GPU_VIRTUAL_ADDRESS_SUPPORT va_support{};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_GPU_VIRTUAL_ADDRESS_SUPPORT, &va_support, sizeof(va_support))));
        EXPECT(va_support.MaxGPUVirtualAddressBitsPerResource == 40);
        EXPECT(va_support.MaxGPUVirtualAddressBitsPerProcess == 48);

        EXPECT(compare_format_info(device.p, d) == 0);
        // Use a real oversized buffer and check that rejecting it leaves the output untouched.
        D3D12_FEATURE_DATA_FORMAT_INFO format_info[2]{{DXGI_FORMAT_D24_UNORM_S8_UINT, 0xff}};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, nullptr, sizeof(format_info[0])) == E_INVALIDARG);
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, format_info, sizeof(format_info)) == E_INVALIDARG);
        EXPECT(format_info[0].PlaneCount == 0xff);
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_INFO, format_info, sizeof(format_info[0])) == S_OK);
        EXPECT(format_info[0].PlaneCount == 2);

        D3D12_FEATURE_DATA_FORMAT_SUPPORT format_support{DXGI_FORMAT_R8G8B8A8_UNORM};
        EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &format_support, sizeof(format_support))));
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_RENDER_TARGET);
        EXPECT(format_support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE);

        // RTX 5070 / 621.14 host answers, without requiring a host device for this regression.
        for (const UINT samples : {0u, 33u, 64u, UINT_MAX}) {
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{DXGI_FORMAT_R8G8B8A8_UNORM, samples,
                                                                D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 99};
            EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality,
                                               sizeof(quality)) == E_FAIL);
            EXPECT(quality.NumQualityLevels == 0);
        }
        for (UINT format = 0; format <= 192; ++format) {
            const bool known = format != UINT(DXGI_FORMAT_R1_UNORM) &&
                               (format <= 115 || (format >= 189 && format <= 191));
            for (const UINT samples : {1u, 3u, 31u}) {
                D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS quality{DXGI_FORMAT(format), samples,
                                                                    D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 99};
                EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &quality,
                                                   sizeof(quality)) == (known ? S_OK : E_FAIL));
                EXPECT(quality.NumQualityLevels == (known ? UINT(samples == 1) : 99u));
            }
        }
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS invalid_format_levels{DXGI_FORMAT_FORCE_UINT, 1,
                                                                          D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 99};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &invalid_format_levels,
                                           sizeof(invalid_format_levels)) == E_FAIL);
        EXPECT(invalid_format_levels.NumQualityLevels == 99);
        // Native (RTX 5070 / 621.24): an unknown format keeps the output even with an invalid count.
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS unknown_zero{DXGI_FORMAT(192), 0,
                                                                   D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 99};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &unknown_zero,
                                           sizeof(unknown_zero)) == E_FAIL);
        EXPECT(unknown_zero.NumQualityLevels == 99);
        for (const auto format : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_BC1_UNORM, DXGI_FORMAT_R32G32B32_FLOAT}) {
            for (const UINT samples : {1u, 3u, 4u, 32u}) {
                for (const UINT flags : {0u, 1u}) {
                    D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS plain{format, samples,
                        D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS(flags)};
                    EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &plain,
                                                       sizeof(plain)) == S_OK);
                    if (flags && format == DXGI_FORMAT_R32G32B32_FLOAT) EXPECT(plain.NumQualityLevels == 0);
                    for (const UINT extra : {2u, 0xfeu}) {
                        auto undefined_flags = plain;
                        undefined_flags.Flags = D3D12_MULTISAMPLE_QUALITY_LEVEL_FLAGS(flags | extra);
                        undefined_flags.NumQualityLevels = 99;
                        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &undefined_flags,
                                                           sizeof(undefined_flags)) == S_OK);
                        EXPECT(undefined_flags.NumQualityLevels == plain.NumQualityLevels);
                    }
                }
            }
        }
        D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS wrong_size[2]{{DXGI_FORMAT_R8G8B8A8_UNORM, 1,
            D3D12_MULTISAMPLE_QUALITY_LEVELS_FLAG_NONE, 99}};
        for (const UINT bytes : {UINT(sizeof(wrong_size[0]) - 1), UINT(sizeof(wrong_size))}) {
            EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, wrong_size,
                                               bytes) == E_INVALIDARG);
            EXPECT(wrong_size[0].NumQualityLevels == 99);
        }

        D3D12_RESOURCE_DESC unsupported_quality =
            texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        unsupported_quality.SampleDesc.Count = 4;
        unsupported_quality.SampleDesc.Quality = 1;
        EXPECT(device->GetResourceAllocationInfo(0, 1, &unsupported_quality).SizeInBytes == UINT64_MAX);

        // UNKNOWN is a buffer format to the host runtime (S_OK, Support1 BUFFER; "format support: UNKNOWN" compares
        // it), and a value past its format table is E_FAIL.
        D3D12_FEATURE_DATA_FORMAT_SUPPORT unknown_format{DXGI_FORMAT_UNKNOWN};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &unknown_format, sizeof(unknown_format)) ==
                   S_OK &&
               unknown_format.Support1 == D3D12_FORMAT_SUPPORT1_BUFFER);
        D3D12_FEATURE_DATA_FORMAT_SUPPORT invalid_format{static_cast<DXGI_FORMAT>(116)};
        EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &invalid_format, sizeof(invalid_format)) ==
               E_FAIL);
        // The table's last entries before and after the gap are formats, whatever this GPU supports of them.
        for (const UINT edge : {115u, 189u, 191u}) {  // B4G4R4A4_UNORM, SAMPLER_FEEDBACK_MIN_MIP_OPAQUE, A4B4G4R4_UNORM
            D3D12_FEATURE_DATA_FORMAT_SUPPORT known{static_cast<DXGI_FORMAT>(edge)};
            EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &known, sizeof(known)) != E_FAIL);
        }

        // D3D-61: XR_BIAS is a display-only format with the R10G10B10A2 bit layout. The layer can create and
        // copy it, and its views can cast to the ordinary 10:10:10:2 format, but it does not claim display or
        // video-processor support.
        {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            D3D12_FEATURE_DATA_FORMAT_SUPPORT xr{DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &xr, sizeof(xr))));
            EXPECT(xr.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
            EXPECT(xr.Support1 & D3D12_FORMAT_SUPPORT1_CAST_WITHIN_BIT_LAYOUT);
            EXPECT(!(xr.Support1 & (D3D12_FORMAT_SUPPORT1_TEXTURE3D | D3D12_FORMAT_SUPPORT1_MIP |
                                    D3D12_FORMAT_SUPPORT1_SHADER_SAMPLE | D3D12_FORMAT_SUPPORT1_RENDER_TARGET |
                                    D3D12_FORMAT_SUPPORT1_BLENDABLE)));
            D3D12_FEATURE_DATA_MULTISAMPLE_QUALITY_LEVELS multsample{DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, 4};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_MULTISAMPLE_QUALITY_LEVELS, &multsample,
                                                         sizeof(multsample))));
            EXPECT(multsample.NumQualityLevels == 0);
            Ref<ID3D12Resource> tex;
            const D3D12_RESOURCE_DESC desc = texture_desc(
                16, 16, DXGI_FORMAT_R10G10B10_XR_BIAS_A2_UNORM, D3D12_RESOURCE_FLAG_NONE, 1);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&tex.p))));
            EXPECT(tex.p != nullptr);

            // The host refuses extra mips, multisampling and 3D; the layer must refuse the same descriptions.
            D3D12_RESOURCE_DESC invalid = desc;
            invalid.MipLevels = 2;
            EXPECT(FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &invalid,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          __uuidof(ID3D12Resource), nullptr)));
            EXPECT(device->GetResourceAllocationInfo(0, 1, &invalid).SizeInBytes == UINT64_MAX);
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};
            UINT rows = 0;
            UINT64 total = 0;
            device->GetCopyableFootprints(&invalid, 0, 1, 0, &layout, &rows, nullptr, &total);
            EXPECT(total == UINT64_MAX);

            invalid = desc;
            invalid.SampleDesc.Count = 4;
            invalid.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            EXPECT(FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &invalid,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          __uuidof(ID3D12Resource), nullptr)));
            EXPECT(device->GetResourceAllocationInfo(0, 1, &invalid).SizeInBytes == UINT64_MAX);

            invalid = desc;
            invalid.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            invalid.DepthOrArraySize = 4;
            EXPECT(FAILED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &invalid,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          __uuidof(ID3D12Resource), nullptr)));
            EXPECT(device->GetResourceAllocationInfo(0, 1, &invalid).SizeInBytes == UINT64_MAX);
        }

        // D3D-60: CheckFeatureSupport and committed resource creation for integer and typeless formats
        {
            const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
            for (DXGI_FORMAT f : {
                     DXGI_FORMAT_R32G32_TYPELESS, DXGI_FORMAT_R32G32_UINT, DXGI_FORMAT_R32G32_SINT,
                     DXGI_FORMAT_R16G16B16A16_UINT, DXGI_FORMAT_R16G16B16A16_SINT,
                     DXGI_FORMAT_R10G10B10A2_UINT, DXGI_FORMAT_R16G16_TYPELESS, DXGI_FORMAT_R16G16_UINT,
                     DXGI_FORMAT_R16G16_SINT, DXGI_FORMAT_R8G8_TYPELESS, DXGI_FORMAT_R8G8_UINT,
                     DXGI_FORMAT_R8G8_SINT, DXGI_FORMAT_B8G8R8X8_TYPELESS, DXGI_FORMAT_B8G8R8X8_UNORM,
                     DXGI_FORMAT_B8G8R8X8_UNORM_SRGB}) {
                D3D12_FEATURE_DATA_FORMAT_SUPPORT fs{f};
                EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &fs, sizeof(fs))));
                EXPECT(fs.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
                Ref<ID3D12Resource> tex;
                const D3D12_RESOURCE_DESC desc = texture_desc(16, 16, f, D3D12_RESOURCE_FLAG_NONE, 1);
                EXPECT(SUCCEEDED(device->CreateCommittedResource(
                    &default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                    IID_PPV_ARGS(&tex.p))));
                EXPECT(tex.p != nullptr);
            }
        }
        // D3D-87, B8G8R8X8: no typed UAV and no typed texture that allows unordered access (the typeless family may),
        // as the host runtime answers; render targets as for B8G8R8A8.
        {
            Ref<ID3D12Device> host;
            if (create_host_device(host)) {
                const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
                for (DXGI_FORMAT f : {DXGI_FORMAT_B8G8R8X8_UNORM, DXGI_FORMAT_B8G8R8X8_UNORM_SRGB,
                                      DXGI_FORMAT_B8G8R8X8_TYPELESS}) {
                    D3D12_FEATURE_DATA_FORMAT_SUPPORT a{f}, b{f};
                    EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &a, sizeof(a))));
                    EXPECT(SUCCEEDED(host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &b, sizeof(b))));
                    constexpr UINT typed_uav = D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
                    constexpr UINT typed_uav2 = D3D12_FORMAT_SUPPORT2_UAV_TYPED_LOAD | D3D12_FORMAT_SUPPORT2_UAV_TYPED_STORE;
                    EXPECT((UINT(a.Support1) & typed_uav) == (UINT(b.Support1) & typed_uav));
                    EXPECT((UINT(a.Support2) & typed_uav2) == (UINT(b.Support2) & typed_uav2));
                    for (auto flag : {D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET}) {
                        const D3D12_RESOURCE_DESC desc = texture_desc(16, 16, f, flag, 1);
                        Ref<ID3D12Resource> ours_resource, host_resource;
                        const HRESULT ours_hr = device->CreateCommittedResource(
                            &default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            IID_PPV_ARGS(&ours_resource.p));
                        const HRESULT host_hr = host->CreateCommittedResource(
                            &default_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                            IID_PPV_ARGS(&host_resource.p));
                        if (SUCCEEDED(ours_hr) != SUCCEEDED(host_hr))
                            std::fprintf(stderr, "B8G8R8X8 format %d flags %d: ours %08lx, host %08lx\n", int(f),
                                         int(flag), static_cast<unsigned long>(ours_hr),
                                         static_cast<unsigned long>(host_hr));
                        EXPECT(SUCCEEDED(ours_hr) == SUCCEEDED(host_hr));
                    }
                }
                std::printf("B8G8R8X8: typed UAV support and UAV/RTV resource creation match the host runtime\n");
            } else {
                std::printf("B8G8R8X8: no host D3D12 device to compare with, skipped\n");
            }
        }

        const int options_mismatches = compare_feature_options(device.p, info);
        EXPECT(options_mismatches == 0);
        // Uncompressed 8-bit sRGB formats support texture reads but cannot type a D3D12 buffer view.
        for (DXGI_FORMAT format : {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
                                   DXGI_FORMAT_B8G8R8X8_UNORM_SRGB}) {
            D3D12_FEATURE_DATA_FORMAT_SUPPORT support{format};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &support, sizeof(support))));
            EXPECT(!(support.Support1 & D3D12_FORMAT_SUPPORT1_BUFFER));
            EXPECT(support.Support1 & D3D12_FORMAT_SUPPORT1_SHADER_LOAD);
            EXPECT(support.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D);
        }
        const int format_support_mismatches = compare_format_support(device.p, info);
        EXPECT(format_support_mismatches == 0);
        {
            VkFormatProperties r4_props{};
            d.instance_fn<PFN_vkGetPhysicalDeviceFormatProperties>("vkGetPhysicalDeviceFormatProperties")(
                d.physical, VK_FORMAT_R4G4B4A4_UNORM_PACK16, &r4_props);
            const bool r4g4b4a4 = (r4_props.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) != 0;
            // The PC-ABI layer device: the proxy returns heap starts by value.
            EXPECT(new_format_round_trips(inner, info, r4g4b4a4) == 0);
        }

        // Push constants limit check (D3D-14): Vulkan spec guarantees >= 128 bytes.
        EXPECT(rt::d3d12::device_max_push_constants_size(device.p) >= 128);
        EXPECT(rt::d3d12::device_max_push_constants_size(inner) == rt::d3d12::device_max_push_constants_size(device.p));
        // The test GPU may have either answer (Intel Arc and a Radeon 680M on driver 22.40.84.06 report 128 bytes);
        // the device must flag the fallback exactly when the 144-byte block does not fit.
        EXPECT(rt::d3d12::device_push_constants_need_fallback(device.p) ==
               (rt::d3d12::device_max_push_constants_size(device.p) < 144));

        // Simulated fallback path when maxPushConstantsSize < 144 bytes:
        {
            rt::d3d12::VulkanDeviceInfo fallback_info = info;
            fallback_info.texture_compression_bc = false;
            fallback_info.max_push_constants_size = 128;
            fallback_info.push_constants_need_fallback = true;
            ID3D12Device* fallback_inner = nullptr;
            EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(fallback_info, IID_PPV_ARGS(&fallback_inner))));
            Ref<ID3D12Device> fallback_device;
            EXPECT(SUCCEEDED(rt::d3d12_xs::create_device_proxy(fallback_inner, __uuidof(ID3D12Device),
                                                               reinterpret_cast<void**>(&fallback_device.p))));
            rt::d3d12::set_device_outer(fallback_inner, fallback_device.p);
            D3D12_FEATURE_DATA_D3D12_OPTIONS8 options8{TRUE};
            EXPECT(fallback_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS8, &options8,
                                                        sizeof(options8)) == S_OK);
            EXPECT(options8.UnalignedBlockTexturesSupported == FALSE);
            EXPECT(rt::d3d12::device_max_push_constants_size(fallback_device.p) == 128);
            EXPECT(rt::d3d12::device_push_constants_need_fallback(fallback_device.p));
        }

        // The Windows 10 footprint skip depends on this classification; check it on every host.
        EXPECT(uses_unaligned_blocks(texture_desc(10, 6, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE)));
        EXPECT(!uses_unaligned_blocks(texture_desc(8, 8, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE)));
        EXPECT(!uses_unaligned_blocks(texture_desc(10, 6, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE)));

        // D3D12_OPTIONS1 (D3D-86): the lane range is the Vulkan device's own subgroup range and the device-wide lane
        // count is deliberately one wave, so those are pinned here rather than only compared. Wave ops can only be
        // promised by a GPU, and a software Vulkan device has no host GPU to compare with either.
        {
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 wave{};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &wave, sizeof(wave))));
            EXPECT(wave.WaveLaneCountMin >= 1);
            EXPECT(wave.WaveLaneCountMax >= wave.WaveLaneCountMin);
            EXPECT(wave.TotalLaneCount == wave.WaveLaneCountMax);
            EXPECT(wave.ExpandedComputeResourceStates);
            const int options1 = compare_options1(device.p, d);
            if (options1 == -2)
                std::printf("D3D12_OPTIONS1: software Vulkan device, host comparison skipped\n");
            else if (options1 < 0)
                std::printf("D3D12_OPTIONS1: no host D3D12 device to compare with, skipped\n");
            EXPECT(options1 <= 0);
            if (options1 != -2) EXPECT(wave.WaveOps);
            D3D12_FEATURE_DATA_D3D12_OPTIONS1 shrunk{};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, &shrunk, sizeof(shrunk) - 4)));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, nullptr, sizeof(shrunk))));
        }
        const int footprint_mismatches = compare_footprints(device.p);
        if (footprint_mismatches < 0)
            std::printf("GetCopyableFootprints: no host D3D12 device to compare with, skipped\n");
        EXPECT(footprint_mismatches <= 0);
        {
            // D3D-122: the same refusals without a host device (CI's llvmpipe), so they can't regress unseen.
            const auto cases = footprint_validity_cases();
            for (size_t i = 0; i < cases.size(); ++i) {
                UINT64 total = 0;
                device->GetCopyableFootprints(&cases[i].first, 0, 1, 0, nullptr, nullptr, nullptr, &total);
                if ((total != UINT64_MAX) != cases[i].second)
                    std::fprintf(stderr, "footprint validity case %zu: total %llu, expected %s\n", i,
                                 (unsigned long long)total, cases[i].second ? "a footprint" : "the failure answer");
                EXPECT((total != UINT64_MAX) == cases[i].second);
            }
        }

        // Shader model and root signature (D3D-86): the answers are only useful if the game can rely on them, so
        // pin both the ceiling and the refusal cases rather than just comparing with the host.
        {
            D3D12_FEATURE_DATA_SHADER_MODEL model{D3D_SHADER_MODEL_6_0};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &model, sizeof(model))));
            EXPECT(model.HighestShaderModel == D3D_SHADER_MODEL_6_0);
            // 5.1 is D3D12's baseline and the host answers it, so a caller asking for the minimum must get it back.
            D3D12_FEATURE_DATA_SHADER_MODEL baseline{D3D_SHADER_MODEL_5_1};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &baseline, sizeof(baseline))));
            EXPECT(baseline.HighestShaderModel == D3D_SHADER_MODEL_5_1);
            D3D12_FEATURE_DATA_SHADER_MODEL above{static_cast<D3D_SHADER_MODEL>(0x69)};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &above, sizeof(above))));
            EXPECT(above.HighestShaderModel == D3D_SHADER_MODEL_6_0);  // clamped to what the layer accepts
            D3D12_FEATURE_DATA_SHADER_MODEL below{static_cast<D3D_SHADER_MODEL>(0)};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &below, sizeof(below))));
            // Between 5.1 and 6.0 no runtime names a model: refused (the host's E_INVALIDARG), not echoed back.
            for (const UINT undefined : {0x52u, 0x5au, 0x5fu}) {
                D3D12_FEATURE_DATA_SHADER_MODEL gap{static_cast<D3D_SHADER_MODEL>(undefined)};
                EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &gap, sizeof(gap)) == E_INVALIDARG);
            }

            D3D12_FEATURE_DATA_ROOT_SIGNATURE signature{D3D_ROOT_SIGNATURE_VERSION_1_1};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &signature, sizeof(signature))));
            EXPECT(signature.HighestVersion == D3D_ROOT_SIGNATURE_VERSION_1_1);
            D3D12_FEATURE_DATA_ROOT_SIGNATURE newer{static_cast<D3D_ROOT_SIGNATURE_VERSION>(0x3)};
            EXPECT(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &newer, sizeof(newer))));
            EXPECT(newer.HighestVersion == D3D_ROOT_SIGNATURE_VERSION_1_1);
            D3D12_FEATURE_DATA_ROOT_SIGNATURE undefined{static_cast<D3D_ROOT_SIGNATURE_VERSION>(0)};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &undefined, sizeof(undefined))));
            // Anything above 1.2 is undefined: E_INVALIDARG with the value left alone, measured on an RTX 4070 Laptop
            // (621.24) and WARP. Checked here rather than against the host, which has crashed on undefined values of
            // the shader model query next to this one.
            for (const UINT undefined_version : {0x4u, 0x5u, 0x10u, 0xffu, 0x7fffffffu, 0x80000000u, 0xffffffffu}) {
                D3D12_FEATURE_DATA_ROOT_SIGNATURE refused{static_cast<D3D_ROOT_SIGNATURE_VERSION>(undefined_version)};
                EXPECT(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &refused, sizeof(refused)) ==
                       E_INVALIDARG);
                EXPECT(static_cast<UINT>(refused.HighestVersion) == undefined_version);
            }
            // A wrong DataSize and a null buffer are refused for both queries.
            D3D12_FEATURE_DATA_SHADER_MODEL short_model{D3D_SHADER_MODEL_6_0};
            D3D12_FEATURE_DATA_ROOT_SIGNATURE short_signature{D3D_ROOT_SIGNATURE_VERSION_1_1};
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &short_model,
                                                      sizeof(short_model) - 1)));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, &short_signature,
                                                      sizeof(short_signature) - 1)));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, nullptr, sizeof(short_model))));
            EXPECT(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_ROOT_SIGNATURE, nullptr, sizeof(short_signature))));
        }
        const int feature_mismatches = compare_feature_support(device.p);
        if (feature_mismatches < 0)
            std::printf("CheckFeatureSupport: no host D3D12 device to compare with, skipped\n");
        EXPECT(feature_mismatches <= 0);
        const int allocation_mismatches = compare_allocation_info(device.p);
        if (allocation_mismatches < 0)
            std::printf("GetResourceAllocationInfo: no host D3D12 device to compare with, skipped\n");
        EXPECT(allocation_mismatches <= 0);
        const int rule_mismatches = compare_command_list_rules(device.p);
        if (rule_mismatches < 0) std::printf("command-list rules: no host D3D12 device to compare with, skipped\n");
        EXPECT(rule_mismatches <= 0);

        // D3D-46: copies against the host runtime (d3d12_copy_compare.h).
        const int copy_mismatches = copycmp::compare_copies(device.p, d.bc);
        if (copy_mismatches < 0) std::printf("Copies: no host D3D12 device to compare with, skipped\n");
        EXPECT(copy_mismatches <= 0);
        const int depth24_mismatches = copycmp::compare_depth24(device.p, d.depth_format_usable(VK_FORMAT_D24_UNORM_S8_UINT));
        if (depth24_mismatches < 0) std::printf("D24 clears: no host D3D12 device to compare with, skipped\n");
        EXPECT(depth24_mismatches <= 0);
        const int tiling_mismatches = compare_tiling(device.p);
        if (tiling_mismatches < 0)
            std::printf("GetResourceTiling: no host D3D12 device with tiled resources, or no layout it accepts; "
                        "skipped\n");
        EXPECT(tiling_mismatches <= 0);
        const int argument_mismatches = compare_invalid_arguments(device.p);
        if (argument_mismatches < 0)
            std::printf("invalid arguments: no host D3D12 device to compare with, skipped\n");
        EXPECT(argument_mismatches <= 0);
        const int feature_error_mismatches = compare_check_feature_support(device.p);
        if (feature_error_mismatches < 0)
            std::printf("CheckFeatureSupport error paths: no host D3D12 device to compare with, skipped\n");
        EXPECT(feature_error_mismatches <= 0);
        const int fence_mismatches = compare_fences(device.p);
        if (fence_mismatches < 0)
            std::printf("Fences: no host D3D12 device to compare with, skipped\n");
        EXPECT(fence_mismatches <= 0);

        // D3D-236: CreateCommandQueue with D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME requires
        // SeIncreaseBasePriorityPrivilege enabled in the token, matching the Windows D3D12 runtime.
        {
            const D3D12_COMMAND_QUEUE_DESC rt_desc{
                D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME,
                D3D12_COMMAND_QUEUE_FLAG_NONE, 0};
            struct RestoreOverride {
                ~RestoreOverride() { rt::d3d12::set_global_realtime_privilege_override_for_test(-1); }
            } restore_override;

            rt::d3d12::set_global_realtime_privilege_override_for_test(0);
            EXPECT(!rt::d3d12::has_global_realtime_privilege());
            Ref<ID3D12CommandQueue> denied_q;
            EXPECT(device->CreateCommandQueue(&rt_desc, IID_PPV_ARGS(&denied_q.p)) == DXGI_ERROR_ACCESS_DENIED);
            EXPECT(!denied_q.p);
            EXPECT(device->CreateCommandQueue(&rt_desc, __uuidof(ID3D12CommandQueue), nullptr) ==
                   DXGI_ERROR_ACCESS_DENIED);

            rt::d3d12::set_global_realtime_privilege_override_for_test(1);
            EXPECT(rt::d3d12::has_global_realtime_privilege());
            EXPECT(device->CreateCommandQueue(&rt_desc, __uuidof(ID3D12CommandQueue), nullptr) == E_UNEXPECTED);
            Ref<ID3D12CommandQueue> granted_q;
            EXPECT(SUCCEEDED(device->CreateCommandQueue(&rt_desc, IID_PPV_ARGS(&granted_q.p))) && granted_q.p);
            if (granted_q.p) {
                EXPECT(granted_q->GetDesc().Priority == D3D12_COMMAND_QUEUE_PRIORITY_GLOBAL_REALTIME);
            }
            granted_q.reset();
            rt::d3d12::set_global_realtime_privilege_override_for_test(-1);

            Ref<ID3D12Device> host;
            create_host_device(host);
            if (host.p) EXPECT(command_queue(device.p, rt_desc) == command_queue(host.p, rt_desc));

            // SecurityAnonymous thread impersonation (OpenThreadToken fails with ERROR_CANT_OPEN_ANONYMOUS, not
            // ERROR_NO_TOKEN) must fail closed instead of falling back to the process token, matching the host runtime.
            const auto check_security_anonymous_impersonation = [&]() -> int {
                HANDLE proc_token = nullptr;
                if (!OpenProcessToken(GetCurrentProcess(), TOKEN_DUPLICATE, &proc_token)) return 0;
                HANDLE anon_imp = nullptr;
                if (DuplicateTokenEx(proc_token, MAXIMUM_ALLOWED, nullptr, SecurityAnonymous, TokenImpersonation,
                                     &anon_imp)) {
                    if (SetThreadToken(nullptr, anon_imp)) {
                        struct RevertImpersonation {
                            ~RevertImpersonation() { RevertToSelf(); }
                        } revert_impersonation;
                        HANDLE opened = nullptr;
                        EXPECT(!OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &opened));
                        EXPECT(GetLastError() == ERROR_CANT_OPEN_ANONYMOUS);
                        if (opened) CloseHandle(opened);
                        EXPECT(!rt::d3d12::has_global_realtime_privilege());
                        EXPECT(command_queue(device.p, rt_desc) == DXGI_ERROR_ACCESS_DENIED);
                        if (host.p) EXPECT(command_queue(host.p, rt_desc) == DXGI_ERROR_ACCESS_DENIED);
                    }
                    CloseHandle(anon_imp);
                }
                CloseHandle(proc_token);
                return 0;
            };
            if (check_security_anonymous_impersonation() != 0) return 1;

            // When the process token holds SeIncreaseBasePriorityPrivilege (e.g. an elevated session), toggle the
            // real token privilege and verify both states (plus SecurityAnonymous impersonation while the process
            // token has the privilege enabled) against the host D3D12 runtime.
            HANDLE token = nullptr;
            LUID luid{};
            if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &token)) {
                if (LookupPrivilegeValueW(nullptr, L"SeIncreaseBasePriorityPrivilege", &luid)) {
                    TOKEN_PRIVILEGES tp{}, prev{};
                    DWORD prev_len = 0;
                    tp.PrivilegeCount = 1;
                    tp.Privileges[0].Luid = luid;
                    tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                    if (AdjustTokenPrivileges(token, FALSE, &tp, sizeof(prev), &prev, &prev_len) &&
                        GetLastError() == ERROR_SUCCESS) {
                        EXPECT(rt::d3d12::has_global_realtime_privilege());
                        EXPECT(command_queue(device.p, rt_desc) == S_OK);
                        if (host.p) EXPECT(command_queue(host.p, rt_desc) == S_OK);
                        if (check_security_anonymous_impersonation() != 0) {
                            AdjustTokenPrivileges(token, FALSE, &prev, 0, nullptr, nullptr);
                            CloseHandle(token);
                            return 1;
                        }
                        tp.Privileges[0].Attributes = 0;
                        if (AdjustTokenPrivileges(token, FALSE, &tp, 0, nullptr, nullptr) &&
                            GetLastError() == ERROR_SUCCESS) {
                            EXPECT(!rt::d3d12::has_global_realtime_privilege());
                            EXPECT(command_queue(device.p, rt_desc) == DXGI_ERROR_ACCESS_DENIED);
                            if (host.p) EXPECT(command_queue(host.p, rt_desc) == DXGI_ERROR_ACCESS_DENIED);
                        }
                        AdjustTokenPrivileges(token, FALSE, &prev, 0, nullptr, nullptr);
                    }
                }
                CloseHandle(token);
            }
        }

        const int depth_quality_mismatches = compare_depth_multisample_quality_levels(device.p, depth_sample_counts);
        if (depth_quality_mismatches < 0)
            std::printf("Depth/stencil MSAA quality queries: no host D3D12 device to compare with, skipped\n");
        EXPECT(depth_quality_mismatches <= 0);

        EXPECT(check_rgb9e5_resolves(device.p, d));
        const int resolve_mismatches = compare_resolves(device.p, color_sample_counts);
        if (resolve_mismatches < 0)
            std::printf("Resolve comparison: no host D3D12 device to compare with, skipped\n");
        EXPECT(resolve_mismatches <= 0);

        // Command objects; their GetDevice is the proxy, never the inner device.
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(device->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator.p))));
        EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                   IID_PPV_ARGS(&list.p))));
        EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        // Every wait on `fence` goes through this: a hand-picked value that is reused or lower than an earlier one
        // makes SetEventOnCompletion return at once, and the block then reads results the GPU hasn't written.
        UINT64 fence_last = 0;
        const auto submit_and_wait = [&] {
            const UINT64 v = ++fence_last;
            return SUCCEEDED(queue->Signal(fence.p, v)) && SUCCEEDED(fence->SetEventOnCompletion(v, nullptr)) &&
                   fence->GetCompletedValue() >= v;
        };
        {
            Ref<ID3D12Device> parent;
            EXPECT(SUCCEEDED(queue->GetDevice(IID_PPV_ARGS(&parent.p))) && parent.p == device.p);
        }
        // Residency and power (D3D-33): no-ops on Vulkan that still check their arguments, through the proxy's
        // forwarded slots. EnqueueMakeResident must signal its fence, or a queue waiting on it would hang.
        {
            Ref<ID3D12Fence> residency;
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&residency.p))));
            ID3D12Pageable* const pageables[] = {fence.p, allocator.p};
            EXPECT(device->MakeResident(2, pageables) == S_OK);
            EXPECT(device->MakeResident(0, nullptr) == S_OK);
            EXPECT(device->MakeResident(1, nullptr) == E_INVALIDARG);
            EXPECT(device->Evict(2, pageables) == S_OK);
            EXPECT(device->Evict(0, nullptr) == S_OK);
            EXPECT(device->Evict(1, nullptr) == E_INVALIDARG);
            EXPECT(device->SetStablePowerState(TRUE) == S_OK);
            EXPECT(device->SetStablePowerState(FALSE) == S_OK);

            Ref<ID3D12Device1> device1;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device1.p))));
            const D3D12_RESIDENCY_PRIORITY priorities[] = {D3D12_RESIDENCY_PRIORITY_MAXIMUM,
                                                           D3D12_RESIDENCY_PRIORITY(D3D12_RESIDENCY_PRIORITY_LOW + 5)};
            // Only committed resources and heaps take a priority, as on the host runtime (d3d12_residency_test): a
            // fence and an allocator don't.
            EXPECT(device1->SetResidencyPriority(2, pageables, priorities) == E_INVALIDARG);
            EXPECT(device1->SetResidencyPriority(0, nullptr, nullptr) == S_OK);
            EXPECT(device1->SetResidencyPriority(1, pageables, nullptr) == E_INVALIDARG);
            EXPECT(device1->SetResidencyPriority(1, nullptr, priorities) == E_INVALIDARG);

            Ref<ID3D12Device3> device3;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device3.p))));
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 2, pageables, residency.p, 7) == S_OK);
            EXPECT(residency->GetCompletedValue() == 7);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_DENY_OVERBUDGET, 0, nullptr, residency.p, 9) ==
                   S_OK);
            EXPECT(residency->GetCompletedValue() == 9);
            // A null fence is accepted (the host runtime's answer) and signals nothing; refused calls leave the fence
            // alone.
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, pageables, nullptr, 11) == S_OK);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAG_NONE, 1, nullptr, residency.p, 11) ==
                   E_INVALIDARG);
            EXPECT(device3->EnqueueMakeResident(D3D12_RESIDENCY_FLAGS(0x2), 1, pageables, residency.p, 11) ==
                   E_INVALIDARG);
            EXPECT(residency->GetCompletedValue() == 9);
            // The GPU sees the signal: a queue waits on it, then signals past it.
            EXPECT(SUCCEEDED(queue->Wait(residency.p, 9)));
            EXPECT(SUCCEEDED(queue->Signal(residency.p, 10)));
            EXPECT(SUCCEEDED(residency->SetEventOnCompletion(10, nullptr)) && residency->GetCompletedValue() == 10);

            Ref<ID3D12Device6> device6;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device6.p))));
            const HANDLE done = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            EXPECT(done);
            BOOL further = TRUE;
            const HRESULT background = device6->SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK, D3D12_MEASUREMENTS_ACTION_COMMIT_RESULTS,
                done, &further);
            const DWORD signalled = WaitForSingleObject(done, 0);
            CloseHandle(done);
            EXPECT(background == S_OK && signalled == WAIT_OBJECT_0 && further == FALSE);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
                                                        D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr) == S_OK);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE(4),
                                                        D3D12_MEASUREMENTS_ACTION_KEEP_ALL, nullptr, nullptr) ==
                   E_INVALIDARG);
            EXPECT(device6->SetBackgroundProcessingMode(D3D12_BACKGROUND_PROCESSING_MODE_ALLOWED,
                                                        D3D12_MEASUREMENTS_ACTION(4), nullptr, nullptr) ==
                   E_INVALIDARG);
        }
        {
            Ref<ID3D12Device5> device5;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device5.p))));
            UINT meta_count = 7;
            EXPECT(device5->EnumerateMetaCommands(&meta_count, nullptr) == S_OK && meta_count == 0);
            EXPECT(device5->EnumerateMetaCommands(nullptr, nullptr) == E_INVALIDARG);
            const GUID unknown_command{0x12345678, 0x1234, 0x1234, {1, 2, 3, 4, 5, 6, 7, 8}};
            UINT total_size = 0, param_count = 0;
            EXPECT(device5->EnumerateMetaCommandParameters(unknown_command, D3D12_META_COMMAND_PARAMETER_STAGE_CREATION,
                                                           &total_size, &param_count, nullptr) == E_INVALIDARG);
            void* meta = reinterpret_cast<void*>(1);
            EXPECT(device5->CreateMetaCommand(unknown_command, 0, nullptr, 0, IID_ID3D12MetaCommand, &meta) ==
                       E_INVALIDARG &&
                   meta == nullptr);
            D3D12_SERIALIZED_DATA_DRIVER_MATCHING_IDENTIFIER identifier{};
            EXPECT(device5->CheckDriverMatchingIdentifier(D3D12_SERIALIZED_DATA_RAYTRACING_ACCELERATION_STRUCTURE,
                                                          &identifier) == D3D12_DRIVER_MATCHING_IDENTIFIER_UNRECOGNIZED);
            EXPECT(device5->CheckDriverMatchingIdentifier(D3D12_SERIALIZED_DATA_TYPE(1), &identifier) ==
                   D3D12_DRIVER_MATCHING_IDENTIFIER_UNSUPPORTED_TYPE);
        }
        {
            Ref<ID3D12Device4> device4;
            Ref<ID3D12GraphicsCommandList> closed;
            EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device4.p))));
            EXPECT(SUCCEEDED(device4->CreateCommandList1(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                         D3D12_COMMAND_LIST_FLAG_NONE, IID_PPV_ARGS(&closed.p))));
            EXPECT(closed->Close() == E_FAIL);  // CreateCommandList1 returns a closed list
        }

        // Descriptor heaps from slot 14, start handles in the Xbox shape.
        D3D12_DESCRIPTOR_HEAP_DESC heap_desc{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 3, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        Ref<ID3D12DescriptorHeap> rtv_heap, dsv_heap;
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        heap_desc = {D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
        EXPECT(SUCCEEDED(device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&dsv_heap.p))));
        EXPECT(rt::d3d12_xs::is_heap_proxy(rtv_heap.p) && rt::d3d12_xs::is_heap_proxy(dsv_heap.p));
        const std::uint64_t rtv_start = reinterpret_cast<XboxHeapHandleFn>(slot(rtv_heap.p, 9))(rtv_heap.p);
        const std::uint64_t dsv_start = reinterpret_cast<XboxHeapHandleFn>(slot(dsv_heap.p, 9))(dsv_heap.p);
        const UINT increment = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        EXPECT(rtv_start && dsv_start && increment >= sizeof(leonida::TargetView));
        const D3D12_CPU_DESCRIPTOR_HANDLE rtv0{SIZE_T(rtv_start)}, rtv1{SIZE_T(rtv_start + increment)},
            rtv565{SIZE_T(rtv_start + 2 * increment)}, dsv0{SIZE_T(dsv_start)};

        // Resources: the render target through slot 76 with the Xbox tiled layout word, the others through the
        // PC CreateCommittedResource.
        const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
        const D3D12_HEAP_PROPERTIES upload_heap{D3D12_HEAP_TYPE_UPLOAD};
        const D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const auto create = [&](const D3D12_HEAP_PROPERTIES& heap, const D3D12_RESOURCE_DESC& desc,
                                D3D12_RESOURCE_STATES state, Ref<ID3D12Resource>& out) {
            return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, nullptr,
                                                   IID_PPV_ARGS(&out.p));
        };
        Ref<ID3D12Resource> refused_buffer;
        D3D12_RESOURCE_DESC invalid_buffer_samples = buffer_desc(256);
        invalid_buffer_samples.SampleDesc.Count = 0;
        EXPECT(create(default_heap, invalid_buffer_samples, D3D12_RESOURCE_STATE_COPY_DEST, refused_buffer) ==
               E_INVALIDARG && !refused_buffer.p);
        invalid_buffer_samples.SampleDesc.Count = 1;
        invalid_buffer_samples.SampleDesc.Quality = 1;
        EXPECT(create(default_heap, invalid_buffer_samples, D3D12_RESOURCE_STATE_COPY_DEST, refused_buffer) ==
               E_INVALIDARG && !refused_buffer.p);

        Ref<ID3D12Resource> rt_resource, packed565, ds, untouched, upload, readback, volume, line, line_array,
            line_refused, line_target;
        D3D12_RESOURCE_DESC xbox_rt =
            texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        xbox_rt.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);
        const IID resource_iid = __uuidof(ID3D12Resource);
        EXPECT(SUCCEEDED(reinterpret_cast<XboxCreateResourceFn>(slot(device.p, 76))(
            device.p, &default_heap, D3D12_HEAP_FLAG_NONE, &xbox_rt, D3D12_RESOURCE_STATE_COMMON, nullptr,
            &resource_iid, reinterpret_cast<void**>(&rt_resource.p))));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(size, size, DXGI_FORMAT_B5G6R5_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                D3D12_RESOURCE_STATE_RENDER_TARGET, packed565)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(size, size, DXGI_FORMAT_D32_FLOAT_S8X24_UINT,
                                             D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                                D3D12_RESOURCE_STATE_DEPTH_WRITE, ds)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(small_size, small_size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COPY_SOURCE, untouched)));
        EXPECT(SUCCEEDED(create(upload_heap, buffer_desc(65536), D3D12_RESOURCE_STATE_GENERIC_READ, upload)));
        EXPECT(SUCCEEDED(create(readback_heap, buffer_desc(65536), D3D12_RESOURCE_STATE_COPY_DEST, readback)));
        EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, volume)));
        EXPECT(volume->GetDesc().MipLevels == 5);
        const auto line_shape = line_desc(size, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
        const auto line_array_shape = line_desc(small_size, 1, 3, DXGI_FORMAT_R8G8B8A8_UNORM);
        EXPECT(SUCCEEDED(create(default_heap, line_shape, D3D12_RESOURCE_STATE_COPY_DEST, line)));
        EXPECT(SUCCEEDED(create(default_heap, line_array_shape, D3D12_RESOURCE_STATE_COPY_DEST, line_array)));
        EXPECT(line->GetDesc().Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE1D && line->GetDesc().Height == 1);
        EXPECT(line_array->GetDesc().DepthOrArraySize == 3);
        EXPECT(rt::d3d12::vulkan_image(line.p) && rt::d3d12::vulkan_image(line_array.p));
        auto refused_line = line_shape;
        refused_line.Height = 2;
        EXPECT(create(default_heap, refused_line, D3D12_RESOURCE_STATE_COMMON, line_refused) == E_INVALIDARG &&
               !line_refused.p);
        // A 1D render target is created, as the PC runtime does (d3d12_texture1d_target_test compares them).
        auto line_target_shape = line_shape;
        line_target_shape.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        EXPECT(SUCCEEDED(create(default_heap, line_target_shape, D3D12_RESOURCE_STATE_RENDER_TARGET, line_target)) &&
               rt::d3d12::vulkan_image(line_target.p));
        EXPECT(rt_resource->GetDesc().Width == size && rt_resource->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_UNKNOWN);
        EXPECT(rt::d3d12::vulkan_image(rt_resource.p) && !rt::d3d12::vulkan_image(upload.p));

        // Upload buffers are mapped and zeroed.
        std::uint8_t* up = nullptr;
        std::uint8_t* down = nullptr;
        EXPECT(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&up))) && up);
        EXPECT(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&down))) && down);
        bool zeroed = true;
        for (int i = 0; i < 65536; ++i) zeroed = zeroed && up[i] == 0;
        EXPECT(zeroed);
        D3D12_HEAP_PROPERTIES heap_properties{};
        EXPECT(SUCCEEDED(upload->GetHeapProperties(&heap_properties, nullptr)) &&
               heap_properties.Type == D3D12_HEAP_TYPE_UPLOAD);
        {
            // D3D-469: the flag stays on the heap, next to the DENY flags a committed buffer gets (0x4c0 on the host).
            const D3D12_HEAP_PROPERTIES atomics_heap{D3D12_HEAP_TYPE_DEFAULT};
            auto atomics_desc = buffer_desc(256);
            atomics_desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
            Ref<ID3D12Resource> atomics;
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&atomics_heap, D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS,
                                                             &atomics_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&atomics.p))));
            D3D12_HEAP_PROPERTIES atomics_got{};
            D3D12_HEAP_FLAGS atomics_flags = D3D12_HEAP_FLAG_NONE;
            EXPECT(SUCCEEDED(atomics->GetHeapProperties(&atomics_got, &atomics_flags)));
            EXPECT(UINT(atomics_flags) == (UINT(D3D12_HEAP_FLAG_ALLOW_SHADER_ATOMICS) |
                                           UINT(D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS)));
        }

        // D3D-96: upload buffers are in VRAM exactly when the GPU has Resizable BAR (and LEONIDA_UPLOAD_SYSMEM is not
        // set), host-visible either way. The copies below read this buffer on the GPU, so both placements are checked.
        {
            VkPhysicalDeviceMemoryProperties memory{};
            d.instance_fn<PFN_vkGetPhysicalDeviceMemoryProperties>("vkGetPhysicalDeviceMemoryProperties")(d.physical,
                                                                                                          &memory);
            char text[4] = {};
            size_t length = 0;
            const bool sysmem = getenv_s(&length, text, "LEONIDA_UPLOAD_SYSMEM") == 0 && length > 1 && text[0] != '0';
            const bool in_vram = rt::d3d12::device_upload_in_vram(device.p);
            EXPECT(in_vram == (leonida::resizable_bar(memory) && !sysmem));
            const VkMemoryPropertyFlags upload_flags = rt::d3d12::vulkan_memory_flags(upload.p);
            EXPECT((upload_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) &&
                   (upload_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
            if (in_vram) EXPECT(upload_flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            EXPECT(rt::d3d12::vulkan_memory_flags(readback.p) & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
            std::printf("upload buffer memory: %s\n", in_vram ? "VRAM (Resizable BAR)" : "system memory");
        }

        // Views into the heaps.
        device->CreateRenderTargetView(rt_resource.p, nullptr, rtv0);
        D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
        rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device->CreateRenderTargetView(rt_resource.p, &rtv_desc, rtv1);
        device->CreateDepthStencilView(ds.p, nullptr, dsv0);
        device->CreateRenderTargetView(packed565.p, nullptr, rtv565);

        // List 1: clears and barriers on never-written resources (their first use zero-fills them), then copies of
        // the colour, depth plane, stencil plane and the untouched texture into the readback buffer.
        UINT64 cursor = 0;
        const auto rt_place = place(device.p, rt_resource.p, 0, cursor);
        const auto packed_place = place(device.p, packed565.p, 0, cursor);
        const auto depth_place = place(device.p, ds.p, 0, cursor);
        const auto stencil_place = place(device.p, ds.p, 1, cursor);
        const auto untouched_place = place(device.p, untouched.p, 0, cursor);
        EXPECT(rt_place.Footprint.RowPitch == 256 && depth_place.Footprint.Format == DXGI_FORMAT_R32_TYPELESS &&
               stencil_place.Footprint.Format == DXGI_FORMAT_R8_TYPELESS);
        std::memset(down, 0xcd, 65536);

        D3D12_RESOURCE_BARRIER barrier =
            transition(rt_resource.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &barrier);
        const FLOAT red[4] = {1, 0, 0, 1}, blue[4] = {0, 0, 1, 1};
        const D3D12_RECT square{2, 2, 6, 6}, top_half{0, 0, 8, 4};
        list->ClearRenderTargetView(rtv0, red, 0, nullptr);
        list->ClearRenderTargetView(rtv1, blue, 1, &square);
        const FLOAT packed_full[4] = {1, 0.25f, 0, 1}, packed_rect[4] = {0, 0.75f, 1, 0};
        list->ClearRenderTargetView(rtv565, packed_full, 0, nullptr);
        list->ClearRenderTargetView(rtv565, packed_rect, 1, &square);
        barrier = transition(packed565.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        list->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 0.25f, 0x42, 1, &top_half);
        barrier = transition(rt_resource.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        barrier = transition(ds.p, D3D12_RESOURCE_STATE_DEPTH_WRITE, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        Ref<ID3D12GraphicsCommandList7> list7;
        EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list7.p))));
        D3D12_TEXTURE_BARRIER readable{};  // an enhanced barrier that only makes the untouched texture readable
        readable.SyncBefore = D3D12_BARRIER_SYNC_NONE;
        readable.SyncAfter = D3D12_BARRIER_SYNC_COPY;
        readable.AccessBefore = D3D12_BARRIER_ACCESS_NO_ACCESS;
        readable.AccessAfter = D3D12_BARRIER_ACCESS_COPY_SOURCE;
        readable.LayoutBefore = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        readable.LayoutAfter = D3D12_BARRIER_LAYOUT_COPY_SOURCE;
        readable.pResource = untouched.p;
        readable.Subresources = {0xffffffff, 0, 0, 0, 0, 0};
        D3D12_BARRIER_GROUP group{};
        group.Type = D3D12_BARRIER_TYPE_TEXTURE;
        group.NumBarriers = 1;
        group.pTextureBarriers = &readable;
        list7->Barrier(1, &group);
        for (const auto& [resource, index, placed] :
             {std::tuple{rt_resource.p, 0u, rt_place}, std::tuple{packed565.p, 0u, packed_place},
              std::tuple{ds.p, 0u, depth_place},
              std::tuple{ds.p, 1u, stencil_place}, std::tuple{untouched.p, 0u, untouched_place}}) {
            const auto dst = footprint(readback.p, placed);
            const auto src = subresource(resource, index);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        EXPECT(list->Close() == S_OK);

        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(submit_and_wait());  // null event: blocks until done

        for (UINT row = 0; row < size; ++row) {
            for (UINT col = 0; col < size; ++col) {
                const bool inside = col >= 2 && col < 6 && row >= 2 && row < 6;
                const std::uint8_t want[4] = {std::uint8_t(inside ? 0 : 255), 0, std::uint8_t(inside ? 255 : 0), 255};
                report("colour texel", row * size + col,
                       down + rt_place.Offset + row * rt_place.Footprint.RowPitch + col * 4, want, 4);
                // B is in bits 0..4, G in 5..10 and R in 11..15. The green levels round to 16 and 47.
                const std::uint16_t packed_word = inside ? 0x05ff : 0xfa00;
                report("B5G6R5 texel", row * size + col,
                       down + packed_place.Offset + row * packed_place.Footprint.RowPitch + col * 2,
                       reinterpret_cast<const std::uint8_t*>(&packed_word), sizeof(packed_word));
                const float depth = row < 4 ? 0.25f : 0.0f;  // the bottom half was zero-filled, never cleared
                report("depth texel", row * size + col,
                       down + depth_place.Offset + row * depth_place.Footprint.RowPitch + col * 4,
                       reinterpret_cast<const std::uint8_t*>(&depth), 4);
                const std::uint8_t stencil = row < 4 ? 0x42 : 0;
                report("stencil texel", row * size + col,
                       down + stencil_place.Offset + row * stencil_place.Footprint.RowPitch + col, &stencil, 1);
            }
        }
        const std::uint8_t zero[4] = {};
        for (UINT row = 0; row < small_size; ++row)
            for (UINT col = 0; col < small_size; ++col)
                report("untouched texel", row * small_size + col,
                       down + untouched_place.Offset + row * untouched_place.Footprint.RowPitch + col * 4, zero, 4);

        // List 2: uploads. A pattern goes from the upload buffer into a texture, which is copied whole into a
        // texture created in COMMON (implicit promotion to COPY_DEST), and in part (a boxed region) into a smaller
        // one; a BC1 texture makes a round trip; a buffer region is copied.
        Ref<ID3D12Resource> source, whole, part, bc;
        const D3D12_RESOURCE_DESC rgba = texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_COPY_DEST, source)));
        EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_COMMON, whole)));
        EXPECT(SUCCEEDED(create(default_heap,
                                texture_desc(small_size, small_size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                D3D12_RESOURCE_STATE_COPY_DEST, part)));
        if (d.bc)
            EXPECT(SUCCEEDED(create(default_heap,
                                    texture_desc(size, size, DXGI_FORMAT_BC1_UNORM, D3D12_RESOURCE_FLAG_NONE),
                                    D3D12_RESOURCE_STATE_COPY_DEST, bc)));

        UINT64 up_cursor = 0;
        const auto pattern_place = place(device.p, source.p, 0, up_cursor);
        for (UINT row = 0; row < size; ++row)
            for (UINT col = 0; col < size; ++col)
                std::memcpy(up + pattern_place.Offset + row * pattern_place.Footprint.RowPitch + col * 4,
                            pattern(col, row).data(), 4);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT bc_up{}, bc_down{};
        if (bc.p) {
            bc_up = place(device.p, bc.p, 0, up_cursor);
            for (UINT block_row = 0; block_row < 2; ++block_row)
                for (UINT i = 0; i < 16; ++i)  // two 8-byte blocks per row
                    up[bc_up.Offset + block_row * bc_up.Footprint.RowPitch + i] = std::uint8_t(0x30 + block_row * 16 + i);
        }
        const UINT64 bytes_at = (up_cursor + 255) / 256 * 256;
        for (UINT i = 0; i < 64; ++i) up[bytes_at + i] = std::uint8_t(0xa0 + i);

        cursor = 0;
        const auto whole_place = place(device.p, whole.p, 0, cursor);
        const auto part_place = place(device.p, part.p, 0, cursor);
        if (bc.p) bc_down = place(device.p, bc.p, 0, cursor);
        const UINT64 bytes_down = (cursor + 255) / 256 * 256;
        std::memset(down, 0xcd, 65536);

        EXPECT(SUCCEEDED(allocator->Reset()));
        EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        {
            const auto dst = subresource(source.p, 0);
            const auto src = footprint(upload.p, pattern_place);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        barrier = transition(source.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        list->CopyResource(whole.p, source.p);
        {
            const auto dst = subresource(part.p, 0);
            const auto src = subresource(source.p, 0);
            const D3D12_BOX box{2, 2, 0, 6, 6, 1};
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
        }
        barrier = transition(whole.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);  // promoted
        list->ResourceBarrier(1, &barrier);
        barrier = transition(part.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        if (bc.p) {
            const auto dst = subresource(bc.p, 0);
            const auto src = footprint(upload.p, bc_up);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            barrier = transition(bc.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
        }
        for (const auto& [resource, placed] : {std::pair{whole.p, whole_place}, std::pair{part.p, part_place},
                                               std::pair{bc.p, bc_down}}) {
            if (!resource) continue;
            const auto dst = footprint(readback.p, placed);
            const auto src = subresource(resource, 0);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        list->CopyBufferRegion(readback.p, bytes_down, upload.p, bytes_at, 64);
        EXPECT(list->Close() == S_OK);
        queue->ExecuteCommandLists(1, lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(submit_and_wait());

        for (UINT row = 0; row < size; ++row)
            for (UINT col = 0; col < size; ++col)
                report("whole-copy texel", row * size + col,
                       down + whole_place.Offset + row * whole_place.Footprint.RowPitch + col * 4,
                       pattern(col, row).data(), 4);
        for (UINT row = 0; row < small_size; ++row)
            for (UINT col = 0; col < small_size; ++col)
                report("boxed-copy texel", row * small_size + col,
                       down + part_place.Offset + row * part_place.Footprint.RowPitch + col * 4,
                       pattern(col + 2, row + 2).data(), 4);
        if (bc.p)
            for (UINT block_row = 0; block_row < 2; ++block_row)
                report("BC1 block row", block_row, down + bc_down.Offset + block_row * bc_down.Footprint.RowPitch,
                       up + bc_up.Offset + block_row * bc_up.Footprint.RowPitch, 16);
        else
            std::printf("BC1 round trip: no textureCompressionBC, skipped\n");
        report("buffer copy", 0, down + bytes_down, up + bytes_at, 64);

        // Mip 1 is 4x3x8. Move a two-row box across slices through each of the three copy paths.
        // The footprint has a third row, so using the box height as the slice stride corrupts every later slice.
        {
            Ref<ID3D12Resource> clone, cropped;
            EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, clone)));
            EXPECT(SUCCEEDED(create(default_heap, volume_desc(0), D3D12_RESOURCE_STATE_COPY_DEST, cropped)));
            cursor = 0;
            const auto layout = place(device.p, volume.p, 1, cursor);
            EXPECT(layout.Footprint.Depth == 8 && layout.Footprint.Height == 3 && layout.Footprint.Width == 4);
            const auto offset = [&](UINT x, UINT y, UINT z) {
                return layout.Offset + (z * 3 + y) * layout.Footprint.RowPitch + x * 4;
            };
            for (UINT z = 0; z < 8; ++z)
                for (UINT y = 0; y < 3; ++y)
                    for (UINT x = 0; x < 4; ++x)
                        std::memcpy(up + offset(x, y, z), pattern(x, z + 8 * y).data(), 4);
            std::memset(down, 0xcd, 65536);
            const auto in = footprint(upload.p, layout), out = footprint(readback.p, layout);
            const auto v = subresource(volume.p, 1), c = subresource(clone.p, 1), target = subresource(cropped.p, 1);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_BOX upload_box{1, 1, 1, 4, 3, 5};
            list->CopyTextureRegion(&v, 0, 0, 2, &in, &upload_box);
            auto readable_volume = transition(volume.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            list->CopyResource(clone.p, volume.p);
            readable_volume = transition(clone.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            const D3D12_BOX clone_box{0, 0, 2, 3, 2, 6};
            list->CopyTextureRegion(&target, 1, 1, 3, &c, &clone_box);
            readable_volume = transition(cropped.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable_volume);
            const D3D12_BOX output_box{1, 1, 3, 4, 3, 7};
            list->CopyTextureRegion(&out, 0, 0, 1, &target, &output_box);
            EXPECT(SUCCEEDED(list->Close()));
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            const std::array<std::uint8_t, 4> untouched_bytes{0xcd, 0xcd, 0xcd, 0xcd};
            for (UINT z = 0; z < 8; ++z)
                for (UINT y = 0; y < 3; ++y)
                    for (UINT x = 0; x < 4; ++x) {
                        const auto want = (x < 3 && y < 2 && z >= 1 && z < 5)
                            ? pattern(x + 1, z + 8 * (y + 1)) : untouched_bytes;
                        report("volume texel", (z * 3 + y) * 4 + x, down + offset(x, y, z), want.data(), 4);
                    }
        }
        volume.reset();

        // A resolve: a 4x multisampled target is cleared and resolved into a single-sampled texture, which is read
        // back. First the shapes D3D12 refuses: a multisampled texture that is not a single-mip render target or
        // depth-stencil, a sample count that is not a power of two, and a multisampled volume.
        {
            Ref<ID3D12Resource> msaa, resolved, refused;
            D3D12_RESOURCE_DESC shape =
                texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 2);
            shape.SampleDesc.Count = 4;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.MipLevels = 1;
            shape.Flags = D3D12_RESOURCE_FLAG_NONE;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            shape.SampleDesc.Count = 3;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = D3D12_MAX_MULTISAMPLE_SAMPLE_COUNT * 2;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = 0;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Count = 4;
            shape.SampleDesc.Quality = 1;  // The Vulkan backend exposes only quality level 0.
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);
            shape.SampleDesc.Quality = 0;
            shape = volume_desc(1);
            shape.SampleDesc.Count = 4;
            EXPECT(create(default_heap, shape, D3D12_RESOURCE_STATE_COMMON, refused) == E_INVALIDARG && !refused.p);

            D3D12_RESOURCE_DESC samples4 =
                texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            samples4.SampleDesc.Count = 4;
            EXPECT(SUCCEEDED(create(default_heap, samples4, D3D12_RESOURCE_STATE_RENDER_TARGET, msaa)));
            EXPECT(SUCCEEDED(create(default_heap, rgba, D3D12_RESOURCE_STATE_RESOLVE_DEST, resolved)));
            EXPECT(msaa->GetDesc().SampleDesc.Count == 4);
            D3D12_RENDER_TARGET_VIEW_DESC msaa_view{};
            msaa_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            msaa_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMS;
            device->CreateRenderTargetView(msaa.p, &msaa_view, rtv1);  // the lists that used this slot have run
            cursor = 0;
            const auto resolved_place = place(device.p, resolved.p, 0, cursor);
            std::memset(down, 0xcd, 65536);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const FLOAT green[4] = {0, 1, 0, 1};
            list->ClearRenderTargetView(rtv1, green, 0, nullptr);
            barrier = transition(msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            list->ResourceBarrier(1, &barrier);
            list->ResolveSubresource(resolved.p, 0, msaa.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            barrier = transition(resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            {
                const auto dst = footprint(readback.p, resolved_place);
                const auto src = subresource(resolved.p, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            const std::uint8_t resolved_green[4] = {0, 255, 0, 255};
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col)
                    report("resolved texel", row * size + col,
                           down + resolved_place.Offset + row * resolved_place.Footprint.RowPitch + col * 4,
                           resolved_green, 4);

            // Move a cropped two-colour region into a larger red texture.
            constexpr UINT region_width = 10, region_height = 9;
            constexpr auto region_format = DXGI_FORMAT_R8G8B8A8_UNORM;
            const auto region_desc =
                texture_desc(region_width, region_height, region_format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            Ref<ID3D12Resource> region_target;
            EXPECT(SUCCEEDED(create(default_heap, region_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, region_target)));
            device->CreateRenderTargetView(region_target.p, nullptr, rtv0);
            cursor = 0;
            const auto region_place = place(device.p, region_target.p, 0, cursor);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            barrier = transition(msaa.p, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            const D3D12_RECT blue_patch{2, 3, 4, 5};
            list->ClearRenderTargetView(rtv1, blue, 1, &blue_patch);
            list->ClearRenderTargetView(rtv0, red, 0, nullptr);
            const D3D12_RESOURCE_BARRIER resolve_barriers[] = {
                transition(msaa.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
                transition(region_target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_DEST)};
            list->ResourceBarrier(2, resolve_barriers);
            D3D12_RECT region{1, 2, 5, 5};
            const auto region_copy = subresource(region_target.p, 0), region_out = footprint(readback.p, region_place);
            const auto resolve_and_readback = [&](UINT x, D3D12_RECT* rect) {
                list7->ResolveSubresourceRegion(region_target.p, 0, x, 1, msaa.p, 0, rect, region_format,
                                                D3D12_RESOLVE_MODE_AVERAGE);
                const auto to_copy = transition(region_target.p, D3D12_RESOURCE_STATE_RESOLVE_DEST,
                                                D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &to_copy);
                list->CopyTextureRegion(&region_out, 0, 0, 0, &region_copy, nullptr);
                if (list->Close() != S_OK) return false;
                queue->ExecuteCommandLists(1, lists);
                return rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success &&
                       submit_and_wait();
            };
            EXPECT(resolve_and_readback(4, &region));
            const std::uint8_t resolved_red[4] = {255, 0, 0, 255}, resolved_blue[4] = {0, 0, 255, 255};
            for (UINT row = 0; row < region_height; ++row)
                for (UINT col = 0; col < region_width; ++col) {
                    const bool in_region = row >= 1 && row < 4 && col >= 4 && col < 8;
                    const bool in_blue = row >= 2 && row < 4 && col >= 5 && col < 7;
                    report("partial resolve texel", row * region_width + col,
                           down + region_place.Offset + row * region_place.Footprint.RowPitch + col * 4,
                           in_blue ? resolved_blue : in_region ? resolved_green : resolved_red, 4);
                }

            // A null rectangle resolves the entire source at the destination offset.
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const auto to_resolve = transition(region_target.p, D3D12_RESOURCE_STATE_COPY_SOURCE,
                                               D3D12_RESOURCE_STATE_RESOLVE_DEST);
            list->ResourceBarrier(1, &to_resolve);
            EXPECT(resolve_and_readback(1, nullptr));
            for (UINT row = 0; row < region_height; ++row)
                for (UINT col = 0; col < region_width; ++col) {
                    const bool in_source = row >= 1 && col >= 1 && col < 9;
                    const bool in_blue = row >= 4 && row < 6 && col >= 3 && col < 5;
                    report("whole source at offset", row * region_width + col,
                           down + region_place.Offset + row * region_place.Footprint.RowPitch + col * 4,
                           in_blue ? resolved_blue : in_source ? resolved_green : resolved_red, 4);
                }

            D3D12_RECT out_of_source{0, 0, LONG(size + 1), 1};
            for (auto [x, rect] : {std::pair{0u, &out_of_source}, std::pair{7u, &region}}) {
                EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
                list->ResourceBarrier(1, &to_resolve);
                list7->ResolveSubresourceRegion(region_target.p, 0, x, 1, msaa.p, 0, rect, region_format,
                                                D3D12_RESOLVE_MODE_AVERAGE);
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_copy);
            }
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list7->ResolveSubresourceRegion(region_target.p, 0, 4, 1, msaa.p, 0, &region, region_format,
                                            D3D12_RESOLVE_MODE_MAX);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list7->ResolveSubresourceRegion(region_target.p, 0, 4, 1, msaa.p, 0, &region,
                                            DXGI_FORMAT_R8G8B8A8_UNORM_SRGB, D3D12_RESOLVE_MODE_AVERAGE);
            EXPECT(list->Close() == E_INVALIDARG);

            // A resolve of a resource into itself is refused when it is recorded, one from a single-sampled
            // texture when it is executed; nothing is submitted.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ResolveSubresource(resolved.p, 0, resolved.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ResolveSubresource(resolved.p, 0, source.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_copy);
        }

        // D3D-26 through COM: AtomicCopyBufferUINT/UINT64 from the upload buffer to the readback buffer, discards of a
        // texture and a buffer, and a WriteBufferImmediate whose address no resource holds, which fails at execution.
        {
            Ref<ID3D12GraphicsCommandList1> list1;
            Ref<ID3D12GraphicsCommandList2> list2;
            EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list1.p))));
            EXPECT(SUCCEEDED(list->QueryInterface(IID_PPV_ARGS(&list2.p))));
            const std::uint32_t value32 = 0x600df00d;
            const std::uint64_t value64 = 0x0123456789abcdefull;
            std::memcpy(up + 4096, &value32, 4);
            std::memcpy(up + 4104, &value64, 8);
            std::memset(down, 0xcd, 65536);
            ID3D12Resource* const dependents[] = {upload.p};
            const D3D12_SUBRESOURCE_RANGE_UINT64 ranges[] = {{0, {4096, 4112}}};
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list1->AtomicCopyBufferUINT(readback.p, 256, upload.p, 4096, 1, dependents, ranges);
            list1->AtomicCopyBufferUINT64(readback.p, 264, upload.p, 4104, 1, dependents, ranges);
            list->DiscardResource(rt_resource.p, nullptr);
            const D3D12_DISCARD_REGION first_mip{0, nullptr, 0, 1};
            list->DiscardResource(rt_resource.p, &first_mip);
            list->DiscardResource(upload.p, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            report("atomic copy UINT", 0, down + 256, reinterpret_cast<const std::uint8_t*>(&value32), 4);
            report("atomic copy UINT64", 0, down + 264, reinterpret_cast<const std::uint8_t*>(&value64), 8);
            const std::uint8_t cd[4] = {0xcd, 0xcd, 0xcd, 0xcd};
            report("next to the atomic copies", 0, down + 252, cd, 4);
            report("next to the atomic copies", 1, down + 272, cd, 4);

            // Misaligned calls are refused when recorded; count == 0 is accepted (matches Windows 11 WARP & RTX 5090).
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list1->AtomicCopyBufferUINT64(readback.p, 260, upload.p, 4104, 1, dependents, ranges);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list1->AtomicCopyBufferUINT(readback.p, 256, upload.p, 4096, 0, nullptr, nullptr);
            EXPECT(list->Close() == S_OK);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER misaligned{0x10002, 1};
            list2->WriteBufferImmediate(1, &misaligned, nullptr);
            EXPECT(list->Close() == E_INVALIDARG);
            // A region with NumSubresources = UINT_MAX is an explicit range, not a null region.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_DISCARD_REGION all_from_one{0, nullptr, 1, UINT_MAX};
            list->DiscardResource(rt_resource.p, &all_from_one);
            EXPECT(list->Close() == E_INVALIDARG);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_DISCARD_REGION all_from_zero{0, nullptr, 0, UINT_MAX};
            list->DiscardResource(rt_resource.p, &all_from_zero);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_subresource);

            // Nothing on main gives a buffer a GPU virtual address yet (GetGPUVirtualAddress, #157/#291), so an
            // immediate write has no target: the list is refused and nothing is submitted.
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER write{0x10000, 1};
            const D3D12_WRITEBUFFERIMMEDIATE_MODE mode = D3D12_WRITEBUFFERIMMEDIATE_MODE_MARKER_OUT;
            list2->WriteBufferImmediate(1, &write, &mode);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::unresolved_resource);
        }

        // A depth-stencil view of a slice range of a 4x multisampled depth array (D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY):
        // the view captures the range, a clear through it replays, and a range past the last slice is refused when
        // the list runs. D3D12 can't copy a multisampled texture to a buffer, so the clear itself isn't read back.
        {
            Ref<ID3D12Resource> ms_depth;
            Ref<ID3D12Fence> ms_fence;
            D3D12_RESOURCE_DESC ms_desc =
                texture_desc(size, size, DXGI_FORMAT_D32_FLOAT, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL, 1, 4);
            ms_desc.SampleDesc.Count = 4;
            EXPECT(SUCCEEDED(create(default_heap, ms_desc, D3D12_RESOURCE_STATE_DEPTH_WRITE, ms_depth)));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ms_fence.p))));
            D3D12_DEPTH_STENCIL_VIEW_DESC ms_view{};
            ms_view.Format = DXGI_FORMAT_D32_FLOAT;
            ms_view.ViewDimension = D3D12_DSV_DIMENSION_TEXTURE2DMSARRAY;
            ms_view.Texture2DMSArray.FirstArraySlice = 1;
            ms_view.Texture2DMSArray.ArraySize = 2;
            device->CreateDepthStencilView(ms_depth.p, &ms_view, dsv0);  // the lists that used this slot have run
            leonida::TargetView captured{};
            std::memcpy(&captured, reinterpret_cast<const void*>(dsv0.ptr), sizeof(captured));
            EXPECT(captured.resource && captured.mip == 0 && captured.first_layer == 1 && captured.layer_count == 2);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 1)) && SUCCEEDED(ms_fence->SetEventOnCompletion(1, nullptr)));

            ms_view.Texture2DMSArray.FirstArraySlice = 3;  // slices 3 and 4 of a 4-slice texture
            device->CreateDepthStencilView(ms_depth.p, &ms_view, dsv0);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->ClearDepthStencilView(dsv0, D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_subresource);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 2)) && SUCCEEDED(ms_fence->SetEventOnCompletion(2, nullptr)));
        }

        // A render-target view of a slice range of a 4x multisampled colour array (D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY):
        // all four slices are cleared red through a whole-array view, then slices 1-2 green through a two-slice view.
        // Every slice is resolved and read back, so only slices 1 and 2 may be green. A range past the last slice is
        // refused when the list runs.
        {
            Ref<ID3D12Resource> ms_colour, ms_resolved;
            Ref<ID3D12Fence> ms_fence;
            constexpr UINT16 slices = 4;
            D3D12_RESOURCE_DESC ms_desc = texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                       D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, 1, slices);
            ms_desc.SampleDesc.Count = 4;
            EXPECT(SUCCEEDED(create(default_heap, ms_desc, D3D12_RESOURCE_STATE_RENDER_TARGET, ms_colour)));
            EXPECT(SUCCEEDED(create(default_heap,
                                    texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1,
                                                 slices),
                                    D3D12_RESOURCE_STATE_RESOLVE_DEST, ms_resolved)));
            EXPECT(SUCCEEDED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&ms_fence.p))));
            D3D12_RENDER_TARGET_VIEW_DESC ms_view{};
            ms_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            ms_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY;
            ms_view.Texture2DMSArray.FirstArraySlice = 0;
            ms_view.Texture2DMSArray.ArraySize = slices;
            device->CreateRenderTargetView(ms_colour.p, &ms_view, rtv0);  // the lists that used these slots have run
            ms_view.Texture2DMSArray.FirstArraySlice = 1;
            ms_view.Texture2DMSArray.ArraySize = 2;
            device->CreateRenderTargetView(ms_colour.p, &ms_view, rtv1);
            leonida::TargetView captured{};
            std::memcpy(&captured, reinterpret_cast<const void*>(rtv1.ptr), sizeof(captured));
            EXPECT(captured.resource && captured.mip == 0 && captured.first_layer == 1 && captured.layer_count == 2);

            cursor = 0;
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT slice_place[slices];
            for (UINT s = 0; s < slices; ++s) slice_place[s] = place(device.p, ms_resolved.p, s, cursor);
            std::memset(down, 0xcd, 65536);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            const FLOAT slice_red[4] = {1, 0, 0, 1}, slice_green[4] = {0, 1, 0, 1};
            list->ClearRenderTargetView(rtv0, slice_red, 0, nullptr);
            list->ClearRenderTargetView(rtv1, slice_green, 0, nullptr);
            barrier = transition(ms_colour.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE);
            list->ResourceBarrier(1, &barrier);
            for (UINT s = 0; s < slices; ++s)
                list->ResolveSubresource(ms_resolved.p, s, ms_colour.p, s, DXGI_FORMAT_R8G8B8A8_UNORM);
            barrier = transition(ms_resolved.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            for (UINT s = 0; s < slices; ++s) {
                const auto dst = footprint(readback.p, slice_place[s]);
                const auto src = subresource(ms_resolved.p, s);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 1)) && SUCCEEDED(ms_fence->SetEventOnCompletion(1, nullptr)));
            const std::uint8_t red_texel[4] = {255, 0, 0, 255}, green_texel[4] = {0, 255, 0, 255};
            for (UINT s = 0; s < slices; ++s)
                for (UINT row = 0; row < size; ++row)
                    for (UINT col = 0; col < size; ++col)
                        report("multisample array slice texel", (s * size + row) * size + col,
                               down + slice_place[s].Offset + row * slice_place[s].Footprint.RowPitch + col * 4,
                               s == 1 || s == 2 ? green_texel : red_texel, 4);

            ms_view.Texture2DMSArray.FirstArraySlice = 3;  // slices 3 and 4 of a 4-slice texture
            device->CreateRenderTargetView(ms_colour.p, &ms_view, rtv1);
            EXPECT(SUCCEEDED(list->Reset(allocator.p, nullptr)));
            barrier = transition(ms_colour.p, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(rtv1, slice_green, 0, nullptr);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::invalid_subresource);
            EXPECT(SUCCEEDED(queue->Signal(ms_fence.p, 2)) && SUCCEEDED(ms_fence->SetEventOnCompletion(2, nullptr)));
        }

        // 1D textures and 1D arrays use the same footprint path; one array layer is enough to prove the image
        // type and the array stride. This runs after the resolve fences so it can use a higher fence value.
        {
            cursor = 0;
            const auto line_up = place(device.p, line.p, 0, cursor);
            const auto line_array_up = place(device.p, line_array.p, 1, cursor);
            EXPECT(line_up.Footprint.Width == size && line_up.Footprint.Height == 1 &&
                   line_up.Footprint.Depth == 1 && line_up.Footprint.RowPitch == 256);
            EXPECT(line_array_up.Footprint.Height == 1 && line_array_up.Footprint.Depth == 1);
            for (UINT col = 0; col < size; ++col)
                std::memcpy(up + line_up.Offset + col * 4, pattern(col, 0).data(), 4);
            for (UINT col = 0; col < small_size; ++col)
                std::memcpy(up + line_array_up.Offset + col * 4, pattern(col, 1).data(), 4);

            cursor = 0;
            const auto line_down = place(device.p, line.p, 0, cursor);
            const auto line_array_down = place(device.p, line_array.p, 1, cursor);
            std::memset(down, 0xcd, 65536);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            {
                const auto dst = subresource(line.p, 0);
                const auto src = footprint(upload.p, line_up);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            {
                const auto dst = subresource(line_array.p, 1);
                const auto src = footprint(upload.p, line_array_up);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            barrier = transition(line.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            barrier = transition(line_array.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            {
                const auto dst = footprint(readback.p, line_down);
                const auto src = subresource(line.p, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            {
                const auto dst = footprint(readback.p, line_array_down);
                const auto src = subresource(line_array.p, 1);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            for (UINT col = 0; col < size; ++col)
                report("1D texel", col, down + line_down.Offset + col * 4, pattern(col, 0).data(), 4);
            for (UINT col = 0; col < small_size; ++col)
                report("1D array texel", col, down + line_array_down.Offset + col * 4,
                       pattern(col, 1).data(), 4);
        }

        // Reserved resources (docs/d3d12_vulkan.md section 4.2): fully backed at creation, so they copy like
        // committed ones and their first use reads zero; D3D12's description rules hold; tile-mapping calls are
        // accepted and change nothing.
        {
            const auto B = D3D12_RESOURCE_DIMENSION_BUFFER, T2 = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            Ref<ID3D12Resource> tiled, tiled_buffer, xbox_tiled, refused;
            const D3D12_RESOURCE_DESC tiled_desc = reserved_desc(T2, size, size, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
            EXPECT(SUCCEEDED(device->CreateReservedResource(&tiled_desc, D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                            IID_PPV_ARGS(&tiled.p))));
            EXPECT(tiled->GetDesc().Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE);
            D3D12_HEAP_PROPERTIES reserved_heap{};
            EXPECT(tiled->GetHeapProperties(&reserved_heap, nullptr) == E_INVALIDARG);  // no heap of its own
            // A null output is E_UNEXPECTED, as on the host runtime (compare_placed_reserved, "reserved:").
            EXPECT(device->CreateReservedResource(&tiled_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                  __uuidof(ID3D12Resource), nullptr) == E_UNEXPECTED);
            const auto refuses = [&](D3D12_RESOURCE_DESC desc) {
                const HRESULT hr = device->CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                  IID_PPV_ARGS(&refused.p));
                return hr == E_INVALIDARG && !refused.p;
            };
            auto bad = tiled_desc;
            bad.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;  // textures need a 64 KiB swizzle layout
            EXPECT(refuses(bad));
            bad.Layout = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE;  // StandardSwizzle64KBSupported is FALSE
            EXPECT(refuses(bad));
            EXPECT(refuses(reserved_desc(T2, 64, 64, 1, 1, DXGI_FORMAT_R32G32B32_FLOAT)));  // 12-byte texels: no tiles
            bad = reserved_desc(B, 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN);
            bad.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;  // buffers are ROW_MAJOR
            EXPECT(refuses(bad));
            const D3D12_RESOURCE_DESC tiled_buffer_desc = reserved_desc(B, 2 * 65536, 1, 1, 1, DXGI_FORMAT_UNKNOWN);
            EXPECT(SUCCEEDED(device->CreateReservedResource(&tiled_buffer_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&tiled_buffer.p))));
            auto xbox_desc = tiled_desc;
            xbox_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);  // the Xbox tiled layout word
            EXPECT(SUCCEEDED(device->CreateReservedResource(&xbox_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                            IID_PPV_ARGS(&xbox_tiled.p))));
            {
                Ref<ID3D12Device4> device4;
                Ref<ID3D12Device10> device10;
                Ref<ID3D12Resource> one, two;
                EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device4.p))));
                EXPECT(SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&device10.p))));
                EXPECT(SUCCEEDED(device4->CreateReservedResource1(&tiled_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                                  nullptr, IID_PPV_ARGS(&one.p))));
                EXPECT(SUCCEEDED(device10->CreateReservedResource2(&tiled_desc, D3D12_BARRIER_LAYOUT_COMMON, nullptr,
                                                                   nullptr, 0, nullptr, IID_PPV_ARGS(&two.p))));
                UINT tiles = 0;
                device->GetResourceTiling(two.p, &tiles, nullptr, nullptr, nullptr, 0, nullptr);
                EXPECT(tiles == 1);  // 8x8 RGBA8: one packed tile
            }

            // Tiling: a resource that is not reserved has none, as on the PC runtime; a first index past the end
            // returns no subresources.
            UINT total = 7, count = 4;
            D3D12_PACKED_MIP_INFO packed{1, 1, 1, 1};
            D3D12_TILE_SHAPE shape{1, 1, 1};
            D3D12_SUBRESOURCE_TILING tilings[4]{};
            device->GetResourceTiling(upload.p, &total, &packed, &shape, &count, 0, tilings);
            EXPECT(total == 0 && count == 0 && shape.WidthInTexels == 0 && packed.NumStandardMips == 0 &&
                   packed.NumTilesForPackedMips == 0);
            count = 4;
            device->GetResourceTiling(tiled_buffer.p, &total, &packed, &shape, &count, 0, tilings);
            EXPECT(total == 2 && count == 1 && shape.WidthInTexels == 65536 && tilings[0].WidthInTiles == 2);
            count = 4;
            device->GetResourceTiling(tiled_buffer.p, nullptr, nullptr, nullptr, &count, 1, tilings);
            EXPECT(count == 0);

            // Mapping calls: accepted, logged once, and the resources keep their memory.
            D3D12_TILED_RESOURCE_COORDINATE origin{};
            D3D12_TILE_REGION_SIZE one_tile{1, FALSE, 0, 0, 0};
            const D3D12_TILE_RANGE_FLAGS unmap = D3D12_TILE_RANGE_FLAG_NULL;
            queue->UpdateTileMappings(tiled.p, 1, &origin, &one_tile, nullptr, 1, &unmap, nullptr, nullptr,
                                      D3D12_TILE_MAPPING_FLAG_NONE);
            queue->CopyTileMappings(tiled_buffer.p, &origin, tiled_buffer.p, &origin, &one_tile,
                                    D3D12_TILE_MAPPING_FLAG_NONE);

            // Upload a pattern into the reserved texture and bytes into the reserved buffer, read both back, and
            // read the never-written Xbox-layout texture, which is zero.
            UINT64 tiled_up_cursor = 0;
            const auto tiled_up = place(device.p, tiled.p, 0, tiled_up_cursor);
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col)
                    std::memcpy(up + tiled_up.Offset + row * tiled_up.Footprint.RowPitch + col * 4,
                                pattern(col, row).data(), 4);
            const UINT64 tiled_bytes_up = (tiled_up_cursor + 255) / 256 * 256;
            for (UINT i = 0; i < 256; ++i) up[tiled_bytes_up + i] = std::uint8_t(0x5a ^ i);
            UINT64 tiled_down_cursor = 0;
            const auto tiled_down = place(device.p, tiled.p, 0, tiled_down_cursor);
            const auto xbox_down = place(device.p, xbox_tiled.p, 0, tiled_down_cursor);
            const UINT64 tiled_bytes_down = (tiled_down_cursor + 255) / 256 * 256;
            std::memset(down, 0xcd, 65536);

            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            {
                const auto dst = subresource(tiled.p, 0);
                const auto src = footprint(upload.p, tiled_up);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            barrier = transition(tiled.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            barrier = transition(xbox_tiled.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            for (const auto& [resource, placed] : {std::pair{tiled.p, tiled_down}, std::pair{xbox_tiled.p, xbox_down}}) {
                const auto dst = footprint(readback.p, placed);
                const auto src = subresource(resource, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            list->CopyBufferRegion(tiled_buffer.p, 65536 + 4096, upload.p, tiled_bytes_up, 256);  // into the second tile
            barrier = transition(tiled_buffer.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            list->CopyBufferRegion(readback.p, tiled_bytes_down, tiled_buffer.p, 65536 + 4096, 256);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col) {
                    report("reserved texel", row * size + col,
                           down + tiled_down.Offset + row * tiled_down.Footprint.RowPitch + col * 4,
                           pattern(col, row).data(), 4);
                    report("unwritten reserved texel", row * size + col,
                           down + xbox_down.Offset + row * xbox_down.Footprint.RowPitch + col * 4, zero, 4);
                }
            report("reserved buffer", 0, down + tiled_bytes_down, up + tiled_bytes_up, 256);
        }

        // Several lists in one ExecuteCommandLists, then fewer. The queue keeps its prepared storage between
        // executions, so the second must replay only the list it is given. B clears blue and A red: {B, A} leaves
        // red, then {B} alone must leave blue (a stale A replayed after it would leave red).
        {
            Ref<ID3D12CommandAllocator> allocator_a, allocator_b;
            Ref<ID3D12GraphicsCommandList> list_a, list_b;
            Ref<ID3D12Resource> target;
            EXPECT(SUCCEEDED(create(default_heap,
                                    texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                 D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                    D3D12_RESOURCE_STATE_RENDER_TARGET, target)));
            device->CreateRenderTargetView(target.p, nullptr, rtv1);  // the lists that used this slot have run
            for (auto [a, l, colour] : {std::tuple{&allocator_a, &list_a, red}, std::tuple{&allocator_b, &list_b, blue}}) {
                EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&a->p))));
                EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, a->p, nullptr,
                                                           IID_PPV_ARGS(&l->p))));
                (*l)->ClearRenderTargetView(rtv1, colour, 0, nullptr);
                EXPECT((*l)->Close() == S_OK);
            }
            ID3D12CommandList* b_then_a[] = {list_b.p, list_a.p};
            queue->ExecuteCommandLists(2, b_then_a);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            ID3D12CommandList* b_only[] = {list_b.p};
            queue->ExecuteCommandLists(1, b_only);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);

            UINT64 target_cursor = 0;
            const auto target_place = place(device.p, target.p, 0, target_cursor);
            std::memset(down, 0xcd, 65536);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            barrier = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &barrier);
            {
                const auto dst = footprint(readback.p, target_place);
                const auto src = subresource(target.p, 0);
                list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
            }
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            const std::uint8_t blue_texel[4] = {0, 0, 255, 255};
            for (UINT row = 0; row < size; ++row)
                for (UINT col = 0; col < size; ++col)
                    report("second ExecuteCommandLists, fewer lists", row * size + col,
                           down + target_place.Offset + row * target_place.Footprint.RowPitch + col * 4, blue_texel, 4);
        }

        // Queries (docs/d3d12_vulkan.md section 4.6). The layer has no draws yet, so occlusion and pipeline
        // statistics count nothing and read zero; what is checked is that every query is reset and run, that
        // results land as UINT64s at the offsets asked for (over a 0xcd fill), that timestamps advance at the
        // reported frequency, that a query keeps its result across executions, that one never ended reads zero
        // instead of stalling the GPU, and that the D3D12 rules for the calls are kept.
        {
            UINT64 frequency = 0;
            EXPECT(SUCCEEDED(queue->GetTimestampFrequency(&frequency)) && frequency >= 1'000'000);
            EXPECT(queue->GetTimestampFrequency(nullptr) == E_INVALIDARG);
            Ref<ID3D12QueryHeap> occlusion, timestamps, statistics, so, refused;
            D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 4, 0};
            EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&occlusion.p))));
            // A null output is E_UNEXPECTED, as on the host runtime (compare_invalid_arguments, "query heap:").
            EXPECT(device->CreateQueryHeap(&query_desc, __uuidof(ID3D12QueryHeap), nullptr) == E_UNEXPECTED);
            {
                Ref<ID3D12Device> parent;
                EXPECT(SUCCEEDED(occlusion->GetDevice(IID_PPV_ARGS(&parent.p))) && parent.p == device.p);
            }
            query_desc = {D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 3, 0};
            EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&timestamps.p))));
            query_desc = {D3D12_QUERY_HEAP_TYPE_OCCLUSION, 0, 0};
            EXPECT(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG && !refused.p);
            // The layer has no stream output, so stream-output statistics heaps are made without a Vulkan pool and
            // every query reads zero (D3D-302).
            query_desc = {D3D12_QUERY_HEAP_TYPE_SO_STATISTICS, 3, 0};
            EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&so.p))));
            query_desc = {D3D12_QUERY_HEAP_TYPE_PIPELINE_STATISTICS, 2, 0};
            if (d.pipeline_statistics_query)
                EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&statistics.p))));
            else
                std::printf("pipeline statistics: no pipelineStatisticsQuery, skipped\n");

            // Readback layout: timestamps at 0, occlusion at 64, statistics at 256, stream output at 512, the second
            // run's at 1024.
            constexpr UINT64 ts_at = 0, occlusion_at = 64, statistics_at = 256, so_at = 512, again_at = 1024;
            static_assert(sizeof(D3D12_QUERY_DATA_SO_STATISTICS) == 16);
            const auto u64 = [&](UINT64 offset) {
                UINT64 value;
                std::memcpy(&value, down + offset, sizeof(value));
                return value;
            };
            std::memset(down, 0xcd, 65536);
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->EndQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            if (statistics.p) list->BeginQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            // Two streams at once, inside the statistics query and crossing the occlusion one: none of them is a
            // Vulkan query, so none overlaps.
            list->BeginQuery(so.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0);
            list->BeginQuery(so.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3, 1);
            // A rectangle clear runs in a render pass inside a query; clears count no samples.
            device->CreateRenderTargetView(rt_resource.p, &rtv_desc, rtv1);  // the resolve test reused it
            list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
            barrier = transition(rt_resource.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
            list->ResourceBarrier(1, &barrier);
            list->ClearRenderTargetView(rtv1, red, 1, &square);
            list->EndQuery(so.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0);
            list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
            list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
            list->CopyBufferRegion(readback.p, 8192, upload.p, 0, 32768);
            list->EndQuery(so.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3, 1);
            list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
            if (statistics.p) list->EndQuery(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0);
            list->EndQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 1);
            list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 3, readback.p, ts_at);  // 2: never
            list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, readback.p, occlusion_at);
            list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1, 1, readback.p, occlusion_at + 8);
            list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 2, 2, readback.p, occlusion_at + 16);
            if (statistics.p)
                list->ResolveQueryData(statistics.p, D3D12_QUERY_TYPE_PIPELINE_STATISTICS, 0, 2, readback.p,
                                       statistics_at);
            list->ResolveQueryData(so.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM0, 0, 1, readback.p, so_at);
            list->ResolveQueryData(so.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM3, 1, 2, readback.p, so_at + 16);  // 2: never
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());

            const UINT64 t0 = u64(ts_at), t1 = u64(ts_at + 8);
            EXPECT(t0 != 0 && t0 != 0xcdcdcdcdcdcdcdcdull && t1 >= t0);
            EXPECT(double(t1 - t0) / double(frequency) < 10.0);  // the copy took less than ten seconds
            EXPECT(u64(ts_at + 16) == 0);                        // never written
            for (UINT i = 0; i < 4; ++i) EXPECT(u64(occlusion_at + 8 * i) == 0);
            for (UINT i = 0; statistics.p && i < 22; ++i) EXPECT(u64(statistics_at + 8 * i) == 0);
            EXPECT(u64(statistics.p ? statistics_at + 176 : statistics_at) == 0xcdcdcdcdcdcdcdcdull);  // not past it
            for (UINT i = 0; i < 6; ++i) EXPECT(u64(so_at + 8 * i) == 0);  // written and needed, three queries
            EXPECT(u64(so_at + 48) == 0xcdcdcdcdcdcdcdcdull);              // not past them

            // A second execution reuses timestamp 0 (reset before it is written again) and resolves timestamp 1,
            // which keeps the result of the first execution.
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->EndQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0);
            list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback.p, again_at);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            EXPECT(u64(again_at) >= t1 && u64(again_at + 8) == t1);

            // A timestamp written by one execution and resolved by the next, with nothing waited for in between, as
            // a game resolves last frame's timestamps. The resolve reads the new value (on MoltenVK only because the
            // queue orders the two; docs/d3d12_vulkan.md section 4.6).
            {
                constexpr UINT64 later_at = 2048;
                Ref<ID3D12CommandAllocator> writer_allocator;
                Ref<ID3D12GraphicsCommandList> writer;
                EXPECT(SUCCEEDED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                                IID_PPV_ARGS(&writer_allocator.p))));
                EXPECT(SUCCEEDED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, writer_allocator.p,
                                                           nullptr, IID_PPV_ARGS(&writer.p))));
                writer->CopyBufferRegion(readback.p, 8192, upload.p, 0, 32768);
                writer->EndQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 2);
                EXPECT(writer->Close() == S_OK);
                EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
                list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 3, readback.p, later_at);
                EXPECT(list->Close() == S_OK);
                ID3D12CommandList* const writers[] = {writer.p};
                queue->ExecuteCommandLists(1, writers);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                EXPECT(submit_and_wait());
                EXPECT(u64(later_at) == u64(again_at) && u64(later_at + 8) == t1);
                EXPECT(u64(later_at + 16) >= u64(again_at) && u64(later_at + 16) != 0xcdcdcdcdcdcdcdcdull);
            }

            // Lists that break the rules. Close reports what it can see; the rest is refused at execution, and
            // nothing is submitted.
            const auto refuse = [&](HRESULT close, leonida::ReplayResult result, auto&& record) {
                if (FAILED(allocator->Reset()) || FAILED(list->Reset(allocator.p, nullptr))) return false;
                record();
                if (list->Close() != close) return false;
                queue->ExecuteCommandLists(1, lists);
                return rt::d3d12::last_execute_result(queue.p) == result;
            };
            using R = leonida::ReplayResult;
            EXPECT(refuse(E_INVALIDARG, R::unsupported,  // begun and never ended; invalid list is refused at execution
                          [&] { list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0); }));
            EXPECT(refuse(E_INVALIDARG, R::unsupported,  // timestamps have no Begin; invalid list is refused at execution
                          [&] { list->BeginQuery(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0); }));
            EXPECT(refuse(E_INVALIDARG, R::unsupported,  // offset not a multiple of 8; invalid list is refused at execution
                          [&] { list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, readback.p, 4); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // past the end of the heap
                          [&] { list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 3, 2, readback.p, 0); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // a type the heap does not hold
                          [&] { list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_TIMESTAMP, 0); }));
            EXPECT(refuse(S_OK, R::invalid_query, [&] {  // a stream-output query in an occlusion heap
                list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1, 0);
                list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM1, 0);
            }));
            EXPECT(refuse(E_INVALIDARG, R::unsupported,  // a stream-output query begun and never ended; invalid list is refused
                          [&] { list->BeginQuery(so.p, D3D12_QUERY_TYPE_SO_STATISTICS_STREAM2, 0); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // into a texture
                          [&] { list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 1, rt_resource.p, 0); }));
            EXPECT(refuse(S_OK, R::invalid_query,  // past the end of the buffer
                          [&] { list->ResolveQueryData(timestamps.p, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, readback.p, 65528); }));
            EXPECT(refuse(S_OK, R::success, [&] {  // overlapping occlusion kinds share physical segments (D3D-252)
                list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
                list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
                list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1);
                list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0);
                list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_OCCLUSION, 0, 1, readback.p, 0);
                list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 1, 1, readback.p, 8);
            }));
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            EXPECT(SUCCEEDED(list->Close()));
        }

        // An occlusion query resolved into a predicate, the way a renderer skips occluded work (docs/d3d12_vulkan.md
        // sections 4.6 and 4.7). There are no draws yet, so the query counts 0 samples: EQUAL_ZERO skips the left
        // clear and NOT_EQUAL_ZERO runs the right one. The predicate goes COPY_DEST -> PREDICATION -> COPY_DEST and is
        // resolved into again, so validation sees both sides of the hand-off between the copy and conditional
        // rendering's read.
        if (!d.conditional_rendering) {
            std::printf("predication: no VK_EXT_conditional_rendering, skipped\n");
        } else {
            Ref<ID3D12QueryHeap> occlusion;
            Ref<ID3D12Resource> predicate, target;
            const D3D12_QUERY_HEAP_DESC query_heap_desc{D3D12_QUERY_HEAP_TYPE_OCCLUSION, 1, 0};
            EXPECT(SUCCEEDED(device->CreateQueryHeap(&query_heap_desc, IID_PPV_ARGS(&occlusion.p))));
            EXPECT(SUCCEEDED(create(default_heap, buffer_desc(256), D3D12_RESOURCE_STATE_COPY_DEST, predicate)));
            EXPECT(SUCCEEDED(create(default_heap,
                                    texture_desc(size, size, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                 D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
                                    D3D12_RESOURCE_STATE_RENDER_TARGET, target)));
            device->CreateRenderTargetView(target.p, nullptr, rtv1);  // the lists that used this slot have run
            UINT64 target_cursor = 0;
            const auto target_place = place(device.p, target.p, 0, target_cursor);
            const FLOAT green[4] = {0, 1, 0, 1};
            const D3D12_RECT left{0, 0, LONG(size / 2), LONG(size)}, right{LONG(size / 2), 0, LONG(size), LONG(size)};
            const auto resolve_predicate = [&] {
                list->BeginQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0);
                list->EndQuery(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0);
                list->ResolveQueryData(occlusion.p, D3D12_QUERY_TYPE_BINARY_OCCLUSION, 0, 1, predicate.p, 0);
            };

            // The predicate's hand-offs, through legacy barriers and through enhanced buffer barriers.
            const auto hand_off = [&](bool enhanced, bool to_predicate) {
                if (!enhanced) {
                    const auto copy_dest = D3D12_RESOURCE_STATE_COPY_DEST, predication = D3D12_RESOURCE_STATE_PREDICATION;
                    barrier = transition(predicate.p, to_predicate ? copy_dest : predication,
                                         to_predicate ? predication : copy_dest);
                    list->ResourceBarrier(1, &barrier);
                    return;
                }
                D3D12_BUFFER_BARRIER buffer{};
                buffer.SyncBefore = to_predicate ? D3D12_BARRIER_SYNC_COPY : D3D12_BARRIER_SYNC_PREDICATION;
                buffer.SyncAfter = to_predicate ? D3D12_BARRIER_SYNC_PREDICATION : D3D12_BARRIER_SYNC_COPY;
                buffer.AccessBefore = to_predicate ? D3D12_BARRIER_ACCESS_COPY_DEST : D3D12_BARRIER_ACCESS_PREDICATION;
                buffer.AccessAfter = to_predicate ? D3D12_BARRIER_ACCESS_PREDICATION : D3D12_BARRIER_ACCESS_COPY_DEST;
                buffer.pResource = predicate.p;
                buffer.Offset = 0;
                buffer.Size = UINT64_MAX;
                D3D12_BARRIER_GROUP buffer_group{};
                buffer_group.Type = D3D12_BARRIER_TYPE_BUFFER;
                buffer_group.NumBarriers = 1;
                buffer_group.pBufferBarriers = &buffer;
                list7->Barrier(1, &buffer_group);
            };
            for (const bool enhanced : {false, true}) {
                std::memset(down, 0xcd, 65536);
                EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
                if (enhanced) {
                    barrier = transition(target.p, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET);
                    list->ResourceBarrier(1, &barrier);
                }
                list->ClearRenderTargetView(rtv1, red, 0, nullptr);
                resolve_predicate();
                hand_off(enhanced, true);
                list->SetPredication(predicate.p, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);  // the value is 0: skipped
                list->ClearRenderTargetView(rtv1, green, 1, &left);
                list->SetPredication(predicate.p, 0, D3D12_PREDICATION_OP_NOT_EQUAL_ZERO);  // runs
                list->ClearRenderTargetView(rtv1, blue, 1, &right);
                list->SetPredication(nullptr, 0, D3D12_PREDICATION_OP_EQUAL_ZERO);
                hand_off(enhanced, false);
                resolve_predicate();  // written again after conditional rendering has read it
                barrier = transition(target.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
                list->ResourceBarrier(1, &barrier);
                {
                    const auto dst = footprint(readback.p, target_place);
                    const auto src = subresource(target.p, 0);
                    list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
                }
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
                EXPECT(submit_and_wait());
                const std::uint8_t red_texel[4] = {255, 0, 0, 255}, blue_texel[4] = {0, 0, 255, 255};
                for (UINT row = 0; row < size; ++row)
                    for (UINT col = 0; col < size; ++col)
                        report(enhanced ? "predicated clear texel (enhanced barriers)" : "predicated clear texel",
                               row * size + col,
                               down + target_place.Offset + row * target_place.Footprint.RowPitch + col * 4,
                               col < size / 2 ? red_texel : blue_texel, 4);
            }
        }

        // D3D-71: a default-heap buffer outside the arena (this device has none) starts zeroed, as D3D12 committed
        // resources do. Its memory is first dirtied through an earlier buffer, which is then released so the next
        // allocation is likely to reuse it. CREATE_NOT_ZEROED skips the fill; a copy into the same list still lands.
        {
            constexpr UINT64 bytes = 4096;
            const D3D12_HEAP_PROPERTIES default_buffer_heap{D3D12_HEAP_TYPE_DEFAULT};
            std::memset(up, 0xab, std::size_t(bytes));
            {
                Ref<ID3D12Resource> dirty;
                EXPECT(SUCCEEDED(create(default_buffer_heap, buffer_desc(bytes), D3D12_RESOURCE_STATE_COMMON, dirty)));
                EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
                list->CopyBufferRegion(dirty.p, 0, upload.p, 0, bytes);
                EXPECT(list->Close() == S_OK);
                queue->ExecuteCommandLists(1, lists);
                EXPECT(submit_and_wait());
            }
            Ref<ID3D12Resource> fresh, not_zeroed;
            EXPECT(SUCCEEDED(create(default_buffer_heap, buffer_desc(bytes), D3D12_RESOURCE_STATE_COMMON, fresh)));
            const D3D12_RESOURCE_DESC zero_desc = buffer_desc(bytes);
            EXPECT(SUCCEEDED(device->CreateCommittedResource(&default_buffer_heap, D3D12_HEAP_FLAG_CREATE_NOT_ZEROED,
                                                             &zero_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&not_zeroed.p))));
            std::memset(down, 0xcd, std::size_t(2 * bytes));
            EXPECT(SUCCEEDED(allocator->Reset()) && SUCCEEDED(list->Reset(allocator.p, nullptr)));
            list->CopyBufferRegion(readback.p, 0, fresh.p, 0, bytes);
            list->CopyBufferRegion(not_zeroed.p, 0, upload.p, 0, bytes);  // written before it is read
            auto readable = transition(not_zeroed.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
            list->ResourceBarrier(1, &readable);
            list->CopyBufferRegion(readback.p, bytes, not_zeroed.p, 0, bytes);
            EXPECT(list->Close() == S_OK);
            queue->ExecuteCommandLists(1, lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(submit_and_wait());
            const std::uint8_t zero[4] = {}, written[4] = {0xab, 0xab, 0xab, 0xab};
            for (UINT64 i = 0; i < bytes; i += 4) report("zero-filled buffer byte", UINT(i), down + i, zero, 4);
            for (UINT64 i = 0; i < bytes; i += 4)
                report("NOT_ZEROED buffer byte after a copy", UINT(i), down + bytes + i, written, 4);
        }

        // Every child holds the proxy; once they are gone the proxy holds the only reference.
        upload->Unmap(0, nullptr);
        readback->Unmap(0, nullptr);
        for (auto* r : {&rt_resource, &packed565, &ds, &untouched, &upload, &readback, &source, &whole, &part, &bc,
                        &line, &line_array, &line_target}) r->reset();
        list7.reset();
        list.reset();
        allocator.reset();
        queue.reset();
        fence.reset();
        rtv_heap.reset();
        dsv_heap.reset();
        EXPECT(rt::d3d12_xs::device_proxy_ref_count(device.p) == 1);
    }  // the device is released here, before the Vulkan device it runs on
    EXPECT(!mismatches);

    // CPU texture copies, once per path. A second D3D12 device on the same Vulkan device, told the extension is
    // off, takes the staging path even on a GPU that has host image copy.
    for (const bool host : {true, false}) {
        if (host && !d.host_image_copy) {
            std::printf("WriteToSubresource/ReadFromSubresource (host image copy): no VK_EXT_host_image_copy, "
                        "skipped\n");
            continue;
        }
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = d.instance_proc;
        info.instance = d.instance;
        info.physical = d.physical;
        info.device = d.device;
        info.queue = d.queue;
        info.queue_family = d.family;
        info.host_image_copy = host;
        Ref<ID3D12Device> device;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&device.p))));
        EXPECT(cpu_copies(device.p, d.bc, host ? "host image copy" : "staging copy") == 0);
    }

    {
        Ref<ID3D12Device> removable;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&removable.p))));
        Ref<ID3D12Device5> removable5;
        EXPECT(removable.p && SUCCEEDED(removable->QueryInterface(IID_PPV_ARGS(&removable5.p))));
        if (removable5.p) {
            EXPECT(removable5->GetDeviceRemovedReason() == S_OK);
            removable5->RemoveDevice();
            EXPECT(removable5->GetDeviceRemovedReason() == DXGI_ERROR_DEVICE_REMOVED);
        }
    }

    // The path D3D12XboxCreateDevice takes with LEONIDA_D3D12=vulkan: a device on its own Vulkan instance.
    {
        Ref<ID3D12Device> standalone;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&standalone.p))));
        // What translated shaders can declare beyond Int64/Float64 (the index's `features`) is enabled wherever the GPU
        // has it: image stores and loads without a format, gather offsets, LOD clamps, a vertex shader's layer and
        // viewport. Creating a module that declares one without it breaks VUID-VkShaderModuleCreateInfo-pCode-08740.
        const rt::d3d12::VulkanDeviceInfo* made = rt::d3d12::vulkan_device_info(standalone.p);
        EXPECT(made != nullptr);
        if (made) {
            const auto features2 = reinterpret_cast<PFN_vkGetPhysicalDeviceFeatures2>(
                made->get_instance_proc(made->instance, "vkGetPhysicalDeviceFeatures2"));
            EXPECT(features2 != nullptr);
            VkPhysicalDeviceVulkan12Features have12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
            VkPhysicalDeviceFeatures2 have{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, &have12};
            if (features2) features2(made->physical, &have);
            const VkPhysicalDeviceFeatures& on = made->features;
            EXPECT(on.shaderStorageImageWriteWithoutFormat == have.features.shaderStorageImageWriteWithoutFormat);
            EXPECT(on.shaderStorageImageReadWithoutFormat == have.features.shaderStorageImageReadWithoutFormat);
            EXPECT(on.shaderImageGatherExtended == have.features.shaderImageGatherExtended);
            EXPECT(on.shaderResourceMinLod == have.features.shaderResourceMinLod);
            EXPECT(made->shader_output_layer == (have12.shaderOutputLayer == VK_TRUE));
            EXPECT(made->shader_output_viewport_index == (have12.shaderOutputViewportIndex == VK_TRUE));
            // The pixel-shader extensions (barycentrics, the POPS interlock, maintenance8's texel offsets): each is
            // enabled exactly when the GPU offers the extension and its feature. A structure is chained into the
            // query only when its extension is offered.
            const auto enumerate = reinterpret_cast<PFN_vkEnumerateDeviceExtensionProperties>(
                made->get_instance_proc(made->instance, "vkEnumerateDeviceExtensionProperties"));
            EXPECT(enumerate != nullptr);
            std::vector<VkExtensionProperties> offered;
            std::uint32_t offered_count = 0;
            if (enumerate && enumerate(made->physical, nullptr, &offered_count, nullptr) == VK_SUCCESS) {
                offered.resize(offered_count);
                enumerate(made->physical, nullptr, &offered_count, offered.data());
                offered.resize(offered_count);
            }
            const auto has = [&](const char* name) {
                return std::any_of(offered.begin(), offered.end(),
                                   [&](const VkExtensionProperties& e) { return std::strcmp(e.extensionName, name) == 0; });
            };
            VkPhysicalDeviceFragmentShaderBarycentricFeaturesKHR bary{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_BARYCENTRIC_FEATURES_KHR};
            VkPhysicalDeviceFragmentShaderInterlockFeaturesEXT lock{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FRAGMENT_SHADER_INTERLOCK_FEATURES_EXT};
            VkPhysicalDeviceFeatures2 ext_query{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
            if (has(VK_KHR_FRAGMENT_SHADER_BARYCENTRIC_EXTENSION_NAME)) {
                bary.pNext = ext_query.pNext;
                ext_query.pNext = &bary;
            }
            if (has(VK_EXT_FRAGMENT_SHADER_INTERLOCK_EXTENSION_NAME)) {
                lock.pNext = ext_query.pNext;
                ext_query.pNext = &lock;
            }
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
            VkPhysicalDeviceMaintenance8FeaturesKHR m8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_8_FEATURES_KHR};
            if (has(VK_KHR_MAINTENANCE_8_EXTENSION_NAME)) {
                m8.pNext = ext_query.pNext;
                ext_query.pNext = &m8;
            }
#endif
            VkPhysicalDeviceShaderClockFeaturesKHR clock{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_CLOCK_FEATURES_KHR};
            if (has(VK_KHR_SHADER_CLOCK_EXTENSION_NAME)) {
                clock.pNext = ext_query.pNext;
                ext_query.pNext = &clock;
            }
            if (features2) features2(made->physical, &ext_query);
            EXPECT(made->shader_subgroup_clock == (clock.shaderSubgroupClock == VK_TRUE));
            EXPECT(made->shader_device_clock == (clock.shaderDeviceClock == VK_TRUE));
            EXPECT(made->fragment_shader_barycentric == (bary.fragmentShaderBarycentric == VK_TRUE));
            EXPECT(made->fragment_shader_interlock == (lock.fragmentShaderPixelInterlock == VK_TRUE));
#ifdef VK_KHR_MAINTENANCE_8_EXTENSION_NAME
            EXPECT(made->maintenance8 == (m8.maintenance8 == VK_TRUE));
#endif
            // The OPTIONS booleans answer what the layer will honour, not what the GPU offers (docs 4.1: a false
            // "supported" is worse at boot; the PSO paths refuse these same missing features). Stencil ref: the
            // layer enables no stencil export and the translator emits no stencil-ref write, so always FALSE.
            D3D12_FEATURE_DATA_D3D12_OPTIONS answers{};
            EXPECT(SUCCEEDED(standalone->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &answers,
                                                             sizeof(answers))));
            EXPECT(answers.DoublePrecisionFloatShaderOps == BOOL(made->features.shaderFloat64));
            EXPECT(answers.OutputMergerLogicOp == BOOL(made->features.logicOp));
            EXPECT(answers.PSSpecifiedStencilRefSupported == FALSE);
            EXPECT(answers.ROVsSupported == BOOL(made->fragment_shader_interlock));
            EXPECT(answers.VPAndRTArrayIndexFromAnyShaderFeedingRasterizerSupportedWithoutGSEmulation ==
                   BOOL(made->shader_output_layer && made->shader_output_viewport_index));
        }
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        EXPECT(SUCCEEDED(standalone->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        EXPECT(SUCCEEDED(standalone->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 5)));
        EXPECT(SUCCEEDED(fence->SetEventOnCompletion(5, nullptr)) && fence->GetCompletedValue() == 5);
        UINT64 frequency = 0;
        Ref<ID3D12QueryHeap> timestamps;
        const D3D12_QUERY_HEAP_DESC query_desc{D3D12_QUERY_HEAP_TYPE_TIMESTAMP, 1, 0};
        EXPECT(SUCCEEDED(queue->GetTimestampFrequency(&frequency)) && frequency);
        EXPECT(SUCCEEDED(standalone->CreateQueryHeap(&query_desc, IID_PPV_ARGS(&timestamps.p))));

        // Heaps and placed resources: no proxy sits on these slots, so this is the game's call path. A heap reserves
        // arena memory, and a placed buffer's console VA is where it sits in it.
        D3D12_HEAP_PROPERTIES placed_props{};
        placed_props.Type = D3D12_HEAP_TYPE_DEFAULT;
        D3D12_HEAP_DESC placed_heap_desc{};
        placed_heap_desc.Properties = placed_props;
        placed_heap_desc.SizeInBytes = 256 * 1024;
        placed_heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        Ref<ID3D12Heap> placed_heap, again;
        EXPECT(SUCCEEDED(standalone->CreateHeap(&placed_heap_desc, IID_PPV_ARGS(&placed_heap.p))));
        EXPECT(placed_heap->GetDesc().SizeInBytes == placed_heap_desc.SizeInBytes &&
               placed_heap->GetDesc().Properties.Type == D3D12_HEAP_TYPE_DEFAULT);
        Ref<ID3D12Resource> placed_a, placed_b, placed_image, refused;
        const D3D12_RESOURCE_DESC placed_buffer = buffer_desc(4096);
        const D3D12_RESOURCE_DESC placed_texture =
            texture_desc(4, 4, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE);
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(placed_heap.p, 0, &placed_buffer,
                                                          D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                          IID_PPV_ARGS(&placed_a.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(placed_heap.p, 65536, &placed_buffer,
                                                          D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                          IID_PPV_ARGS(&placed_b.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(placed_heap.p, 131072, &placed_texture,
                                                          D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                          IID_PPV_ARGS(&placed_image.p))));
        const D3D12_GPU_VIRTUAL_ADDRESS placed_va = placed_a->GetGPUVirtualAddress();
        EXPECT(placed_va && placed_va % D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT == 0);
        EXPECT(placed_b->GetGPUVirtualAddress() == placed_va + 65536);
        EXPECT(placed_image->GetGPUVirtualAddress() == 0);  // only buffers are addressable, as in D3D12
        // Off the 64 KiB grid, past the heap end, and non-ROW_MAJOR buffers: E_INVALIDARG, not placed somewhere wrong.
        EXPECT(standalone->CreatePlacedResource(placed_heap.p, 512, &placed_buffer, D3D12_RESOURCE_STATE_COMMON,
                                                nullptr, IID_PPV_ARGS(&refused.p)) == E_INVALIDARG);
        EXPECT(standalone->CreatePlacedResource(placed_heap.p, 256 * 1024, &placed_buffer,
                                                D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&refused.p)) == E_INVALIDARG);
        D3D12_RESOURCE_DESC tiled_buf = buffer_desc(4096);
        tiled_buf.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        EXPECT(standalone->CreatePlacedResource(placed_heap.p, 0, &tiled_buf, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                IID_PPV_ARGS(&refused.p)) == E_INVALIDARG);
        EXPECT(!refused.p);
        // A placed buffer reads as zeroed through the copy path, at its console VA.
        D3D12_HEAP_PROPERTIES readback_props{};
        readback_props.Type = D3D12_HEAP_TYPE_UPLOAD;
        const D3D12_RESOURCE_DESC readback_desc = buffer_desc(64);
        Ref<ID3D12Resource> zeroes;
        EXPECT(SUCCEEDED(standalone->CreateCommittedResource(&readback_props, D3D12_HEAP_FLAG_NONE, &readback_desc,
                                                             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
                                                             IID_PPV_ARGS(&zeroes.p))));
        Ref<ID3D12CommandAllocator> placed_allocator;
        Ref<ID3D12GraphicsCommandList> placed_list;
        EXPECT(SUCCEEDED(standalone->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&placed_allocator.p))));
        EXPECT(SUCCEEDED(standalone->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, placed_allocator.p, nullptr,
                                                       IID_PPV_ARGS(&placed_list.p))));
        ID3D12CommandList* placed_lists[] = {placed_list.p};
        placed_list->CopyBufferRegion(zeroes.p, 0, placed_a.p, 0, 64);
        EXPECT(SUCCEEDED(placed_list->Close()));
        queue->ExecuteCommandLists(1, placed_lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 6)) && SUCCEEDED(fence->SetEventOnCompletion(6, nullptr)));
        void* zero_bytes = nullptr;
        EXPECT(SUCCEEDED(zeroes->Map(0, nullptr, &zero_bytes)));
        report("placed buffer", 0, static_cast<std::uint8_t*>(zero_bytes), std::array<std::uint8_t, 64>{}.data(), 64);
        zeroes->Unmap(0, nullptr);

        // Placed buffers carry their console VA in VulkanContext (so VA-resolved commands like WriteBufferImmediate
        // find them) and placed readback buffers mark host_readable (so the submission flushes GPU writes to host).
        {
            D3D12_HEAP_DESC rb_heap_desc{};
            rb_heap_desc.Properties.Type = D3D12_HEAP_TYPE_READBACK;
            rb_heap_desc.SizeInBytes = 65536;
            rb_heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            Ref<ID3D12Heap> rb_heap;
            Ref<ID3D12Resource> placed_rb;
            EXPECT(SUCCEEDED(standalone->CreateHeap(&rb_heap_desc, IID_PPV_ARGS(&rb_heap.p))));
            EXPECT(SUCCEEDED(standalone->CreatePlacedResource(rb_heap.p, 0, &placed_buffer,
                                                              D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                              IID_PPV_ARGS(&placed_rb.p))));
            Ref<ID3D12GraphicsCommandList2> placed_list2;
            EXPECT(SUCCEEDED(placed_list->QueryInterface(IID_PPV_ARGS(&placed_list2.p))));
            EXPECT(SUCCEEDED(placed_allocator->Reset()) && SUCCEEDED(placed_list->Reset(placed_allocator.p, nullptr)));
            const D3D12_WRITEBUFFERIMMEDIATE_PARAMETER wp{placed_va + 16, 0xCAFEBABEu};
            const D3D12_WRITEBUFFERIMMEDIATE_MODE wm = D3D12_WRITEBUFFERIMMEDIATE_MODE_DEFAULT;
            placed_list2->WriteBufferImmediate(1, &wp, &wm);
            D3D12_RESOURCE_BARRIER b{};
            b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Transition = {placed_a.p, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE};
            placed_list->ResourceBarrier(1, &b);
            placed_list->CopyBufferRegion(placed_rb.p, 0, placed_a.p, 0, 64);
            EXPECT(SUCCEEDED(placed_list->Close()));
            queue->ExecuteCommandLists(1, placed_lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 7)) && SUCCEEDED(fence->SetEventOnCompletion(7, nullptr)));
            void* rb_bytes = nullptr;
            EXPECT(SUCCEEDED(placed_rb->Map(0, nullptr, &rb_bytes)));
            std::uint32_t word = 0;
            std::memcpy(&word, static_cast<const std::uint8_t*>(rb_bytes) + 16, sizeof(word));
            EXPECT(word == 0xCAFEBABEu);
            placed_rb->Unmap(0, nullptr);
        }

        // The heap owns the range: releasing it puts the next heap of that size back at the same place.
        placed_image.reset();
        placed_b.reset();
        placed_a.reset();
        placed_heap.reset();
        EXPECT(SUCCEEDED(standalone->CreateHeap(&placed_heap_desc, IID_PPV_ARGS(&again.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(again.p, 0, &placed_buffer, D3D12_RESOURCE_STATE_GENERIC_READ,
                                                          nullptr, IID_PPV_ARGS(&placed_a.p))));
        EXPECT(placed_a->GetGPUVirtualAddress() == placed_va);

        // D3D-191: tiled placed resources (D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE and Xbox 0x101).
        D3D12_HEAP_DESC tiled_heap_desc{};
        tiled_heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        tiled_heap_desc.SizeInBytes = 2 * 1024 * 1024;
        tiled_heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        Ref<ID3D12Heap> tiled_heap, upload_placed_heap;
        EXPECT(SUCCEEDED(standalone->CreateHeap(&tiled_heap_desc, IID_PPV_ARGS(&tiled_heap.p))));
        D3D12_HEAP_DESC upload_heap_desc = tiled_heap_desc;
        upload_heap_desc.Properties.Type = D3D12_HEAP_TYPE_UPLOAD;
        EXPECT(SUCCEEDED(standalone->CreateHeap(&upload_heap_desc, IID_PPV_ARGS(&upload_placed_heap.p))));

        // D3D-238: heap descriptions the host refuses (RX 5700 and WARP; ALLOW_DISPLAY on an RTX 4070 SUPER and a
        // UHD 770) and their valid neighbours, HARDWARE_PROTECTED among them. compare_invalid_arguments and
        // compare_heap_neighbours check them against a host.
        {
            const D3D12_HEAP_FLAGS deny_all = D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES |
                                              D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES;
            const D3D12_HEAP_DESC refused_heaps[] = {
                heap_desc_of(4096),
                heap_desc_of(0, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_L0}),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, deny_all),
                heap_desc_of(0, {D3D12_HEAP_TYPE(0)}),
                heap_desc_of(0, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
                                 D3D12_MEMORY_POOL_UNKNOWN}),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK}),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_NONE, 0),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_ALLOW_DISPLAY),
            };
            for (const D3D12_HEAP_DESC& hd : refused_heaps) EXPECT(heap_create(standalone.p, hd) == E_INVALIDARG);
            const D3D12_HEAP_DESC accepted_heaps[] = {
                heap_desc_of(0),
                heap_desc_of(D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT),
                heap_desc_of(D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT, {D3D12_HEAP_TYPE_DEFAULT},
                             D3D12_HEAP_FLAG_NONE, D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_NONE, 1000),
                heap_desc_of(0, {D3D12_HEAP_TYPE_UPLOAD}),
                heap_desc_of(0, {D3D12_HEAP_TYPE_READBACK}),
                heap_desc_of(0, {D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE, D3D12_MEMORY_POOL_L0}),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT},
                             D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES),
                heap_desc_of(0, {D3D12_HEAP_TYPE_DEFAULT}, D3D12_HEAP_FLAG_HARDWARE_PROTECTED),
            };
            for (const D3D12_HEAP_DESC& hd : accepted_heaps) EXPECT(heap_create(standalone.p, hd) == S_OK);
            const int heap_mismatches = compare_heap_neighbours(standalone.p);
            if (heap_mismatches < 0)
                std::printf("CreateHeap valid neighbours: no host D3D12 device to compare with, skipped\n");
            EXPECT(heap_mismatches <= 0);
            const int heap_desc_mismatches = compare_heap_descs(standalone.p);
            if (heap_desc_mismatches < 0)
                std::printf("ID3D12Heap::GetDesc: no host D3D12 device to compare with, skipped\n");
            EXPECT(heap_desc_mismatches <= 0);
            const int placed_mismatches = compare_placed_reserved(standalone.p);
            if (placed_mismatches < 0)
                std::printf("CreatePlacedResource/CreateReservedResource arguments: no host D3D12 device to compare "
                            "with, skipped\n");
            EXPECT(placed_mismatches <= 0);
        }

        auto tiled_2d = [&](UINT w, UINT h, DXGI_FORMAT fmt, UINT16 mips, D3D12_RESOURCE_FLAGS flags,
                            UINT64 align = 0) {
            D3D12_RESOURCE_DESC d = texture_desc(w, h, fmt, flags, mips);
            d.Alignment = align;
            d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
            return d;
        };
        const D3D12_RESOURCE_DESC tiled_rt_desc =
            tiled_2d(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        const D3D12_RESOURCE_DESC tiled_bc_desc =
            tiled_2d(256, 256, DXGI_FORMAT_BC1_UNORM, 0, D3D12_RESOURCE_FLAG_NONE);
        const D3D12_RESOURCE_DESC tiled_ds_desc =
            tiled_2d(256, 256, DXGI_FORMAT_D32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL);
        D3D12_RESOURCE_DESC tiled_3d_desc =
            tiled_2d(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, 0, D3D12_RESOURCE_FLAG_NONE);
        tiled_3d_desc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
        tiled_3d_desc.DepthOrArraySize = 32;
        D3D12_RESOURCE_DESC xbox_tiled_desc =
            tiled_2d(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 1, D3D12_RESOURCE_FLAG_NONE);
        xbox_tiled_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);

        Ref<ID3D12Resource> tiled_rt, tiled_bc, tiled_ds, tiled_3d, xbox_tiled;
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 0, &tiled_rt_desc, D3D12_RESOURCE_STATE_COMMON,
                                                          nullptr, IID_PPV_ARGS(&tiled_rt.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 512 * 1024, &tiled_bc_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&tiled_bc.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 640 * 1024, &tiled_ds_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&tiled_ds.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 1024 * 1024, &tiled_3d_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&tiled_3d.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 0, &xbox_tiled_desc, D3D12_RESOURCE_STATE_COMMON,
                                                          nullptr, IID_PPV_ARGS(&xbox_tiled.p))));

        for (ID3D12Resource* r : {tiled_rt.p, tiled_bc.p, tiled_ds.p, tiled_3d.p}) {
            const D3D12_RESOURCE_DESC got = r->GetDesc();
            EXPECT(r->GetGPUVirtualAddress() == 0);
            EXPECT(got.Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE);
            EXPECT(got.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
            EXPECT(got.MipLevels > 0);
            D3D12_HEAP_PROPERTIES got_hp{};
            D3D12_HEAP_FLAGS got_hf = D3D12_HEAP_FLAG_NONE;
            EXPECT(SUCCEEDED(r->GetHeapProperties(&got_hp, &got_hf)));
            EXPECT(got_hp.Type == D3D12_HEAP_TYPE_DEFAULT);
            const D3D12_RESOURCE_ALLOCATION_INFO ai = standalone->GetResourceAllocationInfo(0, 1, &got);
            UINT tiles = 0;
            D3D12_TILE_SHAPE shape{};
            standalone->GetResourceTiling(r, &tiles, nullptr, &shape, nullptr, 0, nullptr);
            EXPECT(tiles > 0 && shape.WidthInTexels > 0 && ai.Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
            EXPECT(ai.SizeInBytes == UINT64(tiles) * D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES);
        }
        EXPECT(static_cast<UINT>(xbox_tiled->GetDesc().Layout) == 0x101);
        UINT xbox_tiles = 0;
        standalone->GetResourceTiling(xbox_tiled.p, &xbox_tiles, nullptr, nullptr, nullptr, 0, nullptr);
        EXPECT(xbox_tiles == 4);

        // Refusals matching host D3D12: 1D, 4 KiB alignment, 96-bit, two-plane DS, invalid MSAA, upload heap, past-end.
        D3D12_RESOURCE_DESC bad_1d = line_desc(256, 1, 1, DXGI_FORMAT_R8G8B8A8_UNORM);
        bad_1d.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        auto tiled_2d_msaa = [&](UINT samples, D3D12_RESOURCE_FLAGS flags, UINT64 align = 0) {
            D3D12_RESOURCE_DESC d = tiled_2d(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 1, flags, align);
            d.SampleDesc.Count = samples;
            return d;
        };
        const D3D12_RESOURCE_DESC bad_cases[] = {
            bad_1d,
            tiled_2d(16, 16, DXGI_FORMAT_R8G8B8A8_UNORM, 1, D3D12_RESOURCE_FLAG_NONE,
                     D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT),
            tiled_2d(64, 64, DXGI_FORMAT_R32G32B32_FLOAT, 1, D3D12_RESOURCE_FLAG_NONE),
            tiled_2d(64, 64, DXGI_FORMAT_D24_UNORM_S8_UINT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
            tiled_2d(64, 64, DXGI_FORMAT_D32_FLOAT_S8X24_UINT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
            tiled_2d_msaa(2, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET),
            tiled_2d_msaa(4, D3D12_RESOURCE_FLAG_NONE),
            tiled_2d_msaa(4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT),
        };
        for (const D3D12_RESOURCE_DESC& bad : bad_cases) {
            Ref<ID3D12Resource> fail;
            EXPECT(standalone->CreatePlacedResource(tiled_heap.p, 0, &bad, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&fail.p)) == E_INVALIDARG);
            EXPECT(!fail.p);
        }
        {
            Ref<ID3D12Resource> fail;
            EXPECT(standalone->CreatePlacedResource(upload_placed_heap.p, 0, &tiled_rt_desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&fail.p)) == E_INVALIDARG);
            const D3D12_RESOURCE_ALLOCATION_INFO rt_alloc = standalone->GetResourceAllocationInfo(0, 1, &tiled_rt_desc);
            EXPECT(standalone->CreatePlacedResource(tiled_heap.p,
                                                    tiled_heap_desc.SizeInBytes - rt_alloc.SizeInBytes + 65536,
                                                    &tiled_rt_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                    IID_PPV_ARGS(&fail.p)) == E_INVALIDARG);
            EXPECT(!fail.p);
        }

        // Clear and read back a placed tiled RTV to verify GPU execution.
        D3D12_DESCRIPTOR_HEAP_DESC tiled_rtv_desc{};
        tiled_rtv_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        tiled_rtv_desc.NumDescriptors = 1;
        Ref<ID3D12DescriptorHeap> tiled_rtv_heap;
        EXPECT(SUCCEEDED(standalone->CreateDescriptorHeap(&tiled_rtv_desc, IID_PPV_ARGS(&tiled_rtv_heap.p))));
        standalone->CreateRenderTargetView(tiled_rt.p, nullptr, tiled_rtv_heap->GetCPUDescriptorHandleForHeapStart());
        D3D12_HEAP_PROPERTIES rb_props{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC rb_desc = buffer_desc(256);
        Ref<ID3D12Resource> tiled_rb;
        EXPECT(SUCCEEDED(standalone->CreateCommittedResource(&rb_props, D3D12_HEAP_FLAG_NONE, &rb_desc,
                                                             D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                             IID_PPV_ARGS(&tiled_rb.p))));
        Ref<ID3D12CommandAllocator> tiled_alloc;
        Ref<ID3D12GraphicsCommandList> tiled_list;
        EXPECT(SUCCEEDED(standalone->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                            IID_PPV_ARGS(&tiled_alloc.p))));
        EXPECT(SUCCEEDED(standalone->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, tiled_alloc.p, nullptr,
                                                       IID_PPV_ARGS(&tiled_list.p))));
        const FLOAT tiled_color[4] = {64.0f / 255.0f, 128.0f / 255.0f, 191.0f / 255.0f, 1.0f};
        D3D12_RESOURCE_BARRIER b = transition(tiled_rt.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        tiled_list->ResourceBarrier(1, &b);
        tiled_list->ClearRenderTargetView(tiled_rtv_heap->GetCPUDescriptorHandleForHeapStart(), tiled_color, 0, nullptr);
        b = transition(tiled_rt.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        tiled_list->ResourceBarrier(1, &b);
        D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
        fp.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
        const D3D12_TEXTURE_COPY_LOCATION copy_dst = footprint(tiled_rb.p, fp);
        const D3D12_TEXTURE_COPY_LOCATION copy_src = subresource(tiled_rt.p, 0);
        D3D12_BOX box{0, 0, 0, 1, 1, 1};
        tiled_list->CopyTextureRegion(&copy_dst, 0, 0, 0, &copy_src, &box);
        EXPECT(SUCCEEDED(tiled_list->Close()));
        ID3D12CommandList* tiled_lists[] = {tiled_list.p};
        queue->ExecuteCommandLists(1, tiled_lists);
        EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        EXPECT(SUCCEEDED(queue->Signal(fence.p, 8)) && SUCCEEDED(fence->SetEventOnCompletion(8, nullptr)));
        void* tiled_bytes = nullptr;
        EXPECT(SUCCEEDED(tiled_rb->Map(0, nullptr, &tiled_bytes)));
        const std::uint8_t want_rgba[4] = {64, 128, 191, 255};
        report("tiled placed RTV", 0, static_cast<std::uint8_t*>(tiled_bytes), want_rgba, 4);
        EXPECT(std::memcmp(tiled_bytes, want_rgba, 4) == 0);
        tiled_rb->Unmap(0, nullptr);

        // Compare placed 64KB_UNDEFINED_SWIZZLE acceptance, layout, alignment, heap props and tile shape with host.
        Ref<ID3D12Device> host_dev;
        if (create_host_device(host_dev)) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS host_opts{};
            if (SUCCEEDED(host_dev->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &host_opts, sizeof(host_opts))) &&
                host_opts.TiledResourcesTier != D3D12_TILED_RESOURCES_TIER_NOT_SUPPORTED) {
                Ref<ID3D12Heap> host_heap;
                if (SUCCEEDED(host_dev->CreateHeap(&tiled_heap_desc, IID_PPV_ARGS(&host_heap.p)))) {
                    const D3D12_RESOURCE_DESC host_cases[] = {
                        tiled_2d(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, 1, D3D12_RESOURCE_FLAG_NONE),
                        tiled_2d(256, 256, DXGI_FORMAT_D32_FLOAT, 1, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL),
                        tiled_3d_desc,
                    };
                    for (int idx = 0; idx < int(std::size(host_cases)); ++idx) {
                        const D3D12_RESOURCE_DESC& host_desc = host_cases[idx];
                        if (host_desc.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE3D &&
                            host_opts.TiledResourcesTier < D3D12_TILED_RESOURCES_TIER_3)
                            continue;
                        Ref<ID3D12Resource> our_r, host_r;
                        const HRESULT our_hr = standalone->CreatePlacedResource(
                            tiled_heap.p, 0, &host_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&our_r.p));
                        const HRESULT host_hr = host_dev->CreatePlacedResource(
                            host_heap.p, 0, &host_desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&host_r.p));
                        if (FAILED(host_hr)) continue;
                        EXPECT(SUCCEEDED(our_hr));
                        EXPECT(our_r->GetDesc().Layout == host_r->GetDesc().Layout);
                        EXPECT(our_r->GetDesc().Alignment == host_r->GetDesc().Alignment);
                        D3D12_HEAP_PROPERTIES ohp{}, hhp{};
                        EXPECT(SUCCEEDED(our_r->GetHeapProperties(&ohp, nullptr)) &&
                               SUCCEEDED(host_r->GetHeapProperties(&hhp, nullptr)));
                        EXPECT(ohp.Type == hhp.Type);
                        D3D12_TILE_SHAPE os{}, hs{};
                        standalone->GetResourceTiling(our_r.p, nullptr, nullptr, &os, nullptr, 0, nullptr);
                        host_dev->GetResourceTiling(host_r.p, nullptr, nullptr, &hs, nullptr, 0, nullptr);
                        EXPECT(os.WidthInTexels == hs.WidthInTexels && os.HeightInTexels == hs.HeightInTexels &&
                               os.DepthInTexels == hs.DepthInTexels);
                    }
                }
            }
        }

        // D3D-222: multisampled placed textures (UNKNOWN and 4x tiled).
        D3D12_HEAP_DESC msaa_heap_desc{};
        msaa_heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        msaa_heap_desc.SizeInBytes = 16 * 1024 * 1024;
        msaa_heap_desc.Alignment = D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT;
        Ref<ID3D12Heap> msaa_heap;
        EXPECT(SUCCEEDED(standalone->CreateHeap(&msaa_heap_desc, IID_PPV_ARGS(&msaa_heap.p))));
        D3D12_HEAP_DESC deny_rt_desc = tiled_heap_desc;
        deny_rt_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES;
        Ref<ID3D12Heap> deny_rt_heap;
        EXPECT(SUCCEEDED(standalone->CreateHeap(&deny_rt_desc, IID_PPV_ARGS(&deny_rt_heap.p))));

        auto msaa_desc = [&](UINT w, UINT h, UINT samples, UINT64 align, D3D12_TEXTURE_LAYOUT layout,
                             D3D12_RESOURCE_FLAGS flags, DXGI_FORMAT fmt = DXGI_FORMAT_R8G8B8A8_UNORM,
                             UINT16 mips = 1) {
            D3D12_RESOURCE_DESC d = texture_desc(w, h, fmt, flags, mips);
            d.Alignment = align;
            d.SampleDesc.Count = samples;
            d.Layout = layout;
            return d;
        };
        const D3D12_RESOURCE_DESC msaa_unk_rt_desc =
            msaa_desc(256, 256, 4, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        const D3D12_RESOURCE_DESC msaa_unk_small_desc = msaa_desc(
            256, 256, 4, D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT, D3D12_TEXTURE_LAYOUT_UNKNOWN,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        const D3D12_RESOURCE_DESC msaa_unk_ds_desc =
            msaa_desc(256, 256, 4, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                      DXGI_FORMAT_D32_FLOAT);
        const D3D12_RESOURCE_DESC msaa_tiled_rt_desc = tiled_2d_msaa(4, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
        D3D12_RESOURCE_DESC msaa_xbox_rt_desc = msaa_tiled_rt_desc;
        msaa_xbox_rt_desc.Layout = static_cast<D3D12_TEXTURE_LAYOUT>(0x101);

        Ref<ID3D12Resource> msaa_unk_rt, msaa_unk_small, msaa_unk_ds, msaa_tiled_rt, msaa_xbox_rt, msaa_cast_rt;
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(msaa_heap.p, D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT,
                                                          &msaa_unk_rt_desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&msaa_unk_rt.p))));
        EXPECT(msaa_unk_rt->GetGPUVirtualAddress() == 0);
        EXPECT(msaa_unk_rt->GetDesc().Alignment == D3D12_DEFAULT_MSAA_RESOURCE_PLACEMENT_ALIGNMENT);
        EXPECT(msaa_unk_rt->GetDesc().SampleDesc.Count == 4);
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 65536, &msaa_unk_small_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&msaa_unk_small.p))));
        EXPECT(msaa_unk_small->GetDesc().Alignment == D3D12_SMALL_MSAA_RESOURCE_PLACEMENT_ALIGNMENT);
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(msaa_heap.p, 8 * 1024 * 1024, &msaa_unk_ds_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&msaa_unk_ds.p))));
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 65536, &msaa_tiled_rt_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&msaa_tiled_rt.p))));
        EXPECT(msaa_tiled_rt->GetDesc().Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
        UINT msaa_tiles = 0;
        D3D12_TILE_SHAPE msaa_shape{};
        standalone->GetResourceTiling(msaa_tiled_rt.p, &msaa_tiles, nullptr, &msaa_shape, nullptr, 0, nullptr);
        EXPECT(msaa_tiles == 16 && msaa_shape.WidthInTexels == 64 && msaa_shape.HeightInTexels == 64);
        EXPECT(SUCCEEDED(standalone->CreatePlacedResource(tiled_heap.p, 65536, &msaa_xbox_rt_desc,
                                                          D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                          IID_PPV_ARGS(&msaa_xbox_rt.p))));
        EXPECT(msaa_xbox_rt->GetDesc().Alignment == D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT);
        EXPECT(static_cast<UINT>(msaa_xbox_rt->GetDesc().Layout) == 0x101);

        {
            Ref<ID3D12Device10> standalone10;
            EXPECT(SUCCEEDED(standalone->QueryInterface(IID_PPV_ARGS(&standalone10.p))));
            D3D12_RESOURCE_DESC1 cast_msaa{};
            cast_msaa.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            cast_msaa.Width = 64;
            cast_msaa.Height = 64;
            cast_msaa.DepthOrArraySize = 1;
            cast_msaa.MipLevels = 1;
            cast_msaa.Format = DXGI_FORMAT_R8G8B8A8_TYPELESS;
            cast_msaa.SampleDesc.Count = 4;
            cast_msaa.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            const DXGI_FORMAT cast_list[] = {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
            EXPECT(SUCCEEDED(standalone10->CreatePlacedResource2(
                msaa_heap.p, 0, &cast_msaa, D3D12_BARRIER_LAYOUT_COMMON, nullptr, 2, cast_list,
                IID_PPV_ARGS(&msaa_cast_rt.p))));
        }

        // Refusals matching host D3D12 for MSAA placed resources.
        {
            struct BadMsaaCase {
                ID3D12Heap* heap;
                UINT64 offset;
                D3D12_RESOURCE_DESC desc;
            };
            D3D12_RESOURCE_DESC msaa_3d =
                msaa_desc(16, 16, 4, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            msaa_3d.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE3D;
            msaa_3d.DepthOrArraySize = 8;
            const BadMsaaCase bad_msaa[] = {
                // 4 MiB-aligned UNKNOWN MSAA fails on a 64 KiB heap (even at offset 0) and at misaligned offsets.
                {tiled_heap.p, 0, msaa_unk_rt_desc},
                {msaa_heap.p, 65536, msaa_unk_rt_desc},
                {msaa_heap.p, msaa_heap_desc.SizeInBytes - 65536, msaa_unk_small_desc},
                {upload_placed_heap.p, 0, msaa_unk_small_desc},
                {deny_rt_heap.p, 0, msaa_unk_small_desc},
                {deny_rt_heap.p, 0, tiled_rt_desc},
                {msaa_heap.p, 0,
                 msaa_desc(256, 256, 4, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_NONE)},
                {msaa_heap.p, 0,
                 msaa_desc(256, 256, 4, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN,
                           D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL)},
                {msaa_heap.p, 0,
                 msaa_desc(64, 64, 4, 0, D3D12_TEXTURE_LAYOUT_UNKNOWN, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                           DXGI_FORMAT_R8G8B8A8_UNORM, 0)},
                {msaa_heap.p, 0, msaa_3d},
            };
            for (const BadMsaaCase& bad : bad_msaa) {
                Ref<ID3D12Resource> fail;
                EXPECT(standalone->CreatePlacedResource(bad.heap, bad.offset, &bad.desc, D3D12_RESOURCE_STATE_COMMON,
                                                        nullptr, IID_PPV_ARGS(&fail.p)) == E_INVALIDARG);
                EXPECT(!fail.p);
            }
            D3D12_HEAP_DESC only_rt_desc = tiled_heap_desc;
            only_rt_desc.Flags = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES;
            Ref<ID3D12Heap> only_rt_heap;
            EXPECT(SUCCEEDED(standalone->CreateHeap(&only_rt_desc, IID_PPV_ARGS(&only_rt_heap.p))));
            for (const D3D12_RESOURCE_DESC& bad_only_rt :
                 {texture_desc(64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1),
                  buffer_desc(4096)}) {
                Ref<ID3D12Resource> fail;
                EXPECT(standalone->CreatePlacedResource(only_rt_heap.p, 0, &bad_only_rt, D3D12_RESOURCE_STATE_COMMON,
                                                        nullptr, IID_PPV_ARGS(&fail.p)) == E_INVALIDARG);
                EXPECT(!fail.p);
            }
        }

        // Clear placed 4x MSAA RTVs (UNKNOWN + tiled) and DSV, resolve, and read back to verify GPU execution.
        {
            const D3D12_RESOURCE_DESC resolved_desc =
                texture_desc(256, 256, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_NONE, 1);
            Ref<ID3D12Resource> msaa_resolved_unk, msaa_resolved_tiled;
            D3D12_HEAP_PROPERTIES def_hp{D3D12_HEAP_TYPE_DEFAULT};
            EXPECT(SUCCEEDED(standalone->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &resolved_desc,
                                                                 D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr,
                                                                 IID_PPV_ARGS(&msaa_resolved_unk.p))));
            EXPECT(SUCCEEDED(standalone->CreateCommittedResource(&def_hp, D3D12_HEAP_FLAG_NONE, &resolved_desc,
                                                                 D3D12_RESOURCE_STATE_RESOLVE_DEST, nullptr,
                                                                 IID_PPV_ARGS(&msaa_resolved_tiled.p))));

            D3D12_DESCRIPTOR_HEAP_DESC msaa_rtv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            D3D12_DESCRIPTOR_HEAP_DESC msaa_dsv_hd{D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 1, D3D12_DESCRIPTOR_HEAP_FLAG_NONE, 0};
            Ref<ID3D12DescriptorHeap> msaa_rtv_heap, msaa_dsv_heap;
            EXPECT(SUCCEEDED(standalone->CreateDescriptorHeap(&msaa_rtv_hd, IID_PPV_ARGS(&msaa_rtv_heap.p))));
            EXPECT(SUCCEEDED(standalone->CreateDescriptorHeap(&msaa_dsv_hd, IID_PPV_ARGS(&msaa_dsv_heap.p))));
            const UINT rtv_step = standalone->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            const D3D12_CPU_DESCRIPTOR_HANDLE rtv0 = msaa_rtv_heap->GetCPUDescriptorHandleForHeapStart();
            const D3D12_CPU_DESCRIPTOR_HANDLE rtv1{rtv0.ptr + rtv_step};
            standalone->CreateRenderTargetView(msaa_unk_rt.p, nullptr, rtv0);
            standalone->CreateRenderTargetView(msaa_tiled_rt.p, nullptr, rtv1);
            standalone->CreateDepthStencilView(msaa_unk_ds.p, nullptr, msaa_dsv_heap->GetCPUDescriptorHandleForHeapStart());

            EXPECT(SUCCEEDED(tiled_alloc->Reset()) && SUCCEEDED(tiled_list->Reset(tiled_alloc.p, nullptr)));
            D3D12_RESOURCE_BARRIER barriers[] = {
                transition(msaa_unk_rt.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition(msaa_tiled_rt.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET),
                transition(msaa_unk_ds.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_DEPTH_WRITE),
            };
            tiled_list->ResourceBarrier(3, barriers);
            tiled_list->ClearRenderTargetView(rtv0, tiled_color, 0, nullptr);
            tiled_list->ClearRenderTargetView(rtv1, tiled_color, 0, nullptr);
            tiled_list->ClearDepthStencilView(msaa_dsv_heap->GetCPUDescriptorHandleForHeapStart(),
                                              D3D12_CLEAR_FLAG_DEPTH, 0.5f, 0, 0, nullptr);
            D3D12_RESOURCE_BARRIER to_resolve[] = {
                transition(msaa_unk_rt.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
                transition(msaa_tiled_rt.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_RESOLVE_SOURCE),
            };
            tiled_list->ResourceBarrier(2, to_resolve);
            tiled_list->ResolveSubresource(msaa_resolved_unk.p, 0, msaa_unk_rt.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            tiled_list->ResolveSubresource(msaa_resolved_tiled.p, 0, msaa_tiled_rt.p, 0, DXGI_FORMAT_R8G8B8A8_UNORM);
            D3D12_RESOURCE_BARRIER to_copy[] = {
                transition(msaa_resolved_unk.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
                transition(msaa_resolved_tiled.p, D3D12_RESOURCE_STATE_RESOLVE_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE),
            };
            tiled_list->ResourceBarrier(2, to_copy);
            Ref<ID3D12Resource> msaa_rb;
            const D3D12_RESOURCE_DESC msaa_rb_desc = buffer_desc(1024);
            EXPECT(SUCCEEDED(standalone->CreateCommittedResource(&rb_props, D3D12_HEAP_FLAG_NONE, &msaa_rb_desc,
                                                                 D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                                                                 IID_PPV_ARGS(&msaa_rb.p))));
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp_unk{}, fp_tiled{};
            fp_unk.Offset = 0;
            fp_unk.Footprint = {DXGI_FORMAT_R8G8B8A8_UNORM, 1, 1, 1, D3D12_TEXTURE_DATA_PITCH_ALIGNMENT};
            fp_tiled = fp_unk;
            fp_tiled.Offset = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT;
            const D3D12_TEXTURE_COPY_LOCATION dst_unk = footprint(msaa_rb.p, fp_unk);
            const D3D12_TEXTURE_COPY_LOCATION src_unk = subresource(msaa_resolved_unk.p, 0);
            const D3D12_TEXTURE_COPY_LOCATION dst_tiled = footprint(msaa_rb.p, fp_tiled);
            const D3D12_TEXTURE_COPY_LOCATION src_tiled = subresource(msaa_resolved_tiled.p, 0);
            tiled_list->CopyTextureRegion(&dst_unk, 0, 0, 0, &src_unk, &box);
            tiled_list->CopyTextureRegion(&dst_tiled, 0, 0, 0, &src_tiled, &box);
            EXPECT(SUCCEEDED(tiled_list->Close()));
            queue->ExecuteCommandLists(1, tiled_lists);
            EXPECT(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
            EXPECT(SUCCEEDED(queue->Signal(fence.p, 9)) && SUCCEEDED(fence->SetEventOnCompletion(9, nullptr)));
            void* msaa_bytes = nullptr;
            EXPECT(SUCCEEDED(msaa_rb->Map(0, nullptr, &msaa_bytes)));
            const auto* out_bytes = static_cast<const std::uint8_t*>(msaa_bytes);
            report("placed MSAA UNKNOWN RTV resolve", 0, out_bytes, want_rgba, 4);
            report("placed MSAA tiled RTV resolve", 0, out_bytes + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, want_rgba, 4);
            EXPECT(std::memcmp(out_bytes, want_rgba, 4) == 0);
            EXPECT(std::memcmp(out_bytes + D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT, want_rgba, 4) == 0);
            msaa_rb->Unmap(0, nullptr);
        }

        // Host parity for MSAA placed acceptance/refusals, layout, alignment, heap flags and tiled 4x shape.
        if (host_dev.p) {
            Ref<ID3D12Heap> host_heap64, host_heap4m;
            if (SUCCEEDED(host_dev->CreateHeap(&tiled_heap_desc, IID_PPV_ARGS(&host_heap64.p))) &&
                SUCCEEDED(host_dev->CreateHeap(&msaa_heap_desc, IID_PPV_ARGS(&host_heap4m.p)))) {
                struct MsaaParityCase {
                    ID3D12Heap* our_heap;
                    ID3D12Heap* host_heap;
                    UINT64 offset;
                    D3D12_RESOURCE_DESC desc;
                };
                const MsaaParityCase parity[] = {
                    {tiled_heap.p, host_heap64.p, 0, msaa_unk_rt_desc},
                    {msaa_heap.p, host_heap4m.p, 65536, msaa_unk_rt_desc},
                    {msaa_heap.p, host_heap4m.p, 0, msaa_unk_rt_desc},
                    {tiled_heap.p, host_heap64.p, 65536, msaa_unk_small_desc},
                    {tiled_heap.p, host_heap64.p, 65536, msaa_tiled_rt_desc},
                };
                for (const MsaaParityCase& mc : parity) {
                    Ref<ID3D12Resource> our_r, host_r;
                    const HRESULT our_hr = standalone->CreatePlacedResource(
                        mc.our_heap, mc.offset, &mc.desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&our_r.p));
                    const HRESULT host_hr = host_dev->CreatePlacedResource(
                        mc.host_heap, mc.offset, &mc.desc, D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&host_r.p));
                    EXPECT(SUCCEEDED(our_hr) == SUCCEEDED(host_hr));
                    if (FAILED(host_hr)) {
                        EXPECT(our_hr == host_hr);
                        continue;
                    }
                    EXPECT(our_r->GetDesc().Layout == host_r->GetDesc().Layout);
                    EXPECT(our_r->GetDesc().Alignment == host_r->GetDesc().Alignment);
                    EXPECT(our_r->GetDesc().SampleDesc.Count == host_r->GetDesc().SampleDesc.Count);
                    D3D12_HEAP_PROPERTIES ohp{}, hhp{};
                    EXPECT(SUCCEEDED(our_r->GetHeapProperties(&ohp, nullptr)) &&
                           SUCCEEDED(host_r->GetHeapProperties(&hhp, nullptr)));
                    EXPECT(ohp.Type == hhp.Type);
                    if (mc.desc.Layout == D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE) {
                        D3D12_TILE_SHAPE os{}, hs{};
                        standalone->GetResourceTiling(our_r.p, nullptr, nullptr, &os, nullptr, 0, nullptr);
                        host_dev->GetResourceTiling(host_r.p, nullptr, nullptr, &hs, nullptr, 0, nullptr);
                        EXPECT(os.WidthInTexels == hs.WidthInTexels && os.HeightInTexels == hs.HeightInTexels &&
                               os.DepthInTexels == hs.DepthInTexels);
                    }
                }
            }
        }
    }

    // D3D-90, the placed half of castable formats: CreatePlacedResource2 takes NumCastableFormats exactly as
    // CreateCommittedResource3 does (#693). A castable list on a placed texture makes a mutable image whose views
    // may cast: an SRV descriptor is written only for listed formats, an RTV in a listed family format clears
    // through the cast, and the cleared bytes are compared with the host PC runtime's (WARP when the default host
    // adapter cannot cast). Refused lists give the host's HRESULT, and a placement without a list still works.
#define CAST_CHECK(condition) \
    do { \
        if (!(condition)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); return false; } \
    } while (0)
    const auto placed_cast_scenario = [](ID3D12Device10* device10, bool layer, std::uint8_t* out) -> bool {
        D3D12_HEAP_DESC heap_desc{};
        heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap_desc.SizeInBytes = 256 * 1024;
        heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        Ref<ID3D12Heap> heap;
        CAST_CHECK(SUCCEEDED(device10->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
        D3D12_RESOURCE_DESC1 texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = 4;
        texture.Height = 4;
        texture.MipLevels = 1;
        texture.DepthOrArraySize = 1;
        texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        const DXGI_FORMAT srgb_list[] = {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
        Ref<ID3D12Resource> srgb, plain;
        CAST_CHECK(SUCCEEDED(device10->CreatePlacedResource2(heap.p, 0, &texture, D3D12_BARRIER_LAYOUT_COMMON,
                                                              nullptr, 1, srgb_list, IID_PPV_ARGS(&srgb.p))));
        // The control: a count with a null list is no list, so this placement works as CreatePlacedResource's.
        CAST_CHECK(SUCCEEDED(device10->CreatePlacedResource2(heap.p, 65536, &texture, D3D12_BARRIER_LAYOUT_COMMON,
                                                              nullptr, 2, nullptr, IID_PPV_ARGS(&plain.p))));
        D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
        rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
        rtv_heap_desc.NumDescriptors = 2;
        Ref<ID3D12DescriptorHeap> rtv_heap;
        CAST_CHECK(SUCCEEDED(device10->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
        const UINT rtv_stride = device10->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
        D3D12_RENDER_TARGET_VIEW_DESC srgb_view{}, plain_view{};
        srgb_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
        plain_view.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        srgb_view.ViewDimension = plain_view.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
        device10->CreateRenderTargetView(srgb.p, &srgb_view, rtv);
        device10->CreateRenderTargetView(plain.p, &plain_view, {rtv.ptr + rtv_stride});
        D3D12_COMMAND_QUEUE_DESC queue_desc{};
        Ref<ID3D12CommandQueue> queue;
        Ref<ID3D12Fence> fence;
        Ref<ID3D12CommandAllocator> allocator;
        Ref<ID3D12GraphicsCommandList> list;
        CAST_CHECK(SUCCEEDED(device10->CreateCommandQueue(&queue_desc, IID_PPV_ARGS(&queue.p))));
        CAST_CHECK(SUCCEEDED(device10->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        CAST_CHECK(SUCCEEDED(device10->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                              IID_PPV_ARGS(&allocator.p))));
        CAST_CHECK(SUCCEEDED(device10->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator.p, nullptr,
                                                         IID_PPV_ARGS(&list.p))));
        const FLOAT quarter[4] = {0.25f, 0.5f, 0.75f, 1.0f};
        D3D12_RESOURCE_BARRIER barrier = transition(srgb.p, D3D12_RESOURCE_STATE_COMMON,
                                                    D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &barrier);
        list->ClearRenderTargetView(rtv, quarter, 0, nullptr);
        barrier = transition(srgb.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        barrier = transition(plain.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_RENDER_TARGET);
        list->ResourceBarrier(1, &barrier);
        list->ClearRenderTargetView({rtv.ptr + rtv_stride}, quarter, 0, nullptr);
        barrier = transition(plain.p, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE);
        list->ResourceBarrier(1, &barrier);
        D3D12_HEAP_PROPERTIES readback_heap{D3D12_HEAP_TYPE_READBACK};
        const D3D12_RESOURCE_DESC readback_desc = buffer_desc(2048);
        Ref<ID3D12Resource> readback;
        CAST_CHECK(SUCCEEDED(device10->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE,
                                                               &readback_desc, D3D12_RESOURCE_STATE_COPY_DEST,
                                                               nullptr, IID_PPV_ARGS(&readback.p))));
        for (const auto& [resource, offset] : {std::tuple{srgb.p, 0u}, std::tuple{plain.p, 1024u}}) {
            D3D12_PLACED_SUBRESOURCE_FOOTPRINT placed{};
            placed.Offset = offset;
            placed.Footprint.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            placed.Footprint.Width = 4;
            placed.Footprint.Height = 4;
            placed.Footprint.Depth = 1;
            placed.Footprint.RowPitch = 256;
            const D3D12_TEXTURE_COPY_LOCATION dst = footprint(readback.p, placed);
            const D3D12_TEXTURE_COPY_LOCATION src = subresource(resource, 0);
            list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
        }
        CAST_CHECK(SUCCEEDED(list->Close()));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        if (layer) CAST_CHECK(rt::d3d12::last_execute_result(queue.p) == leonida::ReplayResult::success);
        CAST_CHECK(SUCCEEDED(queue->Signal(fence.p, 1)));
        CAST_CHECK(SUCCEEDED(fence->SetEventOnCompletion(1, nullptr)));
        void* bytes = nullptr;
        CAST_CHECK(SUCCEEDED(readback->Map(0, nullptr, &bytes)));
        std::memcpy(out, bytes, 2048);
        readback->Unmap(0, nullptr);
        return true;
    };
    // The refusal answers, also run on the host: a castable list on a buffer, an unmapped format, and a cast
    // whose texel size differs from the resource's.
    const auto placed_cast_refusals = [](ID3D12Device10* device10, std::array<HRESULT, 3>& out) -> bool {
        D3D12_HEAP_DESC heap_desc{};
        heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
        heap_desc.SizeInBytes = 256 * 1024;
        heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
        Ref<ID3D12Heap> heap;
        if (FAILED(device10->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap.p)))) return false;
        D3D12_RESOURCE_DESC1 texture{};
        texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
        texture.Width = 4;
        texture.Height = 4;
        texture.MipLevels = 1;
        texture.DepthOrArraySize = 1;
        texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        texture.SampleDesc.Count = 1;
        texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
        D3D12_RESOURCE_DESC1 buffer{};
        buffer.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        buffer.Width = 4096;
        buffer.Height = 1;
        buffer.DepthOrArraySize = 1;
        buffer.MipLevels = 1;
        buffer.SampleDesc.Count = 1;
        buffer.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        const DXGI_FORMAT srgb_list[] = {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
        const DXGI_FORMAT unknown_list[] = {DXGI_FORMAT_UNKNOWN};
        const DXGI_FORMAT wide_list[] = {DXGI_FORMAT_R16G16B16A16_FLOAT};
        Ref<ID3D12Resource> refused;
        out[0] = device10->CreatePlacedResource2(heap.p, 0, &buffer, D3D12_BARRIER_LAYOUT_UNDEFINED, nullptr, 1,
                                                 srgb_list, IID_PPV_ARGS(&refused.p));
        out[1] = device10->CreatePlacedResource2(heap.p, 0, &texture, D3D12_BARRIER_LAYOUT_COMMON, nullptr, 1,
                                                 unknown_list, IID_PPV_ARGS(&refused.p));
        out[2] = device10->CreatePlacedResource2(heap.p, 0, &texture, D3D12_BARRIER_LAYOUT_COMMON, nullptr, 1,
                                                 wide_list, IID_PPV_ARGS(&refused.p));
        return !refused.p;
    };
    {
        Ref<ID3D12Device> cast;
        EXPECT(SUCCEEDED(rt::d3d12::create_vulkan_device_standalone(IID_PPV_ARGS(&cast.p))));
        Ref<ID3D12Device10> cast10;
        EXPECT(SUCCEEDED(cast.p->QueryInterface(IID_PPV_ARGS(&cast10.p))));
        std::array<std::uint8_t, 2048> ours{}, host{};
        std::array<HRESULT, 3> our_refusals{};
        EXPECT(placed_cast_scenario(cast10.p, true, ours.data()));
        EXPECT(placed_cast_refusals(cast10.p, our_refusals));
        EXPECT(our_refusals[0] == E_INVALIDARG && our_refusals[1] == E_INVALIDARG);

        // CPU-side: the SRV allow-list on placed textures, and the RTV's, as on the committed path.
        {
            D3D12_HEAP_DESC heap_desc{};
            heap_desc.Properties.Type = D3D12_HEAP_TYPE_DEFAULT;
            heap_desc.SizeInBytes = 256 * 1024;
            heap_desc.Alignment = D3D12_DEFAULT_RESOURCE_PLACEMENT_ALIGNMENT;
            Ref<ID3D12Heap> heap;
            EXPECT(SUCCEEDED(cast10->CreateHeap(&heap_desc, IID_PPV_ARGS(&heap.p))));
            D3D12_RESOURCE_DESC1 texture{};
            texture.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
            texture.Width = 4;
            texture.Height = 4;
            texture.MipLevels = 1;
            texture.DepthOrArraySize = 1;
            texture.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
            texture.SampleDesc.Count = 1;
            texture.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
            const DXGI_FORMAT srgb_list[] = {DXGI_FORMAT_R8G8B8A8_UNORM_SRGB};
            Ref<ID3D12Resource> listed;
            EXPECT(SUCCEEDED(cast10->CreatePlacedResource2(heap.p, 0, &texture, D3D12_BARRIER_LAYOUT_COMMON,
                                                           nullptr, 1, srgb_list, IID_PPV_ARGS(&listed.p))));
            EXPECT(listed->GetGPUVirtualAddress() == 0);  // a placed texture keeps a null VA, castable or not
            D3D12_RESOURCE_DESC1 uint_texture = texture;
            uint_texture.Format = DXGI_FORMAT_R8G8B8A8_UINT;
            const DXGI_FORMAT uint_list[] = {DXGI_FORMAT_R32_UINT};
            Ref<ID3D12Resource> cross_family;
            EXPECT(SUCCEEDED(cast10->CreatePlacedResource2(heap.p, 65536, &uint_texture,
                                                           D3D12_BARRIER_LAYOUT_COMMON, nullptr, 1, uint_list,
                                                           IID_PPV_ARGS(&cross_family.p))));
            D3D12_DESCRIPTOR_HEAP_DESC srv_heap_desc{};
            srv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
            srv_heap_desc.NumDescriptors = 3;
            srv_heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
            Ref<ID3D12DescriptorHeap> srv_heap;
            EXPECT(SUCCEEDED(cast10->CreateDescriptorHeap(&srv_heap_desc, IID_PPV_ARGS(&srv_heap.p))));
            const UINT stride = cast10->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            D3D12_CPU_DESCRIPTOR_HANDLE slot = srv_heap->GetCPUDescriptorHandleForHeapStart();
            D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
            srv.ViewDimension = D3D12_SRV_DIMENSION_TEXTURE2D;
            srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
            srv.Texture2D.MipLevels = 1;
            const auto slot_nonzero = [&]() {
                std::uint32_t words[8]{};
                std::memcpy(words, reinterpret_cast<void*>(slot.ptr), sizeof(words));
                for (std::uint32_t word : words)
                    if (word) return true;
                return false;
            };
            srv.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;  // listed: written
            cast.p->CreateShaderResourceView(listed.p, &srv, slot);
            EXPECT(slot_nonzero());
            srv.Format = DXGI_FORMAT_R8G8B8A8_UINT;  // unlisted: a null descriptor
            slot.ptr += stride;
            cast.p->CreateShaderResourceView(listed.p, &srv, slot);
            std::uint32_t words[8]{};
            std::memcpy(words, reinterpret_cast<void*>(slot.ptr), sizeof(words));
            EXPECT(words[0] == 0 && words[1] == 0 && words[2] == 0 && words[3] == 0 && words[4] == 0 &&
                   words[5] == 0 && words[6] == 0 && words[7] == 0);
            srv.Format = DXGI_FORMAT_R32_UINT;  // a listed same-size cross-family cast: written
            slot.ptr += stride;
            cast.p->CreateShaderResourceView(cross_family.p, &srv, slot);
            EXPECT(slot_nonzero());
            // The RTV allow-list: a listed family format is a view, an unlisted one is not.
            D3D12_DESCRIPTOR_HEAP_DESC rtv_heap_desc{};
            rtv_heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_RTV;
            rtv_heap_desc.NumDescriptors = 2;
            Ref<ID3D12DescriptorHeap> rtv_heap;
            EXPECT(SUCCEEDED(cast10->CreateDescriptorHeap(&rtv_heap_desc, IID_PPV_ARGS(&rtv_heap.p))));
            const UINT rtv_stride = cast10->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
            D3D12_CPU_DESCRIPTOR_HANDLE rtv = rtv_heap->GetCPUDescriptorHandleForHeapStart();
            D3D12_RENDER_TARGET_VIEW_DESC rtv_desc{};
            rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UNORM_SRGB;
            rtv_desc.ViewDimension = D3D12_RTV_DIMENSION_TEXTURE2D;
            cast.p->CreateRenderTargetView(listed.p, &rtv_desc, rtv);
            std::memcpy(words, reinterpret_cast<void*>(rtv.ptr), sizeof(words));
            EXPECT(words[0] != 0 || words[1] != 0);
            rtv_desc.Format = DXGI_FORMAT_R8G8B8A8_UINT;  // unlisted
            cast.p->CreateRenderTargetView(listed.p, &rtv_desc, {rtv.ptr + rtv_stride});
            std::memcpy(words, reinterpret_cast<void*>(rtv.ptr + rtv_stride), sizeof(words));
            EXPECT(words[0] == 0 && words[1] == 0 && words[2] == 0 && words[3] == 0);
        }

        // The tolerance host_side applies, pinned here so every GPU checks it: its accept side only runs where the
        // host rounds differently (Intel's SRGB clear), so a change to it would otherwise stay green elsewhere.
        // One code either side of a near-tie is accepted; two codes, or a code past 0.6 ULP, are not.
        EXPECT(within_06_ulp(187, 188, ideal_srgb(0.5) * 255.0) && within_06_ulp(127, 128, 127.5));
        EXPECT(!within_06_ulp(186, 188, ideal_srgb(0.5) * 255.0) && !within_06_ulp(126, 128, 127.5));
        EXPECT(!within_06_ulp(224, 225, ideal_srgb(0.75) * 255.0) &&
               !within_06_ulp(136, 137, ideal_srgb(0.25) * 255.0));
        EXPECT(!within_06_ulp(254, 255, 255.0));

        // The host PC runtime's bytes and refusals: the default adapter when it can cast, WARP otherwise.
        bool compared = false;
        const char* source = "no host";
        Ref<ID3D12Device> host_device;
        const auto host_side = [&](ID3D12Device10* device10, bool same_adapter) {
            std::array<HRESULT, 3> host_refusals{};
            if (!placed_cast_scenario(device10, false, host.data())) return false;
            if (!placed_cast_refusals(device10, host_refusals)) return false;
            unsigned differences = 0;
            // D3D12 does not pin float -> UNORM or float -> SRGB to one code: the D3D11.3 functional spec
            // (3.2.3.6 FLOAT -> UNORM, 3.2.3.8 FLOAT -> SRGB) permits a tolerance of 0.6 ULP on the integer side, so
            // a value near halfway between two codes may map to either. The plain clear's 0.5 * 255 = 127.5 is a
            // tie: WARP and AMD give the upper code, NVIDIA's D3D12 driver and its Vulkan driver the lower (RTX 3080
            // 616.64, RTX 3090 610.88 and RTX 5070 Laptop 617.42). The SRGB-cast clear's 0.5 encodes to 187.516:
            // WARP and NVIDIA's Vulkan driver give 188, Intel's Vulkan driver 187 (UHD 770 101.7092), both within
            // 0.6 ULP. On the layer's own adapter (same LUID) the comparison stays exact; against another adapter
            // or WARP a byte may differ when both sides are within 0.6 ULP of the ideal value.
            // The plain clear's normalised values are rounded by the layer itself (nearest even, WARP's and AMD's
            // code), so on its own NVIDIA adapter the tie byte may be the step above the driver's: accepted only
            // when it is exactly that nearest-even code and within 0.6 ULP.
            const FLOAT clear_value[4] = {0.25f, 0.5f, 0.75f, 1.0f};  // placed_cast_scenario's clear colour
            const auto within_tolerance = [&](std::uint8_t ours_byte, std::uint8_t host_byte, double ideal) {
                return !same_adapter && within_06_ulp(ours_byte, host_byte, ideal);
            };
            unsigned plain_within = 0, srgb_within = 0;
            for (UINT row = 0; row < 4; ++row)
                for (UINT i = 0; i < 16; ++i) {
                    const UINT a = row * 256 + i, b = 1024 + row * 256 + i;
                    const double c = clear_value[i % 4];
                    const double plain_ideal = c * 255.0;
                    const double srgb_ideal = (i % 4 == 3 ? c : ideal_srgb(c)) * 255.0;  // alpha stays linear
                    if (ours[b] != host[b]) {
                        const bool layer_tie = double(ours[b]) == nearest_even_code(plain_ideal) &&
                                               within_06_ulp(ours[b], host[b], plain_ideal);
                        if (within_tolerance(ours[b], host[b], plain_ideal) || layer_tie)
                            ++plain_within;
                        else if (differences++ < 8)
                            std::fprintf(stderr, "placed plain clear row %u byte %u: ours %02x, host %02x\n", row, i,
                                         ours[b], host[b]);
                    }
                    if (ours[a] != host[a]) {
                        if (within_tolerance(ours[a], host[a], srgb_ideal))
                            ++srgb_within;
                        else if (differences++ < 8)
                            std::fprintf(stderr, "placed SRGB-cast clear row %u byte %u: ours %02x, host %02x\n",
                                         row, i, ours[a], host[a]);
                    }
                }
            if (plain_within || srgb_within)
                std::printf("placed clears: %u plain and %u SRGB-cast bytes differ from the host by one code, both "
                            "within the 0.6 ULP tolerance of D3D11.3 spec 3.2.3.6 and 3.2.3.8\n",
                            plain_within, srgb_within);
            for (UINT i = 0; i < 3; ++i)
                if (our_refusals[i] != host_refusals[i]) {
                    ++differences;
                    std::fprintf(stderr, "placed castable refusal %u: ours %08lx, host %08lx\n", i,
                                 static_cast<unsigned long>(our_refusals[i]),
                                 static_cast<unsigned long>(host_refusals[i]));
                }
            return differences == 0;
        };
        const auto can_cast = [&](ID3D12Device* device, Ref<ID3D12Device10>& out10) {
            D3D12_FEATURE_DATA_D3D12_OPTIONS12 options12{};
            return SUCCEEDED(device->QueryInterface(IID_PPV_ARGS(&out10.p))) &&
                   SUCCEEDED(out10->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12, &options12,
                                                        sizeof(options12))) &&
                   options12.RelaxedFormatCastingSupported;
        };
        // The host is the layer's own adapter (matched by LUID, then PCI ids) unless LEONIDA_HOST_ADAPTER names
        // one: drivers round float -> UNORM ties differently, so the same GPU is the fair reference.
        std::string host_adapter, host_source;
        char host_override[8] = {};
        const bool named = GetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", host_override, sizeof(host_override)) > 0;
        const bool have_host =
            named ? create_host_device(host_device, &host_adapter)
                  : create_host_device_for_physical(d.instance, d.physical, d.instance_proc, host_device, host_adapter);
        if (have_host) {
            VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
            VkPhysicalDeviceProperties2 properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
            d.instance_fn<PFN_vkGetPhysicalDeviceProperties2>("vkGetPhysicalDeviceProperties2")(d.physical, &properties);
            const LUID host_luid = host_device->GetAdapterLuid();
            const bool same_adapter =
                id.deviceLUIDValid && std::memcmp(&host_luid, id.deviceLUID, sizeof(host_luid)) == 0;
            Ref<ID3D12Device10> host10;
            if (can_cast(host_device.p, host10)) {
                EXPECT(host_side(host10.p, same_adapter));
                compared = true;
                host_source = "the host runtime on " + host_adapter +
                              (same_adapter ? " (the layer's own adapter)" : " (a different adapter from the layer's)");
                source = host_source.c_str();
            }
        }
        std::string warp_name;
        if (!compared) {
            // WARP, the host path that casts when the default adapter cannot. LEONIDA_HOST_ADAPTER is saved
            // and restored: it belongs to whoever runs the test.
            char previous[128] = {};
            const BOOL had = GetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", previous, sizeof(previous));
            SetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", "warp");
            copycmp::Ptr<ID3D12Device> warp;

            Ref<ID3D12Device10> warp10;
            if (copycmp::try_create_host(warp, warp_name) && can_cast(warp.p, warp10)) {
                EXPECT(host_side(warp10.p, false));
                compared = true;
                source = warp_name.c_str();
            }
            SetEnvironmentVariableA("LEONIDA_HOST_ADAPTER", had && previous[0] ? previous : nullptr);
        }
        std::printf("placed castable formats (D3D-90): SRGB-cast and plain clears read back and compared with %s; "
                    "refusals %s\n",
                    source, compared ? "matched the host's" : "checked on the layer alone");
    }
#undef CAST_CHECK

    if (d.validation_errors()) {
        std::fprintf(stderr, "%u validation errors\n", d.validation_errors());
        return 1;
    }
    std::printf("Vulkan device behind the Xbox-ABI proxy: clears, zero-filled first use, uploads, texture, buffer and "
                "BC copies, a resolve, reserved resources, queries, GetCopyableFootprints, GetResourceTiling and "
                "CheckFeatureSupport error paths and fence semantics all matched\n");
    return 0;
}
