// 96-bit R32G32B32 textures on the Vulkan-backed device (D3D-62), compared with Windows D3D12/WARP.
//
// The device holds them in R32G32B32A32 images and repacks 12-byte texels in a compute pass (the depth24 pass,
// Depth24Job::rgb96) on every path that meets a buffer or the CPU. Checked here, for FLOAT, UINT, SINT and TYPELESS, as a 2D array with mips and as a volume:
//   - GetCopyableFootprints against WARP (12-byte texels, D3D12's pitches and offsets);
//   - uploads of every subresource, an upload of a box to an offset, CopyResource, a texture-to-texture box,
//     readbacks of every subresource and of a box to an offset inside its footprint: every texel bit for bit
//     against the values the test wrote (NaN payloads, denormals, -0 and full-range integers among them) and
//     against WARP, and the readback buffer's bytes between texels left as they were;
//   - WriteToSubresource/ReadFromSubresource (always the staged copy for these textures), on a device with
//     host image copies and on one without, before and after a barrier has moved the texture;
//   - what creation, GetResourceAllocationInfo and CheckFeatureSupport say, next to WARP's answers, and that
//     the three agree with each other;
//   - copies between a 96-bit texture and an RGBA32 one refused;
//   - an upload and a readback on a COPY queue;
//   - how long an upload and a readback take at 64x64 and 256x256, and up to 4096x4096 with LEONIDA_RGB96_LARGE
//     set (single runs, each on a new queue).
// No game data or shaders. Run explicitly (needs Vulkan hardware); LEONIDA_VK_VALIDATION=1 checks Vulkan usage.
#include "host_wine.h"
#include "vk_test_device.h"
#include "../shims/d3d12_com.h"
#include <dxgi1_4.h>
#include <chrono>
#include <cstdint>
#include <cstring>
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

constexpr UINT kTexel = 12;                  // bytes of an R32G32B32 texel in a footprint
constexpr std::uint8_t kUntouched = 0xee;    // what the readback buffer holds before a copy

// Channel c of the texel at (x, y, z) of subresource `sub`. The first texels of each subresource are bit patterns
// a float conversion would not keep: signalling and quiet NaNs with payloads, a denormal, -0, the integer extremes.
std::uint32_t value(UINT sub, UINT x, UINT y, UINT z, UINT c, UINT salt = 0) {
    static constexpr std::uint32_t special[] = {0x7fa00001, 0xffc12345, 0x00000001, 0x80000000, 0xffffffff,
                                                0x80000001, 0x7f800000, 0x007fffff, 0x7fffffff};
    const UINT index = (x + y * 5 + z * 15) * 3 + c;
    if (!salt && index < std::size(special)) return special[(index + sub) % std::size(special)];
    std::uint32_t v = (sub + 1) * 0x9e3779b9u ^ (x * 0x85ebca6bu) ^ (y * 0xc2b2ae35u) ^ (z * 0x27d4eb2fu) ^
                      (c * 0x165667b1u) ^ (salt * 0xd6e8feb8u);
    v ^= v >> 15;
    v *= 0x2c1b3c6du;
    return v ^ v >> 12;
}

D3D12_RESOURCE_DESC buffer_desc(UINT64 size) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    d.Width = size;
    d.Height = d.DepthOrArraySize = d.MipLevels = 1;
    d.SampleDesc.Count = 1;
    d.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    return d;
}

D3D12_RESOURCE_DESC texture_desc(DXGI_FORMAT format, bool volume, UINT width, UINT height, UINT16 depth_or_layers,
                                 UINT16 mips, D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension = volume ? D3D12_RESOURCE_DIMENSION_TEXTURE3D : D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    d.Width = width;
    d.Height = height;
    d.DepthOrArraySize = depth_or_layers;
    d.MipLevels = mips;
    d.Format = format;
    d.SampleDesc.Count = 1;
    d.Flags = flags;
    return d;
}

// A queue, a list and a fence: record with `list`, then run().
struct Gpu {
    ID3D12Device* device;
    Ref<ID3D12CommandQueue> queue;
    Ref<ID3D12CommandAllocator> allocator;
    Ref<ID3D12GraphicsCommandList> list;
    Ref<ID3D12Fence> fence;
    UINT64 signalled = 0;

    bool create(ID3D12Device* d, D3D12_COMMAND_LIST_TYPE type = D3D12_COMMAND_LIST_TYPE_DIRECT) {
        device = d;
        D3D12_COMMAND_QUEUE_DESC desc{};
        desc.Type = type;
        CHECK(SUCCEEDED(d->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue.p))));
        CHECK(SUCCEEDED(d->CreateCommandAllocator(type, IID_PPV_ARGS(&allocator.p))));
        CHECK(SUCCEEDED(d->CreateCommandList(0, type, allocator.p, nullptr, IID_PPV_ARGS(&list.p))));
        CHECK(SUCCEEDED(d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence.p))));
        return true;
    }
    bool run() {
        CHECK(SUCCEEDED(list->Close()));
        ID3D12CommandList* lists[] = {list.p};
        queue->ExecuteCommandLists(1, lists);
        CHECK(SUCCEEDED(queue->Signal(fence.p, ++signalled)));
        const HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        CHECK(event);
        const bool done = SUCCEEDED(fence->SetEventOnCompletion(signalled, event)) &&
                          WaitForSingleObject(event, 60000) == WAIT_OBJECT_0;
        CloseHandle(event);
        CHECK(done);
        CHECK(SUCCEEDED(allocator->Reset()));
        CHECK(SUCCEEDED(list->Reset(allocator.p, nullptr)));
        return true;
    }
    void transition(ID3D12Resource* resource, D3D12_RESOURCE_STATES before, D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        list->ResourceBarrier(1, &barrier);
    }
};

D3D12_TEXTURE_COPY_LOCATION subresource(ID3D12Resource* resource, UINT index) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    location.SubresourceIndex = index;
    return location;
}
D3D12_TEXTURE_COPY_LOCATION placed(ID3D12Resource* resource, const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& footprint) {
    D3D12_TEXTURE_COPY_LOCATION location{};
    location.pResource = resource;
    location.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
    location.PlacedFootprint = footprint;
    return location;
}

struct Shape {
    DXGI_FORMAT format;
    bool volume;
    static constexpr UINT width = 5, height = 3, mips = 2;  // odd sizes: rows end inside the 256-byte pitch
    UINT16 depth_or_layers() const { return volume ? 4 : 2; }
    UINT subresources() const { return volume ? mips : mips * depth_or_layers(); }
    UINT mip(UINT sub) const { return sub % mips; }
    UINT w(UINT sub) const { return std::max(1u, width >> mip(sub)); }
    UINT h(UINT sub) const { return std::max(1u, height >> mip(sub)); }
    UINT d(UINT sub) const { return volume ? std::max(1u, UINT(depth_or_layers()) >> mip(sub)) : 1; }
};

struct Footprints {
    std::vector<D3D12_PLACED_SUBRESOURCE_FOOTPRINT> layouts;
    std::vector<UINT> rows;
    std::vector<UINT64> row_sizes;
    UINT64 total = 0;
    bool operator==(const Footprints& o) const {
        return total == o.total && rows == o.rows && row_sizes == o.row_sizes &&
               std::memcmp(layouts.data(), o.layouts.data(), layouts.size() * sizeof(layouts[0])) == 0;
    }
};

bool host_reports_no_tiling(ID3D12Device* device, const char* device_name, DXGI_FORMAT format) {
    const D3D12_RESOURCE_DESC desc = texture_desc(format, false, 256, 256, 1, 1);
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    Ref<ID3D12Resource> resource;
    const HRESULT created = device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                             D3D12_RESOURCE_STATE_COMMON, nullptr,
                                                             IID_PPV_ARGS(&resource.p));
    CHECK(SUCCEEDED(created));
    UINT total = 0xdeadbeefu, count = 1;
    D3D12_PACKED_MIP_INFO packed{0xdeu, 0xadu, 0xdeadbeefu, 0xdeadbeefu};
    D3D12_TILE_SHAPE shape{0xdeadbeefu, 0xdeadbeefu, 0xdeadbeefu};
    const D3D12_SUBRESOURCE_TILING untouched{0xdeadbeefu, 0xbeefu, 0xbeefu, 0xdeadbeefu};
    D3D12_SUBRESOURCE_TILING subresource = untouched;
    device->GetResourceTiling(resource.p, &total, &packed, &shape, &count, 0, &subresource);
    if (total || shape.WidthInTexels || shape.HeightInTexels || shape.DepthInTexels ||
        packed.NumStandardMips || packed.NumPackedMips || packed.NumTilesForPackedMips ||
        packed.StartTileIndexInOverallResource || count ||
        std::memcmp(&subresource, &untouched, sizeof(subresource)) != 0) {
        std::fprintf(stderr, "%s format %d: unexpected tiling: total %u, shape %ux%ux%u, packed %u/%u/%u/%u, "
                     "subresources %u\n", device_name, int(format), total, shape.WidthInTexels,
                     shape.HeightInTexels, shape.DepthInTexels, packed.NumStandardMips, packed.NumPackedMips,
                     packed.NumTilesForPackedMips, packed.StartTileIndexInOverallResource, count);
        return false;
    }
    return true;
}

// What the copies below leave in the second texture, and in the box read back from it: three words a texel.
struct Texels {
    std::vector<std::vector<std::uint32_t>> subresources;
    std::vector<std::uint32_t> box;  // 3x2 texels
    bool operator==(const Texels&) const = default;
};

// The boxes: an upload of the 2x2 texels at (1,0) of a 3x2 footprint to (2,1) of subresource 0; a 2x1 box at (3,1)
// of the first texture's subresource 0 to the second texture's subresource 1 (mip 1, 2x1); a readback of the 3x2
// box at (1,1) of the second texture's subresource 0 to (1,1) of a 4x3 footprint.
Texels expected(const Shape& s) {
    Texels t;
    t.subresources.resize(s.subresources());
    const auto at = [&](UINT sub, UINT x, UINT y, UINT z) {
        return t.subresources[sub].data() + ((z * s.h(sub) + y) * s.w(sub) + x) * 3;
    };
    for (UINT sub = 0; sub < s.subresources(); ++sub) {
        t.subresources[sub].resize(std::size_t(s.w(sub)) * s.h(sub) * s.d(sub) * 3);
        for (UINT z = 0; z < s.d(sub); ++z)
            for (UINT y = 0; y < s.h(sub); ++y)
                for (UINT x = 0; x < s.w(sub); ++x)
                    for (UINT c = 0; c < 3; ++c) at(sub, x, y, z)[c] = value(sub, x, y, z, c);
    }
    for (UINT y = 0; y < 2; ++y)
        for (UINT x = 0; x < 2; ++x)
            for (UINT c = 0; c < 3; ++c) at(0, 2 + x, 1 + y, 0)[c] = value(0, 1 + x, y, 0, c, 1);
    const std::vector<std::uint32_t> first = t.subresources[0];  // the first texture's subresource 0
    for (UINT x = 0; x < 2; ++x)
        for (UINT c = 0; c < 3; ++c) at(1, x, 0, 0)[c] = first[((1 * Shape::width) + 3 + x) * 3 + c];
    for (UINT y = 0; y < 2; ++y)
        for (UINT x = 0; x < 3; ++x)
            for (UINT c = 0; c < 3; ++c) t.box.push_back(at(0, 1 + x, 1 + y, 0)[c]);
    return t;
}

bool copies(ID3D12Device* device, const Shape& s, Footprints& footprints, Texels& out, bool check_untouched) {
    Gpu gpu;
    CHECK(gpu.create(device));
    const D3D12_RESOURCE_DESC desc = texture_desc(s.format, s.volume, Shape::width, Shape::height, s.depth_or_layers(), Shape::mips);
    const UINT count = s.subresources();
    footprints.layouts.resize(count);
    footprints.rows.resize(count);
    footprints.row_sizes.resize(count);
    device->GetCopyableFootprints(&desc, 0, count, 0, footprints.layouts.data(), footprints.rows.data(),
                                  footprints.row_sizes.data(), &footprints.total);
    CHECK(footprints.total && footprints.total != ~0ull);
    for (UINT sub = 0; sub < count; ++sub) {
        CHECK(footprints.row_sizes[sub] == UINT64(s.w(sub)) * kTexel);
        CHECK(footprints.layouts[sub].Footprint.Format == s.format);
    }

    // After the subresources: a 3x2 footprint for the box upload, a 4x3 one for the box readback.
    const UINT64 extra = (footprints.total + 511) / 512 * 512;
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT box_upload{extra, {s.format, 3, 2, 1, 256}};
    const D3D12_PLACED_SUBRESOURCE_FOOTPRINT box_readback{extra, {s.format, 4, 3, 1, 256}};
    const UINT64 size = extra + 3 * 256;

    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
        readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC bytes = buffer_desc(size);
    Ref<ID3D12Resource> first, second, upload, readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&first.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&second.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));

    std::uint8_t* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped);
    std::memset(mapped, 0xcd, size);
    const auto put = [&](const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& f, UINT x, UINT y, UINT z, UINT c, std::uint32_t v) {
        const UINT64 slice = UINT64(f.Footprint.RowPitch) * f.Footprint.Height;
        std::memcpy(mapped + f.Offset + z * slice + UINT64(y) * f.Footprint.RowPitch + x * kTexel + c * 4, &v, 4);
    };
    for (UINT sub = 0; sub < count; ++sub)
        for (UINT z = 0; z < s.d(sub); ++z)
            for (UINT y = 0; y < s.h(sub); ++y)
                for (UINT x = 0; x < s.w(sub); ++x)
                    for (UINT c = 0; c < 3; ++c) put(footprints.layouts[sub], x, y, z, c, value(sub, x, y, z, c));
    for (UINT y = 0; y < 2; ++y)
        for (UINT x = 0; x < 3; ++x)
            for (UINT c = 0; c < 3; ++c) put(box_upload, x, y, 0, c, value(0, x, y, 0, c, 1));
    upload->Unmap(0, nullptr);
    // D3D12 does not let the CPU write a readback buffer, but its memory is plain memory on both devices: fill it,
    // so bytes a copy should not touch (between a row's texels and its pitch, around a box) can be seen to stay.
    CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped);
    std::memset(mapped, kUntouched, size);
    readback->Unmap(0, nullptr);

    for (UINT sub = 0; sub < count; ++sub) {
        const auto dst = subresource(first.p, sub), src = placed(upload.p, footprints.layouts[sub]);
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    // The box rewrites texels the upload before it wrote, in the same list: the later copy must win.
    {
        const auto dst = subresource(first.p, 0), src = placed(upload.p, box_upload);
        const D3D12_BOX box{1, 0, 0, 3, 2, 1};
        gpu.list->CopyTextureRegion(&dst, 2, 1, 0, &src, &box);
    }
    gpu.transition(first.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    gpu.list->CopyResource(second.p, first.p);
    {
        const auto dst = subresource(second.p, 1), src = subresource(first.p, 0);
        const D3D12_BOX box{3, 1, 0, 5, 2, 1};
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, &box);
    }
    gpu.transition(second.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    for (UINT sub = 0; sub < count; ++sub) {
        const auto dst = placed(readback.p, footprints.layouts[sub]), src = subresource(second.p, sub);
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    {
        const auto dst = placed(readback.p, box_readback), src = subresource(second.p, 0);
        const D3D12_BOX box{1, 1, 0, 4, 3, 1};
        gpu.list->CopyTextureRegion(&dst, 1, 1, 0, &src, &box);
    }
    CHECK(gpu.run());

    const D3D12_RANGE all{0, SIZE_T(size)};
    CHECK(SUCCEEDED(readback->Map(0, &all, reinterpret_cast<void**>(&mapped))) && mapped);
    std::vector<std::uint8_t> touched(size, 0);
    const auto get = [&](const D3D12_PLACED_SUBRESOURCE_FOOTPRINT& f, UINT x, UINT y, UINT z, UINT c) {
        const UINT64 slice = UINT64(f.Footprint.RowPitch) * f.Footprint.Height;
        const UINT64 at = f.Offset + z * slice + UINT64(y) * f.Footprint.RowPitch + x * kTexel + c * 4;
        std::memset(touched.data() + at, 1, 4);
        std::uint32_t v;
        std::memcpy(&v, mapped + at, 4);
        return v;
    };
    out.subresources.assign(count, {});
    for (UINT sub = 0; sub < count; ++sub)
        for (UINT z = 0; z < s.d(sub); ++z)
            for (UINT y = 0; y < s.h(sub); ++y)
                for (UINT x = 0; x < s.w(sub); ++x)
                    for (UINT c = 0; c < 3; ++c) out.subresources[sub].push_back(get(footprints.layouts[sub], x, y, z, c));
    out.box.clear();
    for (UINT y = 0; y < 2; ++y)
        for (UINT x = 0; x < 3; ++x)
            for (UINT c = 0; c < 3; ++c) out.box.push_back(get(box_readback, 1 + x, 1 + y, 0, c));
    UINT64 overwritten = 0;
    for (UINT64 i = 0; i < size; ++i) overwritten += !touched[i] && mapped[i] != kUntouched;
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    if (check_untouched && overwritten) {
        std::fprintf(stderr, "%llu readback bytes outside the copied texels were overwritten\n", overwritten);
        return false;
    }
    return true;
}

// WriteToSubresource/ReadFromSubresource with pitches unlike the texture's, a box, and (staged) after a GPU copy
// has moved the texture out of the layout host copies need.
bool cpu_copies(ID3D12Device* device, DXGI_FORMAT format, bool leave_common) {
    const D3D12_HEAP_PROPERTIES cpu_heap{D3D12_HEAP_TYPE_CUSTOM, D3D12_CPU_PAGE_PROPERTY_WRITE_BACK, D3D12_MEMORY_POOL_L0};
    const D3D12_RESOURCE_DESC desc = texture_desc(format, false, 5, 3, 1, 1);
    Ref<ID3D12Resource> texture;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&cpu_heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON,
                                                    nullptr, IID_PPV_ARGS(&texture.p))));
    constexpr UINT write_pitch = 5 * kTexel + 4, read_pitch = 5 * kTexel + 20;
    std::vector<std::uint8_t> data(write_pitch * 3, 0xcd), back(read_pitch * 3, kUntouched);
    CHECK(SUCCEEDED(texture->ReadFromSubresource(back.data(), read_pitch, 0, 0, nullptr)));
    for (UINT y = 0; y < 3; ++y)
        for (UINT i = 0; i < 5 * kTexel; ++i) CHECK(back[y * read_pitch + i] == 0);  // a new texture is zero-filled
    for (UINT y = 0; y < 3; ++y)
        for (UINT x = 0; x < 5; ++x)
            for (UINT c = 0; c < 3; ++c) {
                const std::uint32_t v = value(7, x, y, 0, c);
                std::memcpy(data.data() + y * write_pitch + x * kTexel + c * 4, &v, 4);
            }
    CHECK(SUCCEEDED(texture->WriteToSubresource(0, nullptr, data.data(), write_pitch, 0)));
    // A 2x1 box at (3,2), from a buffer of its own.
    std::uint32_t patch[6];
    for (UINT i = 0; i < 6; ++i) patch[i] = value(8, i / 3, 0, 0, i % 3, 2);
    const D3D12_BOX box{3, 2, 0, 5, 3, 1};
    CHECK(SUCCEEDED(texture->WriteToSubresource(0, &box, patch, 2 * kTexel, 0)));
    if (leave_common) {
        Gpu gpu;
        CHECK(gpu.create(device));
        gpu.transition(texture.p, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE);
        CHECK(gpu.run());
    }
    std::memset(back.data(), kUntouched, back.size());
    CHECK(SUCCEEDED(texture->ReadFromSubresource(back.data(), read_pitch, 0, 0, nullptr)));
    for (UINT y = 0; y < 3; ++y) {
        for (UINT x = 0; x < 5; ++x)
            for (UINT c = 0; c < 3; ++c) {
                const std::uint32_t want = y == 2 && x >= 3 ? patch[(x - 3) * 3 + c] : value(7, x, y, 0, c);
                std::uint32_t got;
                std::memcpy(&got, back.data() + y * read_pitch + x * kTexel + c * 4, 4);
                if (got != want) {
                    std::fprintf(stderr, "CPU copy (%u,%u) channel %u: got %08x, want %08x\n", x, y, c, got, want);
                    return false;
                }
            }
        for (UINT i = 5 * kTexel; i < read_pitch; ++i) CHECK(back[y * read_pitch + i] == kUntouched);
    }
    return true;
}

// One upload and one readback of a size x size texture, in milliseconds.
bool timed(ID3D12Device* device, UINT size, double& upload_ms, double& readback_ms) {
    Gpu gpu;
    CHECK(gpu.create(device));
    const D3D12_RESOURCE_DESC desc = texture_desc(DXGI_FORMAT_R32G32B32_FLOAT, false, size, size, 1, 1);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
        readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC bytes = buffer_desc(total);
    Ref<ID3D12Resource> texture, upload, readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&texture.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    std::uint8_t* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped);
    for (UINT y = 0; y < size; ++y)
        for (UINT x = 0; x < size * 3; ++x) {
            const std::uint32_t v = y * 0x10001u + x;
            std::memcpy(mapped + footprint.Offset + UINT64(y) * footprint.Footprint.RowPitch + x * 4, &v, 4);
        }
    upload->Unmap(0, nullptr);
    const auto ms = [](auto from) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - from).count();
    };
    auto start = std::chrono::steady_clock::now();
    {
        const auto dst = subresource(texture.p, 0), src = placed(upload.p, footprint);
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(gpu.run());
    upload_ms = ms(start);
    gpu.transition(texture.p, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE);
    start = std::chrono::steady_clock::now();
    {
        const auto dst = placed(readback.p, footprint), src = subresource(texture.p, 0);
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(gpu.run());
    readback_ms = ms(start);
    CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped);
    for (UINT y = 0; y < size; y += std::max(1u, size / 61))
        for (UINT x = 0; x < size * 3; x += 7) {
            std::uint32_t v;
            std::memcpy(&v, mapped + footprint.Offset + UINT64(y) * footprint.Footprint.RowPitch + x * 4, 4);
            CHECK(v == y * 0x10001u + x);
        }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

// An upload and a readback on a COPY queue, where streaming uploads run: the repack is a compute dispatch, which
// the layer's copy queues must take. The texture starts in COMMON and is promoted, as a copy queue requires.
bool copy_queue(ID3D12Device* device) {
    Gpu gpu;
    CHECK(gpu.create(device, D3D12_COMMAND_LIST_TYPE_COPY));
    constexpr UINT width = 7, height = 5;
    const D3D12_RESOURCE_DESC desc = texture_desc(DXGI_FORMAT_R32G32B32_UINT, false, width, height, 1, 1);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, &footprint, nullptr, nullptr, &total);
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT}, upload_heap{D3D12_HEAP_TYPE_UPLOAD},
        readback_heap{D3D12_HEAP_TYPE_READBACK};
    const D3D12_RESOURCE_DESC bytes = buffer_desc(total);
    Ref<ID3D12Resource> texture, upload, readback;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &desc,
                                                    D3D12_RESOURCE_STATE_COMMON, nullptr, IID_PPV_ARGS(&texture.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&upload_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                    D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&upload.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&readback_heap, D3D12_HEAP_FLAG_NONE, &bytes,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback.p))));
    std::uint8_t* mapped = nullptr;
    CHECK(SUCCEEDED(upload->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x)
            for (UINT c = 0; c < 3; ++c) {
                const std::uint32_t v = value(11, x, y, 0, c);
                std::memcpy(mapped + footprint.Offset + UINT64(y) * footprint.Footprint.RowPitch + x * kTexel + c * 4, &v, 4);
            }
    upload->Unmap(0, nullptr);
    {
        const auto dst = subresource(texture.p, 0), src = placed(upload.p, footprint);
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(gpu.run());  // the texture decays to COMMON, and is promoted again as a copy source
    {
        const auto dst = placed(readback.p, footprint), src = subresource(texture.p, 0);
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(gpu.run());
    CHECK(SUCCEEDED(readback->Map(0, nullptr, reinterpret_cast<void**>(&mapped))) && mapped);
    for (UINT y = 0; y < height; ++y)
        for (UINT x = 0; x < width; ++x)
            for (UINT c = 0; c < 3; ++c) {
                std::uint32_t v;
                std::memcpy(&v, mapped + footprint.Offset + UINT64(y) * footprint.Footprint.RowPitch + x * kTexel + c * 4, 4);
                if (v != value(11, x, y, 0, c)) {
                    std::fprintf(stderr, "copy queue (%u,%u) channel %u: got %08x, want %08x\n", x, y, c, v,
                                 value(11, x, y, 0, c));
                    return false;
                }
            }
    const D3D12_RANGE none{};
    readback->Unmap(0, &none);
    return true;
}

HRESULT creates(ID3D12Device* device, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    const D3D12_HEAP_PROPERTIES heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC desc = texture_desc(format, false, 4, 4, 1, 1, flags);
    Ref<ID3D12Resource> texture;
    return device->CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, nullptr,
                                           IID_PPV_ARGS(&texture.p));
}
// GetResourceAllocationInfo has a size, or GetCopyableFootprints a layout, for the same description.
bool allocates(ID3D12Device* device, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    const D3D12_RESOURCE_DESC desc = texture_desc(format, false, 4, 4, 1, 1, flags);
    return device->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes != UINT64_MAX;
}
bool has_footprint(ID3D12Device* device, DXGI_FORMAT format, D3D12_RESOURCE_FLAGS flags) {
    const D3D12_RESOURCE_DESC desc = texture_desc(format, false, 4, 4, 1, 1, flags);
    UINT64 total = 0;
    device->GetCopyableFootprints(&desc, 0, 1, 0, nullptr, nullptr, nullptr, &total);
    return total != UINT64_MAX;
}

// What the layer refuses at ExecuteCommandLists: copies between a 96-bit texture and a four-channel one (their
// images are the same format, their D3D12 formats are not).
bool refusals(ID3D12Device* device) {
    Gpu gpu;
    CHECK(gpu.create(device));
    const D3D12_HEAP_PROPERTIES default_heap{D3D12_HEAP_TYPE_DEFAULT};
    const D3D12_RESOURCE_DESC narrow_desc = texture_desc(DXGI_FORMAT_R32G32B32_FLOAT, false, 4, 4, 1, 1);
    const D3D12_RESOURCE_DESC wide_desc = texture_desc(DXGI_FORMAT_R32G32B32A32_FLOAT, false, 4, 4, 1, 1);
    Ref<ID3D12Resource> narrow, wide;
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &narrow_desc,
                                                    D3D12_RESOURCE_STATE_COPY_SOURCE, nullptr, IID_PPV_ARGS(&narrow.p))));
    CHECK(SUCCEEDED(device->CreateCommittedResource(&default_heap, D3D12_HEAP_FLAG_NONE, &wide_desc,
                                                    D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&wide.p))));
    const auto refused = [&](leonida::ReplayResult want) {
        if (FAILED(gpu.list->Close())) return false;
        ID3D12CommandList* lists[] = {gpu.list.p};
        gpu.queue->ExecuteCommandLists(1, lists);
        const bool ok = rt::d3d12::last_execute_result(gpu.queue.p) == want;
        return ok && SUCCEEDED(gpu.allocator->Reset()) && SUCCEEDED(gpu.list->Reset(gpu.allocator.p, nullptr));
    };
    {
        const auto dst = subresource(wide.p, 0), src = subresource(narrow.p, 0);
        gpu.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
    }
    CHECK(refused(leonida::ReplayResult::invalid_copy));
    gpu.list->CopyResource(wide.p, narrow.p);
    CHECK(refused(leonida::ReplayResult::invalid_copy));
    return true;
}
} // namespace

int main() {
    vktest::Device vk;
    const auto device_status = vk.initialize();
    if (device_status == vktest::DeviceStatus::unavailable) return 77;  // no Vulkan GPU: ctest skips
    if (device_status != vktest::DeviceStatus::ready) return 1;
    unsigned compared = 0;
    {
        Ref<ID3D12Device> ours, ours_staged, host;
        rt::d3d12::VulkanDeviceInfo info{};
        info.get_instance_proc = vk.instance_proc;
        info.instance = vk.instance;
        info.physical = vk.physical;
        info.device = vk.device;
        info.queue = vk.queue;
        info.queue_family = vk.family;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours_staged.p)))) return 1;
        info.host_image_copy = vk.host_image_copy;
        if (FAILED(rt::d3d12::create_vulkan_device(info, IID_PPV_ARGS(&ours.p)))) return 1;
        // Wine has a WARP adapter too, but it is vkd3d, whose footprints differ from Microsoft's.
        if (hostwine::host_is_wine()) return 77;  // WARP is the reference for every case
        Ref<IDXGIFactory4> factory;
        Ref<IDXGIAdapter> warp;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory.p))) ||
            FAILED(factory->EnumWarpAdapter(IID_PPV_ARGS(&warp.p))) ||
            FAILED(D3D12CreateDevice(warp.p, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&host.p)))) {
            std::fprintf(stderr, "WARP reference unavailable\n");
            return 77;  // Wine: ctest skips
        }
        Ref<ID3D12Device> hardware;
        if (FAILED(D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&hardware.p)))) return 77;

        constexpr DXGI_FORMAT formats[] = {DXGI_FORMAT_R32G32B32_FLOAT, DXGI_FORMAT_R32G32B32_UINT,
                                           DXGI_FORMAT_R32G32B32_SINT, DXGI_FORMAT_R32G32B32_TYPELESS};
        for (const DXGI_FORMAT format : formats) {
            if (!host_reports_no_tiling(hardware.p, "hardware", format) ||
                !host_reports_no_tiling(host.p, "WARP", format))
                return 1;
            for (const bool volume : {false, true}) {
                const Shape shape{format, volume};
                Footprints ours_footprints, host_footprints;
                Texels ours_texels, host_texels;
                if (!copies(ours.p, shape, ours_footprints, ours_texels, true) ||
                    !copies(host.p, shape, host_footprints, host_texels, false))
                    return 1;
                const Texels want = expected(shape);
                if (!(ours_footprints == host_footprints)) {
                    std::fprintf(stderr, "format %d %s: footprints differ from WARP (total %llu, WARP %llu)\n", int(format),
                                 volume ? "3D" : "2D", ours_footprints.total, host_footprints.total);
                    return 1;
                }
                if (!(host_texels == want)) {
                    std::fprintf(stderr, "format %d %s: WARP differs from the test's expected texels\n", int(format),
                                 volume ? "3D" : "2D");
                    return 1;
                }
                if (!(ours_texels == want)) {
                    for (std::size_t sub = 0; sub < want.subresources.size(); ++sub)
                        for (std::size_t i = 0; i < want.subresources[sub].size(); ++i)
                            if (ours_texels.subresources[sub][i] != want.subresources[sub][i]) {
                                std::fprintf(stderr, "format %d %s subresource %zu word %zu: got %08x, want %08x\n",
                                             int(format), volume ? "3D" : "2D", sub, i,
                                             ours_texels.subresources[sub][i], want.subresources[sub][i]);
                                return 1;
                            }
                    std::fprintf(stderr, "format %d %s: the box readback differs\n", int(format), volume ? "3D" : "2D");
                    return 1;
                }
                ++compared;
            }
            if (!cpu_copies(ours.p, format, false) || !cpu_copies(ours.p, format, true) ||
                !cpu_copies(ours_staged.p, format, false))
                return 1;

            // Creation and feature support, next to the PC runtime's.
            const HRESULT plain = creates(ours.p, format, D3D12_RESOURCE_FLAG_NONE);
            const HRESULT target = creates(ours.p, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            const HRESULT host_target = creates(host.p, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET);
            if (FAILED(plain) || FAILED(creates(host.p, format, D3D12_RESOURCE_FLAG_NONE)) ||
                !allocates(ours.p, format, D3D12_RESOURCE_FLAG_NONE))
                return 1;
            // Unordered access and depth: refused, as the PC runtime refuses them.
            for (const auto flag : {D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL}) {
                const HRESULT mine = creates(ours.p, format, flag), theirs = creates(host.p, format, flag);
                if (mine != E_INVALIDARG || theirs != E_INVALIDARG || allocates(ours.p, format, flag) ||
                    has_footprint(ours.p, format, flag) || has_footprint(host.p, format, flag)) {
                    std::fprintf(stderr, "format %d flag %d: creation %08lx (WARP %08lx), or a size or a footprint\n",
                                 int(format), int(flag), mine, theirs);
                    return 1;
                }
            }
            // Render targets are not implemented: creation and allocation info must not look as if they were.
            if (SUCCEEDED(target) || allocates(ours.p, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET) ||
                has_footprint(ours.p, format, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET)) {
                std::fprintf(stderr, "format %d: a render target was created or sized\n", int(format));
                return 1;
            }
            D3D12_FEATURE_DATA_FORMAT_SUPPORT mine{format}, theirs{format};
            if (FAILED(ours->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &mine, sizeof(mine))) ||
                FAILED(host->CheckFeatureSupport(D3D12_FEATURE_FORMAT_SUPPORT, &theirs, sizeof(theirs))))
                return 1;
            constexpr auto unsupported = D3D12_FORMAT_SUPPORT1_RENDER_TARGET | D3D12_FORMAT_SUPPORT1_BLENDABLE |
                                         D3D12_FORMAT_SUPPORT1_DEPTH_STENCIL | D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RENDERTARGET |
                                         D3D12_FORMAT_SUPPORT1_MULTISAMPLE_RESOLVE |
                                         D3D12_FORMAT_SUPPORT1_TYPED_UNORDERED_ACCESS_VIEW;
            if (!(mine.Support1 & D3D12_FORMAT_SUPPORT1_TEXTURE2D) || (mine.Support1 & unsupported) || mine.Support2) {
                std::fprintf(stderr, "format %d: FORMAT_SUPPORT %08x/%08x does not match what creation accepts\n",
                             int(format), unsigned(mine.Support1), unsigned(mine.Support2));
                return 1;
            }
            std::printf("format %d: FORMAT_SUPPORT ours %08x/%08x, WARP %08x/%08x; render target: ours %08lx, WARP %08lx\n",
                        int(format), unsigned(mine.Support1), unsigned(mine.Support2), unsigned(theirs.Support1),
                        unsigned(theirs.Support2), target, host_target);
        }
        if (!refusals(ours.p)) return 1;
        if (!copy_queue(ours.p) || !copy_queue(host.p)) return 1;
        // The large sizes need hundreds of MiB and a real GPU: only with LEONIDA_RGB96_LARGE set.
        const bool large = GetEnvironmentVariableA("LEONIDA_RGB96_LARGE", nullptr, 0) != 0;
        for (const UINT size : {64u, 256u, 1024u, 2048u, 4096u}) {
            if (size > 256 && !large) break;
            double upload_ms = 0, readback_ms = 0, host_upload = 0, host_readback = 0;
            if (!timed(ours.p, size, upload_ms, readback_ms) || !timed(host.p, size, host_upload, host_readback)) return 1;
            std::printf("%ux%u: upload %.1f ms, readback %.1f ms (WARP %.1f, %.1f)\n", size, size, upload_ms, readback_ms,
                        host_upload, host_readback);
        }
    }
    if (vk.validation_errors()) return 1;
    std::printf("rgb96: %u texture shapes matched WARP and the expected texels bit for bit; CPU copies with%s; "
                "the hardware adapter and WARP report no tiling for all four formats\n",
                compared, vk.host_image_copy ? " and without host image copies on the device" : "out host image copies");
    return 0;
}
