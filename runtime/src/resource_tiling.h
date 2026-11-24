#pragma once

// How a D3D12 reserved (tiled) resource divides into 64 KiB tiles: ID3D12Device::GetResourceTiling
// (docs/d3d12_vulkan.md section 4.2). No Windows or Vulkan dependency.
//
// What D3D12 defines, and d3d12_vulkan_device_test compares with the PC runtime: the standard tile shape for each
// element size, that a mip filling at least one tile shape in every dimension is standard, the tile counts of every
// standard mip, and the order of the tiles: each array slice holds its standard mips in order, then its packed tiles.
//
// What a driver chooses (D3D11.3 functional spec 5.9.2.8.5: a mip with any dimension below the tile "may or may not
// be packed, depending on the IHV"): where the packed tail starts, and how many tiles the packed mips take. Here
// a mip stays standard while it is wider than half the tile or taller than the tile (depth does not count), as
// AMD's PC runtime does on RDNA2 and RDNA4 (D3D-65; the console is AMD): measured on the 47 layouts of #868, see
// resource_tiling_test. A standard mip may be smaller than a tile and then takes one tile on that axis. An NVIDIA
// GPU packs from the first mip smaller than a tile on any axis and WARP packs nothing; the test accepts all of
// them. The packed tail takes one tile per tile-depth slice it spans: ceil(depth of the first packed mip / tile
// depth), so 1 for 2D, as AMD's PC runtime reports on RDNA2 and RDNA4 (those 47 and 32 deeper volumes, see
// resource_tiling_test), and never fewer than the tail's tight linear size in whole tiles. NVIDIA sometimes reports
// one more. Applications read the numbers rather than assume them, and this layer backs reserved textures fully, so
// any split is consistent.

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace leonida {

inline constexpr std::uint32_t tile_size_in_bytes = 65536;  // D3D12_TILED_RESOURCE_TILE_SIZE_IN_BYTES
inline constexpr std::uint32_t packed_tile = 0xffffffff;    // D3D12_PACKED_TILE

// In texels, or in bytes for a buffer (65536 x 1 x 1).
struct TileShape {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
};

enum class TiledKind { buffer, texture2d, texture3d };

struct TilingDesc {
    TiledKind kind = TiledKind::texture2d;
    std::uint64_t width = 0;            // bytes for a buffer
    std::uint32_t height = 1;
    std::uint32_t depth_or_layers = 1;  // depth for a volume, array size otherwise
    std::uint32_t mips = 1;
    std::uint32_t element_bytes = 0;    // bytes per texel, or per block of a block-compressed format
    std::uint32_t block = 1;            // texels per block side: 4 for BC1-7
    std::uint32_t samples = 1;          // 1, or 4 for 2D single-mip MSAA
};

// D3D12_SUBRESOURCE_TILING: a packed mip has no tiles of its own and starts at packed_tile.
struct SubresourceTiling {
    std::uint32_t width = 0;
    std::uint32_t height = 0;
    std::uint32_t depth = 0;
    std::uint32_t start = 0;
};

struct ResourceTiling {
    std::uint32_t total_tiles = 0;
    TileShape shape;
    std::uint32_t standard_mips = 0;  // D3D12_PACKED_MIP_INFO, per array slice
    std::uint32_t packed_mips = 0;
    std::uint32_t packed_tiles = 0;
    std::uint32_t packed_start = 0;   // the first packed tile of slice 0; 0 without packed mips
    std::vector<SubresourceTiling> subresources;  // mip + layer * mips
};

// The standard tile shape in elements (D3D12 "standard swizzle" tile shapes for 64 KiB tiles; 2D 4x MSAA scales the
// tile shape down by 4 samples per texel, measured on RTX 5070 Ti and WARP), or {} for an unsupported sample count or
// an element size that cannot be tiled (3- and 12-byte formats).
constexpr TileShape element_tile_shape(bool volume, std::uint32_t element_bytes, std::uint32_t samples = 1) {
    if (samples == 4) {
        if (volume) return {};
        switch (element_bytes) {
        case 1: return {128, 128, 1};
        case 2: return {128, 64, 1};
        case 4: return {64, 64, 1};
        case 8: return {64, 32, 1};
        case 16: return {32, 32, 1};
        default: return {};
        }
    }
    if (samples != 1) return {};
    switch (element_bytes) {
    case 1: return volume ? TileShape{64, 32, 32} : TileShape{256, 256, 1};
    case 2: return volume ? TileShape{32, 32, 32} : TileShape{256, 128, 1};
    case 4: return volume ? TileShape{32, 32, 16} : TileShape{128, 128, 1};
    case 8: return volume ? TileShape{32, 16, 16} : TileShape{128, 64, 1};
    case 16: return volume ? TileShape{16, 16, 16} : TileShape{64, 64, 1};
    default: return {};
    }
}

// The 4 KiB tile shape in elements the PC runtime counts to allow D3D12_SMALL_RESOURCE_PLACEMENT_ALIGNMENT (16 of
// them must cover a texture's top mip), or {} as above. Measured with GetResourceAllocationInfo on an RTX 5070 and
// WARP (d3d12_vulkan_device_test).
constexpr TileShape small_tile_shape(bool volume, std::uint32_t element_bytes) {
    switch (element_bytes) {
    case 1: return volume ? TileShape{16, 16, 16} : TileShape{64, 64, 1};
    case 2: return volume ? TileShape{16, 16, 8} : TileShape{64, 32, 1};
    case 4: return volume ? TileShape{16, 8, 8} : TileShape{32, 32, 1};
    case 8: return volume ? TileShape{8, 8, 8} : TileShape{32, 16, 1};
    case 16: return volume ? TileShape{8, 8, 4} : TileShape{16, 16, 1};
    default: return {};
    }
}

// False when the resource cannot be tiled: an untileable element size, no mips, or more than 2^32 tiles.
inline bool resource_tiling(const TilingDesc& desc, ResourceTiling& out) {
    out = {};
    constexpr std::uint64_t max_tiles = 0xffffffffu;
    constexpr std::uint64_t max_packed_bytes = max_tiles * tile_size_in_bytes;
    const auto up = [](std::uint64_t value, std::uint64_t by) -> std::uint64_t {
        return value ? 1 + (value - 1) / by : 0;
    };
    if (desc.kind == TiledKind::buffer) {
        const std::uint64_t tiles = up(desc.width, tile_size_in_bytes);
        if (!tiles || tiles > max_tiles) return false;
        out.total_tiles = std::uint32_t(tiles);
        out.shape = {tile_size_in_bytes, 1, 1};
        out.subresources = {{out.total_tiles, 1, 1, 0}};
        return true;
    }
    const bool volume = desc.kind == TiledKind::texture3d;
    if (desc.samples != 1 && (desc.samples != 4 || volume || desc.mips != 1 || desc.block != 1)) return false;
    const TileShape elements = element_tile_shape(volume, desc.element_bytes, desc.samples);
    if (!elements.width || !desc.mips || !desc.block || !desc.width || !desc.height || !desc.depth_or_layers)
        return false;
    const std::uint64_t shape_w = std::uint64_t(elements.width) * desc.block;
    const std::uint64_t shape_h = std::uint64_t(elements.height) * desc.block;
    if (shape_w > 0xffffffffu || shape_h > 0xffffffffu) return false;
    ResourceTiling result;
    result.shape = {std::uint32_t(shape_w), std::uint32_t(shape_h), elements.depth};
    const std::uint32_t layers = volume ? 1 : desc.depth_or_layers;
    const std::uint32_t depth = volume ? desc.depth_or_layers : 1;

    // One array slice: standard mips while wider than half a tile or taller than a tile, then the packed tail.
    std::vector<SubresourceTiling> slice(desc.mips);
    std::uint64_t standard_tiles = 0, packed_bytes = 0;
    std::uint32_t tail_depth = 0;   // depth of the first packed mip
    for (std::uint32_t mip = 0; mip < desc.mips; ++mip) {
        const std::uint64_t w = mip >= 64 ? 1 : std::max<std::uint64_t>(1, desc.width >> mip);
        const std::uint32_t h = mip >= 32 ? 1u : std::max(1u, desc.height >> mip);
        const std::uint32_t d = mip >= 32 ? 1u : std::max(1u, depth >> mip);
        const bool standard = result.packed_mips == 0 && (w > result.shape.width / 2 || h > result.shape.height);
        if (standard) {
            const std::uint64_t tw = up(w, result.shape.width);
            const std::uint64_t th = up(h, result.shape.height);
            const std::uint64_t td = up(d, result.shape.depth);
            if (tw > max_tiles || th > max_tiles || td > max_tiles) return false;
            const std::uint64_t xy = tw * th;
            if (xy > max_tiles || xy > max_tiles / td) return false;
            const std::uint64_t mip_tiles = xy * td;
            if (mip_tiles > max_tiles - standard_tiles) return false;
            slice[mip] = {std::uint32_t(tw), std::uint32_t(th), std::uint32_t(td), std::uint32_t(standard_tiles)};
            standard_tiles += mip_tiles;
            ++result.standard_mips;
        } else {
            slice[mip] = {0, 0, 0, packed_tile};
            const std::uint64_t row_slice_bytes =
                up(w, desc.block) * up(h, desc.block) * desc.element_bytes * desc.samples;
            if (d > (max_packed_bytes - packed_bytes) / row_slice_bytes) return false;
            packed_bytes += row_slice_bytes * d;
            if (result.packed_mips == 0) tail_depth = d;
            ++result.packed_mips;
        }
    }
    // One tile per tile-depth slice of the tail (AMD), at least its tight size; 1 for 2D.
    const std::uint64_t packed_tiles =
        result.packed_mips ? std::max<std::uint64_t>({1, up(packed_bytes, tile_size_in_bytes),
                                                      up(tail_depth, result.shape.depth)})
                           : 0;
    if (packed_tiles > max_tiles - standard_tiles) return false;
    const std::uint64_t per_slice = standard_tiles + packed_tiles;
    if (per_slice > max_tiles / layers) return false;
    result.packed_tiles = std::uint32_t(packed_tiles);
    result.packed_start = result.packed_mips ? std::uint32_t(standard_tiles) : 0;
    result.total_tiles = std::uint32_t(per_slice * layers);
    result.subresources.reserve(std::size_t(desc.mips) * layers);
    for (std::uint32_t layer = 0; layer < layers; ++layer)
        for (SubresourceTiling s : slice) {
            if (s.start != packed_tile) s.start += std::uint32_t(per_slice * layer);
            result.subresources.push_back(s);
        }
    out = std::move(result);
    return true;
}

} // namespace leonida
