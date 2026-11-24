// GetResourceTiling's arithmetic (src/resource_tiling.h), without a GPU. The expected tile shapes, per-mip tile
// counts and start indices are what the PC D3D12 runtime reported for the same reserved resources on an RTX 5070.
// Where the packed tail starts and how many tiles it takes follow AMD's PC runtime (D3D-65): the AmdCase table holds
// AMD's values, and where another runtime chose differently the comment says so. d3d12_vulkan_device_test makes the
// same comparison against the host runtime on the machine it runs on.
#include "../src/resource_tiling.h"

#include <cstdio>
#include <tuple>

#define CHECK(condition) do {                                        \
    if (!(condition)) {                                              \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                    \
    }                                                                \
} while (0)

namespace {

using leonida::packed_tile;
using leonida::ResourceTiling;
using leonida::TiledKind;
using leonida::TilingDesc;

bool same(const leonida::SubresourceTiling& s, std::uint32_t w, std::uint32_t h, std::uint32_t d, std::uint32_t start) {
    return s.width == w && s.height == h && s.depth == d && s.start == start;
}

}  // namespace

int main() {
    ResourceTiling t;

    // A buffer: 64 KiB tiles in a row, no mips.
    CHECK(resource_tiling({TiledKind::buffer, 200000}, t));
    CHECK(t.total_tiles == 4 && t.shape.width == 65536 && t.shape.height == 1 && t.shape.depth == 1);
    CHECK(t.standard_mips == 0 && t.packed_mips == 0 && t.subresources.size() == 1 && same(t.subresources[0], 4, 1, 1, 0));
    CHECK(!resource_tiling({TiledKind::buffer, 0}, t));

    // 1024x1024 RGBA8, full chain: four standard mips (64 + 16 + 4 + 1 tiles), seven packed in one tile.
    CHECK(resource_tiling({TiledKind::texture2d, 1024, 1024, 1, 11, 4, 1}, t));
    CHECK(t.shape.width == 128 && t.shape.height == 128 && t.shape.depth == 1);
    CHECK(t.standard_mips == 4 && t.packed_mips == 7 && t.packed_tiles == 1 && t.packed_start == 85);
    CHECK(t.total_tiles == 86 && t.subresources.size() == 11);
    CHECK(same(t.subresources[0], 8, 8, 1, 0) && same(t.subresources[1], 4, 4, 1, 64));
    CHECK(same(t.subresources[2], 2, 2, 1, 80) && same(t.subresources[3], 1, 1, 1, 84));
    CHECK(same(t.subresources[4], 0, 0, 0, packed_tile) && same(t.subresources[10], 0, 0, 0, packed_tile));

    // A 1000x300 array of 3: mips 2 (250x75) and 3 (125x37) are below a tile in height but still standard, one
    // tile tall; mip 4 (62x18) is the first at most 64 texels on both axes. Each slice holds its standard tiles,
    // then its packed one. As the AMD PC runtime reports (RDNA2 and RDNA4).
    CHECK(resource_tiling({TiledKind::texture2d, 1000, 300, 3, 10, 4, 1}, t));
    CHECK(t.standard_mips == 4 && t.packed_mips == 6 && t.packed_start == 35 && t.packed_tiles == 1);
    CHECK(t.total_tiles == 3 * 36 && t.subresources.size() == 30);
    CHECK(same(t.subresources[0], 8, 3, 1, 0) && same(t.subresources[1], 4, 2, 1, 24));
    CHECK(same(t.subresources[2], 2, 1, 1, 32) && same(t.subresources[3], 1, 1, 1, 34));
    CHECK(same(t.subresources[10], 8, 3, 1, 36) && same(t.subresources[11], 4, 2, 1, 60));
    CHECK(same(t.subresources[14], 0, 0, 0, packed_tile) && same(t.subresources[20], 8, 3, 1, 72));

    // Element sizes: 1, 2, 8 and 16 bytes.
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 1, 1}, t));
    CHECK(t.shape.width == 256 && t.shape.height == 256 && t.standard_mips == 2 && t.packed_start == 5);
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 2, 1}, t));
    CHECK(t.shape.width == 256 && t.shape.height == 128 && t.standard_mips == 2 && t.packed_start == 10);
    CHECK(same(t.subresources[0], 2, 4, 1, 0) && same(t.subresources[1], 1, 2, 1, 8));
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 8, 1}, t));
    CHECK(t.shape.width == 128 && t.shape.height == 64 && t.standard_mips == 3 && t.packed_start == 42);
    CHECK(resource_tiling({TiledKind::texture2d, 512, 512, 1, 10, 16, 1}, t));
    CHECK(t.shape.width == 64 && t.shape.height == 64 && t.standard_mips == 4 && t.total_tiles == 86);

    // Block-compressed: the shape is in texels, four per block side. BC1 (8-byte blocks) tiles 512x256 texels,
    // BC7 (16-byte blocks) 256x256; a two-slice BC7 array has one packed tile per slice.
    CHECK(resource_tiling({TiledKind::texture2d, 2048, 2048, 1, 12, 8, 4}, t));
    CHECK(t.shape.width == 512 && t.shape.height == 256 && t.standard_mips == 3 && t.packed_mips == 9);
    CHECK(same(t.subresources[0], 4, 8, 1, 0) && same(t.subresources[2], 1, 2, 1, 40) && t.packed_start == 42);
    CHECK(resource_tiling({TiledKind::texture2d, 2048, 1024, 2, 12, 16, 4}, t));
    CHECK(t.shape.width == 256 && t.shape.height == 256 && t.packed_tiles == 1 && t.total_tiles == 88);
    CHECK(t.standard_mips == 4 && t.packed_start == 43);  // mip 3 (256x128) is still standard
    CHECK(same(t.subresources[0], 8, 4, 1, 0) && same(t.subresources[12], 8, 4, 1, 44));
    CHECK(same(t.subresources[14], 2, 1, 1, 84) && same(t.subresources[15], 1, 1, 1, 86));
    CHECK(same(t.subresources[16], 0, 0, 0, packed_tile));

    // One whole tile with one mip: nothing packed, and the packed start is 0.
    CHECK(resource_tiling({TiledKind::texture2d, 128, 128, 1, 1, 4, 1}, t));
    CHECK(t.total_tiles == 1 && t.standard_mips == 1 && t.packed_mips == 0 && t.packed_tiles == 0 && t.packed_start == 0);
    // Smaller than a tile but more than half of it: mip 0 (100x100) is one standard tile, the rest is packed.
    CHECK(resource_tiling({TiledKind::texture2d, 100, 100, 1, 7, 4, 1}, t));
    CHECK(t.standard_mips == 1 && t.packed_mips == 6 && t.packed_start == 1 && t.packed_tiles == 1 && t.total_tiles == 2);
    CHECK(same(t.subresources[0], 1, 1, 1, 0) && same(t.subresources[1], 0, 0, 0, packed_tile));
    // Thinner than a tile: 4096x64 keeps standard mips while it is wider than 64 texels (mip 5, 128x2), then packs.
    CHECK(resource_tiling({TiledKind::texture2d, 4096, 64, 1, 13, 4, 1}, t));
    CHECK(t.standard_mips == 6 && t.packed_mips == 7 && t.packed_tiles == 1 && t.packed_start == 63 && t.total_tiles == 64);
    CHECK(same(t.subresources[0], 32, 1, 1, 0) && same(t.subresources[5], 1, 1, 1, 62));
    // A short chain that stops before any mip is packed.
    CHECK(resource_tiling({TiledKind::texture2d, 1024, 1024, 1, 3, 4, 1}, t));
    CHECK(t.total_tiles == 84 && t.packed_mips == 0 && t.packed_start == 0);

    // Volumes: three-dimensional tiles, and depth also decides when mips pack (mip 3, 32x32x8, is still standard).
    CHECK(resource_tiling({TiledKind::texture3d, 256, 256, 64, 9, 4, 1}, t));
    CHECK(t.shape.width == 32 && t.shape.height == 32 && t.shape.depth == 16);
    CHECK(t.standard_mips == 4 && t.packed_mips == 5 && t.packed_start == 293 && t.total_tiles == 294);
    CHECK(same(t.subresources[0], 8, 8, 4, 0) && same(t.subresources[1], 4, 4, 2, 256) && same(t.subresources[2], 2, 2, 1, 288));
    CHECK(same(t.subresources[3], 1, 1, 1, 292) && same(t.subresources[4], 0, 0, 0, packed_tile));
    CHECK(resource_tiling({TiledKind::texture3d, 128, 128, 128, 8, 1, 1}, t));
    CHECK(t.shape.width == 64 && t.shape.height == 32 && t.shape.depth == 32 && t.standard_mips == 2);
    CHECK(t.packed_start == 36 && t.packed_tiles == 1);  // 37,449 bytes (the PC runtime reports two tiles)
    CHECK(resource_tiling({TiledKind::texture3d, 64, 64, 64, 7, 8, 1}, t));
    CHECK(t.shape.width == 32 && t.shape.height == 16 && t.shape.depth == 16 && t.standard_mips == 2);
    CHECK(same(t.subresources[0], 2, 4, 4, 0) && same(t.subresources[1], 1, 2, 2, 32) && t.packed_start == 36);

    // AMD's split (D3D-65). Every layout below was measured on the AMD PC runtime and is identical on an RX 9070 XT
    // (RDNA4, driver 32.0.31041.1004) and a Ryzen 7 9800X3D iGPU (RDNA2, 32.0.21045.5002): compare_tiling's 17 texture
    // layouts, then 30 edge cases (a mip exactly at or just past half a tile, thin and tall textures, non-square 2D and
    // volume tiles), then 32 volumes whose packed tail is deeper than a tile. Every field is AMD's: the standard and
    // packed mip counts, the packed start, and the packed tiles, one per tile-depth slice the tail spans.
    struct AmdCase {
        const char* name;
        TilingDesc desc;
        std::uint32_t standard, packed, packed_tiles, packed_start;
    };
    const AmdCase amd[] = {
        {"1024x1024 RGBA8", {TiledKind::texture2d, 1024, 1024, 1, 11, 4, 1}, 4, 7, 1, 85},
        {"1000x300 RGBA8 sRGB x3", {TiledKind::texture2d, 1000, 300, 3, 10, 4, 1}, 4, 6, 1, 35},
        {"512x512 R8", {TiledKind::texture2d, 512, 512, 1, 10, 1, 1}, 2, 8, 1, 5},
        {"512x512 R16F", {TiledKind::texture2d, 512, 512, 1, 10, 2, 1}, 2, 8, 1, 10},
        {"512x512 RGBA16F", {TiledKind::texture2d, 512, 512, 1, 10, 8, 1}, 3, 7, 1, 42},
        {"512x512 RGBA32F x2", {TiledKind::texture2d, 512, 512, 2, 10, 16, 1}, 4, 6, 1, 85},
        {"2048x2048 BC1", {TiledKind::texture2d, 2048, 2048, 1, 12, 8, 4}, 3, 9, 1, 42},
        {"2048x1024 BC7 x2", {TiledKind::texture2d, 2048, 1024, 2, 12, 16, 4}, 4, 8, 1, 43},
        {"1000x600 BC3", {TiledKind::texture2d, 1000, 600, 1, 10, 16, 4}, 3, 7, 1, 17},
        {"128x128 RGBA8 1 mip", {TiledKind::texture2d, 128, 128, 1, 1, 4, 1}, 1, 0, 0, 0},
        {"100x100 RGBA8", {TiledKind::texture2d, 100, 100, 1, 7, 4, 1}, 1, 6, 1, 1},
        {"4096x64 BGRA8", {TiledKind::texture2d, 4096, 64, 1, 13, 4, 1}, 6, 7, 1, 63},
        {"1024x1024 R32F x4 3 mips", {TiledKind::texture2d, 1024, 1024, 4, 3, 4, 1}, 3, 0, 0, 0},
        {"256x256x64 RGBA8 vol", {TiledKind::texture3d, 256, 256, 64, 9, 4, 1}, 4, 5, 1, 293},
        {"128^3 R8 vol", {TiledKind::texture3d, 128, 128, 128, 8, 1, 1}, 2, 6, 1, 36},
        {"64^3 RGBA16F vol", {TiledKind::texture3d, 64, 64, 64, 7, 8, 1}, 2, 5, 1, 36},
        {"64^3 RGBA32F vol", {TiledKind::texture3d, 64, 64, 64, 7, 16, 1}, 3, 4, 1, 73},
        {"64x64 RGBA8 half", {TiledKind::texture2d, 64, 64, 1, 7, 4, 1}, 0, 7, 1, 0},
        {"65x65 RGBA8", {TiledKind::texture2d, 65, 65, 1, 7, 4, 1}, 1, 6, 1, 1},
        {"128x64 RGBA8", {TiledKind::texture2d, 128, 64, 1, 8, 4, 1}, 1, 7, 1, 1},
        {"64x128 RGBA8", {TiledKind::texture2d, 64, 128, 1, 8, 4, 1}, 0, 8, 1, 0},
        {"4096x1 RGBA8", {TiledKind::texture2d, 4096, 1, 1, 13, 4, 1}, 6, 7, 1, 63},
        {"1x4096 RGBA8", {TiledKind::texture2d, 1, 4096, 1, 13, 4, 1}, 5, 8, 1, 62},
        {"256x2 RGBA8", {TiledKind::texture2d, 256, 2, 1, 9, 4, 1}, 2, 7, 1, 3},
        {"64x64 RGBA8 x3", {TiledKind::texture2d, 64, 64, 3, 7, 4, 1}, 0, 7, 1, 0},
        {"128x128 R8 half", {TiledKind::texture2d, 128, 128, 1, 8, 1, 1}, 0, 8, 1, 0},
        {"129x129 R8", {TiledKind::texture2d, 129, 129, 1, 8, 1, 1}, 1, 7, 1, 1},
        {"128x128 R16F", {TiledKind::texture2d, 128, 128, 1, 8, 2, 1}, 0, 8, 1, 0},
        {"129x64 R16F", {TiledKind::texture2d, 129, 64, 1, 8, 2, 1}, 1, 7, 1, 1},
        {"64x129 R16F", {TiledKind::texture2d, 64, 129, 1, 8, 2, 1}, 1, 7, 1, 2},
        {"64x128 R16F", {TiledKind::texture2d, 64, 128, 1, 8, 2, 1}, 0, 8, 1, 0},
        {"64x64 RGBA16F", {TiledKind::texture2d, 64, 64, 1, 7, 8, 1}, 0, 7, 1, 0},
        {"64x33 RGBA16F", {TiledKind::texture2d, 64, 33, 1, 7, 8, 1}, 0, 7, 1, 0},
        {"65x8 RGBA16F", {TiledKind::texture2d, 65, 8, 1, 7, 8, 1}, 1, 6, 1, 1},
        {"256x256 BC1", {TiledKind::texture2d, 256, 256, 1, 9, 8, 4}, 0, 9, 1, 0},
        {"260x16 BC1", {TiledKind::texture2d, 260, 16, 1, 9, 8, 4}, 1, 8, 1, 1},
        {"128x128 BC7", {TiledKind::texture2d, 128, 128, 1, 8, 16, 4}, 0, 8, 1, 0},
        {"132x4 BC7", {TiledKind::texture2d, 132, 4, 1, 8, 16, 4}, 1, 7, 1, 1},
        {"32^3 R8 vol", {TiledKind::texture3d, 32, 32, 32, 6, 1, 1}, 0, 6, 1, 0},
        {"33x4x4 R8 vol", {TiledKind::texture3d, 33, 4, 4, 6, 1, 1}, 1, 5, 1, 1},
        {"8x8x64 R8 vol", {TiledKind::texture3d, 8, 8, 64, 7, 1, 1}, 0, 7, 2, 0},
        {"32x17x17 R8 vol", {TiledKind::texture3d, 32, 17, 17, 6, 1, 1}, 0, 6, 1, 0},
        {"16^3 RGBA8 vol", {TiledKind::texture3d, 16, 16, 16, 5, 4, 1}, 0, 5, 1, 0},
        {"16x16x17 RGBA8 vol", {TiledKind::texture3d, 16, 16, 17, 5, 4, 1}, 0, 5, 2, 0},
        {"16x9x9 RGBA8 vol", {TiledKind::texture3d, 16, 9, 9, 5, 4, 1}, 0, 5, 1, 0},
        {"8^3 RGBA32F vol", {TiledKind::texture3d, 8, 8, 8, 4, 16, 1}, 0, 4, 1, 0},
        {"9x1x1 RGBA32F vol", {TiledKind::texture3d, 9, 1, 1, 4, 16, 1}, 1, 3, 1, 1},
        // Packed tails deeper than one tile: one packed tile per tile-depth slice.
        {"8x8x16 R8 vol", {TiledKind::texture3d, 8, 8, 16, 5, 1, 1}, 0, 5, 1, 0},
        {"8x8x32 R8 vol", {TiledKind::texture3d, 8, 8, 32, 6, 1, 1}, 0, 6, 1, 0},
        {"8x8x33 R8 vol", {TiledKind::texture3d, 8, 8, 33, 6, 1, 1}, 0, 6, 2, 0},
        {"8x8x48 R8 vol", {TiledKind::texture3d, 8, 8, 48, 6, 1, 1}, 0, 6, 2, 0},
        {"8x8x65 R8 vol", {TiledKind::texture3d, 8, 8, 65, 7, 1, 1}, 0, 7, 3, 0},
        {"8x8x96 R8 vol", {TiledKind::texture3d, 8, 8, 96, 7, 1, 1}, 0, 7, 3, 0},
        {"8x8x128 R8 vol", {TiledKind::texture3d, 8, 8, 128, 8, 1, 1}, 0, 8, 4, 0},
        {"8x8x129 R8 vol", {TiledKind::texture3d, 8, 8, 129, 8, 1, 1}, 0, 8, 5, 0},
        {"8x8x256 R8 vol", {TiledKind::texture3d, 8, 8, 256, 9, 1, 1}, 0, 9, 8, 0},
        {"16x16x32 RGBA8 vol", {TiledKind::texture3d, 16, 16, 32, 6, 4, 1}, 0, 6, 2, 0},
        {"16x16x33 RGBA8 vol", {TiledKind::texture3d, 16, 16, 33, 6, 4, 1}, 0, 6, 3, 0},
        {"16x16x48 RGBA8 vol", {TiledKind::texture3d, 16, 16, 48, 6, 4, 1}, 0, 6, 3, 0},
        {"16x16x64 RGBA8 vol", {TiledKind::texture3d, 16, 16, 64, 7, 4, 1}, 0, 7, 4, 0},
        {"16x16x128 RGBA8 vol", {TiledKind::texture3d, 16, 16, 128, 8, 4, 1}, 0, 8, 8, 0},
        {"16x16x32 R16F vol", {TiledKind::texture3d, 16, 16, 32, 6, 2, 1}, 0, 6, 1, 0},
        {"16x16x33 R16F vol", {TiledKind::texture3d, 16, 16, 33, 6, 2, 1}, 0, 6, 2, 0},
        {"16x16x64 R16F vol", {TiledKind::texture3d, 16, 16, 64, 7, 2, 1}, 0, 7, 2, 0},
        {"16x16x100 R16F vol", {TiledKind::texture3d, 16, 16, 100, 7, 2, 1}, 0, 7, 4, 0},
        {"16x16x16 RGBA16F vol", {TiledKind::texture3d, 16, 16, 16, 5, 8, 1}, 0, 5, 1, 0},
        {"16x16x17 RGBA16F vol", {TiledKind::texture3d, 16, 16, 17, 5, 8, 1}, 0, 5, 2, 0},
        {"16x16x64 RGBA16F vol", {TiledKind::texture3d, 16, 16, 64, 7, 8, 1}, 0, 7, 4, 0},
        {"8x8x16 RGBA32F vol", {TiledKind::texture3d, 8, 8, 16, 5, 16, 1}, 0, 5, 1, 0},
        {"8x8x17 RGBA32F vol", {TiledKind::texture3d, 8, 8, 17, 5, 16, 1}, 0, 5, 2, 0},
        {"8x8x64 RGBA32F vol", {TiledKind::texture3d, 8, 8, 64, 7, 16, 1}, 0, 7, 4, 0},
        {"32x32x64 R8 vol", {TiledKind::texture3d, 32, 32, 64, 7, 1, 1}, 0, 7, 2, 0},
        {"1x1x256 R8 vol", {TiledKind::texture3d, 1, 1, 256, 9, 1, 1}, 0, 9, 8, 0},
        {"16x4x100 RGBA8 vol", {TiledKind::texture3d, 16, 4, 100, 7, 4, 1}, 0, 7, 7, 0},
        {"64x8x256 R8 vol", {TiledKind::texture3d, 64, 8, 256, 9, 1, 1}, 1, 8, 4, 8},
        {"32x8x64 RGBA8 vol", {TiledKind::texture3d, 32, 8, 64, 7, 4, 1}, 1, 6, 2, 4},
        {"128x16x128 RGBA8 vol", {TiledKind::texture3d, 128, 16, 128, 8, 4, 1}, 3, 5, 1, 42},
        {"64x64x256 R8 vol", {TiledKind::texture3d, 64, 64, 256, 9, 1, 1}, 1, 8, 4, 16},
        {"32x16x64 RGBA16F vol", {TiledKind::texture3d, 32, 16, 64, 7, 8, 1}, 1, 6, 2, 4},
    };
    for (const AmdCase& c : amd) {
        if (!resource_tiling(c.desc, t) || t.standard_mips != c.standard || t.packed_mips != c.packed ||
            t.packed_tiles != c.packed_tiles || t.packed_start != c.packed_start) {
            std::fprintf(stderr, "AMD layout %s: standard %u packed %u packed tiles %u start %u, want %u %u %u %u\n",
                         c.name, t.standard_mips, t.packed_mips, t.packed_tiles, t.packed_start, c.standard, c.packed,
                         c.packed_tiles, c.packed_start);
            return 1;
        }
    }

    // 2D 4x MSAA: tile shape divides by 4 in texels, 1 mip only, unsupported sample counts fail.
    for (const auto& ms : {std::tuple{1u, 128u, 128u, 4u}, std::tuple{2u, 128u, 64u, 8u},
                           std::tuple{4u, 64u, 64u, 16u}, std::tuple{8u, 64u, 32u, 32u},
                           std::tuple{16u, 32u, 32u, 64u}}) {
        const auto [bytes, sw, sh, want_tiles] = ms;
        CHECK(resource_tiling({TiledKind::texture2d, 256, 256, 1, 1, bytes, 1, 4}, t));
        CHECK(t.shape.width == sw && t.shape.height == sh && t.shape.depth == 1 && t.total_tiles == want_tiles);
    }
    CHECK(resource_tiling({TiledKind::texture2d, 64, 64, 1, 1, 1, 1, 4}, t) && t.total_tiles == 1);
    CHECK(!resource_tiling({TiledKind::texture2d, 256, 256, 1, 1, 4, 1, 2}, t));
    CHECK(!resource_tiling({TiledKind::texture2d, 256, 256, 1, 1, 4, 1, 8}, t));
    CHECK(!resource_tiling({TiledKind::texture2d, 256, 256, 1, 2, 4, 1, 4}, t));
    CHECK(!resource_tiling({TiledKind::texture3d, 64, 64, 16, 1, 4, 1, 4}, t));

    // Element sizes that have no tile shape, and empty descriptions.
    CHECK(!resource_tiling({TiledKind::texture2d, 64, 64, 1, 1, 12, 1}, t));
    CHECK(!resource_tiling({TiledKind::texture2d, 64, 64, 1, 0, 4, 1}, t));
    CHECK(!resource_tiling({TiledKind::texture2d, 0, 64, 1, 1, 4, 1}, t));

    // Overflows and shift edge cases must fail cleanly instead of wrapping or crashing.
    // 2^25 * 2^25 * 2^14 = 2^64 wraps per_slice * layers to 0 in unchecked uint64_t arithmetic.
    CHECK(!resource_tiling({TiledKind::texture2d, 1ULL << 31, 1u << 31, 16384, 1, 16, 1}, t));
    // block * elements.width overflowing uint32_t must not produce shape.width == 0 (divide by zero).
    CHECK(!resource_tiling({TiledKind::texture2d, 128, 128, 1, 1, 16, 1u << 26}, t));
    // Buffer width near UINT64_MAX must not wrap in up().
    CHECK(!resource_tiling({TiledKind::buffer, ~std::uint64_t{0}}, t));
    // mip >= 32 must clamp height/depth to 1 without shifting uint32_t by >= 32.
    CHECK(resource_tiling({TiledKind::texture2d, 1024, 1024, 1, 33, 4, 1}, t));
    CHECK(t.standard_mips == 4 && t.packed_mips == 29 && t.packed_tiles == 1 && t.total_tiles == 86);
    std::printf("resource tiling: buffer, 2D, array, block-compressed and volume layouts match\n");
    return 0;
}
