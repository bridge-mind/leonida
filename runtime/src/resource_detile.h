#pragma once

// Detiling and tiling for AMD GFX10 / RDNA2 console-tiled resources (docs/d3d12_vulkan.md section 4.2).
//
// Resources created through Xbox slot 76 carry layout 0x101 (kXboxTiledLayout).
// The data layout inside each 64 KiB tile is GFX10 SW_64KB_D (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO):
// 64 KiB macro-blocks are laid out in row-major order across the surface, and intra-tile coordinates (x, y)
// are swizzled into byte offsets via bit-permutation equations matching RDNA2 / Xbox Series hardware.
//
// Independent X and Y coordinate bit mappings permit O(1) intra-tile offset resolution via compile-time
// lookup tables (lut_x[x % tile_w] | lut_y[y % tile_h]). No Windows or Vulkan dependency.

#include "resource_tiling.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <span>

namespace leonida {

// The Xbox tiled layout word (device slot 76, docs/d3d12_vulkan.md section 4.2).
inline constexpr std::uint32_t xbox_tiled_layout = 0x101;

constexpr bool is_xbox_tiled(std::uint32_t layout) {
    return layout == xbox_tiled_layout;
}

// Intra-tile bit permutation equations for 64 KiB tiles (Mesa addrlib GFX10_SW_64K_D_RBPLUS_PATINFO).
// For each bit 0..15:
//   s > 0: bit is ((x >> (s - 1)) & 1)
//   s < 0: bit is ((y >> (-s - 1)) & 1)
//   s == 0: intra-element byte bit (element_bytes > 1)
template <std::uint32_t ElementBytes>
struct SwizzleLut64K {
    static constexpr std::uint32_t bpe = ElementBytes;
    static constexpr TileShape shape = element_tile_shape(false, bpe);
    static constexpr std::uint32_t tile_w = shape.width;
    static constexpr std::uint32_t tile_h = shape.height;

    std::array<std::uint16_t, tile_w> lut_x{};
    std::array<std::uint16_t, tile_h> lut_y{};

    constexpr SwizzleLut64K() {
        constexpr std::array<int8_t, 16> pat = []() {
            if constexpr (bpe == 1) {
                return std::array<int8_t, 16>{1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8};
            } else if constexpr (bpe == 2) {
                return std::array<int8_t, 16>{0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8};
            } else if constexpr (bpe == 4) {
                // RDNA2/RbPlus SW_64KB_D: bits 6..7 swapped vs Navi 1x ((3, -3) instead of (-3, 3))
                return std::array<int8_t, 16>{0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7};
            } else if constexpr (bpe == 8) {
                return std::array<int8_t, 16>{0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7};
            } else if constexpr (bpe == 16) {
                return std::array<int8_t, 16>{0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6};
            } else {
                return std::array<int8_t, 16>{};
            }
        }();

        for (std::uint32_t x = 0; x < tile_w; ++x) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s > 0) {
                    val |= static_cast<std::uint16_t>(((x >> (s - 1)) & 1) << bit);
                }
            }
            lut_x[x] = val;
        }

        for (std::uint32_t y = 0; y < tile_h; ++y) {
            std::uint16_t val = 0;
            for (std::uint32_t bit = 0; bit < 16; ++bit) {
                const int8_t s = pat[bit];
                if (s < 0) {
                    val |= static_cast<std::uint16_t>(((y >> (-s - 1)) & 1) << bit);
                }
            }
            lut_y[y] = val;
        }
    }
};

inline constexpr SwizzleLut64K<1> swizzle_lut_64k_1bpe{};
inline constexpr SwizzleLut64K<2> swizzle_lut_64k_2bpe{};
inline constexpr SwizzleLut64K<4> swizzle_lut_64k_4bpe{};
inline constexpr SwizzleLut64K<8> swizzle_lut_64k_8bpe{};
inline constexpr SwizzleLut64K<16> swizzle_lut_64k_16bpe{};

namespace detail {

constexpr bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
    if (a != 0 && b > (std::numeric_limits<std::size_t>::max)() / a) return false;
    out = a * b;
    return true;
}

struct SurfaceLayout64K {
    std::uint32_t blocks_w = 0;
    std::uint32_t blocks_h = 0;
    std::uint32_t tiles_x = 0;
    std::uint32_t tiles_y = 0;
    std::size_t linear_bytes = 0;
    std::size_t tiled_bytes = 0;
};

constexpr bool compute_surface_layout_64k(std::uint32_t width, std::uint32_t height,
                                          std::uint32_t element_bytes, std::uint32_t block_dim,
                                          SurfaceLayout64K& out) {
    if (!width || !height || (block_dim != 1 && block_dim != 4)) return false;
    const TileShape shape = element_tile_shape(false, element_bytes);
    if (!shape.width || !shape.height) return false;

    // Overflow-free ceiling division for positive uint32_t operands: 1 + (n - 1) / d.
    const std::uint32_t blocks_w = 1u + (width - 1u) / block_dim;
    const std::uint32_t blocks_h = 1u + (height - 1u) / block_dim;
    const std::uint32_t tiles_x = 1u + (blocks_w - 1u) / shape.width;
    const std::uint32_t tiles_y = 1u + (blocks_h - 1u) / shape.height;

    std::size_t block_count = 0;
    std::size_t linear_bytes = 0;
    std::size_t tile_count = 0;
    std::size_t tiled_bytes = 0;
    if (!checked_mul(blocks_w, blocks_h, block_count) ||
        !checked_mul(block_count, element_bytes, linear_bytes) ||
        !checked_mul(tiles_x, tiles_y, tile_count) ||
        !checked_mul(tile_count, tile_size_in_bytes, tiled_bytes)) {
        return false;
    }

    out = {blocks_w, blocks_h, tiles_x, tiles_y, linear_bytes, tiled_bytes};
    return true;
}

inline bool spans_overlap(const std::byte* a, std::size_t a_len,
                          const std::byte* b, std::size_t b_len) {
    if (!a_len || !b_len) return false;
    const std::less<const std::byte*> lt{};
    return lt(a, b + b_len) && lt(b, a + a_len);
}

}  // namespace detail

// Compute required tiled byte size for a 2D surface of the given dimensions (0 on invalid input or overflow).
constexpr std::size_t tiled_surface_size_64k(std::uint32_t width, std::uint32_t height,
                                            std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    return detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)
               ? layout.tiled_bytes
               : 0;
}

// Compute required linear byte size for a 2D surface of the given dimensions (0 on invalid input or overflow).
constexpr std::size_t linear_surface_size(std::uint32_t width, std::uint32_t height,
                                         std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    return detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)
               ? layout.linear_bytes
               : 0;
}

namespace detail {

template <std::uint32_t Bpe>
void detile_typed(std::span<const std::byte> tiled, std::span<std::byte> linear,
                  std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    constexpr const auto& lut = []() constexpr -> const auto& {
        if constexpr (Bpe == 1) return swizzle_lut_64k_1bpe;
        else if constexpr (Bpe == 2) return swizzle_lut_64k_2bpe;
        else if constexpr (Bpe == 4) return swizzle_lut_64k_4bpe;
        else if constexpr (Bpe == 8) return swizzle_lut_64k_8bpe;
        else if constexpr (Bpe == 16) return swizzle_lut_64k_16bpe;
    }();

    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;

    for (std::uint32_t y = 0; y < blocks_h; ++y) {
        const std::uint32_t ty = y / tile_h;
        const std::uint16_t intra_y = lut.lut_y[y % tile_h];
        const std::size_t macro_y_base = static_cast<std::size_t>(ty) * tiles_x * tile_size_in_bytes;
        const std::size_t linear_row = static_cast<std::size_t>(y) * blocks_w * Bpe;

        for (std::uint32_t x = 0; x < blocks_w; ++x) {
            const std::uint32_t tx = x / tile_w;
            const std::uint16_t intra_x = lut.lut_x[x % tile_w];
            const std::size_t tiled_off = macro_y_base + static_cast<std::size_t>(tx) * tile_size_in_bytes + (intra_x | intra_y);
            const std::size_t linear_off = linear_row + static_cast<std::size_t>(x) * Bpe;

            std::memcpy(&linear[linear_off], &tiled[tiled_off], Bpe);
        }
    }
}

template <std::uint32_t Bpe>
void tile_typed(std::span<const std::byte> linear, std::span<std::byte> tiled,
                std::uint32_t blocks_w, std::uint32_t blocks_h, std::uint32_t tiles_x) {
    constexpr const auto& lut = []() constexpr -> const auto& {
        if constexpr (Bpe == 1) return swizzle_lut_64k_1bpe;
        else if constexpr (Bpe == 2) return swizzle_lut_64k_2bpe;
        else if constexpr (Bpe == 4) return swizzle_lut_64k_4bpe;
        else if constexpr (Bpe == 8) return swizzle_lut_64k_8bpe;
        else if constexpr (Bpe == 16) return swizzle_lut_64k_16bpe;
    }();

    constexpr std::uint32_t tile_w = lut.tile_w;
    constexpr std::uint32_t tile_h = lut.tile_h;

    for (std::uint32_t y = 0; y < blocks_h; ++y) {
        const std::uint32_t ty = y / tile_h;
        const std::uint16_t intra_y = lut.lut_y[y % tile_h];
        const std::size_t macro_y_base = static_cast<std::size_t>(ty) * tiles_x * tile_size_in_bytes;
        const std::size_t linear_row = static_cast<std::size_t>(y) * blocks_w * Bpe;

        for (std::uint32_t x = 0; x < blocks_w; ++x) {
            const std::uint32_t tx = x / tile_w;
            const std::uint16_t intra_x = lut.lut_x[x % tile_w];
            const std::size_t tiled_off = macro_y_base + static_cast<std::size_t>(tx) * tile_size_in_bytes + (intra_x | intra_y);
            const std::size_t linear_off = linear_row + static_cast<std::size_t>(x) * Bpe;

            std::memcpy(&tiled[tiled_off], &linear[linear_off], Bpe);
        }
    }
}

}  // namespace detail

// Detile a 2D surface from console-tiled 64 KiB layout into linear row-major order.
inline bool detile_surface_64k(std::span<const std::byte> tiled, std::span<std::byte> linear,
                               std::uint32_t width, std::uint32_t height,
                               std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    if (!detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)) return false;
    if (linear.size() < layout.linear_bytes || tiled.size() < layout.tiled_bytes) return false;
    if (detail::spans_overlap(tiled.data(), layout.tiled_bytes, linear.data(), layout.linear_bytes)) return false;

    switch (element_bytes) {
    case 1:
        detail::detile_typed<1>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 2:
        detail::detile_typed<2>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 4:
        detail::detile_typed<4>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 8:
        detail::detile_typed<8>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 16:
        detail::detile_typed<16>(tiled, linear, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    default:
        return false;
    }
}

// Tile a 2D surface from linear row-major order into console-tiled 64 KiB layout.
inline bool tile_surface_64k(std::span<const std::byte> linear, std::span<std::byte> tiled,
                             std::uint32_t width, std::uint32_t height,
                             std::uint32_t element_bytes, std::uint32_t block_dim = 1) {
    detail::SurfaceLayout64K layout{};
    if (!detail::compute_surface_layout_64k(width, height, element_bytes, block_dim, layout)) return false;
    if (linear.size() < layout.linear_bytes || tiled.size() < layout.tiled_bytes) return false;
    if (detail::spans_overlap(linear.data(), layout.linear_bytes, tiled.data(), layout.tiled_bytes)) return false;

    switch (element_bytes) {
    case 1:
        detail::tile_typed<1>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 2:
        detail::tile_typed<2>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 4:
        detail::tile_typed<4>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 8:
        detail::tile_typed<8>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    case 16:
        detail::tile_typed<16>(linear, tiled, layout.blocks_w, layout.blocks_h, layout.tiles_x);
        return true;
    default:
        return false;
    }
}

}  // namespace leonida
