#include "../src/gfx10_tiling.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <limits>
#include <vector>

#define CHECK(condition) do {                                      \
    if (!(condition)) {                                            \
        std::fprintf(stderr, "line %d: %s\n", __LINE__, #condition); \
        return 1;                                                  \
    }                                                              \
} while (0)

using leonida::gfx10::TileMode;
using leonida::gfx10::TiledSurface2D;

int check_surface(TileMode mode, std::size_t bpe, std::size_t width, std::size_t height) {
    TiledSurface2D surface;
    CHECK(TiledSurface2D::make(mode, bpe, width, height, surface));
    const auto bw = surface.tile_width(), bh = surface.tile_height();
    const auto pw = (width + bw - 1) / bw * bw, ph = (height + bh - 1) / bh * bh;
    CHECK(surface.tiled_size() == pw * ph * bpe);
    TiledSurface2D padded;
    CHECK(TiledSurface2D::make(mode, bpe, pw, ph, padded));
    std::vector<bool> seen(surface.tiled_size());
    for (std::size_t y = 0; y < ph; ++y) {
        for (std::size_t x = 0; x < pw; ++x) {
            std::size_t offset;
            CHECK(padded.element_offset(x, y, offset));
            CHECK(offset % bpe == 0 && offset <= seen.size() - bpe);
            for (std::size_t byte = 0; byte < bpe; ++byte) {
                CHECK(!seen[offset + byte]);
                seen[offset + byte] = true;
            }
        }
    }
    CHECK(std::all_of(seen.begin(), seen.end(), [](bool value) { return value; }));

    const std::size_t row_bytes = width * bpe;
    for (const auto pitch : {row_bytes, row_bytes + 13}) {
        const std::size_t linear_size = (height - 1) * pitch + row_bytes;
        std::vector<std::byte> linear(linear_size), tiled(surface.tiled_size(), std::byte{0xcc});
        for (std::size_t i = 0; i < linear.size(); ++i) linear[i] = std::byte((i * 37 + i / 251) & 255);
        std::vector<std::byte> decoded(linear_size, std::byte{0xab});
        CHECK(surface.tile(linear, tiled, pitch));
        CHECK(surface.detile(tiled, decoded, pitch));
        std::vector<std::byte> expected(linear_size, std::byte{0xab});
        for (std::size_t y = 0; y < height; ++y)
            std::copy_n(linear.begin() + y * pitch, row_bytes, expected.begin() + y * pitch);
        CHECK(decoded == expected);
        std::vector<std::byte> reference(linear_size, std::byte{0xab});
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                std::size_t offset;
                CHECK(surface.element_offset(x, y, offset));
                std::copy_n(tiled.begin() + offset, bpe, reference.begin() + y * pitch + x * bpe);
            }
        }
        CHECK(decoded == reference);
        std::vector<std::byte> encoded(tiled.size(), std::byte{0xcc});
        CHECK(surface.tile(decoded, encoded, pitch));
        CHECK(encoded == tiled);
        std::vector<bool> active(tiled.size());
        for (std::size_t y = 0; y < height; ++y) {
            for (std::size_t x = 0; x < width; ++x) {
                std::size_t offset;
                CHECK(surface.element_offset(x, y, offset));
                for (std::size_t byte = 0; byte < bpe; ++byte) {
                    CHECK(tiled[offset + byte] == linear[y * pitch + x * bpe + byte]);
                    active[offset + byte] = true;
                }
            }
        }
        for (std::size_t i = 0; i < tiled.size(); ++i) CHECK(active[i] || tiled[i] == std::byte{0xcc});

        const auto before_tiled = tiled, before_decoded = decoded;
        CHECK(!surface.tile(std::span(linear).first(linear_size - 1), tiled, pitch));
        CHECK(!surface.tile(linear, std::span(tiled).first(tiled.size() - 1), pitch));
        CHECK(!surface.detile(std::span(tiled).first(tiled.size() - 1), decoded, pitch));
        CHECK(!surface.detile(tiled, std::span(decoded).first(linear_size - 1), pitch));
        CHECK(!surface.tile(linear, tiled, row_bytes - 1));
        CHECK(!surface.detile(tiled, decoded, row_bytes - 1));
        CHECK(!surface.tile({}, tiled, pitch));
        CHECK(!surface.detile(tiled, {}, pitch));
        CHECK(tiled == before_tiled && decoded == before_decoded);
    }
    std::size_t offset = 123;
    CHECK(!surface.element_offset(width, 0, offset));
    CHECK(!surface.element_offset(0, height, offset));
    CHECK(offset == 123);
    std::vector<std::byte> shared(std::max(surface.tiled_size(), width * height * bpe) + 1);
    const auto before = shared;
    CHECK(!surface.tile(shared, shared, row_bytes));
    CHECK(!surface.detile(shared, shared, row_bytes));
    CHECK(!surface.tile(std::span(shared).subspan(1), std::span(shared).first(shared.size() - 1), row_bytes));
    CHECK(!surface.detile(std::span(shared).first(shared.size() - 1), std::span(shared).subspan(1), row_bytes));
    CHECK(shared == before);
    return 0;
}

int main() {
    // Source byte offsets reconstructed by calling ttpf_mips.detile on synthetic address-labelled bytes,
    // three passes (address >> 0/8/16). Extent 300x130; points (4,4), (32,16), (299,129).
    constexpr std::size_t answers[2][5][3] = {
        {{68, 2304, 59419}, {72, 3072, 100502}, {192, 5120, 201244},
         {384, 45056, 365144}, {768, 86016, 698576}},
        {{36, 2304, 84051}, {72, 3072, 198806}, {192, 9216, 336412},
         {320, 12288, 533048}, {768, 36864, 952432}}};
    constexpr std::array<std::array<std::size_t, 2>, 3> points{{{4, 4}, {32, 16}, {299, 129}}};
    unsigned mode_index = 0;
    for (const auto mode : {TileMode::sw_4kb_s, TileMode::sw_64kb_d}) {
        unsigned bpe_index = 0;
        for (const std::size_t bpe : {1, 2, 4, 8, 16}) {
            TiledSurface2D surface;
            CHECK(TiledSurface2D::make(mode, bpe, 300, 130, surface));
            for (unsigned i = 0; i < points.size(); ++i) {
                std::size_t offset;
                CHECK(surface.element_offset(points[i][0], points[i][1], offset));
                CHECK(offset == answers[mode_index][bpe_index][i]);
            }
            CHECK(check_surface(mode, bpe, 1, 1) == 0);
            CHECK(check_surface(mode, bpe, 33, 17) == 0);
            CHECK(check_surface(mode, bpe, 300, 130) == 0);
            CHECK(check_surface(mode, bpe, surface.tile_width() * 2, surface.tile_height() * 2) == 0);
            ++bpe_index;
        }
        ++mode_index;
    }
    TiledSurface2D invalid;
    for (const std::size_t bpe : {0, 3, 12, 32})
        CHECK(!TiledSurface2D::make(TileMode::sw_4kb_s, bpe, 1, 1, invalid));
    CHECK(!TiledSurface2D::make(TileMode(99), 4, 1, 1, invalid));
    CHECK(!TiledSurface2D::make(TileMode::sw_4kb_s, 4, 0, 1, invalid));
    CHECK(!TiledSurface2D::make(TileMode::sw_4kb_s, 4, 1, 0, invalid));
    constexpr auto max = std::numeric_limits<std::size_t>::max();
    for (const auto mode : {TileMode::sw_4kb_s, TileMode::sw_64kb_d}) {
        CHECK(!TiledSurface2D::make(mode, 16, max, 1, invalid));
        CHECK(!TiledSurface2D::make(mode, 1, max, max, invalid));
        CHECK(!TiledSurface2D::make(mode, 1, 1, max, invalid));
    }
    CHECK(invalid.tiled_size() == 0);
    std::size_t offset;
    CHECK(!invalid.element_offset(0, 0, offset));
    CHECK(!invalid.tile({}, {}, 0));
    CHECK(!invalid.detile({}, {}, 0));
    TiledSurface2D surface;
    CHECK(TiledSurface2D::make(TileMode::sw_4kb_s, 4, 1, 3, surface));
    std::vector<std::byte> tiled(surface.tiled_size()), linear(12);
    CHECK(!surface.tile(linear, tiled, max));
    CHECK(!surface.detile(tiled, linear, max));
    std::puts("gfx10_tiling: 40 surface round trips, padded bijections, 30 Python offsets and invalid arguments passed");
    return 0;
}
