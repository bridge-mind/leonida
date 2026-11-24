"""Measure and decode TTPF (RDS5) textures: 2D compact mip chains, 2D arrays,
streaming cubemaps, and 3D volumes.

The compact profile in `TtpfReader` is the AMD Gfx10 mip-tail layout: mips smaller
than half a tile share one tile at offset 0, and the rest are standalone, smallest
first. `TtpfReader` only trusts a hand-validated tuple table and stops at the
descriptor's `nonTail` field, which is not the layout boundary. Multi-slice
descriptors use the same 16-byte ETPF record with a 7-bit DXGI format
(`(b >> 3) & 0x7F`) and an 11-bit depth (`((b >> 10) & 0x7FF) + 1`):
  - 2D arrays (`dimension = 1, depth > 1, mode = 1`): mip-major SW_64KB_D.
  - Cubemaps (`dimension = 2, depth = 6, mode = 18 + 2 * non_tail`): `non_tail + 1`
    back-to-back streaming snapshots; the final chunk holds the full-res 6-face
    slice-major SW_4KB_S cubemap.
  - 3D volumes (`dimension = 3, depth > 1, mode = 2, mips = 1`): single-mip 3D
    SW_4KB_S tiling.

The layout rules live in `tex_headers.py`; this tool supplies the RDS5 reader,
decoders and commands. It computes the layout from the descriptor, validates it
against the native allocation span, then decodes zlib/raw mips and supported
BCPack codecs (BC1, BC3, BC4, BC5, BC6H and BC7 via `bcpack.py`, strict
confirmed-history mode by default; pass `--allow-provisional` to opt into
unconfirmed built-in history entries, and `--history <path>` to supply locally
generated BC6H/BC7 initial history tables).

Read-only: no texture bytes are printed or committed. `scan` emits aggregate
counters only (both the zlib/raw baseline and the BCPack-enabled projection).
`layout` prints descriptor fields and offsets. `check` prints the BC
block-average error against a 2x2 downsample of the parent mip (across all
slices/faces for arrays and cubemaps; BC1, BC2, BC3, BC4_UNORM, BC5_UNORM, BC6H and
BC7) or 3D Z-continuity vs 2D control for float16 volumes. `export` writes a
DX10 DDS when every source mip uses a supported codec.

    python -I tools/ttpf_mips.py scan <game root, pool folders or pool paths...> [--json out] [--tuples]
    python -I tools/ttpf_mips.py layout <pool.ttpf> <index>
    python -I tools/ttpf_mips.py check <pool.ttpf> <index> [--allow-provisional] [--history path]
    python -I tools/ttpf_mips.py export <pool.ttpf> <index> <out.dds> [--mips N] [--force] [--allow-provisional] [--history path]

Standard library only. The tile patterns and tail formula are the public AMD
Gfx10 4 KiB/64 KiB layouts recorded in docs/textures.md and docs/textures_arrays.md.
Usage: `python -I tools/ttpf_mips.py --help` lists the options.
"""
import argparse
import glob
import json
import math
import mmap
import operator
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcpack  # noqa: E402
import bcpack_tables_bptc as _bptc  # noqa: E402
from tex_headers import (  # noqa: E402
    array_surface_layout, compact_surface_layout, cubemap_surface_layout,
    decode_etpf_chunk_pages, etpf_streaming_padding, gfx10_tile_bytes,
    gfx10_tile_mode, parse_etpf_texture_record, volume_surface_layout,
)

MAX_OUTPUT = 256 * 1024 * 1024
MAX_TEXTURES = 250_000
ETPF_VERSION = 634

# DXGI format -> element bytes, matching RageTextureDecoder.ElementBytes plus DXGI 11
# (R16G16B16A16_UNORM, 8 bytes).
ELEMENT_BYTES = {
    2: 16, 3: 16, 4: 16, 6: 12, 7: 12, 8: 12, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8,
    16: 8, 17: 8, 18: 8, 24: 4, 25: 4, 26: 4, 28: 4, 29: 4, 30: 4, 31: 4, 32: 4,
    34: 4, 35: 4, 36: 4, 37: 4, 38: 4, 41: 4, 42: 4, 43: 4, 49: 2, 50: 2, 51: 2,
    52: 2, 54: 2, 56: 2, 57: 2, 58: 2, 59: 2, 61: 1, 62: 1, 63: 1, 64: 1, 65: 1,
    71: 8, 72: 8, 74: 16, 75: 16, 77: 16, 78: 16, 80: 8, 81: 8, 83: 16, 84: 16,
    87: 4, 88: 4, 91: 4, 93: 4, 95: 16, 96: 16, 98: 16, 99: 16,
}
BC_FORMATS = frozenset(range(70, 85)) | frozenset(range(94, 100))
# Validated 2D/array/cubemap uncompressed DXGI formats decoded in `check` and exported in `export`.
UNCOMPRESSED_2D_FORMATS = frozenset({10, 11, 28, 29, 49, 54, 56, 61, 65, 87, 91})

# AMD Gfx10/RDNA2 (RbPlus) intra-tile bit order (bit index -> source bit), matching
# Mesa addrlib GFX10_SW_4K_S_RBPLUS_PATINFO and GFX10_SW_64K_D_RBPLUS_PATINFO.
PATTERNS_4K = {
    0: (1, 2, 3, 4, -1, -2, -3, -4, -5, 5, -6, 6),
    1: (0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6),
    2: (0, 0, 1, 2, -1, -2, -3, 3, -4, 4, -5, 5),
    3: (0, 0, 0, 1, -1, -2, 2, 3, -3, 4, -4, 5),
    4: (0, 0, 0, 0, -1, -2, 1, 2, -3, 3, -4, 4),
}
PATTERNS_64K = {
    0: (1, 2, 3, -2, -1, -3, 4, -4, -5, 5, -6, 6, -7, 7, -8, 8),
    1: (0, 1, 2, 3, -1, -2, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8),
    # RDNA2/RbPlus SW_64KB_D swaps bits 6..7 vs Navi 1x ((3, -3) instead of (-3, 3)).
    2: (0, 0, 1, 2, -1, -2, 3, -3, -4, 4, -5, 5, -6, 6, -7, 7),
    3: (0, 0, 0, 1, -1, 2, 3, -2, -3, 4, -4, 5, -5, 6, -6, 7),
    4: (0, 0, 0, 0, 1, -1, 2, -2, -3, 3, -4, 4, -5, 5, -6, 6),
}
# AMD Gfx10 3D SW_4KB_S intra-tile bit order: Mesa addrlib GFX10_SW_4K_S3_PATINFO (and the identical
# GFX10_SW_4K_S3_RBPLUS_PATINFO), first nibble01 + nibble2 entries per element size.
# >0 and <100 = X bit (s-1), <0 = Y bit (-s-1), >=101 = Z bit (s-101). The tile extents in
# tex_headers._VOLUME_4K_TILE are 2**(number of X, Y and Z bits) of each row.
VOLUME_PATTERNS_4K = {
    0: (1, 2, 101, -1, 102, -2, 3, 103, -3, 4, 104, -4),
    1: (0, 1, 101, -1, 102, -2, 2, 103, -3, 3, 104, -4),
    2: (0, 0, 1, -1, 101, -2, 2, 102, -3, 3, 103, -4),
    3: (0, 0, 0, 1, 101, -1, 2, 102, -2, 3, 103, -3),
    4: (0, 0, 0, 0, 101, -1, 1, 102, -2, 2, 103, -3),
}
VOLUME_FLOAT_FORMATS = {10: 4, 54: 1}  # DXGI format -> float16 channels per voxel

# The profile tuples `TtpfReader.ValidatedCompactProfile` accepts. Reproduced only
# so `scan` can show the reader's published skip/partial split next to the layout
# result; decoding does not depend on this table.
VALIDATED_COMPACT = {
    (4, 1024, 1024, 9, 3, 8, 0x503, 0x00B00106),
    (6, 512, 512, 8, 3, 8, 0x041, 0x00B800D4),
    (4, 256, 256, 7, 1, 8, 0x103, 0),
    (4, 512, 512, 8, 3, 16, 0x482, 0x015000F5),
    (4, 1024, 512, 8, 3, 8, 0x482, 0x015000F5),
    (4, 512, 1024, 8, 3, 8, 0x483, 0x013000F5),
    (4, 128, 256, 6, 1, 8, 0x083, 0),
    (4, 256, 512, 7, 1, 8, 0x167, 0),
    (4, 512, 256, 7, 1, 8, 0x166, 0),
    (4, 256, 256, 7, 1, 16, 0x166, 0),
    (4, 1024, 512, 8, 3, 16, 0x504, 0x900106),
    (4, 1024, 256, 7, 3, 16, 0x484, 0x11000F5),
    (6, 1024, 256, 7, 3, 8, 0x42, 0x9800D4),
    (6, 64, 1024, 5, 3, 8, 0x83, 0x3800B4),
    (4, 512, 1024, 8, 3, 16, 0x503, 0xB00106),
    (4, 512, 128, 6, 1, 16, 0x168, 0),
    (6, 256, 1024, 7, 3, 8, 0x83, 0x3800D4),
    (4, 1024, 128, 6, 1, 8, 0x168, 0),
    (4, 2048, 256, 7, 3, 16, 0x508, 0x100106),
    (4, 2048, 512, 8, 3, 8, 0x504, 0x900106),
    (6, 512, 256, 7, 3, 16, 0x42, 0x9800D4),
    (6, 1024, 128, 6, 3, 16, 0x84, 0x1800D4),
    (4, 512, 128, 6, 1, 8, 0x104, 0),
    (4, 256, 1024, 7, 3, 16, 0x483, 0x13000F5),
    (6, 1024, 64, 5, 3, 16, 0x83, 0x3800B4),
    (6, 256, 512, 7, 3, 16, 0x41, 0xB800D4),
    (4, 128, 128, 6, 1, 16, 0x82, 0),
    (4, 512, 64, 5, 1, 16, 0x108, 0),
    (4, 256, 128, 6, 1, 16, 0x104, 0),
    (6, 128, 1024, 6, 3, 16, 0x83, 0x3800D4),
    (4, 512, 64, 5, 1, 8, 0x84, 0),
    (4, 128, 512, 6, 1, 16, 0x167, 0),
    (4, 256, 128, 6, 1, 8, 0x82, 0),
    (4, 1024, 64, 5, 1, 8, 0x108, 0),
    (6, 1024, 32, 4, 3, 16, 0x82, 0x5800B4),
    (4, 128, 512, 6, 1, 8, 0x107, 0),
    (4, 512, 2048, 8, 3, 8, 0x507, 0x300106),
    (4, 256, 2048, 7, 3, 8, 0x507, 0x3000F5),
    (4, 512, 32, 4, 1, 16, 0x107, 0),
    (6, 128, 1024, 6, 3, 8, 0x83, 0x3800B4),
    (6, 64, 1024, 5, 3, 16, 0x83, 0x3800B4),
    (4, 256, 256, 5, 1, 16, 0x166, 0),
    (4, 128, 256, 6, 1, 16, 0x103, 0),
    (4, 256, 64, 5, 1, 16, 0x84, 0),
    (4, 16, 256, 3, 1, 16, 0x83, 0),
    (4, 256, 32, 4, 1, 16, 0x84, 0),
    (4, 64, 512, 5, 1, 16, 0x107, 0),
    (4, 2048, 256, 7, 3, 8, 0x484, 0x11000F5),
    (4, 64, 512, 5, 1, 8, 0x107, 0),
    (4, 64, 256, 5, 1, 8, 0x83, 0),
    (6, 256, 512, 9, 3, 16, 0x41, 0xB800D4),
    (6, 32, 1024, 4, 3, 16, 0x82, 0x5800B4),
    (4, 512, 512, 3, 3, 8, 0x502, 0xD00006),
    (36, 1024, 1024, 9, 3, 8, 0x503, 0xB00106),
    (36, 256, 256, 7, 1, 8, 0x103, 0),
}
VALIDATED_AUX = dict(mips=3, non_tail=3, meta=0x1A8, extra=0x01080008, width=1024,
                     height_a=512, fmt_a=61, height_b=1024, bytes_b=8)


class LayoutError(ValueError):
    pass


class BcpackRequired(NotImplementedError):
    pass


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def align(v, n):
    return (v + n - 1) // n * n


def tile_mode(mode):
    """Descriptor mode is a flag bit plus a 4-bit Gfx10 tile mode at bit 1."""
    return gfx10_tile_mode(mode)


def tile_bytes(mode):
    # Tile mode 0 is SW_64KB_D; 1-3 are SW_4KB_S; 9-12 appear only on arrays.
    return gfx10_tile_bytes(mode)


def element_bytes(fmt):
    if fmt not in ELEMENT_BYTES:
        raise LayoutError(f"no element-size mapping for DXGI format {fmt}")
    return ELEMENT_BYTES[fmt]


def surface_layout(width, height, fmt, mips, tile=4096):
    """AMD Gfx10 surface layout: standalone mips smallest-first, then one tail tile."""
    try:
        return compact_surface_layout(width, height, mips, element_bytes(fmt),
                                      4 if fmt in BC_FORMATS else 1, tile)
    except ValueError as exc:
        raise LayoutError(str(exc)) from exc


def _coalesce_x_runs(x_offs, eb):
    """Merge consecutive contiguous X offsets into (start_offset, byte_len) runs."""
    if not x_offs:
        return []
    runs = []
    start = x_offs[0]
    run_len = eb
    for xo in x_offs[1:]:
        if xo == start + run_len:
            run_len += eb
        else:
            runs.append((start, run_len))
            start = xo
            run_len = eb
    runs.append((start, run_len))
    return runs


def detile(layout, mip, tiled):
    """Copy the mip's elements out of the tiled allocation into block-linear order."""
    tile = layout["tile"]
    eb = layout["element_bytes"]
    pat = (PATTERNS_4K if tile == 4096 else PATTERNS_64K)[
        {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[eb]]
    bw, bh = layout["block_w"], layout["block_h"]
    w, h = mip["blocks_w"], mip["blocks_h"]
    out = bytearray(w * h * eb)
    if w == 0 or h == 0:
        return bytes(out)
    tx, ty = mip["tile_x"], mip["tile_y"]
    tiles_x = mip["padded_w"] // bw
    x_bits = [(1 << bit, s - 1) for bit, s in enumerate(pat) if s > 0]
    y_bits = [(1 << bit, -s - 1) for bit, s in enumerate(pat) if s < 0]

    row_stride = tiles_x * tile

    def y_base_of(y):
        o = (y // bh) * row_stride
        yy = y + ty
        for mask, shift in y_bits:
            if (yy >> shift) & 1:
                o |= mask
        return o

    def x_off_of(x):
        o = (x // bw) * tile
        xx = x + tx
        for mask, shift in x_bits:
            if (xx >> shift) & 1:
                o |= mask
        return o

    if h == 1:
        # A single row has no X-table reuse: stream it, allocating no per-column or per-row tables.
        y_base = y_base_of(0)
        for x in range(w):
            off = y_base + x_off_of(x)
            at = x * eb
            out[at:at + eb] = tiled[off:off + eb]
        return bytes(out)

    x_offs = [x_off_of(x) for x in range(w)]
    max_x = max(x_offs)
    max_y = max(map(y_base_of, range(h)))

    if len(tiled) >= max_y + max_x + eb:
        x_runs = _coalesce_x_runs(x_offs, eb)
        src = memoryview(tiled)
        dst = memoryview(out)
        at = 0
        for y in range(h):
            y_base = y_base_of(y)
            for xo, run in x_runs:
                off = y_base + xo
                dst[at:at + run] = src[off:off + run]
                at += run
    else:
        for y in range(h):
            y_base = y_base_of(y)
            row_at = y * w * eb
            for x, xo in enumerate(x_offs):
                off = y_base + xo
                at = row_at + x * eb
                out[at:at + eb] = tiled[off:off + eb]
    return bytes(out)


def detile_volume_slice(layout, tiled, z):
    """Copy one Z slice out of a 3D SW_4KB_S tiled volume into row-major linear order."""
    tile = layout["tile"]
    eb = layout["element_bytes"]
    pat = VOLUME_PATTERNS_4K[{1: 0, 2: 1, 4: 2, 8: 3, 16: 4}[eb]]
    bw, bh, bd = layout["block_w"], layout["block_h"], layout["block_d"]
    mip = layout["mips"][0]
    w, h = mip["blocks_w"], mip["blocks_h"]
    out = bytearray(w * h * eb)
    if w == 0 or h == 0:
        return bytes(out)
    tiles_x = mip["padded_w"] // bw
    z_base = (z // bd) * layout["z_tile_stride"]
    z_in_tile = 0
    for bit, s in enumerate(pat):
        if s >= 101:
            z_in_tile |= ((z >> (s - 101)) & 1) << bit
    x_bits = [(1 << bit, s - 1) for bit, s in enumerate(pat) if 0 < s < 100]
    y_bits = [(1 << bit, -s - 1) for bit, s in enumerate(pat) if s < 0]

    row_stride = tiles_x * tile

    def y_base_of(y):
        o = z_base + (y // bh) * row_stride + z_in_tile
        for mask, shift in y_bits:
            if (y >> shift) & 1:
                o |= mask
        return o

    def x_off_of(x):
        o = (x // bw) * tile
        for mask, shift in x_bits:
            if (x >> shift) & 1:
                o |= mask
        return o

    if h == 1:
        # A single row has no X-table reuse: stream it, allocating no per-column or per-row tables.
        y_base = y_base_of(0)
        for x in range(w):
            off = y_base + x_off_of(x)
            at = x * eb
            out[at:at + eb] = tiled[off:off + eb]
        return bytes(out)

    x_offs = [x_off_of(x) for x in range(w)]
    max_x = max(x_offs)
    max_y = max(map(y_base_of, range(h)))

    if len(tiled) >= max_y + max_x + eb:
        x_runs = _coalesce_x_runs(x_offs, eb)
        src = memoryview(tiled)
        dst = memoryview(out)
        at = 0
        for y in range(h):
            y_base = y_base_of(y)
            for xo, run in x_runs:
                off = y_base + xo
                dst[at:at + run] = src[off:off + run]
                at += run
    else:
        for y in range(h):
            y_base = y_base_of(y)
            row_at = y * w * eb
            for x, xo in enumerate(x_offs):
                off = y_base + xo
                at = row_at + x * eb
                out[at:at + eb] = tiled[off:off + eb]
    return bytes(out)


def detile_volume(layout, tiled):
    """Detile all Z slices of a 3D SW_4KB_S volume into Z-major linear order."""
    return b"".join(detile_volume_slice(layout, tiled, z) for z in range(layout["depth"]))


# --- BC decoding (color only; enough for the downsample check) -------------

# The RGB565 channel levels, one float per value. Each is the same `i * 255.0 / max` the
# per-pixel expression computed, so every decoded float is unchanged.
_R565 = tuple(i * 255.0 / 31 for i in range(32))
_G565 = tuple(i * 255.0 / 63 for i in range(64))
_B565 = _R565
_GREY = tuple((float(v),) * 3 for v in range(256))  # one shared (v, v, v) per BC4 level
_BC1_HEAD = struct.Struct("<HHI").unpack_from
_ROW_MAX_CACHE = 1 << 14  # palettes kept per decode_pixels call; reset when full


def _rgb565(c):
    return _R565[(c >> 11) & 31], _G565[(c >> 5) & 63], _B565[c & 31]


def _row_getters(index_bits):
    """One itemgetter per packed row of four palette indices: getter[row](pal) is that
    row's four palette entries, in pixel order (the lowest index bits are the left pixel)."""
    mask = (1 << index_bits) - 1
    return [operator.itemgetter(*((row >> (index_bits * x)) & mask for x in range(4)))
            for row in range(1 << (4 * index_bits))]


_BC1_ROWS = _row_getters(2)  # a byte of BC1 indices -> one row
_BC4_ROWS = []  # 4096 getters (twelve BC4 index bits per row), built on first use


def _bc4_rows():
    if not _BC4_ROWS:
        _BC4_ROWS.extend(_row_getters(3))
    return _BC4_ROWS


def _bc1_palette(c0, c1, four_color):
    r0, g0, b0 = _R565[(c0 >> 11) & 31], _G565[(c0 >> 5) & 63], _B565[c0 & 31]
    r1, g1, b1 = _R565[(c1 >> 11) & 31], _G565[(c1 >> 5) & 63], _B565[c1 & 31]
    if four_color or c0 > c1:
        return ((r0, g0, b0), (r1, g1, b1),
                ((2 * r0 + r1) / 3, (2 * g0 + g1) / 3, (2 * b0 + b1) / 3),
                ((r0 + 2 * r1) / 3, (g0 + 2 * g1) / 3, (b0 + 2 * b1) / 3))
    return ((r0, g0, b0), (r1, g1, b1),
            ((r0 + r1) / 2, (g0 + g1) / 2, (b0 + b1) / 2), (0.0, 0.0, 0.0))


def _bc4_palette(e0, e1):
    if e0 > e1:
        vals = [e0, e1] + [((7 - i) * e0 + i * e1) // 7 for i in range(1, 7)]
    else:
        vals = [e0, e1] + [((5 - i) * e0 + i * e1) // 5 for i in range(1, 5)] + [0, 255]
    return tuple([_GREY[v] for v in vals])


def decode_bc1_block(data, at, four_color=False):
    c0, c1, idx = _BC1_HEAD(data, at)
    pal, rows = _bc1_palette(c0, c1, four_color), _BC1_ROWS
    return [*rows[idx & 255](pal), *rows[(idx >> 8) & 255](pal),
            *rows[(idx >> 16) & 255](pal), *rows[idx >> 24](pal)]


def decode_bc4_block(data, at):
    e0, e1 = data[at], data[at + 1]
    bits = int.from_bytes(data[at + 2:at + 8], "little")
    pal, rows = _bc4_palette(e0, e1), _bc4_rows()
    return [*rows[bits & 4095](pal), *rows[(bits >> 12) & 4095](pal),
            *rows[(bits >> 24) & 4095](pal), *rows[bits >> 36](pal)]


_BPTC_W2 = (0, 21, 43, 64)
_BPTC_W3 = (0, 9, 18, 27, 37, 46, 55, 64)
_BPTC_W4 = (0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64)
_BPTC_WEIGHTS = {2: _BPTC_W2, 3: _BPTC_W3, 4: _BPTC_W4}

# Khronos BPTC BC7 mode table:
# (subsets, part_bits, rot_bits, isel_bits, cb, ab, ep_pbits, shared_pbits, ib, ib2)
_BC7_MODES = (
    (3, 4, 0, 0, 4, 0, 1, 0, 3, 0),
    (2, 6, 0, 0, 6, 0, 0, 1, 3, 0),
    (3, 6, 0, 0, 5, 0, 0, 0, 2, 0),
    (2, 6, 0, 0, 7, 0, 1, 0, 2, 0),
    (1, 0, 2, 1, 5, 6, 0, 0, 2, 3),
    (1, 0, 2, 0, 7, 8, 0, 0, 2, 2),
    (1, 0, 0, 0, 7, 7, 1, 0, 4, 0),
    (2, 6, 0, 0, 5, 5, 1, 0, 2, 0),
)

_F16_LUT = None
_BC6H_HDR_FORMATS = {95, 96}
_HDR_FORMATS = frozenset({10, 54, 95, 96})


def _f16_lut():
    """Lazily build a 65536-entry uint16 -> float lookup table for IEEE 754 float16."""
    global _F16_LUT
    if _F16_LUT is None:
        raw = struct.pack("<65536H", *range(65536))
        _F16_LUT = struct.unpack("<65536e", raw)
    return _F16_LUT


def _sext(v, bits):
    sign = 1 << (bits - 1)
    return (v ^ sign) - sign


def _bc6h_unquantize(comp, bits, signed):
    """Unquantize one BC6H endpoint component per Khronos Data Format Spec 1.3."""
    if not signed:
        if bits >= 15:
            return comp
        if comp == 0:
            return 0
        if comp == (1 << bits) - 1:
            return 0xFFFF
        return ((comp << 16) + 0x8000) >> bits
    if bits >= 16:
        return -32767 if comp < -32767 else comp
    neg = comp < 0
    c = -comp if neg else comp
    if c == 0:
        unq = 0
    elif c >= (1 << (bits - 1)) - 1:
        unq = 0x7FFF
    else:
        unq = ((c << 15) + 0x4000) >> (bits - 1)
    return -unq if neg else unq


def _bc6h_finish(val, signed, lut):
    """Finish BC6H unquantization to a finite IEEE 754 half-float."""
    if not signed:
        h = (val * 31) >> 6
        return lut[h if h <= 0x7BFF else 0x7BFF]
    neg = val < 0
    v = -val if neg else val
    if v > 0x7FFF:
        v = 0x7FFF
    h = (v * 31) >> 5
    if h > 0x7BFF:
        h = 0x7BFF
    return lut[h | 0x8000] if neg else lut[h]


def decode_bc6h_block(data, at, *, signed=False):
    """Decode one 16-byte BC6H block (UF16/SF16) to 16 linear float16 (r, g, b) tuples."""
    b0 = data[at]
    code = (b0 & 3) if (b0 & 3) < 2 else (b0 & 31)
    info = bcpack._BC6H_INFO.get(code)
    if info is None:
        return [(0.0, 0.0, 0.0)] * 16
    trans, base, deltas, reg = info
    plan, hdr = bcpack._bc6h_plan(code)
    blk = int.from_bytes(data[at:at + 16], "little")
    f = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    part = 0
    for c, q, bit_idx, pos in plan:
        if (blk >> pos) & 1:
            if c < 0:
                part |= 1 << bit_idx
            else:
                f[c][q] |= 1 << bit_idx

    mask_base = (1 << base) - 1
    eps = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    n_ep = 4 if reg == 2 else 2
    for c in range(3):
        dc = deltas[c]
        e0 = _sext(f[c][0], base) if signed else f[c][0]
        eps[c][0] = _bc6h_unquantize(e0, base, signed)
        for q in range(1, n_ep):
            eq = _sext(f[c][q], dc) if (signed or trans) else f[c][q]
            if trans:
                eq = (e0 + eq) & mask_base
                if signed:
                    eq = _sext(eq, base)
            eps[c][q] = _bc6h_unquantize(eq, base, signed)

    lut = _f16_lut()
    pos = hdr
    out = [None] * 16
    if reg == 1:
        weights = _BPTC_W4
        r0, r1 = eps[0][0], eps[0][1]
        g0, g1 = eps[1][0], eps[1][1]
        b0e, b1e = eps[2][0], eps[2][1]
        for p in range(16):
            nb = 3 if p == 0 else 4
            w = weights[(blk >> pos) & ((1 << nb) - 1)]
            pos += nb
            iw = 64 - w
            out[p] = (
                _bc6h_finish((iw * r0 + w * r1 + 32) >> 6, signed, lut),
                _bc6h_finish((iw * g0 + w * g1 + 32) >> 6, signed, lut),
                _bc6h_finish((iw * b0e + w * b1e + 32) >> 6, signed, lut),
            )
    else:
        tab = _bptc.load_bc7()
        sm = tab["partition2"][part]
        anchor1 = tab["anchor2"][part]
        weights = _BPTC_W3
        for p in range(16):
            nb = 2 if (p == 0 or p == anchor1) else 3
            w = weights[(blk >> pos) & ((1 << nb) - 1)]
            pos += nb
            iw = 64 - w
            s2 = sm[p] * 2
            out[p] = (
                _bc6h_finish((iw * eps[0][s2] + w * eps[0][s2 + 1] + 32) >> 6, signed, lut),
                _bc6h_finish((iw * eps[1][s2] + w * eps[1][s2 + 1] + 32) >> 6, signed, lut),
                _bc6h_finish((iw * eps[2][s2] + w * eps[2][s2 + 1] + 32) >> 6, signed, lut),
            )
    return out


def decode_bc7_block(data, at):
    """Decode one 16-byte BC7 block (all 8 modes) to 16 (r, g, b) float tuples in 0..255."""
    b0 = data[at]
    if b0 == 0:
        return [(0.0, 0.0, 0.0)] * 16
    mode = (b0 & -b0).bit_length() - 1
    ns, pb, rb, isb, cb, ab, epb, spb, ib, ib2 = _BC7_MODES[mode]
    blk = int.from_bytes(data[at:at + 16], "little")
    pos = mode + 1

    part = (blk >> pos) & ((1 << pb) - 1) if pb else 0
    pos += pb
    rot = (blk >> pos) & ((1 << rb) - 1) if rb else 0
    pos += rb
    isel = (blk >> pos) & 1 if isb else 0
    pos += isb

    # Read raw endpoints: R, G, B, [A], each with 2*ns values.
    n_ep = 2 * ns
    cb_mask = (1 << cb) - 1
    raw_c = [[0] * n_ep for _ in range(4)]
    for ch in range(3):
        row = raw_c[ch]
        for k in range(n_ep):
            row[k] = (blk >> pos) & cb_mask
            pos += cb
    if ab:
        ab_mask = (1 << ab) - 1
        row = raw_c[3]
        for k in range(n_ep):
            row[k] = (blk >> pos) & ab_mask
            pos += ab

    # Read p-bits and expand endpoints to 8-bit UNORM via MSB replication.
    c_bits = cb
    a_bits = ab
    if epb:
        pbits = [(blk >> (pos + k)) & 1 for k in range(n_ep)]
        pos += n_ep
        c_bits += 1
        if ab:
            a_bits += 1
    elif spb:
        sp = [(blk >> (pos + s)) & 1 for s in range(ns)]
        pos += ns
        pbits = [sp[k >> 1] for k in range(n_ep)]
        c_bits += 1
    else:
        pbits = None

    c_sh = 8 - c_bits
    c_rep = 2 * c_bits - 8
    ep = [[255] * n_ep for _ in range(4)]
    for ch in range(3):
        src = raw_c[ch]
        dst = ep[ch]
        for k in range(n_ep):
            v = (src[k] << 1) | pbits[k] if pbits is not None else src[k]
            dst[k] = (v << c_sh) | (v >> c_rep)
    if ab:
        a_sh = 8 - a_bits
        a_rep = 2 * a_bits - 8
        src = raw_c[3]
        dst = ep[3]
        for k in range(n_ep):
            v = (src[k] << 1) | pbits[k] if (pbits is not None and epb) else src[k]
            dst[k] = (v << a_sh) | (v >> a_rep)

    tab = _bptc.load_bc7()
    if ns == 1:
        sm = None
        anchors = (0,)
    elif ns == 2:
        sm = tab["partition2"][part]
        anchors = (0, tab["anchor2"][part])
    else:
        sm = tab["partition3"][part]
        anchors = (0, tab["anchor3a"][part], tab["anchor3b"][part])

    ib_mask = (1 << ib) - 1
    ib_amask = (1 << (ib - 1)) - 1
    idx1 = [0] * 16
    for p in range(16):
        sub = 0 if sm is None else sm[p]
        if p == anchors[sub]:
            idx1[p] = (blk >> pos) & ib_amask
            pos += ib - 1
        else:
            idx1[p] = (blk >> pos) & ib_mask
            pos += ib

    if ib2:
        ib2_mask = (1 << ib2) - 1
        ib2_amask = (1 << (ib2 - 1)) - 1
        idx2 = [0] * 16
        for p in range(16):
            if p == 0:
                idx2[p] = (blk >> pos) & ib2_amask
                pos += ib2 - 1
            else:
                idx2[p] = (blk >> pos) & ib2_mask
                pos += ib2
        if isel:
            c_idx, c_wt = idx2, _BPTC_WEIGHTS[ib2]
            a_idx, a_wt = idx1, _BPTC_WEIGHTS[ib]
        else:
            c_idx, c_wt = idx1, _BPTC_WEIGHTS[ib]
            a_idx, a_wt = idx2, _BPTC_WEIGHTS[ib2]
    else:
        c_idx = a_idx = idx1
        c_wt = a_wt = _BPTC_WEIGHTS[ib]

    er, eg, eb_, ea = ep
    out = [None] * 16
    for p in range(16):
        s2 = 0 if sm is None else sm[p] * 2
        wc = c_wt[c_idx[p]]
        iwc = 64 - wc
        r = (iwc * er[s2] + wc * er[s2 + 1] + 32) >> 6
        g = (iwc * eg[s2] + wc * eg[s2 + 1] + 32) >> 6
        b = (iwc * eb_[s2] + wc * eb_[s2 + 1] + 32) >> 6
        if rot:
            wa = a_wt[a_idx[p]]
            a = ((64 - wa) * ea[s2] + wa * ea[s2 + 1] + 32) >> 6
            if rot == 1:
                r = a
            elif rot == 2:
                g = a
            else:
                b = a
        out[p] = (float(r), float(g), float(b))
    return out


def decode_pixels(fmt, data, blocks_w, blocks_h):
    """Return one (r, g, b) per pixel, or None for a format this tool cannot decode."""
    if fmt not in BC_FORMATS:
        n = blocks_w * blocks_h
        if fmt in (61, 65):  # R8_UNORM, A8_UNORM
            return [(float(b), float(b), float(b)) for b in data[:n]]
        if fmt in (28, 29):  # R8G8B8A8_UNORM / UNORM_SRGB
            return [(float(data[i]), float(data[i + 1]), float(data[i + 2]))
                    for i in range(0, n * 4, 4)]
        if fmt in (87, 91):  # B8G8R8A8_UNORM / UNORM_SRGB
            return [(float(data[i + 2]), float(data[i + 1]), float(data[i]))
                    for i in range(0, n * 4, 4)]
        if fmt == 49:  # R8G8_UNORM
            return [(float(data[i]), float(data[i + 1]), 0.0) for i in range(0, n * 2, 2)]
        if fmt == 56:  # R16_UNORM
            vals = struct.unpack_from(f"<{n}H", data, 0)
            return [(v / 257.0, v / 257.0, v / 257.0) for v in vals]
        if fmt == 54:  # R16_FLOAT
            lut = _f16_lut()
            vals = struct.unpack_from(f"<{n}H", data, 0)
            return [(lut[v], lut[v], lut[v]) for v in vals]
        if fmt == 11:  # R16G16B16A16_UNORM
            vals = struct.unpack_from(f"<{n * 4}H", data, 0)
            return [(vals[i] / 257.0, vals[i + 1] / 257.0, vals[i + 2] / 257.0)
                    for i in range(0, n * 4, 4)]
        if fmt == 10:  # R16G16B16A16_FLOAT
            lut = _f16_lut()
            vals = struct.unpack_from(f"<{n * 4}H", data, 0)
            return [(lut[vals[i]], lut[vals[i + 1]], lut[vals[i + 2]])
                    for i in range(0, n * 4, 4)]
        return None
    if fmt in (71, 72, 74, 75, 77, 78):
        # BC1 stands alone; BC2/BC3 carry the same color block after eight alpha bytes and use
        # four RGB entries regardless of endpoint order.
        stride, skip, four = (8, 0, False) if fmt in (71, 72) else (16, 8, True)
        return _decode_bc1_rows(data, blocks_w, blocks_h, stride, skip, four)
    if fmt == 80:
        return _decode_bc4_rows(data, blocks_w, blocks_h)
    decode = _BLOCK_DECODERS.get(fmt)
    if decode is None and blocks_w > 0 and blocks_h > 0:
        return None  # a BC format with no decoder here (an empty grid still gives [])
    px = []
    for by in range(blocks_h):
        r0, r1, r2, r3 = [], [], [], []
        for bx in range(blocks_w):
            blk = decode(data, (by * blocks_w + bx) * 16)
            r0 += blk[0:4]
            r1 += blk[4:8]
            r2 += blk[8:12]
            r3 += blk[12:16]
        px += r0
        px += r1
        px += r2
        px += r3
    return px


def _decode_bc5_block(data, at):
    a = decode_bc4_block(data, at)
    b = decode_bc4_block(data, at + 8)
    return [(p[0], q[0], 0.0) for p, q in zip(a, b)]


_BLOCK_DECODERS = {
    83: _decode_bc5_block,
    95: lambda data, at: decode_bc6h_block(data, at, signed=False),
    96: lambda data, at: decode_bc6h_block(data, at, signed=True),
    98: decode_bc7_block,
    99: decode_bc7_block,
}


def _decode_bc1_rows(data, blocks_w, blocks_h, stride, skip, four_color):
    """BC1 color blocks of a whole surface in pixel order, filling the output one block row
    at a time. Equal endpoint pairs share one palette (a reset cache bounds the memory)."""
    px, rows, head, cache = [], _BC1_ROWS, _BC1_HEAD, {}
    for by in range(blocks_h):
        at = by * blocks_w * stride + skip
        r0, r1, r2, r3 = [], [], [], []
        for _ in range(blocks_w):
            c0, c1, idx = head(data, at)
            at += stride
            key = c0 << 16 | c1
            pal = cache.get(key)
            if pal is None:
                if len(cache) >= _ROW_MAX_CACHE:
                    cache.clear()
                pal = cache[key] = _bc1_palette(c0, c1, four_color)
            r0 += rows[idx & 255](pal)
            r1 += rows[(idx >> 8) & 255](pal)
            r2 += rows[(idx >> 16) & 255](pal)
            r3 += rows[idx >> 24](pal)
        px += r0
        px += r1
        px += r2
        px += r3
    return px


def _decode_bc4_rows(data, blocks_w, blocks_h):
    """BC4 blocks of a whole surface in pixel order; same scheme as `_decode_bc1_rows`."""
    px, rows, cache = [], _bc4_rows(), {}
    for by in range(blocks_h):
        at = by * blocks_w * 8
        r0, r1, r2, r3 = [], [], [], []
        for _ in range(blocks_w):
            e0, e1 = data[at], data[at + 1]
            bits = int.from_bytes(data[at + 2:at + 8], "little")
            at += 8
            key = e0 << 8 | e1
            pal = cache.get(key)
            if pal is None:
                if len(cache) >= _ROW_MAX_CACHE:
                    cache.clear()
                pal = cache[key] = _bc4_palette(e0, e1)
            r0 += rows[bits & 4095](pal)
            r1 += rows[(bits >> 12) & 4095](pal)
            r2 += rows[(bits >> 24) & 4095](pal)
            r3 += rows[bits >> 36](pal)
        px += r0
        px += r1
        px += r2
        px += r3
    return px


def block_means(pixels, blocks_w, blocks_h, pixels_w=None, pixels_h=None, block=4):
    if block == 1:
        return list(pixels)
    pw = blocks_w * block if pixels_w is None else pixels_w
    ph = blocks_h * block if pixels_h is None else pixels_h
    width = blocks_w * block  # row pitch of `pixels`
    zero = (0.0, 0.0, 0.0)
    if block < 1:
        return [zero for _ in range(blocks_w * blocks_h)]
    rx, ry = min(pw, width), min(ph, blocks_h * block)  # visible columns / rows of the grid
    if rx > 0 and ry > 0 and len(pixels) < (ry - 1) * width + rx:
        raise IndexError("list index out of range")
    # Columns of each block that fall inside the picture; rows are clamped per block row.
    cols = [min(block, max(0, rx - bx * block)) for bx in range(blocks_w)]
    add, div, get0, get1, get2 = (operator.add, operator.truediv, operator.itemgetter(0),
                                  operator.itemgetter(1), operator.itemgetter(2))
    out = []
    for by in range(blocks_h):
        rows_in = min(block, max(0, ry - by * block))
        # Add the pixels in raster order, each block's sum starting from 0.0 as the scalar
        # loop did: one pass over a row's x-th column of every block at a time.
        acc = [[0.0] * blocks_w, [0.0] * blocks_w, [0.0] * blocks_w]
        for y in range(rows_in):
            start = (by * block + y) * width
            row = pixels[start:start + rx]
            for k, get in enumerate((get0, get1, get2)):
                chan = list(map(get, row))
                sums = acc[k]
                for x in range(min(block, rx)):
                    seq = chan[x::block]
                    if len(seq) == blocks_w:
                        sums = list(map(add, sums, seq))
                    else:
                        sums[:len(seq)] = map(add, sums[:len(seq)], seq)
                acc[k] = sums
        counts = [rows_in * c or 1 for c in cols]
        out.extend(zip(map(div, acc[0], counts), map(div, acc[1], counts),
                       map(div, acc[2], counts)))
    return out


def downsample2(blocks, blocks_w, blocks_h):
    """2x2 mean over a block grid. This only predicts the child mip when the parent block grid
    is exactly twice the child's; `check` uses `predict_child_means` instead."""
    ow, oh = max(1, blocks_w // 2), max(1, blocks_h // 2)
    out = []
    for y in range(oh):
        for x in range(ow):
            acc = [0.0, 0.0, 0.0]
            n = 0
            for dy in range(2):
                for dx in range(2):
                    bx = min(blocks_w - 1, 2 * x + dx)
                    by = min(blocks_h - 1, 2 * y + dy)
                    q = blocks[by * blocks_w + bx]
                    for k in range(3):
                        acc[k] += q[k]
                    n += 1
            out.append(tuple(a / n for a in acc))
    return out


class MipGeometryError(ValueError):
    """Two adjacent mips do not form a 2x2 downsample, or a block list does not fit its mip."""


def _hdr_tonemap(v):
    """Signed Reinhard tone-map onto (-255, 255) for HDR mip-chain comparison."""
    if not math.isfinite(v):
        return 0.0 if math.isnan(v) else (255.0 if v > 0 else -255.0)
    return 255.0 * v / (1.0 + abs(v))


def mean_abs_error(a, b, *, hdr=False):
    if len(a) != len(b):
        raise MipGeometryError(f"cannot compare {len(a)} predicted blocks with {len(b)} mip blocks")
    if not a:
        return None
    if hdr:
        return sum(
            abs(_hdr_tonemap(x) - _hdr_tonemap(y))
            for pa, pb in zip(a, b) for x, y in zip(pa, pb)
        ) / (3 * len(a))
    return sum(abs(x - y) for pa, pb in zip(a, b) for x, y in zip(pa, pb)) / (3 * len(a))


def _visible_grid(mip, block=4):
    """Return the (blocks_w, blocks_h) that hold visible texels, checking the element grid."""
    pw, ph, bw, bh = mip["pixels_w"], mip["pixels_h"], mip["blocks_w"], mip["blocks_h"]
    vw, vh = (pw + block - 1) // block, (ph + block - 1) // block
    if pw < 1 or ph < 1 or bw < vw or bh < vh:
        raise MipGeometryError(
            f"mip {mip.get('index')} geometry: {bw}x{bh} blocks cannot hold {pw}x{ph} pixels")
    return vw, vh


def _visible_rows(layout, mip, data, block=4):
    """Crop block-linear `data` from `detile` to the mip's visible elements, tightly packed.

    Storage grids can be wider or taller than the visible grid (17x13 BC1 stores its 8x6
    child as 3x2 blocks; 17x13 uncompressed stores its 8x6 child as 9x7 elements); standard
    DDS readers expect only the visible rows. Aligned grids are returned unchanged.
    """
    vw, vh = _visible_grid(mip, block)
    bw, eb = mip["blocks_w"], layout["element_bytes"]
    if (vw, vh) == (bw, mip["blocks_h"]):
        return data
    return b"".join(data[y * bw * eb:(y * bw + vw) * eb] for y in range(vh))


def visible_block_means(means, mip, block=4):
    """Select the `block_means` entries whose elements hold visible texels, row-major."""
    vw, vh = _visible_grid(mip, block)
    bw = mip["blocks_w"]
    if len(means) != bw * mip["blocks_h"]:
        raise MipGeometryError(
            f"mip {mip.get('index')} geometry: {len(means)} block means for "
            f"{bw}x{mip['blocks_h']} blocks")
    return [means[by * bw + bx] for by in range(vh) for bx in range(vw)]


def predict_child_means(pixels, parent, child, block=4):
    """Predict the visible element means of `child` as a 2x2 box downsample of `parent` pixels.

    Child texel (x, y) is the mean of parent texels (2x + dx, 2y + dy), dx, dy in {0, 1},
    clamped to the last visible parent row/column. Each child element averages only its
    visible texels, so padding in either mip is never read. Returns one (r, g, b) per element
    of the child's visible grid, ordered like `visible_block_means`.
    """
    pw, ph, cw, ch = parent["pixels_w"], parent["pixels_h"], child["pixels_w"], child["pixels_h"]
    _visible_grid(parent, block)
    vw, vh = _visible_grid(child, block)
    if (cw, ch) != (max(1, pw // 2), max(1, ph // 2)):
        raise MipGeometryError(
            f"mip {child.get('index')} geometry: {cw}x{ch} is not the 2x2 downsample of {pw}x{ph}")
    stride = parent["blocks_w"] * block
    if len(pixels) != stride * parent["blocks_h"] * block:
        raise MipGeometryError(
            f"mip {parent.get('index')} geometry: {len(pixels)} pixels for "
            f"{parent['blocks_w']}x{parent['blocks_h']} blocks")
    if block == 1:
        out = []
        for y in range(ch):
            r0 = min(ph - 1, 2 * y) * stride
            r1 = min(ph - 1, 2 * y + 1) * stride
            for x in range(cw):
                x0 = min(pw - 1, 2 * x)
                x1 = min(pw - 1, 2 * x + 1)
                p00 = pixels[r0 + x0]
                p10 = pixels[r0 + x1]
                p01 = pixels[r1 + x0]
                p11 = pixels[r1 + x1]
                out.append((
                    0.25 * (p00[0] + p10[0] + p01[0] + p11[0]),
                    0.25 * (p00[1] + p10[1] + p01[1] + p11[1]),
                    0.25 * (p00[2] + p10[2] + p01[2] + p11[2]),
                ))
        return out
    out = []
    for by in range(vh):
        for bx in range(vw):
            acc = [0.0, 0.0, 0.0]
            n = 0
            for y in range(by * block, min(by * block + block, ch)):
                for x in range(bx * block, min(bx * block + block, cw)):
                    for dy in range(2):
                        row = min(ph - 1, 2 * y + dy) * stride
                        for dx in range(2):
                            q = pixels[row + min(pw - 1, 2 * x + dx)]
                            for k in range(3):
                                acc[k] += q[k]
                            n += 1
            out.append(tuple(a / n for a in acc))
    return out


# --- RDS5 / ETPF container ------------------------------------------------

class Pool:
    """Read-only RDS5 texture pool, mirroring TtpfReader's catalog checks."""

    def __init__(self, path):
        self._cache_token = object()
        # A pool that fails a check used to keep its file open and mapped (locked on Windows).
        self._fh = self.data = None
        try:
            self._open(path)
        except Exception:
            self.close()
            raise

    def _open(self, path):
        self.path = path
        self._fh = open(path, "rb")
        try:
            self.data = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        d = self.data
        if len(d) < 256 or d[:4] != b"RDS5":
            raise ValueError("not an RDS5 container or truncated header")
        count = u32(d, 4)
        size_word = u64(d, 8)
        self.virtual_size = size_word & 0x00FFFFFFFFFFFFFF
        if (count < 3 or count > 1_000_002 or (size_word >> 56) != 22 or
                self.virtual_size < 4096 or u32(d, 16) not in
                ((count + 31) // 32, (count + 31) // 32 + 1)):
            raise ValueError("unsupported RDS5 index layout")
        self.chunk_count = count - 2
        table_end = 24 + (count - 1) * 8
        self.etpf = align(table_end, 256)
        if table_end > len(d) - 16:
            raise ValueError("RDS5 index exceeds its source")
        table = d[24:table_end]
        if ((u64(table, len(table) - 8) & 0x3FFFFFF) * 256 != len(d) or
                ((u64(table, len(table) - 8) >> 30) & 0x3FFFFF) * 4096 != self.virtual_size):
            raise ValueError("RDS5 sentinel does not match its lengths")
        self.chunks = []
        for i in range(self.chunk_count):
            v, nxt = u64(table, i * 8), u64(table, (i + 1) * 8)
            start, end = (v & 0x3FFFFFF) * 256, (nxt & 0x3FFFFFF) * 256
            virt, vend = ((v >> 30) & 0x3FFFFF) * 4096, ((nxt >> 30) & 0x3FFFFF) * 4096
            pad = v >> 52
            codec = (v >> 26) & 15
            packed, size = end - start - pad, vend - virt
            if (start < self.etpf + 16 or end > len(d) or packed < 1 or packed > MAX_OUTPUT or
                    size < 1 or size > MAX_OUTPUT or vend > self.virtual_size or
                    (codec == 0 and packed != size)):
                raise ValueError(f"RDS5 chunk {i} range is invalid")
            self.chunks.append((virt, size, start, packed, codec, pad))
        eh = d[self.etpf:self.etpf + 16]
        if eh[:4] != b"ETPF" or u32(eh, 4) != ETPF_VERSION:
            raise ValueError("unsupported embedded ETPF version")
        self.textures = u32(eh, 8)
        area = u32(eh, 12)
        metadata = 16 + area + self.textures * 32
        padded = align(metadata, 4096)
        alignment = u32(d, 20)
        source_padding = (alignment >> 28) * 256
        if (self.textures < 1 or self.textures > MAX_TEXTURES or area < self.textures * 16 or
                (alignment & 0x0FFFFFFF) != 0 or padded != self.chunks[0][0] or
                self.etpf + padded + source_padding != self.chunks[0][2]):
            raise ValueError("ETPF tables do not match the data boundary")
        self.area = area
        self.records = d[self.etpf + 16:self.etpf + 16 + area]
        self.ids = d[self.etpf + 16 + area:self.etpf + 16 + area + self.textures * 32]
        self._starts = [c[0] for c in self.chunks]

    def close(self):
        if self.data is not None:
            self.data.close()
        if self._fh is not None:
            self._fh.close()

    def asset_key(self, index):
        """The entry's 64-bit native asset key. The key array follows the 16-byte records and
        is longer than the record area has room for in most pools, so it continues into the
        32-byte ID region; TtpfReader reads it across that boundary, and so does this."""
        at = self.textures * 16 + index * 8
        if at + 8 <= len(self.records):
            return u64(self.records, at)
        if at >= len(self.records):
            return u64(self.ids, at - len(self.records))
        raise ValueError("ETPF asset key straddles the records/ids boundary")

    def descriptor(self, index):
        if not 0 <= index < self.textures:
            raise IndexError(index)
        d = parse_etpf_texture_record(self.records[index * 16:index * 16 + 16])
        d["index"] = index
        d["asset_key_at"] = self.asset_key(index)
        return d

    def chunks_for(self, start, size):
        import bisect
        end = start + size
        i = max(0, bisect.bisect_right(self._starts, start) - 1)
        out = []
        covered = start
        while i < len(self.chunks) and self.chunks[i][0] < end:
            virt, csize, src, packed, codec, pad = self.chunks[i]
            if virt > covered:
                raise ValueError("texture crosses an unmapped virtual range")
            out.append(self.chunks[i])
            covered = min(end, virt + csize)
            i += 1
        if covered != end:
            raise ValueError("texture range is not completely mapped")
        return out

    def codecs(self, start, size):
        return {c[4] for c in self.chunks_for(start, size)}

    def readable(self, start, size, *, allow_bcpack=False):
        allowed = ({0, 8} | bcpack.SUPPORTED_REBUILD_CODECS) if allow_bcpack else {0, 8}
        return self.codecs(start, size) <= allowed

    def read_range(self, start, size, cache=None, *, bcpack_mode="strict", bcpack_history=None):
        if bcpack_mode not in ("strict", "provisional", "off"):
            raise ValueError(f"invalid bcpack_mode: {bcpack_mode!r}")
        cache = {} if cache is None else cache
        out = bytearray(size)
        hist_key = None if bcpack_history is None else (
            id(bcpack_history) if isinstance(bcpack_history, dict) else os.fspath(bcpack_history)
        )
        for virt, csize, src, packed, codec, pad in self.chunks_for(start, size):
            key = (self._cache_token, src, packed, codec, bcpack_mode if codec not in (0, 8) else "raw",
                   hist_key if codec not in (0, 8) else None)
            if key not in cache:
                raw = self.data[src:src + packed]
                if codec == 0:
                    decoded = bytes(raw)
                elif codec == 8:
                    inflater = zlib.decompressobj()
                    decoded = inflater.decompress(raw, csize + 1)
                    if len(decoded) > csize or not inflater.eof:
                        raise ValueError("zlib chunk is truncated or exceeds its declared size")
                elif bcpack_mode != "off" and codec in bcpack.SUPPORTED_REBUILD_CODECS:
                    decoded = bcpack.decode(
                        raw, codec, csize,
                        allow_provisional=(bcpack_mode == "provisional"),
                        history=bcpack_history,
                    )
                else:
                    raise BcpackRequired(
                        f"range touches a BCPack chunk (codec {codec})"
                    )
                if len(decoded) != csize:
                    raise ValueError("decoded chunk length disagrees with its range")
                cache[key] = decoded
            lo, hi = max(start, virt), min(start + size, virt + csize)
            out[lo - start:hi - start] = cache[key][lo - virt:hi - virt]
        return bytes(out)

    def span(self, index, base):
        if _is_unmapped_base(base) or base >= self.virtual_size:
            return None
        for j in range(index + 1, self.textures):
            cand = (u32(self.records, j * 16 + 8) & 0x1FFFFF) * 4096
            if _is_unmapped_base(cand):
                continue
            if cand < base:
                return None
            if cand > base:
                return cand - base
        return self.virtual_size - base


def _is_unmapped_base(base):
    """True for ETPF unmapped-allocation sentinel bases (0 or 0x1FFFFF pages)."""
    return base == 0 or (base >> 12) == 0x1FFFFF


def compact_row(desc):
    """The reader's compact-profile gate (mode families 4/6/36 and aux mode 2)."""
    return desc["mode"] in (4, 6, 36) or (desc["mode"] == 2 and (desc["meta"] or desc["extra"]))


def validated_compact(desc, eb):
    return (desc["mode"], desc["width"], desc["height"], desc["mips"], desc["non_tail"],
            eb, desc["meta"], desc["extra"]) in VALIDATED_COMPACT


def validated_aux(desc):
    a = VALIDATED_AUX
    return (desc["mips"] == a["mips"] and desc["non_tail"] == a["non_tail"] and
            desc["meta"] == a["meta"] and desc["extra"] == a["extra"] and
            desc["width"] == a["width"] and
            ((desc["height"] == a["height_a"] and desc["format"] == a["fmt_a"]) or
             (desc["height"] == a["height_b"] and desc["format"] in BC_FORMATS and
              ELEMENT_BYTES.get(desc["format"]) == a["bytes_b"])))


def validate_chunk_metadata(pool, desc, span):
    """Verify ``(meta, extra)`` streaming chunk sizes and alignment padding.

    Returns ``(ok, pages)`` where ``pages`` is the decoded tuple of 4 KiB chunk
    page counts (empty when ``meta == 0 and extra == 0``).
    """
    meta, extra = desc["meta"], desc["extra"]
    if meta == 0 and extra == 0:
        return True, ()
    pages = decode_etpf_chunk_pages(meta, extra)
    if not pages:
        return False, None
    try:
        chunks = pool.chunks_for(desc["base"], span)
    except ValueError:
        return False, None
    if len(chunks) != len(pages):
        return False, None
    if chunks[0][0] != desc["base"] or sum(c[1] for c in chunks) != span:
        return False, None
    for chunk, expected_pages in zip(chunks, pages):
        if chunk[1] != expected_pages * 4096:
            return False, None
    align_group, pad_pages = etpf_streaming_padding(extra)
    if desc["dimension"] == 2 or desc["non_tail"] == 1:
        if align_group != 0 or pad_pages != 0:
            return False, None
    elif desc["dimension"] == 1 and desc["non_tail"] == 3:
        if align_group == 1:
            if pages[0] + pad_pages != 16:
                return False, None
        elif align_group == 2:
            if len(pages) < 2 or pages[0] + pages[1] + pad_pages != 16:
                return False, None
        elif align_group == 3:
            if len(pages) < 2 or pages[0] + pages[1] + pad_pages != 8:
                return False, None
        else:
            return False, None
    else:
        if align_group != 0 or pad_pages != 0:
            return False, None
    return True, pages


def _mip_range_codecs(pool, base, offset, region, slices=1, slice_stride=0):
    codecs = set()
    for s in range(slices):
        codecs |= pool.codecs(base + offset + s * slice_stride, region)
    return codecs


def _decode_volume_voxels(fmt, data, width, height, depth):
    """Decode float16 volume bytes into a flat list of per-voxel float tuples."""
    channels = VOLUME_FLOAT_FORMATS.get(fmt)
    if channels is None:
        return None
    n = width * height * depth * channels
    vals = struct.unpack_from(f"<{n}e", data, 0)
    if channels == 1:
        return [(v,) for v in vals]
    return [vals[i:i + channels] for i in range(0, n, channels)]


def _volume_z_diff(voxels, width, height, depth):
    """Mean absolute difference between adjacent Z slices (finite voxels only)."""
    plane = width * height
    total = 0.0
    count = 0
    for z in range(depth - 1):
        a0 = z * plane
        b0 = (z + 1) * plane
        for p in range(plane):
            va = voxels[a0 + p]
            vb = voxels[b0 + p]
            for ca, cb in zip(va, vb):
                if math.isfinite(ca) and math.isfinite(cb):
                    total += abs(ca - cb)
                    count += 1
    return (total / count) if count else None


def analyze(pool, index, *, allow_bcpack=True):
    """Descriptor + layout + per-mip codec availability for one texture."""
    d = pool.descriptor(index)
    out = dict(descriptor=d, tile_mode=tile_mode(d["mode"]), tile=tile_bytes(d["mode"]))
    if _is_unmapped_base(d["base"]):
        out["blocked"] = "no mapped allocation"
        return out
    span = pool.span(index, d["base"])
    out["allocation_span"] = span
    allowed = ({0, 8} | bcpack.SUPPORTED_REBUILD_CODECS) if allow_bcpack else {0, 8}

    if d["dimension"] == 1 and d["depth"] == 1:
        out["shape"] = "2d"
        try:
            layout = surface_layout(d["width"], d["height"], d["format"], d["mips"],
                                    out["tile"])
        except LayoutError as e:
            out["blocked"] = str(e)
            return out
        out["layout"] = layout
        out["span_matches"] = span == layout["size"]
        if not out["span_matches"]:
            out["blocked"] = "layout does not match the allocation span"
            return out
        avail = []
        mip_codecs = []
        for mip in layout["mips"]:
            try:
                cs = pool.codecs(d["base"] + mip["offset"], mip["region"])
                mip_codecs.append(sorted(cs))
                avail.append(cs <= allowed)
            except ValueError:
                avail = []
                mip_codecs = []
                break
        out["available"] = avail
        out["mip_codecs"] = mip_codecs
        if avail:
            limit = d["non_tail"] if compact_row(d) else d["mips"]
            limit = max(0, min(limit, len(avail)))
            base_mip = next((i for i in range(limit) if avail[i]), None)
            exported = 0
            if base_mip is not None:
                while base_mip + exported < limit and avail[base_mip + exported]:
                    exported += 1
            out["reader_limit"] = limit
            out["reader_base_mip"] = base_mip
            out["reader_exported"] = exported
            out["full_chain"] = bool(avail) and all(avail)
            out["recoverable_mips"] = sum(1 for a in avail[exported:] if a) if base_mip == 0 else 0
        return out

    if d["dimension"] == 1 and d["depth"] > 1:
        out["shape"] = "2d_array"
        if d["mode"] != 1:
            out["blocked"] = f"unsupported 2D array mode {d['mode']}"
            return out
        try:
            eb = element_bytes(d["format"])
            blk = 4 if d["format"] in BC_FORMATS else 1
            layout = array_surface_layout(d["width"], d["height"], d["depth"], d["mips"],
                                          eb, blk, out["tile"])
        except (LayoutError, ValueError) as e:
            out["blocked"] = str(e)
            return out
        out["layout"] = layout
        out["span_matches"] = span == layout["size"]
        if not out["span_matches"]:
            out["blocked"] = "array layout does not match the allocation span"
            return out
        avail = []
        mip_codecs = []
        for mip in layout["mips"]:
            try:
                cs = _mip_range_codecs(pool, d["base"], mip["offset"], mip["region"],
                                       d["depth"], mip["slice_stride"])
                mip_codecs.append(sorted(cs))
                avail.append(cs <= allowed)
            except ValueError:
                avail = []
                mip_codecs = []
                break
        out["available"] = avail
        out["mip_codecs"] = mip_codecs
        if avail:
            out["reader_limit"] = 0
            out["reader_base_mip"] = None
            out["reader_exported"] = 0
            out["full_chain"] = all(avail)
        return out

    if d["dimension"] == 2:
        out["shape"] = "cubemap"
        if d["depth"] != 6:
            out["blocked"] = f"unsupported cubemap depth {d['depth']}"
            return out
        if d["non_tail"] > 3 or d["mode"] != 18 + 2 * d["non_tail"]:
            out["blocked"] = f"unsupported cubemap mode {d['mode']} (non_tail={d['non_tail']})"
            return out
        ok, pages = validate_chunk_metadata(pool, d, span)
        if not ok:
            out["blocked"] = "cubemap streaming chunk metadata mismatch"
            return out
        if d["non_tail"] > 0 and len(pages) != d["non_tail"] + 1:
            out["blocked"] = "cubemap streaming chunk metadata mismatch"
            return out
        if d["non_tail"] == 0 and len(pages) not in (0, 1):
            out["blocked"] = "cubemap chunk metadata mismatch"
            return out
        try:
            eb = element_bytes(d["format"])
            blk = 4 if d["format"] in BC_FORMATS else 1
            layout = cubemap_surface_layout(d["width"], d["height"], d["mips"], eb, blk,
                                            d["non_tail"], pages if pages else None)
        except (LayoutError, ValueError) as e:
            out["blocked"] = str(e)
            return out
        out["layout"] = layout
        out["span_matches"] = span == layout["size"]
        if not out["span_matches"]:
            out["blocked"] = "cubemap layout does not match the allocation span"
            return out
        avail = []
        mip_codecs = []
        face_base = d["base"] + layout["data_offset"]
        for mip in layout["mips"]:
            try:
                cs = _mip_range_codecs(pool, face_base, mip["offset"], mip["region"],
                                       6, layout["face_size"])
                mip_codecs.append(sorted(cs))
                avail.append(cs <= allowed)
            except ValueError:
                avail = []
                mip_codecs = []
                break
        out["available"] = avail
        out["mip_codecs"] = mip_codecs
        if avail:
            out["reader_limit"] = 0
            out["reader_base_mip"] = None
            out["reader_exported"] = 0
            out["full_chain"] = all(avail)
        return out

    if d["dimension"] == 3:
        out["shape"] = "volume"
        if d["mode"] != 2 or d["mips"] != 1 or d["format"] in BC_FORMATS:
            out["blocked"] = (
                f"unsupported 3D volume mode={d['mode']} mips={d['mips']} format={d['format']}")
            return out
        try:
            eb = element_bytes(d["format"])
            layout = volume_surface_layout(d["width"], d["height"], d["depth"], d["mips"], eb)
        except (LayoutError, ValueError) as e:
            out["blocked"] = str(e)
            return out
        out["layout"] = layout
        out["span_matches"] = span == layout["size"]
        if not out["span_matches"]:
            out["blocked"] = "volume layout does not match the allocation span"
            return out
        try:
            cs = pool.codecs(d["base"], layout["size"])
            avail = [cs <= allowed]
            mip_codecs = [sorted(cs)]
        except ValueError:
            avail = []
            mip_codecs = []
        out["available"] = avail
        out["mip_codecs"] = mip_codecs
        if avail:
            out["reader_limit"] = 0
            out["reader_base_mip"] = None
            out["reader_exported"] = 0
            out["full_chain"] = avail[0]
        return out

    out["shape"] = "unknown"
    out["blocked"] = f"unsupported descriptor dimension {d['dimension']}"
    return out


# --- commands --------------------------------------------------------------

def same_output_path(first, second):
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def find_pools(arg):
    """Pools under a game root (data/ttdb and pack/*/data/ttdb), or, for any other folder such as data/ttdb
    itself or one pack, every .ttpf below it. A file is taken as a pool."""
    if os.path.isfile(arg):
        return [arg]
    root = glob.escape(arg)   # a folder like "Game [v1]" is a path, not a glob pattern
    pools = sorted(glob.glob(os.path.join(root, "data", "ttdb", "**", "*.ttpf"), recursive=True))
    pools += sorted(glob.glob(os.path.join(root, "pack", "*", "data", "ttdb", "**", "*.ttpf"),
                                recursive=True))
    pools = [path for path in pools if os.path.isfile(path)]
    if not pools:
        pools = sorted(path for path in glob.iglob(os.path.join(root, "**", "*.ttpf"), recursive=True)
                       if os.path.isfile(path))
    return pools


def scan_pools(paths, as_json=None, include_tuples=False):
    total = dict(pools=0, entries=0, arrays=0, arrays_2d=0, arrays_2d_full=0,
                 arrays_2d_partial=0, arrays_2d_bcpack=0, cubemaps=0, cubemaps_full=0,
                 cubemaps_partial=0, cubemaps_bcpack=0, volumes=0, volumes_full=0,
                 volumes_bcpack=0, multi_slice_layout_mismatch=0,
                 chunk_metadata_ok=0, chunk_metadata_fail=0,
                 unsupported_format=0, format11=0,
                 layout_mismatch=0, reader_full=0, reader_partial=0,
                 reader_skip_bcpack=0, whitelist_gated=0, gated_layout_ok=0,
                 gated_full_chain=0, gated_full_mips=0, gated_full_bytes=0,
                 gated_tail_only=0, gated_partial=0, gated_no_contiguous=0,
                 gated_no_layout=0, gated_unknown_format=0,
                 gated_other_layout_error=0, gated_span_mismatch=0,
                 gated_unreadable_range=0, gated_tuple_distinct=0, gated_by_mode={},
                 compact_partial_rows=0, compact_missing_mips=0,
                 compact_decodable_mips=0, compact_decodable_bytes=0,
                 compact_missing_standalone=0, compact_missing_tail=0,
                 compact_rows_measured=0, compact_non_tail_before_tail=0,
                 compact_standalone_beyond_limit=0,
                 compact_standalone_decodable_beyond_limit=0,
                 compact_full_rows=0, after_bcpack_full=0,
                 after_bcpack_partial=0, tile_modes={})
    gated_tuples = {}
    bcpack_allowed = {0, 8} | bcpack.SUPPORTED_REBUILD_CODECS
    for path in paths:
        try:
            pool = Pool(path)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{path}: {exc}") from None
        try:
            total["pools"] += 1
            for i in range(pool.textures):
                d = pool.descriptor(i)
                total["entries"] += 1
                tm = str(tile_mode(d["mode"]))
                total["tile_modes"][tm] = total["tile_modes"].get(tm, 0) + 1
                span = pool.span(i, d["base"])
                if span is not None:
                    meta_ok, _ = validate_chunk_metadata(pool, d, span)
                    if meta_ok:
                        total["chunk_metadata_ok"] += 1
                    else:
                        total["chunk_metadata_fail"] += 1
                if d["dimension"] != 1 or d["depth"] != 1:
                    total["arrays"] += 1
                    a = analyze(pool, i, allow_bcpack=False)
                    shape = a.get("shape")
                    prefix = {"2d_array": "arrays_2d", "cubemap": "cubemaps",
                              "volume": "volumes"}.get(shape)
                    if prefix:
                        total[prefix] += 1
                    if "available" not in a or not a["available"]:
                        total["multi_slice_layout_mismatch"] += 1
                        continue
                    avail = a["available"]
                    mip_codecs = a.get("mip_codecs") or []
                    avail_bc = [set(cs) <= bcpack_allowed for cs in mip_codecs]
                    if prefix:
                        if all(avail):
                            total[f"{prefix}_full"] += 1
                        elif any(avail):
                            if f"{prefix}_partial" in total:
                                total[f"{prefix}_partial"] += 1
                        else:
                            total[f"{prefix}_bcpack"] += 1
                    if avail_bc and all(avail_bc):
                        total["after_bcpack_full"] += 1
                    elif any(avail_bc):
                        total["after_bcpack_partial"] += 1
                    continue
                # The reader's gate: modes 4/6/36 and aux mode 2 need a validated tuple;
                # every other non-0/2 mode is rejected by the same branch.
                aux = d["mode"] == 2 and (d["meta"] or d["extra"])
                if d["mode"] in (0, 2):
                    gated = d["mode"] == 2 and aux and not validated_aux(d)
                else:
                    eb = ELEMENT_BYTES.get(d["format"])
                    gated = not (d["mode"] in (4, 6, 36) and eb is not None and
                                 validated_compact(d, eb))
                try:
                    layout = surface_layout(d["width"], d["height"], d["format"],
                                            d["mips"], tile_bytes(d["mode"]))
                except LayoutError:
                    total["unsupported_format"] += 1
                    if d["format"] == 11:
                        total["format11"] += 1
                    if gated:
                        total["whitelist_gated"] += 1
                        total["gated_no_layout"] += 1
                        if d["format"] not in ELEMENT_BYTES:
                            total["gated_unknown_format"] += 1
                        else:
                            total["gated_other_layout_error"] += 1
                    continue
                if d["base"] == 0 or span != layout["size"]:
                    total["layout_mismatch"] += 1
                    if gated:
                        total["whitelist_gated"] += 1
                        total["gated_span_mismatch"] += 1
                    continue
                avail = []
                avail_bc = []
                for mip in layout["mips"]:
                    try:
                        cs = pool.codecs(d["base"] + mip["offset"], mip["region"])
                    except ValueError:
                        avail = []
                        avail_bc = []
                        break
                    avail.append(cs <= {0, 8})
                    avail_bc.append(cs <= bcpack_allowed)
                if not avail:
                    total["layout_mismatch"] += 1
                    if gated:
                        total["whitelist_gated"] += 1
                        total["gated_unreadable_range"] += 1
                    continue
                if all(avail_bc):
                    total["after_bcpack_full"] += 1
                else:
                    ft = layout["first_tail"]
                    usable_tail_bc = (ft < d["mips"] and all(avail_bc[ft:]))
                    lim_bc = (d["mips"] if (not compact_row(d) or usable_tail_bc)
                              else min(ft, len(avail_bc)))
                    bm_bc = next((k for k in range(lim_bc) if avail_bc[k]), None)
                    exp_bc = 0
                    if bm_bc is not None:
                        while bm_bc + exp_bc < lim_bc and avail_bc[bm_bc + exp_bc]:
                            exp_bc += 1
                    if exp_bc:
                        total["after_bcpack_partial"] += 1
                if gated:
                    total["whitelist_gated"] += 1
                    total["gated_layout_ok"] += 1
                    key = str(d["mode"])
                    total["gated_by_mode"][key] = total["gated_by_mode"].get(key, 0) + 1
                    tuple_key = (d["mode"], d["width"], d["height"], d["mips"],
                                 d["non_tail"], layout["element_bytes"])
                    gated_tuples[tuple_key] = gated_tuples.get(tuple_key, 0) + 1
                    first_tail = layout["first_tail"]
                    if all(avail):
                        total["gated_full_chain"] += 1
                        total["gated_full_mips"] += d["mips"]
                        total["gated_full_bytes"] += sum(
                            mip["blocks_w"] * mip["blocks_h"] * layout["element_bytes"]
                            for mip in layout["mips"])
                    else:
                        # After the layout fix the export limit is the usable
                        # tail tile, or the computed tail boundary when it is
                        # not. Count what would still export and what would not.
                        usable_tail = (first_tail < d["mips"] and
                                       all(avail[first_tail:]))
                        if usable_tail:
                            total["gated_tail_only"] += 1
                            limit = d["mips"]
                        else:
                            limit = min(first_tail, len(avail))
                        base_mip = next((k for k in range(limit) if avail[k]), None)
                        exported = 0
                        if base_mip is not None:
                            while base_mip + exported < limit and avail[base_mip + exported]:
                                exported += 1
                        if exported:
                            total["gated_partial"] += 1
                        else:
                            total["gated_no_contiguous"] += 1
                    continue
                is_compact = compact_row(d)
                limit = min(d["non_tail"], len(avail)) if is_compact else len(avail)
                base_mip = next((k for k in range(limit) if avail[k]), None)
                exported = 0
                if base_mip is not None:
                    while base_mip + exported < limit and avail[base_mip + exported]:
                        exported += 1
                if exported == 0:
                    total["reader_skip_bcpack"] += 1
                elif exported < d["mips"]:
                    total["reader_partial"] += 1
                    if is_compact and base_mip == 0 and exported == limit:
                        total["compact_partial_rows"] += 1
                        missing = d["mips"] - exported
                        total["compact_missing_mips"] += missing
                        dec = 0
                        for k in range(exported, d["mips"]):
                            if avail[k]:
                                dec += 1
                                if k < layout["first_tail"]:
                                    total["compact_missing_standalone"] += 1
                                else:
                                    total["compact_missing_tail"] += 1
                        total["compact_decodable_mips"] += dec
                        total["compact_decodable_bytes"] += sum(
                            layout["mips"][k]["blocks_w"] * layout["mips"][k]["blocks_h"] *
                            layout["element_bytes"] for k in range(exported, d["mips"])
                            if avail[k])
                        if dec == missing:
                            total["compact_full_rows"] += 1
                else:
                    total["reader_full"] += 1
                if d["mode"] in (4, 6, 36):
                    total["compact_rows_measured"] += 1
                    if d["non_tail"] < layout["first_tail"]:
                        total["compact_non_tail_before_tail"] += 1
                        total["compact_standalone_beyond_limit"] += (
                            layout["first_tail"] - d["non_tail"])
                        total["compact_standalone_decodable_beyond_limit"] += sum(
                            1 for k in range(d["non_tail"], layout["first_tail"]) if avail[k])
        finally:
            pool.close()
    total["gated_accounted"] = (total["gated_layout_ok"] + total["gated_no_layout"] +
                                total["gated_span_mismatch"] + total["gated_unreadable_range"])
    total["gated_tuple_distinct"] = len(gated_tuples)
    # Projected split once the reader uses the computed 2D + multi-slice layouts:
    # existing full exports plus compact partials, gated 2D records, and resolved
    # arrays/cubemaps/volumes that complete; partials lose completed 2D rows and
    # gain newly partial gated 2D + array rows.
    total["after_full"] = (total["reader_full"] + total["compact_full_rows"] +
                           total["gated_full_chain"] + total["arrays_2d_full"] +
                           total["cubemaps_full"] + total["volumes_full"])
    total["after_partial"] = (total["reader_partial"] - total["compact_full_rows"] +
                              total["gated_partial"] + total["arrays_2d_partial"] +
                              total["cubemaps_partial"])
    total["after_skip"] = total["entries"] - total["after_full"] - total["after_partial"]
    total["after_bcpack_skip"] = (total["entries"] - total["after_bcpack_full"] -
                                  total["after_bcpack_partial"])
    total["bcpack_unlocked_full"] = total["after_bcpack_full"] - total["after_full"]
    if include_tuples:
        total["gated_tuples"] = [
            {"tuple": list(key), "count": count}
            for key, count in sorted(gated_tuples.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
    report = dict(total=total)
    # Printed first: a report path that can't be written used to end the scan in a traceback with nothing shown.
    print(json.dumps(report, indent=1))
    if as_json:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(as_json)), exist_ok=True)
            with open(as_json, "w", encoding="utf-8", newline="\n") as f:
                json.dump(report, f, indent=1)
        except OSError as exc:
            raise ValueError(f"cannot write {as_json}: {exc.strerror or exc}") from None


def describe(pool, index):
    a = analyze(pool, index)
    d = a["descriptor"]
    out = dict(index=index, shape=a.get("shape"), width=d["width"], height=d["height"],
               depth=d["depth"], dimension=d["dimension"], format=d["format"],
               mips=d["mips"], non_tail=d["non_tail"], mode=d["mode"],
               tile_mode=a["tile_mode"], tile=a["tile"], base=d["base"],
               meta=d["meta"], extra=d["extra"])
    if "layout" in a:
        lay = a["layout"]
        out["allocation_size"] = lay["size"]
        out["allocation_span"] = a.get("allocation_span")
        out["span_matches"] = a.get("span_matches")
        out["first_tail_mip"] = lay["first_tail"]
        mip_codecs = a.get("mip_codecs") or []
        if a.get("shape") == "cubemap":
            out["data_offset"] = lay["data_offset"]
            out["face_size"] = lay["face_size"]
            out["snapshots"] = lay["snapshots"]
        elif a.get("shape") == "volume":
            out["block_d"] = lay["block_d"]
            out["padded_d"] = lay["padded_d"]
            out["z_tile_stride"] = lay["z_tile_stride"]
        out["mips_detail"] = [
            dict(index=m["index"], blocks=[m["blocks_w"], m["blocks_h"]], offset=m["offset"],
                 region=m["region"], tail=m["in_tail"], tile_origin=[m["tile_x"], m["tile_y"]],
                 **({"slice_stride": m["slice_stride"]} if "slice_stride" in m else {}),
                 **({"codecs": mip_codecs[m["index"]]} if m["index"] < len(mip_codecs) else {}))
            for m in lay["mips"]]
    if "available" in a:
        out["available"] = a["available"]
        out["reader_limit"] = a["reader_limit"]
        out["reader_exported"] = a["reader_exported"]
        out["full_chain"] = a["full_chain"]
    if a.get("blocked"):
        out["blocked"] = a["blocked"]
    return out


def _check_mip_chain(layout, fmt, slice_bytes_fn):
    """Detile every mip of one 2D slice and return (means_per_mip, mip_errors, predictions).

    predictions[k] is the visible-element prediction for mip k from the decoded pixels of
    mip k - 1 (None for mip 0 or an undecodable parent); mip_errors[k - 1] compares it with
    mip k's visible elements, so odd parent counts and partial edge blocks are covered.
    """
    means, errs, preds = [], [], [None]
    prev_px = None
    block = 4 if fmt in BC_FORMATS else 1
    hdr = fmt in _HDR_FORMATS
    for k, mip in enumerate(layout["mips"]):
        data = detile(layout, mip, slice_bytes_fn(mip))
        px = decode_pixels(fmt, data, mip["blocks_w"], mip["blocks_h"])
        means.append(None if px is None else block_means(
            px, mip["blocks_w"], mip["blocks_h"], mip["pixels_w"], mip["pixels_h"], block))
        if k:
            pred = None if prev_px is None else predict_child_means(
                prev_px, layout["mips"][k - 1], mip, block)
            preds.append(pred)
            errs.append(None if pred is None or px is None else round(
                mean_abs_error(pred, visible_block_means(means[k], mip, block), hdr=hdr), 2))
        prev_px = px
    return means, errs, preds


def _check_mip_chain_means(layout, fmt, slice_bytes_fn):
    """Detile every mip of one 2D slice and return (means_per_mip, mip_errors)."""
    means, errs, _ = _check_mip_chain(layout, fmt, slice_bytes_fn)
    return means, errs


def _tail_origin_control(layout, fmt, tiled, preds):
    """Negative control: the first tail mip decoded at the tile origin instead of its in-tile
    offset, against its prediction from the parent mip. None when it cannot be decoded."""
    ft = layout["first_tail"]
    if not preds or ft >= len(preds) or preds[ft] is None:
        return None
    tm = layout["mips"][ft]
    bad = dict(tm, tile_x=0, tile_y=0)
    data = detile(layout, bad, tiled[:layout["tile"]])
    px = decode_pixels(fmt, data, tm["blocks_w"], tm["blocks_h"])
    if px is None:
        return None
    block = 4 if fmt in BC_FORMATS else 1
    means = block_means(px, tm["blocks_w"], tm["blocks_h"], tm["pixels_w"], tm["pixels_h"], block)
    return round(mean_abs_error(preds[ft], visible_block_means(means, tm, block),
                                hdr=(fmt in _HDR_FORMATS)), 2)


def _read_range_or_block(pool, start, size, cache, bcpack_mode, index, shape=None,
                         bcpack_history=None):
    try:
        return pool.read_range(start, size, cache, bcpack_mode=bcpack_mode,
                               bcpack_history=bcpack_history)
    except BcpackRequired as exc:
        msg = dict(index=index, blocked=str(exc))
    except bcpack.BcpackProvisionalError as exc:
        msg = dict(
            index=index,
            blocked=f"provisional BCPack history required ({exc}); pass --allow-provisional",
        )
    except (OSError, ValueError) as exc:
        msg = dict(index=index, blocked=str(exc))
    if shape is not None:
        msg["shape"] = shape
    print(json.dumps(msg, indent=1))
    return None


def check(pool, index, *, allow_provisional=False, history=None):
    try:
        return _check_texture(pool, index, allow_provisional=allow_provisional, history=history)
    except MipGeometryError as exc:
        print(json.dumps(dict(index=index, blocked=str(exc)), indent=1))
        return 2


def _check_texture(pool, index, *, allow_provisional=False, history=None):
    a = analyze(pool, index)
    if "layout" not in a or "available" not in a or a.get("blocked"):
        print(json.dumps(dict(index=index, blocked=a.get("blocked")), indent=1))
        return 2
    layout = a["layout"]
    d = a["descriptor"]
    shape = a.get("shape", "2d")
    cache = {}
    bcpack_mode = "provisional" if allow_provisional else "strict"

    if shape == "volume":
        if d["format"] not in VOLUME_FLOAT_FORMATS:
            print(json.dumps(dict(index=index, shape=shape, blocked=(
                f"format {d['format']} is not one of the float16 volume formats this tool decodes "
                "(10/54)")), indent=1))
            return 2
        tiled = _read_range_or_block(pool, d["base"], layout["size"], cache, bcpack_mode, index,
                                     shape, history)
        if tiled is None:
            return 2
        linear_3d = detile_volume(layout, tiled)
        vox_3d = _decode_volume_voxels(d["format"], linear_3d, d["width"], d["height"], d["depth"])
        z_diff_3d = _volume_z_diff(vox_3d, d["width"], d["height"], d["depth"])
        # Negative control: treat the buffer as stacked 2D SW_4KB_S planes.
        lay_2d = compact_surface_layout(d["width"], d["height"], 1, layout["element_bytes"], 1, 4096)
        mip_2d = lay_2d["mips"][0]
        plane_bytes = lay_2d["size"]
        buf_2d = bytearray()
        for z in range(d["depth"]):
            off = z * plane_bytes
            if off + plane_bytes <= len(tiled):
                buf_2d += detile(lay_2d, mip_2d, tiled[off:off + plane_bytes])
            else:
                buf_2d += bytes(d["width"] * d["height"] * layout["element_bytes"])
        vox_2d = _decode_volume_voxels(d["format"], bytes(buf_2d), d["width"], d["height"], d["depth"])
        z_diff_2d = _volume_z_diff(vox_2d, d["width"], d["height"], d["depth"])
        result = dict(index=index, shape=shape, width=d["width"], height=d["height"],
                      depth=d["depth"], format=d["format"], mips=d["mips"], tile=layout["tile"],
                      block_d=layout["block_d"], padded_d=layout["padded_d"],
                      volume_3d_z_diff=round(z_diff_3d, 6) if z_diff_3d is not None else None,
                      volume_2d_control_z_diff=round(z_diff_2d, 6) if z_diff_2d is not None else None,
                      full_chain=a.get("full_chain"))
        print(json.dumps(result, indent=1))
        ok = (z_diff_3d is not None and z_diff_2d is not None and
              z_diff_3d < 0.5 * z_diff_2d)
        return 0 if ok else 1

    if shape == "2d_array":
        tiled = _read_range_or_block(pool, d["base"], layout["size"], cache, bcpack_mode, index,
                                     shape, history)
        if tiled is None:
            return 2
        per_slice_max = []
        all_errs = []
        for s in range(d["depth"]):
            means, errs = _check_mip_chain_means(
                layout, d["format"],
                lambda m, sl=s: tiled[m["offset"] + sl * m["slice_stride"]:
                                      m["offset"] + sl * m["slice_stride"] + m["region"]])
            if all(m is None for m in means):
                print(json.dumps(dict(index=index, shape=shape, blocked=(
                    f"format {d['format']} is not one of the formats this tool decodes "
                    "(BC 71/72/74/75/77/78/80/83/95/96/98/99; uncompressed 10/11/28/29/49/54/56/61/65/87/91)")), indent=1))
                return 2
            per_slice_max.append(max((e for e in errs if e is not None), default=None))
            all_errs.extend(errs)
        # Negative control: interpret the same allocation as slice-major.
        single = surface_layout(d["width"], d["height"], d["format"], d["mips"], layout["tile"])
        ctrl_errs = []
        for s in range(d["depth"]):
            base_s = s * layout["slice_size"]
            _, cerrs = _check_mip_chain_means(
                single, d["format"],
                lambda m, b0=base_s: tiled[b0 + m["offset"]:b0 + m["offset"] + m["region"]])
            ctrl_errs.extend(e for e in cerrs if e is not None)
        result = dict(index=index, shape=shape, width=d["width"], height=d["height"],
                      depth=d["depth"], format=d["format"], mips=d["mips"],
                      first_tail=layout["first_tail"], tile=layout["tile"],
                      per_slice_max_error=per_slice_max,
                      max_error=max((e for e in all_errs if e is not None), default=None),
                      slice_major_control_max_error=max(ctrl_errs, default=None),
                      full_chain=a.get("full_chain"))
        print(json.dumps(result, indent=1))
        return 0 if all(e is not None and e <= 6 for e in all_errs) else 1

    if shape == "cubemap":
        tiled = _read_range_or_block(pool, d["base"] + layout["data_offset"], layout["full_bytes"],
                                     cache, bcpack_mode, index, shape, history)
        if tiled is None:
            return 2
        per_face_max = []
        all_errs = []
        face0_preds = None
        for face in range(6):
            fbase = face * layout["face_size"]
            means, errs, preds = _check_mip_chain(
                layout, d["format"],
                lambda m, b0=fbase: tiled[b0 + m["offset"]:b0 + m["offset"] + m["region"]])
            if all(m is None for m in means):
                print(json.dumps(dict(index=index, shape=shape, blocked=(
                    f"format {d['format']} is not one of the formats this tool decodes "
                    "(BC 71/72/74/75/77/78/80/83/95/96/98/99; uncompressed 10/11/28/29/49/54/56/61/65/87/91)")), indent=1))
                return 2
            if face == 0:
                face0_preds = preds
            per_face_max.append(max((e for e in errs if e is not None), default=None))
            all_errs.extend(errs)
        control = None
        if 0 < layout["first_tail"] < d["mips"]:
            control = _tail_origin_control(layout, d["format"], tiled, face0_preds)
        result = dict(index=index, shape=shape, width=d["width"], height=d["height"],
                      depth=6, format=d["format"], mips=d["mips"], non_tail=d["non_tail"],
                      snapshots=layout["snapshots"], data_offset=layout["data_offset"],
                      first_tail=layout["first_tail"], tile=layout["tile"],
                      per_face_max_error=per_face_max,
                      max_error=max((e for e in all_errs if e is not None), default=None),
                      tail_origin_control=control, full_chain=a.get("full_chain"))
        print(json.dumps(result, indent=1))
        return 0 if all(e is not None and e <= 6 for e in all_errs) else 1

    tiled = _read_range_or_block(pool, d["base"], layout["size"], cache, bcpack_mode, index,
                                 bcpack_history=history)
    if tiled is None:
        return 2
    means, errs, preds = _check_mip_chain(
        layout, d["format"],
        lambda m: tiled[m["offset"]:m["offset"] + m["region"]])
    if all(m is None for m in means):
        print(json.dumps(dict(index=index, blocked=(
            f"format {d['format']} is not one of the formats this tool decodes "
            "(BC 71/72/74/75/77/78/80/83/95/96/98/99; uncompressed 10/11/28/29/49/54/56/61/65/87/91)")), indent=1))
        return 2
    control = None
    if 0 < layout["first_tail"] < d["mips"]:
        control = _tail_origin_control(layout, d["format"], tiled, preds)
    result = dict(index=index, width=d["width"], height=d["height"], format=d["format"],
                  mips=d["mips"], non_tail=d["non_tail"], first_tail=layout["first_tail"],
                  tile=layout["tile"], chain_error=errs, max_error=max(
                      (e for e in errs if e is not None), default=None),
                  tail_origin_control=control, full_chain=a.get("full_chain"))
    print(json.dumps(result, indent=1))
    return 0 if all(e is not None and e <= 6 for e in errs) else 1


def dds_bytes(width, height, fmt, mips, linear, payload,
              depth=1, array_size=1, is_cube=False, is_volume=False):
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    flags = 0x1007 | (0x80000 if fmt in BC_FORMATS else 0x8)
    if mips > 1:
        flags |= 0x20000
    if is_volume:
        flags |= 0x800000
    caps = 0x1000
    if mips > 1:
        caps |= 0x400008
    if is_cube or is_volume or array_size > 1:
        caps |= 0x8
    caps2 = 0
    if is_cube:
        caps2 |= 0xFE00
    if is_volume:
        caps2 |= 0x200000
    struct.pack_into("<IIIIIII", hdr, 4, 124, flags, height, width, linear,
                     depth if is_volume else 0, mips)
    struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
    struct.pack_into("<II", hdr, 108, caps, caps2)
    dim = 4 if is_volume else 3
    misc = 0x4 if is_cube else 0
    dx10_array = 1 if (is_cube or is_volume) else array_size
    return bytes(hdr) + struct.pack("<IIIII", fmt, dim, misc, dx10_array, 0) + payload


def texture_dds(pool, a, mips=None, *, allow_provisional=False, history=None):
    """Build the complete DDS in memory before publishing it to disk."""
    if "layout" not in a or "available" not in a:
        raise ValueError(a.get("blocked", "layout unavailable"))
    d = a["descriptor"]
    shape = a.get("shape", "2d")
    if shape != "volume" and d["format"] not in BC_FORMATS and d["format"] not in UNCOMPRESSED_2D_FORMATS:
        raise ValueError(f"format {d['format']} is uncompressed and not validated for 2D export")
    if shape == "volume" and d["format"] not in VOLUME_FLOAT_FORMATS:
        raise ValueError(f"volume format {d['format']} is not validated for export")
    count = d["mips"] if mips is None else mips
    if count < 1 or count > d["mips"]:
        raise ValueError(f"--mips must be 1..{d['mips']}")
    available = a["available"]
    if len(available) < count or not all(available[:count]):
        raise ValueError("not every requested mip uses a supported codec (zlib/raw/BCPack BC1/BC3/BC4/BC5/BC6H/BC7)")
    layout = a["layout"]
    block = 4 if d["format"] in BC_FORMATS else 1
    cache = {}
    bcpack_mode = "provisional" if allow_provisional else "strict"
    payload = bytearray()
    if shape == "volume":
        tiled = pool.read_range(d["base"], layout["size"], cache,
                                bcpack_mode=bcpack_mode, bcpack_history=history)
        payload += detile_volume(layout, tiled)
    elif shape == "2d_array":
        for s in range(d["depth"]):
            for mip in layout["mips"][:count]:
                tiled = pool.read_range(d["base"] + mip["offset"] + s * mip["slice_stride"],
                                        mip["region"], cache,
                                        bcpack_mode=bcpack_mode, bcpack_history=history)
                payload += _visible_rows(layout, mip, detile(layout, mip, tiled), block)
    elif shape == "cubemap":
        face_base = d["base"] + layout["data_offset"]
        for face in range(6):
            for mip in layout["mips"][:count]:
                tiled = pool.read_range(face_base + face * layout["face_size"] + mip["offset"],
                                        mip["region"], cache,
                                        bcpack_mode=bcpack_mode, bcpack_history=history)
                payload += _visible_rows(layout, mip, detile(layout, mip, tiled), block)
    else:
        for mip in layout["mips"][:count]:
            tiled = pool.read_range(d["base"] + mip["offset"], mip["region"], cache,
                                    bcpack_mode=bcpack_mode, bcpack_history=history)
            payload += _visible_rows(layout, mip, detile(layout, mip, tiled), block)
    if 148 + len(payload) > MAX_OUTPUT:
        raise ValueError("DDS exceeds the 256 MiB output limit")
    top_w, top_h = (_visible_grid(layout["mips"][0], block) if d["format"] in BC_FORMATS else (0, 0))
    linear = (top_w * top_h * layout["element_bytes"]
              if d["format"] in BC_FORMATS else d["width"] * layout["element_bytes"])
    return dds_bytes(
        d["width"], d["height"], d["format"], count, linear, bytes(payload),
        depth=d["depth"] if shape == "volume" else 1,
        array_size=d["depth"] if shape == "2d_array" else 1,
        is_cube=(shape == "cubemap"),
        is_volume=(shape == "volume"),
    )


def export(pool, index, out_path, mips=None, force=False, *, allow_provisional=False, history=None):
    a = analyze(pool, index)
    try:
        blob = texture_dds(pool, a, mips, allow_provisional=allow_provisional, history=history)
    except bcpack.BcpackProvisionalError as exc:
        print(f"texture {index}: provisional BCPack history required ({exc}); pass --allow-provisional", file=sys.stderr)
        return 2
    except (BcpackRequired, OSError, ValueError) as exc:
        print(f"texture {index}: {exc}", file=sys.stderr)
        return 2
    if os.path.exists(out_path) and not force:
        print(f"{out_path} exists; pass --force to replace it", file=sys.stderr)
        return 2
    d = a["descriptor"]
    count = d["mips"] if mips is None else mips
    shape = a.get("shape", "2d")
    try:
        with open(out_path, "wb") as f:
            f.write(blob)
    except OSError as exc:
        print(f"{out_path}: cannot write: {exc.strerror or exc}", file=sys.stderr)
        return 1
    dims = (f"{d['width']}x{d['height']}x{d['depth']}" if shape in ("2d_array", "volume")
            else f"{d['width']}x{d['height']}")
    print(f"texture {index}: {shape} {dims} fmt {d['format']} {count} mips "
          f"({len(blob)} bytes) -> {out_path}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    ps = sub.add_parser("scan", help="aggregate counters over pools")
    ps.add_argument("paths", nargs="+")
    ps.add_argument("--json")
    ps.add_argument("--tuples", action="store_true",
                    help="include the distinct rejected compact-profile tuples")
    pl = sub.add_parser("layout", help="descriptor fields and mip offsets")
    pl.add_argument("pool")
    pl.add_argument("index", type=int)
    pc = sub.add_parser("check", help="decode a chain and report downsample error")
    pc.add_argument("pool")
    pc.add_argument("index", type=int)
    pc.add_argument("--allow-provisional", action="store_true",
                    help="allow unconfirmed built-in BCPack history entries")
    pc.add_argument("--history",
                    help="local JSON file or directory supplying BC6H/BC7 (and optional BC1-BC5) initial history tables")
    pe = sub.add_parser("export", help="write the full chain as DDS")
    pe.add_argument("pool")
    pe.add_argument("index", type=int)
    pe.add_argument("out")
    pe.add_argument("--mips", type=int)
    pe.add_argument("--force", action="store_true")
    pe.add_argument("--calibration", metavar="PROFILE.json", help="export using an explicit locally fitted profile")
    pe.add_argument("--allow-provisional", action="store_true",
                    help="allow unconfirmed built-in BCPack history entries")
    pe.add_argument("--history",
                    help="local JSON file or directory supplying BC6H/BC7 (and optional BC1-BC5) initial history tables")
    a = ap.parse_args(argv)
    if a.cmd == "scan":
        pools = []
        for p in a.paths:
            found = find_pools(p)
            if not found:   # a mistyped path would otherwise report an all-zero scan and exit 0
                why = "no such file or folder" if not os.path.exists(p) else "no .ttpf pool in it"
                print(f"ERROR: {p}: {why}", file=sys.stderr)
                return 2
            pools += found
        # The report is written after every pool is read: a pool given as --json used to be replaced by it.
        if a.json:
            if os.path.isdir(a.json):
                print(f"ERROR: --json {a.json} is a directory", file=sys.stderr)
                return 2
            for p in pools:
                if same_output_path(a.json, p):
                    print(f"ERROR: --json would overwrite the input {p}", file=sys.stderr)
                    return 2
        try:
            scan_pools(pools, a.json, a.tuples)
        except ValueError as exc:   # a truncated or unsupported pool: name it instead of a traceback
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        return 0
    if a.cmd == "export":
        # Refuse an output that aliases an input asset or metadata before the pool is
        # opened (so its mmap can never be truncated on disk). --force may replace an
        # ordinary DDS but never a pool, a calibration profile, or a supplied history file.
        for asset, label in ((a.pool, "input pool"),
                             (a.calibration, "--calibration profile"),
                             (a.history, "--history file")):
            if asset is None:
                continue
            if label == "--history file" and not os.path.isfile(asset):
                continue
            if same_output_path(a.out, asset):
                print(f"ERROR: {a.out} would overwrite the {label} {asset}", file=sys.stderr)
                return 2
    try:
        pool = Pool(a.pool)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {a.pool}: {exc}", file=sys.stderr)
        return 2
    try:
        if not 0 <= a.index < pool.textures:
            print(f"ERROR: texture {a.index} is out of range: {a.pool} holds {pool.textures}",
                  file=sys.stderr)
            return 2
        if a.cmd == "layout":
            print(json.dumps(describe(pool, a.index), indent=1))
            return 0
        if a.cmd == "check":
            return check(pool, a.index, allow_provisional=a.allow_provisional, history=a.history)
        if a.calibration:
            if a.history:
                print("ERROR: choose either --calibration or --history", file=sys.stderr)
                return 2
            try:
                from calibrated_texture import export_file
                return export_file(pool, a.index, a.out, a.mips, a.force, a.calibration)
            except (OSError, ImportError, ValueError, TypeError, KeyError, IndexError) as exc:
                print(f"calibrated export failed: {exc}", file=sys.stderr)
                return 2
        return export(pool, a.index, a.out, a.mips, a.force,
                      allow_provisional=a.allow_provisional, history=a.history)
    finally:
        pool.close()


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
