"""Find texture headers in carved CPU-segment blobs and pair them with their GPU blob.

Texture descriptor layout (qwords, found in GTA VI's resource CPU segments):
  q[k-2]  packed u64 (unknown: size/flags)
  q[k-1]  0x980                     (descriptor type)
  q[k]    0x60000000 + offset       (GPU data pointer into the resource's GPU segment)
  q[k+4]  u16 width, u16 height, u16 depth/array, u8 ?, u8 DXGI format
  q[k+5]  tiling / flags (0x0001 = AMD SW_4K_S verified on an R16 texture)
The GPU segment is the carved blob that follows the CPU blob.

This module also carries the Gfx10 compact mip layout rule used by the RDS5/TTPF pools
(`tools/ttpf_mips.py`): tile-mode selection, the packed tail, and the allocation-span check.

Usage:
    python -I tex_headers.py <carve dir> [--archives a,b] [--out textures_index.json]
"""
import argparse
import array
import json
import os
import stat
import struct
import sys
from collections import Counter

_UINT64_CODE = "Q" if array.array("Q").itemsize == 8 else None

DXGI = {
    2: "R32G32B32A32_FLOAT", 3: "R32G32B32A32_UINT", 4: "R32G32B32A32_SINT",
    6: "R32G32B32_FLOAT", 7: "R32G32B32_UINT", 8: "R32G32B32_SINT",
    10: "R16G16B16A16_FLOAT", 11: "R16G16B16A16_UNORM", 12: "R16G16B16A16_UINT",
    13: "R16G16B16A16_SNORM", 14: "R16G16B16A16_SINT",
    16: "R32G32_FLOAT", 17: "R32G32_UINT", 18: "R32G32_SINT",
    24: "R10G10B10A2_UNORM", 25: "R10G10B10A2_UINT", 26: "R11G11B10_FLOAT",
    28: "R8G8B8A8_UNORM", 29: "R8G8B8A8_UNORM_SRGB", 30: "R8G8B8A8_UINT",
    31: "R8G8B8A8_SNORM", 32: "R8G8B8A8_SINT",
    34: "R16G16_FLOAT", 35: "R16G16_UNORM", 36: "R16G16_UINT",
    37: "R16G16_SNORM", 38: "R16G16_SINT",
    41: "R32_FLOAT", 42: "R32_UINT", 43: "R32_SINT",
    49: "R8G8_UNORM", 50: "R8G8_UINT", 51: "R8G8_SNORM", 52: "R8G8_SINT",
    54: "R16_FLOAT", 56: "R16_UNORM", 57: "R16_UINT", 58: "R16_SNORM", 59: "R16_SINT",
    61: "R8_UNORM", 62: "R8_UINT", 63: "R8_SNORM", 64: "R8_SINT",
    71: "BC1_UNORM", 72: "BC1_UNORM_SRGB",
    74: "BC2_UNORM", 75: "BC2_UNORM_SRGB",
    77: "BC3_UNORM", 78: "BC3_UNORM_SRGB",
    80: "BC4_UNORM", 81: "BC4_SNORM",
    83: "BC5_UNORM", 84: "BC5_SNORM",
    87: "B8G8R8A8_UNORM", 91: "B8G8R8A8_UNORM_SRGB",
    95: "BC6H_UF16", 96: "BC6H_SF16",
    98: "BC7_UNORM", 99: "BC7_UNORM_SRGB",
}

GFX10_TO_DXGI = {
    1: 61,
    2: 63,
    5: 62,
    6: 64,
    7: 56,
    8: 58,
    11: 57,
    12: 59,
    13: 54,
    14: 49,
    15: 51,
    18: 50,
    19: 52,
    20: 42,
    21: 43,
    22: 41,
    23: 35,
    24: 37,
    27: 36,
    28: 38,
    29: 34,
    36: 26,
    50: 24,
    54: 25,
    56: 28,
    57: 31,
    60: 30,
    61: 32,
    62: 17,
    63: 18,
    64: 16,
    65: 11,
    66: 13,
    69: 12,
    70: 14,
    71: 10,
    72: 7,
    73: 8,
    74: 6,
    75: 3,
    76: 4,
    77: 2,
    130: 29,
    169: 71,
    170: 72,
    171: 74,
    172: 75,
    173: 77,
    174: 78,
    175: 80,
    176: 81,
    177: 83,
    178: 84,
    179: 95,
    180: 96,
    181: 98,
    182: 99,
}
DXGI_TO_GFX10 = {v: k for k, v in GFX10_TO_DXGI.items()}


def map_gfx10_format(gfx10_code):
    if gfx10_code not in GFX10_TO_DXGI:
        raise ValueError(f"unsupported or invalid GFX10 format: {gfx10_code}")
    dxgi_id = GFX10_TO_DXGI[gfx10_code]
    return dxgi_id, DXGI.get(dxgi_id, f"dxgi{dxgi_id}")


def dxgi_to_gfx10(dxgi_code):
    if dxgi_code not in DXGI_TO_GFX10:
        raise ValueError(f"unsupported DXGI format: {dxgi_code}")
    return DXGI_TO_GFX10[dxgi_code]


def calc_mipmap_count(w, h, depth=1):
    if w <= 0 or h <= 0 or depth <= 0:
        return 0
    return max(w, h, depth).bit_length()


def calc_mipmap_dimensions(w, h, depth=1, mips=0):
    if w <= 0 or h <= 0 or depth <= 0:
        return []
    max_mips = calc_mipmap_count(w, h, depth)
    count = max_mips if mips <= 0 else min(mips, max_mips)
    return [(max(1, w >> i), max(1, h >> i), max(1, depth >> i)) for i in range(count)]


FMT_INFO = {
    71: (8, 4), 72: (8, 4), 80: (8, 4), 81: (8, 4),
    74: (16, 4), 75: (16, 4), 77: (16, 4), 78: (16, 4), 83: (16, 4), 84: (16, 4),
    95: (16, 4), 96: (16, 4), 98: (16, 4), 99: (16, 4),
    61: (1, 1), 62: (1, 1), 63: (1, 1), 64: (1, 1),
    49: (2, 1), 50: (2, 1), 51: (2, 1), 52: (2, 1), 54: (2, 1), 56: (2, 1), 57: (2, 1), 58: (2, 1), 59: (2, 1),
    16: (8, 1), 17: (8, 1), 18: (8, 1),
    24: (4, 1), 25: (4, 1), 26: (4, 1), 28: (4, 1), 29: (4, 1), 30: (4, 1), 31: (4, 1), 32: (4, 1),
    34: (4, 1), 35: (4, 1), 36: (4, 1), 37: (4, 1), 38: (4, 1), 41: (4, 1), 42: (4, 1), 43: (4, 1),
    87: (4, 1), 91: (4, 1),
    6: (12, 1), 7: (12, 1), 8: (12, 1),
    10: (8, 1), 11: (8, 1), 12: (8, 1), 13: (8, 1), 14: (8, 1),
    2: (16, 1), 3: (16, 1), 4: (16, 1),
}


def calc_mip_level_bytes(w, h, depth=1, fmt=28, align=1):
    if fmt not in FMT_INFO:
        raise ValueError(f"unsupported format for byte calculation: {fmt}")
    bpe, blk = FMT_INFO[fmt]
    we = max(1, (w + blk - 1) // blk)
    he = max(1, (h + blk - 1) // blk)
    slice_bytes = we * he * bpe
    total = slice_bytes * max(1, depth)
    if align > 1:
        total = (total + align - 1) // align * align
    return total


# AMD Gfx10 compact mip chain (the TTPF "compact" profile) and multi-slice layouts.
#
# The 9-bit descriptor mode is a low flag bit plus a 4-bit Gfx10 tile mode at
# bit 1. Tile mode 0 is SW_64KB_D; the observed 1-3 and 9-12 values are
# SW_4KB_S. Mips are stored smallest first. The last few mips share one tile at
# offset 0 instead of getting their own allocation.
COMPACT_TILE_4K = 4096
COMPACT_TILE_64K = 65536
COMPACT_MAX_TAIL_MIPS = {COMPACT_TILE_4K: 8, COMPACT_TILE_64K: 12}
_COMPACT_ELEMENT_LOG = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}
_COMPACT_TILE_ORIGIN = {0: (16, 16), 1: (16, 8), 2: (8, 8), 3: (8, 4), 4: (4, 4)}
# AMD Gfx10 3D SW_4KB_S micro-tile extents (blocks_w, blocks_h, blocks_d) by log2(element_bytes).
_VOLUME_4K_TILE = {
    0: (16, 16, 16),
    1: (8, 16, 16),
    2: (8, 16, 8),
    3: (8, 8, 8),
    4: (4, 8, 8),
}
# 5-bit chunk page-count codes packed in the low 20 bits of ((extra << 11) | meta),
# mapping each non-zero code to its RDS5 chunk virtual size in 4 KiB pages.
ETPF_CHUNK_CODE_PAGES = {
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    10: 12, 11: 16, 12: 24, 13: 32, 14: 48, 15: 64, 16: 128,
    21: 18, 22: 80, 23: 96, 24: 144, 26: 272, 27: 384, 28: 528,
}


def gfx10_tile_mode(mode):
    """Return the 4-bit Gfx10 tile mode hidden in the TTPF descriptor mode."""
    if not isinstance(mode, int) or mode < 0:
        raise ValueError(f"invalid descriptor mode: {mode!r}")
    return (mode >> 1) & 0xF


def gfx10_tile_bytes(mode):
    """Tile size for a descriptor mode: 64 KiB only for tile mode 0."""
    return COMPACT_TILE_64K if gfx10_tile_mode(mode) == 0 else COMPACT_TILE_4K


def joaat(s: str) -> int:
    """Standard Jenkins One-at-a-Time 32-bit hash of lowercase ASCII string."""
    h = 0
    for c in s.lower().encode("latin-1", "replace"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def is_system_pointer(val: int, max_len: int) -> bool:
    """Validate that the pointer is strictly in the system page segment (0x50000000..0x5FFFFFFF)."""
    if (val >> 28) != 5:
        return False
    offset = val & 0x0FFFFFFF
    return 0 < offset + 8 <= max_len


def parse_texture_dict_page(data: bytes):
    """Parse a small RAGE texture dictionary system page (e.g. the 400-byte stream in textures.rpf).

    Layout:
      +0x20: 64-bit pointer to hash array (optional in synthetic tests)
      +0x30: 64-bit pointer to grcTexture descriptor pointer array (optional in synthetic tests)
      +0x38: u16 texture count (optional in synthetic tests)
      +0x140..pointers_off: null-separated lower-case ASCII texture names
      pointers_off (+0x170 or +0x178): array of 64-bit pointers to grcTexture descriptors (tagged 0x5...)
      hashes_off (+0x180 or +0x188): array of 32-bit joaat32 name hashes
      Within each grcTexture descriptor:
        +0x10: dtype (0x980)
        +0x18: data pointer (0x60000000 + GPU data offset)
        +0x38: u16 width, u16 height, u16 depth, u8 b6, u8 DXGI format
        +0x40: u64 flags (bits 16..23: mip count)
    """
    if len(data) < 400:
        raise ValueError(f"texture dictionary page truncated: expected >= 400 bytes, got {len(data)}")

    pointers_off = None
    hashes_off = None
    count = 2

    # Check pgDictionary header at 0x20 / 0x30
    if len(data) >= 0x40:
        h_cand = struct.unpack_from("<Q", data, 0x20)[0]
        p_cand = struct.unpack_from("<Q", data, 0x30)[0]
        cnt = struct.unpack_from("<H", data, 0x38)[0]
        if is_system_pointer(h_cand, len(data)) and is_system_pointer(p_cand, len(data)):
            h_off = h_cand & 0x0FFFFFFF
            p_off = p_cand & 0x0FFFFFFF
            if 0 < cnt <= 64 and p_off + cnt * 8 <= len(data) and h_off + cnt * 4 <= len(data):
                pointers_off = p_off
                hashes_off = h_off
                count = cnt

    if pointers_off is None:
        # Fallback probe for shifted or unheadered layouts:
        # 1. Standard layout: pointers start at 0x170
        # 2. Shifted layout (longer names in 0x140..): pointers start at 0x178, so both 0x178 and 0x180 are system pointers
        if is_system_pointer(struct.unpack_from("<Q", data, 0x170)[0], len(data)):
            pointers_off = 0x170
            hashes_off = 0x180
        elif is_system_pointer(struct.unpack_from("<Q", data, 0x178)[0], len(data)) and is_system_pointer(struct.unpack_from("<Q", data, 0x180)[0], len(data)):
            pointers_off = 0x178
            hashes_off = 0x188
        else:
            pointers_off = 0x170
            hashes_off = 0x180

    names_raw = data[0x140:pointers_off]
    names = [s.decode("latin1") for s in names_raw.split(b"\x00") if s]
    names_by_hash = {joaat(n): n for n in names}

    textures = []
    for i in range(count):
        if pointers_off + (i + 1) * 8 > len(data):
            break
        raw_ptr = struct.unpack_from("<Q", data, pointers_off + i * 8)[0]
        thash = struct.unpack_from("<I", data, hashes_off + i * 4)[0] if hashes_off + (i + 1) * 4 <= len(data) else 0

        # Validate that the pointer is strictly in the system page segment (0x5...)
        if (raw_ptr >> 28) != 5:
            continue
        ptr = raw_ptr & 0x0FFFFFFF
        if ptr == 0 or ptr + 0x48 > len(data):
            continue
        dtype = struct.unpack_from("<Q", data, ptr + 0x10)[0]
        data_off = struct.unpack_from("<Q", data, ptr + 0x18)[0] & 0x0FFFFFFF
        w, height, depth, b6, fmt = struct.unpack_from("<HHHBB", data, ptr + 0x38)
        flags = struct.unpack_from("<Q", data, ptr + 0x40)[0]
        mips = (flags >> 16) & 0xFF
        name = names_by_hash.get(thash, f"tex_{thash:08x}")
        textures.append({
            "name": name,
            "hash": thash,
            "width": w,
            "height": height,
            "depth": depth,
            "b6": b6,
            "mips": mips,
            "flags": flags,
            "format": fmt,
            "format_name": DXGI.get(fmt, f"dxgi{fmt}"),
            "data_offset": data_off,
            "dtype": dtype,
        })
    return textures


def parse_etpf_texture_record(record):
    """Parse one 16-byte ETPF texture record (the RDS5 embedded descriptor).

    Word ``b`` packs 3-bit ``non_tail`` (bits 0..2), 7-bit DXGI ``format``
    (bits 3..9), 11-bit ``depth - 1`` (bits 10..20), 2-bit ``dimension``
    (bits 21..22), and 9-bit ``mode`` (bits 23..31).
    """
    if len(record) < 16:
        raise ValueError(f"ETPF texture record truncated: expected 16 bytes, got {len(record)}")
    a, b, c, d = struct.unpack_from("<IIII", record, 0)
    return {
        "width": (a & 0x3FFF) + 1,
        "height": ((a >> 14) & 0x3FFF) + 1,
        "packed_mips": a >> 28,
        "non_tail": b & 7,
        "mips": (a >> 28) + (b & 7),
        "format": (b >> 3) & 0x7F,
        "depth": ((b >> 10) & 0x7FF) + 1,
        "dimension": (b >> 21) & 3,
        "mode": b >> 23,
        "meta": c >> 21,
        "extra": d,
        "base": (c & 0x1FFFFF) * 4096,
    }


def decode_etpf_chunk_pages(meta, extra):
    """Decode the 43-bit ``(extra << 11) | meta`` streaming chunk-page list.

    The low 20 bits hold up to four zero-terminated 5-bit codes giving each
    RDS5 chunk's virtual size in 4 KiB pages. Bits 9..18 of ``extra`` must be
    zero. Returns a tuple of page counts, or ``None`` if any code or reserved
    bit is invalid.
    """
    if not isinstance(meta, int) or not isinstance(extra, int):
        return None
    if meta < 0 or meta > 0x7FF or extra < 0 or extra > 0xFFFFFFFF:
        return None
    if ((extra >> 9) & 0x3FF) != 0:
        return None
    val43 = (extra << 11) | meta
    pages = []
    ended = False
    for slot in range(4):
        code = (val43 >> (5 * slot)) & 0x1F
        if code == 0:
            ended = True
            continue
        if ended:
            return None
        count = ETPF_CHUNK_CODE_PAGES.get(code)
        if count is None:
            return None
        pages.append(count)
    return tuple(pages)


def etpf_streaming_padding(extra):
    """Return ``(align_group, pad_pages)`` from the upper 13 bits of ``extra``."""
    return (extra >> 19) & 3, extra >> 21


def compact_profile_key(descriptor, element_bytes):
    """The exact tuple a validated compact-profile table is keyed on."""
    return (descriptor["mode"], descriptor["width"], descriptor["height"],
            descriptor["mips"], descriptor["non_tail"], element_bytes)


def compact_surface_layout(width, height, mips, element_bytes, block=1,
                           tile=COMPACT_TILE_4K):
    """Compute the Gfx10 compact mip chain for one 2D texture.

    ``element_bytes`` is the size of one pixel or BC block; ``block`` is 4 for
    BC formats and 1 for uncompressed formats. Returns the allocation size,
    the first packed-tail mip, block tile dimensions and one entry per mip
    with its offset and region size. Raises ValueError for a mip count that
    cannot describe the surface or a tile/element size outside the measured
    set.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid surface dimensions {width}x{height}")
    if element_bytes not in _COMPACT_ELEMENT_LOG:
        raise ValueError(f"unsupported element size: {element_bytes}")
    if block not in (1, 4):
        raise ValueError(f"unsupported block dimension: {block}")
    if tile not in COMPACT_MAX_TAIL_MIPS:
        raise ValueError(f"unsupported compact tile size: {tile}")
    levels = max(width, height).bit_length()
    if mips < 1 or mips > levels:
        raise ValueError(f"invalid mip count {mips} for {width}x{height}")

    log = _COMPACT_ELEMENT_LOG[element_bytes]
    half_bits = 6 if tile == COMPACT_TILE_4K else 8
    max_tail = COMPACT_MAX_TAIL_MIPS[tile]
    bw = 1 << (half_bits - log // 2)
    bh = 1 << (half_bits - (log + 1) // 2)
    mw, mh = _COMPACT_TILE_ORIGIN[log]
    ew = (width + block - 1) // block
    eh = (height + block - 1) // block

    tail = mips
    sizes = []
    out = []
    for i in range(mips):
        wc = max(1, (ew + (1 << i) - 1) >> i)
        hc = max(1, (eh + (1 << i) - 1) >> i)
        if mips > 1 and wc <= bw // 2 and hc <= bh and mips - i <= max_tail:
            tail = i
            break
        pw = (wc + bw - 1) // bw * bw
        ph = (hc + bh - 1) // bh * bh
        sizes.append(pw * ph * element_bytes)
        out.append({
            "index": i, "blocks_w": wc, "blocks_h": hc, "padded_w": pw, "padded_h": ph,
            "pixels_w": max(1, width >> i), "pixels_h": max(1, height >> i),
            "offset": 0, "region": pw * ph * element_bytes, "tile_x": 0, "tile_y": 0,
            "in_tail": False,
        })

    off = tile if tail < mips else 0
    for i in range(tail - 1, -1, -1):
        out[i]["offset"] = off
        off += sizes[i]
    for i in range(tail, mips):
        entry = max_tail - 1 - (i - tail)
        moff = (16 << entry) if entry > 6 else entry << 8
        x = y = 0
        for bit in range(6):
            x |= ((moff >> (9 + 2 * bit)) & 1) << bit
            y |= ((moff >> (8 + 2 * bit)) & 1) << bit
        wc = max(1, (ew + (1 << i) - 1) >> i)
        hc = max(1, (eh + (1 << i) - 1) >> i)
        out.append({
            "index": i, "blocks_w": wc, "blocks_h": hc, "padded_w": bw, "padded_h": bh,
            "pixels_w": max(1, width >> i), "pixels_h": max(1, height >> i),
            "offset": 0, "region": tile, "tile_x": x * mw, "tile_y": y * mh,
            "in_tail": True, "tail_offset": moff,
        })
    return {
        "size": off, "tile": tile, "first_tail": tail, "block_w": bw, "block_h": bh,
        "element_bytes": element_bytes, "levels": levels, "mips": out,
    }


def compact_layout_for_dxgi(width, height, fmt, mips, tile=COMPACT_TILE_4K):
    """Compact layout using tex_headers' DXGI element-size table."""
    if fmt not in FMT_INFO:
        raise ValueError(f"unsupported format for compact layout: {fmt}")
    bpe, blk = FMT_INFO[fmt]
    return compact_surface_layout(width, height, mips, bpe, blk, tile)


def array_surface_layout(width, height, depth, mips, element_bytes, block=1,
                         tile=COMPACT_TILE_64K):
    """Compute the mip-major Gfx10 layout for a 2D texture array.

    Every slice shares the same per-mip dimensions; the packed tail (if any)
    occupies ``depth * tile`` bytes at offset 0 (slice ``s`` at ``s * tile``),
    followed by standalone mips from ``first_tail - 1`` down to ``0``, each
    occupying ``depth * region`` bytes (slice ``s`` at ``offset + s * region``).
    """
    if not isinstance(depth, int) or depth < 1:
        raise ValueError(f"invalid array depth: {depth!r}")
    base = compact_surface_layout(width, height, mips, element_bytes, block, tile)
    first_tail = base["first_tail"]
    mips_out = [dict(m) for m in base["mips"]]
    off = depth * tile if first_tail < mips else 0
    for i in range(first_tail - 1, -1, -1):
        mips_out[i]["offset"] = off
        mips_out[i]["slice_stride"] = mips_out[i]["region"]
        off += depth * mips_out[i]["region"]
    for i in range(first_tail, mips):
        mips_out[i]["offset"] = 0
        mips_out[i]["slice_stride"] = tile
    return {
        "size": off,
        "slice_size": base["size"],
        "tile": tile,
        "first_tail": first_tail,
        "block_w": base["block_w"],
        "block_h": base["block_h"],
        "element_bytes": element_bytes,
        "levels": base["levels"],
        "depth": depth,
        "mip_major": True,
        "mips": mips_out,
    }


def array_layout_for_dxgi(width, height, depth, fmt, mips, tile=COMPACT_TILE_64K):
    """Mip-major 2D array layout using tex_headers' DXGI element-size table."""
    if fmt not in FMT_INFO:
        raise ValueError(f"unsupported format for array layout: {fmt}")
    bpe, blk = FMT_INFO[fmt]
    return array_surface_layout(width, height, depth, mips, bpe, blk, tile)


def cubemap_surface_layout(width, height, mips, element_bytes, block=1,
                           non_tail=0, chunk_pages=None):
    """Compute the slice-major SW_4KB_S layout for a 6-face streaming cubemap.

    ``non_tail == 0`` (mode 18) stores one 6-face cubemap at offset 0.
    ``non_tail in 1..3`` (modes 20/22/24) stores ``non_tail + 1`` self-contained
    cubemap snapshots back-to-back in the RDS5 chunk stream; the final chunk
    holds the full-resolution 6-face slice-major cubemap, zero-padded to a
    64 KiB boundary when ``6 * face_size >= 36 * 4096``.
    """
    if not isinstance(non_tail, int) or not 0 <= non_tail <= 3:
        raise ValueError(f"unsupported cubemap non_tail: {non_tail!r}")
    face = compact_surface_layout(width, height, mips, element_bytes, block, COMPACT_TILE_4K)
    face_size = face["size"]
    full_bytes = 6 * face_size
    if non_tail == 0:
        padded_full = full_bytes
        if chunk_pages is not None and len(chunk_pages) != 0:
            if len(chunk_pages) != 1 or chunk_pages[0] * COMPACT_TILE_4K != padded_full:
                raise ValueError(
                    f"non-streaming cubemap chunk_pages {chunk_pages!r} != expected ({padded_full // COMPACT_TILE_4K},)")
        data_offset = 0
        total_size = full_bytes
    else:
        padded_full = ((full_bytes + 65535) // 65536 * 65536
                       if full_bytes >= 36 * COMPACT_TILE_4K else full_bytes)
        if chunk_pages is not None:
            if len(chunk_pages) != non_tail + 1:
                raise ValueError(
                    f"cubemap chunk count mismatch: expected {non_tail + 1}, got {len(chunk_pages)}")
            if chunk_pages[-1] * COMPACT_TILE_4K != padded_full:
                raise ValueError(
                    f"cubemap final chunk size {chunk_pages[-1] * COMPACT_TILE_4K} != expected {padded_full}")
            data_offset = sum(chunk_pages[:-1]) * COMPACT_TILE_4K
            total_size = sum(chunk_pages) * COMPACT_TILE_4K
        else:
            data_offset = 0
            total_size = padded_full
    return {
        "size": total_size,
        "data_offset": data_offset,
        "face_size": face_size,
        "full_bytes": full_bytes,
        "padded_full_bytes": padded_full,
        "snapshots": non_tail + 1,
        "tile": COMPACT_TILE_4K,
        "first_tail": face["first_tail"],
        "block_w": face["block_w"],
        "block_h": face["block_h"],
        "element_bytes": element_bytes,
        "levels": face["levels"],
        "depth": 6,
        "mips": face["mips"],
    }


def cubemap_layout_for_dxgi(width, height, fmt, mips, non_tail=0, chunk_pages=None):
    """Cubemap layout using tex_headers' DXGI element-size table."""
    if fmt not in FMT_INFO:
        raise ValueError(f"unsupported format for cubemap layout: {fmt}")
    bpe, blk = FMT_INFO[fmt]
    return cubemap_surface_layout(width, height, mips, bpe, blk, non_tail, chunk_pages)


def volume_surface_layout(width, height, depth, mips, element_bytes):
    """Compute the AMD Gfx10 3D SW_4KB_S layout for a single-mip volume texture."""
    if width <= 0 or height <= 0 or depth <= 0:
        raise ValueError(f"invalid volume dimensions {width}x{height}x{depth}")
    if mips != 1:
        raise ValueError(f"only single-mip 3D volumes are validated, got mips={mips}")
    if element_bytes not in _COMPACT_ELEMENT_LOG:
        raise ValueError(f"unsupported volume element size: {element_bytes}")
    bw, bh, bd = _VOLUME_4K_TILE[_COMPACT_ELEMENT_LOG[element_bytes]]
    pw = (width + bw - 1) // bw * bw
    ph = (height + bh - 1) // bh * bh
    pd = (depth + bd - 1) // bd * bd
    z_tile_stride = (pw // bw) * (ph // bh) * COMPACT_TILE_4K
    size = (pd // bd) * z_tile_stride
    return {
        "size": size,
        "tile": COMPACT_TILE_4K,
        "first_tail": 1,
        "block_w": bw,
        "block_h": bh,
        "block_d": bd,
        "padded_w": pw,
        "padded_h": ph,
        "padded_d": pd,
        "z_tile_stride": z_tile_stride,
        "element_bytes": element_bytes,
        "depth": depth,
        "levels": calc_mipmap_count(width, height, depth),
        "mips": [{
            "index": 0,
            "blocks_w": width,
            "blocks_h": height,
            "blocks_d": depth,
            "padded_w": pw,
            "padded_h": ph,
            "padded_d": pd,
            "pixels_w": width,
            "pixels_h": height,
            "pixels_d": depth,
            "offset": 0,
            "region": size,
            "tile_x": 0,
            "tile_y": 0,
            "in_tail": False,
        }],
    }


def volume_layout_for_dxgi(width, height, depth, fmt, mips=1):
    """Single-mip 3D SW_4KB_S volume layout using tex_headers' DXGI table."""
    if fmt not in FMT_INFO:
        raise ValueError(f"unsupported format for volume layout: {fmt}")
    bpe, blk = FMT_INFO[fmt]
    if blk != 1:
        raise ValueError(f"block-compressed 3D volumes are not supported: format {fmt}")
    return volume_surface_layout(width, height, depth, mips, bpe)


def compact_span_matches(layout, span):
    """True only when an allocation span is byte-exact for the computed chain."""
    return isinstance(span, int) and span == layout["size"]


DDSD_CAPS = 0x00000001
DDSD_HEIGHT = 0x00000002
DDSD_WIDTH = 0x00000004
DDSD_PITCH = 0x00000008
DDSD_PIXELFORMAT = 0x00001000
DDSD_MIPMAPCOUNT = 0x00020000
DDSD_LINEARSIZE = 0x00080000
DDSD_DEPTH = 0x00800000

DDSCAPS_COMPLEX = 0x00000008
DDSCAPS_MIPMAP = 0x00400000
DDSCAPS_TEXTURE = 0x00001000

DDSCAPS2_CUBEMAP = 0x00000200
DDSCAPS2_CUBEMAP_ALLFACES = 0x0000FE00
DDSCAPS2_VOLUME = 0x00200000

D3D10_RESOURCE_DIMENSION_TEXTURE2D = 3
D3D10_RESOURCE_DIMENSION_TEXTURE3D = 4
D3D10_RESOURCE_MISC_TEXTURECUBE = 0x4


def dds_header(w, h, depth=1, mips=1, fmt=28, is_cube=False, linear_size=0, array_size=1):
    """Build a DX10 DDS header; array_size counts cubes for cubemaps.

    For known uncompressed formats, derive the row pitch. A nonzero linear_size
    overrides the derived top mip size for compressed formats.
    """
    hdr = bytearray(128)
    hdr[0:4] = b"DDS "
    flags = DDSD_CAPS | DDSD_HEIGHT | DDSD_WIDTH | DDSD_PIXELFORMAT
    if mips > 1:
        flags |= DDSD_MIPMAPCOUNT
    if depth > 1:
        flags |= DDSD_DEPTH
    layout = FMT_INFO.get(fmt)
    if layout is not None and layout[1] == 1:
        flags |= DDSD_PITCH
        pitch_or_size = calc_mip_level_bytes(w, 1, fmt=fmt)
    else:
        flags |= DDSD_LINEARSIZE
        pitch_or_size = linear_size
        if layout is not None and linear_size == 0:
            pitch_or_size = calc_mip_level_bytes(w, h, depth=depth, fmt=fmt)

    caps = DDSCAPS_TEXTURE
    if mips > 1:
        caps |= DDSCAPS_COMPLEX | DDSCAPS_MIPMAP
    if depth > 1 or is_cube or array_size > 1:
        caps |= DDSCAPS_COMPLEX

    caps2 = 0
    if depth > 1:
        caps2 |= DDSCAPS2_VOLUME
    if is_cube:
        caps2 |= DDSCAPS2_CUBEMAP_ALLFACES

    struct.pack_into("<IIIIIII", hdr, 4, 124, flags, h, w, pitch_or_size,
                     depth if depth > 1 else 0, mips if mips > 1 else 1)
    struct.pack_into("<II4s", hdr, 76, 32, 4, b"DX10")
    struct.pack_into("<II", hdr, 108, caps, caps2)

    dim = D3D10_RESOURCE_DIMENSION_TEXTURE3D if depth > 1 else D3D10_RESOURCE_DIMENSION_TEXTURE2D
    misc = D3D10_RESOURCE_MISC_TEXTURECUBE if is_cube else 0
    dxt10 = struct.pack("<IIIII", fmt, dim, misc, array_size, 0)
    return bytes(hdr) + dxt10


def dds_bytes(w, h, fmt, payload=b"", mips=1, depth=1, is_cube=False, array_size=1):
    linear_size = 0 if fmt in FMT_INFO else len(payload)
    return dds_header(w, h, depth=depth, mips=mips, fmt=fmt, is_cube=is_cube,
                      linear_size=linear_size, array_size=array_size) + payload


def parse_dds_header(data):
    if len(data) < 128:
        raise ValueError(f"DDS header truncated: expected at least 128 bytes, got {len(data)}")
    if data[:4] != b"DDS ":
        raise ValueError(f"invalid DDS magic: {data[:4]!r}")
    size, flags, h, w, pitch_linear, depth, mips = struct.unpack_from("<IIIIIII", data, 4)
    if size != 124:
        raise ValueError(f"invalid DDS header size: {size}")
    pf_size, pf_flags, fourcc = struct.unpack_from("<II4s", data, 76)
    if pf_size != 32:
        raise ValueError(f"invalid DDS pixel format size: {pf_size}")
    caps, caps2 = struct.unpack_from("<II", data, 108)

    info = {
        "magic": b"DDS ",
        "flags": flags,
        "width": w,
        "height": h,
        "depth": depth,
        "mips": mips,
        "pitch_linear": pitch_linear,
        "fourcc": fourcc,
        "caps": caps,
        "caps2": caps2,
    }

    if fourcc == b"DX10":
        if len(data) < 148:
            raise ValueError(f"truncated DX10 extension header: expected at least 148 bytes, got {len(data)}")
        dxgi_fmt, dim, misc, array_size, misc2 = struct.unpack_from("<IIIII", data, 128)
        info.update({
            "dxgi_format": dxgi_fmt,
            "resource_dimension": dim,
            "misc_flags": misc,
            "array_size": array_size,
            "misc_flags2": misc2,
        })
    return info


def _find_gpu_ptr_qwords_stdlib(buf: bytes):
    """Return (qwords, hit_indices) for 64-bit GPU pointers matching 0x60000000..0x6FFFFFFF."""
    n_qwords = len(buf) // 8
    if n_qwords == 0:
        return (), []
    trimmed = memoryview(buf)[: n_qwords * 8]
    if _UINT64_CODE is not None:
        q = array.array(_UINT64_CODE)
        q.frombytes(trimmed)
        if sys.byteorder != "little":
            q.byteswap()
    else:
        q = struct.unpack(f"<{n_qwords}Q", trimmed)
    hits = [k for k, v in enumerate(q) if 0x60000000 <= v <= 0x6FFFFFFF]
    return q, hits


def _find_gpu_ptr_qwords(buf: bytes):
    """Scan aligned little-endian u64 qwords, using numpy when installed or stdlib fallback."""
    try:
        import numpy as np
    except ImportError:
        return _find_gpu_ptr_qwords_stdlib(buf)
    q = np.frombuffer(buf[: len(buf) // 8 * 8], "<u8")
    hits = np.nonzero(((q >> 28) == 6) & ((q >> 32) == 0))[0]
    return q, hits


def scan_archive(carve, arch):
    with open(os.path.join(carve, arch, "index.json"), encoding="utf-8-sig") as f:
        files = sorted(json.load(f)["files"], key=lambda r: r["off"])
    out = []
    for i in range(len(files) - 1):
        v, g = files[i], files[i + 1]
        # CPU segment: starts with a zero qword but isn't zero-padded GPU data. The GPU segment is whatever
        # blob follows it (texture data often starts directly with BC blocks, not a zero header).
        if not (v["head"][:16] == "0" * 16 and not v["head"].startswith("0" * 32)):
            continue
        if g["head"].startswith("42400000") or g.get("type") == "bink2":
            continue
        # The GPU segment can span several consecutive blobs: everything up to the next CPU/header blob.
        chain = []
        for j in range(i + 1, min(len(files), i + 257)):
            r = files[j]
            if r["head"].startswith("42400000") or r.get("type") == "bink2" or (
                    r["head"][:16] == "0" * 16 and not r["head"].startswith("0" * 32) and chain):
                break
            chain.append([r["off"], r["size"]])
        with open(os.path.join(carve, arch, f"{v['off']:012x}.bin"), "rb") as f:
            V = f.read()
        q, hits = _find_gpu_ptr_qwords(V)
        # Any GPU pointer whose header (4 qwords later) decodes to sane dimensions and a known DXGI format.
        for k in hits:
            k = int(k)
            if k < 2 or k + 5 >= len(q):
                continue
            d = int(q[k + 4]).to_bytes(8, "little")
            w, h, depth = int.from_bytes(d[0:2], "little"), int.from_bytes(d[2:4], "little"), int.from_bytes(d[4:6], "little")
            fmt = d[7]
            if fmt not in DXGI or not (1 <= w <= 16384 and 1 <= h <= 16384 and 1 <= depth <= 2048):
                continue
            flags = int(q[k + 5])
            mips = (flags >> 16) & 0xFF
            dtype = int(q[k - 1])
            # Small dimensions need the known texture type and a mip count that fits the extent.
            if min(w, h) < 4 and (dtype != 0x980 or not 1 <= mips <= calc_mipmap_count(w, h, depth)):
                continue
            # Real texture data is 256-byte aligned (4 KB in every verified case); unaligned hits are other structs.
            if (int(q[k]) & 0xFF) != 0:
                continue
            out.append({"archive": arch, "cpu": v["off"], "gpu": g["off"], "gpu_size": g["size"], "gpu_chain": chain,
                        "data_off": int(q[k]) & 0x0FFFFFFF, "w": w, "h": h, "depth": depth, "b6": d[6], "fmt": fmt,
                        "fmt_name": DXGI.get(fmt, f"dxgi{fmt}"), "flags": flags, "packed": int(q[k - 2]),
                        "dtype": dtype, "mips": mips, "q_index": k})
    return out


def same_output_path(first, second):
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def input_files(carve, archs):
    """Yield (kind, path) for every file the scan reads from the selected carve archives.

    Each archive contributes its ``index.json`` manifest and the ``<offset>.bin`` streams next to it.
    A report path only has to match one of these, not the carve folder argument itself.
    """
    for arch in archs:
        folder = os.path.join(carve, arch)
        yield "manifest", os.path.join(folder, "index.json")
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(".bin"):
                    yield "blob", entry.path


def report_needs_input_check(carve, archs, path):
    """Whether a report path has to be compared with every scanned input.

    A report that already exists as a regular file, is not a symlink and has one link is reachable
    only through its own name. When that name is not ``index.json`` or a ``*.bin`` stream, no input
    can resolve to it, so the per-file realpath/samefile checks can be skipped. A missing report,
    a link, or an input-like name is always compared; an input may also be a symlink to it.
    """
    try:
        st = os.lstat(path)
    except OSError:
        return True
    if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1:
        return True
    name = os.path.basename(os.path.realpath(path)).lower()
    if name == "index.json" or name.endswith(".bin"):
        return True
    for arch in archs:
        with os.scandir(os.path.join(carve, arch)) as entries:
            for entry in entries:
                if entry.is_symlink():
                    leaf = entry.name.lower()
                    if leaf == "index.json" or leaf.endswith(".bin"):
                        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("carve")
    ap.add_argument("--archives")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not os.path.isdir(a.carve):
        ap.error(f"{a.carve} is not a folder")
    archs = a.archives.split(",") if a.archives else sorted(
        d for d in os.listdir(a.carve) if os.path.isfile(os.path.join(a.carve, d, "index.json")))
    missing = [arch for arch in archs if not os.path.isfile(os.path.join(a.carve, arch, "index.json"))]
    if missing:
        ap.error(f"no index.json for {', '.join(missing)} in {a.carve}")
    if not archs:
        ap.error(f"no carved archive (a folder with index.json) in {a.carve}")
    path = a.out or os.path.join(a.carve, "textures_index.json")
    if os.path.isdir(path):
        ap.error(f"{path} is a directory, not a .json file")
    if report_needs_input_check(a.carve, archs, path):
        for kind, source in input_files(a.carve, archs):
            if same_output_path(source, path):
                ap.error(f"output would overwrite an input {kind}: {path}")
    allt = []
    for arch in archs:
        try:
            t = scan_archive(a.carve, arch)
        except (OSError, ValueError, KeyError, TypeError) as e:   # a malformed index.json: one line, as gamedata_index
            where = getattr(e, "filename", None) or os.path.join(a.carve, arch, "index.json")
            print(f"ERROR: cannot read {where}: {getattr(e, 'strerror', None) or e!r}", file=sys.stderr)
            return 2
        print(f"{arch}: {len(t)} texture headers", flush=True)
        allt += t
    parent = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(allt, f)
    except OSError as e:
        print(f"ERROR: cannot write {path}: {e.strerror or e}", file=sys.stderr)
        return 2
    print(f"{len(allt)} textures -> {path}")
    print("formats:", Counter(t["fmt_name"] for t in allt).most_common(12))
    print("flags (low 16 bits):", Counter(hex(t["flags"] & 0xFFFF) for t in allt).most_common(10))
    print("byte6:", Counter(t["b6"] for t in allt).most_common(8))


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
