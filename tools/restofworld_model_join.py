"""Join RPF9 model system pages to their graphics pages without the encrypted TOC.

In the rest-of-world RPF9 archives, `rpf9_carve.py` and `restofworld_index.py` see
individual zlib streams: zero-prefix system pages (`0x50000000`-based pointers) and
`42 40 00 00` / meshlet-group graphics pages (`0x60000000`-based pointers). Grouping
them into complete drawable, drawable-dictionary and fragment resources does not
require the encrypted RPF9 table of contents:

1. A top-level resource starts on a 4,096-byte archive boundary + 16, preceded by
   16 zero bytes, with eight zero bytes at +0x00 and a `0x50000000`-based pointer
   at +0x08 to its `datResourceMap` page header (`0DRP` directory pages have `0DRP`
   at +0x10 and are excluded).
2. At `page_map = u64@+0x08 - 0x50000000`, `u8@(page_map + 8)` gives `system_pages`
   and `u8@(page_map + 9)` gives `graphics_pages`, followed by `system_pages +
   graphics_pages` 8-byte zero runtime slot words at `page_map + 0x10`. When
   `system_pages > 1`, `page_map` sits near the end of the concatenated system
   streams, and every non-final system or graphics stream is a full 64 KiB multiple.
3. The resource owns `system_pages` contiguous zlib streams followed immediately by
   `graphics_pages` contiguous zlib streams (each internal gap is 0..15 zero bytes to
   the next 16-byte archive boundary).
4. Every geometry record across all LODs (`+0x50..+0x68`) of every drawable in the
   system segment cross-validates against the joined graphics segment: vertex buffer
   `vb+0x18` (`0x60000000`-based) lands on a `0x42` or `0x63` vertex block matching
   the FVF position encoding and vertex count, and meshlet descriptor `geo+0x40`
   lands on an index buffer whose meshlet header, running sums and triangle indices
   match the descriptor counts and reach into the final declared graphics page.

No asset names, path strings, hashes or payload bytes are written. Reports contain
archive offsets, stream indices, byte sizes, page counts and geometry totals only.

Usage:
    python -I tools/restofworld_model_join.py <archive.rpf> <out.json|out.csv>
        [--start N] [--limit BYTES] [--max-inflate BYTES] [--quiet]
    python -I tools/restofworld_model_join.py <carved-dir> <out.json|out.csv>
        [--max-inflate BYTES] [--quiet]
"""
import argparse
import csv
import json
import math
import mmap
import os
import re
import struct
import sys
import time
import zlib

ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
SYS_BASE = 0x50000000
GFX_BASE = 0x60000000
MAX_SYS = 32 * 1024 * 1024
MAX_GFX = 224 * 1024 * 1024
MAX_PAGES = 32
MAX_INFLATE = 512 << 20
DECOMPRESS_CHUNK = 1 << 20

COLUMNS = [
    "offset",
    "stream_index",
    "kind",
    "complete",
    "system_pages",
    "graphics_pages",
    "graphics_first_offset",
    "system_stored",
    "system_bytes",
    "graphics_stored",
    "graphics_bytes",
    "required_system_bytes",
    "required_graphics_bytes",
    "drawables",
    "primary_geometries",
    "primary_vertices",
    "primary_triangles",
    "all_lod_geometries",
    "all_lod_vertices",
    "all_lod_triangles",
    "error",
]


class JoinError(ValueError):
    """Raised when a candidate resource fails a structural or cross-page check."""


def _u16(data, at):
    if at < 0 or at + 2 > len(data):
        raise JoinError(f"truncated u16 at +0x{at:x}")
    return struct.unpack_from("<H", data, at)[0]


def _u32(data, at):
    if at < 0 or at + 4 > len(data):
        raise JoinError(f"truncated u32 at +0x{at:x}")
    return struct.unpack_from("<I", data, at)[0]


def _u64(data, at):
    if at < 0 or at + 8 > len(data):
        raise JoinError(f"truncated u64 at +0x{at:x}")
    return struct.unpack_from("<Q", data, at)[0]


def is_sys_ptr(value):
    return SYS_BASE <= value < SYS_BASE + MAX_SYS


def is_gfx_ptr(value):
    return GFX_BASE <= value < GFX_BASE + MAX_GFX


def _valid_bounds(data, at):
    if at < 0 or at + 28 > len(data):
        return False
    mn = struct.unpack_from("<3f", data, at)
    mx = struct.unpack_from("<3f", data, at + 16)
    if not all(math.isfinite(v) and abs(v) < 1e7 for v in mn + mx):
        return False
    return mn[0] <= mx[0] and mn[1] <= mx[1] and mn[2] <= mx[2]


def looks_like_drawable_header(data, at=0, embedded=False):
    if len(data) < at + 176 or _u64(data, at) != 0:
        return False
    pages = _u64(data, at + 8)
    shader = _u64(data, at + 16)
    high = _u64(data, at + 80)
    name = _u64(data, at + 168)
    if (pages != 0) if embedded else (not is_sys_ptr(pages)):
        return False
    if not is_sys_ptr(shader) or not is_sys_ptr(high):
        return False
    if not (is_sys_ptr(name) or (embedded and name == 0)):
        return False
    radius = struct.unpack_from("<f", data, at + 44)[0]
    return _valid_bounds(data, at + 48) and math.isfinite(radius) and 0.0 <= radius < 1e6


def looks_like_fragment_header(data):
    if len(data) < 176 or _u64(data, 0) != 0:
        return False
    for at in (8, 24, 168):
        if not is_sys_ptr(_u64(data, at)):
            return False
    for at in (16, 32, 40):
        ptr = _u64(data, at)
        if ptr != 0 and not is_sys_ptr(ptr):
            return False
    a, b, c, d = _u16(data, 48), _u16(data, 50), _u16(data, 52), _u16(data, 54)
    return 0 < a <= 4096 and b <= 4096 and c <= d <= 4096


def classify_dictionary_header(data):
    """Return 'drawable_dict', 'texture_dict', or None for a pgDictionary root."""
    if len(data) < 64 or _u64(data, 0) != 0:
        return None
    if not (is_sys_ptr(_u64(data, 8)) and is_sys_ptr(_u64(data, 32)) and is_sys_ptr(_u64(data, 48))):
        return None
    count = _u16(data, 56)
    if not (0 < count <= 4096 and count == _u16(data, 58) == _u16(data, 40) == _u16(data, 42)):
        return None
    values = _u64(data, 48) - SYS_BASE
    if values < 64 or values + count * 8 > len(data):
        return None
    first = _u64(data, values)
    if not is_sys_ptr(first):
        return None
    first_off = first - SYS_BASE
    if first_off >= 64 and first_off + 176 <= len(data) and looks_like_drawable_header(data, first_off, embedded=True):
        return "drawable_dict"
    if first_off >= 64 and first_off + 0x70 <= len(data):
        # Standalone texture dictionary: grcTexture entries have self-pointer +0x50 at +0x28 and tile mode 5 at +0x40.
        if _u64(data, first_off + 0x28) == SYS_BASE + first_off + 0x50 and data[first_off + 0x40] == 5:
            return "texture_dict"
    return "other_dict"


def classify_system_root(sys_bytes):
    """Classify an assembled system segment into a model kind or a non-model class."""
    if len(sys_bytes) < 64 or sys_bytes[:8] != b"\x00" * 8:
        return None
    if len(sys_bytes) >= 0x14 and sys_bytes[0x10:0x14] == b"0DRP":
        return "0drp"
    if not is_sys_ptr(_u64(sys_bytes, 8)):
        return None
    if looks_like_drawable_header(sys_bytes, 0, embedded=False):
        return "drawable"
    if looks_like_fragment_header(sys_bytes):
        return "fragment"
    dk = classify_dictionary_header(sys_bytes)
    if dk is not None:
        return dk
    return "other_sys"


def classify_graphics_head(data):
    """Coarse stream-start tag for a graphics stream."""
    if len(data) < 4:
        return "short"
    w0 = struct.unpack_from("<I", data, 0)[0]
    if w0 == 0x4042:
        return "0x4042"
    if w0 == 0x3063:
        return "0x3063"
    if w0 & 0x80000000:
        return "meshlet"
    return "other"


def parse_page_map(sys_bytes):
    """Parse the datResourceMap header pointed to by u64@+0x08."""
    if len(sys_bytes) < 16 or _u64(sys_bytes, 0) != 0:
        raise JoinError("missing zero-prefix resource header")
    p8 = _u64(sys_bytes, 8)
    if not is_sys_ptr(p8):
        raise JoinError("u64@+0x08 is not a system pointer")
    off = p8 - SYS_BASE
    if off + 16 > len(sys_bytes):
        raise JoinError("page-map header extends past system segment")
    w0, w4 = _u32(sys_bytes, off), _u32(sys_bytes, off + 4)
    sys_pages = sys_bytes[off + 8]
    gfx_pages = sys_bytes[off + 9]
    pad10 = _u16(sys_bytes, off + 10)
    pad12 = _u32(sys_bytes, off + 12)
    if not (1 <= sys_pages <= MAX_PAGES and 0 <= gfx_pages <= MAX_PAGES):
        raise JoinError(f"invalid page counts ({sys_pages}, {gfx_pages})")
    total = sys_pages + gfx_pages
    end = off + 16 + 8 * total
    if end > len(sys_bytes):
        raise JoinError("page-map slot array extends past system segment")
    slots = struct.unpack_from(f"<{total}Q", sys_bytes, off + 16)
    if w0 != 0 or w4 != 0 or pad10 != 0 or pad12 != 0 or any(slots):
        raise JoinError("non-zero reserved or runtime slot word in page-map header")
    return {
        "offset": off,
        "end": end,
        "system_pages": sys_pages,
        "graphics_pages": gfx_pages,
    }


class ModelSystemPages:
    """Track the existing model-map exception to the 64-byte scanned-stream cutoff.

    Only a map readable in the first, full-sized model page establishes a run. Callers supply their
    framing checks. Archive and indexed-folder scans check recorded boundaries; loose folders
    treat their ordered files as contiguous. No payload is retained.
    """

    def __init__(self):
        self.remaining = 0

    def reset(self):
        self.remaining = 0

    def minimum_size(self, adjacent):
        return 1 if adjacent and self.remaining == 1 else 64

    def accept(self, data, *, adjacent, root_framed):
        previous = self.remaining
        self.reset()
        if not data or len(data) % 65536:
            return
        if adjacent and previous:
            self.remaining = previous - 1
        elif root_framed and classify_system_root(data) in ("drawable", "drawable_dict", "fragment"):
            try:
                self.remaining = parse_page_map(data)["system_pages"] - 1
            except JoinError:
                pass  # The full join still checks maps that extend into another system page.


def _check_cstring(sys_bytes, ptr_at, skip=0):
    """Validate that a name pointer resolves to printable ASCII without returning the name."""
    p = _u64(sys_bytes, ptr_at)
    if not is_sys_ptr(p):
        raise JoinError(f"invalid string pointer at +0x{ptr_at:x}")
    start = (p - SYS_BASE) + skip
    if start < 0 or start >= len(sys_bytes):
        raise JoinError("string pointer outside system segment")
    window = sys_bytes[start:min(len(sys_bytes), start + 256)]
    nul = window.find(0)
    if not 0 < nul < 256:
        raise JoinError("unterminated or empty string in system segment")
    raw = window[:nul]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JoinError("non-UTF-8 bytes in string") from exc
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in text):
        raise JoinError("control character in string")
    return start + nul + 1


def _validate_meshlet_buffer(gfx_bytes, ib_off, ib_bytes, desc_clusters, desc_tris, desc_refs, desc_binding):
    """Validate a pointed-to meshlet group inside the joined graphics segment."""
    if ib_off < 0 or ib_off + ib_bytes > len(gfx_bytes) or ib_bytes < 32:
        raise JoinError("index buffer outside graphics segment")
    count_flags = _u32(gfx_bytes, ib_off)
    bone_palette = (count_flags & 0x40000000) != 0
    clusters = count_flags & (0xFFFF if bone_palette else 0x7FFFFFFF)
    tris = _u32(gfx_bytes, ib_off + 4)
    refs = _u32(gfx_bytes, ib_off + 8)
    palette_no_marker = (desc_binding & 0xFF) == 0 and (count_flags & 0xFFFF0000) == 0xC0000000
    if not (
        0 < clusters <= 2048
        and 0 < tris <= 65535
        and 0 < refs <= 65535
        and clusters == desc_clusters
        and tris == desc_tris
        and refs == desc_refs
        and (((desc_binding & 0x80000000) != 0) == bone_palette)
        and (not bone_palette or (desc_binding & 0xFF) == 0x80 or palette_no_marker)
    ):
        raise JoinError("meshlet group header disagrees with system descriptor")

    ref_size = 2 if (count_flags & 0x80000000) else 4
    tri_at = ib_off + 12 + clusters * 16
    refs_at = tri_at + tris * 4
    pal_at = refs_at + refs * ref_size
    end_at = pal_at + (clusters * 2 if bone_palette else 0)
    if end_at > ib_off + ib_bytes:
        raise JoinError("meshlet payload exceeds index buffer allocation")

    acc_refs = 0
    acc_tris = 0
    for c_idx in range(clusters):
        rec_at = ib_off + 12 + c_idx * 16
        packed = _u32(gfx_bytes, rec_at)
        offsets = _u32(gfx_bytes, rec_at + 4)
        c_verts = (packed & 0x7F) + 1
        c_tris = ((packed >> 7) & 0xFF) + 1
        v_start = offsets & 0xFFFF
        t_start = offsets >> 16
        if v_start != acc_refs or t_start != acc_tris:
            raise JoinError("meshlet running offsets do not match accumulated counts")
        if c_verts > refs - v_start or c_tris > tris - t_start:
            raise JoinError("meshlet counts exceed group totals")
        for t_word in struct.unpack_from(f"<{c_tris}I", gfx_bytes, tri_at + t_start * 4):
            if (t_word >> 21) != c_idx:
                raise JoinError("packed triangle cluster id mismatch")
            if (
                (t_word & 0x7F) >= c_verts
                or ((t_word >> 7) & 0x7F) >= c_verts
                or ((t_word >> 14) & 0x7F) >= c_verts
            ):
                raise JoinError("packed triangle local index out of range")
        acc_refs += c_verts
        acc_tris += c_tris

    if acc_refs != refs or acc_tris != tris:
        raise JoinError("meshlet sums do not match group header")

    if ref_size == 2:
        max_ref = max(struct.unpack_from(f"<{refs}H", gfx_bytes, refs_at))
    else:
        max_ref = max(struct.unpack_from(f"<{refs}I", gfx_bytes, refs_at))
    return max_ref, bone_palette


def inspect_model_resource(sys_bytes, gfx_bytes, gfx_page_sizes):
    """Validate a joined (system, graphics) model resource and return structural totals."""
    if len(sys_bytes) > MAX_SYS:
        raise JoinError("system segment exceeds limit")
    if len(gfx_bytes) < 1 or len(gfx_bytes) > MAX_GFX:
        raise JoinError("graphics segment empty or exceeds limit")

    pm = parse_page_map(sys_bytes)
    if pm["graphics_pages"] < 1:
        raise JoinError("model resource declares zero graphics pages")
    if len(gfx_page_sizes) != pm["graphics_pages"] or sum(gfx_page_sizes) != len(gfx_bytes):
        raise JoinError("graphics stream list disagrees with declared graphics_pages")

    kind = classify_system_root(sys_bytes)
    if kind not in ("drawable", "drawable_dict", "fragment"):
        raise JoinError(f"not a model resource ({kind})")

    req_sys = pm["end"]
    req_gfx = 0

    def s_off(ptr_at, size):
        nonlocal req_sys
        p = _u64(sys_bytes, ptr_at)
        if not is_sys_ptr(p):
            raise JoinError(f"expected system pointer at +0x{ptr_at:x}")
        at = p - SYS_BASE
        if at < 0 or at + size > len(sys_bytes):
            raise JoinError(f"system pointer target +0x{at:x}..+0x{at + size:x} outside system segment")
        if at + size > req_sys:
            req_sys = at + size
        return at

    def g_off(ptr_at, size):
        nonlocal req_gfx
        p = _u64(sys_bytes, ptr_at)
        if not is_gfx_ptr(p):
            raise JoinError(f"expected graphics pointer at +0x{ptr_at:x}")
        at = p - GFX_BASE
        if size <= 0 or at < 0 or at + size > len(gfx_bytes):
            raise JoinError(f"graphics pointer target +0x{at:x}..+0x{at + size:x} outside graphics segment")
        if at + size > req_gfx:
            req_gfx = at + size
        return at

    drawable_offsets = []
    if kind == "drawable_dict":
        req_sys = max(req_sys, 64)
        count = _u16(sys_bytes, 56)
        hashes = s_off(32, count * 4)
        ptrs = s_off(48, count * 8)
        if hashes < 64 or ptrs < 64:
            raise JoinError("dictionary tables overlap root header")
        seen_children = set()
        for i in range(count):
            child = s_off(ptrs + i * 8, 192)
            if child < 64 or child in seen_children or not looks_like_drawable_header(sys_bytes, child, embedded=True):
                raise JoinError("invalid embedded drawable header in dictionary")
            seen_children.add(child)
            req_sys = max(req_sys, _check_cstring(sys_bytes, child + 168))
            drawable_offsets.append(child)
    elif kind == "fragment":
        req_sys = max(req_sys, 176)
        req_sys = max(req_sys, _check_cstring(sys_bytes, 168))
        child = s_off(24, 192)
        if child == 0 or not looks_like_drawable_header(sys_bytes, child, embedded=True):
            raise JoinError("invalid fragment primary drawable header")
        drawable_offsets.append(child)
    else:
        req_sys = max(req_sys, 192)
        drawable_offsets.append(0)

    prim_geos = prim_verts = prim_tris = 0
    all_geos = all_verts = all_tris = 0
    bone_bound_drawables = 0
    total_meshes_checked = 0

    for d_off in drawable_offsets:
        req_sys = max(req_sys, d_off + 192)
        if _u64(sys_bytes, d_off + 168) != 0:
            req_sys = max(req_sys, _check_cstring(sys_bytes, d_off + 168))

        drawable_bone_bound = False
        seen_lods = set()
        for lod_idx, lod_field in enumerate((80, 88, 96, 104)):
            if _u64(sys_bytes, d_off + lod_field) == 0:
                if lod_idx == 0:
                    raise JoinError("drawable missing primary LOD (+0x50)")
                continue
            lod = s_off(d_off + lod_field, 16)
            if lod in seen_lods:
                continue
            seen_lods.add(lod)
            model_count = _u16(sys_bytes, lod + 8)
            if not (0 < model_count <= 256 and model_count == _u16(sys_bytes, lod + 10)):
                raise JoinError("invalid LOD model array count")
            model_ptrs = s_off(lod, model_count * 8)
            seen_models = set()
            seen_geos = set()

            for mi in range(model_count):
                model = s_off(model_ptrs + mi * 8, 56)
                if model in seen_models:
                    raise JoinError("duplicate model pointer in LOD")
                seen_models.add(model)
                geo_count = _u16(sys_bytes, model + 16)
                if not (
                    0 < geo_count <= 1024
                    and geo_count == _u16(sys_bytes, model + 18) == _u16(sys_bytes, model + 54)
                ):
                    raise JoinError("inconsistent model geometry count")
                bone_count = _u16(sys_bytes, model + 20)
                if bone_count > 4096:
                    raise JoinError("model bone count out of range")
                if lod_idx == 0 and bone_count > 0:
                    drawable_bone_bound = True
                geo_ptrs = s_off(model + 8, geo_count * 8)
                s_off(model + 24, (geo_count + (1 if geo_count > 1 else 0)) * 32)
                if _u64(sys_bytes, model + 40) != 0:
                    s_off(model + 40, bone_count * 4 if bone_count > 0 else 1)

                for gi in range(geo_count):
                    total_meshes_checked += 1
                    if total_meshes_checked > 4096:
                        raise JoinError("mesh count limit exceeded")
                    geo = s_off(geo_ptrs + gi * 8, 160)
                    if geo in seen_geos:
                        raise JoinError("duplicate geometry pointer in LOD")
                    seen_geos.add(geo)

                    vb = s_off(geo + 8, 80)
                    fvf = s_off(vb + 64, 320)
                    flags = _u32(sys_bytes, fvf + 0x138)
                    vcount = flags >> 10
                    pos_size = sys_bytes[fvf + 0xD0]
                    pos_fmt = sys_bytes[fvf + 0x104]
                    if not (
                        (flags & 1) != 0
                        and 0 < vcount <= 1_000_000
                        and (pos_size, pos_fmt) in ((8, 16), (12, 6))
                    ):
                        raise JoinError("unsupported FVF vertex count or position format")

                    vb_stride = _u16(sys_bytes, vb + 12)
                    vb_bytes = _u32(sys_bytes, vb + 8) * vb_stride
                    pos_off = _u32(sys_bytes, fvf)
                    if not (
                        0 < vb_stride <= 32
                        and vb_bytes > 0
                        and pos_off >= (128 if (pos_size, pos_fmt) == (8, 16) else 96)
                        and pos_off <= vb_bytes
                        and vcount * pos_size <= vb_bytes - pos_off
                    ):
                        raise JoinError("invalid vertex buffer span")
                    vb_gfx = g_off(vb + 24, vb_bytes)
                    if _u64(sys_bytes, vb + 40) != 0:
                        s_off(vb + 40, 32)
                    vb_head = _u32(gfx_bytes, vb_gfx)
                    expected_tag = 0x42 if (pos_size, pos_fmt) == (8, 16) else 0x63
                    if (vb_head & 0x7F) != expected_tag:
                        raise JoinError("vertex block tag disagrees with FVF position format")

                    desc = s_off(geo + 64, 16)
                    ib = s_off(desc, 64)
                    ib_stride = _u16(sys_bytes, ib + 12)
                    ib_bytes = _u32(sys_bytes, ib + 8) * ib_stride
                    if ib_stride != 4 or ib_bytes < 32:
                        raise JoinError("invalid index buffer stride or size")
                    ib_gfx = g_off(ib + 24, ib_bytes)
                    if _u64(sys_bytes, ib + 40) != 0:
                        s_off(ib + 40, 32)

                    d_clusters = _u16(sys_bytes, desc + 8)
                    d_tris = _u16(sys_bytes, desc + 10)
                    d_binding = _u32(sys_bytes, desc + 12)
                    d_refs = (d_binding >> 8) & 0x7FFFFF
                    max_ref, bone_pal = _validate_meshlet_buffer(
                        gfx_bytes, ib_gfx, ib_bytes, d_clusters, d_tris, d_refs, d_binding
                    )
                    if max_ref >= vcount:
                        raise JoinError("meshlet vertex reference exceeds FVF vertex count")
                    if lod_idx == 0 and bone_pal:
                        drawable_bone_bound = True

                    all_geos += 1
                    all_verts += vcount
                    all_tris += d_tris
                    if lod_idx == 0:
                        prim_geos += 1
                        prim_verts += vcount
                        prim_tris += d_tris

        # Shader group (+16): shader constant buffer (+48), embedded texture dictionary (+8), materials (+16/+24)
        sg = s_off(d_off + 16, 64)
        if _u64(sys_bytes, sg + 48) != 0:
            sdesc = s_off(sg + 48, 32)
            sbuf = s_off(sdesc, 64)
            sstride = _u16(sys_bytes, sbuf + 12)
            sbytes = _u32(sys_bytes, sbuf + 8) * sstride
            if not (0 < sstride <= 32 and sbytes > 0):
                raise JoinError("invalid shader buffer extent")
            g_off(sbuf + 24, sbytes)
        if _u64(sys_bytes, sg + 8) != 0:
            txd = s_off(sg + 8, 64)
            tcount = _u16(sys_bytes, txd + 56)
            if tcount != _u16(sys_bytes, txd + 58) or tcount > 4096:
                raise JoinError("invalid embedded texture dictionary count")
            if tcount > 0:
                s_off(txd + 32, tcount * 4)
                tarr = s_off(txd + 48, tcount * 8)
                for ti in range(tcount):
                    tex = s_off(tarr + ti * 8, 112)
                    eb = _u16(sys_bytes, tex + 0x0C)
                    tbytes = _u32(sys_bytes, tex + 0x08) * eb
                    if eb == 0 or tbytes <= 0:
                        raise JoinError("invalid embedded texture allocation")
                    g_off(tex + 0x18, tbytes)

        mat_count = _u16(sys_bytes, sg + 24)
        if mat_count > 0:
            if mat_count > 512 or mat_count != _u16(sys_bytes, sg + 26):
                raise JoinError("invalid material array count")
            mat_ptrs = s_off(sg + 16, mat_count * 8)
            for mi in range(mat_count):
                sh = s_off(mat_ptrs + mi * 8, 64)
                if _u32(sys_bytes, sh + 4) == 0x6D657461:
                    info = s_off(sh + 32, 8)
                    params = sys_bytes[info + 4]
                    texs = sys_bytes[info + 1]
                    s_off(sh + 32, 8 + params * 8)
                    if texs > 0:
                        tarr = s_off(sh + 16, texs * 8)
                        for tj in range(texs):
                            if _u64(sys_bytes, tarr + tj * 8) != 0:
                                tex_ref = s_off(tarr + tj * 8, 80)
                                if _u64(sys_bytes, tex_ref + 48) != 0:
                                    asset = s_off(tex_ref + 48, 16)
                                    if asset >= 16 and _u32(sys_bytes, asset - 16) == 6:
                                        req_sys = max(req_sys, asset + 16)

        if _u64(sys_bytes, d_off + 176) != 0:
            rend = s_off(d_off + 176, 80)
            for li in range(5):
                lst = rend + li * 16
                lcnt = _u16(sys_bytes, lst + 8)
                if lcnt != _u16(sys_bytes, lst + 10) or (li < 4 and lcnt > 4096):
                    raise JoinError("invalid renderer list count")
                if lcnt > 0:
                    s_off(lst, lcnt * (4 if li < 4 else 80))

        if drawable_bone_bound:
            bone_bound_drawables += 1

    if req_gfx <= 0:
        raise JoinError("no graphics buffers referenced by drawable")
    prev_gfx_bytes = sum(gfx_page_sizes[:-1])
    if req_gfx <= prev_gfx_bytes:
        raise JoinError("final declared graphics page is not reached by any buffer")

    return {
        "kind": kind,
        "page_map_offset": pm["offset"],
        "system_pages": pm["system_pages"],
        "graphics_pages": pm["graphics_pages"],
        "required_system_bytes": req_sys,
        "required_graphics_bytes": req_gfx,
        "drawables": len(drawable_offsets),
        "dictionary_children": len(drawable_offsets) if kind == "drawable_dict" else 0,
        "primary_geometries": prim_geos,
        "primary_vertices": prim_verts,
        "primary_triangles": prim_tris,
        "all_lod_geometries": all_geos,
        "all_lod_vertices": all_verts,
        "all_lod_triangles": all_tris,
        "bone_bound_drawables": bone_bound_drawables,
    }


def _inflate_exact(buf, offset, end, max_inflate, *, min_inflate=64):
    """Inflate one zlib stream at `offset` inside `buf[offset:end]`."""
    d = zlib.decompressobj()
    p = offset
    total = 0
    chunks = []
    try:
        while not d.eof and p < end:
            block = buf[p:min(p + DECOMPRESS_CHUNK, end)]
            if not block:
                break
            p += len(block)
            while block and not d.eof:
                chunk = d.decompress(block, DECOMPRESS_CHUNK)
                block = d.unconsumed_tail
                if chunk:
                    total += len(chunk)
                    if total > max_inflate:
                        return None, 0, True
                    chunks.append(chunk)
                if not block:
                    break
    except zlib.error:
        return None, 0, False
    if not d.eof or total < min_inflate:
        return None, 0, False
    used = p - offset - len(d.unused_data)
    if used <= 0:
        return None, 0, False
    return b"".join(chunks), used, False


def join_streams(streams, get_payload, require_archive_framing=True):
    """Join model system streams to graphics streams over an indexed stream list."""
    resources = []
    tex_dicts_skipped = 0
    non_model_skipped = 0
    joined_4240 = 0
    total_4240 = sum(1 for s in streams if s["kind"] == "zlib" and s["head_tag"] == "0x4042")

    idx = 0
    n = len(streams)
    while idx < n:
        st = streams[idx]
        if st["kind"] != "zlib" or not st["zero_prefix"] or st["is_0drp"] or not is_sys_ptr(st["u64_at_8"]):
            idx += 1
            continue
        if require_archive_framing and not (st["align4k_plus16"] and st["pre16_zero"]):
            idx += 1
            continue

        first_data = get_payload(idx)
        root_hint = classify_system_root(first_data)
        # Assemble system continuation streams if u64@+0x08 points into a subsequent 64 KiB block.
        p8_off = st["u64_at_8"] - SYS_BASE
        sys_chunks = [first_data]
        cur = idx
        framing_err = None
        while p8_off + 16 > sum(len(c) for c in sys_chunks):
            if len(sys_chunks) >= MAX_PAGES:
                framing_err = "system page count exceeds limit before page-map header"
                break
            if (len(sys_chunks[-1]) & 0xFFFF) != 0:
                framing_err = "non-64KiB system page before continuation"
                break
            cur += 1
            if cur >= n or streams[cur]["kind"] != "zlib":
                framing_err = "missing system continuation stream"
                break
            if require_archive_framing and (not streams[cur]["gap_zero"] or streams[cur]["gap_len"] >= 16):
                framing_err = "non-zero or wide gap inside system continuation"
                break
            sys_chunks.append(get_payload(cur))

        if framing_err is not None:
            if root_hint in ("drawable", "drawable_dict", "fragment"):
                resources.append(_failed_row(st, root_hint, framing_err))
            idx += 1
            continue

        sys_probe = b"".join(sys_chunks)
        try:
            pm = parse_page_map(sys_probe)
        except JoinError as exc:
            kind_guess = classify_system_root(sys_probe)
            if kind_guess in ("drawable", "drawable_dict", "fragment"):
                resources.append(_failed_row(st, kind_guess, str(exc)))
            idx += 1
            continue

        sp = pm["system_pages"]
        gp = pm["graphics_pages"]
        if sp < len(sys_chunks) or idx + sp > n:
            kind_guess = classify_system_root(sys_probe)
            if kind_guess in ("drawable", "drawable_dict", "fragment"):
                resources.append(_failed_row(st, kind_guess, "declared system_pages truncated at archive end"))
            idx += 1
            continue

        # Append any remaining declared system pages beyond sys_chunks.
        for extra_idx in range(idx + len(sys_chunks), idx + sp):
            if (len(sys_chunks[-1]) & 0xFFFF) != 0:
                framing_err = "non-64KiB non-final system stream"
                break
            if streams[extra_idx]["kind"] != "zlib":
                framing_err = "non-zlib stream inside declared system_pages"
                break
            if require_archive_framing and (not streams[extra_idx]["gap_zero"] or streams[extra_idx]["gap_len"] >= 16):
                framing_err = "non-zero or wide gap inside declared system_pages"
                break
            sys_chunks.append(get_payload(extra_idx))

        sys_bytes = b"".join(sys_chunks)
        rkind = classify_system_root(sys_bytes)
        if rkind not in ("drawable", "drawable_dict", "fragment"):
            if framing_err is None and idx + sp + gp <= n and all(
                streams[idx + k]["kind"] == "zlib"
                and (not require_archive_framing or (streams[idx + k]["gap_zero"] and streams[idx + k]["gap_len"] < 16))
                for k in range(1, sp + gp)
            ):
                if rkind == "texture_dict":
                    tex_dicts_skipped += 1
                else:
                    non_model_skipped += 1
                idx += sp + gp
            else:
                idx += 1
            continue

        if framing_err is not None:
            resources.append(_failed_row(st, rkind, framing_err, sp, gp))
            idx += 1
            continue

        if gp < 1 or idx + sp + gp > n:
            resources.append(_failed_row(st, rkind, "missing declared graphics streams", sp, gp))
            idx += 1
            continue

        gfx_meta = streams[idx + sp:idx + sp + gp]
        gfx_chunks = []
        for gi, gst in enumerate(gfx_meta):
            if gst["kind"] != "zlib":
                framing_err = "non-zlib stream inside declared graphics_pages"
                break
            if require_archive_framing and (not gst["gap_zero"] or gst["gap_len"] >= 16):
                framing_err = "non-zero or wide gap before graphics stream"
                break
            gdata = get_payload(idx + sp + gi)
            if gi + 1 < gp and (len(gdata) & 0xFFFF) != 0:
                framing_err = "non-64KiB non-final graphics stream"
                break
            gfx_chunks.append(gdata)

        if framing_err is not None:
            resources.append(_failed_row(st, rkind, framing_err, sp, gp))
            idx += 1
            continue

        gfx_bytes = b"".join(gfx_chunks)
        gfx_sizes = [len(c) for c in gfx_chunks]
        try:
            checked = inspect_model_resource(sys_bytes, gfx_bytes, gfx_sizes)
        except JoinError as exc:
            resources.append(_failed_row(st, rkind, str(exc), sp, gp))
            idx += 1
            continue

        sys_meta = streams[idx:idx + sp]
        joined_4240 += sum(1 for gst in gfx_meta if gst["head_tag"] == "0x4042")
        resources.append({
            "offset": st["offset"],
            "stream_index": st["stream_index"],
            "kind": checked["kind"],
            "complete": True,
            "error": None,
            "page_map_offset": checked["page_map_offset"],
            "system_pages": sp,
            "graphics_pages": gp,
            "system_streams": [
                {
                    "stream_index": s["stream_index"],
                    "offset": s["offset"],
                    "stored": s["stored"],
                    "size": s["size"],
                }
                for s in sys_meta
            ],
            "graphics_streams": [
                {
                    "stream_index": s["stream_index"],
                    "offset": s["offset"],
                    "stored": s["stored"],
                    "size": s["size"],
                    "head": s["head_tag"],
                }
                for s in gfx_meta
            ],
            "system_stored": sum(s["stored"] for s in sys_meta),
            "system_bytes": len(sys_bytes),
            "graphics_stored": sum(s["stored"] for s in gfx_meta),
            "graphics_bytes": len(gfx_bytes),
            "required_system_bytes": checked["required_system_bytes"],
            "required_graphics_bytes": checked["required_graphics_bytes"],
            "drawables": checked["drawables"],
            "dictionary_children": checked["dictionary_children"],
            "primary_geometries": checked["primary_geometries"],
            "primary_vertices": checked["primary_vertices"],
            "primary_triangles": checked["primary_triangles"],
            "all_lod_geometries": checked["all_lod_geometries"],
            "all_lod_vertices": checked["all_lod_vertices"],
            "all_lod_triangles": checked["all_lod_triangles"],
            "bone_bound_drawables": checked["bone_bound_drawables"],
        })
        idx += sp + gp

    stats = {
        "total_4240_starts": total_4240,
        "joined_4240_starts": joined_4240,
        "texture_dictionaries_skipped": tex_dicts_skipped,
        "non_model_system_resources_skipped": non_model_skipped,
    }
    return resources, stats


def _failed_row(st, kind, error, sp=0, gp=0):
    return {
        "offset": st["offset"],
        "stream_index": st["stream_index"],
        "kind": kind,
        "complete": False,
        "error": error,
        "page_map_offset": (st["u64_at_8"] - SYS_BASE) if is_sys_ptr(st["u64_at_8"]) else None,
        "system_pages": sp,
        "graphics_pages": gp,
        "system_streams": [
            {
                "stream_index": st["stream_index"],
                "offset": st["offset"],
                "stored": st["stored"],
                "size": st["size"],
            }
        ],
        "graphics_streams": [],
        "system_stored": st["stored"],
        "system_bytes": st["size"],
        "graphics_stored": 0,
        "graphics_bytes": 0,
        "required_system_bytes": 0,
        "required_graphics_bytes": 0,
        "drawables": 0,
        "dictionary_children": 0,
        "primary_geometries": 0,
        "primary_vertices": 0,
        "primary_triangles": 0,
        "all_lod_geometries": 0,
        "all_lod_vertices": 0,
        "all_lod_triangles": 0,
        "bone_bound_drawables": 0,
    }


def _summarise(resources, streams, stats, mode, source, skipped, elapsed):
    joined = [r for r in resources if r["complete"]]
    by_kind = {}
    page_shapes = {}
    for r in joined:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        shape = f"{r['system_pages']}+{r['graphics_pages']}"
        page_shapes[shape] = page_shapes.get(shape, 0) + 1
    return {
        "mode": mode,
        "source": source,
        "streams": len(streams),
        "zlib_streams": sum(1 for s in streams if s["kind"] == "zlib"),
        "bink2_files": sum(1 for s in streams if s["kind"] == "bink2"),
        "skipped_candidates": skipped,
        "model_candidates": len(resources),
        "joined_resources": len(joined),
        "unjoined_candidates": len(resources) - len(joined),
        "by_kind": dict(sorted(by_kind.items())),
        "page_shapes": dict(sorted(page_shapes.items())),
        "model_system_streams": sum(r["system_pages"] for r in joined),
        "model_graphics_streams": sum(r["graphics_pages"] for r in joined),
        "graphics_4240_starts_total": stats["total_4240_starts"],
        "graphics_4240_starts_joined": stats["joined_4240_starts"],
        "drawables": sum(r["drawables"] for r in joined),
        "dictionary_children": sum(r["dictionary_children"] for r in joined),
        "standalone_models": sum(1 for r in joined if r["kind"] in ("drawable", "fragment")),
        "fragments": sum(1 for r in joined if r["kind"] == "fragment"),
        "bone_bound_drawables": sum(r["bone_bound_drawables"] for r in joined),
        "primary_geometries": sum(r["primary_geometries"] for r in joined),
        "primary_vertices": sum(r["primary_vertices"] for r in joined),
        "primary_triangles": sum(r["primary_triangles"] for r in joined),
        "all_lod_geometries": sum(r["all_lod_geometries"] for r in joined),
        "all_lod_vertices": sum(r["all_lod_vertices"] for r in joined),
        "all_lod_triangles": sum(r["all_lod_triangles"] for r in joined),
        "texture_dictionaries_skipped": stats["texture_dictionaries_skipped"],
        "non_model_system_resources_skipped": stats["non_model_system_resources_skipped"],
        "elapsed_seconds": round(elapsed, 3),
    }


def scan_rpf(path, start=None, limit=None, max_inflate=MAX_INFLATE):
    """Walk an RPF9 archive data area and join model system pages to graphics pages."""
    t0 = time.time()
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise ValueError("empty file")
    if file_size < 8:
        raise ValueError("too short for an RPF9 header")
    streams = []
    skipped = 0
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if m[:4] not in (b"9FPR", b"RPF9"):
                raise ValueError("not an RPF9 archive")
            first = struct.unpack_from("<I", m, 4)[0]
            if first > file_size:
                raise ValueError("RPF9 data offset is outside the archive")
            pos = start if start is not None else first
            if not 0 <= pos <= file_size:
                raise ValueError("start offset is outside the archive")
            end = min(file_size, pos + limit) if limit is not None else file_size
            last = pos
            system_pages = ModelSystemPages()
            while pos < end:
                found = ZHDR.search(m, pos, end)
                if not found:
                    break
                s = found.start()
                if m[s:s + 3] == b"KB2":
                    if s + 0x1C > end:
                        pos = s + 1
                        continue
                    blen = struct.unpack_from("<I", m, s + 4)[0] + 8
                    w, h = struct.unpack_from("<II", m, s + 0x14)
                    if not (64 <= blen <= end - s and 16 <= w <= 8192 and 16 <= h <= 8192):
                        pos = s + 1
                        continue
                    gap = m[last:s]
                    streams.append({
                        "stream_index": len(streams),
                        "offset": s,
                        "stored": blen,
                        "size": blen,
                        "kind": "bink2",
                        "gap_len": len(gap),
                        "gap_zero": gap.count(0) == len(gap),
                        "align4k_plus16": (s & 0xFFF) == 0x10,
                        "pre16_zero": s >= 16 and m[s - 16:s] == b"\x00" * 16,
                        "zero_prefix": False,
                        "is_0drp": False,
                        "u64_at_8": 0,
                        "head_tag": "bink2",
                    })
                    last = pos = s + blen
                    system_pages.reset()
                    continue

                adjacent = s == ((last + 15) & ~15) and not any(m[last:s])
                # A declared final system page can be smaller than the minimum for a scanned stream.
                data, used, too_big = _inflate_exact(
                    m, s, end, max_inflate, min_inflate=system_pages.minimum_size(adjacent))
                if data is None:
                    if too_big:
                        skipped += 1
                    pos = s + 1
                    continue
                gap = m[last:s]
                meta = {
                    "stream_index": len(streams),
                    "offset": s,
                    "stored": used,
                    "size": len(data),
                    "kind": "zlib",
                    "gap_len": len(gap),
                    "gap_zero": gap.count(0) == len(gap),
                    "align4k_plus16": (s & 0xFFF) == 0x10,
                    "pre16_zero": s >= 16 and m[s - 16:s] == b"\x00" * 16,
                    "zero_prefix": data[:8] == b"\x00" * 8,
                    "is_0drp": len(data) >= 0x14 and data[0x10:0x14] == b"0DRP",
                    "u64_at_8": struct.unpack_from("<Q", data, 8)[0] if len(data) >= 16 else 0,
                    "head_tag": classify_graphics_head(data),
                }
                streams.append(meta)
                system_pages.accept(data, adjacent=adjacent,
                                    root_framed=meta["align4k_plus16"] and meta["pre16_zero"])
                last = pos = s + used

            cache = {}

            def get_payload(s_idx):
                if s_idx not in cache:
                    if len(cache) >= 64:
                        cache.clear()
                    meta = streams[s_idx]
                    payload, _, _ = _inflate_exact(m, meta["offset"], meta["offset"] + meta["stored"], max_inflate,
                                                   min_inflate=meta["size"])
                    cache[s_idx] = payload
                return cache[s_idx]

            resources, stats = join_streams(streams, get_payload, require_archive_framing=True)

    summary = _summarise(resources, streams, stats, "rpf9", path, skipped, time.time() - t0)
    return resources, summary


def _carve_rows(path):
    """Read physical stream boundaries in the order recorded by rpf9_carve."""
    try:
        fh = open(os.path.join(path, "index.json"), encoding="utf-8")
    except FileNotFoundError:
        return None
    with fh:
        index = json.load(fh)
    rows = index.get("files") if isinstance(index, dict) else None
    if not isinstance(rows, list):
        raise ValueError("index.json requires a files array")
    result = {}
    last_end = 0
    for row in rows:
        if not isinstance(row, dict) or any(type(row.get(k)) is not int for k in ("off", "size", "stored")):
            raise ValueError("invalid stream dimensions in index.json")
        off, size, stored = row["off"], row["size"], row["stored"]
        if off < last_end or min(size, stored) <= 0:
            raise ValueError("overlapping or invalid stream in index.json")
        last_end = off + stored
        bink = row.get("type") == "bink2"
        pre = row.get("pre")
        if not bink and (not isinstance(pre, str) or
                         len(pre) != 2 * min(16, off) or not re.fullmatch(r"[0-9a-fA-F]*", pre)):
            raise ValueError("invalid predecessor bytes in index.json")
        name = f"{off:012x}." + ("bk2" if bink else "bin")
        result[name] = (off, size, stored, b"" if bink else bytes.fromhex(pre))
    return result


def scan_dir(path, max_inflate=MAX_INFLATE):
    """Join already-carved `.bin` / `.bk2` stream files in offset order."""
    t0 = time.time()
    streams = []
    files = []
    skipped = 0
    system_pages = ModelSystemPages()
    rows = _carve_rows(path)
    previous_end = 0
    names = rows if rows is not None else sorted(os.listdir(path))
    for name in names:
        full = os.path.join(path, name)
        if rows is None and (not os.path.isfile(full) or not name.endswith((".bin", ".bk2"))):
            continue
        fsize = os.path.getsize(full)
        if rows is None:
            match = re.match(r"([0-9a-fA-F]{6,})", name)
            off = int(match.group(1), 16) if match else len(streams)
            stored, gap, gap_zero, pre16_zero = fsize, 0, True, True
        else:
            off, size, stored, pre = rows[name]
            if size != fsize:
                raise ValueError(f"{name}: size differs from index.json")
            gap = off - previous_end
            gap_zero = gap == 0 or (gap <= len(pre) and not any(pre[-gap:]))
            pre16_zero = pre == bytes(16)
        adjacent = rows is None or (off == ((previous_end + 15) & ~15) and gap_zero)
        root_framed = rows is None or ((off & 0xFFF) == 0x10 and pre16_zero)
        if name.endswith(".bin") and fsize > max_inflate:
            skipped += 1
            system_pages.reset()
            continue
        minimum = 64 if name.endswith(".bk2") else system_pages.minimum_size(adjacent)
        if fsize < minimum:
            system_pages.reset()
            continue
        previous_end = off + stored
        with open(full, "rb") as fh:
            head = fh.read(32)
            # The carver writes raw video as .bk2; a compressed .bin payload can also start with KB2.
            is_bink = name.endswith(".bk2") and head[:3] == b"KB2"
            if fsize % 65536 == 0 and not is_bink:
                # Inspect the model header and map without copying a full page just for this metadata pass.
                with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as data:
                    system_pages.accept(data, adjacent=adjacent, root_framed=root_framed)
            else:
                system_pages.reset()
        if fsize >= 64 and is_bink and len(head) >= 0x1C:
            streams.append({
                "stream_index": len(streams),
                "offset": off,
                "stored": stored,
                "size": fsize,
                "kind": "bink2",
                "gap_len": gap,
                "gap_zero": gap_zero,
                "align4k_plus16": (off & 0xFFF) == 0x10,
                "pre16_zero": pre16_zero,
                "zero_prefix": False,
                "is_0drp": False,
                "u64_at_8": 0,
                "head_tag": "bink2",
            })
            files.append(None)
            continue
        streams.append({
            "stream_index": len(streams),
            "offset": off,
            "stored": stored,
            "size": fsize,
            "kind": "zlib",
            "gap_len": gap,
            "gap_zero": gap_zero,
            "align4k_plus16": (off & 0xFFF) == 0x10,
            "pre16_zero": pre16_zero,
            "zero_prefix": head[:8] == b"\x00" * 8,
            "is_0drp": len(head) >= 0x14 and head[0x10:0x14] == b"0DRP",
            "u64_at_8": struct.unpack_from("<Q", head, 8)[0] if len(head) >= 16 else 0,
            "head_tag": classify_graphics_head(head),
        })
        files.append(full)

    cache = {}

    def get_payload(s_idx):
        if s_idx not in cache:
            if len(cache) >= 64:
                cache.clear()
            with open(files[s_idx], "rb") as fh:
                cache[s_idx] = fh.read()
        return cache[s_idx]

    resources, stats = join_streams(streams, get_payload, require_archive_framing=rows is not None)
    summary = _summarise(resources, streams, stats, "directory", path, skipped, time.time() - t0)
    return resources, summary


def write_output(path, resources, summary):
    """Write JSON or CSV based on the output path extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summary": summary, "resources": resources}, fh, indent=1)
            fh.write("\n")
        return
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for r in resources:
            gfx_first = (
                f"0x{r['graphics_streams'][0]['offset']:x}" if r["graphics_streams"] else ""
            )
            writer.writerow({
                "offset": f"0x{r['offset']:x}",
                "stream_index": r["stream_index"],
                "kind": r["kind"],
                "complete": int(r["complete"]),
                "system_pages": r["system_pages"],
                "graphics_pages": r["graphics_pages"],
                "graphics_first_offset": gfx_first,
                "system_stored": r["system_stored"],
                "system_bytes": r["system_bytes"],
                "graphics_stored": r["graphics_stored"],
                "graphics_bytes": r["graphics_bytes"],
                "required_system_bytes": r["required_system_bytes"],
                "required_graphics_bytes": r["required_graphics_bytes"],
                "drawables": r["drawables"],
                "primary_geometries": r["primary_geometries"],
                "primary_vertices": r["primary_vertices"],
                "primary_triangles": r["primary_triangles"],
                "all_lod_geometries": r["all_lod_geometries"],
                "all_lod_vertices": r["all_lod_vertices"],
                "all_lod_triangles": r["all_lod_triangles"],
                "error": "" if r["error"] is None else r["error"],
            })


def offset_arg(text):
    """--start: 0 or more, decimal or 0x hex."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 0 or more") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def byte_count_arg(text):
    """--limit/--max-inflate: at least 1."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _aliases_input(out_path, source_path):
    if os.path.normcase(os.path.realpath(out_path)) == os.path.normcase(os.path.realpath(source_path)):
        return True
    try:
        if os.path.samefile(out_path, source_path):
            return True
    except OSError:
        pass
    if os.path.isdir(source_path):
        index_path = os.path.join(source_path, "index.json")
        if os.path.normcase(os.path.realpath(out_path)) == os.path.normcase(os.path.realpath(index_path)):
            return True
        if not os.path.isfile(out_path):
            return False
        if os.path.isfile(index_path) and os.path.samefile(out_path, index_path):
            return True
        src_root = os.path.normcase(os.path.abspath(source_path))
        target = os.path.normcase(os.path.abspath(out_path))
        if os.path.dirname(target) == src_root and target.endswith((".bin", ".bk2")):
            return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive or a directory of carved .bin/.bk2 files")
    ap.add_argument("out", help="output path ending in .json or .csv")
    ap.add_argument("--start", type=offset_arg, help="first byte to scan (default: RPF data start)")
    ap.add_argument("--limit", type=byte_count_arg, help="number of bytes to scan")
    ap.add_argument("--max-inflate", type=byte_count_arg, default=MAX_INFLATE,
                    help="skip a stream that inflates above this many bytes (default: 512 MiB)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if os.path.isdir(args.source) and (args.start is not None or args.limit is not None):
        ap.error("--start and --limit apply to a single archive, not a directory")
    if os.path.isdir(args.out):
        ap.error(f"{args.out} is a folder, not a .json or .csv file")
    if _aliases_input(args.out, args.source):
        ap.error("output must not overwrite the source")
    if os.path.splitext(args.out)[1].lower() not in (".json", ".csv"):
        ap.error("output path must end in .json or .csv")
    folder = os.path.dirname(os.path.abspath(args.out))
    if not os.path.isdir(folder):
        ap.error(f"{folder} is not a folder, so {args.out} cannot be written")

    try:
        if os.path.isdir(args.source):
            resources, summary = scan_dir(args.source, args.max_inflate)
        else:
            resources, summary = scan_rpf(args.source, args.start, args.limit, args.max_inflate)
    except ValueError as exc:
        print(f"{args.source}: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"{args.source}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    try:
        write_output(args.out, resources, summary)
    except OSError as exc:
        print(f"{args.out}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    if not args.quiet:
        print(json.dumps(summary, indent=1))
    return 1 if summary["unjoined_candidates"] else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
