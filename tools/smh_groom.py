"""Clean-room decoder, strand generator, census tool and GLB exporter for RAGE "smh" strand grooms.

Hair, beards, brows, lashes, body hair and animal fur in this build are stored as GPU strand groom
LOD resources rather than static triangle meshes. Each groom LOD resource is a native RAGE resource
containing a small system page (type ID ``0x7d8f3db3``) and one GPU asset buffer headed by a table
of byte offsets into per-guide, per-node and per-triangle arrays.

Strands are not stored directly in the asset: the ``smh_gen_*_CS_InterpolateStrands`` compute shaders
scatter strand roots across the scalp triangle mesh (whose vertices are the guide roots) using a
deterministic integer RNG, then blend three guides per strand and trim to a per-strand length.
Class 0 (``smh_gen_1``) and class 1 (``smh_gen_2``/``_4``/``_8``/``_16``) are two placement sets of
one groom generated with the same points-per-strand (per-instance ``+0x54`` bits 16-23, 16 bytes/point)
and drawn together as ``(count0 + count1) * (points - 1)`` segments in one batch; neither class is
"more detailed".

Pure Python standard library only (runs under ``python -I``, no third-party packages).

Usage:
    python -I tools/smh_groom.py inspect <file.smh|archive.rpf> [--offset 0x...] [--json]
    python -I tools/smh_groom.py census <path> [<path> ...] [--json] [--limit N]
    python -I tools/smh_groom.py export-glb <file.smh|archive.rpf> <out.glb> [--offset 0x...] [--geometry tubes|ribbons|curves|line-strip] [--strands N]
"""
from __future__ import annotations

import argparse
import array
import collections
import json
import math
import mmap
import os
import struct
import sys
import zlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glb_writer  # noqa: E402

SYS_BASE = 0x50000000
GPU_BASE = 0x60000000
TYPE_ID = 0x7D8F3DB3
PREFIX = b"\x00" * 16 + b"\x78"
HEADER_SIZE = 0xA0
MAX_INFLATE_BYTES = 64 << 20

# System page +0xb8 metadata hashes and +0x48..+0x70 child-rig float sentinels.
SKEL_HEAD_HASH = 0x34C53E77
UNUSED_HASH_FILL = 0x89EA2ED6
FLT_MAX_U32 = 0x7F7FFFFF
NEG_FLT_MAX_U32 = 0xFF7FFFFF

# Measured +0xcc (slot 5 of system page +0xb8) part class hashes across the corpus (30 distinct total).
PART_CLASS_NAMES: dict[int, str] = {
    0xD54A2D40: "main_hair",
    0x27B342A3: "short_hair",
    0xE61E1493: "long_hair",
    0x1CAEB442: "fuzz",
    0xE2500E92: "braids_ties",
    0x44D2709F: "beard",
    0x42C7E1D0: "beard",
    0x83A664E7: "lashes",
    0xA228CC57: "body_hair",
    0x304DD218: "animal_fur",
    0xC50E7D12: "animal_fur",
}

MODE_LINES = 1
MODE_LINE_STRIP = 3
MODE_TRIANGLES = glb_writer.MODE_TRIANGLES

# GPU asset buffer header: byte offset -> (name, bytes per element, domain).
# Domains: "G" = per guide, "N" = per node (guide-major), "T" = per triangle.
ARRAYS: dict[int, tuple[str, int, str]] = {
    0x00: ("attr_a", 4, "G"),
    0x04: ("attr_b", 4, "G"),
    0x08: ("uv", 4, "G"),
    0x0C: ("attr_c", 4, "G"),
    0x10: ("bone_index", 8, "G"),
    0x14: ("bone_weight", 8, "G"),
    0x18: ("ext_18", 12, "G"),
    0x1C: ("ext_1c", 8, "G"),
    0x24: ("ext_24", 4, "G"),
    0x28: ("direction", 4, "G"),
    0x30: ("length", 2, "G"),
    0x34: ("node_4", 4, "N"),
    0x38: ("node_2", 2, "N"),
    0x40: ("triangle", 6, "T"),
    0x44: ("position", 8, "N"),
    0x48: ("tri_count", 4, "T"),
    0x4C: ("tri_region", 1, "T"),
}
REQUIRED_ARRAYS = ("position", "triangle", "tri_count", "length", "bone_index", "bone_weight", "uv")
OPTIONAL_RAW_ARRAYS = (
    ("attr_a", 4),
    ("attr_b", 4),
    ("attr_c", 4),
    ("ext_18", 12),
    ("ext_1c", 8),
    ("ext_24", 4),
    ("direction", 4),
    ("node_4", 4),
    ("node_2", 2),
)
TAIL_OFFSET = 0x20
SENTINEL_OFFSET = 0x2C
EXT_3C_OFFSET = 0x3C
STRAND_CLASS_SEED = (0x7C, 0x80)

PART_NAMES = ("Scalp triangles", "Guides", "Strands (class 0)", "Strands (class 1)")
PART_COLORS = (
    (0.55, 0.55, 0.55, 1.0),
    (0.85, 0.20, 0.20, 1.0),
    (0.32, 0.22, 0.14, 1.0),
    (0.76, 0.64, 0.42, 1.0),
)
GEOMETRY_MODES = ("tubes", "ribbons", "curves", "line-strip")

_M32 = 0xFFFFFFFF
_M24 = 0xFFFFFF
_INV_2_24 = 2.0 ** -24
_ATTEMPT_CONSTS = tuple(
    (
        (attempt * 0x19660D + 0x86945BED) & _M32,
        (((attempt * 0x19660D + 0x86945BED) & _M32) * 0x153E39CE) & _M32,
    )
    for attempt in range(4)
)
_NEEDS_BYTESWAP = sys.byteorder != "little"
_UINT16_CODE = "H" if array.array("H").itemsize == 2 else None
_UINT32_CODE = "I" if array.array("I").itemsize == 4 else ("L" if array.array("L").itemsize == 4 else None)
_FLOAT32_CODE = "f" if array.array("f").itemsize == 4 else None


class GroomError(ValueError):
    """Raised when a groom resource, buffer header or export request is invalid."""


def _f32(value: float) -> float:
    """Round a Python float to IEEE 754 binary32 precision."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _clean_zero(value: float) -> float:
    return 0.0 if value == 0.0 else value


def inflate_stream(src: bytes | bytearray | memoryview | mmap.mmap, pos: int) -> tuple[bytes, int]:
    """Inflate one zlib stream at ``pos`` -> ``(raw_bytes, next_16_aligned_offset)``."""
    decoder = zlib.decompressobj()
    out = bytearray()
    size = len(src)
    while not decoder.eof:
        if pos >= size:
            raise GroomError("incomplete zlib resource stream")
        chunk = src[pos:pos + 65536]
        if not chunk:
            raise GroomError("incomplete zlib resource stream")
        try:
            piece = decoder.decompress(chunk)
        except zlib.error as exc:
            raise GroomError(f"zlib decompression failed at {pos:#x}: {exc}") from exc
        out.extend(piece)
        if len(out) > MAX_INFLATE_BYTES:
            raise GroomError(f"resource stream exceeds {MAX_INFLATE_BYTES} bytes")
        pos += len(chunk) - len(decoder.unused_data)
    return bytes(out), (pos + 15) & -16


def _sys_ptr(sysd: bytes, at: int) -> int:
    value = struct.unpack_from("<Q", sysd, at)[0] - SYS_BASE
    if not (0 <= value < len(sysd)):
        raise GroomError(f"system pointer at {at:#x} points outside the {len(sysd)}-byte page")
    return value


def system_page_is_groom(sysd: bytes | bytearray | memoryview) -> bool:
    """Return True if ``sysd`` starts like an inflated smh groom LOD system page."""
    if len(sysd) < 0xE0 or bytes(sysd[:8]) != b"\x00" * 8:
        return False
    if struct.unpack_from("<I", sysd, 0x1C)[0] != TYPE_ID:
        return False
    desc = struct.unpack_from("<Q", sysd, 0x70)[0] - SYS_BASE
    if not (0 < desc <= len(sysd) - 0x20):
        return False
    words, stride, flag = struct.unpack_from("<IHH", sysd, desc + 8)
    return (
        stride == 4
        and flag == 8
        and struct.unpack_from("<Q", sysd, desc + 0x18)[0] == GPU_BASE
        and words >= 40
    )


def scan_offsets(
    src: bytes | bytearray | memoryview | mmap.mmap,
    start: int = 0,
    stop: int | None = None,
    chunk: int = 64 << 20,
) -> Iterator[int]:
    """Yield the byte offset of every groom LOD resource in ``src`` (16-aligned zero prefix + zlib page)."""
    if isinstance(chunk, bool) or chunk <= 0:
        raise GroomError(f"scan chunk size must be positive, got {chunk}")
    if isinstance(src, memoryview):
        if src.nbytes == 0:
            return
        if src.c_contiguous and 0 not in src.strides:
            src = src.cast("B")
        elif not (src.ndim == 1 and src.itemsize == 1):
            src = src.tobytes()
    size = len(src)
    stop = size if stop is None else min(stop, size)
    first = (max(start, 0) + 15) & -16
    overlap = len(PREFIX) - 1
    for base in range(first, stop, chunk):
        window_end = min(base + chunk, stop)
        limit = window_end - base
        end = min(window_end + overlap, size)
        block = src[base:end]
        if isinstance(block, memoryview):
            block = block.tobytes()
        at = -1
        while True:
            at = block.find(PREFIX, at + 1)
            if at < 0 or at >= limit:
                break
            off = base + at
            if off & 15:
                continue
            comp = src[off + 16:min(off + 16 + 4096, size)]
            if isinstance(comp, memoryview) and not comp.c_contiguous:
                comp = comp.tobytes()
            try:
                head = zlib.decompressobj().decompress(comp, 4096)
            except zlib.error:
                continue
            if system_page_is_groom(head):
                yield off


def _finite_f16(value: float) -> float | None:
    return _clean_zero(value) if math.isfinite(value) else None


def _round_opt(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def _round_style_params(params: dict | None) -> dict | None:
    if params is None:
        return None
    out: dict = {
        "strandWidth": _round_opt(params["strandWidth"]),
        "longestGuideM": _round_opt(params["longestGuideM"]),
        "longestGuideAltM": _round_opt(params["longestGuideAltM"]),
        "braidParam": _round_opt(params["braidParam"]),
    }
    for key in ("curlFrizz0", "curlFrizz1"):
        blk = params[key]
        out[key] = {
            "amounts": [_round_opt(v) for v in blk["amounts"]],
            "radiiM": [_round_opt(v) for v in blk["radiiM"]],
            "frequenciesPerM": [_round_opt(v) for v in blk["frequenciesPerM"]],
        }
    return out


def parse_system_page(system: bytes | bytearray | memoryview) -> dict:
    """Decode structured metadata from an inflated ``smh`` groom LOD system page.

    Decodes bounding box (``+0x30``/``+0x40``), child-rig unset float sentinel detection in
    ``+0x48..+0x70`` (secondary float4 bounds at ``+0x50..+0x6f`` set to ``+FLT_MAX``
    ``0x7f7fffff`` / ``-FLT_MAX`` ``0xff7fffff``), skinning bone hashes (``+0x98/+0xa0``),
    region records (``+0xa8/+0xb0``), the six ``+0xb8`` metadata hashes (slots 0..3 bone
    hashes, slot 4 ``+0xc8`` style preset hash, slot 5 ``+0xcc`` part class hash), strand
    totals (``+0xd0``), and the ``+0xe8..+0x110`` IEEE 754 ``float16`` style parameters
    (when ``len(system) >= 0x110``).
    """
    sys_bytes = bytes(system)
    if len(sys_bytes) < 0xE0:
        raise GroomError(f"system page shorter than 0xe0 bytes ({len(sys_bytes)} bytes)")

    box_min = struct.unpack_from("<3f", sys_bytes, 0x30)
    box_max = struct.unpack_from("<3f", sys_bytes, 0x40)
    box = (box_min, box_max)

    # Child rigs leave the +0x50..+0x6f secondary bounds floats (within +0x48..+0x70) unset
    # at +FLT_MAX (0x7f7fffff, little-endian ff ff 7f 7f) / -FLT_MAX (0xff7fffff).
    rig_float_words = struct.unpack_from("<10I", sys_bytes, 0x48)
    has_unset_rig_floats = any(w in (FLT_MAX_U32, NEG_FLT_MAX_U32) for w in rig_float_words)

    totals = struct.unpack_from("<II", sys_bytes, 0xD0)
    style_hashes = struct.unpack_from("<6I", sys_bytes, 0xB8)
    slot_bone_hashes = style_hashes[:4]
    style_preset_hash = style_hashes[4]
    part_class_hash = style_hashes[5]
    part_class = PART_CLASS_NAMES.get(part_class_hash)

    bone_hashes: tuple[int, ...] = ()
    n_bones = struct.unpack_from("<I", sys_bytes, 0x98)[0]
    if 0 < n_bones <= 256 and struct.unpack_from("<Q", sys_bytes, 0xA0)[0]:
        b_at = _sys_ptr(sys_bytes, 0xA0)
        if b_at + 4 * n_bones <= len(sys_bytes):
            bone_hashes = struct.unpack_from(f"<{n_bones}I", sys_bytes, b_at)

    regions: list[dict] = []
    n_reg = struct.unpack_from("<I", sys_bytes, 0xA8)[0]
    if 0 < n_reg <= 64 and struct.unpack_from("<Q", sys_bytes, 0xB0)[0]:
        r_at = _sys_ptr(sys_bytes, 0xB0)
        for k in range(n_reg):
            off_k = r_at + 12 * k
            if off_k + 12 > len(sys_bytes):
                break
            n_tri, n_guide, n_s0, _pad, rid, n_s1 = struct.unpack_from("<HHHBBI", sys_bytes, off_k)
            regions.append({
                "id": rid,
                "triangles": n_tri,
                "guides": n_guide,
                "strands": [n_s0, n_s1],
            })

    style_params: dict | None = None
    if len(sys_bytes) >= 0x110:
        h = [_finite_f16(v) for v in struct.unpack_from("<20e", sys_bytes, 0xE8)]
        style_params = {
            "strandWidth": h[0],          # +0xe8 (typically 0.1, or 0.03 in some grooms)
            "longestGuideM": h[1],        # +0xea (longest guide in metres)
            "longestGuideAltM": h[3],     # +0xee (longest guide in metres)
            "braidParam": h[4],           # +0xf0 (correlates with braided styles, r = 0.75)
            "curlFrizz0": {               # +0xf8..+0x103 (6 half-floats: 2 amounts, 2 radii in m, 2 freqs/m)
                "amounts": [h[8], h[9]],
                "radiiM": [h[10], h[11]],
                "frequenciesPerM": [h[12], h[13]],
            },
            "curlFrizz1": {               # +0x104..+0x10f (6 half-floats; +0x10a correlates with curly, r = 0.81)
                "amounts": [h[14], h[15]],
                "radiiM": [h[16], h[17]],
                "frequenciesPerM": [h[18], h[19]],
            },
        }

    return {
        "box": box,
        "strand_totals": totals,
        "bone_hashes": bone_hashes,
        "regions": regions,
        "style_hashes": style_hashes,
        "slot_bone_hashes": slot_bone_hashes,
        "style_preset_hash": style_preset_hash,
        "part_class_hash": part_class_hash,
        "part_class": part_class,
        "has_unset_rig_floats": has_unset_rig_floats,
        "style_params": style_params,
    }


@dataclass
class Groom:
    """Decoded smh groom LOD resource."""

    archive: str
    offset: int
    system: bytes = field(repr=False)
    buffer: bytes = field(repr=False)
    header: tuple[int, ...] = field(repr=False)
    guides: int = 0
    triangles_n: int = 0
    nodes: int = 0
    nodes_per_guide: int = 0
    flags: int = 0
    box: tuple[tuple[float, float, float], tuple[float, float, float]] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    strand_totals: tuple[int, int] = (0, 0)
    regions: list[dict] = field(default_factory=list)
    bone_hashes: tuple[int, ...] = ()
    style_hashes: tuple[int, ...] = ()
    style_preset_hash: int | None = None
    part_class_hash: int | None = None
    part_class: str | None = None
    has_unset_rig_floats: bool = False
    style_params: dict | None = None
    positions: list[list[tuple[float, float, float]]] = field(default_factory=list, repr=False)
    node_w: list[list[float]] = field(default_factory=list, repr=False)
    triangles: list[tuple[int, int, int]] = field(default_factory=list, repr=False)
    tri_count: list[tuple[int, int, int, int]] = field(default_factory=list, repr=False)
    tri_region: bytes = field(default=b"", repr=False)
    lengths: list[float] = field(default_factory=list, repr=False)
    uvs: list[tuple[float, float]] = field(default_factory=list, repr=False)
    bone_indices: list[tuple[int, ...]] = field(default_factory=list, repr=False)
    bone_weights_raw: list[tuple[int, ...]] = field(default_factory=list, repr=False)
    length_random: tuple[float, float] = (0.0, 1.0)
    present: dict[str, bool] = field(default_factory=dict, repr=False)
    optional_raw: dict[str, bytes] = field(default_factory=dict, repr=False)
    tail: bytes = field(default_factory=b"", repr=False)
    sentinel_2c: int = 0xFFFFFFFF
    ext_3c_offset: int = 0

    def seed(self, strand_class: int = 0) -> int:
        """Per-class RNG seed word from the GPU buffer header."""
        if strand_class not in (0, 1):
            raise GroomError(f"unsupported strand class {strand_class} (expected 0 or 1)")
        return self.header[STRAND_CLASS_SEED[strand_class] // 4]

    def bone_weights(self) -> list[tuple[float, ...]]:
        """Sum-normalised 8-slot skin weights per guide."""
        out: list[tuple[float, ...]] = []
        for raw in self.bone_weights_raw:
            total = sum(raw)
            denom = float(total) if total > 0 else 1e-6
            out.append(tuple(w / denom for w in raw))
        return out

    def guide_curve_lengths(self) -> list[float]:
        """Measured polyline arc length in metres for each guide."""
        out: list[float] = []
        for guide in self.positions:
            acc = 0.0
            for i in range(len(guide) - 1):
                p0 = guide[i]
                p1 = guide[i + 1]
                acc += math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
            out.append(acc)
        return out

    def tri_count_totals(self) -> list[int]:
        """Sum of per-triangle strand counts for all 4 class slots."""
        totals = [0, 0, 0, 0]
        for c0, c1, c2, c3 in self.tri_count:
            totals[0] += c0
            totals[1] += c1
            totals[2] += c2
            totals[3] += c3
        return totals

    def summary(self) -> dict:
        """Return a JSON-serializable summary of this groom LOD resource."""
        seg = self.guide_curve_lengths()
        min_len = min(seg) if seg else 0.0
        mean_len = (sum(seg) / len(seg)) if seg else 0.0
        max_len = max(seg) if seg else 0.0
        max_err = max((abs(a - b) for a, b in zip(seg, self.lengths)), default=0.0)
        return {
            "archive": self.archive,
            "offset": hex(self.offset),
            "guides": self.guides,
            "nodesPerGuide": self.nodes_per_guide,
            "nodes": self.nodes,
            "triangles": self.triangles_n,
            "flags": hex(self.flags),
            "box": [
                [round(c, 5) for c in self.box[0]],
                [round(c, 5) for c in self.box[1]],
            ],
            "strandTotals": [self.strand_totals[0], self.strand_totals[1]],
            "strandsFromTriangles": self.tri_count_totals(),
            "regions": self.regions,
            "bufferBytes": len(self.buffer),
            "systemBytes": len(self.system),
            "guideLengthM": [round(min_len, 5), round(mean_len, 5), round(max_len, 5)],
            "storedLengthMaxErrorM": round(max_err, 6),
            "lengthRandom": [round(self.length_random[0], 4), round(self.length_random[1], 4)],
            "seeds": [hex(self.seed(0)), hex(self.seed(1))],
            "optionalArrays": sorted(
                name
                for name, present in self.present.items()
                if present and name.startswith(("ext", "node_", "direction"))
            ),
            "tailBytes": len(self.tail),
            "sentinel2c": hex(self.sentinel_2c),
            "hasExt3c": bool(self.ext_3c_offset),
            "boneHashCount": len(self.bone_hashes),
            "styleHashes": [hex(h) for h in self.style_hashes],
            "stylePresetHash": hex(self.style_preset_hash) if self.style_preset_hash is not None else None,
            "partClassHash": hex(self.part_class_hash) if self.part_class_hash is not None else None,
            "partClass": self.part_class,
            "hasUnsetRigFloats": self.has_unset_rig_floats,
            "styleParams": _round_style_params(self.style_params),
        }


def parse(
    system: bytes | bytearray | memoryview | None,
    buffer: bytes | bytearray | memoryview,
    archive: str = "",
    offset: int = 0,
) -> Groom:
    """Decode one groom LOD from its inflated system page (optional) and GPU asset buffer."""
    buf = bytes(buffer)
    if len(buf) < HEADER_SIZE:
        raise GroomError(f"asset buffer shorter than {HEADER_SIZE}-byte header ({len(buf)} bytes)")

    header = struct.unpack_from("<40I", buf, 0)
    guides = header[0x58 // 4] & 0xFFFF
    tris = header[0x58 // 4] >> 16
    nodes = header[0x54 // 4] >> 8
    ppg = header[0x5C // 4] & 0xFF
    if not guides or not tris or not ppg or nodes != guides * ppg:
        raise GroomError(f"inconsistent counts: {guides} guides x {ppg} nodes != {nodes} (triangles={tris})")

    counts = {"G": guides, "N": nodes, "T": tris}
    raw_spans: dict[str, tuple[int, int, int]] = {}
    present: dict[str, bool] = {}
    for at, (name, elem_size, domain) in ARRAYS.items():
        start = header[at // 4]
        present[name] = bool(start)
        if not start:
            continue
        n_elems = counts[domain]
        end = start + elem_size * n_elems
        if start < HEADER_SIZE or end > len(buf):
            raise GroomError(f"array {name} ({start:#x}..{end:#x}) outside {len(buf)}-byte buffer")
        raw_spans[name] = (start, end, n_elems)

    for needed in REQUIRED_ARRAYS:
        if needed not in raw_spans:
            raise GroomError(f"asset buffer missing required {needed} array")

    # Decode guide-major rest positions (half4 x, y, z, w).
    pos_start, pos_end, _ = raw_spans["position"]
    pos_flat = struct.unpack_from(f"<{nodes * 4}e", buf, pos_start)
    if not all(math.isfinite(v) for v in pos_flat):
        raise GroomError("non-finite node position")
    positions: list[list[tuple[float, float, float]]] = []
    node_w: list[list[float]] = []
    idx_p = 0
    for _ in range(guides):
        g_pos: list[tuple[float, float, float]] = []
        g_w: list[float] = []
        for _ in range(ppg):
            g_pos.append((pos_flat[idx_p], pos_flat[idx_p + 1], pos_flat[idx_p + 2]))
            g_w.append(pos_flat[idx_p + 3])
            idx_p += 4
        positions.append(g_pos)
        node_w.append(g_w)

    # Decode scalp triangles (3 x u16 guide indices).
    tri_start, _, _ = raw_spans["triangle"]
    tri_flat = struct.unpack_from(f"<{tris * 3}H", buf, tri_start)
    if tri_flat and max(tri_flat) >= guides:
        raise GroomError(f"triangle references guide {max(tri_flat)} past guide count {guides}")
    triangles = [(tri_flat[i], tri_flat[i + 1], tri_flat[i + 2]) for i in range(0, tris * 3, 3)]

    # Decode per-triangle strand counts (4 x u8) and optional region IDs (u8).
    tc_start, tc_end, _ = raw_spans["tri_count"]
    tc_bytes = buf[tc_start:tc_end]
    tri_count = [
        (tc_bytes[i], tc_bytes[i + 1], tc_bytes[i + 2], tc_bytes[i + 3])
        for i in range(0, tris * 4, 4)
    ]
    if "tri_region" in raw_spans:
        tr_start, tr_end, _ = raw_spans["tri_region"]
        tri_region = buf[tr_start:tr_end]
    else:
        tri_region = bytes(tris)

    # Decode guide lengths (f16) and root UVs (2 x f16).
    len_start, _, _ = raw_spans["length"]
    lengths = list(struct.unpack_from(f"<{guides}e", buf, len_start))
    if not all(math.isfinite(v) for v in lengths):
        raise GroomError("non-finite guide length")

    uv_start, _, _ = raw_spans["uv"]
    uv_flat = struct.unpack_from(f"<{guides * 2}e", buf, uv_start)
    if not all(math.isfinite(v) for v in uv_flat):
        raise GroomError("non-finite guide UV")
    uvs = [(uv_flat[i], uv_flat[i + 1]) for i in range(0, guides * 2, 2)]

    # Decode 8-slot bone indices and weights per guide.
    bi_start, bi_end, _ = raw_spans["bone_index"]
    bi_bytes = buf[bi_start:bi_end]
    bone_indices = [tuple(bi_bytes[i:i + 8]) for i in range(0, guides * 8, 8)]

    bw_start, bw_end, _ = raw_spans["bone_weight"]
    bw_bytes = buf[bw_start:bw_end]
    bone_weights_raw = [tuple(bw_bytes[i:i + 8]) for i in range(0, guides * 8, 8)]

    # Keep raw slices of optional / undecoded arrays.
    optional_raw: dict[str, bytes] = {}
    for name, _width in OPTIONAL_RAW_ARRAYS:
        if name in raw_spans:
            s0, s1, _ = raw_spans[name]
            optional_raw[name] = buf[s0:s1]

    # Length randomisation parameters at header +0x60 (two half floats: amount, min factor).
    len_rand = struct.unpack_from("<2e", buf, 0x60)
    if not (math.isfinite(len_rand[0]) and math.isfinite(len_rand[1])):
        raise GroomError("non-finite length_random header field")

    # Optional variable-length tail block at header +0x20.
    tail = b""
    tail_off = header[TAIL_OFFSET // 4]
    ext_3c_off = header[EXT_3C_OFFSET // 4]
    if tail_off:
        if tail_off < HEADER_SIZE or tail_off > len(buf):
            raise GroomError(f"tail offset {tail_off:#x} outside {len(buf)}-byte buffer")
        starts = sorted([s for s, _, _ in raw_spans.values()] + ([ext_3c_off] if ext_3c_off else []))
        after = [s for s in starts if s > tail_off]
        tail_end = after[0] if after else len(buf)
        tail = buf[tail_off:tail_end]

    # Parse system page fields when present; fall back to computed box/totals for raw GPU buffers.
    sys_bytes = bytes(system) if system else b""
    regions: list[dict] = []
    bone_hashes: tuple[int, ...] = ()
    style_hashes: tuple[int, ...] = ()
    style_preset_hash: int | None = None
    part_class_hash: int | None = None
    part_class: str | None = None
    has_unset_rig_floats = False
    style_params: dict | None = None
    if sys_bytes:
        sys_info = parse_system_page(sys_bytes)
        box = sys_info["box"]
        totals = sys_info["strand_totals"]
        bone_hashes = sys_info["bone_hashes"]
        regions = sys_info["regions"]
        style_hashes = sys_info["style_hashes"]
        style_preset_hash = sys_info["style_preset_hash"]
        part_class_hash = sys_info["part_class_hash"]
        part_class = sys_info["part_class"]
        has_unset_rig_floats = sys_info["has_unset_rig_floats"]
        style_params = sys_info["style_params"]
    else:
        xs = pos_flat[0::4]
        ys = pos_flat[1::4]
        zs = pos_flat[2::4]
        box = ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
        tc_sum0 = sum(c[0] for c in tri_count)
        tc_sum1 = sum(c[1] for c in tri_count)
        totals = (tc_sum0, tc_sum1)

    return Groom(
        archive=archive,
        offset=offset,
        system=sys_bytes,
        buffer=buf,
        header=header,
        guides=guides,
        triangles_n=tris,
        nodes=nodes,
        nodes_per_guide=ppg,
        flags=header[0x50 // 4],
        box=box,
        strand_totals=totals,
        regions=regions,
        bone_hashes=bone_hashes,
        style_hashes=style_hashes,
        style_preset_hash=style_preset_hash,
        part_class_hash=part_class_hash,
        part_class=part_class,
        has_unset_rig_floats=has_unset_rig_floats,
        style_params=style_params,
        positions=positions,
        node_w=node_w,
        triangles=triangles,
        tri_count=tri_count,
        tri_region=tri_region,
        lengths=lengths,
        uvs=uvs,
        bone_indices=bone_indices,
        bone_weights_raw=bone_weights_raw,
        length_random=(len_rand[0], len_rand[1]),
        present=present,
        optional_raw=optional_raw,
        tail=tail,
        sentinel_2c=header[SENTINEL_OFFSET // 4],
        ext_3c_offset=ext_3c_off,
    )


def read_resource(
    src: bytes | bytearray | memoryview | mmap.mmap,
    offset: int = 0,
    archive: str = "",
) -> Groom:
    """Read one groom LOD resource whose 16-byte zero prefix starts at ``offset``."""
    if offset < 0 or offset + 17 > len(src):
        raise GroomError(f"resource offset {offset:#x} out of range")
    if bytes(src[offset:offset + 16]) != b"\x00" * 16:
        raise GroomError(f"expected 16-byte zero prefix at {offset:#x}")
    return _read_zlib_resource(src, offset + 16, archive=archive, resource_offset=offset)


def _read_zlib_resource(
    src: bytes | bytearray | memoryview | mmap.mmap,
    zlib_offset: int,
    *,
    archive: str = "",
    resource_offset: int = 0,
) -> Groom:
    """Inflate the system page at ``zlib_offset`` followed by 16-aligned GPU buffer stream(s)."""
    system, pos = inflate_stream(src, zlib_offset)
    if not system_page_is_groom(system):
        raise GroomError("not a groom LOD resource (system page signature mismatch)")
    desc = _sys_ptr(system, 0x70)
    size = struct.unpack_from("<I", system, desc + 8)[0] * 4
    gpu = bytearray()
    while len(gpu) < size:
        if pos >= len(src) or src[pos:pos + 1] != b"x":
            raise GroomError("asset GPU buffer zlib stream missing or truncated")
        chunk, pos = inflate_stream(src, pos)
        gpu.extend(chunk)
    return parse(system, bytes(gpu[:size]), archive=archive, offset=resource_offset)


def load_groom(path: str | os.PathLike, offset: int | None = None) -> Groom:
    """Load one groom from an RPF archive (at ``offset`` or first match) or a carved ``.smh`` file."""
    path_str = os.fspath(path)
    size = os.path.getsize(path_str)
    if size == 0:
        raise GroomError(f"empty input file: {path_str}")

    with open(path_str, "rb") as handle:
        with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if offset is not None:
                return read_resource(mm, offset, archive=path_str)
            if mm[:17] == PREFIX:
                return read_resource(mm, 0, archive=path_str)
            if mm[:2] in (b"\x78\x01", b"\x78\x5e", b"\x78\x9c", b"\x78\xda"):
                return _read_zlib_resource(mm, 0, archive=path_str, resource_offset=0)
            if mm[:4] in (b"9FPR", b"RPF9"):
                first = next(scan_offsets(mm), None)
                if first is None:
                    raise GroomError(f"no smh groom LOD resources found in archive: {path_str}")
                return read_resource(mm, first, archive=path_str)
            # Standalone raw GPU asset buffer (e.g. a carved .smh GPU buffer).
            return parse(None, mm[:], archive=path_str, offset=0)


# --- Shader-exact Class 0 RNG & barycentric strand generation -----------------------------------------


def mul24(a: int, b: int) -> int:
    """RDNA2 ``v_mad_u32_u24`` low-24x24->32 unsigned multiply."""
    return ((a & _M24) * (b & _M24)) & _M32


def pcg_u24(state: int) -> int:
    """PCG output hash folded to 24 bits (exact port of ``smh_gen_1_CS_InterpolateStrands``)."""
    v = ((state & _M32) * 0x2C9277B5 + 0xAC564B05) & _M32
    w = (((v >> ((v >> 28) + 4)) ^ v) * 0x108EF2D9) & _M32
    return ((w >> 22) ^ (w & _M24)) & _M32


def pcg_float(state: int) -> float:
    """PCG output hash as an exact float in ``[0, 1)`` (``pcg_u24(state) * 2**-24``)."""
    return pcg_u24(state) * _INV_2_24


def scatter(
    groom: Groom,
    count: int | None = None,
    strand_class: int = 0,
    density_scale: float | None = None,
    region_mask: Iterable[int] | None = None,
    max_accepted: int | None = None,
) -> dict:
    """Scatter strand roots with the exact integer RNG of ``smh_gen_1_CS_InterpolateStrands``.

    Args:
        groom: Decoded ``Groom`` resource.
        count: Number of strand threads (``0..count-1``). Defaults to ``groom.strand_totals[strand_class]``
            or the sum of per-triangle counts for ``strand_class``.
        strand_class: ``0`` (placement set 0, exact ``smh_gen_1`` shader port) or ``1`` (placement set 1,
            approximate until ``smh_gen_2``/``_4``/``_8``/``_16`` are ported; both classes are generated
            alike and drawn together as one batch).
        density_scale: Acceptance factor in ``saturate(tri_count * density_scale)``. Defaults to
            ``1.0 / max(max_tri_count, 1.0)`` so the densest triangle is always accepted.
        region_mask: Optional iterable of region IDs (1..255) to cull; region 0 is never culled.
        max_accepted: Optional cap on accepted (non-culled) strand roots to collect. When ``None``
            (default), all ``count`` candidates are returned (including culled entries). When set,
            candidates are evaluated in thread order ``0..count-1``, culled candidates are skipped,
            and iteration stops once ``max_accepted`` accepted roots are collected, bounding memory
            to ``O(max_accepted)`` instead of ``O(count)``.

    Returns:
        Dict with keys ``tri`` (list[int]), ``culled`` (list[bool]),
        ``bary`` (list[tuple[float, float, float]]), and ``seed`` (list[int]).
    """
    if strand_class not in (0, 1):
        raise GroomError(f"unsupported strand class {strand_class} (expected 0 or 1)")

    per_tri = [c[strand_class] for c in groom.tri_count]
    if count is None:
        count = groom.strand_totals[strand_class] or sum(per_tri)
    if count < 0:
        raise GroomError(f"strand count must be non-negative, got {count}")

    if max_accepted is not None and max_accepted <= 0:
        return {"tri": [], "culled": [], "bary": [], "seed": []}
    if count == 0:
        return {"tri": [], "culled": [], "bary": [], "seed": []}

    if density_scale is None:
        max_c = max(per_tri) if per_tri else 0
        density_scale = 1.0 / max(float(max_c), 1.0)
    ds32 = _f32(float(density_scale))
    # Precompute IEEE 754 float32 acceptance thresholds for every possible u8 triangle count (0..255).
    thresh = [min(max(_f32(float(c) * ds32), 0.0), 1.0) for c in range(256)]

    blocked = [False] * 256
    for rid in region_mask or ():
        blocked[int(rid) & 255] = True
    blocked[0] = False

    n_tris = groom.triangles_n
    tri_region = groom.tri_region
    seed_base = (groom.seed(strand_class) * 0x8A30D) & _M32
    c0_s0, c0_s8 = _ATTEMPT_CONSTS[0]
    c1_s0, c1_s8 = _ATTEMPT_CONSTS[1]
    c2_s0, c2_s8 = _ATTEMPT_CONSTS[2]
    c3_s0, c3_s8 = _ATTEMPT_CONSTS[3]
    attempts = ((c0_s0, c0_s8), (c1_s0, c1_s8), (c2_s0, c2_s8), (c3_s0, c3_s8))

    if max_accepted is None:
        tris_out: list[int] = [0] * count
        culled_out: list[bool] = [False] * count
        bary_out: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)] * count
        seed_out: list[int] = [0] * count
    else:
        tris_out = []
        culled_out = []
        bary_out = []
        seed_out = []
        t_append = tris_out.append
        c_append = culled_out.append
        b_append = bary_out.append
        s_append = seed_out.append

    sqrt = math.sqrt
    for t in range(count):
        v3 = (t * 0x19660D + 0xEF3EFB11) & _M32
        tri_t = 0
        v20_t = 0
        culled_t = False
        for s0, s8 in attempts:
            v11 = (s8 + v3) & _M32
            v5 = (v11 * 0x31CD17D1 + 0x153E39CE) & _M32
            v9 = v11 >> 16
            v6 = (v5 * v11 + 0x31CD17D1) & _M32
            v7 = (v6 * v5) & _M32
            v5 ^= v5 >> 16
            v6 ^= v6 >> 16
            v7 = (v7 + s0) & _M32
            v7 ^= v7 >> 16
            v8 = ((v9 ^ v11) + ((v7 * v5) & _M32)) & _M32
            v10 = (((v8 * v6) & _M32) + v5) & _M32
            pick = (v8 >> 16) % n_tris
            v5 = (((v10 * v8) & _M32) + v6) & _M32
            v6 = (((v5 * v10) & _M32) + v7) & _M32

            tri_t = pick
            if blocked[tri_region[pick]]:
                culled_t = True
                continue
            culled_t = False
            v20_t = v5 >> 8
            if ((v6 & _M24) * _INV_2_24) < thresh[per_tri[pick]]:
                break

        if max_accepted is not None and culled_t:
            continue

        # Barycentric sample via 24-bit multiply cascade (v_mad_u32_u24, k = 0xC083C5).
        seed_t = (0x3C1 * (tri_t & 0xFFFF) + seed_base + v20_t * 31) & _M32
        x1 = (0xC083C5 * (seed_t & _M24) + 0x3C6EF35F) & _M32
        x2 = (0xC083C5 * (x1 & _M24) + 0x3C6EF35F) & _M32
        x1 = (0xC083C5 * (x2 & _M24) + x1) & _M32
        x2 = (0xC083C5 * (x1 & _M24) + x2) & _M32
        x1 ^= x1 >> 12
        x2 ^= x2 >> 12
        x1 = (0xC083C5 * (x2 & _M24) + x1) & _M32
        a = (x1 & _M24) ^ (x1 >> 12)
        x1 = (0xC083C5 * (x1 & _M24) + x2) & _M32
        b = (x1 & _M24) ^ (x1 >> 12)

        u1 = a * _INV_2_24
        u2 = b * _INV_2_24
        r = sqrt(u1)
        if max_accepted is None:
            tris_out[t] = tri_t
            culled_out[t] = culled_t
            bary_out[t] = (1.0 - r, (1.0 - u2) * r, u2 * r)
            seed_out[t] = seed_t
        else:
            t_append(tri_t)
            c_append(False)
            b_append((1.0 - r, (1.0 - u2) * r, u2 * r))
            s_append(seed_t)
            if len(tris_out) >= max_accepted:
                break

    return {"tri": tris_out, "culled": culled_out, "bary": bary_out, "seed": seed_out}


def generate_strands(
    groom: Groom,
    count: int | None = None,
    strand_class: int = 0,
    density_scale: float | None = None,
    region_mask: Iterable[int] | None = None,
    max_strands: int | None = None,
) -> dict:
    """Generate rest-pose strand polylines by barycentric blending and length trimming.

    Returns dict with keys:
        - ``points``: ``list[list[tuple[float, float, float]]]`` of shape ``[n, nodes_per_guide]``
          (trimmed strands repeat their tip point on trailing nodes)
        - ``live``: ``list[int]`` count of distinct nodes per strand (``1..nodes_per_guide``)
        - ``length``: ``list[float]`` actual trimmed arc length in metres
        - ``uv``: ``list[tuple[float, float]]`` barycentric root UV
        - ``tri``: ``list[int]`` scalp triangle index
        - ``bary``: ``list[tuple[float, float, float]]`` barycentric weights
    """
    roots = scatter(groom, count, strand_class, density_scale, region_mask, max_strands)
    r_tri = roots["tri"]
    r_culled = roots["culled"]
    r_bary = roots["bary"]
    r_seed = roots["seed"]

    positions = groom.positions
    triangles = groom.triangles
    lengths = groom.lengths
    uvs = groom.uvs
    ppg = groom.nodes_per_guide
    amount, low = groom.length_random
    one_minus_low = 1.0 - low
    eps_len = 2.0 ** -23
    hypot = math.hypot

    points_out: list[list[tuple[float, float, float]]] = []
    live_out: list[int] = []
    len_out: list[float] = []
    uv_out: list[tuple[float, float]] = []
    tri_out: list[int] = []
    bary_out: list[tuple[float, float, float]] = []

    for i in range(len(r_tri)):
        if r_culled[i]:
            continue
        if max_strands is not None and len(points_out) >= max_strands:
            break
        tri_i = r_tri[i]
        b0, b1, b2 = r_bary[i]
        g0, g1, g2 = triangles[tri_i]

        p0_list = positions[g0]
        p1_list = positions[g1]
        p2_list = positions[g2]

        guide_len = b0 * lengths[g0] + b1 * lengths[g1] + b2 * lengths[g2]
        factor = low + pcg_float(r_seed[i]) * one_minus_low
        target = guide_len * (1.0 + (factor - 1.0) * amount)
        if target < 0.0:
            target = 0.0

        # Blend root position.
        a0, a1, a2 = p0_list[0], p1_list[0], p2_list[0]
        prev_x = b0 * a0[0] + b1 * a1[0] + b2 * a2[0]
        prev_y = b0 * a0[1] + b1 * a1[1] + b2 * a2[1]
        prev_z = b0 * a0[2] + b1 * a1[2] + b2 * a2[2]

        cur_x, cur_y, cur_z = prev_x, prev_y, prev_z
        strand_pts: list[tuple[float, float, float]] = [(cur_x, cur_y, cur_z)]
        remaining = target
        total_used = 0.0
        live_nodes = 1

        for k in range(1, ppg):
            q0, q1, q2 = p0_list[k], p1_list[k], p2_list[k]
            nx = b0 * q0[0] + b1 * q1[0] + b2 * q2[0]
            ny = b0 * q0[1] + b1 * q1[1] + b2 * q2[1]
            nz = b0 * q0[2] + b1 * q1[2] + b2 * q2[2]

            dx = nx - prev_x
            dy = ny - prev_y
            dz = nz - prev_z
            prev_x, prev_y, prev_z = nx, ny, nz

            seg_len = hypot(dx, dy, dz)
            if seg_len < eps_len:
                seg_len = eps_len

            if remaining > 0.0:
                use = seg_len if remaining >= seg_len else remaining
                if use < 1e-9:
                    use = 0.0
                if use > 0.0:
                    scale = use / seg_len
                    cur_x += dx * scale
                    cur_y += dy * scale
                    cur_z += dz * scale
                    total_used += use
                    live_nodes += 1
                remaining -= seg_len
            strand_pts.append((cur_x, cur_y, cur_z))

        u0, v0 = uvs[g0]
        u1, v1 = uvs[g1]
        u2, v2 = uvs[g2]
        points_out.append(strand_pts)
        live_out.append(live_nodes)
        len_out.append(total_used)
        uv_out.append((b0 * u0 + b1 * u1 + b2 * u2, b0 * v0 + b1 * v1 + b2 * v2))
        tri_out.append(tri_i)
        bary_out.append((b0, b1, b2))

    return {
        "points": points_out,
        "live": live_out,
        "length": len_out,
        "uv": uv_out,
        "tri": tri_out,
        "bary": bary_out,
    }


# --- Geometry builders & GLB export (pure stdlib) -----------------------------------------------------


def _convert_up(
    points: Iterable[tuple[float, float, float]],
    up_axis: str = "z",
) -> list[tuple[float, float, float]]:
    if up_axis == "y":
        return [(_clean_zero(x), _clean_zero(y), _clean_zero(z)) for x, y, z in points]
    # RAGE Z-up -> glTF Y-up: (x, z, -y).
    return [(_clean_zero(x), _clean_zero(z), _clean_zero(-y)) for x, y, z in points]


def _polyline_frames(
    poly: list[tuple[float, float, float]],
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    """Compute orthonormal (side, up) cross-section frame at each node of one polyline."""
    k = len(poly)
    sides: list[tuple[float, float, float]] = []
    ups: list[tuple[float, float, float]] = []
    hypot = math.hypot
    for j in range(k):
        if k == 1:
            tx, ty, tz = 0.0, 0.0, 1.0
        elif j == 0:
            tx = poly[1][0] - poly[0][0]
            ty = poly[1][1] - poly[0][1]
            tz = poly[1][2] - poly[0][2]
        elif j == k - 1:
            tx = poly[j][0] - poly[j - 1][0]
            ty = poly[j][1] - poly[j - 1][1]
            tz = poly[j][2] - poly[j - 1][2]
        else:
            tx = 0.5 * (poly[j + 1][0] - poly[j - 1][0])
            ty = 0.5 * (poly[j + 1][1] - poly[j - 1][1])
            tz = 0.5 * (poly[j + 1][2] - poly[j - 1][2])
        norm = hypot(tx, ty, tz)
        if norm > 1e-9:
            inv = 1.0 / norm
            tx, ty, tz = tx * inv, ty * inv, tz * inv
        else:
            tx, ty, tz = 0.0, 0.0, 1.0

        # Reference vector avoids gimbal lock near +/-Z.
        if abs(tz) < 0.9:
            # cross(tangent, (0, 0, 1)) = (ty, -tx, 0)
            sx, sy, sz = ty, -tx, 0.0
        else:
            # cross(tangent, (1, 0, 0)) = (0, tz, -ty)
            sx, sy, sz = 0.0, tz, -ty
        sn = hypot(sx, sy, sz)
        if sn > 1e-9:
            inv_s = 1.0 / sn
            sx, sy, sz = sx * inv_s, sy * inv_s, sz * inv_s
        else:
            sx, sy, sz = 1.0, 0.0, 0.0

        # up = cross(tangent, side)
        ux = ty * sz - tz * sy
        uy = tz * sx - tx * sz
        uz = tx * sy - ty * sx
        sides.append((sx, sy, sz))
        ups.append((ux, uy, uz))
    return sides, ups


def build_tubes(
    polylines: list[list[tuple[float, float, float]]],
    live: list[int] | None = None,
    *,
    radius: float = 0.0004,
    tip: float = 0.25,
    sides: int = 3,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[int]]:
    """Build tapered N-sided triangle tubes around polylines -> ``(verts, normals, indices)``."""
    if not polylines or sides < 3:
        return [], [], []
    k = len(polylines[0])
    if k < 2:
        return [], [], []

    ring_trig = [
        (math.cos(s * (2.0 * math.pi / sides)), math.sin(s * (2.0 * math.pi / sides)))
        for s in range(sides)
    ]
    denom = float(k - 1)
    tapers = [(1.0 - (1.0 - tip) * (j / denom)) * radius for j in range(k)]

    verts: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []
    v_append = verts.append
    n_append = normals.append
    i_extend = indices.extend

    for idx, poly in enumerate(polylines):
        base_poly = idx * k * sides
        side_vecs, up_vecs = _polyline_frames(poly)
        for j in range(k):
            px, py, pz = poly[j]
            sx, sy, sz = side_vecs[j]
            ux, uy, uz = up_vecs[j]
            rad = tapers[j]
            for ca, sa in ring_trig:
                rx = ca * sx + sa * ux
                ry = ca * sy + sa * uy
                rz = ca * sz + sa * uz
                v_append((px + rx * rad, py + ry * rad, pz + rz * rad))
                n_append((rx, ry, rz))

        seg_limit = (live[idx] - 1) if live is not None else (k - 1)
        if seg_limit > k - 1:
            seg_limit = k - 1
        for j in range(max(seg_limit, 0)):
            ring0 = base_poly + j * sides
            ring1 = ring0 + sides
            for s in range(sides):
                s_next = (s + 1) % sides
                a = ring0 + s
                b = ring0 + s_next
                c = ring1 + s
                d = ring1 + s_next
                i_extend((a, b, c, b, d, c))

    return verts, normals, indices


def build_ribbons(
    polylines: list[list[tuple[float, float, float]]],
    live: list[int] | None = None,
    *,
    width: float = 0.0008,
    tip: float = 0.25,
) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]], list[int]]:
    """Build flat tapered quad-strip ribbons around polylines -> ``(verts, normals, indices)``."""
    if not polylines:
        return [], [], []
    k = len(polylines[0])
    if k < 2:
        return [], [], []

    half_w = 0.5 * width
    denom = float(k - 1)
    half_tapers = [(1.0 - (1.0 - tip) * (j / denom)) * half_w for j in range(k)]

    verts: list[tuple[float, float, float]] = []
    normals: list[tuple[float, float, float]] = []
    indices: list[int] = []
    v_append = verts.append
    n_append = normals.append
    i_extend = indices.extend

    for idx, poly in enumerate(polylines):
        base_poly = idx * k * 2
        side_vecs, up_vecs = _polyline_frames(poly)
        for j in range(k):
            px, py, pz = poly[j]
            sx, sy, sz = side_vecs[j]
            up = up_vecs[j]
            hw = half_tapers[j]
            v_append((px - sx * hw, py - sy * hw, pz - sz * hw))
            n_append(up)
            v_append((px + sx * hw, py + sy * hw, pz + sz * hw))
            n_append(up)

        seg_limit = (live[idx] - 1) if live is not None else (k - 1)
        if seg_limit > k - 1:
            seg_limit = k - 1
        for j in range(max(seg_limit, 0)):
            r0 = base_poly + j * 2
            r1 = r0 + 2
            i_extend((r0, r0 + 1, r1, r0 + 1, r1 + 1, r1))

    return verts, normals, indices


def build_lines(
    polylines: list[list[tuple[float, float, float]]],
    live: list[int] | None = None,
) -> tuple[list[tuple[float, float, float]], list[int]]:
    """Flatten polylines for glTF ``LINES`` (mode 1) -> ``(verts, paired_indices)``."""
    if not polylines:
        return [], []
    k = len(polylines[0])
    verts: list[tuple[float, float, float]] = []
    indices: list[int] = []
    v_extend = verts.extend
    i_extend = indices.extend
    for idx, poly in enumerate(polylines):
        base = idx * k
        v_extend(poly)
        seg_limit = (live[idx] - 1) if live is not None else (k - 1)
        if seg_limit > k - 1:
            seg_limit = k - 1
        for j in range(max(seg_limit, 0)):
            a = base + j
            i_extend((a, a + 1))
    return verts, indices


def _pack_scalar_indices(indices: list[int], vert_count: int) -> tuple[bytes, int]:
    """Pack any index list (triangles, lines, or line strips) as uint16 or uint32."""
    if not indices:
        raise GroomError("cannot pack empty index list")
    lo = min(indices)
    hi = max(indices)
    if lo < 0 or hi >= vert_count:
        raise GroomError(f"index out of range [0, {vert_count - 1}]: min={lo}, max={hi}")
    if vert_count <= 65535:
        comp_type = glb_writer.COMPONENT_UNSIGNED_SHORT
        if _UINT16_CODE is not None:
            arr = array.array(_UINT16_CODE, indices)
            if _NEEDS_BYTESWAP:
                arr.byteswap()
            return arr.tobytes(), comp_type
        return struct.pack(f"<{len(indices)}H", *indices), comp_type

    comp_type = glb_writer.COMPONENT_UNSIGNED_INT
    if _UINT32_CODE is not None:
        arr = array.array(_UINT32_CODE, indices)
        if _NEEDS_BYTESWAP:
            arr.byteswap()
        return arr.tobytes(), comp_type
    return struct.pack(f"<{len(indices)}I", *indices), comp_type


def _pack_vec3(values: list[tuple[float, float, float]]) -> tuple[bytes, list[float], list[float]]:
    raw_bytes, floats = glb_writer._pack_f32(values, 3)
    xs = floats[0::3]
    ys = floats[1::3]
    zs = floats[2::3]
    return (
        raw_bytes,
        [_clean_zero(min(xs)), _clean_zero(min(ys)), _clean_zero(min(zs))],
        [_clean_zero(max(xs)), _clean_zero(max(ys)), _clean_zero(max(zs))],
    )


def to_glb(
    groom: Groom,
    *,
    max_strands: int = 6000,
    guide_radius: float = 0.0006,
    strand_radius: float = 0.00035,
    scalp: bool = True,
    guides: bool = True,
    classes: Iterable[int] = (0, 1),
    geometry: str = "tubes",
    curves: bool = False,
    vertex_budget: int = 360000,
    up_axis: str = "z",
) -> tuple[bytes, dict]:
    """Serialize a groom LOD to binary glTF 2.0 (``.glb``) bytes plus metadata dict.

    ``geometry`` selects how guides and strands are emitted:
      - ``"tubes"``: 3-sided tapered triangle tubes (glTF ``TRIANGLES``, mode 4).
      - ``"ribbons"``: flat tapered quad-strip ribbons (glTF ``TRIANGLES``, mode 4, double-sided).
      - ``"curves"``: indexed line segments (glTF ``LINES``, mode 1), one primitive per part.
      - ``"line-strip"``: glTF ``LINE_STRIP`` (mode 3), one primitive per polyline.

    Passing ``curves=True`` overrides ``geometry`` to ``"curves"``.
    """
    if curves:
        geometry = "curves"
    if geometry not in GEOMETRY_MODES:
        raise GroomError(f"unsupported geometry mode {geometry!r} (expected one of {GEOMETRY_MODES})")
    up = up_axis.lower()
    if up not in ("z", "y"):
        raise GroomError(f"unsupported up_axis {up_axis!r} (expected 'z' or 'y')")
    cls_tuple = tuple(classes)
    for cls in cls_tuple:
        if cls not in (0, 1):
            raise GroomError(f"unsupported strand class {cls} (expected 0 or 1)")

    bin_buf = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    gltf_materials: list[dict] = []
    part_mat_index: dict[int, int] = {}
    primitives: list[dict] = []
    stats: dict = {"strands": {}, "shown": {}, "geometry": geometry}
    name = f"groom_{groom.offset:x}"

    def get_material(part: int, double_sided: bool = False) -> int:
        if part not in part_mat_index:
            mat: dict = {
                "name": PART_NAMES[part],
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(PART_COLORS[part]),
                    "metallicFactor": 0.0,
                    "roughnessFactor": 0.85,
                },
            }
            if double_sided:
                mat["doubleSided"] = True
            part_mat_index[part] = len(gltf_materials)
            gltf_materials.append(mat)
        return part_mat_index[part]

    def append_primitive(
        part: int,
        verts: list[tuple[float, float, float]],
        normals: list[tuple[float, float, float]] | None,
        indices: list[int],
        mode: int = MODE_TRIANGLES,
        double_sided: bool = False,
    ) -> None:
        if not verts or not indices:
            return
        conv_verts = _convert_up(verts, up)
        pos_bytes, vmin, vmax = _pack_vec3(conv_verts)
        pos_bv = glb_writer._append_buffer_view(
            bin_buf, buffer_views, pos_bytes, target=glb_writer.TARGET_ARRAY_BUFFER,
        )
        pos_acc = len(accessors)
        accessors.append({
            "bufferView": pos_bv,
            "componentType": glb_writer.COMPONENT_FLOAT,
            "count": len(conv_verts),
            "type": "VEC3",
            "min": vmin,
            "max": vmax,
        })
        attributes: dict[str, int] = {"POSITION": pos_acc}
        if normals is not None and len(normals) == len(conv_verts):
            conv_nrm = [
                glb_writer._normalize3(*n) for n in _convert_up(normals, up)
            ]
            nrm_bytes, _ = glb_writer._pack_f32(conv_nrm, 3)
            nrm_bv = glb_writer._append_buffer_view(
                bin_buf, buffer_views, nrm_bytes, target=glb_writer.TARGET_ARRAY_BUFFER,
            )
            nrm_acc = len(accessors)
            accessors.append({
                "bufferView": nrm_bv,
                "componentType": glb_writer.COMPONENT_FLOAT,
                "count": len(conv_nrm),
                "type": "VEC3",
            })
            attributes["NORMAL"] = nrm_acc

        idx_bytes, idx_comp = _pack_scalar_indices(indices, len(conv_verts))
        idx_bv = glb_writer._append_buffer_view(
            bin_buf, buffer_views, idx_bytes, target=glb_writer.TARGET_ELEMENT_ARRAY_BUFFER,
        )
        idx_acc = len(accessors)
        accessors.append({
            "bufferView": idx_bv,
            "componentType": idx_comp,
            "count": len(indices),
            "type": "SCALAR",
        })
        primitives.append({
            "attributes": attributes,
            "indices": idx_acc,
            "material": get_material(part, double_sided=double_sided),
            "mode": mode,
            "extras": {"shaderIndex": part, "part": PART_NAMES[part]},
        })

    def append_polylines(
        part: int,
        polylines: list[list[tuple[float, float, float]]],
        live: list[int] | None,
        radius: float,
    ) -> None:
        if not polylines:
            return
        if geometry == "curves":
            verts, idx = build_lines(polylines, live)
            append_primitive(part, verts, None, idx, mode=MODE_LINES)
        elif geometry == "line-strip":
            # Share one vertex buffer view for the part and emit one LINE_STRIP primitive per strand.
            all_verts: list[tuple[float, float, float]] = []
            ranges: list[tuple[int, int]] = []
            for i, poly in enumerate(polylines):
                n_live = live[i] if live is not None else len(poly)
                n_live = min(n_live, len(poly))
                if n_live < 2:
                    continue
                start_v = len(all_verts)
                all_verts.extend(poly[:n_live])
                ranges.append((start_v, n_live))
            if not ranges:
                return
            conv_verts = _convert_up(all_verts, up)
            pos_bytes, vmin, vmax = _pack_vec3(conv_verts)
            pos_bv = glb_writer._append_buffer_view(
                bin_buf, buffer_views, pos_bytes, target=glb_writer.TARGET_ARRAY_BUFFER,
            )
            pos_acc = len(accessors)
            accessors.append({
                "bufferView": pos_bv,
                "componentType": glb_writer.COMPONENT_FLOAT,
                "count": len(conv_verts),
                "type": "VEC3",
                "min": vmin,
                "max": vmax,
            })
            mat_idx = get_material(part)
            for start_v, n_live in ranges:
                strip_idx = list(range(start_v, start_v + n_live))
                idx_bytes, idx_comp = _pack_scalar_indices(strip_idx, len(conv_verts))
                idx_bv = glb_writer._append_buffer_view(
                    bin_buf, buffer_views, idx_bytes, target=glb_writer.TARGET_ELEMENT_ARRAY_BUFFER,
                )
                idx_acc = len(accessors)
                accessors.append({
                    "bufferView": idx_bv,
                    "componentType": idx_comp,
                    "count": n_live,
                    "type": "SCALAR",
                })
                primitives.append({
                    "attributes": {"POSITION": pos_acc},
                    "indices": idx_acc,
                    "material": mat_idx,
                    "mode": MODE_LINE_STRIP,
                    "extras": {"shaderIndex": part, "part": PART_NAMES[part]},
                })
        elif geometry == "ribbons":
            verts, nrms, idx = build_ribbons(polylines, live, width=radius * 2.0)
            append_primitive(part, verts, nrms, idx, mode=MODE_TRIANGLES, double_sided=True)
        else:
            verts, nrms, idx = build_tubes(polylines, live, radius=radius)
            append_primitive(part, verts, nrms, idx, mode=MODE_TRIANGLES)

    if scalp:
        roots = [guide[0] for guide in groom.positions]
        flat_tris = [i for tri in groom.triangles for i in tri]
        if roots and flat_tris:
            nrms = glb_writer._compute_smooth_normals(roots, flat_tris)
            append_primitive(0, roots, nrms, flat_tris, mode=MODE_TRIANGLES, double_sided=True)

    if guides:
        append_polylines(1, groom.positions, None, guide_radius)

    verts_per_node = 3 if geometry == "tubes" else (2 if geometry == "ribbons" else 1)
    for cls in cls_tuple:
        tri_sum = sum(c[cls] for c in groom.tri_count)
        total = groom.strand_totals[cls] or tri_sum
        stats["strands"][cls] = total
        if not total or not tri_sum:
            continue
        if geometry in ("tubes", "ribbons"):
            budget = min(max_strands, max(200, vertex_budget // max(groom.nodes_per_guide * verts_per_node, 1)))
        else:
            budget = max_strands
        generated = generate_strands(groom, count=total, strand_class=cls, max_strands=budget)
        stats["shown"][cls] = len(generated["points"])
        append_polylines(2 + cls, generated["points"], generated["live"], strand_radius)

    if not primitives:
        raise GroomError("nothing to export")

    bin_pad = (-len(bin_buf)) % 4
    if bin_pad:
        bin_buf.extend(b"\x00" * bin_pad)

    extras = {
        "archive": groom.archive,
        "member": hex(groom.offset),
        "groom": groom.summary(),
        "strandTotals": stats["strands"],
        "strandsShown": stats["shown"],
        "geometry": geometry,
        "note": "Tube/ribbon radii are display sizes, not the game's runtime strand width.",
    }
    gltf: dict = {
        "asset": {"version": "2.0", "generator": "Leonida smh_groom"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"name": name, "mesh": 0}],
        "meshes": [{"name": name, "primitives": primitives}],
        "materials": gltf_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_buf)}],
        "extras": extras,
    }
    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    if json_pad:
        json_bytes += b" " * json_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    blob = b"".join((
        struct.pack("<4sII", glb_writer.GLB_MAGIC, glb_writer.GLB_VERSION, total_length),
        struct.pack("<II", len(json_bytes), glb_writer.CHUNK_TYPE_JSON),
        json_bytes,
        struct.pack("<II", len(bin_buf), glb_writer.CHUNK_TYPE_BIN),
        bytes(bin_buf),
    ))
    return blob, extras


def write_glb(path: str | os.PathLike, groom: Groom, **kwargs) -> dict:
    """Export ``groom`` to a binary ``.glb`` file atomically; returns metadata dict."""
    blob, meta = to_glb(groom, **kwargs)
    glb_writer._write_bytes_atomic(os.fspath(path), blob)
    meta["bytes"] = len(blob)
    return meta


# --- LOD chain grouping & corpus census ---------------------------------------------------------------


def classify_placement(
    box: tuple[tuple[float, float, float], tuple[float, float, float]],
    longest_m: float,
) -> str:
    """Heuristic placement class from stored bounding box (skeleton-root space, Z-up, metres)."""
    (x0, y0, z0), (x1, y1, z1) = box
    width = x1 - x0
    height = z1 - z0
    zc = 0.5 * (z0 + z1)
    if abs(x0 + x1) < 0.08 and width < 0.34 and 0.44 < zc < 0.9 and z1 < 1.05 and y0 > -0.5:
        if z1 > 0.745 or height > 0.24:
            return "Scalp hair"
        if height < 0.075 and zc > 0.62:
            return "Brows and lashes"
        return "Facial hair"
    if (
        width < 0.9
        and -1.05 < z0
        and z1 < 0.52
        and longest_m < 0.08
        and abs(x0 + x1) < 0.12
        and (y1 - y0) < 0.45
    ):
        return "Body hair"
    return "Fur and other"


def _same_lod_asset(a: dict, b: dict) -> bool:
    if a["ppg"] != b["ppg"] or a["flags"] != b["flags"] or b["guides"] >= a["guides"]:
        return False
    for u, v in zip(a["box"], b["box"]):
        for p, q in zip(u, v):
            if abs(p - q) > 0.03:
                return False
    if abs(a["s0"] - b["s0"]) > max(12, 0.05 * a["s0"]):
        return False
    if abs(a["s1"] - b["s1"]) > max(12, 0.05 * a["s1"]):
        return False
    return True


def group_lod_chains(rows: list[dict]) -> list[list[dict]]:
    """Group decoded LOD summary rows into logical multi-LOD assets, rejoining split chains."""
    ok = [r for r in rows if not r.get("error")]
    runs: list[list[dict]] = []
    cur: list[dict] = []
    for row in ok:
        if cur and _same_lod_asset(cur[-1], row) and len(cur) < 8:
            cur.append(row)
        else:
            if cur:
                runs.append(cur)
            cur = [row]
    if cur:
        runs.append(cur)

    assets: list[list[dict]] = []
    used: set[int] = set()
    for i, run in enumerate(runs):
        if i in used:
            continue
        full = list(run)
        # Split LOD chains resume 1 to 3 runs later (LOD 0, intervening groom(s), LODs 1..3).
        for j in (i + 2, i + 3, i + 1):
            if j < len(runs) and j not in used and len(full) < 4 and _same_lod_asset(full[-1], runs[j][0]):
                full.extend(runs[j])
                used.add(j)
                break
        assets.append(full)
    return assets


def _collect_census_targets(paths: Iterable[str]) -> list[str]:
    targets: list[str] = []
    for raw in paths:
        if os.path.isdir(raw):
            for root, dirs, files in os.walk(raw):
                dirs.sort()
                for name in sorted(files):
                    low = name.lower()
                    if low.endswith((".rpf", ".smh")):
                        targets.append(os.path.join(root, name))
        else:
            targets.append(raw)
    return targets


def run_census(paths: Iterable[str], *, limit: int | None = None) -> dict:
    """Scan archives or carved ``.smh`` files and return aggregate corpus census statistics."""
    targets = _collect_census_targets(paths)
    per_source: list[dict] = []
    total_lods = 0
    total_assets = 0
    total_errors = 0
    total_buffer_bytes = 0
    total_s0 = 0
    total_s1 = 0
    class0_exact_match = 0
    class1_exact_match = 0
    sentinel_non_default = 0
    ext_3c_count = 0
    unset_rig_float_assets = 0
    flags_hist: collections.Counter[str] = collections.Counter()
    lod_depth_hist: collections.Counter[int] = collections.Counter()
    opt_array_hist: collections.Counter[str] = collections.Counter()
    placement_hist: collections.Counter[str] = collections.Counter()
    part_class_hist: collections.Counter[str] = collections.Counter()
    style_preset_hist: collections.Counter[str] = collections.Counter()
    guides_min: int | None = None
    guides_max = 0
    ppg_min: int | None = None
    ppg_max = 0
    remaining = limit

    def _row_from_groom(g: Groom, off: int) -> dict:
        tc_totals = g.tri_count_totals()
        longest = max(g.lengths) if g.lengths else 0.0
        return {
            "offset": off,
            "error": None,
            "guides": g.guides,
            "ppg": g.nodes_per_guide,
            "tris": g.triangles_n,
            "flags": g.flags,
            "s0": g.strand_totals[0],
            "s1": g.strand_totals[1],
            "tc0": tc_totals[0],
            "tc1": tc_totals[1],
            "bytes": len(g.buffer),
            "box": g.box,
            "longest": longest,
            "opt": [n for n, p in g.present.items() if p and n.startswith(("ext", "node_", "direction"))],
            "sentinel_2c": g.sentinel_2c,
            "has_ext_3c": bool(g.ext_3c_offset),
            "part_class": g.part_class,
            "part_class_hash": g.part_class_hash,
            "style_preset_hash": g.style_preset_hash,
            "has_unset_rig_floats": g.has_unset_rig_floats,
        }

    for target in targets:
        if remaining is not None and remaining <= 0:
            break
        size = os.path.getsize(target)
        rows: list[dict] = []
        if size == 0:
            continue

        with open(target, "rb") as handle:
            with mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                if target.lower().endswith(".rpf") or mm[:4] in (b"9FPR", b"RPF9"):
                    for off in scan_offsets(mm):
                        if remaining is not None and remaining <= 0:
                            break
                        if remaining is not None:
                            remaining -= 1
                        try:
                            g = read_resource(mm, off, archive=target)
                            rows.append(_row_from_groom(g, off))
                        except (GroomError, ValueError, IndexError) as exc:
                            rows.append({"offset": off, "error": str(exc)})
                else:
                    if remaining is not None:
                        remaining -= 1
                    try:
                        g = load_groom(target)
                        rows.append(_row_from_groom(g, 0))
                    except (GroomError, ValueError, IndexError) as exc:
                        rows.append({"offset": 0, "error": str(exc)})

        ok_rows = [r for r in rows if not r["error"]]
        err_rows = [r for r in rows if r["error"]]
        chains = group_lod_chains(ok_rows)

        src_bytes = sum(r["bytes"] for r in ok_rows)
        src_s0 = sum(chain[0]["s0"] for chain in chains)
        src_s1 = sum(chain[0]["s1"] for chain in chains)

        total_lods += len(ok_rows)
        total_errors += len(err_rows)
        total_assets += len(chains)
        total_buffer_bytes += src_bytes
        total_s0 += src_s0
        total_s1 += src_s1

        for r in ok_rows:
            flags_hist[hex(r["flags"])] += 1
            for name in r["opt"]:
                opt_array_hist[name] += 1
            if r["sentinel_2c"] != 0xFFFFFFFF:
                sentinel_non_default += 1
            if r["has_ext_3c"]:
                ext_3c_count += 1
            guides_min = r["guides"] if guides_min is None else min(guides_min, r["guides"])
            guides_max = max(guides_max, r["guides"])
            ppg_min = r["ppg"] if ppg_min is None else min(ppg_min, r["ppg"])
            ppg_max = max(ppg_max, r["ppg"])

        for chain in chains:
            lod_depth_hist[len(chain)] += 1
            lod0 = chain[0]
            if lod0["tc0"] == lod0["s0"]:
                class0_exact_match += 1
            if lod0["tc1"] == lod0["s1"]:
                class1_exact_match += 1
            placement_hist[classify_placement(lod0["box"], lod0["longest"])] += 1
            if lod0["part_class"] is not None:
                part_class_hist[lod0["part_class"]] += 1
            elif lod0["part_class_hash"] is not None:
                part_class_hist[hex(lod0["part_class_hash"])] += 1
            if lod0["style_preset_hash"] is not None:
                style_preset_hist[hex(lod0["style_preset_hash"])] += 1
            if lod0["has_unset_rig_floats"]:
                unset_rig_float_assets += 1

        per_source.append({
            "source": os.path.basename(target),
            "lods": len(ok_rows),
            "assets": len(chains),
            "errors": len(err_rows),
            "bufferBytes": src_bytes,
            "strandsClass0": src_s0,
            "strandsClass1": src_s1,
        })

    return {
        "sources": per_source,
        "totals": {
            "lods": total_lods,
            "assets": total_assets,
            "errors": total_errors,
            "bufferBytes": total_buffer_bytes,
            "strandsClass0": total_s0,
            "strandsClass1": total_s1,
            "class0TriSumMatchesLod0": class0_exact_match,
            "class1TriSumMatchesLod0": class1_exact_match,
            "sentinel2cNonDefaultLods": sentinel_non_default,
            "ext3cLods": ext_3c_count,
            "unsetRigFloatAssets": unset_rig_float_assets,
            "stylePresetDistinctCount": len(style_preset_hist),
            "guidesRange": [guides_min or 0, guides_max],
            "nodesPerGuideRange": [ppg_min or 0, ppg_max],
        },
        "lodDepthHistogram": {str(k): v for k, v in sorted(lod_depth_hist.items())},
        "placementHistogram": dict(placement_hist.most_common()),
        "partClassHistogram": dict(part_class_hist.most_common()),
        "topStylePresets": dict(style_preset_hist.most_common(12)),
        "optionalArrays": dict(opt_array_hist.most_common()),
        "topFlags": dict(flags_hist.most_common(12)),
    }


# --- CLI ----------------------------------------------------------------------------------------------


def _parse_int(text: str) -> int:
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return value


def _parse_limit(text: str) -> int:
    """--limit: at least 1; 0 stopped before the first target, so census read as an empty corpus."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid integer: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _parse_classes(text: str) -> tuple[int, ...]:
    if not text.strip():
        return ()
    out: list[int] = []
    for part in text.split(","):
        val = int(part.strip())
        if val not in (0, 1):
            raise argparse.ArgumentTypeError(f"strand class must be 0 or 1, got {val}")
        if val not in out:
            out.append(val)
    return tuple(out)


def _format_inspect_text(summary: dict) -> str:
    part_label = summary["partClass"] or "unknown"
    part_hash = summary["partClassHash"] or "none"
    preset_hash = summary["stylePresetHash"] or "none"
    lines = [
        f"Source:        {summary['archive']} @ {summary['offset']}",
        f"Geometry:      {summary['guides']} guides x {summary['nodesPerGuide']} nodes "
        f"({summary['nodes']} nodes), {summary['triangles']} scalp triangles",
        f"Flags:         {summary['flags']}  (sentinel +0x2c={summary['sentinel2c']}, "
        f"ext_3c={summary['hasExt3c']}, unset_rig_floats={summary['hasUnsetRigFloats']})",
        f"Part / preset: part_class={part_label} ({part_hash})  style_preset={preset_hash}",
        f"Buffer:        {summary['bufferBytes']} B GPU, {summary['systemBytes']} B system, {summary['tailBytes']} B tail",
        f"Bounds (m):    min={summary['box'][0]}  max={summary['box'][1]}",
        f"Strands:       system={summary['strandTotals']}  tri_sum={summary['strandsFromTriangles']}",
        f"Guide length:  min={summary['guideLengthM'][0]} m  mean={summary['guideLengthM'][1]} m  "
        f"max={summary['guideLengthM'][2]} m  (max stored error {summary['storedLengthMaxErrorM']} m)",
        f"Length random: amount={summary['lengthRandom'][0]}  min_factor={summary['lengthRandom'][1]}",
        f"Seeds:         class0={summary['seeds'][0]}  class1={summary['seeds'][1]}",
        f"Regions:       {len(summary['regions'])}  Bone hashes: {summary['boneHashCount']}",
        f"Optional:      {', '.join(summary['optionalArrays']) or 'none'}",
    ]
    sp = summary.get("styleParams")
    if sp is not None:
        c0 = sp["curlFrizz0"]
        c1 = sp["curlFrizz1"]
        lines.append(
            f"Style (+0xe8): width={sp['strandWidth']}  longest={sp['longestGuideM']}/{sp['longestGuideAltM']} m  "
            f"braid={sp['braidParam']}  curl0(r={c0['radiiM']}, f={c0['frequenciesPerM']})  "
            f"curl1(r={c1['radiiM']}, f={c1['frequenciesPerM']})"
        )
    return "\n".join(lines)


def _format_census_text(report: dict) -> str:
    t = report["totals"]
    lines = [
        f"Groom census: {t['assets']} assets, {t['lods']} LOD resources, {t['errors']} decode errors",
        f"GPU buffers:  {t['bufferBytes']:,} bytes ({t['bufferBytes'] / (1024 * 1024):.1f} MiB)",
        f"Strands:      class 0 = {t['strandsClass0']:,} (tri-sum exact in {t['class0TriSumMatchesLod0']}/{t['assets']} LOD0), "
        f"class 1 = {t['strandsClass1']:,} (tri-sum exact in {t['class1TriSumMatchesLod0']}/{t['assets']} LOD0)",
        f"Guides range: {t['guidesRange'][0]}..{t['guidesRange'][1]}  "
        f"Nodes/guide range: {t['nodesPerGuideRange'][0]}..{t['nodesPerGuideRange'][1]}",
        f"LOD depths:   {report['lodDepthHistogram']}",
        f"Placement:    {report['placementHistogram']}",
        f"Part classes: {report['partClassHistogram']}  "
        f"(style presets: {t['stylePresetDistinctCount']} distinct, unset rig floats: {t['unsetRigFloatAssets']})",
    ]
    for src in report["sources"]:
        lines.append(
            f"  {src['source']}: {src['assets']} assets, {src['lods']} LODs, "
            f"{src['errors']} errors, {src['bufferBytes']:,} B"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decode, inspect, census and export RAGE smh GPU strand grooms (stdlib only)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ins = sub.add_parser("inspect", help="Inspect one groom LOD resource or carved .smh file.")
    p_ins.add_argument("path", help="Path to carved .smh/.bin file or RPF9 archive")
    p_ins.add_argument("--offset", type=_parse_int, default=None, help="Byte offset inside archive (decimal or 0x hex)")
    p_ins.add_argument("--json", action="store_true", help="Emit JSON summary")

    p_cen = sub.add_parser("census", help="Scan RPF9 archives, carved .smh files or directories for aggregate counts.")
    p_cen.add_argument("paths", nargs="+", help="One or more archive files, .smh files or directories")
    p_cen.add_argument("--json", action="store_true", help="Emit JSON report")
    p_cen.add_argument("--limit", type=_parse_limit, default=None, help="Stop after N groom LOD resources")

    p_glb = sub.add_parser("export-glb", help="Export a groom LOD to binary glTF 2.0 (.glb).")
    p_glb.add_argument("path", help="Path to carved .smh/.bin file or RPF9 archive")
    p_glb.add_argument("output", help="Output .glb file path")
    p_glb.add_argument("--offset", type=_parse_int, default=None, help="Byte offset inside archive (decimal or 0x hex)")
    p_glb.add_argument(
        "--geometry",
        choices=GEOMETRY_MODES,
        default="tubes",
        help="Strand/guide representation: tubes (default), ribbons, curves (LINES), or line-strip (LINE_STRIP)",
    )
    p_glb.add_argument("--curves", action="store_true", help="Shorthand for --geometry curves")
    p_glb.add_argument("--strands", type=_parse_int, default=6000, help="Maximum strands per class to export")
    p_glb.add_argument(
        "--classes",
        type=_parse_classes,
        default=(0, 1),
        help="Comma-separated placement classes to export (default 0,1; both sets are drawn together in one batch)",
    )
    p_glb.add_argument("--no-scalp", action="store_true", help="Omit scalp triangle mesh")
    p_glb.add_argument("--no-guides", action="store_true", help="Omit guide curves/tubes")
    p_glb.add_argument("--guide-radius", type=float, default=0.0006, help="Display tube/ribbon radius for guides (m)")
    p_glb.add_argument("--strand-radius", type=float, default=0.00035, help="Display tube/ribbon radius for strands (m)")
    p_glb.add_argument("--up", choices=("z", "y"), default="z", help="Source up axis (default z -> glTF Y-up)")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "inspect":
            groom = load_groom(args.path, offset=args.offset)
            summary = groom.summary()
            if args.json:
                print(json.dumps(summary, indent=2))
            else:
                print(_format_inspect_text(summary))
            return 0

        if args.command == "census":
            report = run_census(args.paths, limit=args.limit)
            if args.json:
                print(json.dumps(report, indent=2))
            else:
                print(_format_census_text(report))
            return 0

        if args.command == "export-glb":
            if os.path.exists(args.output) and os.path.exists(args.path) and os.path.samefile(args.path, args.output):
                raise GroomError(f"output must not overwrite the input file: {args.path}")
            groom = load_groom(args.path, offset=args.offset)
            meta = write_glb(
                args.output,
                groom,
                max_strands=args.strands,
                guide_radius=args.guide_radius,
                strand_radius=args.strand_radius,
                scalp=not args.no_scalp,
                guides=not args.no_guides,
                classes=args.classes,
                geometry=args.geometry,
                curves=args.curves,
                up_axis=args.up,
            )
            print(json.dumps({
                "output": args.output,
                "bytes": meta["bytes"],
                "geometry": meta["geometry"],
                "strandTotals": meta["strandTotals"],
                "strandsShown": meta["strandsShown"],
            }))
            return 0
    except (GroomError, glb_writer.GlbError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())
