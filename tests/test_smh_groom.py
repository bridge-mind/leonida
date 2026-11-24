"""Pure synthetic unit tests for tools/smh_groom.py (no game files, stdlib only).

Run with:
    python -I tests/test_smh_groom.py
"""
from __future__ import annotations

import gc
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import tracemalloc
import unittest
from unittest import mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_writer  # noqa: E402
import smh_groom  # noqa: E402


def _align16(data: bytes) -> bytes:
    pad = (-len(data)) % 16
    return data + (b"\x00" * pad)


def build_synthetic_groom_buffers(
    *,
    guides_pos: list[list[tuple[float, float, float]]] | None = None,
    triangles: list[tuple[int, int, int]] | None = None,
    tri_count: list[tuple[int, int, int, int]] | None = None,
    tri_region: list[int] | None = None,
    uvs: list[tuple[float, float]] | None = None,
    strand_totals: tuple[int, int] | None = None,
    flags: int = 0x443,
    length_random: tuple[float, float] = (0.0, 1.0),
    seeds: tuple[int, int] = (0x12345678, 0x89ABCDEF),
    regions: list[tuple[int, int, int, int, int]] | None = None,
    bone_hashes: tuple[int, ...] = (0x34C53E77, 0x11223344),
    style_hashes: tuple[int, ...] = (0x34C53E77, 0x89EA2ED6, 0x89EA2ED6, 0x89EA2ED6, 0x74792000, 0xD54A2D40),
    include_optional: bool = True,
    tail_bytes: bytes = b"",
    sentinel_2c: int = 0xFFFFFFFF,
    include_ext_3c: bool = False,
    unset_rig_floats: bool = False,
    style_params_f16: dict[int, float] | None = None,
) -> tuple[bytes, bytes]:
    """Build a synthetic (system_page, gpu_buffer) pair for unit testing."""
    if guides_pos is None:
        # 4 guides in a small square scalp patch, 4 nodes per guide growing straight along +Z.
        roots = [
            (-0.05, 0.00, 0.70),
            (0.05, 0.00, 0.70),
            (0.05, 0.05, 0.70),
            (-0.05, 0.05, 0.70),
        ]
        heights = (0.00, 0.04, 0.08, 0.12)
        guides_pos = [
            [(rx, ry, rz + dz) for dz in heights]
            for (rx, ry, rz) in roots
        ]
    guides = len(guides_pos)
    ppg = len(guides_pos[0])
    nodes = guides * ppg
    if triangles is None:
        triangles = [(0, 1, 2), (0, 2, 3)]
    tris = len(triangles)
    if tri_count is None:
        tri_count = [(12, 4, 0, 0) for _ in range(tris)]
    if tri_region is None:
        tri_region = [0 for _ in range(tris)]
    if uvs is None:
        uvs = [(i / max(guides - 1, 1), 0.5) for i in range(guides)]

    # Compute exact arc lengths per guide.
    guide_lengths: list[float] = []
    for g in guides_pos:
        acc = 0.0
        for j in range(ppg - 1):
            p0, p1 = g[j], g[j + 1]
            acc += math.hypot(p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
        guide_lengths.append(acc)

    gpu = bytearray(smh_groom.HEADER_SIZE)
    header = [0] * 40

    def append_array(hdr_off: int, payload: bytes) -> int:
        # 4-byte align each array.
        pad = (-len(gpu)) % 4
        if pad:
            gpu.extend(b"\x00" * pad)
        start = len(gpu)
        gpu.extend(payload)
        if hdr_off is not None:
            header[hdr_off // 4] = start
        return start

    # Required arrays.
    pos_vals: list[float] = []
    for g_idx, g in enumerate(guides_pos):
        for k_idx, (x, y, z) in enumerate(g):
            pos_vals.extend((x, y, z, float(k_idx) / max(ppg - 1, 1)))
    append_array(0x44, struct.pack(f"<{nodes * 4}e", *pos_vals))
    append_array(0x40, struct.pack(f"<{tris * 3}H", *(i for t in triangles for i in t)))
    append_array(0x48, bytes(c for row in tri_count for c in row))
    append_array(0x4C, bytes(tri_region))
    append_array(0x30, struct.pack(f"<{guides}e", *guide_lengths))
    append_array(0x08, struct.pack(f"<{guides * 2}e", *(v for uv in uvs for v in uv)))
    append_array(0x10, bytes([0, 1, 0, 0, 0, 0, 0, 0] * guides))
    append_array(0x14, bytes([192, 64, 0, 0, 0, 0, 0, 0] * guides))

    if include_optional:
        append_array(0x00, bytes([64, 128, 192, 255] * guides))
        append_array(0x04, bytes([32, 96, 160, 0] * guides))
        append_array(0x0C, bytes([64, 128, 191, 255] * guides))
        append_array(0x18, bytes(12 * guides))
        append_array(0x28, bytes([128, 128, 255, 0] * guides))
        append_array(0x34, bytes(4 * nodes))
        append_array(0x38, bytes(2 * nodes))

    if tail_bytes:
        append_array(smh_groom.TAIL_OFFSET, tail_bytes)
    if include_ext_3c:
        append_array(smh_groom.EXT_3C_OFFSET, b"\xAA" * 16)

    # Pad GPU buffer to multiple of 4 bytes (dword count in descriptor).
    pad = (-len(gpu)) % 4
    if pad:
        gpu.extend(b"\x00" * pad)

    header[smh_groom.SENTINEL_OFFSET // 4] = sentinel_2c
    header[0x50 // 4] = flags
    header[0x54 // 4] = (nodes << 8) | 0x01
    header[0x58 // 4] = guides | (tris << 16)
    header[0x5C // 4] = ppg
    lr_word = struct.unpack("<I", struct.pack("<2e", length_random[0], length_random[1]))[0]
    header[0x60 // 4] = lr_word
    header[0x7C // 4] = seeds[0]
    header[0x80 // 4] = seeds[1]
    struct.pack_into("<40I", gpu, 0, *header)

    # Build system page (>= 0x120 bytes + descriptor + bone hashes + region records).
    xs = [p[0] for g in guides_pos for p in g]
    ys = [p[1] for g in guides_pos for p in g]
    zs = [p[2] for g in guides_pos for p in g]
    if strand_totals is None:
        strand_totals = (sum(c[0] for c in tri_count), sum(c[1] for c in tri_count))
    if regions is None:
        regions = [(tris, guides, strand_totals[0], 0, strand_totals[1])]

    sysd = bytearray(0x120)
    struct.pack_into("<I", sysd, 0x1C, smh_groom.TYPE_ID)
    struct.pack_into("<3f", sysd, 0x30, min(xs), min(ys), min(zs))
    struct.pack_into("<3f", sysd, 0x40, max(xs), max(ys), max(zs))
    if unset_rig_floats:
        struct.pack_into("<4I", sysd, 0x50, *(smh_groom.FLT_MAX_U32,) * 4)
        struct.pack_into("<4I", sysd, 0x60, *(smh_groom.NEG_FLT_MAX_U32,) * 4)
    else:
        struct.pack_into("<4f", sysd, 0x50, min(xs), min(ys), min(zs), 0.0)
        struct.pack_into("<4f", sysd, 0x60, max(xs), max(ys), max(zs), 0.0)

    desc_off = len(sysd)
    sysd.extend(b"\x00" * 0x20)
    struct.pack_into("<Q", sysd, 0x70, smh_groom.SYS_BASE + desc_off)
    struct.pack_into("<IHH", sysd, desc_off + 8, len(gpu) // 4, 4, 8)
    struct.pack_into("<Q", sysd, desc_off + 0x18, smh_groom.GPU_BASE)

    if bone_hashes:
        bh_off = len(sysd)
        sysd.extend(struct.pack(f"<{len(bone_hashes)}I", *bone_hashes))
        struct.pack_into("<I", sysd, 0x98, len(bone_hashes))
        struct.pack_into("<Q", sysd, 0xA0, smh_groom.SYS_BASE + bh_off)

    if regions:
        reg_off = len(sysd)
        for n_tri, n_g, s0, rid, s1 in regions:
            sysd.extend(struct.pack("<HHHBBI", n_tri, n_g, s0, 0, rid, s1))
        struct.pack_into("<I", sysd, 0xA8, len(regions))
        struct.pack_into("<Q", sysd, 0xB0, smh_groom.SYS_BASE + reg_off)

    struct.pack_into("<6I", sysd, 0xB8, *style_hashes)
    struct.pack_into("<IIIHH", sysd, 0xD0, strand_totals[0], strand_totals[1], 0, tris, guides)

    if style_params_f16 is not None:
        for off_f16, val_f16 in style_params_f16.items():
            struct.pack_into("<e", sysd, off_f16, val_f16)

    return bytes(sysd), bytes(gpu)


def pack_native_resource(system: bytes, gpu: bytes, *, split_gpu_chunks: int = 1) -> bytes:
    """Pack (system, gpu) into native RAGE resource wire framing (16 zero prefix + 16-aligned zlib streams)."""
    out = bytearray(b"\x00" * 16)
    out.extend(_align16(zlib.compress(system)))
    if split_gpu_chunks <= 1:
        out.extend(_align16(zlib.compress(gpu)))
    else:
        step = max(1, (len(gpu) + split_gpu_chunks - 1) // split_gpu_chunks)
        for start in range(0, len(gpu), step):
            out.extend(_align16(zlib.compress(gpu[start:start + step])))
    return bytes(out)


def _reference_scatter_one(
    t: int,
    n_tris: int,
    per_tri: list[int],
    tri_region: bytes,
    density_scale: float,
    blocked: set[int],
    seed_word: int,
) -> tuple[int, bool, int, tuple[float, float, float]]:
    """Independent scalar reference implementation of smh_gen_1_CS_InterpolateStrands RNG."""
    m32 = 0xFFFFFFFF
    m24 = 0xFFFFFF
    ds32 = struct.unpack("<f", struct.pack("<f", density_scale))[0]
    v3 = (t * 0x19660D + 0xEF3EFB11) & m32
    tri = 0
    v20 = 0
    culled = False
    for attempt in range(4):
        s0 = (attempt * 0x19660D + 0x86945BED) & m32
        v11 = (s0 * 0x153E39CE + v3) & m32
        v5 = (v11 * 0x31CD17D1 + 0x153E39CE) & m32
        v9 = v11 >> 16
        v6 = (v5 * v11 + 0x31CD17D1) & m32
        v7 = (v6 * v5) & m32
        v5 ^= v5 >> 16
        v6 ^= v6 >> 16
        v7 = (v7 + s0) & m32
        v7 ^= v7 >> 16
        v8 = ((v9 ^ v11) + ((v7 * v5) & m32)) & m32
        v10 = (((v8 * v6) & m32) + v5) & m32
        pick = (v8 >> 16) % n_tris
        v5 = (((v10 * v8) & m32) + v6) & m32
        v6 = (((v5 * v10) & m32) + v7) & m32
        tri = pick
        rid = tri_region[pick]
        if rid != 0 and rid in blocked:
            culled = True
            continue
        culled = False
        v20 = v5 >> 8
        u = (v6 & m24) * (2.0 ** -24)
        p = struct.unpack("<f", struct.pack("<f", float(per_tri[pick]) * ds32))[0]
        p = min(max(p, 0.0), 1.0)
        if u < p:
            break
    seed = (0x3C1 * (tri & 0xFFFF) + ((seed_word * 0x8A30D) & m32) + ((v20 * 31) & m32)) & m32
    k = 0xC083C5
    x1 = (((k & m24) * (seed & m24)) + 0x3C6EF35F) & m32
    x2 = (((k & m24) * (x1 & m24)) + 0x3C6EF35F) & m32
    x1 = (((k & m24) * (x2 & m24)) + x1) & m32
    x2 = (((k & m24) * (x1 & m24)) + x2) & m32
    x1 ^= x1 >> 12
    x2 ^= x2 >> 12
    x1 = (((k & m24) * (x2 & m24)) + x1) & m32
    a = (x1 & m24) ^ (x1 >> 12)
    x1 = (((k & m24) * (x1 & m24)) + x2) & m32
    b = (x1 & m24) ^ (x1 >> 12)
    u1 = a * (2.0 ** -24)
    u2 = b * (2.0 ** -24)
    r = math.sqrt(u1)
    return tri, culled, seed, (1.0 - r, (1.0 - u2) * r, u2 * r)


class ParsingTests(unittest.TestCase):
    def test_system_page_signature_validation(self):
        sysd, _ = build_synthetic_groom_buffers()
        self.assertTrue(smh_groom.system_page_is_groom(sysd))

        # Too short
        self.assertFalse(smh_groom.system_page_is_groom(sysd[:0xDF]))
        # Non-zero vtable prefix
        bad_prefix = bytearray(sysd)
        bad_prefix[0] = 1
        self.assertFalse(smh_groom.system_page_is_groom(bad_prefix))
        # Wrong TYPE_ID
        bad_type = bytearray(sysd)
        struct.pack_into("<I", bad_type, 0x1C, 0xDEADBEEF)
        self.assertFalse(smh_groom.system_page_is_groom(bad_type))
        # Bad descriptor stride/flag/GPU base/word count
        desc = struct.unpack_from("<Q", sysd, 0x70)[0] - smh_groom.SYS_BASE
        for offset, fmt, val in (
            (desc + 8, "<I", 39),
            (desc + 12, "<H", 8),
            (desc + 14, "<H", 4),
            (desc + 0x18, "<Q", smh_groom.GPU_BASE + 16),
        ):
            mut = bytearray(sysd)
            struct.pack_into(fmt, mut, offset, val)
            self.assertFalse(smh_groom.system_page_is_groom(mut))

    def test_parse_full_groom_and_optional_sections(self):
        sysd, gpu = build_synthetic_groom_buffers(
            tri_count=[(10, 3, 0, 0), (14, 5, 0, 0)],
            tri_region=[1, 2],
            regions=[(1, 2, 10, 1, 3), (1, 2, 14, 2, 5)],
            length_random=(0.5, 0.4),
            tail_bytes=b"\x01\x02\x03\x04\x05\x06\x07\x08",
            sentinel_2c=0x0001FFFF,
            include_ext_3c=True,
        )
        groom = smh_groom.parse(sysd, gpu, archive="synth.rpf", offset=0x1000)
        self.assertEqual(groom.guides, 4)
        self.assertEqual(groom.nodes_per_guide, 4)
        self.assertEqual(groom.nodes, 16)
        self.assertEqual(groom.triangles_n, 2)
        self.assertEqual(groom.flags, 0x443)
        self.assertEqual(groom.strand_totals, (24, 8))
        self.assertEqual(groom.tri_count_totals(), [24, 8, 0, 0])
        self.assertEqual(list(groom.tri_region), [1, 2])
        self.assertEqual(len(groom.regions), 2)
        self.assertEqual(groom.regions[0]["id"], 1)
        self.assertEqual(groom.regions[1]["strands"], [14, 5])
        self.assertEqual(groom.bone_hashes, (0x34C53E77, 0x11223344))
        self.assertEqual(len(groom.style_hashes), 6)
        self.assertEqual(groom.sentinel_2c, 0x0001FFFF)
        self.assertTrue(groom.ext_3c_offset > 0)
        self.assertEqual(groom.tail, b"\x01\x02\x03\x04\x05\x06\x07\x08")
        self.assertAlmostEqual(groom.length_random[0], 0.5, places=3)
        self.assertAlmostEqual(groom.length_random[1], 0.4, places=3)

        # Normalized bone weights sum to 1.0 (192/256 = 0.75, 64/256 = 0.25).
        bw = groom.bone_weights()
        self.assertEqual(len(bw), 4)
        self.assertAlmostEqual(bw[0][0], 0.75, places=5)
        self.assertAlmostEqual(bw[0][1], 0.25, places=5)
        self.assertAlmostEqual(sum(bw[0]), 1.0, places=5)

        summary = groom.summary()
        self.assertEqual(summary["guides"], 4)
        self.assertEqual(summary["tailBytes"], 8)
        self.assertTrue(summary["hasExt3c"])
        self.assertLess(summary["storedLengthMaxErrorM"], 0.001)
        self.assertIn("ext_18", summary["optionalArrays"])
        self.assertIn("direction", summary["optionalArrays"])
        self.assertIn("node_4", summary["optionalArrays"])
        self.assertIn("node_2", summary["optionalArrays"])

    def test_parse_standalone_gpu_buffer_without_system_page(self):
        _, gpu = build_synthetic_groom_buffers(include_optional=False)
        groom = smh_groom.parse(None, gpu, archive="raw.smh", offset=0)
        self.assertEqual(groom.system, b"")
        self.assertEqual(groom.strand_totals, (24, 8))
        self.assertIsNone(groom.style_preset_hash)
        self.assertIsNone(groom.part_class_hash)
        self.assertIsNone(groom.part_class)
        self.assertFalse(groom.has_unset_rig_floats)
        self.assertIsNone(groom.style_params)
        self.assertAlmostEqual(groom.box[0][0], -0.05, places=2)
        self.assertAlmostEqual(groom.box[1][2], 0.82, places=2)

    def test_system_page_metadata_hashes_and_part_class_mapping(self):
        for part_hash, expected_name in smh_groom.PART_CLASS_NAMES.items():
            with self.subTest(part_hash=hex(part_hash), expected=expected_name):
                hashes = (
                    smh_groom.SKEL_HEAD_HASH,
                    smh_groom.UNUSED_HASH_FILL,
                    smh_groom.UNUSED_HASH_FILL,
                    smh_groom.UNUSED_HASH_FILL,
                    0x74792000,
                    part_hash,
                )
                sysd, gpu = build_synthetic_groom_buffers(style_hashes=hashes)
                parsed_sys = smh_groom.parse_system_page(sysd)
                self.assertEqual(parsed_sys["slot_bone_hashes"], hashes[:4])
                self.assertEqual(parsed_sys["style_preset_hash"], 0x74792000)
                self.assertEqual(parsed_sys["part_class_hash"], part_hash)
                self.assertEqual(parsed_sys["part_class"], expected_name)

                groom = smh_groom.parse(sysd, gpu)
                self.assertEqual(groom.style_preset_hash, 0x74792000)
                self.assertEqual(groom.part_class_hash, part_hash)
                self.assertEqual(groom.part_class, expected_name)
                summary = groom.summary()
                self.assertEqual(summary["stylePresetHash"], "0x74792000")
                self.assertEqual(summary["partClassHash"], hex(part_hash))
                self.assertEqual(summary["partClass"], expected_name)

        # Unmapped slot 5 hash leaves part_class as None while preserving raw hash.
        sysd_unk, gpu_unk = build_synthetic_groom_buffers(
            style_hashes=(smh_groom.SKEL_HEAD_HASH, 0x89EA2ED6, 0x89EA2ED6, 0x89EA2ED6, 0x12345678, 0xDEADBEEF),
        )
        groom_unk = smh_groom.parse(sysd_unk, gpu_unk)
        self.assertEqual(groom_unk.style_preset_hash, 0x12345678)
        self.assertEqual(groom_unk.part_class_hash, 0xDEADBEEF)
        self.assertIsNone(groom_unk.part_class)

    def test_system_page_unset_child_rig_floats(self):
        sysd_adult, gpu_adult = build_synthetic_groom_buffers(unset_rig_floats=False)
        groom_adult = smh_groom.parse(sysd_adult, gpu_adult)
        self.assertFalse(groom_adult.has_unset_rig_floats)
        self.assertFalse(groom_adult.summary()["hasUnsetRigFloats"])

        sysd_child, gpu_child = build_synthetic_groom_buffers(unset_rig_floats=True)
        parsed_child = smh_groom.parse_system_page(sysd_child)
        self.assertTrue(parsed_child["has_unset_rig_floats"])
        groom_child = smh_groom.parse(sysd_child, gpu_child)
        self.assertTrue(groom_child.has_unset_rig_floats)
        self.assertTrue(groom_child.summary()["hasUnsetRigFloats"])

    def test_system_page_half_float_style_parameters(self):
        style_f16 = {
            0xE8: 0.1,      # strand width
            0xEA: 0.28,     # longest guide (m)
            0xEE: 0.28,     # longest guide copy (m)
            0xF0: 0.75,     # braid-correlated parameter
            0xF8: 0.1,      # curl/frizz block 0: amount 0
            0xFA: 0.05,     # amount 1
            0xFC: 0.003,    # radius 0 (m)
            0xFE: 0.004,    # radius 1 (m)
            0x100: 12.75,   # frequency 0 (/m)
            0x102: 61.0,    # frequency 1 (/m)
            0x104: 0.08,    # curl/frizz block 1: amount 0
            0x106: 0.02,    # amount 1
            0x108: 0.003,   # radius 0 (m)
            0x10A: 0.0007,  # radius 1 (m, curly-correlated)
            0x10C: 70.0,    # frequency 0 (/m)
            0x10E: 74.5,    # frequency 1 (/m)
        }
        sysd, gpu = build_synthetic_groom_buffers(style_params_f16=style_f16)
        groom = smh_groom.parse(sysd, gpu)
        sp = groom.style_params
        self.assertIsNotNone(sp)
        self.assertAlmostEqual(sp["strandWidth"], 0.1, places=3)
        self.assertAlmostEqual(sp["longestGuideM"], 0.28, places=3)
        self.assertAlmostEqual(sp["longestGuideAltM"], 0.28, places=3)
        self.assertAlmostEqual(sp["braidParam"], 0.75, places=3)
        self.assertAlmostEqual(sp["curlFrizz0"]["amounts"][0], 0.1, places=3)
        self.assertAlmostEqual(sp["curlFrizz0"]["amounts"][1], 0.05, places=3)
        self.assertAlmostEqual(sp["curlFrizz0"]["radiiM"][0], 0.003, places=4)
        self.assertAlmostEqual(sp["curlFrizz0"]["radiiM"][1], 0.004, places=4)
        self.assertAlmostEqual(sp["curlFrizz0"]["frequenciesPerM"][0], 12.75, places=2)
        self.assertAlmostEqual(sp["curlFrizz0"]["frequenciesPerM"][1], 61.0, places=2)
        self.assertAlmostEqual(sp["curlFrizz1"]["amounts"][0], 0.08, places=3)
        self.assertAlmostEqual(sp["curlFrizz1"]["amounts"][1], 0.02, places=3)
        self.assertAlmostEqual(sp["curlFrizz1"]["radiiM"][0], 0.003, places=4)
        self.assertAlmostEqual(sp["curlFrizz1"]["radiiM"][1], 0.0007, places=4)
        self.assertAlmostEqual(sp["curlFrizz1"]["frequenciesPerM"][0], 70.0, places=1)
        self.assertAlmostEqual(sp["curlFrizz1"]["frequenciesPerM"][1], 74.5, places=1)

        # Non-finite half-float (NaN = 0x7E00) decodes cleanly as None without crashing.
        mut_nan = bytearray(sysd)
        struct.pack_into("<H", mut_nan, 0x10A, 0x7E00)
        parsed_nan = smh_groom.parse_system_page(mut_nan)
        self.assertIsNone(parsed_nan["style_params"]["curlFrizz1"]["radiiM"][1])

        # Minimal 0xe0-byte system page omits style_params safely.
        short_sys = bytearray(0xE0)
        short_sys[:0xE0] = sysd[:0xE0]
        struct.pack_into("<I", short_sys, 0x98, 0)
        struct.pack_into("<I", short_sys, 0xA8, 0)
        parsed_short = smh_groom.parse_system_page(short_sys)
        self.assertIsNone(parsed_short["style_params"])

    def test_native_resource_multi_chunk_and_scan_offsets(self):
        sys1, gpu1 = build_synthetic_groom_buffers(flags=0x440, seeds=(0x111, 0x222))
        sys2, gpu2 = build_synthetic_groom_buffers(flags=0x460, seeds=(0x333, 0x444))
        res1 = pack_native_resource(sys1, gpu1, split_gpu_chunks=3)
        res2 = pack_native_resource(sys2, gpu2, split_gpu_chunks=1)

        # Build a synthetic archive blob with header + noise + two 16-aligned resources.
        blob = bytearray(b"9FPR" + b"\xAB" * 0x3C)
        off1 = len(blob)
        blob.extend(res1)
        blob.extend(b"\x55" * 0x20)
        off2 = len(blob)
        blob.extend(res2)

        found = list(smh_groom.scan_offsets(bytes(blob)))
        self.assertEqual(found, [off1, off2])

        g1 = smh_groom.read_resource(blob, off1, archive="mem.rpf")
        g2 = smh_groom.read_resource(blob, off2, archive="mem.rpf")
        self.assertEqual(g1.flags, 0x440)
        self.assertEqual(g2.flags, 0x460)
        self.assertEqual(g1.seed(0), 0x111)
        self.assertEqual(g2.seed(1), 0x444)

    def test_scan_offsets_boundaries_alignment_and_memoryview(self):
        sys1, gpu1 = build_synthetic_groom_buffers(flags=0x440, seeds=(0x111, 0x222))
        sys2, gpu2 = build_synthetic_groom_buffers(flags=0x460, seeds=(0x333, 0x444))
        res1 = pack_native_resource(sys1, gpu1)
        res2 = pack_native_resource(sys2, gpu2)

        blob = bytearray(b"9FPR" + b"\xAB" * 0x3C)
        off1 = len(blob)
        blob.extend(res1)
        blob.extend(b"\x55" * 0x20)
        off2 = len(blob)
        blob.extend(res2)
        raw = bytes(blob)

        # Unaligned start rounds up to the next 16-byte boundary and never yields offsets < start.
        self.assertEqual(list(smh_groom.scan_offsets(raw, start=off1 + 1)), [off2])
        self.assertEqual(list(smh_groom.scan_offsets(raw, start=off1 + 15)), [off2])
        self.assertEqual(list(smh_groom.scan_offsets(raw, start=off1)), [off1, off2])
        self.assertEqual(list(smh_groom.scan_offsets(raw, start=-32)), [off1, off2])

        # Non-16-aligned chunk sizes must still check global 16-byte alignment and find every resource.
        for chunk in (1, 17, 73, 100):
            with self.subTest(chunk=chunk):
                self.assertEqual(list(smh_groom.scan_offsets(raw, chunk=chunk)), [off1, off2])

        # Adjacent [0, split) / [split, size) shards partition offsets without losing a straddling prefix.
        for split in (off1, off1 + 1, off1 + 16, off1 + 17, off2, off2 + 1, off2 + 16):
            with self.subTest(split=split):
                left = list(smh_groom.scan_offsets(raw, start=0, stop=split, chunk=73))
                right = list(smh_groom.scan_offsets(raw, start=split, stop=len(raw), chunk=73))
                self.assertTrue(all(o < split for o in left))
                self.assertTrue(all(o >= split for o in right))
                self.assertEqual(left + right, [off1, off2])

        # memoryview inputs work identically to bytes/bytearray (contiguous byte, typed, 2D, empty, and strided).
        self.assertEqual(list(smh_groom.scan_offsets(memoryview(raw), chunk=73)), [off1, off2])

        padded = raw + bytes((-len(raw)) % 4)
        for fmt in ("B", "h", "I"):
            with self.subTest(memoryview_fmt=fmt):
                self.assertEqual(list(smh_groom.scan_offsets(memoryview(padded).cast(fmt), chunk=73)), [off1, off2])
        self.assertEqual(
            list(smh_groom.scan_offsets(memoryview(padded).cast("B", shape=(2, len(padded) // 2)), chunk=73)),
            [off1, off2],
        )
        self.assertEqual(list(smh_groom.scan_offsets(memoryview(b"").cast("I"))), [])

        strided_buf = bytearray(2 * len(raw))
        strided_buf[::2] = raw
        strided_view = memoryview(strided_buf)[::2]
        self.assertEqual(list(smh_groom.scan_offsets(strided_view, chunk=73)), [off1, off2])

        strided_u32_buf = bytearray(2 * len(padded))
        mv_u32 = memoryview(strided_u32_buf).cast("I")
        mv_u32[::2] = memoryview(padded).cast("I")
        self.assertEqual(list(smh_groom.scan_offsets(mv_u32[::2], chunk=73)), [off1, off2])

        # A resource at a misaligned offset is never yielded, even when chunk alignment places it at block offset 0.
        misaligned = b"9FPR" + b"\xAB" * 0x40 + res1
        self.assertEqual(list(smh_groom.scan_offsets(misaligned, chunk=0x44)), [])

        # Non-positive chunk sizes are rejected with GroomError.
        for bad_chunk in (0, -16, False):
            with self.subTest(bad_chunk=bad_chunk):
                with self.assertRaises(smh_groom.GroomError):
                    list(smh_groom.scan_offsets(raw, chunk=bad_chunk))

    def test_load_groom_all_file_formats(self):
        sysd, gpu = build_synthetic_groom_buffers()
        native = pack_native_resource(sysd, gpu)
        raw_zlib = native[16:]
        rpf_like = b"9FPR" + b"\x00" * 0x3C + native

        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in (
                ("native.smh", native),
                ("zlib.smh", raw_zlib),
                ("gpu.smh", gpu),
                ("pack.rpf", rpf_like),
            ):
                p = os.path.join(tmp, name)
                with open(p, "wb") as f:
                    f.write(payload)
                g = smh_groom.load_groom(p)
                self.assertEqual(g.guides, 4, name)
                self.assertEqual(g.triangles_n, 2, name)

            # Explicit offset into archive.
            g_off = smh_groom.load_groom(os.path.join(tmp, "pack.rpf"), offset=0x40)
            self.assertEqual(g_off.offset, 0x40)

    def test_group_lod_chains_contiguous_and_split(self):
        box = ((-0.1, -0.1, 0.6), (0.1, 0.1, 0.85))
        other_box = ((-0.1, -0.1, 0.3), (0.1, 0.1, 0.45))

        def row(off, guides, s0=1000, s1=200, b=box, ppg=8, flags=0x443):
            return {"offset": off, "guides": guides, "ppg": ppg, "flags": flags, "s0": s0, "s1": s1, "box": b, "error": None}

        # Asset A is split around Asset B: A_lod0, B_lod0, B_lod1, A_lod1, A_lod2
        rows = [
            row(0x100, 120, b=box),
            row(0x200, 60, s0=500, s1=100, b=other_box),
            row(0x300, 36, s0=500, s1=100, b=other_box),
            row(0x400, 72, b=box),
            row(0x500, 44, b=box),
        ]
        chains = smh_groom.group_lod_chains(rows)
        self.assertEqual(len(chains), 2)
        self.assertEqual([r["guides"] for r in chains[0]], [120, 72, 44])
        self.assertEqual([r["guides"] for r in chains[1]], [60, 36])


class RngAndInterpolationTests(unittest.TestCase):
    def test_mul24_and_pcg_exact_vectors(self):
        # Top 8 bits are ignored by v_mad_u32_u24.
        self.assertEqual(smh_groom.mul24(0xFF000003, 0xAA000005), 15)
        self.assertEqual(smh_groom.mul24(0xFFFFFF, 0xFFFFFF), (0xFFFFFF * 0xFFFFFF) & 0xFFFFFFFF)

        # Verify PCG bit-exactness and [0, 1) range.
        for state in (0, 1, 0x12345678, 0xDEADBEEF, 0xFFFFFFFF):
            v = (state * 0x2C9277B5 + 0xAC564B05) & 0xFFFFFFFF
            w = (((v >> ((v >> 28) + 4)) ^ v) * 0x108EF2D9) & 0xFFFFFFFF
            expected_u24 = ((w >> 22) ^ (w & 0xFFFFFF)) & 0xFFFFFFFF
            self.assertEqual(smh_groom.pcg_u24(state), expected_u24)
            f = smh_groom.pcg_float(state)
            self.assertGreaterEqual(f, 0.0)
            self.assertLess(f, 1.0)
            self.assertEqual(f, expected_u24 * (2.0 ** -24))

    def test_class0_scatter_matches_independent_reference_bit_for_bit(self):
        sysd, gpu = build_synthetic_groom_buffers(
            triangles=[(0, 1, 2), (0, 2, 3), (1, 2, 3)],
            tri_count=[(30, 5, 0, 0), (10, 2, 0, 0), (50, 8, 0, 0)],
            tri_region=[0, 1, 2],
            seeds=(0x5EED1234, 0xCAFEBABE),
        )
        groom = smh_groom.parse(sysd, gpu)
        ds = 1.0 / 50.0
        res = smh_groom.scatter(groom, count=64, strand_class=0, density_scale=ds, region_mask={1})
        res2 = smh_groom.scatter(groom, count=64, strand_class=0, density_scale=ds, region_mask={1})
        self.assertEqual(res, res2)

        per_tri = [30, 10, 50]
        for t in range(64):
            e_tri, e_culled, e_seed, e_bary = _reference_scatter_one(
                t, 3, per_tri, bytes([0, 1, 2]), ds, {1}, 0x5EED1234,
            )
            self.assertEqual(res["tri"][t], e_tri)
            self.assertEqual(res["culled"][t], e_culled)
            self.assertEqual(res["seed"][t], e_seed)
            for c in range(3):
                self.assertAlmostEqual(res["bary"][t][c], e_bary[c], places=7)
            self.assertAlmostEqual(sum(res["bary"][t]), 1.0, places=6)
            self.assertTrue(all(w >= -1e-7 for w in res["bary"][t]))

    def test_region_zero_is_never_culled(self):
        sysd, gpu = build_synthetic_groom_buffers(tri_region=[0, 0])
        groom = smh_groom.parse(sysd, gpu)
        res = smh_groom.scatter(groom, count=32, strand_class=0, region_mask={0})
        self.assertFalse(any(res["culled"]))

        # Masking the only non-zero region culls every strand.
        sysd2, gpu2 = build_synthetic_groom_buffers(tri_region=[7, 7])
        groom2 = smh_groom.parse(sysd2, gpu2)
        res2 = smh_groom.scatter(groom2, count=32, strand_class=0, region_mask={7})
        self.assertTrue(all(res2["culled"]))

    def test_strand_interpolation_and_length_trimming(self):
        # Full-length strands (length_random = (0.0, 1.0)): exact barycentric blend.
        sysd, gpu = build_synthetic_groom_buffers(length_random=(0.0, 1.0))
        groom = smh_groom.parse(sysd, gpu)
        out = smh_groom.generate_strands(groom, count=40, strand_class=0)
        self.assertEqual(len(out["points"]), 40)
        for i in range(40):
            tri = out["tri"][i]
            b0, b1, b2 = out["bary"][i]
            g0, g1, g2 = groom.triangles[tri]
            for k in range(groom.nodes_per_guide):
                ex = b0 * groom.positions[g0][k][0] + b1 * groom.positions[g1][k][0] + b2 * groom.positions[g2][k][0]
                ey = b0 * groom.positions[g0][k][1] + b1 * groom.positions[g1][k][1] + b2 * groom.positions[g2][k][1]
                ez = b0 * groom.positions[g0][k][2] + b1 * groom.positions[g1][k][2] + b2 * groom.positions[g2][k][2]
                px, py, pz = out["points"][i][k]
                self.assertAlmostEqual(px, ex, places=4)
                self.assertAlmostEqual(py, ey, places=4)
                self.assertAlmostEqual(pz, ez, places=4)
            eu = b0 * groom.uvs[g0][0] + b1 * groom.uvs[g1][0] + b2 * groom.uvs[g2][0]
            ev = b0 * groom.uvs[g0][1] + b1 * groom.uvs[g1][1] + b2 * groom.uvs[g2][1]
            self.assertAlmostEqual(out["uv"][i][0], eu, places=5)
            self.assertAlmostEqual(out["uv"][i][1], ev, places=5)
            self.assertEqual(out["live"][i], groom.nodes_per_guide)
            self.assertAlmostEqual(out["length"][i], 0.12, places=3)

        # Trimmed strands (length_random = (1.0, 0.25)): lengths fall in [0.03, 0.12],
        # and trimmed strands repeat their tip node.
        sysd_t, gpu_t = build_synthetic_groom_buffers(length_random=(1.0, 0.25))
        groom_t = smh_groom.parse(sysd_t, gpu_t)
        trimmed = smh_groom.generate_strands(groom_t, count=80, strand_class=0)
        saw_trimmed = False
        for i in range(80):
            self.assertGreaterEqual(trimmed["length"][i], 0.029)
            self.assertLessEqual(trimmed["length"][i], 0.121)
            live = trimmed["live"][i]
            self.assertGreaterEqual(live, 2)
            self.assertLessEqual(live, groom_t.nodes_per_guide)
            if live < groom_t.nodes_per_guide:
                saw_trimmed = True
                tip = trimmed["points"][i][live - 1]
                for tail_k in range(live, groom_t.nodes_per_guide):
                    self.assertEqual(trimmed["points"][i][tail_k], tip)
        self.assertTrue(saw_trimmed, "expected at least one trimmed strand")

        # max_strands returns exact prefix.
        capped = smh_groom.generate_strands(groom_t, count=80, strand_class=0, max_strands=10)
        self.assertEqual(len(capped["points"]), 10)
        self.assertEqual(capped["points"], trimmed["points"][:10])
        self.assertEqual(capped["live"], trimmed["live"][:10])

    def test_scatter_max_accepted_skips_culled_and_matches_full(self):
        # 10 triangles where 9 are in blocked region 1 and 1 is in unblocked region 2,
        # ensuring early candidates are culled before reaching 1 or 5 accepted strands.
        triangles = [(0, 1, 2), (0, 2, 3)] * 5
        tri_count = [(20, 4, 0, 0)] * 10
        tri_region = [1] * 9 + [2]
        sysd, gpu = build_synthetic_groom_buffers(
            triangles=triangles,
            tri_count=tri_count,
            tri_region=tri_region,
            length_random=(0.5, 0.4),
            seeds=(0x5EED1234, 0xCAFEBABE),
        )
        groom = smh_groom.parse(sysd, gpu)

        # Non-positive max_accepted returns empty lists immediately.
        for cap in (0, -1):
            empty = smh_groom.scatter(groom, count=500, strand_class=0, region_mask={1}, max_accepted=cap)
            self.assertEqual(empty, {"tri": [], "culled": [], "bary": [], "seed": []})

        full = smh_groom.scatter(groom, count=500, strand_class=0, region_mask={1})
        self.assertEqual(len(full["tri"]), 500)
        accepted_idx = [i for i, c in enumerate(full["culled"]) if not c]
        self.assertGreaterEqual(len(accepted_idx), 5)
        self.assertGreater(accepted_idx[0], 0, "expected at least one culled candidate before first accepted")
        self.assertGreater(accepted_idx[4], 4, "expected culled candidates before fifth accepted")

        original_scatter = smh_groom.scatter
        for limit in (1, 5):
            with self.subTest(max_accepted=limit):
                capped_roots = smh_groom.scatter(
                    groom, count=500, strand_class=0, region_mask={1}, max_accepted=limit,
                )
                expected_idx = accepted_idx[:limit]
                for key in ("tri", "culled", "bary", "seed"):
                    self.assertEqual(len(capped_roots[key]), limit)
                self.assertFalse(any(capped_roots["culled"]))
                self.assertEqual(capped_roots["tri"], [full["tri"][i] for i in expected_idx])
                self.assertEqual(capped_roots["bary"], [full["bary"][i] for i in expected_idx])
                self.assertEqual(capped_roots["seed"], [full["seed"][i] for i in expected_idx])

                fast_strands = smh_groom.generate_strands(
                    groom, count=500, strand_class=0, region_mask={1}, max_strands=limit,
                )
                with mock.patch.object(
                    smh_groom,
                    "scatter",
                    lambda g, c, sc, ds, rm, _ma=None: original_scatter(g, c, sc, ds, rm, None),
                ):
                    full_strands = smh_groom.generate_strands(
                        groom, count=500, strand_class=0, region_mask={1}, max_strands=limit,
                    )
                self.assertEqual(fast_strands, full_strands)

        # When max_accepted exceeds total accepted, all accepted roots are returned in order.
        all_accepted = smh_groom.scatter(
            groom, count=500, strand_class=0, region_mask={1}, max_accepted=10000,
        )
        self.assertEqual(len(all_accepted["tri"]), len(accepted_idx))
        self.assertEqual(all_accepted["tri"], [full["tri"][i] for i in accepted_idx])

        # When every region is masked, max_accepted returns empty lists without error.
        none_accepted = smh_groom.scatter(
            groom, count=500, strand_class=0, region_mask={1, 2}, max_accepted=5,
        )
        self.assertEqual(none_accepted, {"tri": [], "culled": [], "bary": [], "seed": []})


class GlbExportTests(unittest.TestCase):
    def test_glb_export_all_geometry_modes(self):
        sysd, gpu = build_synthetic_groom_buffers(length_random=(0.5, 0.4))
        groom = smh_groom.parse(sysd, gpu)

        expected_modes = {
            "tubes": smh_groom.MODE_TRIANGLES,
            "ribbons": smh_groom.MODE_TRIANGLES,
            "curves": smh_groom.MODE_LINES,
            "line-strip": smh_groom.MODE_LINE_STRIP,
        }
        for geom, expected_mode in expected_modes.items():
            with self.subTest(geometry=geom):
                blob, meta = smh_groom.to_glb(groom, max_strands=12, geometry=geom)
                self.assertTrue(blob.startswith(b"glTF"))
                self.assertEqual(len(blob) % 4, 0)
                parsed = glb_writer.read_glb_summary(blob)
                self.assertEqual(parsed["version"], 2)
                self.assertEqual(parsed["bin_bytes"] % 4, 0)
                gltf = parsed["json"]
                prims = gltf["meshes"][0]["primitives"]
                self.assertGreaterEqual(len(prims), 3)
                # First primitive is scalp triangles (mode 4).
                self.assertEqual(prims[0]["mode"], smh_groom.MODE_TRIANGLES)
                self.assertEqual(prims[0]["extras"]["shaderIndex"], 0)
                self.assertIn("NORMAL", prims[0]["attributes"])
                # Subsequent primitives use the selected strand/guide mode.
                for prim in prims[1:]:
                    self.assertEqual(prim["mode"], expected_mode)
                    if geom in ("tubes", "ribbons"):
                        self.assertIn("NORMAL", prim["attributes"])
                    else:
                        self.assertNotIn("NORMAL", prim["attributes"])
                self.assertEqual(meta["geometry"], geom)
                self.assertEqual(meta["strandsShown"][0], 12)

    def test_glb_curves_shorthand_and_y_up_conversion(self):
        sysd, gpu = build_synthetic_groom_buffers()
        groom = smh_groom.parse(sysd, gpu)
        blob_z, _ = smh_groom.to_glb(groom, scalp=True, guides=False, classes=(), curves=True, up_axis="z")
        blob_y, _ = smh_groom.to_glb(groom, scalp=True, guides=False, classes=(), curves=True, up_axis="y")
        acc_z = glb_writer.read_glb_summary(blob_z)["json"]["accessors"][0]
        acc_y = glb_writer.read_glb_summary(blob_y)["json"]["accessors"][0]
        # RAGE root z=0.70 maps to glTF y=0.70 when up_axis="z", and stays z=0.70 when up_axis="y".
        self.assertAlmostEqual(acc_z["min"][1], 0.70, places=2)
        self.assertAlmostEqual(acc_y["min"][2], 0.70, places=2)

    def test_glb_nothing_to_export_raises(self):
        sysd, gpu = build_synthetic_groom_buffers()
        groom = smh_groom.parse(sysd, gpu)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.to_glb(groom, scalp=False, guides=False, classes=())

    def test_glb_max_strands_early_stop_byte_identical_and_bounded_memory(self):
        # 128 triangles x density 255 = 32,640 total candidates; max_strands=1 should
        # produce a byte-identical GLB while keeping scatter allocations O(max_strands).
        groom = smh_groom.parse(
            *build_synthetic_groom_buffers(
                triangles=[(0, 1, 2)] * 128,
                tri_count=[(255, 0, 0, 0)] * 128,
            )
        )
        self.assertEqual(groom.strand_totals[0], 32640)

        roots_capped = smh_groom.scatter(groom, count=32640, strand_class=0, max_accepted=1)
        for key in ("tri", "culled", "bary", "seed"):
            self.assertEqual(len(roots_capped[key]), 1)

        original_scatter = smh_groom.scatter
        if not tracemalloc.is_tracing():
            gc.collect()
            tracemalloc.start()
            try:
                blob_fast, meta_fast = smh_groom.to_glb(
                    groom, max_strands=1, classes=(0,), scalp=False, guides=False, geometry="curves",
                )
                peak_fast = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

            gc.collect()
            tracemalloc.start()
            try:
                with mock.patch.object(
                    smh_groom,
                    "scatter",
                    lambda g, c, sc, ds, rm, _ma=None: original_scatter(g, c, sc, ds, rm, None),
                ):
                    blob_full, meta_full = smh_groom.to_glb(
                        groom, max_strands=1, classes=(0,), scalp=False, guides=False, geometry="curves",
                    )
                peak_full = tracemalloc.get_traced_memory()[1]
            finally:
                tracemalloc.stop()

            self.assertLess(peak_fast * 10, peak_full)
        else:
            blob_fast, meta_fast = smh_groom.to_glb(
                groom, max_strands=1, classes=(0,), scalp=False, guides=False, geometry="curves",
            )
            with mock.patch.object(
                smh_groom,
                "scatter",
                lambda g, c, sc, ds, rm, _ma=None: original_scatter(g, c, sc, ds, rm, None),
            ):
                blob_full, meta_full = smh_groom.to_glb(
                    groom, max_strands=1, classes=(0,), scalp=False, guides=False, geometry="curves",
                )

        self.assertEqual(blob_fast, blob_full)
        self.assertEqual(meta_fast, meta_full)
        self.assertEqual(meta_fast["strandsShown"][0], 1)
        self.assertEqual(meta_fast["strandTotals"][0], 32640)


class ErrorHandlingTests(unittest.TestCase):
    def test_truncated_and_corrupt_buffers(self):
        sysd, gpu = build_synthetic_groom_buffers()

        # Truncated header
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.parse(sysd, gpu[:0x80])

        # Inconsistent node count
        bad_counts = bytearray(gpu)
        struct.pack_into("<I", bad_counts, 0x54, (999 << 8) | 1)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.parse(sysd, bad_counts)

        # Array offset pointing into header
        bad_off = bytearray(gpu)
        struct.pack_into("<I", bad_off, 0x44, 0x20)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.parse(sysd, bad_off)

        # Missing required array (position = 0)
        missing_pos = bytearray(gpu)
        struct.pack_into("<I", missing_pos, 0x44, 0)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.parse(sysd, missing_pos)

        # Truncated buffer cutting into array payload
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.parse(sysd, gpu[:-8])

        # Triangle referencing out-of-range guide
        bad_tri = bytearray(gpu)
        tri_off = struct.unpack_from("<I", gpu, 0x40)[0]
        struct.pack_into("<H", bad_tri, tri_off, 42)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.parse(sysd, bad_tri)

        # Non-finite node position (NaN half float = 0x7E00)
        bad_nan = bytearray(gpu)
        pos_off = struct.unpack_from("<I", gpu, 0x44)[0]
        struct.pack_into("<H", bad_nan, pos_off, 0x7E00)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.parse(sysd, bad_nan)

        # Bad zlib stream in read_resource
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.read_resource(b"\x00" * 16 + b"\x78\x9c\xff\xff\xff", 0)

        # Invalid strand class / negative count
        groom = smh_groom.parse(sysd, gpu)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.scatter(groom, strand_class=2)
        with self.assertRaises(smh_groom.GroomError):
            smh_groom.scatter(groom, count=-1)


class CliTests(unittest.TestCase):
    def test_cli_inspect_census_and_export_glb(self):
        sysd, gpu = build_synthetic_groom_buffers(
            unset_rig_floats=True,
            style_params_f16={0xE8: 0.1, 0xEA: 0.12, 0xEE: 0.12, 0xF0: 0.5, 0x10A: 0.003, 0x10C: 70.0},
        )
        native = pack_native_resource(sysd, gpu)
        rpf_like = b"9FPR" + b"\x00" * 0x3C + native

        with tempfile.TemporaryDirectory() as tmp:
            smh_path = os.path.join(tmp, "sample.smh")
            rpf_path = os.path.join(tmp, "sample.rpf")
            glb_path = os.path.join(tmp, "out.glb")
            with open(smh_path, "wb") as f:
                f.write(native)
            with open(rpf_path, "wb") as f:
                f.write(rpf_like)

            # inspect plain-text and --json via isolated Python subprocess
            proc_txt = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "smh_groom.py"), "inspect", smh_path],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("4 guides x 4 nodes", proc_txt.stdout)
            self.assertIn("part_class=main_hair (0xd54a2d40)", proc_txt.stdout)
            self.assertIn("style_preset=0x74792000", proc_txt.stdout)
            self.assertIn("unset_rig_floats=True", proc_txt.stdout)
            self.assertIn("Style (+0xe8):", proc_txt.stdout)

            proc = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "smh_groom.py"), "inspect", smh_path, "--json"],
                capture_output=True, text=True, check=True,
            )
            info = json.loads(proc.stdout)
            self.assertEqual(info["guides"], 4)
            self.assertEqual(info["triangles"], 2)
            self.assertEqual(info["partClass"], "main_hair")
            self.assertEqual(info["partClassHash"], "0xd54a2d40")
            self.assertEqual(info["stylePresetHash"], "0x74792000")
            self.assertTrue(info["hasUnsetRigFloats"])
            self.assertAlmostEqual(info["styleParams"]["strandWidth"], 0.1, places=3)
            self.assertAlmostEqual(info["styleParams"]["curlFrizz1"]["frequenciesPerM"][0], 70.0, places=1)

            # census --json across directory (finds both .rpf and .smh) and --limit 1
            proc_c = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "smh_groom.py"), "census", tmp, "--json"],
                capture_output=True, text=True, check=True,
            )
            cen = json.loads(proc_c.stdout)
            self.assertEqual(cen["totals"]["lods"], 2)
            self.assertEqual(cen["totals"]["errors"], 0)
            self.assertEqual(cen["totals"]["unsetRigFloatAssets"], 2)
            self.assertEqual(cen["totals"]["stylePresetDistinctCount"], 1)
            self.assertEqual(cen["partClassHistogram"], {"main_hair": 2})

            proc_lim = subprocess.run(
                [sys.executable, "-I", os.path.join(TOOLS, "smh_groom.py"), "census", tmp, "--limit", "1"],
                capture_output=True, text=True, check=True,
            )
            self.assertIn("1 LOD resources", proc_lim.stdout)
            self.assertIn("Part classes:", proc_lim.stdout)

            # export-glb --curves
            proc_g = subprocess.run(
                [
                    sys.executable, "-I", os.path.join(TOOLS, "smh_groom.py"),
                    "export-glb", rpf_path, glb_path, "--offset", "0x40", "--curves", "--strands", "10",
                ],
                capture_output=True, text=True, check=True,
            )
            meta = json.loads(proc_g.stdout)
            self.assertGreater(meta["bytes"], 100)
            self.assertEqual(meta["geometry"], "curves")
            with open(glb_path, "rb") as f:
                summary = glb_writer.read_glb_summary(f.read())
            self.assertEqual(summary["version"], 2)

            # Refusing to overwrite input
            import contextlib
            import io
            err_buf = io.StringIO()
            with contextlib.redirect_stderr(err_buf):
                bad_rc = smh_groom.main(["export-glb", smh_path, smh_path])
            self.assertEqual(bad_rc, 1)
            self.assertIn("must not overwrite", err_buf.getvalue())


    def test_cli_census_limit_below_one_is_refused(self):
        """--limit 0 used to stop at the first target and report an empty corpus with exit 0."""
        with tempfile.TemporaryDirectory() as tmp:
            for value in ("0", "-1"):
                with self.subTest(limit=value):
                    proc = subprocess.run(
                        [sys.executable, "-I", os.path.join(TOOLS, "smh_groom.py"),
                         "census", tmp, "--limit", value],
                        capture_output=True, text=True,
                    )
                    self.assertEqual(proc.returncode, 2)
                    self.assertIn("--limit", proc.stderr)
                    self.assertIn("at least 1", proc.stderr)


if __name__ == "__main__":
    unittest.main()
