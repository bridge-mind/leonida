"""Synthetic tests for tools/restofworld_model_join.py.

    python -I tests/test_restofworld_model_join.py -v

Every byte in this file is constructed synthetically in memory. No game archive,
carved payload, asset name or hash is used.
"""
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "restofworld_model_join.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import restofworld_model_join as joiner  # noqa: E402

SYS_BASE = joiner.SYS_BASE
GFX_BASE = joiner.GFX_BASE


def _align(value, step):
    return (value + step - 1) & ~(step - 1)


def _write_page_map(buf, at, sys_pages, gfx_pages):
    end = at + 16 + 8 * (sys_pages + gfx_pages)
    if len(buf) < end:
        buf.extend(b"\x00" * (end - len(buf)))
    buf[at:end] = b"\x00" * (end - at)
    buf[at + 8] = sys_pages
    buf[at + 9] = gfx_pages
    struct.pack_into("<Q", buf, 8, SYS_BASE + at)
    return end


def _build_vb_block(vcount, quantized=True):
    """Build a synthetic 0x4042 (8-byte pos) or 0x3063 (12-byte pos) vertex block."""
    if quantized:
        head_size = 128
        pos_size = 8
        tag = 0x4042
    else:
        head_size = 96
        pos_size = 12
        tag = 0x3063
    total = _align(head_size + vcount * pos_size, 16)
    buf = bytearray(total)
    struct.pack_into("<I", buf, 0, tag)
    for i in range(vcount):
        base = head_size + i * pos_size
        if quantized:
            struct.pack_into("<4H", buf, base, i + 1, i + 2, i + 3, 0)
        else:
            struct.pack_into("<3f", buf, base, float(i), float(i) * 0.5, 1.0)
    return bytes(buf), head_size, pos_size, (16 if quantized else 6)


def _build_meshlet_ib(clusters_spec, bone_palette=False, palette_no_marker=False):
    """Build a synthetic meshlet index buffer from [(c_verts, [(a, b, c), ...]), ...]."""
    clusters = len(clusters_spec)
    total_tris = sum(len(tris) for _, tris in clusters_spec)
    total_refs = sum(cv for cv, _ in clusters_spec)
    flags = 0x80000000 | clusters
    if bone_palette:
        flags |= 0x40000000
    tri_at = 12 + clusters * 16
    refs_at = tri_at + total_tris * 4
    pal_at = refs_at + total_refs * 2
    raw_end = pal_at + (clusters * 2 if bone_palette else 0)
    ib_bytes = max(32, _align(raw_end, 4))
    buf = bytearray(ib_bytes)
    struct.pack_into("<III", buf, 0, flags, total_tris, total_refs)

    v_acc = 0
    t_acc = 0
    for c_idx, (c_verts, tris) in enumerate(clusters_spec):
        c_tris = len(tris)
        rec = 12 + c_idx * 16
        packed = (c_verts - 1) | ((c_tris - 1) << 7)
        offsets = v_acc | (t_acc << 16)
        struct.pack_into("<IIII", buf, rec, packed, offsets, 0, 0)
        for ti, (i0, i1, i2) in enumerate(tris):
            t_word = (i0 & 0x7F) | ((i1 & 0x7F) << 7) | ((i2 & 0x7F) << 14) | (c_idx << 21)
            struct.pack_into("<I", buf, tri_at + (t_acc + ti) * 4, t_word)
        for vi in range(c_verts):
            struct.pack_into("<H", buf, refs_at + (v_acc + vi) * 2, v_acc + vi)
        if bone_palette:
            struct.pack_into("<H", buf, pal_at + c_idx * 2, 1)
        v_acc += c_verts
        t_acc += c_tris

    if bone_palette:
        binding = 0x80000000 | (total_refs << 8) | (0x00 if palette_no_marker else 0x80)
    else:
        binding = total_refs << 8
    return bytes(buf), clusters, total_tris, total_refs, binding


class _Arena:
    """Simple bump allocator for building synthetic segments."""

    def __init__(self, initial_size=0):
        self.buf = bytearray(initial_size)

    def alloc(self, size, align=16):
        pos = _align(len(self.buf), align)
        if pos > len(self.buf):
            self.buf.extend(b"\x00" * (pos - len(self.buf)))
        self.buf.extend(b"\x00" * size)
        return pos

    def put_bytes(self, payload, align=16):
        pos = self.alloc(len(payload), align=align)
        self.buf[pos:pos + len(payload)] = payload
        return pos

    def pad_to(self, target_size):
        if len(self.buf) < target_size:
            self.buf.extend(b"\x00" * (target_size - len(self.buf)))


def _populate_drawable(
    sys_arena,
    gfx_arena,
    d_off,
    embedded,
    label,
    lod_specs,
    bone_count=0,
    with_shader_cb=False,
    empty_txd=False,
    with_renderer=False,
):
    """Populate a synthetic rmcDrawable at `d_off` inside `sys_arena` and append buffers to `gfx_arena`."""
    struct.pack_into("<Q", sys_arena.buf, d_off + 0, 0)
    if embedded:
        struct.pack_into("<Q", sys_arena.buf, d_off + 8, 0)
    struct.pack_into("<f", sys_arena.buf, d_off + 44, 4.5)
    struct.pack_into("<4f", sys_arena.buf, d_off + 48, -2.0, -2.0, -1.0, 0.0)
    struct.pack_into("<4f", sys_arena.buf, d_off + 64, 2.0, 2.0, 3.0, 0.0)

    if label is not None:
        name_off = sys_arena.put_bytes(label + b"\x00", align=4)
        struct.pack_into("<Q", sys_arena.buf, d_off + 168, SYS_BASE + name_off)

    for lod_idx, geos in enumerate(lod_specs):
        lod_off = sys_arena.alloc(16, align=16)
        model_ptrs_off = sys_arena.alloc(8, align=8)
        model_off = sys_arena.alloc(56, align=16)
        geo_count = len(geos)
        geo_ptrs_off = sys_arena.alloc(geo_count * 8, align=8)
        aabbs_off = sys_arena.alloc((geo_count + (1 if geo_count > 1 else 0)) * 32, align=16)

        struct.pack_into("<Q", sys_arena.buf, d_off + 80 + lod_idx * 8, SYS_BASE + lod_off)
        struct.pack_into("<QHH", sys_arena.buf, lod_off, SYS_BASE + model_ptrs_off, 1, 1)
        struct.pack_into("<Q", sys_arena.buf, model_ptrs_off, SYS_BASE + model_off)
        struct.pack_into("<Q", sys_arena.buf, model_off + 8, SYS_BASE + geo_ptrs_off)
        struct.pack_into("<HHH", sys_arena.buf, model_off + 16, geo_count, geo_count, bone_count)
        struct.pack_into("<Q", sys_arena.buf, model_off + 24, SYS_BASE + aabbs_off)
        struct.pack_into("<H", sys_arena.buf, model_off + 54, geo_count)
        if bone_count > 0:
            bmap_off = sys_arena.alloc(bone_count * 4, align=4)
            struct.pack_into("<Q", sys_arena.buf, model_off + 40, SYS_BASE + bmap_off)

        for gi, gspec in enumerate(geos):
            vcount = gspec["vcount"]
            quantized = gspec.get("quantized", True)
            bone_pal = gspec.get("bone_palette", False)
            clusters_spec = gspec["clusters"]

            if gspec.get("align_gfx_64k"):
                gfx_arena.pad_to(_align(max(1, len(gfx_arena.buf)), 65536))

            vb_raw, pos_off, pos_size, pos_fmt = _build_vb_block(vcount, quantized=quantized)
            vb_gfx_off = gfx_arena.put_bytes(vb_raw, align=16)

            if gspec.get("align_ib_64k"):
                gfx_arena.pad_to(_align(max(1, len(gfx_arena.buf)), 65536))

            ib_raw, n_clusters, n_tris, _n_refs, binding = _build_meshlet_ib(
                clusters_spec, bone_palette=bone_pal
            )
            ib_gfx_off = gfx_arena.put_bytes(ib_raw, align=16)

            geo_off = sys_arena.alloc(160, align=16)
            vb_off = sys_arena.alloc(80, align=16)
            fvf_off = sys_arena.alloc(320, align=16)
            desc_off = sys_arena.alloc(16, align=16)
            ib_off = sys_arena.alloc(64, align=16)
            srv_vb = sys_arena.alloc(32, align=16)
            srv_ib = sys_arena.alloc(32, align=16)

            struct.pack_into("<Q", sys_arena.buf, geo_ptrs_off + gi * 8, SYS_BASE + geo_off)
            struct.pack_into("<Q", sys_arena.buf, geo_off + 8, SYS_BASE + vb_off)
            struct.pack_into("<Q", sys_arena.buf, geo_off + 64, SYS_BASE + desc_off)

            vb_stride = 4
            struct.pack_into("<IH", sys_arena.buf, vb_off + 8, len(vb_raw) // vb_stride, vb_stride)
            struct.pack_into("<Q", sys_arena.buf, vb_off + 24, GFX_BASE + vb_gfx_off)
            struct.pack_into("<Q", sys_arena.buf, vb_off + 40, SYS_BASE + srv_vb)
            struct.pack_into("<Q", sys_arena.buf, vb_off + 64, SYS_BASE + fvf_off)

            struct.pack_into("<I", sys_arena.buf, fvf_off + 0, pos_off)
            sys_arena.buf[fvf_off + 0xD0] = pos_size
            sys_arena.buf[fvf_off + 0x104] = pos_fmt
            struct.pack_into("<I", sys_arena.buf, fvf_off + 0x138, (vcount << 10) | 1)

            struct.pack_into("<QHH", sys_arena.buf, desc_off, SYS_BASE + ib_off, n_clusters, n_tris)
            struct.pack_into("<I", sys_arena.buf, desc_off + 12, binding)

            struct.pack_into("<IH", sys_arena.buf, ib_off + 8, len(ib_raw) // 4, 4)
            struct.pack_into("<Q", sys_arena.buf, ib_off + 24, GFX_BASE + ib_gfx_off)
            struct.pack_into("<Q", sys_arena.buf, ib_off + 40, SYS_BASE + srv_ib)

    sg_off = sys_arena.alloc(64, align=16)
    struct.pack_into("<Q", sys_arena.buf, d_off + 16, SYS_BASE + sg_off)

    if with_shader_cb:
        cb_gfx = gfx_arena.put_bytes(b"\x00" * 64, align=16)
        sdesc = sys_arena.alloc(32, align=16)
        sbuf = sys_arena.alloc(64, align=16)
        struct.pack_into("<Q", sys_arena.buf, sg_off + 48, SYS_BASE + sdesc)
        struct.pack_into("<Q", sys_arena.buf, sdesc, SYS_BASE + sbuf)
        struct.pack_into("<IH", sys_arena.buf, sbuf + 8, 16, 4)
        struct.pack_into("<Q", sys_arena.buf, sbuf + 24, GFX_BASE + cb_gfx)

    if empty_txd:
        txd_off = sys_arena.alloc(64, align=16)
        struct.pack_into("<Q", sys_arena.buf, sg_off + 8, SYS_BASE + txd_off)
        struct.pack_into("<HH", sys_arena.buf, txd_off + 56, 0, 0)

    mat_ptrs = sys_arena.alloc(8, align=8)
    mat_off = sys_arena.alloc(64, align=16)
    minfo = sys_arena.alloc(16, align=8)
    tarr = sys_arena.alloc(8, align=8)
    tref = sys_arena.alloc(80, align=16)
    asset_hdr = sys_arena.alloc(32, align=16)
    struct.pack_into("<QHH", sys_arena.buf, sg_off + 16, SYS_BASE + mat_ptrs, 1, 1)
    struct.pack_into("<Q", sys_arena.buf, mat_ptrs, SYS_BASE + mat_off)
    struct.pack_into("<I", sys_arena.buf, mat_off + 4, 0x6D657461)
    struct.pack_into("<Q", sys_arena.buf, mat_off + 16, SYS_BASE + tarr)
    struct.pack_into("<Q", sys_arena.buf, mat_off + 32, SYS_BASE + minfo)
    sys_arena.buf[minfo + 1] = 1
    sys_arena.buf[minfo + 4] = 1
    struct.pack_into("<Q", sys_arena.buf, tarr, SYS_BASE + tref)
    struct.pack_into("<I", sys_arena.buf, asset_hdr, 6)
    struct.pack_into("<Q", sys_arena.buf, tref + 48, SYS_BASE + asset_hdr + 16)

    if with_renderer:
        rend = sys_arena.alloc(80, align=16)
        r_items = sys_arena.alloc(8, align=8)
        struct.pack_into("<Q", sys_arena.buf, d_off + 176, SYS_BASE + rend)
        struct.pack_into("<QHH", sys_arena.buf, rend, SYS_BASE + r_items, 2, 2)


def make_standalone_drawable_1x1():
    """1 system page + 1 graphics page standalone drawable with 2 LODs."""
    sys_a = _Arena(192)
    gfx_a = _Arena()
    _populate_drawable(
        sys_a,
        gfx_a,
        d_off=0,
        embedded=False,
        label=b"synth_unit_a",
        lod_specs=[
            [
                {
                    "vcount": 6,
                    "quantized": True,
                    "clusters": [
                        (3, [(0, 1, 2)]),
                        (3, [(0, 1, 2), (0, 2, 1)]),
                    ],
                }
            ],
            [
                {
                    "vcount": 3,
                    "quantized": False,
                    "clusters": [(3, [(0, 1, 2)])],
                }
            ],
        ],
        with_shader_cb=True,
        empty_txd=True,
        with_renderer=True,
    )
    pm_at = _align(len(sys_a.buf), 16)
    _write_page_map(sys_a.buf, pm_at, sys_pages=1, gfx_pages=1)
    return [bytes(sys_a.buf)], [bytes(gfx_a.buf)]


def make_drawable_dict_1x2():
    """1 system page + 2 graphics pages drawable dictionary with 2 children (1 bone-bound)."""
    sys_a = _Arena(64)
    gfx_a = _Arena()
    hashes_off = sys_a.alloc(8, align=8)
    ptrs_off = sys_a.alloc(16, align=8)
    d0 = sys_a.alloc(192, align=16)
    d1 = sys_a.alloc(192, align=16)

    struct.pack_into("<Q", sys_a.buf, 0, 0)
    struct.pack_into("<Q", sys_a.buf, 32, SYS_BASE + hashes_off)
    struct.pack_into("<HH", sys_a.buf, 40, 2, 2)
    struct.pack_into("<Q", sys_a.buf, 48, SYS_BASE + ptrs_off)
    struct.pack_into("<HH", sys_a.buf, 56, 2, 2)
    struct.pack_into("<II", sys_a.buf, hashes_off, 101, 202)
    struct.pack_into("<QQ", sys_a.buf, ptrs_off, SYS_BASE + d0, SYS_BASE + d1)

    _populate_drawable(
        sys_a,
        gfx_a,
        d_off=d0,
        embedded=True,
        label=b"synth_dict_c0",
        lod_specs=[[{"vcount": 4, "quantized": True, "clusters": [(4, [(0, 1, 2), (0, 2, 3)])]}]],
    )
    _populate_drawable(
        sys_a,
        gfx_a,
        d_off=d1,
        embedded=True,
        label=b"synth_dict_c1",
        lod_specs=[
            [
                {
                    "vcount": 4,
                    "quantized": True,
                    "bone_palette": True,
                    "align_gfx_64k": True,
                    "clusters": [(4, [(0, 1, 2), (1, 3, 2)])],
                }
            ]
        ],
        bone_count=3,
    )
    pm_at = _align(len(sys_a.buf), 16)
    _write_page_map(sys_a.buf, pm_at, sys_pages=1, gfx_pages=2)
    assert len(gfx_a.buf) > 65536
    return [bytes(sys_a.buf)], [bytes(gfx_a.buf[:65536]), bytes(gfx_a.buf[65536:])]


def make_fragment_2x2():
    """2 system pages + 2 graphics pages fragment whose page-map sits in system page 1."""
    sys_a = _Arena(176)
    gfx_a = _Arena()
    child_off = sys_a.alloc(192, align=16)

    struct.pack_into("<Q", sys_a.buf, 0, 0)
    struct.pack_into("<Q", sys_a.buf, 24, SYS_BASE + child_off)
    struct.pack_into("<4H", sys_a.buf, 48, 2, 1, 0, 1)

    _populate_drawable(
        sys_a,
        gfx_a,
        d_off=child_off,
        embedded=True,
        label=None,
        lod_specs=[
            [{"vcount": 3, "quantized": True, "clusters": [(3, [(0, 1, 2)])]}],
            [
                {
                    "vcount": 5,
                    "quantized": True,
                    "align_ib_64k": True,
                    "clusters": [(5, [(0, 1, 2), (0, 2, 3), (0, 3, 4)])],
                }
            ],
        ],
    )
    # Place the fragment root name and datResourceMap in the second 64 KiB system page.
    sys_a.pad_to(65536 + 32)
    frag_label_off = sys_a.put_bytes(b"synth_frag_0\x00", align=4)
    struct.pack_into("<Q", sys_a.buf, 168, SYS_BASE + frag_label_off)
    pm_at = _align(len(sys_a.buf), 16)
    _write_page_map(sys_a.buf, pm_at, sys_pages=2, gfx_pages=2)
    gfx_a.pad_to(max(len(gfx_a.buf), 65536 + 64))
    return [bytes(sys_a.buf[:65536]), bytes(sys_a.buf[65536:])], [bytes(gfx_a.buf[:65536]), bytes(gfx_a.buf[65536:])]


def make_texture_dict_1x1():
    """Standalone pgDictionary<grcTexture> (non-model) to verify clean skipping."""
    sys_a = _Arena(64)
    hashes = sys_a.alloc(4, align=8)
    ptrs = sys_a.alloc(8, align=8)
    tex = sys_a.alloc(128, align=16)
    struct.pack_into("<Q", sys_a.buf, 32, SYS_BASE + hashes)
    struct.pack_into("<HH", sys_a.buf, 40, 1, 1)
    struct.pack_into("<Q", sys_a.buf, 48, SYS_BASE + ptrs)
    struct.pack_into("<HH", sys_a.buf, 56, 1, 1)
    struct.pack_into("<Q", sys_a.buf, ptrs, SYS_BASE + tex)
    struct.pack_into("<Q", sys_a.buf, tex + 0x28, SYS_BASE + tex + 0x50)
    sys_a.buf[tex + 0x40] = 5
    pm_at = _align(len(sys_a.buf), 16)
    _write_page_map(sys_a.buf, pm_at, sys_pages=1, gfx_pages=1)
    return [bytes(sys_a.buf)], [b"\x99" * 128]


def make_0drp_page():
    buf = bytearray(96)
    struct.pack_into("<Q", buf, 8, SYS_BASE + 0x40)
    buf[0x10:0x14] = b"0DRP"
    struct.pack_into("<I", buf, 0x14, 122)
    return bytes(buf)


def make_bink2():
    buf = bytearray(96)
    buf[:4] = b"KB2n"
    struct.pack_into("<I", buf, 4, len(buf) - 8)
    struct.pack_into("<II", buf, 0x14, 64, 64)
    return bytes(buf)


def pack_synthetic_rpf(entries):
    """Pack top-level entries at 4 KiB + 0x10 boundaries into a synthetic RPF9 byte image.

    Each entry in `entries` is a list of stream payloads (`bytes`); Bink2 entries start
    with `KB2` and are stored raw, while resource streams are zlib-compressed and
    separated by 0..15 zero bytes inside an entry.
    """
    out = bytearray(0x1000)
    out[:4] = b"9FPR"
    struct.pack_into("<I", out, 4, 0x1010)
    out.extend(b"\x00" * 0x10)

    recorded = []
    for group in entries:
        # Advance to next (4096 * k + 0x10) boundary with at least 16 zero bytes before it.
        if (len(out) & 0xFFF) != 0x10 or out[-16:] != b"\x00" * 16:
            target = ((len(out) + 16 + 0xFFF) & ~0xFFF) + 0x10
            if target < len(out) + 16:
                target += 0x1000
            out.extend(b"\x00" * (target - len(out)))
        group_offsets = []
        for idx, raw in enumerate(group):
            if idx > 0:
                pad = (-len(out)) % 16
                out.extend(b"\x00" * pad)
            group_offsets.append(len(out))
            if raw[:3] == b"KB2":
                out.extend(raw)
            else:
                out.extend(zlib.compress(raw))
        recorded.append((group_offsets, group))

    out.extend(b"\x00" * 64)
    return bytes(out), recorded


class ModelJoinTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def _build_full_corpus_entries(self):
        sys1, gfx1 = make_standalone_drawable_1x1()
        sys2, gfx2 = make_drawable_dict_1x2()
        sys3, gfx3 = make_fragment_2x2()
        txd_s, txd_g = make_texture_dict_1x1()
        return [
            [make_bink2()],
            [make_0drp_page()],
            txd_s + txd_g,
            sys1 + gfx1,
            sys2 + gfx2,
            sys3 + gfx3,
        ]

    def test_joins_a_short_declared_final_system_page(self):
        system, graphics = make_standalone_drawable_1x1()
        first = bytearray(system[0].ljust(65536, b"\0"))
        _write_page_map(first, 65536 - 64, sys_pages=2, gfx_pages=1)
        struct.pack_into("<Q", first, 168, SYS_BASE + 65536 + 8)
        last = b"\0" * 8 + b"synth_tail\0".ljust(40, b"\0")
        archive, recorded = pack_synthetic_rpf([[b"unrelated".ljust(48, b"\0")], [bytes(first), last, *graphics]])
        rpf = os.path.join(self.root, "short_system.rpf")
        with open(rpf, "wb") as fh:
            fh.write(archive)

        resources, summary = joiner.scan_rpf(rpf)
        self.assertEqual(summary["joined_resources"], 1)
        self.assertEqual(summary["zlib_streams"], 3)
        self.assertEqual([s["offset"] for s in resources[0]["system_streams"]], recorded[1][0][:2])
        self.assertEqual(resources[0]["system_streams"][1]["size"], 48)

        invalid = bytearray(archive)
        checksum_end = recorded[1][0][1] + len(zlib.compress(last))
        invalid[checksum_end - 1] ^= 1
        with open(rpf, "wb") as fh:
            fh.write(invalid)
        _, rejected = joiner.scan_rpf(rpf)
        self.assertEqual(rejected["joined_resources"], 0)

    def test_joins_1x1_drawable_1x2_dict_and_2x2_fragment_in_rpf(self):
        archive_bytes, recorded = pack_synthetic_rpf(self._build_full_corpus_entries())
        rpf_path = os.path.join(self.root, "models.rpf")
        with open(rpf_path, "wb") as fh:
            fh.write(archive_bytes)

        resources, summary = joiner.scan_rpf(rpf_path)
        self.assertEqual(summary["bink2_files"], 1)
        self.assertEqual(summary["zlib_streams"], 12)
        self.assertEqual(summary["model_candidates"], 3)
        self.assertEqual(summary["joined_resources"], 3)
        self.assertEqual(summary["unjoined_candidates"], 0)
        self.assertEqual(summary["by_kind"], {"drawable": 1, "drawable_dict": 1, "fragment": 1})
        self.assertEqual(summary["page_shapes"], {"1+1": 1, "1+2": 1, "2+2": 1})
        self.assertEqual(summary["model_system_streams"], 4)
        self.assertEqual(summary["model_graphics_streams"], 5)
        self.assertEqual(summary["graphics_4240_starts_total"], 4)
        self.assertEqual(summary["graphics_4240_starts_joined"], 4)
        self.assertEqual(summary["drawables"], 4)
        self.assertEqual(summary["dictionary_children"], 2)
        self.assertEqual(summary["standalone_models"], 2)
        self.assertEqual(summary["fragments"], 1)
        self.assertEqual(summary["bone_bound_drawables"], 1)
        self.assertEqual(summary["primary_geometries"], 4)
        self.assertEqual(summary["primary_vertices"], 6 + 4 + 4 + 3)
        self.assertEqual(summary["primary_triangles"], 3 + 2 + 2 + 1)
        self.assertEqual(summary["all_lod_geometries"], 6)
        self.assertEqual(summary["all_lod_vertices"], (6 + 3) + (4 + 4) + (3 + 5))
        self.assertEqual(summary["all_lod_triangles"], (3 + 1) + (2 + 2) + (1 + 3))
        self.assertEqual(summary["texture_dictionaries_skipped"], 1)
        self.assertEqual([r["offset"] for r in resources], [recorded[3][0][0], recorded[4][0][0], recorded[5][0][0]])

        # Verify CLI JSON and CSV outputs contain LF endings only and zero synthetic label strings.
        out_json = os.path.join(self.root, "joined.json")
        out_csv = os.path.join(self.root, "joined.csv")
        rc_json = subprocess.run(
            [sys.executable, "-I", TOOL, rpf_path, out_json, "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rc_json.returncode, 0, rc_json.stderr)
        rc_csv = subprocess.run(
            [sys.executable, "-I", TOOL, rpf_path, out_csv, "--quiet"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(rc_csv.returncode, 0, rc_csv.stderr)

        for p in (out_json, out_csv):
            with open(p, "rb") as fh:
                raw = fh.read()
            self.assertNotIn(b"\r", raw)
            self.assertNotIn(b"synth_", raw)

        with open(out_json, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded["summary"]["joined_resources"], 3)
        self.assertEqual(len(loaded["resources"]), 3)

        with open(out_csv, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["kind"] for r in rows], ["drawable", "drawable_dict", "fragment"])
        self.assertEqual([r["complete"] for r in rows], ["1", "1", "1"])

    def test_carved_directory_matches_archive_join(self):
        archive_bytes, recorded = pack_synthetic_rpf(self._build_full_corpus_entries())
        rpf_path = os.path.join(self.root, "source.rpf")
        carved_dir = os.path.join(self.root, "carved")
        os.makedirs(carved_dir)
        with open(rpf_path, "wb") as fh:
            fh.write(archive_bytes)

        for offsets, payloads in recorded:
            for off, raw in zip(offsets, payloads):
                ext = ".bk2" if raw[:3] == b"KB2" else ".bin"
                with open(os.path.join(carved_dir, f"{off:08x}{ext}"), "wb") as fh:
                    fh.write(raw)

        _, rpf_summary = joiner.scan_rpf(rpf_path)
        _, dir_summary = joiner.scan_dir(carved_dir)
        for key in (
            "joined_resources",
            "unjoined_candidates",
            "by_kind",
            "page_shapes",
            "drawables",
            "dictionary_children",
            "bone_bound_drawables",
            "primary_geometries",
            "primary_vertices",
            "primary_triangles",
            "all_lod_geometries",
            "all_lod_vertices",
            "all_lod_triangles",
            "texture_dictionaries_skipped",
        ):
            self.assertEqual(dir_summary[key], rpf_summary[key], key)

    def test_empty_directory_writes_valid_json_and_csv(self):
        empty_dir = os.path.join(self.root, "empty")
        os.makedirs(empty_dir)
        out_json = os.path.join(self.root, "empty.json")
        out_csv = os.path.join(self.root, "empty.csv")
        self.assertEqual(joiner.main([empty_dir, out_json, "--quiet"]), 0)
        self.assertEqual(joiner.main([empty_dir, out_csv, "--quiet"]), 0)
        with open(out_json, "rb") as fh:
            raw_json = fh.read()
        with open(out_csv, "rb") as fh:
            raw_csv = fh.read()
        self.assertNotIn(b"\r", raw_json)
        self.assertNotIn(b"\r", raw_csv)
        self.assertEqual(json.loads(raw_json.decode("utf-8"))["summary"]["joined_resources"], 0)
        self.assertEqual(raw_csv.decode("utf-8").splitlines(), [",".join(joiner.COLUMNS)])

    def test_start_limit_and_max_inflate_windowing(self):
        archive_bytes, recorded = pack_synthetic_rpf(self._build_full_corpus_entries())
        rpf_path = os.path.join(self.root, "window.rpf")
        with open(rpf_path, "wb") as fh:
            fh.write(archive_bytes)

        # Restrict window to the standalone 1+1 drawable only.
        start = recorded[3][0][0]
        end = recorded[4][0][0] - 16
        resources, summary = joiner.scan_rpf(rpf_path, start=start, limit=end - start)
        self.assertEqual(summary["joined_resources"], 1)
        self.assertEqual(resources[0]["kind"], "drawable")

        # Restrict max_inflate below 64 KiB so the 64 KiB multi-page streams are skipped.
        _, capped = joiner.scan_rpf(rpf_path, max_inflate=32768)
        self.assertGreater(capped["skipped_candidates"], 0)
        self.assertEqual(capped["joined_resources"], 1)

    def test_rejects_missing_graphics_stream_and_exits_nonzero(self):
        sys1, _gfx1 = make_standalone_drawable_1x1()
        archive_bytes, _ = pack_synthetic_rpf([sys1])
        rpf_path = os.path.join(self.root, "missing_gfx.rpf")
        out_json = os.path.join(self.root, "missing_gfx.json")
        with open(rpf_path, "wb") as fh:
            fh.write(archive_bytes)

        rc = joiner.main([rpf_path, out_json, "--quiet"])
        self.assertEqual(rc, 1)
        with open(out_json, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        self.assertEqual(report["summary"]["joined_resources"], 0)
        self.assertEqual(report["summary"]["unjoined_candidates"], 1)
        self.assertFalse(report["resources"][0]["complete"])
        self.assertIn("missing declared graphics streams", report["resources"][0]["error"])

    def test_rejects_nonzero_internal_gap_unused_page_and_meshlet_mismatch(self):
        # 1. Non-zero gap between system and graphics stream inside RPF9.
        sys1, gfx1 = make_standalone_drawable_1x1()
        archive_bytes, recorded = pack_synthetic_rpf([sys1 + gfx1])
        mutated = bytearray(archive_bytes)
        gfx_off = recorded[0][0][1]
        # Insert a non-zero gap by placing an earlier stream end 4 bytes before gfx_off with 0xFF padding.
        sys_stored = len(zlib.compress(sys1[0]))
        if recorded[0][0][0] + sys_stored < gfx_off:
            mutated[gfx_off - 1] = 0xFF
        else:
            # Rebuild with explicit non-zero byte between sys and gfx.
            head = bytearray(archive_bytes[:recorded[0][0][0] + sys_stored])
            head.extend(b"\xff" * 16)
            head.extend(zlib.compress(gfx1[0]))
            head.extend(b"\x00" * 64)
            mutated = head
        gap_rpf = os.path.join(self.root, "bad_gap.rpf")
        with open(gap_rpf, "wb") as fh:
            fh.write(mutated)
        res_gap, sum_gap = joiner.scan_rpf(gap_rpf)
        self.assertEqual(sum_gap["joined_resources"], 0)
        self.assertEqual(sum_gap["unjoined_candidates"], 1)
        self.assertFalse(res_gap[0]["complete"])

        # 2. Unused trailing graphics page (declares gp=2, but all buffers sit in 64 KiB page 0).
        over_sys = bytearray(sys1[0])
        pm_off = struct.unpack_from("<Q", over_sys, 8)[0] - SYS_BASE
        _write_page_map(over_sys, pm_off, sys_pages=1, gfx_pages=2)
        gfx_64k = gfx1[0] + b"\x00" * (65536 - len(gfx1[0]))
        unused_rpf_bytes, _ = pack_synthetic_rpf([[bytes(over_sys), gfx_64k, b"\x42\x40\x00\x00" + b"\x00" * 124]])
        unused_rpf = os.path.join(self.root, "unused_page.rpf")
        with open(unused_rpf, "wb") as fh:
            fh.write(unused_rpf_bytes)
        res_unused, sum_unused = joiner.scan_rpf(unused_rpf)
        self.assertEqual(sum_unused["joined_resources"], 0)
        self.assertIn("final declared graphics page is not reached", res_unused[0]["error"])

        # 3. Non-64 KiB non-final graphics stream when gp=2.
        non64_bytes, _ = pack_synthetic_rpf([[bytes(over_sys), gfx1[0], b"\x42\x40\x00\x00" + b"\x00" * 124]])
        non64_rpf = os.path.join(self.root, "non64.rpf")
        with open(non64_rpf, "wb") as fh:
            fh.write(non64_bytes)
        res_non64, _ = joiner.scan_rpf(non64_rpf)
        self.assertIn("non-64KiB non-final graphics stream", res_non64[0]["error"])

        # 4. Meshlet header triangle count disagrees with system descriptor.
        bad_gfx = bytearray(gfx1[0])
        # Corrupt totalTris word in every meshlet group header inside bad_gfx.
        for pos in range(0, len(bad_gfx) - 12, 16):
            w0 = struct.unpack_from("<I", bad_gfx, pos)[0]
            if w0 == (0x80000000 | 2):
                struct.pack_into("<I", bad_gfx, pos + 4, 99)
                break
        with self.assertRaises(joiner.JoinError):
            joiner.inspect_model_resource(sys1[0], bytes(bad_gfx), [len(bad_gfx)])

    def test_mmap_closed_on_invalid_archive(self):
        bad_path = os.path.join(self.root, "not_rpf.bin")
        with open(bad_path, "wb") as fh:
            fh.write(b"NOPE" + b"\x00" * 60)

        closed = []
        real_mmap = joiner.mmap.mmap

        class TrackingMmap:
            def __init__(self, *args, **kwargs):
                self._m = real_mmap(*args, **kwargs)

            def __enter__(self):
                self._m.__enter__()
                return self

            def __exit__(self, *exc):
                closed.append(True)
                return self._m.__exit__(*exc)

            def __getitem__(self, item):
                return self._m[item]

        with unittest.mock.patch.object(joiner.mmap, "mmap", TrackingMmap):
            with self.assertRaises(ValueError):
                joiner.scan_rpf(bad_path)
        self.assertEqual(closed, [True])

    def test_cli_argument_validation(self):
        src_rpf = os.path.join(self.root, "in.rpf")
        with open(src_rpf, "wb") as fh:
            fh.write(b"9FPR\x10\x00\x00\x00" + b"\x00" * 64)
        carved_dir = os.path.join(self.root, "dir_src")
        os.makedirs(carved_dir)
        carved_bin = os.path.join(carved_dir, "00001010.bin")
        with open(carved_bin, "wb") as fh:
            fh.write(b"\x00" * 64)
        out_ok = os.path.join(self.root, "out.json")

        bad_invocations = [
            [src_rpf, out_ok, "--start", "-1"],
            [src_rpf, out_ok, "--limit", "0"],
            [src_rpf, out_ok, "--max-inflate", "0"],
            [carved_dir, out_ok, "--start", "0"],
            [carved_dir, out_ok, "--limit", "64"],
            [src_rpf, carved_dir],
            [src_rpf, os.path.join(self.root, "out.txt")],
            [src_rpf, os.path.join(self.root, "missing_dir", "out.json")],
            [src_rpf, src_rpf],
            [carved_dir, carved_bin],
        ]
        for argv in bad_invocations:
            proc = subprocess.run(
                [sys.executable, "-I", TOOL] + argv,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 2, (argv, proc.stderr))
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(os.path.exists(out_ok))


if __name__ == "__main__":
    unittest.main()
