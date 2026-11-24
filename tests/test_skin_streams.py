"""Synthetic-data tests for tools/skin_streams.py (ASSET-12, the joint-index and skin-weight format).

    python -I tests/test_skin_streams.py

Every byte here is built by the test: a synthetic drawable system page, a synthetic graphics page and a
synthetic RPF9-style archive made of page-aligned zlib streams. No game file, value or name is used.
"""
import contextlib
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "skin_streams.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import skin_streams as ss  # noqa: E402


class Page:
    """A little big-endian-free allocator for the synthetic system page."""

    def __init__(self, size=0x1000):
        self.buf = bytearray(size)
        self.at = 0x180   # leave room for the drawable header

    def put(self, data):
        at = self.at
        self.buf[at:at + len(data)] = data
        self.at += len(data)
        return at

    def u16(self, at, value):
        struct.pack_into("<H", self.buf, at, value)

    def u32(self, at, value):
        struct.pack_into("<I", self.buf, at, value)

    def u64(self, at, value):
        struct.pack_into("<Q", self.buf, at, value)

    def ptr(self, at, offset):
        self.u64(at, 0x50000000 + offset)


def build_drawable(page, slots, vertices, bones):
    """Append one model/geometry/FVF chain to the page; returns (fvf_at, vb_at).

    `slots` maps semantic -> (size, fmt, offset, data bytes); slot 0 must be present as the position.
    """
    geometry = page.put(b"\0" * 160)
    vb = page.put(b"\0" * 80)
    fvf = page.put(b"\0" * 0x13C)
    lod = page.put(b"\0" * 16)
    model = page.put(b"\0" * 56)
    geo_array = page.put(struct.pack("<Q", 0x50000000 + geometry))
    model_array = page.put(struct.pack("<Q", 0x50000000 + model))
    # drawable: LOD pointer at +80
    page.ptr(80, lod)
    page.u16(lod + 8, 1)
    page.u16(lod + 10, 1)
    page.ptr(lod, model_array)
    page.u16(model + 16, 1)      # geometry count
    page.u16(model + 18, 1)
    page.u16(model + 54, 1)
    page.u16(model + 20, bones)  # local bone count
    page.ptr(model + 8, geo_array)
    page.ptr(geo_array, geometry)
    page.ptr(geometry + 8, vb)
    page.ptr(vb + 64, fvf)
    gfx_len = max(offset + len(data) for _s, _f, offset, data in slots.values()) + 64
    page.u32(vb + 8, gfx_len)
    page.u16(vb + 12, 1)
    page.u64(vb + 24, 0x60000000)
    page.u32(fvf + 0x138, (vertices << 10) | 1)
    for semantic, (size, fmt, offset, _data) in slots.items():
        page.u32(fvf + 4 * semantic, offset)
        page.buf[fvf + 0xD0 + semantic] = size
        page.buf[fvf + 0x104 + semantic] = fmt
    return fvf, vb, gfx_len


def build_resource(slots, vertices, bones, skip_descriptor=None):
    """Build a full synthetic drawable resource: system page bytes and graphics page bytes."""
    page = Page()
    # header: shader group pointer at +16, radius at +44, bounds at +48
    page.ptr(16, page.put(b"\0" * 64))
    struct.pack_into("<f", page.buf, 44, 1.0)
    struct.pack_into("<3f", page.buf, 48, -1.0, -1.0, -1.0)
    struct.pack_into("<3f", page.buf, 64, 1.0, 1.0, 1.0)
    _fvf, _vb, gfx_len = build_drawable(page, slots, vertices, bones)
    # pages directory: +8 system pages, +9 graphics pages
    directory = page.put(b"\0" * 16)
    page.buf[directory + 8] = 1
    page.buf[directory + 9] = 1
    page.ptr(8, directory)
    page.u64(168, 0)
    gfx = bytearray(gfx_len)
    # native buffer descriptors at the head: one word per slot, offset in bits 7.., size in bits 3..5
    for w, semantic in enumerate(s for s in sorted(slots) if s != skip_descriptor):
        size, _fmt, offset, _data = slots[semantic]
        struct.pack_into("<I", gfx, 4 * w, (offset << 7) | (size << 3) | (semantic & 15))
    for semantic, (_size, _fmt, offset, data) in slots.items():
        gfx[offset:offset + len(data)] = data
    return bytes(page.buf), bytes(gfx)


def quantized_position(count, offset):
    """Slot-0 stream: 8-byte quantized positions (format 16)."""
    return b"\x42" + b"\0" * (offset - 1) + b"".join(struct.pack("<Q", 0) for _ in range(count)) \
        if offset > 8 else b"".join(struct.pack("<Q", 0) for _ in range(count))


def weighted_mesh(layout="weighted-4inf-8bit", vertices=8, bones=6, bad_sum=False, bad_index=False,
                  missing_descriptor=False):
    """A synthetic weighted mesh; returns (system, graphics, mesh) with the mesh's expected layout."""
    influence8 = "8inf" in layout
    packed = layout.endswith("packed")
    sixteen = layout.endswith("16bit")
    ifmt, isz = (3, 16) if packed else ((12, 8) if sixteen else (30, 4))
    w16, w17, i20, i21 = bytearray(), bytearray(), bytearray(), bytearray()
    for v in range(vertices):
        row = [200, 40, 10, 5] + ([0, 0, 0, 0] if influence8 else [])
        if bad_sum and v == 0:
            row = [100, 40, 10, 5] + ([0, 0, 0, 0] if influence8 else [])
        w16 += bytes(row[:4])
        if influence8:
            w17 += bytes(row[4:])
        first = [1, 2, 3, 4]
        if bad_index and v == 1:
            first = [1, 2, 3, bones + 3]

        def pack(values):
            if packed:
                high = 0x3F57000  # the unidentified high field; the reader masks it away
                return struct.pack("<4I", *[(values[c] & 0xFFF) | (high if c == 0 else 0)
                                             for c in range(4)])
            if sixteen:
                return struct.pack("<4H", *values)
            return bytes(values)

        i20 += pack(first)
        if influence8:
            i21 += pack([0, 0, 0, 0])
    slots = {0: (8, 16, 128, quantized_position(vertices, 128)),
             16: (4, 28, 96, bytes(w16)),
             20: (isz, ifmt, 96 + 8 * vertices, bytes(i20))}
    if influence8:
        slots[17] = (4, 28, 96 + 4 * vertices, bytes(w17))
        slots[21] = (isz, ifmt, 96 + 8 * vertices + isz * vertices, bytes(i21))
    if missing_descriptor:
        slots[24] = (4, 28, 96 + 40 * vertices, b"\0" * (4 * vertices))
    skip = (20 if missing_descriptor else None)
    system, gfx = build_resource(slots, vertices, bones, skip_descriptor=skip)
    meshes = ss.walk_resource(system, "drawable")
    assert len(meshes) == 1, meshes
    return system, gfx, meshes[0]


def rigid_mesh(sixteen=False, vertices=8, bones=None, bad_index=False):
    bones = (400 if sixteen else 5) if bones is None else bones
    if not sixteen:
        row = bytes([3] * (vertices - 1) + [bones + 1 if bad_index else 3])
        slots = {0: (8, 16, 128, quantized_position(vertices, 128)), 20: (1, 62, 96, row)}
    else:
        stream = struct.pack(f"<{vertices}H", *([300] * (vertices - 1) + [bones + 7 if bad_index else 300]))
        slots = {0: (8, 16, 128, quantized_position(vertices, 128)), 20: (2, 57, 96, stream)}
    system, gfx = build_resource(slots, vertices, bones)
    meshes = ss.walk_resource(system, "drawable")
    assert len(meshes) == 1, meshes
    return system, gfx, meshes[0]


class TestClassify(unittest.TestCase):
    def test_layout_names(self):
        for layout in ("weighted-4inf-8bit", "weighted-8inf-8bit", "weighted-4inf-packed",
                       "weighted-8inf-packed", "weighted-4inf-16bit", "weighted-8inf-16bit"):
            _s, _g, mesh = weighted_mesh(layout)
            self.assertEqual(ss.classify(mesh), layout)
        for sixteen in (False, True):
            _s, _g, mesh = rigid_mesh(sixteen)
            self.assertEqual(ss.classify(mesh), "rigid-16bit" if sixteen else "rigid-8bit")

    def test_unknown_weight_format_is_unclassified_not_an_error(self):
        system, _gfx, _mesh = weighted_mesh(vertices=5)
        # find the FVF table by its flags word, then break slot 16's format byte: the census must
        # report unclassified, not raise
        flags = (5 << 10) | 1
        fvf = next(at for at in range(0x180, len(system) - 0x13C)
                   if struct.unpack_from("<I", system, at + 0x138)[0] == flags
                   and system[at + 0xD0 + 16] == 4)
        patched = bytearray(system)
        patched[fvf + 0x104 + 16] = 99   # unknown format, size unchanged
        self.assertIsNone(ss.classify(ss.walk_resource(bytes(patched), "drawable")[0]))

    def test_size0_placeholder_slots_are_absent(self):
        _s, _g, mesh = weighted_mesh("weighted-4inf-8bit")
        self.assertNotIn(17, mesh.slots)
        self.assertNotIn(21, mesh.slots)


class TestResourceKindDictionary(unittest.TestCase):
    """Dictionary headers follow the pinned rpf9-explorer's DrawableDictionaryCheckBytes.

    Every byte here is built by the test; no package bytes are used.
    """

    @staticmethod
    def dictionary(pages=0x50000000 + 0x100):
        b = bytearray(192)
        struct.pack_into("<Q", b, 8, pages)
        struct.pack_into("<Q", b, 32, 0x50000000 + 0x200)
        struct.pack_into("<Q", b, 48, 0x50000000 + 0x300)
        for at in (40, 42, 56, 58):
            struct.pack_into("<H", b, at, 2)
        return bytes(b)

    def test_null_page_pointer_is_not_a_dictionary(self):
        self.assertIsNone(ss.resource_kind(self.dictionary(pages=0)))

    def test_non_system_page_pointer_is_not_a_dictionary(self):
        self.assertIsNone(ss.resource_kind(self.dictionary(pages=0x12345678)))

    def test_system_page_pointer_is_a_dictionary(self):
        self.assertEqual(ss.resource_kind(self.dictionary()), "dictionary")


class TestResourceKindDictionaryChild(unittest.TestCase):
    """The dictionary signature follows the pinned rpf9-explorer's DrawableDictionaryCheckBytes.

    When the child-pointer array named by +48 is inside the supplied prefix, its first entry must also
    be a system pointer. Every byte here is built by the test; no package bytes are used.
    """

    @staticmethod
    def dictionary(child_at=0x80, first_child=0x50000000 + 0x300):
        b = bytearray(192)
        struct.pack_into("<Q", b, 0, 0)
        struct.pack_into("<Q", b, 8, 0x50000000 + 0x180)
        struct.pack_into("<Q", b, 32, 0x50000000 + 0x200)
        struct.pack_into("<Q", b, 48, 0x50000000 + child_at)
        for at in (40, 42, 56, 58):
            struct.pack_into("<H", b, at, 2)
        if child_at <= len(b) - 8:
            struct.pack_into("<Q", b, child_at, first_child)
        return bytes(b)

    def test_null_first_child_is_not_a_dictionary(self):
        self.assertIsNone(ss.resource_kind(self.dictionary(first_child=0)))

    def test_non_system_first_child_is_not_a_dictionary(self):
        self.assertIsNone(ss.resource_kind(self.dictionary(first_child=0x12345678)))

    def test_system_first_child_is_a_dictionary(self):
        self.assertEqual(ss.resource_kind(self.dictionary()), "dictionary")

    def test_child_array_beyond_prefix_is_still_a_dictionary(self):
        self.assertEqual(ss.resource_kind(self.dictionary(child_at=0x100)), "dictionary")


class TestValidate(unittest.TestCase):
    def test_weighted_8bit_clean(self):
        _s, gfx, mesh = weighted_mesh("weighted-8inf-8bit", vertices=16, bones=6)
        t = ss.validate(mesh, gfx, "weighted-8inf-8bit")
        self.assertEqual(t["weight-sum-255"], 16)
        self.assertEqual(t["weight-sum-other"], 0)
        self.assertEqual(t["components-out-of-range"], 0)
        self.assertEqual(t["descriptor-mismatch"], 0)
        self.assertEqual(t["weighted-components"], 16 * 4)   # slot-16 weights are all non-zero

    def test_weighted_problems_are_counted(self):
        _s, gfx, mesh = weighted_mesh("weighted-4inf-8bit", bad_sum=True, bad_index=True,
                                      missing_descriptor=True)
        t = ss.validate(mesh, gfx, "weighted-4inf-8bit")
        self.assertEqual(t["weight-sum-other"], 1)
        self.assertEqual(t["components-out-of-range"], 1)
        self.assertGreaterEqual(t["descriptor-mismatch"], 1)

    def test_zero_weight_component_index_is_ignored(self):
        # a zero-weight component carries an out-of-range index and must not be counted as a violation
        system, gfx = build_resource({
            0: (8, 16, 128, quantized_position(4, 128)),
            16: (4, 28, 96, bytes([255, 0, 0, 0] * 4)),
            20: (4, 30, 96 + 16, bytes([1, 200, 200, 200] * 4))}, 4, 6)
        mesh = ss.walk_resource(system, "drawable")[0]
        t = ss.validate(mesh, gfx, "weighted-4inf-8bit")
        self.assertEqual(t["components-out-of-range"], 0)
        self.assertEqual(t["weighted-components"], 4)

    def test_packed_twelve_bit_mask(self):
        _s, gfx, mesh = weighted_mesh("weighted-4inf-packed", vertices=8, bones=6)
        t = ss.validate(mesh, gfx, "weighted-4inf-packed")
        self.assertEqual(t["components-out-of-range"], 0)      # the high field is masked away
        self.assertEqual(t["packed-bits-9-11-set"], 0)
        self.assertEqual(t["weight-sum-255"], 8)

    def test_packed_index_uses_bit_8_when_bones_exceed_255(self):
        slots = {0: (8, 16, 128, quantized_position(2, 128)),
                 16: (4, 28, 96, bytes([255, 0, 0, 0] * 2)),
                 20: (16, 3, 96 + 8, struct.pack("<4I", 300, 0, 0, 0) + struct.pack("<4I", *([0] * 4)))}
        system, gfx = build_resource(slots, 2, 400)
        mesh = ss.walk_resource(system, "drawable")[0]
        self.assertEqual(ss.classify(mesh), "weighted-4inf-packed")
        t = ss.validate(mesh, gfx, "weighted-4inf-packed")
        self.assertEqual(t["components-out-of-range"], 0)
        self.assertEqual(t["packed-bit-8-set"], 1)             # index 300 needs bit 8

    def test_rigid(self):
        for sixteen in (False, True):
            _s, gfx, mesh = rigid_mesh(sixteen)
            layout = "rigid-16bit" if sixteen else "rigid-8bit"
            t = ss.validate(mesh, gfx, layout)
            self.assertEqual(t["components-out-of-range"], 0)
            _s2, gfx2, mesh2 = rigid_mesh(sixteen, bad_index=True)
            self.assertEqual(ss.validate(mesh2, gfx2, layout)["components-out-of-range"], 1)

    def test_short_graphics_buffer(self):
        _s, gfx, mesh = weighted_mesh()
        self.assertEqual(ss.validate(mesh, gfx[:10], "weighted-4inf-8bit").get("buffer-short"), 1)

    def test_weight_sum_min_max(self):
        _s, gfx, mesh = weighted_mesh(bad_sum=True)
        t = ss.validate(mesh, gfx, "weighted-4inf-8bit")
        self.assertEqual(t["weight-sum-min"], 155)
        self.assertEqual(t["weight-sum-max"], 255)

    def test_weight_sum_min_max_without_255_vertex(self):
        # All vertices sum below 255 (200) or above 255 (265): min/max must not be clamped to 255.
        s_low, g_low = build_resource({
            0: (8, 16, 128, quantized_position(4, 128)),
            16: (4, 28, 96, bytes([100, 50, 30, 20] * 4)),
            20: (4, 30, 96 + 16, bytes([0, 1, 2, 3] * 4)),
        }, 4, 6)
        m_low = ss.walk_resource(s_low, "drawable")[0]
        t_low = ss.validate(m_low, g_low, "weighted-4inf-8bit")
        self.assertEqual((t_low["weight-sum-min"], t_low["weight-sum-max"]), (200, 200))

        s_high, g_high = build_resource({
            0: (8, 16, 128, quantized_position(4, 128)),
            16: (4, 28, 96, bytes([200, 40, 10, 5] * 4)),
            17: (4, 28, 96 + 16, bytes([10, 0, 0, 0] * 4)),
            20: (4, 30, 96 + 32, bytes([0, 1, 2, 3] * 4)),
            21: (4, 30, 96 + 48, bytes([0, 0, 0, 0] * 4)),
        }, 4, 6)
        m_high = ss.walk_resource(s_high, "drawable")[0]
        t_high = ss.validate(m_high, g_high, "weighted-8inf-8bit")
        self.assertEqual((t_high["weight-sum-min"], t_high["weight-sum-max"]), (265, 265))

    def test_non_graphics_vb_pointer_is_rejected(self):
        system, gfx, mesh = weighted_mesh("weighted-4inf-8bit", vertices=4, bones=6)
        patched = bytearray(system)
        pos = patched.index(struct.pack("<Q", 0x60000000))
        for bad_ptr in (0, 0x50000000, 0x60000000 - len(gfx) - 16, 0x60000000 + ss.MAX_GFX):
            with self.subTest(bad_ptr=hex(bad_ptr)):
                struct.pack_into("<Q", patched, pos, bad_ptr)
                with self.assertRaises(ss.Skip):
                    ss.walk_resource(bytes(patched), "drawable")
        neg_mesh = ss.Mesh(mesh.vertices, mesh.bones, mesh.slots, -len(gfx) - 16, mesh.gfx_len)
        padded_gfx = b"\x00" * 1024 + gfx + b"\x00" * 16
        self.assertEqual(ss.validate(neg_mesh, padded_gfx, "weighted-4inf-8bit"),
                         {"buffer-short": 1, "vertices": mesh.vertices})


def build_archive(system, gfx, unaligned_copy=None):
    """A synthetic RPF9-style file: page-aligned zlib streams, each preceded by 16 zero bytes."""
    out = bytearray(0x2000)
    def place(data, page):
        at = page + 16
        out[at:at + len(data)] = data
        return at, at + len(data)
    c_sys = zlib.compress(system, 9)
    c_gfx = zlib.compress(gfx, 9)
    at1, end1 = place(c_sys, 0x1000)
    assert end1 <= 0x2000 - 16
    out += b"\0" * (0x3000 + 16 - len(out))
    at2, end2 = place(c_gfx, 0x3000)
    if unaligned_copy:
        out += b"\0" * (0x3800 - len(out))
        out[0x3800:0x3800 + len(unaligned_copy)] = unaligned_copy
    return bytes(out)


class TestScanArchive(unittest.TestCase):
    def run_scan(self, system, gfx, extra=b""):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "synthetic.rpf")
            with open(path, "wb") as fh:
                fh.write(build_archive(system, gfx, unaligned_copy=extra or None))
            stats = ss.scan_archive(path, 64, lambda _text: None)
        return stats

    def test_scan_finds_and_validates(self):
        system, gfx, _mesh = weighted_mesh("weighted-8inf-8bit", vertices=12, bones=7)
        stats = self.run_scan(system, gfx)
        self.assertEqual(stats["resources"], 1)
        self.assertEqual(stats["layouts"]["weighted-8inf-8bit"]["meshes"], 1)
        v = stats["validated"]["weighted-8inf-8bit"]
        self.assertEqual(v["weight-sum-255"], 12)
        self.assertEqual(v["components-out-of-range"], 0)
        self.assertEqual(v["descriptor-mismatch"], 0)

    def test_scan_counts_weighted_16_bit_joint_values(self) -> None:
        slots = {
            0: (8, 16, 96, bytes(24)),
            16: (4, 28, 128, bytes((255, 0, 0, 0)) * 3),
            20: (8, 12, 144, struct.pack("<12H", 300, 700, 0, 0,
                                        403, 700, 0, 0, 399, 700, 0, 0)),
        }
        system, gfx = build_resource(slots, vertices=3, bones=400)
        stats = self.run_scan(system, gfx)
        validated = stats["validated"]["weighted-4inf-16bit"]
        self.assertEqual(validated["descriptor-mismatch"], 0)
        self.assertEqual(validated["weighted-components"], 3)
        self.assertEqual(validated["components-out-of-range"], 1)

    def test_unaligned_resource_is_not_a_candidate(self):
        system, gfx, _mesh = rigid_mesh()
        stats = self.run_scan(system, gfx, extra=zlib.compress(system, 9))
        self.assertEqual(stats["resources"], 1)   # only the aligned copy counts

    def test_empty_archive_is_zero_resources_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "empty.rpf")
            open(path, "wb").close()
            stats = ss.scan_archive(path, 64, lambda _text: None)
            self.assertEqual((stats["resources"], stats["meshes"], stats["candidates"]), (0, 0, 0))
            self.assertEqual(ss.main([path]), 0)

    def test_rigid_scan(self):
        system, gfx, _mesh = rigid_mesh(sixteen=True)
        stats = self.run_scan(system, gfx)
        self.assertEqual(stats["layouts"].get("rigid-16bit", {"meshes": 0})["meshes"], 1)
        self.assertEqual(stats["validated"]["rigid-16bit"]["components-out-of-range"], 0)

    def test_scan_finds_fragment_with_null_optional_pointers(self):
        system, gfx, _mesh = weighted_mesh("weighted-8inf-8bit", vertices=12, bones=7)
        b = bytearray(system)
        for at in (16, 32, 40):
            struct.pack_into("<Q", b, at, 0)
        struct.pack_into("<Q", b, 24, 0x50000000)   # the embedded drawable is at offset 0
        struct.pack_into("<Q", b, 168, 0x50000000 + 0x180)
        for i, value in enumerate((1, 1, 1, 2)):
            struct.pack_into("<H", b, 48 + 2 * i, value)
        stats = self.run_scan(bytes(b), gfx)
        self.assertEqual(stats["resources"], 1)
        self.assertEqual(stats["layouts"]["weighted-8inf-8bit"]["meshes"], 1)


class TestResourceKind(unittest.TestCase):
    """Fragment headers follow the pinned rpf9-explorer decoder (breadbbbb/rpf9-explorer#8).

    Every header byte here is built by the test; no package bytes are used.
    """

    @staticmethod
    def fragment(fields=(1, 1, 1, 1), optional="all", drawable=True):
        b = bytearray(192)
        for at in (8, 24, 168):
            if at != 24 or drawable:
                struct.pack_into("<Q", b, at, 0x50000000 + 0x100)
        if optional == "all":
            for at in (16, 32, 40):
                struct.pack_into("<Q", b, at, 0x50000000 + 0x200)
        for i, value in enumerate(fields):
            struct.pack_into("<H", b, 48 + 2 * i, value)
        return bytes(b)

    def test_null_optional_pointers_are_a_fragment(self):
        self.assertEqual(ss.resource_kind(self.fragment(fields=(1, 1, 1, 2), optional="none")), "fragment")

    def test_unequal_part_fields_are_a_fragment(self):
        self.assertEqual(ss.resource_kind(self.fragment(fields=(1, 2, 3, 9))), "fragment")

    def test_reversed_part_fields_are_rejected(self):
        self.assertIsNone(ss.resource_kind(self.fragment(fields=(1, 1, 9, 3))))

    def test_zero_first_part_count_is_rejected(self):
        self.assertIsNone(ss.resource_kind(self.fragment(fields=(0, 0, 1, 1))))

    def test_non_null_non_system_optional_pointer_is_rejected(self):
        b = bytearray(self.fragment(optional="none"))
        struct.pack_into("<Q", b, 32, 0x12345678)
        self.assertIsNone(ss.resource_kind(bytes(b)))

    def test_null_drawable_pointer_is_rejected(self):
        self.assertIsNone(ss.resource_kind(self.fragment(drawable=False)))


class TestFormatTable(unittest.TestCase):
    def test_format_element_sizes_agree(self):
        for fmt, (_name, size) in ss.FORMATS.items():
            self.assertIn(size, (1, 2, 4, 8, 12, 16), fmt)

    def test_walk_rejects_format_size_mismatch(self):
        system, _gfx = build_resource({
            0: (8, 16, 128, quantized_position(2, 128)),
            16: (4, 30, 96, bytes([255, 0, 0, 0] * 2)),   # format 30 is 4 bytes; declare 4 -> ok
            20: (8, 30, 104, bytes(16))}, 2, 6)            # format 30 with 8 bytes must be refused
        with self.assertRaises(ss.Skip):
            ss.walk_resource(system, "drawable")


class TestCli(unittest.TestCase):
    @staticmethod
    def _archive(tmp, name="synthetic.rpf"):
        system, gfx, _mesh = weighted_mesh("weighted-8inf-8bit", vertices=12, bones=7)
        data = build_archive(system, gfx)
        path = os.path.join(tmp, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path, data

    def test_json_over_an_input_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = self._archive(tmp)
            other = os.path.join(tmp, ".", "synthetic.rpf")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", other],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual(r.stdout, "")
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_that_is_a_directory_is_refused_before_the_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", tmp],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("is a directory", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual(r.stdout, "")

    def test_json_in_a_missing_folder_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            out = os.path.join(tmp, "reports", "sub", "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", out],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as fh:
                report = json.load(fh)
                self.assertIn(path, report)
                self.assertEqual(report[path]["resources"], 1)

    def test_json_that_cannot_be_written_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            blocker = os.path.join(tmp, "blocker")
            with open(blocker, "w", encoding="utf-8") as fh:
                fh.write("a file, not a folder")
            out = os.path.join(blocker, "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", out],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("cannot write", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn("synthetic.rpf: 1 resources", r.stdout)

    def test_invalid_sample_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            for sample in ("0", "-1"):
                with self.subTest(sample=sample):
                    r = subprocess.run([sys.executable, "-I", TOOL, path, "--sample", sample],
                                       capture_output=True, text=True, errors="replace")
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertIn("--sample must be at least 1", r.stderr)
                    self.assertNotIn("Traceback", r.stderr)

    def test_unreadable_archive_reports_its_path_without_writing_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            out = os.path.join(tmp, "report.json")
            stderr = io.StringIO()
            error = PermissionError(13, "input is locked", path)
            with (mock.patch.object(ss, "open", create=True, side_effect=error),
                  contextlib.redirect_stderr(stderr)):
                self.assertEqual(ss.main([path, "--json", out]), 2)
            self.assertEqual(stderr.getvalue().splitlines(),
                             [f"error: cannot read {path}: input is locked"])
            self.assertFalse(os.path.exists(out))

    def test_missing_or_directory_input_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "missing.rpf")
            r = subprocess.run([sys.executable, "-I", TOOL, missing],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("not a file", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

            r2 = subprocess.run([sys.executable, "-I", TOOL, tmp],
                                capture_output=True, text=True, errors="replace")
            self.assertEqual(r2.returncode, 2, r2.stderr)
            self.assertIn("not a file", r2.stderr)
            self.assertNotIn("Traceback", r2.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
