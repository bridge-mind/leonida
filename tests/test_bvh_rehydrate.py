"""Synthetic dehydrated BLASes for tools/bvh_rehydrate.py: leaves, box groups and overrides, BOX16 rounding, the leaf
frame, the geometry transform, chunking, errors and the command line. No game data is read.

    python -I tests/test_bvh_rehydrate.py
"""
import contextlib
import io
import math
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bvh_rehydrate as R  # noqa: E402

BLAS_DEHYDRATED = 0x80000000 | R.FLAG_DEHYDRATED | 0x00020006


def header(total, n_leaves, bounds, nodes, root, elements, parents, compression=R.NO_TRANSFORM, flags=BLAS_DEHYDRATED):
    assert total % 64 == 0
    return struct.pack("<4I6f6I", flags, total // 64 - 1, 0x1234, n_leaves, *bounds, nodes, 0, root, elements,
                       parents, compression)


def leaf_chunk(leaves):
    """leaves: [(geometry, flags, prim0, prim1 or None, vertex order)] -> the five per-chunk arrays."""
    out = bytearray()
    for g, *_ in leaves:
        out += g.to_bytes(3, "little")
    for _, fl, *_ in leaves:
        out.append(fl)
    for _, _, p0, *_ in leaves:
        out += p0.to_bytes(3, "little")
    for _, _, _, p1, _ in leaves:
        out += (0xFFFFFF if p1 is None else p1).to_bytes(3, "little")
    for *_, order in leaves:
        out.append(order)
    return bytes(out)


def box_group(nodes, wide):
    """nodes: [(child pointers, {bit: stored value})] -> one compressed group."""
    words, kids, values = [], [], []
    for children, overrides in nodes:
        mask = 0
        for bit in sorted(overrides):
            mask |= 1 << bit
            values.append(overrides[bit])
        words.append(len(children) << 24 | mask)
        kids += children
    body = struct.pack("<%dI" % len(words), *words) + struct.pack("<%dI" % len(kids), *kids)
    return body + (struct.pack("<%df" % len(values), *values) if wide else struct.pack("<%dH" % len(values), *values))


def level_table(pairs):
    return b"".join(struct.pack("<2I", a, b) for a, b in pairs) + struct.pack("<I", len(pairs))


def ptr(offset, kind):
    return (offset >> 6) << 3 | kind


def build(leaves, root_wide=True, overrides=None, compression=R.NO_TRANSFORM, bounds=(0, 0, 0, 1, 1, 1),
          flags=BLAS_DEHYDRATED):
    """One box node over up to four leaves. Layout: header, parent table at 0x40, leaves at 0x100, node at 0x200."""
    total, elements, parents, nodes = 0x300, 0x100, 0x40, 0x200
    src = bytearray(total)
    leaf_ptrs = [ptr(elements + 64 * i, R.TRI0) for i in range(len(leaves))]
    root = ptr(nodes, R.BOX32 if root_wide else R.BOX16)
    src[0:64] = header(total, len(leaves), bounds, nodes, root, elements, parents, compression, flags)
    chunk = leaf_chunk(leaves)
    src[elements:elements + len(chunk)] = chunk
    grp = box_group([(leaf_ptrs, overrides or {})], root_wide)
    src[nodes:nodes + len(grp)] = grp
    lv = level_table([(128, 128)] if root_wide else [(0, 64)])
    src[total - len(lv):] = lv
    return bytes(src), root, leaf_ptrs


def floats(buf, at, n):
    return struct.unpack_from("<%df" % n, buf, at)


QUADS = R.Geometry(struct.pack("<12f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0), indices=struct.pack("<6I", 0, 1, 2, 2, 1, 3))


class TestLeaves(unittest.TestCase):
    def test_single_and_paired_leaf(self):
        # leaf 0: triangle 0 alone, vertex order 0x09 = (1, 2) -> slots (0, 1, 2); leaf 1: triangles 0 and 1 paired
        src, _, _ = build([(0, 0x01, 0, None, 0x09), (0, 0x81, 0, 1, 0x09)])
        out = R.rehydrate(src, [QUADS])
        self.assertEqual(floats(out, 0x100, 9), (0, 0, 0, 1, 0, 0, 0, 1, 0))
        self.assertTrue(all(math.isnan(x) for x in floats(out, 0x100 + 36, 3)))
        self.assertEqual(struct.unpack_from("<4I", out, 0x100 + 48), (0x0001, 0, R.INVALID, 0x009))
        # the paired leaf's fourth vertex is triangle 1's third index (neither 0x20 nor 0x80 in the order byte)
        self.assertEqual(floats(out, 0x140 + 36, 3), (1, 1, 0))
        self.assertEqual(struct.unpack_from("<4I", out, 0x140 + 48), (0x0081, 0, 1, 0x009))

    def test_vertex_order_moves_slots(self):
        # order 0x06: first = 2, second = 1, third = (2 ^ 3) - 1 = 0 -> vertices land in slots 1, 0, 2
        src, _, _ = build([(0, 0x01, 0, None, 0x06)])
        out = R.rehydrate(src, [QUADS])
        self.assertEqual(floats(out, 0x100, 9), (1, 0, 0, 0, 0, 0, 0, 1, 0))
        self.assertEqual(struct.unpack_from("<I", out, 0x100 + 60)[0], 0x006)

    def test_fourth_vertex_choice(self):
        for order, expected in ((0x29, (1, 0, 0)), (0x89, (0, 1, 0))):   # 0x20 picks index 0 of triangle 1, 0x80 index 1
            geo = R.Geometry(QUADS.vertices, indices=struct.pack("<6I", 0, 1, 2, 1, 2, 3))
            src, _, _ = build([(0, 0x81, 0, 1, order)])
            self.assertEqual(floats(R.rehydrate(src, [geo]), 0x100 + 36, 3), expected, hex(order))

    def test_no_indices_and_index16_and_stride(self):
        flat = struct.pack("<9f", 5, 0, 0, 6, 0, 0, 5, 1, 0)
        src, _, _ = build([(0, 0x01, 0, None, 0x09)])
        self.assertEqual(floats(R.rehydrate(src, [R.Geometry(flat)]), 0x100, 3), (5, 0, 0))
        padded = b"".join(struct.pack("<3fI", *struct.unpack_from("<3f", flat, 12 * i), 0xDEAD) for i in range(3))
        g16 = R.Geometry(padded, stride=16, indices=struct.pack("<3H", 2, 1, 0), index16=True)
        self.assertEqual(floats(R.rehydrate(src, [g16]), 0x100, 9), (5, 1, 0, 6, 0, 0, 5, 0, 0))

    def test_geometry_transform(self):
        rot_y = ((0.0, 0.0, 1.0, 0.0), (0.0, 1.0, 0.0, 0.0), (-1.0, 0.0, 0.0, 10.0))
        g = R.Geometry(struct.pack("<9f", 1, 2, 3, 1, 2, 3, 1, 2, 3), transform=rot_y)
        src, _, _ = build([(0, 0x01, 0, None, 0x09)])
        self.assertEqual(floats(R.rehydrate(src, [g]), 0x100, 3), (3, 2, 9))

    def test_second_chunk_starts_128_kib_later(self):
        n = R.LEAVES_PER_CHUNK + 1
        elements, total = 0x100, 0x100 + R.CHUNK_STRIDE + 0x100
        src = bytearray(total)
        src[0:64] = header(total, n, (0, 0, 0, 1, 1, 1), 0, ptr(elements, R.TRI0), elements, 0x40)
        first = leaf_chunk([(0, 0x01, 0, None, 0x09)] * R.LEAVES_PER_CHUNK)
        src[elements:elements + len(first)] = first
        second = leaf_chunk([(0, 0x01, 1, None, 0x09)])
        at = elements + R.CHUNK_STRIDE
        src[at:at + len(second)] = second
        g = R.Geometry(struct.pack("<18f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 7, 8, 9, 8, 8, 9, 7, 9, 9))
        out = R.rehydrate(bytes(src), [g])
        self.assertEqual(floats(out, at, 3), (7, 8, 9))
        self.assertEqual(floats(out, elements + 64 * (R.LEAVES_PER_CHUNK - 1), 3), (0, 0, 0))


class TestBoxes(unittest.TestCase):
    def test_box32_bounds_parents_and_header(self):
        src, root, leaf_ptrs = build([(0, 0x01, 0, None, 0x09), (0, 0x01, 1, None, 0x09)])
        out = R.rehydrate(src, [QUADS])
        node = 0x200
        self.assertEqual(struct.unpack_from("<4I", out, node), (leaf_ptrs[0], leaf_ptrs[1], R.INVALID, R.INVALID))
        self.assertEqual(floats(out, node + 16, 12), (0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0))
        self.assertEqual(floats(out, node + 64, 6), (R.FLT_MAX,) * 3 + (-R.FLT_MAX,) * 3)   # empty slot
        self.assertEqual(struct.unpack_from("<I", out, 0)[0], BLAS_DEHYDRATED & ~R.FLAG_DEHYDRATED)
        # the root's parent entry: unit (0x200 - 0x100) / 64 = 4 -> 0x40 + 16
        self.assertEqual(struct.unpack_from("<I", out, 0x40 + 16)[0], R.INVALID)
        # TRI0 leaves get a parent entry only in an updatable structure
        self.assertEqual(struct.unpack_from("<I", out, 0x40)[0], 0)
        upd, _, _ = build([(0, 0x01, 0, None, 0x09)], flags=BLAS_DEHYDRATED | R.FLAG_UPDATABLE)
        self.assertEqual(struct.unpack_from("<I", R.rehydrate(upd, [QUADS]), 0x40)[0], root)

    def test_box32_override_replaces_component(self):
        src, _, _ = build([(0, 0x01, 0, None, 0x09), (0, 0x01, 1, None, 0x09)], overrides={6 + 0: 0.25, 3: 0.75})
        out = R.rehydrate(src, [QUADS])
        self.assertEqual(floats(out, 0x200 + 16, 12), (0, 0, 0, 0.75, 1, 0, 0.25, 0, 0, 1, 1, 0))

    def test_box16_rounds_outward_and_overflows_to_inf(self):
        g = R.Geometry(struct.pack("<9f", 0.1, -0.1, 0, 70000.0, 0, 0, 0, 1, 0))
        src, _, _ = build([(0, 0x01, 0, None, 0x09)], root_wide=False)
        h = struct.unpack_from("<6H", R.rehydrate(src, [g]), 0x200 + 16)
        lo = [R.half_to_float(x) for x in h[:3]]
        self.assertLessEqual(lo[0], 0.0)
        self.assertLessEqual(lo[1], -0.1)
        self.assertEqual(h[3], R.HALF_POS_INF)        # 70000 is past the half range
        self.assertGreaterEqual(R.half_to_float(h[4]), 1.0)
        self.assertEqual(h[5], 0)

    def test_box16_override_and_half_helpers(self):
        src, _, _ = build([(0, 0x01, 0, None, 0x09)], root_wide=False, overrides={0: R.float_to_half(-2.0)})
        self.assertEqual(R.half_to_float(struct.unpack_from("<H", R.rehydrate(src, [QUADS]), 0x200 + 16)[0]), -2.0)
        self.assertEqual(R.half_to_float(R.half_down(1.0 + 2 ** -12)), 1.0)
        self.assertEqual(R.half_to_float(R.half_up(1.0 + 2 ** -12)), 1.0 + 2 ** -10)
        self.assertEqual(R.half_down(-70000.0), R.HALF_NEG_INF)
        self.assertEqual(R.half_to_float(R.half_up(1e-6)), 2 ** -14)


class TestParentTableBounds(unittest.TestCase):
    def test_console_offset_to_parents_zero_keeps_the_header(self):
        # a console structure keeps offsetToParents at 0, where a node in unit 0 or 4 puts its entry over the
        # header's flags word; only the entries from the header end on are stored (docs/bvh.md 3.1)
        src, _, _ = build([(0, 0x01, 0, None, 0x09)], flags=BLAS_DEHYDRATED | R.FLAG_UPDATABLE)
        src = bytearray(src)
        struct.pack_into("<I", src, 56, 0)        # offsetToParents
        out = R.rehydrate(bytes(src), [QUADS])
        kept_flags = (BLAS_DEHYDRATED | R.FLAG_UPDATABLE) & ~R.FLAG_DEHYDRATED
        self.assertEqual(struct.unpack_from("<I", out, 0)[0], kept_flags)
        self.assertEqual(struct.unpack_from("<I", out, 16)[0], 0)

    def test_stored_parent_entry_ends_at_offset_to_elements(self):
        hdr = R.Header(header(0x300, 1, (0, 0, 0, 1, 1, 1), 0x200, ptr(0x200, R.BOX32), 0x100, 0x40))
        self.assertEqual(hdr.stored_parent_entry(0x200, 0x300), 0x40 + 16)     # unit 4 is inside the table
        self.assertIsNone(hdr.stored_parent_entry(0x100 + 64 * 48, 0x300))     # entry == offsetToElements


class TestLeafFrame(unittest.TestCase):
    def test_scale_and_translation(self):
        # scale bytes 0x80, 0x81, 0x7f -> 2, 4, 1; translation on: x bounds [10, 12] -> centre 11, y and z straddle 0
        word = 0x80000000 | 0x40000000 | 0x7F << 16 | 0x81 << 8 | 0x80
        g = R.Geometry(struct.pack("<9f", 11, 1, 2, 12, 0, 0, 10, 0, 0))
        src, _, _ = build([(0, 0x01, 0, None, 0x09)], compression=word, bounds=(10, -1, -1, 12, 1, 3))
        self.assertEqual(floats(R.rehydrate(src, [g]), 0x100, 3), (0, 4, 2))

    def test_rotation_table(self):
        self.assertEqual(R.rotation(0), ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        r = R.rotation(16)              # index 16: x and y step 0, z step 1 (pi/8)
        self.assertAlmostEqual(r[0][0], math.cos(math.pi / 8), places=6)


class TestErrorsAndCli(unittest.TestCase):
    def test_errors(self):
        src, _, _ = build([(0, 0x01, 0, None, 0x09)])
        with self.assertRaisesRegex(ValueError, "geometry index 0"):
            R.rehydrate(src, [])
        plain = bytearray(src)
        struct.pack_into("<I", plain, 0, BLAS_DEHYDRATED & ~R.FLAG_DEHYDRATED)
        with self.assertRaisesRegex(ValueError, "dehydrated flag"):
            R.rehydrate(bytes(plain), [QUADS])
        with self.assertRaisesRegex(ValueError, "shorter"):
            R.rehydrate(src[:0x200], [QUADS])
        bad = bytearray(src)
        bad[-12:] = level_table([(100, 128)])
        with self.assertRaisesRegex(ValueError, "out of order"):
            R.rehydrate(bytes(bad), [QUADS])

    def test_header_offset(self):
        src, _, _ = build([(0, 0x01, 0, None, 0x09)])
        self.assertEqual(R.rehydrate(b"\0" * 0x100 + src, [QUADS], 0x100), R.rehydrate(src, [QUADS]))

    def test_command_line(self):
        src, _, _ = build([(0, 0x01, 0, None, 0x09), (0, 0x81, 0, 1, 0x09)])
        with tempfile.TemporaryDirectory() as d:
            paths = {k: os.path.join(d, k) for k in ("in.bin", "vb.bin", "ib.bin", "out.bin", "ref.bin")}
            for k, data in (("in.bin", src), ("vb.bin", QUADS.vertices), ("ib.bin", QUADS.indices),
                            ("ref.bin", R.rehydrate(src, [QUADS]))):
                with open(paths[k], "wb") as f:
                    f.write(data)
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                rc = R.main([paths["in.bin"], "--geom", paths["vb.bin"] + "," + paths["ib.bin"], "--out", paths["out.bin"],
                             "--compare", paths["ref.bin"]])
            self.assertEqual(rc, 0)
            self.assertIn("in 0 bytes", text.getvalue())
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(R.main([paths["in.bin"], "--out", paths["out.bin"]]), 2)
                self.assertEqual(R.main([paths["in.bin"], "--geom", paths["vb.bin"], "--out", paths["out.bin"],
                                         "--transform", "1,2,3"]), 2)

    def test_out_must_not_overwrite_an_input(self):
        src, _, _ = build([(0, 0x01, 0, None, 0x09), (0, 0x81, 0, 1, 0x09)])
        with tempfile.TemporaryDirectory() as d:
            paths = {k: os.path.join(d, k) for k in ("in.bin", "vb.bin", "ib.bin", "out.bin", "ref.bin")}
            for k, data in (("in.bin", src), ("vb.bin", QUADS.vertices), ("ib.bin", QUADS.indices),
                            ("ref.bin", R.rehydrate(src, [QUADS]))):
                with open(paths[k], "wb") as f:
                    f.write(data)
            originals = {k: open(paths[k], "rb").read() for k in ("in.bin", "vb.bin", "ib.bin", "ref.bin")}
            geo = paths["vb.bin"] + "," + paths["ib.bin"]
            for argv, victim in (
                    ([paths["in.bin"], "--geom", geo, "--out", paths["in.bin"]], "in.bin"),
                    ([paths["in.bin"], "--geom", geo, "--out", paths["vb.bin"]], "vb.bin"),
                    ([paths["in.bin"], "--geom", geo, "--out", paths["ref.bin"], "--compare", paths["ref.bin"]],
                     "ref.bin")):
                with self.subTest(victim=victim), contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(R.main(argv), 2)
                with open(paths[victim], "rb") as f:
                    self.assertEqual(f.read(), originals[victim], victim)
            alt = os.path.join(d, "IN.BIN")
            if os.path.exists(alt):     # case-insensitive volume: IN.BIN is in.bin, so it must refuse
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(R.main([paths["in.bin"], "--geom", geo, "--out", alt]), 2)
            text = io.StringIO()
            with contextlib.redirect_stdout(text):
                self.assertEqual(R.main([paths["in.bin"], "--geom", geo, "--out", paths["out.bin"]]), 0)
            self.assertIn("rehydrated", text.getvalue())


if __name__ == "__main__":
    unittest.main()
