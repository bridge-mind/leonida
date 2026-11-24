"""Synthetic-data tests for tools/meshlet_page.py.

    python -I tests/test_meshlet_page.py

Every byte here is built by the test. No game file, value or name is used.
"""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "meshlet_page.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import meshlet_page as mp  # noqa: E402


def pad16(buf):
    buf += b"\0" * (-len(buf) % 16)
    return buf


def block(n, colour=True, uv1=True, normal=True, flags=0x200):
    """A vertex block with n vertices: slots 0 (pos), 1 (rgba), 2 (normal), 3 (uv0), 4 (uv1).

    The streams are stored in slot order 0, 2, 1, 3, 4, as measured.
    """
    order = [(0, 0x63, 12), (2, 0x63, 12), (1, 0x26, 4), (3, 0x42, 8), (4, 0x42, 8)]
    if not normal:
        order = [o for o in order if o[0] != 2]
    if not colour:
        order = [o for o in order if o[0] != 1]
    if not uv1:
        order = [o for o in order if o[0] != 4]
    head = bytearray(0x60)
    off = 0x60
    body = bytearray()
    for slot, tag, stride in order:
        struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
        for v in range(n):
            if slot == 0:
                body += struct.pack("<3f", v * 0.5, -v * 0.25, 1.0)
            elif slot == 1:
                body += bytes((v & 0xFF, 0, 0, 255))
            elif slot == 2:
                body += struct.pack("<3f", 0.0, 0.0, 1.0)
            else:
                body += struct.pack("<2f", v / max(n, 1), 0.5)
        off += stride * n
    struct.pack_into("<I", head, 0x5C, flags)
    return bytes(head + body)


def other_block(n, tags=((0, 0x42, 8), (2, 0x21, 4), (11, 0x27, 4), (1, 0x26, 4), (12, 0x26, 4), (3, 0x28, 4),
                         (4, 0x28, 4), (14, 0x28, 4)), axis=2, bad_size=False, packed=None, bias=(0.0, 0.0, 0.0),
                scale=(0.0, 0.0, 0.0)):
    """A `42 40 00 00` vertex block with n vertices: 0x80-byte header, streams stored in the order given (slot, tag,
    stride), +0x5C = 0x200, +0x60 = bias, +0x6C = axis, +0x70 = scale, +0x7C = 4.0f. bad_size makes the second stream
    one vertex short. packed, if given, is the list of (a, b, c) quantised positions for slot 0."""
    head = bytearray(0x80)
    off = 0x80
    body = bytearray()
    for k, (slot, tag, stride) in enumerate(tags):
        struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
        count = n - 1 if bad_size and k == 1 else n
        if k == 0 and packed is not None:
            body += b"".join(struct.pack("<Q", pack_position(*abc)) for abc in packed)
        else:
            body += bytes((v * 7 + k) & 0xFF for v in range(stride * count))
        off += stride * count
    struct.pack_into("<I", head, 0x5C, 0x200)
    struct.pack_into("<3f", head, 0x60, *bias)
    struct.pack_into("<I", head, 0x6C, axis)
    struct.pack_into("<3f", head, 0x70, *scale)
    struct.pack_into("<f", head, 0x7C, 4.0)
    return bytes(head + body)


def pack_position(a, b, c):
    """The 8-byte position word: a (24 bits) at bits 8-31, b's low byte at bits 0-7 and its high 16 bits at 32-47,
    c (16 bits) at 48-63."""
    return (b & 0xFF) | (a << 8) | ((b >> 8) << 32) | (c << 48)


def group(sizes):
    """A meshlet group. sizes is a list of (vertices, triangles) per meshlet (each <= 128)."""
    recs = bytearray()
    tris = bytearray()
    refs = bytearray()
    vbase = tbase = 0
    for m, (vc, tc) in enumerate(sizes):
        recs += struct.pack("<4I", (vc - 1) | ((tc - 1) << 7), vbase | (tbase << 16), 0x7FF7FF, 0x7FF)
        for t in range(tc):
            a, b, c = t % vc, (t + 1) % vc, (t + 2) % vc
            tris += struct.pack("<I", a | (b << 7) | (c << 14) | (m << 21))
        for v in range(vc):
            refs += struct.pack("<H", vbase + v)
        vbase += vc
        tbase += tc
    return struct.pack("<3I", 0x80000000 | len(sizes), tbase, vbase) + bytes(recs + tris + refs)


def page(*chunks):
    buf = bytearray()
    for c in chunks:
        buf = pad16(buf)
        buf += c
    return bytes(pad16(buf))


class DrawBindingTests(unittest.TestCase):
    def setUp(self):
        self.system = bytearray(704)
        self.graphics = bytearray(640)
        # Two same-sized blocks. The group near the first binds the second.
        for at in (0, 384):
            struct.pack_into("<I", self.graphics, at, 0x4042)
        data = bytes(pad16(bytearray(group([(8, 4)]))))
        self.graphics[256:256 + len(data)] = data
        for at, target in ((8, 160), (64, 240), (224, 320), (240, 256)):
            struct.pack_into("<Q", self.system, at, 0x50000000 + target)
        struct.pack_into("<IH", self.system, 168, 192, 1)
        struct.pack_into("<Q", self.system, 184, 0x60000000 + 384)
        struct.pack_into("<HHI", self.system, 248, 1, 4, 8 << 8)
        struct.pack_into("<IH", self.system, 264, len(data) // 4, 4)
        struct.pack_into("<Q", self.system, 280, 0x60000000 + 256)
        struct.pack_into("<I", self.system, 320, 128)
        self.system[320 + 208] = 8
        self.system[320 + 260] = 16
        struct.pack_into("<I", self.system, 320 + 312, (8 << 10) | 1)

    def test_binding_uses_resource_relative_pointers(self):
        result = mp.draw_binding(self.system, self.graphics, 0)
        self.assertEqual(result, {"vertex_block": 384, "meshlet_group": 256, "vertices": 8})

    def test_binding_to_group_in_later_page(self):
        graphics = self.graphics + self.graphics[256:336]
        struct.pack_into("<Q", self.system, 280, 0x60000000 + 640)
        self.assertEqual(mp.draw_binding(self.system, graphics, 0)["meshlet_group"], 640)

    def add_drawable(self, lod_index):
        self.system.extend(bytes(304))
        lod_field = 784 + 8 * lod_index
        for at, target in ((lod_field, 880), (880, 896), (896, 912), (920, 976), (976, 0)):
            struct.pack_into("<Q", self.system, at, 0x50000000 + target)
        for at in (888, 928):
            struct.pack_into("<HH", self.system, at, 1, 1)
        struct.pack_into("<H", self.system, 966, 1)

    def test_primary_lod_traversal(self):
        self.add_drawable(0)
        self.assertEqual(list(mp.drawable_geometries(self.system, 704)), [0])
        self.system[966] = 2
        with self.assertRaises(mp.PageError):
            list(mp.drawable_geometries(self.system, 704))

    def test_cli_selects_a_nonprimary_lod_from_the_drawable(self):
        self.add_drawable(1)
        with tempfile.TemporaryDirectory() as folder:
            system = os.path.join(folder, "system")
            graphics = os.path.join(folder, "graphics")
            for path, data in ((system, self.system), (graphics, self.graphics)):
                with open(path, "wb") as out:
                    out.write(data)
            cmd = [sys.executable, "-I", TOOL, graphics, "--draw-system", system, "--drawable-offset", "704"]
            primary = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(primary.returncode, 1)
            self.assertIn("unresolved", json.loads(primary.stdout)["bindings"][0])
            secondary = subprocess.run([*cmd, "--lod", "1"], capture_output=True, text=True)
            self.assertEqual(secondary.returncode, 0, secondary.stderr)
            self.assertEqual(json.loads(secondary.stdout)["bindings"],
                             [{"geometry": 0, "vertex_block": 384, "meshlet_group": 256, "vertices": 8}])

    def test_inconsistent_bindings(self):
        for offset, fmt, value in ((64, "Q", 0x50010000), (248, "H", 2),
                                   (280, "Q", 0x60010000), (632, "I", (7 << 10) | 1)):
            with self.subTest(offset=offset):
                damaged = self.system.copy()
                struct.pack_into("<" + fmt, damaged, offset, value)
                with self.assertRaises(mp.PageError):
                    mp.draw_binding(damaged, self.graphics, 0)

    def test_cli_preserves_unresolved_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            system = os.path.join(folder, "system")
            graphics = os.path.join(folder, "graphics")
            for path, data in ((system, self.system), (graphics, self.graphics)):
                with open(path, "wb") as out:
                    out.write(data)
            run = subprocess.run([sys.executable, "-I", TOOL, graphics, "--draw-system", system,
                                  "--geometry-offset", "0", "--geometry-offset", "0xffff"],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            rows = json.loads(run.stdout)["bindings"]
            self.assertEqual(rows[0]["vertex_block"], 384)
            self.assertIn("unresolved", rows[1])


class ParseTests(unittest.TestCase):
    def test_block_then_group(self):
        data = page(block(200), group([(128, 120), (72, 70)]))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block", "group"])
        self.assertEqual(chunks[0]["vertices"], 200)
        self.assertEqual(chunks[0]["flags"], 0x200)
        self.assertTrue(chunks[0]["tags_expected"])
        g = chunks[1]
        self.assertEqual((g["meshlets"], g["triangles"], g["vertex_refs"]), (2, 190, 200))
        self.assertEqual((g["vertex_span"], g["unique_refs"], g["full_meshlets"]), (200, 200, 1))

    def test_three_slot_block_and_odd_ref_count(self):
        # 3 vertices -> odd u16 reference count, so the next chunk needs the 16-byte realignment
        data = page(block(3, colour=False, uv1=False), group([(3, 1)]), block(5, colour=False, uv1=False))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block", "group", "block"])
        self.assertEqual([s[0] for s in chunks[0]["slots"]], [0, 2, 3])

    def test_block_without_normals(self):
        # slots 0, 3, 4 only: the vertex count comes from slot 0's span up to the next stream (slot 3)
        data = page(block(7, colour=False, normal=False), group([(7, 5)]))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block", "group"])
        self.assertEqual(chunks[0]["vertices"], 7)
        self.assertEqual([s[0] for s in chunks[0]["slots"]], [0, 3, 4])
        self.assertTrue(chunks[0]["tags_expected"])
        summary = mp.summarise_page(data, "x")
        self.assertNotIn("error", summary)
        c = summary["checks"]
        self.assertEqual((c["vertices"], c["positions_finite"], c["normals_unit"]), (7, 7, 0))
        self.assertEqual((c["uv_vertices"], c["uv_finite"]), (14, 14))

    def test_group_led_page(self):
        data = page(group([(10, 8)]), block(10))
        self.assertTrue(mp.is_mesh_page(data))
        summary = mp.summarise_page(data, "x")
        self.assertEqual(summary["starts_with"], "group")
        self.assertEqual(summary["sequence"], "GB")
        self.assertEqual(summary["groups_matching_block_here"], 1)

    def test_stop_at_other_block(self):
        other = struct.pack("<I", 0x4042) + b"\x26\x40\x03\x00" + b"\x55" * 40
        summary = mp.summarise_page(page(group([(4, 2)]), other), "x")
        self.assertEqual(summary["sequence"], "G.")
        self.assertEqual(summary["stop"], "42 40 00 00 block")

    def test_4240_block_led_page(self):
        data = page(other_block(40), group([(40, 30)]))
        self.assertTrue(mp.is_mesh_page(data))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block4240", "group"])
        b = chunks[0]
        self.assertEqual((b["vertices"], b["flags"], b["axis"]), (40, 0x200, 2))
        self.assertEqual([s[0] for s in b["slots"]], [0, 1, 2, 3, 4, 11, 12, 14])
        summary = mp.summarise_page(data, "x")
        self.assertEqual((summary["starts_with"], summary["sequence"]), ("block4240", "QG"))
        self.assertEqual((summary["blocks_4240"], summary["vertices_4240"]), (1, 40))
        self.assertEqual(summary["groups_matching_4240_block_here"], 1)
        self.assertEqual((summary["blocks"], summary["checks"].get("vertices", 0)), (0, 0))

    def test_walk_past_4240_block(self):
        # before ASSET-146 the walk stopped at the 42 40 block and never reached the second group
        data = page(block(6), group([(6, 4)]), other_block(9, tags=((0, 0x42, 8), (3, 0x28, 4))), group([(9, 3)]))
        summary = mp.summarise_page(data, "x")
        self.assertEqual(summary["sequence"], "BGQG")
        self.assertIsNone(summary["stop"])
        self.assertEqual((summary["groups"], summary["triangles"]), (2, 7))

    def test_4240_block_that_does_not_fit_stops(self):
        for bad in (other_block(5, bad_size=True), other_block(5, tags=((0, 0x42, 8), (1, 0x99, 4)))):
            summary = mp.summarise_page(page(group([(4, 2)]), bad), "x")
            self.assertEqual((summary["sequence"], summary["stop"]), ("G.", "42 40 00 00 block"))
            self.assertFalse(mp.is_mesh_page(page(bad)))

    def test_63_30_block_with_a_slot_past_4(self):
        # streams in the order 0, 2, 1, 12, 3, as in the one measured block with a slot 12 (tag 0x26): slot 12 takes
        # the tag's 4-byte stride, so the block ends where the group starts
        n = 28
        head = bytearray(0x60)
        body = bytearray()
        off = 0x60
        for slot, tag, stride in ((0, 0x63, 12), (2, 0x63, 12), (1, 0x26, 4), (12, 0x26, 4), (3, 0x42, 8)):
            struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
            body += (struct.pack("<3f", 1.0, 2.0, 3.0) if stride == 12 else bytes(stride)) * n
            off += stride * n
        struct.pack_into("<I", head, 0x5C, 0x200)
        chunks = mp.parse_page(page(bytes(head + body), group([(n, 9)])))
        self.assertEqual([c["kind"] for c in chunks], ["block", "group"])
        self.assertEqual(chunks[0]["vertices"], n)
        self.assertEqual([s[0] for s in chunks[0]["slots"]], [0, 1, 2, 3, 12])

    def test_totals_count_4240_blocks(self):
        pages = [mp.summarise_page(page(other_block(12, axis=1), group([(12, 5)])), "a"),
                 mp.summarise_page(page(other_block(8, axis=0)), "b")]
        t = mp.totals(pages)
        self.assertEqual((t["block_4240_led_pages"], t["blocks_4240"], t["vertices_4240"]), (2, 2, 20))
        self.assertEqual((t["blocks_4240_flags"], t["blocks_4240_axes"]), ([0x200], [0, 1]))
        self.assertEqual(t["groups_matching_4240_block_here"], 1)

    def test_stop_in_truncated_group(self):
        data = page(block(4), group([(4, 2)]))
        g = group([(100, 90)])
        summary = mp.summarise_page(data + g[:64], "x")
        self.assertEqual(summary["sequence"], "BG.")
        self.assertEqual(summary["stop"], "group continues past the stream")

    def test_checks_count_only(self):
        summary = mp.summarise_page(page(block(16), group([(16, 14)])), "x")
        c = summary["checks"]
        self.assertEqual(c["vertices"], 16)
        self.assertEqual(c["positions_finite"], 16)
        self.assertEqual(c["normals_unit"], 16)
        self.assertEqual(c["colour_alpha_255"], 16)
        self.assertEqual(c["uv_vertices"], 32)
        for value in summary.values():
            self.assertNotIsInstance(value, float)


class RejectTests(unittest.TestCase):
    def test_not_a_page(self):
        with self.assertRaises(mp.PageError):
            mp.parse_page(b"\x01\x02\x03\x04" + b"\0" * 60)
        self.assertFalse(mp.is_mesh_page(b"\x63\x30\x00"))

    def test_bad_running_sum(self):
        g = bytearray(group([(8, 6), (8, 6)]))
        struct.pack_into("<I", g, 12 + 16 + 4, 9)   # second meshlet's first vertex reference
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_bad_triangle_meshlet_id(self):
        g = bytearray(group([(8, 6)]))
        tri_at = 12 + 16
        struct.pack_into("<I", g, tri_at, 0 | (1 << 7) | (2 << 14) | (5 << 21))
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_index_past_meshlet(self):
        g = bytearray(group([(8, 6)]))
        struct.pack_into("<I", g, 12 + 16, 0 | (1 << 7) | (9 << 14))
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_header_sum_mismatch(self):
        g = bytearray(group([(8, 6)]))
        struct.pack_into("<I", g, 8, 9)   # one more reference than the meshlets cover
        g += b"\0\0"
        with self.assertRaises(mp.PageError):
            mp.parse_group(bytes(g), 0)

    def test_slot_size_mismatch(self):
        b = bytearray(block(10))
        struct.pack_into("<I", b, 4, 0x26 | (((0x60 + 120 + 8) // 2) << 8))
        with self.assertRaises(mp.PageError):
            mp.parse_block(bytes(b), 0)

    def test_block_with_only_slot_0(self):
        b = bytearray(0x60 + 12 * 4)
        struct.pack_into("<I", b, 0, 0x63 | ((0x60 // 2) << 8))
        with self.assertRaises(mp.PageError):
            mp.parse_block(bytes(b), 0)

    def test_unknown_chunk(self):
        data = page(block(4)) + struct.pack("<I", 0x1234) + b"\0" * 12
        summary = mp.summarise_page(data, "x")
        self.assertIn("unknown chunk", summary["error"])

    def test_other_data_after_padding_stops(self):
        # mesh chunks, zero padding to a 256-byte boundary, then other data (story000's 256 KiB mixed pages)
        mesh = page(other_block(10), group([(10, 8)]))
        tail = bytes(range(1, 65)) * 4
        # unknown magic, and a group header whose meshlet offsets are not running sums
        bad_group = struct.pack("<3I", 0x80000001, 1, 1) + struct.pack("<4I", 0, 5, 0, 0) + bytes(64)
        for junk in (tail, bad_group):
            data = mesh + b"\0" * (-len(mesh) % 256 or 256) + junk
            summary = mp.summarise_page(data, "x")
            self.assertNotIn("error", summary)
            self.assertEqual((summary["sequence"], summary["stop"]), ("QG.", "non-mesh data after the chunks"))
            self.assertEqual((summary["blocks_4240"], summary["groups"]), (1, 1))
            self.assertEqual(mp.totals([summary])["stopped_at_other_data"], 1)
        # with no zero padding in front, or off a 256-byte boundary, it stays an error
        self.assertIn("error", mp.summarise_page(page(other_block(10)) + tail, "x"))
        off = mesh + b"\0" * ((-len(mesh) % 256) + 16)
        self.assertIn("error", mp.summarise_page(off + tail, "x"))


def grid_group(cols, rows):
    """One meshlet over a cols x rows vertex grid (row-major), two consistently wound triangles per cell."""
    tris = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            v = r * cols + c
            tris += [(v, v + 1, v + cols), (v + 1, v + cols + 1, v + cols)]
    vc = cols * rows
    head = struct.pack("<3I", 0x80000000 | 1, len(tris), vc)
    rec = struct.pack("<4I", (vc - 1) | ((len(tris) - 1) << 7), 0, 0x7FF7FF, 0x7FF)
    body = b"".join(struct.pack("<I", a | (b << 7) | (c << 14)) for a, b, c in tris)
    return head + rec + body + b"".join(struct.pack("<H", v) for v in range(vc))


class Positions4240Tests(unittest.TestCase):
    BIAS = (-1.5, 2.25, 0.5)
    SCALE = (2.0 ** -20, 2.0 ** -18, 2.0 ** -10)
    PACKED = [(0, 0, 0), (0xFFFFFF, 0xFFFFFF, 0xFFFF), (0xABCDEF, 0x123456, 0x789A), (1, 0x100, 2)]

    def decode(self, axis):
        data = page(other_block(4, axis=axis, packed=self.PACKED, bias=self.BIAS, scale=self.SCALE))
        return mp.positions_4240(data, mp.parse_page(data)[0])

    def test_packed_position_fields(self):
        want = [tuple(b + s * q for b, s, q in zip(self.BIAS, self.SCALE, abc)) for abc in self.PACKED]
        self.assertEqual(self.decode(2), want)
        # the two 24-bit fields take their own bytes: a from bits 8-31, b from bits 32-47 and 0-7
        self.assertEqual(self.decode(2)[2], (-1.5 + 0xABCDEF * 2.0 ** -20, 2.25 + 0x123456 * 2.0 ** -18,
                                             0.5 + 0x789A * 2.0 ** -10))

    def test_axis_order(self):
        xyz = self.decode(2)
        self.assertEqual(self.decode(0), [(c, b, a) for a, b, c in xyz])
        self.assertEqual(self.decode(1), [(b, c, a) for a, b, c in xyz])

    def test_checks_match_decoding(self):
        # the header shortcut (box inside 300 m) and the vertex-by-vertex path give the decoded counts
        for bias, scale in ((self.BIAS, self.SCALE), ((-250.0, 0.0, 0.0), (2.0 ** -14, 2.0 ** -20, 2.0 ** -10)),
                            ((float("inf"), 0.0, 0.0), self.SCALE)):
            data = page(other_block(4, packed=self.PACKED, bias=bias, scale=scale))
            blk = mp.parse_page(data)[0]
            pos = mp.positions_4240(data, blk)
            finite = [p for p in pos if all(math.isfinite(c) for c in p)]
            want = {"positions_4240": 4, "positions_4240_finite": len(finite),
                    "positions_4240_within_300m": sum(1 for p in finite if all(abs(c) <= 300 for c in p))}
            self.assertEqual(mp.block4240_checks(data, blk), want, bias)
        self.assertEqual(want["positions_4240_finite"], 0)

    def test_bad_axis_and_tag_are_errors(self):
        data = page(other_block(4, axis=3))
        with self.assertRaises(mp.PageError):
            mp.positions_4240(data, mp.parse_page(data)[0])
        self.assertIn("axis word 3", mp.summarise_page(data, "x")["error"])
        data = page(other_block(4, tags=((0, 0x28, 4), (1, 0x26, 4))))
        with self.assertRaises(mp.PageError):
            mp.positions_4240(data, mp.parse_page(data)[0])

    def test_face_agreement(self):
        flat = [(x, y, 0.0) for y in range(3) for x in range(3)]
        tris = [(0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)]
        self.assertEqual(mp.face_agreement(flat, tris), {"shared_edges": 3, "agree": 3, "opposed": 0})
        flipped = tris[:3] + [(2, 4, 5)]
        self.assertEqual(mp.face_agreement(flat, flipped), {"shared_edges": 3, "agree": 2, "opposed": 1})
        # a fold of 90 degrees counts as neither
        self.assertEqual(mp.face_agreement([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0)],
                                           [(0, 1, 2), (1, 0, 3)]), {"shared_edges": 1, "agree": 0, "opposed": 0})

    def test_page_checks_and_faces(self):
        cols, rows = 6, 5
        # a grid in (a, b) at c = 7: axis 0 turns it into a plane of constant x
        packed = [(c * 4096, r * 4096 + 17, 7) for r in range(rows) for c in range(cols)]
        data = page(other_block(cols * rows, axis=0, packed=packed, bias=self.BIAS, scale=self.SCALE),
                    grid_group(cols, rows))
        self.assertEqual([c["kind"] for c in mp.parse_page(data)], ["block4240", "group"])
        plain = mp.summarise_page(data, "p")
        self.assertEqual((plain["checks"]["positions_4240"], plain["checks"]["positions_4240_finite"],
                          plain["checks"]["positions_4240_within_300m"]), (30, 30, 30))
        self.assertNotIn("faces_4240_shared_edges", plain["checks"])
        faces = mp.summarise_page(data, "p", faces=True)["checks"]
        cells = (cols - 1) * (rows - 1)
        shared = cells + (cols - 1) * (rows - 2) + (cols - 2) * (rows - 1)
        self.assertEqual((faces["faces_4240_shared_edges"], faces["faces_4240_agree"], faces["faces_4240_opposed"]),
                         (shared, shared, 0))
        # axis 0 stores (z, y, x): the constant c field is x
        self.assertEqual(len({p[0] for p in mp.positions_4240(data, mp.parse_page(data)[0])}), 1)

    def test_cli_faces(self):
        packed = [(c * 4096, r * 4096, 3) for r in range(4) for c in range(4)]
        data = page(other_block(16, packed=packed, bias=self.BIAS, scale=self.SCALE), grid_group(4, 4))
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "p.bin")
            with open(src, "wb") as fh:
                fh.write(data)
            out = os.path.join(tmp, "r.json")
            r = subprocess.run([sys.executable, "-I", TOOL, src, "--faces", "--json", out, "--quiet"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as fh:
                checks = json.load(fh)["totals"]["checks"]
        self.assertEqual((checks["positions_4240"], checks["faces_4240_agree"], checks["faces_4240_opposed"]),
                         (16, 21, 0))


class MeshletCentreTests(unittest.TestCase):
    BIAS = (-1.5, 2.25, 0.5)
    SCALE = (2.0 ** -20, 2.0 ** -18, 2.0 ** -10)

    def test_meshlet_centre(self):
        self.assertEqual(mp.meshlet_centre(0x007FF7FF, 0x7FF), (0.0, 0.0, 0.0))
        self.assertEqual(mp.meshlet_centre(0x0080F7EF, 0xFFFFF000), (-1.0, 1.0, -127.9375))
        self.assertEqual(mp.meshlet_centre(0x00000FFF, 0x00000FFF), (128.0, -127.9375, 128.0))
        # w2 bit 28 doubles the step: 1/8 m, out to +-256 m
        self.assertEqual(mp.meshlet_centre(0x10000FFF, 0x00000FFF), (256.0, -255.875, 256.0))
        self.assertEqual(mp.meshlet_centre(0x1080F7EF, 0x7FF), (-2.0, 2.0, 0.0))
        # w2 bits 28-31 are an exponent: step 2^e / 16 m
        self.assertEqual([mp.centre_step(e << 28 | 0x007FF7FF) for e in range(5)], [0.0625, 0.125, 0.25, 0.5, 1.0])
        self.assertEqual(mp.meshlet_centre(0x2080F7EF, 0x7FF), (-4.0, 4.0, 0.0))

    def grid_parts(self, steps, bias=BIAS):
        """A 4x4 grid block and one meshlet over it whose stored centre is the vertex-box centre moved by `steps`."""
        packed = [(c * 4096, r * 4096, 3) for r in range(4) for c in range(4)]
        blk = other_block(16, packed=packed, bias=bias, scale=self.SCALE)
        pos = mp.positions_4240(page(blk), mp.parse_page(page(blk))[0])
        centre = [(min(p[i] for p in pos) + max(p[i] for p in pos)) / 2 for i in range(3)]
        f = [round(c * 16) + 2047 + s for c, s in zip(centre, steps)]
        grp = bytearray(grid_group(4, 4))
        struct.pack_into("<2I", grp, 12 + 8, f[0] | f[1] << 12, f[2])
        return blk, bytes(grp)

    def grid_page(self, steps):
        return page(*self.grid_parts(steps))

    def test_centre_agreement(self):
        data = self.grid_page((0, 0, 0))
        checks = mp.summarise_page(data, "p", centres=True)["checks"]
        self.assertEqual((checks["centres_4240_meshlets"], checks["centres_4240_axes"], checks["centres_4240_axes_exact"],
                          checks["centres_4240_axes_within_step"], checks["centres_4240_default_words"]),
                         (1, 3, 3, 3, 0))
        data = self.grid_page((1, 0, 5))
        checks = mp.summarise_page(data, "p", centres=True)["checks"]
        self.assertEqual((checks["centres_4240_axes_exact"], checks["centres_4240_axes_within_step"]), (1, 2))
        self.assertNotIn("centres_4240_meshlets", mp.summarise_page(data, "p")["checks"])

    def test_block_chosen_by_centres(self):
        """Two blocks with the group's vertex count: the one its stored centres fit, wherever it is."""
        blk, grp = self.grid_parts((0, 0, 0))
        moved = self.grid_parts((0, 0, 0), bias=(48.5, 2.25, 0.5))[0]                      # same grid, 50 m away
        for data, by_centres in ((page(blk, moved, grp), 1), (page(moved, blk, grp), 0), (page(moved, grp, blk), 1)):
            checks = mp.summarise_page(data, "p", centres=True)["checks"]
            self.assertEqual((checks["centres_4240_axes_exact"], checks["groups_4240_several_blocks"],
                              checks.get("groups_4240_block_by_centres", 0)), (3, 1, by_centres))
        checks = mp.summarise_page(page(moved, moved, grp), "p", centres=True)["checks"]  # a tie keeps the nearest
        self.assertEqual((checks["centres_4240_axes_exact"], checks.get("groups_4240_block_by_centres", 0)), (2, 0))
        self.assertNotIn("groups_4240_several_blocks", mp.summarise_page(page(blk, grp), "p", centres=True)["checks"])

    def test_non_finite_block_does_not_win(self):
        """A block with NaN positions scores None: the finite block wins in either order, and with no finite block
        the old choice stays (as before this rule)."""
        blk, grp = self.grid_parts((0, 0, 0))
        nan = bytearray(blk)
        struct.pack_into("<f", nan, 0x60, float("nan"))                               # bias x
        for data in (page(bytes(nan), blk, grp), page(blk, bytes(nan), grp)):
            self.assertEqual(mp.summarise_page(data, "p", centres=True)["checks"]["centres_4240_axes_exact"], 3)
        checks = mp.summarise_page(page(bytes(nan), bytes(nan), grp), "p", faces=True)["checks"]
        self.assertEqual((checks["groups_4240_several_blocks"], checks.get("groups_4240_block_by_centres", 0)), (1, 0))
        self.assertIsNone(mp.centre_distance(page(grp), mp.parse_page(page(grp))[0], [(math.nan, 0.0, 0.0)] * 16))

    def test_coarse_centre_agreement(self):
        # the grid page with its stored centre written at 2^e / 16 m steps and the exponent e in w2 bits 28-31
        for e in (1, 2, 3):
            data = bytearray(self.grid_page((0, 0, 0)))
            blk, grp = mp.parse_page(bytes(data))
            pos = mp.positions_4240(bytes(data), blk)
            box = [(min(p[i] for p in pos) + max(p[i] for p in pos)) / 2 for i in range(3)]
            f = [round(c * 16 / 2 ** e) + 2047 for c in box]
            rec = grp["offset"] + 12
            w3 = struct.unpack_from("<I", data, rec + 12)[0] & ~0xFFF
            struct.pack_into("<2I", data, rec + 8, e << 28 | f[0] | f[1] << 12, w3 | f[2])
            c = mp.summarise_page(bytes(data), "p", centres=True)["checks"]
            self.assertEqual((c["centres_4240_coarse_steps"], c["centres_4240_axes_exact"]), (1, 3), e)

    def test_meshlet_radius(self):
        self.assertEqual(mp.meshlet_radius(0x3C00 << 16 | 0x1FFF), 1.0)
        self.assertEqual(mp.meshlet_radius(0x4D00 << 16), 20.0)
        self.assertEqual(mp.meshlet_radius(0x7F), 0.0)

    def test_radius_agreement(self):
        data = bytearray(self.grid_page((0, 0, 0)))
        blk, grp = mp.parse_page(bytes(data))
        pos = mp.positions_4240(bytes(data), blk)
        centre = mp.meshlet_centre(*struct.unpack_from("<2I", data, grp["offset"] + 12 + 8))
        far = max(math.dist(p, centre) for p in pos)
        rec = grp["offset"] + 12
        w0 = struct.unpack_from("<I", data, rec)[0] & 0xFFFF
        for radius, want in ((far * 1.01, (1, 1, 1)), (far * 0.99, (0, 1, 1)), (far * 2 + 1, (1, 1, 0))):
            struct.pack_into("<I", data, rec, w0 | struct.unpack("<H", struct.pack("<e", radius))[0] << 16)
            c = mp.summarise_page(bytes(data), "p", centres=True)["checks"]
            self.assertEqual((c["centres_4240_radius_covers"], c["centres_4240_radius_covers_within_rounding"],
                              c["centres_4240_radius_tight"]), want, radius)

    def test_meshlet_cone(self):
        self.assertIsNone(mp.meshlet_cone(0x7FF))
        up = 32 << 14 | 32 << 20                        # octahedral u = v = 0.5 / 31.5: nearly +z
        axis, cos_half = mp.meshlet_cone(up | 31 << 26)
        self.assertAlmostEqual(axis[2], 1.0, places=3)
        self.assertEqual(cos_half, 1.0)                  # f = 31: sin(half-angle) 0
        self.assertEqual(mp.meshlet_cone(up)[1], 0.0)    # f = 0: half-angle 90 degrees
        self.assertAlmostEqual(mp.meshlet_cone(up | 16 << 26)[1], math.sqrt(1 - (15 / 31) ** 2))
        self.assertAlmostEqual(mp.meshlet_cone(63 << 14 | 32 << 20 | 1 << 26)[0][0], 1.0, places=3)   # +x

    def test_cone_agreement(self):
        # grid_page's grid is a plane of constant z; take the facing from its first triangle
        data = bytearray(self.grid_page((0, 0, 0)))
        blk, grp = mp.parse_page(bytes(data))
        pos = mp.positions_4240(bytes(data), blk)
        p, q, r = (pos[i] for i in next(mp.group_triangles(bytes(data), grp)))
        nz = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        up, down = 32 << 14 | 32 << 20, 0             # octahedral: nearly +z, and u = v = -1 folded onto -z
        facing, away = (up, down) if nz > 0 else (down, up)
        rec = grp["offset"] + 12
        w2, w3 = struct.unpack_from("<2I", data, rec + 8)
        # f = 31 is a zero half-angle: the 6-bit axis is not exactly the face normal, so only the 0.05 allowance holds
        for cone, want in ((facing | 31 << 26, (1, 1, 0, 1)), (away | 31 << 26, (1, 0, 0, 0)),
                           (facing | 16 << 26, (1, 1, 1, 1)), (0, (0, 0, 0, 0))):
            struct.pack_into("<I", data, rec + 12, (w3 & 0xFFF) | cone)
            c = mp.summarise_page(bytes(data), "p", cones=True)["checks"]
            self.assertEqual((c["cones_4240_cones"], c["cones_4240_axis_within_25_degrees"], c["cones_4240_faces_inside"],
                              c["cones_4240_faces_inside_within_0_05"]), want, hex(cone))

    def test_block4240_checks_streaming_and_summary_counts(self):
        # Normal block with 16 vertices within 300m
        packed = [(c * 4096, r * 4096, 3) for r in range(4) for c in range(4)]
        blk16 = other_block(16, packed=packed, bias=(10.0, 20.0, 30.0), scale=(0.001, 0.001, 0.001))
        data = page(blk16)
        parsed = mp.parse_page(data)[0]
        chk = mp.block4240_checks(data, parsed)
        self.assertEqual(chk, {"positions_4240": 16, "positions_4240_finite": 16, "positions_4240_within_300m": 16})

        # Beyond 300m range (bias at 500m)
        blk_far = other_block(16, packed=packed, bias=(500.0, 20.0, 30.0), scale=(0.001, 0.001, 0.001))
        data_far = page(blk_far)
        chk_far = mp.block4240_checks(data_far, mp.parse_page(data_far)[0])
        self.assertEqual(chk_far, {"positions_4240": 16, "positions_4240_finite": 16, "positions_4240_within_300m": 0})

        # Non-finite (NaN bias)
        blk_nan = other_block(16, packed=packed, bias=(float("nan"), 20.0, 30.0), scale=(0.001, 0.001, 0.001))
        data_nan = page(blk_nan)
        chk_nan = mp.block4240_checks(data_nan, mp.parse_page(data_nan)[0])
        self.assertEqual(chk_nan, {"positions_4240": 16, "positions_4240_finite": 0, "positions_4240_within_300m": 0})

        # summarise_page with multiple groups matching block4240
        data_multi = page(blk16, group([(16, 5)]), group([(16, 8)]), group([(10, 2)]))
        summary = mp.summarise_page(data_multi, "multi")
        self.assertEqual(summary["groups_matching_4240_block_here"], 2)


    def test_cli_centres(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "p.bin")
            with open(src, "wb") as fh:
                fh.write(self.grid_page((0, -1, 0)))
            out = os.path.join(tmp, "r.json")
            r = subprocess.run([sys.executable, "-I", TOOL, src, "--centres", "--json", out, "--quiet"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as fh:
                checks = json.load(fh)["totals"]["checks"]
        self.assertEqual((checks["centres_4240_axes_exact"], checks["centres_4240_axes_within_step"]), (2, 3))
        self.assertNotIn("faces_4240_shared_edges", checks)


class SlotTagTests(unittest.TestCase):
    def test_tag_stride(self):
        self.assertEqual([mp.tag_stride(x) for x in (0x63, 0x42, 0x26, 0x28, 0x21, 0x27)], [12, 8, 4, 4, 4, 4])
        self.assertEqual(mp.tag_stride(0x25), 4)  # joint indices
        self.assertIsNone(mp.tag_stride(0x2f))   # unknown type
        self.assertIsNone(mp.tag_stride(0x02))   # stride 0
        self.assertEqual(mp.tag_stride(0x46), 8)  # RGBA8-type tag with an 8-byte stride: sized by the rule

    def test_block_sized_by_tag_rule(self):
        # slot 3 has tag 0x48 (type 8, 8 bytes): not in the old per-tag table, sized by the rule
        tags = ((0, 0x42, 8), (2, 0x21, 4), (3, 0x48, 8), (1, 0x26, 4))
        data = page(other_block(10, tags=tags), group([(10, 6)]))
        chunks = mp.parse_page(data)
        self.assertEqual([c["kind"] for c in chunks], ["block4240", "group"])
        self.assertEqual(chunks[1]["offset"] - chunks[0]["offset"], 0x80 + 10 * (8 + 4 + 8 + 4))
        # a type that is not seen is still not a block: such a page is not a mesh page
        bad = page(other_block(10, tags=((0, 0x42, 8), (2, 0x2f, 4))), group([(10, 6)]))
        self.assertFalse(mp.is_mesh_page(bad))
        self.assertIn("error", mp.summarise_page(bad, "x"))

    def test_joint_index_stream_before_another_slot(self):
        tags = ((0, 0x42, 8), (6, 0x26, 4), (8, 0x25, 4), (1, 0x26, 4))
        vertices = other_block(5, tags=tags)
        data = page(vertices, group([(5, 3)]))
        self.assertTrue(mp.is_mesh_page(data))
        block, meshlets = mp.parse_page(data)
        self.assertEqual(block["vertices"], 5)
        self.assertIn([8, 0x25, 0x80 + 5 * 12], block["slots"])
        self.assertEqual((meshlets["kind"], meshlets["offset"]), ("group", (len(vertices) + 15) & ~15))


class Normals4240Tests(unittest.TestCase):
    UP = 512 | 512 << 10   # octahedral u = v = 0.5 / 511.5: nearly +z

    def block_with_normals(self, words, tag=0x21, packed=None, n=None):
        n = n or len(words)
        tags = ((0, 0x42, 8), (2, tag, 4), (1, 0x26, 4))
        data = bytearray(page(other_block(n, tags=tags, packed=packed or [(0, 0, 0)] * n, scale=(2.0 ** -20,) * 3)))
        blk = mp.parse_page(bytes(data))[0]
        off = dict((i, o) for i, _, o in blk["slots"])[2]
        for k, w in enumerate(words):
            struct.pack_into("<I", data, blk["offset"] + off + 4 * k, w)
        return bytes(data), blk

    def assertVector(self, got, want, places=2):
        for g, w in zip(got, want):
            self.assertAlmostEqual(g, w, places=places)

    def test_octahedral_normals(self):
        data, blk = self.block_with_normals([1023, self.UP, 1023 | 511 << 10, 0 | 1023 << 10, 0xC0000000 | self.UP])
        n = mp.normals_4240(data, blk)
        self.assertVector(n[0], (0, 0, -1))   # u = 1, v = -1 folds onto -z
        self.assertVector(n[1], (0, 0, 1))
        self.assertVector(n[2], (1, 0, 0))
        self.assertVector(n[3], (0, 0, -1))
        self.assertEqual(n[4], n[1])          # bits 20-31 don't change the normal
        for v in n:
            self.assertAlmostEqual(sum(c * c for c in v), 1.0)

    def test_type7_normals_and_other_types(self):
        data, blk = self.block_with_normals([1023 | 512 << 10 | 512 << 20, 512 | 512 << 10 | 0 << 20], tag=0x27)
        n = mp.normals_4240(data, blk)
        self.assertVector(n[0], (1, 0, 0))
        self.assertVector(n[1], (0, 0, -1))
        data, blk = self.block_with_normals([0, 0], tag=0x26)
        self.assertIsNone(mp.normals_4240(data, blk))

    def test_normal_agreement(self):
        flat = [(x, y, 0.0) for y in range(3) for x in range(3)]
        tris = [(0, 1, 3), (1, 4, 3), (1, 2, 4), (2, 5, 4)]
        up = [(0.0, 0.0, 1.0)] * 9
        self.assertEqual(mp.normal_agreement(flat, up, tris),
                         {"vertices": 6, "within_8_degrees": 6, "within_25_degrees": 6, "opposed": 0})
        tilt = math.radians(20)
        mixed = list(up)
        mixed[0] = (0.0, 0.0, -1.0)
        mixed[4] = (math.sin(tilt), 0.0, math.cos(tilt))
        self.assertEqual(mp.normal_agreement(flat, mixed, tris),
                         {"vertices": 6, "within_8_degrees": 4, "within_25_degrees": 5, "opposed": 1})
        with self.assertRaises(mp.PageError):
            mp.normal_agreement(flat, up[:4], tris)

    def test_tangents(self):
        tags = ((0, 0x42, 8), (2, 0x21, 4), (11, 0x27, 4), (18, 0x27, 4))
        data = bytearray(page(other_block(3, tags=tags, packed=[(0, 0, 0)] * 3)))
        blk = mp.parse_page(bytes(data))[0]
        offs = dict((i, o) for i, _, o in blk["slots"])
        x_axis = 1023 | 512 << 10 | 512 << 20          # tangent +x, normal +z: perpendicular
        z_axis = 512 | 512 << 10 | 1023 << 20
        for k, (nw, t11, t18) in enumerate([(self.UP, 3 << 30 | x_axis, x_axis), (self.UP, x_axis, 1 << 30 | x_axis),
                                             (self.UP, 3 << 30 | z_axis, 3 << 30 | x_axis)]):
            struct.pack_into("<I", data, blk["offset"] + offs[2] + 4 * k, nw)
            struct.pack_into("<I", data, blk["offset"] + offs[11] + 4 * k, t11)
            struct.pack_into("<I", data, blk["offset"] + offs[18] + 4 * k, t18)
        tan = mp.tangents_4240(bytes(data), blk)
        self.assertEqual(sorted(tan), [11, 18])
        self.assertVector(tan[11][0][:3], (1, 0, 0))
        self.assertEqual([r[3] for r in tan[11]], [1, -1, 1])
        self.assertEqual([r[3] for r in tan[18]], [-1, 0, 1])
        got = mp.tangent_agreement(mp.normals_4240(bytes(data), blk), tan)
        self.assertEqual(got, {"vertices": 6, "perpendicular": 5, "sign_plus": 3, "sign_minus": 2})
        checks = mp.summarise_page(bytes(data), "p", normals=True)["checks"]
        self.assertEqual(checks["tangents_4240_perpendicular"], 5)

    def test_page_normals_and_cli(self):
        cols, rows = 4, 4
        packed = [(c * 4096, r * 4096, 5) for r in range(rows) for c in range(cols)]   # axis 2: a plane facing +z
        data, _ = self.block_with_normals([self.UP] * (cols * rows), packed=packed)
        data = page(data[:mp.parse_page(data)[0]["offset"] + 0x80 + (8 + 4 + 4) * cols * rows], grid_group(cols, rows))
        checks = mp.summarise_page(data, "p", normals=True)["checks"]
        self.assertEqual((checks["normals_4240_vertices"], checks["normals_4240_within_8_degrees"],
                          checks["normals_4240_opposed"]), (16, 16, 0))
        self.assertNotIn("normals_4240_vertices", mp.summarise_page(data, "p")["checks"])
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "p.bin")
            with open(src, "wb") as fh:
                fh.write(data)
            out = os.path.join(tmp, "r.json")
            r = subprocess.run([sys.executable, "-I", TOOL, src, "--normals", "--json", out, "--quiet"],
                               capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["totals"]["checks"]["normals_4240_within_25_degrees"], 16)


class CliTests(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)

    def test_directory_and_archive(self):
        pages = [page(block(20), group([(20, 18)])), page(group([(6, 4)]), block(6, colour=False, uv1=False)),
                 page(other_block(10), group([(10, 8)]))]
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, "carve")
            os.mkdir(folder)
            for i, data in enumerate(pages):
                with open(os.path.join(folder, f"{i:04x}.bin"), "wb") as fh:
                    fh.write(data)
            with open(os.path.join(folder, "other.bin"), "wb") as fh:
                fh.write(b"\0" * 64)
            out = os.path.join(tmp, "dir.json")
            r = self.run_tool(folder, "--json", out, "--quiet")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as fh:
                totals = json.load(fh)["totals"]
            self.assertEqual((totals["pages"], totals["block_led_pages"], totals["group_led_pages"]), (3, 1, 1))
            self.assertEqual((totals["blocks"], totals["groups"], totals["vertices"]), (2, 3, 26))
            self.assertEqual(totals["groups_matching_block_near"], 2)
            # a page that starts with a 42 40 block is read from a directory too, not only from an archive
            self.assertEqual((totals["block_4240_led_pages"], totals["blocks_4240"]), (1, 1))

            # A minimal RPF9-like file: data offset at +4, then zlib streams (one is not a mesh page).
            rpf = os.path.join(tmp, "a.rpf")
            body = b"".join(zlib.compress(p) for p in pages + [b"\x11" * 128])
            with open(rpf, "wb") as fh:
                fh.write(b"9FPR" + struct.pack("<I", 16) + b"\0" * 8 + body)
            out2 = os.path.join(tmp, "rpf.json")
            r = self.run_tool(rpf, "--json", out2, "--quiet")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out2, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["totals"]["pages"], 3)

    def test_page_report_write_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "page.bin")
            with open(source, "wb") as fh:
                fh.write(page(block(4)))
            out = os.path.join(tmp, "missing", "report.json")

            result = self.run_tool(source, "--json", out, "--quiet")

        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(f"cannot write {out}", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_neighbor_counts_keep_malformed_page_positions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            block_path = os.path.join(tmp, "block.bin")
            bad_path = os.path.join(tmp, "malformed.bin")
            group_path = os.path.join(tmp, "group.bin")
            out = os.path.join(tmp, "report.json")
            for path, data in ((block_path, page(block(4))),
                               (bad_path, b"bad"),
                               (group_path, page(group([(4, 2)])))):
                with open(path, "wb") as fh:
                    fh.write(data)

            for paths, exit_code, matches in (
                ((block_path, group_path), 0, 1),
                ((group_path, block_path), 0, 1),
                ((block_path, bad_path, group_path), 1, 0),
            ):
                with self.subTest(paths=paths):
                    result = self.run_tool(*paths, "--json", out, "--quiet")
                    self.assertEqual(result.returncode, exit_code, result.stderr)
                    with open(out, encoding="utf-8") as fh:
                        totals = json.load(fh)["totals"]
                    self.assertEqual(totals["groups_matching_block_near"], matches)

    def test_missing_input(self):
        r = self.run_tool(os.path.join(tempfile.gettempdir(), "no-such-meshlet-page.bin"))
        self.assertEqual(r.returncode, 2)

    def test_error_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.bin")
            with open(path, "wb") as fh:
                fh.write(page(block(4)) + struct.pack("<I", 0x1234) + b"\0" * 12)
            r = self.run_tool(path, "--quiet")
            self.assertEqual(r.returncode, 1)

    def test_json_that_is_a_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "page.bin")
            with open(source, "wb") as fh:
                fh.write(page(block(4)))
            r = self.run_tool(source, "--json", tmp, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("is a directory", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

    def test_json_over_input_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "test.rpf")
            data = b"9FPR" + struct.pack("<I", 16) + b"\0" * 8 + zlib.compress(page(block(4)))
            with open(rpf, "wb") as fh:
                fh.write(data)
            alias = os.path.join(tmp, ".", "test.rpf")
            r = self.run_tool(rpf, "--json", alias, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(rpf, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_over_input_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "page.bin")
            data = page(block(4))
            with open(source, "wb") as fh:
                fh.write(data)
            r = self.run_tool(source, "--json", source, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(source, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_cannot_create_a_report_in_place_of_the_carve_manifest(self):
        # <input>/index.json is the carve manifest's name even before a carve has written it.
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            with open(os.path.join(carve, "0000.bin"), "wb") as fh:
                fh.write(page(block(4)))
            manifest = os.path.join(carve, "index.json")
            r = self.run_tool(carve, "--json", manifest, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertFalse(os.path.exists(manifest))

    def test_json_over_carve_dir_manifest_or_payload_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            manifest = os.path.join(carve, "index.json")
            with open(manifest, "w", encoding="utf-8") as fh:
                fh.write('{"files": []}')
            bin_path = os.path.join(carve, "0000.bin")
            bin_data = page(block(4))
            with open(bin_path, "wb") as fh:
                fh.write(bin_data)

            # Refuse overwrite of index.json
            r = self.run_tool(carve, "--json", manifest, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(manifest, "r", encoding="utf-8") as fh:
                self.assertEqual(fh.read(), '{"files": []}')

            # Refuse overwrite of carved .bin payload
            r = self.run_tool(carve, "--json", bin_path, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(bin_path, "rb") as fh:
                self.assertEqual(fh.read(), bin_data)

    def test_json_hard_link_to_a_page_in_a_scanned_dir_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            source = os.path.join(carve, "page.bin")
            data = page(block(4))
            with open(source, "wb") as fh:
                fh.write(data)
            report = os.path.join(tmp, "report.json")
            try:
                os.link(source, report)
            except (AttributeError, NotImplementedError, OSError) as exc:
                self.skipTest(f"hard links unavailable: {exc}")

            r = self.run_tool(carve, "--json", report, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            with open(source, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_inside_carve_dir_can_be_written_and_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            bin_path = os.path.join(carve, "0000.bin")
            with open(bin_path, "wb") as fh:
                fh.write(page(block(4)))
            out = os.path.join(carve, "report.json")
            for run in ("new report", "existing report"):
                with self.subTest(run=run):
                    r = self.run_tool(carve, "--json", out, "--quiet")
                    self.assertEqual(r.returncode, 0, r.stderr)
                    with open(out, "r", encoding="utf-8") as fh:
                        self.assertEqual(json.load(fh)["totals"]["pages"], 1)

    def test_json_over_draw_system_inputs_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            sys_path = os.path.join(tmp, "system.bin")
            gfx_path = os.path.join(tmp, "graphics.bin")
            with open(sys_path, "wb") as fh:
                fh.write(b"\0" * 256)
            with open(gfx_path, "wb") as fh:
                fh.write(b"\0" * 256)

            # Refuse overwrite of graphics segment
            r = self.run_tool(gfx_path, "--draw-system", sys_path, "--geometry-offset", "0", "--json", gfx_path, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)

            # Refuse overwrite of system segment
            r = self.run_tool(gfx_path, "--draw-system", sys_path, "--geometry-offset", "0", "--json", sys_path, "--quiet")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
