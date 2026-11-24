"""Synthetic-data tests for tools/vertex_types.py.

    python -I tests/test_vertex_types.py

Every byte here is built by the test. No game file, value or name is used.
"""
import contextlib
import io
import json
import math
import os
import pathlib
import random
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "vertex_types.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import vertex_types as vt  # noqa: E402

GRID = 10  # GRID x GRID vertices, 2 * (GRID - 1)^2 triangles


def pack_position(x, y, z):
    """Inverse of the 0x42 slot-0 packing: x bits 8-31, y low byte bits 0-7 and high 16 bits 32-47, z bits 48-63."""
    return (y & 0xFF) | (x << 8) | ((y >> 8) << 32) | (z << 48)


def grid_triangles(grid=GRID):
    tris = []
    for j in range(grid - 1):
        for i in range(grid - 1):
            a, b, c, d = j * grid + i, j * grid + i + 1, (j + 1) * grid + i, (j + 1) * grid + i + 1
            tris += [(a, b, c), (b, d, c)]
    return tris


def group_for(tris, n):
    """A meshlet group: each meshlet references all n vertices and takes up to 128 of the triangles."""
    assert n <= 128
    chunks = [tris[i:i + 128] for i in range(0, len(tris), 128)]
    recs, words, refs = bytearray(), bytearray(), bytearray()
    vbase = tbase = 0
    for m, part in enumerate(chunks):
        recs += struct.pack("<4I", (n - 1) | ((len(part) - 1) << 7), vbase | (tbase << 16), 0x7FF7FF, 0x7FF)
        for a, b, c in part:
            words += struct.pack("<I", a | (b << 7) | (c << 14) | (m << 21))
        refs += struct.pack(f"<{n}H", *range(n))
        vbase += n
        tbase += len(part)
    return struct.pack("<3I", 0x80000000 | len(chunks), tbase, vbase) + bytes(recs + words + refs)


def block(uv_offset=-5, colour=(10, 20, 30, 255), grid=GRID, with_group=True, normal=bytes((0, 0, 127, 0))):
    """A 42 40 00 00 block on a flat grid: slots 0 (0x42 position), 1 (0x26), 2 (0x21 = snorm8 +z), 3 (0x28),
    followed by its meshlet group unless with_group is false."""
    n = grid * grid
    streams = [
        (0, 0x42, b"".join(struct.pack("<Q", pack_position(100 * (v % grid), 100 * (v // grid), 0)) for v in range(n))),
        (1, 0x26, bytes(colour) * n),
        (2, 0x21, normal * n),
        (3, 0x28, b"".join(struct.pack("<2h", 1000 * (v % grid + uv_offset), 1000 * (v // grid + uv_offset))
                           for v in range(n))),
    ]
    head = bytearray(vt.HEADER)
    off = vt.HEADER
    body = bytearray()
    for slot, tag, data in streams:
        struct.pack_into("<I", head, 4 * slot, tag | ((off // 2) << 8))
        body += data
        off += len(data)
    struct.pack_into("<I", head, 0x5C, 0x200)
    struct.pack_into("<3f", head, 0x60, 0.0, 0.0, 0.0)
    struct.pack_into("<I", head, 0x6C, 2)
    struct.pack_into("<3f", head, 0x70, 0.01, 0.01, 0.01)
    struct.pack_into("<f", head, 0x7C, 4.0)
    buf = bytearray(head + body)
    buf += b"\0" * (-len(buf) % 16)
    if with_group:
        buf += padded_group(grid)
    return bytes(buf)


def padded_group(grid=GRID):
    g = group_for(grid_triangles(grid), grid * grid)
    return g + b"\0" * (-len(g) % 16)


def measure_stream(data):
    T, errors = {}, {}
    vt.measure_stream(data, T, random.Random(1), errors)
    return T, errors


def measure(data):
    T = {}
    vt.measure_block(data, 0, T, random.Random(1))
    return T, vt.summarise(T)


class BlockTests(unittest.TestCase):
    def test_stride_rule(self):
        self.assertEqual([vt.stride(t) for t in (0x42, 0x21, 0x25, 0x26, 0x27, 0x28, 0x8C, 0x63)],
                         [8, 4, 4, 4, 4, 4, 16, 12])

    def test_slots_and_count(self):
        slots, order, flags = vt.parse_slots(block())
        self.assertEqual(order, [0, 1, 2, 3])
        self.assertEqual(slots[0], (0x42, 0x80))
        self.assertEqual(flags, 0x200)
        self.assertEqual(vt.vertex_count(slots, order), GRID * GRID)

    def test_rejects_other_magic(self):
        data = bytearray(block())
        data[0] = 0x63
        with self.assertRaises(vt.BlockError):
            vt.parse_slots(bytes(data))

    def test_signed_uv_is_continuous(self):
        T, out = measure(block(uv_offset=-5))
        uv = out["tags"]["slot3:28"]
        self.assertEqual(uv["edge_jumps_signed"], 0)
        self.assertGreater(uv["edge_jumps_unsigned"], 0)
        self.assertEqual(T["header_words"]["f32_7c"], {"4.0": 1})

    def test_positive_uv_has_no_jumps_either_way(self):
        _, out = measure(block(uv_offset=0))
        uv = out["tags"]["slot3:28"]
        self.assertEqual((uv["edge_jumps_signed"], uv["edge_jumps_unsigned"]), (0, 0))

    def test_mapping_spread_is_zero_on_an_even_grid(self):
        _, out = measure(block())
        self.assertEqual(out["mapping_spread"]["s16"]["median"], 0.0)

    def test_packed_snorm10_normal_clamps_both_negative_endpoint_codes(self) -> None:
        decode = vt.NORMAL_CANDIDATES["snorm10_xyz"]
        values = struct.pack("<2I", 512 | (511 << 10), 513 | (511 << 10))
        diagonal = 1.0 / math.sqrt(2.0)
        expected = (-diagonal, diagonal, 0.0)
        self.assertEqual(decode(values, 0), expected)
        self.assertEqual(decode(values, 1), expected)

    def test_normal_probe_uses_clamped_snorm10_direction_for_face_agreement(self) -> None:
        mesh = bytearray(block(grid=2))
        slots, _, _ = vt.parse_slots(mesh)
        struct.pack_into("<4Q", mesh, slots[0][1], pack_position(0, 0, 0), pack_position(31, 218, 0),
                         pack_position(0, 0, 100), pack_position(31, 218, 100))
        struct.pack_into("<4I", mesh, slots[2][1], *([512 | (511 << 10)] * 4))
        _, report = measure(bytes(mesh))
        # Without the clamp, the packed normal's absolute dot product
        # crosses the 0.8 face-agreement threshold.
        self.assertEqual(report["tags"]["slot2:21"]["facenormal_snorm10_xyz_median"], 0.0)

    def test_constant_colour_and_known_normal(self):
        _, out = measure(block())
        col = out["tags"]["slot1:26"]
        for k in range(4):
            self.assertEqual(col[f"byte{k}_constant_blocks"], 1)
        self.assertEqual(col["byte3_255"], GRID * GRID)
        nrm = out["tags"]["slot2:21"]
        self.assertEqual(nrm["facenormal_snorm8_xyz_median"], 1.0)
        self.assertEqual(nrm["unit_snorm8"], GRID * GRID)

    def test_0x21_octahedral_normal_in_either_width(self):
        # +z in the 10-bit form (bit 31 set, x = y = 512) and in the 8-bit form (bit 31 clear, bytes 0-1 = 128);
        # each scores 1.0 only under its own branch of the decode
        for word in (0x80000000 | 512 | 512 << 10, 128 | 128 << 8):
            _, out = measure(block(normal=struct.pack("<I", word)))
            nrm = out["tags"]["slot2:21"]
            self.assertEqual(nrm["facenormal_oct_bit31_10_or_8_median"], 1.0)

    def test_0x21_tangent_follows_u_and_bit30_is_the_bitangent_sign(self):
        # flat grid: dP/du = +x, dP/dv = +y, normal +z. 10-bit: angle field 0 (-pi) turns b1 = -x into +x;
        # 8-bit: octahedral tangent (127, 64) = +x. cross(+z, +x) = +y runs along +v, so bit 30 set agrees.
        for form, word in (("10", 0x80000000 | 512 | 512 << 10), ("8", 128 | 128 << 8 | 127 << 16 | 64 << 23)):
            for bit30, agree in ((1 << 30, True), (0, False)):
                _, out = measure(block(normal=struct.pack("<I", word | bit30)))
                nrm = out["tags"]["slot2:21"]
                checked = nrm[f"tangent{form}_checked"]
                self.assertGreater(checked, 0)
                self.assertEqual(nrm[f"tangent{form}_along_u"], checked)
                self.assertEqual(nrm[f"tangent{form}_sign_agree"], checked if agree else 0)
                self.assertLess(nrm[f"tangent{form}_rotated_along_u"], checked)

    def test_positions_follow_the_surface(self):
        _, out = measure(block())
        self.assertLess(out["pos_edge_vs_random_median"], 0.5)
        self.assertEqual(out["pos_header_finite"], 1)


class Normal0x21Tests(unittest.TestCase):
    @staticmethod
    def encode(n, bits):
        """Octahedral encode of unit vector n into the 10-bit (bit 31 set) or 8-bit form."""
        x, y, z = n
        s = abs(x) + abs(y) + abs(z)
        x, y = x / s, y / s
        if z < 0:
            x, y = (1 - abs(y)) * (1 if x >= 0 else -1), (1 - abs(x)) * (1 if y >= 0 else -1)
        if bits == 10:
            q = [round((c + 1) * 1023 / 2) for c in (x, y)]
            return 0x80000000 | q[0] | q[1] << 10 | 0x2AA << 20
        q = [round((c + 1) * 127.5) for c in (x, y)]
        return q[0] | q[1] << 8 | 0x55 << 16

    def test_round_trip_both_widths_and_hemispheres(self):
        rng = random.Random(7)
        for _ in range(500):
            v = [rng.gauss(0, 1) for _ in range(3)]
            l = sum(c * c for c in v) ** 0.5
            n = [c / l for c in v]
            for bits, tol in ((10, 0.005), (8, 0.03)):
                d = vt.normal_0x21(self.encode(n, bits))
                self.assertAlmostEqual(sum(a * a for a in d), 1.0, places=9)
                self.assertGreater(sum(a * b for a, b in zip(d, n)), 1 - tol)

    def test_bit31_selects_the_width(self):
        # the same low 20 bits: with bit 31 they are 10-bit x = 1023, y = 512, i.e. +x; without it bytes 0-1 are
        # 0xFF, 0x03, an 8-bit pair near (+1, -1) that folds to a vector close to -z
        w = 1023 | 512 << 10
        hi = vt.normal_0x21(0x80000000 | w)
        lo = vt.normal_0x21(w)
        self.assertGreater(hi[0], 0.999)
        self.assertLess(lo[0], 0.1)
        self.assertLess(lo[2], -0.9)


class Tangent0x21Tests(unittest.TestCase):
    def test_10bit_tangent_is_unit_and_perpendicular_to_the_normal(self):
        rng = random.Random(3)
        for _ in range(2000):
            w = 0x80000000 | rng.getrandbits(31)
            n = vt.normal_0x21(w)
            t, sign = vt.tangent_0x21(w)
            self.assertAlmostEqual(sum(a * a for a in t), 1.0, places=9)
            self.assertLess(abs(sum(a * b for a, b in zip(n, t))), 1e-9)
            self.assertEqual(sign, 1 if (w >> 30) & 1 else -1)

    def test_10bit_angle_turns_the_tangent_about_the_normal(self):
        # normal +z: b1 = -x, b2 = -y, tangent = (-cos, sin, 0) for angle = v * 2pi / 1023 - pi
        base = 0x80000000 | 512 | 512 << 10
        for v in (0, 256, 511, 767, 1023):
            ang = v * 2 * math.pi / 1023 - math.pi
            t, _ = vt.tangent_0x21(base | v << 20)
            self.assertAlmostEqual(t[0], -math.cos(ang), delta=2e-3)
            self.assertAlmostEqual(t[1], math.sin(ang), delta=2e-3)

    def test_negative_z_normal_uses_the_other_basis_sign(self):
        # x = y = 0 folds to -z with the octahedral corner (1023, 1023); the basis stays perpendicular there
        w = 0x80000000 | 1023 | 1023 << 10 | 300 << 20
        n = vt.normal_0x21(w)
        t, _ = vt.tangent_0x21(w)
        self.assertLess(n[2], -0.99)
        self.assertLess(abs(sum(a * b for a, b in zip(n, t))), 1e-9)

    def test_8bit_tangent_is_octahedral_in_bits_16_to_29(self):
        t, sign = vt.tangent_0x21(127 << 16 | 64 << 23)
        self.assertGreater(t[0], 0.999)
        self.assertEqual(sign, -1)
        t, sign = vt.tangent_0x21(64 << 16 | 127 << 23 | 1 << 30)
        self.assertGreater(t[1], 0.999)
        self.assertEqual(sign, 1)


class BlockRunTests(unittest.TestCase):
    """Blocks stored back to back, with their groups after the whole run in the same order."""

    def test_two_blocks_then_two_groups(self):
        data = block(with_group=False) + block(grid=8, with_group=False) + padded_group() + padded_group(8)
        T, errors = measure_stream(data)
        self.assertEqual((T.get("blocks"), T["vertices"], T["run_blocks"], errors), (2, 100 + 64, 2, {}))

    def test_groups_out_of_order_fail_the_density_check(self):
        data = block(with_group=False) + block(grid=8, with_group=False) + padded_group(8) + padded_group()
        T, errors = measure_stream(data)
        self.assertEqual((T.get("blocks"), errors), (None, {"group does not cover the block densely": 2}))

    def test_blocks_without_a_group_in_the_stream_are_counted(self):
        data = block(with_group=False) + block(with_group=False) + block(with_group=False) + padded_group()
        T, errors = measure_stream(data)
        self.assertEqual((T.get("blocks"), errors), (1, {"no group in the stream for a later block of the run": 2}))

    def test_single_block_unchanged(self):
        T, errors = measure_stream(block())
        self.assertEqual((T.get("blocks"), T["run_blocks"], errors), (1, 1, {}))
        with self.assertRaises(vt.BlockError):
            measure_stream(block(with_group=False))


def block_with_streams(extra, grid=GRID):
    """block() with more (slot, tag, per-vertex bytes) streams appended after its own, before the group."""
    n = grid * grid
    buf = bytearray(block(grid=grid, with_group=False))
    for slot, tag, data in extra:
        struct.pack_into("<I", buf, 4 * slot, tag | ((len(buf) // 2) << 8))
        buf += data * n
    buf += b"\0" * (-len(buf) % 16)
    return bytes(buf + padded_group(grid))


def skin_block(weights, joints, pairs=1):
    """block() with weight slots 6(/7) (tag 0x26) and joint slots 8(/9) (tag 0x25); weights and joints are
    8-byte lists (two slots' worth), the first four bytes used when pairs == 1."""
    extra = [(6, 0x26, bytes(weights[:4])), (8, 0x25, bytes(joints[:4]))]
    if pairs == 2:
        extra = [(6, 0x26, bytes(weights[:4])), (7, 0x26, bytes(weights[4:])),
                 (8, 0x25, bytes(joints[:4])), (9, 0x25, bytes(joints[4:]))]
    return block_with_streams(extra)


class SkinTests(unittest.TestCase):
    N = GRID * GRID

    def test_two_pairs_of_sorted_weights_summing_to_255(self):
        _, out = measure(skin_block([100, 80, 40, 20, 10, 5, 0, 0], [3, 7, 1, 2, 9, 4, 0, 0], pairs=2))
        sk = out["skin"]
        self.assertEqual(sk["blocks"], 1)
        self.assertEqual(sk["pairs_2"], 1)
        for k in ("weight_sum_255", "weight_sum_247_255", "weights_descending", "nonzero_indices_distinct",
                  "zero_weight_zero_index"):
            self.assertEqual(sk[k], self.N, k)
        self.assertEqual(sk["index_max_0-15"], 1)
        self.assertEqual(sk["colour_vertices"], self.N)
        self.assertEqual(sk["colour_sum_255"], 0)

    def test_failures_are_counted(self):
        # sum 250, not descending, joint 40 repeated, a zero weight with a non-zero index
        _, out = measure(skin_block([50, 100, 100, 0], [40, 40, 2, 5]))
        sk = out["skin"]
        self.assertEqual(sk["pairs_1"], 1)
        self.assertEqual(sk["weight_sum_255"], 0)
        self.assertEqual(sk["weight_sum_247_255"], self.N)
        self.assertEqual(sk["weights_descending"], 0)
        self.assertEqual(sk["nonzero_indices_distinct"], 0)
        self.assertEqual(sk["zero_weight_zero_index"], 0)
        self.assertEqual(sk["index_max_32-47"], 1)

    def test_no_joint_slot_no_counts(self):
        _, out = measure(block())
        self.assertEqual(out["skin"], {})

    def test_joint_slot_without_weights_is_unpaired(self):
        _, out = measure(block_with_streams([(8, 0x25, bytes((1, 2, 3, 4)))]))
        self.assertEqual(out["skin"], {"unpaired": 1})


class ArchiveTests(unittest.TestCase):
    def test_end_to_end_on_a_fake_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rpf")
            body = zlib.compress(block())
            with open(path, "wb") as fh:
                fh.write(b"RPF9" + struct.pack("<I", 16) + b"\0" * 8 + body + zlib.compress(b"\x63\x30\x00\x00" * 8))
            out_json = os.path.join(tmp, "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", out_json], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out_json, encoding="utf-8") as fh:
                out = json.load(fh)
            self.assertEqual((out["streams"], out["blocks"]), (1, 1))
            self.assertEqual(out["tags"]["slot3:28"]["edge_jumps_signed"], 0)

    def test_block_run_on_a_fake_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fake.rpf")
            run = block(with_group=False) + block(grid=8, with_group=False) + padded_group() + padded_group(8)
            with open(path, "wb") as fh:
                fh.write(b"RPF9" + struct.pack("<I", 16) + b"\0" * 8 + zlib.compress(run))
            out_json = os.path.join(tmp, "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", out_json], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out_json, encoding="utf-8") as fh:
                out = json.load(fh)
            self.assertEqual((out["streams"], out.get("blocks"), out["skipped"]), (1, 2, {}))

    def test_directory_scan_uses_stable_archive_order(self):
        with tempfile.TemporaryDirectory() as root:
            def walk(path, onerror=None):
                directories = ["z", "a"]
                yield path, directories, []
                for directory in directories:
                    yield os.path.join(path, directory), [], ["one.rpf"]

            with (mock.patch.object(vt.os, "walk", side_effect=walk),
                  mock.patch.object(vt, "iter_blocks", return_value=()) as blocks,
                  contextlib.redirect_stdout(io.StringIO()),
                  contextlib.redirect_stderr(io.StringIO())):
                self.assertEqual(vt.main([root]), 0)

            self.assertEqual(blocks.call_args_list, [
                mock.call(os.path.join(root, "a", "one.rpf"), 0),
                mock.call(os.path.join(root, "z", "one.rpf"), 0),
            ])

    def test_negative_limit_is_an_error(self):
        r = subprocess.run([sys.executable, "-I", TOOL, TOOL, "--limit", "-1"], capture_output=True, text=True, errors="replace")
        self.assertEqual(r.returncode, 2)

    @staticmethod
    def _archive(folder):
        path = os.path.join(folder, "fake.rpf")
        data = b"RPF9" + struct.pack("<I", 16) + b"\0" * 8 + zlib.compress(block())
        with open(path, "wb") as fh:
            fh.write(data)
        return path, data

    def test_overlapping_targets_count_distinct_archives_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "nested")
            os.mkdir(sub)
            path, _ = self._archive(sub)
            self._archive(tmp)  # Same basename and bytes, different archive path.
            r = subprocess.run([sys.executable, "-I", TOOL, path, tmp, sub, os.path.join(sub, ".", "fake.rpf")],
                               capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 0, r.stderr)
            out = json.loads(r.stdout)
            self.assertEqual((out["streams"], out["blocks"]), (2, 2))

    def test_json_over_an_input_archive_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = self._archive(tmp)
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", path], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual(r.stdout, "")
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_over_an_archive_found_in_an_input_folder_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "pack")
            os.mkdir(sub)
            path, data = self._archive(sub)
            r = subprocess.run([sys.executable, "-I", TOOL, tmp, "--json", path], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("would overwrite the input", r.stderr)
            with open(path, "rb") as fh:
                self.assertEqual(fh.read(), data)

    def test_json_that_is_a_directory_is_refused_before_the_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", tmp], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("is a directory", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertEqual(r.stdout, "")

    def test_json_in_a_missing_folder_is_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            out = os.path.join(tmp, "reports", "vertex", "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", out], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 0, r.stderr)
            with open(out, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["blocks"], 1)

    def test_json_that_cannot_be_written_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self._archive(tmp)
            blocker = os.path.join(tmp, "blocker")
            with open(blocker, "w", encoding="utf-8") as fh:
                fh.write("a file, not a folder")
            out = os.path.join(blocker, "out.json")
            r = subprocess.run([sys.executable, "-I", TOOL, path, "--json", out], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("cannot write", r.stderr)
            self.assertNotIn("Traceback", r.stderr)
            self.assertIn('"blocks"', r.stdout)


def block27(word, grid=GRID):
    """block() with a tag-0x27 stream in slot 11 (the same word on every vertex), before the group."""
    buf = bytearray(block(grid=grid, with_group=False))
    struct.pack_into("<I", buf, 4 * 11, 0x27 | ((len(buf) // 2) << 8))
    buf += word * (grid * grid)
    buf += bytes(-len(buf) % 16)
    return bytes(buf + padded_group(grid))


def word27(x, y, z, w):
    """Inverse of tangent_0x27 for components in [-1, 1] and w in 0..3."""
    return struct.pack("<I", sum(round((c + 1) / 2 * 1023) << s for c, s in ((x, 0), (y, 10), (z, 20))) | w << 30)


class Tag27Tests(unittest.TestCase):
    N = GRID * GRID

    def test_decode_is_2_10_10_10_unorm_times_2_minus_1(self):
        t, sign = vt.tangent_0x27(struct.unpack("<I", word27(1, 0, 0, 3))[0])
        self.assertGreater(t[0], 0.999)
        self.assertEqual(sign, 1.0)
        t, sign = vt.tangent_0x27(struct.unpack("<I", word27(0, 0, -1, 0))[0])
        self.assertLess(t[2], -0.999)
        self.assertEqual(sign, -1.0)
        self.assertAlmostEqual(vt.tangent_0x27(2 << 30)[1], 1 / 3)

    def test_tangent_along_u_with_w_as_bitangent_sign(self):
        # flat grid: face normal +z, dP/du = +x, dP/dv = +y; cross(+z, +x) = +y, so it points along +v and
        # w = 0 (sign -1) is the "w > 0 exactly when -v" side; w = 3 is the other
        for w, agree in ((0, True), (3, False)):
            _, out = measure(block27(word27(1, 0, 0, w)))
            t = out["tag27"]
            self.assertEqual(t["11:vertices"], self.N)
            self.assertEqual(t["11:unit"], self.N)
            self.assertEqual(t[f"11:w{w}"], self.N)
            self.assertEqual(t["11:perp_face"], t["11:perp_face_n"])
            self.assertEqual(t["11:uv3_along_u"], t["11:uv3_n"])
            self.assertEqual(t["11:uv3_w_plus_minus_v"], t["11:uv3_along_u"] if agree else 0)

    def test_vector_along_the_normal_fails_the_checks(self):
        _, out = measure(block27(word27(0, 0, 1, 3)))
        t = out["tag27"]
        self.assertEqual(t["11:unit"], self.N)
        self.assertEqual(t["11:perp_face"], 0)
        self.assertNotIn("11:uv3_along_u", t)

    def test_non_unit_vectors_are_counted(self):
        _, out = measure(block27(word27(0.5, 0, 0, 3)))
        self.assertEqual(out["tag27"]["11:unit"], 0)

    def test_no_0x27_slot_no_counts(self):
        _, out = measure(block())
        self.assertEqual(out["tag27"], {})


class Rpf9ValidationTests(unittest.TestCase):
    """RPF9 archive header validation in iter_blocks (contract shared with rpf9_names.scan_archive)."""

    @staticmethod
    def _run(args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, errors="replace")

    def test_invalid_headers_are_errors(self):
        cases = [
            ("empty", b""),
            ("short header", b"RPF9"),
            ("wrong magic", b"XRP9" + struct.pack("<I", 16) + b"\0" * 8),
            ("data start before the header", b"RPF9" + struct.pack("<I", 0) + b"\0" * 8),
            ("data start inside the header", b"RPF9" + struct.pack("<I", 4) + b"\0" * 8),
            ("data start past the end", b"RPF9" + struct.pack("<I", 0x7FFFFFFF) + b"\0" * 8),
        ]
        for name, data in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    root = pathlib.Path(tmp)
                    path = root / "bad.rpf"
                    path.write_bytes(data)
                    out = root / "out.json"
                    out.write_text('{"sentinel": true}', encoding="utf-8")
                    r = self._run([str(path), "--json", str(out)])
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertEqual(r.stdout, "")
                    self.assertIn(path.name, r.stderr)
                    self.assertIn("RPF9", r.stderr)
                    self.assertNotIn("Traceback", r.stderr)
                    self.assertEqual(out.read_text(encoding="utf-8"), '{"sentinel": true}')

    def test_valid_header_only_and_reversed_magic(self):
        cases = [
            ("header only", b"RPF9" + struct.pack("<I", 8), 0),
            ("reversed magic", b"9FPR" + struct.pack("<I", 16) + b"\0" * 8 + zlib.compress(block()), 1),
        ]
        for name, data, streams in cases:
            with self.subTest(name=name):
                with tempfile.TemporaryDirectory() as tmp:
                    path = pathlib.Path(tmp) / "ok.rpf"
                    path.write_bytes(data)
                    r = self._run([str(path)])
                    self.assertEqual(r.returncode, 0, r.stderr)
                    self.assertEqual(json.loads(r.stdout)["streams"], streams)

    def test_unlistable_subfolder_is_a_clean_error_not_a_silent_skip(self):
        # os.walk skips a directory it cannot list unless onerror raises; the scan used to report an
        # empty directory and exit 0, hiding the incomplete scan.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            (sub / "x.rpf").write_bytes(b"RPF9" + struct.pack("<I", 8))
            os.chmod(sub, 0o000)
            try:
                try:
                    os.listdir(sub)
                    self.skipTest("the sub-folder is still listable (running as root, or Windows)")
                except OSError:
                    pass
                r = self._run([str(root)])
                self.assertEqual(r.returncode, 2, r.stderr)
                self.assertIn("error:", r.stderr)
                self.assertNotIn("Traceback", r.stderr)
            finally:
                os.chmod(sub, 0o755)

    def test_unlistable_subfolder_fails_when_chmod_is_not_enough(self):
        # The test above self-skips on Windows and as root, and CI only runs this file on
        # windows-latest, so the guard would go unexercised there. Forcing os.scandir to raise for
        # one folder reproduces the same PermissionError os.walk hits for real, on every platform.
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            sub = root / "sub"
            sub.mkdir()
            (sub / "x.rpf").write_bytes(b"RPF9" + struct.pack("<I", 8))
            real_scandir = os.scandir
            locked = os.path.normcase(os.path.abspath(str(sub)))

            def scandir(path, *args, **kwargs):
                if os.path.normcase(os.path.abspath(str(path))) == locked:
                    raise PermissionError(13, "Access is denied", str(path))
                return real_scandir(path, *args, **kwargs)

            err = io.StringIO()
            with mock.patch.object(vt.os, "scandir", scandir):
                with contextlib.redirect_stderr(err):
                    code = vt.main([str(root)])
            self.assertEqual(code, 2, err.getvalue())
            self.assertIn("error:", err.getvalue())
            self.assertIn("sub", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
