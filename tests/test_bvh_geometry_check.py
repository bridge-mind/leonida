"""Synthetic checks for tools/bvh_geometry_check.py: the right geometry passes, a shifted, smaller or re-indexed one
fails the check that should catch it. No game data is read.

    python -I tests/test_bvh_geometry_check.py
"""
import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bvh_geometry_check as C  # noqa: E402
import bvh_rehydrate as R  # noqa: E402
from test_bvh_rehydrate import build  # noqa: E402

# Two triangles sharing the edge (1, 2); bounds x 0..1, y 0..1, z 0
VB = struct.pack("<12f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0)
IB = struct.pack("<6I", 0, 1, 2, 2, 1, 3)
TRUE = R.Geometry(VB, indices=IB)
LEAVES = [(0, 0x81, 0, 1, 0x09)]          # one leaf holding both triangles
BOUNDS = (0, 0, 0, 1, 1, 0)


class TestChecks(unittest.TestCase):
    def test_true_geometry_is_consistent(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        r = C.check(src, [TRUE])
        self.assertTrue(r["consistent"], r)
        self.assertEqual((r["leaves_outside"], r["bounds_gap"], r["pairs"]), (0, 0.0, {"0": [1, 1]}))
        self.assertGreater(r["internal_area"], 0.0)

    def test_shifted_geometry_fails_bounds(self):
        shifted = R.Geometry(VB, indices=IB, transform=((1, 0, 0, 0.5), (0, 1, 0, 0), (0, 0, 1, 0)))
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        r = C.check(src, [shifted])
        self.assertFalse(r["consistent"])
        self.assertEqual(r["leaves_outside"], 1)
        self.assertAlmostEqual(r["bounds_gap"], 0.5, places=6)

    def test_smaller_geometry_inside_the_bounds_still_fails(self):
        src, _, _ = build(LEAVES, bounds=(0, 0, 0, 2, 1, 0))
        r = C.check(src, [TRUE])
        self.assertEqual(r["leaves_outside"], 0)
        self.assertFalse(r["consistent"])

    def test_reindexed_geometry_fails_pairs_without_positions(self):
        # same triangles count, but the second one no longer shares an edge with the first
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        loose = R.Geometry(VB + struct.pack("<6f", 1, 1, 0, 1, 1, 0), indices=struct.pack("<6I", 0, 1, 2, 3, 4, 5))
        r = C.check(src, [loose])
        self.assertEqual(r["pairs"], {"0": [0, 1]})
        self.assertFalse(r["consistent"])

    def test_overrides(self):
        tight = build([(0, 0x01, 0, None, 0x09), (0, 0x01, 1, None, 0x09)], overrides={6: 0.25}, bounds=BOUNDS)[0]
        self.assertEqual(C.check(tight, [TRUE])["overrides_consistent"], 1)
        one_ulp = struct.unpack("<f", struct.pack("<I", 0x3FFFFFFF))[0]          # just below 2.0, slot 1 max x is 1.0
        loose = build([(0, 0x01, 0, None, 0x09), (0, 0x01, 1, None, 0x09)], overrides={9: one_ulp}, bounds=BOUNDS)[0]
        self.assertEqual(C.check(loose, [TRUE])["overrides_inconsistent"], 1)
        near = struct.unpack("<f", struct.pack("<I", 0x3F7FFFFF))[0]             # one ulp below 1.0: within tolerance
        ok = build([(0, 0x01, 0, None, 0x09), (0, 0x01, 1, None, 0x09)], overrides={9: near}, bounds=BOUNDS)[0]
        self.assertEqual(C.check(ok, [TRUE])["overrides_inconsistent"], 0)

    def test_command_line(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        with tempfile.TemporaryDirectory() as d:
            paths = [os.path.join(d, n) for n in ("in.bin", "vb.bin", "ib.bin", "r.json")]
            for p, data in zip(paths, (src, VB, IB)):
                with open(p, "wb") as f:
                    f.write(data)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = C.main([paths[0], "--geom", paths[1] + "," + paths[2], "--json", paths[3]])
            self.assertEqual(rc, 0)
            self.assertIn("geometry 0 1/1", out.getvalue())
            self.assertTrue(os.path.getsize(paths[3]) > 0)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(C.main([paths[0], "--geom", paths[1] + "," + paths[2], "--transform", "1,0,0,9,0,1,0,0,0,0,1,0"]), 1)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(C.main([paths[0]]), 2)

    def test_json_must_not_overwrite_input_or_buffers(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        with tempfile.TemporaryDirectory() as d:
            paths = [os.path.join(d, n) for n in ("in.bin", "vb.bin", "ib.bin", "r.json")]
            for p, data in zip(paths, (src, VB, IB)):
                with open(p, "wb") as f:
                    f.write(data)
            # --json names the dehydrated input itself
            for target in (paths[0], paths[1], paths[2]):
                with contextlib.redirect_stderr(io.StringIO()) as err:
                    rc = C.main([paths[0], "--geom", paths[1] + "," + paths[2], "--json", target])
                self.assertEqual(rc, 2)
                self.assertIn("must not overwrite", err.getvalue())
            # a hard link to the input also counts
            link = os.path.join(d, "link.json")
            try:
                os.link(paths[0], link)
            except (OSError, NotImplementedError, AttributeError) as e:
                self.skipTest(f"hard links unavailable: {e}")
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rc = C.main([paths[0], "--geom", paths[1] + "," + paths[2], "--json", link])
            self.assertEqual(rc, 2)
            self.assertIn("must not overwrite", err.getvalue())


if __name__ == "__main__":
    unittest.main()
