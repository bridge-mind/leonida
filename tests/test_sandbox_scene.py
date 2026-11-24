"""tools/sandbox_scene.py: entity world rows, nearest-first selection and the packed vertex layout."""
import io
import json
import os
import sys
import struct
import tempfile
import tracemalloc
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tools'))
import sandbox_scene

class TestSandboxScene(unittest.TestCase):
    def test_draw_rows(self):
        result = sandbox_scene.draw_rows((0, 0, 0, 1), 1.0, 1.0, (10, 20, 30))
        self.assertAlmostEqual(result[0][0], 1.0, places=6)
        self.assertAlmostEqual(result[0][1], 0.0, places=6)
        self.assertAlmostEqual(result[0][2], 0.0, places=6)
        self.assertAlmostEqual(result[1][0], 0.0, places=6)
        self.assertAlmostEqual(result[1][1], 1.0, places=6)
        self.assertAlmostEqual(result[1][2], 0.0, places=6)
        self.assertAlmostEqual(result[2][0], 0.0, places=6)
        self.assertAlmostEqual(result[2][1], 0.0, places=6)
        self.assertAlmostEqual(result[2][2], 1.0, places=6)
        self.assertAlmostEqual(result[3][0], 10.0, places=6)
        self.assertAlmostEqual(result[3][1], 20.0, places=6)
        self.assertAlmostEqual(result[3][2], 30.0, places=6)

    def test_draw_rows_rotation(self):
        result = sandbox_scene.draw_rows((0, 0, -0.70710678, 0.70710678), 1, 1, (0, 0, 0))
        self.assertAlmostEqual(result[0][0], 0.0, places=6)
        self.assertAlmostEqual(result[0][1], 1.0, places=6)
        self.assertAlmostEqual(result[0][2], 0.0, places=6)
        self.assertAlmostEqual(result[1][0], -1.0, places=6)
        self.assertAlmostEqual(result[1][1], 0.0, places=6)
        self.assertAlmostEqual(result[1][2], 0.0, places=6)
        self.assertAlmostEqual(result[2][0], 0.0, places=6)
        self.assertAlmostEqual(result[2][1], 0.0, places=6)
        self.assertAlmostEqual(result[2][2], 1.0, places=6)

    def test_draw_rows_scale(self):
        result = sandbox_scene.draw_rows((0, 0, 0, 1), 2.0, 3.0, (0, 0, 0))
        self.assertAlmostEqual(result[0][0], 2.0, places=6)
        self.assertAlmostEqual(result[0][1], 0.0, places=6)
        self.assertAlmostEqual(result[0][2], 0.0, places=6)
        self.assertAlmostEqual(result[1][0], 0.0, places=6)
        self.assertAlmostEqual(result[1][1], 2.0, places=6)
        self.assertAlmostEqual(result[1][2], 0.0, places=6)
        self.assertAlmostEqual(result[2][0], 0.0, places=6)
        self.assertAlmostEqual(result[2][1], 0.0, places=6)
        self.assertAlmostEqual(result[2][2], 3.0, places=6)

    def test_select_rows(self):
        rows = [
            {"_pos": [1, 1, 0]},
            {"_pos": [5, 5, 0]},
            {"_pos": [20, 20, 0]},
            {"_pos": [3, 3, 0]}
        ]
        result = sandbox_scene.select_rows(rows, (0, 0), 10, 10)
        self.assertEqual(len(result), 3)
        self.assertAlmostEqual(result[0]["_pos"][0], 1.0, places=6)
        self.assertAlmostEqual(result[0]["_pos"][1], 1.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][0], 3.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][1], 3.0, places=6)
        self.assertAlmostEqual(result[2]["_pos"][0], 5.0, places=6)
        self.assertAlmostEqual(result[2]["_pos"][1], 5.0, places=6)

        result = sandbox_scene.select_rows(rows, (0, 0), 10, 2)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0]["_pos"][0], 1.0, places=6)
        self.assertAlmostEqual(result[0]["_pos"][1], 1.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][0], 3.0, places=6)
        self.assertAlmostEqual(result[1]["_pos"][1], 3.0, places=6)

    def test_full_selection_memory(self):
        # Searching past missing models needs every nearby candidate, without
        # retaining distance/row tuples and duplicate full-size reference lists.
        rows = [{"_pos": (0.0, 0.0)}] * 100_000
        tracemalloc.start()
        try:
            picked = sandbox_scene.select_rows(rows, (0, 0), 1, len(rows))
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(len(picked), len(rows))
        self.assertTrue(all(row is rows[0] for row in picked))
        # Allow eight pointers per row, including temporary sorting storage.
        # The original tuple-based selection needs more than twelve on x64.
        self.assertLess(peak, 8 * struct.calcsize("P") * len(rows))

    def test_mesh_bytes(self):
        prim = {
            "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "normals": [(0, 0, 1)] * 3,
            "uvs": [(0, 0), (1, 0), (0, 1)],
            "indices": [0, 1, 2]
        }
        b = sandbox_scene.mesh_bytes(prim)
        self.assertEqual(len(b), 116)
        self.assertEqual(tuple(struct.unpack_from('<II', b, 0)), (3, 3))
        self.assertEqual(tuple(struct.unpack_from('<8f', b, 40)), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0))
        self.assertEqual(tuple(struct.unpack_from('<3I', b, 104)), (0, 1, 2))

    def test_mesh_bytes_no_normals(self):
        prim = {
            "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "normals": [],
            "indices": [0, 1, 2]
        }
        b = sandbox_scene.mesh_bytes(prim)
        self.assertEqual(tuple(struct.unpack_from('<8f', b, 8)), (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        self.assertEqual(tuple(struct.unpack_from('<8f', b, 40)), (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0))

class TestCliErrors(unittest.TestCase):
    """An unreadable input or an unwritable --out is one ERROR: line and exit 2, not a traceback."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.models = os.path.join(self.tmp.name, "models")
        os.makedirs(self.models)
        self.entities = os.path.join(self.tmp.name, "e.jsonl")
        with open(self.entities, "w", encoding="utf-8") as f:
            f.write(json.dumps({"archetype": 123, "position": [0, 0, 0],
                                "rotation": [0, 0, 0, 1], "lod_level": 0}) + "\n")

    def _run(self, entities, out):
        err = io.StringIO()
        with redirect_stderr(err):
            rc = sandbox_scene.main(["--entities", entities, "--models", self.models,
                                     "--center", "0", "0", "--out", out])
        return rc, err.getvalue()

    def test_out_directory_is_one_error_line(self):
        out = os.path.join(self.tmp.name, "iamdir")
        os.makedirs(out)
        rc, err = self._run(self.entities, out)
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("ERROR:"))
        # the refusal is up front and says why, not a repr'd OSError with doubled backslashes
        self.assertIn("is a folder, not a scene file", err)
        self.assertNotIn("[Errno", err)

    def test_out_under_a_regular_file_is_one_error_line(self):
        reg = os.path.join(self.tmp.name, "reg")
        with open(reg, "w", encoding="utf-8") as f:
            f.write("x")
        rc, err = self._run(self.entities, os.path.join(reg, "o.bin"))
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("ERROR:"))

    def test_missing_entities_is_one_error_line(self):
        rc, err = self._run(os.path.join(self.tmp.name, "nope.jsonl"),
                            os.path.join(self.tmp.name, "o.bin"))
        self.assertEqual(rc, 2)
        self.assertEqual(len(err.strip().splitlines()), 1)
        self.assertTrue(err.startswith("ERROR:"))

    def test_writable_out_still_builds(self):
        out = os.path.join(self.tmp.name, "o.bin")
        err = io.StringIO()
        with redirect_stderr(err):
            rc = sandbox_scene.main(["--entities", self.entities, "--models", self.models,
                                     "--center", "0", "0", "--out", out])
        self.assertEqual(rc, 0)
        self.assertEqual(err.getvalue(), "")
        with open(out, "rb") as f:
            self.assertEqual(f.read(4), b"LSB1")


if __name__ == "__main__":
    unittest.main()
