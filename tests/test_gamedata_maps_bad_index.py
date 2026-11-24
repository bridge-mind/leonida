"""Bad carve indexes must give one error line in gamedata_maps.py, never a traceback.

Run with ``python -I tests/test_gamedata_maps_bad_index.py``. Set ``LEONIDA_GAMEDATA_MAPS`` to another copy
of the tool to use as a negative control (main's version fails the two TypeError cases). All inputs are
hand-built; no game data is used.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = pathlib.Path(os.environ.get("LEONIDA_GAMEDATA_MAPS", str(ROOT / "tools" / "gamedata_maps.py")))


class BadGamedataMapsIndexTest(unittest.TestCase):
    def run_tool(self, carve):
        return subprocess.run([sys.executable, "-I", str(TOOL), str(carve)],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def make_carve(self, folder, doc=None, text=None):
        carve = pathlib.Path(folder) / "x.rpf"
        carve.mkdir()
        index = carve / "index.json"
        if text is not None:
            index.write_text(text, encoding="utf-8")
        else:
            index.write_text(json.dumps(doc), encoding="utf-8")
        return carve

    def assert_one_error(self, result):
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
        self.assertTrue(result.stderr.startswith("error:"), result.stderr)

    def assert_missing_source(self, result, target):
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
        self.assertTrue(result.stderr.startswith(f"ERROR: {target}:"), result.stderr)

    def test_source_is_a_file(self):
        # main's version names the index it could not open ('<file>\index.json') and exits 1.
        with tempfile.TemporaryDirectory() as td:
            source = pathlib.Path(td) / "not-a-carve.txt"
            source.write_text("junk", encoding="utf-8")
            self.assert_missing_source(self.run_tool(source), source)

    def test_folder_without_index(self):
        # A folder of carved .bin files whose index.json is missing, as main also fails on.
        with tempfile.TemporaryDirectory() as td:
            carve = pathlib.Path(td)
            (carve / "000000000001.bin").write_bytes(b"MHIU" + bytes(40))
            self.assert_missing_source(self.run_tool(carve), carve)

    def test_missing_files_key(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, {})))

    def test_files_is_a_string(self):
        # TypeError in read_payloads: the one shape main's except tuple did not catch.
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, {"files": "oops"})))

    def test_files_is_null(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, {"files": None})))

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, text="{oops")))

    def test_row_without_off(self):
        with tempfile.TemporaryDirectory() as td:
            self.assert_one_error(self.run_tool(self.make_carve(td, {"files": [{"size": 4}]})))

    def test_empty_carve_still_reports(self):
        with tempfile.TemporaryDirectory() as td:
            result = self.run_tool(self.make_carve(td, {"files": []}))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("no MHIU or PAMH payloads", result.stdout)


if __name__ == "__main__":
    unittest.main()
