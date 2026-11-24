"""gamedata_index.py report paths: create the parent folder and never overwrite the input index.

Run with ``python -I tests/test_gamedata_index_report_paths.py``. Set ``LEONIDA_GAMEDATA_INDEX`` to
another copy of the tool to use as a negative control (main's version overwrites the input index and
tracebacks on a missing report folder). All inputs are hand-built; no game data is used.
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOL = pathlib.Path(os.environ.get("LEONIDA_GAMEDATA_INDEX", str(ROOT / "tools" / "gamedata_index.py")))


class ReportPathTest(unittest.TestCase):
    def make_carve(self, folder):
        carve = pathlib.Path(folder) / "x.rpf"
        carve.mkdir()
        (carve / "index.json").write_text(json.dumps({"files": []}), encoding="utf-8")
        return carve

    def run_tool(self, carve, *extra):
        return subprocess.run([sys.executable, "-I", str(TOOL), str(carve), *extra],
                              cwd=ROOT, capture_output=True, text=True, timeout=60)

    def test_missing_json_folder_is_created(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            out = pathlib.Path(td) / "missing" / "sub" / "gamedata.json"
            result = self.run_tool(carve, "--json", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file(), result.stderr)
            self.assertIn("summary", json.loads(out.read_text(encoding="utf-8")))

    def test_missing_csv_folder_is_created(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            out = pathlib.Path(td) / "missing" / "sub" / "gamedata.csv"
            result = self.run_tool(carve, "--csv", str(out))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.is_file(), result.stderr)
            self.assertIn("offset", out.read_text(encoding="utf-8").splitlines()[0])

    def test_json_cannot_overwrite_the_input_index(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            index = carve / "index.json"
            before = index.read_text(encoding="utf-8")
            result = self.run_tool(carve, "--json", str(index))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(index.read_text(encoding="utf-8"), before)

    def test_csv_cannot_overwrite_the_input_index(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            index = carve / "index.json"
            before = index.read_text(encoding="utf-8")
            result = self.run_tool(carve, "--csv", str(index))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("would overwrite the input", result.stderr)
            self.assertEqual(index.read_text(encoding="utf-8"), before)

    def test_json_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            outdir = pathlib.Path(td) / "outdir"
            outdir.mkdir()
            result = self.run_tool(carve, "--json", str(outdir))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("is a directory", result.stderr)

    def test_csv_cannot_be_a_directory(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            outdir = pathlib.Path(td) / "outdir"
            outdir.mkdir()
            result = self.run_tool(carve, "--csv", str(outdir))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("is a directory", result.stderr)

    def test_json_and_csv_must_differ(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            out = pathlib.Path(td) / "report.bin"
            result = self.run_tool(carve, "--json", str(out), "--csv", str(out))
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("must use different paths", result.stderr)

    def test_reports_in_a_writable_folder_still_work(self):
        with tempfile.TemporaryDirectory() as td:
            carve = self.make_carve(td)
            js = pathlib.Path(td) / "report.json"
            csv_path = pathlib.Path(td) / "report.csv"
            result = self.run_tool(carve, "--json", str(js), "--csv", str(csv_path))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(js.is_file() and csv_path.is_file(), result.stderr)


if __name__ == "__main__":
    unittest.main()
