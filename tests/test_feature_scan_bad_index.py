"""Malformed index shapes must be one-line errors or empty counts, never a traceback.

Run with ``python -I tests/test_feature_scan_bad_index.py``. Set ``LEONIDA_FEATURE_SCAN``
to another copy of the tool to use as a negative control (main's version fails these
tests). All inputs are built here by hand; no game data is used.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = Path(os.environ.get("LEONIDA_FEATURE_SCAN", str(ROOT / "tools" / "feature_scan.py")))


def make_work(folder, dxbc_rows, spirv=None, spirv_text=None):
    shaders = Path(folder) / "shaders"
    (shaders / "dxbc").mkdir(parents=True)
    (shaders / "spirv").mkdir(parents=True)
    (shaders / "dxbc" / "index.json").write_text(json.dumps(dxbc_rows), encoding="utf-8")
    if spirv_text is not None:
        (shaders / "spirv" / "index.json").write_text(spirv_text, encoding="utf-8")
    elif spirv is not None:
        (shaders / "spirv" / "index.json").write_text(json.dumps(spirv), encoding="utf-8")
    return shaders


class BadFeatureScanIndexTest(unittest.TestCase):
    def run_scan(self, shaders, out=None):
        argv = [sys.executable, "-I", str(TOOL), str(shaders)]
        if out:
            argv += ["--out", str(out)]
        return subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=60)

    def assert_clean(self, result):
        self.assertNotIn("Traceback", result.stderr, result.stderr)

    def test_report_write_failure_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as td:
            shaders = make_work(td, [])
            blocker = Path(td) / "blocker"
            blocker.write_bytes(b"keep")
            for out in (Path(td), blocker / "report.json"):
                with self.subTest(out=out):
                    result = self.run_scan(shaders, out)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(f"ERROR: cannot write {out}", result.stderr)
                    self.assertEqual(len(result.stderr.strip().splitlines()), 1)
                    self.assert_clean(result)
            self.assertEqual(blocker.read_bytes(), b"keep")

    def test_non_object_spirv_indexes_are_empty_counts(self):
        for value in (None, 42, True, "x"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as td:
                shaders = make_work(td, [], spirv_text=json.dumps(value))
                result = self.run_scan(shaders)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assert_clean(result)
                self.assertIn("uses_code_base by stage: {}", result.stdout)

    def test_broken_spirv_json_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as td:
            shaders = make_work(td, [], spirv_text="{oops")
            result = self.run_scan(shaders)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("ERROR: cannot read", result.stderr)
            self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)
            self.assert_clean(result)

    def test_non_object_dxbc_rows_are_counted_as_unreadable(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [1, "x", None, {}, {"name": "good", "file": "good.dxbc", "stage": "ps"}]
            shaders = make_work(td, rows)
            result = self.run_scan(shaders)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assert_clean(result)
            self.assertIn("could not be read", result.stdout)

    def test_bad_fields_do_not_stop_the_scan(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [
                {"file": ["bad.dxbc"], "stage": "vs"},
                {"file": "missing.dxbc", "stage": {"invalid": True}},
                {"file": "missing.dxbc", "stage": 1},
            ]
            spirv = [{"stage": ["bad"], "uses_code_base": True}, {"stage": "ps"}]
            shaders = make_work(td, rows, spirv=spirv)
            out = Path(td) / "report.json"
            result = self.run_scan(shaders, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["dxbc"]["unreadable"], 3)
            self.assertEqual(report["dxbc"]["stages"]["vs"], 1)
            self.assertEqual(report["dxbc"]["stages"]["?"], 2)
            self.assertEqual(report["spirv"]["translated"], {"?": 1, "ps": 1})
            self.assertEqual(report["spirv"]["uses_code_base"], {"?": 1})

    def test_valid_indexes_are_still_counted(self):
        with tempfile.TemporaryDirectory() as td:
            rows = [{"name": "ps_a", "file": "ps_a.dxbc", "stage": "ps"}]
            shaders = make_work(td, rows, spirv={"ps_a": {"stage": "ps", "uses_code_base": True}})
            out = Path(td) / "report.json"
            result = self.run_scan(shaders, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["spirv"]["translated"], {"ps": 1})
            self.assertEqual(report["spirv"]["uses_code_base"], {"ps": 1})


if __name__ == "__main__":
    unittest.main()
