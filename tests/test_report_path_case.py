"""Report paths that differ only in letter case, in the tools that compare two report paths before writing.

    python -I tests/test_report_path_case.py

On a case-insensitive volume (macOS's default APFS, Windows) `o/B.json` and `o/b.json` are one file, so a tool that
accepted both as `--out` and `--csv` wrote the CSV over the JSON. On a case-sensitive volume they are two reports and
must stay allowed. Everything runs in a temporary folder on hand-built inputs; no game data.
"""
from __future__ import annotations

import importlib.util
import os
import pathlib
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import report_paths  # noqa: E402

# tool -> name of its two-path comparison helper
HELPERS = {
    "bvh_index": "same_output_path",
    "anim_index": "same_report_path",
    "gamedata_xml": None,
    "gamedata_index": None,
    "navmesh_decode": "same_report_path",
    "world_census": "same_report_path",
}


def load(name):
    spec = importlib.util.spec_from_file_location(f"case_{name}", TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper(module, name):
    if HELPERS[name]:
        return getattr(module, HELPERS[name])
    found = [v for k, v in vars(module).items()
             if callable(v) and k.startswith("same_") and k.endswith("_path")]
    assert len(found) == 1, (name, found)
    return found[0]


class ReportPathCaseTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        probe = self.dir / "Probe"
        probe.touch()
        self.insensitive = (self.dir / "probe").exists()
        probe.unlink()

    def tearDown(self):
        self._tmp.cleanup()

    def test_new_paths_that_differ_only_in_case(self):
        upper, lower = self.dir / "new" / "B.json", self.dir / "new" / "b.json"
        for name in HELPERS:
            with self.subTest(tool=name):
                same = helper(load(name), name)
                self.assertTrue(same(str(upper), str(upper)))
                self.assertEqual(same(str(upper), str(lower)), self.insensitive)
                self.assertFalse(same(str(upper), str(self.dir / "new" / "c.json")))

    def test_new_paths_that_differ_only_in_accent_form(self):
        if sys.platform != "darwin" or not self.insensitive:
            self.skipTest("APFS/HFS+ name normalization")
        composed, decomposed = self.dir / "café.json", self.dir / "CAFÉ.json"
        for name in HELPERS:
            with self.subTest(tool=name):
                self.assertTrue(helper(load(name), name)(str(composed), str(decomposed)))

    def test_world_census_follows_the_folding_volume(self):
        """The archive guard asks report_paths, so a case-insensitive APFS volume refuses a report that would
        overwrite the archive; its own realpath+normcase copy answered "two files" there."""
        module = load("world_census")
        composed = self.dir / "café.json"
        decomposed = self.dir / unicodedata.normalize("NFD", "café.json")
        folding = [True]
        with mock.patch.object(report_paths, "sys", mock.Mock(platform="darwin")), \
                mock.patch.object(report_paths, "case_insensitive_volume", lambda path: folding[0]):
            self.assertTrue(module.same_report_path(str(composed), str(decomposed)))
            folding[0] = False
            self.assertFalse(module.same_report_path(str(composed), str(decomposed)))

    def test_bvh_index_refuses_case_only_report_pair(self):
        source = self.dir / "in.rpf"
        source.write_bytes(bytes(4096))
        out = self.dir / "o"
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "bvh_index.py"), str(source),
                            "--out", str(out / "B.json"), "--csv", str(out / "b.json")],
                           cwd=ROOT, capture_output=True, text=True, timeout=60)
        if self.insensitive:
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--out and --csv must use different paths", r.stderr)
            self.assertFalse(out.exists() and any(out.iterdir()))
        else:
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(sorted(p.name for p in out.iterdir()), ["B.json", "b.json"])


if __name__ == "__main__":
    unittest.main()
