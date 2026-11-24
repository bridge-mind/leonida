"""Tests for command-line fixture argument handling in w64_graphics.py (TEST-649)."""
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT_PATH = os.path.join(ROOT, "runtime", "test", "shaders", "w64_graphics.py")

spec = importlib.util.spec_from_file_location("w64_graphics", SCRIPT_PATH)
w64_graphics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w64_graphics)


class TestW64GraphicsArgParsing(unittest.TestCase):
    def test_rejects_unknown_fixture_single(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = w64_graphics.main(["dummy_spirv", "bad_fixture_typo"])
        self.assertEqual(rc, 1)
        self.assertIn("Unknown fixture name: bad_fixture_typo", stderr.getvalue())

    def test_rejects_unknown_fixtures_multiple(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = w64_graphics.main(["dummy_spirv", "bad1", "bad2"])
        self.assertEqual(rc, 1)
        self.assertIn("Unknown fixture names: bad1, bad2", stderr.getvalue())

    def test_rejects_unknown_fixture_with_tools_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_tools:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = w64_graphics.main(["dummy_spirv", tmp_tools, "bad_with_tools"])
            self.assertEqual(rc, 1)
            self.assertIn("Unknown fixture name: bad_with_tools", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
