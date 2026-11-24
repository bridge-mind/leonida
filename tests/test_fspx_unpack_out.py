"""Unit tests for the tools/fspx_unpack.py output-directory guard: an unusable --out (a regular file
at --out, a regular file above the unpack target, or a regular file where fspx/disasm must be
created) must give one stderr line and exit 2 at main start, before any input is read, instead of
a raw OSError traceback.

    python -I tests/test_fspx_unpack_out.py

Zero game data: every fixture is a plain synthetic file built here.
"""
import contextlib
import io
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fspx_unpack  # noqa: E402

INPUT_NAME = "payload.bin"
GARBAGE_INPUT = b"not an fspx payload"  # never parsed: the guard refuses before any input is read


class FspxOutputDirGuardTests(unittest.TestCase):
    def run_unpack(self, out, disasm=False):
        with tempfile.TemporaryDirectory() as t:
            source = os.path.join(t, INPUT_NAME)
            with open(source, "wb") as f:
                f.write(GARBAGE_INPUT)
            err, out_text = io.StringIO(), io.StringIO()
            with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out_text):
                rc = fspx_unpack.main([source, "--out", out] + (["--disasm"] if disasm else []))
            return rc, err.getvalue(), out_text.getvalue()

    def test_regular_file_at_out_gives_one_line_and_exit_2(self):
        with tempfile.TemporaryDirectory() as t:
            blocker = os.path.join(t, "blocker")
            with open(blocker, "wb") as f:
                f.write(b"x")
            rc, err, _ = self.run_unpack(blocker)
            self.assertEqual(rc, 2)
            self.assertEqual(err.count("\n"), 1)
            self.assertTrue(err.startswith("cannot create output directory "))
            self.assertIn(os.path.join(blocker, "fspx"), err)
            self.assertFalse(os.path.exists(os.path.join(blocker, "fspx")))

    def test_regular_file_above_out_gives_one_line_and_exit_2(self):
        with tempfile.TemporaryDirectory() as t:
            blocker = os.path.join(t, "blocker")
            with open(blocker, "wb") as f:
                f.write(b"x")
            child = os.path.join(blocker, "child")
            rc, err, _ = self.run_unpack(child)
            self.assertEqual(rc, 2)
            self.assertEqual(err.count("\n"), 1)
            self.assertTrue(err.startswith("cannot create output directory "))
            self.assertFalse(os.path.exists(child))

    def test_regular_file_where_fspx_belongs_gives_one_line_and_exit_2(self):
        with tempfile.TemporaryDirectory() as t:
            fspx_file = os.path.join(t, "fspx")
            with open(fspx_file, "wb") as f:
                f.write(b"x")
            rc, err, _ = self.run_unpack(t)
            self.assertEqual(rc, 2)
            self.assertEqual(err.count("\n"), 1)
            self.assertIn(fspx_file, err)
            with open(fspx_file, "rb") as f:  # the blocker stays a regular file, not replaced
                self.assertEqual(f.read(), b"x")

    def test_disasm_dir_refusal_names_the_disasm_dir(self):
        with tempfile.TemporaryDirectory() as t:
            os.makedirs(t, exist_ok=True)
            disasm_file = os.path.join(t, "disasm")
            with open(disasm_file, "wb") as f:
                f.write(b"x")
            rc, err, _ = self.run_unpack(t, disasm=True)
            self.assertEqual(rc, 2)
            self.assertEqual(err.count("\n"), 1)
            self.assertIn(disasm_file, err)
            self.assertFalse(os.path.isdir(disasm_file))

    def test_valid_out_still_unpacks_and_reports_failed_inputs(self):
        with tempfile.TemporaryDirectory() as t:
            rc, err, out = self.run_unpack(os.path.join(t, "unpack"))
            self.assertEqual(rc, 1)  # the garbage input fails parsing, as before the guard
            self.assertIn("no FSPX payload found", err)
            self.assertIn("0 FSPX program(s) unpacked", out)
            self.assertTrue(os.path.isdir(os.path.join(t, "unpack", "fspx")))
            self.assertTrue(os.path.isfile(os.path.join(t, "unpack", "index.json")))


if __name__ == "__main__":
    unittest.main()
