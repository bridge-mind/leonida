"""saves.py import: an option given without its value is a one-line error, not a StopIteration traceback.

    python -I tests/test_saves_import_args.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "saves.py")


class ImportMissingValueTest(unittest.TestCase):
    def test_trailing_option_without_value(self):
        with tempfile.TemporaryDirectory() as d:
            for flag in ("--out", "--map"):
                with self.subTest(flag=flag):
                    r = subprocess.run([sys.executable, "-I", TOOL, "import", d, flag],
                                       capture_output=True, text=True)
                    self.assertEqual(r.returncode, 2, r.stderr)
                    self.assertNotIn("Traceback", r.stderr)
                    self.assertIn(flag, r.stderr)


if __name__ == "__main__":
    unittest.main()
