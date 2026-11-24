"""anim_index.py --rigs: an unwritable --rigs path is a one-line error, as for the index output, not a traceback.

    python -I tests/test_anim_index_rigs_write.py
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "anim_index.py")
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_anim_index import rbf0_blob  # noqa: E402


class RigsWriteTest(unittest.TestCase):
    def test_unwritable_rigs_path(self):
        with tempfile.TemporaryDirectory() as d:
            carve = os.path.join(d, "carve")
            os.mkdir(carve)
            with open(os.path.join(carve, "00001000.bin"), "wb") as f:
                f.write(rbf0_blob())
            rigs = os.path.join(d, "missing", "rigs.json")
            r = subprocess.run([sys.executable, "-I", TOOL, carve, os.path.join(d, "index.json"),
                                "--rigs", rigs, "--quiet"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("Traceback", r.stderr)
        self.assertIn("rigs.json", r.stderr)


if __name__ == "__main__":
    unittest.main()
