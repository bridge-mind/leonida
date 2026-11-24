"""Regression tests for clean OSError handling when writing carved streams in tools/rpf9_carve.py.

    python -I tests/test_rpf9_carve_errors.py
"""
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "rpf9_carve.py")


class Rpf9CarveWriteErrorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _write_archive(self, name, payload, base=16):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as f:
            f.write(b"9FPR" + struct.pack("<I", base) + b"\x00" * (base - 8) + payload)
        return path

    def test_unwritable_bin_stream_exits_without_traceback(self):
        zdata = zlib.compress(b"A" * 128)
        rpf = self._write_archive("bin.rpf", zdata, base=16)
        out_dir = os.path.join(self.tmp.name, "out_bin")
        os.makedirs(os.path.join(out_dir, f"{16:012x}.bin"))
        p = subprocess.run([sys.executable, "-I", TOOL, rpf, out_dir], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("cannot write", p.stderr)
        self.assertIn(f"{16:012x}.bin", p.stderr)
        self.assertNotIn("Traceback", p.stderr)

    def test_unwritable_bk2_stream_exits_without_traceback(self):
        blen = 64
        bk2 = bytearray(blen)
        bk2[0:4] = b"KB2i"
        struct.pack_into("<II", bk2, 4, blen - 8, 10)
        struct.pack_into("<II", bk2, 0x14, 1280, 720)
        rpf = self._write_archive("bk2.rpf", bytes(bk2), base=16)
        out_dir = os.path.join(self.tmp.name, "out_bk2")
        os.makedirs(os.path.join(out_dir, f"{16:012x}.bk2"))
        p = subprocess.run([sys.executable, "-I", TOOL, rpf, out_dir], capture_output=True, text=True)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("cannot write", p.stderr)
        self.assertIn(f"{16:012x}.bk2", p.stderr)
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
