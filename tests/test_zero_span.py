"""The archive scanners classify the bytes between streams without copying the whole gap.

`gap = m[last:s]` on an mmap copies the gap, so one large gap costs as much memory as its size: story000.rpf
(about 20 GB) could not be indexed on a 16 GB machine. tools/zero_span.py compares the gap with zeros a window
at a time. This checks its counts against the old expression, then runs every scanner that used it on an archive
with a 48 MiB gap and measures the Python heap peak with tracemalloc.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
import zero_span  # noqa: E402

GAP = 48 << 20
PEAK_LIMIT = 24 << 20   # well under the gap: a whole-gap copy alone would be 48 MiB

# Each scanner and its arguments; "{arc}" is the archive and "{out}" a fresh output directory.
SCANNERS = {
    "rpf9_carve": ["{arc}", "{out}/carve", "--index-only"],
    "rpf9_names": ["{arc}", "{out}/names.json", "--quiet"],
    "bvh_index": ["{arc}", "--out", "{out}/bvh.json"],
    "navmesh_index": ["{arc}", "--out", "{out}/navmesh.json"],
    "navmesh_decode": ["{arc}", "--out", "{out}/decode.json", "--quiet"],
    "rbf0_index": ["{arc}", "{out}/rbf0.json"],
    "anim_index": ["{arc}", "{out}/anim.json", "--quiet"],
    "restofworld_index": ["{arc}", "{out}/row.json", "--quiet"],
    "story_index": ["{arc}", "{out}/story.json", "--quiet"],
}

# Runs a tool as __main__ under tracemalloc and prints its exit code and heap peak as JSON on the last line.
RUNNER = r"""
import json, runpy, sys, tracemalloc
tool, args = sys.argv[1], sys.argv[2:]
sys.argv = [tool] + args
tracemalloc.start()
code = 0
try:
    runpy.run_path(tool, run_name="__main__")
except SystemExit as e:
    code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
peak = tracemalloc.get_traced_memory()[1]
sys.stdout.flush()
print("\n" + json.dumps({"code": code, "peak": peak}))
"""


def old_split(buf, start, end):
    gap = buf[start:end]
    return (len(gap), 0) if gap.count(0) == len(gap) else (0, len(gap))


class GapSplitTest(unittest.TestCase):
    def test_counts_match_the_slice_expression(self):
        w = zero_span.WINDOW
        buf = bytearray(3 * w + 100)
        cases = [(0, 0), (5, 5), (10, 3), (0, len(buf)), (w - 1, w + 1), (0, len(buf) + 50),
                 (len(buf) + 10, len(buf) + 20), (w, 2 * w)]
        for nz in (None, 0, w - 1, w, 2 * w + 7, len(buf) - 1):
            data = bytearray(buf)
            if nz is not None:
                data[nz] = 1
            for start, end in cases:
                with self.subTest(nonzero=nz, start=start, end=end):
                    self.assertEqual(zero_span.gap_split(bytes(data), start, end), old_split(bytes(data), start, end))

    def test_all_zero_stops_at_the_first_non_zero_window(self):
        class Probe(bytes):
            def __getitem__(self, key):
                self.reads.append(key)
                return bytes.__getitem__(self, key)
        p = Probe(b"\x01" + bytes(3 * zero_span.WINDOW))
        p.reads = []
        self.assertFalse(zero_span.all_zero(p, 0, len(p)))
        self.assertEqual(len(p.reads), 1)


def make_archive(path):
    """RPF9 data area: one stream, a GAP of zero bytes, one more stream, and a short zero tail."""
    a = zlib.compress(b"A" * 1000)
    b = zlib.compress(b"B" * 2000)
    with open(path, "wb") as f:
        f.write(b"9FPR" + struct.pack("<I", 0x40) + bytes(0x38))
        f.write(a)
        f.seek(GAP, os.SEEK_CUR)
        f.write(b)
        f.write(bytes(4096))


class ScannerPeakMemoryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.arc = os.path.join(cls.tmp.name, "gap.rpf")
        make_archive(cls.arc)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_a_large_zero_gap_is_not_copied(self):
        for name, args in SCANNERS.items():
            with self.subTest(tool=name), tempfile.TemporaryDirectory() as out:
                argv = [x.format(arc=self.arc, out=out) for x in args]
                p = subprocess.run([sys.executable, "-I", "-c", RUNNER, os.path.join(TOOLS, f"{name}.py")] + argv,
                                   capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                res = json.loads(p.stdout.strip().splitlines()[-1])
                self.assertEqual(res["code"], 0, p.stdout[-2000:] + p.stderr[-2000:])
                self.assertLess(res["peak"], PEAK_LIMIT, f"{name}: heap peak {res['peak'] >> 20} MiB")


if __name__ == "__main__":
    unittest.main()
