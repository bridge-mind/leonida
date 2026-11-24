"""--limit ending inside a stream: the index tools leave that stream out instead of counting its whole stored size.

story_index (#712) and navmesh_index (#766) already inflate only inside the scan window; anim_index, rbf0_index,
restofworld_index and bvh_index inflated up to the end of the archive, so a stream that started in the window but
ended after it was counted as readable and readable + zero + unread exceeded the span. rpf9_carve.py has its own
test in test_rpf9_carve.py.
"""
import hashlib
import os
import struct
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import anim_index  # noqa: E402
import bvh_index  # noqa: E402
import rbf0_index  # noqa: E402
import restofworld_index  # noqa: E402

GAP = 0x100


def noisy(n, seed):
    """Bytes zlib can't shrink, so a stream's stored size is about n."""
    out = bytearray()
    while len(out) < n:
        out += hashlib.sha256(bytes([seed]) + len(out).to_bytes(4, "little")).digest()
    return bytes(out[:n])


def summary_of(result):
    """The tools return (rows, summary), (rows, records, summary) or (rows, [summary])."""
    last = result[-1]
    return last[0] if isinstance(last, list) else last


def first_of(summary, *keys):
    for key in keys:
        if key in summary:
            return summary[key]
    raise KeyError(keys)


class LimitWindowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.first = zlib.compress(noisy(4000, 1))
        data = bytearray(0x40 + GAP) + self.first + zlib.compress(noisy(4000, 2))
        data[0:4] = b"9FPR"
        struct.pack_into("<I", data, 4, 0x40)
        self.path = os.path.join(self.tmp.name, "window.rpf")
        with open(self.path, "wb") as f:
            f.write(data)
        self.limit = GAP + len(self.first) // 2   # ends inside the first stream

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_stream_crossing_the_limit_is_left_out(self):
        for tool in (anim_index, rbf0_index, restofworld_index, bvh_index):
            with self.subTest(tool=tool.__name__):
                summary = summary_of(tool.scan_rpf(self.path, limit=self.limit))
                stored = first_of(summary, "compressed_bytes", "stored")
                unread = first_of(summary, "unread_bytes", "unread")
                self.assertEqual(summary["span"], self.limit)
                self.assertEqual(stored, 0)
                self.assertEqual(stored + summary["zero_padding"] + unread, summary["span"])
                self.assertLessEqual(summary["readable_fraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
