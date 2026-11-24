"""world_index.scan_rpf joining a resource stored as three zlib streams (synthetic bytes only).

    python -I tests/test_world_index_three_streams.py
"""
import os
import sys
import tempfile
import unittest
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_world_index import WI, make_resource, make_rpf_streams  # noqa: E402


def stream_ends(parts, start=0x40):
    """(start, end) of each compressed part as make_rpf_streams lays them out."""
    out, pos = [], start
    for part in parts:
        pos = (pos + 15) & ~15
        out.append((pos, pos + len(zlib.compress(part))))
        pos = out[-1][1]
    return out


def padding(spans):
    """Alignment bytes between the first two streams and between the second and third."""
    return sum(spans[i + 1][0] - spans[i][1] for i in range(2))


def padded_split(blob):
    """Three parts whose two alignment gaps add up to 16 bytes or more, and their spans.

    How much padding a split leaves depends on the compressor's output lengths: zlib-ng (Python 3.14 on
    Windows) compresses these parts to other lengths than zlib, so the split points are searched for
    instead of fixed. The search starts at 0x150/0x420, which give 2 + 15 bytes with zlib, and the second
    point stays at or below 0x500 so the CMapData root (at 0x500) is in the third part.
    """
    for first in range(0x150, 0x200):
        for second in range(0x420, 0x501):
            parts = [blob[:first], blob[first:second], blob[second:]]
            spans = stream_ends(parts)
            if padding(spans) >= 16:
                return parts, spans
    return None, None


class ThreeStreamJoinTest(unittest.TestCase):
    def test_third_stream_follows_the_second_streams_real_end(self):
        blob = make_resource()
        # With 16 or more bytes of padding after the first two streams, "first stream start + both streams'
        # lengths", rounded up to 16, lands inside the second stream instead of at the third.
        parts, spans = padded_split(blob)
        self.assertIsNotNone(parts, "no split with 16 bytes of padding under this zlib")
        archive = make_rpf_streams(parts)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "three.rpf")
            with open(path, "wb") as f:
                f.write(archive)
            rows, stats = WI.scan_rpf(path)
        self.assertEqual(stats["streams"], 3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["size"], len(blob))
        self.assertEqual(rows[0]["map_name"], 0x11223344)  # the CMapData block lives in the third stream
        self.assertEqual(stats["scanned_bytes"], sum(e - s for s, e in spans))


if __name__ == "__main__":
    unittest.main()
