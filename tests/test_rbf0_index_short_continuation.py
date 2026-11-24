"""rbf0_index joins a continuation only after a full 64 KiB chunk, as rbf0_tree does.

    python -I tests/test_rbf0_index_short_continuation.py

Every byte here is built by the test.
"""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import rbf0_index  # noqa: E402
import rbf0_tree  # noqa: E402


def big_unclosed():
    """A wide root holding filler strings, longer than 64 KiB, with no close tag."""
    b = bytearray(b"RBF0" + rbf0_index.PROLOGUE)
    b += bytes([0, 0x00]) + struct.pack("<H", 4) + b"Root" + struct.pack("<HHH", 0, 0, 0)
    filler = b"x" * 4000
    first = True
    while len(b) < 70000:
        b += bytes([1, 0x60]) + ((struct.pack("<H", 6) + b"filler") if first else b"")
        b += struct.pack("<H", len(filler)) + filler
        first = False
    return bytes(b)


class ShortContinuationTest(unittest.TestCase):
    def test_no_join_after_a_short_continuation(self):
        blob = big_unclosed()
        parts = [blob[:65536], blob[65536:65600], blob[65600:] + b"\xff\xff"]
        rows, off = [], 0
        for p in parts:
            rows.append({"offset": off, "stored": len(p), "size": len(p), "head": p[:8]})
            off += len(p)
        data = {r["offset"]: p for r, p in zip(rows, parts)}
        _, chunks, parsed = rbf0_index.join_resource(rows, 0, lambda r: data[r["offset"]])
        # The second stream is shorter than 64 KiB, so the resource ends there; the third is another stream.
        self.assertEqual(chunks, 2)
        self.assertEqual(parsed["status"], "truncated")

    def test_agrees_with_rbf0_tree(self):
        import tempfile
        blob = big_unclosed()
        parts = [blob[:65536], blob[65536:65600], blob[65600:] + b"\xff\xff"]
        with tempfile.TemporaryDirectory() as d:
            off = 0
            for p in parts:
                with open(os.path.join(d, f"{off:08x}.bin"), "wb") as f:
                    f.write(p)
                off += len(p)
            _, records, _ = rbf0_index.scan_dir(d)
            docs, _ = rbf0_tree.decode_source(d)
        self.assertEqual((records[0]["chunks"], records[0]["status"]), (docs[0]["chunks"], docs[0]["status"]))



if __name__ == "__main__":
    unittest.main()
