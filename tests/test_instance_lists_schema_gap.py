"""tools/instance_lists.py: a list block whose struct the schema doesn't describe is counted, not a crash.

    python -I tests/test_instance_lists_schema_gap.py

Synthetic bytes only (the resource builder of tests/test_instance_lists.py).
"""
import os
import struct
import sys
import tempfile
import unittest
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

import instance_lists as il  # noqa: E402
from test_instance_lists import resource  # noqa: E402


def without_prop_struct():
    """The fixture with the prop-list struct renamed in the schema; its block still names PROP_LIST."""
    V, P = resource()
    V = bytearray(V)
    st = 0x100 + 32 * list(il.STRUCT_SIZE).index(il.PROP_LIST)
    assert struct.unpack_from("<I", V, st)[0] == il.PROP_LIST
    struct.pack_into("<I", V, st, 0x12345678)
    return bytes(V), P


class SchemaGap(unittest.TestCase):
    def test_measure_counts_the_block_unresolved(self):
        V, P = without_prop_struct()
        T = {}
        self.assertTrue(il.measure(V, P, T))
        self.assertEqual(T["list_blocks_unresolved"], 1)
        self.assertEqual(T["grass"]["lists"], 1)   # the grass list beside it is still read

    def test_scan_keeps_going(self):
        V, P = without_prop_struct()
        blob = bytearray(16)
        blob[:4] = b"RPF9"
        struct.pack_into("<I", blob, 4, 16)                       # data start
        blob += zlib.compress(V)
        blob += b"\0" * (-len(blob) % 16) + zlib.compress(P)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.rpf")
            with open(path, "wb") as f:
                f.write(blob)
            T = {}
            il.scan(path, T)
        self.assertEqual(T.get("map_resources_read"), 1)
        self.assertEqual(T.get("list_blocks_unresolved"), 1)


if __name__ == "__main__":
    unittest.main()
