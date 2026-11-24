"""group_edges on a meshlet group with more than 65,536 vertex references (synthetic bytes only).

    python -I tests/test_vertex_types_wide_group.py
"""
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import vertex_types as vt  # noqa: E402


def wide_group(meshlets=513, n=128):
    """One triangle (0, 1, 2) per meshlet; every reference of meshlet m points at vertex m."""
    recs, words, refs = bytearray(), bytearray(), bytearray()
    for m in range(meshlets):
        vbase, tbase = m * n, m
        # w1 holds the low 16 bits of each running sum, which is what parse_group checks.
        recs += struct.pack("<4I", n - 1, (vbase & 0xFFFF) | ((tbase & 0xFFFF) << 16), 0x7FF7FF, 0x7FF)
        words += struct.pack("<I", 0 | (1 << 7) | (2 << 14) | (m << 21))
        refs += struct.pack(f"<{n}H", *([m] * n))
    return struct.pack("<3I", 0x80000000 | meshlets, meshlets, meshlets * n) + bytes(recs + words + refs)


class WideGroupTests(unittest.TestCase):
    def test_meshlet_past_64k_references_reads_its_own_references(self):
        data = wide_group()
        group, _ = vt.parse_group(data, 0)  # the shared validator accepts the wrapped bases
        tris = vt.group_edges(data, 0, group)
        self.assertEqual(tris, [(m, m, m) for m in range(513)])


if __name__ == "__main__":
    unittest.main()
