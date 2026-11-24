"""SHADER-562: a mesh shader's LDS array covers its constant ds offsets with the access's own element size.

read2/write2 offset0/offset1 count elements: 4 bytes for _b32, 8 for _b64, x64 for st64 (`ds_const_extent`, as the
compute path does since SHADER-309). The mesh loop counted every element as 8 bytes, so a 32 KiB mesh shader whose
st64 b32 offsets stay inside its allocation still doubled its array to 64 KiB.

The programs and the PSV0 part are written here; no game data.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)  # python -I leaves the script's folder off sys.path

import support  # noqa: E402  (support puts tools/ on sys.path, honouring LEONIDA_TOOLS_DIR)

rdna2spirv = support.rdna2spirv


def mesh_dxbc(group_shared, threads=128):
    """DXBC with one PSV0 part: the fields mesh_psv() reads (stage 13, 64/64 triangles, NumThreads, shared bytes)."""
    r = bytearray(48)
    struct.pack_into("<I", r, 0, group_shared)
    struct.pack_into("<HH", r, 12, 64, 64)
    r[24], r[27] = 13, 2
    struct.pack_into("<III", r, 36, threads, 1, 1)
    part = b"PSV0" + struct.pack("<I", 4 + len(r)) + struct.pack("<I", len(r)) + bytes(r)
    head = 32 + 4
    return b"DXBC" + bytes(16) + struct.pack("<IIII", 1, head + len(part), 1, head) + part


def lds_dwords(op, data, off1, group_shared=32768):
    prog = ["s_mov_b64 exec, -1",
            "v_lshlrev_b32_e32 v1, 2, v0",
            f"{op} v1, {data} offset1:{off1}",
            "s_endpgm"]
    support.translate_lines("ms_lds_need", prog, dxbc=mesh_dxbc(group_shared))
    return rdna2spirv.translate.last_gen.lds_dwords


class MeshLdsNeedTests(unittest.TestCase):
    def test_st64_b32_offsets_inside_the_allocation_keep_it(self):
        # offset1:112 st64 b32 = 112 * 64 * 4 = 28,672 B, inside 32 KiB: the array is the allocation, 8,192 dwords
        # (counting 8 bytes per element made it 14,340 dwords, so a 64 KiB array)
        self.assertEqual(lds_dwords("ds_write2st64_b32", "v0, v0", 112), 8192)

    def test_b32_offsets_count_dwords(self):
        # offset1:255 b32 ends at byte 1,023 + 1: 256 dwords, so a small allocation stays small (16 dwords minimum)
        self.assertEqual(lds_dwords("ds_write2_b32", "v0, v0", 255, group_shared=0), 256)

    def test_b64_offsets_count_qwords(self):
        # offset1:112 st64 b64 = 112 * 64 * 8 = 57,344 B + 8: past 32 KiB, so the array grows to 64 KiB
        self.assertEqual(lds_dwords("ds_write2st64_b64", "v[2:3], v[2:3]", 112), 16384)
        # offset1:255 b64 ends at byte 2,047 + 1: 512 dwords
        self.assertEqual(lds_dwords("ds_write2_b64", "v[2:3], v[2:3]", 255, group_shared=0), 512)


if __name__ == "__main__":
    unittest.main()
