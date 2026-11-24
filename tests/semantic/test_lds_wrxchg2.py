"""Value tests for RDNA2 ds_wrxchg2_rtn_b32 and ds_wrxchg2st64_rtn_b32, run through spirv_mini_interp on the CPU.

RDNA2 ISA, "Data Share Instructions": two exchanges addressed as ds_write2_b32 addresses its two dwords,
ADDR0 = vaddr + offset0 * 4 and ADDR1 = vaddr + offset1 * 4 (offset * 256 bytes for st64):
    vdst[0] = M[ADDR0]; M[ADDR0] = DATA
    vdst[1] = M[ADDR1]; M[ADDR1] = DATA2
Each case is a compute shader run as one invocation (lane 0): it seeds the LDS with ds_write_b32, runs the exchange,
reads both dwords back and stores (vdst[0], vdst[1], M[ADDR0], M[ADDR1]) to a buffer. Standard library only; no GPU,
no LLVM, no game files.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_values import run_listing  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402


def run_xchg2(op: str, addr: int, off0: int, off1: int, m0: int, m1: int, d0: int, d1: int, dst: str = "v[6:7]"):
    unit = 4 * (64 if "st64" in op else 1)
    a0, a1 = addr + off0 * unit, addr + off1 * unit
    body = [f"v_mov_b32_e32 v1, {addr:#x}", f"v_mov_b32_e32 v10, {a0:#x}", f"v_mov_b32_e32 v11, {a1:#x}",
            f"v_mov_b32_e32 v2, {m0:#x}", f"v_mov_b32_e32 v3, {m1:#x}",
            "ds_write_b32 v10, v2", "ds_write_b32 v11, v3", "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v4, {d0:#x}", f"v_mov_b32_e32 v5, {d1:#x}",
            "v_mov_b32_e32 v6, 0xdead", "v_mov_b32_e32 v7, 0xbeef",
            f"{op} {dst}, v1, v4, v5 offset0:{off0} offset1:{off1}", "s_waitcnt lgkmcnt(0)",
            "ds_read_b32 v8, v10", "ds_read_b32 v9, v11", "s_waitcnt lgkmcnt(0)"]
    for k, r in enumerate((6, 7, 8, 9)):
        body += [f"v_mov_b32_e32 v31, {4 * k}", f"buffer_store_dword v{r}, v31, s[0:3], 0 offen"]
    mem = run_listing(body)
    return tuple(mem.get(4 * k) for k in range(4))


class LdsWrxchg2(unittest.TestCase):
    def test_two_exchanges(self):
        for m0, m1, d0, d1 in ((1, 2, 3, 4), (0xFFFFFFFF, 0, 0x80000000, 0x7FFFFFFF), (0x12345678, 0x9ABCDEF0, 0, 0)):
            for off0, off1 in ((0, 1), (3, 7), (255, 0)):
                with self.subTest(m=(hex(m0), hex(m1)), d=(hex(d0), hex(d1)), offs=(off0, off1)):
                    got = run_xchg2("ds_wrxchg2_rtn_b32", 0x40, off0, off1, m0, m1, d0, d1)
                    self.assertEqual(got, (m0, m1, d0, d1))

    def test_st64_offsets_are_256_bytes(self):
        # offset1:1 is 256 bytes past vaddr, not 4: the seeded word there comes back.
        self.assertEqual(run_xchg2("ds_wrxchg2st64_rtn_b32", 0x10, 0, 1, 11, 22, 33, 44), (11, 22, 33, 44))
        self.assertEqual(run_xchg2("ds_wrxchg2st64_rtn_b32", 0x8, 2, 3, 5, 6, 7, 8), (5, 6, 7, 8))

    def test_same_dword_twice_returns_the_first_exchange(self):
        # Both offsets name one dword: the second exchange sees DATA from the first and leaves DATA2.
        self.assertEqual(run_xchg2("ds_wrxchg2_rtn_b32", 0x40, 2, 2, 9, 9, 100, 200), (9, 100, 200, 200))

    def test_vdst_aliasing_vaddr_and_data(self):
        # vdst v[4:5] overlaps the data registers: both exchanges read their data before either result is written.
        self.assertEqual(run_xchg2("ds_wrxchg2_rtn_b32", 0x40, 0, 1, 1, 2, 3, 4, dst="v[4:5]")[2:], (3, 4))

    def test_bad_destination_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_listing(["v_mov_b32_e32 v1, 0", "ds_wrxchg2_rtn_b32 v6, v1, v4, v5 offset1:1"])


if __name__ == "__main__":
    unittest.main()
