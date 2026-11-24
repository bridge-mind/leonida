"""Value tests for RDNA2 ds_wrxchg_rtn_b64, ds_wrxchg2_rtn_b64 and ds_wrxchg2st64_rtn_b64, run through
spirv_mini_interp on the CPU.

RDNA2 ISA, "Data Share Instructions": a 64-bit element is two dwords, low first.
    ds_wrxchg_rtn_b64       vdst[0:1] = M[A]; M[A] = DATA                       A = vaddr + offset
    ds_wrxchg2_rtn_b64      vdst[0:1] = M[A0]; M[A0] = DATA; vdst[2:3] = M[A1]; M[A1] = DATA2
                            A0 / A1 = vaddr + offset0 / offset1 * 8 (* 512 for st64), as ds_write2_b64 addresses them
The translator does one 32-bit exchange per dword. Each case is a compute shader run as one invocation: it seeds the
LDS with ds_write_b64, runs the exchange, reads the element(s) back with ds_read_b64 and stores the words to a buffer.
Standard library only; no GPU, no LLVM, no game files.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_lds_atomics_values import run_listing  # noqa: E402  (also puts tools/ on sys.path)
import rdna2spirv  # noqa: E402

M32 = 0xFFFFFFFF


def split(v: int):
    return v & M32, v >> 32


def stores(regs):
    body = []
    for k, r in enumerate(regs):
        body += [f"v_mov_b32_e32 v31, {4 * k}", f"buffer_store_dword v{r}, v31, s[0:3], 0 offen"]
    return body


def seed(reg_lo: int, value: int, addr_reg: int, addr: int):
    lo, hi = split(value)
    return [f"v_mov_b32_e32 v{addr_reg}, {addr:#x}", f"v_mov_b32_e32 v{reg_lo}, {lo:#x}",
            f"v_mov_b32_e32 v{reg_lo + 1}, {hi:#x}", f"ds_write_b64 v{addr_reg}, v[{reg_lo}:{reg_lo + 1}]",
            "s_waitcnt lgkmcnt(0)"]


def run_one(m: int, d: int, addr: int = 0x40, offset: int = 0, dst: str = "v[6:7]"):
    """ds_wrxchg_rtn_b64 on M = m with DATA = d (in v[4:5]); returns (old lo, old hi, new lo, new hi)."""
    body = seed(2, m, 10, addr + offset) + [f"v_mov_b32_e32 v1, {addr:#x}"]
    body += [f"v_mov_b32_e32 v4, {split(d)[0]:#x}", f"v_mov_b32_e32 v5, {split(d)[1]:#x}",
             f"ds_wrxchg_rtn_b64 {dst}, v1, v[4:5]" + (f" offset:{offset}" if offset else ""), "s_waitcnt lgkmcnt(0)",
             "ds_read_b64 v[8:9], v10", "s_waitcnt lgkmcnt(0)"]
    reg = int(dst[2:dst.index(":")])
    mem = run_listing(body + stores((reg, reg + 1, 8, 9)))
    return tuple(mem.get(4 * k) for k in range(4))


def run_two(op: str, m0: int, m1: int, d0: int, d1: int, off0: int, off1: int, addr: int = 0x40):
    unit = 8 * (64 if "st64" in op else 1)
    body = seed(2, m0, 10, addr + off0 * unit) + seed(2, m1, 11, addr + off1 * unit)
    body += [f"v_mov_b32_e32 v1, {addr:#x}"]
    for r, v in ((12, d0), (14, d1)):
        body += [f"v_mov_b32_e32 v{r}, {split(v)[0]:#x}", f"v_mov_b32_e32 v{r + 1}, {split(v)[1]:#x}"]
    body += [f"{op} v[16:19], v1, v[12:13], v[14:15] offset0:{off0} offset1:{off1}", "s_waitcnt lgkmcnt(0)",
             "ds_read_b64 v[20:21], v10", "ds_read_b64 v[22:23], v11", "s_waitcnt lgkmcnt(0)"]
    mem = run_listing(body + stores((16, 17, 18, 19, 20, 21, 22, 23)))
    return tuple(mem.get(4 * k) for k in range(8))


VALUES = [0, (1 << 64) - 1, 0x12345678_9ABCDEF0, 0x80000000_00000001, 0x00000000_FFFFFFFF]


class LdsWrxchgB64(unittest.TestCase):
    def test_single_exchange(self):
        for m in VALUES:
            for d in VALUES[::2]:
                with self.subTest(m=hex(m), d=hex(d)):
                    self.assertEqual(run_one(m, d), split(m) + split(d))

    def test_single_offset_and_aliasing(self):
        m, d = 0x11111111_22222222, 0x33333333_44444444
        self.assertEqual(run_one(m, d, addr=0x20, offset=0x18), split(m) + split(d))
        # vdst v[4:5] is the data pair: both dwords are read before the old value is written back.
        self.assertEqual(run_one(m, d, dst="v[4:5]"), split(m) + split(d))

    def test_two_exchanges(self):
        rng = random.Random(0x2B64)
        for op in ("ds_wrxchg2_rtn_b64", "ds_wrxchg2st64_rtn_b64"):
            for off0, off1 in ((0, 1), (3, 5) if op == "ds_wrxchg2_rtn_b64" else (0, 2)):
                m0, m1, d0, d1 = (rng.getrandbits(64) for _ in range(4))
                with self.subTest(op=op, offs=(off0, off1)):
                    self.assertEqual(run_two(op, m0, m1, d0, d1, off0, off1),
                                     split(m0) + split(m1) + split(d0) + split(d1))

    def test_st64_steps_are_512_bytes(self):
        # offset1:1 is 512 bytes past vaddr for the b64 st64 form; the element seeded there comes back.
        got = run_two("ds_wrxchg2st64_rtn_b64", 1, 2, 3, 4, 0, 1, addr=0x10)
        self.assertEqual(got, (1, 0, 2, 0, 3, 0, 4, 0))

    def test_bad_operands_rejected(self):
        for line in ("ds_wrxchg_rtn_b64 v6, v1, v[4:5]", "ds_wrxchg_rtn_b64 v[6:7], v1, v4",
                     "ds_wrxchg2_rtn_b64 v[16:17], v1, v[12:13], v[14:15] offset1:1"):
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    run_listing(["v_mov_b32_e32 v1, 0", line])


if __name__ == "__main__":
    unittest.main()
