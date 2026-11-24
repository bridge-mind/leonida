"""Value tests for RDNA2 ds_and_b64, ds_or_b64, ds_xor_b64 and their _rtn_ forms, run through spirv_mini_interp.

RDNA2 ISA, "Data Share Instructions": M = M op DATA on the 64-bit element at ADDR = vaddr + offset (the dwords at ADDR
and ADDR + 4, low first); the _rtn_ forms return the old 64 bits in vdst[0:1]. The translator does one 32-bit atomic
per dword, which leaves the same memory because a bitwise op works on each bit alone. Each case is a compute shader
run as one invocation: it seeds both dwords with ds_write_b64, runs the op, reads the element back with ds_read_b64
and stores (vdst lo, vdst hi, M lo, M hi) to a buffer. Standard library only; no GPU, no LLVM, no game files.
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
OPS = {"and": lambda m, d: m & d, "or": lambda m, d: m | d, "xor": lambda m, d: m ^ d}
VALUES = [0, (1 << 64) - 1, 0x00000000_FFFFFFFF, 0xFFFFFFFF_00000000, 0x12345678_9ABCDEF0, 0x80000000_00000001]


def run_b64(line: str, init: int, data: int, addr: int = 0x40, offset: int = 0):
    body = [f"v_mov_b32_e32 v1, {addr:#x}", f"v_mov_b32_e32 v2, {init & M32:#x}", f"v_mov_b32_e32 v3, {init >> 32:#x}",
            f"ds_write_b64 v1, v[2:3] offset:{offset}", "s_waitcnt lgkmcnt(0)",
            f"v_mov_b32_e32 v4, {data & M32:#x}", f"v_mov_b32_e32 v5, {data >> 32:#x}",
            "v_mov_b32_e32 v6, 0xdead", "v_mov_b32_e32 v7, 0xbeef",
            line + (f" offset:{offset}" if offset else ""), "s_waitcnt lgkmcnt(0)",
            f"ds_read_b64 v[8:9], v1 offset:{offset}", "s_waitcnt lgkmcnt(0)"]
    for k, r in enumerate((6, 7, 8, 9)):
        body += [f"v_mov_b32_e32 v31, {4 * k}", f"buffer_store_dword v{r}, v31, s[0:3], 0 offen"]
    mem = run_listing(body)
    return tuple(mem.get(4 * k) for k in range(4))


def split(v: int):
    return v & M32, v >> 32


class LdsBitwiseB64(unittest.TestCase):
    def test_returning_forms(self):
        for kind, fn in OPS.items():
            for m in VALUES:
                for d in VALUES[::2]:
                    with self.subTest(op=kind, m=hex(m), d=hex(d)):
                        got = run_b64(f"ds_{kind}_rtn_b64 v[6:7], v1, v[4:5]", m, d)
                        self.assertEqual(got, split(m) + split(fn(m, d)))

    def test_non_returning_forms_leave_vdst_alone(self):
        for kind, fn in OPS.items():
            for m, d in ((0x12345678_9ABCDEF0, 0x0F0F0F0F_F0F0F0F0), (0, (1 << 64) - 1)):
                with self.subTest(op=kind, m=hex(m), d=hex(d)):
                    self.assertEqual(run_b64(f"ds_{kind}_b64 v1, v[4:5]", m, d), (0xDEAD, 0xBEEF) + split(fn(m, d)))

    def test_offset(self):
        m, d = 0x11111111_22222222, 0x0000FFFF_FFFF0000
        self.assertEqual(run_b64("ds_xor_rtn_b64 v[6:7], v1, v[4:5]", m, d, addr=0x20, offset=0x18),
                         split(m) + split(m ^ d))

    def test_vdst_aliasing_data(self):
        # vdst v[4:5] is also the data pair: the data is read before the old value is written back.
        m, d = 0xF0F0F0F0_0F0F0F0F, 0xFF00FF00_00FF00FF
        got = run_b64("ds_or_rtn_b64 v[4:5], v1, v[4:5]", m, d)
        self.assertEqual(got[2:], split(m | d))

    def test_seeded(self):
        rng = random.Random(0xB64)
        for _ in range(15):
            kind, m, d = rng.choice(list(OPS)), rng.getrandbits(64), rng.getrandbits(64)
            with self.subTest(op=kind, m=hex(m), d=hex(d)):
                self.assertEqual(run_b64(f"ds_{kind}_rtn_b64 v[6:7], v1, v[4:5]", m, d),
                                 split(m) + split(OPS[kind](m, d)))

    def test_bad_operands_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            run_listing(["v_mov_b32_e32 v1, 0", "ds_and_b64 v1, v4"])


if __name__ == "__main__":
    unittest.main()
