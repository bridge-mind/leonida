"""Semantic interpreter tests for VALU ops that had no translation test.

RDNA2 ISA definitions:
  v_cmp_<c>_u64 / _i64   per lane, S0 <c> S1 over 64-bit values, unsigned or signed (the signed order differs only
                         when the high words differ in bit 63);
  v_cvt_f32_ubyte<k>     D = float(S0[8k+7:8k]);
  v_lshlrev_b16 / v_lshrrev_b16 / v_ashrrev_i16   D[15:0] = S1.16 shifted by S0[3:0] (the amount wraps at 16, so
                         16 shifts by 0); only the low half of D is checked here;
  v_bcnt_u32_b32         D = popcount(S0) + S1 (32-bit wrap).
Each case runs a one-lane pixel shader through tools/rdna2spirv.py and tests/semantic/spirv_mini_interp.py and compares
with plain-Python integer references written from the lines above. Compare results leave through v_cndmask_b32_e64.
"""
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

M32 = 0xFFFFFFFF
M64 = (1 << 64) - 1


def s64(x):
    return x - (1 << 64) if x >> 63 else x


def s16(x):
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def run(lines):
    spv = support.translate_lines("ps_valu_untested", lines + ["exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"])
    return si.run_fragment(spv)["%out_mrt0"]


def regs(**kw):
    out = []
    for name, val in kw.items():
        out += [f"s_mov_b32 s{name[1:]}, {val & M32:#x}", f"v_mov_b32 v{name[1:]}, s{name[1:]}"]
    return out


class Cmp64(unittest.TestCase):
    VALUES = [0, 1, M64, 1 << 63, (1 << 63) - 1, 1 << 32, 0xFFFFFFFF, 0x80000000, 0xFFFFFFFF00000000,
              0x00000001FFFFFFFF, 0x7FFFFFFF80000000, 0x123456789ABCDEF0]
    CONDS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
             "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}

    def test_all(self):
        for cond, fn in self.CONDS.items():
            for ty in ("u64", "i64"):
                for x in self.VALUES:
                    for y in self.VALUES:
                        a, b = (x, y) if ty == "u64" else (s64(x), s64(y))
                        lines = (regs(v2=x & M32, v3=x >> 32, v4=y & M32, v5=y >> 32)
                                 + ["v_mov_b32 v10, 0", "v_mov_b32 v11, 0", "v_mov_b32 v12, 0", "v_mov_b32 v13, 0",
                                    "v_mov_b32 v14, 1", f"v_cmp_{cond}_{ty}_e64 s20, v[2:3], v[4:5]",
                                    "v_cndmask_b32_e64 v10, v10, v14, s20"])
                        with self.subTest(cond=cond, ty=ty, x=hex(x), y=hex(y)):
                            self.assertEqual(run(lines)[0], int(fn(a, b)))


class Misc(unittest.TestCase):
    def one(self, op, srcs, old=0, mods=""):
        regs_ = []
        for i, v in enumerate(srcs):
            regs_ += [f"s_mov_b32 s{2 + i}, {v & M32:#x}", f"v_mov_b32 v{2 + i}, s{2 + i}"]
        lines = regs_ + [f"s_mov_b32 s9, {old:#x}", "v_mov_b32 v10, s9", "v_mov_b32 v11, 0", "v_mov_b32 v12, 0",
                         "v_mov_b32 v13, 0", f"{op} v10, " + ", ".join(f"v{2 + i}" for i in range(len(srcs))) + mods]
        return run(lines)[0]

    def test_cvt_f32_ubyte(self):
        for x in (0, M32, 0x80402010, 0x01FF7F80, 0x12345678):
            for k in range(4):
                e = struct.unpack("<I", struct.pack("<f", float((x >> (8 * k)) & 0xFF)))[0]
                with self.subTest(k=k, x=hex(x)):
                    self.assertEqual(self.one(f"v_cvt_f32_ubyte{k}", [x]), e)

    def test_shifts_b16(self):
        for x in (0, 1, 0x8000, 0xFFFF, 0x7FFF, 0xA5C3):
            for s in (0, 1, 15, 16, 17, 31, 0xFFF0, 0x12340010):
                old = 0xABCD0000
                x16 = x & 0xFFFF
                cases = {"v_lshlrev_b16": (x16 << (s & 15)) & 0xFFFF, "v_lshrrev_b16": x16 >> (s & 15),
                         "v_ashrrev_i16": (s16(x16) >> (s & 15)) & 0xFFFF}
                for op, lo in cases.items():
                    with self.subTest(op=op, x=hex(x), s=hex(s)):
                        self.assertEqual(self.one(op, [s, x], old=old) & 0xFFFF, lo)

    def test_bcnt(self):
        for x in (0, 1, M32, 0x80000000, 0x12345678):
            for c in (0, 5, M32):
                with self.subTest(x=hex(x), c=hex(c)):
                    self.assertEqual(self.one("v_bcnt_u32_b32", [x, c]), (bin(x).count("1") + c) & M32)


if __name__ == "__main__":
    unittest.main()
