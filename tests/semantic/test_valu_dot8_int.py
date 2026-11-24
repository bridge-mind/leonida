"""v_dot8_u32_u4 and v_dot8_i32_i4 (RDNA2 VOP3P): D = S2 + sum of the eight nibble products of S0 and S1 (unsigned
nibbles for u4, signed for i4), modulo 2^32; with `clamp` the result saturates to [0, 2^32-1] (u4) or
[-2^31, 2^31-1] (i4).

Each case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
words are written by hand and also recomputed by a plain-Python reference. Standard library only: no GPU, no LLVM,
no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402

MASK32 = 0xFFFFFFFF
INT32_MIN, INT32_MAX = -0x80000000, 0x7FFFFFFF


def pk(*n):
    """Pack eight nibbles, nibble 0 first, into a dword."""
    return sum((x & 0xF) << (4 * i) for i, x in enumerate(n))


def ref(op, s0, s1, s2, clamp=False):
    signed = "i4" in op
    def nib(v, i):
        x = (v >> (4 * i)) & 0xF
        return x - 16 if signed and x & 8 else x
    total = sum(nib(s0, i) * nib(s1, i) for i in range(8))
    if signed:
        total += s2 - (1 << 32) if s2 & 0x80000000 else s2
        if clamp:
            total = max(INT32_MIN, min(INT32_MAX, total))
    else:
        total += s2
        if clamp:
            total = min(MASK32, total)
    return total & MASK32


def run(op, s0, s1, s2, clamp=False, form="vgpr"):
    if form == "vgpr":
        setup = [f"v_mov_b32 v1, {s0:#x}", f"v_mov_b32 v2, {s1:#x}", f"v_mov_b32 v3, {s2:#x}"]
        srcs = "v1, v2, v3"
    else:
        setup = [f"s_mov_b32 s1, {s0:#x}", f"s_mov_b32 s2, {s1:#x}", f"s_mov_b32 s3, {s2:#x}"]
        srcs = "s1, s2, s3"
    lines = setup + [f"{op} v0, {srcs}{' clamp' if clamp else ''}", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_dot8", lines))[0]


U, I = "v_dot8_u32_u4", "v_dot8_i32_i4"
ALL_M8 = pk(*[-8] * 8)      # eight nibbles of -8 (0x8 each)
ALL_7 = pk(*[7] * 8)

# (name, op, S0, S1, S2, clamp, expected word), written by hand.
CASES = [
    ("u4 basic: 0*1+1*2+...+7*8 = 168", U, pk(0, 1, 2, 3, 4, 5, 6, 7), pk(1, 2, 3, 4, 5, 6, 7, 8), 0, False, 168),
    ("u4 basic with accumulator", U, pk(0, 1, 2, 3, 4, 5, 6, 7), pk(1, 2, 3, 4, 5, 6, 7, 8), 500, False, 668),
    ("u4 zero operand keeps S2", U, 0, 0x12345678, 0x1234ABCD, False, 0x1234ABCD),
    ("u4 zero operand keeps S2 (swapped)", U, 0x12345678, 0, 0xCAFEBABE, False, 0xCAFEBABE),
    ("u4 maximum: 8*15*15", U, MASK32, MASK32, 0, False, 1800),
    ("u4 maximum plus 200", U, MASK32, MASK32, 200, False, 2000),
    ("u4 wraps to 0", U, MASK32, MASK32, 0xFFFFF8F8, False, 0),
    ("u4 wraps to 1", U, MASK32, MASK32, 0xFFFFF8F9, False, 1),
    ("u4 without clamp wraps", U, MASK32, MASK32, 0xFFFFFFF0, False, 0x000006F8),
    ("u4 clamp saturates", U, MASK32, MASK32, 0xFFFFFFF0, True, 0xFFFFFFFF),
    ("i4 basic: 1+2+3+4-1-2-3-4 = 0", I, pk(1, 2, 3, 4, -1, -2, -3, -4), pk(*[1] * 8), 0, False, 0),
    ("i4 basic with accumulator", I, pk(1, 2, 3, 4, -1, -2, -3, -4), pk(*[1] * 8), 77, False, 77),
    ("i4 negative operands: 2 * -20 = -40", I, pk(-1, -2, -3, -4, -1, -2, -3, -4), pk(*[2] * 8), 0, False, 0xFFFFFFD8),
    ("i4 max positive: 8*(-8)^2", I, ALL_M8, ALL_M8, 0, False, 512),
    ("i4 max negative: 8*(-8*7)", I, ALL_M8, ALL_7, 0, False, 0xFFFFFE40),
    ("i4 wraps negative", I, ALL_M8, ALL_M8, INT32_MAX, False, 0x800001FF),
    ("i4 clamp to INT32_MAX", I, ALL_M8, ALL_M8, INT32_MAX, True, 0x7FFFFFFF),
    ("i4 clamp to INT32_MIN", I, ALL_M8, ALL_7, 0x80000000, True, 0x80000000),
    ("0xF is 15 unsigned", U, pk(15, 0, 0, 0, 0, 0, 0, 0), pk(1, 0, 0, 0, 0, 0, 0, 0), 0, False, 15),
    ("0xF is -1 signed", I, pk(15, 0, 0, 0, 0, 0, 0, 0), pk(1, 0, 0, 0, 0, 0, 0, 0), 0, False, 0xFFFFFFFF),
]


class Dot8Int(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for name, op, s0, s1, s2, clamp, want in CASES:
            self.assertEqual(ref(op, s0, s1, s2, clamp), want, name)

    def test_translated_cases(self):
        for name, op, s0, s1, s2, clamp, want in CASES:
            with self.subTest(name):
                self.assertEqual(run(op, s0, s1, s2, clamp), want)

    def test_sgpr_sources(self):
        for name, op, s0, s1, s2, clamp, want in CASES[:6] + CASES[10:14]:
            with self.subTest(name):
                self.assertEqual(run(op, s0, s1, s2, clamp, "sgpr"), want)

    def test_random_operands(self):
        rng = random.Random(0x646F7438)
        for _ in range(40):
            s0, s1, s2 = (rng.getrandbits(32) for _ in range(3))
            for op in (U, I):
                for clamp in (False, True):
                    with self.subTest(op=op, clamp=clamp, s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                        self.assertEqual(run(op, s0, s1, s2, clamp), ref(op, s0, s1, s2, clamp))


if __name__ == "__main__":
    unittest.main()
