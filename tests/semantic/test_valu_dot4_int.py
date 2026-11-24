"""v_dot4_u32_u8 and v_dot4_i32_i8 (RDNA2 VOP3P): D = S2 + sum of the four byte products of S0 and S1 (unsigned bytes
for u8, signed for i8), modulo 2^32; with `clamp` the result saturates to [0, 2^32-1] (u8) or [-2^31, 2^31-1] (i8).

Each case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
words are written by hand and also recomputed by a plain-Python reference. v_dot4acc_u32_u8 is not a RDNA2 opcode the
translator knows, and stays refused. Standard library only: no GPU, no LLVM, no game data.
"""
import os
import random
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

import support  # noqa: E402
from rdna2spirv import Unsupported  # noqa: E402

MASK32 = 0xFFFFFFFF
INT32_MIN, INT32_MAX = -0x80000000, 0x7FFFFFFF


def pk(*b):
    """Pack four bytes, byte 0 first, into a dword."""
    return sum((x & 0xFF) << (8 * i) for i, x in enumerate(b))


def ref(op, s0, s1, s2, clamp=False):
    signed = "i8" in op
    def byte(v, i):
        x = (v >> (8 * i)) & 0xFF
        return x - 256 if signed and x & 0x80 else x
    total = sum(byte(s0, i) * byte(s1, i) for i in range(4))
    if signed:
        acc = s2 - (1 << 32) if s2 & 0x80000000 else s2
        total += acc
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
    return support.mrt0_words(support.translate_lines("ps_dot4", lines))[0]


# (name, op, S0, S1, S2, clamp, expected word), written by hand.
CASES = [
    ("u8 basic: 1*5+2*6+3*7+4*8", "v_dot4_u32_u8", pk(1, 2, 3, 4), pk(5, 6, 7, 8), 0, False, 70),
    ("u8 basic with accumulator", "v_dot4_u32_u8", pk(1, 2, 3, 4), pk(5, 6, 7, 8), 1000, False, 1070),
    ("u8 zero operand keeps S2", "v_dot4_u32_u8", 0, 0x12345678, 0xDEADBEEF, False, 0xDEADBEEF),
    ("u8 zero operand keeps S2 (swapped)", "v_dot4_u32_u8", 0x12345678, 0, 0xCAFEBABE, False, 0xCAFEBABE),
    ("u8 maximum: 4*255*255", "v_dot4_u32_u8", MASK32, MASK32, 0, False, 260100),
    ("u8 maximum plus 100", "v_dot4_u32_u8", MASK32, MASK32, 100, False, 260200),
    ("u8 wraps to 0", "v_dot4_u32_u8", MASK32, MASK32, 0xFFFC07FC, False, 0),
    ("u8 wraps to 1", "v_dot4_u32_u8", MASK32, MASK32, 0xFFFC07FD, False, 1),
    ("u8 wraps without clamp", "v_dot4_u32_u8", MASK32, MASK32, 0xFFFF0000, False, 0x0002F804),
    ("u8 clamp saturates", "v_dot4_u32_u8", MASK32, MASK32, 0xFFFF0000, True, 0xFFFFFFFF),
    ("i8 basic", "v_dot4_i32_i8", pk(1, 2, 3, 4), pk(5, 6, 7, 8), 0, False, 70),
    ("i8 basic with accumulator", "v_dot4_i32_i8", pk(1, 2, 3, 4), pk(5, 6, 7, 8), 500, False, 570),
    ("i8 mixed signs: -5-12+21-32", "v_dot4_i32_i8", pk(-1, 2, -3, -4), pk(5, -6, -7, 8), 0, False, 0xFFFFFFE4),
    ("i8 max positive: 4*(-128)^2", "v_dot4_i32_i8", pk(-128, -128, -128, -128), pk(-128, -128, -128, -128), 0, False, 65536),
    ("i8 max negative: 4*(-128*127)", "v_dot4_i32_i8", pk(-128, -128, -128, -128), pk(127, 127, 127, 127), 0, False, 0xFFFF0200),
    ("i8 wraps negative", "v_dot4_i32_i8", pk(-128, -128, -128, -128), pk(-128, -128, -128, -128), INT32_MAX, False, 0x8000FFFF),
    ("i8 clamp to INT32_MAX", "v_dot4_i32_i8", pk(-128, -128, -128, -128), pk(-128, -128, -128, -128), INT32_MAX, True, 0x7FFFFFFF),
    ("i8 clamp to INT32_MIN", "v_dot4_i32_i8", pk(-128, -128, -128, -128), pk(127, 127, 127, 127), 0x80000000, True, 0x80000000),
    ("0xFF is 255 unsigned", "v_dot4_u32_u8", pk(255, 0, 0, 0), pk(1, 0, 0, 0), 0, False, 255),
    ("0xFF is -1 signed", "v_dot4_i32_i8", pk(255, 0, 0, 0), pk(1, 0, 0, 0), 0, False, 0xFFFFFFFF),
]


class Dot4Int(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for name, op, s0, s1, s2, clamp, want in CASES:
            self.assertEqual(ref(op, s0, s1, s2, clamp), want, name)

    def test_translated_cases(self):
        for name, op, s0, s1, s2, clamp, want in CASES:
            with self.subTest(name):
                self.assertEqual(run(op, s0, s1, s2, clamp), want)

    def test_sgpr_sources(self):
        for name, op, s0, s1, s2, clamp, want in CASES[:6] + CASES[10:13]:
            with self.subTest(name):
                self.assertEqual(run(op, s0, s1, s2, clamp, "sgpr"), want)

    def test_random_operands(self):
        rng = random.Random(0x646F7434)
        for _ in range(40):
            s0, s1, s2 = (rng.getrandbits(32) for _ in range(3))
            for op in ("v_dot4_u32_u8", "v_dot4_i32_i8"):
                for clamp in (False, True):
                    with self.subTest(op=op, clamp=clamp, s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                        self.assertEqual(run(op, s0, s1, s2, clamp), ref(op, s0, s1, s2, clamp))

    def test_dot4acc_is_refused_not_guessed(self):
        with self.assertRaises(Unsupported):
            run("v_dot4acc_u32_u8", pk(1, 2, 3, 4), pk(5, 6, 7, 8), 1000)


if __name__ == "__main__":
    unittest.main()
