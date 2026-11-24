"""v_mul_hi_u32 and v_mul_hi_i32 (RDNA2 VOP3): D = high 32 bits of the 64-bit product of two 32-bit sources.

Every case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
word is written down by hand next to each case and also recomputed by the plain-Python reference below.
Standard library only: no GPU, no LLVM, no game data.
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
INT32_MAX = 0x7FFFFFFF
INT32_MIN = 0x80000000


def ref_u32(a, b):
    return ((a & MASK32) * (b & MASK32) >> 32) & MASK32


def ref_i32(a, b):
    sa = a - (1 << 32) if a & 0x80000000 else a
    sb = b - (1 << 32) if b & 0x80000000 else b
    return (sa * sb >> 32) & MASK32


def run(op, a, b, form="vgpr"):
    """Translate and run `op v0, <a>, <b>` on a one-lane pixel shader; return v0 as exported to mrt0."""
    if form == "vgpr":
        setup = [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}"]
        srcs = "v1, v2"
    elif form == "sgpr":
        setup = [f"s_mov_b32 s1, {a:#x}", f"s_mov_b32 s2, {b:#x}"]
        srcs = "s1, s2"
    else:   # inline constant or literal as the second source, v1 as the first
        setup = [f"v_mov_b32 v1, {a:#x}"]
        srcs = f"v1, {b:#x}"
    lines = setup + [f"{op} v0, {srcs}", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_mul_hi", lines))[0]


def run_alias(op, a, b, mode):
    """Write the result over a source: mode "src0" (v1), "src1" (v2) or "square" (v1 = op v1, v1, with b ignored)."""
    dst, srcs = {"src0": ("v1", "v1, v2"), "src1": ("v2", "v1, v2"), "square": ("v1", "v1, v1")}[mode]
    lines = [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}", f"{op} {dst}, {srcs}",
             f"exp mrt0 {dst}, {dst}, {dst}, {dst} done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_mul_hi", lines))[0]


def run_mixed(op, vgpr, sgpr, scalar_first):
    """One VGPR and one SGPR operand, in either order."""
    srcs = "s4, v2" if scalar_first else "v2, s4"
    lines = [f"v_mov_b32 v2, {vgpr:#x}", f"s_mov_b32 s4, {sgpr:#x}", f"{op} v0, {srcs}",
             "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_mul_hi", lines))[0]


# (a, b, v_mul_hi_u32 word, v_mul_hi_i32 word), written by hand.
CASES = [
    (0x80000000, 2, 0x00000001, 0xFFFFFFFF),        # 2^31 * 2 = 2^32; (-2^31) * 2 = -2^32
    (0x10000, 0x10000, 0x00000001, 0x00000001),     # 2^32
    (100, 200, 0x00000000, 0x00000000),
    (0xFFFF, 0xFFFF, 0x00000000, 0x00000000),
    (0, 0xFFFFFFFF, 0x00000000, 0x00000000),
    (0xFFFFFFFF, 1, 0x00000000, 0xFFFFFFFF),        # u: 2^32-1 fits; i: -1 * 1 = -1
    (0xFFFFFFFF, 2, 0x00000001, 0xFFFFFFFF),        # u: 0x1_FFFFFFFE; i: -1 * 2 = -2
    (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0x00000000),   # u: 0xFFFFFFFE_00000001; i: (-1)*(-1) = 1
    (3, 4, 0x00000000, 0x00000000),
    (10, 0xFFFFFFFB, 0x00000009, 0xFFFFFFFF),       # i: 10 * -5 = -50
    (INT32_MIN, 0xFFFFFFFF, 0x7FFFFFFF, 0x00000000),    # i: -2^31 * -1 = 2^31
    (INT32_MIN, INT32_MIN, 0x40000000, 0x40000000),     # u: 2^62; i: (-2^31)^2 = 2^62
    (INT32_MAX, INT32_MAX, 0x3FFFFFFF, 0x3FFFFFFF),     # (2^31-1)^2 = 0x3FFFFFFF_00000001
    (INT32_MIN, INT32_MAX, 0x3FFFFFFF, 0xC0000000),     # i: -2^62 + 2^31 = 0xC0000000_80000000
    (0x12345678, 0x9ABCDEF0, 0x0B00EA4E, 0xF8CC93D6),   # 0x12345678 * 0x9ABCDEF0 and its signed reading
]


class MulHi(unittest.TestCase):
    def test_hand_written_table_matches_reference(self):
        for a, b, u, i in CASES:
            self.assertEqual(ref_u32(a, b), u, (hex(a), hex(b)))
            self.assertEqual(ref_i32(a, b), i, (hex(a), hex(b)))

    def test_translated_cases(self):
        for a, b, u, i in CASES:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(run("v_mul_hi_u32", a, b), u)
                self.assertEqual(run("v_mul_hi_i32", a, b), i)

    def test_sgpr_and_literal_sources(self):
        for a, b, u, i in CASES[:8]:
            for form in ("sgpr", "literal"):
                with self.subTest(a=hex(a), b=hex(b), form=form):
                    self.assertEqual(run("v_mul_hi_u32", a, b, form), u)
                    self.assertEqual(run("v_mul_hi_i32", a, b, form), i)

    def test_destination_overlaps_a_source(self):
        # The high product differs from the overwritten source in these rows, so returning that source fails them.
        for a, b, u, i in CASES:
            for mode in ("src0", "src1"):
                with self.subTest(a=hex(a), b=hex(b), mode=mode):
                    self.assertEqual(run_alias("v_mul_hi_u32", a, b, mode), u)
                    self.assertEqual(run_alias("v_mul_hi_i32", a, b, mode), i)

    def test_in_place_square(self):
        # (a, high word of a*a unsigned, signed), by hand: 0x80000000^2 = 2^62, 0xFFFFFFFF^2 = 0xFFFFFFFE_00000001,
        # 0x10000^2 = 2^32, 0x7FFFFFFF^2 = 0x3FFFFFFF_00000001, 0x12345678^2 = 0x14B66DC_1DF4D840.
        for a, u, i in ((0x80000000, 0x40000000, 0x40000000), (0xFFFFFFFF, 0xFFFFFFFE, 0),
                        (0x10000, 1, 1), (0x7FFFFFFF, 0x3FFFFFFF, 0x3FFFFFFF), (0x12345678, 0x014B66DC, 0x014B66DC)):
            self.assertEqual((ref_u32(a, a), ref_i32(a, a)), (u, i), hex(a))
            with self.subTest(a=hex(a)):
                self.assertEqual(run_alias("v_mul_hi_u32", a, a, "square"), u)
                self.assertEqual(run_alias("v_mul_hi_i32", a, a, "square"), i)

    def test_mixed_vector_and_scalar_sources(self):
        # (vector operand, scalar operand, unsigned word, signed word), by hand.
        for v, sc, u, i in ((0x20000, 0x10000, 2, 2),                       # 2^17 * 2^16 = 2^33
                            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0),
                            (0x80000000, 2, 1, 0xFFFFFFFF),
                            (0xFFFFFFFB, 10, 9, 0xFFFFFFFF),                # -5 * 10 = -50
                            (INT32_MAX, INT32_MIN, 0x3FFFFFFF, 0xC0000000)):
            self.assertEqual((ref_u32(v, sc), ref_i32(v, sc)), (u, i), (hex(v), hex(sc)))
            for scalar_first in (False, True):
                with self.subTest(v=hex(v), s=hex(sc), scalar_first=scalar_first):
                    self.assertEqual(run_mixed("v_mul_hi_u32", v, sc, scalar_first), u)
                    self.assertEqual(run_mixed("v_mul_hi_i32", v, sc, scalar_first), i)

    def test_random_operands(self):
        rng = random.Random(0x6D756C68)
        for _ in range(60):
            a, b = rng.getrandbits(32), rng.getrandbits(32)
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(run("v_mul_hi_u32", a, b), ref_u32(a, b))
                self.assertEqual(run("v_mul_hi_i32", a, b), ref_i32(a, b))

    def test_signedness_is_not_confused(self):
        # -1 * -1: the signed high word is 0, the unsigned one 0xFFFFFFFE.
        self.assertEqual(run("v_mul_hi_i32", 0xFFFFFFFF, 0xFFFFFFFF), 0)
        self.assertEqual(run("v_mul_hi_u32", 0xFFFFFFFF, 0xFFFFFFFF), 0xFFFFFFFE)


if __name__ == "__main__":
    unittest.main()
