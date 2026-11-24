"""Translator tests for RDNA2 v_cvt_f32_i32 and v_cvt_f32_u32 (32-bit integer to binary32 float).

  v_cvt_f32_i32   D = exact (as binary32) signed int32 -> nearest f32, round to nearest even
  v_cvt_f32_u32   D = exact (as binary32) unsigned int32 -> nearest f32, round to nearest even

Every case translates a hand-written pixel-shader listing with tools/rdna2spirv.py and executes the resulting SPIR-V
via tests/semantic/spirv_mini_interp.py. Expected words are written by hand from the ISA formula and also recomputed
by a plain-Python reference, with seeded random operands.

Standard library only: no GPU, no LLVM, no external tools, no game files.

Until this file, test_int_to_float.py ran a private Python oracle (a plain-Python f32 model) that never called the
translator, as recorded in the DEBT table in tests/test_semantic_translator_use.py.
"""
import math
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from support import mrt0_words, translate_lines  # noqa: E402

MASK32 = 0xFFFFFFFF


def bits_from_f32(x):
    """Pack a Python float into its IEEE-754 binary32 32-bit unsigned word."""
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000


def f32_from_bits(bits):
    """Unpack a 32-bit unsigned word to a Python float (IEEE-754 binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_cvt_f32_i32(val):
    """"Plain-Python reference for v_cvt_f32_i32: signed int32 -> f32, round to nearest even."""
    val &= MASK32
    if val >= 0x80000000:
        x = val - 0x100000000  # two's complement to signed
    else:
        x = val
    return bits_from_f32(float(x))


def ref_cvt_f32_u32(val):
    """Plain-Python reference for v_cvt_f32_u32: unsigned int32 -> f32, round to nearest even.

    Python's float is binary64, so any exact conversion is exact; the rounding to binary32 happens in
    `bits_from_f32` via pack/unpack which follows round-to-nearest-even.
    """
    return bits_from_f32(float(val & MASK32))


def run_cvt_f32_i32(val, e64_mods=""):
    """Translate and run v_cvt_f32_i32 on a uniform i32 value. Returns [mrt0 words]."""
    v = val & MASK32
    if e64_mods:
        lines = [
            f"v_mov_b32 v1, {v:#010x}",
            f"v_cvt_f32_i32_e64 v0, {e64_mods}v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    else:
        lines = [
            f"v_mov_b32 v1, {v:#010x}",
            "v_cvt_f32_i32_e32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    return mrt0_words(translate_lines("ps_cvt_f32_i32", lines))[0]


def run_cvt_f32_u32(val, e64_mods=""):
    """Translate and run v_cvt_f32_u32 on a uniform u32 value. Returns one mrt0 word."""
    v = val & MASK32
    if e64_mods:
        lines = [
            f"v_mov_b32 v1, {v:#010x}",
            f"v_cvt_f32_u32_e64 v0, {e64_mods}v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    else:
        lines = [
            f"v_mov_b32 v1, {v:#010x}",
            "v_cvt_f32_u32_e32 v0, v1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
    return mrt0_words(translate_lines("ps_cvt_f32_u32", lines))[0]


# (input word, expected f32 word) for v_cvt_f32_i32.
# Computed from the formula: float(signed_int32), then f32 pack.
I32_HAND = [
    (0x00000000, 0x00000000),       # 0
    (0x00000001, 0x3F800000),       # 1
    (0x00000002, 0x40000000),       # 2
    (0x0000002A, 0x42280000),       # 42
    (0x00000064, 0x42C80000),       # 100
    (0x7FFFFFFF, 0x4F000000),       # max positive i32 = 2^31 - 1
    (0xFFFFFFFF, 0xBF800000),       # -1
    (0xFFFFFFD6, 0xC2280000),       # -42
    (0xFFFFFF9C, 0xC2C80000),       # -100
    (0x80000000, 0xCF000000),       # min negative i32 = -2^31
    (0x01000000, 0x4B800000),       # 2^24 -> exact in f32
    (0x01000001, 0x4B800000),       # 2^24 + 1 -> rounds down (tie, even is the 2^24 alternative)
    (0x00FFFFFF, 0x4B7FFFFF),       # 2^24 - 1 -> exact
    (0x00400000, 0x4A800000),       # 2^22 -> exact
]

# (input word, expected f32 word) for v_cvt_f32_u32.
# Computed from the formula: float(unsigned_int32), then f32 pack.
U32_HAND = [
    (0x00000000, 0x00000000),       # 0
    (0x00000001, 0x3F800000),       # 1
    (0x00000002, 0x40000000),       # 2
    (0x0000002A, 0x42280000),       # 42
    (0x00000064, 0x42C80000),       # 100
    (0x7FFFFFFF, 0x4F000000),       # 2^31 - 1 -> exact in f32
    (0x80000000, 0x4F000000),       # 2^31 -> exact
    (0xFFFFFFFF, 0x4F800000),       # max u32 -> exact
    (0x01000000, 0x4B800000),       # 2^24 -> exact
    (0x01000001, 0x4B800000),       # 2^24 + 1 -> rounds down (tie, even)
    (0x00FFFFFF, 0x4B7FFFFF),       # 2^24 - 1 -> exact
    (0x00400000, 0x4A800000),       # 2^22 -> exact
    (0x80000001, 0x4F000000),       # 2^31 + 1 -> rounds down (tie, even: the 2^31 mantissa ends in 0)
    (0x0000000A, 0x41200000),       # 10
]


class HandWrittenCases(unittest.TestCase):
    def test_v_cvt_f32_i32_hand_table(self):
        for val, expected in I32_HAND:
            with self.subTest(val=hex(val)):
                self.assertEqual(run_cvt_f32_i32(val), expected)

    def test_v_cvt_f32_u32_hand_table(self):
        for val, expected in U32_HAND:
            with self.subTest(val=hex(val)):
                self.assertEqual(run_cvt_f32_u32(val), expected)


class ReferenceCoherence(unittest.TestCase):
    """The hand-written table matches the plain-Python reference for both ops."""

    def test_i32_ref_matches_hand(self):
        for val, expected in I32_HAND:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_cvt_f32_i32(val), expected)

    def test_u32_ref_matches_hand(self):
        for val, expected in U32_HAND:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_cvt_f32_u32(val), expected)

    def test_translator_matches_reference(self):
        rng = random.Random(0xC007C437)
        for _ in range(60):
            for op, ref_fn, run_fn in (
                ("v_cvt_f32_i32", ref_cvt_f32_i32, run_cvt_f32_i32),
                ("v_cvt_f32_u32", ref_cvt_f32_u32, run_cvt_f32_u32),
            ):
                val = rng.getrandbits(32)
                with self.subTest(op=op, val=hex(val)):
                    self.assertEqual(run_fn(val), ref_fn(val))


class SGPRSource(unittest.TestCase):
    def test_sgpr_source(self):
        lines = [
            "s_mov_b32 s1, 0x0000002A",       # 42
            "v_cvt_f32_i32_e64 v0, s1",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_sgpr", lines)
        self.assertEqual(mrt0_words(spv)[0], 0x42280000)


class InlineConstantSource(unittest.TestCase):
    def test_inline_literal(self):
        lines = [
            "v_cvt_f32_i32_e32 v0, 10",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_lit", lines)
        self.assertEqual(mrt0_words(spv)[0], 0x41200000)  # 10.0


class VOP3Modifiers(unittest.TestCase):
    """VOP3 modifiers on v_cvt_f32_i32/v_cvt_f32_u32.

    The RDNA2 ISA applies source modifiers (neg/abs) to the integer source in two's complement (not to the float
    result). The translator currently implements them via OpFAbs/OpFNegate on the bit pattern reinterpreted as float,
    which gives wrong results for negative sources (where the bits are a NaN-narrowing pattern). These tests document
    the actual translator output for positive sources (where the bits represent a normal float).
    """

    def test_omod_mul2(self):
        # 1 * 2 = 2.0
        spv = translate_lines("ps_cvt_omod", [
            "v_mov_b32 v1, 0x00000001",
            "v_cvt_f32_i32_e64 v0, v1 mul:2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(spv)[0], 0x40000000)  # 2.0

    def test_omod_div2(self):
        # 10 / 2 = 5.0
        spv = translate_lines("ps_cvt_omod_div2", [
            "v_mov_b32 v1, 0x0000000A",
            "v_cvt_f32_i32_e64 v0, v1 div:2",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(spv)[0], 0x40A00000)  # 5.0

    def test_omod_mul4(self):
        # 42 * 4 = 168.0
        spv = translate_lines("ps_cvt_omod_mul4", [
            "v_mov_b32 v1, 0x0000002A",
            "v_cvt_f32_i32_e64 v0, v1 mul:4",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(spv)[0], 0x43280000)  # 168.0

    def test_clamp_negative_saturates_to_zero(self):
        # -1.0 clamped -> 0.0
        spv = translate_lines("ps_cvt_clamp_neg", [
            "v_mov_b32 v1, 0xFFFFFFFF",
            "v_cvt_f32_i32_e64 v0, v1 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(spv)[0], 0x00000000)

    def test_clamp_positive_large_saturates_to_one(self):
        # 1000.0 clamped -> 1.0
        spv = translate_lines("ps_cvt_clamp_pos", [
            "v_mov_b32 v1, 0x000003E8",
            "v_cvt_f32_i32_e64 v0, v1 clamp",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ])
        self.assertEqual(mrt0_words(spv)[0], 0x3F800000)  # 1.0

    def test_sgpr_with_modifiers(self):
        # s_mov_b32 s1, 42 -> v_cvt_f32_i32_e64 v0, s1 mul:2
        lines = [
            "s_mov_b32 s1, 0x0000002A",
            "v_cvt_f32_i32_e64 v0, s1 mul:4",
            "exp mrt0 v0, v0, v0, v0 done vm",
            "s_endpgm",
        ]
        spv = translate_lines("ps_cvt_sgpr_mod", lines)
        self.assertEqual(mrt0_words(spv)[0], 0x43280000)  # 168.0


if __name__ == "__main__":
    unittest.main()