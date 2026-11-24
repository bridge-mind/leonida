"""Value tests for v_cvt_u32_f32 (f32 -> u32 with truncation toward zero and saturation):

  v_cvt_u32_f32   D = 0                         if x is NaN or x <= -0.0 (negative)
                      0xFFFFFFFF                if x >= 2^32 (including +inf)
                      trunc(x) as uint32        otherwise (0.0 <= x < 2^32)

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and executes the resulting SPIR-V via tests/semantic/spirv_mini_interp.py.
Standard library only.
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

from support import mrt0_words, translate_lines  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF


def f32_to_bits(x):
    """Pack a Python float into its IEEE-754 binary32 32-bit unsigned word."""
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def bits_to_f32(bits):
    """Unpack a 32-bit unsigned word as an IEEE-754 binary32 Python float."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def ref_cvt_u32_f32(bits):
    """Plain-Python reference for RDNA2 v_cvt_u32_f32 given a binary32 word."""
    v = bits_to_f32(bits)
    if math.isnan(v) or v < 0.0 or (bits & MASK32) == 0x80000000:
        return 0
    if v >= 4294967296.0:  # 2^32
        return 0xFFFFFFFF
    return math.trunc(v) & MASK32


def ref_cvt_i32_f32(bits):
    """Plain-Python reference for RDNA2 v_cvt_i32_f32 given a binary32 word."""
    v = bits_to_f32(bits)
    if math.isnan(v):
        return 0
    if v >= 2147483648.0:  # 2^31
        return 0x7FFFFFFF
    if v <= -2147483648.0:  # -2^31
        return 0x80000000
    return math.trunc(v) & MASK32


def run_cvt(op, src_bits, *, setup=None, src_expr="v1"):
    """Translate and run `op v0, <src_expr>` on a pixel shader; return the uniform mrt0 word."""
    if setup is None:
        setup = [f"v_mov_b32 v1, {src_bits & MASK32:#010x}"]
    lines = list(setup) + [
        f"{op} v0, {src_expr}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    spv = translate_lines(f"ps_{op.split()[0]}", lines)
    words = mrt0_words(spv)
    if len(set(words)) != 1:
        raise AssertionError(f"{op}: exported mrt0 words differ: {[hex(w) for w in words]}")
    return words[0]


# Hand-verified (input_bits, expected_u32)
HAND_CASES = [
    (f32_to_bits(0.0), 0),
    (f32_to_bits(0.5), 0),
    (0x3F7FFFFF, 0),                           # 0.99999994 (largest f32 < 1.0)
    (f32_to_bits(1.0), 1),
    (f32_to_bits(1.9), 1),
    (f32_to_bits(42.75), 42),
    (f32_to_bits(float(2**24)), 2**24),
    (0x4F7FFFFF, 4294967040),                  # largest f32 < 2^32
    (0x4F800000, 0xFFFFFFFF),                  # 2^32 -> saturates
    (f32_to_bits(1e20), 0xFFFFFFFF),
    (0x7F800000, 0xFFFFFFFF),                  # +inf -> saturates
    (0x80000000, 0),                           # -0.0 -> 0
    (f32_to_bits(-0.5), 0),
    (f32_to_bits(-1.0), 0),
    (f32_to_bits(-1e20), 0),
    (0xFF800000, 0),                           # -inf -> 0
    (0x7FC00000, 0),                           # +qNaN -> 0
    (0xFFC00000, 0),                           # -qNaN -> 0
    (0x7F800001, 0),                           # +sNaN -> 0
    (0xFF800001, 0),                           # -sNaN -> 0
]


class ReferenceTests(unittest.TestCase):
    def test_hand_constants(self):
        for bits, expected in HAND_CASES:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(ref_cvt_u32_f32(bits), expected)


class CvtU32F32Tests(unittest.TestCase):
    def test_exact_fractional_overflow_negative_and_nan(self):
        for op in ("v_cvt_u32_f32", "v_cvt_u32_f32_e32", "v_cvt_u32_f32_e64"):
            for bits, expected in HAND_CASES:
                with self.subTest(op=op, bits=hex(bits)):
                    got = run_cvt(op, bits)
                    self.assertEqual(
                        got, expected,
                        f"{op}({bits:#010x}): got {got:#010x}, expected {expected:#010x}",
                    )

    def test_contrast_with_v_cvt_i32_f32(self):
        # [2^31, 2^32): u32 converts exact, i32 saturates to 0x7FFFFFFF
        upper_range = [
            (f32_to_bits(float(2**31)), 2**31, 0x7FFFFFFF),
            (f32_to_bits(3e9), 3000000000, 0x7FFFFFFF),
            (0x4F7FFFFF, 4294967040, 0x7FFFFFFF),
        ]
        for bits, exp_u32, exp_i32 in upper_range:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(run_cvt("v_cvt_u32_f32", bits), exp_u32)
                self.assertEqual(run_cvt("v_cvt_i32_f32", bits), exp_i32)

        # Negative values: u32 gives 0, i32 gives negative signed integer
        neg_cases = [
            (f32_to_bits(-1.5), 0, (-1) & MASK32),
            (f32_to_bits(-42.75), 0, (-42) & MASK32),
            (0xCF000000, 0, 0x80000000),  # -2^31
        ]
        for bits, exp_u32, exp_i32 in neg_cases:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(run_cvt("v_cvt_u32_f32", bits), exp_u32)
                self.assertEqual(run_cvt("v_cvt_i32_f32", bits), exp_i32)

    def test_operand_kinds_and_vop3_modifiers(self):
        cases = [
            ("v_cvt_u32_f32_e32", ["s_mov_b32 s0, 0x422b0000"], "s0", 42),        # 42.75
            ("v_cvt_u32_f32_e32", [], "0x42c88000", 100),                          # 100.25 literal
            ("v_cvt_u32_f32_e32", [], "4.0", 4),                                   # inline float
            ("v_cvt_u32_f32_e32", [], "-1.0", 0),                                  # inline float -1.0 -> 0
            ("v_cvt_u32_f32_e64", ["s_mov_b32 s0, 0x447a8000"], "s0", 1002),       # 1002.0
            ("v_cvt_u32_f32_e64", [], "2.0", 2),
            # VOP3 source modifiers (neg / abs)
            ("v_cvt_u32_f32_e64", [f"v_mov_b32 v1, {f32_to_bits(-7.5):#010x}"], "-v1", 7),
            ("v_cvt_u32_f32_e64", [f"v_mov_b32 v1, {f32_to_bits(-7.5):#010x}"], "|v1|", 7),
            ("v_cvt_u32_f32_e64", [f"v_mov_b32 v1, {f32_to_bits(7.5):#010x}"], "-|v1|", 0),
        ]
        for op, setup, src_expr, expected in cases:
            with self.subTest(op=op, src_expr=src_expr):
                got = run_cvt(op, 0, setup=setup, src_expr=src_expr)
                self.assertEqual(got, expected)

    def test_seeded_random_f32_patterns(self):
        rng = random.Random(0xC7032F32)
        for _ in range(40):
            # Mix exponents across subnormal, normal [0, 2^32), overflow, negative, and NaN/inf
            sign = rng.choice((0, 0x80000000))
            exp = rng.choice((0, 127, 128, 150, 158, 159, 200, 255))
            mant = rng.randrange(0, 1 << 23)
            bits = sign | (exp << 23) | mant
            expected = ref_cvt_u32_f32(bits)
            with self.subTest(bits=hex(bits)):
                got = run_cvt("v_cvt_u32_f32", bits)
                self.assertEqual(got, expected)


if __name__ == "__main__":
    unittest.main()
