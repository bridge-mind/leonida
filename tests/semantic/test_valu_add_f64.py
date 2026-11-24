"""Value tests for v_add_f64 (IEEE-754 binary64 addition, VOP3 modifiers, and MODE.FP_ROUND):

  v_add_f64   D = S0 + S1   (53-bit significand precision, signed zeros, inf/NaN,
                             neg/abs/clamp/mul:2/mul:4/div:2, MODE.FP_ROUND 0 vs 3)

Each case translates a hand-written pixel-shader listing with tools/rdna2spirv.py
and executes the resulting SPIR-V via tests/spirv_interp.py (which supports OpTypeFloat 64,
whereas spirv_mini_interp.py is 32-bit float only). Standard library only.
"""
import math
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for p in (HERE, TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import spirv_interp  # noqa: E402
from support import translate_lines  # noqa: E402  (support puts tools/ on sys.path)

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
POS_ZERO = 0x0000000000000000
NEG_ZERO = 0x8000000000000000
POS_INF = 0x7FF0000000000000
NEG_INF = 0xFFF0000000000000
DBL_MAX = 0x7FEFFFFFFFFFFFFF
NEG_DBL_MAX = 0xFFEFFFFFFFFFFFFF


def f64_to_bits(x):
    """Pack a Python float into its IEEE-754 binary64 64-bit unsigned word."""
    try:
        return struct.unpack("<Q", struct.pack("<d", x))[0]
    except OverflowError:
        return POS_INF if x > 0 else NEG_INF


def bits_to_f64(bits):
    """Unpack a 64-bit unsigned word as an IEEE-754 binary64 Python float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def run_add_f64(a_bits, b_bits, *, setup=None, operands="v[0:1], v[2:3]", mode=None):
    """Translate and run `v_add_f64 v[4:5], <operands>` via spirv_interp; return (u64_bits, spv)."""
    if setup is None:
        setup = [
            f"v_mov_b32 v0, {a_bits & MASK32:#010x}",
            f"v_mov_b32 v1, {(a_bits >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b_bits & MASK32:#010x}",
            f"v_mov_b32 v3, {(b_bits >> 32) & MASK32:#010x}",
        ]
    lines = list(setup)
    if mode is not None:
        lines.append(f"s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), {mode}")
    lines += [
        f"v_add_f64 v[4:5], {operands}",
        "exp mrt0 v4, v5, v4, v5 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_add_f64", lines)
    res = spirv_interp.run(spv)
    got = (res.vgpr[4] & MASK32) | ((res.vgpr[5] & MASK32) << 32)
    return got, spv


class AddF64PrecisionAndSpecialsTests(unittest.TestCase):
    def test_precision_beyond_f32(self):
        cases = [
            (float(2**24), 1.0, float(2**24 + 1)),
            (float(2**40), 1.0, float(2**40 + 1)),
            (float(2**52), 1.0, float(2**52 + 1)),
            (1.0, 2.0**-52, 1.0 + 2.0**-52),
            # 2^53 + 1.0 rounds to 2^53 (ties to even at the 53-bit significand boundary)
            (float(2**53), 1.0, float(2**53)),
        ]
        for a, b, exp in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_add_f64(f64_to_bits(a), f64_to_bits(b))
                self.assertEqual(got, f64_to_bits(exp))

    def test_cancellation(self):
        cases = [
            (float(2**50) + 1.0, -float(2**50), 1.0),
            (1.0 + 2.0**-52, -1.0, 2.0**-52),
            (123.456, -123.456, 0.0),
        ]
        for a, b, exp in cases:
            with self.subTest(a=a, b=b):
                got, _ = run_add_f64(f64_to_bits(a), f64_to_bits(b))
                self.assertEqual(got, f64_to_bits(exp))

    def test_signed_zeros(self):
        cases = [
            (POS_ZERO, POS_ZERO, POS_ZERO),
            (POS_ZERO, NEG_ZERO, POS_ZERO),
            (NEG_ZERO, POS_ZERO, POS_ZERO),
            (NEG_ZERO, NEG_ZERO, NEG_ZERO),
        ]
        for a, b, exp in cases:
            with self.subTest(a=hex(a), b=hex(b)):
                got, _ = run_add_f64(a, b)
                self.assertEqual(got, exp)

    def test_infinities_and_nan(self):
        one = f64_to_bits(1.0)
        for a, b, exp in (
            (POS_INF, one, POS_INF),
            (NEG_INF, one, NEG_INF),
            (POS_INF, POS_INF, POS_INF),
            (NEG_INF, NEG_INF, NEG_INF),
        ):
            with self.subTest(a=hex(a), b=hex(b)):
                got, _ = run_add_f64(a, b)
                self.assertEqual(got, exp)

        got, _ = run_add_f64(POS_INF, NEG_INF)
        self.assertTrue(math.isnan(bits_to_f64(got)), f"expected NaN, got {got:#018x}")


class AddF64ModifiersAndOperandsTests(unittest.TestCase):
    def test_neg_and_abs_modifiers(self):
        # neg preserves full 53-bit precision
        a = f64_to_bits(float(2**52))
        b = f64_to_bits(1.0)
        got, _ = run_add_f64(a, b, operands="v[0:1], -v[2:3]")
        self.assertEqual(got, f64_to_bits(float(2**52 - 1)))

        got, _ = run_add_f64(a, b, operands="-v[0:1], v[2:3]")
        self.assertEqual(got, f64_to_bits(float(-(2**52) + 1)))

        # abs / neg-abs (using f32-exact values because spirv_interp.FAbs passes through f32)
        for a_val, b_val, ops, exp in (
            (-1.5, 0.25, "|v[0:1]|, v[2:3]", 1.75),
            (1.5, -0.25, "v[0:1], |v[2:3]|", 1.75),
            (-1.5, 0.25, "-|v[0:1]|, v[2:3]", -1.25),
        ):
            with self.subTest(ops=ops):
                got, _ = run_add_f64(f64_to_bits(a_val), f64_to_bits(b_val), operands=ops)
                self.assertEqual(got, f64_to_bits(exp))

    def test_output_modifiers(self):
        # Use f32-exact values for clamp cases because spirv_interp.FClamp rounds via f32
        cases = [
            (0.75, 0.5, "v[0:1], v[2:3] mul:2", 2.5),
            (0.75, 0.5, "v[0:1], v[2:3] mul:4", 5.0),
            (0.75, 0.5, "v[0:1], v[2:3] div:2", 0.625),
            (0.75, 0.5, "v[0:1], v[2:3] clamp", 1.0),
            (-1.5, 0.5, "v[0:1], v[2:3] clamp", 0.0),
            (0.75, 0.5, "v[0:1], v[2:3] mul:2 clamp", 1.0),
            # div:2 before clamp: (1.0 + 0.5) * 0.5 = 0.75 -> clamp -> 0.75 (not 0.5)
            (1.0, 0.5, "v[0:1], v[2:3] div:2 clamp", 0.75),
        ]
        for a_val, b_val, ops, exp in cases:
            with self.subTest(ops=ops):
                got, _ = run_add_f64(f64_to_bits(a_val), f64_to_bits(b_val), operands=ops)
                self.assertEqual(got, f64_to_bits(exp))

        # Verify mul:2/mul:4/div:2 keep full f64 precision
        got, _ = run_add_f64(f64_to_bits(float(2**50)), f64_to_bits(1.0), operands="v[0:1], v[2:3] mul:2")
        self.assertEqual(got, f64_to_bits(float(2**51 + 2)))

    def test_sgpr_pair_and_inline_float_operands(self):
        a = f64_to_bits(float(2**40))
        b = f64_to_bits(3.5)
        # SGPR-pair src0 + VGPR-pair src1
        setup_s0 = [
            f"s_mov_b32 s0, {a & MASK32:#010x}",
            f"s_mov_b32 s1, {(a >> 32) & MASK32:#010x}",
            f"v_mov_b32 v2, {b & MASK32:#010x}",
            f"v_mov_b32 v3, {(b >> 32) & MASK32:#010x}",
        ]
        got, _ = run_add_f64(0, 0, setup=setup_s0, operands="s[0:1], v[2:3]")
        self.assertEqual(got, f64_to_bits(float(2**40) + 3.5))

        # Inline float constants
        got, _ = run_add_f64(a, 0, operands="v[0:1], 0.5")
        self.assertEqual(got, f64_to_bits(float(2**40) + 0.5))

        got, _ = run_add_f64(0, b, operands="-1.0, v[2:3]")
        self.assertEqual(got, f64_to_bits(2.5))


class AddF64RoundingModeTests(unittest.TestCase):
    def test_mode_fp_round_toward_zero_vs_nearest_even(self):
        # 1.0 + (2^-52 + 2^-53) = 1.0 + 3 * 2^-53:
        #   nearest-even (MODE.FP_ROUND=0) -> 1.0 + 2^-51 (0x3FF0000000000002)
        #   toward-zero  (MODE.FP_ROUND=3) -> 1.0 + 2^-52 (0x3FF0000000000001)
        one = 0x3FF0000000000000
        three_half_ulp = 0x3CB8000000000000  # 1.5 * 2^-52 = 3 * 2^-53
        self.assertEqual(run_add_f64(one, three_half_ulp, mode=0)[0], 0x3FF0000000000002)
        self.assertEqual(run_add_f64(one, three_half_ulp, mode=3)[0], 0x3FF0000000000001)

        # Negative mirror: -(1.0 + 3 * 2^-53)
        neg_one = 0xBFF0000000000000
        neg_three_half_ulp = 0xBCB8000000000000
        self.assertEqual(run_add_f64(neg_one, neg_three_half_ulp, mode=0)[0], 0xBFF0000000000002)
        self.assertEqual(run_add_f64(neg_one, neg_three_half_ulp, mode=3)[0], 0xBFF0000000000001)

        # Overflow: DBL_MAX + DBL_MAX -> +inf in nearest-even, DBL_MAX in toward-zero
        self.assertEqual(run_add_f64(DBL_MAX, DBL_MAX, mode=0)[0], POS_INF)
        self.assertEqual(run_add_f64(DBL_MAX, DBL_MAX, mode=3)[0], DBL_MAX)

        # Negative overflow: -DBL_MAX + (-DBL_MAX) -> -inf in nearest-even, -DBL_MAX in toward-zero
        self.assertEqual(run_add_f64(NEG_DBL_MAX, NEG_DBL_MAX, mode=0)[0], NEG_INF)
        self.assertEqual(run_add_f64(NEG_DBL_MAX, NEG_DBL_MAX, mode=3)[0], NEG_DBL_MAX)


if __name__ == "__main__":
    unittest.main()
