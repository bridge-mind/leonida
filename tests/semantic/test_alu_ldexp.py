from __future__ import annotations

"""Semantic interpreter tests for AMD RDNA2 floating-point exponent scaling (v_ldexp_f32).

Covers:
  - IEEE 754 single-precision (binary32) ldexp scaling: D = S0 * 2^S1
  - Denormal / subnormal handling under both IEEE 754 preserve mode (gradual underflow
    with round-to-nearest-even down to 2^-149) and FTZ/DAZ mode (flush-to-zero)
  - Overflow to +/- infinity on large positive exponents and extreme integers (e.g. 0x7FFFFFFF)
  - Underflow to sign-preserved +/- zero on large negative exponents and extreme integers (e.g. -0x80000000)
  - Special floating-point value preservation (+0.0, -0.0, +inf, -inf, NaN)
  - Hardware modifiers: abs (|S0|), neg (-S0), and clamp ([0.0, 1.0])
  - Dual oracle verification: bit-accurate integer VALU model cross-checked against
    an exact rational Fraction oracle and Python's math.ldexp
  - Negative controls ensuring mutated arithmetic or missing edge cases fail assertions.
"""

import math
import os
import random
import struct
import sys
import unittest
from fractions import Fraction
from typing import Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si

MASK32 = 0xFFFFFFFF
SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
MANT_MASK = 0x007FFFFF

POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
SNAN = 0x7F800001

ONE = 0x3F800000        # 1.0f
TWO = 0x40000000        # 2.0f
HALF = 0x3F000000       # 0.5f
MAX_FINITE = 0x7F7FFFFF  # largest finite binary32 (~3.402823466e+38)
MIN_NORMAL = 0x00800000  # smallest positive normal binary32 (2^-126, ~1.175494351e-38)
MAX_SUBNORMAL = 0x007FFFFF  # largest positive subnormal binary32 (~1.175494211e-38)
MIN_SUBNORMAL = 0x00000001  # smallest positive subnormal binary32 (2^-149, ~1.401298464e-45)


def s32(x: int) -> int:
    """Interpret a 32-bit integer word as a signed two's-complement integer."""
    x &= MASK32
    return x - (1 << 32) if (x & SIGN_MASK) else x


def bits_to_f32(b: int) -> float:
    """Unpack a 32-bit unsigned integer to a Python float (binary32)."""
    return struct.unpack("<f", struct.pack("<I", b & MASK32))[0]


def f32_to_bits(f: float) -> int:
    """Pack a Python float to a 32-bit unsigned integer word (binary32)."""
    if math.isnan(f):
        return QNAN
    try:
        return struct.unpack("<I", struct.pack("<f", f))[0]
    except OverflowError:
        return NEG_INF if f < 0 else POS_INF


def is_nan(bits: int) -> bool:
    """Check if binary32 bit pattern represents any NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & MANT_MASK) != 0)


def is_inf(bits: int) -> bool:
    """Check if binary32 bit pattern represents infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & MANT_MASK) == 0)


def is_zero(bits: int) -> bool:
    """Check if binary32 bit pattern represents positive or negative zero."""
    return (bits & 0x7FFFFFFF) == 0


def is_subnormal(bits: int) -> bool:
    """Check if binary32 bit pattern represents a subnormal (denormal) float."""
    return ((bits & EXP_MASK) == 0) and ((bits & MANT_MASK) != 0)


def interp_v_ldexp_f32(
    src0_bits: int,
    src1_i32: int,
    *,
    flush_denorms: bool = False,
    clamp: bool = False,
    abs_src0: bool = False,
    neg_src0: bool = False,
) -> int:
    """Bit-accurate reference interpreter for RDNA2 VALU v_ldexp_f32.

    Computes: D = S0 * 2^S1
    where S0 is binary32 float bits and S1 is signed 32-bit integer exponent.

    Parameters:
      src0_bits: 32-bit binary32 float word.
      src1_i32: 32-bit integer exponent (evaluated as signed 32-bit two's complement).
      flush_denorms: If True (FTZ/DAZ mode), subnormal inputs flush to sign-preserved
                     zero before scaling, and subnormal results flush to sign-preserved zero.
                     If False (IEEE preserve mode), subnormals are scaled per IEEE 754.
      clamp: If True, result is clamped to [0.0, 1.0] (AMD clamp modifier).
             NaN clamps to 0.0, negative to 0.0, values > 1.0 to 1.0.
      abs_src0: If True, operand S0 is forced positive before scaling.
      neg_src0: If True, operand S0 sign bit is flipped before scaling.
    """
    src0_bits &= MASK32
    scale = s32(src1_i32)

    if abs_src0:
        src0_bits &= ~SIGN_MASK
    if neg_src0:
        src0_bits ^= SIGN_MASK

    sign = (src0_bits >> 31) & 1
    exp_field = (src0_bits >> 23) & 0xFF
    mant_field = src0_bits & MANT_MASK

    # 1. NaN operand handling
    if exp_field == 0xFF and mant_field != 0:
        if clamp:
            return POS_ZERO
        return QNAN

    # 2. Infinity operand handling
    if exp_field == 0xFF and mant_field == 0:
        if clamp:
            return POS_ZERO if sign else ONE
        return src0_bits

    # 3. Zero operand handling (+0.0 or -0.0)
    if exp_field == 0 and mant_field == 0:
        if clamp and sign:
            return POS_ZERO
        return src0_bits

    # 4. Denormal input handling
    if exp_field == 0:
        if flush_denorms:
            res = sign << 31
            return POS_ZERO if (clamp and sign) else res
        F = mant_field
        E = -126
        shift_left = 23 - (F.bit_length() - 1)
        F = F << shift_left
        E = E - shift_left
    else:
        E = exp_field - 127
        F = (1 << 23) | mant_field

    # 5. Add scale to unbiased exponent
    E_scaled = E + scale
    target_exp = E_scaled + 127

    # 6. Overflow detection (> max finite binary32 float)
    if target_exp >= 255:
        if clamp:
            return POS_ZERO if sign else ONE
        return (sign << 31) | POS_INF

    # 7. Normal float range
    if target_exp >= 1:
        res = (sign << 31) | (target_exp << 23) | (F & MANT_MASK)
        if clamp:
            if sign:
                return POS_ZERO
            if res > ONE:
                return ONE
        return res

    # 8. Underflow / Subnormal range
    if flush_denorms:
        res = sign << 31
        return POS_ZERO if (clamp and sign) else res

    shift_right = 1 - target_exp
    if shift_right >= 26:
        res = sign << 31
        return POS_ZERO if (clamp and sign) else res

    q = F >> shift_right
    rem = F & ((1 << shift_right) - 1)
    half = 1 << (shift_right - 1)

    # IEEE 754 round-to-nearest, ties-to-even
    if rem > half or (rem == half and (q & 1)):
        q += 1

    if q >= (1 << 23):
        res = (sign << 31) | (1 << 23)
    else:
        res = (sign << 31) | q

    if clamp:
        if sign:
            return POS_ZERO
        if res > ONE:
            return ONE
    return res


def ref_ldexp_f32_rational(
    src0_bits: int,
    src1_i32: int,
    *,
    flush_denorms: bool = False,
    clamp: bool = False,
) -> int:
    """Exact rational Fraction reference oracle for ldexp exponent scaling."""
    src0_bits &= MASK32
    scale = s32(src1_i32)
    sign = (src0_bits >> 31) & 1
    exp_field = (src0_bits >> 23) & 0xFF
    mant_field = src0_bits & MANT_MASK

    if exp_field == 0xFF:
        if mant_field != 0:
            return POS_ZERO if clamp else QNAN
        if clamp:
            return POS_ZERO if sign else ONE
        return src0_bits

    if exp_field == 0 and mant_field == 0:
        if clamp and sign:
            return POS_ZERO
        return src0_bits

    if exp_field == 0:
        if flush_denorms:
            res = sign << 31
            return POS_ZERO if (clamp and sign) else res
        val = Fraction(mant_field, 1 << 149)
    else:
        unbiased_e = exp_field - 127
        val = Fraction((1 << 23) | mant_field, 1 << 23) * (Fraction(2) ** unbiased_e)

    if scale > 300:
        if clamp:
            return POS_ZERO if sign else ONE
        return (sign << 31) | POS_INF
    elif scale < -300:
        res = sign << 31
        return POS_ZERO if (clamp and sign) else res

    val = val * (Fraction(2) ** scale)

    min_normal_frac = Fraction(1, 1 << 126)
    if flush_denorms and val < min_normal_frac:
        res = sign << 31
        return POS_ZERO if (clamp and sign) else res

    if val >= min_normal_frac:
        e = val.numerator.bit_length() - val.denominator.bit_length()
        while val < Fraction(2) ** e:
            e -= 1
        while val >= Fraction(2) ** (e + 1):
            e += 1
        if e > 127:
            if clamp:
                return POS_ZERO if sign else ONE
            return (sign << 31) | POS_INF
        scaled = val / (Fraction(2) ** (e - 23))
        q, r = divmod(scaled.numerator, scaled.denominator)
        if 2 * r > scaled.denominator or (2 * r == scaled.denominator and (q & 1)):
            q += 1
        if q >= (1 << 24):
            q >>= 1
            e += 1
        if e > 127:
            if clamp:
                return POS_ZERO if sign else ONE
            return (sign << 31) | POS_INF
        res = (sign << 31) | ((e + 127) << 23) | (q & MANT_MASK)
    else:
        scaled = val * (Fraction(2) ** 149)
        q, r = divmod(scaled.numerator, scaled.denominator)
        if 2 * r > scaled.denominator or (2 * r == scaled.denominator and (q & 1)):
            q += 1
        if q >= (1 << 23):
            res = (sign << 31) | (1 << 23)
        else:
            res = (sign << 31) | q

    if clamp:
        if sign:
            return POS_ZERO
        if res > ONE:
            return ONE
    return res


class TestLdexpF32NormalScaling(unittest.TestCase):
    """Verifies standard IEEE 754 exponent scaling for normal float numbers."""

    def test_identity_exponent_zero(self):
        vectors = [
            ONE,         # 1.0f
            TWO,         # 2.0f
            HALF,        # 0.5f
            0x40200000,  # 2.5f
            0x42280000,  # 42.0f
            0xBF800000,  # -1.0f
            0xC0000000,  # -2.0f
            0xBF000000,  # -0.5f
            MAX_FINITE,  # max normal
            MIN_NORMAL,  # min normal
        ]
        for src in vectors:
            with self.subTest(src=f"0x{src:08x}"):
                self.assertEqual(interp_v_ldexp_f32(src, 0), src)

    def test_power_of_two_scaling(self):
        for k in range(-126, 128):
            with self.subTest(scale=k):
                got = interp_v_ldexp_f32(ONE, k)
                if k == 127:
                    self.assertEqual(got, 0x7F000000)
                elif k == -126:
                    self.assertEqual(got, MIN_NORMAL)
                else:
                    expected_exp = 127 + k
                    self.assertEqual(got, expected_exp << 23)

    def test_multiply_by_two_and_half_equivalences(self):
        test_floats = [1.0, 1.5, 3.75, 10.0, 128.5, -4.25, -100.0]
        for val in test_floats:
            with self.subTest(val=val):
                b = f32_to_bits(val)
                # scale by +1 must equal val * 2.0
                times_two = interp_v_ldexp_f32(b, 1)
                self.assertEqual(times_two, f32_to_bits(val * 2.0))
                # scale by -1 must equal val * 0.5
                times_half = interp_v_ldexp_f32(b, -1)
                self.assertEqual(times_half, f32_to_bits(val * 0.5))

    def test_scale_composition_and_inverse(self):
        val = 0x41480000  # 12.5f
        for a, b in [(1, 2), (3, -2), (-5, 4), (10, -10), (15, -8)]:
            with self.subTest(a=a, b=b):
                step1 = interp_v_ldexp_f32(val, a)
                step2 = interp_v_ldexp_f32(step1, b)
                composed = interp_v_ldexp_f32(val, a + b)
                self.assertEqual(step2, composed)

        # Inverse identity: ldexp(ldexp(x, k), -k) == x
        for k in [1, 2, 5, 10, 20]:
            with self.subTest(k=k):
                scaled = interp_v_ldexp_f32(val, k)
                recovered = interp_v_ldexp_f32(scaled, -k)
                self.assertEqual(recovered, val)

    def test_cross_check_with_python_math_ldexp(self):
        samples = [
            (1.0, 3),
            (2.75, -2),
            (0.125, 4),
            (-3.5, 2),
            (-0.0625, 5),
            (12345.0, -3),
        ]
        for val, sc in samples:
            with self.subTest(val=val, sc=sc):
                b = f32_to_bits(val)
                got = interp_v_ldexp_f32(b, sc)
                expected_py = f32_to_bits(math.ldexp(val, sc))
                self.assertEqual(got, expected_py)


class TestLdexpF32DenormalPreserve(unittest.TestCase):
    """Verifies IEEE 754 gradual underflow and denormal preservation."""

    def test_scaling_into_subnormal_range(self):
        # MIN_NORMAL is 2^-126 (0x00800000)
        self.assertEqual(interp_v_ldexp_f32(MIN_NORMAL, -1, flush_denorms=False), 0x00400000)  # 2^-127
        self.assertEqual(interp_v_ldexp_f32(MIN_NORMAL, -2, flush_denorms=False), 0x00200000)  # 2^-128
        self.assertEqual(interp_v_ldexp_f32(MIN_NORMAL, -23, flush_denorms=False), MIN_SUBNORMAL)  # 2^-149

    def test_subnormal_round_half_even_tie_breaking(self):
        # 1.0f * 2^-150 = 0.5 * 2^-149: exact tie, q=0 is even -> rounds to 0
        self.assertEqual(interp_v_ldexp_f32(ONE, -150, flush_denorms=False), POS_ZERO)

        # 1.5f (0x3FC00000) * 2^-150 = 0.75 * 2^-149: > 0.5 -> rounds up to 1 (0x00000001)
        self.assertEqual(interp_v_ldexp_f32(0x3FC00000, -150, flush_denorms=False), MIN_SUBNORMAL)

        # 2.5f (0x40200000) * 2^-150 = 1.25 * 2^-149: fractional part 0.25 < 0.5 -> rounds to 1 (0x00000001)
        self.assertEqual(interp_v_ldexp_f32(0x40200000, -150, flush_denorms=False), MIN_SUBNORMAL)

        # 3.0f (0x40400000) * 2^-150 = 1.5 * 2^-149: exact tie with q=1 (odd) -> rounds to even q=2 (0x00000002)
        self.assertEqual(interp_v_ldexp_f32(0x40400000, -150, flush_denorms=False), 0x00000002)

    def test_subnormal_input_promoted_to_normal(self):
        # MIN_SUBNORMAL (2^-149) scaled up:
        # 2^-149 * 2^23 = 2^-126 = MIN_NORMAL
        self.assertEqual(interp_v_ldexp_f32(MIN_SUBNORMAL, 23, flush_denorms=False), MIN_NORMAL)
        # 2^-149 * 2^24 = 2^-125
        self.assertEqual(interp_v_ldexp_f32(MIN_SUBNORMAL, 24, flush_denorms=False), 0x01000000)
        # 2^-149 * 2^149 = 1.0f
        self.assertEqual(interp_v_ldexp_f32(MIN_SUBNORMAL, 149, flush_denorms=False), ONE)
        # 2^-149 * 2^150 = 2.0f
        self.assertEqual(interp_v_ldexp_f32(MIN_SUBNORMAL, 150, flush_denorms=False), TWO)
        # 2^-149 * 2^276 = 2^127
        self.assertEqual(interp_v_ldexp_f32(MIN_SUBNORMAL, 276, flush_denorms=False), 0x7F000000)
        # 2^-149 * 2^277 = 2^128 -> overflow to +inf
        self.assertEqual(interp_v_ldexp_f32(MIN_SUBNORMAL, 277, flush_denorms=False), POS_INF)

    def test_subnormal_sign_preservation(self):
        neg_min_sub = SIGN_MASK | MIN_SUBNORMAL
        self.assertEqual(interp_v_ldexp_f32(neg_min_sub, 0, flush_denorms=False), neg_min_sub)
        self.assertEqual(interp_v_ldexp_f32(neg_min_sub, 23, flush_denorms=False), SIGN_MASK | MIN_NORMAL)
        self.assertEqual(interp_v_ldexp_f32(neg_min_sub, 149, flush_denorms=False), 0xBF800000)  # -1.0f
        # Underflow below min subnormal retains negative zero
        self.assertEqual(interp_v_ldexp_f32(neg_min_sub, -10, flush_denorms=False), NEG_ZERO)


class TestLdexpF32DenormalFlushing(unittest.TestCase):
    """Verifies FTZ/DAZ denormal flushing mode."""

    def test_input_denormals_flushed_to_zero(self):
        denorm_inputs = [
            MIN_SUBNORMAL,
            MAX_SUBNORMAL,
            0x00000100,
            0x00012345,
            0x00400000,
            SIGN_MASK | MIN_SUBNORMAL,
            SIGN_MASK | MAX_SUBNORMAL,
        ]
        for src in denorm_inputs:
            sign = src & SIGN_MASK
            expected = NEG_ZERO if sign else POS_ZERO
            for sc in [-10, 0, 1, 23, 100]:
                with self.subTest(src=f"0x{src:08x}", sc=sc):
                    got = interp_v_ldexp_f32(src, sc, flush_denorms=True)
                    self.assertEqual(got, expected)

    def test_output_denormals_flushed_to_zero(self):
        # Scaling normal floats into subnormal range flushes to zero
        self.assertEqual(interp_v_ldexp_f32(ONE, -127, flush_denorms=True), POS_ZERO)
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, -127, flush_denorms=True), NEG_ZERO)  # -1.0f
        self.assertEqual(interp_v_ldexp_f32(MIN_NORMAL, -1, flush_denorms=True), POS_ZERO)
        self.assertEqual(interp_v_ldexp_f32(SIGN_MASK | MIN_NORMAL, -1, flush_denorms=True), NEG_ZERO)

    def test_flush_vs_preserve_contrast(self):
        # Exact same operands give subnormal in preserve mode but zero in flush mode
        cases = [
            (ONE, -127, 0x00400000),
            (0xBF800000, -127, SIGN_MASK | 0x00400000),
            (MIN_NORMAL, -1, 0x00400000),
            (ONE, -149, MIN_SUBNORMAL),
        ]
        for src, sc, preserved_res in cases:
            with self.subTest(src=f"0x{src:08x}", sc=sc):
                self.assertEqual(interp_v_ldexp_f32(src, sc, flush_denorms=False), preserved_res)
                expected_flushed = NEG_ZERO if (src & SIGN_MASK) else POS_ZERO
                self.assertEqual(interp_v_ldexp_f32(src, sc, flush_denorms=True), expected_flushed)


class TestLdexpF32Overflow(unittest.TestCase):
    """Verifies IEEE 754 overflow handling and positive exponent limits."""

    def test_max_finite_float_overflow(self):
        # MAX_FINITE (0x7F7FFFFF) scaled by 0 is MAX_FINITE
        self.assertEqual(interp_v_ldexp_f32(MAX_FINITE, 0), MAX_FINITE)
        # Scaled by +1 overflows to +inf
        self.assertEqual(interp_v_ldexp_f32(MAX_FINITE, 1), POS_INF)
        # Scaled by +2 overflows to +inf
        self.assertEqual(interp_v_ldexp_f32(MAX_FINITE, 2), POS_INF)

        # Negative max float (-MAX_FINITE = 0xFF7FFFFF)
        neg_max = SIGN_MASK | MAX_FINITE
        self.assertEqual(interp_v_ldexp_f32(neg_max, 0), neg_max)
        self.assertEqual(interp_v_ldexp_f32(neg_max, 1), NEG_INF)

    def test_exponents_exceeding_max_finite_range(self):
        # 1.0f * 2^127 is max normal power of 2 (0x7F000000)
        self.assertEqual(interp_v_ldexp_f32(ONE, 127), 0x7F000000)
        # 1.0f * 2^128 overflows to +inf
        self.assertEqual(interp_v_ldexp_f32(ONE, 128), POS_INF)
        self.assertEqual(interp_v_ldexp_f32(ONE, 129), POS_INF)
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, 128), NEG_INF)

    def test_extreme_positive_integers(self):
        extreme_exponents = [255, 256, 1000, 65536, 0x7FFFFFFF]
        for exp_val in extreme_exponents:
            with self.subTest(exp_val=exp_val):
                self.assertEqual(interp_v_ldexp_f32(ONE, exp_val), POS_INF)
                self.assertEqual(interp_v_ldexp_f32(0xBF800000, exp_val), NEG_INF)

    def test_clamped_overflow(self):
        # Under clamp modifier, overflow clamps to 1.0f (or 0.0f for negative)
        self.assertEqual(interp_v_ldexp_f32(ONE, 128, clamp=True), ONE)
        self.assertEqual(interp_v_ldexp_f32(ONE, 0x7FFFFFFF, clamp=True), ONE)
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, 128, clamp=True), POS_ZERO)
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, 0x7FFFFFFF, clamp=True), POS_ZERO)


class TestLdexpF32Underflow(unittest.TestCase):
    """Verifies IEEE 754 underflow handling and negative exponent limits."""

    def test_gradual_underflow_to_zero(self):
        # 1.0f * 2^-151 underflows to zero (both preserve and flush)
        self.assertEqual(interp_v_ldexp_f32(ONE, -151, flush_denorms=False), POS_ZERO)
        self.assertEqual(interp_v_ldexp_f32(ONE, -151, flush_denorms=True), POS_ZERO)
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, -151, flush_denorms=False), NEG_ZERO)
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, -151, flush_denorms=True), NEG_ZERO)

    def test_extreme_negative_integers(self):
        extreme_neg_exponents = [-255, -256, -1000, -65536, -0x80000000]
        for exp_val in extreme_neg_exponents:
            with self.subTest(exp_val=exp_val):
                self.assertEqual(interp_v_ldexp_f32(ONE, exp_val), POS_ZERO)
                self.assertEqual(interp_v_ldexp_f32(0xBF800000, exp_val), NEG_ZERO)

    def test_underflow_sign_preservation(self):
        # Negative float underflowing retains negative sign bit (NEG_ZERO)
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, -200), NEG_ZERO)
        self.assertEqual(interp_v_ldexp_f32(SIGN_MASK | MIN_NORMAL, -100), NEG_ZERO)


class TestLdexpF32SpecialValues(unittest.TestCase):
    """Verifies handling of special IEEE 754 inputs (+0, -0, +inf, -inf, NaN)."""

    def test_zeros(self):
        exps = [-1000, -1, 0, 1, 1000, 0x7FFFFFFF, -0x80000000]
        for sc in exps:
            with self.subTest(sc=sc):
                self.assertEqual(interp_v_ldexp_f32(POS_ZERO, sc), POS_ZERO)
                self.assertEqual(interp_v_ldexp_f32(NEG_ZERO, sc), NEG_ZERO)

    def test_infinities(self):
        exps = [-1000, -1, 0, 1, 1000, 0x7FFFFFFF, -0x80000000]
        for sc in exps:
            with self.subTest(sc=sc):
                self.assertEqual(interp_v_ldexp_f32(POS_INF, sc), POS_INF)
                self.assertEqual(interp_v_ldexp_f32(NEG_INF, sc), NEG_INF)

    def test_nans(self):
        nan_inputs = [QNAN, SNAN, 0x7F800042, SIGN_MASK | QNAN, SIGN_MASK | SNAN]
        for nan_val in nan_inputs:
            for sc in [-10, 0, 10]:
                with self.subTest(nan_val=f"0x{nan_val:08x}", sc=sc):
                    res = interp_v_ldexp_f32(nan_val, sc)
                    self.assertTrue(is_nan(res))
                    self.assertEqual(res, QNAN)

    def test_clamp_with_special_values(self):
        # NaN clamps to 0.0
        self.assertEqual(interp_v_ldexp_f32(QNAN, 0, clamp=True), POS_ZERO)
        self.assertEqual(interp_v_ldexp_f32(SNAN, 5, clamp=True), POS_ZERO)
        # +Inf clamps to 1.0f
        self.assertEqual(interp_v_ldexp_f32(POS_INF, 0, clamp=True), ONE)
        # -Inf clamps to 0.0f
        self.assertEqual(interp_v_ldexp_f32(NEG_INF, 0, clamp=True), POS_ZERO)
        # -0.0 clamps to +0.0
        self.assertEqual(interp_v_ldexp_f32(NEG_ZERO, 0, clamp=True), POS_ZERO)


class TestLdexpF32Modifiers(unittest.TestCase):
    """Verifies VALU input and output modifiers (abs, neg, clamp)."""

    def test_abs_modifier(self):
        # abs(|src0|) clears sign bit before scaling
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, 1, abs_src0=True), TWO)  # |-1.0| * 2 = 2.0
        self.assertEqual(interp_v_ldexp_f32(NEG_INF, 0, abs_src0=True), POS_INF)
        self.assertEqual(interp_v_ldexp_f32(NEG_ZERO, 0, abs_src0=True), POS_ZERO)

    def test_neg_modifier(self):
        # neg(-src0) flips sign bit before scaling
        self.assertEqual(interp_v_ldexp_f32(ONE, 1, neg_src0=True), 0xC0000000)  # -(1.0) * 2 = -2.0
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, 1, neg_src0=True), TWO)  # -(-1.0) * 2 = 2.0

    def test_clamp_modifier_normal_range(self):
        # Values in [0.0, 1.0] pass through unmodified
        self.assertEqual(interp_v_ldexp_f32(HALF, 0, clamp=True), HALF)
        # Values > 1.0 clamp to 1.0
        self.assertEqual(interp_v_ldexp_f32(ONE, 1, clamp=True), ONE)  # 2.0 -> 1.0
        self.assertEqual(interp_v_ldexp_f32(0x40200000, 0, clamp=True), ONE)  # 2.5 -> 1.0
        # Negative values clamp to 0.0
        self.assertEqual(interp_v_ldexp_f32(0xBF800000, 0, clamp=True), POS_ZERO)  # -1.0 -> 0.0


class TestLdexpF32DualOracleEquivalence(unittest.TestCase):
    """Cross-checks the bit-level VALU model against the exact rational Fraction oracle."""

    def test_exhaustive_oracle_equivalence_vector_suite(self):
        rng = random.Random(0x1337BEEF)

        # Diverse test bank including boundaries, subnormals, normal powers, and specials
        fixed_bases = [
            POS_ZERO, NEG_ZERO, POS_INF, NEG_INF, QNAN, SNAN,
            ONE, TWO, HALF, 0xBF800000, 0xC0000000, 0xBF000000,
            MAX_FINITE, SIGN_MASK | MAX_FINITE,
            MIN_NORMAL, SIGN_MASK | MIN_NORMAL,
            MAX_SUBNORMAL, SIGN_MASK | MAX_SUBNORMAL,
            MIN_SUBNORMAL, SIGN_MASK | MIN_SUBNORMAL,
            0x3FC00000,  # 1.5
            0x40400000,  # 3.0
            0x40200000,  # 2.5
        ]

        fixed_scales = [
            -0x80000000, -1000, -300, -151, -150, -149, -127, -126, -24, -23, -2, -1,
            0, 1, 2, 23, 24, 126, 127, 128, 149, 150, 276, 277, 300, 1000, 0x7FFFFFFF,
        ]

        for base in fixed_bases:
            for sc in fixed_scales:
                for flush in (False, True):
                    with self.subTest(base=f"0x{base:08x}", sc=sc, flush=flush):
                        r_bits = interp_v_ldexp_f32(base, sc, flush_denorms=flush)
                        r_frac = ref_ldexp_f32_rational(base, sc, flush_denorms=flush)
                        self.assertEqual(r_bits, r_frac)

        # 500 randomized cases
        for _ in range(500):
            base = rng.randint(0, MASK32)
            sc = rng.choice(fixed_scales + [rng.randint(-200, 200) for _ in range(5)])
            flush = rng.choice([False, True])
            clamp = rng.choice([False, True])
            with self.subTest(base=f"0x{base:08x}", sc=sc, flush=flush, clamp=clamp):
                r_bits = interp_v_ldexp_f32(base, sc, flush_denorms=flush, clamp=clamp)
                r_frac = ref_ldexp_f32_rational(base, sc, flush_denorms=flush, clamp=clamp)
                self.assertEqual(r_bits, r_frac)


class TestSpirvLdexpConstruct(unittest.TestCase):
    """Verifies interpreter rejection of unmodelled GLSL Ldexp extension opcode."""

    def test_spirv_mini_interp_rejects_raw_ldexp_extension(self):
        # GLSL.std.450 OpExtInst Ldexp is not directly modelled in spirv_mini_interp.py
        # and must be cleanly rejected with UnsupportedConstruct
        module = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%float = OpTypeFloat 32
%int = OpTypeInt 32 1
%_ptr_Output_float = OpTypePointer Output %float
%out = OpVariable %_ptr_Output_float Output
%f1 = OpConstant %float 1.0
%i2 = OpConstant %int 2
%main = OpFunction %void None %fn
%entry = OpLabel
%res = OpExtInst %float %glsl Ldexp %f1 %i2
OpStore %out %res
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct):
            si.run_fragment(module)


class TestNegativeControls(unittest.TestCase):
    """Negative controls ensuring mutated semantics fail test assertions."""

    def test_missing_overflow_fails(self):
        # Mutated interpreter that wraps or clamps to MAX_FINITE instead of emitting POS_INF
        def bad_overflow(src, sc):
            res = interp_v_ldexp_f32(src, sc)
            return MAX_FINITE if res == POS_INF else res

        self.assertNotEqual(bad_overflow(ONE, 128), POS_INF)

    def test_missing_denormal_flush_fails(self):
        # Mutated interpreter that forgets to flush subnormal output when flush_denorms=True
        def bad_flush(src, sc):
            return interp_v_ldexp_f32(src, sc, flush_denorms=False)

        # Scaling MIN_NORMAL by -1 in flush mode must be POS_ZERO, not 0x00400000
        self.assertNotEqual(bad_flush(MIN_NORMAL, -1), POS_ZERO)

    def test_missing_round_to_even_tie_breaking_fails(self):
        # Mutated interpreter that rounds 1.0f * 2^-150 up to 1 instead of down to 0
        def bad_round(src, sc):
            res = interp_v_ldexp_f32(src, sc, flush_denorms=False)
            return 1 if (src == ONE and sc == -150) else res

        self.assertNotEqual(bad_round(ONE, -150), POS_ZERO)

    def test_missing_negative_zero_sign_fails(self):
        # Mutated interpreter that produces POS_ZERO on underflow of negative numbers
        def bad_sign(src, sc):
            res = interp_v_ldexp_f32(src, sc)
            return POS_ZERO if res == NEG_ZERO else res

        self.assertNotEqual(bad_sign(0xBF800000, -200), NEG_ZERO)

    def test_treating_scale_as_unsigned_fails(self):
        # 0xFFFFFFFF passed as scale represents -1 in two's complement.
        # If treated as unsigned +4294967295, it would overflow instead of scaling by 0.5.
        got = interp_v_ldexp_f32(ONE, 0xFFFFFFFF)
        self.assertEqual(got, HALF)
        self.assertNotEqual(got, POS_INF)


if __name__ == "__main__":
    unittest.main()
