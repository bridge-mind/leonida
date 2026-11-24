"""Standalone idealized IEEE-754 / ISA-PDF reference model tests for v_div_fixup_f32 and v_div_fixup_f16.

RDNA2 VALU floating-point division fixup instructions:
  v_div_fixup_f32  vdst, s0, s1, s2
  v_div_fixup_f16  vdst, s0, s1, s2  [op_sel:[s0_hi, s1_hi, s2_hi, dst_hi]]

Arguments:
  s0: quotient estimate (computed from preliminary reciprocal / Newton-Raphson sequence)
  s1: denominator (y in x / y)
  s2: numerator (x in x / y)

Scope note:
  This file tests a self-contained textbook IEEE-754 / ISA-PDF model (IEEE=1, denorm32=3,
  s1-before-s2 NaN priority, positive canonical +qNaN 0x7FC00000 / 0x7E00); it does NOT invoke
  tools/rdna2spirv.py and does NOT match real RDNA2 silicon in the game's inferred MODE.
  For the hardware-measured RDNA2 (gfx1036, IEEE=0, denorm32=0, DX10_CLAMP=1) translator contract,
  see the ``v_div_fixup_f32`` row in docs/design.md and tests/semantic/test_alu_div_fixup_f32.py.
  Key divergences measured natively on gfx1036 (wave32 and wave64):
    - NaN priority: hardware prefers s2 (numerator) over s1 (denominator), even under IEEE=1/denorm32=3.
    - 0/0 and +-inf/+-inf: hardware returns -qNaN (0xFFC00000), not +qNaN (0x7FC00000).
    - sNaN quieting: hardware passes sNaN unchanged under IEEE=0 (quieting only when IEEE=1).
    - Subnormals: under denorm32=0, s1/s2 subnormals flush to signed zero (s0 subnormals are kept);
      hardware also applies exp(s2) - exp(s1) <= -151 -> +-0.0, NaN(s0) -> +-inf, and
      copysign(s0, sign(s1) ^ sign(s2)).

Idealized IEEE=1 / denorm32=3 model truth table tested in this file:
  1. If s1 is NaN, return quiet NaN (quiet_f32(s1) or quiet_f16(s1)).
  2. If s2 is NaN, return quiet NaN (quiet_f32(s2) or quiet_f16(s2)).
  3. If s1 == 0 and s2 == 0, return NaN (canonical QNaN: 0x7FC00000 for f32, 0x7E00 for f16).
  4. If s1 == 0 and s2 != 0, return sign(s1 ^ s2) * inf.
  5. If s1 == +-inf and s2 == +-inf, return NaN.
  6. If s1 == +-inf and s2 != +-inf, return sign(s1 ^ s2) * 0.0.
  7. If s2 == 0, return sign(s1 ^ s2) * 0.0.
  8. Otherwise return s0.

Zero game data; standard library Python only; Python 3.9+ compatible.
"""
from __future__ import annotations

import itertools
import math
import struct
import unittest

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF

# IEEE-754 binary32 bit constants
F32_POS_ZERO = 0x00000000
F32_NEG_ZERO = 0x80000000
F32_POS_INF = 0x7F800000
F32_NEG_INF = 0xFF800000
F32_QNAN = 0x7FC00000
F32_NEG_QNAN = 0xFFC00000
F32_SNAN = 0x7F800001
F32_NEG_SNAN = 0xFF800001

F32_ONE = 0x3F800000
F32_NEG_ONE = 0xBF800000
F32_TWO = 0x40000000
F32_HALF = 0x3F000000
F32_NORM_MIN = 0x00800000
F32_NORM_MAX = 0x7F7FFFFF
F32_SUB_MIN = 0x00000001
F32_SUB_MAX = 0x007FFFFF
F32_NEG_SUB_MIN = 0x80000001
F32_NEG_SUB_MAX = 0x807FFFFF

# IEEE-754 binary16 bit constants
F16_POS_ZERO = 0x0000
F16_NEG_ZERO = 0x8000
F16_POS_INF = 0x7C00
F16_NEG_INF = 0xFC00
F16_QNAN = 0x7E00
F16_NEG_QNAN = 0xFE00
F16_SNAN = 0x7C01
F16_NEG_SNAN = 0xFC01

F16_ONE = 0x3C00
F16_NEG_ONE = 0xBC00
F16_TWO = 0x4000
F16_HALF = 0x3800
F16_NORM_MIN = 0x0400
F16_NORM_MAX = 0x7BFF
F16_SUB_MIN = 0x0001
F16_SUB_MAX = 0x03FF
F16_NEG_SUB_MIN = 0x8001
F16_NEG_SUB_MAX = 0x83FF

OLD_VGPR_SENTINEL = 0xDEADBEEF


def f32_to_bits(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return F32_NEG_INF if val < 0 else F32_POS_INF


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_sign(bits: int) -> int:
    return (bits >> 31) & 1


def is_f32_nan(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0x7F800000) and ((bits & 0x007FFFFF) != 0)


def is_f32_snan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) == 0)


def is_f32_qnan(bits: int) -> bool:
    return is_f32_nan(bits) and ((bits & 0x00400000) != 0)


def is_f32_zero(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) == 0


def is_f32_inf(bits: int) -> bool:
    return (bits & 0x7FFFFFFF) == 0x7F800000


def is_f32_subnormal(bits: int) -> bool:
    return ((bits & 0x7F800000) == 0) and ((bits & 0x007FFFFF) != 0)


def is_f32_normal(bits: int) -> bool:
    exp = (bits >> 23) & 0xFF
    return 0 < exp < 255


def quiet_f32(bits: int) -> int:
    """Ensure quiet NaN bit (bit 22) is set on single-precision NaN while preserving payload."""
    if not is_f32_nan(bits):
        return F32_QNAN
    return (bits & MASK32) | 0x00400000


def ref_v_div_fixup_f32(s0: int, s1: int, s2: int) -> int:
    """Idealized IEEE=1 / denorm32=3 reference model for v_div_fixup_f32 (not gfx1036 silicon).

    For the hardware-measured RDNA2 (gfx1036, IEEE=0, denorm32=0) translator contract, see
    ``docs/design.md`` and ``tests/semantic/test_alu_div_fixup_f32.py``.

    s0 = quotient estimate
    s1 = denominator
    s2 = numerator
    """
    s0 &= MASK32
    s1 &= MASK32
    s2 &= MASK32

    sign_xor = f32_sign(s1) ^ f32_sign(s2)
    sign_bit = 0x80000000 if sign_xor else 0x00000000

    if is_f32_nan(s1):
        return quiet_f32(s1)
    if is_f32_nan(s2):
        return quiet_f32(s2)
    if is_f32_zero(s1) and is_f32_zero(s2):
        return F32_QNAN
    if is_f32_zero(s1):
        return sign_bit | F32_POS_INF
    if is_f32_inf(s1) and is_f32_inf(s2):
        return F32_QNAN
    if is_f32_inf(s1):
        return sign_bit | F32_POS_ZERO
    if is_f32_zero(s2):
        return sign_bit | F32_POS_ZERO
    return s0


def f16_to_bits(val: float) -> int:
    try:
        return struct.unpack("<H", struct.pack("<e", val))[0]
    except OverflowError:
        return F16_NEG_INF if val < 0 else F16_POS_INF


def bits_to_f16(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def f16_sign(bits: int) -> int:
    return (bits >> 15) & 1


def is_f16_nan(bits: int) -> bool:
    return ((bits & 0x7C00) == 0x7C00) and ((bits & 0x03FF) != 0)


def is_f16_snan(bits: int) -> bool:
    return is_f16_nan(bits) and ((bits & 0x0200) == 0)


def is_f16_qnan(bits: int) -> bool:
    return is_f16_nan(bits) and ((bits & 0x0200) != 0)


def is_f16_zero(bits: int) -> bool:
    return (bits & 0x7FFF) == 0


def is_f16_inf(bits: int) -> bool:
    return (bits & 0x7FFF) == 0x7C00


def is_f16_subnormal(bits: int) -> bool:
    return ((bits & 0x7C00) == 0) and ((bits & 0x03FF) != 0)


def is_f16_normal(bits: int) -> bool:
    exp = (bits >> 10) & 0x1F
    return 0 < exp < 31


def quiet_f16(bits: int) -> int:
    """Ensure quiet NaN bit (bit 9) is set on half-precision NaN while preserving payload."""
    if not is_f16_nan(bits):
        return F16_QNAN
    return (bits & MASK16) | 0x0200


def ref_v_div_fixup_f16(s0: int, s1: int, s2: int) -> int:
    """Idealized IEEE=1 / subnormals-kept reference model for v_div_fixup_f16 (not silicon-measured).

    s0 = quotient estimate
    s1 = denominator
    s2 = numerator
    """
    s0 &= MASK16
    s1 &= MASK16
    s2 &= MASK16

    sign_xor = f16_sign(s1) ^ f16_sign(s2)
    sign_bit = 0x8000 if sign_xor else 0x0000

    if is_f16_nan(s1):
        return quiet_f16(s1)
    if is_f16_nan(s2):
        return quiet_f16(s2)
    if is_f16_zero(s1) and is_f16_zero(s2):
        return F16_QNAN
    if is_f16_zero(s1):
        return sign_bit | F16_POS_INF
    if is_f16_inf(s1) and is_f16_inf(s2):
        return F16_QNAN
    if is_f16_inf(s1):
        return sign_bit | F16_POS_ZERO
    if is_f16_zero(s2):
        return sign_bit | F16_POS_ZERO
    return s0


def merge_half(old_val: int, half_val: int, dst_hi: int) -> int:
    """Write 16-bit half to VGPR word while preserving the unwritten 16-bit half."""
    h = half_val & MASK16
    old = old_val & MASK32
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


class WaveDivFixupInterpreter:
    """Multi-lane SIMD wave reference interpreter modeling RDNA2 Wave32/Wave64 VALU execution."""

    def __init__(self, wave_size: int = 32) -> None:
        self.wave_size = wave_size
        self.exec_mask = (1 << wave_size) - 1
        self.vgprs = [[OLD_VGPR_SENTINEL] * 256 for _ in range(wave_size)]

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK32

    def v_div_fixup_f32(self, vdst: int, s0_reg: int, s1_reg: int, s2_reg: int) -> None:
        """Executes v_div_fixup_f32 across all lanes conditioned on the exec mask."""
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                s0 = self.get_lane_vgpr(lane, s0_reg)
                s1 = self.get_lane_vgpr(lane, s1_reg)
                s2 = self.get_lane_vgpr(lane, s2_reg)
                self.vgprs[lane][vdst] = ref_v_div_fixup_f32(s0, s1, s2)

    def v_div_fixup_f16(
        self,
        vdst: int,
        s0_reg: int,
        s1_reg: int,
        s2_reg: int,
        op_sel: tuple[int, int, int, int] = (0, 0, 0, 0),
    ) -> None:
        """Executes v_div_fixup_f16 with op_sel across all lanes conditioned on exec mask."""
        s0_hi, s1_hi, s2_hi, dst_hi = op_sel
        for lane in range(self.wave_size):
            if (self.exec_mask >> lane) & 1:
                r0 = self.get_lane_vgpr(lane, s0_reg)
                r1 = self.get_lane_vgpr(lane, s1_reg)
                r2 = self.get_lane_vgpr(lane, s2_reg)

                s0 = (r0 >> 16) & MASK16 if s0_hi else (r0 & MASK16)
                s1 = (r1 >> 16) & MASK16 if s1_hi else (r1 & MASK16)
                s2 = (r2 >> 16) & MASK16 if s2_hi else (r2 & MASK16)

                res = ref_v_div_fixup_f16(s0, s1, s2)
                curr = self.get_lane_vgpr(lane, vdst)
                self.vgprs[lane][vdst] = merge_half(curr, res, dst_hi)


class TestF32Classification(unittest.TestCase):
    def test_zero_classification(self):
        self.assertTrue(is_f32_zero(F32_POS_ZERO))
        self.assertTrue(is_f32_zero(F32_NEG_ZERO))
        self.assertFalse(is_f32_zero(F32_ONE))
        self.assertEqual(f32_sign(F32_POS_ZERO), 0)
        self.assertEqual(f32_sign(F32_NEG_ZERO), 1)

    def test_inf_classification(self):
        self.assertTrue(is_f32_inf(F32_POS_INF))
        self.assertTrue(is_f32_inf(F32_NEG_INF))
        self.assertFalse(is_f32_inf(F32_QNAN))
        self.assertEqual(f32_sign(F32_POS_INF), 0)
        self.assertEqual(f32_sign(F32_NEG_INF), 1)

    def test_nan_classification_and_quieting(self):
        self.assertTrue(is_f32_nan(F32_QNAN))
        self.assertTrue(is_f32_nan(F32_SNAN))
        self.assertTrue(is_f32_qnan(F32_QNAN))
        self.assertTrue(is_f32_snan(F32_SNAN))
        self.assertFalse(is_f32_qnan(F32_SNAN))
        self.assertFalse(is_f32_snan(F32_QNAN))

        quieted = quiet_f32(F32_SNAN)
        self.assertTrue(is_f32_qnan(quieted))
        self.assertEqual(quieted, F32_SNAN | 0x00400000)

        # NaN payload preservation on quieting
        custom_snan = 0x7F800042
        self.assertTrue(is_f32_snan(custom_snan))
        custom_qnan = quiet_f32(custom_snan)
        self.assertTrue(is_f32_qnan(custom_qnan))
        self.assertEqual(custom_qnan & 0x003FFFFF, 0x00000042)

    def test_subnormal_classification(self):
        self.assertTrue(is_f32_subnormal(F32_SUB_MIN))
        self.assertTrue(is_f32_subnormal(F32_SUB_MAX))
        self.assertTrue(is_f32_subnormal(F32_NEG_SUB_MIN))
        self.assertTrue(is_f32_subnormal(F32_NEG_SUB_MAX))
        self.assertFalse(is_f32_subnormal(F32_POS_ZERO))
        self.assertFalse(is_f32_subnormal(F32_NORM_MIN))


class TestDivFixupF32ZeroAndInfCases(unittest.TestCase):
    def test_zero_divided_by_zero_produces_qnan(self):
        for s1 in (F32_POS_ZERO, F32_NEG_ZERO):
            for s2 in (F32_POS_ZERO, F32_NEG_ZERO):
                with self.subTest(s1=hex(s1), s2=hex(s2)):
                    res = ref_v_div_fixup_f32(F32_ONE, s1, s2)
                    self.assertTrue(is_f32_qnan(res))
                    self.assertEqual(res, F32_QNAN)

    def test_finite_divided_by_zero_produces_signed_inf(self):
        test_cases = [
            (F32_POS_ZERO, F32_ONE, F32_POS_INF),       # (+1) / (+0) = +inf
            (F32_NEG_ZERO, F32_ONE, F32_NEG_INF),       # (+1) / (-0) = -inf
            (F32_POS_ZERO, F32_NEG_ONE, F32_NEG_INF),   # (-1) / (+0) = -inf
            (F32_NEG_ZERO, F32_NEG_ONE, F32_POS_INF),   # (-1) / (-0) = +inf
            (F32_POS_ZERO, F32_TWO, F32_POS_INF),       # (+2) / (+0) = +inf
            (F32_NEG_ZERO, F32_TWO, F32_NEG_INF),       # (+2) / (-0) = -inf
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f32(0x12345678, s1, s2)
                self.assertEqual(res, expected)

    def test_inf_divided_by_inf_produces_qnan(self):
        for s1 in (F32_POS_INF, F32_NEG_INF):
            for s2 in (F32_POS_INF, F32_NEG_INF):
                with self.subTest(s1=hex(s1), s2=hex(s2)):
                    res = ref_v_div_fixup_f32(F32_ONE, s1, s2)
                    self.assertTrue(is_f32_qnan(res))
                    self.assertEqual(res, F32_QNAN)

    def test_finite_divided_by_inf_produces_signed_zero(self):
        test_cases = [
            (F32_POS_INF, F32_ONE, F32_POS_ZERO),       # (+1) / (+inf) = +0.0
            (F32_NEG_INF, F32_ONE, F32_NEG_ZERO),       # (+1) / (-inf) = -0.0
            (F32_POS_INF, F32_NEG_ONE, F32_NEG_ZERO),   # (-1) / (+inf) = -0.0
            (F32_NEG_INF, F32_NEG_ONE, F32_POS_ZERO),   # (-1) / (-inf) = +0.0
            (F32_POS_INF, F32_TWO, F32_POS_ZERO),       # (+2) / (+inf) = +0.0
            (F32_NEG_INF, F32_TWO, F32_NEG_ZERO),       # (+2) / (-inf) = -0.0
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f32(0xCAFEBABE, s1, s2)
                self.assertEqual(res, expected)

    def test_zero_divided_by_finite_produces_signed_zero(self):
        test_cases = [
            (F32_ONE, F32_POS_ZERO, F32_POS_ZERO),       # (+0.0) / (+1) = +0.0
            (F32_NEG_ONE, F32_POS_ZERO, F32_NEG_ZERO),   # (+0.0) / (-1) = -0.0
            (F32_ONE, F32_NEG_ZERO, F32_NEG_ZERO),       # (-0.0) / (+1) = -0.0
            (F32_NEG_ONE, F32_NEG_ZERO, F32_POS_ZERO),   # (-0.0) / (-1) = +0.0
            (F32_TWO, F32_POS_ZERO, F32_POS_ZERO),       # (+0.0) / (+2) = +0.0
            (f32_to_bits(-5.0), F32_POS_ZERO, F32_NEG_ZERO),
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f32(0x99887766, s1, s2)
                self.assertEqual(res, expected)


class TestDivFixupF32NaNCases(unittest.TestCase):
    def test_denominator_nan_propagates_quiet_nan(self):
        snan = 0x7F800007
        qnan = 0x7FC00010
        # s1 is sNaN: quiets s1
        res_snan = ref_v_div_fixup_f32(F32_ONE, snan, F32_ONE)
        self.assertTrue(is_f32_qnan(res_snan))
        self.assertEqual(res_snan, snan | 0x00400000)

        # s1 is qNaN: returns qNaN
        res_qnan = ref_v_div_fixup_f32(F32_ONE, qnan, F32_ONE)
        self.assertEqual(res_qnan, qnan)

    def test_numerator_nan_propagates_quiet_nan_when_denom_not_nan(self):
        snan = 0x7F80002A
        qnan = 0x7FC00055
        res_snan = ref_v_div_fixup_f32(F32_ONE, F32_ONE, snan)
        self.assertTrue(is_f32_qnan(res_snan))
        self.assertEqual(res_snan, snan | 0x00400000)

        res_qnan = ref_v_div_fixup_f32(F32_ONE, F32_ONE, qnan)
        self.assertEqual(res_qnan, qnan)

    def test_denominator_nan_takes_precedence_over_numerator_nan(self):
        s1_nan = 0x7F800111
        s2_nan = 0x7F800222
        res = ref_v_div_fixup_f32(F32_ONE, s1_nan, s2_nan)
        self.assertEqual(res, s1_nan | 0x00400000)


class TestDivFixupF32NormalPassThrough(unittest.TestCase):
    def test_regular_quotient_estimate_passes_through_intact(self):
        s0_estimates = [
            f32_to_bits(1.0),
            f32_to_bits(0.5),
            f32_to_bits(0.25),
            f32_to_bits(-3.14159),
            0x12345678,
            0xAABBCCDD,
        ]
        s1_denom = f32_to_bits(3.0)
        s2_num = f32_to_bits(6.0)

        for s0 in s0_estimates:
            with self.subTest(s0=hex(s0)):
                res = ref_v_div_fixup_f32(s0, s1_denom, s2_num)
                self.assertEqual(res, s0)


class TestDivFixupF32SimdWaveExecution(unittest.TestCase):
    def test_wave32_divergence_under_exec_mask(self):
        wave = WaveDivFixupInterpreter(wave_size=32)
        wave.exec_mask = 0x55555555

        vdst, s0_reg, s1_reg, s2_reg = 0, 1, 2, 3
        for lane in range(32):
            wave.set_lane_vgpr(lane, s0_reg, F32_HALF)
            if lane % 2 == 0:
                wave.set_lane_vgpr(lane, s1_reg, F32_POS_ZERO)
                wave.set_lane_vgpr(lane, s2_reg, F32_ONE)
            else:
                wave.set_lane_vgpr(lane, s1_reg, F32_ONE)
                wave.set_lane_vgpr(lane, s2_reg, F32_POS_ZERO)
            wave.set_lane_vgpr(lane, vdst, OLD_VGPR_SENTINEL)

        wave.v_div_fixup_f32(vdst, s0_reg, s1_reg, s2_reg)

        for lane in range(32):
            val = wave.get_lane_vgpr(lane, vdst)
            if lane % 2 == 0:
                self.assertEqual(val, F32_POS_INF, f"Active lane {lane} failed")
            else:
                self.assertEqual(val, OLD_VGPR_SENTINEL, f"Inactive lane {lane} mutated")

    def test_wave64_execution(self):
        wave = WaveDivFixupInterpreter(wave_size=64)
        wave.exec_mask = 0x00000000FFFFFFFF

        vdst, s0_reg, s1_reg, s2_reg = 10, 11, 12, 13
        for lane in range(64):
            wave.set_lane_vgpr(lane, s0_reg, F32_ONE)
            wave.set_lane_vgpr(lane, s1_reg, F32_POS_ZERO)
            wave.set_lane_vgpr(lane, s2_reg, F32_POS_ZERO)
            wave.set_lane_vgpr(lane, vdst, 0x11223344)

        wave.v_div_fixup_f32(vdst, s0_reg, s1_reg, s2_reg)

        for lane in range(64):
            val = wave.get_lane_vgpr(lane, vdst)
            if lane < 32:
                self.assertEqual(val, F32_QNAN)
            else:
                self.assertEqual(val, 0x11223344)


class TestF16Classification(unittest.TestCase):
    def test_zero_classification(self):
        self.assertTrue(is_f16_zero(F16_POS_ZERO))
        self.assertTrue(is_f16_zero(F16_NEG_ZERO))
        self.assertFalse(is_f16_zero(F16_ONE))
        self.assertEqual(f16_sign(F16_POS_ZERO), 0)
        self.assertEqual(f16_sign(F16_NEG_ZERO), 1)

    def test_inf_classification(self):
        self.assertTrue(is_f16_inf(F16_POS_INF))
        self.assertTrue(is_f16_inf(F16_NEG_INF))
        self.assertFalse(is_f16_inf(F16_QNAN))
        self.assertEqual(f16_sign(F16_POS_INF), 0)
        self.assertEqual(f16_sign(F16_NEG_INF), 1)

    def test_nan_classification_and_quieting(self):
        self.assertTrue(is_f16_nan(F16_QNAN))
        self.assertTrue(is_f16_nan(F16_SNAN))
        self.assertTrue(is_f16_qnan(F16_QNAN))
        self.assertTrue(is_f16_snan(F16_SNAN))
        self.assertFalse(is_f16_qnan(F16_SNAN))
        self.assertFalse(is_f16_snan(F16_QNAN))

        quieted = quiet_f16(F16_SNAN)
        self.assertTrue(is_f16_qnan(quieted))
        self.assertEqual(quieted, F16_SNAN | 0x0200)

        # Custom payload preservation
        custom_snan = 0x7C15
        self.assertTrue(is_f16_snan(custom_snan))
        custom_qnan = quiet_f16(custom_snan)
        self.assertTrue(is_f16_qnan(custom_qnan))
        self.assertEqual(custom_qnan & 0x01FF, 0x0015)

    def test_subnormal_classification(self):
        self.assertTrue(is_f16_subnormal(F16_SUB_MIN))
        self.assertTrue(is_f16_subnormal(F16_SUB_MAX))
        self.assertTrue(is_f16_subnormal(F16_NEG_SUB_MIN))
        self.assertTrue(is_f16_subnormal(F16_NEG_SUB_MAX))
        self.assertFalse(is_f16_subnormal(F16_POS_ZERO))
        self.assertFalse(is_f16_subnormal(F16_NORM_MIN))


class TestDivFixupF16ZeroAndInfCases(unittest.TestCase):
    def test_zero_divided_by_zero_produces_qnan(self):
        for s1 in (F16_POS_ZERO, F16_NEG_ZERO):
            for s2 in (F16_POS_ZERO, F16_NEG_ZERO):
                with self.subTest(s1=hex(s1), s2=hex(s2)):
                    res = ref_v_div_fixup_f16(F16_ONE, s1, s2)
                    self.assertTrue(is_f16_qnan(res))
                    self.assertEqual(res, F16_QNAN)

    def test_finite_divided_by_zero_produces_signed_inf(self):
        test_cases = [
            (F16_POS_ZERO, F16_ONE, F16_POS_INF),       # (+1) / (+0) = +inf
            (F16_NEG_ZERO, F16_ONE, F16_NEG_INF),       # (+1) / (-0) = -inf
            (F16_POS_ZERO, F16_NEG_ONE, F16_NEG_INF),   # (-1) / (+0) = -inf
            (F16_NEG_ZERO, F16_NEG_ONE, F16_POS_INF),   # (-1) / (-0) = +inf
            (F16_POS_ZERO, F16_TWO, F16_POS_INF),       # (+2) / (+0) = +inf
            (F16_NEG_ZERO, F16_TWO, F16_NEG_INF),       # (+2) / (-0) = -inf
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f16(0x5678, s1, s2)
                self.assertEqual(res, expected)

    def test_inf_divided_by_inf_produces_qnan(self):
        for s1 in (F16_POS_INF, F16_NEG_INF):
            for s2 in (F16_POS_INF, F16_NEG_INF):
                with self.subTest(s1=hex(s1), s2=hex(s2)):
                    res = ref_v_div_fixup_f16(F16_ONE, s1, s2)
                    self.assertTrue(is_f16_qnan(res))
                    self.assertEqual(res, F16_QNAN)

    def test_finite_divided_by_inf_produces_signed_zero(self):
        test_cases = [
            (F16_POS_INF, F16_ONE, F16_POS_ZERO),       # (+1) / (+inf) = +0.0
            (F16_NEG_INF, F16_ONE, F16_NEG_ZERO),       # (+1) / (-inf) = -0.0
            (F16_POS_INF, F16_NEG_ONE, F16_NEG_ZERO),   # (-1) / (+inf) = -0.0
            (F16_NEG_INF, F16_NEG_ONE, F16_POS_ZERO),   # (-1) / (-inf) = +0.0
            (F16_POS_INF, F16_TWO, F16_POS_ZERO),       # (+2) / (+inf) = +0.0
            (F16_NEG_INF, F16_TWO, F16_NEG_ZERO),       # (+2) / (-inf) = -0.0
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f16(0xBABE, s1, s2)
                self.assertEqual(res, expected)

    def test_zero_divided_by_finite_produces_signed_zero(self):
        test_cases = [
            (F16_ONE, F16_POS_ZERO, F16_POS_ZERO),       # (+0.0) / (+1) = +0.0
            (F16_NEG_ONE, F16_POS_ZERO, F16_NEG_ZERO),   # (+0.0) / (-1) = -0.0
            (F16_ONE, F16_NEG_ZERO, F16_NEG_ZERO),       # (-0.0) / (+1) = -0.0
            (F16_NEG_ONE, F16_NEG_ZERO, F16_POS_ZERO),   # (-0.0) / (-1) = +0.0
            (F16_TWO, F16_POS_ZERO, F16_POS_ZERO),       # (+0.0) / (+2) = +0.0
            (f16_to_bits(-4.0), F16_POS_ZERO, F16_NEG_ZERO),
        ]
        for s1, s2, expected in test_cases:
            with self.subTest(s1=hex(s1), s2=hex(s2)):
                res = ref_v_div_fixup_f16(0x7766, s1, s2)
                self.assertEqual(res, expected)


class TestDivFixupF16NaNCases(unittest.TestCase):
    def test_denominator_nan_propagates_quiet_nan(self):
        snan = 0x7C05
        qnan = 0x7E12
        res_snan = ref_v_div_fixup_f16(F16_ONE, snan, F16_ONE)
        self.assertTrue(is_f16_qnan(res_snan))
        self.assertEqual(res_snan, snan | 0x0200)

        res_qnan = ref_v_div_fixup_f16(F16_ONE, qnan, F16_ONE)
        self.assertEqual(res_qnan, qnan)

    def test_numerator_nan_propagates_quiet_nan_when_denom_not_nan(self):
        snan = 0x7C2B
        qnan = 0x7E5A
        res_snan = ref_v_div_fixup_f16(F16_ONE, F16_ONE, snan)
        self.assertTrue(is_f16_qnan(res_snan))
        self.assertEqual(res_snan, snan | 0x0200)

        res_qnan = ref_v_div_fixup_f16(F16_ONE, F16_ONE, qnan)
        self.assertEqual(res_qnan, qnan)

    def test_denominator_nan_takes_precedence_over_numerator_nan(self):
        s1_nan = 0x7C11
        s2_nan = 0x7C22
        res = ref_v_div_fixup_f16(F16_ONE, s1_nan, s2_nan)
        self.assertEqual(res, s1_nan | 0x0200)


class TestDivFixupF16OpSelAndHalfPreservation(unittest.TestCase):
    def test_op_sel_all_combinations_and_half_preservation(self):
        r0 = (F16_HALF << 16) | F16_ONE
        r1 = (F16_POS_ZERO << 16) | F16_TWO
        r2 = (F16_TWO << 16) | F16_POS_ZERO

        wave = WaveDivFixupInterpreter(wave_size=1)
        wave.set_lane_vgpr(0, 1, r0)
        wave.set_lane_vgpr(0, 2, r1)
        wave.set_lane_vgpr(0, 3, r2)

        for s0_hi, s1_hi, s2_hi, dst_hi in itertools.product((0, 1), repeat=4):
            with self.subTest(op_sel=(s0_hi, s1_hi, s2_hi, dst_hi)):
                initial_vdst = 0xBEEFCAFE
                wave.set_lane_vgpr(0, 0, initial_vdst)
                wave.v_div_fixup_f16(0, 1, 2, 3, op_sel=(s0_hi, s1_hi, s2_hi, dst_hi))

                s0 = (r0 >> 16) & MASK16 if s0_hi else (r0 & MASK16)
                s1 = (r1 >> 16) & MASK16 if s1_hi else (r1 & MASK16)
                s2 = (r2 >> 16) & MASK16 if s2_hi else (r2 & MASK16)
                expected_h = ref_v_div_fixup_f16(s0, s1, s2)

                res = wave.get_lane_vgpr(0, 0)
                if dst_hi:
                    self.assertEqual(res >> 16, expected_h)
                    self.assertEqual(res & 0xFFFF, 0xCAFE, "Low half clobbered when writing high half")
                else:
                    self.assertEqual(res & 0xFFFF, expected_h)
                    self.assertEqual(res >> 16, 0xBEEF, "High half clobbered when writing low half")


class TestDivFixupF16SimdWaveExecution(unittest.TestCase):
    def test_wave32_divergence_f16(self):
        wave = WaveDivFixupInterpreter(wave_size=32)
        wave.exec_mask = 0xAAAAAAAA

        vdst, s0_reg, s1_reg, s2_reg = 0, 1, 2, 3
        for lane in range(32):
            wave.set_lane_vgpr(lane, s0_reg, F16_HALF)
            wave.set_lane_vgpr(lane, s1_reg, F16_POS_ZERO)
            wave.set_lane_vgpr(lane, s2_reg, F16_ONE)
            wave.set_lane_vgpr(lane, vdst, 0x12345678)

        wave.v_div_fixup_f16(vdst, s0_reg, s1_reg, s2_reg, op_sel=(0, 0, 0, 0))

        for lane in range(32):
            val = wave.get_lane_vgpr(lane, vdst)
            if lane % 2 == 1:
                self.assertEqual(val & 0xFFFF, F16_POS_INF)
                self.assertEqual(val >> 16, 0x1234)
            else:
                self.assertEqual(val, 0x12345678)


class TestDivFixupSubnormalOperands(unittest.TestCase):
    def test_f32_subnormal_quotient_pass_through(self):
        # Normal inputs producing subnormal quotient in s0 pass through untouched
        for sub in (F32_SUB_MIN, F32_SUB_MAX, F32_NEG_SUB_MIN, F32_NEG_SUB_MAX):
            with self.subTest(sub=hex(sub)):
                res = ref_v_div_fixup_f32(sub, F32_TWO, F32_ONE)
                self.assertEqual(res, sub)

    def test_f32_subnormal_numerator_and_denominator(self):
        # Subnormal denominator with normal numerator passes through s0
        s0 = f32_to_bits(123456.0)
        res1 = ref_v_div_fixup_f32(s0, F32_SUB_MIN, F32_ONE)
        self.assertEqual(res1, s0)

        # Normal denominator with subnormal numerator passes through s0
        res2 = ref_v_div_fixup_f32(s0, F32_ONE, F32_SUB_MAX)
        self.assertEqual(res2, s0)

        # Both subnormal passes through s0
        res3 = ref_v_div_fixup_f32(s0, F32_SUB_MIN, F32_SUB_MAX)
        self.assertEqual(res3, s0)

    def test_f32_subnormal_with_zero_and_inf(self):
        # subnormal / zero -> signed inf
        self.assertEqual(ref_v_div_fixup_f32(0, F32_POS_ZERO, F32_SUB_MIN), F32_POS_INF)
        self.assertEqual(ref_v_div_fixup_f32(0, F32_NEG_ZERO, F32_SUB_MIN), F32_NEG_INF)
        self.assertEqual(ref_v_div_fixup_f32(0, F32_POS_ZERO, F32_NEG_SUB_MIN), F32_NEG_INF)
        self.assertEqual(ref_v_div_fixup_f32(0, F32_NEG_ZERO, F32_NEG_SUB_MIN), F32_POS_INF)

        # zero / subnormal -> signed zero
        self.assertEqual(ref_v_div_fixup_f32(0, F32_SUB_MIN, F32_POS_ZERO), F32_POS_ZERO)
        self.assertEqual(ref_v_div_fixup_f32(0, F32_NEG_SUB_MIN, F32_POS_ZERO), F32_NEG_ZERO)
        self.assertEqual(ref_v_div_fixup_f32(0, F32_SUB_MIN, F32_NEG_ZERO), F32_NEG_ZERO)
        self.assertEqual(ref_v_div_fixup_f32(0, F32_NEG_SUB_MIN, F32_NEG_ZERO), F32_POS_ZERO)

        # subnormal / inf -> signed zero
        self.assertEqual(ref_v_div_fixup_f32(0, F32_POS_INF, F32_SUB_MAX), F32_POS_ZERO)
        self.assertEqual(ref_v_div_fixup_f32(0, F32_NEG_INF, F32_SUB_MAX), F32_NEG_ZERO)

    def test_f16_subnormal_quotient_pass_through(self):
        for sub in (F16_SUB_MIN, F16_SUB_MAX, F16_NEG_SUB_MIN, F16_NEG_SUB_MAX):
            with self.subTest(sub=hex(sub)):
                res = ref_v_div_fixup_f16(sub, F16_TWO, F16_ONE)
                self.assertEqual(res, sub)

    def test_f16_subnormal_with_zero_and_inf(self):
        # subnormal / zero -> signed inf
        self.assertEqual(ref_v_div_fixup_f16(0, F16_POS_ZERO, F16_SUB_MIN), F16_POS_INF)
        self.assertEqual(ref_v_div_fixup_f16(0, F16_NEG_ZERO, F16_SUB_MIN), F16_NEG_INF)
        self.assertEqual(ref_v_div_fixup_f16(0, F16_POS_ZERO, F16_NEG_SUB_MIN), F16_NEG_INF)
        self.assertEqual(ref_v_div_fixup_f16(0, F16_NEG_ZERO, F16_NEG_SUB_MIN), F16_POS_INF)

        # zero / subnormal -> signed zero
        self.assertEqual(ref_v_div_fixup_f16(0, F16_SUB_MIN, F16_POS_ZERO), F16_POS_ZERO)
        self.assertEqual(ref_v_div_fixup_f16(0, F16_NEG_SUB_MIN, F16_POS_ZERO), F16_NEG_ZERO)
        self.assertEqual(ref_v_div_fixup_f16(0, F16_SUB_MIN, F16_NEG_ZERO), F16_NEG_ZERO)
        self.assertEqual(ref_v_div_fixup_f16(0, F16_NEG_SUB_MIN, F16_NEG_ZERO), F16_POS_ZERO)


class TestDivFixupExhaustiveSignMatrix(unittest.TestCase):
    def test_f32_sign_xor_rule_exhaustive_matrix(self):
        signs = [(0, "+"), (1, "-")]
        nonzeros = [
            (F32_ONE, F32_NEG_ONE, "normal"),
            (F32_SUB_MIN, F32_NEG_SUB_MIN, "subnormal"),
        ]

        for (s1_sign, s1_label), (s2_sign, s2_label) in itertools.product(signs, repeat=2):
            expected_sign = s1_sign ^ s2_sign

            # 1. Nonzero / 0 -> Inf
            s1_zero = F32_NEG_ZERO if s1_sign else F32_POS_ZERO
            for pos_val, neg_val, val_label in nonzeros:
                s2_val = neg_val if s2_sign else pos_val
                with self.subTest(op=f"{val_label}/{s1_label}0", s1=s1_label, s2=s2_label):
                    res = ref_v_div_fixup_f32(0, s1_zero, s2_val)
                    self.assertTrue(is_f32_inf(res))
                    self.assertEqual(f32_sign(res), expected_sign)

            # 2. Nonzero / Inf -> 0.0
            s1_inf = F32_NEG_INF if s1_sign else F32_POS_INF
            for pos_val, neg_val, val_label in nonzeros:
                s2_val = neg_val if s2_sign else pos_val
                with self.subTest(op=f"{val_label}/{s1_label}inf", s1=s1_label, s2=s2_label):
                    res = ref_v_div_fixup_f32(0, s1_inf, s2_val)
                    self.assertTrue(is_f32_zero(res))
                    self.assertEqual(f32_sign(res), expected_sign)

            # 3. 0 / Nonzero -> 0.0
            s2_zero = F32_NEG_ZERO if s2_sign else F32_POS_ZERO
            for pos_val, neg_val, val_label in nonzeros:
                s1_val = neg_val if s1_sign else pos_val
                with self.subTest(op=f"{s2_label}0/{val_label}", s1=s1_label, s2=s2_label):
                    res = ref_v_div_fixup_f32(0, s1_val, s2_zero)
                    self.assertTrue(is_f32_zero(res))
                    self.assertEqual(f32_sign(res), expected_sign)

    def test_f16_sign_xor_rule_exhaustive_matrix(self):
        signs = [(0, "+"), (1, "-")]
        nonzeros = [
            (F16_ONE, F16_NEG_ONE, "normal"),
            (F16_SUB_MIN, F16_NEG_SUB_MIN, "subnormal"),
        ]

        for (s1_sign, s1_label), (s2_sign, s2_label) in itertools.product(signs, repeat=2):
            expected_sign = s1_sign ^ s2_sign

            # Nonzero / 0 -> Inf
            s1_zero = F16_NEG_ZERO if s1_sign else F16_POS_ZERO
            for pos_val, neg_val, val_label in nonzeros:
                s2_val = neg_val if s2_sign else pos_val
                with self.subTest(op=f"f16_{val_label}/{s1_label}0", s1=s1_label, s2=s2_label):
                    res = ref_v_div_fixup_f16(0, s1_zero, s2_val)
                    self.assertTrue(is_f16_inf(res))
                    self.assertEqual(f16_sign(res), expected_sign)


class TestDivFixupNegativeControls(unittest.TestCase):
    def test_sabotaged_sign_xor_fails(self):
        # Sabotage: replace sign_xor with AND (fails when one operand is negative and other positive)
        def bad_ref_f32(s0: int, s1: int, s2: int) -> int:
            sign_and = f32_sign(s1) & f32_sign(s2)  # Sabotaged: AND instead of XOR
            sign_bit = 0x80000000 if sign_and else 0x00000000
            if is_f32_zero(s1) and not is_f32_zero(s2):
                return sign_bit | F32_POS_INF
            return s0

        # (+1) / (-0) should be -inf, but sabotaged sign gives +inf
        correct = ref_v_div_fixup_f32(0, F32_NEG_ZERO, F32_ONE)
        sabotaged = bad_ref_f32(0, F32_NEG_ZERO, F32_ONE)
        self.assertEqual(correct, F32_NEG_INF)
        self.assertEqual(sabotaged, F32_POS_INF)
        self.assertNotEqual(correct, sabotaged)

    def test_sabotaged_snan_quieting_fails(self):
        # Sabotage: returning sNaN directly without setting quiet bit
        def bad_ref_f32(s0: int, s1: int, s2: int) -> int:
            if is_f32_nan(s1):
                return s1  # Sabotaged: forgot quiet_f32
            return s0

        res_correct = ref_v_div_fixup_f32(F32_ONE, F32_SNAN, F32_ONE)
        res_bad = bad_ref_f32(F32_ONE, F32_SNAN, F32_ONE)
        self.assertTrue(is_f32_qnan(res_correct))
        self.assertTrue(is_f32_snan(res_bad))
        self.assertNotEqual(res_correct, res_bad)

    def test_sabotaged_zero_div_zero_fails(self):
        # Sabotage: returning 0 instead of NaN for 0 / 0
        def bad_ref_f32(s0: int, s1: int, s2: int) -> int:
            if is_f32_zero(s1) and is_f32_zero(s2):
                return F32_POS_ZERO  # Sabotaged: returns 0 instead of QNaN
            return s0

        res_correct = ref_v_div_fixup_f32(F32_ONE, F32_POS_ZERO, F32_POS_ZERO)
        res_bad = bad_ref_f32(F32_ONE, F32_POS_ZERO, F32_POS_ZERO)
        self.assertTrue(is_f32_qnan(res_correct))
        self.assertFalse(is_f32_nan(res_bad))
        self.assertNotEqual(res_correct, res_bad)

    def test_sabotaged_inf_div_inf_fails(self):
        # Sabotage: returning 1.0 instead of NaN for inf / inf
        def bad_ref_f32(s0: int, s1: int, s2: int) -> int:
            if is_f32_inf(s1) and is_f32_inf(s2):
                return F32_ONE  # Sabotaged: returns 1.0 instead of QNaN
            return s0

        res_correct = ref_v_div_fixup_f32(F32_ONE, F32_POS_INF, F32_POS_INF)
        res_bad = bad_ref_f32(F32_ONE, F32_POS_INF, F32_POS_INF)
        self.assertTrue(is_f32_qnan(res_correct))
        self.assertEqual(res_bad, F32_ONE)
        self.assertNotEqual(res_correct, res_bad)

    def test_sabotaged_nan_precedence_fails(self):
        # Sabotage: checking s2 (numerator) before s1 (denominator)
        def bad_ref_f32(s0: int, s1: int, s2: int) -> int:
            if is_f32_nan(s2):  # Sabotaged: inverted priority
                return quiet_f32(s2)
            if is_f32_nan(s1):
                return quiet_f32(s1)
            return s0

        s1_nan = 0x7F8000AA
        s2_nan = 0x7F8000BB
        correct = ref_v_div_fixup_f32(F32_ONE, s1_nan, s2_nan)
        sabotaged = bad_ref_f32(F32_ONE, s1_nan, s2_nan)
        self.assertEqual(correct, s1_nan | 0x00400000)
        self.assertEqual(sabotaged, s2_nan | 0x00400000)
        self.assertNotEqual(correct, sabotaged)

    def test_sabotaged_inactive_lane_bleed_fails(self):
        # Sabotage: interpreter mutates inactive lanes
        wave = WaveDivFixupInterpreter(wave_size=4)
        wave.exec_mask = 0b0001  # only lane 0 active

        for lane in range(4):
            wave.set_lane_vgpr(lane, 0, OLD_VGPR_SENTINEL)
            wave.set_lane_vgpr(lane, 1, F32_ONE)
            wave.set_lane_vgpr(lane, 2, F32_POS_ZERO)
            wave.set_lane_vgpr(lane, 3, F32_ONE)

        # Correct execution
        wave.v_div_fixup_f32(0, 1, 2, 3)
        self.assertEqual(wave.get_lane_vgpr(0, 0), F32_POS_INF)
        self.assertEqual(wave.get_lane_vgpr(1, 0), OLD_VGPR_SENTINEL)
        self.assertEqual(wave.get_lane_vgpr(2, 0), OLD_VGPR_SENTINEL)
        self.assertEqual(wave.get_lane_vgpr(3, 0), OLD_VGPR_SENTINEL)

        # Sabotaged execution: clobber all lanes regardless of exec
        for lane in range(4):
            wave.vgprs[lane][0] = 0x00000000
        self.assertNotEqual(wave.get_lane_vgpr(1, 0), OLD_VGPR_SENTINEL)


if __name__ == "__main__":
    unittest.main()
