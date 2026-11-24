from __future__ import annotations
from fractions import Fraction
import struct
import unittest

import support

M32 = 0xFFFFFFFF


def ref_float_to_int(u: int) -> tuple[int, int]:
    u &= M32
    sign = (u >> 31) & 1
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF
    if exp == 0xFF:
        if mant != 0:
            return 0, 0
        if sign:
            return 0, 0x80000000
        return 0xFFFFFFFF, 0x7FFFFFFF
    if exp == 0:
        x = Fraction(0)
    else:
        x = Fraction((1 << 23) | mant) * (Fraction(2) ** (exp - 150))
    if sign:
        x = -x
    iv = int(x)
    if iv >= (1 << 32):
        u_res = 0xFFFFFFFF
    elif iv < 0:
        u_res = 0
    else:
        u_res = iv & M32
    if iv >= (1 << 31):
        s_res = 0x7FFFFFFF
    elif iv < -(1 << 31):
        s_res = 0x80000000
    else:
        s_res = iv & M32
    return u_res, s_res


CASES = (
    (0x00000000, 0, 0),
    (0x80000000, 0, 0),
    (0x00000001, 0, 0),
    (0x3FC00000, 1, 1),
    (0xBFC00000, 0, 0xFFFFFFFF),
    (0x4EFFFFFF, 0x7FFFFF80, 0x7FFFFF80),
    (0x4F000000, 0x80000000, 0x7FFFFFFF),
    (0x4F7FFFFF, 0xFFFFFF00, 0x7FFFFFFF),
    (0x4F800000, 0xFFFFFFFF, 0x7FFFFFFF),
    (0xCEFFFFFF, 0, 0x80000080),
    (0xCF000000, 0, 0x80000000),
    (0xCF000001, 0, 0x80000000),
    (0x7F800000, 0xFFFFFFFF, 0x7FFFFFFF),
    (0xFF800000, 0, 0x80000000),
    (0x7F800001, 0, 0),
    (0xFFC00000, 0, 0),
)


class FloatToIntTest(unittest.TestCase):
    def test_reference_oracle(self):
        for bits, expected_u, expected_s in CASES:
            u_res, s_res = ref_float_to_int(bits)
            self.assertEqual(u_res, expected_u)
            self.assertEqual(s_res, expected_s)

    def test_multi_lane_divergence_and_rounding(self):
        rounding_cases = [
            (0x3F800000, 1, 1),
            (0x3FA00000, 1, 1),
            (0x3FE00000, 1, 1),
            (0xBF800000, 0, 0xFFFFFFFF),
            (0xBFA00000, 0, 0xFFFFFFFF),
            (0xBFE00000, 0, 0xFFFFFFFF),
        ]
        fssm = bytes(0x138) + struct.pack("<4I", 0x808A, 1, 1, 1)
        spv = support.translate_fixture("cs_cvt_f32_sat", fssm=fssm)
        for bits, unsigned, signed in rounding_cases:
            memory = {}
            support.run_fragment(
                spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                push_constants=[0, 0, 8, 0x31016FAC, bits] + [0] * 27, memory=memory)
            self.assertEqual(memory[0], unsigned)
            self.assertEqual(memory[4], signed)

    def test_nan_inf_saturation_and_negative_controls(self):
        special_cases = [
            (0x7F800000, 0xFFFFFFFF, 0x7FFFFFFF), # +inf
            (0xFF800000, 0, 0x80000000),          # -inf
            (0x7F800001, 0, 0),                   # +NaN
            (0xFFC00000, 0, 0),                   # -NaN
            (0x80000000, 0, 0),                   # -0 negative control
            (0x00000001, 0, 0),                   # subnormal negative control
        ]
        fssm = bytes(0x138) + struct.pack("<4I", 0x808A, 1, 1, 1)
        spv = support.translate_fixture("cs_cvt_f32_sat", fssm=fssm)
        for bits, unsigned, signed in special_cases:
            memory = {}
            support.run_fragment(
                spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                push_constants=[0, 0, 8, 0x31016FAC, bits] + [0] * 27, memory=memory)
            self.assertEqual(memory[0], unsigned)
            self.assertEqual(memory[4], signed)

    def test_saturating_conversions(self):
        fssm = bytes(0x138) + struct.pack("<4I", 0x808A, 1, 1, 1)
        spv = support.translate_fixture("cs_cvt_f32_sat", fssm=fssm)
        for bits, unsigned, signed in CASES:
            with self.subTest(bits=f"{bits:08x}"):
                memory = {}
                support.run_fragment(
                    spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                    push_constants=[0, 0, 8, 0x31016FAC, bits] + [0] * 27, memory=memory)
                self.assertEqual(memory, {0: unsigned, 4: signed})


if __name__ == "__main__":
    unittest.main()
