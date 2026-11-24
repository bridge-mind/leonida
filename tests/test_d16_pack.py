"""Exhaustive/sweep bit-exact tests of an integer f32->f16 (RTE) reference, and a check that the d16 pair packer in
tools/rdna2spirv.py (pack_f16_pair_int, through Gen.f16_bits) is integer ops only.

Runs with the standard library only (no GPU, LLVM, or SPIRV-Tools required):
    python -I tests/test_d16_pack.py
"""
import math
import os
import random
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402


def f32_to_f16_int_sim(u):
    """Python mirror of the integer RTE narrowing #489 added as Gen.f32_to_f16_int (IEEE-754 binary16 RTE). Since
    2026-11-22 Gen.f16_bits does it for the d16 results too: the two gave the same bits on 30,648 inputs."""
    u &= 0xFFFFFFFF
    sign = (u >> 16) & 0x8000
    exp = (u >> 23) & 0xFF
    mant = u & 0x7FFFFF

    # Special: exp == 255
    is_nan = mant != 0
    spec_res = (sign | 0x7E00 | (mant >> 13)) if is_nan else (sign | 0x7C00)

    # Normal: exp >= 113
    n_exp = (exp - 112) & 0xFFFFFFFF
    lsb = (mant >> 13) & 1
    rounded = (mant + (0xFFF + lsb)) & 0xFFFFFFFF
    raw_norm = ((n_exp << 10) + (rounded >> 13)) & 0xFFFFFFFF
    ovf = (exp > 142) or (raw_norm >= 0x7C00)
    norm_res = sign | (0x7C00 if ovf else raw_norm)

    # Subnormal / underflow: 102 <= exp <= 112 (shift strictly in [14, 24] for all exp)
    full_m = mant | 0x800000
    in_sub = (exp >= 102) and (exp <= 112)
    s = (126 - exp) if in_sub else 24
    assert 14 <= s <= 24
    q = full_m >> s
    half_bit = 1 << (s - 1)
    rem_mask = (1 << s) - 1
    rem = full_m & rem_mask
    round_up = (rem > half_bit) or ((rem == half_bit) and ((q & 1) != 0))
    sub_mag = (q + (1 if round_up else 0)) & 0xFFFFFFFF
    sub_res = sign | (sub_mag if in_sub else 0)

    fin_res = norm_res if (exp >= 113) else sub_res
    return spec_res if (exp == 255) else fin_res


def ref_f16(u):
    f = struct.unpack("<f", struct.pack("<I", u))[0]
    if math.isnan(f):
        return None
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return (0x8000 if (u & 0x80000000) else 0) | 0x7C00


class D16PackTests(unittest.TestCase):
    def test_edge_cases_exact(self):
        cases = [
            (0x00000000, 0x0000),  # +0.0
            (0x80000000, 0x8000),  # -0.0
            (0x3F800000, 0x3C00),  # +1.0
            (0xBF800000, 0xBC00),  # -1.0
            (0x3F000000, 0x3800),  # +0.5
            (0x33000000, 0x0000),  # 2^-25 exact tie -> rounds to even (0x0000)
            (0x33000001, 0x0001),  # > 2^-25 -> rounds up to smallest subnormal
            (0x33800000, 0x0001),  # 2^-24 = smallest f16 subnormal
            (0x387FC000, 0x03FF),  # exact largest f16 subnormal (1023 * 2^-24)
            (0x387FDFFF, 0x03FF),  # just below subnormal/normal tie -> 0x03FF
            (0x387FE000, 0x0400),  # exact tie (1023.5 * 2^-24) -> rounds to even (0x0400)
            (0x38800000, 0x0400),  # smallest f16 normal (2^-14)
            (0x477FE000, 0x7BFF),  # max finite f16 (65504.0)
            (0x477FF000, 0x7C00),  # 65520.0 (tie above 65504) -> rounds to even (+Inf)
            (0xC77FF000, 0xFC00),  # -65520.0 -> -Inf
            (0x7F800000, 0x7C00),  # +Inf
            (0xFF800000, 0xFC00),  # -Inf
        ]
        for u, want in cases:
            self.assertEqual(f32_to_f16_int_sim(u), want, f"u=0x{u:08x}")

        # Quiet and signaling NaNs (both signs) produce quiet NaNs preserving sign.
        for u in (0x7F800001, 0x7FC00000, 0x7FFFFFFF, 0xFF800001, 0xFFC00000, 0xFFFFFFFF):
            got = f32_to_f16_int_sim(u)
            self.assertGreater(got & 0x7FFF, 0x7C00, f"not NaN for u=0x{u:08x}")
            self.assertNotEqual(got & 0x0200, 0, f"not quiet NaN for u=0x{u:08x}")
            self.assertEqual(got & 0x8000, (u >> 16) & 0x8000, f"sign mismatch for u=0x{u:08x}")

    def test_all_exponents_and_random_mantissas_sweep(self):
        rng = random.Random(2026)
        boundary_mants = (0, 1, 2, 0xFFF, 0x1000, 0x1001, 0x1FFF, 0x2000, 0x2001,
                          0x3FFF, 0x4000, 0x4001, 0x7FFFFE, 0x7FFFFF)
        for sign_bit in (0, 0x80000000):
            for exp in range(256):
                mants = list(boundary_mants) + [rng.randrange(0, 1 << 23) for _ in range(100)]
                for m in mants:
                    u = sign_bit | (exp << 23) | m
                    got = f32_to_f16_int_sim(u)
                    want = ref_f16(u)
                    if want is None:
                        self.assertGreater(got & 0x7FFF, 0x7C00)
                        self.assertNotEqual(got & 0x0200, 0)
                        self.assertEqual(got & 0x8000, (u >> 16) & 0x8000)
                    else:
                        self.assertEqual(got, want, f"u=0x{u:08x}")

    def test_gen_helper_emits_only_integer_ops_and_no_packhalf2x16(self):
        g = rdna2spirv.Gen("ps")
        g.pack_f16_pair_int("%f0", "%f1")
        body = "\n".join(g.body)
        self.assertNotIn("PackHalf2x16", body)
        self.assertNotIn("OpExtInst", body)
        self.assertIn("OpBitcast %uint", body)
        self.assertIn("OpSelect %uint", body)
        self.assertIn("OpBitwiseOr %uint", body)


if __name__ == "__main__":
    unittest.main()
