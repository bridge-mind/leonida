"""v_mul_f16 under MODE.FP_ROUND[3:2] toward zero (s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3).

Standard library only; no GPU, no LLVM, no game files.

docs/design.md ("f16 rounding", SHADER-55/56): after a MODE.FP_ROUND toward-zero write, every f16 ALU result takes the
toward-zero form (`Gen.f16_res`). tests/test_mode_round.py checks that for v_cvt_f16_f32, v_add/sub/fma_f16 and the
packed ops, and that the f16 transcendentals ignore the mode; this file does the same for v_mul_f16, which had no test
under a MODE write. The f16 * f16 product is exact in f32 (11-bit significands), so the one rounding is the f16 one.

Operands are picked so the two modes give different bits (hand-worked, checked by an exact Fraction oracle below):
  0x3e01 * 1.5 = 2305.5/1024 = 2 + 128.75 * 2^-9   -> RNE 0x4081, RTZ 0x4080 (and the same negated)
  65504 * 2.0 = 131008                              -> RNE +-inf (0x7c00 / 0xfc00), RTZ +-65504 (0x7bff / 0xfbff)
  2^-24 * 1.5 = 1.5 * 2^-24 (f16 subnormal)         -> RNE 0x0002, RTZ 0x0001 (and the same negated)
  inf * 2.0 and 1.0 * 1.5 are the same in both modes.
v_mul_f16_e32 writes the low half and keeps the high half of the destination (docs/design.md "16-bit results"),
so each destination is seeded with 0xaaaa5555 and the expected word is 0xaaaa0000 | result.
"""
from fractions import Fraction
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402

RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3"
RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0"
SEED = 0xAAAA5555

# (a, b, RNE result, RTZ result), all f16 bits
CASES = [
    (0x3E01, 0x3E00, 0x4081, 0x4080),   # 1.500977 * 1.5: 0.75 ulp above 0x4080
    (0xBE01, 0x3E00, 0xC081, 0xC080),   # the same negated: toward zero is toward 0x4080's magnitude
    (0x7BFF, 0x4000, 0x7C00, 0x7BFF),   # 65504 * 2 overflows: inf to nearest, max finite toward zero
    (0xFBFF, 0x4000, 0xFC00, 0xFBFF),
    (0x0001, 0x3E00, 0x0002, 0x0001),   # subnormal 1.5 * 2^-24: tie to even 2, truncated to 1
    (0x8001, 0x3E00, 0x8002, 0x8001),
    (0x7C00, 0x4000, 0x7C00, 0x7C00),   # a true inf stays inf toward zero
    (0x3C00, 0x3E00, 0x3E00, 0x3E00),   # exact product: no rounding in either mode
]


def f16_value(h):
    """Exact value of finite f16 bits `h`, or +-inf as a float."""
    s = -1 if h & 0x8000 else 1
    e, m = (h >> 10) & 0x1F, h & 0x3FF
    if e == 0x1F:
        assert m == 0, "NaN is not used here"
        return s * float("inf")
    if e == 0:
        return s * Fraction(m, 1 << 24)
    return s * Fraction(1024 + m, 1024) * Fraction(2) ** (e - 25 + 10)


def ref_mul_f16(a, b, rtz):
    """Plain-Python f16 product of bits `a` and `b`, rounded once to nearest even or toward zero."""
    x, y = f16_value(a), f16_value(b)
    if isinstance(x, float) or isinstance(y, float):
        p = float(x) * float(y)
        return (0x8000 if p < 0 else 0) | 0x7C00
    p = x * y
    sign = 0x8000 if p < 0 else 0
    p = abs(p)
    if p == 0:
        return sign
    e = 14                                       # find the binade: 2^(e-15) <= p < 2^(e-14), clamped to subnormals
    while e > 1 and p < Fraction(2) ** (e - 15):
        e -= 1
    while p >= Fraction(2) ** (e - 14):
        e += 1
    ulp = Fraction(2) ** (e - 25)                # e here is the biased exponent of a normal (1 also covers e == 0)
    q, r = divmod(p, ulp)
    q = int(q)
    if not rtz and (r * 2 > ulp or (r * 2 == ulp and q & 1)):
        q += 1
    bits = q + ((e - 1) << 10) if q >= 1024 or e > 1 else q
    if bits >= 0x7C00:
        bits = 0x7BFF if rtz else 0x7C00
    return sign | bits


def run_quad(cases, prefix=()):
    """Run four v_mul_f16_e32 on one fragment (v0..v3, seeded high halves); return the four mrt0 words."""
    lines = []
    for i, (a, b) in enumerate(cases):
        lines += [f"v_mov_b32 v{10 + 2 * i}, {a:#x}", f"v_mov_b32 v{11 + 2 * i}, {b:#x}", f"v_mov_b32 v{i}, {SEED:#x}"]
    lines += list(prefix)
    lines += [f"v_mul_f16_e32 v{i}, v{10 + 2 * i}, v{11 + 2 * i}" for i in range(4)]
    lines += ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
    return support.mrt0_words(support.translate_lines("ps_mul_f16_mode", lines))


def word(h):
    return (SEED & 0xFFFF0000) | h


class OracleSelfTest(unittest.TestCase):
    def test_hand_worked_table_matches_exact_oracle(self):
        for a, b, rne, rtz in CASES:
            with self.subTest(a=hex(a), b=hex(b)):
                self.assertEqual(ref_mul_f16(a, b, rtz=False), rne)
                self.assertEqual(ref_mul_f16(a, b, rtz=True), rtz)


class MulF16ModeRoundTests(unittest.TestCase):
    def test_default_mode_rounds_to_nearest_even(self):
        for lo in (0, 4):
            quad = CASES[lo:lo + 4]
            with self.subTest(cases=lo):
                got = run_quad([(a, b) for a, b, _, _ in quad])
                self.assertEqual(got, [word(rne) for _, _, rne, _ in quad])

    def test_toward_zero_after_mode_write(self):
        for lo in (0, 4):
            quad = CASES[lo:lo + 4]
            with self.subTest(cases=lo):
                got = run_quad([(a, b) for a, b, _, _ in quad], prefix=[RTZ])
                self.assertEqual(got, [word(rtz) for _, _, _, rtz in quad])

    def test_mode_write_back_to_nearest_even(self):
        # RTZ for v0/v1, then RNE again for v2/v3, on the same two inputs.
        lines = [
            "v_mov_b32 v10, 0x3e01", "v_mov_b32 v11, 0x3e00", "v_mov_b32 v12, 0x7bff", "v_mov_b32 v13, 0x4000",
            f"v_mov_b32 v0, {SEED:#x}", f"v_mov_b32 v1, {SEED:#x}", f"v_mov_b32 v2, {SEED:#x}", f"v_mov_b32 v3, {SEED:#x}",
            RTZ,
            "v_mul_f16_e32 v0, v10, v11",
            "v_mul_f16_e32 v1, v12, v13",
            RNE,
            "v_mul_f16_e32 v2, v10, v11",
            "v_mul_f16_e32 v3, v12, v13",
            "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm",
        ]
        got = support.mrt0_words(support.translate_lines("ps_mul_f16_mode_back", lines))
        self.assertEqual(got, [word(0x4080), word(0x7BFF), word(0x4081), word(0x7C00)])


if __name__ == "__main__":
    unittest.main()
