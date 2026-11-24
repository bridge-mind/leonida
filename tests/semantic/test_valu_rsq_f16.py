"""v_rsq_f16 (RDNA2 VOP1 / VOP3): D.f16 = 1.0 / sqrt(S0.f16), correctly rounded (0 ulp on every non-NaN input on hardware,
docs/design.md "f16 transcendentals"), with rsq(+0) = +inf, rsq(-0) = -inf, rsq(+inf) = +0, and a NaN for a negative
operand (-inf included).

Every case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
word is written down by hand (from the ISA, values checked independently with float64) next to each case, rechecked
against the plain-Python reference in tests/semantic/f16_unary_cases.py, and extended with seeded random operands,
neg/abs, omod/clamp, op_sel, and SGPR / literal / inline-constant sources. A NaN result is compared by class, not
payload. Standard library only: no GPU, no LLVM, no game data.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f16_unary_cases import (HALF, INF, MAXH, MAX_SUB, MIN_NORMAL, MIN_SUB, NINF, ONE, QNAN, QUARTER, TWO,  # noqa: E402,F401
                             UnaryF16, ref_rsq, run, same_f16)  # noqa: F401


class RsqF16(UnaryF16, unittest.TestCase):
    OP = "v_rsq_f16"
    ref = staticmethod(ref_rsq)
    CASES = [
        (ONE, ONE), (0x4400, HALF), (0x4C00, QUARTER), (QUARTER, TWO),
        (TWO, 0x39A8),                              # 0.70710678 = 1.41421 * 2^-1, mantissa 424
        (0x4200, 0x389E),                           # 0.57735027
        (HALF, 0x3DA8),                             # 1.41421356
        (0x4900, 0x350F),                           # rsq(10) = 0.31622777
        (0x5640, 0x2E66),                           # rsq(100) = 0.1
        (MAXH, 0x1C00),                             # rsq(65504) = 0.0039070 -> 2^-8 = 0.00390625
        (MIN_NORMAL, 0x5800),                       # rsq(2^-14) = 2^7 = 128
        (MIN_SUB, 0x6C00),                          # rsq(2^-24) = 2^12 = 4096
        (MAX_SUB, 0x5801),
        (0x0000, INF), (0x8000, NINF),              # rsq(+-0) = +-inf
        (INF, 0x0000),
        (NINF, QNAN), (0xBC00, QNAN), (0x8001, QNAN),   # a negative operand is a NaN (0xfe00 on hardware)
        (QNAN, QNAN),
    ]
    MOD_INPUTS = [0x4400, 0x4900, QUARTER, 0xC400]
    INLINE = [("4.0", 0x4400, HALF), ("1.0", ONE, ONE), ("0.5", HALF, 0x3DA8), ("2.0", TWO, 0x39A8),
              ("-1.0", 0xBC00, QNAN)]
    OP_SEL = [((0x4C00 << 16) | 0x4400, HALF, QUARTER)]         # lo 4.0 / hi 16.0


if __name__ == "__main__":
    unittest.main()
