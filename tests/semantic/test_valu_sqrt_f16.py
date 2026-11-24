"""v_sqrt_f16 (RDNA2 VOP1 / VOP3): D.f16 = sqrt(S0.f16), correctly rounded (0 ulp on every non-NaN input on hardware,
docs/design.md "f16 transcendentals"), with sqrt(+-0) = +-0, sqrt(+inf) = +inf, and a NaN for a negative operand
(-inf included).

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
                             UnaryF16, ref_sqrt, run, same_f16)  # noqa: F401


class SqrtF16(UnaryF16, unittest.TestCase):
    OP = "v_sqrt_f16"
    ref = staticmethod(ref_sqrt)
    CASES = [
        (ONE, ONE), (0x4400, TWO), (0x4C00, 0x4400), (QUARTER, HALF), (0x4880, 0x4200),   # sqrt(9) = 3
        (TWO, 0x3DA8),                              # 1.41421356
        (0x4200, 0x3EEE),                           # 1.7320508
        (0x4900, 0x4253),                           # sqrt(10) = 3.1622777
        (0x5640, 0x4900),                           # sqrt(100) = 10
        (HALF, 0x39A8),
        (MAXH, 0x5BFF),                             # sqrt(65504) = 255.9375 is exact
        (MIN_NORMAL, 0x2000),                       # sqrt(2^-14) = 2^-7
        (MIN_SUB, 0x0C00),                          # sqrt(2^-24) = 2^-12
        (MAX_SUB, 0x1FFF),
        (0x0000, 0x0000), (0x8000, 0x8000),         # sqrt(-0) = -0
        (INF, INF),
        (NINF, QNAN), (0xBC00, QNAN), (0x8001, QNAN),   # a negative operand is a NaN (0xfe00 on hardware)
        (QNAN, QNAN),
    ]
    MOD_INPUTS = [0x4400, 0x4900, QUARTER, 0xC400]
    INLINE = [("4.0", 0x4400, TWO), ("1.0", ONE, ONE), ("0.5", HALF, 0x39A8), ("2.0", TWO, 0x3DA8),
              ("-1.0", 0xBC00, QNAN)]
    OP_SEL = [((0x4880 << 16) | 0x4400, TWO, 0x4200)]          # lo 4.0 / hi 9.0


if __name__ == "__main__":
    unittest.main()
