"""v_fract_f16 (RDNA2 VOP1 / VOP3): D.f16 = S0.f16 - floor(S0.f16), rounded once to binary16.

A result that rounds up to 1.0 (a tiny negative operand) is 0x3bff, the largest word below 1; fract(+-0) = +0, fract(+-inf)
is a NaN (0xfe00 on the hardware, a quiet NaN here: compared by class) and a NaN passes through (docs/design.md,
"Float min/max/med3 with NaN, v_fract").

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
                             UnaryF16, ref_fract, run, same_f16)  # noqa: F401


class FractF16(UnaryF16, unittest.TestCase):
    OP = "v_fract_f16"
    ref = staticmethod(ref_fract)
    CASES = [
        (0x4180, 0x3A00), (0xC180, 0x3400),         # fract(2.75) = 0.75, fract(-2.75) = 0.25
        (HALF, HALF), (0xB800, HALF), (0x3E00, HALF),
        (ONE, 0x0000), (0xBC00, 0x0000), (0x6800, 0x0000), (MAXH, 0x0000), (0xFBFF, 0x0000),
        (0x1400, 0x1400),                           # 2^-10 is already below 1
        (0x9400, 0x3BFE),                           # fract(-2^-10) = 1 - 2^-10 is exact
        (0xBBFF, 0x1000),                           # fract(-0.99951171875) = 2^-11
        (0x3BFF, 0x3BFF),
        (MIN_SUB, MIN_SUB), (MAX_SUB, MAX_SUB),
        (0x8001, 0x3BFF),                           # fract(-2^-24) = 1 - 2^-24 rounds to 1.0, which is written 0x3bff
        (0x8400, 0x3BFF),                           # likewise -2^-14
        (0x0000, 0x0000), (0x8000, 0x0000),         # fract(-0) = +0
        (INF, QNAN), (NINF, QNAN), (QNAN, QNAN),
    ]
    MOD_INPUTS = [0x4180, 0x3E00, 0xC180, 0x3400]
    INLINE = [("0.5", HALF, HALF), ("-0.5", 0xB800, HALF), ("2.0", TWO, 0x0000), ("-1.0", 0xBC00, 0x0000),
              ("0.15915494", 0x3118, 0x3118)]
    OP_SEL = [((0xC180 << 16) | 0x4180, 0x3A00, 0x3400)]       # lo 2.75 / hi -2.75


if __name__ == "__main__":
    unittest.main()
