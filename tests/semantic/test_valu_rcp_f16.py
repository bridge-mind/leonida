"""v_rcp_f16 (RDNA2 VOP1 / VOP3): D.f16 = 1.0 / S0.f16, correctly rounded (0 ulp on every non-NaN input on hardware,
docs/design.md "f16 transcendentals"), with rcp(+-0) = +-inf and rcp(+-inf) = +-0.

Every case is translated by tools/rdna2spirv.py and executed with tests/semantic/spirv_mini_interp.py; the expected
word is written down by hand (from the ISA) next to each case, rechecked against the plain-Python reference in
tests/semantic/f16_unary_cases.py, and extended with seeded random operands, neg/abs, omod/clamp, op_sel, and SGPR /
literal / inline-constant sources. A NaN result is compared by class, not payload. Standard library only.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from f16_unary_cases import (HALF, INF, MAXH, MAX_SUB, MIN_NORMAL, MIN_SUB, NINF, ONE, QNAN, QUARTER, TWO,  # noqa: E402
                             UnaryF16, ref_rcp, run, same_f16)


class RcpF16(UnaryF16, unittest.TestCase):
    OP = "v_rcp_f16"
    ref = staticmethod(ref_rcp)
    CASES = [
        (ONE, ONE), (TWO, HALF), (HALF, TWO), (QUARTER, 0x4400), (0x4400, QUARTER),
        (0xBC00, 0xBC00), (0xC000, 0xB800),         # rcp(-1) = -1, rcp(-2) = -0.5
        (0x4200, 0x3555),                           # 1/3 = 0.33333 -> 0x3555 (0.333251953125)
        (0x4500, 0x3266),                           # 1/5 = 0.2 -> 0x3266 (0.199951171875)
        (0x0000, INF), (0x8000, NINF),              # rcp(+-0) = +-inf
        (INF, 0x0000), (NINF, 0x8000),              # rcp(+-inf) = +-0
        (QNAN, QNAN),
        (MAXH, 0x0100),                             # 1/65504 = 1.5267e-5 = 256.1 * 2^-24 -> the subnormal 0x0100
        (MIN_NORMAL, 0x7400),                       # 1/2^-14 = 2^14 = 16384
        (MAX_SUB, 0x7401),                          # 16777216 / 1023 = 16400.06, ulp 16 there -> 16400
        (0x0100, INF),                              # 1/2^-16 = 65536 overflows
        (MIN_SUB, INF),                             # 1/2^-24 = 2^24 overflows
        (0x8001, NINF),
            ]
    MOD_INPUTS = [0x4400, 0x4000, 0x4900, 0xC400]       # 4, 2, 10, -4
    INLINE = [("2.0", TWO, HALF), ("4.0", 0x4400, QUARTER), ("0.5", HALF, TWO), ("-1.0", 0xBC00, 0xBC00),
              ("1.0", ONE, ONE), ("-4.0", 0xC400, 0xB400)]
    # low half 4.0 / high half 0.5: rcp = 0.25 / 2.0
    OP_SEL = [((HALF << 16) | 0x4400, QUARTER, TWO)]


if __name__ == "__main__":
    unittest.main()
