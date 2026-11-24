"""v_cmp_{f,lt,eq,le,gt,lg,ge,o,u,nge,nlg,ngt,nle,neq,nlt,tru}_f16, e32 (VCC) and e64 (SGPR pair, swapped operands).

RDNA2 ISA (VOPC, _F16): the compare reads S0.f16 and S1.f16, bits 15:0 of the operands only, and the ordered forms
are false when either operand is NaN, so the N* forms are true for an unordered pair; +0 == -0. The shaders are
runtime/test/shaders/ps_vcmp_f16.asm and ps_vcmp_f16_e64.asm (also run on the GPU by check.py and run_gpu_tests.py);
their headers derive the eight words by hand. The reference here is plain Python on the same f16 bit patterns.
"""
import math
import struct
import unittest

import support

EXPECTED_E32 = [0x48A12080, 0x4CE9B1A2, 0x4BA72C98, 0x4FEFBDBA]      # derivation in ps_vcmp_f16.asm
EXPECTED_E64 = [0x44C11100, 0x4CF171C1, 0x47C71D18, 0x4FF77DD9]      # derivation in ps_vcmp_f16_e64.asm
# (A, B) as f16 bit patterns: 1<2, 2>1, +0 == -0, qNaN vs 1, 1 vs sNaN, +inf > -inf, denormal 1 < denormal 2
PAIRS = ((0x3C00, 0x4000), (0x4000, 0x3C00), (0x0000, 0x8000), (0x7E00, 0x3C00), (0x3C00, 0x7C01),
         (0x7C00, 0xFC00), (0x0001, 0x0002))
OPS = (("f", lambda a, b: False), ("lt", lambda a, b: a < b), ("eq", lambda a, b: a == b),
       ("le", lambda a, b: a <= b), ("gt", lambda a, b: a > b), ("lg", lambda a, b: a < b or a > b),
       ("ge", lambda a, b: a >= b), ("o", lambda a, b: not (math.isnan(a) or math.isnan(b))),
       ("u", lambda a, b: math.isnan(a) or math.isnan(b)), ("nge", lambda a, b: not a >= b),
       ("nlg", lambda a, b: not (a < b or a > b)), ("ngt", lambda a, b: not a > b), ("nle", lambda a, b: not a <= b),
       ("neq", lambda a, b: not a == b), ("nlt", lambda a, b: not a < b), ("tru", lambda a, b: True))


def f16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def reference_words(swapped):
    """mrt0 word w: bit 30 | op (4w + j), case k at bit 7*j + k."""
    words = [0x40000000] * 4
    for n, (_, rel) in enumerate(OPS):
        for k, (a, b) in enumerate(PAIRS):
            s0, s1 = (f16(b), f16(a)) if swapped else (f16(a), f16(b))
            if rel(s0, s1):
                words[n // 4] |= 1 << (7 * (n % 4) + k)
    return words


class VcmpF16(unittest.TestCase):
    def test_reference_matches_the_header_words(self):
        self.assertEqual([f"{w:#010x}" for w in reference_words(False)], [f"{w:#010x}" for w in EXPECTED_E32])
        self.assertEqual([f"{w:#010x}" for w in reference_words(True)], [f"{w:#010x}" for w in EXPECTED_E64])

    def run_fixture(self, name):
        return support.mrt0_words(support.translate_fixture(name))

    def test_ps_vcmp_f16(self):
        self.assertEqual([f"{w:#010x}" for w in self.run_fixture("ps_vcmp_f16")], [f"{w:#010x}" for w in EXPECTED_E32])

    def test_ps_vcmp_f16_e64(self):
        self.assertEqual([f"{w:#010x}" for w in self.run_fixture("ps_vcmp_f16_e64")],
                         [f"{w:#010x}" for w in EXPECTED_E64])


if __name__ == "__main__":
    unittest.main()
