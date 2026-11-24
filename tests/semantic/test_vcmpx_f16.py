"""v_cmpx_{f,lt,eq,le,gt,lg,ge,o,u,nge,nlg,ngt,nle,neq,nlt,tru}_f16, e32 and e64 (swapped operands).

RDNA2 ISA (VOPC, _F16): a v_cmpx compares like v_cmp (S0.f16 with S1.f16, ordered compares false for a NaN, +0 == -0)
but writes the result to EXEC (EXEC = EXEC & compare) and to nothing else, so VCC is kept. The shaders are
runtime/test/shaders/ps_vcmpx_f16.asm and ps_vcmpx_f16_e64.asm (also run on the GPU by check.py and
run_gpu_tests.py); their headers derive the eight words by hand. The reference reuses the one of test_vcmp_f16 and
adds bit 29 (VCC kept, 1) next to bit 30 (marker) and bit 28 (a v_cmpx cannot widen EXEC, 0).
"""
import unittest

import support
from test_vcmp_f16 import reference_words

EXPECTED_E32 = [0x68A12080, 0x6CE9B1A2, 0x6BA72C98, 0x6FEFBDBA]      # derivation in ps_vcmpx_f16.asm
EXPECTED_E64 = [0x64C11100, 0x6CF171C1, 0x67C71D18, 0x6FF77DD9]      # derivation in ps_vcmpx_f16_e64.asm


def hexes(words):
    return [f"{w:#010x}" for w in words]


class VcmpxF16(unittest.TestCase):
    def test_reference_matches_the_header_words(self):
        self.assertEqual(hexes(w | 0x20000000 for w in reference_words(False)), hexes(EXPECTED_E32))
        self.assertEqual(hexes(w | 0x20000000 for w in reference_words(True)), hexes(EXPECTED_E64))

    def test_ps_vcmpx_f16(self):
        self.assertEqual(hexes(support.mrt0_words(support.translate_fixture("ps_vcmpx_f16"))), hexes(EXPECTED_E32))

    def test_ps_vcmpx_f16_e64(self):
        self.assertEqual(hexes(support.mrt0_words(support.translate_fixture("ps_vcmpx_f16_e64"))), hexes(EXPECTED_E64))


if __name__ == "__main__":
    unittest.main()
