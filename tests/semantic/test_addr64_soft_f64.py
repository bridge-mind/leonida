"""Combined #2463 address lowering and #2904 software-f64 capability regression.

The test is skipped until both dependencies are present. It uses only this hand-written synthetic program.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools"))

import rdna2spirv  # noqa: E402
from support import translate_lines  # noqa: E402


@unittest.skipUnless(hasattr(rdna2spirv, "ADDR_PAIR") and hasattr(rdna2spirv, "F64_SOFT"),
                     "Depends on #2463 and #2904")
class AddressLoweringSoftF64Tests(unittest.TestCase):
    LINES = ["s_load_dword s4, s[0:1], 0x0", "s_waitcnt lgkmcnt(0)", "v_mov_b32 v0, s4",
             "v_mov_b32 v1, 0x3ff00000", "v_add_f64 v[2:3], v[0:1], v[0:1]",
             "exp mrt0 v2, v3, v0, v1 done vm", "s_endpgm"]

    def translate(self, addr_pair):
        old_addr, old_soft = rdna2spirv.ADDR_PAIR, rdna2spirv.F64_SOFT
        try:
            rdna2spirv.ADDR_PAIR = addr_pair
            rdna2spirv.F64_SOFT = True
            return translate_lines("ps_f64_soft_addr64_case", self.LINES)
        finally:
            rdna2spirv.ADDR_PAIR, rdna2spirv.F64_SOFT = old_addr, old_soft

    def test_lowered_address_and_soft_f64_need_neither_64bit_capability(self):
        text = self.translate(True)
        self.assertNotIn("OpCapability Float64", text)
        self.assertNotIn("OpCapability Int64", text)
        self.assertNotIn("OpTypeFloat 64", text)
        self.assertNotIn("OpTypeInt 64", text)
        self.assertIn("OpBitcast %_ptr_PSB", text)
        self.assertIn("%f64s_add = OpFunction", text)

    def test_int64_address_opt_out_does_not_restore_float64(self):
        text = self.translate(False)
        self.assertNotIn("OpCapability Float64", text)
        self.assertNotIn("OpTypeFloat 64", text)
        self.assertIn("OpCapability Int64", text)
        self.assertIn("OpTypeInt 64", text)
        self.assertIn("%f64s_add = OpFunction", text)


if __name__ == "__main__":
    unittest.main()
