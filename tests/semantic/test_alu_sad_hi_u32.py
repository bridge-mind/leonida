"""v_sad_hi_u32 is not an RDNA2 instruction, and tools/rdna2spirv.py rejects it.

The previous version of this file (944 lines) modelled "v_sad_hi_u32 = high 32 bits of |S0 - S1| + S2" with a private
Python function and hand-written SPIR-V, and never called the translator.  RDNA2 has v_sad_u8, v_sad_hi_u8, v_sad_u16
and v_sad_u32, but no v_sad_hi_u32 (llvm-mc -triple=amdgcn -mcpu=gfx1030: "invalid instruction"), so there is no ISA
semantics to test against; the honest test is that the translator refuses the mnemonic by name instead of inventing
one.  The neighbouring real ops still translate, so the refusal is specific to this mnemonic.
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)
import support  # noqa: E402
import rdna2spirv  # noqa: E402


def program(ins):
    return ["v_mov_b32 v1, 7", "v_mov_b32 v2, 9", "v_mov_b32 v3, 11", ins, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]


class SadHiU32Unsupported(unittest.TestCase):
    def test_refused_by_name_in_every_form(self):
        for ins in ("v_sad_hi_u32 v0, v1, v2, v3", "v_sad_hi_u32 v0, s1, v2, v3", "v_sad_hi_u32 v0, v1, 5, 0xffffffff",
                    "v_sad_hi_u32 v1, v1, v2, v1", "v_sad_hi_u32 v0, v1, v2, v3 clamp"):
            with self.subTest(ins=ins):
                with self.assertRaises(rdna2spirv.Unsupported) as refused:
                    support.translate_lines("ps_sad_hi_u32", program(ins))
                self.assertEqual(str(refused.exception), "v_sad_hi_u32")

    def test_the_real_sad_ops_translate(self):
        for ins in ("v_sad_u32 v0, v1, v2, v3", "v_sad_hi_u8 v0, v1, v2, v3", "v_sad_u8 v0, v1, v2, v3",
                    "v_sad_u16 v0, v1, v2, v3"):
            with self.subTest(ins=ins):
                self.assertIn("OpFunction", support.translate_lines("ps_sad_ok", program(ins)))


if __name__ == "__main__":
    unittest.main()
