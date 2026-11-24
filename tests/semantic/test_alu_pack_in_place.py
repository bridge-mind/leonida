"""v_pack_b32_u16 is not an RDNA2 instruction, and tools/rdna2spirv.py rejects it.

The previous version of this file (799 lines) modelled in-place "v_pack_b32_u16 dst, src0, src1 [op_sel] [clamp]" with
a private Python function and hand-written SPIR-V, and never called the translator.  RDNA2 packs two halves with
v_pack_b32_f16 only (llvm-mc -triple=amdgcn -mcpu=gfx1030: "invalid instruction" for v_pack_b32_u16), so there is no ISA
semantics to test against; the honest test is that the translator refuses the mnemonic by name, in the in-place forms
the old file modelled too.  v_pack_b32_f16, which is real, still translates in the same aliasing forms.
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
    return ["v_mov_b32 v0, 0x11112222", "v_mov_b32 v1, 0x33334444", ins, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]


FORMS = [
    "v_pack_b32_u16 v0, v0, v1",                    # dst aliases src0
    "v_pack_b32_u16 v0, v1, v0",                    # dst aliases src1
    "v_pack_b32_u16 v0, v0, v0",                    # dst aliases both
    "v_pack_b32_u16 v2, v0, v1",
    "v_pack_b32_u16 v0, v0, v1 op_sel:[1,0]",
    "v_pack_b32_u16 v0, v1, v0 op_sel:[0,1]",
    "v_pack_b32_u16 v0, v0, v1 clamp",
]


class PackB32U16Unsupported(unittest.TestCase):
    def test_refused_by_name_in_every_form(self):
        for ins in FORMS:
            with self.subTest(ins=ins):
                with self.assertRaises(rdna2spirv.Unsupported) as refused:
                    support.translate_lines("ps_pack_u16", program(ins))
                self.assertEqual(str(refused.exception), "v_pack_b32_u16")

    def test_the_real_pack_translates_in_place(self):
        for ins in ("v_pack_b32_f16 v0, v0, v1", "v_pack_b32_f16 v0, v1, v0", "v_pack_b32_f16 v0, v0, v0",
                    "v_pack_b32_f16 v0, v0, v1 op_sel:[1,0,0]"):
            with self.subTest(ins=ins):
                self.assertIn("OpFunction", support.translate_lines("ps_pack_ok", program(ins)))


if __name__ == "__main__":
    unittest.main()
