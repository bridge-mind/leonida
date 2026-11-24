"""Parity and edge-case tests for tools/shader_ir.py parse_line(), split_args(), and operand()."""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_ir  # noqa: E402


class ShaderIRParseTests(unittest.TestCase):
    def test_split_args_plain_and_nested(self):
        self.assertEqual(shader_ir.split_args(""), [])
        self.assertEqual(shader_ir.split_args("   "), [])
        self.assertEqual(shader_ir.split_args("v0, v1, 4"), ["v0", " v1", " 4"])
        self.assertEqual(shader_ir.split_args("v0, v1, "), ["v0", " v1"])
        self.assertEqual(shader_ir.split_args("v0, , v1"), ["v0", " ", " v1"])
        self.assertEqual(shader_ir.split_args("[v0, v1], s[4:7], neg(|v2|)"),
                         ["[v0, v1]", " s[4:7]", " neg(|v2|)"])

    def test_operand_fast_paths_and_modifiers(self):
        self.assertEqual(shader_ir.operand("v3"), {"k": "vgpr", "r": 3, "n": 1, "neg": False, "abs": False})
        self.assertEqual(shader_ir.operand("-|s12|"), {"k": "sgpr", "r": 12, "n": 1, "neg": True, "abs": True})
        self.assertEqual(shader_ir.operand("vcc"), {"k": "special", "t": "vcc", "neg": False, "abs": False})
        self.assertEqual(shader_ir.operand("null"), {"k": "special", "t": "null", "neg": False, "abs": False})
        self.assertEqual(shader_ir.operand("0"), {"k": "imm", "v": 0})
        self.assertEqual(shader_ir.operand("-16"), {"k": "imm", "v": -16})
        self.assertEqual(shader_ir.operand("010"), {"k": "imm", "v": 10})
        self.assertEqual(shader_ir.operand("|42|"), {"k": "imm", "v": 42, "abs": True})
        self.assertEqual(shader_ir.operand("-|42|"), {"k": "imm", "v": 42, "abs": True, "neg": True})
        self.assertEqual(shader_ir.operand("0x10"), {"k": "imm", "v": 16})
        self.assertEqual(shader_ir.operand("neg(1)"), {"k": "imm", "v": 1, "neg": True})
        self.assertEqual(shader_ir.operand("abs(v2)"), {"k": "vgpr", "r": 2, "n": 1, "neg": False, "abs": True})
        self.assertEqual(shader_ir.operand("sext(v4)"),
                         {"k": "vgpr", "r": 4, "n": 1, "neg": False, "abs": False, "sext": True})
        self.assertEqual(shader_ir.operand("s[4:7]"), {"k": "sgpr", "r": 4, "n": 4, "neg": False, "abs": False})
        self.assertEqual(shader_ir.operand("[v0, v2]"), {"k": "vlist", "r": [0, 2]})
        self.assertEqual(shader_ir.operand("[s100, vcc_lo, vcc_hi]"),
                         {"k": "slist", "r": [100, "vcc_lo", "vcc_hi"], "n": 3})
        self.assertEqual(shader_ir.operand("attr2.z"), {"k": "attr", "a": 2, "c": 2})
        self.assertEqual(shader_ir.operand("p10"), {"k": "interp", "t": "p10"})
        self.assertEqual(shader_ir.operand("mrt0"), {"k": "target", "t": "mrt0"})
        self.assertEqual(shader_ir.operand("ttmp[4:5]"), {"k": "ttmp", "t": "ttmp[4:5]"})
        self.assertEqual(shader_ir.operand("1.5"), {"k": "fimm", "v": 1.5, "abs": False})
        self.assertEqual(shader_ir.operand("vmcnt(0)"), {"k": "special", "t": "vmcnt(0)"})

    def test_parse_line_modifiers_sendmsg_and_encoding(self):
        ins = shader_ir.parse_line(
            "\tbuffer_load_dword v0, v1, s[4:7], 0 offen offset:16 glc ; encoding: [0x10,0x10,0x30,0xe0,0x01,0x00,0x01,0x80]"
        )
        self.assertEqual(ins["op"], "buffer_load_dword")
        self.assertEqual(ins["mods"], ["offen", "offset:16", "glc"])
        self.assertEqual(ins["enc"], "101030e001000180")
        self.assertEqual(len(ins["args"]), 4)

        sm = shader_ir.parse_line("\ts_sendmsg sendmsg(MSG_GS, GS_OP_EMIT, 0) ; plain comment")
        self.assertEqual(sm["op"], "s_sendmsg")
        self.assertEqual(sm["args"], [{"k": "special", "t": "sendmsg(MSG_GS|GS_OP_EMIT|0)"}])
        self.assertIsNone(sm["enc"])

        f64 = shader_ir.parse_line("\tv_add_f64 v[0:1], v[2:3], 0x1")
        self.assertTrue(f64["args"][2].get("f64_literal"))


if __name__ == "__main__":
    unittest.main()
