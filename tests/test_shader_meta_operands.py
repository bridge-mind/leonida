"""tools/shader_meta.py leaves malformed or out-of-range operands alone instead of raising (standard library only).

    python -I tests/test_shader_meta_operands.py
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_meta  # noqa: E402


def sgpr(r):
    return {"k": "sgpr", "r": r, "n": 1}


def step(ins):
    sym, scale, ptr_stride, vscale = {}, {}, {}, {}
    shader_meta._step_state(sym, scale, ptr_stride, vscale, ins)
    return sym


class ImmediateTest(unittest.TestCase):
    def test_float_immediate_in_range_is_its_bits(self):
        self.assertEqual(shader_meta._imm_u32({"k": "fimm", "v": 1.0}), 0x3F800000)

    def test_float_immediate_no_float32_holds_is_not_a_constant(self):
        self.assertIsNone(shader_meta._imm_u32({"k": "fimm", "v": 1e300}))
        self.assertIsNone(shader_meta._imm_u32({"k": "fimm", "v": -1e300}))


class MovkTest(unittest.TestCase):
    def test_movk_with_an_immediate_records_the_constant(self):
        sym = step({"op": "s_movk_i32", "args": [sgpr(4), {"k": "imm", "v": 0x1234}]})
        self.assertEqual(sym[4], ("const", 0x1234))

    def test_movk_sign_extends_the_16_bit_immediate(self):
        sym = step({"op": "s_movk_i32", "args": [sgpr(4), {"k": "imm", "v": 0xFFFF}]})
        self.assertEqual(sym[4], ("const", 0xFFFFFFFF))

    def test_movk_with_a_register_operand_is_ignored(self):
        self.assertEqual(step({"op": "s_movk_i32", "args": [sgpr(4), sgpr(6)]}).get(4), None)


if __name__ == "__main__":
    unittest.main()
