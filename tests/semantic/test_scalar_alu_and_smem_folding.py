from __future__ import annotations

"""Regression and semantic tests for scalar ALU and s_buffer_load constant folding in the translator.

Verifies that:
- Constant scalar ALU operations (s_and_b32, s_or_b32, s_xor_b32, s_not_b32, s_lshl_b32, s_lshr_b32, s_mul_i32)
  fold at translate time without emitting runtime SPIR-V instructions.
- s_buffer_load with known descriptor limits folds limit calculations at translate time, eliminating
  OpBitFieldUExtract, OpIMul, OpIEqual, and OpSelect.
- In-bounds s_buffer_load dwords emit direct straight-line OpLoad without OpSelectionMerge, OpBranchConditional, or OpPhi.
- Out-of-bounds s_buffer_load dwords emit constant zero without memory load or branch.
- Constant SGPR offsets in scalar loads fold directly into the 64-bit address calculation.
"""
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


class ScalarAluAndSmemFoldingTests(unittest.TestCase):
    def translate_lines(self, lines: list[str]) -> str:
        body = "\n".join("\t" + line for line in lines)
        asm = f"; scalar_fold_test\n{body}\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        with tempfile.TemporaryDirectory() as tmp:
            d_dir = os.path.join(tmp, "disasm")
            os.mkdir(d_dir)
            s_path = os.path.join(d_dir, "ps_scalar.s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(asm)
            with open(os.path.join(tmp, "ps_scalar.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(s_path)

    def test_s_and_b32_and_s_or_b32_constant_folding(self):
        # Bitwise s_and_b32 / s_or_b32 fold constants without runtime bitwise instructions
        lines = [
            "s_and_b32 s4, 0xFF00, 0x0FF0",
            "v_mov_b32 v0, s4",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x0F00)

        lines_or = [
            "s_or_b32 s4, 0xF0, 0x0F",
            "v_mov_b32 v0, s4",
        ]
        spv_or = self.translate_lines(lines_or)
        self.assertNotIn("OpBitwiseOr %uint", spv_or)
        res_or = si.run_fragment(spv_or)["%out_mrt0"][0]
        self.assertEqual(res_or, 0xFF)

    def test_s_xor_b32_and_s_not_b32_constant_folding(self):
        # Bitwise s_xor_b32 and s_not_b32 fold constants
        lines_xor = [
            "s_xor_b32 s4, 0xAAAA5555, 0x5555AAAA",
            "v_mov_b32 v0, s4",
        ]
        spv_xor = self.translate_lines(lines_xor)
        self.assertNotIn("OpBitwiseXor %uint", spv_xor)
        res_xor = si.run_fragment(spv_xor)["%out_mrt0"][0]
        self.assertEqual(res_xor, 0xFFFFFFFF)

        lines_not = [
            "s_not_b32 s4, 0",
            "v_mov_b32 v0, s4",
        ]
        spv_not = self.translate_lines(lines_not)
        self.assertNotIn("OpNot %uint", spv_not)
        res_not = si.run_fragment(spv_not)["%out_mrt0"][0]
        self.assertEqual(res_not, 0xFFFFFFFF)

    def test_s_lshl_b32_s_lshr_b32_s_mul_i32_folding(self):
        # Shifts and multiply fold constants
        lines_shl = [
            "s_lshl_b32 s4, 3, 4",        # 3 << 4 = 48
            "v_mov_b32 v0, s4",
        ]
        spv_shl = self.translate_lines(lines_shl)
        self.assertNotIn("OpShiftLeftLogical %uint", spv_shl)
        res_shl = si.run_fragment(spv_shl)["%out_mrt0"][0]
        self.assertEqual(res_shl, 48)

        lines_mul = [
            "s_mul_i32 s4, 6, 7",
            "v_mov_b32 v0, s4",
        ]
        spv_mul = self.translate_lines(lines_mul)
        self.assertNotIn("OpIMul %uint", spv_mul)
        res_mul = si.run_fragment(spv_mul)["%out_mrt0"][0]
        self.assertEqual(res_mul, 42)

        lines_shr = [
            "s_lshr_b32 s4, 96, 3",       # 96 >> 3 = 12
            "v_mov_b32 v0, s4",
        ]
        spv_shr = self.translate_lines(lines_shr)
        self.assertNotIn("OpShiftRightLogical %uint", spv_shr)
        res_shr = si.run_fragment(spv_shr)["%out_mrt0"][0]
        self.assertEqual(res_shr, 12)

    def test_s_buffer_load_in_bounds_eliminates_branch_diamond(self):
        # When V# limits and offset are constants with offset in bounds:
        # eliminates limit setup and conditional branch diamond.
        g = rdna2spirv.Gen("ps")
        # Load literal V# into s[4:7]:
        # s4: base lo, s5: base hi (0) + stride 4 in bits 16..29 (0x00040000)
        # s6: num_records = 256 (limit = 256 * 4 = 1024 bytes)
        # s7: word3
        g.sgpr_imm[4] = 0x1000
        g.sgpr_imm[5] = 0x00040000
        g.sgpr_imm[6] = 256
        g.sgpr_imm[7] = 0x31000FAC

        ins = parse_line("s_buffer_load_dword s0, s[4:7], 0x10")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)

        # No dynamic limit calculation instructions
        self.assertNotIn("OpBitFieldUExtract %uint", text)
        self.assertNotIn("OpIMul %ulong", text)
        self.assertNotIn("OpSelect %ulong", text)

        # In bounds (0x10 + 4 = 20 <= 1024): no branch or phi
        self.assertNotIn("OpBranchConditional", text)
        self.assertNotIn("OpSelectionMerge", text)
        self.assertNotIn("OpPhi", text)

        # Emits direct aligned load
        self.assertIn("OpAccessChain %_ptr_PSB_uint", text)
        self.assertIn("OpLoad %uint", text)

    def test_s_buffer_load_out_of_bounds_emits_constant_zero(self):
        # When V# limits and offset are constants with offset out of bounds:
        # emits constant 0 without memory load or branches.
        g = rdna2spirv.Gen("ps")
        g.sgpr_imm[4] = 0x1000
        g.sgpr_imm[5] = 0x00040000
        g.sgpr_imm[6] = 16          # limit = 16 * 4 = 64 bytes
        g.sgpr_imm[7] = 0x31000FAC

        ins = parse_line("s_buffer_load_dword s0, s[4:7], 0x80")   # offset 128 > limit 64
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)

        # No branch or PSB memory load emitted for out of bounds
        self.assertNotIn("OpBranchConditional", text)
        self.assertNotIn("OpSelectionMerge", text)
        self.assertNotIn("OpPhi", text)
        self.assertNotIn("OpAccessChain %_ptr_PSB_uint", text)
        self.assertNotIn("Aligned 4", text)

        # Directly stores %c0 to sgpr 0
        self.assertTrue(any(ln.strip().startswith("OpStore %t") and ln.strip().endswith(g.cu(0)) for ln in g.body))

    def test_s_buffer_load_sgpr_offset_folded_from_sgpr_imm(self):
        # When offset is passed via SGPR holding a known literal:
        # folded into constant offset without runtime OpBitwiseAnd / OpUConvert on the offset register.
        g = rdna2spirv.Gen("ps")
        g.sgpr_imm[2] = 0x20        # s2 = 32
        ins = parse_line("s_load_dword s0, s[4:5], s2")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)

        # The offset is folded into constant ulong %cl20
        self.assertIn("%cl20", text)
        # Only base lo is masked with %cfffffffc; offset s2 is not dynamically masked
        self.assertEqual(text.count("%cfffffffc"), 1)

    def test_s_buffer_load_constant_negative_sgpr_offset_does_not_wrap(self):
        # SHADER-328: s8 = -4 with offset:20 is 0xFFFFFFFC + 20 in 64 bits (RDNA2 reads 0, measured in #1686), not
        # 16 after a 32-bit wrap. The constant must give what the run-time path gives: out of bounds, 0.
        g = rdna2spirv.Gen("ps")
        g.sgpr_imm.update({0: 0x1000, 1: 0, 2: 64, 3: 0x31000FAC, 8: 0xFFFFFFFC})   # raw V#, limit 64 bytes
        ins = parse_line("s_buffer_load_dword s10, s[0:3], s8 offset:20")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        text = "\n".join(g.body)
        self.assertNotIn("OpAccessChain %_ptr_PSB_uint", text)     # no read: the dword is out of bounds
        self.assertTrue(any(ln.strip().startswith("OpStore %t") and ln.strip().endswith(g.cu(0)) for ln in g.body))

        # The same offsets inside the limit still read memory.
        g = rdna2spirv.Gen("ps")
        g.sgpr_imm.update({0: 0x1000, 1: 0, 2: 64, 3: 0x31000FAC, 8: 4})
        ins = parse_line("s_buffer_load_dword s10, s[0:3], s8 offset:20")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        self.assertIn("OpAccessChain %_ptr_PSB_uint", "\n".join(g.body))

    def test_s_load_constant_sgpr_offset_is_not_sign_extended(self):
        # The run-time path adds zext(sOFF & ~3) to the 21-bit immediate; a constant sOFF of 2^20 must not pass
        # through the immediate's 21-bit sign extension (it would become -2^20).
        g = rdna2spirv.Gen("ps")
        g.sgpr_imm[2] = 0x100000
        ins = parse_line("s_load_dword s0, s[4:5], s2 offset:0x10")
        g.scalar_load(ins["op"], ins["args"], ins["mods"])
        self.assertIn("%cl100010", "\n".join(g.body))


if __name__ == "__main__":
    unittest.main()
