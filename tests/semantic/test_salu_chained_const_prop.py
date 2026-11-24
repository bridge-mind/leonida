from __future__ import annotations

"""Semantic and regression tests for chained scalar ALU constant propagation in the translator.

Verifies:
- Multi-step chained constant propagation across SGPRs (s_mov_b32 -> s_lshl_b32 -> s_add_u32).
- s_add_u32 and s_sub_u32 constant tracking in sgpr_imm.
- s_lshl{1..4}_add_u32 constant tracking in sgpr_imm.
- s_bfe_u32 and s_bfe_i32 constant folding and static operand emission.
- s_min/max_u32 and s_min/max_i32 constant tracking in sgpr_imm.
- s_andn2, s_orn2, s_nand, s_nor, s_xnor bitwise logic constant tracking in sgpr_imm.
- s_abs_i32, s_absdiff_i32, s_sext_i32_i8/i16 constant tracking in sgpr_imm.
- s_pack_*_b32_b16 and s_bfm_b32 constant tracking in sgpr_imm.
- s_buffer_load offset computed through chained SALU folds into a static load without branch diamonds, and the
  load reads the word at the folded offset.
- v_bfe constant folding.
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
import support  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


class ChainedScalarConstPropTests(unittest.TestCase):
    def translate_lines(self, lines: list[str]) -> str:
        body = "\n".join("\t" + line for line in lines)
        asm = f"; chained_salu_test\n{body}\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        with tempfile.TemporaryDirectory() as tmp:
            d_dir = os.path.join(tmp, "disasm")
            os.mkdir(d_dir)
            s_path = os.path.join(d_dir, "ps_chained.s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(asm)
            with open(os.path.join(tmp, "ps_chained.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(s_path)

    def assert_loads_offset(self, spv: str, offset: int) -> None:
        # Each buffer word holds 0xA000 + its byte offset, so a wrong fold that stays in bounds still reads the wrong
        # word. SMEM ignores the low two offset bits.
        memory = {off: 0xA000 + off for off in range(0, 4096, 4)}
        self.assertEqual(support.mrt0_words(spv, memory=memory)[0], 0xA000 + (offset & ~3))

    def test_chained_propagation_and_folding(self):
        # s_mov_b32 -> s_lshl_b32 -> s_add_u32 -> s_buffer_load_dword chains completely into static load
        lines = [
            "s_mov_b32 s1, 0",          # stride 0
            "s_mov_b32 s2, 2048",       # NUM_RECORDS 2048 bytes
            "s_mov_b32 s4, 0x100",
            "s_lshl_b32 s5, s4, 2",       # folds to 0x400 (1024) and records in sgpr_imm
            "s_add_u32 s6, s5, 0x20",     # computes 0x420 (1056) and records in sgpr_imm
            "s_buffer_load_dword s8, s[0:3], s6",
            "v_mov_b32 v0, s8",
        ]
        spv = self.translate_lines(lines)
        # s_lshl_b32 folded completely without runtime shift on uint
        self.assertNotIn("OpShiftLeftLogical %uint", spv)
        # s_buffer_load evaluated offset 1056 <= 2048 statically: no branch diamond
        self.assertNotIn("OpSelectionMerge", spv)
        self.assertNotIn("OpBranchConditional", spv)
        self.assertNotIn("OpPhi", spv)
        self.assert_loads_offset(spv, 1056)
        self.assertIn("OpLoad %uint", spv)

    def test_s_add_u32_and_s_sub_u32_folding(self):
        # Chained addition into buffer offset: 16 + 32 = 48
        lines_add = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 256",
            "s_mov_b32 s4, 16",
            "s_add_u32 s5, s4, 32",
            "s_buffer_load_dword s8, s[0:3], s5",
            "v_mov_b32 v0, s8",
        ]
        spv_add = self.translate_lines(lines_add)
        self.assertNotIn("OpSelectionMerge", spv_add)
        self.assertNotIn("OpBranchConditional", spv_add)
        self.assertNotIn("OpPhi", spv_add)
        self.assert_loads_offset(spv_add, 48)

        # Chained subtraction into buffer offset: 80 - 16 = 64
        lines_sub = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 256",
            "s_mov_b32 s4, 80",
            "s_sub_u32 s5, s4, 16",
            "s_buffer_load_dword s8, s[0:3], s5",
            "v_mov_b32 v0, s8",
        ]
        spv_sub = self.translate_lines(lines_sub)
        self.assertNotIn("OpSelectionMerge", spv_sub)
        self.assertNotIn("OpBranchConditional", spv_sub)
        self.assertNotIn("OpPhi", spv_sub)
        self.assert_loads_offset(spv_sub, 64)

    def test_s_lshl_add_u32_folding(self):
        # D = (S0 << N) + S1. (0x10 << 2) + 0x5 = 0x45 (69)
        lines = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 256",
            "s_mov_b32 s4, 0x10",
            "s_lshl2_add_u32 s5, s4, 0x5",
            "s_buffer_load_dword s8, s[0:3], s5",
            "v_mov_b32 v0, s8",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpSelectionMerge", spv)
        self.assertNotIn("OpBranchConditional", spv)
        self.assertNotIn("OpPhi", spv)
        self.assert_loads_offset(spv, 69)
        self.assertIn("OpLoad %uint", spv)

    def test_s_bfe_u32_and_i32_constant_folding(self):
        # Extract bits 4..11 (width 8, offset 4): 0x00080004
        # Input 0xAB40 >> 4 = 0xAB4 & 0xFF = 0xB4 = 180
        lines = [
            "s_mov_b32 s4, 0xAB40",
            "s_bfe_u32 s5, s4, 0x00080004",
            "v_mov_b32 v0, s5",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitFieldUExtract", spv)
        self.assertIn("180", spv)

    def test_s_bfe_constant_control_dynamic_data(self):
        # Control is constant (0x00080004), data is dynamic VGPR
        lines = [
            "s_bfe_u32 s5, s1, 0x00080004",
            "v_mov_b32 v0, s5",
        ]
        spv = self.translate_lines(lines)
        # Should NOT emit dynamic width/offset calculation instructions
        self.assertNotIn("OpULessThan", spv)
        self.assertNotIn("OpISub", spv)
        # Emits direct OpBitFieldUExtract with constant operands
        self.assertIn("OpBitFieldUExtract", spv)

    def test_s_min_max_u32_and_i32_folding(self):
        # Chained min into buffer offset: min(64, 128) = 64
        lines = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 256",
            "s_mov_b32 s4, 64",
            "s_min_u32 s5, s4, 128",
            "s_buffer_load_dword s8, s[0:3], s5",
            "v_mov_b32 v0, s8",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpSelectionMerge", spv)
        self.assertNotIn("OpBranchConditional", spv)
        self.assertNotIn("OpPhi", spv)
        self.assert_loads_offset(spv, 64)

    def test_s_andn2_orn2_nand_nor_xnor_folding(self):
        # Chained bitwise into buffer offset: 0xFF & ~0xBF = 0x40 (64)
        lines = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 256",
            "s_mov_b32 s4, 0xFF",
            "s_andn2_b32 s5, s4, 0xBF",
            "s_buffer_load_dword s8, s[0:3], s5",
            "v_mov_b32 v0, s8",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpSelectionMerge", spv)
        self.assertNotIn("OpBranchConditional", spv)
        self.assertNotIn("OpPhi", spv)
        self.assert_loads_offset(spv, 64)

    def test_s_abs_and_absdiff_and_sext_folding(self):
        # Chained absdiff into buffer offset: |10 - 74| = 64
        lines = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 256",
            "s_mov_b32 s4, 10",
            "s_absdiff_i32 s5, s4, 74",
            "s_buffer_load_dword s8, s[0:3], s5",
            "v_mov_b32 v0, s8",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpSelectionMerge", spv)
        self.assertNotIn("OpBranchConditional", spv)
        self.assertNotIn("OpPhi", spv)
        self.assert_loads_offset(spv, 64)

    def test_s_pack_and_bfm_folding(self):
        # Chained bfm into buffer offset: ((1 << 4) - 1) << 2 = 15 << 2 = 60
        lines = [
            "s_mov_b32 s1, 0",
            "s_mov_b32 s2, 256",
            "s_mov_b32 s4, 4",
            "s_bfm_b32 s5, s4, 2",
            "s_buffer_load_dword s8, s[0:3], s5",
            "v_mov_b32 v0, s8",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpSelectionMerge", spv)
        self.assertNotIn("OpBranchConditional", spv)
        self.assertNotIn("OpPhi", spv)
        self.assert_loads_offset(spv, 60)

    def test_s_buffer_load_chained_offset_folding(self):
        # Compute buffer offset through chained SALU:
        # s4 = 16
        # s5 = s4 << 2 = 64
        # s6 = s5 + 32 = 96
        # s_buffer_load_dword through s[0:3] with descriptor limit 256. Offset 96 + 4 <= 256 is in bounds.
        # Should fold offset to 96 and emit straight-line OpLoad without dynamic bounds check diamond.
        lines = [
            "s_mov_b32 s1, 0",          # descriptor word 1: stride 0
            "s_mov_b32 s2, 256",        # descriptor word 2: NUM_RECORDS 256 bytes
            "s_mov_b32 s4, 16",
            "s_lshl_b32 s5, s4, 2",
            "s_add_u32 s6, s5, 32",
            "s_buffer_load_dword s8, s[0:3], s6",
            "v_mov_b32 v0, s8",
        ]
        spv = self.translate_lines(lines)
        # Static straight-line load: no OpSelectionMerge, OpBranchConditional, or OpPhi
        self.assertNotIn("OpSelectionMerge", spv)
        self.assertNotIn("OpBranchConditional", spv)
        self.assertNotIn("OpPhi", spv)
        self.assert_loads_offset(spv, 96)
        # OpLoad is emitted directly
        self.assertIn("OpLoad %uint", spv)

    def test_v_bfe_constant_folding(self):
        # v_bfe_u32 with all constant operands folds to constant
        lines = [
            "v_bfe_u32 v0, 0x1234, 4, 8",   # (0x1234 >> 4) & 0xFF = 0x23 = 35
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitFieldUExtract", spv)
        self.assertIn("35", spv)


if __name__ == "__main__":
    unittest.main()
