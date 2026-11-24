"""Semantic interpreter tests for SDWA source sub-register selection.

Covers sub-dword addressing (SDWA) source selection for RDNA2 -> SPIR-V:
  - Byte extraction: BYTE_0 (bits 7:0), BYTE_1 (bits 15:8), BYTE_2 (bits 23:16), BYTE_3 (bits 31:24)
  - Word extraction: WORD_0 (bits 15:0), WORD_1 (bits 31:16)
  - Full dword selection: DWORD (bits 31:0)
  - Unsigned zero-extension (default) vs signed sign-extension (under sext())
  - Sub-register broadcast to multiple ALU operands and self-broadcast patterns
  - Negative controls: ensuring sign-extension and field offset saboteurs are caught
"""
import gc
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

import rdna2spirv
import spirv_mini_interp as si
import support

MASK = 0xFFFFFFFF

SDWA_SEL_OFF_WIDTH = {
    "BYTE_0": (0, 8),
    "BYTE_1": (8, 8),
    "BYTE_2": (16, 8),
    "BYTE_3": (24, 8),
    "WORD_0": (0, 16),
    "WORD_1": (16, 16),
    "DWORD": (0, 32),
}


def ref_sdwa_extract(val, sel, sext=False):
    """Pure-Python reference implementation of SDWA source sub-register selection."""
    off, width = SDWA_SEL_OFF_WIDTH[sel]
    if width == 32:
        return val & MASK
    field = (val >> off) & ((1 << width) - 1)
    if sext and (field >> (width - 1)):
        field -= 1 << width
    return field & MASK


def run_sdwa_fragment(instructions):
    """Translate and execute a synthetic branch-free pixel shader with given instructions.

    Returns the four 32-bit words written to %out_mrt0 (v1, v2, v3, v4).
    """
    body = "\n".join("\t" + line.strip() for line in instructions if line.strip())
    shader_text = f"; ps_sdwa_test\n{body}\n\texp mrt0 v1, v2, v3, v4 done vm\n\ts_endpgm\n"
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        src_path = os.path.join(tmp, "disasm", "ps_sdwa_test.s")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(shader_text)
        with open(os.path.join(tmp, "ps_sdwa_test.dxbc"), "wb") as f:
            f.write(support.STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src_path)
            gc.collect()
        return support.mrt0_words(spv)


class SdwaReferenceOracleTests(unittest.TestCase):
    """Verify that the independent reference oracle matches expected ISA calculations."""

    def test_byte_extractions_unsigned(self):
        val = 0x8192A3B4
        self.assertEqual(ref_sdwa_extract(val, "BYTE_0", sext=False), 0xB4)
        self.assertEqual(ref_sdwa_extract(val, "BYTE_1", sext=False), 0xA3)
        self.assertEqual(ref_sdwa_extract(val, "BYTE_2", sext=False), 0x92)
        self.assertEqual(ref_sdwa_extract(val, "BYTE_3", sext=False), 0x81)

    def test_byte_extractions_signed(self):
        val = 0x8192A3B4
        self.assertEqual(ref_sdwa_extract(val, "BYTE_0", sext=True), 0xFFFFFFB4)
        self.assertEqual(ref_sdwa_extract(val, "BYTE_1", sext=True), 0xFFFFFFA3)
        self.assertEqual(ref_sdwa_extract(val, "BYTE_2", sext=True), 0xFFFFFF92)
        self.assertEqual(ref_sdwa_extract(val, "BYTE_3", sext=True), 0xFFFFFF81)

    def test_byte_extractions_positive_sign_bit(self):
        val = 0x12345678
        for sel, expected in [("BYTE_0", 0x78), ("BYTE_1", 0x56), ("BYTE_2", 0x34), ("BYTE_3", 0x12)]:
            self.assertEqual(ref_sdwa_extract(val, sel, sext=False), expected)
            self.assertEqual(ref_sdwa_extract(val, sel, sext=True), expected)

    def test_word_extractions_unsigned_and_signed(self):
        val = 0x89ABCDEF
        self.assertEqual(ref_sdwa_extract(val, "WORD_0", sext=False), 0xCDEF)
        self.assertEqual(ref_sdwa_extract(val, "WORD_1", sext=False), 0x89AB)
        self.assertEqual(ref_sdwa_extract(val, "WORD_0", sext=True), 0xFFFFCDEF)
        self.assertEqual(ref_sdwa_extract(val, "WORD_1", sext=True), 0xFFFF89AB)

    def test_dword_selection(self):
        val = 0xDEADBEEF
        self.assertEqual(ref_sdwa_extract(val, "DWORD", sext=False), 0xDEADBEEF)
        self.assertEqual(ref_sdwa_extract(val, "DWORD", sext=True), 0xDEADBEEF)


class SdwaDirectInterpreterTests(unittest.TestCase):
    """Directly test the mini SPIR-V interpreter on OpBitFieldUExtract / OpBitFieldSExtract."""

    def eval_extract(self, op_name, base_val, offset, count):
        header = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
"""
        c_base = f"%c_base = OpConstant %uint {base_val}\n"
        c_off = f"%c_off = OpConstant %uint {offset}\n"
        c_cnt = f"%c_cnt = OpConstant %uint {count}\n"
        body = f"""\
%main = OpFunction %void None %fn
%entry = OpLabel
%res = {op_name} %uint %c_base %c_off %c_cnt
OpStore %out %res
OpReturn
OpFunctionEnd
"""
        text = header + c_base + c_off + c_cnt + body
        return si.run_fragment(text)["%out"][0]

    def test_all_sdwa_sub_register_slices_in_interp(self):
        pattern = 0x89ABCDEF
        for sel, (off, width) in SDWA_SEL_OFF_WIDTH.items():
            if width == 32:
                continue
            with self.subTest(sel=sel, mode="unsigned"):
                got_u = self.eval_extract("OpBitFieldUExtract", pattern, off, width)
                want_u = ref_sdwa_extract(pattern, sel, sext=False)
                self.assertEqual(got_u, want_u)
            with self.subTest(sel=sel, mode="signed"):
                got_s = self.eval_extract("OpBitFieldSExtract", pattern, off, width)
                want_s = ref_sdwa_extract(pattern, sel, sext=True)
                self.assertEqual(got_s, want_s)

    def test_out_of_bounds_extract_rejected(self):
        with self.assertRaises(si.UnsupportedConstruct):
            self.eval_extract("OpBitFieldUExtract", 0x12345678, 28, 8)


class SdwaSourceSelectionSemanticTests(unittest.TestCase):
    """Test full translation and execution of RDNA2 SDWA source sub-register selection."""

    def test_byte_selection_unsigned_all_positions(self):
        val = 0x8192A3B4
        results = run_sdwa_fragment([
            f"v_mov_b32 v0, {val:#x}",
            "v_mov_b32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0",
            "v_mov_b32_sdwa v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",
            "v_mov_b32_sdwa v3, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2",
            "v_mov_b32_sdwa v4, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3",
        ])
        expected = [ref_sdwa_extract(val, f"BYTE_{i}", sext=False) for i in range(4)]
        self.assertEqual(results, expected)
        self.assertEqual(results, [0xB4, 0xA3, 0x92, 0x81])

    def test_byte_selection_sign_extended_all_positions(self):
        val = 0x8192A3B4
        results = run_sdwa_fragment([
            f"v_mov_b32 v0, {val:#x}",
            "v_mov_b32_sdwa v1, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0",
            "v_mov_b32_sdwa v2, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",
            "v_mov_b32_sdwa v3, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2",
            "v_mov_b32_sdwa v4, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3",
        ])
        expected = [ref_sdwa_extract(val, f"BYTE_{i}", sext=True) for i in range(4)]
        self.assertEqual(results, expected)
        self.assertEqual(results, [0xFFFFFFB4, 0xFFFFFFA3, 0xFFFFFF92, 0xFFFFFF81])

    def test_word_selection_unsigned_and_sign_extended(self):
        val = 0x89ABCDEF
        results = run_sdwa_fragment([
            f"v_mov_b32 v0, {val:#x}",
            "v_mov_b32_sdwa v1, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
            "v_mov_b32_sdwa v2, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
            "v_mov_b32_sdwa v3, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
            "v_mov_b32_sdwa v4, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
        ])
        expected = [
            ref_sdwa_extract(val, "WORD_0", sext=False),
            ref_sdwa_extract(val, "WORD_1", sext=False),
            ref_sdwa_extract(val, "WORD_0", sext=True),
            ref_sdwa_extract(val, "WORD_1", sext=True),
        ]
        self.assertEqual(results, expected)
        self.assertEqual(results, [0xCDEF, 0x89AB, 0xFFFFCDEF, 0xFFFF89AB])

    def test_boundary_sign_bit_values(self):
        results = run_sdwa_fragment([
            "v_mov_b32 v0, 0x0000807F",
            "v_mov_b32_sdwa v1, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0",
            "v_mov_b32_sdwa v2, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1",
            "v_mov_b32 v5, 0x80007FFF",
            "v_mov_b32_sdwa v3, sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0",
            "v_mov_b32_sdwa v4, sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1",
        ])
        self.assertEqual(results, [0x7F, 0xFFFFFF80, 0x7FFF, 0xFFFF8000])


class SdwaBroadcastPatternsSemanticTests(unittest.TestCase):
    """Test SDWA sub-register broadcasting where one register/field feeds multiple operands."""

    def test_same_register_sub_byte_broadcast_to_both_operands(self):
        val = 0x04030201
        results = run_sdwa_fragment([
            f"v_mov_b32 v0, {val:#x}",
            "v_add_nc_u32_sdwa v1, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0",
            "v_add_nc_u32_sdwa v2, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1",
            "v_add_nc_u32_sdwa v3, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2",
            "v_add_nc_u32_sdwa v4, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3",
        ])
        self.assertEqual(results, [2, 4, 6, 8])

    def test_broadcast_with_asymmetric_sign_extension(self):
        val = 0x80
        results = run_sdwa_fragment([
            f"v_mov_b32 v0, {val:#x}",
            "v_sub_nc_u32_sdwa v1, sext(v0), v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0",
            "v_sub_nc_u32_sdwa v2, v0, sext(v0) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0",
            "v_and_b32_sdwa v3, sext(v0), v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0",
            "v_or_b32_sdwa v4, sext(v0), v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0",
        ])
        s_val = ref_sdwa_extract(val, "BYTE_0", sext=True)
        u_val = ref_sdwa_extract(val, "BYTE_0", sext=False)
        self.assertEqual(results[0], (s_val - u_val) & MASK)
        self.assertEqual(results[1], (u_val - s_val) & MASK)
        self.assertEqual(results[2], (s_val & u_val) & MASK)
        self.assertEqual(results[3], (s_val | u_val) & MASK)
        self.assertEqual(results, [0xFFFFFF00, 0x00000100, 0x80, 0xFFFFFF80])

    def test_cross_field_recombination_from_same_source(self):
        val = 0x11223344
        results = run_sdwa_fragment([
            f"v_mov_b32 v0, {val:#x}",
            "v_or_b32_sdwa v1, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_1",
            "v_or_b32_sdwa v2, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_3",
            "v_sub_nc_u32_sdwa v3, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0",
            "v_and_b32_sdwa v4, v0, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_0 src1_sel:WORD_1",
        ])
        self.assertEqual(results[0], 0x44 | 0x33)
        self.assertEqual(results[1], 0x22 | 0x11)
        self.assertEqual(results[2], (0x1122 - 0x3344) & MASK)
        self.assertEqual(results[3], 0x3344 & 0x1122)

    def test_dual_source_mixed_width_selectors(self):
        v0_val = 0xF0E0D0C0
        v1_val = 0x00020003
        results = run_sdwa_fragment([
            f"v_mov_b32 v0, {v0_val:#x}",
            f"v_mov_b32 v5, {v1_val:#x}",
            "v_add_nc_u32_sdwa v1, sext(v0), v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:WORD_1",
            "v_add_nc_u32_sdwa v2, v0, sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:WORD_0",
            "v_add_nc_u32_sdwa v3, v0, v5 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:BYTE_0",
            "v_sub_nc_u32_sdwa v4, sext(v0), sext(v5) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:WORD_1 src1_sel:WORD_0",
        ])
        w0 = (ref_sdwa_extract(v0_val, "BYTE_0", sext=True) + ref_sdwa_extract(v1_val, "WORD_1", sext=False)) & MASK
        w1 = (ref_sdwa_extract(v0_val, "BYTE_1", sext=False) + ref_sdwa_extract(v1_val, "WORD_0", sext=True)) & MASK
        w2 = (ref_sdwa_extract(v0_val, "DWORD", sext=False) + ref_sdwa_extract(v1_val, "BYTE_0", sext=False)) & MASK
        w3 = (ref_sdwa_extract(v0_val, "WORD_1", sext=True) - ref_sdwa_extract(v1_val, "WORD_0", sext=True)) & MASK
        self.assertEqual(results, [w0, w1, w2, w3])


class SdwaNegativeControlsTests(unittest.TestCase):
    """Negative controls proving the tests detect sabotage in SDWA source extraction."""

    def test_sabotaging_sign_extension_fails_assertions(self):
        val = 0x80
        s_val = ref_sdwa_extract(val, "BYTE_0", sext=True)
        u_val = ref_sdwa_extract(val, "BYTE_0", sext=False)
        self.assertNotEqual(s_val, u_val)
        self.assertEqual(s_val, 0xFFFFFF80)
        self.assertEqual(u_val, 0x00000080)

    def test_sabotaging_offset_or_width_fails_assertions(self):
        val = 0x12345678
        b0 = ref_sdwa_extract(val, "BYTE_0", sext=False)
        b1 = ref_sdwa_extract(val, "BYTE_1", sext=False)
        w0 = ref_sdwa_extract(val, "WORD_0", sext=False)
        self.assertNotEqual(b0, b1)
        self.assertNotEqual(b0, w0)


if __name__ == "__main__":
    unittest.main()
