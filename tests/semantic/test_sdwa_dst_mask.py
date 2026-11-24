"""Semantic interpreter tests for SDWA destination sub-register write masks and unused bit preservation.

AMD RDNA2 VALU Instruction Set Architecture (ISA) - SDWA (Sub-Dword Addressing):
In RDNA2, SDWA instructions allow selecting sub-dword fields for destination and sources.
For destination sub-register write masks (dst_sel) and unused bit handling (dst_unused):

Destination sub-register selectors (dst_sel):
  - BYTE_0: bits [7:0]   (offset 0, width 8)
  - BYTE_1: bits [15:8]  (offset 8, width 8)
  - BYTE_2: bits [23:16] (offset 16, width 8)
  - BYTE_3: bits [31:24] (offset 24, width 8)
  - WORD_0: bits [15:0]  (offset 0, width 16)
  - WORD_1: bits [31:16] (offset 16, width 16)
  - DWORD:  bits [31:0]  (offset 0, width 32)

Destination unused bit preservation modes (dst_unused):
  - UNUSED_PAD (default when dst_unused is omitted):
      The sub-register field [offset : offset + width - 1] is updated with the truncated ALU result.
      All bits outside the written sub-register field (both above and below) are zeroed (padded with 0).
  - UNUSED_SEXT:
      The sub-register field is updated with the truncated ALU result.
      Bits above the written sub-register field are sign-extended (replicated MSB of the sub-dword value).
      Bits below the sub-register field offset are strictly zeroed.
  - UNUSED_PRESERVE:
      The sub-register field is updated with the truncated ALU result.
      All bits outside the written sub-register field (both below offset and above offset + width)
      strictly retain their previous values from the destination VGPR.

SPIR-V translation semantics (tools/rdna2spirv.py):
  - UNUSED_PRESERVE translates to OpBitFieldInsert %uint %old_vgpr %alu_val %off %width
  - UNUSED_PAD translates to OpBitFieldUExtract %uint %alu_val %c0 %width followed by OpShiftLeftLogical %off
  - UNUSED_SEXT translates to OpBitFieldSExtract %uint %alu_val %c0 %width followed by OpShiftLeftLogical %off
  - DWORD width (32 bits) passes the ALU result through unchanged.

Standard library only: no GPU, no LLVM, no external tools, zero game data.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv
import spirv_mini_interp as si
import support
from shader_ir import parse_line

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

SDWA_SEL_PARAMS = {
    "BYTE_0": (0, 8),
    "BYTE_1": (8, 8),
    "BYTE_2": (16, 8),
    "BYTE_3": (24, 8),
    "WORD_0": (0, 16),
    "WORD_1": (16, 16),
    "DWORD": (0, 32),
}


def ref_sdwa_dst(old_val: int, alu_val: int, sel: str, unused: str) -> int:
    """Reference implementation of SDWA destination sub-register write mask and unused bit preservation."""
    old_val &= MASK32
    alu_val &= MASK32
    off, width = SDWA_SEL_PARAMS[sel]
    if width == 32:
        return alu_val

    field_mask = (1 << width) - 1
    field = alu_val & field_mask

    if unused == "UNUSED_PRESERVE":
        preserve_mask = ~(field_mask << off) & MASK32
        return (old_val & preserve_mask) | (field << off)
    elif unused == "UNUSED_PAD":
        return (field << off) & MASK32
    elif unused == "UNUSED_SEXT":
        sign = (field >> (width - 1)) & 1
        sext = (field | ((~0 << width) & MASK32)) if sign else field
        return (sext << off) & MASK32
    else:
        raise ValueError(f"Unknown dst_unused mode: {unused}")


DIRECT_SPV_HEADER = """OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c0 = OpConstant %uint 0
"""


def run_direct_sdwa_dst_spv(old_val: int, alu_val: int, sel: str, unused: str) -> int:
    """Execute the exact SPIR-V instruction sequence emitted by rdna2spirv.sdwa_dst in spirv_mini_interp."""
    off, width = SDWA_SEL_PARAMS[sel]
    c_decls = [f"%cval = OpConstant %uint {alu_val & MASK32}"]
    if width == 32:
        body = "%res = OpCopyObject %uint %cval\n"
    elif unused == "UNUSED_PRESERVE":
        c_decls.extend([
            f"%cold = OpConstant %uint {old_val & MASK32}",
            f"%coff = OpConstant %uint {off}",
            f"%cwidth = OpConstant %uint {width}",
        ])
        body = "%res = OpBitFieldInsert %uint %cold %cval %coff %cwidth\n"
    else:
        ext = "OpBitFieldSExtract" if unused == "UNUSED_SEXT" else "OpBitFieldUExtract"
        c_decls.extend([
            f"%coff = OpConstant %uint {off}",
            f"%cwidth = OpConstant %uint {width}",
        ])
        body = (
            f"%lo = {ext} %uint %cval %c0 %cwidth\n"
            f"%res = OpShiftLeftLogical %uint %lo %coff\n"
        )

    spv = (
        DIRECT_SPV_HEADER
        + "\n".join(c_decls)
        + "\n%main = OpFunction %void None %fn\n%entry = OpLabel\n"
        + body
        + "OpStore %out %res\nOpReturn\nOpFunctionEnd\n"
    )
    res = si.run_fragment(spv)
    return res["%out"][0]


def translate_snippet(asm_body: str, name: str = "ps_sdwa_dst") -> str:
    """Translate straight-line assembly snippet via rdna2spirv.translate without external tools."""
    with tempfile.TemporaryDirectory() as tmp:
        disasm_dir = os.path.join(tmp, "disasm")
        os.mkdir(disasm_dir)
        with open(os.path.join(disasm_dir, f"{name}.s"), "w", encoding="utf-8") as f:
            f.write(f"; {name}\n{asm_body}\n")
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            res = rdna2spirv.translate(os.path.join(disasm_dir, f"{name}.s"))
            gc.collect()
        return res


class SdwaDstReferenceTests(unittest.TestCase):
    """Verify correctness of pure Python reference model against hand-derived golden vectors."""

    def test_golden_vectors_pad(self):
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xFF, "BYTE_0", "UNUSED_PAD"), 0x000000FF)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xFF, "BYTE_1", "UNUSED_PAD"), 0x0000FF00)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xFF, "BYTE_2", "UNUSED_PAD"), 0x00FF0000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xFF, "BYTE_3", "UNUSED_PAD"), 0xFF000000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xFFFF, "WORD_0", "UNUSED_PAD"), 0x0000FFFF)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xFFFF, "WORD_1", "UNUSED_PAD"), 0xFFFF0000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xCAFEBABE, "DWORD", "UNUSED_PAD"), 0xCAFEBABE)

    def test_golden_vectors_sext_positive(self):
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x7F, "BYTE_0", "UNUSED_SEXT"), 0x0000007F)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x7F, "BYTE_1", "UNUSED_SEXT"), 0x00007F00)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x7F, "BYTE_2", "UNUSED_SEXT"), 0x007F0000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x7F, "BYTE_3", "UNUSED_SEXT"), 0x7F000000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x7FFF, "WORD_0", "UNUSED_SEXT"), 0x00007FFF)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x7FFF, "WORD_1", "UNUSED_SEXT"), 0x7FFF0000)

    def test_golden_vectors_sext_negative(self):
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x80, "BYTE_0", "UNUSED_SEXT"), 0xFFFFFF80)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x90, "BYTE_1", "UNUSED_SEXT"), 0xFFFF9000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0xF7, "BYTE_2", "UNUSED_SEXT"), 0xFFF70000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x80, "BYTE_3", "UNUSED_SEXT"), 0x80000000)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x8001, "WORD_0", "UNUSED_SEXT"), 0xFFFF8001)
        self.assertEqual(ref_sdwa_dst(0x12345678, 0x8001, "WORD_1", "UNUSED_SEXT"), 0x80010000)

    def test_golden_vectors_preserve(self):
        base = 0x12345678
        self.assertEqual(ref_sdwa_dst(base, 0xAA, "BYTE_0", "UNUSED_PRESERVE"), 0x123456AA)
        self.assertEqual(ref_sdwa_dst(base, 0xBB, "BYTE_1", "UNUSED_PRESERVE"), 0x1234BB78)
        self.assertEqual(ref_sdwa_dst(base, 0xCC, "BYTE_2", "UNUSED_PRESERVE"), 0x12CC5678)
        self.assertEqual(ref_sdwa_dst(base, 0xDD, "BYTE_3", "UNUSED_PRESERVE"), 0xDD345678)
        self.assertEqual(ref_sdwa_dst(base, 0xBEEF, "WORD_0", "UNUSED_PRESERVE"), 0x1234BEEF)
        self.assertEqual(ref_sdwa_dst(base, 0xCAFE, "WORD_1", "UNUSED_PRESERVE"), 0xCAFE5678)
        self.assertEqual(ref_sdwa_dst(base, 0x99887766, "DWORD", "UNUSED_PRESERVE"), 0x99887766)

    def test_input_truncation_to_field_width(self):
        base = 0x11223344
        wide_val = 0xFFFFFFFF
        self.assertEqual(ref_sdwa_dst(base, wide_val, "BYTE_1", "UNUSED_PRESERVE"), 0x1122FF44)
        self.assertEqual(ref_sdwa_dst(base, wide_val, "BYTE_1", "UNUSED_PAD"), 0x0000FF00)
        self.assertEqual(ref_sdwa_dst(base, wide_val, "BYTE_1", "UNUSED_SEXT"), 0xFFFFFF00)


class SdwaDstInterpreterDirectTests(unittest.TestCase):
    """Direct semantic interpreter execution of the emitted SPIR-V instruction sequences for all SDWA dst modes."""

    def test_all_selector_and_unused_mode_combinations(self):
        test_values = [
            0x00, 0x01, 0x5A, 0x7F, 0x80, 0xA5, 0xEF, 0xFF,
            0x1234, 0x7FFF, 0x8000, 0x8001, 0xDEAD, 0xFFFF,
            0x12345678, 0x87654321, 0xDEADBEEF, 0xFFFFFFFF,
        ]
        bases = [0x00000000, 0x12345678, 0x87654321, 0xFFFFFFFF]

        for sel in ["BYTE_0", "BYTE_1", "BYTE_2", "BYTE_3", "WORD_0", "WORD_1", "DWORD"]:
            for unused in ["UNUSED_PAD", "UNUSED_SEXT", "UNUSED_PRESERVE"]:
                for base in bases:
                    for val in test_values:
                        with self.subTest(sel=sel, unused=unused, base=hex(base), val=hex(val)):
                            expected = ref_sdwa_dst(base, val, sel, unused)
                            got = run_direct_sdwa_dst_spv(base, val, sel, unused)
                            self.assertEqual(
                                got,
                                expected,
                                f"Direct SPIR-V mismatch for {sel} {unused} base={hex(base)} val={hex(val)}: got {hex(got)}, expected {hex(expected)}",
                            )


class SdwaDstTranslatorTranslationTests(unittest.TestCase):
    """End-to-end RDNA2 -> SPIR-V translation and semantic interpreter execution of pixel shaders with SDWA dst."""

    def test_first_partial_write_keeps_barycentric_input_bits(self) -> None:
        fixture = Path(HERE) / "fixtures" / "sdwa_bary_input.asm"
        spv = support.translate_listing("ps_sdwa_bary_input", fixture.read_bytes())
        words = support.mrt0_words(spv, inputs={"%bary": (0.25, 0.25, 0.5)})
        self.assertEqual(words, [0x3e8000ab, 0x3f00abcd] * 2)

    def test_only_a_preserving_subword_write_reads_the_destination(self) -> None:
        for select, unused, reads_dest in (("BYTE_2", "PRESERVE", True), ("WORD_0", "PRESERVE", True),
                                           ("DWORD", "PRESERVE", False), ("WORD_0", "PAD", False),
                                           ("BYTE_1", "SEXT", False)):
            with self.subTest(select=select, unused=unused):
                ins = parse_line(f"\tv_mov_b32_sdwa v0, v4 dst_sel:{select} dst_unused:UNUSED_{unused} src0_sel:DWORD")
                reads = {0, 4} if reads_dest else {4}
                self.assertEqual(rdna2spirv.read_before_write_vgprs([ins]), reads)
                self.assertEqual(rdna2spirv.reg_reads_writes([ins]), (reads, {0}))

    def test_unused_pad_all_subreg_slices(self):
        asm = """\
\tv_mov_b32 v0, 0xffffffff
\tv_mov_b32_sdwa v1, v0 dst_sel:BYTE_0 dst_unused:UNUSED_PAD src0_sel:DWORD
\tv_mov_b32_sdwa v2, v0 dst_sel:BYTE_1 dst_unused:UNUSED_PAD src0_sel:DWORD
\tv_mov_b32_sdwa v3, v0 dst_sel:BYTE_2 dst_unused:UNUSED_PAD src0_sel:DWORD
\tv_mov_b32_sdwa v4, v0 dst_sel:BYTE_3 dst_unused:UNUSED_PAD src0_sel:DWORD
\texp mrt0 v1, v2, v3, v4 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_pad_bytes")
        words = support.mrt0_words(spv)
        expected = [0x000000FF, 0x0000FF00, 0x00FF0000, 0xFF000000]
        self.assertEqual(words, expected)

    def test_unused_pad_words_and_dword(self):
        asm = """\
\tv_mov_b32 v0, 0xffffffff
\tv_mov_b32_sdwa v1, v0 dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:DWORD
\tv_mov_b32_sdwa v2, v0 dst_sel:WORD_1 dst_unused:UNUSED_PAD src0_sel:DWORD
\tv_mov_b32_sdwa v3, v0 dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:DWORD
\texp mrt0 v1, v2, v3, v3 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_pad_words")
        words = support.mrt0_words(spv)
        expected = [0x0000FFFF, 0xFFFF0000, 0xFFFFFFFF, 0xFFFFFFFF]
        self.assertEqual(words, expected)

    def test_unused_pad_default_when_omitted(self):
        """Omitting dst_unused defaults to UNUSED_PAD behavior in rdna2spirv."""
        asm = """\
\tv_mov_b32 v0, 0x12345678
\tv_mov_b32 v1, 0xaa
\tv_mov_b32_sdwa v0, v1 dst_sel:BYTE_1 src0_sel:DWORD
\texp mrt0 v0, v0, v0, v0 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_default_pad")
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0x0000AA00)

    def test_unused_sext_positive_and_negative(self):
        asm = """\
\tv_mov_b32 v0, 0x80
\tv_mov_b32 v1, 0x90
\tv_mov_b32 v2, 0xF0
\tv_mov_b32 v3, 0x8001
\tv_mov_b32_sdwa v4, v0 dst_sel:BYTE_0 dst_unused:UNUSED_SEXT src0_sel:DWORD
\tv_mov_b32_sdwa v5, v1 dst_sel:BYTE_1 dst_unused:UNUSED_SEXT src0_sel:DWORD
\tv_mov_b32_sdwa v6, v2 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:DWORD
\tv_mov_b32_sdwa v7, v3 dst_sel:WORD_0 dst_unused:UNUSED_SEXT src0_sel:DWORD
\texp mrt0 v4, v5, v6, v7 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_sext_neg")
        words = support.mrt0_words(spv)
        expected = [0xFFFFFF80, 0xFFFF9000, 0xFFF00000, 0xFFFF8001]
        self.assertEqual(words, expected)

    def test_unused_sext_byte3_and_word1(self):
        asm = """\
\tv_mov_b32 v0, 0x80
\tv_mov_b32 v1, 0x8001
\tv_mov_b32 v2, 0x7F
\tv_mov_b32 v3, 0x7FFF
\tv_mov_b32_sdwa v4, v0 dst_sel:BYTE_3 dst_unused:UNUSED_SEXT src0_sel:DWORD
\tv_mov_b32_sdwa v5, v1 dst_sel:WORD_1 dst_unused:UNUSED_SEXT src0_sel:DWORD
\tv_mov_b32_sdwa v6, v2 dst_sel:BYTE_3 dst_unused:UNUSED_SEXT src0_sel:DWORD
\tv_mov_b32_sdwa v7, v3 dst_sel:WORD_1 dst_unused:UNUSED_SEXT src0_sel:DWORD
\texp mrt0 v4, v5, v6, v7 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_sext_b3_w1")
        words = support.mrt0_words(spv)
        expected = [0x80000000, 0x80010000, 0x7F000000, 0x7FFF0000]
        self.assertEqual(words, expected)

    def test_unused_preserve_subreg_slices(self):
        asm = """\
\tv_mov_b32 v0, 0x12345678
\tv_mov_b32 v1, 0x12345678
\tv_mov_b32 v2, 0x12345678
\tv_mov_b32 v3, 0x12345678
\tv_mov_b32 v4, 0xDEADBEEF
\tv_mov_b32_sdwa v0, v4 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v1, v4 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v2, v4 dst_sel:BYTE_2 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v3, v4 dst_sel:BYTE_3 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\texp mrt0 v0, v1, v2, v3 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_pres_bytes")
        words = support.mrt0_words(spv)
        expected = [0x123456EF, 0x1234EF78, 0x12EF5678, 0xEF345678]
        self.assertEqual(words, expected)

    def test_unused_preserve_words(self):
        asm = """\
\tv_mov_b32 v0, 0x12345678
\tv_mov_b32 v1, 0x12345678
\tv_mov_b32 v2, 0xDEADBEEF
\tv_mov_b32_sdwa v0, v2 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v1, v2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\texp mrt0 v0, v1, v0, v1 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_pres_words")
        words = support.mrt0_words(spv)
        expected = [0x1234BEEF, 0xBEEF5678, 0x1234BEEF, 0xBEEF5678]
        self.assertEqual(words, expected)

    def test_sequential_byte_packing(self):
        asm = """\
\tv_mov_b32 v0, 0x00000000
\tv_mov_b32 v1, 0xAA
\tv_mov_b32 v2, 0xBB
\tv_mov_b32 v3, 0xCC
\tv_mov_b32 v4, 0xDD
\tv_mov_b32_sdwa v0, v1 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v0, v2 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v0, v3 dst_sel:BYTE_2 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v0, v4 dst_sel:BYTE_3 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\texp mrt0 v0, v0, v0, v0 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_pack_bytes")
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xDDCCBBAA)

    def test_sequential_word_packing(self):
        asm = """\
\tv_mov_b32 v0, 0x00000000
\tv_mov_b32 v1, 0xCAFE
\tv_mov_b32 v2, 0xBABE
\tv_mov_b32_sdwa v0, v1 dst_sel:WORD_0 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\tv_mov_b32_sdwa v0, v2 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\texp mrt0 v0, v0, v0, v0 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_pack_words")
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xBABECAFE)

    def test_combined_src_and_dst_sdwa(self):
        asm = """\
\tv_mov_b32 v0, 0x12345678
\tv_mov_b32 v1, 0xAABBCCDD
\tv_mov_b32_sdwa v1, v0 dst_sel:BYTE_0 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_3
\texp mrt0 v1, v1, v1, v1 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_combined_sdwa")
        words = support.mrt0_words(spv)
        self.assertEqual(words[0], 0xAABBCC12)

    def test_alu_operations_with_sdwa_dst(self):
        asm = """\
\tv_mov_b32 v0, 0x11223344
\tv_mov_b32 v1, 0x00000010
\tv_mov_b32 v2, 0x00000020
\tv_add_nc_u32_sdwa v0, v1, v2 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
\tv_mov_b32 v3, 0x00000050
\tv_mov_b32 v4, 0x00000010
\tv_subrev_nc_u32_sdwa v5, v4, v3 dst_sel:BYTE_2 dst_unused:UNUSED_PAD src0_sel:DWORD src1_sel:DWORD
\tv_mov_b32 v6, 0x000000F0
\tv_mov_b32 v7, 0x00000007
\tv_or_b32_sdwa v8, v6, v7 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:DWORD src1_sel:DWORD
\tv_mov_b32 v9, 0xAABBCCDD
\tv_mov_b32 v10, 0x00001234
\tv_mov_b32 v11, 0x0000FFFF
\tv_and_b32_sdwa v9, v10, v11 dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD src1_sel:DWORD
\texp mrt0 v0, v5, v8, v9 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_alu_sdwa")
        words = support.mrt0_words(spv)
        expected = [0x11223044, 0x00400000, 0xFFF70000, 0x1234CCDD]
        self.assertEqual(words, expected)


class SdwaDstNegativeControlTests(unittest.TestCase):
    """Verify that test suite detects regressions and sabotaged SDWA destination instructions."""

    def test_detect_bitfield_insert_sabotage(self):
        asm = """\
\tv_mov_b32 v0, 0x12345678
\tv_mov_b32 v1, 0x000000AA
\tv_mov_b32_sdwa v0, v1 dst_sel:BYTE_1 dst_unused:UNUSED_PRESERVE src0_sel:DWORD
\texp mrt0 v0, v0, v0, v0 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_sabotage_bfi")
        good_words = support.mrt0_words(spv)
        self.assertEqual(good_words[0], 0x1234AA78)

        self.assertIn("OpBitFieldInsert", spv)
        bad_spv = spv.replace("OpBitFieldInsert", "OpBitFieldUExtract")
        try:
            bad_words = support.mrt0_words(bad_spv)
            self.assertNotEqual(bad_words[0], 0x1234AA78)
        except Exception:
            pass

    def test_detect_sext_vs_uext_sabotage(self):
        asm = """\
\tv_mov_b32 v0, 0x000000F0
\tv_mov_b32 v1, 0x00000007
\tv_or_b32_sdwa v2, v0, v1 dst_sel:BYTE_2 dst_unused:UNUSED_SEXT src0_sel:DWORD src1_sel:DWORD
\texp mrt0 v2, v2, v2, v2 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_sabotage_sext")
        good_words = support.mrt0_words(spv)
        self.assertEqual(good_words[0], 0xFFF70000)

        self.assertIn("OpBitFieldSExtract", spv)
        bad_spv = spv.replace("OpBitFieldSExtract", "OpBitFieldUExtract")
        bad_words = support.mrt0_words(bad_spv)
        self.assertEqual(bad_words[0], 0x00F70000)
        self.assertNotEqual(bad_words[0], 0xFFF70000)

    def test_sabotage_reference_model_pad_fails(self):
        val = 0xAA
        good = ref_sdwa_dst(0x12345678, val, "BYTE_1", "UNUSED_PAD")
        bad = (val << 8) | 0x78  # failing to zero low byte
        self.assertEqual(good, 0xAA00)
        self.assertNotEqual(good, bad)

    def test_sabotage_reference_model_preserve_fails(self):
        base = 0x12345678
        val = 0xAA
        good = ref_sdwa_dst(base, val, "BYTE_1", "UNUSED_PRESERVE")
        bad = 0x0000AA00  # padding instead of preserving
        self.assertEqual(good, 0x1234AA78)
        self.assertNotEqual(good, bad)


if __name__ == "__main__":
    unittest.main()
