"""Semantic interpreter tests for RDNA2 bitfield extract instructions (v_bfe_u32, v_bfe_i32).

Verifies translation via tools/rdna2spirv.py and execution via tests/semantic/spirv_mini_interp.py,
as well as direct SPIR-V OpBitFieldUExtract and OpBitFieldSExtract interpreter edge cases:
  * Signed vs unsigned bitfield extraction
  * Width = 0 edge cases (must return 0 for both signed and unsigned)
  * Single-bit extractions across all 32 lane bit positions
  * Width and offset masking modulo 32 per RDNA ISA (offset & 31, width & 31)
  * Exact boundary conditions (offset + width == 32)
  * Fields that run past bit 31 (offset + width > 32): the field stops at bit 31, as LLVM's constantFoldBFE
    computes (Src0 >> Offset, arithmetic for v_bfe_i32), and the translation keeps the SPIR-V count in range
  * s_bfe_u32 / s_bfe_i32, whose width (bits 22:16 of src1) goes up to 127
  * Rejection of out-of-bounds bitfields (offset + width > 32, undefined in SPIR-V)
  * Comparison against an independent plain-Python reference oracle
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
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

SPIRV_HEADER = """\
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


def ref_bfe_u32(base: int, offset: int, width: int) -> int:
    """RDNA2 v_bfe_u32 reference oracle: unsigned bitfield extract."""
    base &= MASK32
    offset &= 31
    width &= 31
    if width == 0:
        return 0
    return ((base >> offset) & ((1 << width) - 1)) & MASK32


def ref_bfe_i32(base: int, offset: int, width: int) -> int:
    """RDNA2 v_bfe_i32 reference oracle: signed bitfield extract."""
    base &= MASK32
    offset &= 31
    width &= 31
    if width == 0:
        return 0
    signed = base - (1 << 32) if base >> 31 else base
    # Arithmetic shift: a field that runs past bit 31 is filled with the sign bit.
    field = (signed >> offset) & ((1 << width) - 1)
    if field & (1 << (width - 1)):
        field -= 1 << width
    return field & MASK32


_TMP_DIR = None
_SRC_PATH = None


def setUpModule():
    global _TMP_DIR, _SRC_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(disasm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_bfe.s")
    dxbc_path = os.path.join(_TMP_DIR.name, "ps_bfe.dxbc")
    with open(dxbc_path, "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def run_rdna_bfe(op: str, base: int, offset: int, width: int, use_vgpr: bool = True) -> int:
    """Translate and execute a synthetic pixel shader exercising v_bfe_u32 or v_bfe_i32."""
    global _TMP_DIR, _SRC_PATH
    if _TMP_DIR is None:
        setUpModule()

    if use_vgpr:
        asm_content = (
            "; test_bfe\n"
            f"\tv_mov_b32 v1, {base & MASK32:#x}\n"
            f"\tv_mov_b32 v2, {offset:#x}\n"
            f"\tv_mov_b32 v3, {width:#x}\n"
            f"\t{op} v0, v1, v2, v3\n"
            "\texp mrt0 v0, v0, v0, v0 done vm\n"
            "\ts_endpgm\n"
        )
    else:
        asm_content = (
            "; test_bfe\n"
            f"\tv_mov_b32 v1, {base & MASK32:#x}\n"
            f"\t{op} v0, v1, {offset}, {width}\n"
            "\texp mrt0 v0, v0, v0, v0 done vm\n"
            "\ts_endpgm\n"
        )

    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm_content)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)

    out = si.run_fragment(spv)
    return out["%out_mrt0"][0]


def run_spirv_bfe(opcode: str, base: int, offset: int, width: int) -> int:
    """Execute direct OpBitFieldUExtract or OpBitFieldSExtract via spirv_mini_interp."""
    body = (
        SPIRV_HEADER
        + f"%c_base = OpConstant %uint {base & MASK32}\n"
        + f"%c_off = OpConstant %uint {offset}\n"
        + f"%c_wid = OpConstant %uint {width}\n"
        + "%main = OpFunction %void None %fn\n"
        + "%entry = OpLabel\n"
        + f"%res = {opcode} %uint %c_base %c_off %c_wid\n"
        + "OpStore %out %res\n"
        + "OpReturn\n"
        + "OpFunctionEnd\n"
    )
    return si.run_fragment(body)["%out"][0]


class AluBfeSemanticTests(unittest.TestCase):
    def test_reference_oracle_basics(self):
        self.assertEqual(ref_bfe_u32(0x12345678, 4, 8), 0x67)
        self.assertEqual(ref_bfe_i32(0xFFFFFFF0, 2, 4), 0xFFFFFFFC)
        self.assertEqual(ref_bfe_u32(0xFFFFFFFF, 0, 4), 0x0000000F)
        self.assertEqual(ref_bfe_i32(0xFFFFFFFF, 0, 4), 0xFFFFFFFF)
        self.assertEqual(ref_bfe_u32(0xFFFFFFFF, 0, 0), 0)
        self.assertEqual(ref_bfe_i32(0xFFFFFFFF, 0, 0), 0)
        self.assertEqual(ref_bfe_u32(0x80000000, 31, 1), 1)
        self.assertEqual(ref_bfe_i32(0x80000000, 31, 1), 0xFFFFFFFF)
        self.assertEqual(ref_bfe_u32(0x7FFFFFFF, 31, 1), 0)
        self.assertEqual(ref_bfe_i32(0x7FFFFFFF, 31, 1), 0)

    def test_v_bfe_u32_basic_extractions(self):
        cases = [
            (0x12345678, 4, 8, 0x67),
            (0xDEADBEEF, 8, 12, 0xDBE),
            (0xABCD1234, 16, 16, 0xABCD),
            (0xFEDCBA98, 0, 4, 0x8),
            (0xFEDCBA98, 0, 8, 0x98),
            (0xFEDCBA98, 0, 16, 0xBA98),
            (0x87654321, 28, 4, 0x8),
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(run_rdna_bfe("v_bfe_u32", base, off, wid, use_vgpr=True), expected)
                self.assertEqual(run_rdna_bfe("v_bfe_u32", base, off, wid, use_vgpr=False), expected)

    def test_v_bfe_u32_zero_extension_never_sign_extends(self):
        cases = [
            (0xFFFFFFFF, 0, 4, 0x0000000F),
            (0x80000000, 28, 4, 0x00000008),
            (0x80000000, 31, 1, 0x00000001),
            (0x00008000, 12, 4, 0x00000008),
            (0xFFFFFF00, 8, 8, 0x000000FF),
            (0x00800000, 16, 8, 0x00000080),
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(run_rdna_bfe("v_bfe_u32", base, off, wid), expected)

    def test_v_bfe_u32_width_zero_gives_zero(self):
        for off in (0, 1, 7, 15, 16, 31):
            with self.subTest(off=off):
                self.assertEqual(run_rdna_bfe("v_bfe_u32", 0xFFFFFFFF, off, 0), 0)
                self.assertEqual(run_rdna_bfe("v_bfe_u32", 0x12345678, off, 0), 0)

    def test_v_bfe_i32_positive_fields_zero_extended(self):
        cases = [
            (0x76543210, 28, 4, 0x00000007),
            (0x0000007F, 0, 8, 0x0000007F),
            (0x00007FFF, 0, 16, 0x00007FFF),
            (0x55555555, 2, 2, 0x00000001),
            (0x12345678, 4, 4, 0x00000007),
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(run_rdna_bfe("v_bfe_i32", base, off, wid, use_vgpr=True), expected)
                self.assertEqual(run_rdna_bfe("v_bfe_i32", base, off, wid, use_vgpr=False), expected)

    def test_v_bfe_i32_negative_fields_sign_extended(self):
        cases = [
            (0xFFFFFFF0, 2, 4, 0xFFFFFFFC),  # 0b1100 -> -4 (matches ps_ops fixture)
            (0x00000080, 0, 8, 0xFFFFFF80),  # 0x80 -> -128
            (0x000000FF, 0, 8, 0xFFFFFFFF),  # 0xFF -> -1
            (0x00008000, 0, 16, 0xFFFF8000),  # 0x8000 -> -32768
            (0x0000FFFF, 0, 16, 0xFFFFFFFF),  # 0xFFFF -> -1
            (0x80000000, 31, 1, 0xFFFFFFFF),  # 1-bit -> -1
            (0x00000001, 0, 1, 0xFFFFFFFF),  # 1-bit -> -1
            (0xAAAAAAAA, 2, 2, 0xFFFFFFFE),  # 0b10 -> -2
        ]
        for base, off, wid, expected in cases:
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(run_rdna_bfe("v_bfe_i32", base, off, wid, use_vgpr=True), expected)
                self.assertEqual(run_rdna_bfe("v_bfe_i32", base, off, wid, use_vgpr=False), expected)

    def test_v_bfe_i32_width_zero_gives_zero(self):
        for off in (0, 1, 7, 15, 16, 31):
            with self.subTest(off=off):
                self.assertEqual(run_rdna_bfe("v_bfe_i32", 0xFFFFFFFF, off, 0), 0)
                self.assertEqual(run_rdna_bfe("v_bfe_i32", 0x80000000, off, 0), 0)

    def test_operand_masking_mod_32(self):
        self.assertEqual(run_rdna_bfe("v_bfe_u32", 0x12345678, 36, 8), 0x67)  # 36 & 31 = 4
        self.assertEqual(run_rdna_bfe("v_bfe_u32", 0x12345678, 4, 40), 0x67)  # 40 & 31 = 8
        self.assertEqual(run_rdna_bfe("v_bfe_u32", 0x12345678, 4, 32), 0)  # 32 & 31 = 0
        self.assertEqual(run_rdna_bfe("v_bfe_i32", 0xFFFFFFF0, 34, 4), 0xFFFFFFFC)  # 34 & 31 = 2
        self.assertEqual(run_rdna_bfe("v_bfe_i32", 0xFFFFFFF0, 2, 36), 0xFFFFFFFC)  # 36 & 31 = 4
        self.assertEqual(run_rdna_bfe("v_bfe_i32", 0xFFFFFFF0, 2, 32), 0)  # 32 & 31 = 0

    def test_boundary_offset_plus_width_32(self):
        # When offset + width == 32, exactly 32 bits are spanned; must be valid in SPIR-V.
        self.assertEqual(run_rdna_bfe("v_bfe_u32", 0xFFFFFFFF, 0, 31), 0x7FFFFFFF)
        self.assertEqual(run_rdna_bfe("v_bfe_u32", 0xFFFFFFFF, 1, 31), 0x7FFFFFFF)
        self.assertEqual(run_rdna_bfe("v_bfe_u32", 0x12345678, 16, 16), 0x1234)
        self.assertEqual(run_rdna_bfe("v_bfe_u32", 0x80000000, 31, 1), 1)

        self.assertEqual(run_rdna_bfe("v_bfe_i32", 0xFFFFFFFF, 0, 31), 0xFFFFFFFF)
        self.assertEqual(run_rdna_bfe("v_bfe_i32", 0x7FFFFFFF, 1, 31), 0x3FFFFFFF)
        self.assertEqual(run_rdna_bfe("v_bfe_i32", 0x80000000, 1, 31), 0xC0000000)
        self.assertEqual(run_rdna_bfe("v_bfe_i32", 0x80000000, 31, 1), 0xFFFFFFFF)

    def test_out_of_bounds_rejected_by_interpreter(self):
        # SPIR-V specification declares offset + count > 32 undefined.
        # spirv_mini_interp strictly rejects undefined constructs with UnsupportedConstruct.
        invalid_combinations = [
            (16, 17),  # 16 + 17 = 33
            (31, 2),   # 31 + 2 = 33
            (20, 20),  # 20 + 20 = 40
            (1, 32),   # 1 + 32 = 33 (in direct SPIR-V)
        ]
        for off, wid in invalid_combinations:
            with self.subTest(off=off, wid=wid):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_spirv_bfe("OpBitFieldUExtract", 0xFFFFFFFF, off, wid)
                with self.assertRaises(si.UnsupportedConstruct):
                    run_spirv_bfe("OpBitFieldSExtract", 0xFFFFFFFF, off, wid)

    def test_all_single_bit_positions(self):
        # Test every single bit position (offset = 0..31, width = 1)
        for pos in range(32):
            base = 1 << pos
            with self.subTest(pos=pos):
                u_res = run_rdna_bfe("v_bfe_u32", base, pos, 1)
                i_res = run_rdna_bfe("v_bfe_i32", base, pos, 1)
                self.assertEqual(u_res, 1)
                self.assertEqual(i_res, 0xFFFFFFFF)

                u_zero = run_rdna_bfe("v_bfe_u32", 0, pos, 1)
                i_zero = run_rdna_bfe("v_bfe_i32", 0, pos, 1)
                self.assertEqual(u_zero, 0)
                self.assertEqual(i_zero, 0)

    def test_sweep_against_reference_oracle(self):
        test_bases = [
            0x00000000,
            0xFFFFFFFF,
            0x12345678,
            0xAAAAAAAA,
            0x80000000,
        ]
        test_offsets = [0, 1, 4, 16, 20, 31]
        test_widths = [0, 1, 4, 8, 16, 17, 31]

        for base in test_bases:
            for off in test_offsets:
                for wid in test_widths:
                    exp_u = ref_bfe_u32(base, off, wid)
                    exp_i = ref_bfe_i32(base, off, wid)
                    got_u = run_rdna_bfe("v_bfe_u32", base, off, wid)
                    got_i = run_rdna_bfe("v_bfe_i32", base, off, wid)
                    self.assertEqual(got_u, exp_u, f"v_bfe_u32 mismatch: base={base:#x}, off={off}, wid={wid}")
                    self.assertEqual(got_i, exp_i, f"v_bfe_i32 mismatch: base={base:#x}, off={off}, wid={wid}")

    def test_field_past_bit_31_stops_there(self):
        # offset + width > 32: the bits past 31 are zero (u32) or the sign bit (i32), not undefined.
        for base, off, wid, want_u, want_i in (
            (0xF0000000, 28, 8, 0xF, 0xFFFFFFFF),
            (0x80000000, 31, 2, 1, 0xFFFFFFFF),
            (0x7FFFFFFF, 31, 2, 0, 0),
            (0x12345678, 16, 17, 0x1234, 0x1234),
            (0xABCD0000, 20, 20, 0xABC, 0xFFFFFABC),
        ):
            with self.subTest(base=hex(base), off=off, wid=wid):
                self.assertEqual(ref_bfe_u32(base, off, wid), want_u)
                self.assertEqual(ref_bfe_i32(base, off, wid), want_i)
                self.assertEqual(run_rdna_bfe("v_bfe_u32", base, off, wid), want_u)
                self.assertEqual(run_rdna_bfe("v_bfe_i32", base, off, wid), want_i)
                self.assertEqual(run_rdna_bfe("v_bfe_u32", base, off, wid, use_vgpr=False), want_u)
                self.assertEqual(run_rdna_bfe("v_bfe_i32", base, off, wid, use_vgpr=False), want_i)


def run_rdna_s_bfe(op: str, base: int, offset: int, width: int) -> int:
    """Translate and execute s_bfe_u32 / s_bfe_i32 with src1 = offset | width << 16, return the result."""
    if _TMP_DIR is None:
        setUpModule()
    asm_content = (
        "; test_s_bfe\n"
        f"\ts_mov_b32 s1, {base & MASK32:#x}\n"
        f"\ts_mov_b32 s2, {(offset & 31) | (width & 0x7F) << 16:#x}\n"
        f"\t{op} s0, s1, s2\n"
        "\tv_mov_b32 v0, s0\n"
        "\texp mrt0 v0, v0, v0, v0 done vm\n"
        "\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8") as f:
        f.write(asm_content)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_SRC_PATH)
    return si.run_fragment(spv)["%out_mrt0"][0]


def ref_s_bfe(signed: bool, base: int, offset: int, width: int) -> int:
    """s_bfe reference: the v_bfe field, with a 7-bit width; 32 or more takes every bit from offset up."""
    width = min(width & 0x7F, 32)
    if width == 32:
        if offset == 0:
            return base & MASK32
        width = 32 - offset   # same field: it is cut at bit 31 anyway
    return (ref_bfe_i32 if signed else ref_bfe_u32)(base, offset, width)


class ScalarBfeTests(unittest.TestCase):
    def test_s_bfe_in_range_and_past_bit_31(self):
        for base in (0x12345678, 0x80000000, 0xFEDCBA98):
            for off in (0, 4, 16, 28, 31):
                for wid in (0, 1, 8, 16, 17, 31, 32, 64, 127):
                    for op, signed in (("s_bfe_u32", False), ("s_bfe_i32", True)):
                        with self.subTest(op=op, base=hex(base), off=off, wid=wid):
                            self.assertEqual(run_rdna_s_bfe(op, base, off, wid), ref_s_bfe(signed, base, off, wid))


if __name__ == "__main__":
    unittest.main()
