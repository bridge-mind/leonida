"""Semantic interpreter tests for bitfield insert (v_bfi_b32) with bitmask composition.

AMD RDNA2 VALU Instruction Set Architecture (ISA):
  v_bfi_b32 dst, src0, src1, src2
  Operation: dst = (src0 & src1) | (~src0 & src2)  (all 32-bit unsigned integers)
  - src0: bitmask indicating which bits to take from src1
  - src1: replacement field (bits where mask is 1)
  - src2: base value (bits where mask is 0 are preserved from base)

Bitmask composition:
  In graphics shaders and compute kernels, bitfield insertion (BFI) packs sub-register fields
  (e.g., color components, packed vertex attributes, compressed texture indices, bit flags)
  into a destination 32-bit dword. For any field of width W at offset O:
    mask = ((1 << W) - 1) << O
    field = (value << O) & mask
    result = v_bfi_b32(mask, field, base)

In SPIR-V (as emitted by tools/rdna2spirv.py):
  v_bfi_b32 translates to:
    %p = OpBitwiseAnd %uint %src0 %src1
    %n = OpNot %uint %src0
    %q = OpBitwiseAnd %uint %n %src2
    %r = OpBitwiseOr %uint %p %q

Standard library only: no GPU, no LLVM, no external tools, zero game data.
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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv
import spirv_mini_interp as si

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def ref_v_bfi_b32(src0_mask, src1_field, src2_base):
    """Reference implementation of RDNA2 v_bfi_b32 ALU instruction."""
    m = src0_mask & MASK32
    f = src1_field & MASK32
    b = src2_base & MASK32
    return ((m & f) | ((~m & MASK32) & b)) & MASK32


def make_mask(offset, width):
    """Generate a contiguous 32-bit bitmask of given width at given offset."""
    if width <= 0:
        return 0
    if width >= 32:
        return MASK32
    return (((1 << width) - 1) << offset) & MASK32


def compose_field(base, value, offset, width):
    """Canonical bitfield insertion using v_bfi_b32 semantics."""
    mk = make_mask(offset, width)
    fld = (value << offset) & mk
    return ref_v_bfi_b32(mk, fld, base)


def run_spv_with_not(spv_text):
    """Run SPIR-V assembly in spirv_mini_interp, mapping OpNot to OpISub (%c_all - %x)."""
    lines = spv_text.splitlines()
    fn_idx = next(i for i, ln in enumerate(lines) if "OpFunction" in ln)
    lines.insert(fn_idx, "%_c_all_ones = OpConstant %uint 4294967295")
    for i in range(fn_idx + 1, len(lines)):
        if "OpNot %uint" in lines[i]:
            parts = lines[i].split()
            res_id = parts[0]
            src_id = parts[4]
            lines[i] = f"{res_id} = OpISub %uint %_c_all_ones {src_id}"
    return si.run_fragment("\n".join(lines))


def translate_snippet(asm_body, name="ps_bfi_test"):
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


class BfiPureSemanticsTests(unittest.TestCase):
    """Test bitwise mathematical properties and boundaries of v_bfi_b32."""

    def test_bitwise_definition_random_vectors(self):
        vectors = [
            (0x00FF0000, 0x12345678, 0xA0B0C0D0, 0xA034C0D0),
            (0x000000FF, 0xAABBCCDD, 0x11223344, 0x112233DD),
            (0xFF000000, 0xAABBCCDD, 0x11223344, 0xAA223344),
            (0x0000FF00, 0xAABBCCDD, 0x11223344, 0x1122CC44),
            (0xF0F0F0F0, 0xAAAAAAAA, 0x55555555, 0xA5A5A5A5),
            (0x0F0F0F0F, 0xAAAAAAAA, 0x55555555, 0x5A5A5A5A),
            (0x12345678, 0xFFFFFFFF, 0x00000000, 0x12345678),
            (0x12345678, 0x00000000, 0xFFFFFFFF, 0xEDCBA987),
        ]
        for mask, fld, base, expected in vectors:
            with self.subTest(mask=hex(mask), fld=hex(fld), base=hex(base)):
                self.assertEqual(ref_v_bfi_b32(mask, fld, base), expected)

    def test_mask_zero_preserves_base_completely(self):
        for base in [0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0x13579BDF, 0x80000001]:
            with self.subTest(base=hex(base)):
                self.assertEqual(ref_v_bfi_b32(0, 0xCAFEBABE, base), base)

    def test_mask_all_ones_replaces_base_completely(self):
        for fld in [0x00000000, 0xFFFFFFFF, 0xDEADBEEF, 0x13579BDF, 0x80000001]:
            with self.subTest(fld=hex(fld)):
                self.assertEqual(ref_v_bfi_b32(MASK32, fld, 0xCAFEBABE), fld)

    def test_identity_when_field_equals_base(self):
        for val in [0x00000000, 0xFFFFFFFF, 0x12345678, 0x87654321, 0x55555555]:
            for mask in [0x00000000, 0xFFFFFFFF, 0xF0F0F0F0, 0x00FF00FF, 0xAAAAAAAA]:
                self.assertEqual(ref_v_bfi_b32(mask, val, val), val)

    def test_duality_with_inverted_mask(self):
        mask = 0x3CA55AC3
        s1 = 0xDEADBEEF
        s2 = 0x01234567
        inv_mask = (~mask) & MASK32
        self.assertEqual(ref_v_bfi_b32(mask, s1, s2), ref_v_bfi_b32(inv_mask, s2, s1))

    def test_single_bit_insertion_all_positions(self):
        base = 0x00000000
        for bit in range(32):
            mask = 1 << bit
            set_res = ref_v_bfi_b32(mask, MASK32, base)
            self.assertEqual(set_res, 1 << bit)
            clear_res = ref_v_bfi_b32(mask, 0, MASK32)
            self.assertEqual(clear_res, MASK32 ^ (1 << bit))

    def test_sign_bit_and_lsb_manipulation(self):
        msb_mask = 0x80000000
        lsb_mask = 0x00000001
        self.assertEqual(ref_v_bfi_b32(msb_mask, msb_mask, 0x7FFFFFFF), MASK32)
        self.assertEqual(ref_v_bfi_b32(msb_mask, 0, MASK32), 0x7FFFFFFF)
        self.assertEqual(ref_v_bfi_b32(lsb_mask, 0, 0xFFFFFFFF), 0xFFFFFFFE)
        self.assertEqual(ref_v_bfi_b32(lsb_mask, 1, 0xFFFFFFFE), 0xFFFFFFFF)

    def test_input_truncation_to_32bit(self):
        large_mask = 0x1000000FF  # 33-bit
        large_fld = 0x212345678   # 34-bit
        large_base = 0x3A0B0C0D0  # 34-bit
        expected = 0xA0B0C078
        self.assertEqual(ref_v_bfi_b32(large_mask, large_fld, large_base), expected)


class BitmaskCompositionTests(unittest.TestCase):
    """Test bitfield insert composition, packing workflows, and multi-field chaining."""

    def test_byte_packing_sequence(self):
        b0, b1, b2, b3 = 0x78, 0x56, 0x34, 0x12
        w0 = ref_v_bfi_b32(0x000000FF, b0, 0)
        w1 = ref_v_bfi_b32(0x0000FF00, b1 << 8, w0)
        w2 = ref_v_bfi_b32(0x00FF0000, b2 << 16, w1)
        w3 = ref_v_bfi_b32(0xFF000000, b3 << 24, w2)
        self.assertEqual(w3, 0x12345678)

    def test_in_place_byte_replacement(self):
        base = 0xAABBCCDD
        self.assertEqual(ref_v_bfi_b32(0x000000FF, 0x11, base), 0xAABBCC11)
        self.assertEqual(ref_v_bfi_b32(0x0000FF00, 0x22 << 8, base), 0xAABB22DD)
        self.assertEqual(ref_v_bfi_b32(0x00FF0000, 0x33 << 16, base), 0xAA33CCDD)
        self.assertEqual(ref_v_bfi_b32(0xFF000000, 0x44 << 24, base), 0x44BBCCDD)

    def test_rgb565_color_packing(self):
        r, g, b = 29, 58, 14
        expected_rgb565 = (r & 0x1F) | ((g & 0x3F) << 5) | ((b & 0x1F) << 11)
        c0 = compose_field(0, r, 0, 5)
        c1 = compose_field(c0, g, 5, 6)
        c2 = compose_field(c1, b, 11, 5)
        self.assertEqual(c2, expected_rgb565)

    def test_packed_normal_10_10_10_2(self):
        nx, ny, nz, nw = 0x2AA, 0x155, 0x3FF, 0x2
        expected = (nx & 0x3FF) | ((ny & 0x3FF) << 10) | ((nz & 0x3FF) << 20) | ((nw & 0x3) << 30)
        p0 = compose_field(0, nx, 0, 10)
        p1 = compose_field(p0, ny, 10, 10)
        p2 = compose_field(p1, nz, 20, 10)
        p3 = compose_field(p2, nw, 30, 2)
        self.assertEqual(p3, expected)

    def test_disjoint_masks_commutativity(self):
        mask_a = 0x000000FF
        mask_b = 0x00FF0000
        val_a = 0x000000AA
        val_b = 0x00BB0000
        base = 0x11223344
        order1 = ref_v_bfi_b32(mask_a, val_a, ref_v_bfi_b32(mask_b, val_b, base))
        order2 = ref_v_bfi_b32(mask_b, val_b, ref_v_bfi_b32(mask_a, val_a, base))
        self.assertEqual(order1, order2)
        self.assertEqual(order1, 0x11BB33AA)

    def test_idempotent_reinsertion(self):
        mask = 0x0FFF0000
        fld = 0x0ABC0000
        base = 0x76543210
        first = ref_v_bfi_b32(mask, fld, base)
        second = ref_v_bfi_b32(mask, fld, first)
        self.assertEqual(first, second)

    def test_all_widths_and_offsets(self):
        base = 0x5A5A5A5A
        pattern = 0xFFFFFFFF
        for width in range(1, 33):
            for offset in range(0, 33 - width):
                res = compose_field(base, pattern, offset, width)
                mask = make_mask(offset, width)
                self.assertEqual(res & mask, mask)
                self.assertEqual(res & ~mask & MASK32, base & ~mask & MASK32)


class SpirvEquivalenceTests(unittest.TestCase):
    """Test equivalence between v_bfi_b32 and SPIR-V OpBitFieldInsert / mini-interpreter."""

    def test_equivalence_with_spirv_op_bitfield_insert(self):
        test_cases = [
            (0, 4, 0x5, 0xFFFFFFF0),
            (4, 8, 0xAB, 0x12345678),
            (16, 16, 0xBEEF, 0x12345678),
            (8, 12, 0x789, 0xA0B0C0D0),
            (0, 32, 0xDEADBEEF, 0xCAFEBABE),
            (10, 0, 0x123, 0x87654321),
        ]
        for offset, count, insert, base in test_cases:
            with self.subTest(offset=offset, count=count):
                mk = make_mask(offset, count)
                field = (insert << offset) & mk
                bfi_res = ref_v_bfi_b32(mk, field, base)

                ins_masked = ((1 << count) - 1) if count < 32 else MASK32
                spirv_mk = ((1 << count) - 1) << offset if count < 32 else MASK32
                spirv_res = ((base & ~spirv_mk) | ((insert << offset) & spirv_mk)) & MASK32
                self.assertEqual(bfi_res, spirv_res)

    def test_mini_interp_handwritten_bitfield_insert(self):
        spv = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c_base = OpConstant %uint 4294967280
%c_ins = OpConstant %uint 1
%c_off = OpConstant %uint 0
%c_cnt = OpConstant %uint 4
%main = OpFunction %void None %fn
%entry = OpLabel
%res = OpBitFieldInsert %uint %c_base %c_ins %c_off %c_cnt
OpStore %out %res
OpReturn
OpFunctionEnd
"""
        out = si.run_fragment(spv)["%out"][0]
        self.assertEqual(out, 0xFFFFFFF1)
        mk = make_mask(0, 4)
        self.assertEqual(ref_v_bfi_b32(mk, 1, 0xFFFFFFF0), 0xFFFFFFF1)


class TranslatorIntegrationTests(unittest.TestCase):
    """Test end-to-end translation and execution of RDNA2 v_bfi_b32 shaders."""

    def test_translate_v_bfi_b32_emits_bitwise_quad(self):
        asm = """\
\tv_mov_b32 v1, 0x00ff0000
\tv_mov_b32 v2, 0x12345678
\tv_mov_b32 v3, 0xaaaaaaaa
\tv_bfi_b32 v0, v1, v2, v3
\texp mrt0 v0, v0, v0, v0 done vm
\ts_endpgm"""
        spv = translate_snippet(asm, "ps_bfi_emit")
        self.assertIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpNot %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)

    def test_execute_translated_v_bfi_b32_shader(self):
        asm = """\
\tv_mov_b32 v1, 0x000000ff
\tv_mov_b32 v2, 0x12345678
\tv_mov_b32 v3, 0xa0b0c0d0
\tv_bfi_b32 v0, v1, v2, v3
\texp mrt0 v0, v0, v0, v0 done vm
\ts_endpgm"""
        spv = translate_snippet(asm, "ps_bfi_exec")
        out = run_spv_with_not(spv)
        expected = 0xA0B0C078
        self.assertEqual(list(out["%out_mrt0"]), [expected, expected, expected, expected])

    def test_execute_multi_byte_bfi_pipeline(self):
        asm = """\
\tv_mov_b32 v0, 0
\tv_mov_b32 v1, 0x000000ff
\tv_mov_b32 v2, 0x78
\tv_bfi_b32 v0, v1, v2, v0
\tv_mov_b32 v3, 0x0000ff00
\tv_mov_b32 v4, 0x5600
\tv_bfi_b32 v0, v3, v4, v0
\tv_mov_b32 v5, 0x00ff0000
\tv_mov_b32 v6, 0x340000
\tv_bfi_b32 v0, v5, v6, v0
\tv_mov_b32 v7, 0xff000000
\tv_mov_b32 v8, 0x12000000
\tv_bfi_b32 v0, v7, v8, v0
\texp mrt0 v0, v0, v0, v0 done vm
\ts_endpgm"""
        spv = translate_snippet(asm, "ps_bfi_pipeline")
        out = run_spv_with_not(spv)
        expected = 0x12345678
        self.assertEqual(list(out["%out_mrt0"]), [expected, expected, expected, expected])


class NegativeControlsAndMutationTests(unittest.TestCase):
    """Test that deviations from v_bfi_b32 semantics are detected."""

    def test_sabotage_missing_not_fails(self):
        mask = 0x00FF0000
        fld = 0x12345678
        base = 0xA0B0C0D0
        correct = ref_v_bfi_b32(mask, fld, base)
        sabotaged = (mask & fld) | (mask & base)
        self.assertNotEqual(correct, sabotaged)

    def test_sabotage_swapped_field_and_base_fails(self):
        mask = 0x00FF0000
        fld = 0x12345678
        base = 0xA0B0C0D0
        correct = ref_v_bfi_b32(mask, fld, base)
        swapped = ref_v_bfi_b32(mask, base, fld)
        self.assertNotEqual(correct, swapped)

    def test_sabotage_and_instead_of_or_fails(self):
        mask = 0x00FF0000
        fld = 0x12345678
        base = 0xA0B0C0D0
        correct = ref_v_bfi_b32(mask, fld, base)
        sabotaged = ((mask & fld) & ((~mask & MASK32) & base)) & MASK32
        self.assertEqual(sabotaged, 0)
        self.assertNotEqual(correct, sabotaged)


if __name__ == "__main__":
    unittest.main()
