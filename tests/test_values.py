"""Computed values for the instruction families listed in rdna2spirv.py's "Supported" docstring.

Each case runs a tiny pixel shader through the translator and the SPIR-V interpreter (tests/spirv_interp.py) and
compares v3 with a reference written in plain Python from the RDNA2 ISA definition of the instruction. No GPU,
llvm-mc or spirv-tools needed. These tests check values; spirv-val validity is covered by test_translate.py.
"""
import math
import struct

import pytest

import spirv_interp as si
from listing import translate_listing

M = 0xFFFFFFFF


def fb(x):
    return si.f2u(si.f32(x))


def run_v3(tmp_path, asm, v1=0, v2=0, v4=0):
    pre = f"v_mov_b32 v1, {v1:#x}\nv_mov_b32 v2, {v2:#x}\nv_mov_b32 v4, {v4:#x}\n"
    return si.run(translate_listing(tmp_path, pre + asm)).vgpr[3]


def s(x):
    return x & M


# id: (asm, v1, v2, v4, expected bits of v3)
CASES = {
    # float ALU
    "v_add_f32":   ("v_add_f32 v3, v1, v2", fb(1.5), fb(2.25), 0, fb(3.75)),
    "v_sub_f32":   ("v_sub_f32 v3, v1, v2", fb(1.5), fb(2.25), 0, fb(-0.75)),
    "v_subrev_f32": ("v_subrev_f32 v3, v1, v2", fb(1.5), fb(2.25), 0, fb(0.75)),
    "v_mul_f32":   ("v_mul_f32 v3, v1, v2", fb(1.5), fb(-2.0), 0, fb(-3.0)),
    "v_fma_f32":   ("v_fma_f32 v3, v1, v2, v4", fb(1.5), fb(2.0), fb(0.25), fb(3.25)),
    "v_max_f32":   ("v_max_f32 v3, v1, v2", fb(1.5), fb(-2.0), 0, fb(1.5)),
    "v_min_f32":   ("v_min_f32 v3, v1, v2", fb(1.5), fb(-2.0), 0, fb(-2.0)),
    "v_rcp_f32":   ("v_rcp_f32 v3, v1", fb(4.0), 0, 0, fb(0.25)),
    "v_rsq_f32":   ("v_rsq_f32 v3, v1", fb(4.0), 0, 0, fb(0.5)),
    "v_sqrt_f32":  ("v_sqrt_f32 v3, v1", fb(9.0), 0, 0, fb(3.0)),
    "v_exp_f32":   ("v_exp_f32 v3, v1", fb(3.0), 0, 0, fb(8.0)),
    "v_log_f32":   ("v_log_f32 v3, v1", fb(8.0), 0, 0, fb(3.0)),
    "v_floor_f32": ("v_floor_f32 v3, v1", fb(-1.5), 0, 0, fb(-2.0)),
    "v_ceil_f32":  ("v_ceil_f32 v3, v1", fb(-1.5), 0, 0, fb(-1.0)),
    "v_trunc_f32": ("v_trunc_f32 v3, v1", fb(-1.5), 0, 0, fb(-1.0)),
    "v_fract_f32": ("v_fract_f32 v3, v1", fb(-1.25), 0, 0, fb(0.75)),
    "v_med3_f32":  ("v_med3_f32 v3, v1, v2, v4", fb(5.0), fb(1.0), fb(3.0), fb(3.0)),
    "v_med3_f32_clamp": ("v_med3_f32 v3, v1, v2, v4 clamp", fb(2.0), fb(0.5), fb(1.5), fb(1.0)),
    "v_med3_f32_mul_2": ("v_med3_f32 v3, v1, v2, v4 mul:2", fb(2.0), fb(0.5), fb(1.5), fb(3.0)),
    "v_med3_f32_clamp_div_2": ("v_med3_f32 v3, v1, v2, v4 clamp div:2", fb(2.0), fb(0.5), fb(1.5), fb(0.75)),
    "neg_modifier": ("v_add_f32 v3, -v1, v2", fb(1.5), fb(2.0), 0, fb(0.5)),
    "abs_modifier": ("v_add_f32 v3, |v1|, v2", fb(-1.5), fb(2.0), 0, fb(3.5)),
    "clamp":       ("v_add_f32_e64 v3, v1, v2 clamp", fb(1.5), fb(2.0), 0, fb(1.0)),
    "mul_2":       ("v_add_f32_e64 v3, v1, v2 mul:2", fb(1.5), fb(2.0), 0, fb(7.0)),
    # conversions
    "v_cvt_f32_u32": ("v_cvt_f32_u32 v3, v1", 7, 0, 0, fb(7.0)),
    "v_cvt_f32_i32": ("v_cvt_f32_i32 v3, v1", s(-7), 0, 0, fb(-7.0)),
    "v_cvt_u32_f32": ("v_cvt_u32_f32 v3, v1", fb(7.9), 0, 0, 7),
    "v_cvt_i32_f32": ("v_cvt_i32_f32 v3, v1", fb(-7.9), 0, 0, s(-7)),
    # integer ALU
    "v_add_nc_u32": ("v_add_nc_u32 v3, v1, v2", 0xFFFFFFFF, 2, 0, 1),
    "v_sub_nc_u32": ("v_sub_nc_u32 v3, v1, v2", 5, 7, 0, s(-2)),
    "v_and_b32":   ("v_and_b32 v3, v1, v2", 0xF0F0, 0xFF00, 0, 0xF000),
    "v_or_b32":    ("v_or_b32 v3, v1, v2", 0xF0F0, 0x0F00, 0, 0xFFF0),
    "v_xor_b32":   ("v_xor_b32 v3, v1, v2", 0xFF, 0x0F, 0, 0xF0),
    "v_lshlrev_b32": ("v_lshlrev_b32 v3, v1, v2", 4, 0x0F, 0, 0xF0),
    "v_lshrrev_b32": ("v_lshrrev_b32 v3, v1, v2", 4, 0xF0, 0, 0x0F),
    "v_ashrrev_i32": ("v_ashrrev_i32 v3, v1, v2", 4, 0x80000000, 0, 0xF8000000),
    "v_lshlrev_b32_wraps_amount": ("v_lshlrev_b32 v3, v1, v2", 36, 1, 0, 16),     # shift amount is v1 & 31
    "v_bfe_u32":   ("v_bfe_u32 v3, v1, v2, v4", 0xABCD1234, 8, 12, 0xD12),
    "v_mul_u32_u24": ("v_mul_u32_u24 v3, v1, v2", 0x01000003, 5, 0, 15),         # only the low 24 bits are used
    "v_mad_u32_u24": ("v_mad_u32_u24 v3, v1, v2, v4", 3, 5, 100, 115),
    "v_not_b32":   ("v_not_b32 v3, v1", 0x0F0F0F0F, 0, 0, 0xF0F0F0F0),
    "v_or3_b32":   ("v_or3_b32 v3, v1, v2, v4", 1, 2, 4, 7),
    "v_add3_u32":  ("v_add3_u32 v3, v1, v2, v4", 1, 2, 4, 7),
    "v_xor3_b32":  ("v_xor3_b32 v3, v1, v2, v4", 7, 3, 1, 5),
    "v_bcnt_u32_b32": ("v_bcnt_u32_b32 v3, v1, v2", 0xFF00FF, 1, 0, 17),
    # f16 packing
    "v_pack_b32_f16": ("v_pack_b32_f16 v3, v1, v2", 0xAAAA3C00, 0xBBBB4000, 0, 0x40003C00),
    "v_pack_b32_f16_sel": ("v_pack_b32_f16 v3, v1, v2 op_sel:[1,0,0]", 0x3C00BBBB, 0xAAAA4000, 0, 0x4000 << 16 | 0x3C00),
    "v_pack_b32_f16_sel_hi": ("v_pack_b32_f16 v3, v1, v2 op_sel:[0,1,0]", 0xAAAA3C00, 0x4000BBBB, 0, 0x4000 << 16 | 0x3C00),
    # SHADER-260: a `neg` on this instruction flips the sign bit of the half it packs (gfx1036, IEEE mode 0), not
    # of the whole 32-bit source, and only that half.
    "v_pack_b32_f16_neg_lo": ("v_pack_b32_f16 v3, neg(v1), v2", 0xAAAA3C00, 0xBBBB4000, 0, 0x4000BC00),
    "v_pack_b32_f16_neg_hi_sel": ("v_pack_b32_f16 v3, v1, neg(v2) op_sel:[0,1,0]", 0xAAAA3C00, 0x4000BBBB, 0,
                                  0xC0003C00),
    # `abs` is the ISA's sign clear (|S|), applied before `neg`; it has no hardware run of this instruction yet.
    "v_pack_b32_f16_abs_lo": ("v_pack_b32_f16 v3, abs(v1), v2", 0xAAAABC00, 0xBBBB4000, 0, 0x40003C00),
    "v_pack_b32_f16_abs_snan": ("v_pack_b32_f16 v3, abs(v1), v2", 0xAAAAFC01, 0xBBBB4000, 0, 0x40007C01),
    "v_pack_b32_f16_neg_abs": ("v_pack_b32_f16 v3, -abs(v1), v2", 0xAAAA3C00, 0xBBBB4000, 0, 0x4000BC00),
    # 16-bit float add keeps the other half of the destination (GFX10 partial write; see docs/design.md).
    # These pin the translator's keep; they do not reproduce the AMD driver quirk in valu_fuzz f16
    # seed 3 programme 38, where the same keep is lost on the GPU (driver 26.11.1, RX 9070).
    "v_add_f16_e64_keep_hi": ("v_mov_b32 v3, 0xCD524000\nv_add_f16_e64 v3, v3, v1", 0x3C00, 0, 0, 0xCD524200),
    "v_add_f16_e32_keep_hi": ("v_mov_b32 v3, 0xCD524000\nv_add_f16 v3, v1, v2", 0x3C00, 0x4200, 0, 0xCD524400),
    # 16-bit integer (VOP3 form keeps the high half of the destination; see docs/design.md)
    "v_add_u16_e64": ("v_add_u16_e64 v3, v1, v2", 0xFFFF, 2, 0, 1),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_value(case, tmp_path):
    asm, v1, v2, v4, want = CASES[case]
    got = run_v3(tmp_path, asm, v1, v2, v4)
    assert got == want, f"{asm}: got {got:#010x}, want {want:#010x}"


def test_scalar_alu(tmp_path):
    asm = ("s_mov_b32 s40, 0x12\ns_add_u32 s41, s40, 0xff\ns_lshl_b32 s42, s41, 4\ns_and_b32 s43, s42, 0xfff0\n"
           "s_or_b32 s44, s43, 1\ns_lshr_b32 s45, s44, 1\ns_mul_i32 s46, s45, 3\nv_mov_b32 v3, s46\n")
    s41 = 0x12 + 0xFF
    s42 = s41 << 4
    s43 = s42 & 0xFFF0
    s45 = (s43 | 1) >> 1
    got = si.run(translate_listing(tmp_path, asm)).vgpr[3]
    assert got == (s45 * 3) & M


def test_inline_float_constants(tmp_path):
    asm = "v_mov_b32 v1, 0.5\nv_add_f32 v3, v1, 4.0"
    assert si.run(translate_listing(tmp_path, asm)).vgpr[3] == fb(4.5)


def test_export_reaches_mrt0(tmp_path):
    res = si.run(translate_listing(tmp_path, "v_mov_b32 v3, 0x3f800000"))
    assert res.outputs["out_mrt0"] == (1.0, 1.0, 1.0, 1.0)
