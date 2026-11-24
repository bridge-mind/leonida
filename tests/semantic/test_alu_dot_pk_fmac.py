"""Semantic interpreter tests for RDNA2 integer v_dot2/4/8 instructions and v_pk_fmac_f16 (SHADER-300).

Covers the six remaining instructions after the #1754 split:
  - v_dot2_i32_i16, v_dot2_u32_u16 (2x16-bit packed dot products, wrap and 32-bit clamp)
  - v_dot4_u32_u8 (4x8-bit unsigned packed dot product, wrap and 32-bit clamp)
  - v_dot8_i32_i4, v_dot8_u32_u4 (8x4-bit packed dot products, wrap and 32-bit clamp)
  - v_pk_fmac_f16 (VOP2 packed f16x2 fused multiply-accumulate into vdst with single f16 rounding)

Standard library only; no GPU, no LLVM, no SPIRV-Tools, no game files.
"""
import math
import os
import random
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.dont_write_bytecode = True
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import shader_ir  # noqa: E402
from support import mrt0_words, rdna2spirv, translate_lines  # noqa: E402
from test_f16_fma_rounding import double_rounded, near_tie_triples, ref_fma  # noqa: E402

MASK32 = 0xFFFFFFFF

PS_DOT_PK_FMAC_ASM = (
    # Channel 0 (v10): v_dot2_i32_i16 and v_dot2_u32_u16
    #   v4 = 3*(-2) + (-4)*5 + 100 = 74 (0x4a)
    #   v5 = 1*3 + 32768*1 + 5 = 32776 (0x8008)
    #   v10 = (v4 << 16) + v5 = 0x004a8008
    "v_mov_b32 v1, 0xfffc0003",
    "v_mov_b32 v2, 0x0005fffe",
    "v_mov_b32 v3, 100",
    "v_dot2_i32_i16 v4, v1, v2, v3",
    "v_mov_b32 v1, 0x80000001",
    "v_mov_b32 v2, 0x00010003",
    "v_mov_b32 v3, 5",
    "v_dot2_u32_u16 v5, v1, v2, v3",
    "v_lshlrev_b32 v4, 16, v4",
    "v_add_nc_u32 v10, v4, v5",
    # Channel 1 (v11): v_dot4_u32_u8 (wrapping unsigned bytes + clamp saturation)
    #   v6 = 2*4 + 1*3 + 255*1 + 128*2 + 10 = 532 (0x0214)
    #   v7 = sat_u32(255*4 + 0xffffff00) = 0xffffffff -> low 16 bits 0xffff
    #   v11 = (v6 << 16) + (v7 & 0xffff) = 0x0214ffff
    "v_mov_b32 v6, 0x80ff0102",
    "v_mov_b32 v7, 0x02010304",
    "v_dot4_u32_u8 v6, v6, v7, 10",
    "v_mov_b32 v1, 0xffffffff",
    "v_mov_b32 v2, 0x01010101",
    "v_mov_b32 v3, 0xffffff00",
    "v_dot4_u32_u8 v7, v1, v2, v3 clamp",
    "v_lshlrev_b32 v6, 16, v6",
    "v_and_b32 v7, 0xffff, v7",
    "v_add_nc_u32 v11, v6, v7",
    # Channel 2 (v12): v_dot8_i32_i4 and v_dot8_u32_u4
    #   v8 = 6*2 + 5*1 + 4*2 + 3*1 + 2*2 + 1*1 + (-1)*2 + (-8)*1 + 10 = 33 (0x21)
    #   v9 = 1*3 + 8*2 + 15*1 + 4 = 38 (0x26)
    #   v12 = (v8 << 8) + v9 = 0x00002126
    "v_mov_b32 v1, 0x8f123456",
    "v_mov_b32 v2, 0x12121212",
    "v_mov_b32 v3, 10",
    "v_dot8_i32_i4 v8, v1, v2, v3",
    "v_mov_b32 v1, 0xf8000001",
    "v_mov_b32 v2, 0x12000003",
    "v_mov_b32 v3, 4",
    "v_dot8_u32_u4 v9, v1, v2, v3",
    "v_lshlrev_b32 v8, 8, v8",
    "v_add_nc_u32 v12, v8, v9",
    # Channel 3 (v13): v_pk_fmac_f16
    #   lo: fma(2.0, -1.0, 0.5) = -1.5 (0xbe00)
    #   hi: fma(-2.0, 1.0, 3.0) = +1.0 (0x3c00) -> 0x3c00be00
    "v_mov_b32 v13, 0x42003800",
    "v_mov_b32 v14, 0xc0004000",
    "v_mov_b32 v15, 0x3c00bc00",
    "v_pk_fmac_f16 v13, v14, v15",
    "exp mrt0 v10, v11, v12, v13 done vm",
    "s_endpgm",
)
PS_DOT_PK_FMAC_EXPECTED = (0x004A8008, 0x0214FFFF, 0x00002126, 0x3C00BE00)

DOT_OPS = (
    "v_dot2_i32_i16",
    "v_dot2_u32_u16",
    "v_dot4_u32_u8",
    "v_dot8_i32_i4",
    "v_dot8_u32_u4",
)

# Hardware-measured on RDNA2 gfx1036 (Ryzen 9 7950X3D iGPU, tools/hw_oracle, @aslan2142):
# 60 operand sets as (S0, S1, S2, D, D_clamp).
MEASURED_DOT_GFX1036 = {
    "v_dot4_u32_u8": [
        (0x04030201, 0x08070605, 0x00000000, 0x00000046, 0x00000046),
        (0x04030201, 0x08070605, 0x000003E8, 0x0000042E, 0x0000042E),
        (0x000000FF, 0x00000002, 0x00000000, 0x000001FE, 0x000001FE),
        (0xFF000000, 0x02000000, 0x00000000, 0x000001FE, 0x000001FE),
        (0x00FF0000, 0x0000FF00, 0x00000000, 0x00000000, 0x00000000),
        (0x80808080, 0x7F7F7F7F, 0x00000005, 0x0000FE05, 0x0000FE05),
        (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x0003F804, 0x0003F804),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x0003F803, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0x8003F803, 0x8003F803),
        (0x01010101, 0x01010101, 0xFFFFFFFC, 0x00000000, 0xFFFFFFFF),
        (0x12345678, 0x9ABCDEF0, 0x80000000, 0x8000EC18, 0x8000EC18),
        (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF),
    ],
    "v_dot2_i32_i16": [
        (0x00020001, 0x00040003, 0x00000000, 0x0000000B, 0x0000000B),
        (0x00020001, 0x00040003, 0x000003E8, 0x000003F3, 0x000003F3),
        (0x0000FFFF, 0x00000002, 0x00000000, 0xFFFFFFFE, 0xFFFFFFFE),
        (0xFFFF0000, 0x00020000, 0x00000000, 0xFFFFFFFE, 0xFFFFFFFE),
        (0x80000000, 0x00008000, 0x00000000, 0x00000000, 0x00000000),
        (0x80008000, 0x7FFF7FFF, 0x00000000, 0x80010000, 0x80010000),
        (0x80008000, 0x80008000, 0x00000000, 0x80000000, 0x7FFFFFFF),
        (0x7FFF7FFF, 0x7FFF7FFF, 0x7FFFFFFF, 0xFFFE0001, 0x7FFFFFFF),
        (0x80008000, 0x7FFF7FFF, 0x80000000, 0x00010000, 0x80000000),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000001, 0x00000001),
        (0x8000FFFF, 0x00017FFF, 0x00000064, 0xFFFF0065, 0xFFFF0065),
        (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF),
    ],
    "v_dot2_u32_u16": [
        (0x00020001, 0x00040003, 0x00000000, 0x0000000B, 0x0000000B),
        (0x00020001, 0x00040003, 0x000003E8, 0x000003F3, 0x000003F3),
        (0x0000FFFF, 0x00000002, 0x00000000, 0x0001FFFE, 0x0001FFFE),
        (0xFFFF0000, 0x00020000, 0x00000000, 0x0001FFFE, 0x0001FFFE),
        (0x80000000, 0x00008000, 0x00000000, 0x00000000, 0x00000000),
        (0x80008000, 0x80008000, 0x00000000, 0x80000000, 0x80000000),
        (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0xFFFC0002, 0xFFFFFFFF),
        (0xFFFF0001, 0xFFFF0001, 0x00000000, 0xFFFE0002, 0xFFFE0002),
        (0x00010001, 0x00010001, 0xFFFFFFFF, 0x00000001, 0xFFFFFFFF),
        (0x80008000, 0x80008000, 0x7FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        (0x12345678, 0x9ABCDEF0, 0x80000000, 0xD64DC6B0, 0xD64DC6B0),
        (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF),
    ],
    "v_dot8_i32_i4": [
        (0x87654321, 0x12345678, 0x00000000, 0x00000058, 0x00000058),
        (0x76543210, 0x01234567, 0x000003E8, 0x00000420, 0x00000420),
        (0x0000000F, 0x00000002, 0x00000000, 0xFFFFFFFE, 0xFFFFFFFE),
        (0xF0000000, 0x20000000, 0x00000000, 0xFFFFFFFE, 0xFFFFFFFE),
        (0x000000F0, 0x0000000F, 0x00000000, 0x00000000, 0x00000000),
        (0x88888888, 0x88888888, 0x00000000, 0x00000200, 0x00000200),
        (0x88888888, 0x77777777, 0x00000000, 0xFFFFFE40, 0xFFFFFE40),
        (0x77777777, 0x77777777, 0x7FFFFFFF, 0x80000187, 0x7FFFFFFF),
        (0x88888888, 0x77777777, 0x80000000, 0x7FFFFE40, 0x80000000),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000007, 0x00000007),
        (0xF0F0F0F0, 0x0F0F0F0F, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF),
    ],
    "v_dot8_u32_u4": [
        (0x87654321, 0x12345678, 0x00000000, 0x00000078, 0x00000078),
        (0x76543210, 0x01234567, 0x000003E8, 0x00000420, 0x00000420),
        (0x0000000F, 0x00000002, 0x00000000, 0x0000001E, 0x0000001E),
        (0xF0000000, 0x20000000, 0x00000000, 0x0000001E, 0x0000001E),
        (0x000000F0, 0x0000000F, 0x00000000, 0x00000000, 0x00000000),
        (0x88888888, 0x77777777, 0x00000000, 0x000001C0, 0x000001C0),
        (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000, 0x00000708, 0x00000708),
        (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000707, 0xFFFFFFFF),
        (0xFFFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF, 0x80000707, 0x80000707),
        (0x11111111, 0x11111111, 0xFFFFFFF8, 0x00000000, 0xFFFFFFFF),
        (0xF0F0F0F0, 0x0F0F0F0F, 0x00000000, 0x00000000, 0x00000000),
        (0x00000000, 0x12345678, 0xDEADBEEF, 0xDEADBEEF, 0xDEADBEEF),
    ],
}

# Hardware-measured on RDNA2 gfx1036 in default MODE (HW_REG_MODE = 0x000003C0: round-to-nearest-even,
# f16 denormals kept, IEEE and DX10_CLAMP on), as (S0, S1, D_old, D).
MEASURED_PK_FMAC_DEFAULT_GFX1036 = [
    (0x3C014000, 0x3C034200, 0xBC003C00, 0x1C014700),
    (0x3C013C01, 0x3C013C03, 0xBC02BC00, 0x00101C01),
    (0x3C013C01, 0x3E003C01, 0x8001BC02, 0x3E010010),
    (0x3E013C01, 0x3E013E00, 0x00008001, 0x40823E01),
    (0xBE013E01, 0x3E013E01, 0x80000000, 0xC0824082),
    (0x0000BE01, 0x3C003E01, 0x80008000, 0x0000C082),
    (0x80000000, 0x3C003C00, 0x80008000, 0x80000000),
    (0x3C008000, 0xBC003C00, 0x3C008000, 0x00008000),
    (0x80003C00, 0x3C00BC00, 0x00003C00, 0x00000000),
    (0x7C008000, 0x40003C00, 0x3C000000, 0x7C000000),
    (0x7C007C00, 0xC0004000, 0x7C003C00, 0xFE007C00),
    (0x7C007C00, 0x0000C000, 0x3C007C00, 0xFE00FE00),
    (0x7BFF7C00, 0x40000000, 0x00003C00, 0x7C00FE00),
    (0x7E557BFF, 0x3C004000, 0x3C000000, 0x7E557C00),
    (0x7C017E55, 0x3C003C00, 0x3C003C00, 0x7E017E55),
    (0x3C007C01, 0x3C003C00, 0x7D233C00, 0x7F237E01),
    (0x04003C00, 0x38003C00, 0x00007D23, 0x02007F23),
    (0x00010400, 0x44003800, 0x00000000, 0x00040200),
    (0x00030001, 0x38004400, 0x00000000, 0x00020004),
    (0x3C000003, 0x3C003800, 0x64000000, 0x64010002),
    (0x44003C00, 0xC2003C00, 0x49006400, 0xC0006401),
    (0x40004400, 0x4200C200, 0x3C004900, 0x4700C000),
]


def sext(val, width):
    mask = (1 << width) - 1
    v = val & mask
    return v - (1 << width) if v >> (width - 1) else v


def ref_dot(op, s0, s1, acc, clamp=False):
    width = 16 if "_dot2_" in op else (8 if "_dot4" in op else 4)
    signed = op.endswith(("i16", "i4"))
    mask = (1 << width) - 1
    total = sext(acc, 32) if signed else (acc & MASK32)
    for off in range(0, 32, width):
        a = (s0 >> off) & mask
        b = (s1 >> off) & mask
        if signed:
            a, b = sext(a, width), sext(b, width)
        total += a * b
    if clamp:
        if signed:
            total = max(-0x80000000, min(0x7FFFFFFF, total))
        else:
            total = max(0, min(0xFFFFFFFF, total))
    return total & MASK32


def h2f(bits):
    return struct.unpack("<e", struct.pack("<H", bits & 0xFFFF))[0]


def ref_pk_fmac_half(a_bits, b_bits, c_bits):
    a, b, c = h2f(a_bits), h2f(b_bits), h2f(c_bits)
    if math.isnan(a) or math.isnan(b) or math.isnan(c):
        return None
    if (math.isinf(a) and b == 0.0) or (a == 0.0 and math.isinf(b)):
        return None
    if math.isinf(a) or math.isinf(b):
        prod_sgn = (a_bits ^ b_bits) & 0x8000
        if math.isinf(c) and ((c_bits & 0x8000) != prod_sgn):
            return None
        return prod_sgn | 0x7C00
    if math.isinf(c):
        return c_bits & 0xFFFF
    exact = ref_fma(a_bits, b_bits, c_bits)
    if (exact & 0x7FFF) == 0:
        psgn = ((a_bits ^ b_bits) & 0x8000) if (a == 0.0 or b == 0.0) else None
        if psgn is not None and c == 0.0 and psgn == 0x8000 and (c_bits & 0x8000) == 0x8000:
            return 0x8000
        return 0x0000
    return exact


def ref_pk_fmac_f16(s0, s1, dst, s0_fimm=False):
    lo_s0 = s0 & 0xFFFF
    hi_s0 = 0 if s0_fimm else ((s0 >> 16) & 0xFFFF)
    lo = ref_pk_fmac_half(lo_s0, s1 & 0xFFFF, dst & 0xFFFF)
    hi = ref_pk_fmac_half(hi_s0 & 0xFFFF, (s1 >> 16) & 0xFFFF, (dst >> 16) & 0xFFFF)
    return lo, hi


def run_batch(cases, op, clamp=False, s0_kind="vgpr", acc_in_s=False):
    """Execute up to 4 (s0, s1, acc) cases in one shader and return the list of mrt0 uint32 words."""
    lines = []
    suffix = " clamp" if clamp else ""
    for k, (s0, s1, acc) in enumerate(cases):
        s1_reg = f"v{8 + k}"
        lines.append(f"v_mov_b32 {s1_reg}, {s1 & MASK32:#x}")
        if s0_kind == "fimm":
            s0_op = repr(s0) if not isinstance(s0, str) else s0
        elif s0_kind == "imm":
            s0_op = str(s0)
        elif s0_kind == "sgpr":
            lines.append(f"s_mov_b32 s{k}, {s0 & MASK32:#x}")
            s0_op = f"s{k}"
        else:
            lines.append(f"v_mov_b32 v{4 + k}, {s0 & MASK32:#x}")
            s0_op = f"v{4 + k}"

        if op == "v_pk_fmac_f16":
            lines.append(f"v_mov_b32 v{k}, {acc & MASK32:#x}")
            lines.append(f"{op} v{k}, {s0_op}, {s1_reg}{suffix}")
        else:
            if acc_in_s:
                lines.append(f"s_mov_b32 s{4 + k}, {acc & MASK32:#x}")
                acc_op = f"s{4 + k}"
            else:
                lines.append(f"v_mov_b32 v{12 + k}, {acc & MASK32:#x}")
                acc_op = f"v{12 + k}"
            lines.append(f"{op} v{k}, {s0_op}, {s1_reg}, {acc_op}{suffix}")
    regs = ", ".join(f"v{k}" for k in range(4))
    lines += [f"exp mrt0 {regs} done vm", "s_endpgm"]
    return mrt0_words(translate_lines("ps_dot_pk_batch", lines))[:len(cases)]


def run_chunks(cases, op, **kwargs):
    out = []
    for i in range(0, len(cases), 4):
        chunk = cases[i:i + 4]
        padded = chunk + [(0, 0, 0)] * (4 - len(chunk))
        out.extend(run_batch(padded, op, **kwargs)[:len(chunk)])
    return out


class DotAndPkFmacTests(unittest.TestCase):
    def test_fixture(self):
        got = tuple(mrt0_words(translate_lines("ps_dot_pk_fmac", PS_DOT_PK_FMAC_ASM)))
        self.assertEqual(got, PS_DOT_PK_FMAC_EXPECTED)

    def test_reference(self):
        self.assertEqual(ref_dot("v_dot2_i32_i16", 0x80008000, 0x80008000, 0), 0x80000000)
        self.assertEqual(ref_dot("v_dot2_i32_i16", 0x80008000, 0x80008000, 0, clamp=True), 0x7FFFFFFF)
        self.assertEqual(ref_dot("v_dot2_i32_i16", 0x80008000, 0x80008000, 0xFFFFFFFF, clamp=True), 0x7FFFFFFF)
        self.assertEqual(ref_dot("v_dot2_i32_i16", 0x80008000, 0x80008000, 0xFFFFFFFE, clamp=True), 0x7FFFFFFE)
        self.assertEqual(ref_dot("v_dot2_u32_u16", 0x80008000, 0x80008000, 0), 0x80000000)
        self.assertEqual(ref_dot("v_dot2_u32_u16", 0xFFFF0000, 0xFFFF0000, 0x00030000), 0x00010001)
        self.assertEqual(ref_dot("v_dot2_u32_u16", 0xFFFF0000, 0xFFFF0000, 0x00030000, clamp=True), 0xFFFFFFFF)
        self.assertEqual(ref_dot("v_dot4_u32_u8", 0x80FF017F, 0x0102FE01, 10), 127 + 254 + 510 + 128 + 10)
        self.assertEqual(ref_dot("v_dot8_i32_i4", 0x8F000007, 0x11000008, 0), (7 * (-8) + (-1) * 1 + (-8) * 1) & MASK32)
        self.assertEqual(ref_dot("v_dot8_u32_u4", 0x8F000007, 0x11000008, 0), 7 * 8 + 15 * 1 + 8 * 1)
        self.assertEqual(ref_pk_fmac_f16(0xC0004000, 0x3C00BC00, 0x42003800), (0xBE00, 0x3C00))

    def test_dot2_values_and_clamp(self):
        # Signed product sum (-32768)^2 + (-32768)^2 = +2^31 crosses INT32_MAX before adding acc.
        edge_cases = [
            (0x80008000, 0x80008000, 0),
            (0x80008000, 0x80008000, 1),
            (0x80008000, 0x80008000, 0xFFFFFFFF),   # +2^31 - 1 == INT32_MAX (not overflowed)
            (0x80008000, 0x80008000, 0xFFFFFFFE),   # +2^31 - 2 == 0x7ffffffe
            (0x80008000, 0x80008000, 0x80000000),   # +2^31 + INT32_MIN == 0
            (0x80008000, 0x7FFF7FFF, 0x80000000),   # -0x7fff0000 + INT32_MIN -> underflow
            (0x80008000, 0x7FFF7FFF, 0x80010000),   # exact INT32_MIN (not overflowed)
            (0x80008000, 0x7FFF7FFF, 0x8000FFFE),   # INT32_MIN - 2 -> underflow
            (0xFFFF8000, 0xFFFF8000, 0x7FFFFFFF),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x00020000),   # u16 wrap & clamp boundary
            (0xFFFF0000, 0xFFFF0000, 0x0001FFFE),   # exact 0xffffffff for u16
            (0xFFFF0000, 0xFFFF0000, 0x0001FFFF),   # 0x100000000 -> clamp 0xffffffff
        ]
        for op in ("v_dot2_i32_i16", "v_dot2_u32_u16"):
            for clamp in (False, True):
                got = run_chunks(edge_cases, op, clamp=clamp)
                want = [ref_dot(op, s0, s1, acc, clamp=clamp) for s0, s1, acc in edge_cases]
                with self.subTest(op=op, clamp=clamp):
                    self.assertEqual(got, want)

    def test_dot4_u32_u8_values_and_clamp(self):
        cases = [
            (0x80FF017F, 0x01010101, 0),
            (0x80000000, 0x01000000, 0x7FFFFFFF),
            (0x00800000, 0x00010000, 0x80000000),
            (0x0000FF00, 0x00000200, 0xFFFFFFFF),
            (0x00000080, 0x00000080, 0x7FFFC000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFC03FC),   # exact 0xffffffff unsigned
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFC03FD),   # unsigned overflow -> 0xffffffff
            (0xFFFFFFFF, 0x00000001, 0xFFFFFF00),   # 255 + 0xffffff00 = 0xffffffff exact unsigned
            (0xFFFFFFFF, 0x00000001, 0xFFFFFF01),   # 255 + 0xffffff01 -> overflow clamp
            (0x12345678, 0x9ABCDEF0, 0),
            (0x80808080, 0x80808080, 0xFFFE0000),
            (0xFF00FF00, 0x01000100, 42),
        ]
        for clamp in (False, True):
            got = run_chunks(cases, "v_dot4_u32_u8", clamp=clamp)
            want = [ref_dot("v_dot4_u32_u8", s0, s1, acc, clamp=clamp) for s0, s1, acc in cases]
            with self.subTest(clamp=clamp):
                self.assertEqual(got, want)

    def test_dot8_values_and_clamp(self):
        cases = [
            (0x80000000, 0x10000000, 0),
            (0x0F000000, 0x02000000, 0x7FFFFFFF),
            (0x00700000, 0x00800000, 0x80000000),
            (0x00080000, 0x00080000, 0x7FFFFFC0),
            (0x0000F000, 0x0000F000, 0xFFFFFFFF),
            (0x00000900, 0x00000500, 100),
            (0x00000080, 0x00000070, 0x80000037),
            (0x0000000F, 0x0000000E, 0xFFFFFF2E),   # u4: 15*14 = 210, + 0xffffff2e == 0x100000000
            (0x88888888, 0x88888888, 0x7FFFFE00),   # 512 + (2^31 - 512) == 2^31 -> sat 0x7fffffff
            (0x88888888, 0x77777777, 0x80000100),   # -448 + (INT32_MIN + 256) == INT32_MIN - 192 -> sat INT32_MIN
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFF8F7),   # 8*225 = 1800, + 0xfffff8f7 == 0xffffffff exact u4
            (0x12345678, 0xFEDCBA98, 42),
        ]
        for op in ("v_dot8_i32_i4", "v_dot8_u32_u4"):
            for clamp in (False, True):
                got = run_chunks(cases, op, clamp=clamp)
                want = [ref_dot(op, s0, s1, acc, clamp=clamp) for s0, s1, acc in cases]
                with self.subTest(op=op, clamp=clamp):
                    self.assertEqual(got, want)

    def test_measured_dot_gfx1036(self):
        for op, rows in MEASURED_DOT_GFX1036.items():
            cases = [(s0, s1, s2) for s0, s1, s2, _, _ in rows]
            want_plain = [d for _, _, _, d, _ in rows]
            want_clamp = [d_clamp for _, _, _, _, d_clamp in rows]
            with self.subTest(op=op, clamp=False):
                self.assertEqual(run_chunks(cases, op, clamp=False), want_plain)
            with self.subTest(op=op, clamp=True):
                self.assertEqual(run_chunks(cases, op, clamp=True), want_clamp)

    def test_measured_pk_fmac_default_gfx1036(self):
        # Default MODE 0x000003C0: verify both source orders (S0, S1) and (S1, S0).
        # Non-NaN halves must match gfx1036 bit-for-bit; NaN halves must produce a quiet f16 NaN
        # (SPIR-V UnpackHalf2x16/PackHalf2x16/Fma do not preserve NaN payload bits or sign).
        want = [d for _, _, _, d in MEASURED_PK_FMAC_DEFAULT_GFX1036]
        cases_ab = [(s0, s1, d_old) for s0, s1, d_old, _ in MEASURED_PK_FMAC_DEFAULT_GFX1036]
        cases_ba = [(s1, s0, d_old) for s0, s1, d_old, _ in MEASURED_PK_FMAC_DEFAULT_GFX1036]
        for order, cases in (("s0,s1", cases_ab), ("s1,s0", cases_ba)):
            got = run_chunks(cases, "v_pk_fmac_f16")
            for idx, (g, w) in enumerate(zip(got, want)):
                for shift, half_name in ((0, "lo"), (16, "hi")):
                    gh = (g >> shift) & 0xFFFF
                    wh = (w >> shift) & 0xFFFF
                    with self.subTest(order=order, idx=idx, half=half_name):
                        if (wh & 0x7FFF) > 0x7C00:
                            self.assertEqual(gh & 0x7E00, 0x7E00, f"expected quiet f16 NaN, got {gh:#06x}")
                        else:
                            self.assertEqual(gh, wh)

    def test_sources_and_default_opsel(self):
        self.assertEqual(run_batch([(5, 0x00010002, 0x10), (0x00010002, 0x00030004, 10), (0, 0, 0), (0, 0, 0)],
                                   "v_dot2_i32_i16", s0_kind="imm", acc_in_s=True)[:2],
                         [ref_dot("v_dot2_i32_i16", 5, 0x00010002, 0x10),
                          ref_dot("v_dot2_i32_i16", 0x00010002, 0x00030004, 10)])
        self.assertEqual(run_batch([(0x80FF0102, 0x01020304, 0x80000010), (0x01020304, 0x05060708, -5),
                                    (0, 0, 0), (0, 0, 0)],
                                   "v_dot4_u32_u8", s0_kind="sgpr", acc_in_s=True)[:2],
                         [ref_dot("v_dot4_u32_u8", 0x80FF0102, 0x01020304, 0x80000010),
                          ref_dot("v_dot4_u32_u8", 0x01020304, 0x05060708, -5)])
        # Hardware default is op_sel:[0,0,0] and op_sel_hi:[1,1,1]; writing either or both explicitly matches plain.
        lines = [
            "s_mov_b32 s0, 0x00020001",
            "v_mov_b32 v1, 0x00040003",
            "v_dot2_i32_i16 v2, s0, v1, 0 op_sel:[0,0,0] op_sel_hi:[1,1,1]",   # 1*3 + 2*4 = 11
            "v_dot2_u32_u16 v3, s0, v1, 5 op_sel_hi:[1,1,1]",                  # 1*3 + 2*4 + 5 = 16
            "s_mov_b32 s1, 0x01020304",
            "v_mov_b32 v4, 0x01000100",
            "v_dot4_u32_u8 v5, s1, v4, 7 op_sel:[0,0,0] op_sel_hi:[1,1,1]",    # 3*1 + 1*1 + 7 = 11
            "v_dot8_i32_i4 v6, s1, v4, 3 op_sel:[0,0,0]",                      # 3*1 + 1*1 + 3 = 7
            "exp mrt0 v2, v3, v5, v6 done vm",
            "s_endpgm",
        ]
        got = mrt0_words(translate_lines("ps_dot_opsel_default", lines))
        self.assertEqual(got, [11, 16, 11, 7])

    def test_pk_fmac_f16_values_and_rounding(self):
        cases = [
            (0x40003C00, 0x4200C000, 0x38004400),  # lo: 1.0*(-2.0)+4.0 = 2.0; hi: 2.0*3.0+0.5 = 6.5
            (0x80000000, 0xBC003C00, 0x80008000),  # signed zeros
            (0x00100004, 0x40003C00, 0x00018001),  # subnormals
            (0x7BFF7BFF, 0x40004000, 0x3C00BC00),  # overflow to +inf
            (0x40003C00, 0x40004000, 0xFC007C00),  # +inf / -inf
        ]
        got = run_chunks(cases, "v_pk_fmac_f16")
        for (s0, s1, dst), g in zip(cases, got):
            want_lo, want_hi = ref_pk_fmac_f16(s0, s1, dst)
            with self.subTest(s0=hex(s0), s1=hex(s1), dst=hex(dst)):
                self.assertEqual(g & 0xFFFF, want_lo)
                self.assertEqual((g >> 16) & 0xFFFF, want_hi)

        triples = near_tie_triples(random.Random(55), 32)
        differing = [t for t in triples if double_rounded(*t) != ref_fma(*t)]
        self.assertGreaterEqual(len(differing), 4)
        tie_cases = []
        for i in range(0, min(len(differing), 8), 2):
            a0, b0, c0 = differing[i]
            a1, b1, c1 = differing[i + 1] if i + 1 < len(differing) else differing[i]
            tie_cases.append(((a1 << 16) | a0, (b1 << 16) | b0, (c1 << 16) | c0))
        got_ties = run_chunks(tie_cases, "v_pk_fmac_f16")
        for (s0, s1, dst), g in zip(tie_cases, got_ties):
            want_lo, want_hi = ref_pk_fmac_f16(s0, s1, dst)
            with self.subTest(tie=hex(s0)):
                self.assertEqual(g & 0xFFFF, want_lo)
                self.assertEqual((g >> 16) & 0xFFFF, want_hi)

    def test_pk_fmac_f16_sources_aliasing_and_nan(self):
        got_fimm = run_batch([(1.0, 0x42004000, 0x3C003800), (-0.5, 0x3C00C000, 0x40004000),
                              (0.0, 0, 0), (0.0, 0, 0)], "v_pk_fmac_f16", s0_kind="fimm")[:2]
        self.assertEqual(got_fimm, [
            ref_pk_fmac_f16(0x3C00, 0x42004000, 0x3C003800)[0] | (ref_pk_fmac_f16(0x3C00, 0x42004000, 0x3C003800)[1] << 16),
            ref_pk_fmac_f16(0xB800, 0x3C00C000, 0x40004000)[0] | (ref_pk_fmac_f16(0xB800, 0x3C00C000, 0x40004000)[1] << 16),
        ])

        lines = [
            "s_mov_b32 s0, 0x40003c00",           # lo=1.0, hi=2.0
            "v_mov_b32 v0, 0x3c004000",           # lo=2.0, hi=1.0
            "v_mov_b32 v1, 0xc0003800",           # lo=0.5, hi=-2.0
            "v_pk_fmac_f16 v0, s0, v1",           # lo=1.0*0.5+2.0=2.5 (0x4100), hi=2.0*(-2.0)+1.0=-3.0 (0xc200)
            "v_mov_b32 v2, 0x38004000",           # lo=2.0, hi=0.5
            "v_pk_fmac_f16 v2, v2, v2",           # lo=2.0*2.0+2.0=6.0 (0x4600), hi=0.5*0.5+0.5=0.75 (0x3a00)
            "v_mov_b32 v3, 0x3c007e01",           # lo=NaN, hi=1.0
            "v_pk_fmac_f16 v3, 0x3c003c00, v1",   # lo=NaN, hi=1.0*(-2.0)+1.0=-1.0 (0xbc00)
            "v_and_b32 v4, 0xffff, v3",
            "v_lshrrev_b32 v5, 16, v3",
            "exp mrt0 v0, v2, v4, v5 done vm",
            "s_endpgm",
        ]
        got = mrt0_words(translate_lines("ps_pk_fmac_alias", lines))
        self.assertEqual(got[0], 0xC2004100)
        self.assertEqual(got[1], 0x3A004600)
        self.assertGreater(got[2] & 0x7FFF, 0x7C00)
        self.assertEqual(got[3], 0xBC00)

    def test_dpp_form_translates(self):
        spv = translate_lines("ps_pk_fmac_dpp", [
            "v_pk_fmac_f16_dpp v3, v4, v5 dpp8:[1,2,3,4,5,6,7,0] fi:1",
            "exp mrt0 v3, v3, v3, v3 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpGroupNonUniform", spv)

    def test_read_before_write_and_rw_sets(self):
        insns_fmac = [shader_ir.parse_line(ln) for ln in (
            "v_pk_fmac_f16 v7, v1, v2",
            "v_mov_b32 v9, 1",
            "v_pk_fmac_f16 v9, v1, v2",
        )]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(insns_fmac), {1, 2, 7})
        rr, ww = rdna2spirv.reg_reads_writes([insns_fmac[0]])
        self.assertEqual(rr, {1, 2, 7})
        self.assertEqual(ww, {7})

        insns_dpp = [shader_ir.parse_line("v_pk_fmac_f16_dpp v10, v1, v2 quad_perm:[1,0,3,2]")]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(insns_dpp), {1, 2, 10})

    def test_random_sweep(self):
        rng = random.Random(0xD0716)
        for op in DOT_OPS:
            cases = [(rng.getrandbits(32), rng.getrandbits(32), rng.getrandbits(32)) for _ in range(16)]
            for clamp in (False, True):
                got = run_chunks(cases, op, clamp=clamp)
                want = [ref_dot(op, s0, s1, acc, clamp=clamp) for s0, s1, acc in cases]
                with self.subTest(op=op, clamp=clamp):
                    self.assertEqual(got, want)

        pk_cases = []
        for _ in range(24):
            def rand_f16():
                e = rng.choice([0, 1, 10, 14, 15, 16, 17, 20])
                return (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)
            s0 = (rand_f16() << 16) | rand_f16()
            s1 = (rand_f16() << 16) | rand_f16()
            dst = (rand_f16() << 16) | rand_f16()
            pk_cases.append((s0, s1, dst))
        got_pk = run_chunks(pk_cases, "v_pk_fmac_f16")
        for (s0, s1, dst), g in zip(pk_cases, got_pk):
            want_lo, want_hi = ref_pk_fmac_f16(s0, s1, dst)
            self.assertEqual(g & 0xFFFF, want_lo)
            self.assertEqual((g >> 16) & 0xFFFF, want_hi)

    def test_rejected_forms(self):
        bad_lines = (
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[1,0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[0,0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel:[0,0,0] op_sel_hi:[0,0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[0,1,1]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[1,0,1]",
            "v_dot2_i32_i16 v0, v1, v2, v3 op_sel_hi:[1,1,0]",
            "v_dot2_u32_u16 v0, v1, v2, v3 op_sel_hi:[0,0,0]",
            "v_dot4_u32_u8 v0, v1, v2, v3 op_sel:[1,0,0]",
            "v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,0,0]",
            "v_dot4_u32_u8 v0, v1, v2, v3 op_sel_hi:[0,1,0]",
            "v_dot4_u32_u8 v0, v1, v2, v3 neg_lo:[1,0,0]",
            "v_dot2_i32_i16 v0, v1, v2, v3 neg_lo:[0,0,1] neg_hi:[0,0,1]",
            "v_dot4_u32_u8 v0, -v1, v2, v3",
            "v_dot2_i32_i16 v0, |v1|, v2, v3",
            "v_dot8_i32_i4 v0, v1, v2, v3 mul:2",
            "v_dot4_u32_u8_dpp v0, v1, v2, v3 quad_perm:[1,0,3,2]",
            "v_dot4_u32_u8_sdwa v0, v1, v2, v3 src0_sel:DWORD",
            "v_pk_fmac_f16 v0, v1, v2 clamp",
            "v_pk_fmac_f16 v0, v1, v2 mul:2",
            "v_pk_fmac_f16 v0, v1, v2 op_sel:[0,0,0]",
            "v_pk_fmac_f16 v0, -v1, v2",
            "v_pk_fmac_f16 v0, |v1|, v2",
            "v_pk_fmac_f16 vcc_lo, v1, v2",
            "v_pk_fmac_f16_sdwa v0, v1, v2 src0_sel:DWORD",
        )
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_dot_pk_bad", [
                        "v_mov_b32 v1, 1", "v_mov_b32 v2, 2", "v_mov_b32 v3, 3", line,
                        "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm",
                    ])


if __name__ == "__main__":
    unittest.main()
