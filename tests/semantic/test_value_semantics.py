"""Value-level regression tests for the recent RDNA2 -> SPIR-V semantic fixes.

Each pixel shader ends with `exp mrt0 v.., v.., v.., v..`, so four 32-bit words are checked per shader. The
expectations are written down by hand as constants and recomputed by plain-Python reference functions (checked
against each other in OracleTests). Neither the constants nor the references use the translator or the SPIR-V
interpreter.

ISA semantics used (AMD RDNA2 ISA reference, VALU/VOP3/VOP3P chapters):
  v_mul_legacy_f32      0 * anything, inf/NaN included, is +0.0; otherwise a normal IEEE binary32 multiply.
  v_lshlrev_b16         D.lo = (S1.lo << (S0.lo & 15)) & 0xffff; the VOP3 form keeps D's high half.
  v_lshrrev_b16         same with >>; v_ashrrev_i16 sign-extends S1.lo before the arithmetic shift.
  v_pk_lshrev_b16 /     per half: D.h = (S1.half <</>> (S0.half & 15)) & 0xffff. op_sel picks the source half
  v_pk_lshlrev_b16 /      used for the low result half, op_sel_hi for the high result half.
  v_pk_ashrrev_i16
  v_add_nc_u16          D.lo = (S0.lo + S1.lo) & 0xffff; the VOP3 form keeps D's high half (carry is dropped).
  v_sin_f16/v_cos_f16   input is a number of turns: sin/cos(S * 2*pi), rounded to binary16; high half kept.
  v_add/sub/mul_f16,    low halves in, D.lo out, D's high half kept; neg/abs in the VOP3 form apply to the low
  v_fmac_f16              half that is read. v_fmac_f16: D.lo = S0.lo * S1.lo + D.lo. The ps_f16_vop2 words were
                          measured on an RX 6800 (gfx1030), wave32 and wave64 (see its .asm header).
  v_cvt_f16_f32         VOP3 omod and clamp apply to the f16, after the rounding: round, omod, clamp (an RX 6800,
                          see ps_outmods_f16.asm). With an omod a zero or denormal f16 gives +0.
  s_buffer_load_dwordx2 with destination vcc: dword 0 -> vcc_lo, dword 1 -> vcc_hi.
  v_interp_mov_f32      reads the flat (provoking-vertex) input attribute; the translator must declare it Flat.
  v_perm_b32            byte i = BYTE_PERMUTE({S0, S1}, S2 byte i) over 8 bytes, S1 = bytes 0-3: selector 0-7 picks
                          that byte; 8/9/10/11 give 0xff if bit 7 of byte 1/3/5/7 is set, else 0; 12 gives 0;
                          13 and above give 0xff.
  v_alignbit_b32        D = ({S0, S1} >> (S2 & 31)) & 0xffffffff.
  v_med3_i32/u32        the middle of three values, compared signed or unsigned.
  v_mul_hi_u32/i32      bits 63:32 of the unsigned or signed 64-bit product.
  v_bcnt_u32_b32        D = popcount(S0) + S1.
  v_ffbl_b32/ffbh_u32   index of the lowest set bit / count of leading zeros; 0xffffffff when S0 = 0.
  v_xad_u32/add3_u32    D = (S0 ^ S1) + S2 / D = S0 + S1 + S2, both mod 2^32.
  s_min/max_i32/u32     D = S0 < S1 ? S0 : S1 (max: >), SCC = 1 when S0 is chosen, so a tie gives S1 and SCC 0.
  v_fmamk_f32/madmk_f32 D = S0 * K + S1, K the literal (printed vdst, src0, K, vsrc1); fmamk is fused.
  v_fmaak_f32/madak_f32 D = S0 * S1 + K (printed vdst, src0, vsrc1, K).
  s_ff1_i32_b32 /       index of the lowest set / clear bit of S0, -1 (0xffffffff) when there is none. An inline
  s_ff0_i32_b32           constant or literal source is a 32-bit value, not a lane mask (SHADER-46).
  v_mad_u32_u16 /       D = S0.lo * S1.lo + S2 mod 2^32, the halves zero- (u16) or sign-extended (i16); op_sel bit
  v_mad_i32_i16           0/1 reads the high half of S0/S1. D is a whole dword (nothing of the old value is kept).
                          Measured on gfx1036 (ps_valu16_mad_sel.asm): op_sel bit 2 adds S2[31:16] zero-extended
                          (i16 too), bit 3 writes the result << 16, clamp saturates to u32 / i32 before that shift.
  v_subrev_f16          D.lo = S1.lo - S0.lo; the high half of D is kept in the VOP2 and the VOP3 form.
  16-bit SDWA ops       with dst_sel:DWORD the 16-bit result is written zero-extended: the high half of D is
    (v_add/mul_f16,       cleared whatever dst_unused says (measured on gfx1036, ps_sdwa_dword_hi.asm).
    v_cvt_u16_f16,
    v_cvt_f16_u16)
  s_*_saveexec_b32/b64, a constant used as a lane mask gives lane L its bit L: of the 32-bit value for a b32 op,
  s_andn2_wrexec_*,       of the 64-bit value for a b64 op (inline constants sign-extended, literals with a zero
  s_mov/s_not/s_and_b64   high word). cs_w32_mask_imm / cs_w64_mask_imm run every lane of a wave (LaneMaskConstants).
    into EXEC or VCC
  v_add_lshl_u32        D = (S0 + S1) << (S2 & 31).   v_and_or_b32  D = (S0 & S1) | S2.   v_xor3_b32  D = S0 ^ S1 ^ S2.
  v_bfi_b32             D = (S0 & S1) | (~S0 & S2).   v_bfm_b32  D = ((1 << (S0 & 31)) - 1) << (S1 & 31).
  v_bfrev_b32           D = S0 with its 32 bits in reverse order.
  v_mad_i32_i24         D = sext24(S0) * sext24(S1) + S2 (low 32 bits); bits 24-31 of S0/S1 are ignored.
  v_mul_hi_u32_u24      D = (zext24(S0) * zext24(S1)) >> 32.
  v_min/max_i32/u32,    the smallest / largest operand, compared signed (_i32) or unsigned (_u32).
  v_min3/max3_i32/u32
  v_cmp(x)_*_f32        GE/LG/O and the other ordered compares are false when either source is NaN; the N forms
                          (NGE = !(S0 >= S1), NGT, NLE, NLT, NEQ) are their negations, so true on NaN; F = 0, TRU = 1.
  v_cmp(x)_*_u64/_i64   the whole 64-bit values compared unsigned / signed (the low word is never signed).
  s_andn2_saveexec_b32  D = EXEC; EXEC = S0 & ~EXEC; SCC = (EXEC != 0).
  s_andn2_wrexec_b32    EXEC = S0 & ~EXEC; D = EXEC (the new one); SCC = (EXEC != 0).

The appended cases for ps_ops5/ps_ops6 also pin the op_sel source-half selection for packed shifts, which the
GPU-checked ps_ops4 does not exercise. cs_bounds (compute) pins the buffer num_records bounds checks: an
out-of-bounds store must not modify memory and an out-of-bounds load must return 0. cs_fmt_1010102 (compute) pins
the channel order of the 10:10:10:2 typed-buffer formats: the names list the channel widths from the top bit, so
2_10_10_10 (50) is R10G10B10A2 and 10_10_10_2 (44) has a 2-bit x (Mesa gfx10_format_table.py, ac_formats.c).
"""
import math
import os
import struct
import sys
import unittest
from unittest import mock

import support
import spirv_mini_interp as si

# Independent constants, written down by hand (derivations are in the reference functions).
EXPECTED_LEGACY_MUL = [0x00000000, 0x00000000, 0x00000000, 0x40C00000]
EXPECTED_OPS4 = [0x34000010, 0x00120800, 0xBEEF0000, 0x12343C00]
EXPECTED_OPS5 = [0x12340010, 0xBEEF0800, 0xAAAAFFFF, 0xDEADBC00]
EXPECTED_OPS6 = [0xF800FFFF, 0x01000010, 0x00800800, 0x8000FFF0]
EXPECTED_OPS7 = [0x00FF8077, 0xFF0000FF, 0x89ABCDEF, 0x00070003]
EXPECTED_OPS8 = [0x80000000, 0xBFFFFFFF, 0x000F100E, 0x0F0F0F1F]
EXPECTED_VCC_X2 = [0x11223344, 0x55667788, 0x00000000, 0x00000000]
EXPECTED_F16_VOP2 = [0xAAAA4000, 0xAAAAC000, 0xAAAA3A00, 0x42003F00]        # hardware-measured
EXPECTED_CVT_F16_OUT_MODS = [0xAAAA3C00, 0xAAAA3E00, 0x7C000000, 0x3C000000]
EXPECTED_SDWA_DWORD_HI = [0x00004000, 0x00000003, 0x00004700, 0x0000C000]   # hardware-measured (gfx1036)
EXPECTED_FLAT = [0x00000101, 0x00000102, 0x3F000000, 0x3E800000]
EXPECTED_SCC_MINMAX = [0x00030205, 0x00030207, 0x00030209, 0x00030203]
EXPECTED_FMA_LITERAL = [0x41B80000, 0x41800000, 0x40E00000, 0x41A40000]
EXPECTED_BOUNDS = {0x00: 0x12345678, 0x04: 0x12345678, 0x08: 0x00000000, 0x0C: 0xEEEEEEEE}
EXPECTED_FMT_1010102 = {0x00: 0x00000000, 0x04: 0x3F800000, 0x08: 0x00000000, 0x0C: 0x3F800000,
                        0x10: 0x3F800000, 0x14: 0x00000000, 0x18: 0x3F800000, 0x1C: 0x00000000,
                        0x20: 0x00000000, 0x24: 0x3F800000, 0x28: 0x00000000, 0x2C: 0x3F800000,
                        0x30: 0xC00FFC00, 0x34: 0x003FF003, 0x38: 0xE00003FF, 0x3C: 0xFFE00003}
EXPECTED_SFF_IMM = [0x00000002, 0x00000001, 0x00000014, 0x00000010]
EXPECTED_VALU16_MAD = [0x00000002, 0x00030002, 0x00008000, 0x00000004]
EXPECTED_VALU16_SUBREV = [0xAAAABC00, 0xAAAAC000, 0xAAAA3C00, 0x42003800]
EXPECTED_VALU16_MAD_SEL = [0x00008003, 0xDB880000, 0xFFFFFFFF, 0x80000000]     # hardware-measured
# ps_sdwa_f16_const, measured on gfx1036 (the .asm header): f16 inline constant, then src_sel
EXPECTED_SDWA_F16_CONST = [0xAAAA3800, 0xAAAA1C00, 0xAAAA3E00, 0xAAAA73F0]
# ps_vint_*: 32-bit integer VALU ops on the heap VINT_HEAP (h0..h7, the words check.py's VINT_ENV seeds).
VINT_HEAP = (0xAB800000, 0x12FFFFFE, 0x00000021, 0x80000001, 0x7FFFFFFE, 0xF0F0F0F0, 0x12345678, 0xFFFFFFFF)
EXPECTED_VINT = {"ps_vint_logic3": [0xFFFFFFFE, 0xBBB05070, 0x1D3B5977, 0x123F5F7E],
                 "ps_vint_bits": [0xFFFFFFFE, 0x40000000, 0x1E6A2C48, 0x000001D5],
                 "ps_vint_mul24": [0x13345678, 0xFFFFFFFD, 0x00007FFF, 0x0000FFFF],
                 "ps_vint_minmax": [0x7FFFFFFE, 0x80000001, 0x80000001, 0x7FFFFFFE],
                 "ps_vint_minmax3": [0xF0F0F0F0, 0xAB800000, 0xF0F0F0F0, 0xAB800000]}
EXPECTED_VCMP_F32_NAN = [0x0000376C, 0x0000FD5B, 0x0000D36C, 0x00000FE7]   # derivation in ps_vcmp_f32_nan.asm
EXPECTED_VCMP_64 = [0x0000B3C4, 0x000091E6, 0x00006787, 0x00007C93]        # derivation in ps_vcmp_64.asm
EXPECTED_ANDN2_EXEC = [1, 8, 16, 2064, 2068, 3, 20, 1544]                   # derivation in cs_w32_andn2_exec.asm
VCMP_F32_HEAP = (0x7FC10001, 0x3F800000, 0x40000000)                        # NaN, 1.0, 2.0
# Four (A, B) pairs as A.lo A.hi B.lo B.hi (ps_vcmp_64.asm header).
VCMP_64_HEAP = (0x80000000, 0, 1, 0, 0xFFFFFFFF, 0xFFFFFFFF, 1, 0, 1, 0, 0x80000000, 0, 7, 5, 7, 5)


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def _half(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _unhalf(h):
    return struct.unpack("<e", struct.pack("<H", h))[0]


def _s16(x):
    return x - 0x10000 if x & 0x8000 else x


def _legacy_mul(x, y):
    """v_mul_legacy_f32: zero wins over inf and NaN, including a negative zero."""
    return 0 if x == 0.0 or y == 0.0 else _f32_bits(x * y)


def reference_legacy_mul():
    return [_legacy_mul(0.0, float("inf")), _legacy_mul(0.0, float("nan")),
            _legacy_mul(-0.0, float("-inf")), _legacy_mul(2.0, 3.0)]


def _pk_shift(kind, sh, val, sel=(0, 0), selhi=(1, 1)):
    """One packed 16-bit shift. sel selects (shift-half, value-half) for the low result, selhi for the high."""
    def half(x, n):
        return (x >> (16 * n)) & 0xFFFF

    def one(shalf, vhalf):
        amount = half(sh, shalf) & 15
        x = half(val, vhalf)
        if kind == "ashr":
            return (_s16(x) >> amount) & 0xFFFF
        return ((x << amount) if kind == "lshl" else (x >> amount)) & 0xFFFF

    return one(sel[0], sel[1]) | (one(selhi[0], selhi[1]) << 16)


def reference_ops4():
    v0, shift = 0x12348001, 0x00080004
    sin = _half(math.sin(_unhalf(0x3400) * 2.0 * math.pi))          # 0.25 turns -> 1.0
    add = (0xBEEF0000 & 0xFFFF0000) | ((0x0001 + 0xFFFF) & 0xFFFF)  # carry out of the low half is dropped
    return [_pk_shift("lshl", shift, v0), _pk_shift("lshr", shift, v0), add,
            0x12340000 | sin]


def reference_ops5():
    lshl = 0x12340000 | ((0x8001 << 4) & 0xFFFF)
    lshr = 0xBEEF0000 | ((0x8001 >> 4) & 0xFFFF)
    ashr = 0xAAAA0000 | ((_s16(0xFFF0) >> 4) & 0xFFFF)
    cos = 0xDEAD0000 | _half(math.cos(_unhalf(0x3800) * 2.0 * math.pi))   # 0.5 turns -> -1.0
    return [lshl, lshr, ashr, cos]


def reference_ops6():
    return [_pk_shift("ashr", 0x00040004, 0x8000FFF0),
            # op_sel:[1,0,0] -> low result shifts by the high half of src0; op_sel_hi:[0,0,0] -> high result
            # shifts by the low half. Both use the low half of the value.
            _pk_shift("lshl", 0x00040008, 0x12348001, sel=(1, 0), selhi=(0, 0)),
            _pk_shift("lshr", 0x00040008, 0xBEEF8001, sel=(1, 0), selhi=(0, 0)),
            _pk_shift("ashr", 0, 0x8000FFF0)]


def _perm(s0, s1, sel):
    data = list(s1.to_bytes(4, "little")) + list(s0.to_bytes(4, "little"))
    out = 0
    for i in range(4):
        k = (sel >> (8 * i)) & 0xFF
        if k >= 13:
            b = 0xFF
        elif k == 12:
            b = 0
        elif k >= 8:
            b = 0xFF if data[2 * (k - 8) + 1] & 0x80 else 0
        else:
            b = data[k]
        out |= b << (8 * i)
    return out


def _s(v):
    return v - (1 << 32) if v & 0x80000000 else v


def reference_ops7():
    m = 0xFFFFFFFF
    med_i = sorted([_s(0xFFFFFFFB), 7, 3])[1] & m
    med_u = sorted([0xFFFFFFFB, 7, 3])[1]
    return [_perm(0x80112233, 0x4455E677, 0x0C0D0700),
            _perm(0x80112233, 0x4455E677, 0x0B0A0908),          # sign of bytes 1, 3, 5, 7: E6, 44, 22, 80
            (((0x12345678 << 32) | 0x9ABCDEF0) >> (36 & 31)) & m,
            ((med_u << 16) + med_i) & m]


def reference_ops8():
    m = 0xFFFFFFFF
    bcnt = bin(0xF0F00001).count("1") + 5
    ffbl = (0x00010000 & -0x00010000).bit_length() - 1
    ffbh = 32 - 0x00010000.bit_length()
    xad = ((0xF0F0F0F0 ^ m) + 1) & m
    signed_high = ((_s(0xFFFFFFFE) * 0x40000000) >> 32) & m
    return [(0x80000001 * 0xFFFFFFFF) >> 32,
            signed_high ^ 0x40000000,
            ((ffbh << 16) + (ffbl << 8) + bcnt) & m,
            (xad + 16 + m) & m]                                  # v_ffbh_u32(0) = 0xffffffff


def reference_scc_minmax():
    def minmax(op, x, y):
        """(D, SCC) of s_min/s_max _i32/_u32 on 32-bit words: SCC = 1 only when S0 is the one chosen."""
        kx, ky = (_s(x), _s(y)) if op.endswith("i32") else (x, y)
        pick = kx < ky if op.startswith("min") else kx > ky
        return x if pick else y, int(pick)

    words = []
    m1 = 0xFFFFFFFF                                   # the inline constant -1 as a register word
    for op, tie, strict in (("min_i32", 5, (m1, 1)), ("max_i32", 7, (1, m1)),
                            ("min_u32", 9, (1, m1)), ("max_u32", 3, (m1, 1))):
        d, scc_tie = minmax(op, tie, tie)
        _, scc_strict = minmax(op, *strict)
        words.append(d | ((1 if scc_tie else 2) << 8) | ((3 if scc_strict else 4) << 16))
    return words


def reference_fma_literal():
    # Every product and sum is exact in f32, so fused and unfused give the same bits.
    k1, k2 = 10.0, 0.5
    return [_f32_bits(2.0 * k1 + 3.0), _f32_bits(2.0 * 3.0 + k1),     # fmamk, fmaak
            _f32_bits(4.0 * k2 + 5.0), _f32_bits(4.0 * 5.0 + k2)]     # madmk, madak


def reference_f16_vop2():
    a, b, acc = _unhalf(0x3E00), _unhalf(0x3800), _unhalf(0x3C00)       # low halves: 1.5, 0.5, accumulator 1.0
    return [0xAAAA0000 | _half(a + b), 0xAAAA0000 | _half(-a - abs(b)), 0xAAAA0000 | _half(a * b),
            0x42000000 | _half(a * b + acc)]


def reference_cvt_f16_out_mods():
    def f16(x):         # round to nearest even; struct refuses what rounds past 65504, which is inf
        try:
            return _half(x)
        except OverflowError:
            return 0xFC00 if x < 0 else 0x7C00

    def cvt(x, omod=None, clamp=False):
        h = f16(x)
        if omod is not None and h & 0x7C00 != 0x7C00:
            v = _unhalf(h) * omod if h & 0x7C00 else 0.0
            h = f16(v) if abs(v) >= 2.0 ** -14 else 0         # zero and denormal: +0
        return _half(min(max(_unhalf(h), 0.0), 1.0)) if clamp else h

    return [0xAAAA0000 | cvt(1.5, clamp=True), 0xAAAA0000 | cvt(0.375, 4),
            cvt(2.0 ** -15, 2) | cvt(65520.0, 0.5) << 16, cvt(-0.0, 2) | cvt(2.0, 2, clamp=True) << 16]


def reference_vcc_x2():
    return [0x11223344, 0x55667788, 0, 0]


def reference_flat():
    return [0x101, 0x101 + 1, _f32_bits(0.5), _f32_bits(0.25)]


def reference_bounds():
    # num_records = 16: store 0 and load 0 are in bounds, the store and load at offset 16 are not. The OOB load
    # feeds a store of 0 into offset 8; the OOB store must leave offset 16 untouched (it is not in the map).
    return {0x00: 0x12345678, 0x04: 0x12345678, 0x08: 0, 0x0C: 0xEEEEEEEE}


WIDTHS_1010102 = {50: (10, 10, 10, 2),     # 2_10_10_10_UNORM, R10G10B10A2
                  44: (2, 10, 10, 10)}     # 10_10_10_2_UNORM


def _unorm_unpack(word, widths):
    out, bit = [], 0
    for w in widths:
        out.append(_f32_bits(((word >> bit) & ((1 << w) - 1)) / ((1 << w) - 1)))
        bit += w
    return out


def _unorm_pack(values, widths):
    word, bit = 0, 0
    for v, w in zip(values, widths):
        word |= round(min(max(v, 0.0), 1.0) * ((1 << w) - 1)) << bit
        bit += w
    return word


def reference_fmt_1010102():
    # Every field is 0 or full scale, so the decoded floats are exact.
    src50, src44, stored = 0xC00FFC00, 0x003FF003, (1.0, 0.0, 0.5, 1.0)
    words = (_unorm_unpack(src50, WIDTHS_1010102[50]) + _unorm_unpack(src44, WIDTHS_1010102[44])
             + _unorm_unpack(src50, WIDTHS_1010102[50])
             + [src50, src44, _unorm_pack(stored, WIDTHS_1010102[50]), _unorm_pack(stored, WIDTHS_1010102[44])])
    return {4 * i: w for i, w in enumerate(words)}


def _ff1(x):
    """Index of the lowest set bit of a 32-bit value, 0xffffffff if none."""
    x &= 0xFFFFFFFF
    return (x & -x).bit_length() - 1 if x else 0xFFFFFFFF


def reference_sff_imm():
    # ps_sff_imm: s_ff1 of the inline constant 4 and of the literal 0x00100000; s_ff0 of the inline constant 61
    # (0b111101) and of the literal 0x0000ffff (the lowest clear bit is the lowest set bit of ~S0).
    return [_ff1(4), _ff1(~61), _ff1(0x00100000), _ff1(~0x0000FFFF)]


def _mad_32_16(signed, s0, s1, s2, sel=(0, 0)):
    """v_mad_u32_u16 / v_mad_i32_i16: the selected 16-bit halves, extended, multiplied, plus S2, mod 2^32."""
    a, b = (s0 >> (16 * sel[0])) & 0xFFFF, (s1 >> (16 * sel[1])) & 0xFFFF
    if signed:
        a, b = _s16(a), _s16(b)
    return (a * b + s2) & 0xFFFFFFFF


def reference_valu16_mad():
    v0, v1, v3, v5, v6, v7 = 0x1234FFFF, 0x5678FFFE, 0x00030000, 0x00018000, 0xFFFF7FFF, 0x40000000
    return [_mad_32_16(False, v0, v1, v3), _mad_32_16(True, v0, v1, v3), _mad_32_16(True, v5, v6, v7),
            _mad_32_16(True, v5, v6, 5, sel=(1, 1))]


def reference_valu16_mad_sel():
    # op_sel bit 2 adds S2[31:16] zero-extended; bit 3 shifts the result left by 16; clamp saturates the sum.
    v0, v1, v3, v5, v6, v7 = 0x1234FFFF, 0x5678FFFE, 0x8001FFF0, 0x00018000, 0xFFFF7FFF, 0x80000000
    x = _mad_32_16(True, v0, v1, v3 >> 16)
    y = (_mad_32_16(False, v0, v1, v3, sel=(1, 0)) << 16) & 0xFFFFFFFF
    z = min((v0 & 0xFFFF) * (v0 & 0xFFFF) + v3, 0xFFFFFFFF)
    w = max(-(1 << 31), min((1 << 31) - 1, -32768 * 32767 - (1 << 31))) & 0xFFFFFFFF   # v5.lo * v6.lo + v7
    return [x, y, z, w]


def reference_sdwa_dword_hi():
    # Low/high halves of the sources: v2.lo 1.0, v4.lo 3.0, v4.hi -2.0, v6.hi 7. Each result is the 16-bit value
    # alone; nothing of the destinations' 0xdeadbeef is left.
    one, three, m_two = _unhalf(0x3C00), _unhalf(0x4200), _unhalf(0xC000)
    return [_half(one + one), int(three) & 0xFFFF, _half(7.0), _half(one * m_two)]


def reference_valu16_subrev():
    a, b, d = _unhalf(0x3E00), _unhalf(0x3800), _unhalf(0x3C00)        # low halves: v0 1.5, v1 0.5, v5 1.0
    return [0xAAAA0000 | _half(b - a), 0xAAAA0000 | _half(-b - a), 0xAAAA0000 | _half(min(1.0, b + a)),
            0x42000000 | _half(a - d)]


def reference_sdwa_f16_const():
    # An f16 float inline constant is its f16 bits over a zero high half; an integer op reads 1.0 as its f32 bits.
    # src_sel then picks a zero-extended field of those 32 bits, like a register's.
    def sel(word, off, width):
        return (word >> off) & ((1 << width) - 1)
    v1, s8 = 0x64003800, 0x3C000000
    x = _unhalf(sel(_half(1.0), 16, 16)) + _unhalf(sel(v1, 0, 16))          # +0.0 + 0.5
    y = _unhalf(sel(_half(2.0), 8, 8)) * _unhalf(sel(v1, 16, 16))           # 2^-18 * 1024.0
    z = _unhalf(sel(s8, 16, 16)) + _unhalf(sel(v1, 0, 16))                  # 1.0 + 0.5
    w = float(sel(struct.unpack("<I", struct.pack("<f", 1.0))[0], 16, 16))  # u16 0x3f80 -> f16
    return [0xAAAA0000 | _half(v) for v in (x, y, z, w)]


def reference_vint():
    """The ps_vint_* words from the ISA definitions, on the operands each .asm passes (hN = VINT_HEAP[N])."""
    h = VINT_HEAP
    m = 0xFFFFFFFF
    def i32(x):
        return x - (1 << 32) if x & 0x80000000 else x
    def s24(x):
        x &= 0xFFFFFF
        return x - (1 << 24) if x & 0x800000 else x
    def bfm(size, offset):
        return (((1 << (size & 31)) - 1) << (offset & 31)) & m
    def bfrev(x):
        return int(f"{x:032b}"[::-1], 2)
    return {"ps_vint_logic3": [((h[3] + h[4]) << (h[2] & 31)) & m, (h[6] & h[5]) | h[0], h[5] ^ h[6] ^ h[7],
                               (h[5] & h[6]) | (~h[5] & h[1] & m)],
            "ps_vint_bits": [bfm(h[7], h[2]), bfm(h[2], h[1]), bfrev(h[6]), bfrev(h[0])],
            "ps_vint_mul24": [(s24(h[0]) * s24(h[1]) + h[6]) & m, (s24(h[4]) * s24(h[3]) + h[7]) & m,
                              ((h[0] & 0xFFFFFF) * (h[1] & 0xFFFFFF)) >> 32,
                              ((h[7] & 0xFFFFFF) * (h[7] & 0xFFFFFF)) >> 32],
            "ps_vint_minmax": [max(h[3], h[4], key=i32), min(h[3], h[4], key=i32), max(h[3], h[4]), min(h[3], h[4])],
            "ps_vint_minmax3": [max(h[4], h[3], h[5]), max(h[0], h[6], h[3]), min(h[5], h[4], h[6], key=i32),
                                min(h[7], h[6], h[0], key=i32)]}
def _pack_nibbles(ops, pairs, test):
    """Bit 4*k + i of the word: op k on pair i."""
    return sum(int(test(op, *pair)) << (4 * k + i) for k, op in enumerate(ops) for i, pair in enumerate(pairs))
def reference_vcmp_f32_nan():
    nan = float("nan")
    pairs = ((nan, 1.0), (1.0, 2.0), (2.0, 1.0), (1.0, 1.0))
    ordered = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
               "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "lg": lambda a, b: a < b or a > b,
               "o": lambda a, b: not (math.isnan(a) or math.isnan(b)), "f": lambda a, b: False}
    def test(op, a, b):
        if op == "tru":
            return True
        if op.startswith("n"):                        # nge/ngt/nle/nlt/neq: the negated ordered compare
            return not ordered[op[1:]](a, b)
        return ordered[op](a, b)
    return [_pack_nibbles(ops, pairs, test) for ops in (("ge", "lg", "neq", "nge"), ("ngt", "nle", "nlt", "tru"),
                                                         ("ge", "lg", "nge", "nlt"), ("neq", "o", "tru", "f"))]
def reference_vcmp_64():
    w = VCMP_64_HEAP
    pairs = [((w[4 * i + 1] << 32) | w[4 * i], (w[4 * i + 3] << 32) | w[4 * i + 2]) for i in range(4)]
    def test(op, a, b):
        cond, ty = op.split("_")
        if ty == "i64":
            a, b = (x - (1 << 64) if x >> 63 else x for x in (a, b))
        return {"lt": a < b, "le": a <= b, "gt": a > b, "ge": a >= b, "eq": a == b, "ne": a != b}[cond]
    return [_pack_nibbles(ops, pairs, test) for ops in (("lt_u64", "le_u64", "gt_u64", "ge_u64"),
                                                         ("lt_i64", "le_i64", "gt_i64", "ge_i64"),
                                                         ("ne_u64", "eq_i64", "ne_i64", "lt_i64"),
                                                         ("gt_u64", "ge_i64", "le_u64", "ne_i64"))]
def reference_andn2_exec():
    def popcount(m):
        return bin(m).count("1")
    def ff1(m):
        return (m & -m).bit_length() - 1 if m else -1
    lanes = range(32)
    mask = lambda f: sum(1 << i for i in lanes if f(i))                         # noqa: E731
    out = []
    # 1) saveexec, VCC = bit 3 clear, EXEC = 0xffff
    vcc, exec_ = mask(lambda i: i & 8 == 0), 0x0000FFFF
    d, exec_ = exec_, vcc & ~exec_ & 0xFFFFFFFF
    out += [1 if exec_ else 2, popcount(exec_), popcount(d), popcount(exec_) << 8 | ff1(exec_)]
    # 2) wrexec, VCC = lanes 16..31, EXEC = 0x0f0f0f0f
    vcc, exec_ = mask(lambda i: i >= 16), 0x0F0F0F0F
    exec_ = vcc & ~exec_ & 0xFFFFFFFF
    d = exec_
    out += [popcount(exec_) << 8 | ff1(exec_), 3 if exec_ else 4, ff1(d)]
    # 3) both with an S0 inside EXEC: EXEC and SCC become 0
    vcc = mask(lambda i: i < 8)
    e1, e2 = vcc & ~0x0000FFFF & 0xFFFFFFFF, vcc & ~0xFFFFFFFF & 0xFFFFFFFF
    out.append((5 if e1 else 6) << 8 | (7 if e2 else 8))
    return out
def _isg1_dxbc(elements):
    """A DXBC container with one ISG1 input signature. elements: (register, mask, system value, component type)."""
    payload = bytearray(struct.pack("<II", len(elements), 8))
    for reg, mask, sysv, ctype in elements:
        e = bytearray(32)
        struct.pack_into("<6I", e, 0, 0, 0, 0, sysv, ctype, reg)
        e[24], e[25] = mask, 1                                # read/write mask: the shader reads it
        payload += e
    part = b"ISG1" + struct.pack("<I", len(payload)) + bytes(payload)
    part += bytes((-len(part)) % 4)
    header = b"DXBC" + bytes(16) + struct.pack("<III", 1, 0, 1)
    return header + struct.pack("<I", len(header) + 8) + bytes(4) + part


def _fssm(rsrc2, group):
    """The compute-shader header words the translator reads: COMPUTE_PGM_RSRC2 and the thread-group size."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


# attr0 = TEXCOORD0 (x/y interpolated), attr1 = TEXCOORD1 (read through v_interp_mov only, so Flat).
PS_FLAT_DXBC = _isg1_dxbc([(0, 0x3, 0, 3), (1, 0x1, 0, 3)])
FLAT_ATTR1_X = struct.unpack("<f", struct.pack("<I", 0x101))[0]
# 4 user SGPRs (the output V#), workgroup id in s4, 512 bytes of LDS, 64x1x1 threads: the header cs_bounds was
# written for in runtime/test/shaders/build.py.
CS_BOUNDS_FSSM = _fssm(0x8088, (64, 1, 1))


class OracleTests(unittest.TestCase):
    """The hand-written constants and the plain-Python reference agree (so the oracle is self-consistent)."""

    def test_reference_matches_constants(self):
        self.assertEqual(reference_legacy_mul(), EXPECTED_LEGACY_MUL)
        self.assertEqual(reference_ops4(), EXPECTED_OPS4)
        self.assertEqual(reference_ops5(), EXPECTED_OPS5)
        self.assertEqual(reference_ops6(), EXPECTED_OPS6)
        self.assertEqual(reference_ops7(), EXPECTED_OPS7)
        self.assertEqual(reference_ops8(), EXPECTED_OPS8)
        self.assertEqual(reference_vcc_x2(), EXPECTED_VCC_X2)
        self.assertEqual(reference_f16_vop2(), EXPECTED_F16_VOP2)
        self.assertEqual(reference_cvt_f16_out_mods(), EXPECTED_CVT_F16_OUT_MODS)
        self.assertEqual(reference_sdwa_dword_hi(), EXPECTED_SDWA_DWORD_HI)
        self.assertEqual(reference_flat(), EXPECTED_FLAT)
        self.assertEqual(reference_scc_minmax(), EXPECTED_SCC_MINMAX)
        self.assertEqual(reference_fma_literal(), EXPECTED_FMA_LITERAL)
        self.assertEqual(reference_bounds(), EXPECTED_BOUNDS)
        self.assertEqual(reference_fmt_1010102(), EXPECTED_FMT_1010102)
        self.assertEqual(reference_sff_imm(), EXPECTED_SFF_IMM)
        self.assertEqual(reference_valu16_mad(), EXPECTED_VALU16_MAD)
        self.assertEqual(reference_valu16_subrev(), EXPECTED_VALU16_SUBREV)
        self.assertEqual(reference_valu16_mad_sel(), EXPECTED_VALU16_MAD_SEL)
        self.assertEqual(reference_sdwa_f16_const(), EXPECTED_SDWA_F16_CONST)
        self.assertEqual(reference_vint(), EXPECTED_VINT)
        self.assertEqual(reference_vcmp_f32_nan(), EXPECTED_VCMP_F32_NAN)
        self.assertEqual(reference_vcmp_64(), EXPECTED_VCMP_64)
        self.assertEqual(reference_andn2_exec(), EXPECTED_ANDN2_EXEC)


class TranslatorValueSemantics(unittest.TestCase):
    def check(self, name, expected, dxbc=None, **run_options):
        spv = support.translate_fixture(name, dxbc=dxbc) if dxbc is not None else support.translate_fixture(name)
        words = support.mrt0_words(spv, **run_options)
        self.assertEqual(len(words), 4, "mrt0 has four components")
        for i, (got, want) in enumerate(zip(words, expected)):
            with self.subTest(shader=name, component="xyzw"[i]):
                self.assertEqual(f"{got:#010x}", f"{want:#010x}")

    def test_v_mul_legacy_f32_zero_times_inf_nan(self):
        # 0*inf, 0*NaN and -0*-inf all give +0.0; 2*3 is an ordinary multiply.
        self.check("ps_legacy_mul", EXPECTED_LEGACY_MUL)

    def test_packed_b16_shifts_add_nc_u16_and_sin_f16(self):
        # Per-half shifts with independent amounts (4 and 8), a 16-bit add whose carry is dropped while the
        # destination's high half is kept, and sin_f16 reading its input as turns.
        self.check("ps_ops4", EXPECTED_OPS4)

    def test_scalar_b16_shifts_and_cos_f16(self):
        # VOP3 forms keep the destination's high half; ashrrev sign-extends the 16-bit value before shifting.
        self.check("ps_ops5", EXPECTED_OPS5)

    def test_packed_b16_shifts_with_op_sel(self):
        # Packed ashrrev (both halves negative) and lshl/lshr selecting source halves through op_sel/op_sel_hi.
        self.check("ps_ops6", EXPECTED_OPS6)

    def test_perm_alignbit_and_med3(self):
        # v_perm_b32 with plain byte picks, the 0x00/0xff constants and the sign selectors 8-11; v_alignbit_b32
        # with a shift of 36 (taken mod 32); v_med3_i32 and v_med3_u32 on the same operands (-5, 7, 3).
        self.check("ps_ops7", EXPECTED_OPS7)

    def test_mul_hi_bit_counts_xad_add3(self):
        # High halves of unsigned and signed products, popcount with an accumulator, lowest/highest set bit,
        # v_ffbh_u32(0) = -1 feeding a wrapping v_add3_u32, and v_xad_u32.
        self.check("ps_ops8", EXPECTED_OPS8)

    def test_s_min_max_scc_on_a_tie(self):
        # Equal operands give S1 and SCC 0 for s_min/s_max _i32 and _u32; -1 vs 1 sets SCC only when compared
        # with the op's own signedness. A tie that set SCC turns byte 1 of a word from 2 into 1.
        self.check("ps_scc_minmax", EXPECTED_SCC_MINMAX)
    def test_fmamk_madmk_multiply_by_the_literal(self):
        # fmamk/madmk multiply S0 by the literal and add vsrc1; fmaak/madak multiply S0 by vsrc1 and add the
        # literal. Swapping K and vsrc1 changes every word (16 and 20.5 instead of 23 and 7, and back).
        self.check("ps_fma_literal", EXPECTED_FMA_LITERAL)

    def test_f16_vop2_partial_writes(self):
        # Low halves in (the high halves hold other values), low half out, the destination's high half kept;
        # v_fmac_f16 accumulates into the low half of its destination.
        self.check("ps_f16_vop2", EXPECTED_F16_VOP2)

    def test_cvt_f16_f32_output_modifiers(self):
        # clamp, mul:4, and the RX 6800's order: 65520 div:2 is inf, 2^-15 mul:2 and -0 mul:2 are +0, 2.0 mul:2 clamp 1.0
        self.check("ps_outmods_f16", EXPECTED_CVT_F16_OUT_MODS)

    def test_sdwa_16bit_result_dst_sel_dword_zeroes_high_half(self):
        # v_add_f16 / v_cvt_u16_f16 / v_cvt_f16_u16 / v_mul_f16 SDWA into 0xdeadbeef with dst_sel:DWORD and
        # dst_unused PRESERVE, PAD, PRESERVE, SEXT: the high half is zero in each, unlike the plain VOP1/VOP2 forms.
        self.check("ps_sdwa_dword_hi", EXPECTED_SDWA_DWORD_HI)

    def test_mad_u32_u16_i32_i16_wrap_and_sign_extend(self):
        # The u16 product plus S2 wraps at 32 bits; i16 sign-extends both halves (also the op_sel-selected high
        # halves); the destination dword is replaced, its 0xaaaa5555 is not kept.
        self.check("ps_valu16_mad", EXPECTED_VALU16_MAD)

    def test_mad_u32_u16_i32_i16_op_sel_s2_d_and_clamp(self):
        # EGEN-HW-MAD-U32-U16: S2's high half zero-extended even in the i16 op (0x8001, not -32767), the result
        # shifted into D's high half, and u32 / i32 saturation; all four used to be Unsupported.
        self.check("ps_valu16_mad_sel", EXPECTED_VALU16_MAD_SEL)

    def test_subrev_f16_reversed_and_keeps_high_half(self):
        # S1 - S0 in the _e32 and _e64 forms (neg, clamp), low half out, the destination's high half kept, also
        # when the destination is a source.
        self.check("ps_valu16_subrev", EXPECTED_VALU16_SUBREV)

    def test_sdwa_src_sel_narrows_an_f16_inline_constant(self):
        # 1.0 in v_add_f16_sdwa is 0x00003c00, so src0_sel:WORD_1 reads +0.0 and BYTE_1 of 2.0 is the denormal
        # 0x0040; an SGPR source and v_cvt_f16_u16 reading the f32 bits of 1.0 are narrowed the same way.
        self.check("ps_sdwa_f16_const", EXPECTED_SDWA_F16_CONST)

    def test_f16_denormal_operand_does_not_rely_on_unpack_half(self):
        # On NVIDIA (RTX 5060 Ti, 610.88) UnpackHalf2x16 gives 0 for the denormal 0x0040 that word y's BYTE_1
        # select produces. Here every denormal is flushed, so y stays 0x1c00 only if they skip UnpackHalf2x16.
        real = si.half_to_float

        def flushing(h):
            return math.copysign(0.0, -1.0 if h & 0x8000 else 1.0) if (h & 0x7C00) == 0 else real(h)

        with mock.patch.object(si, "half_to_float", flushing):
            self.check("ps_sdwa_f16_const", EXPECTED_SDWA_F16_CONST)

    def test_s_buffer_load_dwordx2_into_vcc(self):
        # The V# in s[0:3] points at 0x1000; the two dwords must come back in vcc_lo and vcc_hi, not be lost.
        push = [0x1000, 0, 16, 0] + [0] * 28
        self.check("ps_vcc_x2", EXPECTED_VCC_X2, push_constants=push,
                   memory={0x1000: 0x11223344, 0x1004: 0x55667788})

    def test_flat_input_and_interpolated_input(self):
        # attr1.x is read only through v_interp_mov, so it carries the provoking vertex's 0x101 unchanged and
        # integer maths works on it; attr0.xy is read through v_interp and keeps the supplied values.
        self.check("ps_flat", EXPECTED_FLAT, dxbc=PS_FLAT_DXBC,
                   inputs={"%in_attr0": (0.5, 0.25, 0.0, 1.0), "%in_attr1": (FLAT_ATTR1_X,) * 4})

    def test_flat_input_is_declared_flat(self):
        spv = support.translate_fixture("ps_flat", dxbc=PS_FLAT_DXBC)
        self.assertIn("OpDecorate %in_attr1 Flat", spv)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)

    def test_buffer_num_records_bounds(self):
        # cs_bounds builds a V# with base 0, stride 0, num_records 16 and runs one thread. The output V# in
        # s[0:3] is the same memory. The out-of-bounds store at offset 16 must not land anywhere, and the OOB
        # load must come back as 0.
        spv = support.translate_fixture("cs_bounds", fssm=CS_BOUNDS_FSSM)
        memory = {0x0C: 0xEEEEEEEE}
        outputs = support.run_fragment(
            spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 16, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual(outputs, {})
        for address, want in EXPECTED_BOUNDS.items():
            with self.subTest(address=address):
                self.assertEqual(memory.get(address), want)
        with self.subTest(what="out-of-bounds store"):
            self.assertNotIn(0x10, memory)

    def test_buffer_format_10_10_10_2_channel_order(self):
        # The layout of out[] is in the comment at the top of cs_fmt_1010102.asm.
        spv = support.translate_fixture("cs_fmt_1010102", fssm=CS_BOUNDS_FSSM)
        memory = {}
        outputs = support.run_fragment(
            spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 0x100, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual(outputs, {})
        for address, want in EXPECTED_FMT_1010102.items():
            with self.subTest(address=address):
                self.assertEqual(f"{memory.get(address, 0):#010x}", f"{want:#010x}")

    def test_narrow_integer_typed_buffers(self):
        # cs_fmt_int_narrow.asm in the interpreter against the words check.py expects from the GPU, which are the
        # same program's output on gfx1036 run natively: a 136-word heap filled with 0xee as vk_compute_test's.
        sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
        try:
            import check
        finally:
            sys.path.pop(0)
        spv = support.translate_fixture("cs_fmt_int_narrow", fssm=CS_BOUNDS_FSSM)
        memory = {4 * i: 0xEEEEEEEE for i in range(len(check.FMT_INT_NARROW_WORDS))}
        support.run_fragment(
            spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        for i, word in enumerate(check.FMT_INT_NARROW_WORDS):
            with self.subTest(word=i):
                self.assertEqual(memory[4 * i], word)

    def test_signed_typed_buffers(self):
        spv = support.translate_fixture("cs_fmt_sint32", fssm=CS_BOUNDS_FSSM)
        memory = {}
        support.run_fragment(
            spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 0x100, 0x31016FAC] + [0] * 28, memory=memory)
        expected = {
            0x0C: 1,                         # MUBUF integer DST_SEL=1
            0x18: 0xFFFFFFFD,                # MUBUF 96-bit format, third component
            0x34: 0x7FFFFFFF, 0x38: 0,       # MTBUF 64-bit format with absent third component
            0x48: 0xFFFFFFFD, 0x4C: 0,       # MTBUF 96-bit format with absent fourth component
            0x58: 0xFFFFFFFD, 0x5C: 17,      # MTBUF 128-bit format
            0x60: 0x7FFF8000, 0x64: 0x0011FFFD,
            0x68: 0x7FFF8000, 0x6C: 0x0011FFFD,
            0x70: 0xFFFFFFFD, 0x74: 0xFFFFFFFD,
            0x90: 0x80000000, 0xA8: 0xFFFFFFFD, 0xBC: 17,
            0xC0: 0xFFFFFFFD, 0xC4: 0xFFFF8000, 0xC8: 0xFFFFFFFD,
        }
        for address, word in expected.items():
            with self.subTest(address=hex(address)):
                self.assertEqual(memory.get(address), word)
        for address in (0x98, 0xAC, 0xCC):
            self.assertNotIn(address, memory)

    def test_d16_xyz_typed_buffers(self):
        # cs_fmt_d16_xyz.asm (derivation in its header): buffer_load_format_d16_xyz, tbuffer_load_format_d16_xy /
        # _d16_xyz and buffer_store_format_d16_xyz.
        def f16(x):
            return struct.unpack("<H", struct.pack("<e", x))[0]

        def f32(x):
            return struct.unpack("<I", struct.pack("<f", x))[0]

        def pack(lo, hi):
            return (hi & 0xFFFF) << 16 | (lo & 0xFFFF)

        sat16 = lambda v: min(v, 0xFFFF)             # unsigned 32-bit channel narrowed to 16 bits (docs/design.md)
        keep_hi = lambda seed, lo: (seed & 0xFFFF0000) | lo   # an odd final component writes bits 15:0 only
        spv = support.translate_fixture("cs_fmt_d16_xyz", fssm=CS_BOUNDS_FSSM)
        memory = {}
        support.run_fragment(
            spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 0x140, 0x31016FAC] + [0] * 28, memory=memory)
        expected = {
            0x00: pack(0x1234, 0xABCD), 0x04: keep_hi(0xA5A55A5A, 0x7FFF),                 # L1 16_16_16_16_UINT
            0x08: pack(5, sat16(0x10000)), 0x0C: keep_hi(0x12345678, sat16(0xFFFFFFFF)),    # L2 32_32_32_UINT
            0x10: pack(f16(1.0), f16(-2.0)),                                                  # L3 16_16_FLOAT
            0x14: pack(f16(1.0), f16(-2.0)), 0x18: keep_hi(0xBEEF7777, 0x3555),             # L4 16_16_16_16_FLOAT
            0x1C: pack(f16(255 / 255), f16(0 / 255)), 0x20: keep_hi(0xCAFE1111, f16(1.0)),  # L5 8_8_8_8_UNORM
            0x24: pack(5, 0xFFFF), 0x28: keep_hi(0x0BAD0BAD, 0xFFFF),                       # L6 MTBUF 32_32_32_UINT
            0x40: 0x0007, 0x44: 0x8001, 0x48: 0xFFFE,                                         # S1 to 32_32_32_UINT
            0x50: f32(1.0), 0x54: f32(-2.0), 0x58: f32(0.5),                                  # S2 to 32_32_32_FLOAT
        }
        self.assertEqual(expected[0x00], 0xABCD1234)
        self.assertEqual(expected[0x10], 0xC0003C00)
        for address, word in expected.items():
            with self.subTest(address=hex(address)):
                self.assertEqual(f"{memory.get(address, 0):#010x}", f"{word:#010x}")
        for address in (0x2C, 0x4C, 0x5C):
            self.assertNotIn(address, memory)

    def test_s_ff0_s_ff1_read_an_immediate_as_a_value(self):
        # SHADER-46: an inline constant or literal is a 32-bit value. Read as a lane mask (all ones or zero from
        # bit 0), s_ff0 of 61 and of 0x0000ffff gave -1; the s_ff1 words guard the path it already got right.
        self.check("ps_sff_imm", EXPECTED_SFF_IMM)

    def test_int_valu_sign_extension_shift_masks_and_min_max_signedness(self):
        # The ps_vint_* GPU fixtures, inputs from memory: 24-bit operands with bit 23 set and garbage in bits
        # 24-31, shift amounts and field sizes of 32 and more (taken mod 32), and the same operand pair compared
        # signed and unsigned. The operand choices and the wrong answers they catch are in the .asm headers.
        push = [0x1000, 0, 32, 0] + [0] * 28
        for name, want in EXPECTED_VINT.items():
            self.check(name, want, push_constants=push,
                       memory={0x1000 + 4 * i: w for i, w in enumerate(VINT_HEAP)})

    def test_vcmp_f32_ordered_and_unordered_with_nan(self):
        # ge/lg/o (ordered) are false with a NaN source, nge/ngt/nle/nlt/neq (unordered) true; v_cmp e32 into VCC,
        # e64 into an SGPR, and v_cmpx, whose EXEC is copied out and put back. Inputs come from memory.
        self.check("ps_vcmp_f32_nan", EXPECTED_VCMP_F32_NAN, push_constants=[0x1000, 0, 16, 0] + [0] * 28,
                   memory={0x1000 + 4 * i: w for i, w in enumerate(VCMP_F32_HEAP)})

    def test_vcmp_64_bit_integer_compares(self):
        # lt/le/gt/ge of u64 and i64, ne/eq: the high words decide, equal high words compare the low words
        # unsigned for both signednesses (0x80000000 > 1); v_cmpx forms too.
        self.check("ps_vcmp_64", EXPECTED_VCMP_64, push_constants=[0x1000, 0, 64, 0] + [0] * 28,
                   memory={0x1000 + 4 * i: w for i, w in enumerate(VCMP_64_HEAP)})

    def test_s_andn2_saveexec_and_wrexec_b32(self):
        # A 32-thread wave32: saveexec returns the old EXEC, wrexec the new one; EXEC = S0 & ~EXEC and SCC is
        # whether it is non-zero (cs_w32_andn2_exec.asm).
        spv = support.translate_fixture("cs_w32_andn2_exec", fssm=_fssm(0x8088, (32, 1, 1)))
        memory = si.run_workgroup(
            spv, 32, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28, memory={})
        self.assertEqual(sorted(memory), list(range(0, 32, 4)), "lane 0 stores out[0..7] only")
        self.assertEqual([memory[4 * k] for k in range(8)], EXPECTED_ANDN2_EXEC)


class ReadLog(dict):
    """A {byte_address: word} memory for run_fragment that records the address of every load."""

    def __init__(self, *args):
        super().__init__(*args)
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        return super().get(key, default)


VEC_REGION = 0x1000


def _pattern_word(offset):
    # byte n of the region holds 0x40 + n (as in the measurement), so an unaligned dword is its four exact bytes
    return int.from_bytes(bytes((0x40 + offset + k) & 0xFF for k in range(4)), "little")


class RawBufferTailBounds(unittest.TestCase):
    """Raw MUBUF loads (V# stride 0) are bounded per dword, measured on an AMD Radeon RX 6800 (gfx1030), raw
    compute kernel, wave32 and wave64, V# word3 0x31016fac (OOB_SELECT 3):

        O = inst_offset + soffset + (offen ? vaddr : 0)        one byte offset, nothing truncated
        dword i = data at base + O + 4*i (byte exact) if O + 4*i + 4 <= NUM_RECORDS, else 0 (and not read)

    Each case is a one-thread compute program written by hand around one buffer_load (row names of the oracle's
    model in the comments; rule in docs/design.md, Buffer bounds checking). It runs in spirv_mini_interp with a
    memory whose every byte address holds the dword starting there; the test asserts the words stored and the
    exact addresses loaded. Since 2e8101b the interpreter raises on a PhysicalStorageBuffer access that is not
    4-byte aligned (undefined under the translator's Aligned 4), so a row that loads at an unaligned O sets
    base_low to make base + O aligned: the bound sees only O, so the measured O / NUM_RECORDS pairs are kept, but
    byte-exact data at an unaligned address is no longer checked here. The last tests pin paths the measurement did not cover and that keep their old check
    of the first byte: structured buffers and idxen (OOB_SELECT 1).
    """

    def run_program(self, load_lines, n, *, nr, base_low=0, word1=0, word3=0x31016FAC, offset=0, extra=()):
        lines = ["v_cmpx_eq_u32_e32 0, v0",
                 f"s_mov_b32 s4, {VEC_REGION + base_low:#x}", f"s_mov_b32 s5, {word1:#x}",
                 f"s_mov_b32 s6, {nr:#x}", f"s_mov_b32 s7, {word3:#x}",
                 f"v_mov_b32 v1, {offset:#x}", *extra, *load_lines, "s_waitcnt vmcnt(0)"]
        lines += [f"buffer_store_dword v{2 + i}, off, s[0:3], 0 offset:{4 * i}" for i in range(n)]
        lines.append("s_endpgm")
        spv = support.translate_lines("cs_vec_tail", lines, fssm=CS_BOUNDS_FSSM)
        memory = ReadLog({VEC_REGION + k: _pattern_word(k - base_low) for k in range(-4, 0x40)})
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=[0, 0, 0x100, 0x31016FAC] + [0] * 28, memory=memory)
        words = [dict.get(memory, 4 * i) for i in range(n)]
        return words, [a - VEC_REGION - base_low for a in memory.reads]

    def check(self, line, n, want, **kw):
        """want: per dword, the byte offset O + 4*i whose data it holds, or None for 0 (and no load)."""
        words, reads = self.run_program([line], n, **kw)
        self.assertEqual([f"{w:#010x}" for w in words],
                         [f"{0 if o is None else _pattern_word(o):#010x}" for o in want], line)
        self.assertEqual([hex(r) for r in reads], [hex(o) for o in want if o is not None], line)

    def test_single_dword_needs_all_four_bytes(self):
        # vx1-k0[1], lane 15: offset 0xf, NR 18 -> 0 (a start-only check, 0xf < 18, would read)
        self.check("buffer_load_dword v2, v1, s[4:7], 0 offen", 1, [None], nr=18, offset=0xF)
        # offset 0xe, NR 18: bytes 0xe..0x11 fit -> data (base_low 2: base + O aligned, see the class docstring)
        self.check("buffer_load_dword v2, v1, s[4:7], 0 offen", 1, [0xE], nr=18, offset=0xE, base_low=2)

    def test_multi_dword_load_is_bounded_per_dword(self):
        # vx4a-k0[4..7], lane 3: x4 at offset 3, NR 18 -> d0..d2 data, d3 (bytes 15..18) 0
        # (base_low here and below makes base + O aligned for the interpreter, see the class docstring)
        self.check("buffer_load_dwordx4 v[2:5], v1, s[4:7], 0 offen", 4, [3, 7, 0xB, None], nr=18, offset=3,
                   base_low=1)
        # lane 2: d3 = bytes 14..17 fits -> all four data
        self.check("buffer_load_dwordx4 v[2:5], v1, s[4:7], 0 offen", 4, [2, 6, 0xA, 0xE], nr=18, offset=2,
                   base_low=2)
        # vmix2-k0[4..7]: x4 off offset:13, NR 20 -> data, 0, 0, 0
        self.check("buffer_load_dwordx4 v[2:5], off, s[4:7], 0 offset:13", 4, [13, None, None, None], nr=20,
                   base_low=3)
        # x2 at 8, NR 0xe: d0 data, d1 0
        self.check("buffer_load_dwordx2 v[2:3], v1, s[4:7], 0 offen", 2, [8, None], nr=0xE, offset=8)

    def test_offsets_are_summed_before_the_check(self):
        # vx1-k*[7..13] style: soffset 3 + inst_offset 2 + vaddr 0xb = 0x10; NR 0x16 -> x2: data, 0
        self.check("buffer_load_dwordx2 v[2:3], v1, s[4:7], s8 offen offset:2", 2, [0x10, None],
                   nr=0x16, offset=0xB, extra=("s_mov_b32 s8, 3",))

    def test_base_low_bits_kept_in_the_address_not_in_the_bound(self):
        # vx4a-k3[4], [7] measured base & 3 = 3 with x4 at offset 3, NR 18 -> d0..d2 data at base + 3.., d3 0.
        # Stand-in at offset 5 so that base + O is aligned for the interpreter (class docstring): d0..d2 data,
        # d3 (bytes 0x11..0x14) 0; a bound that added base & 3 would also zero d2 (3 + 5 + 8 + 4 = 0x14 > 18).
        self.check("buffer_load_dwordx4 v[2:5], v1, s[4:7], 0 offen", 4, [5, 9, 0xD, None],
                   nr=18, offset=5, base_low=3)

    def test_num_records_zero_reads_nothing(self):
        self.check("buffer_load_dwordx4 v[2:5], v1, s[4:7], 0 offen", 4, [None] * 4, nr=0, offset=0)

    # Modes 1 and 2 (OOB_SELECT) check the index or NUM_RECORDS once for the whole load; modes 0 and 3 are per dword
    # (docs/buffer_bounds.md; tests/semantic/test_buffer_oob_select.py has the full table).
    def test_structured_buffer_keeps_its_first_byte_check(self):
        # OOB_SELECT 1, stride 4, NR 2 records (8 bytes): no idxen, so index 0 < 2 and every dword is loaded
        self.check("buffer_load_dwordx4 v[2:5], v1, s[4:7], 0 offen", 4, [4, 8, 0xC, 0x10],
                   nr=2, offset=4, word1=4 << 16, word3=0x11016FAC)

    def test_idxen_keeps_its_first_byte_check(self):
        # OOB_SELECT 1, idxen with index 0 and offset:0xc, NR 0xe: index 0 < 0xe, so both dwords are loaded
        self.check("buffer_load_dwordx2 v[2:3], v1, s[4:7], 0 idxen offset:12", 2, [0xC, 0x10], nr=0xE, offset=0,
                   word3=0x11016FAC)

    def test_store_is_per_dword(self):
        # x2 at 0xc with NR 0x10: the first dword ends at 0x10 and lands, the second (ends at 0x14) is dropped
        lines = ["v_cmpx_eq_u32_e32 0, v0", f"s_mov_b32 s4, {VEC_REGION:#x}", "s_mov_b32 s5, 0",
                 "s_mov_b32 s6, 0x10", "s_mov_b32 s7, 0x31016fac", "v_mov_b32 v1, 0xc",
                 "v_mov_b32 v2, 0x11111111", "v_mov_b32 v3, 0x22222222",
                 "buffer_store_dwordx2 v[2:3], v1, s[4:7], 0 offen", "s_endpgm"]
        spv = support.translate_lines("cs_vec_store", lines, fssm=CS_BOUNDS_FSSM)
        memory = {}
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=[0, 0, 0x100, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual(memory, {VEC_REGION + 0xC: 0x11111111})


class MeasuredBoundsShaders(unittest.TestCase):
    """The four GPU tests of the measured SMEM / raw MUBUF rules (runtime/test/shaders/cs_smem_bounds.asm,
    cs_smem_stride_align.asm, cs_smem_sload_neg.asm, cs_buf_tail.asm), run here in the interpreter on a 256-byte
    heap filled with 0xee like vk_compute_test's. The words are the ones derived in each .asm header."""

    WANT = {
        "cs_smem_bounds": [0, 0x5EED0010, 0x5EED000C, 0, 0x5EED000C, 0x5EED0010, 0, 0],
        "cs_smem_stride_align": [0x5EED0010, 0, 0, 0x5EED0000, 0x5EED0000, 0x5EED0000, 0x5EED0000, 0x5EED0004],
        "cs_smem_sload_neg": [0x5EED0004, 0x5EED0000, 0x5EED0004, 0x5EED0004, 0x5EED0008, 0x5EED000C,
                              0xEEEEEEEE, 0xEEEEEEEE],
        # SHADER-328: constant and run-time SGPR offsets agree; out[4..7] are not written (heap fill)
        "cs_smem_sconst_off": [0, 0x5EED0018, 0x5EED0018, 0x5EED0014, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE, 0xEEEEEEEE],
        "cs_buf_tail": [0, 0x5EED000C, 0x5EED0008, 0x5EED000C, 0, 0, 0x5EED0010, 0],
        "cs_oob_select": [0x5EED000C, 0, 0x5EED0014, 0, 0x5EED0018, 0xEEEEEEEE, 0, 0],
        "cs_oob_tail": [0, 0, 0x5EED0010, 0, 0xBEEF0000, 0xEEEEEEEE, 0, 0x5EED0010],
    }

    def test_words(self):
        for name, want in self.WANT.items():
            with self.subTest(shader=name):
                spv = support.translate_fixture(name, fssm=CS_BOUNDS_FSSM)
                memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
                support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                                     push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
                self.assertEqual([f"{memory[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in want])
                self.assertEqual(set(memory), set(range(0, 0x100, 4)), "nothing written outside the heap")


class LaneMaskConstants(unittest.TestCase):
    """runtime/test/shaders/cs_w32_mask_imm.asm and cs_w64_mask_imm.asm: a constant read as a lane mask (saveexec,
    wrexec, s_and_b64 / s_not_b64 / s_mov_b64 into EXEC or VCC) gives lane L its bit L, not bit 0 for every lane.
    Run on all lanes of one wave (32 on a 32-wide subgroup, 64 on a 64-wide one, no --wave64-emulate) with a heap
    of 0xee bytes; the words are the GPU check's (check.py CS_DUMPS), derived by hand in each .asm header."""

    WANT = {"cs_w32_mask_imm": (32, [2056, 20, 2, 6152, 2050, 4112, 258, 32]),
            "cs_w64_mask_imm": (64, [2056, 1084, 828, 2052, 319, 2104, 516, 2064])}
    # Bit 0 of the constant for every lane, the translation before the fix (in the .asm headers).
    OLD = {"cs_w32_mask_imm": [0xFFFFFF00, 32, 32, 0xFFFFFF00, 2072, 0xFFFFFF00, 0xFFFFFF00, 32],
           "cs_w64_mask_imm": [0xFFFFFF00] * 7 + [2080]}

    def run_words(self, name, lanes):
        sys.path.insert(0, os.path.join(support.ROOT, "runtime", "test", "shaders"))
        from compute_fixture import compute_header
        header = compute_header(os.path.join(support.ROOT, "runtime", "test", "shaders", name + ".asm"))
        spv = support.translate_fixture(name, fssm=bytes(header))
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, lanes, lanes,
                         inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual(set(memory), set(range(0, 0x100, 4)), "nothing written outside the heap")
        return [memory[4 * i] for i in range(8)]

    def test_words(self):
        for name, (lanes, want) in self.WANT.items():
            with self.subTest(shader=name):
                got = self.run_words(name, lanes)
                self.assertEqual([f"{w:#010x}" for w in got], [f"{w:#010x}" for w in want])
                self.assertNotEqual(got, self.OLD[name])

    def test_words_follow_from_the_masks(self):
        # Plain-Python reference: word = set lanes | lowest lane << 8 of each case's resulting mask.
        def word(m):
            return 0xFFFFFF00 if m == 0 else bin(m).count("1") | ((m & -m).bit_length() - 1) << 8
        m32, m64 = 0xFFFFFFFF, (1 << 64) - 1
        w32 = [word(0xFF00 & m32), word(0x55555555 | 0xFF), word(5 & m32), word(0xFF00FF00 & ~0xFFFF & m32),
               word(0x3FF & ~0xFF), word(-16 & m32 & ~0xFFFF), word(0x80000002 | 0), word(-1 & m32)]
        w64 = [word(0xFF00 & m64), word(-16 & m64), word(-8 & m64 & ~0x10), word(0xFF00 & ~0xF0F0 & m64),
               word(-2 & m64), word(~0xFF & m64), word(0x3C), bin(0xFF00FF).count("1") | 8 << 8]
        self.assertEqual(w32, self.WANT["cs_w32_mask_imm"][1])
        self.assertEqual(w64, self.WANT["cs_w64_mask_imm"][1])


if __name__ == "__main__":
    unittest.main()
