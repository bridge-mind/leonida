"""MODE.FP_ROUND[3:2] (f16/f64 rounding) written with s_setreg_imm32_b32, run in the SPIR-V interpreter.

The game switches f16/f64 rounding to toward zero (value 3) and back to nearest even (value 0) inside programs.
Values are picked so the two modes give different bits: 1 + 0.75 * 2^-10 rounds up to the next f16 under nearest
even and down under toward zero; 1 - 2^-60 rounds to 1.0 in f64 under nearest even and to 1 - 2^-53 toward zero.
"""
import pytest

import rdna2spirv
import spirv_interp as si
from listing import translate_listing

RTZ = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 3\n"
RNE = "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 0\n"
X = 0x3F801800             # f32 1 + 0.75 * 2^-10
NX = X | 0x80000000


def vgprs(tmp_path, asm):
    return si.run(translate_listing(tmp_path, asm)).vgpr


@pytest.mark.parametrize("prefix,src,want", [
    ("", X, 0x3C01),                  # no MODE write: nearest even, as before
    (RTZ, X, 0x3C00),
    (RTZ, NX, 0xBC00),
    (RNE, X, 0x3C01),
    (RTZ, 0x4788B800, 0x7BFF),        # 70000.0: toward zero saturates to 65504 instead of inf
    (RTZ, 0x7F800000, 0x7C00),        # inf stays inf
    (RTZ, 0x33C00000, 0x0001),        # 1.5 * 2^-24: f16 subnormal, truncated (nearest even gives 0x0002)
    ("", 0x33C00000, 0x0002),
    (RTZ, 0x33000000, 0x0000),        # 2^-25: below the smallest f16 subnormal
    (RTZ, 0xC7800000, 0xFBFF),        # -65536.0 saturates to -65504
])
def test_cvt_f16_f32(prefix, src, want, tmp_path):
    asm = f"v_mov_b32 v1, {src:#x}\n{prefix}v_cvt_f16_f32_e32 v2, v1\n"
    assert vgprs(tmp_path, asm)[2] & 0xFFFF == want


def test_mode_switches_back(tmp_path):
    asm = f"v_mov_b32 v1, {X:#x}\n{RTZ}v_cvt_f16_f32_e32 v2, v1\n{RNE}v_cvt_f16_f32_e32 v3, v1\n"
    v = vgprs(tmp_path, asm)
    assert (v[2] & 0xFFFF, v[3] & 0xFFFF) == (0x3C00, 0x3C01)


def test_pk_add_f16_both_halves(tmp_path):
    # lo: 1.0 + 1.5 * 2^-11 (0.75 f16 ulp past 1.0), hi: the same negated; toward zero gives 1.0 and -1.0
    one, small = 0x3C00, 0x1200          # 1.0, 1.5 * 2^-11
    asm = (f"v_mov_b32 v1, {(one | 0x8000) << 16 | one:#x}\nv_mov_b32 v2, {(small | 0x8000) << 16 | small:#x}\n"
           f"{RTZ}v_pk_add_f16 v3, v1, v2\n")
    assert vgprs(tmp_path, asm)[3] == 0xBC00 << 16 | 0x3C00


# 2 - 2^-24: an f32 tie that rounds up to 2.0, so toward zero must look at the exact sum (0x3FFF), not the f32 one.
TWO, TINY, NTINY = 0x4000, 0x0001, 0x8001


@pytest.mark.parametrize("prefix,asm,want", [
    ("", "v_sub_f16_e32 v3, v1, v2", 0x4000),
    (RTZ, "v_sub_f16_e32 v3, v1, v2", 0x3FFF),
    (RTZ, "v_add_f16_e32 v3, v1, v4", 0x3FFF),
    (RTZ, "v_add_f16_e32 v3, v1, v2", 0x4000),                 # 2 + 2^-24 toward zero is 2.0
    (RTZ, "v_fma_f16 v3, v1, v5, v4", 0x3FFF),                 # 2 * 1 - 2^-24
    (RTZ, "v_add_f16_e64 v3, v1, v4 op_sel:[0,0,1]", 0x3FFF << 16 | 0x5555),   # high destination half
])
def test_f16_sum_rounds_from_the_exact_value(prefix, asm, want, tmp_path):
    setup = (f"v_mov_b32 v1, {TWO:#x}\nv_mov_b32 v2, {TINY:#x}\nv_mov_b32 v4, {NTINY:#x}\nv_mov_b32 v5, 0x3c00\n"
             "v_mov_b32 v3, 0x5555\n")
    got = vgprs(tmp_path, setup + prefix + asm + "\n")[3]
    assert (got if want > 0xFFFF else got & 0xFFFF) == want


def test_pk_fma_f16_exact_toward_zero(tmp_path):
    # both halves: 2 * 1 + (-2^-24) -> 0x3FFF, and -(2 * 1) + 2^-24 -> 0xBFFF
    asm = (f"v_mov_b32 v1, {(TWO | 0x8000) << 16 | TWO:#x}\nv_mov_b32 v2, 0x3c003c00\n"
           f"v_mov_b32 v4, {TINY << 16 | NTINY:#x}\n{RTZ}v_pk_fma_f16 v3, v1, v2, v4\n")
    assert vgprs(tmp_path, asm)[3] == 0xBFFF << 16 | 0x3FFF


@pytest.mark.parametrize("prefix,want", [("", (0, 0x3FF00000)), (RTZ, (0xFFFFFFFF, 0x3FEFFFFF))])
def test_add_f64(prefix, want, tmp_path):
    asm = ("v_mov_b32 v0, 0\nv_mov_b32 v1, 0x3ff00000\nv_mov_b32 v2, 0\nv_mov_b32 v3, 0xbc300000\n"
           f"{prefix}v_add_f64 v[4:5], v[0:1], v[2:3]\n")
    v = vgprs(tmp_path, asm)
    assert (v[4], v[5]) == want


@pytest.mark.parametrize("a,b,prefix,want", [
    (0x4000, 0x0001, "", 0x4000),            # 2 + 2^-24: a quarter f16 ulp above 2.0; both modes give 2.0
    (0x4000, 0x0001, RTZ, 0x4000),
    (0x7BFF, 0x4C00, "", 0x7C00),            # 65504 + 16 = 65520: the tie rounds to inf, toward zero stays 65504
    (0x7BFF, 0x4C00, RTZ, 0x7BFF),
])
def test_f16_add_hardware_rows(a, b, prefix, want, tmp_path):
    # words measured on an RX 6800 (SHADER-35)
    asm = f"v_mov_b32 v1, {a:#x}\nv_mov_b32 v2, {b:#x}\n{prefix}v_add_f16_e32 v3, v1, v2\n"
    assert vgprs(tmp_path, asm)[3] & 0xFFFF == want


@pytest.mark.parametrize("prefix,want", [("", 0xBC01), (RTZ, 0xBC00)])
def test_f16_sub_hardware_row(prefix, want, tmp_path):
    # -1 - (2^-11 + 2^-21): above the half-ulp tie, so nearest rounds away (RX 6800)
    asm = f"v_mov_b32 v1, 0xbc00\nv_mov_b32 v2, 0x1001\n{prefix}v_sub_f16_e32 v3, v1, v2\n"
    assert vgprs(tmp_path, asm)[3] & 0xFFFF == want


@pytest.mark.parametrize("prefix,want", [("", (0, 0x3FF00000)), (RTZ, (0xFFFFFFFF, 0x3FEFFFFF))])
def test_add_f64_with_omod_not_asserted(prefix, want, tmp_path):
    # (1 - 2^-60) + 0 toward zero, no omod. The hardware ignores omod when f64 denormals are kept (3ff00000:0 and
    # 3fefffff:ffffffff unscaled with mul:2, RX 6800), the translator applies it either way: that is unmodelled
    # (docs/design.md), so no omod order is claimed here.
    asm = ("v_mov_b32 v0, 0\nv_mov_b32 v1, 0x3ff00000\nv_mov_b32 v2, 0\nv_mov_b32 v3, 0xbc300000\n"
           f"{prefix}v_add_f64 v[4:5], v[0:1], v[2:3]\n")
    v = vgprs(tmp_path, asm)
    assert (v[4], v[5]) == want


@pytest.mark.parametrize("op,src", [
    ("v_rcp_f16_e32", 0x3037), ("v_rsq_f16_e32", 0x3037), ("v_sqrt_f16_e32", 0x4200), ("v_sin_f16_e32", 0x3037),
    ("v_cos_f16_e32", 0x3037), ("v_exp_f16_e32", 0x3037), ("v_log_f16_e32", 0x4a37),
])
def test_transcendental_f16_ignores_mode(op, src, tmp_path):
    # RX 6800 (rcp/sqrt/rsq/sin) + gfx1036 exhaustive 65,536-input sweep (SHADER-132, wave32 & wave64):
    # all seven f16 transcendentals give identical bits in RNE and RTZ.
    got = []
    for prefix in ("", RNE, RTZ):
        asm = f"v_mov_b32 v1, {src:#x}\n{prefix}{op} v2, v1\n"
        got.append(vgprs(tmp_path, asm)[2] & 0xFFFF)
    assert got[0] == got[1] == got[2]


def test_transcendental_f16_nearest_after_rtz_write(tmp_path):
    # MODE is set toward zero before the ops: all seven still give the nearest-even words (gfx1036, SHADER-132)
    for op, src, want in (("v_rcp_f16_e32", 0x3037, 0x4798), ("v_rsq_f16_e32", 0x3037, 0x4183),
                          ("v_sqrt_f16_e32", 0x4200, 0x3EEE), ("v_sin_f16_e32", 0x3037, 0x39E4),
                          ("v_cos_f16_e32", 0x3037, 0x396A), ("v_exp_f16_e32", 0x3037, 0x3C62),
                          ("v_log_f16_e32", 0x4A37, 0x4345)):
        asm = f"v_mov_b32 v1, {src:#x}\n{RTZ}{op} v2, v1\n"
        assert vgprs(tmp_path, asm)[2] & 0xFFFF == want, op


@pytest.mark.parametrize("prefix,want", [("", 0x4000), (RTZ, 0x3FFF)])
def test_fma_mix_single_rounding(prefix, want, tmp_path):
    # 1 * 2 - 2^-30 with f32 operands: toward zero the hardware rounds the exact value once to f16 (0x3fff); rounding
    # to f32 first (2.0) then narrowing would give 0x4000 (RX 6800)
    asm = ("v_mov_b32 v1, 0x3f800000\nv_mov_b32 v2, 0x40000000\nv_mov_b32 v4, 0xb0800000\n"
           f"{prefix}v_fma_mixlo_f16 v3, v1, v2, v4\n")
    assert vgprs(tmp_path, asm)[3] & 0xFFFF == want


def test_fma_mix_toward_zero_negative_and_hi(tmp_path):
    # -(1 * 2) + 2^-30 -> -0x3fff in the high half, the low half kept
    asm = ("v_mov_b32 v1, 0x3f800000\nv_mov_b32 v2, 0xc0000000\nv_mov_b32 v4, 0x30800000\nv_mov_b32 v3, 0x1234\n"
           f"{RTZ}v_fma_mixhi_f16 v3, v1, v2, v4\n")
    assert vgprs(tmp_path, asm)[3] == 0xBFFF << 16 | 0x1234


def test_add_f64_overflow_saturates(tmp_path):
    asm = ("v_mov_b32 v0, -1\nv_mov_b32 v1, 0x7fefffff\n"
           f"{RTZ}v_add_f64 v[4:5], v[0:1], v[0:1]\n")
    v = vgprs(tmp_path, asm)
    assert (v[4], v[5]) == (0xFFFFFFFF, 0x7FEFFFFF)


@pytest.mark.parametrize("x,y,prefix,want", [
    # Opposite-sign operands near -DBL_MAX with |x| < |y| (#2891): unordered TwoSum overflows (r - x -> -inf),
    # yielding e = NaN and leaving the RNE low word 0xc12af666 instead of the exact RTZ word 0xc12af665.
    (0x7FDD9BB47DAA1333, 0xFFEFFFFFFFFFFFFF, RTZ, 0xFFE13225C12AF665),
    (0xFFEFFFFFFFFFFFFF, 0x7FDD9BB47DAA1333, RTZ, 0xFFE13225C12AF665),
    (0x7FDD9BB47DAA1333, 0xFFEFFFFFFFFFFFFF, RNE, 0xFFE13225C12AF666),
    # Opposite-sign near +DBL_MAX with |x| < |y| and RNE lo == 0: FastTwoSum must borrow across lo -> hi - 1.
    (0xFFCFFFFFFFFFFFFE, 0x7FEFFFFFFFFFFFFF, RTZ, 0x7FE7FFFFFFFFFFFF),
    (0x7FEFFFFFFFFFFFFF, 0xFFCFFFFFFFFFFFFE, RTZ, 0x7FE7FFFFFFFFFFFF),
    (0xFFCFFFFFFFFFFFFE, 0x7FEFFFFFFFFFFFFF, RNE, 0x7FE8000000000000),
    # Negative finite overflow saturates to -DBL_MAX; half-ulp above DBL_MAX overflows RNE to inf and steps back.
    (0xFFEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF, RTZ, 0xFFEFFFFFFFFFFFFF),
    (0x7FEFFFFFFFFFFFFF, 0x7CA0000000000000, RTZ, 0x7FEFFFFFFFFFFFFF),
    # Real infinity and exact cancellation stay inf / +0.0 (no spurious -1 ulp step).
    (0x7FF0000000000000, 0xC000000000000000, RTZ, 0x7FF0000000000000),
    (0xC000000000000000, 0xFFF0000000000000, RTZ, 0xFFF0000000000000),
    (0x7FEFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF, RTZ, 0x0000000000000000),
    (0x8000000000000000, 0x8000000000000000, RTZ, 0x8000000000000000),
])
def test_add_f64_extreme_magnitude_and_specials(x, y, prefix, want, tmp_path):
    asm = (f"v_mov_b32 v0, {x & 0xFFFFFFFF:#x}\nv_mov_b32 v1, {x >> 32:#x}\n"
           f"v_mov_b32 v2, {y & 0xFFFFFFFF:#x}\nv_mov_b32 v3, {y >> 32:#x}\n"
           f"{prefix}v_add_f64 v[4:5], v[0:1], v[2:3]\n")
    v = vgprs(tmp_path, asm)
    assert (v[4] | (v[5] << 32)) == want


def test_add_f64_nan_and_no_contraction(tmp_path):
    import math
    import re
    import struct

    for a, b in ((0x7FF8000000000001, 0x3FF0000000000000), (0x3FF0000000000000, 0x7FF8000000000001)):
        asm = (f"v_mov_b32 v0, {a & 0xFFFFFFFF:#x}\nv_mov_b32 v1, {a >> 32:#x}\n"
               f"v_mov_b32 v2, {b & 0xFFFFFFFF:#x}\nv_mov_b32 v3, {b >> 32:#x}\n"
               f"{RTZ}v_add_f64 v[4:5], v[0:1], v[2:3]\n")
        spv = translate_listing(tmp_path, asm)
        v = si.run(spv).vgpr
        got = v[4] | (v[5] << 32)
        assert math.isnan(struct.unpack("<d", struct.pack("<Q", got))[0])
        fsubs = re.findall(r"^\s*(%\w+) = OpFSub %double ", spv, re.M)
        assert len(fsubs) == 2
        for res_id in fsubs:
            assert re.search(rf"(?m)^\s*OpDecorate {re.escape(res_id)} NoContraction$", spv)


@pytest.mark.parametrize("x,y", [
    (0x0000000000000001, 0x0010000000000000),   # subnormal + DBL_MIN
    (0x8000000000000001, 0x0010000000000000),   # -DBL_TRUE_MIN + DBL_MIN
    (0x000FFFFFFFFFFFFF, 0x8000000000000001),   # max subnormal - DBL_TRUE_MIN
    (0x3FF0000000000000, 0x3C90000000000000),   # 1.0 + 0.5 ulp tie (even LSB stays 1.0)
    (0x3FF0000000000001, 0x3C90000000000000),   # (1 + 1 ulp) + 0.5 ulp tie (RNE rounds away; RTZ truncates)
    (0xBFF0000000000001, 0xBC90000000000000),   # negative odd tie
    (0x7FDFFFFFFFFFFFFF, 0xFFEFFFFFFFFFFFFF),   # ~0.5*DBL_MAX - DBL_MAX (|x| < |y| opposite sign)
    (0x7FE0000000000001, 0xFFEFFFFFFFFFFFFF),   # 2^1023 + 1 ulp - DBL_MAX
])
def test_add_f64_rtz_against_fraction_oracle(x, y, tmp_path):
    from fractions import Fraction
    import struct

    def u2d(u):
        return struct.unpack("<d", struct.pack("<Q", u))[0]

    def d2u(d):
        return struct.unpack("<Q", struct.pack("<d", d))[0]

    exact = Fraction(u2d(x)) + Fraction(u2d(y))
    if exact == 0:
        want = 0
    else:
        mag = abs(exact)
        rmag = float(mag)
        u = d2u(rmag) - (1 if Fraction(rmag) > mag else 0)
        want = u | (1 << 63) if exact < 0 else u
    asm = (f"v_mov_b32 v0, {x & 0xFFFFFFFF:#x}\nv_mov_b32 v1, {x >> 32:#x}\n"
           f"v_mov_b32 v2, {y & 0xFFFFFFFF:#x}\nv_mov_b32 v3, {y >> 32:#x}\n"
           f"{RTZ}v_add_f64 v[4:5], v[0:1], v[2:3]\n")
    v = vgprs(tmp_path, asm)
    assert (v[4] | (v[5] << 32)) == want


@pytest.mark.parametrize("asm", [
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 0, 2), 3\n",       # f32 rounding: not modelled
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 4, 4), 15\n",      # denormal controls: not modelled
    "s_setreg_imm32_b32 hwreg(HW_REG_MODE, 2, 2), 1\n",       # round toward +inf
    "s_setreg_b32 hwreg(HW_REG_MODE, 2, 2), s0\n",            # value from a register
])
def test_unmodelled_mode_writes_are_refused(asm, tmp_path):
    with pytest.raises(rdna2spirv.Unsupported):
        translate_listing(tmp_path, asm)


@pytest.mark.parametrize("x,y,z,want", [
    (0x99800000, 0x19800000, 0x3F800000, 0x3BFF),    # 1 - 2^-152: the product error underflows in f32
    (0x9F000000, 0x1F000000, 0x3F800000, 0x3BFF),    # 1 - 2^-130: the product is an f32 subnormal (flushed on some drivers)
    (0x1F000000, 0x1F000000, 0x3F800000, 0x3C00),    # 1 + 2^-130 toward zero is 1.0
    (0x99800000, 0x19800000, 0xBF800000, 0xBC00),    # -1 - 2^-152 toward zero is -1.0
    (0x19800000, 0x19800000, 0xBF800000, 0xBBFF),    # -1 + 2^-152 toward zero is -(1 - 2^-11)
])
def test_fma_mix_tiny_product(x, y, z, want, tmp_path):
    asm = (f"v_mov_b32 v1, {x:#x}\nv_mov_b32 v2, {y:#x}\nv_mov_b32 v4, {z:#x}\n"
           f"{RTZ}v_fma_mixlo_f16 v3, v1, v2, v4\n")
    assert vgprs(tmp_path, asm)[3] & 0xFFFF == want


@pytest.mark.parametrize("x,y,z,want", [
    (0x5F800000, 0x5F800000, 0x00000000, 0x7BFF),    # 2^128 is finite exactly: f32 Fma gives inf, toward zero 65504
    (0xDF800000, 0x5F800000, 0x00000000, 0xFBFF),
    (0x7F800000, 0x3F800000, 0x00000000, 0x7C00),    # a real inf stays inf
])
def test_fma_mix_f32_overflow_toward_zero(x, y, z, want, tmp_path):
    asm = (f"v_mov_b32 v1, {x:#x}\nv_mov_b32 v2, {y:#x}\nv_mov_b32 v4, {z:#x}\n"
           f"{RTZ}v_fma_mixlo_f16 v3, v1, v2, v4\n")
    assert vgprs(tmp_path, asm)[3] & 0xFFFF == want
