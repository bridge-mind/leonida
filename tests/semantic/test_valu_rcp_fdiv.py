"""Semantic value tests for OpFDiv in spirv_mini_interp and RDNA2 reciprocal VALU ops:
  - OpFDiv (scalar %float and vector %v2float) directly in tests/semantic/spirv_mini_interp.py
  - v_rcp_f32:       D.f32 = 1.0 / S0.f32
  - v_rcp_iflag_f32: D.f32 = 1.0 / S0.f32  (no FP flags)
  - v_rcp_f16:       D.f16 = 1.0 / S0.f16  (preserves the untouched 16-bit half of D)
  - Newton-Raphson division iteration pipeline (v_rcp_f32 + v_fma_f32 + v_mul_f32)


Per the SPIR-V specification and tests/semantic/README.md, OpFDiv by zero (+-0.0) is undefined and raises
spirv_mini_interp.UnsupportedConstruct("OpFDiv by zero (undefined in SPIR-V)"). Accordingly, finite non-zero,
subnormal, +-inf, and NaN operands are value-checked against independent Fraction-based binary32/binary16
oracles, while +-0.0 divisors/inputs of the f32 ops are verified to raise UnsupportedConstruct (v_rcp_f16 selects the
ISA's +-inf for +-0 itself).

NaN payloads are not pinned because values passing through Python floats lose exact NaN payload bits; NaN
results are asserted as binary32/binary16 NaNs. Standard library only; hand-written RDNA2 listings only.
"""
from __future__ import annotations

from fractions import Fraction
import gc
import itertools
import math
import os
import re
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
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]
OLD = 0xBEEF1234

# Representative binary16 encodings.
H_POS_ZERO = 0x0000
H_NEG_ZERO = 0x8000
H_SUB_MIN = 0x0001          # 2^-24
H_NEG_SUB_MIN = 0x8001      # -2^-24
H_SUB_2M16 = 0x0100         # 2^-16 (1/x = 65536 >= 65520 -> +inf)
H_SUB_2M15 = 0x0200         # 2^-15 (1/x = 32768 = 0x7800)
H_SUB_MAX = 0x03FF          # 1023 * 2^-24
H_NORM_MIN = 0x0400         # 2^-14 (1/x = 16384 = 0x7400)
H_0P25 = 0x3400             # 0.25
H_NEG_0P25 = 0xB400         # -0.25
H_0P5 = 0x3800              # 0.5
H_NEG_0P5 = 0xB800          # -0.5
H_0P75 = 0x3A00             # 0.75
H_1P0 = 0x3C00              # 1.0
H_NEG_1P0 = 0xBC00          # -1.0
H_NEG_1P25 = 0xBD00         # -1.25
H_1P5 = 0x3E00              # 1.5
H_2P0 = 0x4000              # 2.0
H_NEG_2P0 = 0xC000          # -2.0
H_3P0 = 0x4200              # 3.0
H_NEG_3P0 = 0xC200          # -3.0
H_4P0 = 0x4400              # 4.0
H_NEG_4P0 = 0xC400          # -4.0
H_8P0 = 0x4800              # 8.0
H_16P0 = 0x4C00             # 16.0
H_1024 = 0x6400             # 1024.0
H_MAX = 0x7BFF              # 65504.0
H_NEG_MAX = 0xFBFF          # -65504.0
H_POS_INF = 0x7C00          # +inf
H_NEG_INF = 0xFC00          # -inf
H_QNAN = 0x7E01             # +qNaN
H_NEG_QNAN = 0xFE01         # -qNaN
H_SNAN = 0x7C01             # +sNaN


def f32_bits(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000


def is_f32_nan_bits(bits: int) -> bool:
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def is_f16_nan(half: int) -> bool:
    return (half & 0x7FFF) > 0x7C00


def merge_half(old: int, half: int, dst_hi: int) -> int:
    h = half & 0xFFFF
    return ((h << 16) | (old & 0xFFFF)) if dst_hi else ((old & 0xFFFF0000) | h)


# -----------------------------------------------------------------------------
# Independent Exact-Rational Binary32 / Binary16 Reference Oracles
# -----------------------------------------------------------------------------

def _rne_int(q: Fraction) -> int:
    n = q.numerator // q.denominator
    rem = q - n
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and (n & 1)):
        n += 1
    return n


def f32_to_fraction(bits: int) -> Fraction:
    bits &= MASK32
    sign = -1 if (bits >> 31) else 1
    exp = (bits >> 23) & 0xFF
    man = bits & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError(f"non-finite f32 bits {bits:#010x}")
    if exp == 0:
        return sign * Fraction(man, 1 << 23) * Fraction(2) ** -126
    return sign * Fraction((1 << 23) | man, 1 << 23) * Fraction(2) ** (exp - 127)


def round_fraction_to_f32(val: Fraction, neg_zero: bool = False) -> int:
    if val == 0:
        return 0x80000000 if neg_zero else 0x00000000
    sign = 0x80000000 if val < 0 else 0
    a = abs(val)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -126:
        n = _rne_int(a / Fraction(2) ** -149)
        if n == 0:
            return sign
        if n < (1 << 23):
            return sign | n
        return sign | (1 << 23)
    n = _rne_int(a / Fraction(2) ** (e - 23))
    if n == (1 << 24):
        n >>= 1
        e += 1
    if e > 127:
        return sign | 0x7F800000
    return sign | ((e + 127) << 23) | (n & 0x7FFFFF)


def f16_to_fraction(half: int) -> Fraction:
    half &= 0xFFFF
    sign = -1 if (half >> 15) else 1
    exp = (half >> 10) & 0x1F
    man = half & 0x3FF
    if exp == 0x1F:
        raise ValueError(f"non-finite f16 bits {half:#06x}")
    if exp == 0:
        return sign * Fraction(man, 1 << 10) * Fraction(2) ** -14
    return sign * Fraction((1 << 10) | man, 1 << 10) * Fraction(2) ** (exp - 15)


def round_fraction_to_f16(val: Fraction, neg_zero: bool = False) -> int:
    if val == 0:
        return 0x8000 if neg_zero else 0x0000
    sign = 0x8000 if val < 0 else 0
    a = abs(val)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    if e < -14:
        n = _rne_int(a / Fraction(2) ** -24)
        return sign | n
    n = _rne_int(a / Fraction(2) ** (e - 10))
    if n == (1 << 11):
        n >>= 1
        e += 1
    if e > 15:
        return sign | 0x7C00
    return sign | ((e + 15) << 10) | (n & 0x3FF)


def ref_rcp_f32(x_bits: int, *, scale: Fraction = Fraction(1), clamp: bool = False) -> int:
    """Reference oracle for non-zero f32 reciprocal (with optional output scale/clamp)."""
    bits = x_bits & MASK32
    sign = bits & 0x80000000
    absu = bits & 0x7FFFFFFF
    if absu == 0:
        raise ValueError("division by zero is undefined in SPIR-V")
    if absu > 0x7F800000:
        if clamp:
            raise ValueError("clamp of NaN is undefined in GLSL.std.450")
        return 0x7FC00000
    if absu == 0x7F800000:
        # 1 / (+-inf) = +-0.0; omod preserves sign of zero, clamp orders -0 below +0 -> +0.
        return 0 if clamp else sign
    # Match OpFDiv(1, x) rounded to binary32 before clamp_mods (scale, then FClamp [0, 1]).
    r_bits = round_fraction_to_f32(Fraction(1) / f32_to_fraction(bits))
    if (r_bits & 0x7FFFFFFF) == 0x7F800000:
        r_sign = r_bits & 0x80000000
        if clamp:
            return 0 if r_sign else 0x3F800000
        return r_bits
    val = f32_to_fraction(r_bits) * scale
    if clamp:
        if val <= 0:
            return 0x00000000
        if val >= 1:
            return 0x3F800000
    return round_fraction_to_f32(val)


def ref_rcp_f16(h_bits: int, *, scale: Fraction = Fraction(1), clamp: bool = False) -> int:
    """Reference oracle for non-zero f16 reciprocal (f32 OpFDiv + clamp_mods + f16_bits)."""
    h = h_bits & 0xFFFF
    sign = (h & 0x8000) << 16
    absh = h & 0x7FFF
    if absh == 0:
        raise ValueError("division by zero is undefined in SPIR-V")
    if absh > 0x7C00:
        if clamp:
            raise ValueError("clamp of NaN is undefined in GLSL.std.450")
        return (h & 0x8000) | 0x7E00
    if absh == 0x7C00:
        return 0 if clamp else (h & 0x8000)
    # Widening f16 -> f32 is exact, then f32 OpFDiv + omod/clamp, then f16 nearest-even narrowing.
    r_bits = round_fraction_to_f32(Fraction(1) / f16_to_fraction(h))
    r_sign = r_bits & 0x80000000
    if (r_bits & 0x7FFFFFFF) == 0x7F800000:
        if clamp:
            return 0 if r_sign else 0x3C00
        return (r_sign >> 16) | 0x7C00
    val = f32_to_fraction(r_bits) * scale
    if clamp:
        if val <= 0:
            return 0x0000
        if val >= 1:
            return 0x3C00
    r_f32 = round_fraction_to_f32(val, neg_zero=bool(sign))
    return round_fraction_to_f16(f32_to_fraction(r_f32), neg_zero=bool(r_f32 & 0x80000000))


def f32_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_ulp_diff(a_bits: int, b_bits: int) -> int:
    a_bits &= MASK32
    b_bits &= MASK32
    if a_bits == b_bits:
        return 0
    if (a_bits & 0x7FFFFFFF) == 0 and (b_bits & 0x7FFFFFFF) == 0:
        return 0
    if (a_bits >> 31) != (b_bits >> 31):
        return (a_bits & 0x7FFFFFFF) + (b_bits & 0x7FFFFFFF)
    return abs((a_bits & 0x7FFFFFFF) - (b_bits & 0x7FFFFFFF))


def ref_exact_fdiv_f32(a_bits: int, b_bits: int) -> int:
    """Exact-rational IEEE 754 float32 division reference oracle."""
    a_bits &= MASK32
    b_bits &= MASK32
    sign = (a_bits ^ b_bits) & 0x80000000
    absu_b = b_bits & 0x7FFFFFFF
    absu_a = a_bits & 0x7FFFFFFF
    if absu_b == 0:
        raise ValueError("division by zero is undefined in SPIR-V")
    if absu_a > 0x7F800000 or absu_b > 0x7F800000:
        return 0x7FC00000
    if absu_a == 0x7F800000 and absu_b == 0x7F800000:
        return 0x7FC00000
    if absu_b == 0x7F800000:
        return sign
    if absu_a == 0x7F800000:
        return sign | 0x7F800000
    if absu_a == 0:
        return sign
    fa = f32_to_fraction(a_bits)
    fb = f32_to_fraction(b_bits)
    return round_fraction_to_f32(fa / fb, neg_zero=bool(sign))


def ref_newton_raphson_steps_f32(a_bits: int, b_bits: int) -> tuple[int, int, int, int]:
    """Simulate RDNA2 4-step Newton-Raphson division in software.

    Step 1: Initial reciprocal estimate y0 = rcp(b)
    Step 2: Error term e = 1.0 - b * y0 via FMA (fma(-b, y0, 1.0))
    Step 3: Refined reciprocal y1 = y0 + y0 * e via FMA (fma(y0, e, y0))
    Step 4: Quotient q = a * y1 via FMul
    """
    a_bits &= MASK32
    b_bits &= MASK32
    if (b_bits & 0x7FFFFFFF) == 0:
        raise ValueError("division by zero is undefined in SPIR-V")
    y0_bits = ref_rcp_f32(b_bits)
    y0 = f32_to_float(y0_bits)
    b = f32_to_float(b_bits)
    a = f32_to_float(a_bits)

    e = si.fma_f32(-b, y0, 1.0)
    e_bits = f32_bits(e)

    y1 = si.fma_f32(y0, e, y0)
    y1_bits = f32_bits(y1)

    q = struct.unpack("<f", struct.pack("<f", a * y1))[0]
    q_bits = f32_bits(q)
    return y0_bits, e_bits, y1_bits, q_bits


def ref_newton_raphson_fdiv_f32(a_bits: int, b_bits: int) -> int:
    return ref_newton_raphson_steps_f32(a_bits, b_bits)[3]


NR_FDIV_HAND_WORKED = (
    ("one_half", 0x3F800000, 0x40000000, (0x3F000000, 0x00000000, 0x3F000000, 0x3F000000)),
    ("seven_thirds", 0x40E00000, 0x40400000, (0x3EAAAAAB, 0xB3000000, 0x3EAAAAAB, 0x40155556)),
    ("ten_fourths", 0x41200000, 0x40800000, (0x3E800000, 0x00000000, 0x3E800000, 0x40200000)),
    ("neg_fifteen_fifths", 0xC1700000, 0x40A00000, (0x3E4CCCCD, 0xB2800000, 0x3E4CCCCD, 0xC0400000)),
    ("pos_neg_div", 0x42C90000, 0xC1A20000, (0xBD4A4588, 0xB2000000, 0xBD4A4588, 0xC09ED098)),
    ("pos_zero", 0x00000000, 0x40800000, (0x3E800000, 0x00000000, 0x3E800000, 0x00000000)),
    ("neg_zero", 0x80000000, 0x40800000, (0x3E800000, 0x00000000, 0x3E800000, 0x80000000)),
)



RCP_F32_HAND_WORKED = (
    ("pos_one", 0x3F800000, 0x3F800000),
    ("neg_one", 0xBF800000, 0xBF800000),
    ("pos_two", 0x40000000, 0x3F000000),
    ("neg_two", 0xC0000000, 0xBF000000),
    ("pos_four", 0x40800000, 0x3E800000),
    ("neg_quarter", 0xBE800000, 0xC0800000),
    ("pos_2p20", 0x49800000, 0x35800000),
    ("neg_2pm20", 0xB5800000, 0xC9800000),
    ("pos_1p5", 0x3FC00000, 0x3F2AAAAB),
    ("neg_3p0", 0xC0400000, 0xBEAAAAAB),
    ("pos_0p3", 0x3E99999A, 0x40555555),
    ("subnorm_2m128_to_inf", 0x00200000, 0x7F800000),
    ("neg_subnorm_min_to_ninf", 0x80000001, 0xFF800000),
    ("2p127_to_subnorm", 0x7F000000, 0x00400000),
    ("pos_inf", 0x7F800000, 0x00000000),
    ("neg_inf", 0xFF800000, 0x80000000),
)

RCP_F16_HAND_WORKED = (
    ("pos_one", H_1P0, H_1P0),
    ("neg_one", H_NEG_1P0, H_NEG_1P0),
    ("pos_two", H_2P0, H_0P5),
    ("neg_two", H_NEG_2P0, H_NEG_0P5),
    ("pos_four", H_4P0, H_0P25),
    ("neg_quarter", H_NEG_0P25, H_NEG_4P0),
    ("pos_1p5", H_1P5, 0x3955),
    ("pos_3p0", H_3P0, 0x3555),
    ("neg_3p0", H_NEG_3P0, 0xB555),
    ("sub_min_overflow", H_SUB_MIN, H_POS_INF),
    ("neg_sub_min_overflow", H_NEG_SUB_MIN, H_NEG_INF),
    ("sub_2m16_overflow", H_SUB_2M16, H_POS_INF),
    ("sub_2m15", H_SUB_2M15, 0x7800),
    ("sub_max", H_SUB_MAX, 0x7401),
    ("norm_min", H_NORM_MIN, 0x7400),
    ("f16_max", H_MAX, 0x0100),
    ("neg_f16_max", H_NEG_MAX, 0x8100),
    ("pos_inf", H_POS_INF, H_POS_ZERO),
    ("neg_inf", H_NEG_INF, H_NEG_ZERO),
)


# -----------------------------------------------------------------------------
# Translation & Execution Helpers
# -----------------------------------------------------------------------------

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None
_CS_FSSM = None


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_valu_rcp_fdiv.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_valu_rcp_fdiv.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_valu_rcp_fdiv.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_valu_rcp_fdiv.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_valu_rcp_fdiv\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps_v0(lines: list[str]) -> int:
    return si.run_fragment(translate_ps(lines))["%out_mrt0"][0]


def run_op(line: str, v1: int, old: int = OLD, s4: int = 0) -> int:
    return run_ps_v0([
        f"s_mov_b32 s4, {s4:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v0, {old:#x}",
        line,
    ])


def run_cs(lines: list[str], group_size: int, subgroup_size: int, memory: dict[int, int]) -> dict[int, int]:
    if _TMP_DIR is None:
        setUpModule()
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))
    asm = "; test_cs_valu_rcp_fdiv\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    return si.run_workgroup(
        spv, group_size, subgroup_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP + [0] * 28,
        memory=memory,
    )


# -----------------------------------------------------------------------------
# 1. Reference Oracle Self-Tests
# -----------------------------------------------------------------------------

class OracleSelfTests(unittest.TestCase):
    def test_hand_worked_rcp_f32_and_f16_match_fraction_oracles(self):
        for name, x_bits, want in RCP_F32_HAND_WORKED:
            with self.subTest(kind="f32", name=name):
                self.assertEqual(ref_rcp_f32(x_bits), want)
        for name, h_bits, want in RCP_F16_HAND_WORKED:
            with self.subTest(kind="f16", name=name):
                self.assertEqual(ref_rcp_f16(h_bits), want)


class NewtonRaphsonOracleTests(unittest.TestCase):
    def test_hand_worked_newton_raphson_steps(self):
        for name, a_bits, b_bits, want_steps in NR_FDIV_HAND_WORKED:
            with self.subTest(case=name):
                got_steps = ref_newton_raphson_steps_f32(a_bits, b_bits)
                self.assertEqual(got_steps, want_steps)

    def test_exact_rational_division_oracle(self):
        cases = (
            (f32_bits(1.0), f32_bits(2.0), f32_bits(0.5)),
            (f32_bits(8.0), f32_bits(4.0), f32_bits(2.0)),
            (f32_bits(-9.0), f32_bits(3.0), f32_bits(-3.0)),
            (f32_bits(100.0), f32_bits(-25.0), f32_bits(-4.0)),
            (0x00000000, f32_bits(5.0), 0x00000000),
            (0x80000000, f32_bits(5.0), 0x80000000),
            (f32_bits(1.0), 0x7F800000, 0x00000000),
            (f32_bits(-1.0), 0x7F800000, 0x80000000),
        )
        for a_bits, b_bits, want in cases:
            with self.subTest(a=hex(a_bits), b=hex(b_bits)):
                self.assertEqual(ref_exact_fdiv_f32(a_bits, b_bits), want)

    def test_zero_divisor_raises_in_oracles(self):
        for z in (0x00000000, 0x80000000):
            with self.subTest(zero=hex(z)):
                with self.assertRaises(ValueError):
                    ref_newton_raphson_steps_f32(f32_bits(1.0), z)
                with self.assertRaises(ValueError):
                    ref_exact_fdiv_f32(f32_bits(1.0), z)

    def test_newton_raphson_oracle_accuracy_within_one_ulp(self):
        test_pairs = (
            (1.0, 3.0),
            (7.0, 3.0),
            (22.0, 7.0),
            (355.0, 113.0),
            (12345.67, 0.00123),
            (-9876.54, 321.0),
            (1e12, 3.0),
            (1e-12, 7.0),
            (0.15625, 1.25),
            (1024.0, 10.0),
            (-4096.0, 13.0),
        )
        for a_f, b_f in test_pairs:
            with self.subTest(a=a_f, b=b_f):
                ab = f32_bits(a_f)
                bb = f32_bits(b_f)
                nr = ref_newton_raphson_fdiv_f32(ab, bb)
                exact = ref_exact_fdiv_f32(ab, bb)
                ulp = f32_ulp_diff(nr, exact)
                self.assertLessEqual(ulp, 1, f"ULP diff {ulp} exceeds 1 for {a_f}/{b_f}: nr={nr:#010x} exact={exact:#010x}")


# -----------------------------------------------------------------------------
# 2. Direct OpFDiv Unit Tests in spirv_mini_interp.py
# -----------------------------------------------------------------------------

FDIV_HEADER = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v2float = OpTypeVector %float 2
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%_ptr_PC = OpTypePointer PushConstant %uint
%pc = OpVariable %_ptr_PC PushConstant
%c0 = OpConstant %uint 0
%c1 = OpConstant %uint 1
%c2 = OpConstant %uint 2
%c3 = OpConstant %uint 3
%f0 = OpConstant %float 0
%f0p5 = OpConstant %float 0.5
%f1 = OpConstant %float 1
%f2 = OpConstant %float 2
%f3 = OpConstant %float 3
%f4 = OpConstant %float 4
%f255 = OpConstant %float 255
%u_neg0 = OpConstant %uint 2147483648
%u_inf = OpConstant %uint 2139095040
%u_ninf = OpConstant %uint 4286578688
%u_qnan = OpConstant %uint 2143289344
%u_snan = OpConstant %uint 2139095041
%u_fltmax = OpConstant %uint 2139095039
%u_nfltmax = OpConstant %uint 4286578687
%main = OpFunction %void None %fn
%entry = OpLabel
%fm0 = OpBitcast %float %u_neg0
%inf = OpBitcast %float %u_inf
%ninf = OpBitcast %float %u_ninf
%qnan = OpBitcast %float %u_qnan
%snan = OpBitcast %float %u_snan
%fltmax = OpBitcast %float %u_fltmax
%nfltmax = OpBitcast %float %u_nfltmax
%fm2 = OpFNegate %float %f2
%fm3 = OpFNegate %float %f3
%fm1 = OpFNegate %float %f1
"""
FDIV_FOOTER = "OpReturn\nOpFunctionEnd\n"


def run_fdiv_scalar(num: str, den: str) -> int:
    text = (
        FDIV_HEADER
        + f"%r = OpFDiv %float {num} {den}\n"
        + "%res = OpBitcast %uint %r\n"
        + "OpStore %out %res\n"
        + FDIV_FOOTER
    )
    return si.run_fragment(text)["%out"][0]


def run_fdiv_vec2(n0: str, n1: str, d0: str, d1: str) -> tuple[int, int]:
    hdr = FDIV_HEADER.replace(
        'OpEntryPoint Fragment %main "main" %out',
        'OpEntryPoint Fragment %main "main" %out %out1',
    ).replace(
        "%out = OpVariable %_ptr_Output_uint Output",
        "%out = OpVariable %_ptr_Output_uint Output\n%out1 = OpVariable %_ptr_Output_uint Output",
    )
    body = (
        f"%vn = OpCompositeConstruct %v2float {n0} {n1}\n"
        + f"%vd = OpCompositeConstruct %v2float {d0} {d1}\n"
        + "%vr = OpFDiv %v2float %vn %vd\n"
        + "%r0 = OpCompositeExtract %float %vr 0\n"
        + "%r1 = OpCompositeExtract %float %vr 1\n"
        + "%b0 = OpBitcast %uint %r0\n"
        + "%b1 = OpBitcast %uint %r1\n"
        + "OpStore %out %b0\n"
        + "OpStore %out1 %b1\n"
    )
    res = si.run_fragment(hdr + body + FDIV_FOOTER)
    return res["%out"][0], res["%out1"][0]


class OpFDivInterpreterTests(unittest.TestCase):
    def test_scalar_finite_signed_zero_inf_and_overflow(self):
        cases = (
            ("%f1", "%f4", f32_bits(0.25)),
            ("%fm3", "%f2", f32_bits(-1.5)),
            ("%f3", "%f255", 0x3C40C0C1),
            # Signed zero dividends keep exact IEEE sign rules.
            ("%f0", "%f2", 0x00000000),
            ("%f0", "%fm2", 0x80000000),
            ("%fm0", "%f2", 0x80000000),
            ("%fm0", "%fm2", 0x00000000),
            # Division by +-inf -> +-0.0.
            ("%f1", "%inf", 0x00000000),
            ("%fm1", "%inf", 0x80000000),
            ("%f1", "%ninf", 0x80000000),
            ("%fm1", "%ninf", 0x00000000),
            ("%f0", "%ninf", 0x80000000),
            ("%fm0", "%inf", 0x80000000),
            ("%fm0", "%ninf", 0x00000000),
            # +-inf divided by finite non-zero -> +-inf.
            ("%inf", "%f2", 0x7F800000),
            ("%inf", "%fm2", 0xFF800000),
            ("%ninf", "%f2", 0xFF800000),
            ("%ninf", "%fm2", 0x7F800000),
            # Binary32 overflow -> +-inf.
            ("%fltmax", "%f0p5", 0x7F800000),
            ("%nfltmax", "%f0p5", 0xFF800000),
        )
        for num, den, want in cases:
            with self.subTest(num=num, den=den):
                self.assertEqual(run_fdiv_scalar(num, den), want)

    def test_scalar_inf_over_inf_and_nan_propagation(self):
        nan_cases = (
            ("%inf", "%inf"),
            ("%inf", "%ninf"),
            ("%ninf", "%inf"),
            ("%ninf", "%ninf"),
            ("%qnan", "%f2"),
            ("%f2", "%qnan"),
            ("%qnan", "%inf"),
            ("%inf", "%qnan"),
            ("%f0", "%qnan"),
            ("%snan", "%f2"),
            ("%f2", "%snan"),
        )
        for num, den in nan_cases:
            with self.subTest(num=num, den=den):
                got = run_fdiv_scalar(num, den)
                self.assertTrue(is_f32_nan_bits(got), f"expected NaN for {num}/{den}, got {got:#010x}")

    def test_vector_v2float_division(self):
        self.assertEqual(run_fdiv_vec2("%f1", "%fm3", "%f4", "%fm2"), (f32_bits(0.25), f32_bits(1.5)))
        self.assertEqual(run_fdiv_vec2("%f1", "%fm0", "%ninf", "%ninf"), (0x80000000, 0x00000000))
        g0, g1 = run_fdiv_vec2("%f1", "%inf", "%ninf", "%ninf")
        self.assertEqual(g0, 0x80000000)
        self.assertTrue(is_f32_nan_bits(g1), f"expected NaN in lane 1, got {g1:#010x}")

    def test_zero_divisor_and_non_float_are_rejected(self):
        for num, den in (
            ("%f1", "%f0"),
            ("%f1", "%fm0"),
            ("%fm1", "%f0"),
            ("%fm1", "%fm0"),
            ("%f0", "%f0"),
            ("%fm0", "%fm0"),
            ("%inf", "%f0"),
            ("%qnan", "%f0"),
            ("%qnan", "%fm0"),
        ):
            with self.subTest(num=num, den=den):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_fdiv_scalar(num, den)

        for n0, n1, d0, d1 in (
            ("%f1", "%f2", "%f0", "%f2"),
            ("%f1", "%f2", "%f2", "%fm0"),
        ):
            with self.subTest(vec=(n0, n1, d0, d1)):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_fdiv_vec2(n0, n1, d0, d1)

        with self.assertRaises(si.UnsupportedConstruct):
            si.run_fragment(
                FDIV_HEADER + "%res = OpFDiv %uint %c2 %c1\nOpStore %out %res\n" + FDIV_FOOTER
            )


# -----------------------------------------------------------------------------
# 3. End-to-End v_rcp_f32 & v_rcp_iflag_f32 Value Tests
# -----------------------------------------------------------------------------

class RcpF32AndIflagExecutionTests(unittest.TestCase):
    OPS = ("v_rcp_f32", "v_rcp_iflag_f32")

    def test_hand_worked_and_edge_values_across_encodings(self):
        for op in self.OPS:
            for suffix in ("_e32", "_e64", ""):
                for name, x_bits, want in RCP_F32_HAND_WORKED:
                    with self.subTest(op=op + suffix, case=name):
                        got = run_op(f"{op}{suffix} v0, v1", x_bits)
                        self.assertEqual(got, want)

    def test_nan_inputs_preserve_nan(self):
        for op in self.OPS:
            for suffix in ("_e32", "_e64"):
                for x_bits in (0x7FC00000, 0xFFC00001, 0x7F800001, 0xFF800001):
                    with self.subTest(op=op + suffix, x=hex(x_bits)):
                        got = run_op(f"{op}{suffix} v0, v1", x_bits)
                        self.assertTrue(is_f32_nan_bits(got), f"expected NaN, got {got:#010x}")

    def test_zero_inputs_raise_unsupported_construct(self):
        for op in self.OPS:
            for suffix in ("_e32", "_e64"):
                for x_bits in (0x00000000, 0x80000000):
                    with self.subTest(op=op + suffix, x=hex(x_bits)):
                        with self.assertRaises(si.UnsupportedConstruct):
                            run_op(f"{op}{suffix} v0, v1", x_bits)
            with self.subTest(op=op, mod="-|0|"):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_op(f"{op}_e64 v0, -|0|", 0)

    def test_vop3_output_and_source_modifiers(self):
        cases = (
            # (insn_suffix_and_src, v1_bits, scale, clamp)
            ("v1 clamp", f32_bits(0.5), Fraction(1), True),
            ("v1 clamp", f32_bits(-0.5), Fraction(1), True),
            ("v1 clamp", f32_bits(4.0), Fraction(1), True),
            ("v1 mul:2", f32_bits(4.0), Fraction(2), False),
            ("v1 mul:2", f32_bits(-0.5), Fraction(2), False),
            ("v1 mul:4", f32_bits(4.0), Fraction(4), False),
            ("v1 mul:4", f32_bits(-2.0), Fraction(4), False),
            ("v1 div:2", f32_bits(2.0), Fraction(1, 2), False),
            ("v1 div:2", f32_bits(-0.5), Fraction(1, 2), False),
            # omod applies before clamp: 1/0.5 * 0.5 = 1.0; 1/4.0 * 2 = 0.5; 1/0.5 * 2 -> clamped 1.0
            ("v1 mul:2 clamp", f32_bits(0.5), Fraction(2), True),
            ("v1 mul:2 clamp", f32_bits(4.0), Fraction(2), True),
            ("v1 mul:2 clamp", f32_bits(-0.5), Fraction(2), True),
            ("v1 clamp div:2", f32_bits(0.5), Fraction(1, 2), True),
            ("v1 clamp div:2", f32_bits(-0.5), Fraction(1, 2), True),
        )
        for op in self.OPS:
            for args, x_bits, scale, clamp in cases:
                with self.subTest(op=op, args=args, x=hex(x_bits)):
                    got = run_op(f"{op}_e64 v0, {args}", x_bits)
                    want = ref_rcp_f32(x_bits, scale=scale, clamp=clamp)
                    self.assertEqual(got, want)

        # Source modifiers (-v1, |v1|, -|v1|) alone and combined with output modifiers.
        src_cases = (
            ("-v1", f32_bits(2.0), f32_bits(-0.5)),
            ("|v1|", f32_bits(-4.0), f32_bits(0.25)),
            ("-|v1|", f32_bits(-2.0), f32_bits(-0.5)),
            ("-|v1|", f32_bits(2.0), f32_bits(-0.5)),
            ("-v1 clamp", f32_bits(-0.5), f32_bits(1.0)),
            ("|v1| mul:2", f32_bits(-0.5), f32_bits(4.0)),
            ("-|v1| mul:2 clamp", f32_bits(0.5), f32_bits(0.0)),
            ("-|v1| div:2", f32_bits(-0.5), f32_bits(-1.0)),
        )
        for op in self.OPS:
            for args, x_bits, want in src_cases:
                with self.subTest(op=op, args=args, x=hex(x_bits)):
                    self.assertEqual(run_op(f"{op}_e64 v0, {args}", x_bits), want)

    def test_inline_constants_literals_sgprs_and_aliasing(self):
        for op in self.OPS:
            # Inline float constants, VOP3 constant modifiers, and 32-bit literal.
            self.assertEqual(run_op(f"{op}_e32 v0, 0.5", 0), f32_bits(2.0))
            self.assertEqual(run_op(f"{op}_e32 v0, -0.5", 0), f32_bits(-2.0))
            self.assertEqual(run_op(f"{op}_e32 v0, 4.0", 0), f32_bits(0.25))
            self.assertEqual(run_op(f"{op}_e64 v0, -|2.0|", 0), f32_bits(-0.5))
            self.assertEqual(run_op(f"{op}_e64 v0, neg(0x3f800000)", 0), f32_bits(-1.0))
            self.assertEqual(run_op(f"{op}_e32 v0, 0x40400000", 0), 0x3EAAAAAB)

            # SGPR source and modified SGPR source.
            self.assertEqual(run_op(f"{op}_e32 v0, s4", 0, s4=f32_bits(-4.0)), f32_bits(-0.25))
            self.assertEqual(run_op(f"{op}_e64 v0, |s4| mul:2", 0, s4=f32_bits(-4.0)), f32_bits(0.5))

            # In-place aliasing: dst == src.
            self.assertEqual(run_ps_v0(["v_mov_b32 v0, 0x40000000", f"{op}_e32 v0, v0"]), f32_bits(0.5))
            self.assertEqual(run_ps_v0(["v_mov_b32 v0, 0xc0800000", f"{op}_e64 v0, -v0 mul:4"]), f32_bits(1.0))


# -----------------------------------------------------------------------------
# 4. End-to-End v_rcp_f16 Value Tests
# -----------------------------------------------------------------------------

class RcpF16ExecutionTests(unittest.TestCase):
    def test_hand_worked_values_across_e32_e64_and_bare(self):
        for suffix in ("_e32", "_e64", ""):
            for name, h_bits, want_half in RCP_F16_HAND_WORKED:
                with self.subTest(suffix=suffix, case=name):
                    got = run_op(f"v_rcp_f16{suffix} v0, v1", 0xCAFE0000 | h_bits)
                    self.assertEqual(got, merge_half(OLD, want_half, 0))

    def test_nan_preserves_nan_and_other_half(self):
        for suffix in ("_e32", "_e64"):
            for h in (H_QNAN, H_NEG_QNAN, H_SNAN):
                with self.subTest(suffix=suffix, h=hex(h)):
                    got = run_op(f"v_rcp_f16{suffix} v0, v1", 0xCAFE0000 | h)
                    self.assertEqual(got & 0xFFFF0000, OLD & 0xFFFF0000)
                    self.assertTrue(is_f16_nan(got & 0xFFFF), f"expected NaN in low half, got {got:#010x}")

    def test_zero_inputs_give_signed_infinity(self):
        # The ISA's rcp(+-0) = +-inf. The translator selects it itself rather than dividing by zero (undefined in
        # SPIR-V, so the interpreter raises); the f32 forms above still leave it to the driver.
        for line, v1, want, dst_hi in (
            ("v_rcp_f16_e32 v0, v1", (H_1P0 << 16) | H_POS_ZERO, 0x7C00, 0),
            ("v_rcp_f16_e32 v0, v1", (H_1P0 << 16) | H_NEG_ZERO, 0xFC00, 0),
            ("v_rcp_f16_e64 v0, v1 op_sel:[1,0]", (H_POS_ZERO << 16) | H_1P0, 0x7C00, 0),
            ("v_rcp_f16_e64 v0, v1 op_sel:[1,1]", (H_NEG_ZERO << 16) | H_1P0, 0xFC00, 1),
        ):
            with self.subTest(line=line, v1=hex(v1)):
                self.assertEqual(run_op(line, v1), merge_half(OLD, want, dst_hi))

    def test_op_sel_all_combinations(self):
        # v1 = (hi: -4.0, lo: 0.5) -> rcp(hi) = -0.25, rcp(lo) = 2.0.
        v1 = (H_NEG_4P0 << 16) | H_0P5
        for src_hi, dst_hi in itertools.product((0, 1), repeat=2):
            h_in = H_NEG_4P0 if src_hi else H_0P5
            line = f"v_rcp_f16_e64 v0, v1 op_sel:[{src_hi},{dst_hi}]"
            with self.subTest(line=line):
                self.assertEqual(run_op(line, v1), merge_half(OLD, ref_rcp_f16(h_in), dst_hi))

    def test_vop3_source_and_output_modifiers(self):
        v1 = (H_NEG_0P5 << 16) | H_4P0
        self.assertEqual(run_op("v_rcp_f16_e64 v0, -v1 op_sel:[0,0]", v1), merge_half(OLD, H_NEG_0P25, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, |v1| op_sel:[1,1]", v1), merge_half(OLD, H_2P0, 1))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, -|v1| op_sel:[1,0]", v1), merge_half(OLD, H_NEG_2P0, 0))

        # Output modifiers: on RDNA2 with denorms kept, omod is ignored on f16 results; clamp applies to [0.0, 1.0].
        self.assertEqual(run_op("v_rcp_f16_e64 v0, v1 mul:2", v1), merge_half(OLD, H_0P25, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, v1 mul:4", v1), merge_half(OLD, H_0P25, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, v1 div:2", v1), merge_half(OLD, H_0P25, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, v1 op_sel:[1,1] mul:2", v1), merge_half(OLD, H_NEG_2P0, 1))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, v1 op_sel:[1,1] clamp", v1), merge_half(OLD, H_POS_ZERO, 1))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, |v1| op_sel:[1,0] mul:2 clamp", v1), merge_half(OLD, H_1P0, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, |v1| op_sel:[1,1] clamp div:2", v1), merge_half(OLD, H_1P0, 1))

    def test_inline_constants_sgpr_and_aliasing(self):
        self.assertEqual(run_op("v_rcp_f16_e32 v0, 0.5", 0), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_rcp_f16_e32 v0, -0.5", 0), merge_half(OLD, H_NEG_2P0, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, -1.0", 0), merge_half(OLD, H_NEG_1P0, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, -|0x4000|", 0), merge_half(OLD, H_NEG_0P5, 0))

        s4 = (H_NEG_4P0 << 16) | H_0P5
        self.assertEqual(run_op("v_rcp_f16_e32 v0, s4", 0, s4=s4), merge_half(OLD, H_2P0, 0))
        self.assertEqual(run_op("v_rcp_f16_e64 v0, s4 op_sel:[1,1]", 0, s4=s4), merge_half(OLD, H_NEG_0P25, 1))

        # In-place aliasing: dst == src reading one half and writing the other keeps the original unread half.
        v0_init = (H_0P5 << 16) | H_4P0
        self.assertEqual(
            run_ps_v0([f"v_mov_b32 v0, {v0_init:#x}", "v_rcp_f16_e64 v0, v0 op_sel:[1,0]"]),
            (H_0P5 << 16) | H_2P0,
        )
        self.assertEqual(
            run_ps_v0([f"v_mov_b32 v0, {v0_init:#x}", "v_rcp_f16_e32 v0, v0"]),
            (H_0P5 << 16) | H_0P25,
        )


# -----------------------------------------------------------------------------
# 5. EXEC Masking & Negative Control Tests
# -----------------------------------------------------------------------------

class RcpExecMaskingTests(unittest.TestCase):
    def test_single_lane_exec_masking(self):
        for line in (
            "v_rcp_f32_e32 v0, v1",
            "v_rcp_iflag_f32_e32 v0, v1",
            "v_rcp_f16_e32 v0, v1",
            "v_rcp_f16_e64 v0, v1 op_sel:[1,1]",
        ):
            with self.subTest(line=line):
                got = run_ps_v0([
                    f"v_mov_b32 v1, {(H_2P0 << 16) | H_4P0:#x}",
                    f"v_mov_b32 v0, {OLD:#x}",
                    "s_mov_b32 exec_lo, 0",
                    line,
                    "s_mov_b32 exec_lo, 1",
                ])
                self.assertEqual(got, OLD)

        # Disabled ops leave v0 unchanged; re-enabling EXEC executes the subsequent op.
        got_toggled = run_ps_v0([
            f"v_mov_b32 v1, {(H_0P5 << 16) | H_4P0:#x}",
            f"v_mov_b32 v0, {OLD:#x}",
            "s_mov_b32 exec_lo, 0",
            "v_rcp_f16_e64 v0, v1 op_sel:[0,0]",
            "v_rcp_f16_e64 v0, v1 op_sel:[1,1]",
            "s_mov_b32 exec_lo, 1",
            "v_rcp_f16_e64 v0, v1 op_sel:[1,0]",
        ])
        self.assertEqual(got_toggled, merge_half(OLD, H_2P0, 0))

    def test_multilane_compute_exec_masking(self):
        # 4 lanes in wave32: disable odd lanes (lane & 1 != 0) before executing v_rcp_*.
        # Active even lanes (0, 2) compute the reciprocal; inactive odd lanes (1, 3) keep OLD.
        cases = (
            ("v_rcp_f32_e32 v3, v2", f32_bits(4.0), f32_bits(0.25)),
            ("v_rcp_iflag_f32_e64 v3, v2 mul:2", f32_bits(4.0), f32_bits(0.5)),
            ("v_rcp_f16_e64 v3, v2 op_sel:[1,1]", (H_0P5 << 16) | H_4P0, merge_half(OLD, H_2P0, 1)),
        )
        for insn, src_word, active_want in cases:
            with self.subTest(insn=insn):
                mem = {}
                lines = [
                    "v_lshlrev_b32 v4, 2, v0",          # lane byte offset = lane * 4
                    f"v_mov_b32 v2, {src_word:#x}",
                    f"v_mov_b32 v3, {OLD:#x}",
                    "v_and_b32 v5, v0, 1",
                    "v_cmpx_eq_u32 v5, 0",              # keep even lanes active
                    insn,
                    "s_mov_b32 exec_lo, -1",
                    "buffer_store_dword v3, v4, s[0:3], 0 offen",
                ]
                run_cs(lines, group_size=4, subgroup_size=32, memory=mem)
                self.assertEqual(
                    [mem[4 * i] for i in range(4)],
                    [active_want, OLD, active_want, OLD],
                )


# -----------------------------------------------------------------------------
# 6. Newton-Raphson Multi-Step Division Emulation (v_rcp_f32 + v_fma_f32 + v_mul_f32)
# -----------------------------------------------------------------------------

def run_nr_fdiv_ps(a_bits: int, b_bits: int) -> int:
    """Execute 4-step Newton-Raphson division in a pixel shader and return quotient bits."""
    return run_ps_v0([
        f"v_mov_b32 v0, {a_bits:#x}",
        f"v_mov_b32 v1, {b_bits:#x}",
        "v_rcp_f32_e32 v2, v1",
        "v_fma_f32 v3, -v1, v2, 1.0",
        "v_fma_f32 v4, v2, v3, v2",
        "v_mul_f32_e32 v0, v0, v4",
    ])


def run_nr_fdiv_steps_cs(a_bits: int, b_bits: int) -> tuple[int, int, int, int]:
    """Execute 4-step Newton-Raphson division in a compute shader and capture all intermediate registers."""
    mem: dict[int, int] = {}
    lines = [
        f"v_mov_b32 v0, {a_bits:#x}",
        f"v_mov_b32 v1, {b_bits:#x}",
        "v_rcp_f32_e32 v2, v1",
        "v_fma_f32 v3, -v1, v2, 1.0",
        "v_fma_f32 v4, v2, v3, v2",
        "v_mul_f32_e32 v5, v0, v4",
        "v_mov_b32 v6, 0",
        "buffer_store_dword v2, v6, s[0:3], 0 offen offset:0",
        "buffer_store_dword v3, v6, s[0:3], 0 offen offset:4",
        "buffer_store_dword v4, v6, s[0:3], 0 offen offset:8",
        "buffer_store_dword v5, v6, s[0:3], 0 offen offset:12",
    ]
    run_cs(lines, group_size=1, subgroup_size=32, memory=mem)
    return mem[0], mem[4], mem[8], mem[12]


class NewtonRaphsonMultiStepEmulationTests(unittest.TestCase):
    def test_hand_worked_pipeline_steps_match_oracle(self):
        for name, a_bits, b_bits, want_steps in NR_FDIV_HAND_WORKED:
            with self.subTest(case=name):
                got_steps = run_nr_fdiv_steps_cs(a_bits, b_bits)
                self.assertEqual(got_steps, want_steps)

    def test_end_to_end_quotients_match_oracle(self):
        cases = (
            (f32_bits(1.0), f32_bits(2.0)),
            (f32_bits(3.5), f32_bits(1.75)),
            (f32_bits(10.0), f32_bits(3.0)),
            (f32_bits(-15.0), f32_bits(4.0)),
            (f32_bits(100.5), f32_bits(-20.25)),
            (f32_bits(1e6), f32_bits(3.0)),
            (f32_bits(2.5e-5), f32_bits(4.0e5)),
            (f32_bits(1.0), f32_bits(7.0)),
            (f32_bits(22.0), f32_bits(7.0)),
            (f32_bits(-0.125), f32_bits(8.0)),
            (0x00000000, f32_bits(4.0)),
            (0x80000000, f32_bits(4.0)),
            (0x00000000, f32_bits(-4.0)),
            (0x80000000, f32_bits(-4.0)),
        )
        for a_bits, b_bits in cases:
            with self.subTest(a=hex(a_bits), b=hex(b_bits)):
                got = run_nr_fdiv_ps(a_bits, b_bits)
                want = ref_newton_raphson_fdiv_f32(a_bits, b_bits)
                self.assertEqual(got, want)

    def test_vop3_encodings_and_register_reuse(self):
        # In-place register reuse: v0 = dividend, v1 = divisor, v2 = y0 -> y1, v0 = quotient
        lines = [
            f"v_mov_b32 v0, {f32_bits(10.0):#x}",
            f"v_mov_b32 v1, {f32_bits(2.0):#x}",
            "v_rcp_f32_e64 v2, v1",
            "v_fma_f32 v3, -v1, v2, 1.0",
            "v_fma_f32 v2, v2, v3, v2",
            "v_mul_f32_e64 v0, v0, v2",
        ]
        self.assertEqual(run_ps_v0(lines), f32_bits(5.0))

        # Negative dividend and divisor with VOP3
        lines_neg = [
            f"v_mov_b32 v0, {f32_bits(-24.0):#x}",
            f"v_mov_b32 v1, {f32_bits(-6.0):#x}",
            "v_rcp_f32_e64 v2, v1",
            "v_fma_f32 v3, -v1, v2, 1.0",
            "v_fma_f32 v2, v2, v3, v2",
            "v_mul_f32_e64 v0, v0, v2",
        ]
        self.assertEqual(run_ps_v0(lines_neg), f32_bits(4.0))

    def test_sgpr_and_inline_constant_operands(self):
        # SGPR divisor
        lines_sgpr = [
            f"s_mov_b32 s4, {f32_bits(4.0):#x}",
            f"v_mov_b32 v0, {f32_bits(2.0):#x}",
            "v_rcp_f32_e32 v2, s4",
            "v_fma_f32 v3, -s4, v2, 1.0",
            "v_fma_f32 v4, v2, v3, v2",
            "v_mul_f32_e32 v0, v0, v4",
        ]
        self.assertEqual(run_ps_v0(lines_sgpr), f32_bits(0.5))

        # Inline float constants as divisor (2.0)
        lines_imm = [
            f"v_mov_b32 v0, {f32_bits(12.0):#x}",
            "v_rcp_f32_e32 v2, 2.0",
            "v_fma_f32 v3, -2.0, v2, 1.0",
            "v_fma_f32 v4, v2, v3, v2",
            "v_mul_f32_e32 v0, v0, v4",
        ]
        self.assertEqual(run_ps_v0(lines_imm), f32_bits(6.0))

    def test_multilane_simd_divergence_under_exec_mask(self):
        mem: dict[int, int] = {}
        lines = [
            "v_lshlrev_b32 v6, 2, v0",
            f"v_mov_b32 v7, {f32_bits(10.0):#x}",
            f"v_mov_b32 v8, {f32_bits(2.0):#x}",
            "v_cmp_eq_u32 vcc, v0, 2",
            f"v_cndmask_b32 v7, v7, {f32_bits(-15.0):#x}, vcc",
            f"v_cndmask_b32 v8, v8, {f32_bits(5.0):#x}, vcc",
            f"v_mov_b32 v5, {OLD:#x}",
            "v_and_b32 v9, v0, 1",
            "v_cmpx_eq_u32 v9, 0",
            "v_rcp_f32_e32 v2, v8",
            "v_fma_f32 v3, -v8, v2, 1.0",
            "v_fma_f32 v4, v2, v3, v2",
            "v_mul_f32_e32 v5, v7, v4",
            "s_mov_b32 exec_lo, -1",
            "buffer_store_dword v5, v6, s[0:3], 0 offen",
        ]
        run_cs(lines, group_size=4, subgroup_size=32, memory=mem)
        self.assertEqual(
            [mem[4 * i] for i in range(4)],
            [f32_bits(5.0), OLD, f32_bits(-3.0), OLD],
        )


class NewtonRaphsonAccuracyAndEdgeCaseTests(unittest.TestCase):
    def test_quotient_accuracy_within_one_ulp_wide_dynamic_range(self):
        cases = (
            (1.0e-30, 2.0),
            (3.14159265e-25, 1.41421356e-20),
            (1.0e-15, 3.0),
            (1.0e-5, 7.0),
            (1.0, 3.0),
            (7.0, 3.0),
            (22.0, 7.0),
            (355.0, 113.0),
            (1.234567e5, 9.87654e-5),
            (1.0e15, 3.0e5),
            (1.0e25, 2.5e10),
            (1.0e30, 4.0e15),
            (-1.0e20, 3.0),
            (1.0e20, -7.0),
            (-5.5e15, -1.1e5),
        )
        for a_f, b_f in cases:
            with self.subTest(a=a_f, b=b_f):
                ab = f32_bits(a_f)
                bb = f32_bits(b_f)
                got = run_nr_fdiv_ps(ab, bb)
                exact = ref_exact_fdiv_f32(ab, bb)
                ulp = f32_ulp_diff(got, exact)
                self.assertLessEqual(
                    ulp, 1,
                    f"Accuracy exceeds 1 ULP for {a_f}/{b_f}: got {got:#010x}, exact {exact:#010x}, diff {ulp}"
                )

    def test_subnormal_quotients_and_inputs(self):
        cases = (
            ("subnormal_div_2", 0x00000004, f32_bits(2.0), 0x00000002),
            ("neg_subnormal_div_2", 0x80000004, f32_bits(2.0), 0x80000002),
            ("subnormal_div_half", 0x00000002, f32_bits(0.5), 0x00000004),
            ("normal_underflow_to_subnormal", 0x00400000, f32_bits(2.0), 0x00200000),
            ("subnormal_divisor_to_normal", f32_bits(1.0), 0x00400000, 0x7F000000),
        )
        for name, ab, bb, want in cases:
            with self.subTest(case=name):
                got = run_nr_fdiv_ps(ab, bb)
                self.assertEqual(got, want)

    def test_sign_rule_preservation(self):
        cases = (
            (f32_bits(6.0), f32_bits(2.0), f32_bits(3.0)),
            (f32_bits(6.0), f32_bits(-2.0), f32_bits(-3.0)),
            (f32_bits(-6.0), f32_bits(2.0), f32_bits(-3.0)),
            (f32_bits(-6.0), f32_bits(-2.0), f32_bits(3.0)),
            (0x00000000, f32_bits(2.0), 0x00000000),
            (0x80000000, f32_bits(2.0), 0x80000000),
            (0x00000000, f32_bits(-2.0), 0x80000000),
            (0x80000000, f32_bits(-2.0), 0x00000000),
        )
        for ab, bb, want in cases:
            with self.subTest(a=hex(ab), b=hex(bb)):
                self.assertEqual(run_nr_fdiv_ps(ab, bb), want)

    def test_zero_and_infinite_divisor_behavior(self):
        # Division by +-0.0 is undefined in SPIR-V and raises UnsupportedConstruct
        for z in (0x00000000, 0x80000000):
            with self.subTest(zero=hex(z)):
                with self.assertRaises(si.UnsupportedConstruct):
                    run_nr_fdiv_ps(f32_bits(1.0), z)

        # Infinite divisor: 1 / +-inf -> +-0.0, step 2 FMA calculates 1 - inf * 0 -> NaN
        for inf_bits in (0x7F800000, 0xFF800000):
            with self.subTest(inf=hex(inf_bits)):
                got = run_nr_fdiv_ps(f32_bits(1.0), inf_bits)
                self.assertTrue(is_f32_nan_bits(got), f"expected NaN from inf divisor, got {got:#010x}")

    def test_negative_controls_for_pipeline_sabotage(self):
        a = f32_bits(7.0)
        b = f32_bits(3.0)
        lines = [
            f"v_mov_b32 v0, {a:#x}",
            f"v_mov_b32 v1, {b:#x}",
            "v_rcp_f32_e32 v2, v1",
            "v_fma_f32 v3, -v1, v2, 1.0",
            "v_fma_f32 v4, v2, v3, v2",
            "v_mul_f32_e32 v0, v0, v4",
        ]
        spv = translate_ps(lines)
        correct_res = si.run_fragment(spv)["%out_mrt0"][0]
        want = ref_newton_raphson_fdiv_f32(a, b)
        self.assertEqual(correct_res, want)

        # Sabotage 1: Corrupt error-term FMA to FMul
        bad_fma, n1 = re.subn(r"OpExtInst %float %glsl Fma\b", "OpFMul %float", spv, count=1)
        self.assertEqual(n1, 1)
        bad_res1 = si.run_fragment(bad_fma)["%out_mrt0"][0]
        self.assertNotEqual(bad_res1, want)
        self.assertGreater(f32_ulp_diff(bad_res1, want), 1000)

        # Sabotage 2: Corrupt final FMul to FAdd
        bad_mul, n2 = re.subn(r"OpFMul %float\b", "OpFAdd %float", spv, count=1)
        self.assertEqual(n2, 1)
        bad_res2 = si.run_fragment(bad_mul)["%out_mrt0"][0]
        self.assertNotEqual(bad_res2, want)
        self.assertGreater(f32_ulp_diff(bad_res2, want), 1000)

        # Sabotage 3: Inverting FMA sign (+b instead of -b in error term)
        lines_bad_sign = [
            f"v_mov_b32 v0, {a:#x}",
            f"v_mov_b32 v1, {b:#x}",
            "v_rcp_f32_e32 v2, v1",
            "v_fma_f32 v3, v1, v2, 1.0",
            "v_fma_f32 v4, v2, v3, v2",
            "v_mul_f32_e32 v0, v0, v4",
        ]
        bad_sign_res = run_ps_v0(lines_bad_sign)
        self.assertNotEqual(bad_sign_res, want)
        self.assertGreater(f32_ulp_diff(bad_sign_res, want), 1000)

        # Sabotage 4: Dropping base term in refinement (y0*e instead of y0 + y0*e)
        lines_bad_refine = [
            f"v_mov_b32 v0, {a:#x}",
            f"v_mov_b32 v1, {b:#x}",
            "v_rcp_f32_e32 v2, v1",
            "v_fma_f32 v3, -v1, v2, 1.0",
            "v_fma_f32 v4, v2, v3, 0.0",
            "v_mul_f32_e32 v0, v0, v4",
        ]
        bad_refine_res = run_ps_v0(lines_bad_refine)
        self.assertNotEqual(bad_refine_res, want)
        self.assertGreater(f32_ulp_diff(bad_refine_res, want), 1000)


class NegativeControlTests(unittest.TestCase):
    def test_sabotaged_spirv_is_detected(self):
        cases = (
            ("v_rcp_f32_e32 v0, v1", f32_bits(4.0), f32_bits(0.25)),
            ("v_rcp_iflag_f32_e64 v0, v1 mul:2", f32_bits(4.0), f32_bits(0.5)),
            ("v_rcp_f16_e64 v0, v1 op_sel:[0,1]", (H_1P0 << 16) | H_4P0, merge_half(OLD, H_0P25, 1)),
        )
        for line, v1_bits, want in cases:
            with self.subTest(line=line):
                spv = translate_ps([f"v_mov_b32 v0, {OLD:#x}", f"v_mov_b32 v1, {v1_bits:#x}", line])
                self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], want)
                bad, n = re.subn(r"OpFDiv %float %f1\b", "OpFMul %float %f1", spv, count=1)
                self.assertEqual(n, 1)
                self.assertNotEqual(si.run_fragment(bad)["%out_mrt0"][0], want)

        # Sabotaging FClamp on v_rcp_iflag_f32_e64 clamp changes clamped 1/0.5 = 2.0 -> 1.0.
        spv_clamp = translate_ps([
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {f32_bits(0.5):#x}",
            "v_rcp_iflag_f32_e64 v0, v1 clamp",
        ])
        self.assertEqual(si.run_fragment(spv_clamp)["%out_mrt0"][0], f32_bits(1.0))
        bad_clamp, n = re.subn(r"OpExtInst %float %glsl FClamp (%\w+) %f0 %f1", r"OpCopyObject %float \1", spv_clamp)
        self.assertEqual(n, 1)
        self.assertEqual(si.run_fragment(bad_clamp)["%out_mrt0"][0], f32_bits(2.0))


if __name__ == "__main__":
    unittest.main()
