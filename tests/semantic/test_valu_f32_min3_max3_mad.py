"""Semantic value tests for 3-input f32 VALU ops:
  v_min3_f32       D = min(S0, S1, S2)
  v_max3_f32       D = max(S0, S1, S2)
  v_mad_f32        D = f32(f32(S0 * S1) + S2)                  (unfused, NoContraction)
  v_fmac_f32       D = fma(S0, S1, D)                          (fused, accumulates into vdst)
  v_mad_legacy_f32 D = f32(legacy_mul(S0, S1) + S2)            (0 * anything -> +0.0, unfused)

Each test builds a hand-written RDNA2 pixel or compute shader listing, translates it with tools/rdna2spirv.py,
executes the SPIR-V in tests/semantic/spirv_mini_interp.py, and compares the result words against an independent
plain-Python IEEE-754 binary32 / RDNA2 ISA reference oracle (stdlib only, no game data).

Observed translator / interpreter findings (kept out of failing assertions and documented here):
  1. v_min3_f32 / v_max3_f32 with NaN operands: tools/rdna2spirv.py emits GLSL.std.450 NMin / NMax, which give the
     other operand for a NaN, as RDNA2 hardware does. Those rows are asserted in tests/semantic/
     test_float_nan_fract.py; NaN inputs stay out of the value assertions here (clamp-on-NaN is still FClamp, which
     spirv_mini_interp.py rejects). NaN semantics are tested on v_mad_f32, v_fmac_f32, and v_mad_legacy_f32.
  2. v_min3_f32 / v_max3_f32 signed-zero ties: RDNA2 hardware orders -0.0 below +0.0 (measured on an AMD Radeon
     RX 6800: min(+0, -0) = -0 and max(+0, -0) = +0 in either operand order), and spirv_mini_interp.py's NMin / NMax
     do the same, so every zero triple asserts exact bit patterns.
  3. Fma double-rounding in spirv_mini_interp.py: spirv_mini_interp.py implements GLSL.std.450 Fma as Python
     binary64 `f32(x * y + z)`. Binary32 products require at most 48 <= 53 significand bits and never overflow
     binary64, so intermediate rounding differences vs unfused v_mad_f32 are modeled accurately except for extreme
     exponent-gap (>53-bit) midpoint ties (e.g. 2^30*(1+3*2^-12) * 2^30*(1+2^-12) - 2^-10), which double-round.
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

# Hand-built compute header (RSRC2: 4 user SGPRs, tid=x, group 4x1x1) matching cs_bounds in test_value_semantics.py.
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 4, 1, 1)
CS_FSSM_BYTES = bytes(CS_FSSM)

# Canonical binary32 bit patterns used across tests.
POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
SUBNORM_MIN = 0x00000001       # 2^-149
NEG_SUBNORM_MIN = 0x80000001   # -2^-149
FLT_MIN = 0x00800000           # 2^-126
NEG_FLT_MIN = 0x80800000       # -2^-126
HALF = 0x3F000000              # 0.5
ONE = 0x3F800000               # 1.0
NEG_ONE = 0xBF800000           # -1.0
ONE_HALF = 0x3FC00000          # 1.5
TWO = 0x40000000               # 2.0
NEG_TWO = 0xC0000000           # -2.0
TWO_HALF = 0x40200000          # 2.5
THREE = 0x40400000             # 3.0
FOUR = 0x40800000              # 4.0
FLT_MAX = 0x7F7FFFFF           # (2 - 2^-23) * 2^127
NEG_FLT_MAX = 0xFF7FFFFF       # -(2 - 2^-23) * 2^127
POS_INF = 0x7F800000           # +inf
NEG_INF = 0xFF800000           # -inf
QNAN = 0x7FC00000              # canonical positive qNaN
NEG_QNAN = 0xFFC00000          # negative qNaN
SNAN = 0x7F800001              # signalling NaN


def bits_to_f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return POS_INF if val > 0 else NEG_INF


def is_nan_bits(bits: int) -> bool:
    bits &= MASK32
    return (bits & 0x7F800000) == 0x7F800000 and (bits & 0x007FFFFF) != 0


def exact_f32_to_fraction(bits: int) -> Fraction:
    """Decode a finite binary32 bit pattern into an exact rational Fraction."""
    bits &= MASK32
    sign = -1 if (bits >> 31) else 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError(f"non-finite bits {bits:#010x}")
    if exp == 0:
        return Fraction(sign * frac, 1 << 149)
    mant = (1 << 23) | frac
    shift = exp - 150
    if shift >= 0:
        return Fraction(sign * (mant << shift), 1)
    return Fraction(sign * mant, 1 << (-shift))


def round_fraction_to_f32_bits(val: Fraction, zero_sign: int = 0) -> int:
    """Round an exact Fraction to IEEE-754 binary32 (round-to-nearest-even)."""
    if val == 0:
        return NEG_ZERO if zero_sign else POS_ZERO
    sign = 1 if val < 0 else 0
    aval = -val if sign else val
    num, den = aval.numerator, aval.denominator
    # Binary exponent floor(log2(aval)):
    e = num.bit_length() - den.bit_length()
    if (num << max(0, -e)) < (den << max(0, e)):
        e -= 1

    if e >= 128:
        return (sign << 31) | POS_INF

    # Scale to integer significand q with 1-ulp unit = 2^shift_exp.
    if e < -126:
        # Subnormal range: quantum is 2^-149
        target_shift = 149
    else:
        # Normal range: quantum is 2^(e - 23)
        target_shift = 23 - e

    if target_shift >= 0:
        scaled_num = num << target_shift
    else:
        scaled_num = num
        den = den << (-target_shift)

    q, rem = divmod(scaled_num, den)
    twice_rem = rem << 1
    if twice_rem > den or (twice_rem == den and (q & 1)):
        q += 1

    if e < -126:
        if q >= (1 << 23):
            # Rounded up into smallest normal (exp=1, frac=0)
            return (sign << 31) | (1 << 23)
        return (sign << 31) | q

    if q >= (1 << 24):
        q >>= 1
        e += 1
        if e >= 128:
            return (sign << 31) | POS_INF

    exp_field = e + 127
    frac_field = q & 0x7FFFFF
    return (sign << 31) | (exp_field << 23) | frac_field


def _zero_sign_bit(bits: int) -> int:
    return 1 if (bits & MASK32) == NEG_ZERO else 0


def _zero_ordered(v: float):
    """Sort key with -0.0 below +0.0, the order RDNA2 v_min/v_max use (measured on an AMD Radeon RX 6800:
    min(+0, -0) = -0 and max(+0, -0) = +0 in either operand order; docs/design.md, float min/max/med3)."""
    return (v, math.copysign(1.0, v))


def ref_min3_f32(xb: int, yb: int, zb: int) -> int:
    """Reference for v_min3_f32 = min(min(x, y), z) on non-NaN f32s, with -0.0 < +0.0."""
    x, y, z = bits_to_f32(xb), bits_to_f32(yb), bits_to_f32(zb)
    if math.isnan(x) or math.isnan(y) or math.isnan(z):
        raise ValueError("NaN operands are covered by tests/semantic/test_float_nan_fract.py")
    return f32_to_bits(min((x, y, z), key=_zero_ordered))


def ref_max3_f32(xb: int, yb: int, zb: int) -> int:
    """Reference for v_max3_f32 = max(max(x, y), z) on non-NaN f32s, with -0.0 < +0.0."""
    x, y, z = bits_to_f32(xb), bits_to_f32(yb), bits_to_f32(zb)
    if math.isnan(x) or math.isnan(y) or math.isnan(z):
        raise ValueError("NaN operands are covered by tests/semantic/test_float_nan_fract.py")
    return f32_to_bits(max((x, y, z), key=_zero_ordered))


def _mul_bits_f32(xb: int, yb: int) -> int:
    x, y = bits_to_f32(xb), bits_to_f32(yb)
    if math.isnan(x) or math.isnan(y):
        return QNAN
    if (x == 0.0 and math.isinf(y)) or (math.isinf(x) and y == 0.0):
        return QNAN
    sign = ((xb ^ yb) >> 31) & 1
    if math.isinf(x) or math.isinf(y):
        return (sign << 31) | POS_INF
    if x == 0.0 or y == 0.0:
        return NEG_ZERO if sign else POS_ZERO
    return round_fraction_to_f32_bits(exact_f32_to_fraction(xb) * exact_f32_to_fraction(yb), zero_sign=sign)


def _add_bits_f32(ab: int, bb: int) -> int:
    a, b = bits_to_f32(ab), bits_to_f32(bb)
    if math.isnan(a) or math.isnan(b):
        return QNAN
    if math.isinf(a) and math.isinf(b):
        if (ab >> 31) != (bb >> 31):
            return QNAN
        return ab & MASK32
    if math.isinf(a):
        return ab & MASK32
    if math.isinf(b):
        return bb & MASK32
    if a == 0.0 and b == 0.0:
        # IEEE-754 RTN: (-0.0) + (-0.0) = -0.0; (+0.0) + (-0.0) = +0.0.
        return NEG_ZERO if (_zero_sign_bit(ab) and _zero_sign_bit(bb)) else POS_ZERO
    total = exact_f32_to_fraction(ab) + exact_f32_to_fraction(bb)
    # Exact cancellation of non-zero opposites yields +0.0 under RTN.
    return round_fraction_to_f32_bits(total, zero_sign=0)


def ref_mad_f32(xb: int, yb: int, zb: int) -> int:
    """Unfused IEEE-754 binary32 multiply followed by binary32 add."""
    return _add_bits_f32(_mul_bits_f32(xb, yb), zb)


def ref_fmac_f32(xb: int, yb: int, accb: int) -> int:
    """True IEEE-754 binary32 fused multiply-add: fma(x, y, acc)."""
    x, y, acc = bits_to_f32(xb), bits_to_f32(yb), bits_to_f32(accb)
    if math.isnan(x) or math.isnan(y) or math.isnan(acc):
        return QNAN
    if (x == 0.0 and math.isinf(y)) or (math.isinf(x) and y == 0.0):
        return QNAN
    prod_sign = ((xb ^ yb) >> 31) & 1
    if math.isinf(x) or math.isinf(y):
        prod_inf = (prod_sign << 31) | POS_INF
        return _add_bits_f32(prod_inf, accb)
    if math.isinf(acc):
        return accb & MASK32
    prod = exact_f32_to_fraction(xb) * exact_f32_to_fraction(yb)
    total = prod + exact_f32_to_fraction(accb)
    if total == 0:
        # (-0.0) + (-0.0) -> -0.0 only if product is signed zero with sign=1 and acc is -0.0.
        if (x == 0.0 or y == 0.0) and acc == 0.0 and prod_sign == 1 and _zero_sign_bit(accb) == 1:
            return NEG_ZERO
        return POS_ZERO
    return round_fraction_to_f32_bits(total, zero_sign=0)


def ref_mad_legacy_f32(xb: int, yb: int, zb: int) -> int:
    """RDNA2 v_mad_legacy_f32: 0 * anything (inf/NaN/-0) -> +0.0, then unfused binary32 add."""
    x, y = bits_to_f32(xb), bits_to_f32(yb)
    if (not math.isnan(x) and x == 0.0) or (not math.isnan(y) and y == 0.0):
        prod_bits = POS_ZERO
    else:
        prod_bits = _mul_bits_f32(xb, yb)
    return _add_bits_f32(prod_bits, zb)


def apply_src_mod(bits: int, neg: bool = False, absm: bool = False) -> int:
    """RDNA2 source modifier order: abs first, then neg (operates on sign bit, even for inf/NaN/0)."""
    bits &= MASK32
    if absm:
        bits &= 0x7FFFFFFF
    if neg:
        bits ^= 0x80000000
    return bits


def apply_out_mods(bits: int, omod: str | None = None, clamp: bool = False) -> int:
    """RDNA2 output modifiers: mul:2 / mul:4 / div:2 applied before clamp [0.0, 1.0]."""
    if omod is not None:
        factor = {"mul:2": TWO, "mul:4": FOUR, "div:2": HALF}[omod]
        bits = _mul_bits_f32(bits, factor)
    if clamp:
        v = bits_to_f32(bits)
        if math.isnan(v):
            raise ValueError("clamp on NaN is undefined in GLSL FClamp")
        if v < 0.0:
            return POS_ZERO
        if v > 1.0:
            return ONE
    return bits & MASK32


_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _PS_PATH = os.path.join(disasm_dir, "ps_f32_valu3.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_f32_valu3.s")
    with open(os.path.join(_TMP_DIR.name, "ps_f32_valu3.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(fssm_dir, "cs_f32_valu3.fssm"), "wb") as f:
        f.write(CS_FSSM_BYTES)


def tearDownModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH
    if _TMP_DIR is not None:
        gc.collect()
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _PS_PATH = None
        _CS_PATH = None


def translate_ps(lines: list[str], export_regs: tuple[str, str, str, str] = ("v0", "v0", "v0", "v0")) -> str:
    if _TMP_DIR is None:
        setUpModule()
    assert _PS_PATH is not None
    asm = (
        "; test_valu_f32_min3_max3_mad\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + f"\texp mrt0 {', '.join(export_regs)} done vm\n"
        + "\ts_endpgm\n"
    )
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
    return spv


def run_ps_mrt0(lines: list[str], export_regs: tuple[str, str, str, str] = ("v0", "v0", "v0", "v0")) -> list[int]:
    spv = translate_ps(lines, export_regs=export_regs)
    return list(si.run_fragment(spv)["%out_mrt0"])


def run_three_input(op: str, a_bits: int, b_bits: int, c_bits: int, mods: str = "") -> int:
    """Translate and run a single 3-operand instruction (`v_min3_f32`, `v_max3_f32`, `v_mad_f32`,
    `v_mad_legacy_f32`, or `v_fmac_f32` where `c_bits` initializes the `v0` accumulator)."""
    mod_s = f" {mods}" if mods else ""
    if op.startswith(("v_fmac_f32", "v_mac_f32", "v_mac_legacy_f32")):
        lines = [
            f"v_mov_b32 v1, {a_bits & MASK32:#x}",
            f"v_mov_b32 v2, {b_bits & MASK32:#x}",
            f"v_mov_b32 v0, {c_bits & MASK32:#x}",
            f"{op} v0, v1, v2{mod_s}",
        ]
    else:
        lines = [
            f"v_mov_b32 v1, {a_bits & MASK32:#x}",
            f"v_mov_b32 v2, {b_bits & MASK32:#x}",
            f"v_mov_b32 v3, {c_bits & MASK32:#x}",
            f"{op} v0, v1, v2, v3{mod_s}",
        ]
    return run_ps_mrt0(lines)[0]


def run_cs_workgroup_exec(lines: list[str], lanes: int = 4) -> dict[int, int]:
    """Translate and run a compute shader storing `v1` per lane at byte address `4 * lane`."""
    if _TMP_DIR is None:
        setUpModule()
    assert _CS_PATH is not None
    asm = "; test_valu_f32_min3_max3_mad_cs\n" + "".join(f"\t{ln}\n" for ln in lines) + "\ts_endpgm\n"
    with open(_CS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)
        gc.collect()
    mem: dict[int, int] = {}
    si.run_workgroup(
        spv,
        lanes,
        32,
        inputs_for=lambda i: {"%local_id": (i, 0, 0), "%wg_id": (0, 0, 0)},
        push_constants=[0, 0, 64, 0x31016FAC] + [0] * 28,
        memory=mem,
    )
    return mem


class ReferenceOracleTests(unittest.TestCase):
    """Cross-check plain-Python Fraction/IEEE-754 reference against hand-derived binary32 bit constants."""

    def test_hand_worked_constants(self):
        # v_min3_f32 / v_max3_f32
        self.assertEqual(ref_min3_f32(ONE, NEG_TWO, HALF), NEG_TWO)
        self.assertEqual(ref_max3_f32(ONE, NEG_TWO, THREE), THREE)
        self.assertEqual(ref_min3_f32(POS_INF, FLT_MAX, NEG_INF), NEG_INF)
        self.assertEqual(ref_max3_f32(POS_INF, FLT_MAX, NEG_INF), POS_INF)
        self.assertEqual(ref_min3_f32(NEG_ZERO, NEG_ZERO, ONE), NEG_ZERO)
        self.assertEqual(ref_max3_f32(POS_ZERO, POS_ZERO, NEG_ONE), POS_ZERO)

        # Fused vs unfused divergence: (1 + 2^-20)*(1 - 2^-20) - 1 = -2^-40 (0xAB800000)
        a = 0x3F800008  # 1 + 2^-20
        b = 0x3F7FFFF0  # 1 - 2^-20
        self.assertEqual(ref_mad_f32(a, b, NEG_ONE), POS_ZERO)
        self.assertEqual(ref_fmac_f32(a, b, NEG_ONE), 0xAB800000)

        # (1 + 2^-23)^2 - (1 + 2^-22) = 2^-46 (0x28800000)
        one_plus_ulp = 0x3F800001
        neg_one_plus_2ulp = 0xBF800002
        self.assertEqual(ref_mad_f32(one_plus_ulp, one_plus_ulp, neg_one_plus_2ulp), POS_ZERO)
        self.assertEqual(ref_fmac_f32(one_plus_ulp, one_plus_ulp, neg_one_plus_2ulp), 0x28800000)

        # Intermediate overflow: 2^100 * 2^28 - 2^127
        p100 = 0x71800000   # 2^100
        p28 = 0x4D800000    # 2^28
        n127 = 0xFF000000   # -2^127
        self.assertEqual(ref_mad_f32(p100, p28, n127), POS_INF)
        self.assertEqual(ref_fmac_f32(p100, p28, n127), 0x7F000000)

        # Legacy 0 * anything -> +0.0
        self.assertEqual(ref_mad_legacy_f32(POS_ZERO, POS_INF, NEG_TWO), NEG_TWO)
        self.assertEqual(ref_mad_legacy_f32(NEG_ZERO, NEG_INF, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_mad_legacy_f32(POS_ZERO, QNAN, THREE), THREE)
        self.assertEqual(ref_mad_legacy_f32(NEG_ZERO, TWO, NEG_ZERO), POS_ZERO)
        self.assertEqual(ref_mad_f32(NEG_ZERO, TWO, NEG_ZERO), NEG_ZERO)
        self.assertEqual(ref_fmac_f32(NEG_ZERO, TWO, NEG_ZERO), NEG_ZERO)
        self.assertTrue(is_nan_bits(ref_mad_f32(POS_ZERO, POS_INF, ONE)))
        self.assertTrue(is_nan_bits(ref_fmac_f32(POS_ZERO, POS_INF, ONE)))

        # Source and output modifiers match reference composition: neg/abs apply before mul+add, omod before clamp.
        self.assertEqual(
            apply_out_mods(
                ref_mad_f32(apply_src_mod(NEG_TWO, neg=True), apply_src_mod(THREE, neg=True, absm=True), FOUR),
                "mul:2",
                clamp=True,
            ),
            POS_ZERO,  # (2.0 * (-3.0) + 4.0) * 2.0 = -4.0 -> clamp -> 0.0
        )
        self.assertEqual(
            apply_out_mods(
                ref_mad_f32(apply_src_mod(NEG_TWO, neg=True), apply_src_mod(0xC0400000, absm=True), FOUR),
                "mul:2",
                clamp=True,
            ),
            ONE,  # (2.0 * 3.0 + 4.0) * 2.0 = 20.0 -> clamp -> 1.0
        )
        self.assertEqual(apply_out_mods(0x3EC00000, "mul:2", clamp=True), 0x3F400000)  # 0.375 * 2 = 0.75
        self.assertEqual(apply_out_mods(0x3F400000, "mul:2", clamp=True), ONE)         # 0.75 * 2 = 1.5 -> 1.0
        self.assertEqual(apply_out_mods(THREE, "div:2", clamp=True), ONE)              # 1.5 -> 1.0
        self.assertEqual(apply_out_mods(NEG_TWO, None, clamp=True), POS_ZERO)


class Min3Max3F32Tests(unittest.TestCase):
    """Semantic tests for v_min3_f32 and v_max3_f32 (NaNs excluded; see module docstring)."""

    def test_every_permutation_across_value_classes(self):
        triples = (
            (NEG_INF, NEG_FLT_MAX, FLT_MAX),
            (NEG_INF, NEG_ZERO, POS_INF),
            (NEG_TWO, ONE_HALF, THREE),
            (NEG_FLT_MIN, NEG_SUBNORM_MIN, SUBNORM_MIN),
            (FLT_MIN, HALF, FLT_MAX),
            (NEG_ONE, NEG_ONE, TWO),
            (THREE, HALF, THREE),
            (TWO_HALF, TWO_HALF, TWO_HALF),
        )
        for op, ref_fn in (("v_min3_f32", ref_min3_f32), ("v_max3_f32", ref_max3_f32)):
            for triple in triples:
                for perm in sorted(set(itertools.permutations(triple))):
                    with self.subTest(op=op, perm=[hex(w) for w in perm]):
                        self.assertEqual(run_three_input(op, *perm), ref_fn(*perm))

    def test_grid_sweep_batched(self):
        # 7^3 = 343 triples per op, batched 4 components per pixel shader export.
        grid = (NEG_INF, NEG_TWO, NEG_ONE, POS_ZERO, ONE, TWO_HALF, POS_INF)
        triples = list(itertools.product(grid, repeat=3))
        for op, ref_fn in (("v_min3_f32", ref_min3_f32), ("v_max3_f32", ref_max3_f32)):
            for start in range(0, len(triples), 4):
                chunk = triples[start:start + 4]
                lines, expected = [], []
                for slot, (a, b, c) in enumerate(chunk):
                    base = 10 + 4 * slot
                    lines += [
                        f"v_mov_b32 v{base + 1}, {a:#x}",
                        f"v_mov_b32 v{base + 2}, {b:#x}",
                        f"v_mov_b32 v{base + 3}, {c:#x}",
                        f"{op} v{base}, v{base + 1}, v{base + 2}, v{base + 3}",
                    ]
                    expected.append(ref_fn(a, b, c))
                regs = [f"v{10 + 4 * slot}" for slot in range(len(chunk))]
                while len(regs) < 4:
                    regs.append(regs[0])
                    expected.append(expected[0])
                with self.subTest(op=op, batch=start // 4):
                    self.assertEqual(run_ps_mrt0(lines, tuple(regs)), expected)

    def test_signed_zeros(self):
        # Unambiguous signed-zero triples (same-sign zeros or strictly ordered non-zero extreme).
        exact_cases = [
            ("v_min3_f32", (NEG_ZERO, NEG_ZERO, ONE), NEG_ZERO),
            ("v_min3_f32", (ONE, NEG_ZERO, NEG_ZERO), NEG_ZERO),
            ("v_min3_f32", (POS_ZERO, POS_ZERO, ONE), POS_ZERO),
            ("v_min3_f32", (POS_ZERO, NEG_ZERO, NEG_ONE), NEG_ONE),
            ("v_max3_f32", (NEG_ZERO, NEG_ZERO, NEG_ONE), NEG_ZERO),
            ("v_max3_f32", (NEG_ONE, NEG_ZERO, NEG_ZERO), NEG_ZERO),
            ("v_max3_f32", (POS_ZERO, POS_ZERO, NEG_ONE), POS_ZERO),
            ("v_max3_f32", (POS_ZERO, NEG_ZERO, ONE), ONE),
        ]
        for op, xs, want in exact_cases:
            with self.subTest(op=op, xs=[hex(w) for w in xs]):
                self.assertEqual(run_three_input(op, *xs), want)

        # Mixed (+0.0, -0.0) cases: -0.0 is the smaller zero, as on hardware.
        for op, ref_fn, third in (("v_min3_f32", ref_min3_f32, ONE), ("v_max3_f32", ref_max3_f32, NEG_ONE)):
            for perm in sorted(set(itertools.permutations((POS_ZERO, NEG_ZERO, third)))):
                with self.subTest(op=op, perm=[hex(w) for w in perm]):
                    got = run_three_input(op, *perm)
                    self.assertIn(got, (POS_ZERO, NEG_ZERO))
                    self.assertEqual(got, ref_fn(*perm))

    def test_e64_inline_constants_sgprs_source_and_output_modifiers(self):
        # _e64 suffix, inline float constants (including 0.15915494), SGPRs, neg/abs, mul/div/clamp.
        words = run_ps_mrt0([
            f"s_mov_b32 s4, {THREE:#x}",
            f"v_mov_b32 v1, {NEG_TWO:#x}",
            "v_min3_f32_e64 v10, -v1, 0.15915494, |-4.0|",            # min(2.0, 1/(2*pi), 4.0) -> 0.15915494
            "v_max3_f32_e64 v11, -|s4|, -0.5, neg(2.0) div:2 clamp", # max(-3.0, -0.5, -2.0)=-0.5 -> -0.25 -> +0.0
            "v_min3_f32 v12, s4, 0.5, 4.0 mul:4 clamp",              # 0.5 * 4 -> 2.0 -> clamp -> 1.0
            f"v_mov_b32 v2, {FLT_MAX:#x}",
            "v_max3_f32 v13, v2, 1.0, 2.0 mul:2",                    # FLT_MAX * 2 -> +inf
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [0x3E22F983, POS_ZERO, ONE, POS_INF])

        # clamp on +-inf saturates to [0.0, 1.0].
        self.assertEqual(run_three_input("v_max3_f32", POS_INF, ONE, TWO, mods="clamp"), ONE)
        self.assertEqual(run_three_input("v_min3_f32", NEG_INF, ONE, TWO, mods="clamp"), POS_ZERO)

        # div:2 on FLT_MAX halves exponent cleanly.
        self.assertEqual(run_three_input("v_max3_f32", FLT_MAX, ONE, TWO, mods="div:2"), 0x7EFFFFFF)

    def test_destination_aliasing(self):
        for op, ref_fn in (("v_min3_f32", ref_min3_f32), ("v_max3_f32", ref_max3_f32)):
            for expr, a, b, c in (
                ("v0, v0, v1, v2", NEG_TWO, THREE, ONE),
                ("v0, v1, v0, v2", ONE, NEG_TWO, THREE),
                ("v0, v1, v2, v0", THREE, ONE, NEG_TWO),
                ("v0, v0, v0, v1", FOUR, FOUR, HALF),
            ):
                with self.subTest(op=op, expr=expr):
                    got = run_ps_mrt0([
                        f"v_mov_b32 v0, {a:#x}",
                        f"v_mov_b32 v1, {b:#x}",
                        f"v_mov_b32 v2, {c:#x}",
                        f"{op} {expr}",
                    ])[0]
                    # Order (v0, v1, v2) corresponds to the register values placed into the source slots.
                    regs = {"v0": a, "v1": b, "v2": c}
                    s0, s1, s2 = (regs[r.strip()] for r in expr.split(",")[1:])
                    self.assertEqual(got, ref_fn(s0, s1, s2))


class MadAndFmacF32Tests(unittest.TestCase):
    """Semantic tests for unfused v_mad_f32 and fused accumulating v_fmac_f32."""

    def test_fused_vs_unfused_divergence(self):
        # Normalized cases where intermediate product rounding, overflow, or underflow separates mad from fmac.
        cases = [
            # (a, b, c, want_mad, want_fmac)
            (0x3F800008, 0x3F7FFFF0, NEG_ONE, POS_ZERO, 0xAB800000),            # (1+2^-20)*(1-2^-20) - 1 = -2^-40
            (0x3F800001, 0x3F800001, 0xBF800002, POS_ZERO, 0x28800000),         # (1+2^-23)^2 - (1+2^-22) = 2^-46
            (0x3FC00001, 0x3FBFFFFF, 0xC0100000, POS_ZERO, 0xA8800000),         # (1.5+2^-23)*(1.5-2^-23) - 2.25 = -2^-46
            (0x71800000, 0x4D800000, 0xFF000000, POS_INF, 0x7F000000),          # 2^100*2^28 - 2^127: ovf vs finite
            (0xF1800000, 0x4D800000, 0x7F000000, NEG_INF, 0xFF000000),          # -2^100*2^28 + 2^127: -ovf vs finite
            # Subnormal operands or products: see DenormalFlushTests (RDNA2 mad flushes them).
        ]
        for a, b, c, want_mad, want_fmac in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(ref_mad_f32(a, b, c), want_mad)
                self.assertEqual(ref_fmac_f32(a, b, c), want_fmac)
                self.assertNotEqual(want_mad, want_fmac)
                self.assertEqual(run_three_input("v_mad_f32", a, b, c), want_mad)
                self.assertEqual(run_three_input("v_fmac_f32", a, b, c), want_fmac)

    def test_signed_zeros_infinities_and_nans(self):
        exact_cases = [
            (NEG_ZERO, TWO, NEG_ZERO, NEG_ZERO),          # (-0)*2 + (-0) -> -0
            (POS_ZERO, TWO, NEG_ZERO, POS_ZERO),          # (+0)*2 + (-0) -> +0
            (NEG_TWO, THREE, 0x40C00000, POS_ZERO),       # -6 + 6 -> +0
            (NEG_ZERO, NEG_ONE, NEG_ZERO, POS_ZERO),      # (+0) + (-0) -> +0
            (ONE, NEG_ZERO, NEG_ZERO, NEG_ZERO),          # 1*(-0) + (-0) -> -0
            (POS_INF, TWO, POS_INF, POS_INF),
            (NEG_INF, TWO, ONE, NEG_INF),
            (TWO, THREE, NEG_INF, NEG_INF),
            (NEG_INF, NEG_TWO, NEG_ONE, POS_INF),
            (ONE_HALF, TWO_HALF, HALF, 0x40880000),       # 1.5 * 2.5 + 0.5 = 4.25
        ]
        for op, ref_fn in (("v_mad_f32", ref_mad_f32), ("v_fmac_f32", ref_fmac_f32)):
            for a, b, c, want in exact_cases:
                with self.subTest(op=op, a=hex(a), b=hex(b), c=hex(c)):
                    self.assertEqual(ref_fn(a, b, c), want)
                    self.assertEqual(run_three_input(op, a, b, c), want)

        nan_cases = [
            (POS_ZERO, POS_INF, ONE),
            (NEG_INF, NEG_ZERO, TWO),
            (POS_ZERO, QNAN, THREE),
            (QNAN, ONE, TWO),
            (ONE, TWO, SNAN),
            (POS_INF, ONE, NEG_INF),
            (NEG_INF, ONE, POS_INF),
        ]
        for op, ref_fn in (("v_mad_f32", ref_mad_f32), ("v_fmac_f32", ref_fmac_f32)):
            for a, b, c in nan_cases:
                with self.subTest(op=op, nan_case=(hex(a), hex(b), hex(c))):
                    self.assertTrue(is_nan_bits(ref_fn(a, b, c)))
                    self.assertTrue(is_nan_bits(run_three_input(op, a, b, c)))

    def test_fmac_accumulator_aliasing_and_chaining(self):
        # vdst aliases src0, src1, or both, and chains across consecutive accumulations.
        chained = run_ps_mrt0([
            f"v_mov_b32 v0, {TWO:#x}",
            f"v_mov_b32 v1, {THREE:#x}",
            "v_fmac_f32 v0, v0, v1",        # 2*3 + 2 = 8.0
            f"v_mov_b32 v2, {TWO:#x}",
            f"v_mov_b32 v3, {FOUR:#x}",
            "v_fmac_f32 v2, v3, v2",        # 4*2 + 2 = 10.0
            f"v_mov_b32 v4, {THREE:#x}",
            "v_fmac_f32 v4, v4, v4",        # 3*3 + 3 = 12.0
            f"v_mov_b32 v5, {HALF:#x}",
            f"v_mov_b32 v6, {ONE_HALF:#x}",
            f"v_mov_b32 v7, {TWO:#x}",
            f"v_mov_b32 v8, {TWO_HALF:#x}",
            "v_fmac_f32_e32 v5, v6, v7",    # 1.5*2.0 + 0.5 = 3.5
            "v_fmac_f32_e32 v5, v8, v7",    # 2.5*2.0 + 3.5 = 8.5
        ], ("v0", "v2", "v4", "v5"))
        self.assertEqual(chained, [f32_to_bits(v) for v in (8.0, 10.0, 12.0, 8.5)])

    def test_mad_destination_aliasing(self):
        for expr, want in (
            ("v0, v0, v1, v2", f32_to_bits(2.0 * 3.0 + 4.0)),
            ("v0, v1, v0, v2", f32_to_bits(3.0 * 2.0 + 4.0)),
            ("v0, v1, v2, v0", f32_to_bits(3.0 * 4.0 + 2.0)),
            ("v0, v0, v0, v0", f32_to_bits(2.0 * 2.0 + 2.0)),
        ):
            with self.subTest(expr=expr):
                got = run_ps_mrt0([
                    f"v_mov_b32 v0, {TWO:#x}",
                    f"v_mov_b32 v1, {THREE:#x}",
                    f"v_mov_b32 v2, {FOUR:#x}",
                    f"v_mad_f32 {expr}",
                ])[0]
                self.assertEqual(got, want)

    def test_variants_modifiers_and_nocontraction(self):
        # _e32 literal in src0, _e64 source/output modifiers, SGPRs, inline float constants.
        words = run_ps_mrt0([
            f"s_mov_b32 s4, {THREE:#x}",
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v10, {ONE:#x}",
            "v_fmac_f32_e32 v10, 0x3fc00000, v1",           # 1.5 * 2.0 + 1.0 = 4.0
            f"v_mov_b32 v11, {HALF:#x}",
            "v_fmac_f32_e64 v11, -v1, |s4| mul:2",          # (-2.0 * 3.0 + 0.5) * 2 = -11.0
            f"v_mov_b32 v12, {THREE:#x}",
            "v_fmac_f32_e64 v12, s4, -0.5 div:2 clamp",     # (3.0 * -0.5 + 3.0) * 0.5 = 0.75
            "v_mad_f32_e64 v13, -v1, -|s4|, 4.0 mul:2 clamp",  # (-2 * -3 + 4) * 2 = 20 -> clamp -> 1.0
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [f32_to_bits(4.0), f32_to_bits(-11.0), f32_to_bits(0.75), ONE])

        # Verify v_mad_f32's OpFMul and OpFAdd carry NoContraction so drivers cannot fuse them into FMA.
        spv = translate_ps([
            f"v_mov_b32 v1, {ONE:#x}",
            f"v_mov_b32 v2, {TWO:#x}",
            f"v_mov_b32 v3, {THREE:#x}",
            "v_mad_f32 v0, v1, v2, v3",
        ])
        fmul_ids = re.findall(r"(%\w+)\s*=\s*OpFMul %float", spv)
        fadd_ids = re.findall(r"(%\w+)\s*=\s*OpFAdd %float", spv)
        self.assertEqual(len(fmul_ids), 1)
        self.assertEqual(len(fadd_ids), 1)
        self.assertIn(f"OpDecorate {fmul_ids[0]} NoContraction", spv)
        self.assertIn(f"OpDecorate {fadd_ids[0]} NoContraction", spv)
        self.assertNotIn("Fma", spv)


def flush_bits(b: int) -> int:
    """RDNA2 flush of v_mad_f32, v_mac_f32 and v_mad_legacy_f32: a subnormal reads as signed zero."""
    b &= MASK32
    return b & 0x80000000 if (b & 0x7FFFFFFF) < 0x00800000 else b


def ref_mad_f32_ftz(xb: int, yb: int, zb: int) -> int:
    """v_mad_f32 with the denormal flush: inputs, the rounded product and the sum flush."""
    prod = flush_bits(_mul_bits_f32(flush_bits(xb), flush_bits(yb)))
    return flush_bits(_add_bits_f32(prod, flush_bits(zb)))


def ref_mad_legacy_f32_ftz(xb: int, yb: int, zb: int) -> int:
    """v_mad_legacy_f32 with the denormal flush, then the legacy 0 * anything -> +0.0 rule."""
    x, y = flush_bits(xb), flush_bits(yb)
    prod = POS_ZERO if (x & 0x7FFFFFFF) == 0 or (y & 0x7FFFFFFF) == 0 else flush_bits(_mul_bits_f32(x, y))
    return flush_bits(_add_bits_f32(prod, flush_bits(zb)))


class DenormalFlushTests(unittest.TestCase):
    """v_mad_f32, v_mac_f32 and v_mad_legacy_f32 have no f32 denormal support on RDNA2: a subnormal operand,
    the rounded product or the result reads as signed zero, whatever the MODE register says (SHADER-118). Source:
    LLVM AMDGPU SIISelLowering.cpp, SITargetLowering::isFMADLegal (llvm-project
    1922c0c9244e41ed056f71e897512c72e24c7022), which only forms these instructions when f32 denormals are
    flushed. The translator emulates that with Gen.flush_f32, so the RDNA2 column below is what it computes.
    9,297 of the 26,503 shaders in shaders.rpf use at least one of the three (440,858 occurrences, 0.99% of the
    corpus), counted with tools/shader_profile.py over the gfx1013 disassembly.
    """

    CASES = [
        # (op, a, b, c, mods, want)  -- c is the addend (v_mad*) or the accumulator (v_mac_f32)
        ("v_mad_f32", 0x17800000, 0x1C800000, SUBNORM_MIN, "", POS_ZERO),    # 2^-80*2^-70 + 2^-149
        ("v_mad_f32", 0x17800000, 0x21800000, 0x00000200, "", POS_ZERO),     # product 2^-140 (subnormal)
        ("v_mad_f32", NEG_SUBNORM_MIN, ONE, NEG_SUBNORM_MIN, "", NEG_ZERO),  # flushed to -0, -0 + -0 = -0
        ("v_mac_f32", SUBNORM_MIN, ONE, SUBNORM_MIN, "", POS_ZERO),          # 2^-149*1 + 2^-149
        ("v_mac_f32", ONE, ONE, SUBNORM_MIN, "", ONE),                       # 1 + 2^-149 = 1
        ("v_mad_legacy_f32", SUBNORM_MIN, POS_INF, NEG_ONE, "", NEG_ONE),    # 2^-149 -> +0, 0*inf -> +0
        ("v_mad_legacy_f32", NEG_SUBNORM_MIN, POS_INF, ONE, "", ONE),        # -2^-149 -> -0, 0*inf -> +0
        ("v_mad_f32", 0x00800000, ONE, POS_ZERO, "div:2", POS_ZERO),        # FLT_MIN/2 flushed after omod
        ("v_mad_f32", 0x00C00000, ONE, NEG_FLT_MIN, "mul:2", POS_ZERO),     # (1.5*2^-126 - 2^-126) flushed before mul:2
        ("v_mad_f32", 0x00C00000, ONE, NEG_FLT_MIN, "mul:4", POS_ZERO),     # (1.5*2^-126 - 2^-126) flushed before mul:4
        ("v_mad_legacy_f32", 0x00C00000, ONE, NEG_FLT_MIN, "mul:2", POS_ZERO),
    ]

    def test_flush_matches_the_rdna2_column(self):
        refs = {"v_mad_f32": ref_mad_f32_ftz, "v_mac_f32": ref_mad_f32_ftz,
                "v_mad_legacy_f32": ref_mad_legacy_f32_ftz}
        for op, a, b, c, mods, want in self.CASES:
            with self.subTest(op=op, a=hex(a), b=hex(b), c=hex(c), mods=mods):
                got = run_three_input(op, a, b, c, mods)
                self.assertEqual(got, want)
                if mods == "":
                    self.assertEqual(refs[op](a, b, c), want)

    def test_normal_values_nan_and_inf_pass_through(self):
        for op in ("v_mad_f32", "v_mac_f32"):
            with self.subTest(op=op):
                self.assertEqual(run_three_input(op, TWO, THREE, FOUR), f32_to_bits(10.0))
                self.assertTrue(is_nan_bits(run_three_input(op, QNAN, ONE, ONE)))
                self.assertTrue(is_nan_bits(run_three_input(op, POS_INF, ONE, NEG_INF)))  # inf - inf
                self.assertEqual(run_three_input(op, NEG_ZERO, TWO, NEG_ZERO), NEG_ZERO)
        # The legacy 0 * anything -> +0 rule still runs on the flushed operands.
        self.assertEqual(run_three_input("v_mad_legacy_f32", TWO, THREE, FOUR), f32_to_bits(10.0))
        self.assertTrue(is_nan_bits(run_three_input("v_mad_legacy_f32", QNAN, ONE, ONE)))
        self.assertTrue(is_nan_bits(run_three_input("v_mad_legacy_f32", POS_INF, ONE, NEG_INF)))
        self.assertEqual(run_three_input("v_mad_legacy_f32", NEG_ZERO, TWO, NEG_ZERO), POS_ZERO)

    def test_denormal_divergence_is_still_detectable(self):
        # Sabotage: turn every flush select into a pass-through; the subnormal cases above now differ.
        cases = [
            ("v_mad_f32", 0x17800000, 0x1C800000, SUBNORM_MIN, POS_ZERO),
            ("v_mac_f32", SUBNORM_MIN, ONE, SUBNORM_MIN, POS_ZERO),
            ("v_mad_legacy_f32", SUBNORM_MIN, POS_INF, NEG_ONE, NEG_ONE),
        ]
        for op, a, b, c, want in cases:
            with self.subTest(op=op):
                if op == "v_mac_f32":
                    lines = [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}",
                             f"v_mov_b32 v0, {c:#x}", f"{op} v0, v1, v2"]
                else:
                    lines = [f"v_mov_b32 v1, {a:#x}", f"v_mov_b32 v2, {b:#x}",
                             f"v_mov_b32 v3, {c:#x}", f"{op} v0, v1, v2, v3"]
                spv = translate_ps(lines)
                self.assertEqual(si.run_fragment(spv)["%out_mrt0"][0], want)
                unflushed, count = re.subn(r"(%\w+) = OpSelect %float (%\w+) (%\w+) (%\w+)",
                                           r"\1 = OpCopyObject %float \4", spv)
                self.assertGreaterEqual(count, 3)
                self.assertNotEqual(si.run_fragment(unflushed)["%out_mrt0"][0], want)


class MadLegacyF32Tests(unittest.TestCase):
    """Semantic tests for v_mad_legacy_f32 (0 * inf/NaN/-0 -> +0.0, then unfused f32 add)."""

    def test_zero_times_inf_and_nan_rules(self):
        cases = [
            # 0 in src0 or src1 (either +0.0 or -0.0) * inf/NaN/0 -> +0.0, then + src2
            (POS_ZERO, POS_INF, NEG_TWO, NEG_TWO),
            (POS_INF, POS_ZERO, NEG_TWO, NEG_TWO),
            (NEG_ZERO, NEG_INF, THREE, THREE),
            (NEG_INF, NEG_ZERO, THREE, THREE),
            (POS_ZERO, NEG_INF, POS_INF, POS_INF),
            (NEG_ZERO, POS_INF, NEG_INF, NEG_INF),
            (POS_ZERO, QNAN, ONE_HALF, ONE_HALF),
            (QNAN, NEG_ZERO, NEG_ONE, NEG_ONE),
            (NEG_ZERO, SNAN, TWO, TWO),
            (SNAN, POS_ZERO, HALF, HALF),
            (POS_ZERO, POS_INF, NEG_ZERO, POS_ZERO),         # +0.0 + (-0.0) = +0.0
            (NEG_ZERO, TWO, NEG_ZERO, POS_ZERO),             # legacy (-0.0)*2 -> +0.0; +0.0 + (-0.0) = +0.0
            (TWO, NEG_ZERO, NEG_ZERO, POS_ZERO),             # legacy 2*(-0.0) -> +0.0; +0.0 + (-0.0) = +0.0
            # Non-zero tiny values must NOT trigger zero-override.
            (FLT_MIN, POS_INF, ONE, POS_INF),
            # Subnormal multipliers: see DenormalFlushTests (RDNA2 flushes them to 0 first).
            # Unfused rounding matches v_mad_f32 when neither multiplier is 0.
            (0x3F800008, 0x3F7FFFF0, NEG_ONE, POS_ZERO),
            (0x71800000, 0x4D800000, 0xFF000000, POS_INF),
        ]
        for a, b, c, want in cases:
            with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                self.assertEqual(ref_mad_legacy_f32(a, b, c), want)
                self.assertEqual(run_three_input("v_mad_legacy_f32", a, b, c), want)

        # Legacy only overrides the product; + z with NaN still yields NaN.
        for a, b, c in ((POS_ZERO, ONE, QNAN), (TWO, THREE, NEG_QNAN), (ONE, QNAN, QNAN)):
            with self.subTest(nan_z=(hex(a), hex(b), hex(c))):
                self.assertTrue(is_nan_bits(ref_mad_legacy_f32(a, b, c)))
                self.assertTrue(is_nan_bits(run_three_input("v_mad_legacy_f32", a, b, c)))

    def test_e64_modifiers_inline_constants_aliasing_and_nocontraction(self):
        words = run_ps_mrt0([
            f"s_mov_b32 s4, {POS_INF:#x}",
            f"v_mov_b32 v1, {POS_ZERO:#x}",
            "v_mad_legacy_f32_e64 v10, -v1, s4, -1.5",                 # (-0)*inf + (-1.5) -> -1.5
            "v_mad_legacy_f32 v11, neg(0), s4, 0.25 mul:4 clamp",      # (+0) + 0.25 -> *4 = 1.0
            "v_mad_legacy_f32 v12, 0, s4, 2.0 div:2",                  # (+0) + 2.0 -> /2 = 1.0
            f"v_mov_b32 v13, {QNAN:#x}",
            "v_mad_legacy_f32 v13, v13, 0, 2.5",                       # dst aliases src0: qNaN * 0 + 2.5 = 2.5
        ], ("v10", "v11", "v12", "v13"))
        self.assertEqual(words, [f32_to_bits(-1.5), ONE, ONE, TWO_HALF])

        spv = translate_ps([
            f"v_mov_b32 v1, {POS_ZERO:#x}",
            f"v_mov_b32 v2, {POS_INF:#x}",
            f"v_mov_b32 v3, {ONE:#x}",
            "v_mad_legacy_f32 v0, v1, v2, v3",
        ])
        fmul_ids = re.findall(r"(%\w+)\s*=\s*OpFMul %float", spv)
        fadd_ids = re.findall(r"(%\w+)\s*=\s*OpFAdd %float", spv)
        self.assertEqual(len(fmul_ids), 1)
        self.assertEqual(len(fadd_ids), 1)
        self.assertIn(f"OpDecorate {fmul_ids[0]} NoContraction", spv)
        self.assertIn(f"OpDecorate {fadd_ids[0]} NoContraction", spv)


class ExecMaskingTests(unittest.TestCase):
    """Inactive lanes under EXEC must leave destination registers untouched across all 5 ops."""

    def test_pixel_shader_inactive_exec_preserves_destination(self):
        sentinel = 0x3E4CCCCD  # 0.2f (safe non-NaN bit pattern)
        insns = [
            "v_min3_f32 v0, v1, v2, v3",
            "v_max3_f32 v0, v1, v2, v3",
            "v_mad_f32 v0, v1, v2, v3",
            "v_fmac_f32_e32 v0, v1, v2",
            "v_fmac_f32_e64 v0, v1, v2",
            "v_mad_legacy_f32 v0, v1, v2, v3",
        ]
        for disable_exec in ("s_mov_b32 exec_lo, 0", "v_cmpx_eq_u32_e32 vcc_lo, 0, 1"):
            for insn in insns:
                with self.subTest(disable=disable_exec, insn=insn):
                    got = run_ps_mrt0([
                        f"v_mov_b32 v0, {sentinel:#x}",
                        f"v_mov_b32 v1, {TWO:#x}",
                        f"v_mov_b32 v2, {THREE:#x}",
                        f"v_mov_b32 v3, {FOUR:#x}",
                        disable_exec,
                        insn,
                        "s_mov_b32 exec_lo, 1",
                    ])[0]
                    self.assertEqual(got, sentinel)

    def test_compute_workgroup_partial_exec_mask(self):
        sentinel = ONE
        cases = [
            ("v_min3_f32 v1, v2, v4, v5", NEG_TWO),
            ("v_max3_f32 v1, v2, v4, v5", FOUR),
            ("v_mad_f32 v1, v2, v4, v5", f32_to_bits(3.0 * (-2.0) + 4.0)),
            ("v_fmac_f32 v1, v2, v4", f32_to_bits(3.0 * (-2.0) + 1.0)),
            ("v_mad_legacy_f32 v1, v2, v4, v5", f32_to_bits(3.0 * (-2.0) + 4.0)),
        ]
        for insn, active_val in cases:
            with self.subTest(insn=insn):
                mem = run_cs_workgroup_exec([
                    "v_lshlrev_b32 v3, 2, v0",
                    f"v_mov_b32 v1, {sentinel:#x}",
                    f"v_mov_b32 v2, {THREE:#x}",
                    f"v_mov_b32 v4, {NEG_TWO:#x}",
                    f"v_mov_b32 v5, {FOUR:#x}",
                    "s_mov_b32 exec_lo, 5",      # lanes 0 and 2 active; lanes 1 and 3 inactive
                    insn,
                    "s_mov_b32 exec_lo, 0xf",
                    "buffer_store_dword v1, v3, s[0:3], 0 offen",
                ])
                self.assertEqual(mem, {0x0: active_val, 0x4: sentinel, 0x8: active_val, 0xC: sentinel})


class NegativeControlsTests(unittest.TestCase):
    """Sabotage translated SPIR-V output and confirm value assertions catch each mutation."""

    def test_sabotaged_spirv_is_detected(self):
        # 1. v_min3_f32 mutated to NMax
        spv_min3 = translate_ps([
            f"v_mov_b32 v1, {ONE:#x}",
            f"v_mov_b32 v2, {TWO:#x}",
            f"v_mov_b32 v3, {THREE:#x}",
            "v_min3_f32 v0, v1, v2, v3",
        ])
        self.assertEqual(si.run_fragment(spv_min3)["%out_mrt0"][0], ONE)
        self.assertNotEqual(si.run_fragment(spv_min3.replace("glsl NMin", "glsl NMax"))["%out_mrt0"][0], ONE)

        # 2. v_max3_f32 mutated to NMin
        spv_max3 = translate_ps([
            f"v_mov_b32 v1, {ONE:#x}",
            f"v_mov_b32 v2, {TWO:#x}",
            f"v_mov_b32 v3, {THREE:#x}",
            "v_max3_f32 v0, v1, v2, v3",
        ])
        self.assertEqual(si.run_fragment(spv_max3)["%out_mrt0"][0], THREE)
        self.assertNotEqual(si.run_fragment(spv_max3.replace("glsl NMax", "glsl NMin"))["%out_mrt0"][0], THREE)

        # 3. v_mad_f32 OpFAdd mutated to OpFSub
        spv_mad = translate_ps([
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v2, {THREE:#x}",
            f"v_mov_b32 v3, {FOUR:#x}",
            "v_mad_f32 v0, v1, v2, v3",
        ])
        want_mad = f32_to_bits(10.0)
        self.assertEqual(si.run_fragment(spv_mad)["%out_mrt0"][0], want_mad)
        self.assertNotEqual(si.run_fragment(spv_mad.replace("OpFAdd %float", "OpFSub %float"))["%out_mrt0"][0], want_mad)

        # 4. v_fmac_f32 Fma replaced with unfused OpFMul + OpFAdd fails on intermediate cancellation case
        spv_fmac = translate_ps([
            "v_mov_b32 v1, 0x3f800008",
            "v_mov_b32 v2, 0x3f7ffff0",
            f"v_mov_b32 v0, {NEG_ONE:#x}",
            "v_fmac_f32 v0, v1, v2",
        ])
        want_fmac = 0xAB800000
        self.assertEqual(si.run_fragment(spv_fmac)["%out_mrt0"][0], want_fmac)
        bad_fmac, count = re.subn(
            r"(%\w+) = OpExtInst %float %glsl Fma (%\w+) (%\w+) (%\w+)",
            r"%unfused_m = OpFMul %float \2 \3\n         \1 = OpFAdd %float %unfused_m \4",
            spv_fmac,
        )
        self.assertEqual(count, 1)
        self.assertNotEqual(si.run_fragment(bad_fmac)["%out_mrt0"][0], want_fmac)

        # 5. v_mad_legacy_f32 zero check weakened from LogicalOr to LogicalAnd breaks 0 * inf + 2.0
        spv_leg = translate_ps([
            f"v_mov_b32 v1, {POS_ZERO:#x}",
            f"v_mov_b32 v2, {POS_INF:#x}",
            f"v_mov_b32 v3, {TWO:#x}",
            "v_mad_legacy_f32 v0, v1, v2, v3",
        ])
        self.assertEqual(si.run_fragment(spv_leg)["%out_mrt0"][0], TWO)
        bad_leg = spv_leg.replace("OpLogicalOr %bool", "OpLogicalAnd %bool")
        self.assertTrue(is_nan_bits(si.run_fragment(bad_leg)["%out_mrt0"][0]))

        # 6. clamp output modifier dropped
        spv_clamp = translate_ps([
            f"v_mov_b32 v1, {TWO:#x}",
            f"v_mov_b32 v2, {THREE:#x}",
            f"v_mov_b32 v3, {FOUR:#x}",
            "v_mad_f32 v0, v1, v2, v3 clamp",
        ])
        self.assertEqual(si.run_fragment(spv_clamp)["%out_mrt0"][0], ONE)
        bad_clamp, clamp_cnt = re.subn(
            r"(%\w+) = OpExtInst %float %glsl FClamp (%\w+) %f0 %f1",
            r"\1 = OpCopyObject %float \2",
            spv_clamp,
        )
        self.assertEqual(clamp_cnt, 1)
        self.assertNotEqual(si.run_fragment(bad_clamp)["%out_mrt0"][0], ONE)


class MacLegacyF32Tests(unittest.TestCase):
    """v_mac_legacy_f32 (GFX10.1 VOP2): v_mad_legacy_f32 with D as the addend, D = legacy(S0 * S1) + D, with the same
    denormal flush. Until 2026-11-22 the translator rejected it."""

    VALUES = [POS_ZERO, NEG_ZERO, SUBNORM_MIN, NEG_SUBNORM_MIN, FLT_MIN, HALF, ONE, NEG_ONE, TWO, NEG_TWO, THREE,
              FLT_MAX, POS_INF, NEG_INF, QNAN]

    def test_matches_mad_legacy_reference(self):
        for a in self.VALUES:
            for b in self.VALUES[::2]:
                for c in (POS_ZERO, NEG_ZERO, NEG_ONE, TWO_HALF, POS_INF, SUBNORM_MIN):
                    with self.subTest(a=hex(a), b=hex(b), c=hex(c)):
                        want = ref_mad_legacy_f32_ftz(a, b, c)
                        got = run_three_input("v_mac_legacy_f32", a, b, c)
                        if is_nan_bits(want):
                            self.assertTrue(is_nan_bits(got), hex(got))
                        else:
                            self.assertEqual(got, want)

    def test_zero_times_inf_is_zero_and_d_is_the_addend(self):
        self.assertEqual(run_three_input("v_mac_legacy_f32", POS_ZERO, POS_INF, NEG_TWO), NEG_TWO)
        self.assertEqual(run_three_input("v_mac_legacy_f32", QNAN, NEG_ZERO, THREE), THREE)
        self.assertEqual(run_three_input("v_mac_legacy_f32", TWO, THREE, FOUR), f32_to_bits(10.0))
        self.assertEqual(run_three_input("v_mac_legacy_f32", NEG_ZERO, TWO, NEG_ZERO), POS_ZERO)

    def test_e64_modifiers_and_accumulator_aliasing(self):
        # VOP3 form: neg on a source, clamp on the result.
        self.assertEqual(run_three_input("v_mac_legacy_f32_e64", TWO, THREE, FOUR, "clamp"), ONE)
        lines = [f"v_mov_b32 v1, {TWO:#x}", f"v_mov_b32 v2, {THREE:#x}", f"v_mov_b32 v0, {FOUR:#x}",
                 "v_mac_legacy_f32_e64 v0, -v1, v2"]
        self.assertEqual(run_ps_mrt0(lines)[0], NEG_TWO)                        # -2*3 + 4
        # D read as src0 and as the addend: 3*3 + 3 = 12; chained: 12*2 + 12 = 36.
        lines = [f"v_mov_b32 v0, {THREE:#x}", f"v_mov_b32 v2, {TWO:#x}",
                 "v_mac_legacy_f32 v0, v0, v0", "v_mac_legacy_f32 v0, v0, v2"]
        self.assertEqual(run_ps_mrt0(lines)[0], f32_to_bits(36.0))


if __name__ == "__main__":
    unittest.main()
