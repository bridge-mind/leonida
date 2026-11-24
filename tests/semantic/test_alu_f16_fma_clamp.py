"""Semantic interpreter tests for RDNA2 half-precision FMA with IEEE clamp saturation:
v_fma_f16 (VOP3 scalar form) and v_pk_fma_f16 (VOP3P packed SIMD form) with clamp.

Features and specifications:
- Exact half-precision (IEEE 754 binary16) FMA oracle using Fraction-based arithmetic.
- IEEE clamp saturation: output is clamped to [0.0, 1.0].
  Values < 0.0 and -0.0 clamp to +0.0 (0x0000).
  Values > 1.0 clamp to +1.0 (0x3C00).
  Values in [0.0, 1.0] are preserved exactly as rounded to binary16.
- Destination register half preservation on scalar f16 ops: GFX10 f16 ops keep the untouched half.
- VOP3 op_sel destination bit and input modifiers (negation, absolute value).
- Standard library Python only; zero game data.
"""
from __future__ import annotations

import gc
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings
from fractions import Fraction

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
from support import STUB_DXBC  # noqa: E402

MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
OLD = 0xA55A5AA5
VSHARP_CS = [0, 0, 0xFFFFFFFF, 0x31016FAC]

# IEEE 754 binary16 constants
POS_ZERO = 0x0000          # +0.0
NEG_ZERO = 0x8000          # -0.0
POS_ONE = 0x3C00           # +1.0
NEG_ONE = 0xBC00           # -1.0
POS_HALF = 0x3800          # +0.5
NEG_HALF = 0xB800          # -0.5
POS_QUARTER = 0x3400       # +0.25
NEG_QUARTER = 0xB400       # -0.25
POS_THREE_QUARTERS = 0x3A00 # +0.75
NEG_THREE_QUARTERS = 0xBA00 # -0.75
POS_TWO = 0x4000           # +2.0
NEG_TWO = 0xC000           # -2.0
POS_THREE = 0x4200         # +3.0
NEG_THREE = 0xC200         # -3.0
POS_FOUR = 0x4400          # +4.0
POS_INF = 0x7C00           # +inf
NEG_INF = 0xFC00           # -inf
QNAN_CANONICAL = 0x7E00    # canonical quiet NaN
QNAN_ALT = 0x7E11          # quiet NaN with payload
SNAN = 0x7D00              # signaling NaN
NEG_QNAN = 0xFE22          # negative quiet NaN

# Subnormal binary16 constants
POS_MIN_SUBNORM = 0x0001   # +2^-24 (~5.96e-8), smallest positive subnormal
POS_MAX_SUBNORM = 0x03FF   # largest positive subnormal (~6.097e-5)
NEG_MIN_SUBNORM = 0x8001   # -2^-24, smallest negative subnormal
NEG_MAX_SUBNORM = 0x83FF   # largest negative subnormal

# Normal binary16 boundaries
POS_MIN_NORMAL = 0x0400    # +2^-14 (~6.104e-5), smallest positive normal
POS_MAX_NORMAL = 0x7BFF    # +65504.0, largest finite binary16
NEG_MAX_NORMAL = 0xFBFF    # -65504.0


def f16_to_float(bits: int) -> float:
    """Decode 16-bit integer as IEEE 754 half-precision float."""
    return struct.unpack("<e", struct.pack("<H", bits & MASK16))[0]


def float_to_f16_bits(f: float) -> int:
    """Encode float into IEEE 754 half-precision bits."""
    if math.isnan(f):
        return QNAN_CANONICAL
    try:
        return struct.unpack("<H", struct.pack("<e", f))[0]
    except OverflowError:
        return POS_INF if f > 0 else NEG_INF


def is_nan16(bits: int) -> bool:
    """Check if 16-bit integer represents a binary16 NaN."""
    bits &= MASK16
    return (bits & 0x7C00) == 0x7C00 and (bits & 0x03FF) != 0


def is_inf16(bits: int) -> bool:
    """Check if 16-bit integer represents a binary16 infinity."""
    bits &= MASK16
    return (bits & 0x7FFF) == 0x7C00


def is_subnormal16(bits: int) -> bool:
    """Check if 16-bit integer represents a binary16 subnormal."""
    bits &= MASK16
    return (bits & 0x7C00) == 0 and (bits & 0x03FF) != 0


def f16_bits_to_fraction(bits: int) -> Fraction:
    """Decode finite binary16 bit pattern into exact Fraction."""
    bits &= MASK16
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x03FF
    sign = -1 if (bits & 0x8000) else 1
    if exp == 0:
        if mant == 0:
            return Fraction(0)
        return Fraction(sign * mant, 1 << 24)
    return Fraction(sign * ((1 << 10) | mant)) * (Fraction(2) ** (exp - 25))


def fraction_to_f16(x: Fraction, *, flush_denorms: bool = False) -> int:
    """Round an exact Fraction to IEEE 754 binary16 bits (round to nearest even)."""
    if x == 0:
        return POS_ZERO
    sign = NEG_ZERO if x < 0 else POS_ZERO
    x = abs(x)

    if flush_denorms and x < Fraction(1, 1 << 14):
        return sign

    lo, hi = 0, 0x7C00
    while lo < hi:
        mid = (lo + hi) // 2
        val = Fraction(f16_to_float(mid)) if mid < 0x7C00 else Fraction(10 ** 9)
        if val >= x:
            hi = mid
        else:
            lo = mid + 1

    if lo == 0x7C00:
        return sign | (POS_INF if x >= 65520 else POS_MAX_NORMAL)

    above = lo
    if lo == 0 or Fraction(f16_to_float(above)) == x:
        return sign | above

    below = lo - 1
    da = Fraction(f16_to_float(above)) - x
    db = x - Fraction(f16_to_float(below))
    if da < db:
        pick = above
    elif db < da:
        pick = below
    else:
        pick = above if (above % 2 == 0) else below
    return sign | pick


def apply_src_mods(bits: int, *, abs_: bool = False, neg: bool = False) -> int:
    """Apply VOP3 source modifiers (abs applied before neg)."""
    bits &= MASK16
    if abs_:
        bits &= 0x7FFF
    if neg:
        bits ^= 0x8000
    return bits


def ref_fma_f16(
    a_bits: int,
    b_bits: int,
    c_bits: int,
    *,
    clamp: bool = False,
    abs_s0: bool = False,
    neg_s0: bool = False,
    abs_s1: bool = False,
    neg_s1: bool = False,
    abs_s2: bool = False,
    neg_s2: bool = False,
    flush_denorms: bool = False,
) -> int:
    """Independent reference oracle for half-precision FMA (a * b + c) with clamp saturation.

    Implements exact IEEE 754 binary16 arithmetic, round to nearest even, and
    optional clamp saturation to [0.0, 1.0].
    """
    a = apply_src_mods(a_bits, abs_=abs_s0, neg=neg_s0)
    b = apply_src_mods(b_bits, abs_=abs_s1, neg=neg_s1)
    c = apply_src_mods(c_bits, abs_=abs_s2, neg=neg_s2)

    # NaN operands
    if is_nan16(a) or is_nan16(b) or is_nan16(c):
        return QNAN_CANONICAL

    a_is_zero = (a & 0x7FFF) == 0
    b_is_zero = (b & 0x7FFF) == 0
    c_is_zero = (c & 0x7FFF) == 0

    a_is_inf = is_inf16(a)
    b_is_inf = is_inf16(b)
    c_is_inf = is_inf16(c)

    # 0 * inf or inf * 0 is invalid -> NaN
    if (a_is_zero and b_is_inf) or (a_is_inf and b_is_zero):
        return QNAN_CANONICAL

    prod_sign = (a ^ b) & 0x8000

    # Infinite product
    if a_is_inf or b_is_inf:
        if c_is_inf:
            c_sign = c & 0x8000
            if prod_sign != c_sign:
                # (+inf) + (-inf) -> NaN
                return QNAN_CANONICAL
        res = prod_sign | POS_INF
        if clamp:
            return POS_ZERO if prod_sign else POS_ONE
        return res

    # Finite product with infinite addend
    if c_is_inf:
        c_sign = c & 0x8000
        if clamp:
            return POS_ZERO if c_sign else POS_ONE
        return c

    # Denormal input flush if requested
    if flush_denorms:
        if is_subnormal16(a):
            a = a & 0x8000
            a_is_zero = True
        if is_subnormal16(b):
            b = b & 0x8000
            b_is_zero = True
        if is_subnormal16(c):
            c = c & 0x8000
            c_is_zero = True

    # Exact product is zero
    if a_is_zero or b_is_zero:
        if c_is_zero:
            c_sign = c & 0x8000
            # -0 + -0 = -0; all other combinations in RNE produce +0
            res = NEG_ZERO if (prod_sign and c_sign) else POS_ZERO
            if clamp:
                return POS_ZERO
            return res
        # Product is zero, result is c
        res = c
        if clamp:
            if (res & 0x8000) and (res != NEG_ZERO):
                return POS_ZERO
            if res == NEG_ZERO:
                return POS_ZERO
            # If c > 1.0 -> clamp to 1.0
            if f16_bits_to_fraction(c) > Fraction(1):
                return POS_ONE
            if f16_bits_to_fraction(c) < Fraction(0):
                return POS_ZERO
            return res
        return res

    # Finite non-zero operands: compute exact rational a * b + c
    qa = f16_bits_to_fraction(a)
    qb = f16_bits_to_fraction(b)
    qc = f16_bits_to_fraction(c)
    exact = qa * qb + qc

    if exact == 0:
        # Exact cancellation to zero produces +0.0 in round-to-nearest
        res = POS_ZERO
    else:
        res = fraction_to_f16(exact, flush_denorms=flush_denorms)

    if clamp:
        if (res & 0x8000) != 0:
            return POS_ZERO
        if is_inf16(res) or f16_bits_to_fraction(res) > Fraction(1):
            return POS_ONE
        return res

    return res


def merge_half(val: int, dst_hi: int, old: int = OLD) -> int:
    """Merge a 16-bit result into low or high half of 32-bit register, keeping untouched half."""
    val &= MASK16
    return (val << 16) | (old & 0xFFFF) if dst_hi else (old & 0xFFFF0000) | val


def pack2(lo: int, hi: int) -> int:
    """Pack two 16-bit half-precision values into a 32-bit dword."""
    return ((hi & MASK16) << 16) | (lo & MASK16)


def ref_pk_fma_f16(
    v1: int,
    v2: int,
    v3: int,
    *,
    clamp: bool = False,
    op_sel: tuple[int, int, int] = (0, 0, 0),
    op_sel_hi: tuple[int, int, int] = (1, 1, 1),
    neg_lo: tuple[int, int, int] = (0, 0, 0),
    neg_hi: tuple[int, int, int] = (0, 0, 0),
    flush_denorms: bool = False,
) -> int:
    """Reference oracle for packed half-precision SIMD FMA (v_pk_fma_f16).

    RDNA2 VOP3P packed half-precision FMA:
    Result lo: source i takes its half from op_sel[i] (0 = lo, 1 = hi), negated by neg_lo[i].
    Result hi: source i takes its half from op_sel_hi[i] (0 = lo, 1 = hi), negated by neg_hi[i].
    Both halves independently evaluate FMA and optional clamp.
    """
    def extract_half(val: int, half_sel: int, neg: int) -> int:
        h = (val >> 16) if half_sel else (val & MASK16)
        return apply_src_mods(h, neg=bool(neg))

    a_lo = extract_half(v1, op_sel[0], neg_lo[0])
    b_lo = extract_half(v2, op_sel[1], neg_lo[1])
    c_lo = extract_half(v3, op_sel[2], neg_lo[2])
    res_lo = ref_fma_f16(a_lo, b_lo, c_lo, clamp=clamp, flush_denorms=flush_denorms)

    a_hi = extract_half(v1, op_sel_hi[0], neg_hi[0])
    b_hi = extract_half(v2, op_sel_hi[1], neg_hi[1])
    c_hi = extract_half(v3, op_sel_hi[2], neg_hi[2])
    res_hi = ref_fma_f16(a_hi, b_hi, c_hi, clamp=clamp, flush_denorms=flush_denorms)

    return pack2(res_lo, res_hi)


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
    _PS_PATH = os.path.join(disasm_dir, "ps_alu_f16_fma.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_alu_f16_fma.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_alu_f16_fma.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_alu_f16_fma.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "cs_alu_f16_fma.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines: list[str]) -> str:
    if _TMP_DIR is None:
        setUpModule()
    asm = "; test_alu_f16_fma_clamp\n" + "".join(f"\t{ln}\n" for ln in lines) + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    with open(_PS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_PS_PATH)
        gc.collect()
        return spv


def run_ps(lines: list[str]) -> int:
    spv = translate_ps(lines)
    return si.run_fragment(spv)["%out_mrt0"][0]


def run_op(
    op_line: str,
    v1: int = 0,
    v2: int = 0,
    v3: int = 0,
    *,
    old: int = OLD,
    s4: int = 0,
    s5: int = 0,
    s6: int = 0,
) -> int:
    lines = [
        f"v_mov_b32 v0, {old:#x}",
        f"v_mov_b32 v1, {v1:#x}",
        f"v_mov_b32 v2, {v2:#x}",
        f"v_mov_b32 v3, {v3:#x}",
        f"s_mov_b32 s4, {s4:#x}",
        f"s_mov_b32 s5, {s5:#x}",
        f"s_mov_b32 s6, {s6:#x}",
        op_line,
    ]
    return run_ps(lines)


# -----------------------------------------------------------------------------
# Commit 1 Tests: Reference Oracle and Scalar v_fma_f16 with Clamp Saturation
# -----------------------------------------------------------------------------

class ReferenceOracleTests(unittest.TestCase):
    """Test unit validation of the exact Fraction-based half-precision FMA oracle."""

    def test_oracle_basic_fma(self):
        # 2.0 * 2.0 + 2.0 = 6.0 (0x4600 unclamped; clamped to 1.0 = 0x3C00)
        self.assertEqual(ref_fma_f16(POS_TWO, POS_TWO, POS_TWO, clamp=False), 0x4600)
        self.assertEqual(ref_fma_f16(POS_TWO, POS_TWO, POS_TWO, clamp=True), POS_ONE)

        # 0.5 * 0.5 + 0.25 = 0.5 (0x3800, inside [0, 1])
        self.assertEqual(ref_fma_f16(POS_HALF, POS_HALF, POS_QUARTER, clamp=False), POS_HALF)
        self.assertEqual(ref_fma_f16(POS_HALF, POS_HALF, POS_QUARTER, clamp=True), POS_HALF)

        # -2.0 * 1.0 + 0.5 = -1.5 (0xBE00 unclamped; clamped to +0.0 = 0x0000)
        self.assertEqual(ref_fma_f16(NEG_TWO, POS_ONE, POS_HALF, clamp=False), 0xBE00)
        self.assertEqual(ref_fma_f16(NEG_TWO, POS_ONE, POS_HALF, clamp=True), POS_ZERO)

    def test_oracle_clamp_edges(self):
        # 1.0 * 1.0 + 0.0 = 1.0 (exact upper clamp boundary)
        self.assertEqual(ref_fma_f16(POS_ONE, POS_ONE, POS_ZERO, clamp=True), POS_ONE)

        # 0.0 * 0.0 + 0.0 = 0.0 (exact lower clamp boundary)
        self.assertEqual(ref_fma_f16(POS_ZERO, POS_ZERO, POS_ZERO, clamp=True), POS_ZERO)

        # Large positive overflow: 65504 * 2.0 + 0 = +inf (clamped to 1.0)
        self.assertEqual(ref_fma_f16(POS_MAX_NORMAL, POS_TWO, POS_ZERO, clamp=False), POS_INF)
        self.assertEqual(ref_fma_f16(POS_MAX_NORMAL, POS_TWO, POS_ZERO, clamp=True), POS_ONE)

        # Large negative overflow: -65504 * 2.0 + 0 = -inf (clamped to +0.0)
        self.assertEqual(ref_fma_f16(NEG_MAX_NORMAL, POS_TWO, POS_ZERO, clamp=False), NEG_INF)
        self.assertEqual(ref_fma_f16(NEG_MAX_NORMAL, POS_TWO, POS_ZERO, clamp=True), POS_ZERO)

    def test_oracle_source_modifiers(self):
        # -v1 * v2 + v3: - (1.0) * 0.5 + 0.75 = 0.25 (0x3400)
        got = ref_fma_f16(POS_ONE, POS_HALF, POS_THREE_QUARTERS, neg_s0=True, clamp=True)
        self.assertEqual(got, POS_QUARTER)

        # | -2.0 | * 0.5 + 0.0 = 1.0
        got_abs = ref_fma_f16(NEG_TWO, POS_HALF, POS_ZERO, abs_s0=True, clamp=True)
        self.assertEqual(got_abs, POS_ONE)


class ScalarFmaClampTests(unittest.TestCase):
    """End-to-end tests for scalar v_fma_f16 with IEEE clamp saturation."""

    def test_scalar_fma_clamp_saturation_upper(self):
        # 2.0 * 2.0 + 2.0 = 6.0 > 1.0 -> saturates to 1.0 (0x3C00)
        # Low half of v0 gets 0x3C00, high half preserved (OLD & 0xFFFF0000)
        got = run_op("v_fma_f16 v0, v1, v2, v3 clamp", v1=POS_TWO, v2=POS_TWO, v3=POS_TWO)
        expected = merge_half(POS_ONE, dst_hi=0)
        self.assertEqual(got, expected)

    def test_scalar_fma_clamp_saturation_lower(self):
        # 2.0 * (-1.0) + 0.5 = -1.5 < 0.0 -> saturates to 0.0 (0x0000)
        got = run_op("v_fma_f16 v0, v1, v2, v3 clamp", v1=POS_TWO, v2=NEG_ONE, v3=POS_HALF)
        expected = merge_half(POS_ZERO, dst_hi=0)
        self.assertEqual(got, expected)

    def test_scalar_fma_interior_values(self):
        # 0.5 * 0.5 + 0.25 = 0.5 in [0.0, 1.0] -> 0x3800
        got = run_op("v_fma_f16 v0, v1, v2, v3 clamp", v1=POS_HALF, v2=POS_HALF, v3=POS_QUARTER)
        expected = merge_half(POS_HALF, dst_hi=0)
        self.assertEqual(got, expected)

        # 0.75 * 0.5 + 0.125 (0x3000) = 0.5
        got2 = run_op("v_fma_f16 v0, v1, v2, v3 clamp", v1=POS_THREE_QUARTERS, v2=POS_HALF, v3=0x3000)
        self.assertEqual(got2, merge_half(POS_HALF, dst_hi=0))

    def test_scalar_fma_exact_boundaries(self):
        # 1.0 * 1.0 + 0.0 = 1.0 (0x3C00)
        got1 = run_op("v_fma_f16 v0, v1, v2, v3 clamp", v1=POS_ONE, v2=POS_ONE, v3=POS_ZERO)
        self.assertEqual(got1, merge_half(POS_ONE, dst_hi=0))

        # 0.0 * 1.0 + 0.0 = 0.0 (0x0000)
        got0 = run_op("v_fma_f16 v0, v1, v2, v3 clamp", v1=POS_ZERO, v2=POS_ONE, v3=POS_ZERO)
        self.assertEqual(got0, merge_half(POS_ZERO, dst_hi=0))

    def test_scalar_fma_destination_high_half_preservation(self):
        # op_sel:[0,0,0,1] writes high half of v0, keeping low half of v0
        got = run_op("v_fma_f16 v0, v1, v2, v3 clamp op_sel:[0,0,0,1]", v1=POS_TWO, v2=POS_TWO, v3=POS_TWO)
        expected = merge_half(POS_ONE, dst_hi=1)
        self.assertEqual(got, expected)

    def test_scalar_fma_source_modifiers(self):
        # -v1 * v2 + v3: -(1.0) * 0.5 + 0.75 = 0.25 (0x3400)
        got = run_op("v_fma_f16 v0, -v1, v2, v3 clamp", v1=POS_ONE, v2=POS_HALF, v3=POS_THREE_QUARTERS)
        self.assertEqual(got, merge_half(POS_QUARTER, dst_hi=0))

        # |-v1| * v2 + v3: | -2.0 | * 0.25 + 0.25 = 0.75 (0x3A00)
        got_abs = run_op("v_fma_f16 v0, |v1|, v2, v3 clamp", v1=NEG_TWO, v2=POS_QUARTER, v3=POS_QUARTER)
        self.assertEqual(got_abs, merge_half(POS_THREE_QUARTERS, dst_hi=0))

    def test_scalar_fma_sgpr_and_inline_constants(self):
        # Inline constants 0.5 (0x3800), 1.0 (0x3C00), and SGPR source
        # s4 = 0.5: 0.5 * 1.0 + 0.5 = 1.0 (0x3C00)
        got = run_op("v_fma_f16 v0, s4, 1.0, 0.5 clamp", s4=POS_HALF)
        self.assertEqual(got, merge_half(POS_ONE, dst_hi=0))


# -----------------------------------------------------------------------------
# Commit 2 Tests: Packed Half-Precision v_pk_fma_f16 SIMD Saturation Tests
# -----------------------------------------------------------------------------

class PackedReferenceOracleTests(unittest.TestCase):
    """Test unit validation of the packed half-precision SIMD FMA oracle."""

    def test_packed_oracle_basic(self):
        # lo: 2.0 * 2.0 + 2.0 = 6.0 -> clamped to 1.0 (0x3C00)
        # hi: 0.5 * 0.5 + 0.25 = 0.5 (0x3800)
        v1 = pack2(POS_TWO, POS_HALF)
        v2 = pack2(POS_TWO, POS_HALF)
        v3 = pack2(POS_TWO, POS_QUARTER)
        self.assertEqual(ref_pk_fma_f16(v1, v2, v3, clamp=True), pack2(POS_ONE, POS_HALF))
        self.assertEqual(ref_pk_fma_f16(v1, v2, v3, clamp=False), pack2(0x4600, POS_HALF))

    def test_packed_oracle_swizzle_and_negation(self):
        # v1: lo=2.0, hi=0.5
        # v2: lo=1.0, hi=-1.0
        # v3: lo=0.25, hi=0.75
        v1 = pack2(POS_TWO, POS_HALF)
        v2 = pack2(POS_ONE, NEG_ONE)
        v3 = pack2(POS_QUARTER, POS_THREE_QUARTERS)

        # op_sel:(1, 0, 0): lo takes v1.hi (0.5), v2.lo (1.0), v3.lo (0.25) -> 0.5 * 1.0 + 0.25 = 0.75
        # op_sel_hi:(0, 1, 1): hi takes v1.lo (2.0), v2.hi (-1.0), v3.hi (0.75) -> 2.0 * (-1.0) + 0.75 = -1.25 -> clamped to 0.0
        got = ref_pk_fma_f16(v1, v2, v3, clamp=True, op_sel=(1, 0, 0), op_sel_hi=(0, 1, 1))
        self.assertEqual(got, pack2(POS_THREE_QUARTERS, POS_ZERO))

        # neg_lo:(1, 0, 0): lo takes -v1.lo (-2.0), v2.lo (1.0), v3.lo (0.25) -> -1.75 -> clamped to 0.0
        got_neg = ref_pk_fma_f16(v1, v2, v3, clamp=True, neg_lo=(1, 0, 0))
        self.assertEqual(got_neg & MASK16, POS_ZERO)

    def test_packed_oracle_independent_clamp(self):
        # lo clamps to 0.0; hi clamps to 1.0
        v1 = pack2(NEG_TWO, POS_TWO)
        v2 = pack2(POS_ONE, POS_TWO)
        v3 = pack2(POS_HALF, POS_TWO)
        self.assertEqual(ref_pk_fma_f16(v1, v2, v3, clamp=True), pack2(POS_ZERO, POS_ONE))


class PackedFmaF16ClampTests(unittest.TestCase):
    """End-to-end tests for packed v_pk_fma_f16 SIMD execution with IEEE clamp saturation."""

    def test_packed_clamp_both_lanes_saturating(self):
        # lo: 2.0 * (-1.0) + 0.5 = -1.5 -> clamped to 0.0 (0x0000)
        # hi: 2.0 * 2.0 + 2.0 = 6.0 -> clamped to 1.0 (0x3C00)
        v1 = pack2(POS_TWO, POS_TWO)
        v2 = pack2(NEG_ONE, POS_TWO)
        v3 = pack2(POS_HALF, POS_TWO)
        got = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=v1, v2=v2, v3=v3)
        self.assertEqual(got, pack2(POS_ZERO, POS_ONE))

        # Inverted saturation: lo clamps to 1.0, hi clamps to 0.0
        v1_inv = pack2(POS_TWO, POS_TWO)
        v2_inv = pack2(POS_TWO, NEG_ONE)
        v3_inv = pack2(POS_TWO, POS_HALF)
        got_inv = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=v1_inv, v2=v2_inv, v3=v3_inv)
        self.assertEqual(got_inv, pack2(POS_ONE, POS_ZERO))

    def test_packed_clamp_one_lane_interior_one_saturating(self):
        # lo interior (0.5 * 0.5 + 0.25 = 0.5), hi saturates upper (2.0 * 2.0 + 2.0 = 6.0 -> 1.0)
        v1 = pack2(POS_HALF, POS_TWO)
        v2 = pack2(POS_HALF, POS_TWO)
        v3 = pack2(POS_QUARTER, POS_TWO)
        got1 = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=v1, v2=v2, v3=v3)
        self.assertEqual(got1, pack2(POS_HALF, POS_ONE))

        # lo interior (0.5 * 0.5 + 0.25 = 0.5), hi saturates lower (2.0 * -1.0 + 0.5 = -1.5 -> 0.0)
        v1_b = pack2(POS_HALF, POS_TWO)
        v2_b = pack2(POS_HALF, NEG_ONE)
        v3_b = pack2(POS_QUARTER, POS_HALF)
        got2 = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=v1_b, v2=v2_b, v3=v3_b)
        self.assertEqual(got2, pack2(POS_HALF, POS_ZERO))

        # lo saturates upper (1.0), hi interior (0.5)
        got3 = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=pack2(POS_TWO, POS_HALF), v2=pack2(POS_TWO, POS_HALF), v3=pack2(POS_TWO, POS_QUARTER))
        self.assertEqual(got3, pack2(POS_ONE, POS_HALF))

        # lo saturates lower (0.0), hi interior (0.5)
        got4 = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=pack2(POS_TWO, POS_HALF), v2=pack2(NEG_ONE, POS_HALF), v3=pack2(POS_HALF, POS_QUARTER))
        self.assertEqual(got4, pack2(POS_ZERO, POS_HALF))

    def test_packed_clamp_both_lanes_interior(self):
        # lo: 0.5 * 0.5 + 0.0 = 0.25 (0x3400)
        # hi: 0.75 * 0.5 + 0.125 = 0.5 (0x3800)
        v1 = pack2(POS_HALF, POS_THREE_QUARTERS)
        v2 = pack2(POS_HALF, POS_HALF)
        v3 = pack2(POS_ZERO, 0x3000)
        got = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=v1, v2=v2, v3=v3)
        self.assertEqual(got, pack2(POS_QUARTER, POS_HALF))

    def test_packed_clamp_both_lanes_same_saturation(self):
        # Both clamp to 0.0
        v1 = pack2(NEG_TWO, NEG_THREE)
        v2 = pack2(POS_ONE, POS_ONE)
        v3 = pack2(POS_HALF, POS_HALF)
        got_zero = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=v1, v2=v2, v3=v3)
        self.assertEqual(got_zero, pack2(POS_ZERO, POS_ZERO))

        # Both clamp to 1.0
        v1_h = pack2(POS_TWO, POS_THREE)
        v2_h = pack2(POS_TWO, POS_TWO)
        v3_h = pack2(POS_ONE, POS_ONE)
        got_one = run_op("v_pk_fma_f16 v0, v1, v2, v3 clamp", v1=v1_h, v2=v2_h, v3=v3_h)
        self.assertEqual(got_one, pack2(POS_ONE, POS_ONE))

    def test_packed_fma_unclamped_exceeds_range(self):
        # Without clamp, results must not saturate:
        # lo: 2.0 * 2.0 + 2.0 = 6.0 (0x4600)
        # hi: 2.0 * (-1.0) + 0.5 = -1.5 (0xBE00)
        v1 = pack2(POS_TWO, POS_TWO)
        v2 = pack2(POS_TWO, NEG_ONE)
        v3 = pack2(POS_TWO, POS_HALF)
        got = run_op("v_pk_fma_f16 v0, v1, v2, v3", v1=v1, v2=v2, v3=v3)
        self.assertEqual(got, pack2(0x4600, 0xBE00))

    def test_packed_fma_opsel_modifiers(self):
        # Swizzling with op_sel and op_sel_hi under clamp
        v1 = pack2(POS_TWO, POS_HALF)
        v2 = pack2(POS_ONE, NEG_ONE)
        v3 = pack2(POS_QUARTER, POS_THREE_QUARTERS)
        got = run_op(
            "v_pk_fma_f16 v0, v1, v2, v3 clamp op_sel:[1,0,0] op_sel_hi:[0,1,1]",
            v1=v1,
            v2=v2,
            v3=v3,
        )
        expected = ref_pk_fma_f16(v1, v2, v3, clamp=True, op_sel=(1, 0, 0), op_sel_hi=(0, 1, 1))
        self.assertEqual(got, expected)

    def test_packed_fma_neg_modifiers(self):
        # Negation modifiers neg_lo and neg_hi under clamp
        v1 = pack2(POS_ONE, POS_TWO)
        v2 = pack2(POS_HALF, POS_HALF)
        v3 = pack2(POS_THREE_QUARTERS, POS_QUARTER)
        got = run_op(
            "v_pk_fma_f16 v0, v1, v2, v3 clamp neg_lo:[1,0,0] neg_hi:[0,1,0]",
            v1=v1,
            v2=v2,
            v3=v3,
        )
        expected = ref_pk_fma_f16(v1, v2, v3, clamp=True, neg_lo=(1, 0, 0), neg_hi=(0, 1, 0))
        self.assertEqual(got, expected)

    def test_packed_fma_inactive_exec_lane(self):
        # Inactive EXEC lane must preserve existing destination contents
        lines = [
            f"v_mov_b32 v0, {OLD:#x}",
            f"v_mov_b32 v1, {pack2(POS_TWO, POS_TWO):#x}",
            f"v_mov_b32 v2, {pack2(POS_TWO, POS_TWO):#x}",
            f"v_mov_b32 v3, {pack2(POS_TWO, POS_TWO):#x}",
            "s_mov_b32 s10, exec_lo",
            "s_mov_b32 exec_lo, 0",
            "v_pk_fma_f16 v0, v1, v2, v3 clamp",
            "s_mov_b32 exec_lo, s10",
        ]
        got = run_ps(lines)
        self.assertEqual(got, OLD)


if __name__ == "__main__":
    unittest.main()
