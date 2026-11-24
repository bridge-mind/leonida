"""Semantic interpreter tests for single-precision FMA with IEEE clamp saturation (v_fma_f32).

RDNA2 VALU Instruction Set Architecture (ISA):
  v_fma_f32 dst, src0, src1, src2 [clamp] [omod]
  Operation: Computes IEEE-754 binary32 fused multiply-add:
    D = (src0 * src1) + src2
  with a single intermediate rounding step (round-to-nearest-even, RTE).

Modifiers and Semantics:
  - Source Modifiers:
      neg: negates operand (flips sign bit, including infinities, zeros, and NaNs)
      abs: absolute value (clears sign bit)
      Applied as: abs first, then neg (-|x|).
  - Output Modifiers (omod):
      mul:2: multiplies result by 2.0
      mul:4: multiplies result by 4.0
      div:2: multiplies result by 0.5
      Applied before clamp, matching hardware behavior.
  - IEEE Clamp Saturation (clamp):
      Saturates the final computed result into the unit interval [0.0, 1.0].
      - Values > 1.0 (and +inf) saturate to 1.0f (0x3F800000).
      - Values < 0.0 (and -inf) saturate to +0.0f (0x00000000).
      - Negative zero (-0.0f, 0x80000000) saturates to +0.0f (0x00000000).
      - NaNs are clamped/suppressed to +0.0f under hardware clamp saturation.
      - Values within [0.0, 1.0] pass through unmodified.

In SPIR-V Lowering (tools/rdna2spirv.py):
  v_fma_f32 lowers to:
    %r = OpExtInst %float %glsl Fma %src0 %src1 %src2
  With output modifiers and clamp:
    %scaled = OpFMul %float %r %scale
    %clamped = OpExtInst %float %glsl FClamp %scaled %f0 %f1

Standard library only: Python 3.9+ compatible, zero game data, zero proprietary assets.
"""
from __future__ import annotations

import itertools
import math
import os
import random
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

MASK32 = 0xFFFFFFFF
SIGN_MASK = 0x80000000
EXP_MASK = 0x7F800000
MANT_MASK = 0x007FFFFF

POS_ZERO = 0x00000000
NEG_ZERO = 0x80000000
ONE = 0x3F800000          # 1.0f
HALF = 0x3F000000         # 0.5f
ONE_HALF = 0x3FC00000     # 1.5f
TWO = 0x40000000          # 2.0f
THREE = 0x40400000        # 3.0f
FOUR = 0x40800000         # 4.0f
POS_INF = 0x7F800000
NEG_INF = 0xFF800000
QNAN = 0x7FC00000
SNAN = 0x7F800001
MAX_FINITE = 0x7F7FFFFF
MIN_NORMAL = 0x00800000
MIN_SUBNORMAL = 0x00000001
MAX_SUBNORMAL = 0x007FFFFF

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
OLD = 0xCAFE55AA


def bits_to_f32(bits: int) -> float:
    """Unpack a 32-bit unsigned integer to a Python float (binary32)."""
    return struct.unpack("<f", struct.pack("<I", bits & MASK32))[0]


def f32_to_bits(val: float) -> int:
    """Pack a Python float to a 32-bit unsigned integer word (binary32)."""
    if math.isnan(val):
        return QNAN
    try:
        return struct.unpack("<I", struct.pack("<f", val))[0]
    except OverflowError:
        return NEG_INF if val < 0 else POS_INF


def is_nan_bits(bits: int) -> bool:
    """Check if binary32 bit pattern represents any NaN."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & MANT_MASK) != 0)


def is_inf_bits(bits: int) -> bool:
    """Check if binary32 bit pattern represents infinity."""
    bits &= MASK32
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & MANT_MASK) == 0)


def is_zero_bits(bits: int) -> bool:
    """Check if binary32 bit pattern represents positive or negative zero."""
    return (bits & 0x7FFFFFFF) == 0


def is_subnormal_bits(bits: int) -> bool:
    """Check if binary32 bit pattern represents a subnormal (denormal) float."""
    bits &= MASK32
    return ((bits & EXP_MASK) == 0) and ((bits & MANT_MASK) != 0)


def exact_f32_to_fraction(bits: int) -> Fraction:
    """Decode a finite binary32 bit pattern into an exact rational Fraction."""
    bits &= MASK32
    sign = -1 if (bits >> 31) else 1
    exp = (bits >> 23) & 0xFF
    frac = bits & MANT_MASK
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
    """Round an exact rational Fraction to IEEE-754 binary32 (round-to-nearest-even)."""
    if val == 0:
        return NEG_ZERO if zero_sign else POS_ZERO
    sign = 1 if val < 0 else 0
    aval = -val if sign else val
    num, den = aval.numerator, aval.denominator

    e = num.bit_length() - den.bit_length()
    if (num << max(0, -e)) < (den << max(0, e)):
        e -= 1

    if e >= 128:
        return (sign << 31) | POS_INF

    if e < -126:
        target_shift = 149
    else:
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
            return (sign << 31) | (1 << 23)
        return (sign << 31) | q

    if q >= (1 << 24):
        q >>= 1
        e += 1
        if e >= 128:
            return (sign << 31) | POS_INF

    exp_field = e + 127
    frac_field = q & MANT_MASK
    return (sign << 31) | (exp_field << 23) | frac_field


def apply_src_mod(bits: int, neg: bool = False, absm: bool = False) -> int:
    """RDNA2 source modifier order: abs first, then neg (operates on sign bit)."""
    bits &= MASK32
    if absm:
        bits &= ~SIGN_MASK
    if neg:
        bits ^= SIGN_MASK
    return bits


def apply_out_mods(bits: int, omod: str | None = None, clamp: bool = False) -> int:
    """RDNA2 output modifiers: mul:2 / mul:4 / div:2 applied before clamp [0.0, 1.0]."""
    bits &= MASK32
    if is_nan_bits(bits):
        if clamp:
            return POS_ZERO
        return bits

    if omod is not None:
        f = bits_to_f32(bits)
        factor = {"mul:2": 2.0, "mul:4": 4.0, "div:2": 0.5}[omod]
        f_scaled = f * factor
        bits = f32_to_bits(f_scaled)

    if clamp:
        if is_nan_bits(bits):
            return POS_ZERO
        if (bits & SIGN_MASK) != 0:
            return POS_ZERO
        f = bits_to_f32(bits)
        if f > 1.0 or math.isinf(f):
            return ONE
        return bits

    return bits


def ref_fma_f32_rational(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    *,
    clamp: bool = False,
    omod: str | None = None,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
    src2_neg: bool = False,
    src2_abs: bool = False,
) -> int:
    """Exact rational Fraction reference oracle for single-precision FMA with clamp.

    Computes: D = (S0 * S1) + S2
    with exact unrounded product, exact unrounded sum, and a single round-to-nearest-even.
    """
    s0 = apply_src_mod(s0_bits, neg=src0_neg, absm=src0_abs)
    s1 = apply_src_mod(s1_bits, neg=src1_neg, absm=src1_abs)
    s2 = apply_src_mod(s2_bits, neg=src2_neg, absm=src2_abs)

    # 1. NaN propagation / Invalid operation detection
    if is_nan_bits(s0) or is_nan_bits(s1) or is_nan_bits(s2):
        return POS_ZERO if clamp else QNAN

    # Check for inf * 0 -> NaN
    inf0 = is_inf_bits(s0)
    inf1 = is_inf_bits(s1)
    zero0 = is_zero_bits(s0)
    zero1 = is_zero_bits(s1)
    if (inf0 and zero1) or (zero0 and inf1):
        return POS_ZERO if clamp else QNAN

    prod_sign = ((s0 ^ s1) >> 31) & 1

    # Product is infinity
    if inf0 or inf1:
        prod_inf = (prod_sign << 31) | POS_INF
        if is_inf_bits(s2):
            if ((s2 >> 31) & 1) != prod_sign:
                return POS_ZERO if clamp else QNAN
            res = prod_inf
        else:
            res = prod_inf
        return apply_out_mods(res, omod=omod, clamp=clamp)

    # S2 is infinity
    if is_inf_bits(s2):
        return apply_out_mods(s2, omod=omod, clamp=clamp)

    # Both product and S2 are finite
    frac0 = exact_f32_to_fraction(s0)
    frac1 = exact_f32_to_fraction(s1)
    frac2 = exact_f32_to_fraction(s2)

    total_frac = (frac0 * frac1) + frac2

    if total_frac == 0:
        # Sign of zero in IEEE-754:
        # Sum of two zeros with same sign retains sign; (+0) + (-0) = +0;
        # Cancellation of opposite non-zero quantities gives +0 under round-to-nearest.
        prod_is_zero = (zero0 or zero1)
        s2_is_zero = is_zero_bits(s2)
        if prod_is_zero and s2_is_zero:
            s2_sign = (s2 >> 31) & 1
            zsign = 1 if (prod_sign == 1 and s2_sign == 1) else 0
        else:
            zsign = 0
        res = NEG_ZERO if zsign else POS_ZERO
    else:
        res = round_fraction_to_f32_bits(total_frac)

    return apply_out_mods(res, omod=omod, clamp=clamp)


def ref_fma_f32_double(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    *,
    clamp: bool = False,
    omod: str | None = None,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
    src2_neg: bool = False,
    src2_abs: bool = False,
) -> int:
    """Double-precision (binary64) cross-check oracle for single-precision FMA.

    Binary32 float product requires at most 24 + 24 = 48 significand bits, which fits
    without rounding into binary64's 53-bit significand for normal non-overflowing ranges.
    """
    s0 = apply_src_mod(s0_bits, neg=src0_neg, absm=src0_abs)
    s1 = apply_src_mod(s1_bits, neg=src1_neg, absm=src1_abs)
    s2 = apply_src_mod(s2_bits, neg=src2_neg, absm=src2_abs)

    if is_nan_bits(s0) or is_nan_bits(s1) or is_nan_bits(s2):
        return POS_ZERO if clamp else QNAN

    f0, f1, f2 = bits_to_f32(s0), bits_to_f32(s1), bits_to_f32(s2)
    prod = f0 * f1
    if math.isnan(prod):
        return POS_ZERO if clamp else QNAN

    total = prod + f2
    if math.isnan(total):
        return POS_ZERO if clamp else QNAN

    if total == 0.0:
        # Determine zero sign
        prod_sign = ((s0 ^ s1) >> 31) & 1
        s2_sign = (s2 >> 31) & 1
        if is_zero_bits(s0) or is_zero_bits(s1):
            if is_zero_bits(s2) and prod_sign == 1 and s2_sign == 1:
                res = NEG_ZERO
            else:
                res = POS_ZERO
        else:
            res = POS_ZERO
    else:
        res = f32_to_bits(total)

    return apply_out_mods(res, omod=omod, clamp=clamp)


def ref_fma_f32(
    s0_bits: int,
    s1_bits: int,
    s2_bits: int,
    *,
    clamp: bool = False,
    omod: str | None = None,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
    src2_neg: bool = False,
    src2_abs: bool = False,
) -> int:
    """Canonical reference oracle for v_fma_f32."""
    return ref_fma_f32_rational(
        s0_bits,
        s1_bits,
        s2_bits,
        clamp=clamp,
        omod=omod,
        src0_neg=src0_neg,
        src0_abs=src0_abs,
        src1_neg=src1_neg,
        src1_abs=src1_abs,
        src2_neg=src2_neg,
        src2_abs=src2_abs,
    )


# -----------------------------------------------------------------------------
# Test Execution Harness (SPIR-V translation and execution)
# -----------------------------------------------------------------------------

_TMP_DIR = None
_SRC_PATH = None
_CS_PATH = None
_CS_FSSM = None
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def _make_fssm(rsrc2: int, group: tuple[int, int, int]) -> bytes:
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


def setUpModule():
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    _TMP_DIR = tempfile.TemporaryDirectory()
    disasm_dir = os.path.join(_TMP_DIR.name, "disasm")
    fssm_dir = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(disasm_dir)
    os.mkdir(fssm_dir)
    _SRC_PATH = os.path.join(disasm_dir, "ps_fma_f32_clamp.s")
    _CS_PATH = os.path.join(disasm_dir, "cs_fma_f32_clamp.s")
    _CS_FSSM = os.path.join(fssm_dir, "cs_fma_f32_clamp.fssm")
    with open(os.path.join(_TMP_DIR.name, "ps_fma_f32_clamp.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR, _SRC_PATH, _CS_PATH, _CS_FSSM
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None
        _SRC_PATH = None
        _CS_PATH = None
        _CS_FSSM = None


def translate_ps(lines: list[str]) -> str:
    """Translate synthetic pixel shader lines into SPIR-V text."""
    if _TMP_DIR is None:
        setUpModule()
    asm = (
        "; test_alu_f32_fma_clamp\n"
        + "".join(f"\t{ln}\n" for ln in lines)
        + "\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
    )
    with open(_SRC_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_SRC_PATH)


def run_ps_fma(
    s0: int,
    s1: int,
    s2: int,
    *,
    clamp: bool = False,
    omod: str | None = None,
    src0_expr: str = "v1",
    src1_expr: str = "v2",
    src2_expr: str = "v3",
) -> tuple[str, int]:
    """Execute v_fma_f32 via translated SPIR-V fragment shader."""
    mods = []
    if omod:
        mods.append(omod)
    if clamp:
        mods.append("clamp")
    mod_str = (" " + " ".join(mods)) if mods else ""

    lines = [
        f"v_mov_b32 v1, {s0:#x}",
        f"v_mov_b32 v2, {s1:#x}",
        f"v_mov_b32 v3, {s2:#x}",
        f"v_fma_f32_e64 v0, {src0_expr}, {src1_expr}, {src2_expr}{mod_str}",
    ]
    spv = translate_ps(lines)
    res = si.run_fragment(spv)["%out_mrt0"][0]
    return spv, res


def run_cs_fma_workgroup(
    group_size: int,
    in_s0: list[int],
    in_s1: list[int],
    in_s2: list[int],
    *,
    clamp: bool = False,
) -> list[int]:
    """Execute multi-lane compute shader evaluating v_fma_f32 with clamp across a workgroup."""
    if _TMP_DIR is None:
        setUpModule()

    # Buffer layout:
    # 0x000 .. 0x0FF: s0 inputs (4 bytes per lane)
    # 0x100 .. 0x1FF: s1 inputs
    # 0x200 .. 0x2FF: s2 inputs
    # 0x300 .. 0x3FF: dst outputs
    with open(_CS_FSSM, "wb") as f:
        f.write(_make_fssm(0x0088, (group_size, 1, 1)))

    mod_str = " clamp" if clamp else ""
    asm_lines = [
        "; cs_fma_f32_clamp",
        "\tv_lshlrev_b32_e32 v4, 2, v0",            # v4 = lane * 4
        "\tv_add_nc_u32_e32 v5, 0x100, v4",        # v5 = 0x100 + lane * 4
        "\tv_add_nc_u32_e32 v6, 0x200, v4",        # v6 = 0x200 + lane * 4
        "\tv_add_nc_u32_e32 v7, 0x300, v4",        # v7 = 0x300 + lane * 4
        "\tbuffer_load_dword v1, v4, s[0:3], 0 offen",
        "\tbuffer_load_dword v2, v5, s[0:3], 0 offen",
        "\tbuffer_load_dword v3, v6, s[0:3], 0 offen",
        "\ts_waitcnt vmcnt(0)",
        f"\tv_fma_f32_e64 v0, v1, v2, v3{mod_str}",
        "\tbuffer_store_dword v0, v7, s[0:3], 0 offen",
        "\ts_endpgm",
    ]
    with open(_CS_PATH, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(asm_lines) + "\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(_CS_PATH)

    mem: dict[int, int] = {}
    for i in range(group_size):
        mem[i * 4] = in_s0[i] & MASK32
        mem[0x100 + i * 4] = in_s1[i] & MASK32
        mem[0x200 + i * 4] = in_s2[i] & MASK32
        mem[0x300 + i * 4] = OLD

    res_mem = si.run_workgroup(
        spv,
        group_size,
        group_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0)},
        push_constants=VSHARP + [0] * 28,
        memory=mem,
    )
    return [res_mem.get(0x300 + i * 4, OLD) for i in range(group_size)]


# -----------------------------------------------------------------------------
# SIMD Wave Vector Interpreter
# -----------------------------------------------------------------------------

class VectorALUWaveSimd:
    """Deterministic SIMD wave simulator modeling RDNA2 VALU execution.

    Models:
      - 64 SIMD vector lanes (or 32 lanes)
      - Vector general purpose registers (VGPRs: v0..v63)
      - Scalar general purpose registers (SGPRs: s0..s105)
      - Execution mask (EXEC: 64-bit mask gating active lanes)
      - Saturation clamping to [0.0, 1.0] across all active vector lanes
    """

    def __init__(self, wave_size: int = 64, num_vgprs: int = 64, num_sgprs: int = 106):
        self.wave_size = wave_size
        self.num_vgprs = num_vgprs
        self.num_sgprs = num_sgprs
        self.vgprs = [[OLD] * wave_size for _ in range(num_vgprs)]
        self.sgprs = [0] * num_sgprs
        self.exec_mask = (1 << wave_size) - 1

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def set_vgpr_lane(self, reg: int, lane: int, val: int) -> None:
        self.vgprs[reg][lane] = val & MASK32

    def get_vgpr_lane(self, reg: int, lane: int) -> int:
        return self.vgprs[reg][lane] & MASK32

    def set_vgpr_vector(self, reg: int, values: list[int]) -> None:
        for lane in range(min(self.wave_size, len(values))):
            self.vgprs[reg][lane] = values[lane] & MASK32

    def get_vgpr_vector(self, reg: int) -> list[int]:
        return [self.vgprs[reg][lane] & MASK32 for lane in range(self.wave_size)]

    def set_sgpr(self, reg: int, val: int) -> None:
        self.sgprs[reg] = val & MASK32

    def get_sgpr(self, reg: int) -> int:
        return self.sgprs[reg] & MASK32

    def execute_v_fma_f32(
        self,
        dst_reg: int,
        src0: int | str,
        src1: int | str,
        src2: int | str,
        *,
        clamp: bool = False,
        omod: str | None = None,
        src0_neg: bool = False,
        src0_abs: bool = False,
        src1_neg: bool = False,
        src1_abs: bool = False,
        src2_neg: bool = False,
        src2_abs: bool = False,
    ) -> None:
        """Execute v_fma_f32 across all wave lanes with EXEC mask gating."""
        for lane in range(self.wave_size):
            if not ((self.exec_mask >> lane) & 1):
                continue

            def resolve(src: int | str) -> int:
                if isinstance(src, int):
                    return src & MASK32
                if src.startswith("v"):
                    return self.vgprs[int(src[1:])][lane]
                if src.startswith("s"):
                    return self.sgprs[int(src[1:])]
                raise ValueError(f"Unknown operand: {src}")

            s0 = resolve(src0)
            s1 = resolve(src1)
            s2 = resolve(src2)
            self.vgprs[dst_reg][lane] = ref_fma_f32(
                s0,
                s1,
                s2,
                clamp=clamp,
                omod=omod,
                src0_neg=src0_neg,
                src0_abs=src0_abs,
                src1_neg=src1_neg,
                src1_abs=src1_abs,
                src2_neg=src2_neg,
                src2_abs=src2_abs,
            )


# -----------------------------------------------------------------------------
# Unit Tests: Oracle Verification
# -----------------------------------------------------------------------------

class TestFmaF32ReferenceOracle(unittest.TestCase):
    """Verifies correctness of the dual reference oracle (rational vs double) and FMA semantics."""

    def test_oracle_equivalence_on_normal_vectors(self):
        """Cross-checks the exact rational Fraction oracle against the double-precision model."""
        test_values = [
            POS_ZERO,
            ONE,
            TWO,
            HALF,
            ONE_HALF,
            THREE,
            FOUR,
            0x3E800000,  # 0.25f
            0x3E000000,  # 0.125f
            0xBF800000,  # -1.0f
            0xC0000000,  # -2.0f
            0xBF000000,  # -0.5f
            0x3F400000,  # 0.75f
            0x40A00000,  # 5.0f
        ]
        for s0, s1, s2 in itertools.product(test_values[:8], test_values[:8], test_values[:8]):
            for clamp in (False, True):
                r_frac = ref_fma_f32_rational(s0, s1, s2, clamp=clamp)
                r_dbl = ref_fma_f32_double(s0, s1, s2, clamp=clamp)
                self.assertEqual(
                    r_frac,
                    r_dbl,
                    f"Mismatch for s0={s0:#x}, s1={s1:#x}, s2={s2:#x}, clamp={clamp}",
                )

    def test_basic_fma_arithmetic(self):
        """Standard FMA operations: D = (S0 * S1) + S2."""
        # 2.0 * 3.0 + 4.0 = 10.0
        self.assertEqual(ref_fma_f32(TWO, THREE, FOUR), f32_to_bits(10.0))
        # 0.5 * 0.5 + 0.25 = 0.5
        self.assertEqual(ref_fma_f32(HALF, HALF, f32_to_bits(0.25)), HALF)
        # 1.5 * -2.0 + 4.0 = 1.0
        self.assertEqual(ref_fma_f32(ONE_HALF, f32_to_bits(-2.0), FOUR), ONE)
        # 0.0 * 5.0 + 1.0 = 1.0
        self.assertEqual(ref_fma_f32(POS_ZERO, f32_to_bits(5.0), ONE), ONE)

    def test_fused_fma_vs_unfused_mad_precision(self):
        """Fused FMA computes product with unbounded precision, avoiding intermediate rounding."""
        # Case 1: Positive in-bounds residual 2^-24 vs unfused premature rounding to zero.
        # x = 1.0 + 2^-12, y = 1.0 + 2^-12, z = -(1.0 + 2^-11).
        # x * y = 1 + 2^-11 + 2^-24.
        # In unfused MAD: 1 + 2^-11 + 2^-24 rounds to 1 + 2^-11, so (1 + 2^-11) - (1 + 2^-11) = 0.0.
        # In fused FMA: exact product + z = 2^-24 (0x33800000), which stays 2^-24 under clamp.
        x1 = f32_to_bits(1.0 + 2.0 ** -12)
        y1 = f32_to_bits(1.0 + 2.0 ** -12)
        z1 = f32_to_bits(-(1.0 + 2.0 ** -11))

        prod_unfused1 = f32_to_bits(bits_to_f32(x1) * bits_to_f32(y1))
        mad_unfused1 = f32_to_bits(bits_to_f32(prod_unfused1) + bits_to_f32(z1))
        self.assertEqual(mad_unfused1, POS_ZERO)

        fma_fused1 = ref_fma_f32(x1, y1, z1)
        self.assertEqual(fma_fused1, 0x33800000)  # 2^-24
        self.assertEqual(ref_fma_f32(x1, y1, z1, clamp=True), 0x33800000)

        # Case 2: Negative residual -2^-46 clamped to +0.0 vs unclamped negative preservation.
        # x = 1.0 + 2^-23, y = 1.0 - 2^-23, z = -1.0.
        # Exact product is 1.0 - 2^-46.
        # In unfused MAD: product rounds to 1.0, giving 0.0.
        # In fused FMA: exact (1 - 2^-46) - 1.0 = -2^-46 (0xA8800000).
        # Clamping saturates negative -2^-46 to +0.0f (0x00000000).
        x2 = f32_to_bits(1.0 + 2.0 ** -23)
        y2 = f32_to_bits(1.0 - 2.0 ** -23)
        z2 = f32_to_bits(-1.0)

        prod_unfused2 = f32_to_bits(bits_to_f32(x2) * bits_to_f32(y2))
        mad_unfused2 = f32_to_bits(bits_to_f32(prod_unfused2) + bits_to_f32(z2))
        self.assertEqual(mad_unfused2, POS_ZERO)

        fma_fused2 = ref_fma_f32(x2, y2, z2)
        self.assertEqual(fma_fused2, 0xA8800000)  # -2^-46
        self.assertEqual(ref_fma_f32(x2, y2, z2, clamp=True), POS_ZERO)

    def test_clamp_saturation_boundaries(self):
        """Clamp saturates to [0.0, 1.0]: values > 1.0 clamp to 1.0, values < 0.0 clamp to +0.0."""
        # Interior values: pass through
        for val in (0.0, 0.125, 0.25, 0.5, 0.75, 1.0):
            b = f32_to_bits(val)
            self.assertEqual(ref_fma_f32(b, ONE, POS_ZERO, clamp=True), b)

        # Values > 1.0 clamp to 1.0f (0x3F800000)
        for val in (1.0001, 1.5, 2.0, 10.0, 1e20):
            b = f32_to_bits(val)
            self.assertEqual(ref_fma_f32(b, ONE, POS_ZERO, clamp=True), ONE)
            self.assertEqual(ref_fma_f32(ONE, b, POS_ZERO, clamp=True), ONE)

        # Values < 0.0 clamp to +0.0f (0x00000000)
        for val in (-0.0001, -0.5, -1.0, -10.0, -1e20):
            b = f32_to_bits(val)
            self.assertEqual(ref_fma_f32(b, ONE, POS_ZERO, clamp=True), POS_ZERO)
            self.assertEqual(ref_fma_f32(ONE, b, POS_ZERO, clamp=True), POS_ZERO)

        # Infinities clamp to 1.0 and +0.0
        self.assertEqual(ref_fma_f32(POS_INF, ONE, POS_ZERO, clamp=True), ONE)
        self.assertEqual(ref_fma_f32(NEG_INF, ONE, POS_ZERO, clamp=True), POS_ZERO)

    def test_source_modifiers_neg_abs(self):
        """VOP3 source modifiers on S0, S1, S2."""
        # -(2.0) * 3.0 + 10.0 = 4.0
        self.assertEqual(ref_fma_f32(TWO, THREE, f32_to_bits(10.0), src0_neg=True), FOUR)
        # |-2.0| * 3.0 + 4.0 = 10.0
        self.assertEqual(ref_fma_f32(f32_to_bits(-2.0), THREE, FOUR, src0_abs=True), f32_to_bits(10.0))
        # -|-2.0| * 3.0 + 10.0 = 4.0
        self.assertEqual(
            ref_fma_f32(f32_to_bits(-2.0), THREE, f32_to_bits(10.0), src0_neg=True, src0_abs=True),
            FOUR,
        )

        # Combined with clamp:
        # -(2.0) * 3.0 + 4.0 = -2.0 -> clamped to 0.0
        self.assertEqual(ref_fma_f32(TWO, THREE, FOUR, src0_neg=True, clamp=True), POS_ZERO)
        # 0.5 * 0.5 - 0.5 = -0.25 -> clamped to 0.0
        self.assertEqual(ref_fma_f32(HALF, HALF, HALF, src2_neg=True, clamp=True), POS_ZERO)

    def test_output_modifiers_and_clamp(self):
        """Output modifiers (mul:2, mul:4, div:2) are applied before clamp."""
        # 0.25 * 1.0 + 0.125 = 0.375
        # with mul:2 -> 0.75 (within [0.0, 1.0])
        s0, s1, s2 = f32_to_bits(0.25), ONE, f32_to_bits(0.125)
        self.assertEqual(ref_fma_f32(s0, s1, s2, omod="mul:2", clamp=True), f32_to_bits(0.75))

        # with mul:4 -> 1.5 -> clamped to 1.0
        self.assertEqual(ref_fma_f32(s0, s1, s2, omod="mul:4", clamp=True), ONE)

        # 1.5 * 1.0 + 0.5 = 2.0
        # with div:2 -> 1.0 (clamped to 1.0)
        self.assertEqual(ref_fma_f32(ONE_HALF, ONE, HALF, omod="div:2", clamp=True), ONE)


# -----------------------------------------------------------------------------
# Unit Tests: Shader Translation and Execution
# -----------------------------------------------------------------------------

class TestFmaF32ShaderTranslation(unittest.TestCase):
    """Verifies that RDNA2 v_fma_f32 translates to valid SPIR-V and executes accurately."""

    def test_translated_spirv_structure(self):
        """Translating v_fma_f32 with and without clamp generates GLSL Fma and FClamp."""
        spv_unclamped, _ = run_ps_fma(HALF, HALF, HALF, clamp=False)
        self.assertIn("OpExtInst %float %glsl Fma", spv_unclamped)
        self.assertNotIn("OpExtInst %float %glsl FClamp", spv_unclamped)

        spv_clamped, _ = run_ps_fma(HALF, HALF, HALF, clamp=True)
        self.assertIn("OpExtInst %float %glsl Fma", spv_clamped)
        self.assertIn("OpExtInst %float %glsl FClamp", spv_clamped)

    def test_shader_execution_normal_vectors(self):
        """End-to-end execution of unclamped v_fma_f32 matches reference oracle."""
        cases = [
            (f32_to_bits(0.5), f32_to_bits(0.5), f32_to_bits(0.25)),
            (f32_to_bits(2.0), f32_to_bits(3.0), f32_to_bits(4.0)),
            (f32_to_bits(-1.5), f32_to_bits(2.0), f32_to_bits(5.0)),
            (f32_to_bits(0.0), f32_to_bits(10.0), f32_to_bits(0.75)),
        ]
        for s0, s1, s2 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                _, got = run_ps_fma(s0, s1, s2, clamp=False)
                want = ref_fma_f32(s0, s1, s2, clamp=False)
                self.assertEqual(got, want)

    def test_shader_execution_clamped_vectors(self):
        """End-to-end execution of clamped v_fma_f32 saturates into [0.0, 1.0]."""
        cases = [
            # In-bounds: 0.5 * 0.5 + 0.25 = 0.5
            (f32_to_bits(0.5), f32_to_bits(0.5), f32_to_bits(0.25)),
            # Over-range: 2.0 * 2.0 + 1.0 = 5.0 -> clamped to 1.0
            (f32_to_bits(2.0), f32_to_bits(2.0), f32_to_bits(1.0)),
            # Under-range: -1.0 * 2.0 + 0.5 = -1.5 -> clamped to 0.0
            (f32_to_bits(-1.0), f32_to_bits(2.0), f32_to_bits(0.5)),
            # Edge: exactly 1.0
            (f32_to_bits(1.0), f32_to_bits(1.0), f32_to_bits(0.0)),
            # Edge: exactly 0.0
            (f32_to_bits(0.0), f32_to_bits(0.0), f32_to_bits(0.0)),
        ]
        for s0, s1, s2 in cases:
            with self.subTest(s0=hex(s0), s1=hex(s1), s2=hex(s2)):
                _, got = run_ps_fma(s0, s1, s2, clamp=True)
                want = ref_fma_f32(s0, s1, s2, clamp=True)
                self.assertEqual(got, want)


# -----------------------------------------------------------------------------
# Unit Tests: SIMD Multi-Lane Execution & [0.0, 1.0] Saturation Clamping
# -----------------------------------------------------------------------------

class TestFmaF32SimdMultiLane(unittest.TestCase):
    """Verifies SIMD wave vector execution and [0.0, 1.0] saturation clamping."""

    def test_wave64_linear_sweep_saturation_clamping(self):
        """64-lane SIMD wave evaluating a linear sweep across clamp boundaries."""
        simd = VectorALUWaveSimd(wave_size=64)

        # S0: Linear ramp from -1.5 to +1.5 across 64 lanes
        # S1: Uniform 1.0f
        # S2: Uniform 0.0f
        # S0 * S1 + S2 = S0
        ramp = [f32_to_bits(-1.5 + (3.0 * i / 63.0)) for i in range(64)]
        simd.set_vgpr_vector(1, ramp)
        simd.set_vgpr_vector(2, [ONE] * 64)
        simd.set_vgpr_vector(3, [POS_ZERO] * 64)

        # 1. Unclamped execution: full dynamic range preserved
        simd.execute_v_fma_f32(0, "v1", "v2", "v3", clamp=False)
        out_unclamped = simd.get_vgpr_vector(0)
        self.assertEqual(out_unclamped, ramp)

        # 2. Clamped execution: saturates to [0.0, 1.0]
        simd.execute_v_fma_f32(0, "v1", "v2", "v3", clamp=True)
        out_clamped = simd.get_vgpr_vector(0)

        for i in range(64):
            val = -1.5 + (3.0 * i / 63.0)
            got = out_clamped[i]
            if val < 0.0:
                self.assertEqual(got, POS_ZERO, f"Lane {i} (val={val}) must saturate to +0.0")
            elif val > 1.0:
                self.assertEqual(got, ONE, f"Lane {i} (val={val}) must saturate to 1.0")
            else:
                self.assertEqual(got, ramp[i], f"Lane {i} (val={val}) must pass through unmodified")

    def test_wave32_and_wave64_exec_mask_gating(self):
        """Active lanes compute clamped FMA; inactive lanes strictly retain prior register state."""
        for wave_size in (32, 64):
            with self.subTest(wave_size=wave_size):
                simd = VectorALUWaveSimd(wave_size=wave_size)
                simd.set_vgpr_vector(1, [TWO] * wave_size)   # 2.0
                simd.set_vgpr_vector(2, [TWO] * wave_size)   # 2.0
                simd.set_vgpr_vector(3, [ONE] * wave_size)   # 1.0 -> 5.0 -> clamped to 1.0

                # 1. Even lanes active
                even_mask = 0x5555555555555555 if wave_size == 64 else 0x55555555
                simd.set_exec(even_mask)
                simd.execute_v_fma_f32(0, "v1", "v2", "v3", clamp=True)
                out = simd.get_vgpr_vector(0)
                for lane in range(wave_size):
                    if lane % 2 == 0:
                        self.assertEqual(out[lane], ONE, f"Active lane {lane} should clamp to 1.0")
                    else:
                        self.assertEqual(out[lane], OLD, f"Inactive lane {lane} must retain OLD")

                # 2. Inactive wave (exec = 0): no lanes modified
                simd2 = VectorALUWaveSimd(wave_size=wave_size)
                simd2.set_exec(0)
                simd2.execute_v_fma_f32(0, "v1", "v2", "v3", clamp=True)
                self.assertEqual(simd2.get_vgpr_vector(0), [OLD] * wave_size)

    def test_wave_multi_channel_color_clamping(self):
        """Simulates 64 pixels computing clamped color intensity: clamp((base * light) + ambient)."""
        simd = VectorALUWaveSimd(wave_size=64)

        base = []
        light = []
        ambient = []
        expected = []

        for i in range(64):
            if i < 16:
                b, l, a = 0.5, 0.8, 0.1  # 0.5 * 0.8 + 0.1 = 0.5
                exp = f32_to_bits(0.5)
            elif i < 32:
                b, l, a = 0.9, 1.8, 0.2  # 0.9 * 1.8 + 0.2 = 1.82 -> clamped to 1.0
                exp = ONE
            elif i < 48:
                b, l, a = 0.2, 0.5, -0.4  # 0.2 * 0.5 - 0.4 = -0.3 -> clamped to 0.0
                exp = POS_ZERO
            else:
                b, l, a = 1.0, 1.0, 0.0   # 1.0 * 1.0 + 0.0 = 1.0
                exp = ONE
            base.append(f32_to_bits(b))
            light.append(f32_to_bits(l))
            ambient.append(f32_to_bits(a))
            expected.append(exp)

        simd.set_vgpr_vector(1, base)
        simd.set_vgpr_vector(2, light)
        simd.set_vgpr_vector(3, ambient)

        simd.execute_v_fma_f32(0, "v1", "v2", "v3", clamp=True)
        self.assertEqual(simd.get_vgpr_vector(0), expected)

    def test_wave_scalar_sgpr_broadcast(self):
        """Scalar SGPR operand broadcast across all lanes in the wave combined with per-lane VGPRs."""
        simd = VectorALUWaveSimd(wave_size=64)

        v1_in = [f32_to_bits(float(i) * 0.05 - 0.5) for i in range(64)]
        simd.set_vgpr_vector(1, v1_in)

        simd.set_sgpr(4, TWO)
        simd.set_sgpr(5, f32_to_bits(0.25))

        simd.execute_v_fma_f32(0, "v1", "s4", "s5", clamp=True)
        out = simd.get_vgpr_vector(0)

        for lane in range(64):
            x = float(lane) * 0.05 - 0.5
            want = ref_fma_f32(v1_in[lane], TWO, f32_to_bits(0.25), clamp=True)
            self.assertEqual(out[lane], want, f"Lane {lane} mismatch for {x} * 2.0 + 0.25")

    def test_compute_shader_multi_lane_workgroup(self):
        """End-to-end multi-lane compute shader execution via spirv_mini_interp.run_workgroup."""
        group_size = 16
        in_s0 = [f32_to_bits(float(i) - 4.0) for i in range(group_size)]  # -4.0 .. 11.0
        in_s1 = [HALF] * group_size                                       # * 0.5
        in_s2 = [f32_to_bits(0.25)] * group_size                          # + 0.25

        res = run_cs_fma_workgroup(group_size, in_s0, in_s1, in_s2, clamp=True)

        for i in range(group_size):
            want = ref_fma_f32(in_s0[i], in_s1[i], in_s2[i], clamp=True)
            self.assertEqual(res[i], want, f"Compute shader lane {i} mismatch")


if __name__ == "__main__":
    unittest.main()
