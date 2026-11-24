"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_exp_i32_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision unbiased exponent extraction as
specified in the AMD RDNA2 / GFX10 Instruction Set Architecture:
  - v_frexp_exp_i32_f64: D.i = Exponent(S0.d)
    Extracts the unbiased integer exponent of S0.d such that S0.d = mantissa * 2^exponent with
    |mantissa| in [0.5, 1.0).
    The result is returned as a 32-bit signed integer (two's complement) written to a single 32-bit VGPR.

RDNA2 Double-Precision Exponent Extraction Rules:
  1. Register Operands:
     Source operand S0 is a 64-bit float occupying an aligned register pair v[s:s+1] or s[s:s+1],
     or a 64-bit immediate literal. Destination D is a single 32-bit vector register v[d].
  2. Normal Floats:
     In IEEE 754 double precision, exponent bias is 1023 and 0.5 has biased exponent 1022.
     For normal doubles (1 <= biased_exponent <= 2046):
       unbiased_exponent = biased_exponent - 1022
     Cross-checks identically with Python math.frexp(x)[1].
  3. Special Values:
     * +/-0.0: returns 0
     * +/-INF: returns 0
     * NaN (quiet or signaling): returns 0
  4. Subnormals (Denormals):
     In IEEE mode (default), subnormal numbers (biased_exponent == 0, fraction != 0) are normalized:
       unbiased_exponent = -1022 - leading_zeros(fraction)
     Under flush-to-zero (FTZ), subnormals return 0.
  5. VOP1 / VOP3 Encoding Forms & Modifiers:
     Available in VOP1 (e32) and VOP3 (e64) encodings.
     Input modifiers: abs (|src|) and neg (-src). Since the exponent depends only on magnitude,
     abs and neg preserve the computed exponent value.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

import dataclasses
import enum
import math
import os
import struct
import sys
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
SIGN_MASK_F64: int = 0x8000000000000000
EXP_MASK_F64: int = 0x7FF0000000000000
MANT_MASK_F64: int = 0x000FFFFFFFFFFFFF
EXP_SHIFT_F64: int = 52
EXP_BIAS_F64: int = 1023
EXP_FREXP_BIASED_F64: int = 1022  # 2^-1 has biased exponent 1022 (0x3FE) in double precision

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO_F64: int = 0x0000000000000000
NEG_ZERO_F64: int = 0x8000000000000000
POS_ONE_F64: int = 0x3FF0000000000000       # 1.0 -> exp = 1
NEG_ONE_F64: int = 0xBFF0000000000000       # -1.0 -> exp = 1
HALF_F64: int = 0x3FE0000000000000          # 0.5 -> exp = 0
NEG_HALF_F64: int = 0xBFE0000000000000      # -0.5 -> exp = 0
POS_TWO_F64: int = 0x4000000000000000       # 2.0 -> exp = 2
NEG_TWO_F64: int = 0xC000000000000000       # -2.0 -> exp = 2
POS_FOUR_F64: int = 0x4010000000000000      # 4.0 -> exp = 3
QUARTER_F64: int = 0x3FD0000000000000       # 0.25 -> exp = -1

DBL_MIN: int = 0x0010000000000000           # 2^-1022 -> exp = -1021
NEG_DBL_MIN: int = 0x8010000000000000       # -2^-1022 -> exp = -1021
DBL_MAX: int = 0x7FEFFFFFFFFFFFFF           # ~1.797e+308 -> exp = 1024
NEG_DBL_MAX: int = 0xFFEFFFFFFFFFFFFF
POS_INF_F64: int = 0x7FF0000000000000       # +infinity -> exp = 0
NEG_INF_F64: int = 0xFFF0000000000000       # -infinity -> exp = 0
QNAN_F64: int = 0x7FF8000000000000          # Quiet NaN -> exp = 0
NEG_QNAN_F64: int = 0xFFF8000000000000      # Negative quiet NaN -> exp = 0
SNAN_F64: int = 0x7FF0000000000001          # Signaling NaN -> exp = 0
SUBNORM_MIN_F64: int = 0x0000000000000001   # 2^-1074 -> exp = -1073
SUBNORM_MAX_F64: int = 0x000FFFFFFFFFFFFF   # (1 - 2^-52)*2^-1022 -> exp = -1022
NEG_SUBNORM_MIN_F64: int = 0x8000000000000001
NEG_SUBNORM_MAX_F64: int = 0x800FFFFFFFFFFFFF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned word as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned word as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & SIGN_MASK_F64) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    """Decode 64-bit unsigned integer to IEEE 754 double-precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Encode double float to IEEE 754 double-precision 64-bit unsigned integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return POS_INF_F64 if val > 0 else NEG_INF_F64


def is_nan_f64(bits: int) -> bool:
    """True if 64-bit integer represents an IEEE 754 double NaN."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0


def is_subnormal_f64(bits: int) -> bool:
    """True if bits represent an IEEE 754 double subnormal number."""
    bits &= MASK64
    return (bits & EXP_MASK_F64) == 0 and (bits & MANT_MASK_F64) != 0


def apply_f64_modifiers(bits: int, neg: bool = False, abs_val: bool = False) -> int:
    """Applies VOP3 input modifiers: abs (|x|) first, then neg (-x)."""
    bits &= MASK64
    if abs_val:
        bits &= ~SIGN_MASK_F64
    if neg:
        bits ^= SIGN_MASK_F64
    return bits & MASK64


def ref_frexp_exp_i32_f64(s0_bits: int, ftz: bool = False) -> int:
    """Pure bitwise reference oracle for RDNA2 v_frexp_exp_i32_f64.

    Rules:
      - If NaN or +/-INF: returns 0.
      - If +/-0.0: returns 0.
      - If normal double: returns (biased_exp - 1022) as 32-bit two's complement.
      - If subnormal double: returns (-1022 - leading_zeros) under IEEE mode, or 0 under FTZ.
    """
    s0_bits &= MASK64
    exp_biased = (s0_bits & EXP_MASK_F64) >> EXP_SHIFT_F64
    frac = s0_bits & MANT_MASK_F64

    # Special case: +/-INF or NaN
    if exp_biased == 0x7FF:
        return 0

    # Special case: zero or subnormal
    if exp_biased == 0:
        if frac == 0 or ftz:
            return 0
        # In double precision, mantissa has 52 bits
        lz = 52 - frac.bit_length()
        unbiased = -1022 - lz
        return unbiased & MASK32

    # Normal double float
    unbiased = exp_biased - EXP_FREXP_BIASED_F64
    return unbiased & MASK32


def ref_frexp_exp_i32_ext(
    s0_bits: int,
    neg: bool = False,
    abs_val: bool = False,
    ftz: bool = False,
) -> int:
    """Reference oracle for v_frexp_exp_i32_f64 with input modifiers."""
    mod = apply_f64_modifiers(s0_bits, neg=neg, abs_val=abs_val)
    return ref_frexp_exp_i32_f64(mod, ftz=ftz)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register file domain."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Symbolic register operand representing a 32-bit register or 64-bit pair."""
    kind: RegKind
    index: int
    count: int = 1

    def __post_init__(self) -> None:
        if self.index < 0 or self.index >= 256:
            raise ValueError(f"Register index out of range [0, 255]: {self.index}")
        if self.count not in (1, 2):
            raise ValueError(f"Unsupported register count (must be 1 or 2): {self.count}")
        if self.count == 2 and self.index + 1 >= 256:
            raise ValueError(f"64-bit register pair exceeds register file boundary: {self.index}")

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR

    @property
    def is_sgpr(self) -> bool:
        return self.kind == RegKind.SGPR

    def format_asm(self) -> str:
        if self.count == 2:
            return f"{self.kind.value}[{self.index}:{self.index + 1}]"
        return f"{self.kind.value}{self.index}"


@dataclasses.dataclass(frozen=True)
class Immediate:
    """Symbolic 64-bit immediate float operand."""
    bits: int

    def __init__(self, val: int | float) -> None:
        if isinstance(val, float):
            b = f64_to_bits(val)
        else:
            b = val & MASK64
        object.__setattr__(self, "bits", b)

    @property
    def as_float(self) -> float:
        return bits_to_f64(self.bits)

    def format_asm(self) -> str:
        return f"{self.bits:#018x}"


@dataclasses.dataclass(frozen=True)
class OperandModifiers:
    """Input modifiers for VALU floating-point instructions (abs and neg)."""
    neg: bool = False
    abs_val: bool = False

    def apply(self, bits: int) -> int:
        return apply_f64_modifiers(bits, neg=self.neg, abs_val=self.abs_val)

    def format_asm(self, base_str: str) -> str:
        res = base_str
        if self.abs_val:
            res = f"|{res}|"
        if self.neg:
            res = f"-{res}"
        return res


class Opcode(enum.Enum):
    """VALU exponent extraction opcode."""
    V_FREXP_EXP_I32_F64 = "v_frexp_exp_i32_f64"


class Encoding(enum.Enum):
    """Instruction encoding format."""
    VOP1 = "e32"
    VOP3 = "e64"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a v_frexp_exp_i32_f64 instruction."""
    opcode: Opcode
    dst: Register
    src0: Register | Immediate
    src0_mods: OperandModifiers = dataclasses.field(default_factory=OperandModifiers)
    encoding: Encoding = Encoding.VOP3

    def __post_init__(self) -> None:
        if not self.dst.is_vgpr:
            raise ValueError(f"VALU instruction destination must be a VGPR: {self.dst}")
        if self.dst.count != 1:
            raise ValueError(f"v_frexp_exp_i32_f64 destination must be 32-bit (count=1): {self.dst}")
        if isinstance(self.src0, Register) and self.src0.count != 2:
            raise ValueError(f"v_frexp_exp_i32_f64 source register must be a 64-bit pair (count=2): {self.src0}")
        # VOP1 (e32) does not support input modifiers
        has_mods = self.src0_mods.neg or self.src0_mods.abs_val
        if self.encoding == Encoding.VOP1 and has_mods:
            raise ValueError("VOP1 encoding does not support input modifiers (requires VOP3/e64)")

    def to_assembly(self) -> str:
        s0_str = self.src0.format_asm()
        s0_fmt = self.src0_mods.format_asm(s0_str)
        dst_str = self.dst.format_asm()
        enc_suffix = f"_{self.encoding.value}" if self.encoding == Encoding.VOP1 else ""
        return f"{self.opcode.value}{enc_suffix} {dst_str}, {s0_fmt}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of VALU instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class ConditionCodes:
    """Architectural integer condition codes for v_frexp_exp_i32_f64 output."""
    zero: bool
    negative: bool
    positive: bool


@dataclasses.dataclass(frozen=True)
class FPStatusFlags:
    """IEEE 754 floating-point status flags for the input operand."""
    invalid: bool       # e.g. signaling NaN input
    denormal: bool      # e.g. subnormal double input
    inexact: bool = False
    overflow: bool = False
    underflow: bool = False


def compute_condition_codes(result: int) -> ConditionCodes:
    """Computes condition codes evaluating the extracted 32-bit integer exponent."""
    signed_val = s32(result)
    return ConditionCodes(
        zero=(signed_val == 0),
        negative=(signed_val < 0),
        positive=(signed_val > 0),
    )


def compute_fp_status_flags(s0_bits: int) -> FPStatusFlags:
    """Computes IEEE 754 status flags for double-precision input."""
    bits = s0_bits & MASK64
    invalid = bool((bits & EXP_MASK_F64) == EXP_MASK_F64 and (bits & MANT_MASK_F64) != 0 and not (bits & 0x0008000000000000))
    denormal = is_subnormal_f64(bits)
    return FPStatusFlags(
        invalid=invalid,
        denormal=denormal,
        inexact=False,
        overflow=False,
        underflow=False,
    )


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarVALUInterpreter:
    """Deterministic scalar interpreter for GFX10/RDNA2 v_frexp_exp_i32_f64."""

    def __init__(self, mode_ftz: bool = False) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.mode_ftz: bool = mode_ftz
        self.vcc: int = 0
        self.last_condition_codes: ConditionCodes | None = None
        self.last_status_flags: FPStatusFlags | None = None
        self.trace: list[str] = []

    def read_reg_u32(self, kind: RegKind, idx: int) -> int:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        return (self.vgprs[idx] if kind == RegKind.VGPR else self.sgprs[idx]) & MASK32

    def write_reg_u32(self, kind: RegKind, idx: int, val: int) -> None:
        if idx < 0 or (kind == RegKind.VGPR and idx >= 256) or (kind == RegKind.SGPR and idx >= 106):
            raise IndexError(f"Register index out of bounds: {kind.value}{idx}")
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_reg_u64(self, kind: RegKind, idx: int) -> int:
        lo = self.read_reg_u32(kind, idx)
        hi = self.read_reg_u32(kind, idx + 1)
        return join_dwords(lo, hi)

    def write_reg_u64(self, kind: RegKind, idx: int, val: int) -> None:
        lo, hi = split_dwords(val)
        self.write_reg_u32(kind, idx, lo)
        self.write_reg_u32(kind, idx + 1, hi)

    def read_operand(self, op: Register | Immediate, mods: OperandModifiers) -> int:
        if isinstance(op, Immediate):
            raw = op.bits
        elif isinstance(op, Register):
            raw = self.read_reg_u64(op.kind, op.index)
        else:
            raise TypeError(f"Unsupported operand type: {type(op)}")
        return mods.apply(raw)

    def step(self, inst: Instruction) -> int:
        """Executes a single v_frexp_exp_i32_f64 instruction deterministically."""
        # Read 64-bit float source before writing 32-bit integer destination
        s0_val = self.read_operand(inst.src0, inst.src0_mods)

        if inst.opcode == Opcode.V_FREXP_EXP_I32_F64:
            res = ref_frexp_exp_i32_f64(s0_val, ftz=self.mode_ftz)
        else:
            raise NotImplementedError(f"Unsupported opcode in scalar interpreter: {inst.opcode}")

        self.write_reg_u32(inst.dst.kind, inst.dst.index, res)
        self.last_condition_codes = compute_condition_codes(res)
        self.last_status_flags = compute_fp_status_flags(s0_val)
        # Note: v_frexp_exp_i32_f64 does not generate carry or borrow; VCC is isolated and unchanged
        self.trace.append(f"{inst.to_assembly()} => {res:#010x} ({s32(res)})")
        return res

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


# ---------------------------------------------------------------------------
# SIMD Wavefront Interpreter (Multi-Lane Divergence & 32/64-Bit EXEC Masking)
# ---------------------------------------------------------------------------

class SIMDWavefrontInterpreter:
    """Multi-lane SIMD interpreter modeling Wave32 and Wave64 execution on RDNA2."""

    def __init__(self, wave_size: int = 64, mode_ftz: bool = False) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size} (must be 32 or 64)")
        self.wave_size: int = wave_size
        self.mode_ftz: bool = mode_ftz
        self.exec_mask: int = (1 << wave_size) - 1
        # Per-lane VGPR files: [wave_size][256] 32-bit registers
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        # Wavefront-shared SGPR file: [106] 32-bit registers
        self.sgprs: list[int] = [0] * 106
        self.vcc: int = 0

    def set_exec_mask(self, mask: int) -> None:
        valid_mask = (1 << self.wave_size) - 1
        self.exec_mask = mask & valid_mask

    def read_lane_reg_u32(self, lane: int, kind: RegKind, idx: int) -> int:
        if kind == RegKind.VGPR:
            return self.vgprs[lane][idx] & MASK32
        return self.sgprs[idx] & MASK32

    def write_lane_reg_u32(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        val32 = val & MASK32
        if kind == RegKind.VGPR:
            self.vgprs[lane][idx] = val32
        else:
            self.sgprs[idx] = val32

    def read_lane_reg_u64(self, lane: int, kind: RegKind, idx: int) -> int:
        lo = self.read_lane_reg_u32(lane, kind, idx)
        hi = self.read_lane_reg_u32(lane, kind, idx + 1)
        return join_dwords(lo, hi)

    def write_lane_reg_u64(self, lane: int, kind: RegKind, idx: int, val: int) -> None:
        lo, hi = split_dwords(val)
        self.write_lane_reg_u32(lane, kind, idx, lo)
        self.write_lane_reg_u32(lane, kind, idx + 1, hi)

    def read_lane_operand(self, lane: int, op: Register | Immediate, mods: OperandModifiers) -> int:
        if isinstance(op, Immediate):
            raw = op.bits
        elif isinstance(op, Register):
            raw = self.read_lane_reg_u64(lane, op.kind, op.index)
        else:
            raise TypeError(f"Unsupported operand type: {type(op)}")
        return mods.apply(raw)

    def step(self, inst: Instruction) -> list[int | None]:
        """Executes a VALU instruction across all lanes in the wavefront governed by EXEC mask."""
        # 1. Latch inputs for all lanes (read-before-write semantics)
        latched_inputs: list[int] = []
        for lane in range(self.wave_size):
            s0_val = self.read_lane_operand(lane, inst.src0, inst.src0_mods)
            latched_inputs.append(s0_val)

        # 2. Compute and write back strictly for active lanes
        lane_results: list[int | None] = []
        for lane in range(self.wave_size):
            is_active = bool((self.exec_mask >> lane) & 1)
            if not is_active:
                lane_results.append(None)
                continue

            s0_val = latched_inputs[lane]
            if inst.opcode == Opcode.V_FREXP_EXP_I32_F64:
                res = ref_frexp_exp_i32_f64(s0_val, ftz=self.mode_ftz)
            else:
                raise NotImplementedError(f"Unsupported opcode in SIMD interpreter: {inst.opcode}")

            self.write_lane_reg_u32(lane, inst.dst.kind, inst.dst.index, res)
            lane_results.append(res)

        return lane_results


class TestValuFrexpExpI32F64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_powers_of_two(self) -> None:
        self.assertEqual(s32(ref_frexp_exp_i32_f64(HALF_F64)), 0)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(POS_ONE_F64)), 1)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(POS_TWO_F64)), 2)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(POS_FOUR_F64)), 3)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(QUARTER_F64)), -1)

    def test_cross_check_math_frexp_normals(self) -> None:
        test_floats = [
            1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.141592653589793, -100.25,
            1e10, 1e-10, 1e300, 1e-300,
        ]
        for f in test_floats:
            with self.subTest(val=f):
                expected_exp = math.frexp(f)[1]
                bits = f64_to_bits(f)
                computed_exp = s32(ref_frexp_exp_i32_f64(bits))
                self.assertEqual(computed_exp, expected_exp)

    def test_zeros_return_zero(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ZERO_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_ZERO_F64), 0)

    def test_infinities_return_zero(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(POS_INF_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_INF_F64), 0)

    def test_nans_return_zero(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(QNAN_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_QNAN_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(SNAN_F64), 0)

    def test_subnormals_unbiased_exponent(self) -> None:
        # SUBNORM_MAX: leading bit is at 51, lz = 0 -> exp = -1022
        self.assertEqual(s32(ref_frexp_exp_i32_f64(SUBNORM_MAX_F64)), -1022)
        # SUBNORM_MIN: fraction is 1, lz = 51 -> exp = -1073
        self.assertEqual(s32(ref_frexp_exp_i32_f64(SUBNORM_MIN_F64)), -1073)

    def test_modifiers_abs_neg(self) -> None:
        # Sign does not alter exponent magnitude
        self.assertEqual(ref_frexp_exp_i32_ext(NEG_ONE_F64, abs_val=True), 1)
        self.assertEqual(ref_frexp_exp_i32_ext(POS_ONE_F64, neg=True), 1)
        self.assertEqual(ref_frexp_exp_i32_ext(POS_TWO_F64, neg=True, abs_val=True), 2)

    def test_ftz_mode(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(SUBNORM_MIN_F64, ftz=True), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(SUBNORM_MAX_F64, ftz=True), 0)
        # Normal floats unaffected by FTZ
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ONE_F64, ftz=True), 1)


class TestValuFrexpExpI32F64AST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_register_types_and_validation(self) -> None:
        dst = Register(RegKind.VGPR, 0, count=1)
        src = Register(RegKind.VGPR, 2, count=2)
        sgpr_src = Register(RegKind.SGPR, 4, count=2)

        self.assertTrue(dst.is_vgpr)
        self.assertFalse(dst.is_sgpr)
        self.assertEqual(dst.format_asm(), "v0")
        self.assertEqual(src.format_asm(), "v[2:3]")
        self.assertEqual(sgpr_src.format_asm(), "s[4:5]")

    def test_immediate_representation(self) -> None:
        imm = Immediate(1.0)
        self.assertEqual(imm.bits, POS_ONE_F64)
        self.assertEqual(imm.as_float, 1.0)
        self.assertEqual(imm.format_asm(), f"{POS_ONE_F64:#018x}")

    def test_instruction_formatting(self) -> None:
        inst_vop3 = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
            src0_mods=OperandModifiers(neg=True, abs_val=True),
        )
        self.assertEqual(inst_vop3.to_assembly(), "v_frexp_exp_i32_f64 v0, -|v[2:3]|")

        inst_vop1 = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 5, count=1),
            src0=Register(RegKind.VGPR, 6, count=2),
            encoding=Encoding.VOP1,
        )
        self.assertEqual(inst_vop1.to_assembly(), "v_frexp_exp_i32_f64_e32 v5, v[6:7]")

    def test_instruction_validation(self) -> None:
        # Invalid destination kind (must be VGPR)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.SGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

        # Invalid destination count (must be 1 for i32)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

        # Invalid source count (must be 2 for 64-bit float)
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=1),
            )

        # VOP1 does not support input modifiers
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=2),
                src0_mods=OperandModifiers(neg=True),
                encoding=Encoding.VOP1,
            )

    def test_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        ))
        prog.append(Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 1, count=1),
            src0=Immediate(2.0),
        ))
        expected = (
            "v_frexp_exp_i32_f64 v0, v[2:3]\n"
            f"v_frexp_exp_i32_f64 v1, {POS_TWO_F64:#018x}"
        )
        self.assertEqual(prog.to_assembly(), expected)


class TestValuFrexpExpI32F64ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar VALU instruction interpreter."""

    def test_scalar_register_read_write(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u32(RegKind.VGPR, 0, 0x12345678)
        interp.write_reg_u64(RegKind.VGPR, 2, POS_TWO_F64)
        interp.write_reg_u64(RegKind.SGPR, 4, NEG_HALF_F64)

        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), 0x12345678)
        self.assertEqual(interp.read_reg_u64(RegKind.VGPR, 2), POS_TWO_F64)
        self.assertEqual(interp.read_reg_u64(RegKind.SGPR, 4), NEG_HALF_F64)

    def test_scalar_execution_from_vgpr_pair(self) -> None:
        interp = ScalarVALUInterpreter()
        # Set v[2:3] = 4.0 -> exp = 3
        interp.write_reg_u64(RegKind.VGPR, 2, POS_FOUR_F64)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(s32(res), 3)
        self.assertEqual(s32(interp.read_reg_u32(RegKind.VGPR, 0)), 3)

    def test_scalar_execution_from_sgpr_pair(self) -> None:
        interp = ScalarVALUInterpreter()
        # Set s[0:1] = 0.5 -> exp = 0
        interp.write_reg_u64(RegKind.SGPR, 0, HALF_F64)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 5, count=1),
            src0=Register(RegKind.SGPR, 0, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(s32(res), 0)
        self.assertEqual(s32(interp.read_reg_u32(RegKind.VGPR, 5)), 0)

    def test_scalar_execution_from_immediate(self) -> None:
        interp = ScalarVALUInterpreter()
        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(1.0),
        )
        res = interp.step(inst)
        self.assertEqual(s32(res), 1)
        self.assertEqual(s32(interp.read_reg_u32(RegKind.VGPR, 0)), 1)

    def test_scalar_modifiers_abs_neg(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, NEG_TWO_F64)

        # v0 = frexp_exp(-|-2.0|) -> exp(-2.0) = 2
        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
            src0_mods=OperandModifiers(neg=True, abs_val=True),
        )
        res = interp.step(inst)
        self.assertEqual(s32(res), 2)
        self.assertEqual(s32(interp.read_reg_u32(RegKind.VGPR, 0)), 2)

    def test_scalar_program_execution_chain(self) -> None:
        interp = ScalarVALUInterpreter()
        interp.write_reg_u64(RegKind.VGPR, 2, POS_FOUR_F64)   # 4.0 -> exp = 3
        interp.write_reg_u64(RegKind.VGPR, 4, QUARTER_F64)    # 0.25 -> exp = -1

        prog = Program()
        prog.append(Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        ))
        prog.append(Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 1, count=1),
            src0=Register(RegKind.VGPR, 4, count=2),
        ))

        results = interp.run_program(prog)
        self.assertEqual([s32(r) for r in results], [3, -1])
        self.assertEqual(s32(interp.read_reg_u32(RegKind.VGPR, 0)), 3)
        self.assertEqual(s32(interp.read_reg_u32(RegKind.VGPR, 1)), -1)
        self.assertEqual(len(interp.trace), 2)


class TestValuFrexpExpI32F64SIMD(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_inputs(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        for lane in range(32):
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 2, POS_TWO_F64)

        results = simd.step(inst)
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(s32(results[lane]), 2)
            self.assertEqual(s32(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)), 2)

    def test_wave64_multi_lane_divergence(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=64)
        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        expected: list[int] = []
        for lane in range(64):
            # Powers of 2: 2^(lane - 30)
            val = math.ldexp(1.0, lane - 30)
            bits = f64_to_bits(val)
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 2, bits)
            expected.append(s32(ref_frexp_exp_i32_f64(bits)))

        results = simd.step(inst)
        for lane in range(64):
            self.assertEqual(s32(results[lane]), expected[lane])
            self.assertEqual(s32(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)), expected[lane])

    def test_wave32_exec_masking_partial(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        exec_mask = 0x55555555  # Even lanes active
        simd.set_exec_mask(exec_mask)

        canary = 0xDEADBEEF
        for lane in range(32):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 2, POS_FOUR_F64)  # exp = 3

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        results = simd.step(inst)

        for lane in range(32):
            if (exec_mask >> lane) & 1:
                self.assertEqual(s32(results[lane]), 3)
                self.assertEqual(s32(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)), 3)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary)

    def test_wave64_exec_masking_all_inactive(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.set_exec_mask(0)

        canary = 0xBAADF00D
        for lane in range(64):
            simd.write_lane_reg_u32(lane, RegKind.VGPR, 0, canary)
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 2, POS_ONE_F64)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 2, count=2),
        )
        results = simd.step(inst)
        self.assertEqual(results, [None] * 64)
        for lane in range(64):
            self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), canary)

    def test_simd_with_shared_sgpr_scalar_source(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        # Shared SGPR source s[4:5] = 2.0 -> exp = 2
        simd.write_lane_reg_u64(0, RegKind.SGPR, 4, POS_TWO_F64)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.SGPR, 4, count=2),
        )
        results = simd.step(inst)
        for lane in range(32):
            self.assertEqual(s32(results[lane]), 2)
            self.assertEqual(s32(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)), 2)


class TestValuFrexpExpI32F64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_aliasing_dst_equals_src0_low_dword(self) -> None:
        # dst = v0, src0 = v[0:1]
        interp = ScalarVALUInterpreter()
        # Set v[0:1] = 4.0 -> exp = 3
        interp.write_reg_u64(RegKind.VGPR, 0, POS_FOUR_F64)
        lo_orig, hi_orig = split_dwords(POS_FOUR_F64)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 0, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(s32(res), 3)
        # v0 overwritten with result 3
        self.assertEqual(s32(interp.read_reg_u32(RegKind.VGPR, 0)), 3)
        # v1 (high dword) remains undisturbed
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), hi_orig)

    def test_aliasing_dst_equals_src0_high_dword(self) -> None:
        # dst = v1, src0 = v[0:1]
        interp = ScalarVALUInterpreter()
        # Set v[0:1] = 0.25 -> exp = -1
        interp.write_reg_u64(RegKind.VGPR, 0, QUARTER_F64)
        lo_orig, hi_orig = split_dwords(QUARTER_F64)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 1, count=1),
            src0=Register(RegKind.VGPR, 0, count=2),
        )
        res = interp.step(inst)
        self.assertEqual(s32(res), -1)
        # v0 (low dword) remains undisturbed
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 0), lo_orig)
        # v1 overwritten with result -1 (0xFFFFFFFF)
        self.assertEqual(interp.read_reg_u32(RegKind.VGPR, 1), 0xFFFFFFFF)

    def test_unrelated_registers_preserved(self) -> None:
        interp = ScalarVALUInterpreter()
        for idx in range(10, 20):
            interp.write_reg_u32(RegKind.VGPR, idx, 0x55000000 + idx)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(2.0),
        )
        interp.step(inst)

        for idx in range(10, 20):
            self.assertEqual(interp.read_reg_u32(RegKind.VGPR, idx), 0x55000000 + idx)

    def test_simd_aliasing_with_divergent_exec_mask(self) -> None:
        simd = SIMDWavefrontInterpreter(wave_size=32)
        exec_mask = 0x0000000F  # Lanes 0..3 active
        simd.set_exec_mask(exec_mask)

        for lane in range(32):
            simd.write_lane_reg_u64(lane, RegKind.VGPR, 0, POS_TWO_F64)  # exp = 2

        lo_orig, hi_orig = split_dwords(POS_TWO_F64)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Register(RegKind.VGPR, 0, count=2),
        )
        simd.step(inst)

        for lane in range(32):
            if lane < 4:
                # Active: v0 updated to 2, v1 unchanged
                self.assertEqual(s32(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0)), 2)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), hi_orig)
            else:
                # Inactive: v0 and v1 both preserved
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 0), lo_orig)
                self.assertEqual(simd.read_lane_reg_u32(lane, RegKind.VGPR, 1), hi_orig)


class TestValuFrexpExpI32F64ConditionCodes(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow isolation."""

    def test_condition_codes_zero_pos_neg(self) -> None:
        interp = ScalarVALUInterpreter()

        # Input 0.5 -> exp = 0
        interp.step(Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(0.5),
        ))
        cc_zero = interp.last_condition_codes
        self.assertIsNotNone(cc_zero)
        assert cc_zero is not None
        self.assertTrue(cc_zero.zero)
        self.assertFalse(cc_zero.negative)
        self.assertFalse(cc_zero.positive)

        # Input 2.0 -> exp = 2 (positive)
        interp.step(Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(2.0),
        ))
        cc_pos = interp.last_condition_codes
        self.assertIsNotNone(cc_pos)
        assert cc_pos is not None
        self.assertFalse(cc_pos.zero)
        self.assertFalse(cc_pos.negative)
        self.assertTrue(cc_pos.positive)

        # Input 0.25 -> exp = -1 (negative)
        interp.step(Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(0.25),
        ))
        cc_neg = interp.last_condition_codes
        self.assertIsNotNone(cc_neg)
        assert cc_neg is not None
        self.assertFalse(cc_neg.zero)
        self.assertTrue(cc_neg.negative)
        self.assertFalse(cc_neg.positive)

    def test_status_flags_snan_invalid(self) -> None:
        flags_snan = compute_fp_status_flags(SNAN_F64)
        self.assertTrue(flags_snan.invalid)
        self.assertFalse(flags_snan.denormal)

        flags_qnan = compute_fp_status_flags(QNAN_F64)
        self.assertFalse(flags_qnan.invalid)

    def test_status_flags_denormal(self) -> None:
        flags_sub = compute_fp_status_flags(SUBNORM_MIN_F64)
        self.assertTrue(flags_sub.denormal)
        self.assertFalse(flags_sub.invalid)

        flags_norm = compute_fp_status_flags(POS_ONE_F64)
        self.assertFalse(flags_norm.denormal)
        self.assertFalse(flags_norm.invalid)

    def test_carry_borrow_vcc_isolation(self) -> None:
        interp = ScalarVALUInterpreter()
        canary = 0xCAFEBABE
        interp.vcc = canary

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(1.0),
        )
        interp.step(inst)
        self.assertEqual(interp.vcc, canary)

        simd = SIMDWavefrontInterpreter(wave_size=64)
        simd.vcc = canary
        simd.step(inst)
        self.assertEqual(simd.vcc, canary)


class TestValuFrexpExpI32F64IEEE754Boundaries(unittest.TestCase):
    """Exhaustive unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zeros(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ZERO_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_ZERO_F64), 0)

    def test_infinities(self) -> None:
        self.assertEqual(ref_frexp_exp_i32_f64(POS_INF_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_INF_F64), 0)

    def test_nans_payloads(self) -> None:
        nan_payloads = [
            QNAN_F64,
            NEG_QNAN_F64,
            0x7FF8000000000001,
            0xFFF8000012345678,
            SNAN_F64,
            0xFFF0000000000001,
        ]
        for nan in nan_payloads:
            with self.subTest(nan=f"{nan:#018x}"):
                self.assertEqual(ref_frexp_exp_i32_f64(nan), 0)

    def test_normal_extremes(self) -> None:
        # DBL_MIN: 2^-1022 -> exp = -1021
        self.assertEqual(s32(ref_frexp_exp_i32_f64(DBL_MIN)), -1021)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(NEG_DBL_MIN)), -1021)

        # DBL_MAX: biased exp 2046 -> exp = 1024
        self.assertEqual(s32(ref_frexp_exp_i32_f64(DBL_MAX)), 1024)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(NEG_DBL_MAX)), 1024)

    def test_subnormals_range(self) -> None:
        # SUBNORM_MIN: 2^-1074 -> exp = -1073
        self.assertEqual(s32(ref_frexp_exp_i32_f64(SUBNORM_MIN_F64)), -1073)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(NEG_SUBNORM_MIN_F64)), -1073)

        # SUBNORM_MAX: frac bit 51 set -> lz = 0 -> exp = -1022
        self.assertEqual(s32(ref_frexp_exp_i32_f64(SUBNORM_MAX_F64)), -1022)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(NEG_SUBNORM_MAX_F64)), -1022)

        # Intermediate subnormals with varying leading zeros
        for lz in range(52):
            frac = 1 << (51 - lz)
            subnorm_bits = frac  # biased_exp = 0
            expected_exp = -1022 - lz
            self.assertEqual(s32(ref_frexp_exp_i32_f64(subnorm_bits)), expected_exp)

    def test_subnormals_ftz(self) -> None:
        for lz in range(0, 52, 5):
            frac = 1 << (51 - lz)
            self.assertEqual(ref_frexp_exp_i32_f64(frac, ftz=True), 0)

    def test_modifiers_on_boundary_values(self) -> None:
        boundary_values = [
            POS_ZERO_F64, NEG_ZERO_F64, POS_INF_F64, NEG_INF_F64,
            QNAN_F64, SNAN_F64, DBL_MIN, NEG_DBL_MIN, DBL_MAX, NEG_DBL_MAX,
            SUBNORM_MIN_F64, SUBNORM_MAX_F64,
        ]
        for val in boundary_values:
            base_exp = ref_frexp_exp_i32_f64(val)
            self.assertEqual(ref_frexp_exp_i32_ext(val, neg=True), base_exp)
            self.assertEqual(ref_frexp_exp_i32_ext(val, abs_val=True), base_exp)
            self.assertEqual(ref_frexp_exp_i32_ext(val, neg=True, abs_val=True), base_exp)


class TestValuFrexpExpI32F64NegativeControls(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and invalid AST inputs."""

    def test_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, -1, count=1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 256, count=1)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 255, count=2)

    def test_invalid_register_counts(self) -> None:
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 0, count=0)
        with self.assertRaises(ValueError):
            Register(RegKind.VGPR, 0, count=3)

    def test_invalid_destination_kind(self) -> None:
        # VALU destination cannot be SGPR
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.SGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=2),
            )

    def test_invalid_operand_counts(self) -> None:
        # Destination must be count=1
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=2),
                src0=Register(RegKind.VGPR, 2, count=2),
            )
        # Source must be count=2
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=1),
            )

    def test_vop1_rejects_modifiers(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(
                opcode=Opcode.V_FREXP_EXP_I32_F64,
                dst=Register(RegKind.VGPR, 0, count=1),
                src0=Register(RegKind.VGPR, 2, count=2),
                src0_mods=OperandModifiers(neg=True),
                encoding=Encoding.VOP1,
            )

    def test_scalar_interpreter_out_of_bounds_access(self) -> None:
        interp = ScalarVALUInterpreter()
        with self.assertRaises(IndexError):
            interp.read_reg_u32(RegKind.VGPR, 256)
        with self.assertRaises(IndexError):
            interp.write_reg_u32(RegKind.SGPR, 106, 0)

    def test_simd_interpreter_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefrontInterpreter(wave_size=128)


class TestValuFrexpExpI32F64MutationDetection(unittest.TestCase):
    """Mutation tests: verifies that intentional sabotages/mutations of the oracle are caught."""

    def test_mutation_wrong_bias(self) -> None:
        # Sabotage: biased exponent - 1023 instead of biased exponent - 1022
        def mutated_frexp_exp(bits: int) -> int:
            exp_biased = (bits & EXP_MASK_F64) >> EXP_SHIFT_F64
            if exp_biased in (0, 0x7FF):
                return 0
            return (exp_biased - 1023) & MASK32

        with self.assertRaises(AssertionError):
            self.assertEqual(s32(mutated_frexp_exp(POS_ONE_F64)), 1)

    def test_mutation_subnormal_omission(self) -> None:
        # Sabotage: flushing subnormals to zero in IEEE mode
        def mutated_frexp_exp(bits: int) -> int:
            exp_biased = (bits & EXP_MASK_F64) >> EXP_SHIFT_F64
            if exp_biased == 0:
                return 0
            return (exp_biased - 1022) & MASK32

        with self.assertRaises(AssertionError):
            self.assertEqual(s32(mutated_frexp_exp(SUBNORM_MIN_F64)), -1073)

    def test_mutation_special_value_leak(self) -> None:
        # Sabotage: computing unbiased exponent for infinities and NaNs instead of returning 0
        def mutated_frexp_exp(bits: int) -> int:
            exp_biased = (bits & EXP_MASK_F64) >> EXP_SHIFT_F64
            return (exp_biased - 1022) & MASK32

        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_frexp_exp(POS_INF_F64), 0)
        with self.assertRaises(AssertionError):
            self.assertEqual(mutated_frexp_exp(QNAN_F64), 0)

    def test_mutation_simd_exec_mask_leak(self) -> None:
        # Sabotage: inactive lanes write to destination register
        simd = SIMDWavefrontInterpreter(wave_size=32)
        simd.set_exec_mask(0)  # all inactive
        canary = 0x11223344
        simd.write_lane_reg_u32(0, RegKind.VGPR, 0, canary)

        inst = Instruction(
            opcode=Opcode.V_FREXP_EXP_I32_F64,
            dst=Register(RegKind.VGPR, 0, count=1),
            src0=Immediate(2.0),
        )
        simd.step(inst)
        self.assertEqual(simd.read_lane_reg_u32(0, RegKind.VGPR, 0), canary)

        # Mutated unconditional write
        simd.write_lane_reg_u32(0, RegKind.VGPR, 0, 2)
        with self.assertRaises(AssertionError):
            self.assertEqual(simd.read_lane_reg_u32(0, RegKind.VGPR, 0), canary)


if __name__ == "__main__":
    unittest.main()
