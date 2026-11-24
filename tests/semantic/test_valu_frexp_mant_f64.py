from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_mant_f64.

Instruction:
  v_frexp_mant_f64 dst, src0 (VOP1 / VOP3)
  D.f64 = mantissa(S0.f64)

Architecture:
  - RDNA2 Vector ALU (GFX10) double-precision mantissa extraction.
  - S0: 64-bit IEEE 754 binary64 floating-point operand (VGPR pair, SGPR pair, or literal).
  - D:  64-bit IEEE 754 binary64 floating-point destination (VGPR pair).
  - Result:
      * For finite non-zero S0: returns mantissa in [0.5, 1.0) with sign of S0.
      * For +0.0 / -0.0: returns +0.0 / -0.0.
      * For +inf / -inf: returns +inf / -inf.
      * For NaN: returns quiet NaN (payload preserved/quieted).
      * Subnormal inputs are normalized in IEEE preserve mode before extracting mantissa.
  - Modifiers (VOP3):
      * Input modifiers on S0: abs (|S0|), neg (-S0).
      * Output modifiers: clamp (clamps result to [0.0, 1.0]), omod (mul:2, mul:4, div:2).
"""

from fractions import Fraction
import math
import os
import re
import struct
import sys
import unittest
from dataclasses import dataclass
from enum import Enum, auto

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F64 = 0x8000000000000000
EXP_MASK_F64 = 0x7FF0000000000000
MANT_MASK_F64 = 0x000FFFFFFFFFFFFF
BIAS_F64 = 1023
FREXP_EXP_F64 = 1022  # Biased exponent for range [0.5, 1.0)

POS_ZERO_F64 = 0x0000000000000000
NEG_ZERO_F64 = 0x8000000000000000
POS_INF_F64 = 0x7FF0000000000000
NEG_INF_F64 = 0xFFF0000000000000
QNAN_F64 = 0x7FF8000000000000
SNAN_F64 = 0x7FF0000000000001

ONE_F64 = 0x3FF0000000000000        # 1.0
TWO_F64 = 0x4000000000000000        # 2.0
HALF_F64 = 0x3FE0000000000000       # 0.5
MAX_FINITE_F64 = 0x7FEFFFFFFFFFFFFF # ~1.7976931348623157e+308
MIN_NORMAL_F64 = 0x0010000000000000 # 2^-1022 (~2.2250738585072014e-308)
MAX_SUBNORMAL_F64 = 0x000FFFFFFFFFFFFF
MIN_SUBNORMAL_F64 = 0x0000000000000001 # 2^-1074 (~4.9406564584124654e-324)


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit dword."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit qword."""
    return val & MASK64


def s32(val: int) -> int:
    """Interpret a 32-bit unsigned dword as a signed two's-complement integer."""
    val &= MASK32
    return val - (1 << 32) if (val & 0x80000000) else val


def s64(val: int) -> int:
    """Interpret a 64-bit unsigned qword as a signed two's-complement integer."""
    val &= MASK64
    return val - (1 << 64) if (val & (1 << 63)) else val


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def bits_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE 754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def f64_to_bits(val: float) -> int:
    """Pack an IEEE 754 double precision float into an unsigned 64-bit integer word."""
    if math.isnan(val):
        return QNAN_F64
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return NEG_INF_F64 if val < 0.0 else POS_INF_F64


def is_nan_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents any NaN."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & MANT_MASK_F64) != 0)


def is_inf_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents +/- infinity."""
    return ((bits & EXP_MASK_F64) == EXP_MASK_F64) and ((bits & MANT_MASK_F64) == 0)


def is_zero_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents +/- 0.0."""
    return (bits & ~SIGN_MASK_F64) == 0


def is_subnormal_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents a subnormal float."""
    return ((bits & EXP_MASK_F64) == 0) and ((bits & MANT_MASK_F64) != 0)


def is_normal_f64(bits: int) -> bool:
    """Check if binary64 bit pattern represents a normalized finite float."""
    exp = (bits & EXP_MASK_F64) >> 52
    return 1 <= exp <= 2046


def ref_frexp_mant_f64(
    src0_bits: int,
    *,
    clamp: bool = False,
    abs_src0: bool = False,
    neg_src0: bool = False,
    omod: str = "none",
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_frexp_mant_f64.

    Extracts the normalized mantissa in [0.5, 1.0) with the sign of S0.
    """
    src0_bits &= MASK64

    if abs_src0:
        src0_bits &= ~SIGN_MASK_F64
    if neg_src0:
        src0_bits ^= SIGN_MASK_F64

    sign = (src0_bits >> 63) & 1
    exp_field = (src0_bits >> 52) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    # 1. NaN handling
    if exp_field == 0x7FF and mant_field != 0:
        if clamp:
            return POS_ZERO_F64
        return src0_bits | 0x0008000000000000

    # 2. Infinity handling
    if exp_field == 0x7FF and mant_field == 0:
        if clamp:
            return POS_ZERO_F64 if sign else ONE_F64
        return src0_bits

    # 3. Zero handling (+/- 0.0)
    if exp_field == 0 and mant_field == 0:
        if clamp and sign:
            return POS_ZERO_F64
        return src0_bits

    # 4. Extract mantissa bits
    if exp_field == 0:
        # Subnormal input: normalize mantissa
        F = mant_field
        shift_left = 52 - (F.bit_length() - 1)
        normalized_mant = (F << shift_left) & MANT_MASK_F64
    else:
        # Normal input: directly use mantissa bits
        normalized_mant = mant_field

    # Base exponent for [0.5, 1.0) is 1022 (unbiased -1)
    target_exp = FREXP_EXP_F64

    # Apply omod multiplier
    if omod == "mul:2":
        target_exp += 1
    elif omod == "mul:4":
        target_exp += 2
    elif omod == "div:2":
        target_exp -= 1

    res = (sign << 63) | (target_exp << 52) | normalized_mant

    if clamp:
        if sign:
            return POS_ZERO_F64
        if res > ONE_F64:
            return ONE_F64

    return res


def ref_frexp_mant_f64_math(src0_bits: int) -> int:
    """Cross-check using Python's math.frexp."""
    src0_bits &= MASK64
    if is_nan_f64(src0_bits):
        return src0_bits | 0x0008000000000000
    if is_inf_f64(src0_bits) or is_zero_f64(src0_bits):
        return src0_bits

    f = bits_to_f64(src0_bits)
    m, _ = math.frexp(f)
    return f64_to_bits(m)


def ref_frexp_mant_f64_fraction(src0_bits: int) -> int:
    """Exact rational mathematical cross-check using fractions.Fraction."""
    src0_bits &= MASK64
    if is_nan_f64(src0_bits):
        return src0_bits | 0x0008000000000000
    if is_inf_f64(src0_bits) or is_zero_f64(src0_bits):
        return src0_bits

    sign = (src0_bits >> 63) & 1
    exp_field = (src0_bits >> 52) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    if exp_field == 0:
        frac = Fraction(mant_field, 1 << 52) * (Fraction(2) ** -1022)
    else:
        frac = (Fraction(1) + Fraction(mant_field, 1 << 52)) * (Fraction(2) ** (exp_field - 1023))

    frac_abs = abs(frac)
    # Find power of two such that frac_abs * 2^p is in [0.5, 1.0)
    p = 0
    while frac_abs >= 1:
        frac_abs /= 2
        p -= 1
    while frac_abs < Fraction(1, 2):
        frac_abs *= 2
        p += 1

    res_float = float(frac_abs) if sign == 0 else -float(frac_abs)
    return f64_to_bits(res_float)


class DataType(Enum):
    """Data types supported by RDNA2 VALU instructions."""
    F64 = auto()


class OperandKind(Enum):
    """Classification of instruction operands."""
    VGPR_PAIR = auto()
    SGPR_PAIR = auto()
    LITERAL_F64 = auto()
    INLINE_CONST = auto()


@dataclass
class Operand:
    """Typed operand representation in symbolic AST."""
    kind: OperandKind
    data_type: DataType
    reg_index: int = 0
    literal_val: int = 0
    abs_mod: bool = False
    neg_mod: bool = False

    def validate(self) -> None:
        if self.kind == OperandKind.VGPR_PAIR:
            if not (0 <= self.reg_index < 255):
                raise ValueError(f"VGPR pair base {self.reg_index} out of valid range [0, 255)")
        elif self.kind == OperandKind.SGPR_PAIR:
            if not (0 <= self.reg_index < 105):
                raise ValueError(f"SGPR pair base {self.reg_index} out of valid range [0, 105)")

    def to_asm(self) -> str:
        s = ""
        if self.kind == OperandKind.VGPR_PAIR:
            s = f"v[{self.reg_index}:{self.reg_index + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            s = f"s[{self.reg_index}:{self.reg_index + 1}]"
        elif self.kind == OperandKind.LITERAL_F64:
            s = f"{bits_to_f64(self.literal_val)}"
        elif self.kind == OperandKind.INLINE_CONST:
            s = str(self.literal_val)

        if self.abs_mod:
            s = f"|{s}|"
        if self.neg_mod:
            s = f"-{s}"
        return s


@dataclass
class InstructionAST:
    """Symbolic AST node for v_frexp_mant_f64."""
    opcode: str
    dst: Operand
    src0: Operand
    clamp: bool = False
    omod: str = "none"

    def validate(self) -> None:
        if self.opcode != "v_frexp_mant_f64":
            raise ValueError(f"Unsupported opcode {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR or self.dst.data_type != DataType.F64:
            raise TypeError("Destination must be a 64-bit VGPR pair (DataType.F64)")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination does not support input modifiers")
        if self.src0.data_type != DataType.F64:
            raise TypeError("src0 must be DataType.F64")
        if self.omod not in ("none", "mul:2", "mul:4", "div:2"):
            raise ValueError(f"Invalid omod modifier '{self.omod}'")
        self.dst.validate()
        self.src0.validate()

    def to_asm(self) -> str:
        tokens = [f"{self.opcode} {self.dst.to_asm()}, {self.src0.to_asm()}"]
        if self.clamp:
            tokens.append("clamp")
        if self.omod != "none":
            tokens.append(self.omod)
        return " ".join(tokens)


def parse_vop_frexp_mant_f64(asm_line: str) -> InstructionAST:
    """Parse assembly string into an InstructionAST."""
    line = asm_line.strip()
    if not line.startswith("v_frexp_mant_f64"):
        raise ValueError(f"Expected instruction starting with 'v_frexp_mant_f64', got: {line}")

    if ";" in line:
        line = line.split(";", 1)[0].strip()
    if "//" in line:
        line = line.split("//", 1)[0].strip()

    parts = line.split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Incomplete instruction: {line}")

    opcode = parts[0]
    rest = parts[1]

    tokens = [t.strip() for t in rest.split(",") if t.strip()]
    if len(tokens) < 2:
        raise ValueError(f"v_frexp_mant_f64 requires 2 operand sections, got: {len(tokens)}")

    dst_str = tokens[0]
    last_section = tokens[1].split()
    src0_str = last_section[0]
    mod_tokens = [m.lower() for m in last_section[1:]]

    clamp = "clamp" in mod_tokens
    omod = "none"
    for m in mod_tokens:
        if m in ("mul:2", "mul:4", "div:2"):
            omod = m

    # Parse DST
    m_dst = re.match(r"^v\[(\d+):(\d+)\]$", dst_str.strip())
    if not m_dst:
        raise ValueError(f"Invalid 64-bit VGPR destination format: {dst_str}")
    d_lo, d_hi = int(m_dst.group(1)), int(m_dst.group(2))
    if d_hi != d_lo + 1:
        raise ValueError(f"Destination register pair must be contiguous: {dst_str}")
    dst_op = Operand(kind=OperandKind.VGPR_PAIR, data_type=DataType.F64, reg_index=d_lo)

    # Parse SRC0
    s0_str = src0_str.strip()
    neg_s0 = False
    abs_s0 = False
    if s0_str.startswith("-"):
        neg_s0 = True
        s0_str = s0_str[1:].strip()
    if s0_str.startswith("|") and s0_str.endswith("|"):
        abs_s0 = True
        s0_str = s0_str[1:-1].strip()

    m_v0 = re.match(r"^v\[(\d+):(\d+)\]$", s0_str)
    m_s0 = re.match(r"^s\[(\d+):(\d+)\]$", s0_str)
    if m_v0:
        lo, hi = int(m_v0.group(1)), int(m_v0.group(2))
        if hi != lo + 1:
            raise ValueError(f"Source register pair must be contiguous: {s0_str}")
        src0_op = Operand(
            kind=OperandKind.VGPR_PAIR,
            data_type=DataType.F64,
            reg_index=lo,
            abs_mod=abs_s0,
            neg_mod=neg_s0,
        )
    elif m_s0:
        lo, hi = int(m_s0.group(1)), int(m_s0.group(2))
        if hi != lo + 1:
            raise ValueError(f"Source register pair must be contiguous: {s0_str}")
        src0_op = Operand(
            kind=OperandKind.SGPR_PAIR,
            data_type=DataType.F64,
            reg_index=lo,
            abs_mod=abs_s0,
            neg_mod=neg_s0,
        )
    else:
        try:
            if s0_str.startswith("0x") or s0_str.startswith("0X"):
                val64 = int(s0_str, 16)
            else:
                val64 = f64_to_bits(float(s0_str))
            src0_op = Operand(
                kind=OperandKind.LITERAL_F64,
                data_type=DataType.F64,
                literal_val=val64,
                abs_mod=abs_s0,
                neg_mod=neg_s0,
            )
        except ValueError:
            raise ValueError(f"Unrecognized src0 operand format: {s0_str}")

    ast = InstructionAST(
        opcode=opcode,
        dst=dst_op,
        src0=src0_op,
        clamp=clamp,
        omod=omod,
    )
    ast.validate()
    return ast


@dataclass
class FPStatusFlags:
    """IEEE 754 floating-point status exception flags."""
    invalid: bool = False
    div_by_zero: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False

    def merge(self, other: FPStatusFlags) -> None:
        self.invalid |= other.invalid
        self.div_by_zero |= other.div_by_zero
        self.overflow |= other.overflow
        self.underflow |= other.underflow
        self.inexact |= other.inexact


def is_snan_f64(bits: int) -> bool:
    """Check if binary64 bits represent a signaling NaN."""
    if not is_nan_f64(bits):
        return False
    return ((bits >> 51) & 1) == 0


def calc_frexp_mant_f64_status_flags(src0_bits: int, res_bits: int) -> FPStatusFlags:
    """Determine IEEE 754 exception flags for v_frexp_mant_f64."""
    flags = FPStatusFlags()
    if is_snan_f64(src0_bits):
        flags.invalid = True
    return flags


class ScalarVALUInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU instructions."""

    def __init__(self) -> None:
        self.num_vgprs = 256
        self.num_sgprs = 106
        self.vgpr = [0] * self.num_vgprs
        self.sgpr = [0] * self.num_sgprs
        self.vcc = 0
        self.exec = MASK64
        self.status_flags = FPStatusFlags()

    def reset(self) -> None:
        self.vgpr = [0] * self.num_vgprs
        self.sgpr = [0] * self.num_sgprs
        self.vcc = 0
        self.exec = MASK64
        self.status_flags = FPStatusFlags()

    def get_vgpr_u32(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index {idx} out of range [0, {self.num_vgprs})")
        return self.vgpr[idx] & MASK32

    def set_vgpr_u32(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index {idx} out of range [0, {self.num_vgprs})")
        self.vgpr[idx] = val & MASK32

    def get_vgpr_u64(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"64-bit VGPR base {idx} out of range [0, {self.num_vgprs - 1})")
        lo = self.get_vgpr_u32(idx)
        hi = self.get_vgpr_u32(idx + 1)
        return join_dwords(lo, hi)

    def set_vgpr_u64(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"64-bit VGPR base {idx} out of range [0, {self.num_vgprs - 1})")
        lo, hi = split_dwords(val64)
        self.set_vgpr_u32(idx, lo)
        self.set_vgpr_u32(idx + 1, hi)

    def get_sgpr_u32(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index {idx} out of range [0, {self.num_sgprs})")
        return self.sgpr[idx] & MASK32

    def set_sgpr_u32(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index {idx} out of range [0, {self.num_sgprs})")
        self.sgpr[idx] = val & MASK32

    def get_sgpr_u64(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {idx} out of range [0, {self.num_sgprs - 1})")
        lo = self.get_sgpr_u32(idx)
        hi = self.get_sgpr_u32(idx + 1)
        return join_dwords(lo, hi)

    def set_sgpr_u64(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {idx} out of range [0, {self.num_sgprs - 1})")
        lo, hi = split_dwords(val64)
        self.set_sgpr_u32(idx, lo)
        self.set_sgpr_u32(idx + 1, hi)

    def eval_operand_f64(self, op: Operand) -> int:
        if op.data_type != DataType.F64:
            raise TypeError(f"Expected DataType.F64, got {op.data_type}")
        if op.kind == OperandKind.VGPR_PAIR:
            val = self.get_vgpr_u64(op.reg_index)
        elif op.kind == OperandKind.SGPR_PAIR:
            val = self.get_sgpr_u64(op.reg_index)
        elif op.kind == OperandKind.LITERAL_F64:
            val = op.literal_val & MASK64
        else:
            raise ValueError(f"Unsupported operand kind for f64: {op.kind}")

        if op.abs_mod:
            val &= ~SIGN_MASK_F64
        if op.neg_mod:
            val ^= SIGN_MASK_F64
        return val

    def execute(self, ast: InstructionAST) -> int:
        ast.validate()
        if ast.opcode == "v_frexp_mant_f64":
            s0_val = self.eval_operand_f64(ast.src0)
            res = ref_frexp_mant_f64(
                s0_val,
                clamp=ast.clamp,
                omod=ast.omod,
            )
            flags = calc_frexp_mant_f64_status_flags(s0_val, res)
            self.status_flags.merge(flags)
            self.set_vgpr_u64(ast.dst.reg_index, res)
            return res
        raise NotImplementedError(f"Opcode {ast.opcode} not implemented in scalar interpreter")

    def execute_asm(self, asm_line: str) -> int:
        ast = parse_vop_frexp_mant_f64(asm_line)
        return self.execute(ast)


class SIMDVALUInterpreter:
    """SIMD multi-lane execution model supporting Wave32 and Wave64 with EXEC masking."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Invalid wave_size: {wave_size}. Must be 32 or 64.")
        self.wave_size = wave_size
        self.num_vgprs = 256
        self.num_sgprs = 106
        self.vgpr = [[0] * self.wave_size for _ in range(self.num_vgprs)]
        self.sgpr = [0] * self.num_sgprs
        self.vcc = 0
        self.exec = (1 << self.wave_size) - 1
        self.status_flags = FPStatusFlags()

    def reset(self) -> None:
        self.vgpr = [[0] * self.wave_size for _ in range(self.num_vgprs)]
        self.sgpr = [0] * self.num_sgprs
        self.vcc = 0
        self.exec = (1 << self.wave_size) - 1
        self.status_flags = FPStatusFlags()

    def get_vgpr_u32(self, reg: int, lane: int) -> int:
        if not (0 <= reg < self.num_vgprs):
            raise IndexError(f"VGPR index {reg} out of range [0, {self.num_vgprs})")
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size})")
        return self.vgpr[reg][lane] & MASK32

    def set_vgpr_u32(self, reg: int, lane: int, val: int) -> None:
        if not (0 <= reg < self.num_vgprs):
            raise IndexError(f"VGPR index {reg} out of range [0, {self.num_vgprs})")
        if not (0 <= lane < self.wave_size):
            raise IndexError(f"Lane {lane} out of range [0, {self.wave_size})")
        self.vgpr[reg][lane] = val & MASK32

    def get_vgpr_u64(self, reg_base: int, lane: int) -> int:
        if not (0 <= reg_base < self.num_vgprs - 1):
            raise IndexError(f"64-bit VGPR base {reg_base} out of range [0, {self.num_vgprs - 1})")
        lo = self.get_vgpr_u32(reg_base, lane)
        hi = self.get_vgpr_u32(reg_base + 1, lane)
        return join_dwords(lo, hi)

    def set_vgpr_u64(self, reg_base: int, lane: int, val64: int) -> None:
        if not (0 <= reg_base < self.num_vgprs - 1):
            raise IndexError(f"64-bit VGPR base {reg_base} out of range [0, {self.num_vgprs - 1})")
        lo, hi = split_dwords(val64)
        self.set_vgpr_u32(reg_base, lane, lo)
        self.set_vgpr_u32(reg_base + 1, lane, hi)

    def get_sgpr_u32(self, reg: int) -> int:
        if not (0 <= reg < self.num_sgprs):
            raise IndexError(f"SGPR index {reg} out of range [0, {self.num_sgprs})")
        return self.sgpr[reg] & MASK32

    def set_sgpr_u32(self, reg: int, val: int) -> None:
        if not (0 <= reg < self.num_sgprs):
            raise IndexError(f"SGPR index {reg} out of range [0, {self.num_sgprs})")
        self.sgpr[reg] = val & MASK32

    def get_sgpr_u64(self, reg_base: int) -> int:
        if not (0 <= reg_base < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {reg_base} out of range [0, {self.num_sgprs - 1})")
        lo = self.get_sgpr_u32(reg_base)
        hi = self.get_sgpr_u32(reg_base + 1)
        return join_dwords(lo, hi)

    def set_sgpr_u64(self, reg_base: int, val64: int) -> None:
        if not (0 <= reg_base < self.num_sgprs - 1):
            raise IndexError(f"64-bit SGPR base {reg_base} out of range [0, {self.num_sgprs - 1})")
        lo, hi = split_dwords(val64)
        self.set_sgpr_u32(reg_base, lo)
        self.set_sgpr_u32(reg_base + 1, hi)

    def eval_operand_f64_lane(self, op: Operand, lane: int) -> int:
        if op.kind == OperandKind.VGPR_PAIR:
            val = self.get_vgpr_u64(op.reg_index, lane)
        elif op.kind == OperandKind.SGPR_PAIR:
            val = self.get_sgpr_u64(op.reg_index)
        elif op.kind == OperandKind.LITERAL_F64:
            val = op.literal_val & MASK64
        else:
            raise ValueError(f"Unsupported operand kind for f64: {op.kind}")

        if op.abs_mod:
            val &= ~SIGN_MASK_F64
        if op.neg_mod:
            val ^= SIGN_MASK_F64
        return val

    def execute(self, ast: InstructionAST) -> list[int]:
        ast.validate()
        results: list[int] = []
        effective_exec = self.exec & ((1 << self.wave_size) - 1)

        src0_latched = [0] * self.wave_size
        for lane in range(self.wave_size):
            if (effective_exec >> lane) & 1:
                src0_latched[lane] = self.eval_operand_f64_lane(ast.src0, lane)

        for lane in range(self.wave_size):
            if not ((effective_exec >> lane) & 1):
                results.append(self.get_vgpr_u64(ast.dst.reg_index, lane))
                continue

            res = ref_frexp_mant_f64(
                src0_latched[lane],
                clamp=ast.clamp,
                omod=ast.omod,
            )
            flags = calc_frexp_mant_f64_status_flags(src0_latched[lane], res)
            self.status_flags.merge(flags)
            self.set_vgpr_u64(ast.dst.reg_index, lane, res)
            results.append(res)

        return results

    def execute_asm(self, asm_line: str) -> list[int]:
        ast = parse_vop_frexp_mant_f64(asm_line)
        return self.execute(ast)


# --- Mutated Oracles for Mutation Testing and Negative Controls ---

def mutant_exponent_off_by_one(src0: int) -> int:
    """Mutant A: uses biased exponent 1023 ([1.0, 2.0)) instead of 1022 ([0.5, 1.0))."""
    res = ref_frexp_mant_f64(src0)
    if is_zero_f64(res) or is_nan_f64(res) or is_inf_f64(res):
        return res
    # Force exponent 1023
    sign = res & SIGN_MASK_F64
    mant = res & MANT_MASK_F64
    return sign | (1023 << 52) | mant


def mutant_no_subnormal_normalization(src0: int) -> int:
    """Mutant B: skips normalization of subnormal mantissa."""
    src0 &= MASK64
    exp_field = (src0 >> 52) & 0x7FF
    if exp_field == 0 and (src0 & MANT_MASK_F64) != 0:
        # Erroneously uses raw mantissa without shifting
        return (src0 & SIGN_MASK_F64) | (FREXP_EXP_F64 << 52) | (src0 & MANT_MASK_F64)
    return ref_frexp_mant_f64(src0)


def mutant_no_clamp(src0: int) -> int:
    """Mutant C: ignores clamp modifier."""
    return ref_frexp_mant_f64(src0, clamp=False)


def mutant_leak_vcc(interp: ScalarVALUInterpreter, ast: InstructionAST) -> None:
    """Mutant D: corrupts VCC register during VALU FP execution."""
    interp.execute(ast)
    interp.vcc ^= 0x1


def mutant_exec_leak(sim: SIMDVALUInterpreter, ast: InstructionAST) -> None:
    """Mutant E: writes inactive lanes under EXEC mask."""
    sim.execute(ast)
    sim.set_vgpr_u64(ast.dst.reg_index, 0, 0x1234567890ABCDEF)


class TestValuFrexpMantF64NegativeControlsAndMutations(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for v_frexp_mant_f64."""

    def test_syntax_rejection_invalid_opcode(self):
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_add_f64 v[0:1], v[2:3]")

    def test_syntax_rejection_insufficient_operands(self):
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1]")

    def test_syntax_rejection_non_contiguous_dst(self):
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:2], v[2:3]")

    def test_syntax_rejection_non_contiguous_src0(self):
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], v[2:4]")

    def test_syntax_rejection_out_of_range_registers(self):
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[255:256], v[2:3]")
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], s[105:106]")

    def test_syntax_rejection_invalid_modifier(self):
        with self.assertRaises(ValueError):
            ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
            ast.omod = "mul:3"
            ast.validate()

    def test_syntax_rejection_invalid_types(self):
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_frexp_mant_f64 v0, v[2:3]")
        with self.assertRaises(ValueError):
            parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], v2")

    def test_mutation_detection_exponent_off_by_one(self):
        correct = ref_frexp_mant_f64(ONE_F64)
        mutant = mutant_exponent_off_by_one(ONE_F64)
        self.assertEqual(bits_to_f64(correct), 0.5)
        self.assertEqual(bits_to_f64(mutant), 1.0)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_no_subnormal_normalization(self):
        correct = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        mutant = mutant_no_subnormal_normalization(MIN_SUBNORMAL_F64)
        self.assertEqual(bits_to_f64(correct), 0.5)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_no_clamp(self):
        val = f64_to_bits(-2.0)
        correct_clamped = ref_frexp_mant_f64(val, clamp=True)
        mutant = mutant_no_clamp(val)
        self.assertEqual(bits_to_f64(correct_clamped), 0.0)
        self.assertEqual(bits_to_f64(mutant), -0.5)
        self.assertNotEqual(correct_clamped, mutant)

    def test_mutation_detection_corrupt_vcc(self):
        interp = ScalarVALUInterpreter()
        interp.vcc = 0x5555555555555555
        interp.set_vgpr_u64(2, ONE_F64)
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
        interp.execute(ast)
        self.assertEqual(interp.vcc, 0x5555555555555555)
        mutant_leak_vcc(interp, ast)
        self.assertNotEqual(interp.vcc, 0x5555555555555555)

    def test_mutation_detection_exec_mask_leak(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        sim.exec = 0x0
        sim.set_vgpr_u64(0, 0, 0xAAAAAAAAAAAAAAAA)
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
        sim.execute(ast)
        self.assertEqual(sim.get_vgpr_u64(0, 0), 0xAAAAAAAAAAAAAAAA)
        mutant_exec_leak(sim, ast)
        self.assertNotEqual(sim.get_vgpr_u64(0, 0), 0xAAAAAAAAAAAAAAAA)


class TestValuFrexpMantF64IEEE754Boundaries(unittest.TestCase):
    """Deep verification of IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_normalization_extremes(self):
        # MIN_SUBNORMAL_F64 (2^-1074) -> mantissa is 0.5
        res_min = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        self.assertEqual(bits_to_f64(res_min), 0.5)
        self.assertEqual(res_min, HALF_F64)

        # Negative MIN_SUBNORMAL -> -0.5
        res_neg_min = ref_frexp_mant_f64(MIN_SUBNORMAL_F64 | SIGN_MASK_F64)
        self.assertEqual(bits_to_f64(res_neg_min), -0.5)

        # MAX_SUBNORMAL_F64 (all 52 mantissa bits set, exp=0)
        # Normalized: implicit 1 at bit 52, leaving 51 ones and lowest bit 0
        res_max = ref_frexp_mant_f64(MAX_SUBNORMAL_F64)
        self.assertEqual(res_max, (FREXP_EXP_F64 << 52) | (MANT_MASK_F64 ^ 1))
        self.assertAlmostEqual(bits_to_f64(res_max), 1.0 - (2.0 ** -52), places=15)

    def test_normal_boundary_extremes(self):
        # MIN_NORMAL_F64 (2^-1022) -> 0.5
        res_min_norm = ref_frexp_mant_f64(MIN_NORMAL_F64)
        self.assertEqual(bits_to_f64(res_min_norm), 0.5)

        # MAX_FINITE_F64 -> mantissa is 1.0 - 2^-53
        res_max_finite = ref_frexp_mant_f64(MAX_FINITE_F64)
        self.assertEqual(res_max_finite, (FREXP_EXP_F64 << 52) | MANT_MASK_F64)

    def test_signed_zero_preservation(self):
        self.assertEqual(ref_frexp_mant_f64(POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64), NEG_ZERO_F64)
        # With abs: -0.0 becomes +0.0
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64, abs_src0=True), POS_ZERO_F64)
        # With neg: +0.0 becomes -0.0
        self.assertEqual(ref_frexp_mant_f64(POS_ZERO_F64, neg_src0=True), NEG_ZERO_F64)
        # With clamp: -0.0 becomes +0.0
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64, clamp=True), POS_ZERO_F64)

    def test_signed_infinity_preservation(self):
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64), NEG_INF_F64)
        # With clamp: +inf -> 1.0, -inf -> +0.0
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64, clamp=True), ONE_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64, clamp=True), POS_ZERO_F64)

    def test_nan_payloads_and_quietening(self):
        qnan_custom = QNAN_F64 | 0x00000000CAFEBABE
        res_qnan = ref_frexp_mant_f64(qnan_custom)
        self.assertTrue(is_nan_f64(res_qnan))
        self.assertEqual(res_qnan & 0x00000000CAFEBABE, 0x00000000CAFEBABE)

        snan_custom = SNAN_F64 | 0x0000000012345678
        res_snan = ref_frexp_mant_f64(snan_custom)
        self.assertTrue(is_nan_f64(res_snan))
        self.assertFalse(is_snan_f64(res_snan), "Signaling NaN must be quieted")

    def test_clamp_behavior_on_all_special_values(self):
        self.assertEqual(ref_frexp_mant_f64(QNAN_F64, clamp=True), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(SNAN_F64, clamp=True), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64, clamp=True), ONE_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64, clamp=True), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64, clamp=True), POS_ZERO_F64)
        # Negative normal clamps to +0.0
        self.assertEqual(ref_frexp_mant_f64(f64_to_bits(-5.0), clamp=True), POS_ZERO_F64)
        # Positive normal [0.5, 1.0) stays untouched
        f_pos = f64_to_bits(1.5)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_pos, clamp=True)), 0.75)


class TestValuFrexpMantF64StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow behavior."""

    def test_vcc_strictly_preserved_scalar(self):
        interp = ScalarVALUInterpreter()
        initial_vcc = 0x123456789ABCDEF0
        interp.vcc = initial_vcc
        interp.set_vgpr_u64(2, f64_to_bits(1.0))
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertEqual(interp.vcc, initial_vcc, "VCC must remain unmodified by v_frexp_mant_f64")

    def test_vcc_strictly_preserved_simd_wave64(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        initial_vcc = 0xCAFEBABE0000FFFF
        sim.vcc = initial_vcc
        for lane in range(64):
            sim.set_vgpr_u64(2, lane, f64_to_bits(2.0))
        sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertEqual(sim.vcc, initial_vcc, "Wave64 VCC must remain untouched")

    def test_fp_flags_exact_normal_no_exceptions(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, f64_to_bits(1.0))
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertFalse(interp.status_flags.invalid)
        self.assertFalse(interp.status_flags.overflow)
        self.assertFalse(interp.status_flags.underflow)
        self.assertFalse(interp.status_flags.inexact)

    def test_fp_flags_invalid_snan(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, SNAN_F64)
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertTrue(interp.status_flags.invalid)

    def test_fp_flags_no_invalid_on_qnan(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, QNAN_F64)
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertFalse(interp.status_flags.invalid)

    def test_fp_flags_no_overflow_underflow_on_extremes(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, MAX_FINITE_F64)
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertFalse(interp.status_flags.overflow)
        self.assertFalse(interp.status_flags.underflow)


class TestValuFrexpMantF64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_full_aliasing_dst_src0_scalar(self):
        interp = ScalarVALUInterpreter()
        # In-place mantissa: v[0:1] = v_frexp_mant_f64 v[0:1]
        interp.set_vgpr_u64(0, f64_to_bits(14.0))  # 14 = 0.875 * 2^4
        res = interp.execute_asm("v_frexp_mant_f64 v[0:1], v[0:1]")
        self.assertEqual(bits_to_f64(res), 0.875)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), 0.875)

    def test_partial_overlap_dst_hi_src0_lo_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[1:2] = v_frexp_mant_f64 v[0:1]
        interp.set_vgpr_u64(0, f64_to_bits(6.0))  # 6 = 0.75 * 2^3
        res = interp.execute_asm("v_frexp_mant_f64 v[1:2], v[0:1]")
        self.assertEqual(bits_to_f64(res), 0.75)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(1)), 0.75)

    def test_partial_overlap_dst_lo_src0_hi_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[0:1] = v_frexp_mant_f64 v[1:2]
        interp.set_vgpr_u64(1, f64_to_bits(-10.0))  # -10 = -0.625 * 2^4
        res = interp.execute_asm("v_frexp_mant_f64 v[0:1], v[1:2]")
        self.assertEqual(bits_to_f64(res), -0.625)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), -0.625)

    def test_simd_wave64_aliasing_with_exec_mask(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0xABABABABCDCDEFEF
        for lane in range(64):
            if lane % 2 == 0:
                sim.set_vgpr_u64(0, lane, f64_to_bits(4.0))
            else:
                sim.set_vgpr_u64(0, lane, canary)

        sim.exec = 0x5555555555555555
        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[0:1]")

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(bits_to_f64(results[lane]), 0.5)
                self.assertEqual(bits_to_f64(sim.get_vgpr_u64(0, lane)), 0.5)
            else:
                self.assertEqual(results[lane], canary)

    def test_simd_wave32_partial_overlap_divergent(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        for lane in range(32):
            sim.set_vgpr_u64(1, lane, f64_to_bits(1.0 + lane))

        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[1:2]")
        for lane in range(32):
            m, _ = math.frexp(1.0 + lane)
            self.assertEqual(bits_to_f64(results[lane]), m)


class TestValuFrexpMantF64SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        base = f64_to_bits(12.0)
        for lane in range(32):
            sim.set_vgpr_u64(2, lane, base)

        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(bits_to_f64(results[lane]), 0.75)
            self.assertEqual(bits_to_f64(sim.get_vgpr_u64(0, lane)), 0.75)

    def test_wave64_uniform_execution(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        base = f64_to_bits(3.0)
        for lane in range(64):
            sim.set_vgpr_u64(6, lane, base)

        results = sim.execute_asm("v_frexp_mant_f64 v[4:5], v[6:7]")
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertEqual(bits_to_f64(results[lane]), 0.75)

    def test_wave32_divergent_lanes(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        expected: list[float] = []
        for lane in range(32):
            val = float(lane + 1)
            sim.set_vgpr_u64(0, lane, f64_to_bits(val))
            m, _ = math.frexp(val)
            expected.append(m)

        results = sim.execute_asm("v_frexp_mant_f64 v[4:5], v[0:1]")
        for lane in range(32):
            self.assertEqual(bits_to_f64(results[lane]), expected[lane])

    def test_wave64_divergent_lanes(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        expected: list[float] = []
        for lane in range(64):
            val = -0.125 * (lane + 1)
            sim.set_vgpr_u64(2, lane, f64_to_bits(val))
            m, _ = math.frexp(val)
            expected.append(m)

        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        for lane in range(64):
            self.assertEqual(bits_to_f64(results[lane]), expected[lane])

    def test_exec_mask_partial_inactivity_wave32(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        canary = 0xDEADBEEFCAFEBABE
        for lane in range(32):
            sim.set_vgpr_u64(0, lane, canary)
            sim.set_vgpr_u64(2, lane, f64_to_bits(4.0))

        sim.exec = 0x55555555
        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(bits_to_f64(results[lane]), 0.5)
            else:
                self.assertEqual(results[lane], canary)

    def test_exec_mask_partial_inactivity_wave64(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0x1122334455667788
        for lane in range(64):
            sim.set_vgpr_u64(0, lane, canary)
            sim.set_vgpr_u64(2, lane, f64_to_bits(8.0))

        sim.exec = 0x00000000FFFFFFFF
        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")

        for lane in range(64):
            if lane < 32:
                self.assertEqual(bits_to_f64(results[lane]), 0.5)
            else:
                self.assertEqual(results[lane], canary)

    def test_exec_mask_all_inactive(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0xCAFEBABE12345678
        for lane in range(64):
            sim.set_vgpr_u64(0, lane, canary)
            sim.set_vgpr_u64(2, lane, f64_to_bits(1.0))

        sim.exec = 0x0
        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        for lane in range(64):
            self.assertEqual(results[lane], canary)


class TestValuFrexpMantF64ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self):
        self.interp = ScalarVALUInterpreter()

    def test_scalar_interpreter_vgpr_sources(self):
        # v[2:3] = 6.0 = 0.75 * 2^3 -> mantissa is 0.75
        self.interp.set_vgpr_u64(2, f64_to_bits(6.0))
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 0.75)
        self.assertEqual(self.interp.get_vgpr_u64(0), res)

    def test_scalar_interpreter_sgpr_sources(self):
        # s[0:1] = -12.0 -> mantissa is -0.75
        self.interp.set_sgpr_u64(0, f64_to_bits(-12.0))
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[4:5], s[0:1]")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), -0.75)

    def test_scalar_interpreter_literal_operand(self):
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[2:3], 32.0")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 0.5)

    def test_scalar_interpreter_input_modifiers(self):
        # -|v[2:3]| with v[2:3] = 2.0 -> mantissa is -0.5
        self.interp.set_vgpr_u64(2, f64_to_bits(2.0))
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], -|v[2:3]|")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), -0.5)

    def test_scalar_interpreter_output_modifiers_clamp_omod(self):
        # 1.0 -> mantissa 0.5, with mul:2 -> 1.0, clamp -> 1.0
        self.interp.set_vgpr_u64(0, f64_to_bits(1.0))
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[4:5], v[0:1] clamp mul:2")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 1.0)


class TestValuFrexpMantF64SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        dst = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=0)
        src0 = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=2)
        ast = InstructionAST("v_frexp_mant_f64", dst, src0, clamp=True, omod="mul:2")
        ast.validate()
        self.assertEqual(ast.to_asm(), "v_frexp_mant_f64 v[0:1], v[2:3] clamp mul:2")

    def test_parser_vgpr_sgpr_operands(self):
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[4:5], v[6:7]")
        self.assertEqual(ast.dst.reg_index, 4)
        self.assertEqual(ast.src0.kind, OperandKind.VGPR_PAIR)
        self.assertEqual(ast.src0.reg_index, 6)
        self.assertFalse(ast.clamp)
        self.assertEqual(ast.omod, "none")

        ast_sgpr = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], s[2:3]")
        self.assertEqual(ast_sgpr.src0.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(ast_sgpr.src0.reg_index, 2)

    def test_parser_literal_operand(self):
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], 1.5")
        self.assertEqual(ast.src0.kind, OperandKind.LITERAL_F64)
        self.assertEqual(bits_to_f64(ast.src0.literal_val), 1.5)

    def test_parser_modifiers_abs_neg_clamp_omod(self):
        ast = parse_vop_frexp_mant_f64("v_frexp_mant_f64 v[0:1], -|v[2:3]| clamp mul:4")
        self.assertTrue(ast.src0.abs_mod)
        self.assertTrue(ast.src0.neg_mod)
        self.assertTrue(ast.clamp)
        self.assertEqual(ast.omod, "mul:4")
        self.assertEqual(ast.to_asm(), "v_frexp_mant_f64 v[0:1], -|v[2:3]| clamp mul:4")

    def test_ast_type_checking(self):
        bad_dst = Operand(OperandKind.SGPR_PAIR, DataType.F64, reg_index=0)
        src0 = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=2)
        with self.assertRaises(TypeError):
            InstructionAST("v_frexp_mant_f64", bad_dst, src0).validate()


class TestValuFrexpMantF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_normal_values(self):
        # 1.0 = 0.5 * 2^1 -> mantissa is 0.5
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64)), 0.5)
        # 1.5 = 0.75 * 2^1 -> mantissa is 0.75
        f_1p5 = f64_to_bits(1.5)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_1p5)), 0.75)
        # -4.0 = -0.5 * 2^3 -> mantissa is -0.5
        f_neg4 = f64_to_bits(-4.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg4)), -0.5)

    def test_oracle_powers_of_two(self):
        for p in range(-50, 50):
            val = math.ldexp(1.0, p)
            val_bits = f64_to_bits(val)
            self.assertEqual(bits_to_f64(ref_frexp_mant_f64(val_bits)), 0.5)

    def test_oracle_cross_check_math_frexp(self):
        test_floats = [
            0.125, 0.3333333333333333, 1.0, 2.5, 42.0, 1000.125,
            -7.5, -0.0625, 1.7976931348623157e+308, 2.2250738585072014e-308
        ]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            oracle_res = ref_frexp_mant_f64(f_bits)
            math_res = ref_frexp_mant_f64_math(f_bits)
            self.assertEqual(oracle_res, math_res)

    def test_oracle_cross_check_fraction(self):
        for f in [1.5, 3.25, -0.75, 10.125, 0.03125]:
            f_bits = f64_to_bits(f)
            self.assertEqual(ref_frexp_mant_f64(f_bits), ref_frexp_mant_f64_fraction(f_bits))

    def test_oracle_special_values(self):
        # +0.0 and -0.0 preserved
        self.assertEqual(ref_frexp_mant_f64(POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64), NEG_ZERO_F64)
        # +inf and -inf preserved
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64), NEG_INF_F64)
        # NaNs preserved and quieted
        self.assertTrue(is_nan_f64(ref_frexp_mant_f64(QNAN_F64)))
        self.assertTrue(is_nan_f64(ref_frexp_mant_f64(SNAN_F64)))

    def test_oracle_subnormal_normalization(self):
        # Smallest subnormal 2^-1074: normalized mantissa must be 0.5
        res_sub_min = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        self.assertEqual(bits_to_f64(res_sub_min), 0.5)

        # Negative subnormal: normalized mantissa is -0.5
        res_neg_sub = ref_frexp_mant_f64(MIN_SUBNORMAL_F64 | SIGN_MASK_F64)
        self.assertEqual(bits_to_f64(res_neg_sub), -0.5)

    def test_oracle_modifiers(self):
        # abs modifier
        f_neg2 = f64_to_bits(-2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg2, abs_src0=True)), 0.5)
        # neg modifier
        f_pos2 = f64_to_bits(2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_pos2, neg_src0=True)), -0.5)
        # clamp modifier
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg2, clamp=True)), 0.0)
        # omod modifiers
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:2")), 1.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:4")), 2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="div:2")), 0.25)


if __name__ == "__main__":
    unittest.main()
