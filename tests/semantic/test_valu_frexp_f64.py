from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_frexp_f64.

Instructions:
  1. v_frexp_mant_f64 dst, src0 (VOP1 e32 / VOP3 e64)
     D.f64 = Mantissa(S0.f64)
     Decomposes S0 such that S0 = mantissa * (2 ^ exponent) with |mantissa| in [0.5, 1.0).
     For normal inputs, the IEEE 754 11-bit biased exponent field [62:52] is replaced with 1022
     (0x3FE, representing 2^-1), preserving the original sign bit (bit 63) and 52-bit fraction (bits [51:0]).
     Special values:
       * +/-0.0: returns +/-0.0 (sign bit preserved).
       * +/-INF: returns +/-INF unchanged.
       * NaN: returns NaN unchanged (quiet / signaling payload bits preserved).
       * Subnormals: normalized into [0.5, 1.0) with exponent set to 1022.

  2. v_frexp_exp_i32_f64 dst, src0 (VOP1 e32 / VOP3 e64)
     D.i32 = Exponent(S0.f64)
     Extracts the unbiased integer base-2 exponent of S0, returned as a signed 32-bit integer.
     For normal inputs, computes (biased_exponent - 1022).
     Special values:
       * +/-0.0: returns 0.
       * +/-INF: returns 0.
       * NaN: returns 0.
       * Subnormals: returns -1022 - leading_zeros.

Architecture:
  - RDNA2 Vector ALU (GFX10) double-precision floating-point extraction.
  - S0: 64-bit IEEE 754 binary64 operand (VGPR pair, SGPR pair, or literal / inline constant).
  - D (mantissa): 64-bit IEEE 754 binary64 destination (VGPR pair).
  - D (exponent): 32-bit signed integer destination (single VGPR).
  - Modifiers:
      * Input modifiers on S0: abs (|S0|), neg (-S0).
      * Output modifiers on mantissa: clamp (clamps to [0.0, 1.0]), omod (mul:2, mul:4, div:2).
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
EXP_SHIFT_F64 = 52

EXP_FREXP_BIASED_F64 = 1022  # 2^-1 has biased exponent 1022 in IEEE 754 double precision
EXP_FREXP_BITS_F64 = EXP_FREXP_BIASED_F64 << EXP_SHIFT_F64  # 0x3FE0000000000000 (0.5)

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


def is_snan_f64(bits: int) -> bool:
    """Check if binary64 bits represent a signaling NaN (quiet bit 51 is zero)."""
    if not is_nan_f64(bits):
        return False
    return ((bits >> 51) & 1) == 0


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
    exp = (bits & EXP_MASK_F64) >> EXP_SHIFT_F64
    return 1 <= exp <= 2046


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


def calc_frexp_f64_status_flags(src0_bits: int, opcode: str) -> FPStatusFlags:
    """Determine IEEE 754 exception flags for v_frexp_mant_f64 / v_frexp_exp_i32_f64."""
    flags = FPStatusFlags()
    # Signaling NaN generates invalid operation exception
    if is_snan_f64(src0_bits):
        flags.invalid = True
    return flags


def ref_frexp_mant_f64(
    src0_bits: int,
    *,
    clamp: bool = False,
    abs_src0: bool = False,
    neg_src0: bool = False,
    omod: str = "none",
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_frexp_mant_f64.

    Extracts significand normalized into [0.5, 1.0) with original sign preserved.
    """
    src0_bits &= MASK64
    if abs_src0:
        src0_bits &= ~SIGN_MASK_F64
    if neg_src0:
        src0_bits ^= SIGN_MASK_F64

    sign = src0_bits & SIGN_MASK_F64
    exp_field = (src0_bits >> EXP_SHIFT_F64) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    # 1. NaN handling
    if exp_field == 0x7FF and mant_field != 0:
        if clamp:
            return POS_ZERO_F64
        # Quiet signaling NaN if needed, preserve payload
        return src0_bits | 0x0008000000000000

    # 2. Infinity handling
    if exp_field == 0x7FF and mant_field == 0:
        if clamp:
            return POS_ZERO_F64 if sign != 0 else ONE_F64
        return src0_bits

    # 3. Zero handling
    if exp_field == 0 and mant_field == 0:
        if clamp and sign != 0:
            return POS_ZERO_F64
        return src0_bits

    # 4. Subnormal handling: normalized into [0.5, 1.0)
    if exp_field == 0 and mant_field != 0:
        lz = 52 - mant_field.bit_length()
        norm_mant = (mant_field << (lz + 1)) & MANT_MASK_F64
        res_bits = sign | EXP_FREXP_BITS_F64 | norm_mant
    else:
        # 5. Normal float: replace exponent with 1022 (0x3FE)
        res_bits = sign | EXP_FREXP_BITS_F64 | mant_field

    # Output modifiers
    if omod != "none":
        val = bits_to_f64(res_bits)
        if omod == "mul:2":
            val *= 2.0
        elif omod == "mul:4":
            val *= 4.0
        elif omod == "div:2":
            val *= 0.5
        res_bits = f64_to_bits(val)

    if clamp:
        val = bits_to_f64(res_bits)
        if math.isnan(val) or val < 0.0:
            res_bits = POS_ZERO_F64
        elif val > 1.0:
            res_bits = ONE_F64

    return res_bits


def ref_frexp_exp_i32_f64(
    src0_bits: int,
    *,
    abs_src0: bool = False,
    neg_src0: bool = False,
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_frexp_exp_i32_f64.

    Extracts unbiased exponent as signed 32-bit two's-complement integer.
    Special values (+/-0, +/-inf, NaN) return 0.
    """
    src0_bits &= MASK64
    if abs_src0:
        src0_bits &= ~SIGN_MASK_F64
    if neg_src0:
        src0_bits ^= SIGN_MASK_F64

    exp_field = (src0_bits >> EXP_SHIFT_F64) & 0x7FF
    mant_field = src0_bits & MANT_MASK_F64

    # Special values: NaN, Inf, Zero return 0
    if exp_field == 0x7FF or (exp_field == 0 and mant_field == 0):
        return 0

    # Subnormal
    if exp_field == 0:
        lz = 52 - mant_field.bit_length()
        unbiased = -1022 - lz
        return u32(unbiased)

    # Normal float: unbiased = exp_field - 1022
    unbiased = exp_field - 1022
    return u32(unbiased)


def ref_frexp_reconstruct_f64(src0_bits: int) -> float:
    """Reconstruct S0 = mantissa * 2^exponent from frexp decomposition."""
    mant_bits = ref_frexp_mant_f64(src0_bits)
    exp_i32 = s32(ref_frexp_exp_i32_f64(src0_bits))
    m = bits_to_f64(mant_bits)
    return math.ldexp(m, exp_i32)


class DataType(Enum):
    """Data types supported by RDNA2 VALU instructions."""
    F64 = auto()
    I32 = auto()
    U32 = auto()
    U64 = auto()


class OperandKind(Enum):
    """Classification of instruction operands."""
    VGPR_PAIR = auto()
    VGPR_SINGLE = auto()
    SGPR_PAIR = auto()
    SGPR_SINGLE = auto()
    LITERAL_I32 = auto()
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
        if self.kind in (OperandKind.VGPR_PAIR, OperandKind.VGPR_SINGLE):
            limit = 255 if self.kind == OperandKind.VGPR_PAIR else 256
            if not (0 <= self.reg_index < limit):
                raise ValueError(f"VGPR index {self.reg_index} out of valid range [0, {limit})")
        elif self.kind in (OperandKind.SGPR_PAIR, OperandKind.SGPR_SINGLE):
            limit = 105 if self.kind == OperandKind.SGPR_PAIR else 106
            if not (0 <= self.reg_index < limit):
                raise ValueError(f"SGPR index {self.reg_index} out of valid range [0, {limit})")

    def to_asm(self) -> str:
        s = ""
        if self.kind == OperandKind.VGPR_PAIR:
            s = f"v[{self.reg_index}:{self.reg_index + 1}]"
        elif self.kind == OperandKind.VGPR_SINGLE:
            s = f"v{self.reg_index}"
        elif self.kind == OperandKind.SGPR_PAIR:
            s = f"s[{self.reg_index}:{self.reg_index + 1}]"
        elif self.kind == OperandKind.SGPR_SINGLE:
            s = f"s{self.reg_index}"
        elif self.kind == OperandKind.LITERAL_I32:
            s = str(s32(self.literal_val))
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
    """Symbolic AST node for v_frexp_mant_f64 and v_frexp_exp_i32_f64."""
    opcode: str
    dst: Operand
    src0: Operand
    clamp: bool = False
    omod: str = "none"

    def validate(self) -> None:
        if self.opcode not in ("v_frexp_mant_f64", "v_frexp_exp_i32_f64"):
            raise ValueError(f"Unsupported opcode {self.opcode}")

        if self.opcode == "v_frexp_mant_f64":
            if self.dst.kind != OperandKind.VGPR_PAIR or self.dst.data_type != DataType.F64:
                raise TypeError("v_frexp_mant_f64 destination must be a 64-bit VGPR pair (DataType.F64)")
        elif self.opcode == "v_frexp_exp_i32_f64":
            if self.dst.kind != OperandKind.VGPR_SINGLE or self.dst.data_type != DataType.I32:
                raise TypeError("v_frexp_exp_i32_f64 destination must be a 32-bit single VGPR (DataType.I32)")
            if self.clamp:
                raise ValueError("v_frexp_exp_i32_f64 does not support clamp modifier")
            if self.omod != "none":
                raise ValueError("v_frexp_exp_i32_f64 does not support omod modifier")

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


def parse_valu_frexp_f64(asm_line: str) -> InstructionAST:
    """Parse assembly string for v_frexp_mant_f64 or v_frexp_exp_i32_f64 into InstructionAST."""
    line = asm_line.strip()
    m_op = re.match(r"^(v_frexp_mant_f64|v_frexp_exp_i32_f64)(?:_e32|_e64)?\b", line)
    if not m_op:
        raise ValueError(f"Expected instruction starting with v_frexp_mant_f64 or v_frexp_exp_i32_f64, got: {line}")

    opcode = m_op.group(1)
    remainder = line[m_op.end():].strip()

    tokens = remainder.split()
    operands_tokens = []
    clamp = False
    omod = "none"

    for tok in tokens:
        if tok == "clamp":
            clamp = True
        elif tok in ("mul:2", "mul:4", "div:2"):
            omod = tok
        else:
            operands_tokens.append(tok)

    op_str = " ".join(operands_tokens)
    parts = [p.strip() for p in op_str.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected 2 operands (dst, src0), got {len(parts)} in '{line}'")

    dst_str, src0_str = parts[0], parts[1]

    # Parse DST
    m_vpair = re.match(r"^v\[(\d+):(\d+)\]$", dst_str)
    m_vsingle = re.match(r"^v(\d+)$", dst_str)

    if opcode == "v_frexp_mant_f64":
        if not m_vpair:
            raise ValueError(f"v_frexp_mant_f64 destination must be VGPR pair v[lo:hi], got: {dst_str}")
        lo, hi = int(m_vpair.group(1)), int(m_vpair.group(2))
        if hi != lo + 1:
            raise ValueError(f"Destination register pair must be contiguous: {dst_str}")
        dst_op = Operand(kind=OperandKind.VGPR_PAIR, data_type=DataType.F64, reg_index=lo)
    else:  # v_frexp_exp_i32_f64
        if m_vpair:
            raise ValueError(f"v_frexp_exp_i32_f64 destination must be single VGPR, got pair: {dst_str}")
        if not m_vsingle:
            raise ValueError(f"v_frexp_exp_i32_f64 destination must be VGPR vN, got: {dst_str}")
        dst_op = Operand(kind=OperandKind.VGPR_SINGLE, data_type=DataType.I32, reg_index=int(m_vsingle.group(1)))

    # Parse SRC0
    neg_s0 = False
    abs_s0 = False
    s0_clean = src0_str
    if s0_clean.startswith("-"):
        neg_s0 = True
        s0_clean = s0_clean[1:].strip()
    if s0_clean.startswith("|") and s0_clean.endswith("|"):
        abs_s0 = True
        s0_clean = s0_clean[1:-1].strip()
    if s0_clean.startswith("-"):
        neg_s0 = not neg_s0
        s0_clean = s0_clean[1:].strip()

    m_v0 = re.match(r"^v\[(\d+):(\d+)\]$", s0_clean)
    m_s0 = re.match(r"^s\[(\d+):(\d+)\]$", s0_clean)
    if m_v0:
        lo, hi = int(m_v0.group(1)), int(m_v0.group(2))
        if hi != lo + 1:
            raise ValueError(f"Source register pair must be contiguous: {src0_str}")
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
            raise ValueError(f"Source register pair must be contiguous: {src0_str}")
        src0_op = Operand(
            kind=OperandKind.SGPR_PAIR,
            data_type=DataType.F64,
            reg_index=lo,
            abs_mod=abs_s0,
            neg_mod=neg_s0,
        )
    else:
        try:
            if s0_clean.startswith("0x") or s0_clean.startswith("0X"):
                val64 = int(s0_clean, 16)
            else:
                val64 = f64_to_bits(float(s0_clean))
            src0_op = Operand(
                kind=OperandKind.LITERAL_F64,
                data_type=DataType.F64,
                literal_val=val64,
                abs_mod=abs_s0,
                neg_mod=neg_s0,
            )
        except ValueError:
            raise ValueError(f"Unrecognized src0 operand format: {src0_str}")

    ast = InstructionAST(
        opcode=opcode,
        dst=dst_op,
        src0=src0_op,
        clamp=clamp,
        omod=omod,
    )
    ast.validate()
    return ast


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
        s0_val = self.eval_operand_f64(ast.src0)

        flags = calc_frexp_f64_status_flags(s0_val, ast.opcode)
        self.status_flags.merge(flags)

        if ast.opcode == "v_frexp_mant_f64":
            res = ref_frexp_mant_f64(
                s0_val,
                clamp=ast.clamp,
                omod=ast.omod,
            )
            self.set_vgpr_u64(ast.dst.reg_index, res)
            return res
        elif ast.opcode == "v_frexp_exp_i32_f64":
            res = ref_frexp_exp_i32_f64(s0_val)
            self.set_vgpr_u32(ast.dst.reg_index, res)
            return res

        raise NotImplementedError(f"Opcode {ast.opcode} not implemented in scalar interpreter")

    def execute_asm(self, asm_line: str) -> int:
        ast = parse_valu_frexp_f64(asm_line)
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

        # Latch src0 across active lanes first
        src0_latched: list[int] = [0] * self.wave_size
        for lane in range(self.wave_size):
            if (self.exec >> lane) & 1:
                src0_latched[lane] = self.eval_operand_f64_lane(ast.src0, lane)

        for lane in range(self.wave_size):
            if not ((self.exec >> lane) & 1):
                # Inactive lane: preserve existing destination register contents
                if ast.opcode == "v_frexp_mant_f64":
                    results.append(self.get_vgpr_u64(ast.dst.reg_index, lane))
                else:
                    results.append(self.get_vgpr_u32(ast.dst.reg_index, lane))
                continue

            s0_val = src0_latched[lane]
            flags = calc_frexp_f64_status_flags(s0_val, ast.opcode)
            self.status_flags.merge(flags)

            if ast.opcode == "v_frexp_mant_f64":
                res = ref_frexp_mant_f64(
                    s0_val,
                    clamp=ast.clamp,
                    omod=ast.omod,
                )
                self.set_vgpr_u64(ast.dst.reg_index, lane, res)
            elif ast.opcode == "v_frexp_exp_i32_f64":
                res = ref_frexp_exp_i32_f64(s0_val)
                self.set_vgpr_u32(ast.dst.reg_index, lane, res)
            else:
                raise NotImplementedError(f"Opcode {ast.opcode} not implemented")

            results.append(res)

        return results

    def execute_asm(self, asm_line: str) -> list[int]:
        ast = parse_valu_frexp_f64(asm_line)
        return self.execute(ast)


# --- Mutated Oracles for Mutation Testing and Negative Controls ---

def mutant_frexp_bias_off_by_one(src0: int) -> int:
    """Mutant A: uses incorrect biased exponent 1021 instead of 1022 for 0.5f64."""
    mant_bits = ref_frexp_mant_f64(src0)
    if is_normal_f64(mant_bits):
        # Corrupt exponent field to 1021
        return (mant_bits & ~EXP_MASK_F64) | (1021 << EXP_SHIFT_F64)
    return mant_bits


def mutant_frexp_flush_subnormal(src0: int) -> int:
    """Mutant B: prematurely flushes subnormals to zero instead of normalizing."""
    if is_subnormal_f64(src0):
        return src0 & SIGN_MASK_F64
    return ref_frexp_mant_f64(src0)


def mutant_frexp_no_clamp(src0: int) -> int:
    """Mutant C: ignores clamp modifier on mantissa."""
    return ref_frexp_mant_f64(src0, clamp=False)


def mutant_frexp_exp_wrong_special(src0: int) -> int:
    """Mutant D: returns non-zero for NaN/Infinity on exponent extraction."""
    if is_nan_f64(src0) or is_inf_f64(src0):
        return 0x7FF
    return ref_frexp_exp_i32_f64(src0)


def mutant_leak_vcc(interp: ScalarVALUInterpreter, ast: InstructionAST) -> None:
    """Mutant E: improperly corrupts VCC register during VALU execution."""
    interp.execute(ast)
    interp.vcc ^= 0x1


def mutant_exec_leak(sim: SIMDVALUInterpreter, ast: InstructionAST) -> None:
    """Mutant F: writes inactive lanes under EXEC mask."""
    sim.execute(ast)
    # Overwrite lane 0 regardless of EXEC
    if ast.opcode == "v_frexp_mant_f64":
        sim.set_vgpr_u64(ast.dst.reg_index, 0, 0x1234567890ABCDEF)
    else:
        sim.set_vgpr_u32(ast.dst.reg_index, 0, 0x12345678)


class TestValuFrexpF64NegativeControlsAndMutations(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for v_frexp_f64."""

    def test_syntax_rejection_invalid_opcode(self):
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_add_f64 v[0:1], v[2:3]")

    def test_syntax_rejection_insufficient_operands(self):
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1]")

    def test_syntax_rejection_non_contiguous_dst_mant(self):
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_mant_f64 v[0:2], v[2:3]")

    def test_syntax_rejection_non_contiguous_src0(self):
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], v[2:4]")

    def test_syntax_rejection_dst_type_mismatch_mant(self):
        # Single VGPR not allowed for 64-bit mantissa
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_mant_f64 v0, v[2:3]")

    def test_syntax_rejection_dst_type_mismatch_exp(self):
        # VGPR pair not allowed for 32-bit integer exponent
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_exp_i32_f64 v[0:1], v[2:3]")

    def test_syntax_rejection_out_of_range_registers(self):
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_mant_f64 v[255:256], v[2:3]")
        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], s[105:106]")

    def test_syntax_rejection_invalid_modifier(self):
        with self.assertRaises(ValueError):
            ast = parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
            ast.omod = "mul:3"
            ast.validate()

        with self.assertRaises(ValueError):
            parse_valu_frexp_f64("v_frexp_exp_i32_f64 v0, v[2:3] clamp")

    def test_mutation_detection_bias_off_by_one(self):
        correct = ref_frexp_mant_f64(ONE_F64)
        mutant = mutant_frexp_bias_off_by_one(ONE_F64)
        self.assertEqual(bits_to_f64(correct), 0.5)
        self.assertNotEqual(correct, mutant)
        self.assertEqual((mutant >> EXP_SHIFT_F64) & 0x7FF, 1021)

    def test_mutation_detection_flush_subnormal(self):
        correct = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        mutant = mutant_frexp_flush_subnormal(MIN_SUBNORMAL_F64)
        self.assertEqual(bits_to_f64(correct), 0.5)
        self.assertEqual(mutant, POS_ZERO_F64)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_no_clamp(self):
        neg_val = f64_to_bits(-2.0)
        correct_clamped = ref_frexp_mant_f64(neg_val, clamp=True)
        mutant = mutant_frexp_no_clamp(neg_val)
        self.assertEqual(correct_clamped, POS_ZERO_F64)
        self.assertEqual(bits_to_f64(mutant), -0.5)
        self.assertNotEqual(correct_clamped, mutant)

    def test_mutation_detection_exp_special_values(self):
        correct = ref_frexp_exp_i32_f64(POS_INF_F64)
        mutant = mutant_frexp_exp_wrong_special(POS_INF_F64)
        self.assertEqual(correct, 0)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_corrupt_vcc(self):
        interp = ScalarVALUInterpreter()
        interp.vcc = 0x5555555555555555
        interp.set_vgpr_u64(2, ONE_F64)
        ast = parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
        interp.execute(ast)
        self.assertEqual(interp.vcc, 0x5555555555555555)

        mutant_leak_vcc(interp, ast)
        self.assertNotEqual(interp.vcc, 0x5555555555555555)

    def test_mutation_detection_exec_mask_leak(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        sim.exec = 0x0
        sim.set_vgpr_u64(0, 0, 0xAAAAAAAAAAAAAAAA)
        ast = parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
        sim.execute(ast)
        self.assertEqual(sim.get_vgpr_u64(0, 0), 0xAAAAAAAAAAAAAAAA)

        mutant_exec_leak(sim, ast)
        self.assertNotEqual(sim.get_vgpr_u64(0, 0), 0xAAAAAAAAAAAAAAAA)


class TestValuFrexpF64IEEE754Boundaries(unittest.TestCase):
    """Deep verification of IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_normalization_boundaries(self):
        # MIN_SUBNORMAL_F64 = 2^-1074 -> mantissa 0.5, exp -1073
        m_min = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        e_min = ref_frexp_exp_i32_f64(MIN_SUBNORMAL_F64)
        self.assertEqual(bits_to_f64(m_min), 0.5)
        self.assertEqual(s32(e_min), -1073)

        # MAX_SUBNORMAL_F64 = 2^-1022 - 2^-1074 -> exp -1022
        m_max = ref_frexp_mant_f64(MAX_SUBNORMAL_F64)
        e_max = ref_frexp_exp_i32_f64(MAX_SUBNORMAL_F64)
        self.assertEqual(s32(e_max), -1022)
        # Fraction bits in mantissa should match MAX_SUBNORMAL shifted left by 1
        expected_mant = EXP_FREXP_BITS_F64 | ((MAX_SUBNORMAL_F64 << 1) & MANT_MASK_F64)
        self.assertEqual(m_max, expected_mant)

        # Powers of 2 across subnormals
        for p in range(52):
            val_bits = 1 << p
            m = ref_frexp_mant_f64(val_bits)
            e = s32(ref_frexp_exp_i32_f64(val_bits))
            self.assertEqual(bits_to_f64(m), 0.5)
            self.assertEqual(e, -1073 + p)

    def test_all_normal_exponent_boundaries(self):
        # Test lower, middle, upper normal exponents
        for exp in (1, 2, 1022, 1023, 1024, 2045, 2046):
            word = exp << EXP_SHIFT_F64
            m = ref_frexp_mant_f64(word)
            e = s32(ref_frexp_exp_i32_f64(word))
            self.assertEqual(bits_to_f64(m), 0.5)
            self.assertEqual(e, exp - 1022)

    def test_walking_fraction_bits(self):
        for bit_idx in range(52):
            frac = 1 << bit_idx
            word = (1023 << EXP_SHIFT_F64) | frac
            m = ref_frexp_mant_f64(word)
            e = s32(ref_frexp_exp_i32_f64(word))
            self.assertEqual(m & MANT_MASK_F64, frac)
            self.assertEqual((m >> EXP_SHIFT_F64) & 0x7FF, 1022)
            self.assertEqual(e, 1)

    def test_signed_zero_preservation(self):
        self.assertEqual(ref_frexp_mant_f64(POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64), NEG_ZERO_F64)
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ZERO_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_ZERO_F64), 0)

    def test_signed_infinity_preservation(self):
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64), NEG_INF_F64)
        self.assertEqual(ref_frexp_exp_i32_f64(POS_INF_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_INF_F64), 0)

    def test_nan_payloads_and_quieting(self):
        # Quiet NaN payload preserved
        qnan_payload = QNAN_F64 | 0x0000123456789ABC
        m_qnan = ref_frexp_mant_f64(qnan_payload)
        self.assertTrue(is_nan_f64(m_qnan))
        self.assertEqual(m_qnan & 0x0000123456789ABC, 0x0000123456789ABC)
        self.assertEqual(ref_frexp_exp_i32_f64(qnan_payload), 0)

        # Signaling NaN quiet bit set
        snan_payload = SNAN_F64 | 0x0000FEDCBA987654
        m_snan = ref_frexp_mant_f64(snan_payload)
        self.assertTrue(is_nan_f64(m_snan))
        self.assertFalse(is_snan_f64(m_snan), "Signaling NaN must be quieted")
        self.assertEqual(ref_frexp_exp_i32_f64(snan_payload), 0)

    def test_clamp_behavior_on_all_special_values(self):
        # NaNs clamp to +0.0
        self.assertEqual(ref_frexp_mant_f64(QNAN_F64, clamp=True), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(SNAN_F64, clamp=True), POS_ZERO_F64)
        # +inf clamps to 1.0, -inf clamps to +0.0
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64, clamp=True), ONE_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64, clamp=True), POS_ZERO_F64)
        # -0.0 clamps to +0.0
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64, clamp=True), POS_ZERO_F64)
        # Positive mantissas in [0.5, 1.0) remain unchanged
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, clamp=True)), 0.5)
        # Negative mantissas clamp to +0.0
        self.assertEqual(ref_frexp_mant_f64(ONE_F64 | SIGN_MASK_F64, clamp=True), POS_ZERO_F64)

    def test_omod_interactions_with_clamp(self):
        # 0.5 * mul:4 = 2.0 -> clamped to 1.0
        res_mul4 = ref_frexp_mant_f64(ONE_F64, clamp=True, omod="mul:4")
        self.assertEqual(bits_to_f64(res_mul4), 1.0)

        # 0.5 * mul:2 = 1.0 -> within [0.0, 1.0] -> 1.0
        res_mul2 = ref_frexp_mant_f64(ONE_F64, clamp=True, omod="mul:2")
        self.assertEqual(bits_to_f64(res_mul2), 1.0)

        # 0.75 * div:2 = 0.375 -> within [0.0, 1.0] -> 0.375
        f_1p5 = f64_to_bits(1.5)
        res_div2 = ref_frexp_mant_f64(f_1p5, clamp=True, omod="div:2")
        self.assertEqual(bits_to_f64(res_div2), 0.375)


class TestValuFrexpF64StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow generation."""

    def test_vcc_strictly_preserved_scalar(self):
        interp = ScalarVALUInterpreter()
        initial_vcc = 0x123456789ABCDEF0
        interp.vcc = initial_vcc
        interp.set_vgpr_u64(2, f64_to_bits(1.0))

        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertEqual(interp.vcc, initial_vcc, "VCC must remain unmodified by v_frexp_mant_f64")

        interp.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
        self.assertEqual(interp.vcc, initial_vcc, "VCC must remain unmodified by v_frexp_exp_i32_f64")

    def test_vcc_strictly_preserved_simd_wave64(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        initial_vcc = 0xCAFEBABE0000FFFF
        sim.vcc = initial_vcc
        for lane in range(64):
            sim.set_vgpr_u64(2, lane, f64_to_bits(2.0))

        sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertEqual(sim.vcc, initial_vcc, "Wave64 VCC must remain untouched by mantissa")

        sim.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
        self.assertEqual(sim.vcc, initial_vcc, "Wave64 VCC must remain untouched by exponent")

    def test_fp_flags_exact_normal_no_exceptions(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, f64_to_bits(3.5))
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        interp.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
        self.assertFalse(interp.status_flags.invalid)
        self.assertFalse(interp.status_flags.overflow)
        self.assertFalse(interp.status_flags.underflow)
        self.assertFalse(interp.status_flags.inexact)

    def test_fp_flags_invalid_snan_mant(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, SNAN_F64)
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        self.assertTrue(interp.status_flags.invalid)

    def test_fp_flags_invalid_snan_exp(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, SNAN_F64)
        interp.execute_asm("v_frexp_exp_i32_f64 v0, v[2:3]")
        self.assertTrue(interp.status_flags.invalid)

    def test_fp_flags_no_invalid_on_qnan(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, QNAN_F64)
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        interp.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
        self.assertFalse(interp.status_flags.invalid)

    def test_fp_flags_subnormals_exact(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, MIN_SUBNORMAL_F64)
        interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        interp.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
        # Frexp decomposition is exact; no underflow or overflow flags are raised
        self.assertFalse(interp.status_flags.underflow)
        self.assertFalse(interp.status_flags.overflow)
        self.assertFalse(interp.status_flags.invalid)

    def test_fp_flags_infinities_and_zeros(self):
        interp = ScalarVALUInterpreter()
        for val in (POS_ZERO_F64, NEG_ZERO_F64, POS_INF_F64, NEG_INF_F64):
            interp.reset()
            interp.set_vgpr_u64(2, val)
            interp.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
            interp.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
            self.assertFalse(interp.status_flags.invalid)
            self.assertFalse(interp.status_flags.overflow)
            self.assertFalse(interp.status_flags.underflow)


class TestValuFrexpF64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_full_aliasing_mant_dst_src0_scalar(self):
        interp = ScalarVALUInterpreter()
        initial_val = 6.0  # mantissa 0.75
        interp.set_vgpr_u64(0, f64_to_bits(initial_val))
        res = interp.execute_asm("v_frexp_mant_f64 v[0:1], v[0:1]")
        self.assertEqual(bits_to_f64(res), 0.75)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), 0.75)

    def test_aliasing_exp_dst_low_src0_scalar(self):
        interp = ScalarVALUInterpreter()
        # v0 = v_frexp_exp_i32_f64 v[0:1] (dst is low dword of src0)
        interp.set_vgpr_u64(0, f64_to_bits(12.0))  # 12.0 = 0.75 * 2^4 -> exp = 4
        res = interp.execute_asm("v_frexp_exp_i32_f64 v0, v[0:1]")
        self.assertEqual(s32(res), 4)
        self.assertEqual(s32(interp.get_vgpr_u32(0)), 4)

    def test_aliasing_exp_dst_high_src0_scalar(self):
        interp = ScalarVALUInterpreter()
        # v1 = v_frexp_exp_i32_f64 v[0:1] (dst is high dword of src0)
        interp.set_vgpr_u64(0, f64_to_bits(0.25))  # 0.25 = 0.5 * 2^-1 -> exp = -1
        res = interp.execute_asm("v_frexp_exp_i32_f64 v1, v[0:1]")
        self.assertEqual(s32(res), -1)
        self.assertEqual(s32(interp.get_vgpr_u32(1)), -1)

    def test_partial_overlap_dst_hi_src0_lo_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[1:2] = v_frexp_mant_f64 v[0:1]
        interp.set_vgpr_u64(0, f64_to_bits(10.0))  # 10.0 = 0.625 * 2^4
        res = interp.execute_asm("v_frexp_mant_f64 v[1:2], v[0:1]")
        self.assertEqual(bits_to_f64(res), 0.625)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(1)), 0.625)

    def test_partial_overlap_dst_lo_src0_hi_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[0:1] = v_frexp_mant_f64 v[1:2]
        interp.set_vgpr_u64(1, f64_to_bits(0.125))  # 0.125 = 0.5 * 2^-2
        res = interp.execute_asm("v_frexp_mant_f64 v[0:1], v[1:2]")
        self.assertEqual(bits_to_f64(res), 0.5)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), 0.5)

    def test_simd_wave64_aliasing_with_exec_mask(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0xABABABABCDCDEFEF
        for lane in range(64):
            if lane % 2 == 0:
                sim.set_vgpr_u64(0, lane, f64_to_bits(2.0))
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
                self.assertEqual(sim.get_vgpr_u64(0, lane), canary)

    def test_simd_wave32_partial_overlap_divergent(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        expected: list[float] = []
        for lane in range(32):
            val = (lane + 1) * 1.5
            sim.set_vgpr_u64(1, lane, f64_to_bits(val))
            py_m, _ = math.frexp(val)
            expected.append(py_m)

        results = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[1:2]")
        for lane in range(32):
            self.assertEqual(bits_to_f64(results[lane]), expected[lane])
            self.assertEqual(bits_to_f64(sim.get_vgpr_u64(0, lane)), expected[lane])

    def test_simd_wave64_exp_aliasing_with_exec_mask(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0xCAFE1234
        for lane in range(64):
            sim.set_vgpr_u64(0, lane, f64_to_bits(8.0))  # 8.0 = 0.5 * 2^4 -> exp = 4
            if lane % 2 != 0:
                sim.set_vgpr_u32(0, lane, canary)

        sim.exec = 0x5555555555555555
        results = sim.execute_asm("v_frexp_exp_i32_f64 v0, v[0:1]")
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(s32(results[lane]), 4)
                self.assertEqual(s32(sim.get_vgpr_u32(0, lane)), 4)
            else:
                self.assertEqual(results[lane], canary)
                self.assertEqual(sim.get_vgpr_u32(0, lane), canary)


class TestValuFrexpF64SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        base = f64_to_bits(6.0)
        for lane in range(32):
            sim.set_vgpr_u64(2, lane, base)

        results_m = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        results_e = sim.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
        self.assertEqual(len(results_m), 32)
        self.assertEqual(len(results_e), 32)
        for lane in range(32):
            self.assertEqual(bits_to_f64(results_m[lane]), 0.75)
            self.assertEqual(s32(results_e[lane]), 3)

    def test_wave64_uniform_execution(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        base = f64_to_bits(0.125)
        for lane in range(64):
            sim.set_vgpr_u64(6, lane, base)

        results_m = sim.execute_asm("v_frexp_mant_f64 v[4:5], v[6:7]")
        results_e = sim.execute_asm("v_frexp_exp_i32_f64 v8, v[6:7]")
        self.assertEqual(len(results_m), 64)
        for lane in range(64):
            self.assertEqual(bits_to_f64(results_m[lane]), 0.5)
            self.assertEqual(s32(results_e[lane]), -2)

    def test_wave32_divergent_lanes(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        expected_m: list[float] = []
        expected_e: list[int] = []
        for lane in range(32):
            val = float(lane + 1) * 0.75
            sim.set_vgpr_u64(0, lane, f64_to_bits(val))
            py_m, py_e = math.frexp(val)
            expected_m.append(py_m)
            expected_e.append(py_e)

        results_m = sim.execute_asm("v_frexp_mant_f64 v[4:5], v[0:1]")
        results_e = sim.execute_asm("v_frexp_exp_i32_f64 v6, v[0:1]")
        for lane in range(32):
            self.assertEqual(bits_to_f64(results_m[lane]), expected_m[lane])
            self.assertEqual(s32(results_e[lane]), expected_e[lane])

    def test_wave64_divergent_lanes(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        expected_m: list[float] = []
        expected_e: list[int] = []
        for lane in range(64):
            val = 0.375 * (lane + 1)
            sim.set_vgpr_u64(2, lane, f64_to_bits(val))
            py_m, py_e = math.frexp(val)
            expected_m.append(py_m)
            expected_e.append(py_e)

        results_m = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        results_e = sim.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")
        for lane in range(64):
            self.assertEqual(bits_to_f64(results_m[lane]), expected_m[lane])
            self.assertEqual(s32(results_e[lane]), expected_e[lane])

    def test_exec_mask_partial_inactivity_wave32(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        canary64 = 0xDEADBEEFCAFEBABE
        canary32 = 0x55AA55AA
        for lane in range(32):
            sim.set_vgpr_u64(0, lane, canary64)
            sim.set_vgpr_u32(4, lane, canary32)
            sim.set_vgpr_u64(2, lane, f64_to_bits(1.0))

        # Only even lanes active: 0x55555555
        sim.exec = 0x55555555
        results_m = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")
        results_e = sim.execute_asm("v_frexp_exp_i32_f64 v4, v[2:3]")

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(bits_to_f64(results_m[lane]), 0.5)
                self.assertEqual(s32(results_e[lane]), 1)
                self.assertEqual(bits_to_f64(sim.get_vgpr_u64(0, lane)), 0.5)
                self.assertEqual(sim.get_vgpr_u32(4, lane), 1)
            else:
                self.assertEqual(results_m[lane], canary64)
                self.assertEqual(results_e[lane], canary32)
                self.assertEqual(sim.get_vgpr_u64(0, lane), canary64)
                self.assertEqual(sim.get_vgpr_u32(4, lane), canary32)

    def test_exec_mask_partial_inactivity_wave64(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary64 = 0x1122334455667788
        for lane in range(64):
            sim.set_vgpr_u64(0, lane, canary64)
            sim.set_vgpr_u64(2, lane, f64_to_bits(2.0))

        # Low 32 lanes active, high 32 lanes inactive
        sim.exec = 0x00000000FFFFFFFF
        results_m = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[2:3]")

        for lane in range(64):
            if lane < 32:
                self.assertEqual(bits_to_f64(results_m[lane]), 0.5)
            else:
                self.assertEqual(results_m[lane], canary64)
                self.assertEqual(sim.get_vgpr_u64(0, lane), canary64)

    def test_exec_mask_all_inactive(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary64 = 0xCAFEBABE12345678
        canary32 = 0x77777777
        for lane in range(64):
            sim.set_vgpr_u64(0, lane, canary64)
            sim.set_vgpr_u32(2, lane, canary32)
            sim.set_vgpr_u64(4, lane, f64_to_bits(1.0))

        sim.exec = 0x0
        results_m = sim.execute_asm("v_frexp_mant_f64 v[0:1], v[4:5]")
        results_e = sim.execute_asm("v_frexp_exp_i32_f64 v2, v[4:5]")
        for lane in range(64):
            self.assertEqual(results_m[lane], canary64)
            self.assertEqual(results_e[lane], canary32)
            self.assertEqual(sim.get_vgpr_u64(0, lane), canary64)
            self.assertEqual(sim.get_vgpr_u32(2, lane), canary32)


class TestValuFrexpF64ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self):
        self.interp = ScalarVALUInterpreter()

    def test_scalar_interpreter_mant_vgpr_sources(self):
        # Set v[2:3] = 6.0
        self.interp.set_vgpr_u64(2, f64_to_bits(6.0))
        ast = parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], v[2:3]")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 0.75)
        self.assertEqual(self.interp.get_vgpr_u64(0), res)

    def test_scalar_interpreter_exp_vgpr_sources(self):
        # Set v[2:3] = 6.0 (6.0 = 0.75 * 2^3)
        self.interp.set_vgpr_u64(2, f64_to_bits(6.0))
        ast = parse_valu_frexp_f64("v_frexp_exp_i32_f64 v0, v[2:3]")
        res = self.interp.execute(ast)
        self.assertEqual(s32(res), 3)
        self.assertEqual(s32(self.interp.get_vgpr_u32(0)), 3)

    def test_scalar_interpreter_sgpr_sources(self):
        # Set s[0:1] = 0.5
        self.interp.set_sgpr_u64(0, f64_to_bits(0.5))
        ast_m = parse_valu_frexp_f64("v_frexp_mant_f64 v[6:7], s[0:1]")
        res_m = self.interp.execute(ast_m)
        self.assertEqual(bits_to_f64(res_m), 0.5)

        ast_e = parse_valu_frexp_f64("v_frexp_exp_i32_f64 v8, s[0:1]")
        res_e = self.interp.execute(ast_e)
        self.assertEqual(s32(res_e), 0)

    def test_scalar_interpreter_literal_source(self):
        ast_m = parse_valu_frexp_f64("v_frexp_mant_f64 v[2:3], 1.5")
        res_m = self.interp.execute(ast_m)
        self.assertEqual(bits_to_f64(res_m), 0.75)

        ast_e = parse_valu_frexp_f64("v_frexp_exp_i32_f64 v4, 1.5")
        res_e = self.interp.execute(ast_e)
        self.assertEqual(s32(res_e), 1)

    def test_scalar_interpreter_input_modifiers(self):
        self.interp.set_vgpr_u64(2, f64_to_bits(4.0))
        # -|v[2:3]| -> mantissa = -0.5
        ast_m = parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], -|v[2:3]|")
        res_m = self.interp.execute(ast_m)
        self.assertEqual(bits_to_f64(res_m), -0.5)

        # exp of -|v[2:3]| is still 3 (4.0 = 0.5 * 2^3)
        ast_e = parse_valu_frexp_f64("v_frexp_exp_i32_f64 v5, -|v[2:3]|")
        res_e = self.interp.execute(ast_e)
        self.assertEqual(s32(res_e), 3)

    def test_scalar_interpreter_output_modifiers_clamp_omod(self):
        # 0.5 with mul:4 -> 2.0; with clamp -> 1.0
        self.interp.set_vgpr_u64(0, f64_to_bits(0.5))
        ast = parse_valu_frexp_f64("v_frexp_mant_f64 v[4:5], v[0:1] clamp mul:4")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 1.0)

    def test_scalar_interpreter_asm_execution(self):
        self.interp.set_vgpr_u64(10, f64_to_bits(2.5))
        res_m = self.interp.execute_asm("v_frexp_mant_f64 v[0:1], v[10:11]")
        res_e = self.interp.execute_asm("v_frexp_exp_i32_f64 v2, v[10:11]")
        self.assertEqual(bits_to_f64(res_m), 0.625)
        self.assertEqual(s32(res_e), 2)


class TestValuFrexpF64SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        dst_mant = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=0)
        src0_mant = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=2)
        ast_mant = InstructionAST("v_frexp_mant_f64", dst_mant, src0_mant, clamp=True, omod="mul:2")
        ast_mant.validate()
        self.assertEqual(ast_mant.to_asm(), "v_frexp_mant_f64 v[0:1], v[2:3] clamp mul:2")

        dst_exp = Operand(OperandKind.VGPR_SINGLE, DataType.I32, reg_index=5)
        src0_exp = Operand(OperandKind.SGPR_PAIR, DataType.F64, reg_index=4)
        ast_exp = InstructionAST("v_frexp_exp_i32_f64", dst_exp, src0_exp)
        ast_exp.validate()
        self.assertEqual(ast_exp.to_asm(), "v_frexp_exp_i32_f64 v5, s[4:5]")

    def test_parser_mant_f64_vgpr_sgpr_operands(self):
        ast = parse_valu_frexp_f64("v_frexp_mant_f64 v[4:5], v[6:7]")
        self.assertEqual(ast.dst.reg_index, 4)
        self.assertEqual(ast.src0.kind, OperandKind.VGPR_PAIR)
        self.assertEqual(ast.src0.reg_index, 6)
        self.assertFalse(ast.clamp)
        self.assertEqual(ast.omod, "none")

        ast_sgpr = parse_valu_frexp_f64("v_frexp_mant_f64_e32 v[0:1], s[2:3]")
        self.assertEqual(ast_sgpr.src0.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(ast_sgpr.src0.reg_index, 2)

    def test_parser_exp_i32_f64_operands(self):
        ast = parse_valu_frexp_f64("v_frexp_exp_i32_f64 v3, v[8:9]")
        self.assertEqual(ast.dst.kind, OperandKind.VGPR_SINGLE)
        self.assertEqual(ast.dst.reg_index, 3)
        self.assertEqual(ast.src0.kind, OperandKind.VGPR_PAIR)
        self.assertEqual(ast.src0.reg_index, 8)

    def test_parser_literal_and_constant_operands(self):
        ast = parse_valu_frexp_f64("v_frexp_mant_f64 v[0:1], 1.5")
        self.assertEqual(ast.src0.kind, OperandKind.LITERAL_F64)
        self.assertEqual(bits_to_f64(ast.src0.literal_val), 1.5)

        ast_exp = parse_valu_frexp_f64("v_frexp_exp_i32_f64 v2, 2.0")
        self.assertEqual(ast_exp.src0.kind, OperandKind.LITERAL_F64)
        self.assertEqual(bits_to_f64(ast_exp.src0.literal_val), 2.0)

    def test_parser_modifiers_abs_neg_clamp_omod(self):
        ast = parse_valu_frexp_f64("v_frexp_mant_f64_e64 v[0:1], -|v[2:3]| clamp mul:4")
        self.assertTrue(ast.src0.abs_mod)
        self.assertTrue(ast.src0.neg_mod)
        self.assertTrue(ast.clamp)
        self.assertEqual(ast.omod, "mul:4")
        self.assertEqual(ast.to_asm(), "v_frexp_mant_f64 v[0:1], -|v[2:3]| clamp mul:4")

    def test_ast_type_checking_and_invariants(self):
        # Invalid dst type for mantissa (must be VGPR_PAIR F64)
        bad_dst_mant = Operand(OperandKind.VGPR_SINGLE, DataType.F64, reg_index=0)
        src0 = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=2)
        with self.assertRaises(TypeError):
            InstructionAST("v_frexp_mant_f64", bad_dst_mant, src0).validate()

        # Invalid dst type for exponent (must be VGPR_SINGLE I32)
        bad_dst_exp = Operand(OperandKind.VGPR_PAIR, DataType.I32, reg_index=0)
        with self.assertRaises(TypeError):
            InstructionAST("v_frexp_exp_i32_f64", bad_dst_exp, src0).validate()

        # Clamp not supported on v_frexp_exp_i32_f64
        good_dst_exp = Operand(OperandKind.VGPR_SINGLE, DataType.I32, reg_index=0)
        with self.assertRaises(ValueError):
            InstructionAST("v_frexp_exp_i32_f64", good_dst_exp, src0, clamp=True).validate()


class TestValuFrexpF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_normal_values(self):
        # 1.0 -> mantissa = 0.5, exp = 1
        m1 = ref_frexp_mant_f64(ONE_F64)
        e1 = ref_frexp_exp_i32_f64(ONE_F64)
        self.assertEqual(bits_to_f64(m1), 0.5)
        self.assertEqual(s32(e1), 1)

        # 2.0 -> mantissa = 0.5, exp = 2
        m2 = ref_frexp_mant_f64(TWO_F64)
        e2 = ref_frexp_exp_i32_f64(TWO_F64)
        self.assertEqual(bits_to_f64(m2), 0.5)
        self.assertEqual(s32(e2), 2)

        # 0.5 -> mantissa = 0.5, exp = 0
        m05 = ref_frexp_mant_f64(HALF_F64)
        e05 = ref_frexp_exp_i32_f64(HALF_F64)
        self.assertEqual(bits_to_f64(m05), 0.5)
        self.assertEqual(s32(e05), 0)

        # 1.5 -> mantissa = 0.75, exp = 1
        f_1p5 = f64_to_bits(1.5)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_1p5)), 0.75)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_1p5)), 1)

        # -4.0 -> mantissa = -0.5, exp = 3
        f_neg4 = f64_to_bits(-4.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg4)), -0.5)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_neg4)), 3)

    def test_oracle_powers_of_two(self):
        for p in range(-50, 50):
            val = math.ldexp(1.0, p)
            val_bits = f64_to_bits(val)
            m_bits = ref_frexp_mant_f64(val_bits)
            e_bits = ref_frexp_exp_i32_f64(val_bits)
            self.assertEqual(bits_to_f64(m_bits), 0.5)
            self.assertEqual(s32(e_bits), p + 1)

    def test_oracle_cross_check_math_frexp(self):
        test_floats = [0.125, 0.3333333333333333, 1.0, 2.5, 42.0, 1000.125, -7.5, -0.0625]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            oracle_m = bits_to_f64(ref_frexp_mant_f64(f_bits))
            oracle_e = s32(ref_frexp_exp_i32_f64(f_bits))
            py_m, py_e = math.frexp(f)
            self.assertEqual(oracle_m, py_m)
            self.assertEqual(oracle_e, py_e)

    def test_oracle_exact_reconstruction(self):
        test_floats = [
            1.5, 3.25, -0.75, 10.125, -12345.6789, 1.0e-100, 1.0e100,
            bits_to_f64(MIN_NORMAL_F64),
            bits_to_f64(MAX_SUBNORMAL_F64),
            bits_to_f64(MIN_SUBNORMAL_F64),
        ]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            reconstructed = ref_frexp_reconstruct_f64(f_bits)
            self.assertEqual(reconstructed, f)

    def test_oracle_special_values(self):
        # +0.0 and -0.0
        self.assertEqual(ref_frexp_mant_f64(POS_ZERO_F64), POS_ZERO_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_ZERO_F64), NEG_ZERO_F64)
        self.assertEqual(ref_frexp_exp_i32_f64(POS_ZERO_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_ZERO_F64), 0)

        # +inf and -inf
        self.assertEqual(ref_frexp_mant_f64(POS_INF_F64), POS_INF_F64)
        self.assertEqual(ref_frexp_mant_f64(NEG_INF_F64), NEG_INF_F64)
        self.assertEqual(ref_frexp_exp_i32_f64(POS_INF_F64), 0)
        self.assertEqual(ref_frexp_exp_i32_f64(NEG_INF_F64), 0)

        # NaNs
        res_qnan_m = ref_frexp_mant_f64(QNAN_F64)
        self.assertTrue(is_nan_f64(res_qnan_m))
        self.assertEqual(ref_frexp_exp_i32_f64(QNAN_F64), 0)

        res_snan_m = ref_frexp_mant_f64(SNAN_F64)
        self.assertTrue(is_nan_f64(res_snan_m))
        self.assertEqual(ref_frexp_exp_i32_f64(SNAN_F64), 0)

    def test_oracle_subnormal_normalization(self):
        # Max subnormal
        max_sub_m = ref_frexp_mant_f64(MAX_SUBNORMAL_F64)
        max_sub_e = ref_frexp_exp_i32_f64(MAX_SUBNORMAL_F64)
        py_m, py_e = math.frexp(bits_to_f64(MAX_SUBNORMAL_F64))
        self.assertEqual(bits_to_f64(max_sub_m), py_m)
        self.assertEqual(s32(max_sub_e), py_e)
        self.assertEqual(s32(max_sub_e), -1022)

        # Min subnormal
        min_sub_m = ref_frexp_mant_f64(MIN_SUBNORMAL_F64)
        min_sub_e = ref_frexp_exp_i32_f64(MIN_SUBNORMAL_F64)
        py_min_m, py_min_e = math.frexp(bits_to_f64(MIN_SUBNORMAL_F64))
        self.assertEqual(bits_to_f64(min_sub_m), py_min_m)
        self.assertEqual(s32(min_sub_e), py_min_e)
        self.assertEqual(s32(min_sub_e), -1073)

    def test_oracle_modifiers(self):
        f_neg2 = f64_to_bits(-2.0)
        # abs modifier on mantissa
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_neg2, abs_src0=True)), 0.5)
        # neg modifier on positive mantissa
        f_pos2 = f64_to_bits(2.0)
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_pos2, neg_src0=True)), -0.5)
        # abs and neg on exponent (no effect on magnitude/value)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_neg2, abs_src0=True)), 2)
        self.assertEqual(s32(ref_frexp_exp_i32_f64(f_pos2, neg_src0=True)), 2)

        # clamp modifier on mantissa
        # positive mantissa in [0.5, 1.0) is unchanged by clamp
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(f_pos2, clamp=True)), 0.5)
        # negative mantissa is clamped to +0.0
        self.assertEqual(ref_frexp_mant_f64(f_neg2, clamp=True), POS_ZERO_F64)

        # omod modifiers
        # 0.5 * 2 = 1.0
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:2")), 1.0)
        # 0.5 * 4 = 2.0
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="mul:4")), 2.0)
        # 0.5 / 2 = 0.25
        self.assertEqual(bits_to_f64(ref_frexp_mant_f64(ONE_F64, omod="div:2")), 0.25)


if __name__ == "__main__":
    unittest.main()
