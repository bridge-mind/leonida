from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_ldexp_f64.

Instruction:
  v_ldexp_f64 dst, src0, src1 (VOP3)
  D.f64 = S0.f64 * (2 ^ S1.i32)

Architecture:
  - RDNA2 Vector ALU (GFX10) double-precision load exponent.
  - S0: 64-bit IEEE 754 binary64 floating-point operand (VGPR pair, SGPR pair, or literal).
  - S1: 32-bit signed two's complement integer exponent (VGPR, SGPR, or literal).
  - D:  64-bit IEEE 754 binary64 floating-point destination (VGPR pair).
  - Modifiers:
      * Input modifiers on S0: abs (|S0|), neg (-S0).
      * Output modifiers: clamp (clamps result to [0.0, 1.0]), omod (mul:2, mul:4, div:2).

Translator-backed cases (SHADER-1039): a hand-written pixel-shader listing goes through
tools/rdna2spirv.py and is executed with tests/semantic/spirv_mini_interp.py. The lowering is pure 32-bit
integer work, so that interpreter can run it and the module carries no OpCapability Float64. The expected
words there are written by hand from the rules above, never taken from the oracle below, so a rule the
translator and the oracle share cannot hide a mistake.
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

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
for _p in (HERE, TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# The translator-backed cases at the end run a hand-written listing through tools/rdna2spirv.py, so the
# file is a translator regression test and not only a private oracle (SHADER-1039).
from support import mrt0_words, translate_lines  # noqa: E402  (tests/semantic is sys.path, see above)
import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

SIGN_MASK_F64 = 0x8000000000000000
EXP_MASK_F64 = 0x7FF0000000000000
MANT_MASK_F64 = 0x000FFFFFFFFFFFFF
BIAS_F64 = 1023

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


def ref_ldexp_f64(
    src0_bits: int,
    src1_i32: int,
    *,
    clamp: bool = False,
    abs_src0: bool = False,
    neg_src0: bool = False,
    omod: str = "none",
) -> int:
    """Bit-accurate reference mathematical oracle for RDNA2 v_ldexp_f64.

    Computes: D = S0 * 2^S1
    Where S0 is IEEE 754 binary64 and S1 is signed 32-bit integer.
    """
    src0_bits &= MASK64
    scale = s32(src1_i32)

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
        # Quiet the NaN if signaling
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

    # 4. Extract mantissa and unbiased exponent
    if exp_field == 0:
        # Subnormal input
        F = mant_field
        E = -1022
        shift_left = 52 - (F.bit_length() - 1)
        F = F << shift_left
        E = E - shift_left
    else:
        # Normal input
        E = exp_field - BIAS_F64
        F = (1 << 52) | mant_field

    # Apply omod multiplier to exponent
    omod_exp_shift = 0
    if omod == "mul:2":
        omod_exp_shift = 1
    elif omod == "mul:4":
        omod_exp_shift = 2
    elif omod == "div:2":
        omod_exp_shift = -1

    E_scaled = E + scale + omod_exp_shift
    target_exp = E_scaled + BIAS_F64

    # 5. Overflow check
    if target_exp >= 2047:
        if clamp:
            return POS_ZERO_F64 if sign else ONE_F64
        return (sign << 63) | POS_INF_F64

    # 6. Normal float range
    if target_exp >= 1:
        res = (sign << 63) | (target_exp << 52) | (F & MANT_MASK_F64)
        if clamp:
            if sign:
                return POS_ZERO_F64
            if res > ONE_F64:
                return ONE_F64
        return res

    # 7. Subnormal range / Gradual underflow (round-to-nearest-even)
    shift_right = 1 - target_exp
    if shift_right > 55:
        # Complete underflow
        return POS_ZERO_F64 if (clamp and sign) else (sign << 63)

    q = F >> shift_right
    mask = (1 << shift_right) - 1
    half = 1 << (shift_right - 1)
    remainder = F & mask

    # Round to nearest even
    if remainder > half:
        q += 1
    elif remainder == half:
        if q & 1:
            q += 1

    if q >= (1 << 52):
        # Rounded up into normal range (min normal)
        res = (sign << 63) | (1 << 52)
    else:
        res = (sign << 63) | (q & MANT_MASK_F64)

    if clamp:
        if sign:
            return POS_ZERO_F64
        if res > ONE_F64:
            return ONE_F64
    return res


def ref_ldexp_f64_rational(src0_bits: int, src1_i32: int) -> int:
    """Exact rational mathematical oracle using fractions.Fraction."""
    src0_bits &= MASK64
    scale = s32(src1_i32)

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

    scaled_frac = frac * (Fraction(2) ** scale)
    scaled_float = float(scaled_frac)
    res_bits = f64_to_bits(scaled_float)
    if sign:
        res_bits |= SIGN_MASK_F64
    return res_bits


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
    """Symbolic AST node for v_ldexp_f64."""
    opcode: str
    dst: Operand
    src0: Operand
    src1: Operand
    clamp: bool = False
    omod: str = "none"

    def validate(self) -> None:
        if self.opcode != "v_ldexp_f64":
            raise ValueError(f"Unsupported opcode {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR or self.dst.data_type != DataType.F64:
            raise TypeError("Destination must be a 64-bit VGPR pair (DataType.F64)")
        if self.dst.abs_mod or self.dst.neg_mod:
            raise ValueError("Destination does not support input modifiers")
        if self.src0.data_type != DataType.F64:
            raise TypeError("src0 must be DataType.F64")
        if self.src1.data_type != DataType.I32:
            raise TypeError("src1 must be DataType.I32")
        if self.src1.abs_mod or self.src1.neg_mod:
            raise ValueError("src1 (integer exponent) does not support float abs/neg modifiers")
        if self.omod not in ("none", "mul:2", "mul:4", "div:2"):
            raise ValueError(f"Invalid omod modifier '{self.omod}'")
        self.dst.validate()
        self.src0.validate()
        self.src1.validate()

    def to_asm(self) -> str:
        tokens = [f"{self.opcode} {self.dst.to_asm()}, {self.src0.to_asm()}, {self.src1.to_asm()}"]
        if self.clamp:
            tokens.append("clamp")
        if self.omod != "none":
            tokens.append(self.omod)
        return " ".join(tokens)


def parse_vop3_ldexp_f64(asm_line: str) -> InstructionAST:
    """Parse assembly string into an InstructionAST."""
    line = asm_line.strip()
    if not line.startswith("v_ldexp_f64"):
        raise ValueError(f"Expected instruction starting with 'v_ldexp_f64', got: {line}")

    # Remove comments
    if ";" in line:
        line = line.split(";", 1)[0].strip()
    if "//" in line:
        line = line.split("//", 1)[0].strip()

    parts = line.split(None, 1)
    if len(parts) < 2:
        raise ValueError(f"Incomplete instruction: {line}")

    opcode = parts[0]
    rest = parts[1]

    # Tokenize operands and modifiers
    tokens = [t.strip() for t in rest.split(",") if t.strip()]
    if len(tokens) < 3:
        raise ValueError(f"v_ldexp_f64 requires 3 operand sections, got: {len(tokens)}")

    dst_str = tokens[0]
    src0_str = tokens[1]
    last_section = tokens[2].split()
    src1_str = last_section[0]
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
        # Literal float or hex constant
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

    # Parse SRC1 (i32)
    s1_str = src1_str.strip()
    m_v1 = re.match(r"^v(\d+)$", s1_str)
    m_s1 = re.match(r"^s(\d+)$", s1_str)
    if m_v1:
        src1_op = Operand(kind=OperandKind.VGPR_SINGLE, data_type=DataType.I32, reg_index=int(m_v1.group(1)))
    elif m_s1:
        src1_op = Operand(kind=OperandKind.SGPR_SINGLE, data_type=DataType.I32, reg_index=int(m_s1.group(1)))
    else:
        # Literal integer
        if s1_str.startswith("0x") or s1_str.startswith("0X"):
            lit_val = int(s1_str, 16)
        else:
            lit_val = int(s1_str)
        src1_op = Operand(kind=OperandKind.LITERAL_I32, data_type=DataType.I32, literal_val=lit_val)

    ast = InstructionAST(
        opcode=opcode,
        dst=dst_op,
        src0=src0_op,
        src1=src1_op,
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

    def eval_operand_i32(self, op: Operand) -> int:
        if op.data_type != DataType.I32:
            raise TypeError(f"Expected DataType.I32, got {op.data_type}")
        if op.kind == OperandKind.VGPR_SINGLE:
            return s32(self.get_vgpr_u32(op.reg_index))
        elif op.kind == OperandKind.SGPR_SINGLE:
            return s32(self.get_sgpr_u32(op.reg_index))
        elif op.kind in (OperandKind.LITERAL_I32, OperandKind.INLINE_CONST):
            return s32(op.literal_val)
        else:
            raise ValueError(f"Unsupported operand kind for i32: {op.kind}")

    def execute(self, ast: InstructionAST) -> int:
        ast.validate()
        if ast.opcode == "v_ldexp_f64":
            s0_val = self.eval_operand_f64(ast.src0)
            s1_val = self.eval_operand_i32(ast.src1)
            res = ref_ldexp_f64(
                s0_val,
                s1_val,
                clamp=ast.clamp,
                omod=ast.omod,
            )
            # Track FP status flags
            flags = calc_ldexp_f64_status_flags(s0_val, s1_val, res)
            self.status_flags.merge(flags)
            self.set_vgpr_u64(ast.dst.reg_index, res)
            return res
        raise NotImplementedError(f"Opcode {ast.opcode} not implemented in scalar interpreter")

    def execute_asm(self, asm_line: str) -> int:
        ast = parse_vop3_ldexp_f64(asm_line)
        return self.execute(ast)


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
    # Quiet bit is bit 51 (MSB of mantissa). If 0, it is signaling NaN.
    return ((bits >> 51) & 1) == 0


def calc_ldexp_f64_status_flags(src0_bits: int, src1_i32: int, res_bits: int) -> FPStatusFlags:
    """Determine IEEE 754 exception flags for v_ldexp_f64 computation."""
    flags = FPStatusFlags()
    scale = s32(src1_i32)

    # Signaling NaN generates invalid exception
    if is_snan_f64(src0_bits):
        flags.invalid = True
        return flags

    if is_nan_f64(src0_bits) or is_inf_f64(src0_bits) or is_zero_f64(src0_bits):
        return flags

    # Non-zero finite operand
    if is_inf_f64(res_bits):
        flags.overflow = True
        flags.inexact = True
    elif is_subnormal_f64(res_bits):
        flags.underflow = True
        # Check if inexact
        exp_field = (src0_bits >> 52) & 0x7FF
        orig_exp = (exp_field - BIAS_F64) if exp_field != 0 else -1022
        if (orig_exp + scale) < -1074:
            flags.inexact = True
        else:
            # Check if bits were rounded
            shift = 1 - ((orig_exp + scale) + BIAS_F64)
            mant = src0_bits & MANT_MASK_F64
            if exp_field != 0:
                mant |= (1 << 52)
            if shift > 0 and (mant & ((1 << shift) - 1)) != 0:
                flags.inexact = True
    elif is_zero_f64(res_bits):
        flags.underflow = True
        flags.inexact = True

    return flags


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

    def eval_operand_i32_lane(self, op: Operand, lane: int) -> int:
        if op.kind == OperandKind.VGPR_SINGLE:
            return s32(self.get_vgpr_u32(op.reg_index, lane))
        elif op.kind == OperandKind.SGPR_SINGLE:
            return s32(self.get_sgpr_u32(op.reg_index))
        elif op.kind in (OperandKind.LITERAL_I32, OperandKind.INLINE_CONST):
            return s32(op.literal_val)
        else:
            raise ValueError(f"Unsupported operand kind for i32: {op.kind}")

    def execute(self, ast: InstructionAST) -> list[int]:
        ast.validate()
        results: list[int] = []
        effective_exec = self.exec & ((1 << self.wave_size) - 1)

        # Latch inputs across all lanes before write-back (hardware pipeline semantics)
        src0_latched = [0] * self.wave_size
        src1_latched = [0] * self.wave_size

        for lane in range(self.wave_size):
            if (effective_exec >> lane) & 1:
                src0_latched[lane] = self.eval_operand_f64_lane(ast.src0, lane)
                src1_latched[lane] = self.eval_operand_i32_lane(ast.src1, lane)

        # Compute and write back
        for lane in range(self.wave_size):
            if not ((effective_exec >> lane) & 1):
                results.append(self.get_vgpr_u64(ast.dst.reg_index, lane))
                continue

            res = ref_ldexp_f64(
                src0_latched[lane],
                src1_latched[lane],
                clamp=ast.clamp,
                omod=ast.omod,
            )
            # Accumulate FP flags across active lanes
            flags = calc_ldexp_f64_status_flags(src0_latched[lane], src1_latched[lane], res)
            self.status_flags.merge(flags)
            self.set_vgpr_u64(ast.dst.reg_index, lane, res)
            results.append(res)

        return results

    def execute_asm(self, asm_line: str) -> list[int]:
        ast = parse_vop3_ldexp_f64(asm_line)
        return self.execute(ast)


# --- Mutated Oracles for Mutation Testing and Negative Controls ---

def mutant_bias_off_by_one(src0: int, src1: int) -> int:
    """Mutant A: uses incorrect bias 1022 instead of 1023 on re-bias."""
    scale = s32(src1)
    exp = ((src0 >> 52) & 0x7FF) - BIAS_F64
    rebiased = exp + scale + 1022  # Off by one: 1022 instead of 1023
    return (src0 & SIGN_MASK_F64) | (rebiased << 52) | (src0 & MANT_MASK_F64)


def mutant_flush_subnormal(src0: int, src1: int) -> int:
    """Mutant B: prematurely flushes all subnormals to zero."""
    res = ref_ldexp_f64(src0, src1)
    if is_subnormal_f64(res):
        return src0 & SIGN_MASK_F64
    return res


def mutant_no_clamp(src0: int, src1: int) -> int:
    """Mutant C: ignores clamp modifier."""
    return ref_ldexp_f64(src0, src1, clamp=False)


def mutant_truncation_rounding(src0: int, src1: int) -> int:
    """Mutant D: uses truncation (round toward zero) instead of round-to-nearest-even."""
    scale = s32(src1)
    exp_field = (src0 >> 52) & 0x7FF
    mant_field = src0 & MANT_MASK_F64
    sign = (src0 >> 63) & 1
    E = (exp_field - BIAS_F64) if exp_field != 0 else -1022
    F = (1 << 52) | mant_field if exp_field != 0 else mant_field
    target_exp = E + scale + BIAS_F64
    if target_exp < 1:
        shift_right = 1 - target_exp
        q = F >> shift_right  # purely truncated without rounding
        return (sign << 63) | (q & MANT_MASK_F64)
    return ref_ldexp_f64(src0, src1)


def mutant_leak_vcc(interp: ScalarVALUInterpreter, ast: InstructionAST) -> None:
    """Mutant E: improperly corrupts VCC register during VALU FP execution."""
    interp.execute(ast)
    interp.vcc ^= 0x1


def mutant_exec_leak(sim: SIMDVALUInterpreter, ast: InstructionAST) -> None:
    """Mutant F: writes inactive lanes under EXEC mask."""
    sim.execute(ast)
    # Overwrite lane 0 regardless of EXEC
    sim.set_vgpr_u64(ast.dst.reg_index, 0, 0x1234567890ABCDEF)


class TestValuLdexpF64NegativeControlsAndMutations(unittest.TestCase):
    """Negative controls, syntax rejection, and mutation detection for v_ldexp_f64."""

    def test_syntax_rejection_invalid_opcode(self):
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_add_f64 v[0:1], v[2:3], v4")

    def test_syntax_rejection_insufficient_operands(self):
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], v[2:3]")

    def test_syntax_rejection_non_contiguous_dst(self):
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_ldexp_f64 v[0:2], v[2:3], v4")

    def test_syntax_rejection_non_contiguous_src0(self):
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], v[2:4], v4")

    def test_syntax_rejection_out_of_range_registers(self):
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_ldexp_f64 v[255:256], v[2:3], v4")
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], s[105:106], v4")

    def test_syntax_rejection_invalid_modifier(self):
        # Invalid omod
        with self.assertRaises(ValueError):
            ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], v[2:3], v4")
            ast.omod = "mul:3"
            ast.validate()

    def test_syntax_rejection_invalid_types(self):
        # Single VGPR as destination (must be 64-bit pair)
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_ldexp_f64 v0, v[2:3], v4")
        # Single VGPR as src0 (must be 64-bit pair)
        with self.assertRaises(ValueError):
            parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], v2, v4")

    def test_mutation_detection_bias_off_by_one(self):
        # Correct oracle gives 8.0 for 1.0 * 2^3
        correct = ref_ldexp_f64(ONE_F64, 3)
        mutant = mutant_bias_off_by_one(ONE_F64, 3)
        # Mutant computes with wrong bias, causing discrepancy
        self.assertEqual(bits_to_f64(correct), 8.0)
        # In fact, with exp = -1022 + 3 + 1022 = exp, it dropped scale or gave wrong exponent!
        # Let's ensure discrepancy is caught
        mutant_scaled = mutant_bias_off_by_one(ONE_F64, 1)
        self.assertNotEqual(mutant_scaled, ref_ldexp_f64(ONE_F64, 1))

    def test_mutation_detection_flush_subnormal(self):
        # Subnormal must be preserved in IEEE mode, not flushed to zero
        correct = ref_ldexp_f64(MIN_NORMAL_F64, -1)
        self.assertTrue(is_subnormal_f64(correct))
        mutant = mutant_flush_subnormal(MIN_NORMAL_F64, -1)
        self.assertEqual(mutant, POS_ZERO_F64)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_no_clamp(self):
        val = f64_to_bits(4.0)
        correct_clamped = ref_ldexp_f64(val, 0, clamp=True)
        mutant = mutant_no_clamp(val, 0)
        self.assertEqual(bits_to_f64(correct_clamped), 1.0)
        self.assertEqual(bits_to_f64(mutant), 4.0)
        self.assertNotEqual(correct_clamped, mutant)

    def test_mutation_detection_truncation_vs_tiesto_even(self):
        # Odd remainder that rounds up under TiesToEven
        f_odd = MIN_NORMAL_F64 | 2
        correct = ref_ldexp_f64(f_odd, -53)
        mutant = mutant_truncation_rounding(f_odd, -53)
        self.assertEqual(correct, MIN_SUBNORMAL_F64)
        self.assertEqual(mutant, POS_ZERO_F64)
        self.assertNotEqual(correct, mutant)

    def test_mutation_detection_corrupt_vcc(self):
        interp = ScalarVALUInterpreter()
        interp.vcc = 0x5555555555555555
        interp.set_vgpr_u64(2, ONE_F64)
        interp.set_vgpr_u32(4, 1)
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], v[2:3], v4")
        # Under normal execution VCC is preserved
        interp.execute(ast)
        self.assertEqual(interp.vcc, 0x5555555555555555)
        # Mutant corrupts VCC
        mutant_leak_vcc(interp, ast)
        self.assertNotEqual(interp.vcc, 0x5555555555555555)

    def test_mutation_detection_exec_mask_leak(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        sim.exec = 0x0  # all lanes disabled
        sim.set_vgpr_u64(0, 0, 0xAAAAAAAAAAAAAAAA)
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], v[2:3], v4")
        # Normal execution preserves lane 0
        sim.execute(ast)
        self.assertEqual(sim.get_vgpr_u64(0, 0), 0xAAAAAAAAAAAAAAAA)
        # Mutant writes inactive lane
        mutant_exec_leak(sim, ast)
        self.assertNotEqual(sim.get_vgpr_u64(0, 0), 0xAAAAAAAAAAAAAAAA)


class TestValuLdexpF64IEEE754Boundaries(unittest.TestCase):
    """Deep verification of IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_exact_tiesto_even_rounding(self):
        # Construct an unscaled float that underflows to a half-way tie in subnormal range
        # Min normal is 2^-1022 (exp_field = 1, mant = 0).
        # Scale = -53 -> target_exp = -52 -> shift_right = 53.
        # Mantissa F has 53 bits (1 << 52).
        # Shifted by 53: q = 0, remainder = 1 << 52, half = 1 << 52.
        # This is an exact half-way tie with q = 0 (even)!
        # In TiesToEven rounding, it rounds down to 0!
        res_even = ref_ldexp_f64(MIN_NORMAL_F64, -53)
        self.assertEqual(res_even, POS_ZERO_F64)

        # Now test with F = (1 << 52) | (1 << 1), so after shift by 53,
        # remainder > half, so it rounds up to 1 (MIN_SUBNORMAL_F64)!
        f_odd = MIN_NORMAL_F64 | 2
        res_odd = ref_ldexp_f64(f_odd, -53)
        self.assertEqual(res_odd, MIN_SUBNORMAL_F64)

    def test_subnormal_scaling_transitions(self):
        # Scaling MIN_SUBNORMAL (2^-1074) up to normal range
        res = ref_ldexp_f64(MIN_SUBNORMAL_F64, 52)
        self.assertEqual(res, MIN_NORMAL_F64)
        self.assertEqual(bits_to_f64(res), 2.2250738585072014e-308)

        # Scaling MAX_SUBNORMAL up by 1
        res_max_sub = ref_ldexp_f64(MAX_SUBNORMAL_F64, 1)
        self.assertTrue(is_normal_f64(res_max_sub))

        # Scaling MIN_SUBNORMAL down by 1 -> rounds to 0
        res_sub_down = ref_ldexp_f64(MIN_SUBNORMAL_F64, -1)
        self.assertEqual(res_sub_down, POS_ZERO_F64)

    def test_extreme_scale_integers(self):
        # Extreme large positive exponents (e.g. 0x7FFFFFFF, 1000000)
        for s in (0x7FFFFFFF, 1000000, 2048, 1024):
            self.assertEqual(ref_ldexp_f64(ONE_F64, s), POS_INF_F64)
            self.assertEqual(ref_ldexp_f64(ONE_F64 | SIGN_MASK_F64, s), NEG_INF_F64)

        # Extreme large negative exponents (e.g. -0x80000000, -1000000)
        for s in (-0x80000000, -1000000, -2048, -1075):
            self.assertEqual(ref_ldexp_f64(ONE_F64, s), POS_ZERO_F64)
            self.assertEqual(ref_ldexp_f64(ONE_F64 | SIGN_MASK_F64, s), NEG_ZERO_F64)

    def test_max_finite_overflow_boundaries(self):
        # MAX_FINITE with scale 0 is MAX_FINITE
        self.assertEqual(ref_ldexp_f64(MAX_FINITE_F64, 0), MAX_FINITE_F64)
        # MAX_FINITE with scale +1 overflows to +inf
        self.assertEqual(ref_ldexp_f64(MAX_FINITE_F64, 1), POS_INF_F64)
        # Negative MAX_FINITE with scale +1 overflows to -inf
        self.assertEqual(ref_ldexp_f64(MAX_FINITE_F64 | SIGN_MASK_F64, 1), NEG_INF_F64)

    def test_signed_zero_preservation_under_extremes(self):
        # +0.0 and -0.0 must be preserved for all scale values
        for s in (-0x80000000, -1000, -1, 0, 1, 1000, 0x7FFFFFFF):
            self.assertEqual(ref_ldexp_f64(POS_ZERO_F64, s), POS_ZERO_F64)
            self.assertEqual(ref_ldexp_f64(NEG_ZERO_F64, s), NEG_ZERO_F64)

    def test_signed_infinity_preservation(self):
        for s in (-1000, -1, 0, 1, 1000):
            self.assertEqual(ref_ldexp_f64(POS_INF_F64, s), POS_INF_F64)
            self.assertEqual(ref_ldexp_f64(NEG_INF_F64, s), NEG_INF_F64)

    def test_nan_payloads_and_quietening(self):
        # Quiet NaN payload preserved
        qnan_payload = QNAN_F64 | 0x0000000012345678
        res_qnan = ref_ldexp_f64(qnan_payload, 5)
        self.assertTrue(is_nan_f64(res_qnan))
        self.assertEqual(res_qnan & 0x0000000012345678, 0x0000000012345678)

        # Signaling NaN quiet bit set
        snan_payload = SNAN_F64 | 0x0000000087654321
        res_snan = ref_ldexp_f64(snan_payload, -3)
        self.assertTrue(is_nan_f64(res_snan))
        self.assertFalse(is_snan_f64(res_snan), "Signaling NaN must be quieted")

    def test_clamp_behavior_on_all_special_values(self):
        # NaNs clamp to +0.0
        self.assertEqual(ref_ldexp_f64(QNAN_F64, 0, clamp=True), POS_ZERO_F64)
        self.assertEqual(ref_ldexp_f64(SNAN_F64, 0, clamp=True), POS_ZERO_F64)
        # +inf clamps to 1.0, -inf clamps to +0.0
        self.assertEqual(ref_ldexp_f64(POS_INF_F64, 0, clamp=True), ONE_F64)
        self.assertEqual(ref_ldexp_f64(NEG_INF_F64, 0, clamp=True), POS_ZERO_F64)
        # -0.0 clamps to +0.0
        self.assertEqual(ref_ldexp_f64(NEG_ZERO_F64, 0, clamp=True), POS_ZERO_F64)
        # Values > 1.0 clamp to 1.0
        self.assertEqual(ref_ldexp_f64(TWO_F64, 0, clamp=True), ONE_F64)
        # Values < 0.0 clamp to +0.0
        self.assertEqual(ref_ldexp_f64(TWO_F64 | SIGN_MASK_F64, 0, clamp=True), POS_ZERO_F64)


class TestValuLdexpF64StatusFlags(unittest.TestCase):
    """Unit tests for condition codes, status flags, and carry/borrow behavior."""

    def test_vcc_strictly_preserved_scalar(self):
        interp = ScalarVALUInterpreter()
        initial_vcc = 0x123456789ABCDEF0
        interp.vcc = initial_vcc
        interp.set_vgpr_u64(2, f64_to_bits(1.0))
        interp.set_vgpr_u32(4, 5)

        interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertEqual(interp.vcc, initial_vcc, "VCC must remain unmodified by v_ldexp_f64")

    def test_vcc_strictly_preserved_simd_wave64(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        initial_vcc = 0xCAFEBABE0000FFFF
        sim.vcc = initial_vcc
        for lane in range(64):
            sim.set_vgpr_u64(2, lane, f64_to_bits(2.0))
            sim.set_vgpr_u32(4, lane, 1)

        sim.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertEqual(sim.vcc, initial_vcc, "Wave64 VCC must remain untouched")

    def test_fp_flags_exact_normal_no_exceptions(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, f64_to_bits(1.0))
        interp.set_vgpr_u32(4, 3)
        interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertFalse(interp.status_flags.invalid)
        self.assertFalse(interp.status_flags.overflow)
        self.assertFalse(interp.status_flags.underflow)
        self.assertFalse(interp.status_flags.inexact)

    def test_fp_flags_overflow(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, MAX_FINITE_F64)
        interp.set_vgpr_u32(4, 5)  # Overflows to +inf
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertEqual(res, POS_INF_F64)
        self.assertTrue(interp.status_flags.overflow)
        self.assertTrue(interp.status_flags.inexact)

    def test_fp_flags_underflow_subnormal(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, MIN_NORMAL_F64)
        interp.set_vgpr_u32(4, -2)  # Underflows into subnormal
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertTrue(is_subnormal_f64(res))
        self.assertTrue(interp.status_flags.underflow)

    def test_fp_flags_underflow_flush_to_zero(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, MIN_NORMAL_F64)
        interp.set_vgpr_u32(4, -100)  # Complete underflow to 0.0
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertEqual(res, POS_ZERO_F64)
        self.assertTrue(interp.status_flags.underflow)
        self.assertTrue(interp.status_flags.inexact)

    def test_fp_flags_invalid_snan(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, SNAN_F64)
        interp.set_vgpr_u32(4, 0)
        interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertTrue(interp.status_flags.invalid)

    def test_fp_flags_no_invalid_on_qnan(self):
        interp = ScalarVALUInterpreter()
        interp.set_vgpr_u64(2, QNAN_F64)
        interp.set_vgpr_u32(4, 0)
        interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertFalse(interp.status_flags.invalid)


class TestValuLdexpF64RegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_full_aliasing_dst_src0_scalar(self):
        interp = ScalarVALUInterpreter()
        # In-place scaling: v[0:1] = v_ldexp_f64 v[0:1], v2
        initial_val = 3.5
        interp.set_vgpr_u64(0, f64_to_bits(initial_val))
        interp.set_vgpr_u32(2, 2)
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[0:1], v2")
        self.assertEqual(bits_to_f64(res), 14.0)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), 14.0)

    def test_aliasing_dst_low_src1_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[0:1] = v_ldexp_f64 v[2:3], v0 (dst low overlaps src1)
        interp.set_vgpr_u32(0, 3)  # src1 exponent = 3
        interp.set_vgpr_u64(2, f64_to_bits(1.5))
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v0")
        self.assertEqual(bits_to_f64(res), 12.0)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), 12.0)

    def test_aliasing_dst_high_src1_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[0:1] = v_ldexp_f64 v[2:3], v1 (dst high overlaps src1)
        interp.set_vgpr_u32(1, -2)  # src1 exponent = -2
        interp.set_vgpr_u64(2, f64_to_bits(8.0))
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v1")
        self.assertEqual(bits_to_f64(res), 2.0)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), 2.0)

    def test_partial_overlap_dst_hi_src0_lo_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[1:2] = v_ldexp_f64 v[0:1], v3
        # v1 is shared between src0 high dword and dst low dword!
        val = 10.0
        interp.set_vgpr_u64(0, f64_to_bits(val))
        interp.set_vgpr_u32(3, 1)
        res = interp.execute_asm("v_ldexp_f64 v[1:2], v[0:1], v3")
        self.assertEqual(bits_to_f64(res), 20.0)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(1)), 20.0)

    def test_partial_overlap_dst_lo_src0_hi_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[0:1] = v_ldexp_f64 v[1:2], v3
        val = 0.75
        interp.set_vgpr_u64(1, f64_to_bits(val))
        interp.set_vgpr_u32(3, 4)
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[1:2], v3")
        self.assertEqual(bits_to_f64(res), 12.0)
        self.assertEqual(bits_to_f64(interp.get_vgpr_u64(0)), 12.0)

    def test_full_aliasing_dual_src0_src1_scalar(self):
        interp = ScalarVALUInterpreter()
        # v[0:1] = v_ldexp_f64 v[0:1], v0
        # Low dword of v[0:1] is also the exponent
        # Let's craft a double whose low dword is a valid integer exponent (e.g. 2)
        # In little endian IEEE 754: v0 is bits 0..31 of float, v1 is bits 32..63
        # If v0 = 2, and v1 = 0x3FF00000 (1.0)
        # Then src0 is a valid float close to 1.0, and exponent is 2!
        interp.set_vgpr_u32(0, 2)
        interp.set_vgpr_u32(1, 0x3FF00000)
        float_val = bits_to_f64(join_dwords(2, 0x3FF00000))
        expected = math.ldexp(float_val, 2)
        res = interp.execute_asm("v_ldexp_f64 v[0:1], v[0:1], v0")
        self.assertEqual(bits_to_f64(res), expected)

    def test_simd_wave64_aliasing_with_exec_mask(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0xABABABABCDCDEFEF
        for lane in range(64):
            # v[0:1] initialized with 2.0 on even lanes, canary on odd lanes
            if lane % 2 == 0:
                sim.set_vgpr_u64(0, lane, f64_to_bits(2.0))
            else:
                sim.set_vgpr_u64(0, lane, canary)
            sim.set_vgpr_u32(2, lane, 3)

        # Only even lanes active
        sim.exec = 0x5555555555555555
        # In-place execution: v[0:1] = v_ldexp_f64 v[0:1], v2
        results = sim.execute_asm("v_ldexp_f64 v[0:1], v[0:1], v2")

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(bits_to_f64(results[lane]), 16.0)
                self.assertEqual(bits_to_f64(sim.get_vgpr_u64(0, lane)), 16.0)
            else:
                self.assertEqual(results[lane], canary)
                self.assertEqual(sim.get_vgpr_u64(0, lane), canary)

    def test_simd_wave32_partial_overlap_divergent(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        for lane in range(32):
            sim.set_vgpr_u64(1, lane, f64_to_bits(1.0 + lane))
            sim.set_vgpr_u32(3, lane, 1)

        # v[0:1] = v_ldexp_f64 v[1:2], v3
        results = sim.execute_asm("v_ldexp_f64 v[0:1], v[1:2], v3")
        for lane in range(32):
            self.assertEqual(bits_to_f64(results[lane]), (1.0 + lane) * 2.0)


class TestValuLdexpF64SIMDMultiLane(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_uniform_execution(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        base = f64_to_bits(1.5)
        for lane in range(32):
            sim.set_vgpr_u64(2, lane, base)
            sim.set_vgpr_u32(4, lane, 3)

        results = sim.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        self.assertEqual(len(results), 32)
        for lane in range(32):
            self.assertEqual(bits_to_f64(results[lane]), 12.0)
            self.assertEqual(bits_to_f64(sim.get_vgpr_u64(0, lane)), 12.0)

    def test_wave64_uniform_execution(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        base = f64_to_bits(0.125)
        for lane in range(64):
            sim.set_vgpr_u64(6, lane, base)
            sim.set_vgpr_u32(8, lane, 4)

        results = sim.execute_asm("v_ldexp_f64 v[4:5], v[6:7], v8")
        self.assertEqual(len(results), 64)
        for lane in range(64):
            self.assertEqual(bits_to_f64(results[lane]), 2.0)

    def test_wave32_divergent_lanes(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        expected: list[float] = []
        for lane in range(32):
            val = float(lane + 1)
            exp = (lane % 5) - 2  # exponents in [-2, 2]
            sim.set_vgpr_u64(0, lane, f64_to_bits(val))
            sim.set_vgpr_u32(2, lane, exp)
            expected.append(math.ldexp(val, exp))

        results = sim.execute_asm("v_ldexp_f64 v[4:5], v[0:1], v2")
        for lane in range(32):
            self.assertEqual(bits_to_f64(results[lane]), expected[lane])

    def test_wave64_divergent_lanes(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        expected: list[float] = []
        for lane in range(64):
            val = 0.5 * (lane + 1)
            exp = (lane % 7) - 3
            sim.set_vgpr_u64(2, lane, f64_to_bits(val))
            sim.set_vgpr_u32(4, lane, exp)
            expected.append(math.ldexp(val, exp))

        results = sim.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        for lane in range(64):
            self.assertEqual(bits_to_f64(results[lane]), expected[lane])

    def test_exec_mask_partial_inactivity_wave32(self):
        sim = SIMDVALUInterpreter(wave_size=32)
        canary = 0xDEADBEEFCAFEBABE
        for lane in range(32):
            sim.set_vgpr_u64(0, lane, canary)
            sim.set_vgpr_u64(2, lane, f64_to_bits(1.0))
            sim.set_vgpr_u32(4, lane, 2)

        # Only even lanes active: 0x55555555
        sim.exec = 0x55555555
        results = sim.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(bits_to_f64(results[lane]), 4.0)
                self.assertEqual(bits_to_f64(sim.get_vgpr_u64(0, lane)), 4.0)
            else:
                self.assertEqual(results[lane], canary)
                self.assertEqual(sim.get_vgpr_u64(0, lane), canary)

    def test_exec_mask_partial_inactivity_wave64(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0x1122334455667788
        for lane in range(64):
            sim.set_vgpr_u64(0, lane, canary)
            sim.set_vgpr_u64(2, lane, f64_to_bits(2.0))
            sim.set_vgpr_u32(4, lane, 1)

        # Low 32 lanes active, high 32 lanes inactive
        sim.exec = 0x00000000FFFFFFFF
        results = sim.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")

        for lane in range(64):
            if lane < 32:
                self.assertEqual(bits_to_f64(results[lane]), 4.0)
            else:
                self.assertEqual(results[lane], canary)
                self.assertEqual(sim.get_vgpr_u64(0, lane), canary)

    def test_exec_mask_all_inactive(self):
        sim = SIMDVALUInterpreter(wave_size=64)
        canary = 0xCAFEBABE12345678
        for lane in range(64):
            sim.set_vgpr_u64(0, lane, canary)
            sim.set_vgpr_u64(2, lane, f64_to_bits(1.0))
            sim.set_vgpr_u32(4, lane, 1)

        sim.exec = 0x0
        results = sim.execute_asm("v_ldexp_f64 v[0:1], v[2:3], v4")
        for lane in range(64):
            self.assertEqual(results[lane], canary)
            self.assertEqual(sim.get_vgpr_u64(0, lane), canary)


class TestValuLdexpF64ScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction interpreter."""

    def setUp(self):
        self.interp = ScalarVALUInterpreter()

    def test_scalar_interpreter_vgpr_sources(self):
        # Set v[2:3] = 3.0, v4 = 2
        self.interp.set_vgpr_u64(2, f64_to_bits(3.0))
        self.interp.set_vgpr_u32(4, 2)
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], v[2:3], v4")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 12.0)
        self.assertEqual(self.interp.get_vgpr_u64(0), res)

    def test_scalar_interpreter_sgpr_sources(self):
        # Set s[0:1] = 0.5, s2 = 4 -> 8.0
        self.interp.set_sgpr_u64(0, f64_to_bits(0.5))
        self.interp.set_sgpr_u32(2, 4)
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[6:7], s[0:1], s2")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 8.0)
        self.assertEqual(self.interp.get_vgpr_u64(6), res)

    def test_scalar_interpreter_literal_exponent(self):
        self.interp.set_vgpr_u64(0, f64_to_bits(1.25))
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[2:3], v[0:1], 3")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 10.0)

    def test_scalar_interpreter_input_modifiers(self):
        # -|v[2:3]| * 2^1 with v[2:3] = 4.0 -> -8.0
        self.interp.set_vgpr_u64(2, f64_to_bits(4.0))
        self.interp.set_vgpr_u32(5, 1)
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], -|v[2:3]|, v5")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), -8.0)

    def test_scalar_interpreter_output_modifiers_clamp_omod(self):
        self.interp.set_vgpr_u64(0, f64_to_bits(0.25))
        self.interp.set_vgpr_u32(2, 1)  # 0.25 * 2^1 = 0.5
        # With mul:4 -> 2.0; with clamp -> 1.0
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[4:5], v[0:1], v2 clamp mul:4")
        res = self.interp.execute(ast)
        self.assertEqual(bits_to_f64(res), 1.0)

    def test_scalar_interpreter_asm_execution(self):
        self.interp.set_vgpr_u64(10, f64_to_bits(2.5))
        res = self.interp.execute_asm("v_ldexp_f64 v[0:1], v[10:11], -1")
        self.assertEqual(bits_to_f64(res), 1.25)


class TestValuLdexpF64SymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        dst = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=0)
        src0 = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=2)
        src1 = Operand(OperandKind.SGPR_SINGLE, DataType.I32, reg_index=4)
        ast = InstructionAST("v_ldexp_f64", dst, src0, src1, clamp=True, omod="mul:2")
        ast.validate()
        self.assertEqual(ast.to_asm(), "v_ldexp_f64 v[0:1], v[2:3], s4 clamp mul:2")

    def test_parser_vgpr_sgpr_operands(self):
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[4:5], v[6:7], v8")
        self.assertEqual(ast.dst.reg_index, 4)
        self.assertEqual(ast.src0.kind, OperandKind.VGPR_PAIR)
        self.assertEqual(ast.src0.reg_index, 6)
        self.assertEqual(ast.src1.kind, OperandKind.VGPR_SINGLE)
        self.assertEqual(ast.src1.reg_index, 8)
        self.assertFalse(ast.clamp)
        self.assertEqual(ast.omod, "none")

    def test_parser_literal_and_constant_operands(self):
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], s[2:3], 5")
        self.assertEqual(ast.src0.kind, OperandKind.SGPR_PAIR)
        self.assertEqual(ast.src1.kind, OperandKind.LITERAL_I32)
        self.assertEqual(ast.src1.literal_val, 5)

    def test_parser_modifiers_abs_neg_clamp_omod(self):
        ast = parse_vop3_ldexp_f64("v_ldexp_f64 v[0:1], -|v[2:3]|, s0 clamp mul:4")
        self.assertTrue(ast.src0.abs_mod)
        self.assertTrue(ast.src0.neg_mod)
        self.assertTrue(ast.clamp)
        self.assertEqual(ast.omod, "mul:4")
        self.assertEqual(ast.to_asm(), "v_ldexp_f64 v[0:1], -|v[2:3]|, s0 clamp mul:4")

    def test_ast_type_checking_and_invariants(self):
        # Invalid dst type
        bad_dst = Operand(OperandKind.VGPR_SINGLE, DataType.F64, reg_index=0)
        src0 = Operand(OperandKind.VGPR_PAIR, DataType.F64, reg_index=2)
        src1 = Operand(OperandKind.SGPR_SINGLE, DataType.I32, reg_index=4)
        with self.assertRaises(TypeError):
            InstructionAST("v_ldexp_f64", bad_dst, src0, src1).validate()

        # Invalid src1 type
        bad_src1 = Operand(OperandKind.SGPR_SINGLE, DataType.F64, reg_index=4)
        with self.assertRaises(TypeError):
            InstructionAST("v_ldexp_f64", Operand(OperandKind.VGPR_PAIR, DataType.F64, 0), src0, bad_src1).validate()


class TestValuLdexpF64Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural model."""

    def test_oracle_normal_values(self):
        # 1.0 * 2^3 = 8.0
        res = ref_ldexp_f64(ONE_F64, 3)
        self.assertEqual(bits_to_f64(res), 8.0)
        # 1.5 * 2^2 = 6.0
        f_1p5 = f64_to_bits(1.5)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_1p5, 2)), 6.0)
        # -4.0 * 2^-2 = -1.0
        f_neg4 = f64_to_bits(-4.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_neg4, -2)), -1.0)

    def test_oracle_powers_of_two(self):
        for p in range(-50, 50):
            res = ref_ldexp_f64(ONE_F64, p)
            expected = math.ldexp(1.0, p)
            self.assertEqual(bits_to_f64(res), expected)

    def test_oracle_cross_check_math_ldexp(self):
        test_floats = [0.125, 0.3333333333333333, 1.0, 2.5, 42.0, 1000.125, -7.5, -0.0625]
        test_scales = [-10, -5, -2, -1, 0, 1, 2, 5, 10]
        for f in test_floats:
            f_bits = f64_to_bits(f)
            for s in test_scales:
                oracle_res = bits_to_f64(ref_ldexp_f64(f_bits, s))
                expected = math.ldexp(f, s)
                self.assertEqual(oracle_res, expected)

    def test_oracle_cross_check_rational(self):
        for f in [1.5, 3.25, -0.75, 10.125]:
            f_bits = f64_to_bits(f)
            for s in [-4, -1, 0, 3, 7]:
                self.assertEqual(ref_ldexp_f64(f_bits, s), ref_ldexp_f64_rational(f_bits, s))

    def test_oracle_special_values(self):
        # +0.0 and -0.0 preserved
        self.assertEqual(ref_ldexp_f64(POS_ZERO_F64, 10), POS_ZERO_F64)
        self.assertEqual(ref_ldexp_f64(NEG_ZERO_F64, 10), NEG_ZERO_F64)
        # +inf and -inf preserved
        self.assertEqual(ref_ldexp_f64(POS_INF_F64, 5), POS_INF_F64)
        self.assertEqual(ref_ldexp_f64(NEG_INF_F64, 5), NEG_INF_F64)
        # NaNs preserved and quieted
        res_qnan = ref_ldexp_f64(QNAN_F64, 2)
        self.assertTrue(is_nan_f64(res_qnan))
        res_snan = ref_ldexp_f64(SNAN_F64, 2)
        self.assertTrue(is_nan_f64(res_snan))

    def test_oracle_subnormal_and_gradual_underflow(self):
        # Smallest normal shifted down into subnormal
        sub_bits = ref_ldexp_f64(MIN_NORMAL_F64, -1)
        self.assertTrue(is_subnormal_f64(sub_bits))
        self.assertEqual(sub_bits, 0x0008000000000000)
        # Subnormal shifted back to normal
        norm_bits = ref_ldexp_f64(sub_bits, 1)
        self.assertEqual(norm_bits, MIN_NORMAL_F64)
        # Complete underflow to zero
        zero_bits = ref_ldexp_f64(MIN_NORMAL_F64, -100)
        self.assertEqual(zero_bits, POS_ZERO_F64)

    def test_oracle_overflow(self):
        # Overflow to infinity
        res_overflow = ref_ldexp_f64(MAX_FINITE_F64, 2)
        self.assertEqual(res_overflow, POS_INF_F64)
        res_neg_overflow = ref_ldexp_f64(MAX_FINITE_F64 | SIGN_MASK_F64, 2)
        self.assertEqual(res_neg_overflow, NEG_INF_F64)

    def test_oracle_modifiers(self):
        # abs modifier
        f_neg2 = f64_to_bits(-2.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_neg2, 1, abs_src0=True)), 4.0)
        # neg modifier
        f_pos2 = f64_to_bits(2.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_pos2, 1, neg_src0=True)), -4.0)
        # clamp modifier
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_pos2, 2, clamp=True)), 1.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(f_neg2, 1, clamp=True)), 0.0)
        # omod modifiers
        self.assertEqual(bits_to_f64(ref_ldexp_f64(ONE_F64, 0, omod="mul:2")), 2.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(ONE_F64, 0, omod="mul:4")), 4.0)
        self.assertEqual(bits_to_f64(ref_ldexp_f64(ONE_F64, 0, omod="div:2")), 0.5)


# --- translator-backed cases (SHADER-1039) -------------------------------------------------------
# Each case translates a hand-written pixel shader with tools/rdna2spirv.py and runs it through
# tests/semantic/spirv_mini_interp.py; the expected word is the rule above applied by hand.


def run_ldexp(src_bits, exp_bits, src_text="v[0:1]", tail="", setup=()):
    """Translate and run `v_ldexp_f64 v[4:5], <src_text>, v2<tail>`; return (bits, SPIR-V text)."""
    lines = list(setup) + [
        "v_mov_b32 v0, %#010x" % (src_bits & MASK32),
        "v_mov_b32 v1, %#010x" % ((src_bits >> 32) & MASK32),
        "v_mov_b32 v2, %#010x" % (exp_bits & MASK32),
        "v_ldexp_f64 v[4:5], %s, v2%s" % (src_text, tail),
        "exp mrt0 v4, v5, v4, v5 done vm",
        "s_endpgm",
    ]
    spv = translate_lines("ps_ldexp_f64", lines)
    words = mrt0_words(spv)
    return (words[0] | words[1] << 32), spv


class TranslatedLdexpF64(unittest.TestCase):
    """The translated instruction against words written by hand from the rules above (SHADER-1039)."""

    CASES = [
        # (source bits, exponent, expected bits, what it pins)
        (ONE_F64, 5, 0x4040000000000000, "1.0 * 2^5 = 32.0"),
        (0x3FF8000000000000, -1, 0x3FE8000000000000, "1.5 * 2^-1 = 0.75"),
        (0xBFF8000000000000, 2, 0xC018000000000000, "-1.5 * 2^2 = -6.0"),
        (ONE_F64, -1074, MIN_SUBNORMAL_F64, "1.0 * 2^-1074 is the smallest subnormal"),
        (MIN_SUBNORMAL_F64, 1, 0x0000000000000002, "a subnormal input is normalised before scaling"),
        (ONE_F64, -1075, POS_ZERO_F64, "half an ulp ties to even"),
        (0x3FF8000000000000, -1075, MIN_SUBNORMAL_F64, "three quarters of an ulp rounds up"),
        (POS_ZERO_F64, 900, POS_ZERO_F64, "+0 passes through"),
        (NEG_ZERO_F64, 900, NEG_ZERO_F64, "-0 keeps its sign"),
        (POS_INF_F64, 900, POS_INF_F64, "+inf passes through"),
        (NEG_INF_F64, -900, NEG_INF_F64, "-inf passes through"),
        (QNAN_F64, 7, QNAN_F64, "qNaN passes through unchanged"),
        (MAX_FINITE_F64, 2, POS_INF_F64, "overflow gives +inf"),
    ]

    def test_scaling_specials_and_subnormals(self):
        for src_bits, exponent, want, why in self.CASES:
            with self.subTest(why=why):
                got, spv = run_ldexp(src_bits, exponent)
                self.assertEqual(got, want, why)
                self.assertNotIn("OpCapability Float64", spv, "the lowering is 32-bit integer only")

    def test_source_modifiers(self):
        got, _ = run_ldexp(0xC008000000000000, 3, src_text="|v[0:1]|")      # |-3.0| * 8
        self.assertEqual(got, 0x4038000000000000)
        got, _ = run_ldexp(0x3FF8000000000000, 1, src_text="-v[0:1]")      # -(1.5 * 2)
        self.assertEqual(got, 0xC008000000000000)

    def test_clamp(self):
        # dx10_clamp: NaN, -0 and negatives give +0.0; anything above 1.0 gives 1.0.
        for src_bits, exponent, want in ((ONE_F64, 2000, ONE_F64),
                                         (0xBFF0000000000000, 3, POS_ZERO_F64),
                                         (QNAN_F64, 3, POS_ZERO_F64),
                                         (0x3FE0000000000000, 1, ONE_F64)):
            with self.subTest(src=hex(src_bits), exponent=exponent):
                got, _ = run_ldexp(src_bits, exponent, tail=" clamp")
                self.assertEqual(got, want)

    def test_32_bit_literal_lands_in_the_high_word(self):
        # RDNA2 places a 32-bit literal constant in bits [63:32] of an f64 operand (measured on gfx1036).
        got, _ = run_ldexp(0, 3, src_text="0x3ff00000")
        self.assertEqual(got, 0x4020000000000000)                          # 1.0 * 2^3

    def test_sgpr_pair_source(self):
        setup = ("s_mov_b32 s0, 0x00000000", "s_mov_b32 s1, 0xc0000000")   # -2.0
        got, _ = run_ldexp(0, 2, src_text="s[0:1]", setup=setup)
        self.assertEqual(got, 0xC020000000000000)                          # -2.0 * 4

    def test_mode_dependent_omod_is_refused(self):
        # Hardware ignores mul:2/mul:4/div:2 with denorm16_64 = 3 and applies them with 0, so the
        # translator refuses them rather than guessing.
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "output modifier"):
            run_ldexp(ONE_F64, 5, tail=" mul:2")

    def test_single_register_destination_is_refused(self):
        lines = ["v_mov_b32 v0, 0", "v_mov_b32 v1, 0x3ff00000", "v_mov_b32 v2, 3",
                 "v_ldexp_f64 v4, v[0:1], v2", "exp mrt0 v4, v4, v4, v4 done vm", "s_endpgm"]
        with self.assertRaisesRegex(rdna2spirv.Unsupported, "destination"):
            translate_lines("ps_ldexp_bad_dst", lines)


if __name__ == "__main__":
    unittest.main()
