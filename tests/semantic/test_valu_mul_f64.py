"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_mul_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point quantities (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operands
  v[s0:s0+1] and v[s1:s1+1] (or scalar SGPR pairs s[k:k+1], literal 64-bit constants,
  or inline constants).

  Instruction:
    v_mul_f64 v[d:d+1], src0, src1 [clamp] [omod]

  IEEE-754 binary64 encoding:
    - 1 sign bit (bit 63)
    - 11 exponent bits with bias 1023 (bits 62:52)
    - 52 mantissa / fraction bits (bits 51:0)

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
import enum
import math
import struct
import unittest

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# IEEE-754 binary64 bit constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_QUIET_BIT = 0x0008000000000000

F64_EXP_MASK = 0x7FF0000000000000
F64_MANT_MASK = 0x000FFFFFFFFFFFFF
F64_SIGN_MASK = 0x8000000000000000

F64_ONE = 0x3FF0000000000000          # 1.0
F64_NEG_ONE = 0xBFF0000000000000      # -1.0
F64_TWO = 0x4000000000000000          # 2.0
F64_HALF = 0x3FE0000000000000         # 0.5
F64_MIN_NORMAL = 0x0010000000000000   # 2^-1022
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF   # (2 - 2^-52) * 2^1023
F64_MIN_SUBNORMAL = 0x0000000000000001  # 2^-1074
F64_MAX_SUBNORMAL = 0x000FFFFFFFFFFFFF

OLD_VGPR_SENTINEL = 0xDEADBEEF


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit range [0, 2^32 - 1]."""
    return val & MASK32


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit range [0, 2^64 - 1]."""
    return val & MASK64


def split_dwords(val64: int) -> tuple[int, int]:
    """Split an unsigned 64-bit integer into (low_dword, high_dword)."""
    val = val64 & MASK64
    return val & MASK32, (val >> 32) & MASK32


def join_dwords(lo: int, hi: int) -> int:
    """Join (low_dword, high_dword) into an unsigned 64-bit integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_to_u64(val: float) -> int:
    """Pack an IEEE-754 double precision float into an unsigned 64-bit integer."""
    try:
        return struct.unpack("<Q", struct.pack("<d", val))[0]
    except OverflowError:
        return F64_NEG_INF if val < 0 else F64_POS_INF


def u64_to_f64(bits: int) -> float:
    """Unpack an unsigned 64-bit integer into an IEEE-754 double precision float."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def is_f64_nan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents an IEEE-754 NaN."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) != 0)


def is_f64_snan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a signaling NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) == 0)


def is_f64_qnan(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a quiet NaN."""
    return is_f64_nan(bits) and ((bits & F64_QUIET_BIT) != 0)


def is_f64_zero(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +0.0 or -0.0."""
    return (bits & (F64_EXP_MASK | F64_MANT_MASK)) == 0


def is_f64_inf(bits: int) -> bool:
    """Return True if the 64-bit pattern represents +inf or -inf."""
    return ((bits & F64_EXP_MASK) == F64_EXP_MASK) and ((bits & F64_MANT_MASK) == 0)


def is_f64_subnormal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a subnormal/denormal float."""
    return ((bits & F64_EXP_MASK) == 0) and ((bits & F64_MANT_MASK) != 0)


def is_f64_normal(bits: int) -> bool:
    """Return True if the 64-bit pattern represents a normalized finite float."""
    exp = (bits >> 52) & 0x7FF
    return 0 < exp < 0x7FF


def f64_sign(bits: int) -> int:
    """Return the sign bit (0 or 1)."""
    return (bits >> 63) & 1


def quiet_f64(bits: int) -> int:
    """Force an IEEE-754 double NaN to quiet by setting the quiet bit."""
    return (bits & MASK64) | F64_QUIET_BIT


def apply_input_modifiers_f64(bits: int, *, neg: bool = False, abs_val: bool = False) -> int:
    """Apply VOP3 input modifiers (abs, neg).

    In RDNA2 ISA:
      When both abs and neg are set, abs is evaluated before neg:
      i.e. neg(abs(x)) forces the sign bit to 1 (negative).
    """
    res = bits & MASK64
    if abs_val:
        res &= ~F64_SIGN_MASK
    if neg:
        res ^= F64_SIGN_MASK
    return res


def ref_mul_f64(
    src0_bits: int,
    src1_bits: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 double-precision multiplication v_mul_f64.

    Evaluates:
      D.f64 = src0 * src1
    accounting for IEEE-754 binary64 rules, input modifiers (neg, abs), and clamp [0.0, 1.0].
    """
    s0 = apply_input_modifiers_f64(src0_bits, neg=src0_neg, abs_val=src0_abs)
    s1 = apply_input_modifiers_f64(src1_bits, neg=src1_neg, abs_val=src1_abs)

    sign_res = f64_sign(s0) ^ f64_sign(s1)

    # NaN propagation
    if is_f64_nan(s0):
        res = quiet_f64(s0)
    elif is_f64_nan(s1):
        res = quiet_f64(s1)
    # 0 * Inf -> Canonical QNaN
    elif (is_f64_zero(s0) and is_f64_inf(s1)) or (is_f64_inf(s0) and is_f64_zero(s1)):
        res = F64_CANONICAL_QNAN
    # Inf * anything non-zero -> Inf with expected sign
    elif is_f64_inf(s0) or is_f64_inf(s1):
        res = F64_NEG_INF if sign_res else F64_POS_INF
    # Zero * anything finite -> Zero with expected sign
    elif is_f64_zero(s0) or is_f64_zero(s1):
        res = F64_NEG_ZERO if sign_res else F64_POS_ZERO
    else:
        # Standard finite multiplication
        val0 = u64_to_f64(s0)
        val1 = u64_to_f64(s1)
        prod = val0 * val1
        res = f64_to_u64(prod)
        if is_f64_zero(res):
            res = F64_NEG_ZERO if sign_res else F64_POS_ZERO
        elif is_f64_inf(res):
            res = F64_NEG_INF if sign_res else F64_POS_INF

    # Clamp modifier: clamps result to [0.0, 1.0].
    # In RDNA2 VALU: if result is NaN, clamp produces +0.0.
    if clamp:
        if is_f64_nan(res):
            res = F64_POS_ZERO
        else:
            val = u64_to_f64(res)
            if val < 0.0 or is_f64_zero(res):
                res = F64_POS_ZERO
            elif val > 1.0:
                res = F64_ONE

    return res & MASK64


def ref_mul_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    s1_lo: int,
    s1_hi: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
) -> tuple[int, int]:
    """Decomposed dword interface for v_mul_f64."""
    s0 = join_dwords(s0_lo, s0_hi)
    s1 = join_dwords(s1_lo, s1_hi)
    res = ref_mul_f64(
        s0,
        s1,
        clamp=clamp,
        src0_neg=src0_neg,
        src0_abs=src0_abs,
        src1_neg=src1_neg,
        src1_abs=src1_abs,
    )
    return split_dwords(res)


class VALURegisterFile:
    """Architectural model for RDNA2 VALU 32-bit vector register file."""

    def __init__(self, num_vgprs: int = 256) -> None:
        self.num_vgprs = num_vgprs
        self.regs = [0] * num_vgprs

    def get_vgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        return self.regs[idx] & MASK32

    def set_vgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_vgprs):
            raise IndexError(f"VGPR index out of range: {idx}")
        self.regs[idx] = val & MASK32

    def get_vgpr_pair(self, idx: int) -> int:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo = self.get_vgpr(idx)
        hi = self.get_vgpr(idx + 1)
        return join_dwords(lo, hi)

    def set_vgpr_pair(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_vgprs - 1):
            raise IndexError(f"VGPR pair index out of range: v[{idx}:{idx+1}]")
        lo, hi = split_dwords(val64)
        self.set_vgpr(idx, lo)
        self.set_vgpr(idx + 1, hi)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions for v_mul_f64
# ==============================================================================

class OperandKind(enum.Enum):
    VGPR_PAIR = "vgpr_pair"
    SGPR_PAIR = "sgpr_pair"
    LITERAL_64 = "literal_64"
    INLINE_IMM = "inline_imm"


@dataclass
class ASTOperand:
    kind: OperandKind
    reg_idx: int = 0
    imm_val: int = 0
    neg: bool = False
    abs_val: bool = False

    def format_asm(self) -> str:
        if self.kind == OperandKind.VGPR_PAIR:
            s = f"v[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            s = f"s[{self.reg_idx}:{self.reg_idx + 1}]"
        elif self.kind == OperandKind.LITERAL_64:
            s = f"0x{self.imm_val:016x}"
        elif self.kind == OperandKind.INLINE_IMM:
            s = f"{self.imm_val}"
        else:
            s = "unknown"
        if self.abs_val:
            s = f"|{s}|"
        if self.neg:
            s = f"-{s}"
        return s


@dataclass
class ASTInstruction:
    opcode: str
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand
    clamp: bool = False
    omod: int = 0

    def validate(self) -> None:
        if self.opcode != "v_mul_f64":
            raise ValueError(f"Invalid opcode for v_mul_f64 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise ValueError(f"Destination must be a VGPR pair, got {self.dst.kind}")
        if not (0 <= self.dst.reg_idx <= 254):
            raise ValueError(f"Destination VGPR pair index out of range: {self.dst.reg_idx}")
        if self.dst.neg or self.dst.abs_val:
            raise ValueError("Destination operand cannot have neg or abs modifiers")
        for name, src in [("src0", self.src0), ("src1", self.src1)]:
            if src.kind == OperandKind.VGPR_PAIR:
                if not (0 <= src.reg_idx <= 254):
                    raise ValueError(f"{name} VGPR pair index out of range: {src.reg_idx}")
            elif src.kind == OperandKind.SGPR_PAIR:
                if not (0 <= src.reg_idx <= 104):
                    raise ValueError(f"{name} SGPR pair index out of range: {src.reg_idx}")

    def format_asm(self) -> str:
        s = f"{self.opcode} {self.dst.format_asm()}, {self.src0.format_asm()}, {self.src1.format_asm()}"
        if self.clamp:
            s += " clamp"
        return s


class F64Expression(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, env: dict[str, int]) -> int:
        pass


@dataclass
class F64Constant(F64Expression):
    bits: int

    def evaluate(self, env: dict[str, int]) -> int:
        return self.bits & MASK64


@dataclass
class F64Variable(F64Expression):
    name: str

    def evaluate(self, env: dict[str, int]) -> int:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in environment")
        return env[self.name] & MASK64


@dataclass
class F64UnaryOp(F64Expression):
    op: str  # "abs", "neg", "neg_abs"
    operand: F64Expression

    def evaluate(self, env: dict[str, int]) -> int:
        val = self.operand.evaluate(env)
        if self.op == "abs":
            return apply_input_modifiers_f64(val, abs_val=True)
        elif self.op == "neg":
            return apply_input_modifiers_f64(val, neg=True)
        elif self.op == "neg_abs":
            return apply_input_modifiers_f64(val, neg=True, abs_val=True)
        raise ValueError(f"Unknown unary operator: {self.op}")


@dataclass
class F64Mul(F64Expression):
    lhs: F64Expression
    rhs: F64Expression
    clamp: bool = False

    def evaluate(self, env: dict[str, int]) -> int:
        v_lhs = self.lhs.evaluate(env)
        v_rhs = self.rhs.evaluate(env)
        return ref_mul_f64(v_lhs, v_rhs, clamp=self.clamp)


def instruction_to_expression(inst: ASTInstruction) -> F64Mul:
    """Translate an ASTInstruction to a typed symbolic expression tree."""
    inst.validate()

    def build_operand_expr(op: ASTOperand, prefix: str) -> F64Expression:
        if op.kind in (OperandKind.VGPR_PAIR, OperandKind.SGPR_PAIR):
            expr: F64Expression = F64Variable(f"{prefix}_{op.reg_idx}")
        elif op.kind in (OperandKind.LITERAL_64, OperandKind.INLINE_IMM):
            expr = F64Constant(op.imm_val)
        else:
            raise ValueError(f"Unsupported operand kind: {op.kind}")

        if op.abs_val and op.neg:
            expr = F64UnaryOp("neg_abs", expr)
        elif op.abs_val:
            expr = F64UnaryOp("abs", expr)
        elif op.neg:
            expr = F64UnaryOp("neg", expr)
        return expr

    lhs = build_operand_expr(inst.src0, "src0")
    rhs = build_operand_expr(inst.src1, "src1")
    return F64Mul(lhs, rhs, clamp=inst.clamp)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter for v_mul_f64
# ==============================================================================

def parse_asm_operand(tok: str) -> ASTOperand:
    tok = tok.strip()
    neg = False
    abs_val = False
    if tok.startswith("-"):
        neg = True
        tok = tok[1:].strip()
    if tok.startswith("|") and tok.endswith("|"):
        abs_val = True
        tok = tok[1:-1].strip()

    if tok.startswith("v[") and tok.endswith("]"):
        inner = tok[2:-1]
        parts = inner.split(":")
        reg_idx = int(parts[0])
        return ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=reg_idx, neg=neg, abs_val=abs_val)
    elif tok.startswith("s[") and tok.endswith("]"):
        inner = tok[2:-1]
        parts = inner.split(":")
        reg_idx = int(parts[0])
        return ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=reg_idx, neg=neg, abs_val=abs_val)
    elif tok.startswith("0x") or tok.startswith("0X"):
        imm = int(tok, 16)
        return ASTOperand(kind=OperandKind.LITERAL_64, imm_val=imm, neg=neg, abs_val=abs_val)
    else:
        try:
            if "." in tok or "e" in tok.lower():
                imm = f64_to_u64(float(tok))
                return ASTOperand(kind=OperandKind.LITERAL_64, imm_val=imm, neg=neg, abs_val=abs_val)
            imm = int(tok)
            return ASTOperand(kind=OperandKind.INLINE_IMM, imm_val=imm, neg=neg, abs_val=abs_val)
        except ValueError:
            raise ValueError(f"Unrecognized assembly operand token: {tok}")


def parse_asm_line(line: str) -> ASTInstruction:
    clean = line.split(";")[0].split("//")[0].strip()
    tokens = [t for t in clean.replace(",", " ").split() if t]
    if not tokens:
        raise ValueError("Empty assembly line")
    opcode = tokens[0]
    if opcode != "v_mul_f64":
        raise ValueError(f"Expected opcode v_mul_f64, got {opcode}")

    clamp = False
    raw_ops = []
    for tok in tokens[1:]:
        if tok.lower() == "clamp":
            clamp = True
        else:
            raw_ops.append(tok)

    if len(raw_ops) != 3:
        raise ValueError(f"v_mul_f64 requires 3 operands (dst, src0, src1), got {len(raw_ops)}: {raw_ops}")

    dst = parse_asm_operand(raw_ops[0])
    src0 = parse_asm_operand(raw_ops[1])
    src1 = parse_asm_operand(raw_ops[2])
    inst = ASTInstruction(opcode=opcode, dst=dst, src0=src0, src1=src1, clamp=clamp)
    inst.validate()
    return inst


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU v_mul_f64."""

    def __init__(self, num_vgprs: int = 256, num_sgprs: int = 106) -> None:
        self.vgpr = VALURegisterFile(num_vgprs=num_vgprs)
        self.num_sgprs = num_sgprs
        self.sgprs = [0] * num_sgprs
        self.vcc = 0
        self.scc = 0

    def get_sgpr(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: {idx}")
        return self.sgprs[idx] & MASK32

    def set_sgpr(self, idx: int, val: int) -> None:
        if not (0 <= idx < self.num_sgprs):
            raise IndexError(f"SGPR index out of range: {idx}")
        self.sgprs[idx] = val & MASK32

    def get_sgpr_pair(self, idx: int) -> int:
        if not (0 <= idx < self.num_sgprs - 1):
            raise IndexError(f"SGPR pair index out of range: s[{idx}:{idx+1}]")
        lo = self.get_sgpr(idx)
        hi = self.get_sgpr(idx + 1)
        return join_dwords(lo, hi)

    def set_sgpr_pair(self, idx: int, val64: int) -> None:
        if not (0 <= idx < self.num_sgprs - 1):
            raise IndexError(f"SGPR pair index out of range: s[{idx}:{idx+1}]")
        lo, hi = split_dwords(val64)
        self.set_sgpr(idx, lo)
        self.set_sgpr(idx + 1, hi)

    def resolve_operand(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.VGPR_PAIR:
            val = self.vgpr.get_vgpr_pair(op.reg_idx)
        elif op.kind == OperandKind.SGPR_PAIR:
            val = self.get_sgpr_pair(op.reg_idx)
        elif op.kind == OperandKind.LITERAL_64:
            val = op.imm_val
        elif op.kind == OperandKind.INLINE_IMM:
            val = op.imm_val
        else:
            raise ValueError(f"Unknown operand kind: {op.kind}")
        return apply_input_modifiers_f64(val, neg=op.neg, abs_val=op.abs_val)

    def execute_ast(self, inst: ASTInstruction) -> int:
        inst.validate()
        s0 = self.resolve_operand(inst.src0)
        s1 = self.resolve_operand(inst.src1)
        res = ref_mul_f64(s0, s1, clamp=inst.clamp)
        self.vgpr.set_vgpr_pair(inst.dst.reg_idx, res)
        return res

    def execute_asm(self, line: str) -> int:
        inst = parse_asm_line(line)
        return self.execute_ast(inst)


# ==============================================================================
# SIMD Wavefront Execution Model (Wave32 / Wave64) for v_mul_f64
# ==============================================================================

def ref_simd_mul_f64(
    src0_lanes: list[int],
    src1_lanes: list[int],
    wave_size: int = 64,
    exec_mask: int | None = None,
    prev_dst: list[int] | None = None,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
    src1_neg: bool = False,
    src1_abs: bool = False,
) -> list[int]:
    """Reference SIMD execution oracle across wavefront lanes."""
    if wave_size not in (32, 64):
        raise ValueError(f"Unsupported wave size: {wave_size}")
    if len(src0_lanes) != wave_size or len(src1_lanes) != wave_size:
        raise ValueError(f"Operand length must match wave_size ({wave_size})")

    active_mask = (1 << wave_size) - 1 if exec_mask is None else (exec_mask & ((1 << wave_size) - 1))
    fallback = [0] * wave_size if prev_dst is None else prev_dst
    if len(fallback) != wave_size:
        raise ValueError(f"prev_dst length must match wave_size ({wave_size})")

    output = list(fallback)
    for lane in range(wave_size):
        if (active_mask >> lane) & 1:
            output[lane] = ref_mul_f64(
                src0_lanes[lane],
                src1_lanes[lane],
                clamp=clamp,
                src0_neg=src0_neg,
                src0_abs=src0_abs,
                src1_neg=src1_neg,
                src1_abs=src1_abs,
            )
    return output


class SIMDWavefront:
    """SIMD wavefront simulator modeling multi-lane execution and EXEC masking."""

    def __init__(self, wave_size: int = 64, num_vgprs: int = 256) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.num_vgprs = num_vgprs
        self.lanes = [VALURegisterFile(num_vgprs=num_vgprs) for _ in range(wave_size)]
        self.exec_mask = (1 << wave_size) - 1
        self.vcc = 0
        self.sgprs = [0] * 106

    def store_vgpr_pair_wave(self, reg_idx: int, values: list[int]) -> None:
        if len(values) != self.wave_size:
            raise ValueError(f"Value count ({len(values)}) does not match wave_size ({self.wave_size})")
        for lane in range(self.wave_size):
            self.lanes[lane].set_vgpr_pair(reg_idx, values[lane])

    def load_vgpr_pair_wave(self, reg_idx: int) -> list[int]:
        return [self.lanes[lane].get_vgpr_pair(reg_idx) for lane in range(self.wave_size)]

    def step_v_mul_f64(
        self,
        dst_reg: int,
        src0_reg: int,
        src1_reg: int,
        *,
        clamp: bool = False,
        src0_neg: bool = False,
        src0_abs: bool = False,
        src1_neg: bool = False,
        src1_abs: bool = False,
    ) -> list[int]:
        active_mask = self.exec_mask & ((1 << self.wave_size) - 1)
        for lane in range(self.wave_size):
            if (active_mask >> lane) & 1:
                s0 = self.lanes[lane].get_vgpr_pair(src0_reg)
                s1 = self.lanes[lane].get_vgpr_pair(src1_reg)
                res = ref_mul_f64(
                    s0,
                    s1,
                    clamp=clamp,
                    src0_neg=src0_neg,
                    src0_abs=src0_abs,
                    src1_neg=src1_neg,
                    src1_abs=src1_abs,
                )
                self.lanes[lane].set_vgpr_pair(dst_reg, res)
        return self.load_vgpr_pair_wave(dst_reg)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Isolation for v_mul_f64
# ==============================================================================

@dataclass
class StatusFlags:
    invalid_op: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    nan: bool = False


def compute_mul_f64_status_flags(src0: int, src1: int, result: int) -> StatusFlags:
    flags = StatusFlags()
    s0 = src0 & MASK64
    s1 = src1 & MASK64
    res = result & MASK64

    # Invalid operation: SNaN operand or 0 * Inf
    if is_f64_snan(s0) or is_f64_snan(s1):
        flags.invalid_op = True
    elif (is_f64_zero(s0) and is_f64_inf(s1)) or (is_f64_inf(s0) and is_f64_zero(s1)):
        flags.invalid_op = True

    if is_f64_nan(res):
        flags.nan = True
    else:
        flags.negative = bool(f64_sign(res))
        if is_f64_zero(res):
            flags.zero = True
            if not is_f64_zero(s0) and not is_f64_zero(s1) and not is_f64_inf(s0) and not is_f64_inf(s1):
                flags.underflow = True
        else:
            # Check overflow: finite * finite -> inf
            if not is_f64_inf(s0) and not is_f64_inf(s1) and is_f64_inf(res):
                flags.overflow = True
                flags.inexact = True
            # Check underflow: finite non-zero * finite non-zero -> subnormal
            elif not is_f64_zero(s0) and not is_f64_zero(s1) and is_f64_subnormal(res):
                flags.underflow = True

    return flags


def ref_cmp_f64(op: str, src0: int, src1: int) -> bool:
    """Evaluate double-precision IEEE-754 comparison condition codes."""
    if op in ("u", "unordered"):
        return is_f64_nan(src0) or is_f64_nan(src1)

    if is_f64_nan(src0) or is_f64_nan(src1):
        return op in ("neq", "ne", "!=")

    v0 = u64_to_f64(src0)
    v1 = u64_to_f64(src1)
    if op in ("eq", "=="):
        return v0 == v1
    elif op in ("ne", "neq", "!="):
        return v0 != v1
    elif op in ("lt", "<"):
        return v0 < v1
    elif op in ("le", "<="):
        return v0 <= v1
    elif op in ("gt", ">"):
        return v0 > v1
    elif op in ("ge", ">="):
        return v0 >= v1
    raise ValueError(f"Unknown comparison operator: {op}")


def generate_condition_mask_f64(op: str, src0_lanes: list[int], src1_lanes: list[int]) -> int:
    """Generate a 32-bit or 64-bit condition mask (VCC) from per-lane comparisons."""
    mask = 0
    for lane, (s0, s1) in enumerate(zip(src0_lanes, src1_lanes)):
        if ref_cmp_f64(op, s0, s1):
            mask |= (1 << lane)
    return mask


class TestVMulF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_multiplication(self) -> None:
        # 2.0 * 3.5 = 7.0
        r = ref_mul_f64(f64_to_u64(2.0), f64_to_u64(3.5))
        self.assertEqual(u64_to_f64(r), 7.0)

        # -1.5 * 4.0 = -6.0
        r = ref_mul_f64(f64_to_u64(-1.5), f64_to_u64(4.0))
        self.assertEqual(u64_to_f64(r), -6.0)

        # -2.5 * -8.0 = 20.0
        r = ref_mul_f64(f64_to_u64(-2.5), f64_to_u64(-8.0))
        self.assertEqual(u64_to_f64(r), 20.0)

    def test_zero_rules(self) -> None:
        # +0.0 * +0.0 = +0.0
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_POS_ZERO), F64_POS_ZERO)
        # +0.0 * -0.0 = -0.0
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_NEG_ZERO), F64_NEG_ZERO)
        # -0.0 * -0.0 = +0.0
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, F64_NEG_ZERO), F64_POS_ZERO)
        # +0.0 * 5.0 = +0.0
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, f64_to_u64(5.0)), F64_POS_ZERO)
        # -0.0 * 5.0 = -0.0
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, f64_to_u64(5.0)), F64_NEG_ZERO)

    def test_inf_and_zero_rules(self) -> None:
        # 0 * Inf -> Canonical QNaN
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_POS_INF), F64_CANONICAL_QNAN)
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, F64_POS_INF), F64_CANONICAL_QNAN)
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_POS_ZERO), F64_CANONICAL_QNAN)

        # Inf * Inf -> Inf with XOR sign
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_POS_INF), F64_POS_INF)
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_NEG_INF), F64_NEG_INF)
        self.assertEqual(ref_mul_f64(F64_NEG_INF, F64_NEG_INF), F64_POS_INF)

        # Inf * finite non-zero -> Inf with XOR sign
        self.assertEqual(ref_mul_f64(F64_POS_INF, f64_to_u64(2.0)), F64_POS_INF)
        self.assertEqual(ref_mul_f64(F64_POS_INF, f64_to_u64(-2.0)), F64_NEG_INF)

    def test_input_modifiers(self) -> None:
        # neg: 2.0 * neg(3.0) = -6.0
        r = ref_mul_f64(f64_to_u64(2.0), f64_to_u64(3.0), src1_neg=True)
        self.assertEqual(u64_to_f64(r), -6.0)

        # abs: -2.0 * abs(-3.0) = -6.0
        r = ref_mul_f64(f64_to_u64(-2.0), f64_to_u64(-3.0), src1_abs=True)
        self.assertEqual(u64_to_f64(r), -6.0)

        # abs + neg on src0: neg(abs(-2.0)) = -2.0; * 3.0 = -6.0
        r = ref_mul_f64(f64_to_u64(-2.0), f64_to_u64(3.0), src0_abs=True, src0_neg=True)
        self.assertEqual(u64_to_f64(r), -6.0)

    def test_clamp_modifier(self) -> None:
        # In range [0.0, 1.0]: 0.5 * 1.5 = 0.75
        r = ref_mul_f64(f64_to_u64(0.5), f64_to_u64(1.5), clamp=True)
        self.assertEqual(u64_to_f64(r), 0.75)

        # > 1.0 clamped to 1.0: 2.0 * 3.0 = 6.0 -> 1.0
        r = ref_mul_f64(f64_to_u64(2.0), f64_to_u64(3.0), clamp=True)
        self.assertEqual(r, F64_ONE)

        # < 0.0 clamped to +0.0: -2.0 * 3.0 = -6.0 -> +0.0
        r = ref_mul_f64(f64_to_u64(-2.0), f64_to_u64(3.0), clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

        # NaN clamped to +0.0
        r = ref_mul_f64(F64_POS_ZERO, F64_POS_INF, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(12.345)
        s1 = f64_to_u64(67.89)
        s0_lo, s0_hi = split_dwords(s0)
        s1_lo, s1_hi = split_dwords(s1)

        lo, hi = ref_mul_f64_dwords(s0_lo, s0_hi, s1_lo, s1_hi)
        res64 = join_dwords(lo, hi)
        expected = ref_mul_f64(s0, s1)
        self.assertEqual(res64, expected)

    def test_register_file_pair_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        val = 0x01234567_89ABCDEF
        rf.set_vgpr_pair(4, val)

        self.assertEqual(rf.get_vgpr(4), 0x89ABCDEF)
        self.assertEqual(rf.get_vgpr(5), 0x01234567)
        self.assertEqual(rf.get_vgpr_pair(4), val)

        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)


class TestSymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST definitions and typed expressions."""

    def test_ast_creation_and_formatting(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=4)
        inst = ASTInstruction(opcode="v_mul_f64", dst=dst, src0=src0, src1=src1)
        inst.validate()
        self.assertEqual(inst.format_asm(), "v_mul_f64 v[0:1], v[2:3], s[4:5]")

        # With clamp and modifiers
        src0_mod = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2, neg=True, abs_val=True)
        inst_clamp = ASTInstruction(opcode="v_mul_f64", dst=dst, src0=src0_mod, src1=src1, clamp=True)
        inst_clamp.validate()
        self.assertEqual(inst_clamp.format_asm(), "v_mul_f64 v[0:1], -|v[2:3]|, s[4:5] clamp")

    def test_typed_expression_evaluation(self) -> None:
        c1 = F64Constant(f64_to_u64(2.5))
        c2 = F64Constant(f64_to_u64(4.0))
        mul_expr = F64Mul(lhs=c1, rhs=c2)
        res = mul_expr.evaluate({})
        self.assertEqual(u64_to_f64(res), 10.0)

        # Unary operations
        neg_c1 = F64UnaryOp("neg", c1)
        res_neg = F64Mul(lhs=neg_c1, rhs=c2).evaluate({})
        self.assertEqual(u64_to_f64(res_neg), -10.0)

    def test_instruction_to_expression(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2, neg=True)
        src1 = ASTOperand(kind=OperandKind.LITERAL_64, imm_val=f64_to_u64(3.0))
        inst = ASTInstruction(opcode="v_mul_f64", dst=dst, src0=src0, src1=src1, clamp=False)

        expr = instruction_to_expression(inst)
        env = {"src0_2": f64_to_u64(4.0)}
        res = expr.evaluate(env)
        # -4.0 * 3.0 = -12.0
        self.assertEqual(u64_to_f64(res), -12.0)

    def test_ast_validation_errors(self) -> None:
        # Invalid opcode
        bad_op = ASTInstruction(
            opcode="v_add_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
            src1=ASTOperand(OperandKind.VGPR_PAIR, 4),
        )
        with self.assertRaises(ValueError):
            bad_op.validate()

        # Destination cannot be SGPR
        bad_dst = ASTInstruction(
            opcode="v_mul_f64",
            dst=ASTOperand(OperandKind.SGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
            src1=ASTOperand(OperandKind.VGPR_PAIR, 4),
        )
        with self.assertRaises(ValueError):
            bad_dst.validate()

        # Destination out of bounds
        oob_dst = ASTInstruction(
            opcode="v_mul_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 256),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
            src1=ASTOperand(OperandKind.VGPR_PAIR, 4),
        )
        with self.assertRaises(ValueError):
            oob_dst.validate()


class TestScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_execute_vgpr_sources(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(3.0))
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(4.5))

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=4)
        inst = ASTInstruction(opcode="v_mul_f64", dst=dst, src0=src0, src1=src1)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), 13.5)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_sgpr_sources(self) -> None:
        interp = ScalarInterpreter()
        interp.set_sgpr_pair(0, f64_to_u64(-2.0))
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(7.0))

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=4)
        src0 = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=0)
        src1 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        inst = ASTInstruction(opcode="v_mul_f64", dst=dst, src0=src0, src1=src1)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), -14.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(4), res)

    def test_execute_literal_source(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(5.0))

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        src1 = ASTOperand(kind=OperandKind.LITERAL_64, imm_val=f64_to_u64(0.2))
        inst = ASTInstruction(opcode="v_mul_f64", dst=dst, src0=src0, src1=src1)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), 1.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_text_asm(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(6, f64_to_u64(1.25))
        interp.set_sgpr_pair(8, f64_to_u64(8.0))

        res = interp.execute_asm("v_mul_f64 v[0:1], v[6:7], s[8:9]")
        self.assertEqual(u64_to_f64(res), 10.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), f64_to_u64(10.0))

    def test_execute_modifiers_and_clamp(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(-1.5))
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(-0.5))

        # -|v[2:3]| is -1.5, multiplied by -0.5 is 0.75, within clamp [0, 1]
        res = interp.execute_asm("v_mul_f64 v[0:1], -|v[2:3]|, v[4:5] clamp")
        self.assertEqual(u64_to_f64(res), 0.75)

        # > 1.0 clamped
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(-2.0))
        # -1.5 * -2.0 = 3.0 -> clamped to 1.0
        res2 = interp.execute_asm("v_mul_f64 v[0:1], -|v[2:3]|, v[4:5] clamp")
        self.assertEqual(res2, F64_ONE)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_active(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        s0_vals = [f64_to_u64(float(i + 1)) for i in range(32)]
        s1_vals = [f64_to_u64(2.5) for _ in range(32)]
        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)

        res = sim.step_v_mul_f64(0, 2, 4)
        for i in range(32):
            self.assertEqual(u64_to_f64(res[i]), (i + 1) * 2.5)

    def test_wave64_all_active(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        s0_vals = [f64_to_u64(float(i + 1)) for i in range(64)]
        s1_vals = [f64_to_u64(0.125) for _ in range(64)]
        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)

        res = sim.step_v_mul_f64(0, 2, 4)
        for i in range(64):
            self.assertEqual(u64_to_f64(res[i]), (i + 1) * 0.125)

    def test_wave32_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 32)

        s0_vals = [f64_to_u64(3.0)] * 32
        s1_vals = [f64_to_u64(4.0)] * 32
        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)

        # Alternating lanes active: 0b0101...
        sim.exec_mask = 0x55555555
        res = sim.step_v_mul_f64(0, 2, 4)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(u64_to_f64(res[lane]), 12.0)
            else:
                self.assertEqual(res[lane], sentinel_pair)

    def test_wave64_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 64)

        s0_vals = [f64_to_u64(2.0)] * 64
        s1_vals = [f64_to_u64(5.0)] * 64
        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)

        # Lower 32 lanes active, upper 32 inactive
        sim.exec_mask = 0x00000000_FFFFFFFF
        res = sim.step_v_mul_f64(0, 2, 4)

        for lane in range(32):
            self.assertEqual(u64_to_f64(res[lane]), 10.0)
        for lane in range(32, 64):
            self.assertEqual(res[lane], sentinel_pair)

    def test_wave32_upper_exec_bits_ignored(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 32)
        sim.store_vgpr_pair_wave(2, [f64_to_u64(2.0)] * 32)
        sim.store_vgpr_pair_wave(4, [f64_to_u64(3.0)] * 32)

        # In Wave32, upper 32 bits of exec_mask must have no effect
        sim.exec_mask = 0xDEADBEEF_FFFFFFFF
        res = sim.step_v_mul_f64(0, 2, 4)
        for lane in range(32):
            self.assertEqual(u64_to_f64(res[lane]), 6.0)

    def test_simd_oracle_validation(self) -> None:
        wave_size = 64
        s0_vals = [f64_to_u64(float(i + 1.5)) for i in range(wave_size)]
        s1_vals = [f64_to_u64(float(i + 0.5)) for i in range(wave_size)]
        exec_mask = 0xAAAAAAAAAAAAAAAA

        prev_dst = [join_dwords(0xCAFE, i) for i in range(wave_size)]
        expected = ref_simd_mul_f64(s0_vals, s1_vals, wave_size=wave_size, exec_mask=exec_mask, prev_dst=prev_dst)

        sim = SIMDWavefront(wave_size=wave_size)
        sim.store_vgpr_pair_wave(0, prev_dst)
        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.store_vgpr_pair_wave(4, s1_vals)
        sim.exec_mask = exec_mask

        actual = sim.step_v_mul_f64(0, 2, 4)
        self.assertEqual(actual, expected)


class TestRegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_dst_equals_src0(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(3.0))
        interp.vgpr.set_vgpr_pair(6, f64_to_u64(5.0))

        # v[4:5] = v[4:5] * v[6:7]
        res = interp.execute_asm("v_mul_f64 v[4:5], v[4:5], v[6:7]")
        self.assertEqual(u64_to_f64(res), 15.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(4), f64_to_u64(15.0))
        # src1 v[6:7] must remain intact
        self.assertEqual(u64_to_f64(interp.vgpr.get_vgpr_pair(6)), 5.0)

    def test_inplace_dst_equals_src1(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(7.0))
        interp.vgpr.set_vgpr_pair(6, f64_to_u64(2.0))

        # v[6:7] = v[4:5] * v[6:7]
        res = interp.execute_asm("v_mul_f64 v[6:7], v[4:5], v[6:7]")
        self.assertEqual(u64_to_f64(res), 14.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(6), f64_to_u64(14.0))
        # src0 v[4:5] must remain intact
        self.assertEqual(u64_to_f64(interp.vgpr.get_vgpr_pair(4)), 7.0)

    def test_inplace_self_squaring(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(-4.0))

        # v[2:3] = v[2:3] * v[2:3] -> 16.0
        res = interp.execute_asm("v_mul_f64 v[2:3], v[2:3], v[2:3]")
        self.assertEqual(u64_to_f64(res), 16.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(2), f64_to_u64(16.0))

    def test_partially_overlapping_register_pairs(self) -> None:
        interp = ScalarInterpreter()
        # Setup:
        # v0 = 0xAAAA0000, v1 = 0x0000BBBB -> src0 v[0:1]
        # v2 = 0xCCCC0000, v3 = 0x0000DDDD -> src1 v[2:3]
        s0_val = f64_to_u64(3.0)
        s1_val = f64_to_u64(6.0)
        interp.vgpr.set_vgpr_pair(0, s0_val)
        interp.vgpr.set_vgpr_pair(2, s1_val)

        v0_before = interp.vgpr.get_vgpr(0)
        v3_before = interp.vgpr.get_vgpr(3)

        # Execute with dst as unaligned v[1:2] overlapping both src0 (via v1) and src1 (via v2)
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=1)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src1 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        inst = ASTInstruction(opcode="v_mul_f64", dst=dst, src0=src0, src1=src1)
        res = interp.execute_ast(inst)

        self.assertEqual(u64_to_f64(res), 18.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(1), f64_to_u64(18.0))
        # Non-overlapping boundary dwords v[0] and v[3] must remain completely unchanged
        self.assertEqual(interp.vgpr.get_vgpr(0), v0_before)
        self.assertEqual(interp.vgpr.get_vgpr(3), v3_before)

    def test_simd_inplace_divergent_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        init_vals = [f64_to_u64(float(i + 1)) for i in range(32)]
        scale_vals = [f64_to_u64(10.0) for _ in range(32)]
        sim.store_vgpr_pair_wave(0, init_vals)
        sim.store_vgpr_pair_wave(2, scale_vals)

        # Active on odd lanes only: 0xAAAAAAAA
        sim.exec_mask = 0xAAAAAAAA
        res = sim.step_v_mul_f64(0, 0, 2)

        for lane in range(32):
            if lane % 2 == 1:
                # Active lane: multiplied
                self.assertEqual(u64_to_f64(res[lane]), (lane + 1) * 10.0)
            else:
                # Inactive lane: preserved original
                self.assertEqual(res[lane], init_vals[lane])

    def test_successive_inplace_chain(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(0, f64_to_u64(1.0))
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(2.0))

        # Repeatedly double v[0:1] 5 times: 1.0 -> 2.0 -> 4.0 -> 8.0 -> 16.0 -> 32.0
        for _ in range(5):
            interp.execute_asm("v_mul_f64 v[0:1], v[0:1], v[2:3]")

        self.assertEqual(u64_to_f64(interp.vgpr.get_vgpr_pair(0)), 32.0)


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow isolation."""

    def test_status_flags_invalid_operation(self) -> None:
        # 0.0 * Inf
        res = ref_mul_f64(F64_POS_ZERO, F64_POS_INF)
        flags = compute_mul_f64_status_flags(F64_POS_ZERO, F64_POS_INF, res)
        self.assertTrue(flags.invalid_op)
        self.assertTrue(flags.nan)

        # SNaN * 2.0 (SNaN has quiet bit = 0)
        snan = (F64_POS_INF | 0x0001000000000000) & ~F64_QUIET_BIT
        res_snan = ref_mul_f64(snan, F64_TWO)
        flags_snan = compute_mul_f64_status_flags(snan, F64_TWO, res_snan)
        self.assertTrue(flags_snan.invalid_op)
        self.assertTrue(flags_snan.nan)

    def test_status_flags_overflow(self) -> None:
        # (1e200) * (1e200) overflows binary64
        val = f64_to_u64(1e200)
        res = ref_mul_f64(val, val)
        flags = compute_mul_f64_status_flags(val, val, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.inexact)
        self.assertEqual(res, F64_POS_INF)

    def test_status_flags_underflow(self) -> None:
        # (1e-200) * (1e-200) underflows to 0.0
        val = f64_to_u64(1e-200)
        res = ref_mul_f64(val, val)
        flags = compute_mul_f64_status_flags(val, val, res)
        self.assertTrue(flags.underflow)
        self.assertTrue(flags.zero)

    def test_status_flags_zero_and_negative(self) -> None:
        # -2.0 * 3.0 = -6.0 (negative, non-zero)
        v0 = f64_to_u64(-2.0)
        v1 = f64_to_u64(3.0)
        r1 = ref_mul_f64(v0, v1)
        flags1 = compute_mul_f64_status_flags(v0, v1, r1)
        self.assertTrue(flags1.negative)
        self.assertFalse(flags1.zero)

        # -0.0 * 3.0 = -0.0 (negative zero)
        r2 = ref_mul_f64(F64_NEG_ZERO, v1)
        flags2 = compute_mul_f64_status_flags(F64_NEG_ZERO, v1, r2)
        self.assertTrue(flags2.negative)
        self.assertTrue(flags2.zero)

    def test_vcc_and_scc_preserved(self) -> None:
        # Floating point v_mul_f64 does not generate carry/borrow and never writes VCC or SCC
        interp = ScalarInterpreter()
        interp.vcc = 0x12345678_9ABCDEF0
        interp.scc = 1
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(4.0))
        interp.vgpr.set_vgpr_pair(4, f64_to_u64(5.0))

        interp.execute_asm("v_mul_f64 v[0:1], v[2:3], v[4:5]")
        self.assertEqual(interp.vcc, 0x12345678_9ABCDEF0)
        self.assertEqual(interp.scc, 1)

        sim = SIMDWavefront(wave_size=32)
        sim.vcc = 0xCAFEBABE
        sim.store_vgpr_pair_wave(2, [f64_to_u64(2.0)] * 32)
        sim.store_vgpr_pair_wave(4, [f64_to_u64(3.0)] * 32)
        sim.step_v_mul_f64(0, 2, 4)
        self.assertEqual(sim.vcc, 0xCAFEBABE)

    def test_condition_code_generation(self) -> None:
        # Generate condition codes from comparisons on multiplication results
        v_zero = [F64_POS_ZERO] * 4
        # Four lanes: products will be [-4.0, 0.0, 4.0, NaN]
        s0 = [f64_to_u64(-2.0), f64_to_u64(0.0), f64_to_u64(2.0), F64_POS_ZERO]
        s1 = [f64_to_u64(2.0), f64_to_u64(5.0), f64_to_u64(2.0), F64_POS_INF]
        prods = [ref_mul_f64(a, b) for a, b in zip(s0, s1)]

        # Compare < 0: only lane 0 (-4.0 < 0) -> bit 0
        mask_lt = generate_condition_mask_f64("<", prods, v_zero)
        self.assertEqual(mask_lt, 0b0001)

        # Compare == 0: only lane 1 (0.0 == 0) -> bit 1
        mask_eq = generate_condition_mask_f64("==", prods, v_zero)
        self.assertEqual(mask_eq, 0b0010)

        # Compare > 0: only lane 2 (4.0 > 0) -> bit 2
        mask_gt = generate_condition_mask_f64(">", prods, v_zero)
        self.assertEqual(mask_gt, 0b0100)

        # Compare unordered: only lane 3 (NaN) -> bit 3
        mask_u = generate_condition_mask_f64("u", prods, v_zero)
        self.assertEqual(mask_u, 0b1000)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Test suite for IEEE-754 boundary extremes, subnormals, and NaN handling."""

    def test_nan_snan_quieting(self) -> None:
        # Positive signaling NaN
        snan_pos = 0x7FF0000000000001
        self.assertTrue(is_f64_snan(snan_pos))
        res_pos = ref_mul_f64(snan_pos, F64_ONE)
        self.assertTrue(is_f64_qnan(res_pos))
        self.assertEqual(res_pos, 0x7FF8000000000001)

        # Negative signaling NaN
        snan_neg = 0xFFF0000012345678
        self.assertTrue(is_f64_snan(snan_neg))
        res_neg = ref_mul_f64(F64_TWO, snan_neg)
        self.assertTrue(is_f64_qnan(res_neg))
        self.assertEqual(res_neg, 0xFFF8000012345678)

    def test_nan_payload_propagation(self) -> None:
        # Quiet NaN payload from src0
        qnan_a = 0x7FF8AABBCCDDEEFF
        res_a = ref_mul_f64(qnan_a, f64_to_u64(5.0))
        self.assertEqual(res_a, qnan_a)

        # Quiet NaN payload from src1
        qnan_b = 0xFFF8112233445566
        res_b = ref_mul_f64(f64_to_u64(-3.0), qnan_b)
        self.assertEqual(res_b, qnan_b)

    def test_nan_dual_precedence(self) -> None:
        # When both are NaN, src0 takes precedence
        qnan0 = 0x7FF8AAAA00000000
        qnan1 = 0x7FF8BBBB00000000
        res = ref_mul_f64(qnan0, qnan1)
        self.assertEqual(res, qnan0)

    def test_signed_zero_combinations(self) -> None:
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_POS_ZERO), F64_POS_ZERO)
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, F64_NEG_ZERO), F64_NEG_ZERO)
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, F64_POS_ZERO), F64_NEG_ZERO)
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, F64_NEG_ZERO), F64_POS_ZERO)

        # With finite numbers
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, f64_to_u64(10.0)), F64_POS_ZERO)
        self.assertEqual(ref_mul_f64(F64_POS_ZERO, f64_to_u64(-10.0)), F64_NEG_ZERO)
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, f64_to_u64(10.0)), F64_NEG_ZERO)
        self.assertEqual(ref_mul_f64(F64_NEG_ZERO, f64_to_u64(-10.0)), F64_POS_ZERO)

    def test_signed_infinity_combinations(self) -> None:
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_POS_INF), F64_POS_INF)
        self.assertEqual(ref_mul_f64(F64_POS_INF, F64_NEG_INF), F64_NEG_INF)
        self.assertEqual(ref_mul_f64(F64_NEG_INF, F64_POS_INF), F64_NEG_INF)
        self.assertEqual(ref_mul_f64(F64_NEG_INF, F64_NEG_INF), F64_POS_INF)

        # With finite numbers
        self.assertEqual(ref_mul_f64(F64_POS_INF, f64_to_u64(2.0)), F64_POS_INF)
        self.assertEqual(ref_mul_f64(F64_POS_INF, f64_to_u64(-2.0)), F64_NEG_INF)
        self.assertEqual(ref_mul_f64(F64_NEG_INF, f64_to_u64(2.0)), F64_NEG_INF)
        self.assertEqual(ref_mul_f64(F64_NEG_INF, f64_to_u64(-2.0)), F64_POS_INF)

    def test_max_normal_overflow(self) -> None:
        # DBL_MAX * 1.0 = DBL_MAX
        self.assertEqual(ref_mul_f64(F64_MAX_NORMAL, F64_ONE), F64_MAX_NORMAL)
        # DBL_MAX * -1.0 = -DBL_MAX
        self.assertEqual(ref_mul_f64(F64_MAX_NORMAL, F64_NEG_ONE), F64_MAX_NORMAL | F64_SIGN_MASK)
        # DBL_MAX * 2.0 = +inf (overflow)
        self.assertEqual(ref_mul_f64(F64_MAX_NORMAL, F64_TWO), F64_POS_INF)
        # DBL_MAX * -2.0 = -inf (overflow)
        self.assertEqual(ref_mul_f64(F64_MAX_NORMAL, f64_to_u64(-2.0)), F64_NEG_INF)

    def test_min_normal_underflow_to_subnormal(self) -> None:
        # DBL_MIN * 1.0 = DBL_MIN
        self.assertEqual(ref_mul_f64(F64_MIN_NORMAL, F64_ONE), F64_MIN_NORMAL)
        # DBL_MIN * 0.5 = subnormal 2^-1023
        res = ref_mul_f64(F64_MIN_NORMAL, F64_HALF)
        self.assertTrue(is_f64_subnormal(res))
        self.assertEqual(res, 0x0008000000000000)

    def test_subnormal_boundaries(self) -> None:
        # Min subnormal (2^-1074) * 1.0 = min subnormal
        self.assertEqual(ref_mul_f64(F64_MIN_SUBNORMAL, F64_ONE), F64_MIN_SUBNORMAL)
        # Min subnormal * 2.0 = 2^-1073 (0x2)
        self.assertEqual(ref_mul_f64(F64_MIN_SUBNORMAL, F64_TWO), 0x0000000000000002)
        # Min subnormal * 0.5 = underflow to +0.0
        self.assertEqual(ref_mul_f64(F64_MIN_SUBNORMAL, F64_HALF), F64_POS_ZERO)
        # -Min subnormal * 0.5 = underflow to -0.0
        self.assertEqual(ref_mul_f64(F64_MIN_SUBNORMAL | F64_SIGN_MASK, F64_HALF), F64_NEG_ZERO)

    def test_round_to_nearest_even(self) -> None:
        v_base = 1.0 + 2.0**(-52)
        bits_base = f64_to_u64(v_base)
        res = ref_mul_f64(bits_base, F64_ONE)
        self.assertEqual(res, bits_base)


# ==============================================================================
# Mutants for Mutation Detection
# ==============================================================================

def mutant_mul_as_add(s0: int, s1: int, **kwargs) -> int:
    """Mutant 1: Replaces multiplication with addition."""
    v0 = u64_to_f64(s0)
    v1 = u64_to_f64(s1)
    return f64_to_u64(v0 + v1)


def mutant_ignore_neg(s0: int, s1: int, *, src0_neg: bool = False, src1_neg: bool = False, **kwargs) -> int:
    """Mutant 2: Ignores negative modifiers."""
    return ref_mul_f64(s0, s1, src0_neg=False, src1_neg=False, **kwargs)


def mutant_ignore_abs(s0: int, s1: int, *, src0_abs: bool = False, src1_abs: bool = False, **kwargs) -> int:
    """Mutant 3: Ignores absolute value modifiers."""
    return ref_mul_f64(s0, s1, src0_abs=False, src1_abs=False, **kwargs)


def mutant_ignore_clamp(s0: int, s1: int, *, clamp: bool = False, **kwargs) -> int:
    """Mutant 4: Ignores clamp."""
    return ref_mul_f64(s0, s1, clamp=False, **kwargs)


def mutant_clamp_nan_to_nan(s0: int, s1: int, *, clamp: bool = False, **kwargs) -> int:
    """Mutant 5: Clamps numbers but fails to clamp NaN to +0.0."""
    res = ref_mul_f64(s0, s1, clamp=False, **kwargs)
    if clamp and not is_f64_nan(res):
        val = u64_to_f64(res)
        if val < 0.0:
            return F64_POS_ZERO
        elif val > 1.0:
            return F64_ONE
    return res


def mutant_snan_not_quieted(s0: int, s1: int, **kwargs) -> int:
    """Mutant 6: Returns SNaN without quieting."""
    if is_f64_snan(s0):
        return s0
    if is_f64_snan(s1):
        return s1
    return ref_mul_f64(s0, s1, **kwargs)


def mutant_zero_inf_returns_zero(s0: int, s1: int, **kwargs) -> int:
    """Mutant 7: 0 * Inf returns 0 instead of canonical QNaN."""
    if (is_f64_zero(s0) and is_f64_inf(s1)) or (is_f64_inf(s0) and is_f64_zero(s1)):
        return F64_POS_ZERO
    return ref_mul_f64(s0, s1, **kwargs)


def mutant_f32_truncation(s0: int, s1: int, **kwargs) -> int:
    """Mutant 8: Truncates operands to single-precision float."""
    f0 = struct.unpack("<f", struct.pack("<f", u64_to_f64(s0)))[0]
    f1 = struct.unpack("<f", struct.pack("<f", u64_to_f64(s1)))[0]
    return f64_to_u64(float(f0 * f1))


class TestNegativeControlsAndSyntaxRejection(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_bad_opcodes(self) -> None:
        for bad_op in ("v_mul_f32", "v_add_f64", "s_mul_f64", "v_mul_i64", "foo"):
            with self.assertRaises(ValueError):
                parse_asm_line(f"{bad_op} v[0:1], v[2:3], v[4:5]")

    def test_syntax_rejection_bad_operands(self) -> None:
        # Malformed register syntax
        for bad_line in (
            "v_mul_f64 v0, v[2:3], v[4:5]",
            "v_mul_f64 v[0:1], invalid_reg, v[4:5]",
            "v_mul_f64 v[0:1], v[2:3], xyz",
        ):
            with self.assertRaises(ValueError):
                parse_asm_line(bad_line)

    def test_syntax_rejection_operand_count(self) -> None:
        # Too few operands
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_f64 v[0:1], v[2:3]")
        # Too many operands
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_f64 v[0:1], v[2:3], v[4:5], v[6:7]")

    def test_syntax_rejection_destination_modifiers(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_f64 -v[0:1], v[2:3], v[4:5]")
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_f64 |v[0:1]|, v[2:3], v[4:5]")

    def test_syntax_rejection_empty_lines(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("")
        with self.assertRaises(ValueError):
            parse_asm_line("   ; comment only")

    def test_negative_control_add_instead_of_mul(self) -> None:
        # Multiplying 3.0 * 4.0 should be 12.0, not 7.0
        v0 = f64_to_u64(3.0)
        v1 = f64_to_u64(4.0)
        good = ref_mul_f64(v0, v1)
        bad = mutant_mul_as_add(v0, v1)
        self.assertNotEqual(good, bad)
        self.assertEqual(u64_to_f64(good), 12.0)
        self.assertEqual(u64_to_f64(bad), 7.0)

    def test_negative_control_single_precision_loss(self) -> None:
        # A precision pattern that fits binary64 but loses bits in binary32:
        # 1.0000000000001 * 1.0000000000001
        v = 1.0 + 1e-12
        v_bits = f64_to_u64(v)
        good = ref_mul_f64(v_bits, v_bits)
        bad = mutant_f32_truncation(v_bits, v_bits)
        self.assertNotEqual(good, bad)

    def test_mutation_detection_all_mutants_caught(self) -> None:
        # Verify that our test suite discriminator catches every mutant:
        # Mutant 1: add instead of mul
        self.assertNotEqual(ref_mul_f64(f64_to_u64(3.0), f64_to_u64(4.0)),
                            mutant_mul_as_add(f64_to_u64(3.0), f64_to_u64(4.0)))

        # Mutant 2: ignore neg
        self.assertNotEqual(ref_mul_f64(f64_to_u64(3.0), f64_to_u64(4.0), src0_neg=True),
                            mutant_ignore_neg(f64_to_u64(3.0), f64_to_u64(4.0), src0_neg=True))

        # Mutant 3: ignore abs
        self.assertNotEqual(ref_mul_f64(f64_to_u64(-3.0), f64_to_u64(4.0), src0_abs=True),
                            mutant_ignore_abs(f64_to_u64(-3.0), f64_to_u64(4.0), src0_abs=True))

        # Mutant 4: ignore clamp
        self.assertNotEqual(ref_mul_f64(f64_to_u64(2.0), f64_to_u64(3.0), clamp=True),
                            mutant_ignore_clamp(f64_to_u64(2.0), f64_to_u64(3.0), clamp=True))

        # Mutant 5: clamp NaN to NaN instead of +0.0
        self.assertNotEqual(ref_mul_f64(F64_POS_ZERO, F64_POS_INF, clamp=True),
                            mutant_clamp_nan_to_nan(F64_POS_ZERO, F64_POS_INF, clamp=True))

        # Mutant 6: SNaN not quieted
        snan = 0x7FF0000000000001
        self.assertNotEqual(ref_mul_f64(snan, F64_ONE),
                            mutant_snan_not_quieted(snan, F64_ONE))

        # Mutant 7: 0 * Inf returns 0 instead of QNaN
        self.assertNotEqual(ref_mul_f64(F64_POS_ZERO, F64_POS_INF),
                            mutant_zero_inf_returns_zero(F64_POS_ZERO, F64_POS_INF))

        # Mutant 8: f32 truncation
        v = f64_to_u64(1.0000000000001)
        self.assertNotEqual(ref_mul_f64(v, v),
                            mutant_f32_truncation(v, v))


if __name__ == "__main__":
    unittest.main()
