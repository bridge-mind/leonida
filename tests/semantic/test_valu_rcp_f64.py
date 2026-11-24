"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_rcp_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point quantities (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operand
  v[s0:s0+1] (or scalar SGPR pair s[k:k+1], literal 64-bit constant, or inline constant).

  Instruction:
    v_rcp_f64 v[d:d+1], src0 [clamp] [omod]

  Reciprocal definition:
    D.f64 = 1.0 / S0.f64

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
F64_FOUR = 0x4010000000000000         # 4.0
F64_QUARTER = 0x3FD0000000000000      # 0.25
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


def apply_f64_clamp(bits: int) -> int:
    """Clamp double precision value to [0.0, 1.0]. NaN clamps to +0.0."""
    if is_f64_nan(bits):
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0 or is_f64_zero(bits):
        return F64_POS_ZERO
    elif val > 1.0 or is_f64_inf(bits):
        return F64_ONE
    return bits & MASK64


def ref_rcp_f64(
    src0_bits: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 double-precision reciprocal v_rcp_f64.

    Evaluates:
      D.f64 = 1.0 / S0.f64
    accounting for IEEE-754 binary64 rules, input modifiers (neg, abs), and clamp [0.0, 1.0].
    """
    s0 = apply_input_modifiers_f64(src0_bits, neg=src0_neg, abs_val=src0_abs)
    sign = f64_sign(s0)

    # NaN input yields quiet NaN
    if is_f64_nan(s0):
        res = quiet_f64(s0)
    # 1.0 / +-0.0 = +-Inf
    elif is_f64_zero(s0):
        res = F64_NEG_INF if sign else F64_POS_INF
    # 1.0 / +-Inf = +-0.0
    elif is_f64_inf(s0):
        res = F64_NEG_ZERO if sign else F64_POS_ZERO
    else:
        val0 = u64_to_f64(s0)
        try:
            recip = 1.0 / val0
            res = f64_to_u64(recip)
            if is_f64_zero(res):
                res = F64_NEG_ZERO if sign else F64_POS_ZERO
            elif is_f64_inf(res):
                res = F64_NEG_INF if sign else F64_POS_INF
        except OverflowError:
            res = F64_NEG_INF if sign else F64_POS_INF

    if clamp:
        res = apply_f64_clamp(res)

    return res & MASK64


def ref_rcp_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> tuple[int, int]:
    """Decomposed dword interface for v_rcp_f64."""
    s0 = join_dwords(s0_lo, s0_hi)
    res = ref_rcp_f64(s0, clamp=clamp, src0_neg=src0_neg, src0_abs=src0_abs)
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
# Symbolic AST Opcode Definitions and Typed Expressions for v_rcp_f64
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
    clamp: bool = False
    omod: int = 0

    def validate(self) -> None:
        if self.opcode != "v_rcp_f64":
            raise ValueError(f"Invalid opcode for v_rcp_f64 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise ValueError(f"Destination must be a VGPR pair, got {self.dst.kind}")
        if not (0 <= self.dst.reg_idx <= 254):
            raise ValueError(f"Destination VGPR pair index out of range: {self.dst.reg_idx}")
        if self.dst.neg or self.dst.abs_val:
            raise ValueError("Destination operand cannot have neg or abs modifiers")
        if self.src0.kind == OperandKind.VGPR_PAIR:
            if not (0 <= self.src0.reg_idx <= 254):
                raise ValueError(f"src0 VGPR pair index out of range: {self.src0.reg_idx}")
        elif self.src0.kind == OperandKind.SGPR_PAIR:
            if not (0 <= self.src0.reg_idx <= 104):
                raise ValueError(f"src0 SGPR pair index out of range: {self.src0.reg_idx}")

    def format_asm(self) -> str:
        s = f"{self.opcode} {self.dst.format_asm()}, {self.src0.format_asm()}"
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
class F64Rcp(F64Expression):
    operand: F64Expression
    clamp: bool = False

    def evaluate(self, env: dict[str, int]) -> int:
        val = self.operand.evaluate(env)
        return ref_rcp_f64(val, clamp=self.clamp)


def instruction_to_expression(inst: ASTInstruction) -> F64Rcp:
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

    src_expr = build_operand_expr(inst.src0, "src0")
    return F64Rcp(operand=src_expr, clamp=inst.clamp)


# ==============================================================================
# Deterministic Scalar Instruction Interpreter for v_rcp_f64
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
    if opcode != "v_rcp_f64":
        raise ValueError(f"Expected opcode v_rcp_f64, got {opcode}")

    clamp = False
    raw_ops = []
    for tok in tokens[1:]:
        if tok.lower() == "clamp":
            clamp = True
        else:
            raw_ops.append(tok)

    if len(raw_ops) != 2:
        raise ValueError(f"v_rcp_f64 requires 2 operands (dst, src0), got {len(raw_ops)}: {raw_ops}")

    dst = parse_asm_operand(raw_ops[0])
    src0 = parse_asm_operand(raw_ops[1])
    inst = ASTInstruction(opcode=opcode, dst=dst, src0=src0, clamp=clamp)
    inst.validate()
    return inst


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU v_rcp_f64."""

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
        res = ref_rcp_f64(s0, clamp=inst.clamp)
        self.vgpr.set_vgpr_pair(inst.dst.reg_idx, res)
        return res

    def execute_asm(self, line: str) -> int:
        inst = parse_asm_line(line)
        return self.execute_ast(inst)


# ==============================================================================
# SIMD Wavefront Execution Model (Wave32 / Wave64) for v_rcp_f64
# ==============================================================================

def ref_simd_rcp_f64(
    src0_lanes: list[int],
    wave_size: int = 64,
    exec_mask: int | None = None,
    prev_dst: list[int] | None = None,
    *,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> list[int]:
    """Reference SIMD execution oracle across wavefront lanes."""
    if wave_size not in (32, 64):
        raise ValueError(f"Unsupported wave size: {wave_size}")
    if len(src0_lanes) != wave_size:
        raise ValueError(f"Operand length must match wave_size ({wave_size})")

    active_mask = (1 << wave_size) - 1 if exec_mask is None else (exec_mask & ((1 << wave_size) - 1))
    fallback = [0] * wave_size if prev_dst is None else prev_dst
    if len(fallback) != wave_size:
        raise ValueError(f"prev_dst length must match wave_size ({wave_size})")

    output = list(fallback)
    for lane in range(wave_size):
        if (active_mask >> lane) & 1:
            output[lane] = ref_rcp_f64(
                src0_lanes[lane],
                clamp=clamp,
                src0_neg=src0_neg,
                src0_abs=src0_abs,
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

    def step_v_rcp_f64(
        self,
        dst_reg: int,
        src0_reg: int,
        *,
        clamp: bool = False,
        src0_neg: bool = False,
        src0_abs: bool = False,
    ) -> list[int]:
        active_mask = self.exec_mask & ((1 << self.wave_size) - 1)
        for lane in range(self.wave_size):
            if (active_mask >> lane) & 1:
                s0 = self.lanes[lane].get_vgpr_pair(src0_reg)
                res = ref_rcp_f64(
                    s0,
                    clamp=clamp,
                    src0_neg=src0_neg,
                    src0_abs=src0_abs,
                )
                self.lanes[lane].set_vgpr_pair(dst_reg, res)
        return self.load_vgpr_pair_wave(dst_reg)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_all_active(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        s0_vals = [f64_to_u64(float(i + 1)) for i in range(32)]
        sim.store_vgpr_pair_wave(2, s0_vals)

        res = sim.step_v_rcp_f64(0, 2)
        for i in range(32):
            self.assertEqual(u64_to_f64(res[i]), 1.0 / (i + 1))

    def test_wave64_all_active(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        s0_vals = [f64_to_u64(float(i + 1) * 2.0) for i in range(64)]
        sim.store_vgpr_pair_wave(2, s0_vals)

        res = sim.step_v_rcp_f64(0, 2)
        for i in range(64):
            self.assertEqual(u64_to_f64(res[i]), 1.0 / ((i + 1) * 2.0))

    def test_wave32_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 32)

        s0_vals = [f64_to_u64(4.0)] * 32
        sim.store_vgpr_pair_wave(2, s0_vals)

        # Alternating lanes active: 0b0101...
        sim.exec_mask = 0x55555555
        res = sim.step_v_rcp_f64(0, 2)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(u64_to_f64(res[lane]), 0.25)
            else:
                self.assertEqual(res[lane], sentinel_pair)

    def test_wave64_divergent_exec_mask(self) -> None:
        sim = SIMDWavefront(wave_size=64)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 64)

        s0_vals = [f64_to_u64(2.0)] * 64
        sim.store_vgpr_pair_wave(2, s0_vals)

        # Lower 32 lanes active, upper 32 inactive
        sim.exec_mask = 0x00000000_FFFFFFFF
        res = sim.step_v_rcp_f64(0, 2)

        for lane in range(32):
            self.assertEqual(u64_to_f64(res[lane]), 0.5)
        for lane in range(32, 64):
            self.assertEqual(res[lane], sentinel_pair)

    def test_wave32_upper_exec_bits_ignored(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel_pair = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel_pair] * 32)
        sim.store_vgpr_pair_wave(2, [f64_to_u64(2.0)] * 32)

        # In Wave32, upper 32 bits of exec_mask must have no effect
        sim.exec_mask = 0xDEADBEEF_FFFFFFFF
        res = sim.step_v_rcp_f64(0, 2)
        for lane in range(32):
            self.assertEqual(u64_to_f64(res[lane]), 0.5)

    def test_simd_oracle_validation(self) -> None:
        wave_size = 64
        s0_vals = [f64_to_u64(float(i + 1.25)) for i in range(wave_size)]
        exec_mask = 0xAAAAAAAAAAAAAAAA

        prev_dst = [join_dwords(0xCAFE, i) for i in range(wave_size)]
        expected = ref_simd_rcp_f64(s0_vals, wave_size=wave_size, exec_mask=exec_mask, prev_dst=prev_dst)

        sim = SIMDWavefront(wave_size=wave_size)
        sim.store_vgpr_pair_wave(0, prev_dst)
        sim.store_vgpr_pair_wave(2, s0_vals)
        sim.exec_mask = exec_mask

        actual = sim.step_v_rcp_f64(0, 2)
        self.assertEqual(actual, expected)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation for v_rcp_f64
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_dst_equals_src0(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(4, F64_FOUR)

        # v[4:5] = 1.0 / v[4:5] -> 0.25
        res = interp.execute_asm("v_rcp_f64 v[4:5], v[4:5]")
        self.assertEqual(u64_to_f64(res), 0.25)
        self.assertEqual(interp.vgpr.get_vgpr_pair(4), f64_to_u64(0.25))

    def test_inplace_self_inversion_cycle(self) -> None:
        interp = ScalarInterpreter()
        initial = f64_to_u64(16.0)
        interp.vgpr.set_vgpr_pair(2, initial)

        # 1st reciprocal: 16.0 -> 0.0625
        res1 = interp.execute_asm("v_rcp_f64 v[2:3], v[2:3]")
        self.assertEqual(u64_to_f64(res1), 0.0625)
        self.assertEqual(interp.vgpr.get_vgpr_pair(2), res1)

        # 2nd reciprocal: 0.0625 -> 16.0
        res2 = interp.execute_asm("v_rcp_f64 v[2:3], v[2:3]")
        self.assertEqual(u64_to_f64(res2), 16.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(2), initial)

    def test_partially_overlapping_register_pairs(self) -> None:
        interp = ScalarInterpreter()
        s0_val = f64_to_u64(4.0)
        interp.vgpr.set_vgpr_pair(0, s0_val)
        interp.vgpr.set_vgpr(2, 0x11223344)

        v0_before = interp.vgpr.get_vgpr(0)

        # Execute with dst as unaligned v[1:2] overlapping src0 v[0:1] (via v1)
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=1)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0)
        res = interp.execute_ast(inst)

        self.assertEqual(u64_to_f64(res), 0.25)
        self.assertEqual(interp.vgpr.get_vgpr_pair(1), f64_to_u64(0.25))
        # Non-overlapping boundary dword v[0] must remain completely unchanged
        self.assertEqual(interp.vgpr.get_vgpr(0), v0_before)

    def test_reverse_overlapping_register_pairs(self) -> None:
        interp = ScalarInterpreter()
        s0_val = f64_to_u64(8.0)
        interp.vgpr.set_vgpr_pair(1, s0_val)
        interp.vgpr.set_vgpr(3, 0x55667788)

        v3_before = interp.vgpr.get_vgpr(3)

        # Execute with dst v[0:1] overlapping src0 v[1:2] (via v1)
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=1)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0)
        res = interp.execute_ast(inst)

        self.assertEqual(u64_to_f64(res), 0.125)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), f64_to_u64(0.125))
        # Boundary dword v[3] remains completely intact
        self.assertEqual(interp.vgpr.get_vgpr(3), v3_before)

    def test_simd_inplace_divergent_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        init_vals = [f64_to_u64(float(i + 1) * 2.0) for i in range(32)]
        sim.store_vgpr_pair_wave(0, init_vals)

        # Active on odd lanes only: 0xAAAAAAAA
        sim.exec_mask = 0xAAAAAAAA
        res = sim.step_v_rcp_f64(0, 0)

        for lane in range(32):
            if lane % 2 == 1:
                # Active lane: inverted
                self.assertEqual(u64_to_f64(res[lane]), 1.0 / ((lane + 1) * 2.0))
            else:
                # Inactive lane: preserved original
                self.assertEqual(res[lane], init_vals[lane])

    def test_consecutive_chain_distinct_registers(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(0, f64_to_u64(2.0))

        # v[2:3] = 1 / v[0:1] = 0.5
        interp.execute_asm("v_rcp_f64 v[2:3], v[0:1]")
        # v[4:5] = 1 / v[2:3] = 2.0
        interp.execute_asm("v_rcp_f64 v[4:5], v[2:3]")
        # v[6:7] = 1 / v[4:5] = 0.5
        interp.execute_asm("v_rcp_f64 v[6:7], v[4:5]")

        self.assertEqual(u64_to_f64(interp.vgpr.get_vgpr_pair(0)), 2.0)
        self.assertEqual(u64_to_f64(interp.vgpr.get_vgpr_pair(2)), 0.5)
        self.assertEqual(u64_to_f64(interp.vgpr.get_vgpr_pair(4)), 2.0)
        self.assertEqual(u64_to_f64(interp.vgpr.get_vgpr_pair(6)), 0.5)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Isolation for v_rcp_f64
# ==============================================================================

@dataclass
class StatusFlags:
    invalid_op: bool = False
    div_by_zero: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    nan: bool = False


def compute_rcp_f64_status_flags(src0: int, result: int) -> StatusFlags:
    flags = StatusFlags()
    s0 = src0 & MASK64
    res = result & MASK64

    # Invalid operation: SNaN operand
    if is_f64_snan(s0):
        flags.invalid_op = True

    # Division by zero: 1.0 / +-0.0
    if is_f64_zero(s0):
        flags.div_by_zero = True

    if is_f64_nan(res):
        flags.nan = True
    else:
        flags.negative = bool(f64_sign(res))
        if is_f64_zero(res):
            flags.zero = True
            # Finite non-zero input producing zero is underflow
            if not is_f64_inf(s0):
                flags.underflow = True
                flags.inexact = True
        elif is_f64_inf(res):
            # Finite non-zero input producing inf is overflow
            if not is_f64_zero(s0):
                flags.overflow = True
                flags.inexact = True
        else:
            # Check inexact: test if val0 * res == 1.0 exactly
            v0 = u64_to_f64(s0)
            vr = u64_to_f64(res)
            if v0 * vr != 1.0:
                flags.inexact = True

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


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Test suite for condition codes, status flags, and carry/borrow isolation."""

    def test_status_flags_div_by_zero(self) -> None:
        # 1.0 / +0.0 = +Inf
        res = ref_rcp_f64(F64_POS_ZERO)
        flags = compute_rcp_f64_status_flags(F64_POS_ZERO, res)
        self.assertTrue(flags.div_by_zero)
        self.assertFalse(flags.invalid_op)
        self.assertFalse(flags.negative)

        # 1.0 / -0.0 = -Inf
        res_neg = ref_rcp_f64(F64_NEG_ZERO)
        flags_neg = compute_rcp_f64_status_flags(F64_NEG_ZERO, res_neg)
        self.assertTrue(flags_neg.div_by_zero)
        self.assertTrue(flags_neg.negative)

    def test_status_flags_invalid_operation(self) -> None:
        # SNaN input
        snan = (F64_POS_INF | 0x0001000000000000) & ~F64_QUIET_BIT
        res_snan = ref_rcp_f64(snan)
        flags_snan = compute_rcp_f64_status_flags(snan, res_snan)
        self.assertTrue(flags_snan.invalid_op)
        self.assertTrue(flags_snan.nan)

    def test_status_flags_overflow(self) -> None:
        # Subnormal input reciprocal overflows to infinity
        res = ref_rcp_f64(F64_MIN_SUBNORMAL)
        flags = compute_rcp_f64_status_flags(F64_MIN_SUBNORMAL, res)
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.inexact)
        self.assertEqual(res, F64_POS_INF)

    def test_status_flags_zero_and_negative(self) -> None:
        # 1.0 / -2.0 = -0.5 (negative, non-zero)
        v0 = f64_to_u64(-2.0)
        r1 = ref_rcp_f64(v0)
        flags1 = compute_rcp_f64_status_flags(v0, r1)
        self.assertTrue(flags1.negative)
        self.assertFalse(flags1.zero)

        # 1.0 / -Inf = -0.0 (negative zero)
        r2 = ref_rcp_f64(F64_NEG_INF)
        flags2 = compute_rcp_f64_status_flags(F64_NEG_INF, r2)
        self.assertTrue(flags2.negative)
        self.assertTrue(flags2.zero)

    def test_vcc_and_scc_preserved(self) -> None:
        # Floating point v_rcp_f64 does not generate carry/borrow and never modifies VCC or SCC
        interp = ScalarInterpreter()
        interp.vcc = 0x12345678_9ABCDEF0
        interp.scc = 1
        interp.vgpr.set_vgpr_pair(2, F64_FOUR)

        interp.execute_asm("v_rcp_f64 v[0:1], v[2:3]")
        self.assertEqual(interp.vcc, 0x12345678_9ABCDEF0)
        self.assertEqual(interp.scc, 1)

        sim = SIMDWavefront(wave_size=32)
        sim.vcc = 0xCAFEBABE
        sim.store_vgpr_pair_wave(2, [F64_TWO] * 32)
        sim.step_v_rcp_f64(0, 2)
        self.assertEqual(sim.vcc, 0xCAFEBABE)

    def test_condition_code_generation(self) -> None:
        v_zero = [F64_POS_ZERO] * 4
        # Four inputs: [-2.0, +Inf, 2.0, NaN]
        # Their reciprocals: [-0.5, +0.0, 0.5, NaN]
        s0 = [f64_to_u64(-2.0), F64_POS_INF, f64_to_u64(2.0), F64_CANONICAL_QNAN]
        rcps = [ref_rcp_f64(x) for x in s0]

        # Compare < 0: only lane 0 (-0.5 < 0) -> bit 0
        mask_lt = generate_condition_mask_f64("<", rcps, v_zero)
        self.assertEqual(mask_lt, 0b0001)

        # Compare == 0: only lane 1 (+0.0 == 0) -> bit 1
        mask_eq = generate_condition_mask_f64("==", rcps, v_zero)
        self.assertEqual(mask_eq, 0b0010)

        # Compare > 0: only lane 2 (0.5 > 0) -> bit 2
        mask_gt = generate_condition_mask_f64(">", rcps, v_zero)
        self.assertEqual(mask_gt, 0b0100)

        # Compare unordered: only lane 3 (NaN) -> bit 3
        mask_u = generate_condition_mask_f64("u", rcps, v_zero)
        self.assertEqual(mask_u, 0b1000)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling for v_rcp_f64
# ==============================================================================

DBL_MIN = 0x0010000000000000          # 2^-1022
DBL_MAX = 0x7FEFFFFFFFFFFFFF          # ~1.7976931348623157e+308
NEG_DBL_MIN = 0x8010000000000000
NEG_DBL_MAX = 0xFFEFFFFFFFFFFFFF

SUBNORM_MIN = 0x0000000000000001      # 2^-1074
SUBNORM_MAX = 0x000FFFFFFFFFFFFF
NEG_SUBNORM_MIN = 0x8000000000000001
NEG_SUBNORM_MAX = 0x800FFFFFFFFFFFFF

QNAN_PAYLOAD_A = 0x7FF8000000000001
QNAN_PAYLOAD_B = 0x7FF8AAAAAAAAAAAA
SNAN_F64 = 0x7FF0000000000001
NEG_SNAN_F64 = 0xFFF0000000000001


class TestIEEE754Boundaries(unittest.TestCase):
    """Exhaustive unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_signed_zero_boundary(self) -> None:
        self.assertEqual(ref_rcp_f64(F64_POS_ZERO), F64_POS_INF)
        self.assertEqual(ref_rcp_f64(F64_NEG_ZERO), F64_NEG_INF)

        # Input modifiers on zeros
        self.assertEqual(ref_rcp_f64(F64_NEG_ZERO, src0_abs=True), F64_POS_INF)
        self.assertEqual(ref_rcp_f64(F64_POS_ZERO, src0_neg=True), F64_NEG_INF)
        self.assertEqual(ref_rcp_f64(F64_NEG_ZERO, src0_neg=True), F64_POS_INF)

    def test_infinities_boundary(self) -> None:
        self.assertEqual(ref_rcp_f64(F64_POS_INF), F64_POS_ZERO)
        self.assertEqual(ref_rcp_f64(F64_NEG_INF), F64_NEG_ZERO)

        # Input modifiers on infinities
        self.assertEqual(ref_rcp_f64(F64_NEG_INF, src0_abs=True), F64_POS_ZERO)
        self.assertEqual(ref_rcp_f64(F64_POS_INF, src0_neg=True), F64_NEG_ZERO)
        self.assertEqual(ref_rcp_f64(F64_NEG_INF, src0_neg=True), F64_POS_ZERO)

    def test_subnormals_overflow(self) -> None:
        # Minimum subnormal (2^-1074) reciprocal overflows binary64 to signed infinity
        self.assertEqual(ref_rcp_f64(SUBNORM_MIN), F64_POS_INF)
        self.assertEqual(ref_rcp_f64(NEG_SUBNORM_MIN), F64_NEG_INF)

        # Maximum subnormal (2^-1022 * (1 - 2^-52)) reciprocal is a finite normal float (~4.49e307)
        res_sub_max = ref_rcp_f64(SUBNORM_MAX)
        self.assertTrue(is_f64_normal(res_sub_max))
        self.assertAlmostEqual(u64_to_f64(res_sub_max), 1.0 / u64_to_f64(SUBNORM_MAX))

    def test_normal_extrema(self) -> None:
        # 1.0 / DBL_MIN = 2^1022
        exp_val = 2.0 ** 1022
        self.assertEqual(ref_rcp_f64(DBL_MIN), f64_to_u64(exp_val))
        self.assertEqual(ref_rcp_f64(NEG_DBL_MIN), f64_to_u64(-exp_val))

        # 1.0 / DBL_MAX produces a tiny positive subnormal (~5.56e-309 < 2^-1022)
        res_max = ref_rcp_f64(DBL_MAX)
        self.assertEqual(f64_sign(res_max), 0)
        self.assertTrue(is_f64_subnormal(res_max))
        self.assertAlmostEqual(u64_to_f64(res_max), 1.0 / u64_to_f64(DBL_MAX))

        # 1.0 / NEG_DBL_MAX produces a tiny negative subnormal
        res_neg_max = ref_rcp_f64(NEG_DBL_MAX)
        self.assertEqual(f64_sign(res_neg_max), 1)
        self.assertTrue(is_f64_subnormal(res_neg_max))
        self.assertAlmostEqual(u64_to_f64(res_neg_max), 1.0 / u64_to_f64(NEG_DBL_MAX))

    def test_nan_handling_and_quieting(self) -> None:
        nans = [F64_CANONICAL_QNAN, QNAN_PAYLOAD_A, QNAN_PAYLOAD_B, SNAN_F64, NEG_SNAN_F64]
        for n in nans:
            with self.subTest(nan=f"{n:#018x}"):
                res = ref_rcp_f64(n)
                self.assertTrue(is_f64_nan(res))
                self.assertTrue(is_f64_qnan(res))

    def test_boundary_extremes_with_clamp(self) -> None:
        # +0.0 -> +Inf clamped to 1.0
        self.assertEqual(ref_rcp_f64(F64_POS_ZERO, clamp=True), F64_ONE)
        # -0.0 -> -Inf clamped to +0.0
        self.assertEqual(ref_rcp_f64(F64_NEG_ZERO, clamp=True), F64_POS_ZERO)
        # +Inf -> +0.0 clamped to +0.0
        self.assertEqual(ref_rcp_f64(F64_POS_INF, clamp=True), F64_POS_ZERO)
        # -Inf -> -0.0 clamped to +0.0
        self.assertEqual(ref_rcp_f64(F64_NEG_INF, clamp=True), F64_POS_ZERO)
        # Subnormals clamped
        self.assertEqual(ref_rcp_f64(SUBNORM_MIN, clamp=True), F64_ONE)
        self.assertEqual(ref_rcp_f64(NEG_SUBNORM_MIN, clamp=True), F64_POS_ZERO)
        # NaNs clamped to +0.0
        self.assertEqual(ref_rcp_f64(F64_CANONICAL_QNAN, clamp=True), F64_POS_ZERO)
        self.assertEqual(ref_rcp_f64(SNAN_F64, clamp=True), F64_POS_ZERO)


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection for v_rcp_f64
# ==============================================================================

class TestNegativeControlsAndMutations(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_empty_and_whitespace(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("")
        with self.assertRaises(ValueError):
            parse_asm_line("   ; comment only")

    def test_syntax_rejection_wrong_opcode(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("v_rcp_f32 v[0:1], v[2:3]")
        with self.assertRaises(ValueError):
            parse_asm_line("v_mul_f64 v[0:1], v[2:3]")

    def test_syntax_rejection_wrong_operand_count(self) -> None:
        # Zero source operands
        with self.assertRaises(ValueError):
            parse_asm_line("v_rcp_f64 v[0:1]")
        # Two source operands (v_rcp_f64 takes only one source operand!)
        with self.assertRaises(ValueError):
            parse_asm_line("v_rcp_f64 v[0:1], v[2:3], v[4:5]")

    def test_syntax_rejection_invalid_operand_tokens(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_operand("invalid_token")
        with self.assertRaises(ValueError):
            parse_asm_operand("m[0:1]")

    def test_syntax_rejection_out_of_bounds_registers(self) -> None:
        # Destination VGPR > 254
        inst_bad_dst = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, reg_idx=256),
            src0=ASTOperand(OperandKind.VGPR_PAIR, reg_idx=0),
        )
        with self.assertRaises(ValueError):
            inst_bad_dst.validate()

        # Source SGPR > 104
        inst_bad_src = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, reg_idx=0),
            src0=ASTOperand(OperandKind.SGPR_PAIR, reg_idx=106),
        )
        with self.assertRaises(ValueError):
            inst_bad_src.validate()

    def test_syntax_rejection_illegal_dst_modifier(self) -> None:
        inst_neg_dst = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, reg_idx=0, neg=True),
            src0=ASTOperand(OperandKind.VGPR_PAIR, reg_idx=2),
        )
        with self.assertRaises(ValueError):
            inst_neg_dst.validate()

        inst_abs_dst = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, reg_idx=0, abs_val=True),
            src0=ASTOperand(OperandKind.VGPR_PAIR, reg_idx=2),
        )
        with self.assertRaises(ValueError):
            inst_abs_dst.validate()

    def test_negative_simd_wave_sizes_and_lengths(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=128)

        # ref_simd_rcp_f64 with invalid wave size
        with self.assertRaises(ValueError):
            ref_simd_rcp_f64([0] * 16, wave_size=16)

        # ref_simd_rcp_f64 with length mismatch
        with self.assertRaises(ValueError):
            ref_simd_rcp_f64([0] * 32, wave_size=64)

    def test_negative_register_file_bounds(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        with self.assertRaises(IndexError):
            rf.get_vgpr(-1)
        with self.assertRaises(IndexError):
            rf.get_vgpr(16)
        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)
        with self.assertRaises(IndexError):
            rf.set_vgpr_pair(15, 0)

    def test_mutation_zero_reciprocal_sign_detection(self) -> None:
        # A buggy oracle that fails to propagate sign on 1.0 / -0.0
        def mutated_oracle(src: int) -> int:
            if is_f64_zero(src):
                return F64_POS_INF  # Bug: always positive infinity
            return ref_rcp_f64(src)

        correct = ref_rcp_f64(F64_NEG_ZERO)
        buggy = mutated_oracle(F64_NEG_ZERO)
        self.assertNotEqual(buggy, correct)
        self.assertEqual(correct, F64_NEG_INF)

    def test_mutation_clamp_nan_to_zero_detection(self) -> None:
        # A buggy clamp that leaves NaN intact instead of clamping to +0.0
        def mutated_clamp(src: int) -> int:
            res = ref_rcp_f64(src)
            return res  # Bug: misses clamp on NaN

        correct = ref_rcp_f64(F64_CANONICAL_QNAN, clamp=True)
        buggy = mutated_clamp(F64_CANONICAL_QNAN)
        self.assertNotEqual(buggy, correct)
        self.assertEqual(correct, F64_POS_ZERO)

    def test_mutation_clamp_negative_to_zero_detection(self) -> None:
        # A buggy clamp that does not clamp negative values to 0.0
        def mutated_clamp(src: int) -> int:
            res = ref_rcp_f64(src)
            return res  # Bug: misses lower clamp bound 0.0

        val = f64_to_u64(-2.0)
        correct = ref_rcp_f64(val, clamp=True)
        buggy = mutated_clamp(val)
        self.assertNotEqual(buggy, correct)
        self.assertEqual(correct, F64_POS_ZERO)

    def test_mutation_inactive_lane_tampering_detection(self) -> None:
        # Verify SIMD step strictly protects inactive lanes
        sim = SIMDWavefront(wave_size=32)
        sentinel = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel] * 32)
        sim.store_vgpr_pair_wave(2, [f64_to_u64(2.0)] * 32)
        sim.exec_mask = 0x00000001  # Only lane 0 active

        res = sim.step_v_rcp_f64(0, 2)
        self.assertEqual(u64_to_f64(res[0]), 0.5)
        for lane in range(1, 32):
            self.assertEqual(res[lane], sentinel)

    def test_mutation_abs_neg_modifier_order(self) -> None:
        # RDNA2 rule: neg(abs(x)) forces sign to 1.
        # If an implementation mistakenly computed abs(neg(x)), sign would be 0 (+0.5).
        # We verify that neg=True, abs_val=True forces negative result (-0.5).
        res = ref_rcp_f64(f64_to_u64(-2.0), src0_neg=True, src0_abs=True)
        self.assertEqual(u64_to_f64(res), -0.5)
        self.assertEqual(f64_sign(res), 1)


class TestScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def test_execute_vgpr_source(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, F64_FOUR)

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), 0.25)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_sgpr_source(self) -> None:
        interp = ScalarInterpreter()
        interp.set_sgpr_pair(4, f64_to_u64(-2.0))

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=4)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), -0.5)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_literal_source(self) -> None:
        interp = ScalarInterpreter()

        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.LITERAL_64, imm_val=f64_to_u64(0.125))
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0)

        res = interp.execute_ast(inst)
        self.assertEqual(u64_to_f64(res), 8.0)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), res)

    def test_execute_text_asm(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(6, f64_to_u64(8.0))

        res = interp.execute_asm("v_rcp_f64 v[0:1], v[6:7] ; take reciprocal")
        self.assertEqual(u64_to_f64(res), 0.125)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), f64_to_u64(0.125))

    def test_execute_modifiers_and_clamp(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, f64_to_u64(-2.0))

        # -|v[2:3]| is -2.0, reciprocal is -0.5, clamp produces +0.0
        res = interp.execute_asm("v_rcp_f64 v[0:1], -|v[2:3]| clamp")
        self.assertEqual(res, F64_POS_ZERO)

        # Reciprocal of 2.0 is 0.5, within clamp [0, 1]
        interp.vgpr.set_vgpr_pair(2, F64_TWO)
        res2 = interp.execute_asm("v_rcp_f64 v[0:1], v[2:3] clamp")
        self.assertEqual(u64_to_f64(res2), 0.5)


class TestSymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST definitions and typed expressions."""

    def test_ast_creation_and_formatting(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0)
        inst.validate()
        self.assertEqual(inst.format_asm(), "v_rcp_f64 v[0:1], v[2:3]")

        # With clamp and modifiers
        src0_mod = ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=4, neg=True, abs_val=True)
        inst_clamp = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0_mod, clamp=True)
        inst_clamp.validate()
        self.assertEqual(inst_clamp.format_asm(), "v_rcp_f64 v[0:1], -|s[4:5]| clamp")

    def test_typed_expression_evaluation(self) -> None:
        c1 = F64Constant(F64_TWO)
        rcp_expr = F64Rcp(operand=c1)
        res = rcp_expr.evaluate({})
        self.assertEqual(u64_to_f64(res), 0.5)

        # Unary modifier
        neg_c1 = F64UnaryOp("neg", c1)
        res_neg = F64Rcp(operand=neg_c1).evaluate({})
        self.assertEqual(u64_to_f64(res_neg), -0.5)

    def test_instruction_to_expression(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2, neg=True)
        inst = ASTInstruction(opcode="v_rcp_f64", dst=dst, src0=src0, clamp=False)

        expr = instruction_to_expression(inst)
        env = {"src0_2": F64_FOUR}
        res = expr.evaluate(env)
        # 1.0 / (-4.0) = -0.25
        self.assertEqual(u64_to_f64(res), -0.25)

    def test_ast_validation_errors(self) -> None:
        # Invalid opcode
        bad_op = ASTInstruction(
            opcode="v_mul_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
        )
        with self.assertRaises(ValueError):
            bad_op.validate()

        # Destination cannot be SGPR
        bad_dst = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.SGPR_PAIR, 0),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
        )
        with self.assertRaises(ValueError):
            bad_dst.validate()

        # Destination out of bounds
        oob_dst = ASTInstruction(
            opcode="v_rcp_f64",
            dst=ASTOperand(OperandKind.VGPR_PAIR, 256),
            src0=ASTOperand(OperandKind.VGPR_PAIR, 2),
        )
        with self.assertRaises(ValueError):
            oob_dst.validate()


class TestVRcpF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_basic_reciprocal(self) -> None:
        # 1.0 / 1.0 = 1.0
        r = ref_rcp_f64(F64_ONE)
        self.assertEqual(u64_to_f64(r), 1.0)

        # 1.0 / 2.0 = 0.5
        r = ref_rcp_f64(F64_TWO)
        self.assertEqual(u64_to_f64(r), 0.5)

        # 1.0 / 0.5 = 2.0
        r = ref_rcp_f64(F64_HALF)
        self.assertEqual(u64_to_f64(r), 2.0)

        # 1.0 / 4.0 = 0.25
        r = ref_rcp_f64(F64_FOUR)
        self.assertEqual(u64_to_f64(r), 0.25)

        # 1.0 / -4.0 = -0.25
        r = ref_rcp_f64(f64_to_u64(-4.0))
        self.assertEqual(u64_to_f64(r), -0.25)

    def test_zero_reciprocal_rules(self) -> None:
        # 1.0 / +0.0 = +Inf
        self.assertEqual(ref_rcp_f64(F64_POS_ZERO), F64_POS_INF)
        # 1.0 / -0.0 = -Inf
        self.assertEqual(ref_rcp_f64(F64_NEG_ZERO), F64_NEG_INF)

    def test_inf_reciprocal_rules(self) -> None:
        # 1.0 / +Inf = +0.0
        self.assertEqual(ref_rcp_f64(F64_POS_INF), F64_POS_ZERO)
        # 1.0 / -Inf = -0.0
        self.assertEqual(ref_rcp_f64(F64_NEG_INF), F64_NEG_ZERO)

    def test_nan_reciprocal_rules(self) -> None:
        # QNaN propagated with quiet bit
        qnan = F64_CANONICAL_QNAN
        self.assertTrue(is_f64_nan(ref_rcp_f64(qnan)))
        self.assertTrue(is_f64_qnan(ref_rcp_f64(qnan)))

        # SNaN quieted
        snan = (F64_POS_INF | 0x0001000000000000) & ~F64_QUIET_BIT
        r_snan = ref_rcp_f64(snan)
        self.assertTrue(is_f64_nan(r_snan))
        self.assertTrue(is_f64_qnan(r_snan))

    def test_input_modifiers(self) -> None:
        # neg: 1.0 / neg(2.0) = -0.5
        r = ref_rcp_f64(F64_TWO, src0_neg=True)
        self.assertEqual(u64_to_f64(r), -0.5)

        # abs: 1.0 / abs(-2.0) = 0.5
        r = ref_rcp_f64(f64_to_u64(-2.0), src0_abs=True)
        self.assertEqual(u64_to_f64(r), 0.5)

        # neg + abs: neg(abs(-2.0)) = -2.0 -> 1.0 / -2.0 = -0.5
        r = ref_rcp_f64(f64_to_u64(-2.0), src0_neg=True, src0_abs=True)
        self.assertEqual(u64_to_f64(r), -0.5)

    def test_clamp_modifier(self) -> None:
        # In range [0.0, 1.0]: 1.0 / 2.0 = 0.5 -> 0.5
        r = ref_rcp_f64(F64_TWO, clamp=True)
        self.assertEqual(u64_to_f64(r), 0.5)

        # > 1.0 clamped to 1.0: 1.0 / 0.5 = 2.0 -> 1.0
        r = ref_rcp_f64(F64_HALF, clamp=True)
        self.assertEqual(r, F64_ONE)

        # +Inf clamped to 1.0: 1.0 / +0.0 = +Inf -> 1.0
        r = ref_rcp_f64(F64_POS_ZERO, clamp=True)
        self.assertEqual(r, F64_ONE)

        # Negative clamped to +0.0: 1.0 / -2.0 = -0.5 -> +0.0
        r = ref_rcp_f64(f64_to_u64(-2.0), clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

        # NaN clamped to +0.0
        r = ref_rcp_f64(F64_CANONICAL_QNAN, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(12.5)
        lo, hi = split_dwords(s0)
        r_lo, r_hi = ref_rcp_f64_dwords(lo, hi)
        res = join_dwords(r_lo, r_hi)
        self.assertEqual(u64_to_f64(res), 0.08)

    def test_register_file_pair_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        val = 0x01234567_89ABCDEF
        rf.set_vgpr_pair(4, val)

        self.assertEqual(rf.get_vgpr(4), 0x89ABCDEF)
        self.assertEqual(rf.get_vgpr(5), 0x01234567)
        self.assertEqual(rf.get_vgpr_pair(4), val)

        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)


if __name__ == "__main__":
    unittest.main()
