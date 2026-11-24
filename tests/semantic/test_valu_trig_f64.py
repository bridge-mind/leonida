"""Semantic reference tests and mathematical oracle for AMD RDNA2 v_trig_f64.

RDNA2 Vector Arithmetic Logic Unit (VALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, vector registers (VGPRs) are 32 bits wide.
  Operations on 64-bit double-precision floating-point quantities (binary64, IEEE 754)
  operate across aligned pairs of adjacent registers: v[d:d+1] from source operand
  v[s0:s0+1] (or scalar SGPR pair s[k:k+1], literal 64-bit constants, or inline constants).

  Instruction:
    v_trig_f64 v[d:d+1], src0 [clamp]
    (with trigonometric function / argument evaluation in units of 2*pi revolutions / turns)

  Mathematical Specification:
    1. Angle Domain:
       Input src0 is specified in cycles / revolutions (turns), where 1.0 turn = 2*pi radians.
       The hardware reduces the input argument modulo 1.0 to the fundamental domain [-0.5, 0.5]
       using IEEE roundTiesToEven:
         reduced_turns = turns - round_even(turns)
       This exact range reduction ensures zero transcendental error drift across multi-turn rotations.
    2. Primary Evaluation:
       The primary trigonometric function evaluates:
         radians = reduced_turns * 2*pi
         D.f64 = sin(radians)    (or cos(radians) in dual / cos-mode)
    3. IEEE 754 & RDNA2 Rules:
       - Signed Zero:
           sin(+0.0) = +0.0,  sin(-0.0) = -0.0
           cos(+0.0) = 1.0,   cos(-0.0) = 1.0
       - Infinities:
           sin(+-inf) = Canonical QNaN
           cos(+-inf) = Canonical QNaN
       - NaNs:
           NaN operands are quieted and propagated as Canonical Quiet NaNs.
       - Subnormals:
           Denormal / subnormal inputs are preserved without flushing in IEEE compliance mode.
       - Cardinal Angles (in turns):
           turns = 0.00  -> sin =  0.0, cos =  1.0
           turns = 0.25  -> sin =  1.0, cos =  0.0  (pi/2 rad)
           turns = 0.50  -> sin =  0.0, cos = -1.0  (pi rad)
           turns = 0.75  -> sin = -1.0, cos =  0.0  (3*pi/2 rad)
       - VOP3 Input Modifiers:
           abs (|x|) clears sign bit 63.
           neg (-x) toggles sign bit 63.
           neg(abs(x)) forces the sign bit to 1 (negative).
       - Output Clamp:
           clamp forces result into [0.0, 1.0]. Negative results or -0.0 clamp to +0.0.
           In RDNA2 VALU, clamping NaN yields +0.0.

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
import enum
import math
import struct
import unittest

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
TWO_PI: float = 6.283185307179586476925286766559

# IEEE-754 binary64 bit constants
F64_POS_ZERO: int = 0x0000000000000000
F64_NEG_ZERO: int = 0x8000000000000000
F64_POS_INF: int = 0x7FF0000000000000
F64_NEG_INF: int = 0xFFF0000000000000
F64_CANONICAL_QNAN: int = 0x7FF8000000000000
F64_QUIET_BIT: int = 0x0008000000000000

F64_EXP_MASK: int = 0x7FF0000000000000
F64_MANT_MASK: int = 0x000FFFFFFFFFFFFF
F64_SIGN_MASK: int = 0x8000000000000000

F64_ONE: int = 0x3FF0000000000000          # 1.0
F64_NEG_ONE: int = 0xBFF0000000000000      # -1.0
F64_TWO: int = 0x4000000000000000          # 2.0
F64_HALF: int = 0x3FE0000000000000         # 0.5
F64_QUARTER: int = 0x3FD0000000000000      # 0.25 (1/4 turn = 90 deg = pi/2 rad)
F64_NEG_QUARTER: int = 0xBFD0000000000000  # -0.25 (-1/4 turn = -90 deg = -pi/2 rad)
F64_THREE_QUARTERS: int = 0x3FE8000000000000  # 0.75 (3/4 turn = 270 deg = 3*pi/2 rad)

F64_MIN_NORMAL: int = 0x0010000000000000   # 2^-1022
F64_MAX_NORMAL: int = 0x7FEFFFFFFFFFFFFF   # (2 - 2^-52) * 2^1023
F64_MIN_SUBNORMAL: int = 0x0000000000000001  # 2^-1074
F64_MAX_SUBNORMAL: int = 0x000FFFFFFFFFFFFF

OLD_VGPR_SENTINEL: int = 0xDEADBEEF


class TrigMode(enum.Enum):
    SIN = 0
    COS = 1


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


def round_even(x: float) -> float:
    """Round float to nearest integer, breaking ties to even."""
    return float(round(x))


def ref_turn_reduce_f64(turns: float) -> float:
    """Reduce turn count to [-0.5, 0.5] turns using roundTiesToEven."""
    if math.isnan(turns) or math.isinf(turns):
        return float("nan")
    if turns == 0.0:
        return turns
    return turns - round_even(turns)


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
    """Apply VOP3 clamp modifier [0.0, 1.0]. NaN or negatives clamp to +0.0."""
    if is_f64_nan(bits):
        return F64_POS_ZERO
    val = u64_to_f64(bits)
    if val < 0.0 or is_f64_zero(bits):
        return F64_POS_ZERO
    elif val > 1.0:
        return F64_ONE
    return bits & MASK64


def ref_trig_f64(
    src0_bits: int,
    *,
    mode: TrigMode = TrigMode.SIN,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 double-precision v_trig_f64.

    Evaluates:
      D.f64 = trig(src0 * 2*pi)
    in units of revolutions / turns with exact [-0.5, 0.5] range reduction.
    """
    s0 = apply_input_modifiers_f64(src0_bits, neg=src0_neg, abs_val=src0_abs)

    # NaN input produces canonical quiet NaN
    if is_f64_nan(s0):
        res = quiet_f64(s0)
    # Infinities produce canonical quiet NaN
    elif is_f64_inf(s0):
        res = F64_CANONICAL_QNAN
    # Zero input rules
    elif is_f64_zero(s0):
        if mode == TrigMode.SIN:
            res = F64_NEG_ZERO if f64_sign(s0) else F64_POS_ZERO
        else:
            res = F64_ONE
    else:
        turns = u64_to_f64(s0)
        reduced = ref_turn_reduce_f64(turns)

        if mode == TrigMode.SIN:
            if reduced == 0.0:
                res = F64_NEG_ZERO if f64_sign(s0) else F64_POS_ZERO
            elif reduced == 0.25:
                res = F64_ONE
            elif reduced == -0.25:
                res = F64_NEG_ONE
            elif abs(reduced) == 0.5:
                res = F64_POS_ZERO
            else:
                rad = reduced * TWO_PI
                res = f64_to_u64(math.sin(rad))
                if is_f64_zero(res):
                    res = F64_NEG_ZERO if reduced < 0 else F64_POS_ZERO
        else:  # COS
            if reduced == 0.0:
                res = F64_ONE
            elif abs(reduced) == 0.25:
                res = F64_POS_ZERO
            elif abs(reduced) == 0.5:
                res = F64_NEG_ONE
            else:
                rad = reduced * TWO_PI
                res = f64_to_u64(math.cos(rad))

    if clamp:
        res = apply_f64_clamp(res)

    return res & MASK64


def ref_trig_f64_dwords(
    s0_lo: int,
    s0_hi: int,
    *,
    mode: TrigMode = TrigMode.SIN,
    clamp: bool = False,
    src0_neg: bool = False,
    src0_abs: bool = False,
) -> tuple[int, int]:
    """Decomposed dword interface for v_trig_f64."""
    s0 = join_dwords(s0_lo, s0_hi)
    res = ref_trig_f64(
        s0,
        mode=mode,
        clamp=clamp,
        src0_neg=src0_neg,
        src0_abs=src0_abs,
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


class TestVTrigF64ReferenceOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural model."""

    def test_cardinal_angles_sin(self) -> None:
        # 0.0 turn -> sin(0) = 0.0
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.SIN), F64_POS_ZERO)
        # 0.25 turn -> sin(pi/2) = 1.0
        self.assertEqual(ref_trig_f64(F64_QUARTER, mode=TrigMode.SIN), F64_ONE)
        # 0.50 turn -> sin(pi) = 0.0
        self.assertEqual(ref_trig_f64(F64_HALF, mode=TrigMode.SIN), F64_POS_ZERO)
        # 0.75 turn -> sin(3*pi/2) = -1.0
        self.assertEqual(ref_trig_f64(F64_THREE_QUARTERS, mode=TrigMode.SIN), F64_NEG_ONE)
        # 1.00 turn -> sin(2*pi) = 0.0
        self.assertEqual(ref_trig_f64(F64_ONE, mode=TrigMode.SIN), F64_POS_ZERO)

    def test_cardinal_angles_cos(self) -> None:
        # 0.0 turn -> cos(0) = 1.0
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.COS), F64_ONE)
        # 0.25 turn -> cos(pi/2) = 0.0
        self.assertEqual(ref_trig_f64(F64_QUARTER, mode=TrigMode.COS), F64_POS_ZERO)
        # 0.50 turn -> cos(pi) = -1.0
        self.assertEqual(ref_trig_f64(F64_HALF, mode=TrigMode.COS), F64_NEG_ONE)
        # 0.75 turn -> cos(3*pi/2) = 0.0
        self.assertEqual(ref_trig_f64(F64_THREE_QUARTERS, mode=TrigMode.COS), F64_POS_ZERO)
        # 1.00 turn -> cos(2*pi) = 1.0
        self.assertEqual(ref_trig_f64(F64_ONE, mode=TrigMode.COS), F64_ONE)

    def test_signed_zeros(self) -> None:
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.SIN), F64_POS_ZERO)
        self.assertEqual(ref_trig_f64(F64_NEG_ZERO, mode=TrigMode.SIN), F64_NEG_ZERO)
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.COS), F64_ONE)
        self.assertEqual(ref_trig_f64(F64_NEG_ZERO, mode=TrigMode.COS), F64_ONE)

    def test_infinities_produce_qnan(self) -> None:
        self.assertEqual(ref_trig_f64(F64_POS_INF, mode=TrigMode.SIN), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_NEG_INF, mode=TrigMode.SIN), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_POS_INF, mode=TrigMode.COS), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_NEG_INF, mode=TrigMode.COS), F64_CANONICAL_QNAN)

    def test_nan_propagation(self) -> None:
        snan = 0x7FF0000000000001
        res_sin = ref_trig_f64(snan, mode=TrigMode.SIN)
        self.assertTrue(is_f64_qnan(res_sin))
        res_cos = ref_trig_f64(snan, mode=TrigMode.COS)
        self.assertTrue(is_f64_qnan(res_cos))

    def test_input_modifiers(self) -> None:
        # neg: sin(-0.25 turn) = -1.0; with neg modifier: sin(-(-0.25)) = sin(0.25) = 1.0
        r = ref_trig_f64(F64_NEG_QUARTER, mode=TrigMode.SIN, src0_neg=True)
        self.assertEqual(r, F64_ONE)

        # abs: sin(|-0.25|) = sin(0.25) = 1.0
        r = ref_trig_f64(F64_NEG_QUARTER, mode=TrigMode.SIN, src0_abs=True)
        self.assertEqual(r, F64_ONE)

        # abs + neg: neg(abs(0.25)) = -0.25 -> sin(-0.25) = -1.0
        r = ref_trig_f64(F64_QUARTER, mode=TrigMode.SIN, src0_abs=True, src0_neg=True)
        self.assertEqual(r, F64_NEG_ONE)

    def test_clamp_modifier(self) -> None:
        # In range [0.0, 1.0]: sin(0.125 turn) = sin(pi/4) ~= 0.7071
        r = ref_trig_f64(f64_to_u64(0.125), mode=TrigMode.SIN, clamp=True)
        self.assertAlmostEqual(u64_to_f64(r), math.sin(math.pi / 4), places=7)

        # Negative clamped to +0.0: sin(-0.25) = -1.0 -> clamped to +0.0
        r = ref_trig_f64(F64_NEG_QUARTER, mode=TrigMode.SIN, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

        # NaN clamped to +0.0
        r = ref_trig_f64(F64_POS_INF, mode=TrigMode.SIN, clamp=True)
        self.assertEqual(r, F64_POS_ZERO)

    def test_turn_periodicity(self) -> None:
        # Any value x and x + 100.0 turns should yield the exact same result
        base_turns = 0.1738294
        for k in (1.0, 10.0, 1000.0, -50.0):
            t1 = f64_to_u64(base_turns)
            t2 = f64_to_u64(base_turns + k)
            self.assertAlmostEqual(
                u64_to_f64(ref_trig_f64(t1, mode=TrigMode.SIN)),
                u64_to_f64(ref_trig_f64(t2, mode=TrigMode.SIN)),
                places=9,
            )

    def test_pythagorean_identity(self) -> None:
        # sin^2(x) + cos^2(x) == 1.0 for arbitrary turn values
        for angle in (0.05, 0.1337, 0.33333333, -0.42, 12.3456):
            bits = f64_to_u64(angle)
            s = u64_to_f64(ref_trig_f64(bits, mode=TrigMode.SIN))
            c = u64_to_f64(ref_trig_f64(bits, mode=TrigMode.COS))
            self.assertAlmostEqual(s * s + c * c, 1.0, places=12)

    def test_dword_decomposition(self) -> None:
        s0 = f64_to_u64(0.25)
        s0_lo, s0_hi = split_dwords(s0)
        lo, hi = ref_trig_f64_dwords(s0_lo, s0_hi, mode=TrigMode.SIN)
        res64 = join_dwords(lo, hi)
        self.assertEqual(res64, F64_ONE)

    def test_register_file_pair_access(self) -> None:
        rf = VALURegisterFile(num_vgprs=16)
        val = 0x01234567_89ABCDEF
        rf.set_vgpr_pair(4, val)

        self.assertEqual(rf.get_vgpr(4), 0x89ABCDEF)
        self.assertEqual(rf.get_vgpr(5), 0x01234567)
        self.assertEqual(rf.get_vgpr_pair(4), val)

        with self.assertRaises(IndexError):
            rf.get_vgpr_pair(15)


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions for v_trig_f64
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
    mode: TrigMode = TrigMode.SIN
    clamp: bool = False
    omod: int = 0

    def validate(self) -> None:
        if self.opcode != "v_trig_f64":
            raise ValueError(f"Invalid opcode for v_trig_f64 AST: {self.opcode}")
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
class F64Trig(F64Expression):
    operand: F64Expression
    mode: TrigMode = TrigMode.SIN
    clamp: bool = False

    def evaluate(self, env: dict[str, int]) -> int:
        val = self.operand.evaluate(env)
        return ref_trig_f64(val, mode=self.mode, clamp=self.clamp)


def instruction_to_expression(inst: ASTInstruction) -> F64Trig:
    """Translate an ASTInstruction to a typed symbolic expression tree."""
    inst.validate()

    op = inst.src0
    if op.kind in (OperandKind.VGPR_PAIR, OperandKind.SGPR_PAIR):
        expr: F64Expression = F64Variable(f"src0_{op.reg_idx}")
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

    return F64Trig(expr, mode=inst.mode, clamp=inst.clamp)


class TestSymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST definitions and typed expressions."""

    def test_ast_creation_and_formatting(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2)
        inst = ASTInstruction(opcode="v_trig_f64", dst=dst, src0=src0)
        inst.validate()
        self.assertEqual(inst.format_asm(), "v_trig_f64 v[0:1], v[2:3]")

        # With clamp and modifiers
        src0_mod = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2, neg=True, abs_val=True)
        inst_clamp = ASTInstruction(opcode="v_trig_f64", dst=dst, src0=src0_mod, clamp=True)
        inst_clamp.validate()
        self.assertEqual(inst_clamp.format_asm(), "v_trig_f64 v[0:1], -|v[2:3]| clamp")

    def test_typed_expression_evaluation(self) -> None:
        c1 = F64Constant(F64_QUARTER)
        trig_expr = F64Trig(operand=c1, mode=TrigMode.SIN)
        res = trig_expr.evaluate({})
        self.assertEqual(res, F64_ONE)

        # COS mode
        trig_cos = F64Trig(operand=c1, mode=TrigMode.COS)
        res_cos = trig_cos.evaluate({})
        self.assertEqual(res_cos, F64_POS_ZERO)

        # Unary operations: neg(0.25) -> sin(-0.25) = -1.0
        neg_c1 = F64UnaryOp("neg", c1)
        res_neg = F64Trig(operand=neg_c1, mode=TrigMode.SIN).evaluate({})
        self.assertEqual(res_neg, F64_NEG_ONE)

    def test_instruction_to_expression(self) -> None:
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2, neg=True)
        inst = ASTInstruction(opcode="v_trig_f64", dst=dst, src0=src0, mode=TrigMode.SIN)
        expr = instruction_to_expression(inst)

        env = {"src0_2": F64_QUARTER}
        res = expr.evaluate(env)
        self.assertEqual(res, F64_NEG_ONE)

    def test_ast_validation_errors(self) -> None:
        # Bad opcode
        bad_inst = ASTInstruction(
            opcode="v_sin_f32",
            dst=ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0),
            src0=ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2),
        )
        with self.assertRaises(ValueError):
            bad_inst.validate()

        # Destination must be VGPR pair
        bad_dst = ASTInstruction(
            opcode="v_trig_f64",
            dst=ASTOperand(kind=OperandKind.SGPR_PAIR, reg_idx=0),
            src0=ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2),
        )
        with self.assertRaises(ValueError):
            bad_dst.validate()

        # Destination cannot have neg modifier
        bad_mod = ASTInstruction(
            opcode="v_trig_f64",
            dst=ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0, neg=True),
            src0=ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=2),
        )
        with self.assertRaises(ValueError):
            bad_mod.validate()


# ==============================================================================
# Deterministic Scalar Instruction Interpreter for v_trig_f64
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
    if opcode != "v_trig_f64":
        raise ValueError(f"Expected opcode v_trig_f64, got {opcode}")

    clamp = False
    mode = TrigMode.SIN
    raw_ops = []
    for tok in tokens[1:]:
        ltok = tok.lower()
        if ltok == "clamp":
            clamp = True
        elif ltok == "cos":
            mode = TrigMode.COS
        elif ltok == "sin":
            mode = TrigMode.SIN
        else:
            raw_ops.append(tok)

    if len(raw_ops) != 2:
        raise ValueError(f"v_trig_f64 requires 2 operands (dst, src0), got {len(raw_ops)}: {raw_ops}")

    dst = parse_asm_operand(raw_ops[0])
    src0 = parse_asm_operand(raw_ops[1])
    inst = ASTInstruction(opcode=opcode, dst=dst, src0=src0, mode=mode, clamp=clamp)
    inst.validate()
    return inst


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU v_trig_f64."""

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
        res = ref_trig_f64(s0, mode=inst.mode, clamp=inst.clamp)
        self.vgpr.set_vgpr_pair(inst.dst.reg_idx, res)
        return res

    def execute_asm(self, line: str) -> int:
        inst = parse_asm_line(line)
        return self.execute_ast(inst)


class TestScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar interpreter execution."""

    def test_execute_vgpr_source(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, F64_QUARTER)
        res = interp.execute_asm("v_trig_f64 v[0:1], v[2:3]")
        self.assertEqual(res, F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_ONE)

    def test_execute_sgpr_source(self) -> None:
        interp = ScalarInterpreter()
        interp.set_sgpr_pair(4, F64_QUARTER)
        res = interp.execute_asm("v_trig_f64 v[6:7], s[4:5]")
        self.assertEqual(res, F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(6), F64_ONE)

    def test_execute_literal_source(self) -> None:
        interp = ScalarInterpreter()
        res = interp.execute_asm(f"v_trig_f64 v[0:1], 0x{F64_QUARTER:016x}")
        self.assertEqual(res, F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_ONE)

    def test_execute_with_modifiers_and_clamp(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, F64_QUARTER)
        # neg modifier makes turns -0.25 -> sin(-0.25) = -1.0; clamp makes it +0.0
        res = interp.execute_asm("v_trig_f64 v[0:1], -v[2:3] clamp")
        self.assertEqual(res, F64_POS_ZERO)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_POS_ZERO)

    def test_execute_cos_mode(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(2, F64_POS_ZERO)
        res = interp.execute_asm("v_trig_f64 v[0:1], v[2:3] cos")
        self.assertEqual(res, F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_ONE)

    def test_sequential_program_execution(self) -> None:
        interp = ScalarInterpreter()
        # Step 1: v[0:1] = sin(0.25 turns) = 1.0
        interp.execute_asm(f"v_trig_f64 v[0:1], 0x{F64_QUARTER:016x}")
        # Step 2: v[2:3] = cos(0.00 turns) = 1.0
        interp.execute_asm("v_trig_f64 v[2:3], 0x0 cos")
        # Step 3: v[4:5] = sin(v[0:1]) where v[0:1] is 1.0 -> sin(1.0 turn) = 0.0
        interp.execute_asm("v_trig_f64 v[4:5], v[0:1]")

        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(2), F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(4), F64_POS_ZERO)


# ==============================================================================
# SIMD Wavefront Execution Model (Wave32 / Wave64) for v_trig_f64
# ==============================================================================

def ref_simd_trig_f64(
    src0_lanes: list[int],
    wave_size: int = 64,
    exec_mask: int | None = None,
    prev_dst: list[int] | None = None,
    *,
    mode: TrigMode = TrigMode.SIN,
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
            output[lane] = ref_trig_f64(
                src0_lanes[lane],
                mode=mode,
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

    def step_v_trig_f64(
        self,
        dst_reg: int,
        src0_reg: int,
        *,
        mode: TrigMode = TrigMode.SIN,
        clamp: bool = False,
        src0_neg: bool = False,
        src0_abs: bool = False,
    ) -> list[int]:
        active_mask = self.exec_mask & ((1 << self.wave_size) - 1)
        for lane in range(self.wave_size):
            if (active_mask >> lane) & 1:
                s0 = self.lanes[lane].get_vgpr_pair(src0_reg)
                res = ref_trig_f64(
                    s0,
                    mode=mode,
                    clamp=clamp,
                    src0_neg=src0_neg,
                    src0_abs=src0_abs,
                )
                self.lanes[lane].set_vgpr_pair(dst_reg, res)
        return self.load_vgpr_pair_wave(dst_reg)


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and EXEC masking."""

    def test_wave32_full_exec(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        angles = [f64_to_u64(i / 32.0) for i in range(32)]
        wave.store_vgpr_pair_wave(2, angles)

        res = wave.step_v_trig_f64(0, 2, mode=TrigMode.SIN)
        expected = ref_simd_trig_f64(angles, wave_size=32, mode=TrigMode.SIN)
        self.assertEqual(res, expected)

    def test_wave64_full_exec(self) -> None:
        wave = SIMDWavefront(wave_size=64)
        angles = [f64_to_u64(i / 64.0) for i in range(64)]
        wave.store_vgpr_pair_wave(4, angles)

        res = wave.step_v_trig_f64(0, 4, mode=TrigMode.COS)
        expected = ref_simd_trig_f64(angles, wave_size=64, mode=TrigMode.COS)
        self.assertEqual(res, expected)

    def test_wave32_partial_exec_preservation(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        sentinel = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        wave.store_vgpr_pair_wave(0, [sentinel] * 32)

        angles = [F64_QUARTER] * 32
        wave.store_vgpr_pair_wave(2, angles)

        # Alternating mask: even lanes active
        wave.exec_mask = 0x55555555
        res = wave.step_v_trig_f64(0, 2, mode=TrigMode.SIN)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(res[lane], F64_ONE, f"Active lane {lane} failed to update")
            else:
                self.assertEqual(res[lane], sentinel, f"Inactive lane {lane} was mutated")

    def test_wave64_upper_half_exec(self) -> None:
        wave = SIMDWavefront(wave_size=64)
        sentinel = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        wave.store_vgpr_pair_wave(0, [sentinel] * 64)

        angles = [F64_QUARTER] * 64
        wave.store_vgpr_pair_wave(2, angles)

        # Only upper 32 lanes active
        wave.exec_mask = 0xFFFFFFFF00000000
        res = wave.step_v_trig_f64(0, 2, mode=TrigMode.SIN)

        for lane in range(64):
            if lane >= 32:
                self.assertEqual(res[lane], F64_ONE, f"Active upper lane {lane} failed to update")
            else:
                self.assertEqual(res[lane], sentinel, f"Inactive lower lane {lane} was overwritten")

    def test_wave_exec_all_zero(self) -> None:
        wave = SIMDWavefront(wave_size=32)
        sentinel = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        wave.store_vgpr_pair_wave(0, [sentinel] * 32)
        wave.store_vgpr_pair_wave(2, [F64_QUARTER] * 32)

        wave.exec_mask = 0x00000000
        res = wave.step_v_trig_f64(0, 2, mode=TrigMode.SIN)
        self.assertEqual(res, [sentinel] * 32)

    def test_wave_single_active_lane(self) -> None:
        wave = SIMDWavefront(wave_size=64)
        sentinel = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        wave.store_vgpr_pair_wave(0, [sentinel] * 64)
        wave.store_vgpr_pair_wave(2, [F64_QUARTER] * 64)

        # Only lane 42 active
        wave.exec_mask = 1 << 42
        res = wave.step_v_trig_f64(0, 2, mode=TrigMode.SIN)

        for lane in range(64):
            if lane == 42:
                self.assertEqual(res[lane], F64_ONE)
            else:
                self.assertEqual(res[lane], sentinel)

    def test_wave_size_validation(self) -> None:
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDWavefront(wave_size=128)
        with self.assertRaises(ValueError):
            ref_simd_trig_f64([0] * 16, wave_size=16)


# ==============================================================================
# Register Aliasing and In-Place Operand Preservation for v_trig_f64
# ==============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_inplace_dst_equals_src0(self) -> None:
        interp = ScalarInterpreter()
        interp.vgpr.set_vgpr_pair(4, F64_QUARTER)

        # v[4:5] = v_trig_f64(v[4:5]) -> sin(0.25 turns) = 1.0
        res = interp.execute_asm("v_trig_f64 v[4:5], v[4:5]")
        self.assertEqual(res, F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(4), F64_ONE)

    def test_partially_overlapping_register_pairs(self) -> None:
        interp = ScalarInterpreter()
        # Setup: src0 at v[0:1]
        interp.vgpr.set_vgpr_pair(0, F64_QUARTER)
        v0_before = interp.vgpr.get_vgpr(0)

        # Execute with dst as unaligned v[1:2] overlapping src0 via v[1]
        dst = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=1)
        src0 = ASTOperand(kind=OperandKind.VGPR_PAIR, reg_idx=0)
        inst = ASTInstruction(opcode="v_trig_f64", dst=dst, src0=src0, mode=TrigMode.SIN)
        res = interp.execute_ast(inst)

        self.assertEqual(res, F64_ONE)
        self.assertEqual(interp.vgpr.get_vgpr_pair(1), F64_ONE)
        # Non-overlapping boundary dword v[0] must remain completely unchanged
        self.assertEqual(interp.vgpr.get_vgpr(0), v0_before)

    def test_adjacent_register_isolation(self) -> None:
        interp = ScalarInterpreter()
        sentinel_lo = 0x11112222
        sentinel_hi = 0x33334444
        sentinel = join_dwords(sentinel_lo, sentinel_hi)

        interp.vgpr.set_vgpr_pair(0, sentinel)
        interp.vgpr.set_vgpr_pair(2, F64_QUARTER)
        interp.vgpr.set_vgpr_pair(4, sentinel)

        # Execute v_trig_f64 writing only to v[2:3]
        interp.execute_asm("v_trig_f64 v[2:3], v[2:3]")

        self.assertEqual(interp.vgpr.get_vgpr_pair(2), F64_ONE)
        # Adjacent v[0:1] and v[4:5] must not be corrupted
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), sentinel)
        self.assertEqual(interp.vgpr.get_vgpr_pair(4), sentinel)

    def test_simd_inplace_divergent_mask(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        init_vals = [f64_to_u64(0.25) if i % 2 == 1 else f64_to_u64(0.12345) for i in range(32)]
        sim.store_vgpr_pair_wave(0, init_vals)

        # Active on odd lanes only: 0xAAAAAAAA
        sim.exec_mask = 0xAAAAAAAA
        res = sim.step_v_trig_f64(0, 0, mode=TrigMode.SIN)

        for lane in range(32):
            if lane % 2 == 1:
                # Active lane: updated to sin(0.25) = 1.0
                self.assertEqual(res[lane], F64_ONE, f"Lane {lane} failed to update")
            else:
                # Inactive lane: preserved original
                self.assertEqual(res[lane], init_vals[lane], f"Lane {lane} was mutated")

    def test_successive_inplace_trig_chain(self) -> None:
        interp = ScalarInterpreter()
        # Start with 0.25 -> sin(0.25) = 1.0 -> sin(1.0) = 0.0 -> sin(0.0) = 0.0
        interp.vgpr.set_vgpr_pair(0, F64_QUARTER)
        interp.execute_asm("v_trig_f64 v[0:1], v[0:1]")
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_ONE)

        interp.execute_asm("v_trig_f64 v[0:1], v[0:1]")
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_POS_ZERO)

        interp.execute_asm("v_trig_f64 v[0:1], v[0:1]")
        self.assertEqual(interp.vgpr.get_vgpr_pair(0), F64_POS_ZERO)


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Isolation for v_trig_f64
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


def compute_trig_f64_status_flags(src0: int, result: int) -> StatusFlags:
    flags = StatusFlags()
    s0 = src0 & MASK64
    res = result & MASK64

    # Invalid operation: SNaN operand or Infinite operand (sin/cos of Inf is invalid)
    if is_f64_snan(s0) or is_f64_inf(s0):
        flags.invalid_op = True

    if is_f64_nan(res):
        flags.nan = True
    else:
        flags.negative = bool(f64_sign(res))
        if is_f64_zero(res):
            flags.zero = True
        else:
            # Check inexact: transcendental results on non-cardinal inputs are inexact
            val = u64_to_f64(s0)
            red = ref_turn_reduce_f64(val)
            if red not in (0.0, 0.25, -0.25, 0.5, -0.5):
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

    def test_status_flags_invalid_operation(self) -> None:
        # sin(Inf) -> Invalid operation exception and NaN
        res_inf = ref_trig_f64(F64_POS_INF, mode=TrigMode.SIN)
        flags_inf = compute_trig_f64_status_flags(F64_POS_INF, res_inf)
        self.assertTrue(flags_inf.invalid_op)
        self.assertTrue(flags_inf.nan)

        # SNaN -> Invalid operation exception
        snan = (F64_POS_INF | 0x0001000000000000) & ~F64_QUIET_BIT
        res_snan = ref_trig_f64(snan, mode=TrigMode.SIN)
        flags_snan = compute_trig_f64_status_flags(snan, res_snan)
        self.assertTrue(flags_snan.invalid_op)
        self.assertTrue(flags_snan.nan)

    def test_status_flags_zero_and_negative(self) -> None:
        # sin(-0.25 turns) = -1.0 (negative, non-zero)
        res_neg = ref_trig_f64(F64_NEG_QUARTER, mode=TrigMode.SIN)
        flags_neg = compute_trig_f64_status_flags(F64_NEG_QUARTER, res_neg)
        self.assertTrue(flags_neg.negative)
        self.assertFalse(flags_neg.zero)

        # sin(-0.0 turns) = -0.0 (negative zero)
        res_nzero = ref_trig_f64(F64_NEG_ZERO, mode=TrigMode.SIN)
        flags_nzero = compute_trig_f64_status_flags(F64_NEG_ZERO, res_nzero)
        self.assertTrue(flags_nzero.negative)
        self.assertTrue(flags_nzero.zero)

    def test_status_flags_inexact(self) -> None:
        # 0.125 turn = 45 degrees -> sin(pi/4) is irrational and inexact
        t = f64_to_u64(0.125)
        res = ref_trig_f64(t, mode=TrigMode.SIN)
        flags = compute_trig_f64_status_flags(t, res)
        self.assertTrue(flags.inexact)

    def test_vcc_and_scc_preserved(self) -> None:
        # Floating point v_trig_f64 does not generate carry/borrow and never writes VCC or SCC
        interp = ScalarInterpreter()
        interp.vcc = 0x12345678_9ABCDEF0
        interp.scc = 1
        interp.vgpr.set_vgpr_pair(2, F64_QUARTER)

        interp.execute_asm("v_trig_f64 v[0:1], v[2:3]")
        self.assertEqual(interp.vcc, 0x12345678_9ABCDEF0)
        self.assertEqual(interp.scc, 1)

        sim = SIMDWavefront(wave_size=32)
        sim.vcc = 0xCAFEBABE
        sim.store_vgpr_pair_wave(2, [F64_QUARTER] * 32)
        sim.step_v_trig_f64(0, 2)
        self.assertEqual(sim.vcc, 0xCAFEBABE)

    def test_condition_code_generation(self) -> None:
        v_zero = [F64_POS_ZERO] * 4
        # Four angles in turns: [-0.25 (sin=-1), 0.0 (sin=0), 0.25 (sin=1), Inf (sin=NaN)]
        srcs = [F64_NEG_QUARTER, F64_POS_ZERO, F64_QUARTER, F64_POS_INF]
        sins = [ref_trig_f64(a, mode=TrigMode.SIN) for a in srcs]

        # Compare < 0: only lane 0 (-1.0 < 0) -> bit 0
        mask_lt = generate_condition_mask_f64("<", sins, v_zero)
        self.assertEqual(mask_lt, 0b0001)

        # Compare == 0: only lane 1 (0.0 == 0) -> bit 1
        mask_eq = generate_condition_mask_f64("==", sins, v_zero)
        self.assertEqual(mask_eq, 0b0010)

        # Compare > 0: only lane 2 (1.0 > 0) -> bit 2
        mask_gt = generate_condition_mask_f64(">", sins, v_zero)
        self.assertEqual(mask_gt, 0b0100)

        # Compare unordered: only lane 3 (NaN) -> bit 3
        mask_u = generate_condition_mask_f64("u", sins, v_zero)
        self.assertEqual(mask_u, 0b1000)


# ==============================================================================
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling for v_trig_f64
# ==============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Test suite for IEEE-754 boundary extremes, subnormals, and NaN handling."""

    def test_nan_snan_quieting(self) -> None:
        # Positive signaling NaN
        snan_pos = 0x7FF0000000000001
        self.assertTrue(is_f64_snan(snan_pos))
        res_pos = ref_trig_f64(snan_pos, mode=TrigMode.SIN)
        self.assertTrue(is_f64_qnan(res_pos))

        # Negative signaling NaN
        snan_neg = 0xFFF0000000000001
        self.assertTrue(is_f64_snan(snan_neg))
        res_neg = ref_trig_f64(snan_neg, mode=TrigMode.COS)
        self.assertTrue(is_f64_qnan(res_neg))

    def test_qnan_propagation(self) -> None:
        qnan_custom = 0x7FF8000000001234
        self.assertTrue(is_f64_qnan(qnan_custom))
        res_sin = ref_trig_f64(qnan_custom, mode=TrigMode.SIN)
        self.assertTrue(is_f64_qnan(res_sin))
        res_cos = ref_trig_f64(qnan_custom, mode=TrigMode.COS)
        self.assertTrue(is_f64_qnan(res_cos))

    def test_infinities_return_canonical_qnan(self) -> None:
        self.assertEqual(ref_trig_f64(F64_POS_INF, mode=TrigMode.SIN), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_NEG_INF, mode=TrigMode.SIN), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_POS_INF, mode=TrigMode.COS), F64_CANONICAL_QNAN)
        self.assertEqual(ref_trig_f64(F64_NEG_INF, mode=TrigMode.COS), F64_CANONICAL_QNAN)

    def test_signed_zeros_preservation(self) -> None:
        # sin(+0.0) = +0.0, sin(-0.0) = -0.0
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.SIN), F64_POS_ZERO)
        self.assertEqual(ref_trig_f64(F64_NEG_ZERO, mode=TrigMode.SIN), F64_NEG_ZERO)
        # cos(+0.0) = 1.0, cos(-0.0) = 1.0
        self.assertEqual(ref_trig_f64(F64_POS_ZERO, mode=TrigMode.COS), F64_ONE)
        self.assertEqual(ref_trig_f64(F64_NEG_ZERO, mode=TrigMode.COS), F64_ONE)

    def test_subnormal_inputs(self) -> None:
        # Smallest subnormal
        sub_min = F64_MIN_SUBNORMAL
        self.assertTrue(is_f64_subnormal(sub_min))
        res_min = ref_trig_f64(sub_min, mode=TrigMode.SIN)
        # In IEEE mode, not flushed to zero: sin(2*pi*x) > 0
        self.assertFalse(is_f64_zero(res_min))
        self.assertEqual(f64_sign(res_min), 0)

        # Negative subnormal
        neg_sub_min = F64_MIN_SUBNORMAL | F64_SIGN_MASK
        res_neg = ref_trig_f64(neg_sub_min, mode=TrigMode.SIN)
        self.assertFalse(is_f64_zero(res_neg))
        self.assertEqual(f64_sign(res_neg), 1)

        # Largest subnormal: cos(subnormal) == 1.0
        sub_max = F64_MAX_SUBNORMAL
        res_cos = ref_trig_f64(sub_max, mode=TrigMode.COS)
        self.assertEqual(res_cos, F64_ONE)

    def test_normal_boundary_extremes(self) -> None:
        # Smallest normal float: 2^-1022
        min_norm = F64_MIN_NORMAL
        self.assertTrue(is_f64_normal(min_norm))
        res_sin = ref_trig_f64(min_norm, mode=TrigMode.SIN)
        self.assertFalse(is_f64_zero(res_sin))
        self.assertFalse(is_f64_nan(res_sin))

        # Numbers >= 2^53 are exact integers with zero fractional part
        int_power = f64_to_u64(2.0**53 + 4.0)
        self.assertEqual(ref_trig_f64(int_power, mode=TrigMode.SIN), F64_POS_ZERO)
        self.assertEqual(ref_trig_f64(int_power, mode=TrigMode.COS), F64_ONE)

        # Largest normal float: 1.7976931348623157e+308
        max_norm = F64_MAX_NORMAL
        self.assertTrue(is_f64_normal(max_norm))
        res_max = ref_trig_f64(max_norm, mode=TrigMode.SIN)
        # As max_norm is an integer with zero fractional turn, sin is zero
        self.assertEqual(res_max, F64_POS_ZERO)


# ==============================================================================
# Mutants for Mutation Detection
# ==============================================================================

def mutant_radians_not_turns(s0: int, *, mode: TrigMode = TrigMode.SIN, **kwargs) -> int:
    """Mutant 1: Treats input directly as radians instead of turns (missing 2*pi scale)."""
    val = u64_to_f64(s0)
    if is_f64_nan(s0) or is_f64_inf(s0):
        return F64_CANONICAL_QNAN
    res = math.sin(val) if mode == TrigMode.SIN else math.cos(val)
    return f64_to_u64(res)


def mutant_ignore_neg(s0: int, *, src0_neg: bool = False, **kwargs) -> int:
    """Mutant 2: Ignores negative modifier."""
    return ref_trig_f64(s0, src0_neg=False, **kwargs)


def mutant_ignore_abs(s0: int, *, src0_abs: bool = False, **kwargs) -> int:
    """Mutant 3: Ignores absolute value modifier."""
    return ref_trig_f64(s0, src0_abs=False, **kwargs)


def mutant_ignore_clamp(s0: int, *, clamp: bool = False, **kwargs) -> int:
    """Mutant 4: Ignores clamp modifier."""
    return ref_trig_f64(s0, clamp=False, **kwargs)


def mutant_clamp_nan_to_nan(s0: int, *, clamp: bool = False, **kwargs) -> int:
    """Mutant 5: Clamps numbers but fails to clamp NaN to +0.0."""
    res = ref_trig_f64(s0, clamp=False, **kwargs)
    if clamp and not is_f64_nan(res):
        val = u64_to_f64(res)
        if val < 0.0:
            return F64_POS_ZERO
        elif val > 1.0:
            return F64_ONE
    return res


def mutant_snan_not_quieted(s0: int, **kwargs) -> int:
    """Mutant 6: Returns SNaN without setting the quiet bit."""
    if is_f64_snan(s0):
        return s0
    return ref_trig_f64(s0, **kwargs)


def mutant_inf_returns_zero(s0: int, **kwargs) -> int:
    """Mutant 7: Returns 0.0 on infinite inputs instead of Canonical QNaN."""
    if is_f64_inf(s0):
        return F64_POS_ZERO
    return ref_trig_f64(s0, **kwargs)


def mutant_f32_truncation(s0: int, **kwargs) -> int:
    """Mutant 8: Truncates intermediate calculation to 32-bit single precision float."""
    f0 = struct.unpack("<f", struct.pack("<f", u64_to_f64(s0)))[0]
    red = ref_turn_reduce_f64(float(f0))
    res = math.sin(red * TWO_PI)
    f_res = struct.unpack("<f", struct.pack("<f", res))[0]
    return f64_to_u64(float(f_res))


# ==============================================================================
# Negative Controls, Syntax Rejection, and Mutation Detection
# ==============================================================================

class TestNegativeControlsAndSyntaxRejection(unittest.TestCase):
    """Test suite for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_bad_opcodes(self) -> None:
        for bad_op in ("v_trig_f32", "v_sin_f32", "s_trig_f64", "v_add_f64", "unknown_op"):
            with self.assertRaises(ValueError):
                parse_asm_line(f"{bad_op} v[0:1], v[2:3]")

    def test_syntax_rejection_bad_operands(self) -> None:
        for bad_line in (
            "v_trig_f64 v0, v[2:3]",
            "v_trig_f64 v[0:1], invalid_reg",
            "v_trig_f64 v[0:1], v[300:301]",
        ):
            with self.assertRaises(ValueError):
                parse_asm_line(bad_line)

    def test_syntax_rejection_operand_count(self) -> None:
        # Too few operands
        with self.assertRaises(ValueError):
            parse_asm_line("v_trig_f64 v[0:1]")
        # Too many operands
        with self.assertRaises(ValueError):
            parse_asm_line("v_trig_f64 v[0:1], v[2:3], v[4:5]")

    def test_syntax_rejection_destination_modifiers(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("v_trig_f64 -v[0:1], v[2:3]")
        with self.assertRaises(ValueError):
            parse_asm_line("v_trig_f64 |v[0:1]|, v[2:3]")

    def test_syntax_rejection_empty_lines(self) -> None:
        with self.assertRaises(ValueError):
            parse_asm_line("")
        with self.assertRaises(ValueError):
            parse_asm_line("   ; comment only")

    def test_negative_control_radians_instead_of_turns(self) -> None:
        # sin(0.25 turns) = 1.0; sin(0.25 rad) ~= 0.2474
        s0 = F64_QUARTER
        good = ref_trig_f64(s0, mode=TrigMode.SIN)
        bad = mutant_radians_not_turns(s0, mode=TrigMode.SIN)
        self.assertNotEqual(good, bad)
        self.assertEqual(good, F64_ONE)
        self.assertNotEqual(bad, F64_ONE)

    def test_negative_control_f32_truncation_detected(self) -> None:
        # An angle whose low 29 mantissa bits matter in binary64
        angle = 0.125 + 2.0**(-30)
        s0 = f64_to_u64(angle)
        good = ref_trig_f64(s0)
        bad = mutant_f32_truncation(s0)
        self.assertNotEqual(good, bad, "Single precision truncation mutation must be detected")

    def test_mutation_detection_all_mutants_caught(self) -> None:
        # Mutant 1: radians not turns
        self.assertNotEqual(ref_trig_f64(F64_QUARTER), mutant_radians_not_turns(F64_QUARTER))

        # Mutant 2: ignore neg
        self.assertNotEqual(
            ref_trig_f64(F64_QUARTER, src0_neg=True),
            mutant_ignore_neg(F64_QUARTER, src0_neg=True),
        )

        # Mutant 3: ignore abs
        self.assertNotEqual(
            ref_trig_f64(F64_NEG_QUARTER, src0_abs=True),
            mutant_ignore_abs(F64_NEG_QUARTER, src0_abs=True),
        )

        # Mutant 4: ignore clamp
        self.assertNotEqual(
            ref_trig_f64(F64_NEG_QUARTER, clamp=True),
            mutant_ignore_clamp(F64_NEG_QUARTER, clamp=True),
        )

        # Mutant 5: clamp NaN to NaN instead of +0.0
        self.assertNotEqual(
            ref_trig_f64(F64_POS_INF, clamp=True),
            mutant_clamp_nan_to_nan(F64_POS_INF, clamp=True),
        )

        # Mutant 6: SNaN not quieted
        snan = 0x7FF0000000000001
        self.assertNotEqual(ref_trig_f64(snan), mutant_snan_not_quieted(snan))

        # Mutant 7: Inf returns 0 instead of QNaN
        self.assertNotEqual(ref_trig_f64(F64_POS_INF), mutant_inf_returns_zero(F64_POS_INF))

    def test_mutation_detects_exec_mask_leakage(self) -> None:
        sim = SIMDWavefront(wave_size=32)
        sentinel = join_dwords(OLD_VGPR_SENTINEL, OLD_VGPR_SENTINEL)
        sim.store_vgpr_pair_wave(0, [sentinel] * 32)
        sim.store_vgpr_pair_wave(2, [F64_QUARTER] * 32)

        # Only lane 0 active
        sim.exec_mask = 0x00000001
        res = sim.step_v_trig_f64(0, 2)

        self.assertEqual(res[0], F64_ONE)
        for lane in range(1, 32):
            self.assertEqual(res[lane], sentinel, f"Lane {lane} leaked despite inactive EXEC mask")


if __name__ == "__main__":
    unittest.main()
