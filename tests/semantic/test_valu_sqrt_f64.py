from __future__ import annotations

import math
import struct
import unittest
from dataclasses import dataclass, field
from enum import Enum, auto
from fractions import Fraction
from typing import List, Optional, Tuple, Union

# IEEE 754 binary64 (double precision) architectural constants
SIGN_MASK: int = 0x8000_0000_0000_0000
EXP_MASK: int = 0x7FF0_0000_0000_0000
FRAC_MASK: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT: int = 0x0008_0000_0000_0000
CANONICAL_QNAN: int = 0x7FF8_0000_0000_0000

POS_ZERO: int = 0x0000_0000_0000_0000
NEG_ZERO: int = 0x8000_0000_0000_0000
POS_INF: int = 0x7FF0_0000_0000_0000
NEG_INF: int = 0xFFF0_0000_0000_0000

ONE_F64: int = 0x3FF0_0000_0000_0000


def f64_to_bits(val: float) -> int:
    """Convert a Python float (IEEE 754 double) to 64-bit integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit integer bits to a Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & 0xFFFF_FFFF_FFFF_FFFF))[0]


def split_f64_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 63
    exponent = (bits & EXP_MASK) >> 52
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def words_to_f64_bits(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit integer representation."""
    return ((hi & 0xFFFF_FFFF) << 32) | (lo & 0xFFFF_FFFF)


def f64_bits_to_words(bits: int) -> Tuple[int, int]:
    """Split a 64-bit integer into two 32-bit VGPR words (lo, hi)."""
    lo = bits & 0xFFFF_FFFF
    hi = (bits >> 32) & 0xFFFF_FFFF
    return lo, hi


def is_nan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 NaN."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) != 0)


def is_qnan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Quiet NaN."""
    return is_nan(bits) and bool(bits & QNAN_BIT)


def is_snan(bits: int) -> bool:
    """Return True if bits represent an IEEE 754 Signaling NaN."""
    return is_nan(bits) and not bool(bits & QNAN_BIT)


def is_inf(bits: int) -> bool:
    """Return True if bits represent IEEE 754 positive or negative infinity."""
    return ((bits & EXP_MASK) == EXP_MASK) and ((bits & FRAC_MASK) == 0)


def is_subnormal(bits: int) -> bool:
    """Return True if bits represent a subnormal (denormal) double-precision float."""
    return ((bits & EXP_MASK) == 0) and ((bits & FRAC_MASK) != 0)


def is_zero(bits: int) -> bool:
    """Return True if bits represent positive or negative zero."""
    return (bits & ~SIGN_MASK) == 0


def apply_f64_modifiers(bits: int, abs_mod: bool, neg_mod: bool) -> int:
    """Apply VOP3 input modifiers: abs clears sign bit; neg toggles sign bit."""
    res = bits & 0xFFFF_FFFF_FFFF_FFFF
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def oracle_v_sqrt_f64(
    src_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
    clamp: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_sqrt_f64 double-precision square root.

    Computes D = sqrt(src) conforming to IEEE 754-2008 double precision (binary64)
    with Round-to-Nearest-Even (RNE), applying input modifiers and clamp.
    """
    s = apply_f64_modifiers(src_bits, abs_mod, neg_mod)

    # 1. NaN handling
    if is_nan(s):
        if clamp:
            return POS_ZERO
        # Quiet signaling NaN or preserve quiet NaN
        return s | QNAN_BIT

    # 2. Signed zero handling: sqrt(+0) = +0, sqrt(-0) = -0
    if is_zero(s):
        if s == NEG_ZERO:
            return POS_ZERO if clamp else NEG_ZERO
        return POS_ZERO

    # 3. Negative values (< 0): Invalid operation -> Canonical Quiet NaN
    if bool(s & SIGN_MASK):
        return POS_ZERO if clamp else CANONICAL_QNAN

    # 4. Positive infinity: sqrt(+inf) = +inf
    if s == POS_INF:
        return ONE_F64 if clamp else POS_INF

    # 5. Finite positive values
    f_val = bits_to_f64(s)
    res_f = math.sqrt(f_val)
    res = f64_to_bits(res_f)

    # 6. Clamp modifier ([0.0, 1.0])
    if clamp:
        if res_f < 0.0 or res == NEG_ZERO:
            res = POS_ZERO
        elif res_f > 1.0 or res == POS_INF:
            res = ONE_F64

    return res


# ==============================================================================
# Symbolic AST Opcode Definitions and Typed Expressions
# ==============================================================================

class RegType(Enum):
    VGPR = auto()
    SGPR = auto()


@dataclass(frozen=True)
class RegPair:
    """Represents an architectural 64-bit register pair (e.g., v[0:1] or s[4:5])."""
    index: int
    reg_type: RegType

    def __post_init__(self) -> None:
        if not (0 <= self.index <= 254):
            raise ValueError(f"Register pair base index {self.index} out of range [0, 254]")

    @property
    def lo_reg(self) -> int:
        return self.index

    @property
    def hi_reg(self) -> int:
        return self.index + 1

    @property
    def name(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Literal64:
    """Represents a 64-bit literal or inline float constant."""
    bits: int

    @classmethod
    def from_float(cls, val: float) -> Literal64:
        return cls(f64_to_bits(val))

    @property
    def float_value(self) -> float:
        return bits_to_f64(self.bits)

    @property
    def name(self) -> str:
        if self.bits == POS_ZERO:
            return "0.0"
        if self.bits == NEG_ZERO:
            return "-0.0"
        f = self.float_value
        if math.isnan(f):
            return "nan"
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f"{f:g}"


@dataclass(frozen=True)
class VOP3Operand:
    """Typed expression operand with optional input modifiers (abs, neg)."""
    source: Union[RegPair, Literal64]
    abs_mod: bool = False
    neg_mod: bool = False

    def disassemble(self) -> str:
        s = self.source.name
        if self.abs_mod:
            s = f"|{s}|"
        if self.neg_mod:
            s = f"-{s}"
        return s


@dataclass(frozen=True)
class VSqrtF64Instruction:
    """Symbolic AST node representing a RDNA2 v_sqrt_f64 instruction."""
    dst: RegPair
    src: VOP3Operand
    clamp: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise TypeError(f"Destination of VALU instruction must be VGPR, got {self.dst.reg_type}")

    def disassemble(self) -> str:
        s = f"v_sqrt_f64 {self.dst.name}, {self.src.disassemble()}"
        if self.clamp:
            s += " clamp"
        return s


# ==============================================================================
# Deterministic Scalar Instruction Interpreter
# ==============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register file for a single wavefront lane."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def get_vgpr64(self, index: int) -> int:
        return words_to_f64_bits(self.vgprs[index], self.vgprs[index + 1])

    def set_vgpr64(self, index: int, bits: int) -> None:
        lo, hi = f64_bits_to_words(bits)
        self.vgprs[index] = lo
        self.vgprs[index + 1] = hi

    def get_sgpr64(self, index: int) -> int:
        return words_to_f64_bits(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr64(self, index: int, bits: int) -> None:
        lo, hi = f64_bits_to_words(bits)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_sqrt_f64."""

    def resolve_operand(self, op: VOP3Operand, state: ScalarMachineState) -> int:
        if isinstance(op.source, RegPair):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr64(op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr64(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal64):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VSqrtF64Instruction, state: ScalarMachineState) -> int:
        # Atomic read phase
        s_raw = self.resolve_operand(instr.src, state)

        # Computation phase
        result_bits = oracle_v_sqrt_f64(
            src_bits=s_raw,
            abs_mod=instr.src.abs_mod,
            neg_mod=instr.src.neg_mod,
            clamp=instr.clamp,
        )

        # Write-back phase
        state.set_vgpr64(instr.dst.index, result_bits)
        return result_bits


# ==============================================================================
# SIMD Multi-Lane Execution Engine and EXEC Masking
# ==============================================================================

@dataclass
class SIMDState:
    """Multi-lane SIMD machine state representing Wave32 or Wave64."""
    num_lanes: int
    exec_mask: Optional[int] = None
    vgprs: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 256)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.vgprs:
            self.vgprs = [[0] * 256 for _ in range(self.num_lanes)]
        if self.exec_mask is None:
            self.exec_mask = (1 << self.num_lanes) - 1

    def get_vgpr64(self, lane: int, index: int) -> int:
        return words_to_f64_bits(self.vgprs[lane][index], self.vgprs[lane][index + 1])

    def set_vgpr64(self, lane: int, index: int, bits: int) -> None:
        lo, hi = f64_bits_to_words(bits)
        self.vgprs[lane][index] = lo
        self.vgprs[lane][index + 1] = hi

    def get_sgpr64(self, index: int) -> int:
        return words_to_f64_bits(self.sgprs[index], self.sgprs[index + 1])

    def set_sgpr64(self, index: int, bits: int) -> None:
        lo, hi = f64_bits_to_words(bits)
        self.sgprs[index] = lo
        self.sgprs[index + 1] = hi


class SIMDInterpreter:
    """SIMD instruction interpreter handling multi-lane divergence and EXEC masking."""

    def resolve_lane_operand(self, op: VOP3Operand, state: SIMDState, lane: int) -> int:
        if isinstance(op.source, RegPair):
            if op.source.reg_type == RegType.VGPR:
                return state.get_vgpr64(lane, op.source.index)
            elif op.source.reg_type == RegType.SGPR:
                return state.get_sgpr64(op.source.index)
            else:
                raise ValueError(f"Unknown register type: {op.source.reg_type}")
        elif isinstance(op.source, Literal64):
            return op.source.bits
        else:
            raise TypeError(f"Unsupported operand source type: {type(op.source)}")

    def step(self, instr: VSqrtF64Instruction, state: SIMDState) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * state.num_lanes
        # Atomic read phase across all lanes
        s_vals = [
            self.resolve_lane_operand(instr.src, state, lane)
            for lane in range(state.num_lanes)
        ]

        # Execution phase
        for lane in range(state.num_lanes):
            if (state.exec_mask & (1 << lane)) != 0:
                res = oracle_v_sqrt_f64(
                    src_bits=s_vals[lane],
                    abs_mod=instr.src.abs_mod,
                    neg_mod=instr.src.neg_mod,
                    clamp=instr.clamp,
                )
                state.set_vgpr64(lane, instr.dst.index, res)
                results[lane] = res

        return results


# ==============================================================================
# Condition Codes, Status Flags, and Carry/Borrow Generation
# ==============================================================================

@dataclass(frozen=True)
class FPStatusFlags:
    """Architectural floating-point condition codes and IEEE 754 status flags."""
    invalid: bool = False
    overflow: bool = False
    underflow: bool = False
    inexact: bool = False
    zero: bool = False
    negative: bool = False
    denormal: bool = False
    infinite: bool = False
    nan: bool = False


def evaluate_v_sqrt_f64_flags(
    src_bits: int,
    res_bits: int,
    abs_mod: bool = False,
    neg_mod: bool = False,
    clamp: bool = False,
) -> FPStatusFlags:
    """Evaluate IEEE 754 architectural flags and condition codes for v_sqrt_f64."""
    s = apply_f64_modifiers(src_bits, abs_mod, neg_mod)

    s_nan = is_nan(s)
    s_snan = is_snan(s)
    s_inf = is_inf(s)
    s_neg = bool(s & SIGN_MASK)
    s_zero = is_zero(s)

    invalid = False
    if s_snan:
        invalid = True
    elif (s_neg and not s_zero) or (s_inf and s_neg):
        invalid = True

    nan_flag = is_nan(res_bits)
    inf_flag = is_inf(res_bits)
    zero_flag = is_zero(res_bits)
    neg_flag = bool(res_bits & SIGN_MASK)
    denorm_flag = is_subnormal(res_bits)

    overflow = False
    underflow = False
    if denorm_flag:
        underflow = True

    inexact = False
    if not (s_nan or s_inf or nan_flag or inf_flag or invalid or s_zero):
        f = bits_to_f64(s)
        res_f = bits_to_f64(res_bits)
        if not clamp:
            if res_f * res_f != f:
                inexact = True
        else:
            clamped_math = min(1.0, max(0.0, math.sqrt(f)))
            if clamped_math != res_f or res_f * res_f != f:
                inexact = True

    return FPStatusFlags(
        invalid=invalid,
        overflow=overflow,
        underflow=underflow,
        inexact=inexact,
        zero=zero_flag,
        negative=neg_flag,
        denormal=denorm_flag,
        infinite=inf_flag,
        nan=nan_flag,
    )


def compute_f64_sqrt_borrow(src_bits: int, res_bits: int) -> Tuple[int, int]:
    """Demonstrate non-restoring square root integer mantissa remainder and borrow generation."""
    _, exp, frac = split_f64_bits(src_bits)
    _, res_exp, res_frac = split_f64_bits(res_bits)
    m_src = (1 << 52) | frac if exp != 0 else frac
    m_res = (1 << 52) | res_frac if res_exp != 0 else res_frac
    radicand = m_src << 52
    squared = m_res * m_res
    remainder = radicand - squared
    borrow = 1 if remainder < 0 else 0
    return remainder, borrow


# ==============================================================================
# Unit Tests
# ==============================================================================

class TestF64ModelAndOracle(unittest.TestCase):
    """Unit tests for binary64 bit model, predicates, and reference oracle."""

    def test_bit_representation_roundtrip(self) -> None:
        values = [0.0, -0.0, 1.0, -1.0, 0.25, 4.0, 1e-100, 1e100, math.pi]
        for val in values:
            with self.subTest(val=val):
                bits = f64_to_bits(val)
                recovered = bits_to_f64(bits)
                if math.isnan(val):
                    self.assertTrue(math.isnan(recovered))
                else:
                    self.assertEqual(val, recovered)
                    self.assertEqual(math.copysign(1.0, val), math.copysign(1.0, recovered))

    def test_word_packing_vgpr_pair(self) -> None:
        val = 9876543.21098765
        bits = f64_to_bits(val)
        lo, hi = f64_bits_to_words(bits)
        self.assertEqual(lo, bits & 0xFFFF_FFFF)
        self.assertEqual(hi, (bits >> 32) & 0xFFFF_FFFF)
        combined = words_to_f64_bits(lo, hi)
        self.assertEqual(combined, bits)
        self.assertEqual(bits_to_f64(combined), val)

    def test_classification_predicates(self) -> None:
        self.assertTrue(f64_to_bits(0.0) == POS_ZERO)
        self.assertTrue(f64_to_bits(-0.0) == NEG_ZERO)
        self.assertTrue(is_zero(POS_ZERO))
        self.assertTrue(is_zero(NEG_ZERO))
        self.assertFalse(is_zero(f64_to_bits(1.0)))

        self.assertTrue(is_inf(POS_INF))
        self.assertTrue(is_inf(NEG_INF))
        self.assertFalse(is_inf(POS_ZERO))

        snan_bits = 0x7FF0_0000_0000_0001
        qnan_bits = CANONICAL_QNAN
        self.assertTrue(is_nan(snan_bits))
        self.assertTrue(is_snan(snan_bits))
        self.assertFalse(is_qnan(snan_bits))

        self.assertTrue(is_nan(qnan_bits))
        self.assertTrue(is_qnan(qnan_bits))
        self.assertFalse(is_snan(qnan_bits))

        subnorm_bits = 0x0000_0000_0000_0001
        self.assertTrue(is_subnormal(subnorm_bits))
        self.assertFalse(is_subnormal(POS_ZERO))
        self.assertFalse(is_subnormal(f64_to_bits(1.0)))

    def test_exact_finite_sqrt(self) -> None:
        c4 = oracle_v_sqrt_f64(f64_to_bits(4.0))
        self.assertEqual(bits_to_f64(c4), 2.0)

        c9 = oracle_v_sqrt_f64(f64_to_bits(9.0))
        self.assertEqual(bits_to_f64(c9), 3.0)

        c025 = oracle_v_sqrt_f64(f64_to_bits(0.25))
        self.assertEqual(bits_to_f64(c025), 0.5)

        c100 = oracle_v_sqrt_f64(f64_to_bits(100.0))
        self.assertEqual(bits_to_f64(c100), 10.0)

    def test_signed_zero_rules(self) -> None:
        # IEEE 754: sqrt(+0) = +0, sqrt(-0) = -0
        self.assertEqual(oracle_v_sqrt_f64(POS_ZERO), POS_ZERO)
        self.assertEqual(oracle_v_sqrt_f64(NEG_ZERO), NEG_ZERO)

    def test_negative_values_produce_canonical_qnan(self) -> None:
        # sqrt(-4.0) -> CANONICAL_QNAN
        self.assertEqual(oracle_v_sqrt_f64(f64_to_bits(-4.0)), CANONICAL_QNAN)
        self.assertEqual(oracle_v_sqrt_f64(NEG_INF), CANONICAL_QNAN)

    def test_input_modifiers_abs_neg(self) -> None:
        val_neg = f64_to_bits(-16.0)
        # abs on negative: |-16.0| = 16.0 -> sqrt(16.0) = 4.0
        res_abs = oracle_v_sqrt_f64(val_neg, abs_mod=True)
        self.assertEqual(bits_to_f64(res_abs), 4.0)

        # neg on negative: -(-16.0) = 16.0 -> sqrt(16.0) = 4.0
        res_neg = oracle_v_sqrt_f64(val_neg, neg_mod=True)
        self.assertEqual(bits_to_f64(res_neg), 4.0)

        # neg on positive: -(16.0) = -16.0 -> canonical qNaN
        res_neg_pos = oracle_v_sqrt_f64(f64_to_bits(16.0), neg_mod=True)
        self.assertEqual(res_neg_pos, CANONICAL_QNAN)

        # abs and neg together: -|16.0| = -16.0 -> canonical qNaN
        res_both = oracle_v_sqrt_f64(val_neg, abs_mod=True, neg_mod=True)
        self.assertEqual(res_both, CANONICAL_QNAN)

    def test_clamp_modifier(self) -> None:
        # sqrt(4.0) = 2.0 -> clamped to 1.0
        res_hi = oracle_v_sqrt_f64(f64_to_bits(4.0), clamp=True)
        self.assertEqual(res_hi, ONE_F64)

        # sqrt(0.25) = 0.5 -> within [0, 1], unchanged
        res_mid = oracle_v_sqrt_f64(f64_to_bits(0.25), clamp=True)
        self.assertEqual(bits_to_f64(res_mid), 0.5)

        # sqrt(-0.0) with clamp -> +0.0
        res_nz = oracle_v_sqrt_f64(NEG_ZERO, clamp=True)
        self.assertEqual(res_nz, POS_ZERO)

        # sqrt(-4.0) with clamp -> +0.0
        res_neg = oracle_v_sqrt_f64(f64_to_bits(-4.0), clamp=True)
        self.assertEqual(res_neg, POS_ZERO)

    def test_infinity_sqrt(self) -> None:
        self.assertEqual(oracle_v_sqrt_f64(POS_INF), POS_INF)
        self.assertEqual(oracle_v_sqrt_f64(POS_INF, clamp=True), ONE_F64)
        self.assertEqual(oracle_v_sqrt_f64(NEG_INF), CANONICAL_QNAN)
        self.assertEqual(oracle_v_sqrt_f64(NEG_INF, clamp=True), POS_ZERO)

    def test_nan_quietening_and_propagation(self) -> None:
        snan = 0x7FF0_0000_1234_5678
        res = oracle_v_sqrt_f64(snan)
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)

        qnan = 0x7FF8_0000_8765_4321
        res_q = oracle_v_sqrt_f64(qnan)
        self.assertEqual(res_q, qnan)


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_regpair_validation_and_naming(self) -> None:
        v0 = RegPair(0, RegType.VGPR)
        self.assertEqual(v0.name, "v[0:1]")
        self.assertEqual(v0.lo_reg, 0)
        self.assertEqual(v0.hi_reg, 1)

        s8 = RegPair(8, RegType.SGPR)
        self.assertEqual(s8.name, "s[8:9]")
        self.assertEqual(s8.lo_reg, 8)
        self.assertEqual(s8.hi_reg, 9)

        with self.assertRaises(ValueError):
            RegPair(255, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(-1, RegType.VGPR)

    def test_operand_disassembly(self) -> None:
        v = RegPair(2, RegType.VGPR)
        op_plain = VOP3Operand(v)
        self.assertEqual(op_plain.disassemble(), "v[2:3]")

        op_abs = VOP3Operand(v, abs_mod=True)
        self.assertEqual(op_abs.disassemble(), "|v[2:3]|")

        op_neg = VOP3Operand(v, neg_mod=True)
        self.assertEqual(op_neg.disassemble(), "-v[2:3]")

        op_neg_abs = VOP3Operand(v, abs_mod=True, neg_mod=True)
        self.assertEqual(op_neg_abs.disassemble(), "-|v[2:3]|")

        lit = Literal64.from_float(4.0)
        op_lit = VOP3Operand(lit)
        self.assertEqual(op_lit.disassemble(), "4")

    def test_v_sqrt_f64_ast_construction_and_validation(self) -> None:
        dst = RegPair(0, RegType.VGPR)
        src = VOP3Operand(RegPair(2, RegType.VGPR), neg_mod=True, abs_mod=True)
        instr = VSqrtF64Instruction(dst=dst, src=src, clamp=True)

        self.assertEqual(instr.disassemble(), "v_sqrt_f64 v[0:1], -|v[2:3]| clamp")

        # Invalid destination type (SGPR)
        with self.assertRaises(TypeError):
            VSqrtF64Instruction(dst=RegPair(0, RegType.SGPR), src=src)

    def test_structural_equality(self) -> None:
        i1 = VSqrtF64Instruction(
            RegPair(0, RegType.VGPR),
            VOP3Operand(RegPair(2, RegType.VGPR)),
        )
        i2 = VSqrtF64Instruction(
            RegPair(0, RegType.VGPR),
            VOP3Operand(RegPair(2, RegType.VGPR)),
        )
        self.assertEqual(i1, i2)


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarInterpreter()
        self.state = ScalarMachineState()

    def test_scalar_vgpr_source(self) -> None:
        self.state.set_vgpr64(2, f64_to_bits(25.0))
        instr = VSqrtF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src=VOP3Operand(RegPair(2, RegType.VGPR)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f64(res_bits), 5.0)
        self.assertEqual(self.state.get_vgpr64(0), res_bits)

    def test_scalar_sgpr_source(self) -> None:
        self.state.set_sgpr64(4, f64_to_bits(144.0))
        instr = VSqrtF64Instruction(
            dst=RegPair(6, RegType.VGPR),
            src=VOP3Operand(RegPair(4, RegType.SGPR)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f64(res_bits), 12.0)
        self.assertEqual(self.state.get_vgpr64(6), res_bits)

    def test_scalar_literal_source(self) -> None:
        instr = VSqrtF64Instruction(
            dst=RegPair(8, RegType.VGPR),
            src=VOP3Operand(Literal64.from_float(0.0625)),
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(bits_to_f64(res_bits), 0.25)
        self.assertEqual(self.state.get_vgpr64(8), res_bits)

    def test_scalar_with_modifiers_and_clamp(self) -> None:
        self.state.set_vgpr64(0, f64_to_bits(-16.0))
        # |-16.0| = 16.0 -> sqrt(16.0) = 4.0 -> clamp -> 1.0
        instr = VSqrtF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR), abs_mod=True),
            clamp=True,
        )
        res_bits = self.interp.step(instr, self.state)
        self.assertEqual(res_bits, ONE_F64)
        self.assertEqual(self.state.get_vgpr64(4), ONE_F64)

    def test_scalar_determinism(self) -> None:
        runs = []
        for _ in range(50):
            st = ScalarMachineState()
            st.set_vgpr64(0, 0x400921FB54442D18)  # pi
            instr = VSqrtF64Instruction(
                dst=RegPair(2, RegType.VGPR),
                src=VOP3Operand(RegPair(0, RegType.VGPR)),
            )
            runs.append(self.interp.step(instr, st))
        self.assertTrue(all(r == runs[0] for r in runs))


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD execution, lane divergence, and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.interp = SIMDInterpreter()

    def test_wave32_full_exec(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            val = float(lane * lane)
            state.set_vgpr64(lane, 0, f64_to_bits(val))
        instr = VSqrtF64Instruction(
            dst=RegPair(2, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            self.assertIsNotNone(results[lane])
            self.assertEqual(bits_to_f64(state.get_vgpr64(lane, 2)), float(lane))

    def test_wave32_divergent_masking(self) -> None:
        # Alternating mask: even lanes active (0x55555555)
        sentinel = 0xDEAD_BEEF_CAFE_BABE
        state = SIMDState(num_lanes=32, exec_mask=0x5555_5555)
        for lane in range(32):
            state.set_vgpr64(lane, 0, f64_to_bits(float(lane * lane)))
            state.set_vgpr64(lane, 4, sentinel)

        instr = VSqrtF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(32):
            if lane % 2 == 0:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f64(state.get_vgpr64(lane, 4)), float(lane))
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr64(lane, 4), sentinel)

    def test_wave64_full_exec(self) -> None:
        state = SIMDState(num_lanes=64, exec_mask=0xFFFF_FFFF_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr64(lane, 0, f64_to_bits(float(lane * 4)))
        instr = VSqrtF64Instruction(
            dst=RegPair(2, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(64):
            self.assertEqual(bits_to_f64(state.get_vgpr64(lane, 2)), math.sqrt(float(lane * 4)))

    def test_wave64_half_disabled(self) -> None:
        # Lower 32 lanes active, upper 32 lanes inactive
        sentinel = 0x1122_3344_5566_7788
        state = SIMDState(num_lanes=64, exec_mask=0x0000_0000_FFFF_FFFF)
        for lane in range(64):
            state.set_vgpr64(lane, 0, f64_to_bits(64.0))
            state.set_vgpr64(lane, 4, sentinel)

        instr = VSqrtF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        results = self.interp.step(instr, state)
        for lane in range(64):
            if lane < 32:
                self.assertIsNotNone(results[lane])
                self.assertEqual(bits_to_f64(state.get_vgpr64(lane, 4)), 8.0)
            else:
                self.assertIsNone(results[lane])
                self.assertEqual(state.get_vgpr64(lane, 4), sentinel)

    def test_wave_exec_all_zero(self) -> None:
        sentinel = 0xAA55_AA55_AA55_AA55
        state = SIMDState(num_lanes=32, exec_mask=0x0000_0000)
        for lane in range(32):
            state.set_vgpr64(lane, 0, sentinel)
        instr = VSqrtF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src=VOP3Operand(Literal64.from_float(100.0)),
        )
        results = self.interp.step(instr, state)
        self.assertTrue(all(r is None for r in results))
        for lane in range(32):
            self.assertEqual(state.get_vgpr64(lane, 0), sentinel)

    def test_simd_sgpr_uniform_broadcast(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        state.set_sgpr64(10, f64_to_bits(10000.0))

        instr = VSqrtF64Instruction(
            dst=RegPair(2, RegType.VGPR),
            src=VOP3Operand(RegPair(10, RegType.SGPR)),
        )
        self.interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(bits_to_f64(state.get_vgpr64(lane, 2)), 100.0)


class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.scalar_interp = ScalarInterpreter()
        self.simd_interp = SIMDInterpreter()

    def test_dst_alias_src_inplace(self) -> None:
        state = ScalarMachineState()
        state.set_vgpr64(0, f64_to_bits(16.0))
        state.set_vgpr64(2, f64_to_bits(999.0))
        instr = VSqrtF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f64(state.get_vgpr64(0)), 4.0)
        # Unrelated register untouched
        self.assertEqual(bits_to_f64(state.get_vgpr64(2)), 999.0)

    def test_inplace_iterative_convergence(self) -> None:
        # Applying sqrt repeatedly converges towards 1.0
        state = ScalarMachineState()
        state.set_vgpr64(0, f64_to_bits(65536.0))
        instr = VSqrtF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        # 65536 -> 256 -> 16 -> 4 -> 2
        for expected in [256.0, 16.0, 4.0, 2.0]:
            self.scalar_interp.step(instr, state)
            self.assertEqual(bits_to_f64(state.get_vgpr64(0)), expected)

    def test_partial_overlap_misaligned_alias(self) -> None:
        # Destination v[1:2] partially overlaps source v[0:1] (shares v[1])
        state = ScalarMachineState()
        bits_src = f64_to_bits(16.0)
        state.set_vgpr64(0, bits_src)
        lo0, hi0 = f64_bits_to_words(bits_src)

        instr = VSqrtF64Instruction(
            dst=RegPair(1, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        res_bits = self.scalar_interp.step(instr, state)
        self.assertEqual(bits_to_f64(res_bits), 4.0)
        self.assertEqual(state.get_vgpr64(1), res_bits)
        # v[0] should remain unchanged lo0
        self.assertEqual(state.vgprs[0], lo0)

    def test_simd_multi_lane_aliasing(self) -> None:
        state = SIMDState(num_lanes=32, exec_mask=0xFFFF_FFFF)
        for lane in range(32):
            state.set_vgpr64(lane, 0, f64_to_bits(float(lane * lane)))

        instr = VSqrtF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src=VOP3Operand(RegPair(0, RegType.VGPR)),
        )
        self.simd_interp.step(instr, state)
        for lane in range(32):
            self.assertEqual(bits_to_f64(state.get_vgpr64(lane, 0)), float(lane))


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for status flags, condition codes, and carry/borrow generation."""

    def test_exact_finite_flags(self) -> None:
        s = f64_to_bits(4.0)
        res = oracle_v_sqrt_f64(s)
        flags = evaluate_v_sqrt_f64_flags(s, res)
        self.assertFalse(flags.invalid)
        self.assertFalse(flags.overflow)
        self.assertFalse(flags.underflow)
        self.assertFalse(flags.inexact)
        self.assertFalse(flags.zero)
        self.assertFalse(flags.negative)
        self.assertFalse(flags.denormal)
        self.assertFalse(flags.infinite)
        self.assertFalse(flags.nan)

    def test_inexact_flag(self) -> None:
        s = f64_to_bits(2.0)
        res = oracle_v_sqrt_f64(s)
        flags = evaluate_v_sqrt_f64_flags(s, res)
        self.assertTrue(flags.inexact)
        self.assertFalse(flags.invalid)

    def test_invalid_operation_flag(self) -> None:
        # Negative radicand
        s_neg = f64_to_bits(-1.0)
        res = oracle_v_sqrt_f64(s_neg)
        flags = evaluate_v_sqrt_f64_flags(s_neg, res)
        self.assertTrue(flags.invalid)
        self.assertTrue(flags.nan)

        # Signaling NaN operand
        snan = 0x7FF0_0000_0000_0001
        res_snan = oracle_v_sqrt_f64(snan)
        flags_snan = evaluate_v_sqrt_f64_flags(snan, res_snan)
        self.assertTrue(flags_snan.invalid)
        self.assertTrue(flags_snan.nan)

        # Negative infinity
        res_ninf = oracle_v_sqrt_f64(NEG_INF)
        flags_ninf = evaluate_v_sqrt_f64_flags(NEG_INF, res_ninf)
        self.assertTrue(flags_ninf.invalid)
        self.assertTrue(flags_ninf.nan)

    def test_zero_and_negative_flags(self) -> None:
        # +0.0
        res_pz = oracle_v_sqrt_f64(POS_ZERO)
        flags_pz = evaluate_v_sqrt_f64_flags(POS_ZERO, res_pz)
        self.assertTrue(flags_pz.zero)
        self.assertFalse(flags_pz.negative)

        # -0.0
        res_nz = oracle_v_sqrt_f64(NEG_ZERO)
        flags_nz = evaluate_v_sqrt_f64_flags(NEG_ZERO, res_nz)
        self.assertTrue(flags_nz.zero)
        self.assertTrue(flags_nz.negative)

    def test_infinite_flag(self) -> None:
        res = oracle_v_sqrt_f64(POS_INF)
        flags = evaluate_v_sqrt_f64_flags(POS_INF, res)
        self.assertTrue(flags.infinite)
        self.assertFalse(flags.invalid)

    def test_sqrt_borrow_generation(self) -> None:
        # Perfect square: 4.0
        s4 = f64_to_bits(4.0)
        res2 = oracle_v_sqrt_f64(s4)
        rem, borrow = compute_f64_sqrt_borrow(s4, res2)
        self.assertEqual(borrow, 0)


class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_sqrt_transition_to_normal(self) -> None:
        # Smallest positive subnormal: 2^-1074 (bits: 0x1)
        # sqrt(2^-1074) = 2^-537, which is a normal double!
        sub_min = 0x0000_0000_0000_0001
        res = oracle_v_sqrt_f64(sub_min)
        self.assertFalse(is_subnormal(res))
        self.assertFalse(is_zero(res))
        # Exponent of 2^-537 in IEEE 754 (bias 1023): 1023 - 537 = 486
        sign, exp, frac = split_f64_bits(res)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 486)
        self.assertEqual(frac, 0)

    def test_largest_subnormal_sqrt(self) -> None:
        # Largest subnormal: 0x000F_FFFF_FFFF_FFFF = (1 - 2^-52) * 2^-1022
        max_sub = 0x000F_FFFF_FFFF_FFFF
        res = oracle_v_sqrt_f64(max_sub)
        self.assertFalse(is_subnormal(res))
        # Exponent is 511 (just below 2^-511)
        sign, exp, _ = split_f64_bits(res)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 511)

    def test_nan_signaling_to_quiet_conversion(self) -> None:
        snan = 0x7FF0_0000_0000_0005  # fraction non-zero, quiet bit 51 is 0
        self.assertTrue(is_snan(snan))
        res = oracle_v_sqrt_f64(snan)
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)

    def test_nan_payload_preservation(self) -> None:
        payload = 0x0008_DEAD_BEEF_CAFE
        qnan = EXP_MASK | payload
        res = oracle_v_sqrt_f64(qnan)
        self.assertEqual(res, qnan)

    def test_nan_with_modifiers(self) -> None:
        qnan = 0x7FF8_0000_0000_0001
        # abs clears sign bit
        res_abs = oracle_v_sqrt_f64(qnan | SIGN_MASK, abs_mod=True)
        self.assertEqual(res_abs & SIGN_MASK, 0)
        self.assertTrue(is_qnan(res_abs))

        # neg toggles sign bit
        res_neg = oracle_v_sqrt_f64(qnan, neg_mod=True)
        self.assertEqual(res_neg & SIGN_MASK, SIGN_MASK)

    def test_nan_with_clamp(self) -> None:
        qnan = CANONICAL_QNAN
        res = oracle_v_sqrt_f64(qnan, clamp=True)
        self.assertEqual(res, POS_ZERO)

        snan = 0x7FF0_0000_0000_0001
        res_snan = oracle_v_sqrt_f64(snan, clamp=True)
        self.assertEqual(res_snan, POS_ZERO)

    def test_infinity_arithmetic_boundaries(self) -> None:
        self.assertEqual(oracle_v_sqrt_f64(POS_INF), POS_INF)
        self.assertEqual(oracle_v_sqrt_f64(NEG_INF), CANONICAL_QNAN)
        self.assertEqual(oracle_v_sqrt_f64(POS_INF, clamp=True), ONE_F64)
        self.assertEqual(oracle_v_sqrt_f64(NEG_INF, clamp=True), POS_ZERO)

    def test_maximum_finite_f64(self) -> None:
        max_f64 = 0x7FEF_FFFF_FFFF_FFFF
        res = oracle_v_sqrt_f64(max_f64)
        f_res = bits_to_f64(res)
        self.assertAlmostEqual(f_res, 1.3407807929942597e+154, delta=1e140)

    def test_smallest_normal_f64(self) -> None:
        # 2^-1022 (0x0010_0000_0000_0000) -> sqrt is 2^-511 (0x2000_0000_0000_0000)
        min_norm = 0x0010_0000_0000_0000
        res = oracle_v_sqrt_f64(min_norm)
        expected = 0x2000_0000_0000_0000
        self.assertEqual(res, expected)


class TestNegativeControlsAndMutationDetection(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_register_indices(self) -> None:
        with self.assertRaises(ValueError):
            RegPair(255, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(-1, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(300, RegType.SGPR)

    def test_syntax_rejection_invalid_destination(self) -> None:
        with self.assertRaises(TypeError):
            VSqrtF64Instruction(
                dst=RegPair(0, RegType.SGPR),
                src=VOP3Operand(RegPair(2, RegType.VGPR)),
            )

    def test_syntax_rejection_invalid_lane_count(self) -> None:
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=48)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=128)

    def test_negative_control_divergence(self) -> None:
        s = f64_to_bits(16.0)
        res = oracle_v_sqrt_f64(s)
        self.assertNotEqual(res, f64_to_bits(5.0))
        self.assertNotEqual(res, f64_to_bits(16.0))

    def test_mutation_detection_identity_mutant(self) -> None:
        def mutant_oracle(src: int) -> int:
            return src

        s = f64_to_bits(16.0)
        golden = oracle_v_sqrt_f64(s)
        mutant = mutant_oracle(s)
        self.assertNotEqual(golden, mutant)

    def test_mutation_detection_snan_unquieted(self) -> None:
        def mutant_oracle(src: int) -> int:
            return src

        snan = 0x7FF0_0000_0000_0001
        golden = oracle_v_sqrt_f64(snan)
        mutant = mutant_oracle(snan)
        self.assertTrue(is_qnan(golden))
        self.assertFalse(is_qnan(mutant))

    def test_mutation_detection_neg_zero_to_pos_zero(self) -> None:
        def mutant_oracle(src: int) -> int:
            return POS_ZERO if is_zero(src) else oracle_v_sqrt_f64(src)

        golden = oracle_v_sqrt_f64(NEG_ZERO)
        mutant = mutant_oracle(NEG_ZERO)
        self.assertEqual(golden, NEG_ZERO)
        self.assertNotEqual(golden, mutant)

    def test_mutation_detection_unclamped_mutant(self) -> None:
        def mutant_oracle(src: int, clamp: bool = False) -> int:
            return oracle_v_sqrt_f64(src, clamp=False)

        s = f64_to_bits(4.0)
        golden = oracle_v_sqrt_f64(s, clamp=True)
        mutant = mutant_oracle(s, clamp=True)
        self.assertEqual(golden, ONE_F64)
        self.assertNotEqual(golden, mutant)

    def test_mutation_detection_ignoring_abs_modifier(self) -> None:
        def mutant_oracle(src: int, abs_mod: bool = False) -> int:
            return oracle_v_sqrt_f64(src, abs_mod=False)

        val = f64_to_bits(-9.0)
        golden = oracle_v_sqrt_f64(val, abs_mod=True)
        mutant = mutant_oracle(val, abs_mod=True)
        self.assertEqual(bits_to_f64(golden), 3.0)
        self.assertEqual(mutant, CANONICAL_QNAN)
        self.assertNotEqual(golden, mutant)


if __name__ == "__main__":
    unittest.main()
