"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_rsq_f64.

Models IEEE 754-2008 / IEEE 754-2019 binary64 double-precision reciprocal square root
and AMD RDNA2 / GFX10 Vector Arithmetic Logic Unit (VALU) architectural semantics:
  D.d = 1.0 / sqrt(S0.d)

RDNA2 Double-Precision Floating-Point ISA Rules:
  1. Register Pair Allocation:
     Double-precision 64-bit values occupy aligned pairs of 32-bit vector registers:
     v[d:d+1] where low 32 bits reside in v[d] and high 32 bits in v[d+1].
  2. Reciprocal Square Root Operation:
     Computes the IEEE 754 reciprocal square root of S0.
     - Positive finite numbers: 1.0 / sqrt(S0). Exact powers of 4 and 2 yield exact results:
       rsq(1.0) = 1.0, rsq(4.0) = 0.5, rsq(16.0) = 0.25, rsq(0.25) = 2.0, rsq(0.0625) = 4.0.
     - Zeros: rsq(+0.0) = +Infinity, rsq(-0.0) = +Infinity.
     - Positive Infinity: rsq(+Infinity) = +0.0.
     - Negative numbers & -Infinity: square root of negative is invalid, producing Quiet NaN.
     - NaNs: Quiet NaN propagates; Signaling NaN is quieted.
  3. Modifiers (VOP3 encoding):
     - Input modifiers: abs (|S0|) clears sign bit 63; neg (-S0) toggles sign bit 63.
     - Output modifier: clamp saturates result to [0.0, 1.0]. Clamping negative/NaN yields +0.0.
  4. Subnormals & Flush-to-Zero:
     Under IEEE mode, subnormal double values are computed with full precision.
     Under flush-to-zero (FTZ), subnormal inputs flush to signed zero, yielding +Infinity.

Standard library Python only; zero third-party dependencies; zero game data.
Python 3.9+ compatible (from __future__ import annotations).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
import math
import struct
import unittest
from typing import List, Optional, Tuple, Union

MASK32: int = 0xFFFF_FFFF
MASK64: int = 0xFFFF_FFFF_FFFF_FFFF
SIGN_MASK: int = 0x8000_0000_0000_0000
EXP_MASK: int = 0x7FF0_0000_0000_0000
FRAC_MASK: int = 0x000F_FFFF_FFFF_FFFF
QNAN_BIT: int = 0x0008_0000_0000_0000
CANONICAL_QNAN: int = 0x7FF8_0000_0000_0000

# Canonical IEEE 754 binary64 bit patterns
POS_ZERO: int = 0x0000_0000_0000_0000
NEG_ZERO: int = 0x8000_0000_0000_0000
POS_INF: int = 0x7FF0_0000_0000_0000
NEG_INF: int = 0xFFF0_0000_0000_0000

ONE_F64: int = 0x3FF0_0000_0000_0000          # 1.0
FOUR_F64: int = 0x4010_0000_0000_0000         # 4.0
SIXTEEN_F64: int = 0x4030_0000_0000_0000      # 16.0
SIXTY_FOUR_F64: int = 0x4050_0000_0000_0000   # 64.0
QUARTER_F64: int = 0x3FD0_0000_0000_0000      # 0.25
SIXTEENTH_F64: int = 0x3FB0_0000_0000_0000    # 0.0625
HALF_F64: int = 0x3FE0_0000_0000_0000         # 0.5
TWO_F64: int = 0x4000_0000_0000_0000          # 2.0
THREE_F64: int = 0x4008_0000_0000_0000        # 3.0
NINE_F64: int = 0x4022_0000_0000_0000         # 9.0
HUNDRED_F64: int = 0x4059_0000_0000_0000      # 100.0

NEG_ONE_F64: int = 0xBFF0_0000_0000_0000      # -1.0
NEG_FOUR_F64: int = 0xC010_0000_0000_0000     # -4.0
NEG_QUARTER_F64: int = 0xBFD0_0000_0000_0000  # -0.25

DBL_MIN: int = 0x0010_0000_0000_0000          # Smallest positive normal: 2^-1022
DBL_MAX: int = 0x7FEF_FFFF_FFFF_FFFF          # Largest finite positive float
SUBNORM_MIN: int = 0x0000_0000_0000_0001      # Smallest positive subnormal: 2^-1074
SUBNORM_MAX: int = 0x000F_FFFF_FFFF_FFFF      # Largest positive subnormal


def f64_to_bits(val: float) -> int:
    """Convert Python float (IEEE 754 double) to 64-bit unsigned integer bits."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def bits_to_f64(bits: int) -> float:
    """Convert 64-bit unsigned integer bits to Python float (IEEE 754 double)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def split_f64_bits(bits: int) -> Tuple[int, int, int]:
    """Unpack bits into (sign, exponent, fraction) components."""
    sign = (bits & SIGN_MASK) >> 63
    exponent = (bits & EXP_MASK) >> 52
    fraction = bits & FRAC_MASK
    return sign, exponent, fraction


def words_to_f64_bits(lo: int, hi: int) -> int:
    """Combine two 32-bit VGPR words (lo, hi) into a 64-bit integer representation."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def f64_bits_to_words(bits: int) -> Tuple[int, int]:
    """Split a 64-bit integer into two 32-bit VGPR words (lo, hi)."""
    lo = bits & MASK32
    hi = (bits >> 32) & MASK32
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
    res = bits & MASK64
    if abs_mod:
        res &= ~SIGN_MASK
    if neg_mod:
        res ^= SIGN_MASK
    return res


def oracle_v_rsq_f64(
    src_bits: int,
    abs_src: bool = False,
    neg_src: bool = False,
    clamp: bool = False,
    flush_subnormals: bool = False,
) -> int:
    """Reference mathematical oracle for RDNA2 v_rsq_f64.

    Computes D = 1.0 / sqrt(src) according to IEEE 754-2008 / RDNA2 VALU rules:
      - abs_src / neg_src input modifiers applied before root calculation.
      - Flush-to-zero flushes subnormal input to signed zero.
      - NaN input produces Quiet NaN (or +0.0 if clamp is enabled).
      - Negative non-zero and -Infinity inputs produce Canonical Quiet NaN (or +0.0 if clamp is enabled).
      - +0.0 and -0.0 produce +Infinity (clamped to 1.0 if clamp is enabled).
      - +Infinity produces +0.0.
      - Finite positive numbers produce 1.0 / sqrt(x), clamped to [0.0, 1.0] if clamp is enabled.
    """
    s = apply_f64_modifiers(src_bits, abs_src, neg_src)

    # 1. Subnormal flush-to-zero mode
    if flush_subnormals and is_subnormal(s):
        s = NEG_ZERO if (s & SIGN_MASK) else POS_ZERO

    # 2. NaN inputs propagate quieted NaN
    if is_nan(s):
        if clamp:
            return POS_ZERO
        return s | QNAN_BIT

    # 3. Negative non-zero or negative infinity yields Quiet NaN
    if (s & SIGN_MASK) and not is_zero(s):
        if clamp:
            return POS_ZERO
        return CANONICAL_QNAN

    # 4. Zero inputs (+0.0, -0.0) produce +Infinity
    if is_zero(s):
        if clamp:
            return ONE_F64
        return POS_INF

    # 5. Positive Infinity yields +0.0
    if s == POS_INF:
        return POS_ZERO

    # 6. Finite positive calculation
    val = bits_to_f64(s)
    res_val = 1.0 / math.sqrt(val)

    if clamp:
        if res_val < 0.0:
            res_val = 0.0
        elif res_val > 1.0:
            res_val = 1.0

    return f64_to_bits(res_val)


class TestF64ModelAndOracle(unittest.TestCase):
    """Unit tests for IEEE 754 binary64 bit functions and the reference v_rsq_f64 oracle."""

    def test_exact_powers_of_four_and_two(self) -> None:
        """Exact square roots and reciprocals must be bit-exact in binary64."""
        self.assertEqual(oracle_v_rsq_f64(ONE_F64), ONE_F64)
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64), HALF_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTEEN_F64), QUARTER_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTY_FOUR_F64), f64_to_bits(0.125))
        self.assertEqual(oracle_v_rsq_f64(QUARTER_F64), TWO_F64)
        self.assertEqual(oracle_v_rsq_f64(SIXTEENTH_F64), FOUR_F64)

    def test_positive_finite_non_integers(self) -> None:
        """Test reciprocal square root for general positive floating point values."""
        res_nine = bits_to_f64(oracle_v_rsq_f64(NINE_F64))
        self.assertAlmostEqual(res_nine, 1.0 / 3.0, places=15)

        res_hundred = bits_to_f64(oracle_v_rsq_f64(HUNDRED_F64))
        self.assertAlmostEqual(res_hundred, 0.1, places=15)

        res_two = bits_to_f64(oracle_v_rsq_f64(TWO_F64))
        self.assertAlmostEqual(res_two, 1.0 / math.sqrt(2.0), places=15)

    def test_zero_inputs_produce_positive_infinity(self) -> None:
        """Both +0.0 and -0.0 yield +Infinity in AMD RDNA2 RSQ semantics."""
        self.assertEqual(oracle_v_rsq_f64(POS_ZERO), POS_INF)
        self.assertEqual(oracle_v_rsq_f64(NEG_ZERO), POS_INF)

    def test_positive_infinity_yields_positive_zero(self) -> None:
        """1.0 / sqrt(+Inf) yields +0.0."""
        self.assertEqual(oracle_v_rsq_f64(POS_INF), POS_ZERO)

    def test_negative_inputs_produce_canonical_qnan(self) -> None:
        """Square root of negative real numbers produces Quiet NaN."""
        self.assertEqual(oracle_v_rsq_f64(NEG_ONE_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_QUARTER_F64), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(NEG_INF), CANONICAL_QNAN)

    def test_nan_inputs_propagate_quieted(self) -> None:
        """NaN operands propagate quieted NaN."""
        qnan_in = CANONICAL_QNAN | 0x1234
        snan_in = POS_INF | 0x0001_0000  # Signaling NaN
        self.assertTrue(is_qnan(oracle_v_rsq_f64(qnan_in)))
        self.assertTrue(is_qnan(oracle_v_rsq_f64(snan_in)))

    def test_input_modifiers_abs_and_neg(self) -> None:
        """Input modifiers abs and neg transform operand before root calculation."""
        # abs(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, abs_src=True), HALF_F64)
        # neg(4.0) -> -4.0 -> rsq gives NaN
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, neg_src=True), CANONICAL_QNAN)
        # neg(-4.0) -> +4.0 -> rsq gives 0.5
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, neg_src=True), HALF_F64)
        # abs and neg together on positive: neg(abs(4.0)) -> -4.0 -> NaN
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, abs_src=True, neg_src=True), CANONICAL_QNAN)

    def test_clamp_modifier(self) -> None:
        """Clamp modifier saturates result to [0.0, 1.0]."""
        # 0.25 -> 2.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(QUARTER_F64, clamp=True), ONE_F64)
        # 4.0 -> 0.5 -> within [0.0, 1.0] -> unchanged
        self.assertEqual(oracle_v_rsq_f64(FOUR_F64, clamp=True), HALF_F64)
        # +0.0 -> +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(POS_ZERO, clamp=True), ONE_F64)
        # Negative -> NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f64(NEG_FOUR_F64, clamp=True), POS_ZERO)
        # NaN -> clamped to +0.0
        self.assertEqual(oracle_v_rsq_f64(CANONICAL_QNAN, clamp=True), POS_ZERO)

    def test_subnormal_handling(self) -> None:
        """Subnormal precision preservation in IEEE mode and flush in FTZ mode."""
        sub = SUBNORM_MAX
        res_ieee = oracle_v_rsq_f64(sub, flush_subnormals=False)
        self.assertFalse(is_nan(res_ieee))
        self.assertFalse(is_inf(res_ieee))
        self.assertGreater(bits_to_f64(res_ieee), 1.0)

        # In FTZ mode, subnormal flushes to +0.0, so rsq gives +Infinity
        res_ftz = oracle_v_rsq_f64(sub, flush_subnormals=True)
        self.assertEqual(res_ftz, POS_INF)

    def test_word_conversion_helpers(self) -> None:
        """Splitting and joining 32-bit dwords must roundtrip 64-bit values accurately."""
        lo, hi = f64_bits_to_words(FOUR_F64)
        rejoined = words_to_f64_bits(lo, hi)
        self.assertEqual(rejoined, FOUR_F64)

        sign, exp, frac = split_f64_bits(FOUR_F64)
        self.assertEqual(sign, 0)
        self.assertEqual(exp, 1025)
        self.assertEqual(frac, 0)


# ============================================================================
# Section 2: Symbolic AST Opcode Definitions and Typed Expressions
# ============================================================================

class RegType(Enum):
    """Architectural register file identifier."""
    VGPR = auto()
    SGPR = auto()
    LITERAL = auto()


@dataclass(frozen=True)
class RegPair:
    """An aligned 64-bit pair of 32-bit registers (v[d:d+1] or s[s:s+1])."""
    index: int
    reg_type: RegType = RegType.VGPR

    def __post_init__(self) -> None:
        if self.reg_type == RegType.VGPR:
            if not (0 <= self.index <= 254):
                raise ValueError(f"VGPR index {self.index} out of valid range [0, 254] for a 64-bit pair")
        elif self.reg_type == RegType.SGPR:
            if not (0 <= self.index <= 104):
                raise ValueError(f"SGPR index {self.index} out of valid range [0, 104] for a 64-bit pair")
        else:
            raise ValueError(f"Invalid register type for RegPair: {self.reg_type}")

    @property
    def low(self) -> int:
        """Low 32-bit dword register index."""
        return self.index

    @property
    def high(self) -> int:
        """High 32-bit dword register index."""
        return self.index + 1

    def __str__(self) -> str:
        prefix = "v" if self.reg_type == RegType.VGPR else "s"
        return f"{prefix}[{self.index}:{self.index + 1}]"


@dataclass(frozen=True)
class Literal64:
    """A 64-bit immediate constant bit pattern."""
    bits: int

    def __post_init__(self) -> None:
        if not (0 <= self.bits <= MASK64):
            raise ValueError(f"Literal64 bits {hex(self.bits)} out of range [0, 2^64-1]")

    @classmethod
    def from_float(cls, val: float) -> Literal64:
        return cls(f64_to_bits(val))

    def to_float(self) -> float:
        return bits_to_f64(self.bits)

    def __str__(self) -> str:
        return f"0x{self.bits:016x}"


@dataclass(frozen=True)
class VOP3Operand:
    """A VOP3 source operand with optional input modifiers (abs and neg)."""
    reg_pair: Optional[RegPair] = None
    literal: Optional[Literal64] = None
    abs_mod: bool = False
    neg_mod: bool = False

    def __post_init__(self) -> None:
        if (self.reg_pair is None) == (self.literal is None):
            raise ValueError("VOP3Operand must specify exactly one of reg_pair or literal")

    @classmethod
    def from_vgpr(cls, index: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP3Operand:
        return cls(reg_pair=RegPair(index, RegType.VGPR), abs_mod=abs_mod, neg_mod=neg_mod)

    @classmethod
    def from_sgpr(cls, index: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP3Operand:
        return cls(reg_pair=RegPair(index, RegType.SGPR), abs_mod=abs_mod, neg_mod=neg_mod)

    @classmethod
    def from_lit(cls, bits: int, abs_mod: bool = False, neg_mod: bool = False) -> VOP3Operand:
        return cls(literal=Literal64(bits), abs_mod=abs_mod, neg_mod=neg_mod)

    def __str__(self) -> str:
        core = str(self.reg_pair) if self.reg_pair is not None else str(self.literal)
        if self.abs_mod:
            core = f"|{core}|"
        if self.neg_mod:
            core = f"-{core}"
        return core


@dataclass(frozen=True)
class VRSQF64Instruction:
    """Symbolic AST node for RDNA2 VALU v_rsq_f64 instruction."""
    dst: RegPair
    src0: VOP3Operand
    clamp: bool = False
    flush_denorms: bool = False

    def __post_init__(self) -> None:
        if self.dst.reg_type != RegType.VGPR:
            raise ValueError(f"Destination must be a VGPR pair, got {self.dst.reg_type}")

    def disasm(self) -> str:
        """Render GCN/RDNA assembly representation."""
        res = f"v_rsq_f64 {self.dst}, {self.src0}"
        if self.clamp:
            res += " clamp"
        return res


class TestSymbolicAST(unittest.TestCase):
    """Unit tests for symbolic AST opcode definitions and typed expressions."""

    def test_reg_pair_creation_and_bounds(self) -> None:
        """Test register pair construction, indexing, and boundary validation."""
        v = RegPair(0, RegType.VGPR)
        self.assertEqual(v.low, 0)
        self.assertEqual(v.high, 1)
        self.assertEqual(str(v), "v[0:1]")

        s = RegPair(10, RegType.SGPR)
        self.assertEqual(s.low, 10)
        self.assertEqual(s.high, 11)
        self.assertEqual(str(s), "s[10:11]")

        # Boundary checks
        RegPair(254, RegType.VGPR)
        RegPair(104, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegPair(255, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(105, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegPair(-1, RegType.VGPR)

    def test_literal64_creation(self) -> None:
        """Test 64-bit immediate literal operand."""
        lit = Literal64(0x3FF0_0000_0000_0000)
        self.assertEqual(lit.to_float(), 1.0)
        self.assertEqual(str(lit), "0x3ff0000000000000")

        lit2 = Literal64.from_float(4.0)
        self.assertEqual(lit2.bits, FOUR_F64)

        with self.assertRaises(ValueError):
            Literal64(-1)
        with self.assertRaises(ValueError):
            Literal64(1 << 64)

    def test_vop3_operand_formatting(self) -> None:
        """Test formatting of VOP3 operands with input modifiers."""
        op_plain = VOP3Operand.from_vgpr(4)
        self.assertEqual(str(op_plain), "v[4:5]")

        op_abs = VOP3Operand.from_vgpr(4, abs_mod=True)
        self.assertEqual(str(op_abs), "|v[4:5]|")

        op_neg = VOP3Operand.from_vgpr(4, neg_mod=True)
        self.assertEqual(str(op_neg), "-v[4:5]")

        op_neg_abs = VOP3Operand.from_vgpr(4, abs_mod=True, neg_mod=True)
        self.assertEqual(str(op_neg_abs), "-|v[4:5]|")

        op_sgpr = VOP3Operand.from_sgpr(0)
        self.assertEqual(str(op_sgpr), "s[0:1]")

        op_lit = VOP3Operand.from_lit(ONE_F64)
        self.assertEqual(str(op_lit), "0x3ff0000000000000")

        with self.assertRaises(ValueError):
            VOP3Operand(reg_pair=None, literal=None)

    def test_instruction_disasm(self) -> None:
        """Test disassembly output of VRSQF64Instruction."""
        instr_basic = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        self.assertEqual(instr_basic.disasm(), "v_rsq_f64 v[0:1], v[2:3]")

        instr_mod = VRSQF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(6, abs_mod=True, neg_mod=True),
            clamp=True,
        )
        self.assertEqual(instr_mod.disasm(), "v_rsq_f64 v[4:5], -|v[6:7]| clamp")

        # Destination must be VGPR
        with self.assertRaises(ValueError):
            VRSQF64Instruction(
                dst=RegPair(0, RegType.SGPR),
                src0=VOP3Operand.from_vgpr(2),
            )


# ============================================================================
# Section 3: Deterministic Scalar Instruction Interpreter
# ============================================================================

@dataclass
class ScalarMachineState:
    """Architectural register state of an AMD RDNA2 compute unit single thread."""
    vgprs: List[int] = field(default_factory=lambda: [0] * 256)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)
    pc: int = 0
    cycle_count: int = 0

    def read_vgpr(self, index: int) -> int:
        return self.vgprs[index] & MASK32

    def write_vgpr(self, index: int, val: int) -> None:
        self.vgprs[index] = val & MASK32

    def read_sgpr(self, index: int) -> int:
        return self.sgprs[index] & MASK32

    def write_sgpr(self, index: int, val: int) -> None:
        self.sgprs[index] = val & MASK32

    def read_vgpr_pair(self, index: int) -> int:
        lo = self.read_vgpr(index)
        hi = self.read_vgpr(index + 1)
        return words_to_f64_bits(lo, hi)

    def write_vgpr_pair(self, index: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.write_vgpr(index, lo)
        self.write_vgpr(index + 1, hi)

    def read_sgpr_pair(self, index: int) -> int:
        lo = self.read_sgpr(index)
        hi = self.read_sgpr(index + 1)
        return words_to_f64_bits(lo, hi)

    def write_sgpr_pair(self, index: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.write_sgpr(index, lo)
        self.write_sgpr(index + 1, hi)

    def read_operand_raw_bits(self, op: VOP3Operand) -> int:
        if op.literal is not None:
            return op.literal.bits
        assert op.reg_pair is not None
        if op.reg_pair.reg_type == RegType.VGPR:
            return self.read_vgpr_pair(op.reg_pair.index)
        elif op.reg_pair.reg_type == RegType.SGPR:
            return self.read_sgpr_pair(op.reg_pair.index)
        raise ValueError(f"Unknown register type {op.reg_pair.reg_type}")


class ScalarInterpreter:
    """Deterministic scalar instruction interpreter for RDNA2 VALU instructions."""

    def __init__(self, state: Optional[ScalarMachineState] = None) -> None:
        self.state = state if state is not None else ScalarMachineState()

    def execute(self, instr: VRSQF64Instruction) -> int:
        """Step a single VRSQF64Instruction against scalar state."""
        raw_bits = self.state.read_operand_raw_bits(instr.src0)
        res_bits = oracle_v_rsq_f64(
            raw_bits,
            abs_src=instr.src0.abs_mod,
            neg_src=instr.src0.neg_mod,
            clamp=instr.clamp,
            flush_subnormals=instr.flush_denorms,
        )
        self.state.write_vgpr_pair(instr.dst.index, res_bits)
        self.state.pc += 4
        self.state.cycle_count += 1
        return res_bits


class TestScalarInterpreter(unittest.TestCase):
    """Unit tests for deterministic scalar instruction execution."""

    def test_vgpr_to_vgpr_execution(self) -> None:
        """Execute v_rsq_f64 reading from VGPR pair and writing to VGPR pair."""
        state = ScalarMachineState()
        # Set v[2:3] = 4.0
        state.write_vgpr_pair(2, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, HALF_F64)
        self.assertEqual(state.read_vgpr_pair(0), HALF_F64)
        self.assertEqual(state.read_vgpr_pair(2), FOUR_F64)
        self.assertEqual(state.pc, 4)
        self.assertEqual(state.cycle_count, 1)

    def test_sgpr_source_execution(self) -> None:
        """Execute v_rsq_f64 with SGPR pair as source operand."""
        state = ScalarMachineState()
        # Set s[4:5] = 16.0
        state.write_sgpr_pair(4, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_sgpr(4),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, QUARTER_F64)
        self.assertEqual(state.read_vgpr_pair(0), QUARTER_F64)

    def test_literal_source_execution(self) -> None:
        """Execute v_rsq_f64 with immediate 64-bit constant literal."""
        state = ScalarMachineState()
        instr = VRSQF64Instruction(
            dst=RegPair(10, RegType.VGPR),
            src0=VOP3Operand.from_lit(QUARTER_F64),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, TWO_F64)
        self.assertEqual(state.read_vgpr_pair(10), TWO_F64)

    def test_input_modifiers_in_interpreter(self) -> None:
        """Execute v_rsq_f64 with abs and neg input modifiers via interpreter."""
        state = ScalarMachineState()
        state.write_vgpr_pair(6, NEG_FOUR_F64)

        # abs(-4.0) -> +4.0 -> rsq gives 0.5
        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(6, abs_mod=True),
        )
        interp = ScalarInterpreter(state)
        self.assertEqual(interp.execute(instr), HALF_F64)
        self.assertEqual(state.read_vgpr_pair(0), HALF_F64)

    def test_clamp_modifier_in_interpreter(self) -> None:
        """Execute v_rsq_f64 with clamp modifier via interpreter."""
        state = ScalarMachineState()
        state.write_vgpr_pair(2, QUARTER_F64)  # rsq(0.25) = 2.0 -> clamped to 1.0

        instr = VRSQF64Instruction(
            dst=RegPair(8, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
            clamp=True,
        )
        interp = ScalarInterpreter(state)
        self.assertEqual(interp.execute(instr), ONE_F64)
        self.assertEqual(state.read_vgpr_pair(8), ONE_F64)

    def test_register_isolation(self) -> None:
        """Verify that writing to destination does not modify unaddressed registers."""
        state = ScalarMachineState()
        state.write_vgpr(0, 0x1111_1111)
        state.write_vgpr(1, 0x2222_2222)
        state.write_vgpr(2, 0x3333_3333)
        state.write_vgpr(3, 0x4444_4444)
        state.write_vgpr_pair(4, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(2, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(4),
        )
        interp = ScalarInterpreter(state)
        interp.execute(instr)

        # v[0:1] must remain untouched
        self.assertEqual(state.read_vgpr(0), 0x1111_1111)
        self.assertEqual(state.read_vgpr(1), 0x2222_2222)
        # v[2:3] has updated result
        self.assertEqual(state.read_vgpr_pair(2), HALF_F64)


# ============================================================================
# Section 4: SIMD Multi-Lane Divergence and 32/64-Bit EXEC Masking
# ============================================================================

@dataclass
class SIMDState:
    """Architectural state for a wavefront (Wave32 or Wave64) with EXEC masking."""
    num_lanes: int = 64
    exec_mask: int = MASK64
    lanes: List[List[int]] = field(default_factory=list)
    sgprs: List[int] = field(default_factory=lambda: [0] * 106)

    def __post_init__(self) -> None:
        if self.num_lanes not in (32, 64):
            raise ValueError(f"num_lanes must be 32 or 64, got {self.num_lanes}")
        if not self.lanes:
            self.lanes = [[0] * 256 for _ in range(self.num_lanes)]
        if self.num_lanes == 32:
            self.exec_mask &= MASK32

    def is_lane_active(self, lane_idx: int) -> bool:
        """True if the lane's bit is enabled in the wavefront EXEC mask."""
        return bool(self.exec_mask & (1 << lane_idx))

    def read_lane_vgpr(self, lane_idx: int, reg_idx: int) -> int:
        return self.lanes[lane_idx][reg_idx] & MASK32

    def write_lane_vgpr(self, lane_idx: int, reg_idx: int, val: int) -> None:
        self.lanes[lane_idx][reg_idx] = val & MASK32

    def read_lane_vgpr_pair(self, lane_idx: int, reg_idx: int) -> int:
        lo = self.read_lane_vgpr(lane_idx, reg_idx)
        hi = self.read_lane_vgpr(lane_idx, reg_idx + 1)
        return words_to_f64_bits(lo, hi)

    def write_lane_vgpr_pair(self, lane_idx: int, reg_idx: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.write_lane_vgpr(lane_idx, reg_idx, lo)
        self.write_lane_vgpr(lane_idx, reg_idx + 1, hi)

    def read_sgpr_pair(self, reg_idx: int) -> int:
        lo = self.sgprs[reg_idx] & MASK32
        hi = self.sgprs[reg_idx + 1] & MASK32
        return words_to_f64_bits(lo, hi)

    def write_sgpr_pair(self, reg_idx: int, val64: int) -> None:
        lo, hi = f64_bits_to_words(val64)
        self.sgprs[reg_idx] = lo
        self.sgprs[reg_idx + 1] = hi

    def read_lane_operand_bits(self, lane_idx: int, op: VOP3Operand) -> int:
        if op.literal is not None:
            return op.literal.bits
        assert op.reg_pair is not None
        if op.reg_pair.reg_type == RegType.VGPR:
            return self.read_lane_vgpr_pair(lane_idx, op.reg_pair.index)
        elif op.reg_pair.reg_type == RegType.SGPR:
            return self.read_sgpr_pair(op.reg_pair.index)
        raise ValueError(f"Unknown register type {op.reg_pair.reg_type}")


class SIMDInterpreter:
    """SIMD wavefront execution engine modeling divergent multi-lane execution."""

    def __init__(self, state: Optional[SIMDState] = None) -> None:
        self.state = state if state is not None else SIMDState()

    def execute(self, instr: VRSQF64Instruction) -> List[Optional[int]]:
        """Execute VRSQF64Instruction across all active lanes conditioned on EXEC mask."""
        results: List[Optional[int]] = []
        for lane in range(self.state.num_lanes):
            if not self.state.is_lane_active(lane):
                results.append(None)
                continue
            raw_bits = self.state.read_lane_operand_bits(lane, instr.src0)
            res = oracle_v_rsq_f64(
                raw_bits,
                abs_src=instr.src0.abs_mod,
                neg_src=instr.src0.neg_mod,
                clamp=instr.clamp,
                flush_subnormals=instr.flush_denorms,
            )
            self.state.write_lane_vgpr_pair(lane, instr.dst.index, res)
            results.append(res)
        return results


class TestSIMDMultiLaneDivergence(unittest.TestCase):
    """Unit tests for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def test_wave32_full_execution(self) -> None:
        """Wave32 wavefront executes across all 32 lanes with divergent per-lane inputs."""
        simd = SIMDState(num_lanes=32, exec_mask=MASK32)
        # Populate each lane with a distinct power of 4 or known square
        test_inputs = [
            ONE_F64, FOUR_F64, SIXTEEN_F64, SIXTY_FOUR_F64,
            QUARTER_F64, SIXTEENTH_F64, POS_ZERO, POS_INF,
        ] * 4  # 32 inputs
        for lane, inp in enumerate(test_inputs):
            simd.write_lane_vgpr_pair(lane, 2, inp)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertEqual(len(res), 32)
        for lane, inp in enumerate(test_inputs):
            expected = oracle_v_rsq_f64(inp)
            self.assertEqual(res[lane], expected)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), expected)

    def test_wave64_full_execution(self) -> None:
        """Wave64 wavefront executes across all 64 lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=MASK64)
        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 4, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(6, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(4),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertEqual(len(res), 64)
        for lane in range(64):
            self.assertEqual(res[lane], HALF_F64)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 6), HALF_F64)

    def test_checkerboard_divergence_even_lanes(self) -> None:
        """EXEC mask enables even lanes only; odd lanes must remain completely unmodified."""
        exec_even = 0x5555_5555_5555_5555
        simd = SIMDState(num_lanes=64, exec_mask=exec_even)
        SENTINEL = 0xDEAD_BEEF_CAFE_BABE

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(res[lane], HALF_F64)
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), HALF_F64)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_wave_half_divergence(self) -> None:
        """EXEC mask enables lower 32 lanes (half-wave) only."""
        exec_lo32 = 0x0000_0000_FFFF_FFFF
        simd = SIMDState(num_lanes=64, exec_mask=exec_lo32)
        SENTINEL = 0x1234_5678_9ABC_DEF0

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(32):
            self.assertEqual(res[lane], QUARTER_F64)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), QUARTER_F64)

        for lane in range(32, 64):
            self.assertIsNone(res[lane])
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_single_lane_active(self) -> None:
        """EXEC mask enables exactly one lane (lane 17)."""
        target_lane = 17
        simd = SIMDState(num_lanes=64, exec_mask=(1 << target_lane))
        SENTINEL = 0xAAAA_BBBB_CCCC_DDDD

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, QUARTER_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane == target_lane:
                self.assertEqual(res[lane], TWO_F64)
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), TWO_F64)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_zero_exec_mask_noop(self) -> None:
        """An EXEC mask of 0 is a complete no-op across all registers and lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=0)
        SENTINEL = 0x5555_AAAA_5555_AAAA

        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, SENTINEL)
            simd.write_lane_vgpr_pair(lane, 2, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(2),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        self.assertTrue(all(r is None for r in res))
        for lane in range(64):
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), SENTINEL)

    def test_scalar_broadcast_to_active_simd_lanes(self) -> None:
        """Source operand from SGPR pair is broadcast identically across all active lanes."""
        simd = SIMDState(num_lanes=64, exec_mask=MASK64)
        simd.write_sgpr_pair(10, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src0=VOP3Operand.from_sgpr(10),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            self.assertEqual(res[lane], QUARTER_F64)
            self.assertEqual(simd.read_lane_vgpr_pair(lane, 4), QUARTER_F64)


# ============================================================================
# Section 5: Register Aliasing and In-Place Operand Preservation
# ============================================================================

class TestRegisterAliasing(unittest.TestCase):
    """Unit tests for register aliasing and in-place operand preservation."""

    def test_complete_in_place_aliasing_scalar(self) -> None:
        """In-place execution where destination and source are identical (v[0:1] = rsq(v[0:1]))."""
        state = ScalarMachineState()
        state.write_vgpr_pair(0, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(0),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, HALF_F64)
        self.assertEqual(state.read_vgpr_pair(0), HALF_F64)

    def test_partial_overlap_dst_low_src_high(self) -> None:
        """Partial overlap: dst=v[0:1], src=v[1:2] sharing register v[1]."""
        state = ScalarMachineState()
        # Set v[1:2] = 16.0
        state.write_vgpr_pair(1, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(1),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, QUARTER_F64)
        self.assertEqual(state.read_vgpr_pair(0), QUARTER_F64)

    def test_partial_overlap_dst_high_src_low(self) -> None:
        """Partial overlap: dst=v[2:3], src=v[1:2] sharing register v[2]."""
        state = ScalarMachineState()
        # Set v[1:2] = 0.25
        state.write_vgpr_pair(1, QUARTER_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(2, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(1),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, TWO_F64)
        self.assertEqual(state.read_vgpr_pair(2), TWO_F64)

    def test_successive_in_place_evaluations(self) -> None:
        """Two successive in-place rsq evaluations: rsq(rsq(16.0)) == rsq(0.25) == 2.0."""
        state = ScalarMachineState()
        state.write_vgpr_pair(4, SIXTEEN_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(4, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(4),
        )
        interp = ScalarInterpreter(state)

        # Step 1: rsq(16.0) -> 0.25
        r1 = interp.execute(instr)
        self.assertEqual(r1, QUARTER_F64)
        self.assertEqual(state.read_vgpr_pair(4), QUARTER_F64)

        # Step 2: rsq(0.25) -> 2.0
        r2 = interp.execute(instr)
        self.assertEqual(r2, TWO_F64)
        self.assertEqual(state.read_vgpr_pair(4), TWO_F64)

    def test_in_place_simd_multi_lane(self) -> None:
        """Wavefront executing in-place v[0:1] = rsq(v[0:1]) with lane divergence."""
        simd = SIMDState(num_lanes=64, exec_mask=0x5555_5555_5555_5555)  # even lanes
        for lane in range(64):
            simd.write_lane_vgpr_pair(lane, 0, FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(0, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(0),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(res[lane], HALF_F64)
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), HALF_F64)
            else:
                self.assertIsNone(res[lane])
                self.assertEqual(simd.read_lane_vgpr_pair(lane, 0), FOUR_F64)

    def test_in_place_negative_to_nan(self) -> None:
        """In-place replacement of negative float by Canonical QNaN."""
        state = ScalarMachineState()
        state.write_vgpr_pair(6, NEG_FOUR_F64)

        instr = VRSQF64Instruction(
            dst=RegPair(6, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(6),
        )
        interp = ScalarInterpreter(state)
        res = interp.execute(instr)

        self.assertEqual(res, CANONICAL_QNAN)
        self.assertEqual(state.read_vgpr_pair(6), CANONICAL_QNAN)


# ============================================================================
# Section 6: Condition Codes, Status Flags, and Carry/Borrow Generation
# ============================================================================

@dataclass
class FPStatusFlags:
    """IEEE 754 / RDNA2 floating-point exception sticky flags."""
    invalid: bool = False       # IOC: Invalid operation (negative operand, sNaN)
    div_by_zero: bool = False   # DZC: Division by zero (+0.0, -0.0 input)
    overflow: bool = False      # OFC: Overflow
    underflow: bool = False     # UFC: Underflow
    inexact: bool = False       # IXC: Inexact transcendental/rounding result

    def accumulate(self, other: FPStatusFlags) -> None:
        """Accumulate (bitwise OR) flags into hardware sticky status register."""
        self.invalid |= other.invalid
        self.div_by_zero |= other.div_by_zero
        self.overflow |= other.overflow
        self.underflow |= other.underflow
        self.inexact |= other.inexact

    def to_bits(self) -> int:
        """Encode flags into hardware status register bitmask."""
        bits = 0
        if self.invalid:
            bits |= 1 << 0
        if self.div_by_zero:
            bits |= 1 << 1
        if self.overflow:
            bits |= 1 << 2
        if self.underflow:
            bits |= 1 << 3
        if self.inexact:
            bits |= 1 << 4
        return bits


def compute_v_rsq_f64_flags(
    src_bits: int,
    abs_src: bool = False,
    neg_src: bool = False,
    clamp: bool = False,
    flush_subnormals: bool = False,
) -> Tuple[int, FPStatusFlags]:
    """Compute v_rsq_f64 result along with its architectural FP exception flags."""
    flags = FPStatusFlags()
    s = apply_f64_modifiers(src_bits, abs_src, neg_src)

    # Subnormal input under FTZ
    if flush_subnormals and is_subnormal(s):
        flags.underflow = True
        s = NEG_ZERO if (s & SIGN_MASK) else POS_ZERO

    # Signaling NaN input raises Invalid
    if is_snan(s):
        flags.invalid = True
        res = POS_ZERO if clamp else (s | QNAN_BIT)
        return res, flags

    # Quiet NaN input propagates without Invalid
    if is_nan(s):
        res = POS_ZERO if clamp else (s | QNAN_BIT)
        return res, flags

    # Negative non-zero and -Infinity input raises Invalid
    if (s & SIGN_MASK) and not is_zero(s):
        flags.invalid = True
        res = POS_ZERO if clamp else CANONICAL_QNAN
        return res, flags

    # Zero input raises Division-by-Zero
    if is_zero(s):
        flags.div_by_zero = True
        res = ONE_F64 if clamp else POS_INF
        return res, flags

    # Positive Infinity
    if s == POS_INF:
        return POS_ZERO, flags

    # Positive finite numbers
    val = bits_to_f64(s)
    res_val = 1.0 / math.sqrt(val)

    # Check exact powers of 2/4
    # An input is an exact power of 4 if log2(val) is an even integer
    is_exact = False
    if val > 0.0:
        log2_val = math.log2(val)
        if log2_val.is_integer() and (int(log2_val) % 2 == 0):
            is_exact = True

    if not is_exact:
        flags.inexact = True

    if clamp:
        if res_val < 0.0:
            res_val = 0.0
        elif res_val > 1.0:
            res_val = 1.0

    res_bits = f64_to_bits(res_val)
    if is_subnormal(res_bits):
        flags.underflow = True

    return res_bits, flags


class TestConditionCodesAndStatusFlags(unittest.TestCase):
    """Unit tests for condition codes, IEEE 754 status flags, and VCC condition masks."""

    def test_invalid_flag_negative_inputs(self) -> None:
        """Negative operands raise the Invalid Operation (IOC) status flag."""
        _, flags_neg4 = compute_v_rsq_f64_flags(NEG_FOUR_F64)
        self.assertTrue(flags_neg4.invalid)
        self.assertFalse(flags_neg4.div_by_zero)

        _, flags_neginf = compute_v_rsq_f64_flags(NEG_INF)
        self.assertTrue(flags_neginf.invalid)

    def test_invalid_flag_signaling_nan(self) -> None:
        """Signaling NaN operand raises the Invalid Operation (IOC) status flag."""
        snan = POS_INF | 0x0001_0000
        _, flags = compute_v_rsq_f64_flags(snan)
        self.assertTrue(flags.invalid)

    def test_quiet_nan_does_not_raise_invalid(self) -> None:
        """Quiet NaN operand propagates without asserting Invalid Operation."""
        _, flags = compute_v_rsq_f64_flags(CANONICAL_QNAN)
        self.assertFalse(flags.invalid)

    def test_div_by_zero_flag_zeros(self) -> None:
        """Both +0.0 and -0.0 raise Division by Zero (DZC)."""
        _, flags_pos0 = compute_v_rsq_f64_flags(POS_ZERO)
        self.assertTrue(flags_pos0.div_by_zero)
        self.assertFalse(flags_pos0.invalid)

        _, flags_neg0 = compute_v_rsq_f64_flags(NEG_ZERO)
        self.assertTrue(flags_neg0.div_by_zero)

    def test_inexact_flag_non_perfect_squares(self) -> None:
        """Non-exact reciprocal square roots (2.0, 3.0, 10.0) assert Inexact (IXC)."""
        _, flags_two = compute_v_rsq_f64_flags(TWO_F64)
        self.assertTrue(flags_two.inexact)

        _, flags_three = compute_v_rsq_f64_flags(THREE_F64)
        self.assertTrue(flags_three.inexact)

    def test_exact_roots_do_not_raise_inexact(self) -> None:
        """Exact roots of powers of 4 (1.0, 4.0, 16.0, 0.25, 0.0625) do not assert Inexact."""
        _, flags_1 = compute_v_rsq_f64_flags(ONE_F64)
        self.assertFalse(flags_1.inexact)

        _, flags_4 = compute_v_rsq_f64_flags(FOUR_F64)
        self.assertFalse(flags_4.inexact)

        _, flags_16 = compute_v_rsq_f64_flags(SIXTEEN_F64)
        self.assertFalse(flags_16.inexact)

        _, flags_quarter = compute_v_rsq_f64_flags(QUARTER_F64)
        self.assertFalse(flags_quarter.inexact)

        _, flags_sixteenth = compute_v_rsq_f64_flags(SIXTEENTH_F64)
        self.assertFalse(flags_sixteenth.inexact)

    def test_cumulative_sticky_flags_accumulation(self) -> None:
        """Verify that sticky status register correctly accumulates flags across instructions."""
        sticky = FPStatusFlags()

        # Step 1: exact operation -> no flags
        _, f1 = compute_v_rsq_f64_flags(FOUR_F64)
        sticky.accumulate(f1)
        self.assertEqual(sticky.to_bits(), 0)

        # Step 2: division by zero -> div_by_zero flag (bit 1)
        _, f2 = compute_v_rsq_f64_flags(POS_ZERO)
        sticky.accumulate(f2)
        self.assertEqual(sticky.to_bits(), 1 << 1)

        # Step 3: negative operand -> invalid flag (bit 0)
        _, f3 = compute_v_rsq_f64_flags(NEG_FOUR_F64)
        sticky.accumulate(f3)
        self.assertEqual(sticky.to_bits(), (1 << 0) | (1 << 1))

        # Step 4: inexact operation -> inexact flag (bit 4)
        _, f4 = compute_v_rsq_f64_flags(TWO_F64)
        sticky.accumulate(f4)
        self.assertEqual(sticky.to_bits(), (1 << 0) | (1 << 1) | (1 << 4))

    def test_vcc_condition_masks(self) -> None:
        """Compute condition mask across SIMD wave lanes (emulating V_CMP / VCC)."""
        simd = SIMDState(num_lanes=64, exec_mask=MASK64)
        # Assign various values across lanes
        for lane in range(64):
            if lane % 4 == 0:
                simd.write_lane_vgpr_pair(lane, 0, POS_ZERO)  # -> +inf
            elif lane % 4 == 1:
                simd.write_lane_vgpr_pair(lane, 0, POS_INF)   # -> +0.0
            elif lane % 4 == 2:
                simd.write_lane_vgpr_pair(lane, 0, NEG_FOUR_F64)  # -> NaN
            else:
                simd.write_lane_vgpr_pair(lane, 0, FOUR_F64)  # -> 0.5 (positive finite)

        instr = VRSQF64Instruction(
            dst=RegPair(2, RegType.VGPR),
            src0=VOP3Operand.from_vgpr(0),
        )
        interp = SIMDInterpreter(simd)
        res = interp.execute(instr)

        # Build condition bitmasks
        vcc_zero = 0
        vcc_inf = 0
        vcc_nan = 0
        vcc_finite_pos = 0

        for lane, r in enumerate(res):
            assert r is not None
            if is_zero(r):
                vcc_zero |= (1 << lane)
            elif is_inf(r):
                vcc_inf |= (1 << lane)
            elif is_nan(r):
                vcc_nan |= (1 << lane)
            else:
                vcc_finite_pos |= (1 << lane)

        # Verify orthogonal bitmasks partition the wave lanes
        self.assertEqual(vcc_zero | vcc_inf | vcc_nan | vcc_finite_pos, MASK64)
        self.assertEqual(vcc_zero & vcc_inf, 0)
        self.assertEqual(vcc_zero & vcc_nan, 0)
        self.assertEqual(vcc_inf & vcc_nan, 0)


# ============================================================================
# Section 7: IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ============================================================================

class TestIEEE754BoundaryExtremes(unittest.TestCase):
    """Unit tests for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_dbl_min_normal_boundary(self) -> None:
        """rsq(2^-1022) = 2^511, exact power of 2 representation."""
        res_bits = oracle_v_rsq_f64(DBL_MIN)
        self.assertFalse(is_nan(res_bits))
        self.assertFalse(is_inf(res_bits))
        expected_val = 2.0 ** 511
        self.assertAlmostEqual(bits_to_f64(res_bits), expected_val, delta=expected_val * 1e-15)

    def test_dbl_max_normal_boundary(self) -> None:
        """rsq(DBL_MAX) produces small finite positive normal float."""
        res_bits = oracle_v_rsq_f64(DBL_MAX)
        self.assertFalse(is_nan(res_bits))
        self.assertFalse(is_inf(res_bits))
        res_val = bits_to_f64(res_bits)
        self.assertGreater(res_val, 0.0)
        self.assertLess(res_val, 1e-150)

    def test_subnorm_min_gradual_underflow(self) -> None:
        """rsq(2^-1074) evaluates to 2^537 under IEEE-754 gradual underflow."""
        res_bits = oracle_v_rsq_f64(SUBNORM_MIN, flush_subnormals=False)
        self.assertFalse(is_nan(res_bits))
        self.assertFalse(is_inf(res_bits))
        expected_val = 2.0 ** 537
        self.assertAlmostEqual(bits_to_f64(res_bits), expected_val, delta=expected_val * 1e-15)

    def test_subnorm_max_gradual_underflow(self) -> None:
        """Largest positive subnormal computes without NaN under IEEE mode."""
        res_bits = oracle_v_rsq_f64(SUBNORM_MAX, flush_subnormals=False)
        self.assertFalse(is_nan(res_bits))
        self.assertFalse(is_inf(res_bits))
        self.assertGreater(bits_to_f64(res_bits), 2.0 ** 511)

    def test_subnormal_flush_to_zero_mode(self) -> None:
        """Under FTZ mode, both smallest and largest subnormals flush to 0, producing +Inf."""
        res_min_ftz = oracle_v_rsq_f64(SUBNORM_MIN, flush_subnormals=True)
        self.assertEqual(res_min_ftz, POS_INF)

        res_max_ftz = oracle_v_rsq_f64(SUBNORM_MAX, flush_subnormals=True)
        self.assertEqual(res_max_ftz, POS_INF)

    def test_qnan_payload_preservation(self) -> None:
        """Quiet NaN payloads are preserved and have the quiet bit asserted."""
        payload_a = 0x7FF8_0000_1234_5678
        res_a = oracle_v_rsq_f64(payload_a)
        self.assertEqual(res_a, payload_a)

        payload_b = 0x7FF9_BEEF_CAFE_0001
        res_b = oracle_v_rsq_f64(payload_b)
        self.assertEqual(res_b, payload_b)

    def test_snan_quieting(self) -> None:
        """Signaling NaN is converted to a Quiet NaN with quiet bit set."""
        snan = 0x7FF0_0000_0000_0001
        res = oracle_v_rsq_f64(snan)
        self.assertTrue(is_qnan(res))
        self.assertEqual(res, snan | QNAN_BIT)

    def test_negative_subnormals_yield_qnan(self) -> None:
        """Negative subnormals represent negative numbers and yield Canonical QNaN."""
        neg_sub_min = SUBNORM_MIN | SIGN_MASK
        neg_sub_max = SUBNORM_MAX | SIGN_MASK
        self.assertEqual(oracle_v_rsq_f64(neg_sub_min), CANONICAL_QNAN)
        self.assertEqual(oracle_v_rsq_f64(neg_sub_max), CANONICAL_QNAN)

    def test_monotonic_decrease_across_normals(self) -> None:
        """Reciprocal square root is strictly monotonically decreasing for positive inputs."""
        inputs = [
            0.0001, 0.01, 0.25, 0.5, 1.0, 2.0, 4.0, 16.0, 100.0, 10000.0, 1e8, 1e20,
        ]
        prev_res = math.inf
        for inp in inputs:
            curr_res = bits_to_f64(oracle_v_rsq_f64(f64_to_bits(inp)))
            self.assertLess(curr_res, prev_res)
            prev_res = curr_res

    def test_clamp_saturation_at_boundaries(self) -> None:
        """Clamp modifier saturates extreme values to [0.0, 1.0]."""
        # SUBNORM_MIN produces 2^537 >> 1.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(SUBNORM_MIN, clamp=True), ONE_F64)
        # DBL_MIN produces 2^511 >> 1.0 -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(DBL_MIN, clamp=True), ONE_F64)
        # DBL_MAX produces ~7.4e-155 -> within [0.0, 1.0] -> unchanged
        self.assertEqual(oracle_v_rsq_f64(DBL_MAX, clamp=True), oracle_v_rsq_f64(DBL_MAX, clamp=False))
        # +0.0 produces +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(POS_ZERO, clamp=True), ONE_F64)
        # -0.0 produces +inf -> clamped to 1.0
        self.assertEqual(oracle_v_rsq_f64(NEG_ZERO, clamp=True), ONE_F64)


# ============================================================================
# Section 8: Negative Controls, Syntax Rejection, and Mutation Detection
# ============================================================================

def mutant_omit_sqrt(src_bits: int, **kwargs: bool) -> int:
    """Mutant 1: computes 1.0 / x instead of 1.0 / sqrt(x)."""
    val = bits_to_f64(src_bits)
    if val <= 0.0:
        return CANONICAL_QNAN
    return f64_to_bits(1.0 / val)


def mutant_ignore_negative_nan(src_bits: int, **kwargs: bool) -> int:
    """Mutant 2: calculates root of absolute value instead of NaN for negative inputs."""
    s = src_bits & MASK64
    val = abs(bits_to_f64(s))
    if val == 0.0:
        return POS_INF
    return f64_to_bits(1.0 / math.sqrt(val))


def mutant_zero_to_zero(src_bits: int, **kwargs: bool) -> int:
    """Mutant 3: returns +0.0 instead of +Infinity on zero inputs."""
    if is_zero(src_bits):
        return POS_ZERO
    return oracle_v_rsq_f64(src_bits, **kwargs)


def mutant_omit_clamp(src_bits: int, **kwargs: bool) -> int:
    """Mutant 4: fails to clamp values > 1.0 to 1.0."""
    return oracle_v_rsq_f64(src_bits, clamp=False)


def mutant_invert_sign(src_bits: int, **kwargs: bool) -> int:
    """Mutant 5: negates the output sign."""
    res = oracle_v_rsq_f64(src_bits, **kwargs)
    return res ^ SIGN_MASK


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Unit tests for negative controls, syntax rejection, and mutation detection."""

    def test_syntax_rejection_invalid_vgpr_index(self) -> None:
        """Register pairs with indices > 254 must be rejected with ValueError."""
        with self.assertRaises(ValueError):
            RegPair(255, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(300, RegType.VGPR)
        with self.assertRaises(ValueError):
            RegPair(-1, RegType.VGPR)

    def test_syntax_rejection_invalid_sgpr_index(self) -> None:
        """SGPR pairs with indices > 104 must be rejected with ValueError."""
        with self.assertRaises(ValueError):
            RegPair(105, RegType.SGPR)
        with self.assertRaises(ValueError):
            RegPair(-5, RegType.SGPR)

    def test_syntax_rejection_invalid_literal_range(self) -> None:
        """Literal values outside 64-bit unsigned bounds must be rejected."""
        with self.assertRaises(ValueError):
            Literal64(-1)
        with self.assertRaises(ValueError):
            Literal64(1 << 64)

    def test_syntax_rejection_invalid_destination_type(self) -> None:
        """VALU instruction destination cannot be SGPR or literal."""
        with self.assertRaises(ValueError):
            VRSQF64Instruction(
                dst=RegPair(0, RegType.SGPR),
                src0=VOP3Operand.from_vgpr(2),
            )

    def test_syntax_rejection_invalid_vop3_operand_spec(self) -> None:
        """VOP3Operand must specify exactly one of reg_pair or literal."""
        with self.assertRaises(ValueError):
            VOP3Operand(reg_pair=None, literal=None)
        with self.assertRaises(ValueError):
            VOP3Operand(
                reg_pair=RegPair(0, RegType.VGPR),
                literal=Literal64(0),
            )

    def test_syntax_rejection_invalid_simd_lane_count(self) -> None:
        """SIMDState must only accept 32 or 64 lanes."""
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=16)
        with self.assertRaises(ValueError):
            SIMDState(num_lanes=48)

    def test_mutation_detection_omit_sqrt(self) -> None:
        """Verify that omitting square root is detected."""
        golden = oracle_v_rsq_f64(FOUR_F64)  # 0.5
        mutant = mutant_omit_sqrt(FOUR_F64)   # 0.25
        self.assertNotEqual(golden, mutant)

    def test_mutation_detection_ignore_negative_nan(self) -> None:
        """Verify that failing to produce NaN on negative input is detected."""
        golden = oracle_v_rsq_f64(NEG_FOUR_F64)
        mutant = mutant_ignore_negative_nan(NEG_FOUR_F64)
        self.assertEqual(golden, CANONICAL_QNAN)
        self.assertNotEqual(mutant, CANONICAL_QNAN)

    def test_mutation_detection_zero_to_zero(self) -> None:
        """Verify that zero returning 0 instead of Infinity is detected."""
        golden = oracle_v_rsq_f64(POS_ZERO)
        mutant = mutant_zero_to_zero(POS_ZERO)
        self.assertEqual(golden, POS_INF)
        self.assertEqual(mutant, POS_ZERO)
        self.assertNotEqual(golden, mutant)

    def test_mutation_detection_omit_clamp(self) -> None:
        """Verify that omitting clamp saturation is detected."""
        golden = oracle_v_rsq_f64(QUARTER_F64, clamp=True)  # 1.0
        mutant = mutant_omit_clamp(QUARTER_F64, clamp=True)  # 2.0
        self.assertEqual(golden, ONE_F64)
        self.assertEqual(mutant, TWO_F64)
        self.assertNotEqual(golden, mutant)

    def test_mutation_detection_invert_sign(self) -> None:
        """Verify that inverted output sign is detected."""
        golden = oracle_v_rsq_f64(FOUR_F64)
        mutant = mutant_invert_sign(FOUR_F64)
        self.assertEqual(golden, HALF_F64)
        self.assertEqual(mutant, HALF_F64 | SIGN_MASK)

    def test_mathematical_property_rsq_squared_identity(self) -> None:
        """Mathematical identity: (rsq(x))^2 == 1.0 / x for positive floats."""
        test_values = [0.1, 0.5, 1.0, 2.0, 3.5, 7.8, 10.0, 42.0, 100.0, 1024.0]
        for val in test_values:
            bits = f64_to_bits(val)
            rsq_val = bits_to_f64(oracle_v_rsq_f64(bits))
            reconstructed_inv = rsq_val * rsq_val
            expected_inv = 1.0 / val
            rel_error = abs(reconstructed_inv - expected_inv) / expected_inv
            self.assertLess(rel_error, 1e-14)

    def test_mathematical_scaling_property(self) -> None:
        """Scaling identity: rsq(4 * x) == 0.5 * rsq(x)."""
        test_values = [0.05, 0.25, 1.0, 4.0, 9.0, 25.0, 100.0]
        for val in test_values:
            rsq_x = bits_to_f64(oracle_v_rsq_f64(f64_to_bits(val)))
            rsq_4x = bits_to_f64(oracle_v_rsq_f64(f64_to_bits(4.0 * val)))
            self.assertAlmostEqual(rsq_4x, 0.5 * rsq_x, places=14)


if __name__ == "__main__":
    unittest.main()
