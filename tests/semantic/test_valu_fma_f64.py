from __future__ import annotations

"""Semantic interpreter tests and reference mathematical oracle for RDNA2 v_fma_f64.

Tests the double-precision fused multiply-add instruction:
    v_fma_f64 D, S0, S1, S2
Lowered semantics:
    D = (S0 * S1) + S2
with single-rounding precision per IEEE 754-2008 (binary64).
Supports input modifiers:
    - abs (|src|: clear sign bit)
    - neg (-src: invert sign bit)
Supports output modifiers:
    - clamp: saturates result to [0.0, 1.0], with NaN clamped to +0.0
    - omod: mul:2, mul:4, div:2
Architectural model:
    64-bit floating point registers mapped onto aligned VGPR/SGPR pairs:
    [lo: 32-bit dword, hi: 32-bit dword].
"""

from enum import Enum, auto
from fractions import Fraction
import math
import struct
import sys
import unittest

# IEEE 754 double precision bitwise constants
F64_POS_ZERO = 0x0000000000000000
F64_NEG_ZERO = 0x8000000000000000
F64_ONE = 0x3FF0000000000000
F64_NEG_ONE = 0xBFF0000000000000
F64_TWO = 0x4000000000000000
F64_HALF = 0x3FE0000000000000
F64_POS_INF = 0x7FF0000000000000
F64_NEG_INF = 0xFFF0000000000000
F64_CANONICAL_QNAN = 0x7FF8000000000000
F64_SNAN = 0x7FF0000000000001
F64_SUB_MIN = 0x0000000000000001
F64_SUB_MAX = 0x000FFFFFFFFFFFFF
F64_MIN_NORMAL = 0x0010000000000000
F64_MAX_NORMAL = 0x7FEFFFFFFFFFFFFF

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def f64_to_u64(val: float) -> int:
    """Pack Python float (IEEE 754 binary64) into unsigned 64-bit integer."""
    return struct.unpack("<Q", struct.pack("<d", val))[0]


def u64_to_f64(bits: int) -> float:
    """Unpack unsigned 64-bit integer into Python float (IEEE 754 binary64)."""
    return struct.unpack("<d", struct.pack("<Q", bits & MASK64))[0]


def u64_to_dwords(val: int) -> tuple[int, int]:
    """Split 64-bit word into (low 32-bit dword, high 32-bit dword)."""
    return val & MASK32, (val >> 32) & MASK32


def dwords_to_u64(lo: int, hi: int) -> int:
    """Combine (low 32-bit dword, high 32-bit dword) into 64-bit word."""
    return (lo & MASK32) | ((hi & MASK32) << 32)


def apply_src_mods_u64(val_u64: int, *, neg: bool = False, abs_: bool = False) -> int:
    """Apply bit-exact RDNA2 source modifiers (|src|, -src) to a 64-bit float bit pattern."""
    bits = val_u64 & MASK64
    if abs_:
        bits &= ~F64_NEG_ZERO  # clear sign bit
    if neg:
        bits ^= F64_NEG_ZERO   # invert sign bit
    return bits


def apply_clamp_f64(val: float) -> float:
    """Apply RDNA2 clamp modifier [0.0, 1.0] to float64, mapping NaNs to +0.0."""
    if math.isnan(val):
        return 0.0
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def apply_omod_f64(val: float, omod: int) -> float:
    """Apply RDNA2 output modifier: 0: none, 1: mul:2, 2: mul:4, 3: div:2."""
    if omod == 0:
        return val
    elif omod == 1:
        return val * 2.0
    elif omod == 2:
        return val * 4.0
    elif omod == 3:
        return val * 0.5
    else:
        raise ValueError(f"Invalid omod value: {omod}")


def raw_fma_f64(v0: float, v1: float, v2: float) -> float:
    """Compute exact IEEE 754 double precision FMA (v0 * v1) + v2."""
    if math.isnan(v0) or math.isnan(v1) or math.isnan(v2):
        return math.nan

    if (math.isinf(v0) and v1 == 0.0) or (v0 == 0.0 and math.isinf(v1)):
        return math.nan

    p_inf = math.isinf(v0) or math.isinf(v1)
    if p_inf:
        p_sign = -1.0 if (math.copysign(1.0, v0) * math.copysign(1.0, v1) < 0) else 1.0
        if math.isinf(v2):
            v2_sign = math.copysign(1.0, v2)
            if p_sign != v2_sign:
                return math.nan
        return math.copysign(float("inf"), p_sign)

    if math.isinf(v2):
        return v2

    if v0 == 0.0 or v1 == 0.0:
        prod_sign = -1.0 if (math.copysign(1.0, v0) * math.copysign(1.0, v1) < 0) else 1.0
        if v2 == 0.0:
            v2_sign = math.copysign(1.0, v2)
            if prod_sign < 0 and v2_sign < 0:
                return -0.0
            return 0.0
        return v2

    frac = Fraction(v0) * Fraction(v1) + Fraction(v2)
    if frac == 0:
        return 0.0
    try:
        return float(frac)
    except OverflowError:
        return math.copysign(float("inf"), 1.0 if frac > 0 else -1.0)


def ref_fma_f64(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> float:
    """Reference mathematical oracle for v_fma_f64 calculating (s0 * s1) + s2 in binary64."""
    u0 = apply_src_mods_u64(f64_to_u64(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u64(f64_to_u64(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u64(f64_to_u64(s2), neg=neg2, abs_=abs2)

    v0 = u64_to_f64(u0)
    v1 = u64_to_f64(u1)
    v2 = u64_to_f64(u2)

    res = raw_fma_f64(v0, v1, v2)

    if omod != 0:
        res = apply_omod_f64(res, omod)
    if clamp:
        res = apply_clamp_f64(res)

    return res


def ref_fma_f64_u64(
    u0: int,
    u1: int,
    u2: int,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> int:
    """Bit-exact reference mathematical oracle for v_fma_f64 using 64-bit integer words."""
    res_f = ref_fma_f64(
        u64_to_f64(u0),
        u64_to_f64(u1),
        u64_to_f64(u2),
        neg0=neg0,
        neg1=neg1,
        neg2=neg2,
        abs0=abs0,
        abs1=abs1,
        abs2=abs2,
        clamp=clamp,
        omod=omod,
    )
    return f64_to_u64(res_f)


class RegisterPair:
    """Architectural model of a 64-bit register pair (e.g. v[0:1] or s[0:1])."""

    def __init__(self, lo: int = 0, hi: int = 0) -> None:
        self.lo = lo & MASK32
        self.hi = hi & MASK32

    @classmethod
    def from_u64(cls, val: int) -> RegisterPair:
        lo, hi = u64_to_dwords(val)
        return cls(lo, hi)

    @classmethod
    def from_f64(cls, val: float) -> RegisterPair:
        return cls.from_u64(f64_to_u64(val))

    @property
    def as_u64(self) -> int:
        return dwords_to_u64(self.lo, self.hi)

    @property
    def as_f64(self) -> float:
        return u64_to_f64(self.as_u64)

    def set_u64(self, val: int) -> None:
        self.lo, self.hi = u64_to_dwords(val)

    def set_f64(self, val: float) -> None:
        self.set_u64(f64_to_u64(val))

    def __repr__(self) -> str:
        return f"RegisterPair(lo={self.lo:#010x}, hi={self.hi:#010x}, f64={self.as_f64})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RegisterPair):
            return NotImplemented
        return self.lo == other.lo and self.hi == other.hi


class TestFmaF64MathematicalOracle(unittest.TestCase):
    """Test suite for reference mathematical oracle and architectural register models."""

    def test_basic_fma_identities(self):
        # 1.0 * 1.0 + 0.0 == 1.0
        self.assertEqual(ref_fma_f64(1.0, 1.0, 0.0), 1.0)
        # 2.0 * 3.0 + 4.0 == 10.0
        self.assertEqual(ref_fma_f64(2.0, 3.0, 4.0), 10.0)
        # -2.5 * 4.0 + 1.25 == -8.75
        self.assertEqual(ref_fma_f64(-2.5, 4.0, 1.25), -8.75)
        # 0.0 * 100.0 + 5.5 == 5.5
        self.assertEqual(ref_fma_f64(0.0, 100.0, 5.5), 5.5)

    def test_fused_precision_single_rounding(self):
        # FMA retains full intermediate product precision without intermediate rounding.
        # Let a = 1.0 + 2**-52, b = 1.0 - 2**-52, c = -1.0
        # (a * b) = 1.0 - 2**-104
        # Unfused: (a * b) rounds to 1.0, so (a * b) - 1.0 == 0.0
        # Fused: (a * b) - 1.0 retains -2**-104, rounding to -2**-104 != 0.0
        a = 1.0 + (2.0 ** -52)
        b = 1.0 - (2.0 ** -52)
        c = -1.0
        unfused = (a * b) + c
        fused = ref_fma_f64(a, b, c)
        self.assertEqual(unfused, 0.0)
        self.assertNotEqual(fused, 0.0)
        self.assertEqual(fused, -(2.0 ** -104))

    def test_source_modifiers(self):
        # abs(|src|): clears sign bit
        self.assertEqual(ref_fma_f64(-3.5, 2.0, 1.0, abs0=True), 8.0)
        self.assertEqual(ref_fma_f64(3.5, -2.0, 1.0, abs1=True), 8.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, -1.0, abs2=True), 8.0)

        # neg(-src): inverts sign bit
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg0=True), -6.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg1=True), -6.0)
        self.assertEqual(ref_fma_f64(3.5, 2.0, 1.0, neg2=True), 6.0)

        # combined neg and abs: -|src|
        self.assertEqual(ref_fma_f64(-3.5, 2.0, 1.0, neg0=True, abs0=True), -6.0)

    def test_output_modifiers_clamp_and_omod(self):
        # clamp saturates to [0.0, 1.0]
        self.assertEqual(ref_fma_f64(2.0, 3.0, 0.0, clamp=True), 1.0)
        self.assertEqual(ref_fma_f64(-2.0, 3.0, 0.0, clamp=True), 0.0)
        self.assertEqual(ref_fma_f64(0.5, 1.0, 0.25, clamp=True), 0.75)

        # omod: mul:2, mul:4, div:2
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=1), 7.0)   # (3.5) * 2 = 7.0
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=2), 14.0)  # (3.5) * 4 = 14.0
        self.assertEqual(ref_fma_f64(1.5, 2.0, 0.5, omod=3), 1.75)  # (3.5) * 0.5 = 1.75

        # omod combined with clamp: (omod applies before clamp)
        self.assertEqual(ref_fma_f64(0.2, 2.0, 0.0, omod=1, clamp=True), 0.8)
        self.assertEqual(ref_fma_f64(0.4, 2.0, 0.0, omod=2, clamp=True), 1.0)

    def test_register_pair_architecture(self):
        rp = RegisterPair.from_f64(1.0)
        self.assertEqual(rp.as_f64, 1.0)
        self.assertEqual(rp.as_u64, F64_ONE)
        self.assertEqual(rp.lo, 0x00000000)
        self.assertEqual(rp.hi, 0x3FF00000)

        rp.set_f64(-2.0)
        self.assertEqual(rp.as_f64, -2.0)
        self.assertEqual(rp.hi, 0xC0000000)

        # bit-exact oracle roundtrip
        u_res = ref_fma_f64_u64(F64_TWO, F64_HALF, F64_ONE)
        self.assertEqual(u_res, f64_to_u64(2.0))


class Opcode(Enum):
    V_FMA_F64 = "v_fma_f64"


class OperandKind(Enum):
    VGPR_PAIR = auto()
    SGPR_PAIR = auto()
    INLINE_CONST = auto()
    LITERAL_CONST = auto()


class TypeKind(Enum):
    FLOAT64 = "f64"
    UINT64 = "u64"
    UINT32 = "u32"


class OMod(Enum):
    NONE = 0
    MUL2 = 1
    MUL4 = 2
    DIV2 = 3

    def to_asm(self) -> str:
        if self == OMod.MUL2:
            return "mul:2"
        elif self == OMod.MUL4:
            return "mul:4"
        elif self == OMod.DIV2:
            return "div:2"
        return ""


class ASTOperand:
    """Symbolic AST operand representing a register pair, constant, or literal with modifiers."""

    def __init__(
        self,
        kind: OperandKind,
        index: int = 0,
        value: float | int | None = None,
        *,
        neg: bool = False,
        abs_: bool = False,
        type_kind: TypeKind = TypeKind.FLOAT64,
    ) -> None:
        self.kind = kind
        self.index = index
        self.value = value
        self.neg = neg
        self.abs_ = abs_
        self.type_kind = type_kind

    @classmethod
    def vgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.VGPR_PAIR, index=index, neg=neg, abs_=abs_)

    @classmethod
    def sgpr(cls, index: int, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.SGPR_PAIR, index=index, neg=neg, abs_=abs_)

    @classmethod
    def inline_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.INLINE_CONST, value=value, neg=neg, abs_=abs_)

    @classmethod
    def literal_const(cls, value: float, *, neg: bool = False, abs_: bool = False) -> ASTOperand:
        return cls(OperandKind.LITERAL_CONST, value=value, neg=neg, abs_=abs_)

    def disasm(self) -> str:
        if self.kind == OperandKind.VGPR_PAIR:
            body = f"v[{self.index}:{self.index + 1}]"
        elif self.kind == OperandKind.SGPR_PAIR:
            body = f"s[{self.index}:{self.index + 1}]"
        elif self.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            body = str(self.value)
        else:
            raise ValueError(f"Unknown operand kind: {self.kind}")

        if self.abs_:
            body = f"|{body}|"
        if self.neg:
            body = f"-{body}"
        return body

    def __repr__(self) -> str:
        return f"ASTOperand({self.disasm()}, type={self.type_kind.value})"


class ASTInstruction:
    """Symbolic AST instruction representation for v_fma_f64."""

    def __init__(
        self,
        opcode: Opcode,
        dst: ASTOperand,
        src0: ASTOperand,
        src1: ASTOperand,
        src2: ASTOperand,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
    ) -> None:
        self.opcode = opcode
        self.dst = dst
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def validate(self) -> None:
        if self.opcode != Opcode.V_FMA_F64:
            raise ValueError(f"Invalid opcode for v_fma_f64 AST: {self.opcode}")
        if self.dst.kind != OperandKind.VGPR_PAIR:
            raise TypeError("Destination must be a VGPR pair")
        if self.dst.type_kind != TypeKind.FLOAT64:
            raise TypeError(f"Destination type must be FLOAT64, got {self.dst.type_kind}")
        if not (0 <= self.dst.index <= 254):
            raise ValueError(f"Destination VGPR pair index out of range: {self.dst.index}")
        if self.dst.neg or self.dst.abs_:
            raise ValueError("Destination operand cannot have neg/abs modifiers")

        for i, src in enumerate((self.src0, self.src1, self.src2)):
            if src.type_kind != TypeKind.FLOAT64:
                raise TypeError(f"Source operand {i} type must be FLOAT64, got {src.type_kind}")
            if src.kind == OperandKind.VGPR_PAIR and not (0 <= src.index <= 254):
                raise ValueError(f"Source {i} VGPR pair index out of range: {src.index}")
            if src.kind == OperandKind.SGPR_PAIR and not (0 <= src.index <= 104):
                raise ValueError(f"Source {i} SGPR pair index out of range: {src.index}")

    def disasm(self) -> str:
        parts = [
            self.opcode.value,
            f"{self.dst.disasm()},",
            f"{self.src0.disasm()},",
            f"{self.src1.disasm()},",
            self.src2.disasm(),
        ]
        if self.omod != OMod.NONE:
            parts.append(self.omod.to_asm())
        if self.clamp:
            parts.append("clamp")
        return " ".join(parts)

    def __repr__(self) -> str:
        return f"ASTInstruction({self.disasm()})"


class TypedExpr:
    """Base class for typed symbolic expressions."""

    def __init__(self, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        self.type_kind = type_kind

    def evaluate(self, env: dict[str, float]) -> float:
        raise NotImplementedError

    def simplify(self) -> TypedExpr:
        return self


class VariableExpr(TypedExpr):
    """Named variable in symbolic expression tree (e.g. representing a register value)."""

    def __init__(self, name: str, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        super().__init__(type_kind)
        self.name = name

    def evaluate(self, env: dict[str, float]) -> float:
        if self.name not in env:
            raise KeyError(f"Variable {self.name} not found in evaluation environment")
        return env[self.name]

    def __repr__(self) -> str:
        return f"Var({self.name})"


class ConstantExpr(TypedExpr):
    """Constant literal in symbolic expression tree."""

    def __init__(self, value: float, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        super().__init__(type_kind)
        self.value = value

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def __repr__(self) -> str:
        return f"Const({self.value})"


class UnaryOpExpr(TypedExpr):
    """Unary modifier applied to a typed expression ('neg' or 'abs')."""

    def __init__(self, op: str, expr: TypedExpr, type_kind: TypeKind = TypeKind.FLOAT64) -> None:
        super().__init__(type_kind)
        if op not in ("neg", "abs"):
            raise ValueError(f"Unknown unary operator: {op}")
        self.op = op
        self.expr = expr

    def evaluate(self, env: dict[str, float]) -> float:
        val = self.expr.evaluate(env)
        if self.op == "abs":
            return math.fabs(val)
        elif self.op == "neg":
            return -val
        return val

    def simplify(self) -> TypedExpr:
        sub = self.expr.simplify()
        if isinstance(sub, ConstantExpr):
            if self.op == "abs":
                return ConstantExpr(math.fabs(sub.value), self.type_kind)
            elif self.op == "neg":
                return ConstantExpr(-sub.value, self.type_kind)
        return UnaryOpExpr(self.op, sub, self.type_kind)

    def __repr__(self) -> str:
        return f"{self.op}({self.expr})"


class FMAExpr(TypedExpr):
    """Symbolic FMA ternary typed expression: (src0 * src1) + src2 with modifiers."""

    def __init__(
        self,
        src0: TypedExpr,
        src1: TypedExpr,
        src2: TypedExpr,
        *,
        clamp: bool = False,
        omod: OMod = OMod.NONE,
        type_kind: TypeKind = TypeKind.FLOAT64,
    ) -> None:
        super().__init__(type_kind)
        self.src0 = src0
        self.src1 = src1
        self.src2 = src2
        self.clamp = clamp
        self.omod = omod

    def evaluate(self, env: dict[str, float]) -> float:
        v0 = self.src0.evaluate(env)
        v1 = self.src1.evaluate(env)
        v2 = self.src2.evaluate(env)
        return ref_fma_f64(v0, v1, v2, clamp=self.clamp, omod=self.omod.value)

    def simplify(self) -> TypedExpr:
        s0 = self.src0.simplify()
        s1 = self.src1.simplify()
        s2 = self.src2.simplify()
        if isinstance(s0, ConstantExpr) and isinstance(s1, ConstantExpr) and isinstance(s2, ConstantExpr):
            res = ref_fma_f64(s0.value, s1.value, s2.value, clamp=self.clamp, omod=self.omod.value)
            return ConstantExpr(res, self.type_kind)
        return FMAExpr(s0, s1, s2, clamp=self.clamp, omod=self.omod, type_kind=self.type_kind)

    def __repr__(self) -> str:
        return f"FMA({self.src0}, {self.src1}, {self.src2}, clamp={self.clamp}, omod={self.omod.name})"


class TestFmaF64SymbolicAST(unittest.TestCase):
    """Test suite for symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_validation(self):
        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7]")

    def test_ast_with_modifiers_and_disasm(self):
        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2, neg=True, abs_=True),
            src1=ASTOperand.sgpr(4),
            src2=ASTOperand.inline_const(1.0),
            clamp=True,
            omod=OMod.MUL2,
        )
        inst.validate()
        self.assertEqual(inst.disasm(), "v_fma_f64 v[0:1], -|v[2:3]|, s[4:5], 1.0 mul:2 clamp")

    def test_ast_validation_errors(self):
        # Invalid destination kind
        bad_dst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.sgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(TypeError):
            bad_dst.validate()

        # Destination has modifiers (illegal in RDNA2)
        bad_dst_mod = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0, neg=True),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(ValueError):
            bad_dst_mod.validate()

        # Destination out of bounds
        bad_index = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(256),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        with self.assertRaises(ValueError):
            bad_index.validate()

    def test_typed_expressions_and_evaluation(self):
        # Symbolic: (a * b) + c
        expr = FMAExpr(
            VariableExpr("a"),
            VariableExpr("b"),
            VariableExpr("c"),
        )
        env = {"a": 2.5, "b": 4.0, "c": -3.0}
        self.assertEqual(expr.evaluate(env), 7.0)

    def test_typed_expression_constant_folding(self):
        # FMA(Const(2.0), Const(3.0), Const(4.0)) -> Const(10.0)
        c_expr = FMAExpr(
            ConstantExpr(2.0),
            ConstantExpr(3.0),
            ConstantExpr(4.0),
        )
        folded = c_expr.simplify()
        self.assertIsInstance(folded, ConstantExpr)
        self.assertEqual(folded.value, 10.0)

        # Unary folding
        u_expr = UnaryOpExpr("neg", ConstantExpr(5.5))
        u_folded = u_expr.simplify()
        self.assertIsInstance(u_folded, ConstantExpr)
        self.assertEqual(u_folded.value, -5.5)


class ScalarRegisterFile:
    """Register file for a single scalar lane with 256 VGPRs and 106 SGPRs."""

    def __init__(self) -> None:
        self.vgprs = [0] * 256
        self.sgprs = [0] * 106

    def read_vgpr(self, idx: int) -> int:
        return self.vgprs[idx]

    def write_vgpr(self, idx: int, val: int) -> None:
        self.vgprs[idx] = val & MASK32

    def read_sgpr(self, idx: int) -> int:
        return self.sgprs[idx]

    def write_sgpr(self, idx: int, val: int) -> None:
        self.sgprs[idx] = val & MASK32

    def read_vgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.vgprs[idx], self.vgprs[idx + 1])

    def write_vgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.vgprs[idx] = lo
        self.vgprs[idx + 1] = hi

    def read_vgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.read_vgpr_pair_u64(idx))

    def write_vgpr_pair_f64(self, idx: int, val: float) -> None:
        self.write_vgpr_pair_u64(idx, f64_to_u64(val))

    def read_sgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.sgprs[idx], self.sgprs[idx + 1])

    def write_sgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi

    def read_sgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.read_sgpr_pair_u64(idx))

    def write_sgpr_pair_f64(self, idx: int, val: float) -> None:
        self.write_sgpr_pair_u64(idx, f64_to_u64(val))


class FmaF64ScalarInterpreter:
    """Deterministic scalar instruction interpreter for v_fma_f64."""

    def __init__(self, reg_file: ScalarRegisterFile | None = None) -> None:
        self.reg_file = reg_file if reg_file is not None else ScalarRegisterFile()

    def resolve_source_f64(self, src: ASTOperand) -> float:
        if src.kind == OperandKind.VGPR_PAIR:
            val = self.reg_file.read_vgpr_pair_f64(src.index)
        elif src.kind == OperandKind.SGPR_PAIR:
            val = self.reg_file.read_sgpr_pair_f64(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val = float(src.value) if src.value is not None else 0.0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        # Apply source modifiers
        u = apply_src_mods_u64(f64_to_u64(val), neg=src.neg, abs_=src.abs_)
        return u64_to_f64(u)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        s0 = self.resolve_source_f64(inst.src0)
        s1 = self.resolve_source_f64(inst.src1)
        s2 = self.resolve_source_f64(inst.src2)

        res = ref_fma_f64(s0, s1, s2, clamp=inst.clamp, omod=inst.omod.value)
        self.reg_file.write_vgpr_pair_f64(inst.dst.index, res)


class TestFmaF64ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.rf = ScalarRegisterFile()
        self.interp = FmaF64ScalarInterpreter(self.rf)

    def test_scalar_interpreter_vgprs(self):
        # v[0:1] = v[2:3] * v[4:5] + v[6:7]
        self.rf.write_vgpr_pair_f64(2, 3.0)
        self.rf.write_vgpr_pair_f64(4, 4.0)
        self.rf.write_vgpr_pair_f64(6, 5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_pair_f64(0), 17.0)

    def test_scalar_interpreter_sgpr_and_constants(self):
        # v[0:1] = s[0:1] * v[2:3] + inline_const(10.0)
        self.rf.write_sgpr_pair_f64(0, 2.5)
        self.rf.write_vgpr_pair_f64(2, 4.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.sgpr(0),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.inline_const(10.0),
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_pair_f64(0), 20.0)

    def test_scalar_interpreter_modifiers(self):
        # v[0:1] = -|v[2:3]| * v[4:5] + 5.0 with clamp
        # v[2:3] = -2.0 -> abs -> 2.0 -> neg -> -2.0
        # -2.0 * 3.0 + 5.0 = -1.0 -> clamp -> 0.0
        self.rf.write_vgpr_pair_f64(2, -2.0)
        self.rf.write_vgpr_pair_f64(4, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2, neg=True, abs_=True),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.inline_const(5.0),
            clamp=True,
        )
        self.interp.execute(inst)
        self.assertEqual(self.rf.read_vgpr_pair_f64(0), 0.0)

    def test_scalar_interpreter_deterministic_repeatability(self):
        # Verify that repeating operations produces identical bit patterns
        self.rf.write_vgpr_pair_f64(2, 1.23456789012345)
        self.rf.write_vgpr_pair_f64(4, 9.87654321098765)
        self.rf.write_vgpr_pair_f64(6, -3.14159265358979)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        self.interp.execute(inst)
        res1 = self.rf.read_vgpr_pair_u64(0)

        self.interp.execute(inst)
        res2 = self.rf.read_vgpr_pair_u64(0)
        self.assertEqual(res1, res2)


class SIMDWaveState:
    """Architectural model of a SIMD wavefront (Wave32 or Wave64) with EXEC masking."""

    def __init__(self, wave_size: int = 64) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self._exec_mask = (1 << wave_size) - 1
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.sgprs = [0] * 106

    @property
    def exec_mask(self) -> int:
        return self._exec_mask

    @exec_mask.setter
    def exec_mask(self, val: int) -> None:
        mask_limit = (1 << self.wave_size) - 1
        self._exec_mask = val & mask_limit

    @property
    def exec_lo(self) -> int:
        return self._exec_mask & MASK32

    @exec_lo.setter
    def exec_lo(self, val: int) -> None:
        self._exec_mask = (self._exec_mask & ~MASK32) | (val & MASK32)

    @property
    def exec_hi(self) -> int:
        return (self._exec_mask >> 32) & MASK32

    @exec_hi.setter
    def exec_hi(self, val: int) -> None:
        if self.wave_size == 32:
            raise ValueError("Cannot set exec_hi on Wave32")
        self._exec_mask = (self._exec_mask & MASK32) | ((val & MASK32) << 32)

    def set_lane_vgpr_pair_u64(self, lane: int, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.vgprs[lane][idx] = lo
        self.vgprs[lane][idx + 1] = hi

    def get_lane_vgpr_pair_u64(self, lane: int, idx: int) -> int:
        return dwords_to_u64(self.vgprs[lane][idx], self.vgprs[lane][idx + 1])

    def set_lane_vgpr_pair_f64(self, lane: int, idx: int, val: float) -> None:
        self.set_lane_vgpr_pair_u64(lane, idx, f64_to_u64(val))

    def get_lane_vgpr_pair_f64(self, lane: int, idx: int) -> float:
        return u64_to_f64(self.get_lane_vgpr_pair_u64(lane, idx))

    def set_sgpr_pair_u64(self, idx: int, val: int) -> None:
        lo, hi = u64_to_dwords(val)
        self.sgprs[idx] = lo
        self.sgprs[idx + 1] = hi

    def get_sgpr_pair_u64(self, idx: int) -> int:
        return dwords_to_u64(self.sgprs[idx], self.sgprs[idx + 1])

    def set_sgpr_pair_f64(self, idx: int, val: float) -> None:
        self.set_sgpr_pair_u64(idx, f64_to_u64(val))

    def get_sgpr_pair_f64(self, idx: int) -> float:
        return u64_to_f64(self.get_sgpr_pair_u64(idx))

    def _resolve_operand_f64(self, src: ASTOperand, lane: int) -> float:
        if src.kind == OperandKind.VGPR_PAIR:
            val = self.get_lane_vgpr_pair_f64(lane, src.index)
        elif src.kind == OperandKind.SGPR_PAIR:
            val = self.get_sgpr_pair_f64(src.index)
        elif src.kind in (OperandKind.INLINE_CONST, OperandKind.LITERAL_CONST):
            val = float(src.value) if src.value is not None else 0.0
        else:
            raise ValueError(f"Unsupported operand kind: {src.kind}")

        u = apply_src_mods_u64(f64_to_u64(val), neg=src.neg, abs_=src.abs_)
        return u64_to_f64(u)

    def execute(self, inst: ASTInstruction) -> None:
        inst.validate()
        for lane in range(self.wave_size):
            if (self._exec_mask >> lane) & 1:
                s0 = self._resolve_operand_f64(inst.src0, lane)
                s1 = self._resolve_operand_f64(inst.src1, lane)
                s2 = self._resolve_operand_f64(inst.src2, lane)
                res = ref_fma_f64(s0, s1, s2, clamp=inst.clamp, omod=inst.omod.value)
                self.set_lane_vgpr_pair_f64(lane, inst.dst.index, res)


class TestFmaF64SIMDMultiLane(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    CANARY_OLD = 0xDEADBEEFCAFEBABE

    def test_wave32_full_exec(self):
        wave = SIMDWaveState(wave_size=32)
        wave.exec_mask = 0xFFFFFFFF
        for lane in range(32):
            wave.set_lane_vgpr_pair_f64(lane, 2, float(lane))
            wave.set_lane_vgpr_pair_f64(lane, 4, 2.0)
            wave.set_lane_vgpr_pair_f64(lane, 6, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        wave.execute(inst)

        for lane in range(32):
            expected = float(lane * 2 + 1)
            self.assertEqual(wave.get_lane_vgpr_pair_f64(lane, 0), expected)

    def test_wave64_full_exec(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0xFFFFFFFF
        for lane in range(64):
            wave.set_lane_vgpr_pair_f64(lane, 2, float(lane) * 0.5)
            wave.set_lane_vgpr_pair_f64(lane, 4, 4.0)
            wave.set_lane_vgpr_pair_f64(lane, 6, -10.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        wave.execute(inst)

        for lane in range(64):
            expected = float(lane * 0.5) * 4.0 - 10.0
            self.assertEqual(wave.get_lane_vgpr_pair_f64(lane, 0), expected)

    def test_wave64_divergence_exec_masking(self):
        wave = SIMDWaveState(wave_size=64)
        # Even lanes in low 32, odd lanes in high 32
        wave.exec_lo = 0x55555555
        wave.exec_hi = 0xAAAAAAAA

        for lane in range(64):
            wave.set_lane_vgpr_pair_u64(lane, 0, self.CANARY_OLD)
            wave.set_lane_vgpr_pair_f64(lane, 2, 3.0)
            wave.set_lane_vgpr_pair_f64(lane, 4, 5.0)
            wave.set_lane_vgpr_pair_f64(lane, 6, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        wave.execute(inst)

        for lane in range(64):
            is_active = bool((wave.exec_mask >> lane) & 1)
            if is_active:
                self.assertEqual(wave.get_lane_vgpr_pair_f64(lane, 0), 17.0)
            else:
                self.assertEqual(wave.get_lane_vgpr_pair_u64(lane, 0), self.CANARY_OLD)

    def test_wave64_halves_masking(self):
        wave = SIMDWaveState(wave_size=64)
        # Only low 32 lanes active
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0x00000000

        for lane in range(64):
            wave.set_lane_vgpr_pair_u64(lane, 0, self.CANARY_OLD)
            wave.set_lane_vgpr_pair_f64(lane, 2, 2.0)
            wave.set_lane_vgpr_pair_f64(lane, 4, 2.0)
            wave.set_lane_vgpr_pair_f64(lane, 6, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        wave.execute(inst)

        for lane in range(32):
            self.assertEqual(wave.get_lane_vgpr_pair_f64(lane, 0), 6.0)
        for lane in range(32, 64):
            self.assertEqual(wave.get_lane_vgpr_pair_u64(lane, 0), self.CANARY_OLD)

    def test_single_active_lane(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 1 << 17  # only lane 17

        for lane in range(64):
            wave.set_lane_vgpr_pair_u64(lane, 0, self.CANARY_OLD)
            wave.set_lane_vgpr_pair_f64(lane, 2, 7.0)
            wave.set_lane_vgpr_pair_f64(lane, 4, 3.0)
            wave.set_lane_vgpr_pair_f64(lane, 6, 1.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        wave.execute(inst)

        for lane in range(64):
            if lane == 17:
                self.assertEqual(wave.get_lane_vgpr_pair_f64(lane, 0), 22.0)
            else:
                self.assertEqual(wave.get_lane_vgpr_pair_u64(lane, 0), self.CANARY_OLD)

    def test_zero_exec_preserves_all_lanes(self):
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 0

        for lane in range(64):
            wave.set_lane_vgpr_pair_u64(lane, 0, self.CANARY_OLD)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.inline_const(1.0),
            src1=ASTOperand.inline_const(1.0),
            src2=ASTOperand.inline_const(1.0),
        )
        wave.execute(inst)

        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr_pair_u64(lane, 0), self.CANARY_OLD)


class TestFmaF64RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_accumulator_aliasing_dst_equals_src2(self):
        # Accumulator in-place: v[0:1] = v[2:3] * v[4:5] + v[0:1]
        rf = ScalarRegisterFile()
        interp = FmaF64ScalarInterpreter(rf)

        rf.write_vgpr_pair_f64(0, 10.0)  # accumulator initial value
        rf.write_vgpr_pair_f64(2, 2.0)
        rf.write_vgpr_pair_f64(4, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(0),
        )

        # Iteration 1: 10.0 + 6.0 = 16.0
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_pair_f64(0), 16.0)

        # Iteration 2: 16.0 + 6.0 = 22.0
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_pair_f64(0), 22.0)

        # Iteration 3: 22.0 + 6.0 = 28.0
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_pair_f64(0), 28.0)

    def test_multiplicand_aliasing_dst_equals_src0(self):
        # v[0:1] = v[0:1] * v[2:3] + v[4:5]
        rf = ScalarRegisterFile()
        interp = FmaF64ScalarInterpreter(rf)

        rf.write_vgpr_pair_f64(0, 5.0)
        rf.write_vgpr_pair_f64(2, 2.0)
        rf.write_vgpr_pair_f64(4, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(2),
            src2=ASTOperand.vgpr(4),
        )
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_pair_f64(0), 13.0)

    def test_multiplier_aliasing_dst_equals_src1(self):
        # v[0:1] = v[2:3] * v[0:1] + v[4:5]
        rf = ScalarRegisterFile()
        interp = FmaF64ScalarInterpreter(rf)

        rf.write_vgpr_pair_f64(0, 3.0)
        rf.write_vgpr_pair_f64(2, 4.0)
        rf.write_vgpr_pair_f64(4, -5.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(4),
        )
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_pair_f64(0), 7.0)

    def test_self_fma_dst_equals_all_sources(self):
        # v[0:1] = v[0:1] * v[0:1] + v[0:1] -> x^2 + x
        rf = ScalarRegisterFile()
        interp = FmaF64ScalarInterpreter(rf)

        rf.write_vgpr_pair_f64(0, 3.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(0),
            src1=ASTOperand.vgpr(0),
            src2=ASTOperand.vgpr(0),
        )
        interp.execute(inst)
        self.assertEqual(rf.read_vgpr_pair_f64(0), 12.0)

    def test_partial_dword_misaligned_aliasing(self):
        # Misaligned overlap: dst is v[1:2], reading src0 from v[0:1]
        # v0 and v1 encode 2.0 (f64: lo=0, hi=0x40000000)
        rf = ScalarRegisterFile()
        interp = FmaF64ScalarInterpreter(rf)

        rf.write_vgpr_pair_f64(0, 2.0)   # v0=0, v1=0x40000000
        rf.write_vgpr_pair_f64(4, 5.0)   # src1
        rf.write_vgpr_pair_f64(6, 1.0)   # src2

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(1),       # dst writes v1 and v2
            src0=ASTOperand.vgpr(0),      # src0 reads v0 and v1
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(6),
        )
        interp.execute(inst)
        # Result = 2.0 * 5.0 + 1.0 = 11.0, committed to v[1:2]
        self.assertEqual(rf.read_vgpr_pair_f64(1), 11.0)
        # v0 remains untouched
        self.assertEqual(rf.read_vgpr(0), 0)

    def test_simd_multi_lane_aliased_accumulation(self):
        # Multi-lane wave with dst == src2 accumulation under active/inactive EXEC
        wave = SIMDWaveState(wave_size=64)
        wave.exec_lo = 0xFF00FF00
        wave.exec_hi = 0x00FF00FF

        initial_acc = 100.0
        for lane in range(64):
            wave.set_lane_vgpr_pair_f64(lane, 0, initial_acc)
            wave.set_lane_vgpr_pair_f64(lane, 2, float(lane))
            wave.set_lane_vgpr_pair_f64(lane, 4, 2.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.vgpr(2),
            src1=ASTOperand.vgpr(4),
            src2=ASTOperand.vgpr(0),
        )
        wave.execute(inst)

        for lane in range(64):
            is_active = bool((wave.exec_mask >> lane) & 1)
            if is_active:
                expected = float(lane * 2) + initial_acc
                self.assertEqual(wave.get_lane_vgpr_pair_f64(lane, 0), expected)
            else:
                self.assertEqual(wave.get_lane_vgpr_pair_f64(lane, 0), initial_acc)


class StatusFlags:
    """IEEE 754 exception flags and architectural condition codes for an operation."""

    def __init__(
        self,
        *,
        invalid: bool = False,
        overflow: bool = False,
        underflow: bool = False,
        inexact: bool = False,
        is_negative: bool = False,
        is_zero: bool = False,
        is_inf: bool = False,
        is_nan: bool = False,
        is_subnormal: bool = False,
    ) -> None:
        self.invalid = invalid
        self.overflow = overflow
        self.underflow = underflow
        self.inexact = inexact
        self.is_negative = is_negative
        self.is_zero = is_zero
        self.is_inf = is_inf
        self.is_nan = is_nan
        self.is_subnormal = is_subnormal

    def __repr__(self) -> str:
        flags = [k for k, v in self.__dict__.items() if v]
        return f"StatusFlags({', '.join(flags)})"


def compute_fma_status_flags(
    s0: float,
    s1: float,
    s2: float,
    *,
    neg0: bool = False,
    neg1: bool = False,
    neg2: bool = False,
    abs0: bool = False,
    abs1: bool = False,
    abs2: bool = False,
    clamp: bool = False,
    omod: int = 0,
) -> tuple[float, StatusFlags]:
    """Compute FMA result along with IEEE 754 status flags and condition codes."""
    u0 = apply_src_mods_u64(f64_to_u64(s0), neg=neg0, abs_=abs0)
    u1 = apply_src_mods_u64(f64_to_u64(s1), neg=neg1, abs_=abs1)
    u2 = apply_src_mods_u64(f64_to_u64(s2), neg=neg2, abs_=abs2)

    v0 = u64_to_f64(u0)
    v1 = u64_to_f64(u1)
    v2 = u64_to_f64(u2)

    invalid = False
    overflow = False
    underflow = False
    inexact = False

    # Check for invalid operation exceptions
    if math.isnan(v0) or math.isnan(v1) or math.isnan(v2):
        # Signaling NaN or NaN arithmetic
        if (u0 & F64_CANONICAL_QNAN == F64_POS_INF) or (u1 & F64_CANONICAL_QNAN == F64_POS_INF) or (u2 & F64_CANONICAL_QNAN == F64_POS_INF):
            invalid = True

    if (math.isinf(v0) and v1 == 0.0) or (v0 == 0.0 and math.isinf(v1)):
        invalid = True

    p_inf = math.isinf(v0) or math.isinf(v1)
    if p_inf and math.isinf(v2):
        p_sign = -1.0 if (math.copysign(1.0, v0) * math.copysign(1.0, v1) < 0) else 1.0
        if p_sign != math.copysign(1.0, v2):
            invalid = True

    res = ref_fma_f64(
        s0, s1, s2,
        neg0=neg0, neg1=neg1, neg2=neg2,
        abs0=abs0, abs1=abs1, abs2=abs2,
        clamp=clamp, omod=omod,
    )

    # Inexact & Overflow/Underflow determination for finite inputs
    if not math.isnan(res) and math.isfinite(v0) and math.isfinite(v1) and math.isfinite(v2):
        exact_frac = Fraction(v0) * Fraction(v1) + Fraction(v2)
        if omod == 1:
            exact_frac *= Fraction(2)
        elif omod == 2:
            exact_frac *= Fraction(4)
        elif omod == 3:
            exact_frac /= Fraction(2)

        if math.isinf(res):
            overflow = True
            inexact = True
        else:
            if Fraction(res) != exact_frac:
                inexact = True
            u_res = f64_to_u64(res)
            # Check subnormal
            exp_bits = (u_res >> 52) & 0x7FF
            mant_bits = u_res & 0x000FFFFFFFFFFFFF
            if exp_bits == 0 and mant_bits != 0:
                underflow = True

    u_res = f64_to_u64(res)
    is_neg = (u_res & F64_NEG_ZERO) != 0
    is_nan = math.isnan(res)
    is_inf = math.isinf(res)
    is_zero = (res == 0.0)
    exp = (u_res >> 52) & 0x7FF
    mant = u_res & 0x000FFFFFFFFFFFFF
    is_subnormal = (exp == 0 and mant != 0)

    flags = StatusFlags(
        invalid=invalid,
        overflow=overflow,
        underflow=underflow,
        inexact=inexact,
        is_negative=is_neg,
        is_zero=is_zero,
        is_inf=is_inf,
        is_nan=is_nan,
        is_subnormal=is_subnormal,
    )
    return res, flags


def evaluate_vcc_predicate(wave: SIMDWaveState, reg_idx: int, predicate: str) -> int:
    """Generate 64-bit VCC mask evaluating condition on reg_idx across all active lanes."""
    vcc = 0
    for lane in range(wave.wave_size):
        if (wave.exec_mask >> lane) & 1:
            val = wave.get_lane_vgpr_pair_f64(lane, reg_idx)
            cond = False
            if predicate == "eq_zero":
                cond = (val == 0.0)
            elif predicate == "ge_zero":
                cond = not math.isnan(val) and val >= 0.0
            elif predicate == "lt_zero":
                cond = not math.isnan(val) and val < 0.0
            elif predicate == "is_nan":
                cond = math.isnan(val)
            elif predicate == "is_finite":
                cond = math.isfinite(val)
            else:
                raise ValueError(f"Unknown predicate: {predicate}")

            if cond:
                vcc |= (1 << lane)
    return vcc


class TestFmaF64StatusFlagsAndConditionCodes(unittest.TestCase):
    """Test suite for IEEE 754 status flags, condition codes, and VCC carry/predicate generation."""

    def test_inexact_flag_detection(self):
        # Exact calculation: 2.0 * 3.0 + 1.0 = 7.0 (exact in float)
        _, flags_exact = compute_fma_status_flags(2.0, 3.0, 1.0)
        self.assertFalse(flags_exact.inexact)

        # Inexact calculation: inputs are exact, but product has term 2**-104 that is rounded
        a = 1.0 + (2.0 ** -52)
        _, flags_inexact = compute_fma_status_flags(a, a, 0.0)
        self.assertTrue(flags_inexact.inexact)

    def test_overflow_flag(self):
        max_f64 = u64_to_f64(F64_MAX_NORMAL)
        res, flags = compute_fma_status_flags(max_f64, 2.0, 0.0)
        self.assertTrue(math.isinf(res))
        self.assertTrue(flags.overflow)
        self.assertTrue(flags.inexact)
        self.assertTrue(flags.is_inf)

    def test_underflow_flag(self):
        min_normal = u64_to_f64(F64_MIN_NORMAL)
        res, flags = compute_fma_status_flags(min_normal, 0.5, 0.0)
        self.assertTrue(flags.is_subnormal)
        self.assertTrue(flags.underflow)

    def test_invalid_operation_flags(self):
        # inf * 0.0 -> invalid
        res, flags1 = compute_fma_status_flags(float("inf"), 0.0, 1.0)
        self.assertTrue(math.isnan(res))
        self.assertTrue(flags1.invalid)
        self.assertTrue(flags1.is_nan)

        # (+inf) * 1.0 + (-inf) -> invalid
        res, flags2 = compute_fma_status_flags(float("inf"), 1.0, float("-inf"))
        self.assertTrue(math.isnan(res))
        self.assertTrue(flags2.invalid)
        self.assertTrue(flags2.is_nan)

    def test_condition_codes_signed_zeros(self):
        # -0.0 * 1.0 + -0.0 -> -0.0
        res, flags = compute_fma_status_flags(-0.0, 1.0, -0.0)
        self.assertTrue(flags.is_zero)
        self.assertTrue(flags.is_negative)

        # 1.0 * 1.0 + -1.0 -> +0.0
        res, flags2 = compute_fma_status_flags(1.0, 1.0, -1.0)
        self.assertTrue(flags2.is_zero)
        self.assertFalse(flags2.is_negative)

    def test_vcc_predicate_mask_generation(self):
        wave = SIMDWaveState(wave_size=64)
        # EXEC active for lanes 0..31
        wave.exec_lo = 0xFFFFFFFF
        wave.exec_hi = 0x00000000

        # Alternate positive and negative numbers in v[0:1]
        for lane in range(64):
            val = 5.0 if (lane % 2 == 0) else -5.0
            wave.set_lane_vgpr_pair_f64(lane, 0, val)

        # Evaluate ge_zero: should match even lanes in 0..31 (0x55555555)
        vcc_ge = evaluate_vcc_predicate(wave, 0, "ge_zero")
        self.assertEqual(vcc_ge, 0x55555555)

        # Evaluate lt_zero: should match odd lanes in 0..31 (0xAAAAAAAA)
        vcc_lt = evaluate_vcc_predicate(wave, 0, "lt_zero")
        self.assertEqual(vcc_lt, 0xAAAAAAAA)

        # High lanes 32..63 are inactive in EXEC, so VCC bits 32..63 must be 0
        self.assertEqual(vcc_ge >> 32, 0)
        self.assertEqual(vcc_lt >> 32, 0)


class TestFmaF64IEEE754Boundaries(unittest.TestCase):
    """Test suite for IEEE 754 boundary extremes, subnormals, and NaN handling."""

    def test_subnormal_operands_and_results(self):
        sub_min = u64_to_f64(F64_SUB_MIN)  # 2**-1074
        min_norm = u64_to_f64(F64_MIN_NORMAL)  # 2**-1022

        # subnormal * 1.0 + 0.0 retains subnormal
        res = ref_fma_f64_u64(F64_SUB_MIN, F64_ONE, F64_POS_ZERO)
        self.assertEqual(res, F64_SUB_MIN)

        # subnormal * 2.0 + 0.0 scales subnormal mantissa
        res2 = ref_fma_f64_u64(F64_SUB_MIN, F64_TWO, F64_POS_ZERO)
        self.assertEqual(res2, 0x0000000000000002)

        # normal * 0.5 produces maximum subnormal
        res_sub = ref_fma_f64_u64(F64_MIN_NORMAL, F64_HALF, F64_POS_ZERO)
        self.assertEqual(res_sub, 0x0008000000000000)

        # subnormal underflow rounds to zero
        # sub_min * 0.25 rounds to 0.0
        u_quarter = f64_to_u64(0.25)
        res_zero = ref_fma_f64_u64(F64_SUB_MIN, u_quarter, F64_POS_ZERO)
        self.assertEqual(res_zero, F64_POS_ZERO)

    def test_signed_zeros_bitwise_representation(self):
        # +0.0 * +0.0 + +0.0 = +0.0 (0x0000000000000000)
        res_pp = ref_fma_f64_u64(F64_POS_ZERO, F64_POS_ZERO, F64_POS_ZERO)
        self.assertEqual(res_pp, F64_POS_ZERO)

        # -0.0 * 1.0 + -0.0 = -0.0 (0x8000000000000000)
        res_nn = ref_fma_f64_u64(F64_NEG_ZERO, F64_ONE, F64_NEG_ZERO)
        self.assertEqual(res_nn, F64_NEG_ZERO)

        # -0.0 * -1.0 + +0.0 = +0.0
        res_np = ref_fma_f64_u64(F64_NEG_ZERO, F64_NEG_ONE, F64_POS_ZERO)
        self.assertEqual(res_np, F64_POS_ZERO)

        # Cancellation: 1.0 * 1.0 + -1.0 = +0.0 in round-to-nearest
        res_cancel = ref_fma_f64_u64(F64_ONE, F64_ONE, F64_NEG_ONE)
        self.assertEqual(res_cancel, F64_POS_ZERO)

    def test_infinities_propagation_and_invalids(self):
        # (+inf) * 2.0 + 10.0 = +inf
        u_ten = f64_to_u64(10.0)
        res_inf = ref_fma_f64_u64(F64_POS_INF, F64_TWO, u_ten)
        self.assertEqual(res_inf, F64_POS_INF)

        # (-inf) * 2.0 + 10.0 = -inf
        res_ninf = ref_fma_f64_u64(F64_NEG_INF, F64_TWO, u_ten)
        self.assertEqual(res_ninf, F64_NEG_INF)

        # (+inf) * -2.0 + 10.0 = -inf
        res_ninf2 = ref_fma_f64_u64(F64_POS_INF, f64_to_u64(-2.0), u_ten)
        self.assertEqual(res_ninf2, F64_NEG_INF)

        # (+inf) * 0.0 + 1.0 -> NaN
        res_nan1 = ref_fma_f64(float("inf"), 0.0, 1.0)
        self.assertTrue(math.isnan(res_nan1))

        # (+inf) * 1.0 + (-inf) -> NaN
        res_nan2 = ref_fma_f64(float("inf"), 1.0, float("-inf"))
        self.assertTrue(math.isnan(res_nan2))

    def test_nan_handling_and_clamp_mapping(self):
        # NaN propagation: any NaN input produces NaN
        self.assertTrue(math.isnan(ref_fma_f64(float("nan"), 1.0, 2.0)))
        self.assertTrue(math.isnan(ref_fma_f64(1.0, float("nan"), 2.0)))
        self.assertTrue(math.isnan(ref_fma_f64(1.0, 2.0, float("nan"))))

        # RDNA2 clamp modifier: clamp maps NaN to +0.0 (0x0000000000000000)
        clamped_nan = ref_fma_f64(float("nan"), 1.0, 2.0, clamp=True)
        self.assertEqual(clamped_nan, 0.0)
        self.assertEqual(f64_to_u64(clamped_nan), F64_POS_ZERO)

        # Source modifiers on NaN bit patterns
        # -NaN inverts sign bit
        qnan_neg = apply_src_mods_u64(F64_CANONICAL_QNAN, neg=True)
        self.assertEqual(qnan_neg, F64_CANONICAL_QNAN | F64_NEG_ZERO)

        # | -NaN | clears sign bit
        qnan_abs = apply_src_mods_u64(F64_CANONICAL_QNAN | F64_NEG_ZERO, abs_=True)
        self.assertEqual(qnan_abs, F64_CANONICAL_QNAN)

    def test_boundary_max_normal_extremes(self):
        max_f64 = u64_to_f64(F64_MAX_NORMAL)

        # MAX_NORMAL * 1.0 + 0.0 == MAX_NORMAL
        res_max = ref_fma_f64(max_f64, 1.0, 0.0)
        self.assertEqual(res_max, max_f64)

        # MAX_NORMAL * 1.0 - ULP is less than MAX_NORMAL
        # 1 ULP at 2**1024 is 2**(1024 - 53) = 2**971
        ulp_max = 2.0 ** 971
        res_minus_ulp = ref_fma_f64(max_f64, 1.0, -ulp_max)
        self.assertLess(res_minus_ulp, max_f64)

        # MAX_NORMAL * 2.0 overflows to +inf
        res_overflow = ref_fma_f64(max_f64, 2.0, 0.0)
        self.assertEqual(res_overflow, float("inf"))


def parse_operand_asm(op_str: str) -> ASTOperand:
    """Parse a single operand string with potential neg/abs modifiers."""
    op_str = op_str.strip()
    neg = False
    abs_ = False

    if op_str.startswith("-"):
        neg = True
        op_str = op_str[1:].strip()

    if op_str.startswith("|") and op_str.endswith("|"):
        abs_ = True
        op_str = op_str[1:-1].strip()

    if op_str.startswith("v[") and op_str.endswith("]"):
        inner = op_str[2:-1]
        parts = inner.split(":")
        if len(parts) != 2:
            raise ValueError(f"Malformed VGPR pair: {op_str}")
        lo = int(parts[0])
        hi = int(parts[1])
        if hi != lo + 1:
            raise ValueError(f"VGPR pair must be consecutive registers [n:n+1], got {op_str}")
        return ASTOperand.vgpr(lo, neg=neg, abs_=abs_)

    if op_str.startswith("s[") and op_str.endswith("]"):
        inner = op_str[2:-1]
        parts = inner.split(":")
        if len(parts) != 2:
            raise ValueError(f"Malformed SGPR pair: {op_str}")
        lo = int(parts[0])
        hi = int(parts[1])
        if hi != lo + 1:
            raise ValueError(f"SGPR pair must be consecutive registers [n:n+1], got {op_str}")
        return ASTOperand.sgpr(lo, neg=neg, abs_=abs_)

    # Try parsing constant
    try:
        val = float(op_str)
        return ASTOperand.inline_const(val, neg=neg, abs_=abs_)
    except ValueError:
        raise ValueError(f"Unrecognized operand syntax: {op_str}")


def parse_fma_f64_asm(line: str) -> ASTInstruction:
    """Parse a complete v_fma_f64 assembly instruction string."""
    line = line.strip()
    if not line:
        raise ValueError("Empty assembly line")

    tokens = line.split(None, 1)
    if not tokens or tokens[0] != Opcode.V_FMA_F64.value:
        raise ValueError(f"Expected opcode '{Opcode.V_FMA_F64.value}', got '{tokens[0] if tokens else ''}'")

    if len(tokens) < 2:
        raise ValueError("Missing operands in assembly line")

    rest = tokens[1].strip()
    # Separate operands from optional trailing modifiers (clamp, mul:2, etc.)
    # Operands are comma-separated
    parts = [p.strip() for p in rest.split(",")]
    if len(parts) < 4:
        raise ValueError(f"Expected at least 4 comma-separated fields, got {len(parts)}")

    dst_str = parts[0]
    src0_str = parts[1]
    src1_str = parts[2]
    # Fourth part contains src2 and optional modifiers
    last_tokens = parts[3].split()
    if not last_tokens:
        raise ValueError("Missing third source operand")

    src2_str = last_tokens[0]
    mod_tokens = last_tokens[1:]

    clamp = False
    omod = OMod.NONE

    for mod in mod_tokens:
        if mod == "clamp":
            clamp = True
        elif mod == "mul:2":
            omod = OMod.MUL2
        elif mod == "mul:4":
            omod = OMod.MUL4
        elif mod == "div:2":
            omod = OMod.DIV2
        else:
            raise ValueError(f"Unknown instruction modifier: {mod}")

    dst = parse_operand_asm(dst_str)
    src0 = parse_operand_asm(src0_str)
    src1 = parse_operand_asm(src1_str)
    src2 = parse_operand_asm(src2_str)

    inst = ASTInstruction(
        opcode=Opcode.V_FMA_F64,
        dst=dst,
        src0=src0,
        src1=src1,
        src2=src2,
        clamp=clamp,
        omod=omod,
    )
    inst.validate()
    return inst


class TestFmaF64NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, and mutation detection."""

    def test_syntax_rejection_malformed_instructions(self):
        # Empty line
        with self.assertRaises(ValueError):
            parse_fma_f64_asm("")

        # Wrong opcode
        with self.assertRaises(ValueError):
            parse_fma_f64_asm("v_add_f64 v[0:1], v[2:3], v[4:5], v[6:7]")

        # Missing operands
        with self.assertRaises(ValueError):
            parse_fma_f64_asm("v_fma_f64 v[0:1], v[2:3], v[4:5]")

        # Destination is SGPR
        with self.assertRaises(TypeError):
            parse_fma_f64_asm("v_fma_f64 s[0:1], v[2:3], v[4:5], v[6:7]")

        # Destination has negative modifier
        with self.assertRaises(ValueError):
            parse_fma_f64_asm("v_fma_f64 -v[0:1], v[2:3], v[4:5], v[6:7]")

        # Non-consecutive register pair
        with self.assertRaises(ValueError):
            parse_fma_f64_asm("v_fma_f64 v[0:2], v[2:3], v[4:5], v[6:7]")

        # Out of bounds register index
        with self.assertRaises(ValueError):
            parse_fma_f64_asm("v_fma_f64 v[256:257], v[2:3], v[4:5], v[6:7]")

        # Unknown modifier
        with self.assertRaises(ValueError):
            parse_fma_f64_asm("v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7] clamp_invalid")

    def test_mutation_detection_unfused_double_rounding(self):
        # Mutation 1: Replace fused multiply-add with sequential unfused rounding
        def mutated_unfused_fma(s0: float, s1: float, s2: float) -> float:
            # Unfused: rounds intermediate product to binary64
            prod = s0 * s1
            return prod + s2

        a = 1.0 + (2.0 ** -52)
        b = 1.0 - (2.0 ** -52)
        c = -1.0

        oracle_result = ref_fma_f64(a, b, c)
        mutated_result = mutated_unfused_fma(a, b, c)

        # Mutation must be detected and distinguished by test oracle!
        self.assertNotEqual(oracle_result, mutated_result)
        self.assertEqual(mutated_result, 0.0)
        self.assertEqual(oracle_result, -(2.0 ** -104))

    def test_mutation_detection_inverted_clamp_bounds(self):
        # Mutation 2: Inverted clamp bounds [1.0, 0.0]
        def mutated_clamp(val: float) -> float:
            return min(max(val, 1.0), 0.0)

        # For val = 0.5, correct clamp returns 0.5. Inverted clamp returns 0.0.
        correct = apply_clamp_f64(0.5)
        mutated = mutated_clamp(0.5)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_nan_clamp_omission(self):
        # Mutation 3: Clamp fails to map NaN to +0.0
        def mutated_clamp_nan(val: float) -> float:
            if val < 0.0:
                return 0.0
            if val > 1.0:
                return 1.0
            return val  # leaves NaN untouched

        correct = apply_clamp_f64(float("nan"))
        mutated = mutated_clamp_nan(float("nan"))
        self.assertEqual(correct, 0.0)
        self.assertTrue(math.isnan(mutated))
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_exec_mask_leak(self):
        # Mutation 4: Wave executor ignores EXEC mask and mutates inactive lanes
        wave = SIMDWaveState(wave_size=64)
        wave.exec_mask = 0  # all lanes inactive
        old_val = 0x1122334455667788
        for lane in range(64):
            wave.set_lane_vgpr_pair_u64(lane, 0, old_val)

        # Simulating buggy execution that writes even when EXEC bit is 0
        def buggy_execute(wave: SIMDWaveState, inst: ASTInstruction) -> None:
            for lane in range(wave.wave_size):
                # Bug: ignoring self.exec_mask check
                wave.set_lane_vgpr_pair_f64(lane, inst.dst.index, 999.0)

        inst = ASTInstruction(
            opcode=Opcode.V_FMA_F64,
            dst=ASTOperand.vgpr(0),
            src0=ASTOperand.inline_const(1.0),
            src1=ASTOperand.inline_const(1.0),
            src2=ASTOperand.inline_const(1.0),
        )

        # Check correct wave behavior
        wave.execute(inst)
        for lane in range(64):
            self.assertEqual(wave.get_lane_vgpr_pair_u64(lane, 0), old_val)

        # Check that mutation modifies lane and is detected
        buggy_execute(wave, inst)
        self.assertNotEqual(wave.get_lane_vgpr_pair_u64(0, 0), old_val)


if __name__ == "__main__":
    unittest.main()
