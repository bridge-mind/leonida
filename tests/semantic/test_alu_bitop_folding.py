from __future__ import annotations

"""Regression and semantic tests for bitwise Boolean algebra and ALU bitop folding.

Features covered:
- RDNA2 bitwise Boolean logic folding and canonicalization:
  - Idempotence: x & x == x, x | x == x.
  - Annihilator (dominance): x & 0 == 0, x | 0xFFFFFFFF == 0xFFFFFFFF.
  - Identity: x & 0xFFFFFFFF == x, x | 0 == x, x ^ 0 == x.
  - Self-inverse: x ^ x == 0.
  - Involution: ~(~x) == x.
  - Absorption: x | (x & y) == x, x & (x | y) == x.
  - Commutativity, associativity, distributivity, De Morgan's laws, and complementarity.
- Instruction interpreter modeling constant folding and canonicalization:
  - AST representation of bitwise expressions (ConstExpr, RegExpr, UnaryExpr, BinaryExpr).
  - Multi-instruction ALU folding sequences: v_and_b32, v_or_b32, v_xor_b32, v_not_b32.
  - Symbolic simplification and concrete register state execution.
  - Validation against SPIR-V translation and mini-interpreter execution.
- Multi-lane SIMD divergence under EXEC mask:
  - Wavefront (Wave32) vector register execution modeling.
  - Inactive lane preservation (lanes with EXEC bit == 0 are strictly preserved).
  - Divergent execution masks (all-active, all-inactive, checkerboard, halves, single-lane).
  - Multi-instruction ALU execution across divergent lanes.
- Exhaustive truth tables and composite Boolean functions:
  - 1-bit full truth tables (AND, OR, XOR, NOT, NAND, NOR, XNOR, ANDN2, ORN2).
  - 32-bit sliced independent bit position validation.
  - 3-input Boolean functions (majority, multiplexer, parity/xor3).
- Negative controls and fault rejection:
  - Non-identities, non-cancellation, non-annihilation.
  - Inactive lane isolation negative control.
  - Parser and interpreter syntax fault rejection.
- Translator constant folding for shift/bfe/align operands:
  - Constant shift count modulo 32 mask elimination.
  - Constant v_bfe_u32/i32 offset and width extraction directly to OpBitField[US]Extract.
  - Constant v_alignbyte_b32 and v_alignbit_b32 zero/nonzero folding.
  - dedupe_types early termination and type substitution.
"""

import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

MASK32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)

CANONICAL_TEST_VECTORS = [
    0x00000000,
    0x00000001,
    0x00000002,
    0x0000007F,
    0x00000080,
    0x000000FF,
    0x00000100,
    0x00007FFF,
    0x00008000,
    0x0000FFFF,
    0x00010000,
    0x00FFFFFF,
    0x01000000,
    0x55555555,
    0xAAAAAAAA,
    0x33333333,
    0xCCCCCCCC,
    0x0F0F0F0F,
    0xF0F0F0F0,
    0x00FF00FF,
    0xFF00FF00,
    0x0000FFFF,
    0xFFFF0000,
    0x12345678,
    0x9ABCDEF0,
    0xDEADBEEF,
    0xCAFEBABE,
    0x80000000,
    0x7FFFFFFF,
    0xFFFFFFFE,
    0xFFFFFFFF,
]

SINGLE_BIT_SET = [1 << i for i in range(32)]
SINGLE_BIT_CLR = [(~(1 << i)) & MASK32 for i in range(32)]


def ref_u32(val: int) -> int:
    """Normalize integer to 32-bit unsigned domain."""
    return val & MASK32


def ref_and(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for v_and_b32 / s_and_b32."""
    return (a & b) & MASK32


def ref_or(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for v_or_b32 / s_or_b32."""
    return (a | b) & MASK32


def ref_xor(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for v_xor_b32 / s_xor_b32."""
    return (a ^ b) & MASK32


def ref_not(a: int) -> int:
    """Plain-Python 32-bit reference for v_not_b32 / s_not_b32."""
    return (~a) & MASK32


def ref_andn2(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_andn2_b32 (a & ~b)."""
    return (a & (~b & MASK32)) & MASK32


def ref_orn2(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_orn2_b32 (a | ~b)."""
    return (a | (~b & MASK32)) & MASK32


def ref_nand(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_nand_b32 (~(a & b))."""
    return (~(a & b)) & MASK32


def ref_nor(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_nor_b32 (~(a | b))."""
    return (~(a | b)) & MASK32


def ref_xnor(a: int, b: int) -> int:
    """Plain-Python 32-bit reference for s_xnor_b32 (~(a ^ b))."""
    return (~(a ^ b)) & MASK32


class BitwiseAlgebraOracle:
    """Formal mathematical oracle verifying Boolean algebraic identities."""

    @staticmethod
    def check_idempotence_and(x: int) -> bool:
        return ref_and(x, x) == ref_u32(x)

    @staticmethod
    def check_idempotence_or(x: int) -> bool:
        return ref_or(x, x) == ref_u32(x)

    @staticmethod
    def check_annihilator_and(x: int) -> bool:
        return ref_and(x, 0) == 0

    @staticmethod
    def check_annihilator_or(x: int) -> bool:
        return ref_or(x, MASK32) == MASK32

    @staticmethod
    def check_identity_and(x: int) -> bool:
        return ref_and(x, MASK32) == ref_u32(x)

    @staticmethod
    def check_identity_or(x: int) -> bool:
        return ref_or(x, 0) == ref_u32(x)

    @staticmethod
    def check_identity_xor(x: int) -> bool:
        return ref_xor(x, 0) == ref_u32(x)

    @staticmethod
    def check_self_inverse_xor(x: int) -> bool:
        return ref_xor(x, x) == 0

    @staticmethod
    def check_involution_not(x: int) -> bool:
        return ref_not(ref_not(x)) == ref_u32(x)

    @staticmethod
    def check_absorption_or_and(x: int, y: int) -> bool:
        return ref_or(x, ref_and(x, y)) == ref_u32(x)

    @staticmethod
    def check_absorption_and_or(x: int, y: int) -> bool:
        return ref_and(x, ref_or(x, y)) == ref_u32(x)

    @staticmethod
    def check_commutativity_and(x: int, y: int) -> bool:
        return ref_and(x, y) == ref_and(y, x)

    @staticmethod
    def check_commutativity_or(x: int, y: int) -> bool:
        return ref_or(x, y) == ref_or(y, x)

    @staticmethod
    def check_commutativity_xor(x: int, y: int) -> bool:
        return ref_xor(x, y) == ref_xor(y, x)

    @staticmethod
    def check_associativity_and(x: int, y: int, z: int) -> bool:
        return ref_and(ref_and(x, y), z) == ref_and(x, ref_and(y, z))

    @staticmethod
    def check_associativity_or(x: int, y: int, z: int) -> bool:
        return ref_or(ref_or(x, y), z) == ref_or(x, ref_or(y, z))

    @staticmethod
    def check_associativity_xor(x: int, y: int, z: int) -> bool:
        return ref_xor(ref_xor(x, y), z) == ref_xor(x, ref_xor(y, z))

    @staticmethod
    def check_distributivity_and_over_or(x: int, y: int, z: int) -> bool:
        return ref_and(x, ref_or(y, z)) == ref_or(ref_and(x, y), ref_and(x, z))

    @staticmethod
    def check_distributivity_or_over_and(x: int, y: int, z: int) -> bool:
        return ref_or(x, ref_and(y, z)) == ref_and(ref_or(x, y), ref_or(x, z))

    @staticmethod
    def check_demorgan_and(x: int, y: int) -> bool:
        return ref_not(ref_and(x, y)) == ref_or(ref_not(x), ref_not(y))

    @staticmethod
    def check_demorgan_or(x: int, y: int) -> bool:
        return ref_not(ref_or(x, y)) == ref_and(ref_not(x), ref_not(y))

    @staticmethod
    def check_complement_and(x: int) -> bool:
        return ref_and(x, ref_not(x)) == 0

    @staticmethod
    def check_complement_or(x: int) -> bool:
        return ref_or(x, ref_not(x)) == MASK32

    @staticmethod
    def check_complement_xor(x: int) -> bool:
        return ref_xor(x, ref_not(x)) == MASK32


class Expr:
    """Base expression node for bitwise ALU symbolic representation."""

    def eval(self, env: dict[str, int]) -> int:
        raise NotImplementedError

    def fold(self) -> Expr:
        raise NotImplementedError


class ConstExpr(Expr):
    """32-bit constant literal."""

    def __init__(self, val: int) -> None:
        self.val = ref_u32(val)

    def eval(self, env: dict[str, int]) -> int:
        return self.val

    def fold(self) -> Expr:
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ConstExpr) and self.val == other.val

    def __hash__(self) -> int:
        return hash(("const", self.val))

    def __repr__(self) -> str:
        return f"Const(0x{self.val:08X})"


class RegExpr(Expr):
    """Register variable operand."""

    def __init__(self, name: str) -> None:
        self.name = name.lower()

    def eval(self, env: dict[str, int]) -> int:
        return ref_u32(env.get(self.name, 0))

    def fold(self) -> Expr:
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RegExpr) and self.name == other.name

    def __hash__(self) -> int:
        return hash(("reg", self.name))

    def __repr__(self) -> str:
        return f"Reg({self.name})"


class UnaryExpr(Expr):
    """Unary bitwise operation (v_not_b32 / s_not_b32)."""

    def __init__(self, op: str, operand: Expr) -> None:
        self.op = op
        self.operand = operand

    def eval(self, env: dict[str, int]) -> int:
        val = self.operand.eval(env)
        if self.op in ("v_not_b32", "s_not_b32", "not"):
            return ref_not(val)
        raise ValueError(f"Unknown unary op {self.op}")

    def fold(self) -> Expr:
        sub = self.operand.fold()
        if isinstance(sub, ConstExpr):
            return ConstExpr(ref_not(sub.val))
        if isinstance(sub, UnaryExpr) and sub.op in ("v_not_b32", "s_not_b32", "not"):
            return sub.operand.fold()
        return UnaryExpr(self.op, sub)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, UnaryExpr) and self.op == other.op and self.operand == other.operand

    def __hash__(self) -> int:
        return hash(("unary", self.op, self.operand))

    def __repr__(self) -> str:
        return f"Unary({self.op}, {self.operand})"


class BinaryExpr(Expr):
    """Binary bitwise operation (v_and_b32, v_or_b32, v_xor_b32)."""

    def __init__(self, op: str, lhs: Expr, rhs: Expr) -> None:
        self.op = op
        self.lhs = lhs
        self.rhs = rhs

    def eval(self, env: dict[str, int]) -> int:
        l = self.lhs.eval(env)
        r = self.rhs.eval(env)
        if self.op in ("v_and_b32", "s_and_b32", "and"):
            return ref_and(l, r)
        elif self.op in ("v_or_b32", "s_or_b32", "or"):
            return ref_or(l, r)
        elif self.op in ("v_xor_b32", "s_xor_b32", "xor"):
            return ref_xor(l, r)
        raise ValueError(f"Unknown binary op {self.op}")

    def fold(self) -> Expr:
        l = self.lhs.fold()
        r = self.rhs.fold()
        op = self.op

        if isinstance(l, ConstExpr) and isinstance(r, ConstExpr):
            if op in ("v_and_b32", "s_and_b32", "and"):
                return ConstExpr(ref_and(l.val, r.val))
            elif op in ("v_or_b32", "s_or_b32", "or"):
                return ConstExpr(ref_or(l.val, r.val))
            elif op in ("v_xor_b32", "s_xor_b32", "xor"):
                return ConstExpr(ref_xor(l.val, r.val))

        if isinstance(l, ConstExpr) and not isinstance(r, ConstExpr):
            l, r = r, l
        elif not isinstance(l, ConstExpr) and not isinstance(r, ConstExpr):
            if repr(l) > repr(r):
                l, r = r, l

        if op in ("v_and_b32", "s_and_b32", "and"):
            if l == r:
                return l
            if isinstance(r, ConstExpr) and r.val == 0:
                return ConstExpr(0)
            if isinstance(r, ConstExpr) and r.val == MASK32:
                return l
            if (isinstance(r, UnaryExpr) and r.operand == l) or (isinstance(l, UnaryExpr) and l.operand == r):
                return ConstExpr(0)
            if isinstance(r, BinaryExpr) and r.op in ("v_or_b32", "s_or_b32", "or"):
                if r.lhs == l or r.rhs == l:
                    return l
            if isinstance(l, BinaryExpr) and l.op in ("v_or_b32", "s_or_b32", "or"):
                if l.lhs == r or l.rhs == r:
                    return r

        elif op in ("v_or_b32", "s_or_b32", "or"):
            if l == r:
                return l
            if isinstance(r, ConstExpr) and r.val == MASK32:
                return ConstExpr(MASK32)
            if isinstance(r, ConstExpr) and r.val == 0:
                return l
            if (isinstance(r, UnaryExpr) and r.operand == l) or (isinstance(l, UnaryExpr) and l.operand == r):
                return ConstExpr(MASK32)
            if isinstance(r, BinaryExpr) and r.op in ("v_and_b32", "s_and_b32", "and"):
                if r.lhs == l or r.rhs == l:
                    return l
            if isinstance(l, BinaryExpr) and l.op in ("v_and_b32", "s_and_b32", "and"):
                if l.lhs == r or l.rhs == r:
                    return r

        elif op in ("v_xor_b32", "s_xor_b32", "xor"):
            if l == r:
                return ConstExpr(0)
            if isinstance(r, ConstExpr) and r.val == 0:
                return l
            if isinstance(r, ConstExpr) and r.val == MASK32:
                return UnaryExpr("v_not_b32", l).fold()
            if (isinstance(r, UnaryExpr) and r.operand == l) or (isinstance(l, UnaryExpr) and l.operand == r):
                return ConstExpr(MASK32)

        return BinaryExpr(op, l, r)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BinaryExpr) and self.op == other.op and self.lhs == other.lhs and self.rhs == other.rhs

    def __hash__(self) -> int:
        return hash(("binary", self.op, self.lhs, self.rhs))

    def __repr__(self) -> str:
        return f"Binary({self.op}, {self.lhs}, {self.rhs})"


class BitopFoldingInterpreter:
    """RDNA2 instruction interpreter tracking registers, expressions, and folding."""

    def __init__(self) -> None:
        self.regs: dict[str, int] = {}
        self.exprs: dict[str, Expr] = {}

    def set_reg(self, reg: str, val: int) -> None:
        r = reg.lower().strip()
        v = ref_u32(val)
        self.regs[r] = v
        self.exprs[r] = ConstExpr(v)

    def set_reg_symbolic(self, reg: str, name: str) -> None:
        r = reg.lower().strip()
        self.exprs[r] = RegExpr(name)

    def get_reg(self, reg: str) -> int:
        return self.regs.get(reg.lower().strip(), 0)

    def get_expr(self, reg: str) -> Expr:
        r = reg.lower().strip()
        return self.exprs.get(r, RegExpr(r))

    def _parse_operand(self, token: str) -> tuple[int, Expr]:
        token = token.strip()
        try:
            val = int(token, 0)
            u = ref_u32(val)
            return u, ConstExpr(u)
        except ValueError:
            pass
        reg = token.lower()
        val = self.regs.get(reg, 0)
        expr = self.exprs.get(reg, RegExpr(reg))
        return val, expr

    def execute_instruction(self, line: str) -> tuple[str, int, Expr]:
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            return ("", 0, ConstExpr(0))
        parts = line.split(None, 1)
        raw_op = parts[0].lower()
        op = raw_op.replace("_e32", "").replace("_e64", "")
        operands = [p.strip() for p in parts[1].split(",")] if len(parts) > 1 else []

        if op in ("v_mov_b32", "s_mov_b32"):
            if len(operands) < 2:
                raise ValueError(f"Missing operands in {line}")
            dst = operands[0].lower()
            val, expr = self._parse_operand(operands[1])
            self.regs[dst] = val
            folded = expr.fold()
            self.exprs[dst] = folded
            return (dst, val, folded)

        elif op in ("v_not_b32", "s_not_b32"):
            if len(operands) < 2:
                raise ValueError(f"Missing operands in {line}")
            dst = operands[0].lower()
            val, expr = self._parse_operand(operands[1])
            res = ref_not(val)
            self.regs[dst] = res
            folded = UnaryExpr("v_not_b32", expr).fold()
            self.exprs[dst] = folded
            return (dst, res, folded)

        elif op in ("v_and_b32", "s_and_b32", "v_or_b32", "s_or_b32", "v_xor_b32", "s_xor_b32"):
            if len(operands) < 3:
                raise ValueError(f"Missing operands in {line}")
            dst = operands[0].lower()
            v0, e0 = self._parse_operand(operands[1])
            v1, e1 = self._parse_operand(operands[2])
            if op in ("v_and_b32", "s_and_b32"):
                res = ref_and(v0, v1)
            elif op in ("v_or_b32", "s_or_b32"):
                res = ref_or(v0, v1)
            elif op in ("v_xor_b32", "s_xor_b32"):
                res = ref_xor(v0, v1)
            else:
                raise ValueError(f"Unknown binary op {op}")
            self.regs[dst] = res
            folded = BinaryExpr(op, e0, e1).fold()
            self.exprs[dst] = folded
            return (dst, res, folded)

        else:
            raise ValueError(f"Unsupported instruction: {line}")

    def execute_sequence(self, lines: list[str]) -> None:
        for line in lines:
            line = line.strip()
            if line:
                self.execute_instruction(line)


class SimdWaveInterpreter:
    """Wave32 SIMD vector interpreter modeling 32 lanes under an EXEC mask."""

    def __init__(self, num_lanes: int = 32, num_vgprs: int = 64) -> None:
        self.num_lanes = num_lanes
        self.num_vgprs = num_vgprs
        self.vgprs: list[list[int]] = [[0] * num_lanes for _ in range(num_vgprs)]
        self.exec_mask: int = (1 << num_lanes) - 1

    def set_exec(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.num_lanes) - 1)

    def get_exec(self) -> int:
        return self.exec_mask

    def set_vgpr_lane(self, v_idx: int, lane: int, val: int) -> None:
        self.vgprs[v_idx][lane] = ref_u32(val)

    def set_vgpr_all_lanes(self, v_idx: int, values: list[int]) -> None:
        assert len(values) == self.num_lanes
        self.vgprs[v_idx] = [ref_u32(v) for v in values]

    def set_vgpr_broadcast(self, v_idx: int, val: int) -> None:
        u = ref_u32(val)
        self.vgprs[v_idx] = [u] * self.num_lanes

    def get_vgpr_lane(self, v_idx: int, lane: int) -> int:
        return self.vgprs[v_idx][lane]

    def get_vgpr_all_lanes(self, v_idx: int) -> list[int]:
        return list(self.vgprs[v_idx])

    def _resolve_lane_val(self, operand: str, lane: int) -> int:
        tok = operand.strip()
        try:
            return ref_u32(int(tok, 0))
        except ValueError:
            pass
        if tok.lower().startswith("v"):
            v_idx = int(tok[1:])
            return self.vgprs[v_idx][lane]
        raise ValueError(f"Invalid operand token: {tok}")

    def execute_vector_instruction(self, line: str) -> None:
        line = line.strip()
        if not line or line.startswith(";") or line.startswith("//"):
            return
        parts = line.split(None, 1)
        raw_op = parts[0].lower()
        op = raw_op.replace("_e32", "").replace("_e64", "")
        operands = [p.strip() for p in parts[1].split(",")] if len(parts) > 1 else []

        if not operands:
            raise ValueError(f"No operands in instruction: {line}")

        dst_tok = operands[0].lower()
        if not dst_tok.startswith("v"):
            raise ValueError(f"Vector destination must be VGPR (e.g. v0): {dst_tok}")
        dst_idx = int(dst_tok[1:])

        if op in ("v_mov_b32", "s_mov_b32"):
            if len(operands) < 2:
                raise ValueError(f"Missing source operand in {line}")
            for lane in range(self.num_lanes):
                if self.exec_mask & (1 << lane):
                    val = self._resolve_lane_val(operands[1], lane)
                    self.vgprs[dst_idx][lane] = val

        elif op in ("v_not_b32", "s_not_b32"):
            if len(operands) < 2:
                raise ValueError(f"Missing source operand in {line}")
            for lane in range(self.num_lanes):
                if self.exec_mask & (1 << lane):
                    val = self._resolve_lane_val(operands[1], lane)
                    self.vgprs[dst_idx][lane] = ref_not(val)

        elif op in ("v_and_b32", "s_and_b32", "v_or_b32", "s_or_b32", "v_xor_b32", "s_xor_b32"):
            if len(operands) < 3:
                raise ValueError(f"Missing binary operands in {line}")
            for lane in range(self.num_lanes):
                if self.exec_mask & (1 << lane):
                    v0 = self._resolve_lane_val(operands[1], lane)
                    v1 = self._resolve_lane_val(operands[2], lane)
                    if op in ("v_and_b32", "s_and_b32"):
                        res = ref_and(v0, v1)
                    elif op in ("v_or_b32", "s_or_b32"):
                        res = ref_or(v0, v1)
                    elif op in ("v_xor_b32", "s_xor_b32"):
                        res = ref_xor(v0, v1)
                    self.vgprs[dst_idx][lane] = res

        else:
            raise ValueError(f"Unsupported SIMD instruction: {line}")

    def execute_sequence(self, lines: list[str]) -> None:
        for line in lines:
            self.execute_vector_instruction(line)


class BitwiseAlgebraOracleTests(unittest.TestCase):
    """Test suite executing the reference oracle against exhaustive vector sets."""

    def test_idempotence(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_idempotence_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_idempotence_or(vec))

    def test_annihilator(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_annihilator_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_annihilator_or(vec))

    def test_identity(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_identity_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_identity_or(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_identity_xor(vec))

    def test_self_inverse(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_self_inverse_xor(vec))

    def test_involution(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_involution_not(vec))

    def test_absorption(self):
        vectors = CANONICAL_TEST_VECTORS[:12]
        for x in vectors:
            for y in vectors:
                with self.subTest(x=hex(x), y=hex(y)):
                    self.assertTrue(BitwiseAlgebraOracle.check_absorption_or_and(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_absorption_and_or(x, y))

    def test_commutativity(self):
        vectors = CANONICAL_TEST_VECTORS[:10]
        for x in vectors:
            for y in vectors:
                with self.subTest(x=hex(x), y=hex(y)):
                    self.assertTrue(BitwiseAlgebraOracle.check_commutativity_and(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_commutativity_or(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_commutativity_xor(x, y))

    def test_associativity(self):
        vectors = [0x0, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA, 0x12345678, 0x80000000]
        for x in vectors:
            for y in vectors:
                for z in vectors:
                    with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                        self.assertTrue(BitwiseAlgebraOracle.check_associativity_and(x, y, z))
                        self.assertTrue(BitwiseAlgebraOracle.check_associativity_or(x, y, z))
                        self.assertTrue(BitwiseAlgebraOracle.check_associativity_xor(x, y, z))

    def test_distributivity(self):
        vectors = [0x0, 0xFFFFFFFF, 0x55555555, 0xAAAAAAAA, 0x12345678, 0x80000000]
        for x in vectors:
            for y in vectors:
                for z in vectors:
                    with self.subTest(x=hex(x), y=hex(y), z=hex(z)):
                        self.assertTrue(BitwiseAlgebraOracle.check_distributivity_and_over_or(x, y, z))
                        self.assertTrue(BitwiseAlgebraOracle.check_distributivity_or_over_and(x, y, z))

    def test_demorgan_laws(self):
        vectors = CANONICAL_TEST_VECTORS[:12]
        for x in vectors:
            for y in vectors:
                with self.subTest(x=hex(x), y=hex(y)):
                    self.assertTrue(BitwiseAlgebraOracle.check_demorgan_and(x, y))
                    self.assertTrue(BitwiseAlgebraOracle.check_demorgan_or(x, y))

    def test_complementarity(self):
        for vec in CANONICAL_TEST_VECTORS + SINGLE_BIT_SET + SINGLE_BIT_CLR:
            with self.subTest(vec=hex(vec)):
                self.assertTrue(BitwiseAlgebraOracle.check_complement_and(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_complement_or(vec))
                self.assertTrue(BitwiseAlgebraOracle.check_complement_xor(vec))

    def test_composite_folding_identities(self):
        vectors = CANONICAL_TEST_VECTORS[:12]
        for x in vectors:
            self.assertEqual(ref_xor(x, MASK32), ref_not(x))
            for y in vectors:
                lhs1 = ref_and(ref_or(x, y), ref_or(x, ref_not(y)))
                self.assertEqual(lhs1, ref_u32(x))
                lhs2 = ref_or(ref_and(x, y), ref_and(x, ref_not(y)))
                self.assertEqual(lhs2, ref_u32(x))
                lhs3 = ref_xor(x, ref_and(x, y))
                self.assertEqual(lhs3, ref_andn2(x, y))


class BitopFoldingInterpreterTests(unittest.TestCase):
    """Test suite verifying instruction interpreter constant folding and sequences."""

    def test_constant_folding_basic(self):
        interp = BitopFoldingInterpreter()
        interp.execute_sequence([
            "v_mov_b32 v0, 0x12345678",
            "v_mov_b32 v1, 0x0000FFFF",
            "v_and_b32 v2, v0, v1",
            "v_or_b32 v3, v0, v1",
            "v_xor_b32 v4, v0, v1",
            "v_not_b32 v5, v0",
        ])
        self.assertEqual(interp.get_reg("v2"), 0x00005678)
        self.assertEqual(interp.get_expr("v2"), ConstExpr(0x00005678))
        self.assertEqual(interp.get_reg("v3"), 0x1234FFFF)
        self.assertEqual(interp.get_expr("v3"), ConstExpr(0x1234FFFF))
        self.assertEqual(interp.get_reg("v4"), 0x1234A987)
        self.assertEqual(interp.get_expr("v4"), ConstExpr(0x1234A987))
        self.assertEqual(interp.get_reg("v5"), 0xEDCBA987)
        self.assertEqual(interp.get_expr("v5"), ConstExpr(0xEDCBA987))

    def test_idempotence_folding(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg_symbolic("v0", "v0")
        interp.execute_instruction("v_and_b32 v1, v0, v0")
        self.assertEqual(interp.get_expr("v1"), RegExpr("v0"))

        interp.execute_instruction("v_or_b32 v2, v0, v0")
        self.assertEqual(interp.get_expr("v2"), RegExpr("v0"))

    def test_annihilator_folding(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg_symbolic("v0", "v0")
        interp.execute_instruction("v_and_b32 v1, v0, 0")
        self.assertEqual(interp.get_expr("v1"), ConstExpr(0))
        interp.execute_instruction("v_and_b32 v2, 0, v0")
        self.assertEqual(interp.get_expr("v2"), ConstExpr(0))

        interp.execute_instruction("v_or_b32 v3, v0, 0xFFFFFFFF")
        self.assertEqual(interp.get_expr("v3"), ConstExpr(MASK32))
        interp.execute_instruction("v_or_b32 v4, 0xFFFFFFFF, v0")
        self.assertEqual(interp.get_expr("v4"), ConstExpr(MASK32))

    def test_identity_folding(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg_symbolic("v0", "v0")
        interp.execute_instruction("v_and_b32 v1, v0, 0xFFFFFFFF")
        self.assertEqual(interp.get_expr("v1"), RegExpr("v0"))
        interp.execute_instruction("v_and_b32 v2, 0xFFFFFFFF, v0")
        self.assertEqual(interp.get_expr("v2"), RegExpr("v0"))

        interp.execute_instruction("v_or_b32 v3, v0, 0")
        self.assertEqual(interp.get_expr("v3"), RegExpr("v0"))
        interp.execute_instruction("v_or_b32 v4, 0, v0")
        self.assertEqual(interp.get_expr("v4"), RegExpr("v0"))

        interp.execute_instruction("v_xor_b32 v5, v0, 0")
        self.assertEqual(interp.get_expr("v5"), RegExpr("v0"))
        interp.execute_instruction("v_xor_b32 v6, 0, v0")
        self.assertEqual(interp.get_expr("v6"), RegExpr("v0"))

    def test_self_inverse_folding(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg_symbolic("v0", "v0")
        interp.execute_instruction("v_xor_b32 v1, v0, v0")
        self.assertEqual(interp.get_expr("v1"), ConstExpr(0))

    def test_involution_folding(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg_symbolic("v0", "v0")
        interp.execute_sequence([
            "v_not_b32 v1, v0",
            "v_not_b32 v2, v1",
            "v_not_b32 v3, v2",
        ])
        self.assertEqual(interp.get_expr("v2"), RegExpr("v0"))
        self.assertEqual(interp.get_expr("v3"), UnaryExpr("v_not_b32", RegExpr("v0")))

    def test_absorption_folding(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg_symbolic("v0", "v0")
        interp.set_reg_symbolic("v1", "v1")
        interp.execute_sequence([
            "v_and_b32 v2, v0, v1",
            "v_or_b32 v3, v0, v2",
            "v_or_b32 v4, v0, v1",
            "v_and_b32 v5, v0, v4",
        ])
        self.assertEqual(interp.get_expr("v3"), RegExpr("v0"))
        self.assertEqual(interp.get_expr("v5"), RegExpr("v0"))

    def test_self_complement_folding(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg_symbolic("v0", "v0")
        interp.execute_sequence([
            "v_not_b32 v1, v0",
            "v_and_b32 v2, v0, v1",
            "v_or_b32 v3, v0, v1",
            "v_xor_b32 v4, v0, v1",
        ])
        self.assertEqual(interp.get_expr("v2"), ConstExpr(0))
        self.assertEqual(interp.get_expr("v3"), ConstExpr(MASK32))
        self.assertEqual(interp.get_expr("v4"), ConstExpr(MASK32))

    def test_multi_instruction_alu_sequences(self):
        interp = BitopFoldingInterpreter()
        interp.set_reg("v0", 0x12345678)
        interp.set_reg("v1", 0x9ABCDEF0)
        interp.execute_sequence([
            "v_and_b32 v2, v0, v1",
            "v_or_b32 v3, v0, v2",
            "v_xor_b32 v4, v3, v0",
            "v_or_b32 v5, v4, 0x55555555",
            "v_and_b32 v6, v5, 0xFFFFFFFF",
        ])
        self.assertEqual(interp.get_reg("v3"), 0x12345678)
        self.assertEqual(interp.get_reg("v4"), 0)
        self.assertEqual(interp.get_reg("v5"), 0x55555555)
        self.assertEqual(interp.get_reg("v6"), 0x55555555)
        self.assertEqual(interp.get_expr("v6"), ConstExpr(0x55555555))

    def test_translator_spirv_lowering_equivalence(self):
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_mov_b32 v2, 0x9ABCDEF0",
            "v_and_b32 v3, v1, v2",
            "v_or_b32 v4, v1, v3",
            "v_xor_b32 v0, v4, v1",
        ]
        body = "\n".join("\t" + line for line in lines)
        asm = f"; alu_lowering_test\n{body}\n\texp mrt0 v0, v3, v4, v1 done vm\n\ts_endpgm\n"
        with tempfile.TemporaryDirectory() as tmp:
            d_dir = os.path.join(tmp, "disasm")
            os.mkdir(d_dir)
            s_path = os.path.join(d_dir, "ps_bitop_eval.s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(asm)
            with open(os.path.join(tmp, "ps_bitop_eval.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
        out = si.run_fragment(spv)["%out_mrt0"]
        self.assertEqual(out[0], 0)
        self.assertEqual(out[1], ref_and(0x12345678, 0x9ABCDEF0))
        self.assertEqual(out[2], 0x12345678)
        self.assertEqual(out[3], 0x12345678)


class SimdWaveDivergenceTests(unittest.TestCase):
    """Test suite verifying multi-lane SIMD divergence under various EXEC masks."""

    def test_simd_all_active_exec(self):
        simd = SimdWaveInterpreter()
        simd.set_exec(0xFFFFFFFF)
        for lane in range(32):
            simd.set_vgpr_lane(1, lane, 0x1000 + lane)
            simd.set_vgpr_lane(2, lane, 0x00FF)
        simd.execute_vector_instruction("v_and_b32 v0, v1, v2")
        for lane in range(32):
            self.assertEqual(simd.get_vgpr_lane(0, lane), (0x1000 + lane) & 0x00FF)

    def test_simd_all_inactive_exec(self):
        simd = SimdWaveInterpreter()
        simd.set_exec(0x00000000)
        simd.set_vgpr_broadcast(0, 0xCAFEBABE)
        simd.set_vgpr_broadcast(1, 0x12345678)
        simd.execute_vector_instruction("v_mov_b32 v0, v1")
        for lane in range(32):
            self.assertEqual(simd.get_vgpr_lane(0, lane), 0xCAFEBABE)

    def test_simd_checkerboard_divergence(self):
        simd = SimdWaveInterpreter()
        simd.set_exec(0x55555555)  # Even lanes active, odd lanes inactive
        simd.set_vgpr_broadcast(0, 0xDEADBEEF)
        for lane in range(32):
            simd.set_vgpr_lane(1, lane, 0xAA00 + lane)
        simd.execute_vector_instruction("v_mov_b32 v0, v1")
        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(simd.get_vgpr_lane(0, lane), 0xAA00 + lane)
            else:
                self.assertEqual(simd.get_vgpr_lane(0, lane), 0xDEADBEEF)

    def test_simd_halves_divergence(self):
        simd = SimdWaveInterpreter()
        simd.set_exec(0x0000FFFF)  # Lower 16 lanes active, upper 16 inactive
        simd.set_vgpr_broadcast(0, 0x77777777)
        simd.set_vgpr_broadcast(1, 0x11111111)
        simd.set_vgpr_broadcast(2, 0x22222222)
        simd.execute_vector_instruction("v_or_b32 v0, v1, v2")
        for lane in range(32):
            if lane < 16:
                self.assertEqual(simd.get_vgpr_lane(0, lane), 0x33333333)
            else:
                self.assertEqual(simd.get_vgpr_lane(0, lane), 0x77777777)

    def test_simd_single_lane_divergence(self):
        for target_lane in (0, 7, 15, 31):
            with self.subTest(lane=target_lane):
                simd = SimdWaveInterpreter()
                simd.set_exec(1 << target_lane)
                simd.set_vgpr_broadcast(0, 0)
                simd.set_vgpr_broadcast(1, 0xFFFFFFFF)
                simd.execute_vector_instruction("v_mov_b32 v0, v1")
                for lane in range(32):
                    if lane == target_lane:
                        self.assertEqual(simd.get_vgpr_lane(0, lane), 0xFFFFFFFF)
                    else:
                        self.assertEqual(simd.get_vgpr_lane(0, lane), 0)

    def test_simd_divergent_folding_inactive_isolation(self):
        # Even when constant folding simplifies an instruction (e.g. x & 0 -> 0),
        # inactive lanes must strictly retain their previous destination value.
        simd = SimdWaveInterpreter()
        simd.set_exec(0x00FF00FF)
        simd.set_vgpr_broadcast(0, 0x12345678)
        simd.set_vgpr_broadcast(1, 0x9ABCDEF0)
        simd.execute_vector_instruction("v_and_b32 v0, v1, 0")
        for lane in range(32):
            if (0x00FF00FF & (1 << lane)) != 0:
                self.assertEqual(simd.get_vgpr_lane(0, lane), 0)
            else:
                self.assertEqual(simd.get_vgpr_lane(0, lane), 0x12345678)

    def test_simd_multi_instruction_divergence(self):
        simd = SimdWaveInterpreter()
        # Step 1: execute on lower half
        simd.set_exec(0x0000FFFF)
        for lane in range(32):
            simd.set_vgpr_lane(0, lane, 0x0)
            simd.set_vgpr_lane(1, lane, lane)
            simd.set_vgpr_lane(2, lane, 0x10)
        simd.execute_vector_instruction("v_xor_b32 v0, v1, v2")

        # Step 2: execute on upper half with complementary operation
        simd.set_exec(0xFFFF0000)
        simd.execute_vector_instruction("v_not_b32 v0, v1")

        for lane in range(32):
            if lane < 16:
                self.assertEqual(simd.get_vgpr_lane(0, lane), lane ^ 0x10)
            else:
                self.assertEqual(simd.get_vgpr_lane(0, lane), ref_not(lane))


class BitwiseTruthTableTests(unittest.TestCase):
    """Test suite validating full Boolean truth tables and composite logic."""

    def test_1bit_truth_tables(self):
        table = [
            (0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1),
            (0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0),
            (1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1),
            (1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1),
        ]
        for a, b, exp_and, exp_or, exp_xor, exp_not_a, exp_nand, exp_nor, exp_xnor, exp_andn2, exp_orn2 in table:
            with self.subTest(a=a, b=b):
                self.assertEqual(ref_and(a, b), exp_and)
                self.assertEqual(ref_or(a, b), exp_or)
                self.assertEqual(ref_xor(a, b), exp_xor)
                self.assertEqual(ref_not(a) & 1, exp_not_a)
                self.assertEqual(ref_nand(a, b) & 1, exp_nand)
                self.assertEqual(ref_nor(a, b) & 1, exp_nor)
                self.assertEqual(ref_xnor(a, b) & 1, exp_xnor)
                self.assertEqual(ref_andn2(a, b), exp_andn2)
                self.assertEqual(ref_orn2(a, b) & 1, exp_orn2)

    def test_32bit_sliced_truth_tables(self):
        for bit in range(32):
            m = 1 << bit
            with self.subTest(bit=bit):
                # (0, 0)
                self.assertEqual(ref_and(0, 0) & m, 0)
                self.assertEqual(ref_or(0, 0) & m, 0)
                self.assertEqual(ref_xor(0, 0) & m, 0)
                # (m, 0)
                self.assertEqual(ref_and(m, 0) & m, 0)
                self.assertEqual(ref_or(m, 0) & m, m)
                self.assertEqual(ref_xor(m, 0) & m, m)
                # (0, m)
                self.assertEqual(ref_and(0, m) & m, 0)
                self.assertEqual(ref_or(0, m) & m, m)
                self.assertEqual(ref_xor(0, m) & m, m)
                # (m, m)
                self.assertEqual(ref_and(m, m) & m, m)
                self.assertEqual(ref_or(m, m) & m, m)
                self.assertEqual(ref_xor(m, m) & m, 0)

    def test_3input_boolean_functions(self):
        # Majority: maj(a, b, c) = (a & b) | (b & c) | (a & c)
        def maj(a: int, b: int, c: int) -> int:
            return ref_or(ref_or(ref_and(a, b), ref_and(b, c)), ref_and(a, c))

        # Multiplexer: mux(sel, a, b) = (sel & a) | (~sel & b)
        def mux(sel: int, a: int, b: int) -> int:
            return ref_or(ref_and(sel, a), ref_and(ref_not(sel), b))

        # Parity: parity(a, b, c) = a ^ b ^ c
        def parity(a: int, b: int, c: int) -> int:
            return ref_xor(ref_xor(a, b), c)

        for a in (0, 1):
            for b in (0, 1):
                for c in (0, 1):
                    # Majority is 1 if at least two inputs are 1
                    exp_maj = 1 if (a + b + c) >= 2 else 0
                    self.assertEqual(maj(a, b, c), exp_maj)

                    # Parity is odd parity
                    exp_parity = (a ^ b ^ c) & 1
                    self.assertEqual(parity(a, b, c), exp_parity)

        # Mux test across 32-bit words
        for sel in (0x00000000, 0xFFFFFFFF, 0x55555555, 0x0000FFFF):
            x = 0x12345678
            y = 0x9ABCDEF0
            res = mux(sel, x, y)
            for bit in range(32):
                expected_bit = (x if (sel & (1 << bit)) else y) & (1 << bit)
                self.assertEqual(res & (1 << bit), expected_bit)


class BitwiseNegativeControlTests(unittest.TestCase):
    """Test suite validating negative controls, non-identities, and fault rejection."""

    def test_non_identities(self):
        # x & y != x in general when x has bits not in y
        x = 0x12345678
        y = 0x0000FFFF
        self.assertNotEqual(ref_and(x, y), x)

        # x | y != x when y has bits not in x
        x = 0x0000FFFF
        y = 0x12345678
        self.assertNotEqual(ref_or(x, y), x)

    def test_non_cancellation(self):
        # x ^ y != 0 when x != y
        for i in range(len(CANONICAL_TEST_VECTORS) - 1):
            x = CANONICAL_TEST_VECTORS[i]
            y = CANONICAL_TEST_VECTORS[i + 1]
            if x != y:
                self.assertNotEqual(ref_xor(x, y), 0)

    def test_non_annihilation(self):
        # x & c != 0 when x and c share set bits
        x = 0x55555555
        c = 0x11111111
        self.assertNotEqual(ref_and(x, c), 0)

    def test_inactive_lane_isolation(self):
        simd = SimdWaveInterpreter()
        simd.set_exec(0x00000001)  # Only lane 0 active
        simd.set_vgpr_broadcast(0, 0xAAAAAAAAAAAAAAAA)
        simd.set_vgpr_broadcast(1, 0x55555555)
        simd.execute_vector_instruction("v_mov_b32 v0, v1")
        # Lane 0 changed to 0x55555555
        self.assertEqual(simd.get_vgpr_lane(0, 0), 0x55555555)
        # Lanes 1..31 must NOT have changed
        for lane in range(1, 32):
            self.assertNotEqual(simd.get_vgpr_lane(0, lane), 0x55555555)
            self.assertEqual(simd.get_vgpr_lane(0, lane), 0xAAAAAAAA)

    def test_parser_fault_rejection(self):
        interp = BitopFoldingInterpreter()
        with self.assertRaises(ValueError):
            interp.execute_instruction("v_invalid_opcode_b32 v0, v1, v2")
        with self.assertRaises(ValueError):
            interp.execute_instruction("v_and_b32 v0")

        simd = SimdWaveInterpreter()
        with self.assertRaises(ValueError):
            simd.execute_vector_instruction("v_unknown_op v0, v1")
        with self.assertRaises(ValueError):
            simd.execute_vector_instruction("v_and_b32 s0, v1, v2")


class BitopConstantFoldingTests(unittest.TestCase):
    def translate_lines(self, lines: list[str]) -> str:
        body = "\n".join("\t" + line for line in lines)
        asm = f"; bitop_test\n{body}\n\texp mrt0 v0, v0, v0, v0 done vm\n\ts_endpgm\n"
        with tempfile.TemporaryDirectory() as tmp:
            d_dir = os.path.join(tmp, "disasm")
            os.mkdir(d_dir)
            s_path = os.path.join(d_dir, "ps_bitop.s")
            with open(s_path, "w", encoding="utf-8") as f:
                f.write(asm)
            with open(os.path.join(tmp, "ps_bitop.dxbc"), "wb") as f:
                f.write(STUB_DXBC)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                return rdna2spirv.translate(s_path)

    def test_constant_shift_eliminates_runtime_mask(self):
        lines = [
            "v_mov_b32 v1, 0x1234",
            "v_lshlrev_b32 v0, 4, v1",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12340)

    def test_constant_bfe_eliminates_runtime_bounds(self):
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_bfe_u32 v0, v1, 8, 12",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertNotIn("OpULessThan %bool", spv)
        self.assertIn("OpBitFieldUExtract %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x456)

    def test_constant_alignbyte_zero_shift_folds_directly(self):
        lines = [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0x12345678",
            "v_alignbyte_b32 v0, v1, v2, 0",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpShiftRightLogical", spv)
        self.assertNotIn("OpShiftLeftLogical", spv)
        self.assertNotIn("OpBitwiseOr", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12345678)

    def test_constant_alignbyte_nonzero_shift_emits_direct_shifts(self):
        lines = [
            "v_mov_b32 v1, 0x8899AABB",
            "v_mov_b32 v2, 0x11223344",
            "v_alignbyte_b32 v0, v1, v2, 1",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertIn("OpShiftRightLogical %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0xBB112233)

    def test_constant_alignbit_zero_shift_folds_directly(self):
        lines = [
            "v_mov_b32 v1, 0xDEADBEEF",
            "v_mov_b32 v2, 0x12345678",
            "v_alignbit_b32 v0, v1, v2, 0",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpShiftRightLogical", spv)
        self.assertNotIn("OpShiftLeftLogical", spv)
        self.assertNotIn("OpBitwiseOr", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x12345678)

    def test_constant_alignbit_nonzero_shift_emits_direct_shifts(self):
        lines = [
            "v_mov_b32 v1, 0x12345678",
            "v_mov_b32 v2, 0x9ABCDEF0",
            "v_alignbit_b32 v0, v1, v2, 16",
        ]
        spv = self.translate_lines(lines)
        self.assertNotIn("OpBitwiseAnd %uint", spv)
        self.assertNotIn("OpISub %uint", spv)
        self.assertIn("OpShiftRightLogical %uint", spv)
        self.assertIn("OpShiftLeftLogical %uint", spv)
        self.assertIn("OpBitwiseOr %uint", spv)
        res = si.run_fragment(spv)["%out_mrt0"][0]
        self.assertEqual(res, 0x56789ABC)

    def test_dedupe_types_merges_duplicate_types_and_substitutes(self):
        spv_sample = (
            "OpCapability Shader\n"
            "%uint = OpTypeInt 32 0\n"
            "%v4uint = OpTypeVector %uint 4\n"
            "%v4uint_st = OpTypeVector %uint 4\n"
            "%ptr_v4uint = OpTypePointer StorageBuffer %v4uint\n"
            "%ptr_v4uint_st = OpTypePointer StorageBuffer %v4uint_st\n"
            "%fn = OpTypeFunction %void\n"
            "%main = OpFunction %void None %fn\n"
            "%entry = OpLabel\n"
            "%var = OpVariable %ptr_v4uint_st StorageBuffer\n"
            "%val = OpLoad %v4uint_st %var\n"
            "OpReturn\n"
            "OpFunctionEnd"
        )
        deduped = rdna2spirv.dedupe_types(spv_sample)
        self.assertNotIn("%v4uint_st =", deduped)
        self.assertNotIn("%ptr_v4uint_st =", deduped)
        self.assertIn("%var = OpVariable %ptr_v4uint StorageBuffer", deduped)
        self.assertIn("%val = OpLoad %v4uint %var", deduped)


if __name__ == "__main__":
    unittest.main()
