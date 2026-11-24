from __future__ import annotations

"""Semantic reference tests and translator hardware verification for AMD RDNA2 s_andn2_b64.

RDNA2 Scalar Arithmetic Logic Unit (SALU) Architecture:
  In AMD RDNA2 (GFX10/GFX1030) ISA, the s_andn2_b64 instruction belongs to SOP2 format.
  Instruction:
    s_andn2_b64 sdst, ssrc0, ssrc1

  Operands:
    sdst  = 64-bit destination SGPR pair s[2k:2k+1] (or EXEC)
    ssrc0 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC
    ssrc1 = 64-bit source operand: SGPR pair, literal 64-bit constant, inline constant, EXEC, or VCC

  Semantics:
    sdst = ssrc0 & ~ssrc1
    SCC = 1 if sdst != 0 else 0

Standard library only: Python 3.9+ compatible, zero game data, zero external dependencies.
Exercises tools/rdna2spirv.py via support.translate_lines(...) and support.mrt0_words for hardware-accurate translation.
"""

from dataclasses import dataclass
from enum import Enum, auto
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import support  # noqa: E402
import rdna2spirv  # noqa: E402

MASK32: int = 0xFFFFFFFF
MASK64: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_ACTIVE: int = 0xFFFFFFFFFFFFFFFF
ALL_LANES_INACTIVE: int = 0x0000000000000000


def u64(val: int) -> int:
    """Normalize integer to unsigned 64-bit integer range [0, 2^64 - 1]."""
    return val & MASK64


def u32(val: int) -> int:
    """Normalize integer to unsigned 32-bit integer range [0, 2^32 - 1]."""
    return val & MASK32


def ref_andn2_b64(ssrc0: int, ssrc1: int) -> tuple[int, int]:
    """Mathematical reference oracle for AMD RDNA2 s_andn2_b64 instruction.

    Args:
        ssrc0: 64-bit first source operand (uint64)
        ssrc1: 64-bit second source operand (uint64)

    Returns:
        tuple (sdst, scc):
          - sdst: 64-bit result = ssrc0 & ~ssrc1
          - scc: scalar condition code = 1 if sdst != 0 else 0
    """
    s0 = u64(ssrc0)
    s1 = u64(ssrc1)
    sdst = u64(s0 & (~s1 & MASK64))
    scc = 1 if sdst != 0 else 0
    return sdst, scc


class TestSaluAndn2B64Oracle(unittest.TestCase):
    """Test suite for reference mathematical oracle of s_andn2_b64."""

    def test_oracle_basic(self):
        s0 = 0x123456789ABCDEF0
        s1 = 0x00FF00FF0F0F0F0F
        sdst, scc = ref_andn2_b64(s0, s1)
        expected = s0 & (~s1 & MASK64)
        self.assertEqual(sdst, expected)
        self.assertEqual(scc, 1)

    def test_oracle_zero_s0(self):
        s1 = 0x123456789ABCDEF0
        sdst, scc = ref_andn2_b64(0, s1)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_all_ones_s1(self):
        s0 = 0x123456789ABCDEF0
        sdst, scc = ref_andn2_b64(s0, ALL_LANES_ACTIVE)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_oracle_zero_s1(self):
        s0 = 0x123456789ABCDEF0
        sdst, scc = ref_andn2_b64(s0, 0)
        self.assertEqual(sdst, s0)
        self.assertEqual(scc, 1)

    def test_oracle_operand_asymmetry(self):
        # s0 & ~s1 != s1 & ~s0 (non-commutative)
        s0 = 0x00000000FFFFFFFF
        s1 = 0x0000FFFF0000FFFF
        res_01, _ = ref_andn2_b64(s0, s1)
        res_10, _ = ref_andn2_b64(s1, s0)
        self.assertNotEqual(res_01, res_10)
        self.assertEqual(res_01, 0x00000000FFFF0000)
        self.assertEqual(res_10, 0x0000FFFF00000000)


class Opcode(Enum):
    S_ANDN2_B64 = auto()


class OperandKind(Enum):
    SGPR = auto()
    EXEC = auto()
    VCC = auto()
    LITERAL64 = auto()
    INLINE_CONST = auto()


@dataclass(frozen=True)
class ASTOperand:
    kind: OperandKind
    index: int = 0
    count: int = 2
    value: int = 0

    @classmethod
    def sgpr(cls, index: int, count: int = 2) -> ASTOperand:
        if count != 2:
            raise ValueError(f"s_andn2_b64 requires 64-bit register pair (count=2), got {count}")
        if index < 0 or index > 104 or (index + count) > 106:
            raise ValueError(f"SGPR index out of range: s[{index}:{index + count - 1}]")
        if index % 2 != 0:
            raise ValueError(f"SGPR pair must be 64-bit aligned (even index), got s{index}")
        return cls(kind=OperandKind.SGPR, index=index, count=count)

    @classmethod
    def exec_mask(cls) -> ASTOperand:
        return cls(kind=OperandKind.EXEC, count=2)

    @classmethod
    def vcc(cls) -> ASTOperand:
        return cls(kind=OperandKind.VCC, count=2)

    @classmethod
    def literal64(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.LITERAL64, count=2, value=u64(val))

    @classmethod
    def inline_const(cls, val: int) -> ASTOperand:
        return cls(kind=OperandKind.INLINE_CONST, count=2, value=u64(val))


@dataclass(frozen=True)
class ASTInstruction:
    opcode: Opcode
    dst: ASTOperand
    src0: ASTOperand
    src1: ASTOperand

    def validate(self) -> None:
        if self.opcode != Opcode.S_ANDN2_B64:
            raise ValueError(f"Unsupported opcode: {self.opcode}")
        if self.dst.kind not in (OperandKind.SGPR, OperandKind.EXEC, OperandKind.VCC):
            raise TypeError(f"Destination must be SGPR pair, EXEC, or VCC, got {self.dst.kind}")
        if self.dst.count != 2:
            raise ValueError(f"Destination must be 64-bit (2 registers), got {self.dst.count}")
        if self.src0.count != 2 or self.src1.count != 2:
            raise ValueError(f"Sources must be 64-bit operands, got {self.src0.count}, {self.src1.count}")


class TestSaluAndn2B64AST(unittest.TestCase):
    """Test suite for AST opcode definitions and operand validation."""

    def test_ast_construction_valid_operands(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B64,
            dst=ASTOperand.sgpr(4, 2),
            src0=ASTOperand.sgpr(6, 2),
            src1=ASTOperand.sgpr(8, 2),
        )
        inst.validate()
        self.assertEqual(inst.opcode, Opcode.S_ANDN2_B64)
        self.assertEqual(inst.dst.kind, OperandKind.SGPR)
        self.assertEqual(inst.dst.index, 4)
        self.assertEqual(inst.src0.index, 6)
        self.assertEqual(inst.src1.index, 8)

    def test_ast_rejects_unaligned_sgpr_pair(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(5, 2)

    def test_ast_rejects_single_sgpr(self):
        with self.assertRaises(ValueError):
            ASTOperand.sgpr(4, 1)

    def test_ast_rejects_invalid_dst_kind(self):
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B64,
            dst=ASTOperand.literal64(0x123),
            src0=ASTOperand.sgpr(6, 2),
            src1=ASTOperand.sgpr(8, 2),
        )
        with self.assertRaises(TypeError):
            inst.validate()


class ScalarRegisterState:
    """Simulated state of GCN/RDNA Scalar Registers (SGPRs, EXEC, and SCC)."""

    def __init__(self, exec_mask: int = ALL_LANES_ACTIVE):
        self.sgpr = [0] * 106
        self.exec_mask = u64(exec_mask)
        self.vcc = 0
        self.scc = 0

    def read_operand_u64(self, op: ASTOperand) -> int:
        if op.kind == OperandKind.SGPR:
            lo = self.sgpr[op.index]
            hi = self.sgpr[op.index + 1]
            return (hi << 32) | lo
        elif op.kind == OperandKind.EXEC:
            return self.exec_mask
        elif op.kind == OperandKind.VCC:
            return self.vcc
        elif op.kind in (OperandKind.LITERAL64, OperandKind.INLINE_CONST):
            return op.value
        raise ValueError(f"Unsupported operand kind for read: {op.kind}")

    def write_sgpr_u64(self, index: int, val: int) -> None:
        val = u64(val)
        self.sgpr[index] = val & MASK32
        self.sgpr[index + 1] = (val >> 32) & MASK32

    def execute(self, inst: ASTInstruction) -> tuple[int, int]:
        inst.validate()
        s0 = self.read_operand_u64(inst.src0)
        s1 = self.read_operand_u64(inst.src1)
        sdst_val, scc = ref_andn2_b64(s0, s1)

        if inst.dst.kind == OperandKind.SGPR:
            self.write_sgpr_u64(inst.dst.index, sdst_val)
        elif inst.dst.kind == OperandKind.EXEC:
            self.exec_mask = sdst_val
        elif inst.dst.kind == OperandKind.VCC:
            self.vcc = sdst_val

        self.scc = scc
        return sdst_val, scc


class TestSaluAndn2B64ScalarInterpreter(unittest.TestCase):
    """Test suite for deterministic scalar instruction execution."""

    def test_scalar_interpreter_execution(self):
        state = ScalarRegisterState()
        state.write_sgpr_u64(0, 0x123456789ABCDEF0)
        state.write_sgpr_u64(2, 0x00FF00FF0F0F0F0F)
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B64,
            dst=ASTOperand.sgpr(4, 2),
            src0=ASTOperand.sgpr(0, 2),
            src1=ASTOperand.sgpr(2, 2),
        )
        sdst, scc = state.execute(inst)
        expected = 0x123456789ABCDEF0 & (~0x00FF00FF0F0F0F0F & MASK64)
        self.assertEqual(sdst, expected)
        self.assertEqual(state.sgpr[4], expected & MASK32)
        self.assertEqual(state.sgpr[5], (expected >> 32) & MASK32)
        self.assertEqual(state.scc, 1)


class TestSaluAndn2B64SimdDivergence(unittest.TestCase):
    """Test suite for SIMD multi-lane divergence and 64-bit EXEC masking."""

    def test_wave64_lane_mask_isolation(self):
        lanes = lambda lo, hi: sum(1 << l for l in range(lo, hi))
        s0 = lanes(0, 48)  # lanes 0..47
        s1 = lanes(16, 64) # lanes 16..63
        # s0 & ~s1 = lanes 0..15
        sdst, scc = ref_andn2_b64(s0, s1)
        self.assertEqual(sdst, lanes(0, 16))
        self.assertEqual(scc, 1)

    def test_divergent_mask_checkerboard(self):
        even_lanes = 0x5555555555555555
        odd_lanes = 0xAAAAAAAAAAAAAAAA
        # even & ~odd = even & even = even
        sdst, scc = ref_andn2_b64(even_lanes, odd_lanes)
        self.assertEqual(sdst, even_lanes)
        self.assertEqual(scc, 1)

        # even & ~even = even & odd = 0
        sdst_zero, scc_zero = ref_andn2_b64(even_lanes, even_lanes)
        self.assertEqual(sdst_zero, 0)
        self.assertEqual(scc_zero, 0)


class TestSaluAndn2B64RegisterAliasing(unittest.TestCase):
    """Test suite for register aliasing and in-place operand preservation."""

    def test_dst_aliases_src0(self):
        # s[0:1] = s[0:1] & ~s[2:3]
        state = ScalarRegisterState()
        s0_val = 0x123456789ABCDEF0
        s1_val = 0x00000000FFFFFFFF
        state.write_sgpr_u64(0, s0_val)
        state.write_sgpr_u64(2, s1_val)

        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B64,
            dst=ASTOperand.sgpr(0, 2),
            src0=ASTOperand.sgpr(0, 2),
            src1=ASTOperand.sgpr(2, 2),
        )
        sdst, scc = state.execute(inst)
        self.assertEqual(sdst, 0x1234567800000000)
        self.assertEqual(state.sgpr[0], 0x00000000)
        self.assertEqual(state.sgpr[1], 0x12345678)
        self.assertEqual(scc, 1)

    def test_dst_aliases_src1(self):
        # s[2:3] = s[0:1] & ~s[2:3]
        state = ScalarRegisterState()
        s0_val = 0x123456789ABCDEF0
        s1_val = 0x00000000FFFFFFFF
        state.write_sgpr_u64(0, s0_val)
        state.write_sgpr_u64(2, s1_val)

        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B64,
            dst=ASTOperand.sgpr(2, 2),
            src0=ASTOperand.sgpr(0, 2),
            src1=ASTOperand.sgpr(2, 2),
        )
        sdst, scc = state.execute(inst)
        self.assertEqual(sdst, 0x1234567800000000)
        self.assertEqual(state.sgpr[2], 0x00000000)
        self.assertEqual(state.sgpr[3], 0x12345678)
        self.assertEqual(scc, 1)

    def test_self_cancellation_aliasing(self):
        # s[0:1] = s[0:1] & ~s[0:1] = 0
        state = ScalarRegisterState()
        state.write_sgpr_u64(0, 0x123456789ABCDEF0)
        inst = ASTInstruction(
            opcode=Opcode.S_ANDN2_B64,
            dst=ASTOperand.sgpr(0, 2),
            src0=ASTOperand.sgpr(0, 2),
            src1=ASTOperand.sgpr(0, 2),
        )
        sdst, scc = state.execute(inst)
        self.assertEqual(sdst, 0)
        self.assertEqual(state.sgpr[0], 0)
        self.assertEqual(state.sgpr[1], 0)
        self.assertEqual(scc, 0)


class TestSaluAndn2B64ConditionCodes(unittest.TestCase):
    """Test suite for condition codes, status flags, and SCC generation."""

    def test_scc_set_when_result_nonzero(self):
        sdst, scc = ref_andn2_b64(1, 0)
        self.assertEqual(sdst, 1)
        self.assertEqual(scc, 1)

    def test_scc_cleared_when_result_zero(self):
        sdst, scc = ref_andn2_b64(0x1234, 0x1234)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)

    def test_scc_cleared_when_src0_is_zero(self):
        sdst, scc = ref_andn2_b64(0, 0x5555555555555555)
        self.assertEqual(sdst, 0)
        self.assertEqual(scc, 0)


class TestSaluAndn2B64BitwiseEdgeCases(unittest.TestCase):
    """Test suite for 64-bit IEEE-754 bit representations and boundary bitmasks."""

    def test_ieee754_double_precision_bit_patterns(self):
        cases = [
            0x7FF0000000000000,  # +inf
            0xFFF0000000000000,  # -inf
            0x7FF8000000000000,  # Quiet NaN
            0x7FF0000000000001,  # Signaling NaN
            0x0000000000000000,  # +0.0
            0x8000000000000000,  # -0.0
            0x0000000000000001,  # Minimum subnormal
            0x000FFFFFFFFFFFFF,  # Maximum subnormal
        ]
        for pattern0 in cases:
            for pattern1 in cases:
                with self.subTest(p0=hex(pattern0), p1=hex(pattern1)):
                    sdst, scc = ref_andn2_b64(pattern0, pattern1)
                    expected = pattern0 & (~pattern1 & MASK64)
                    self.assertEqual(sdst, expected)
                    self.assertEqual(scc, int(expected != 0))

    def test_boundary_alternating_bitmasks(self):
        checkerboards = [
            0x5555555555555555,
            0xAAAAAAAAAAAAAAAA,
            0x3333333333333333,
            0xCCCCCCCCCCCCCCCC,
            0x0F0F0F0F0F0F0F0F,
            0xF0F0F0F0F0F0F0F0,
            0x00FF00FF00FF00FF,
            0xFF00FF00FF00FF00,
        ]
        for cb0 in checkerboards:
            for cb1 in checkerboards:
                sdst, scc = ref_andn2_b64(cb0, cb1)
                self.assertEqual(sdst, cb0 & (~cb1 & MASK64))


def parse_s_andn2_b64_asm(asm_line: str) -> ASTInstruction:
    """Parse a single assembly instruction line for s_andn2_b64."""
    clean = asm_line.strip()
    if not clean:
        raise ValueError("Empty assembly line")
    parts = clean.split(None, 1)
    mnemonic = parts[0].lower()
    if mnemonic != "s_andn2_b64":
        raise ValueError(f"Expected mnemonic s_andn2_b64, got {mnemonic}")
    if len(parts) < 2:
        raise ValueError("Missing operands")
    operands_str = [op.strip() for op in parts[1].split(",")]
    if len(operands_str) != 3:
        raise ValueError(f"Expected 3 operands (dst, src0, src1), got {len(operands_str)}")

    def parse_op(op_str: str) -> ASTOperand:
        op_lower = op_str.lower()
        if op_lower == "exec":
            return ASTOperand.exec_mask()
        elif op_lower == "vcc":
            return ASTOperand.vcc()
        elif op_lower.startswith("s[") and op_lower.endswith("]"):
            idx_pair = op_lower[2:-1].split(":")
            if len(idx_pair) != 2:
                raise ValueError(f"Invalid SGPR pair syntax: {op_str}")
            start, end = int(idx_pair[0]), int(idx_pair[1])
            if end != start + 1 or start % 2 != 0:
                raise ValueError(f"SGPR pair must be even-aligned 2-register range: {op_str}")
            return ASTOperand.sgpr(start, 2)
        elif op_str.startswith("0x") or op_str.startswith("-") or op_str.isdigit():
            val = int(op_str, 0)
            return ASTOperand.literal64(val)
        else:
            raise ValueError(f"Invalid operand: {op_str}")

    dst = parse_op(operands_str[0])
    src0 = parse_op(operands_str[1])
    src1 = parse_op(operands_str[2])

    inst = ASTInstruction(Opcode.S_ANDN2_B64, dst, src0, src1)
    inst.validate()
    return inst


# Mutant functions for mutation testing
def mutant_and(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 1: uses regular AND (omits ~ negation)."""
    res = (s0 & s1) & MASK64
    return res, 1 if res != 0 else 0


def mutant_orn2(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 2: uses OR instead of AND."""
    res = (s0 | (~s1 & MASK64)) & MASK64
    return res, 1 if res != 0 else 0


def mutant_swapped_operands(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 3: swaps operand order (s1 & ~s0)."""
    res = (s1 & (~s0 & MASK64)) & MASK64
    return res, 1 if res != 0 else 0


def mutant_invert_scc(s0: int, s1: int) -> tuple[int, int]:
    """Mutant 4: inverts SCC logic."""
    res = (s0 & (~s1 & MASK64)) & MASK64
    return res, 1 if res == 0 else 0


class TestSaluAndn2B64NegativeControlsAndMutations(unittest.TestCase):
    """Test suite for syntax rejection, negative controls, mutation detection, and translator execution."""

    def test_syntax_rejection_malformed_syntax(self):
        with self.assertRaises(ValueError):
            parse_s_andn2_b64_asm("")

        with self.assertRaises(ValueError):
            parse_s_andn2_b64_asm("s_orn2_b64 s[0:1], s[2:3], s[4:5]")

        with self.assertRaises(ValueError):
            parse_s_andn2_b64_asm("s_andn2_b64 s[0:1], s[2:3]")

        with self.assertRaises(ValueError):
            parse_s_andn2_b64_asm("s_andn2_b64 s[1:2], s[2:3], s[4:5]")

    def test_mutation_detection_and(self):
        s0, s1 = 0x0F, 0x0F
        correct = ref_andn2_b64(s0, s1)
        mutated = mutant_and(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_orn2(self):
        s0, s1 = 0x00, 0x00
        correct = ref_andn2_b64(s0, s1)
        mutated = mutant_orn2(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_swapped_operands(self):
        s0, s1 = 0x0F, 0x00
        correct = ref_andn2_b64(s0, s1)
        mutated = mutant_swapped_operands(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_mutation_detection_invert_scc(self):
        s0, s1 = 0x00, 0x00
        correct = ref_andn2_b64(s0, s1)
        mutated = mutant_invert_scc(s0, s1)
        self.assertNotEqual(correct, mutated)

    def test_andn2_spirv_emission(self):
        """Verify that support.translate_lines emits expected SPIR-V logic for s_andn2_b64."""
        lines = [
            "s_mov_b32 s0, 0x12345678",
            "s_mov_b32 s1, 0x9abcdef0",
            "s_mov_b32 s2, 0x00ff00ff",
            "s_mov_b32 s3, 0x0f0f0f0f",
            "s_andn2_b64 s[4:5], s[0:1], s[2:3]",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_test_andn2_b64_emit", lines)
        self.assertIn("OpBitwiseAnd", spv)
        self.assertIn("OpNot", spv)

    def test_andn2_hardware_execution(self):
        """Verify hardware-accurate execution of s_andn2_b64 through support.mrt0_words."""
        s0_lo, s0_hi = 0x12345678, 0x9ABCDEF0
        s1_lo, s1_hi = 0x00FF00FF, 0x0F0F0F0F
        expected_lo = (s0_lo & (~s1_lo & MASK32)) & MASK32
        expected_hi = (s0_hi & (~s1_hi & MASK32)) & MASK32

        lines = [
            f"s_mov_b32 s0, {s0_lo:#x}",
            f"s_mov_b32 s1, {s0_hi:#x}",
            f"s_mov_b32 s2, {s1_lo:#x}",
            f"s_mov_b32 s3, {s1_hi:#x}",
            "s_andn2_b64 s[4:5], s[0:1], s[2:3]",
            "s_cselect_b32 s6, 1, 0",
            "v_mov_b32_e32 v0, s4",
            "v_mov_b32_e32 v1, s5",
            "v_mov_b32_e32 v2, s6",
            "v_mov_b32_e32 v3, 0",
            "exp mrt0, v0, v1, v2, v3 done",
            "s_endpgm",
        ]
        spv = support.translate_lines("ps_andn2_b64_hw", lines)
        words = support.mrt0_words(spv)

        self.assertEqual(words[0], expected_lo)
        self.assertEqual(words[1], expected_hi)
        self.assertEqual(words[2], 1)
        self.assertEqual(words[3], 0)


if __name__ == "__main__":
    unittest.main()
