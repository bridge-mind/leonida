from __future__ import annotations

import dataclasses
import enum
import os
import struct
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

try:
    import rdna2spirv  # noqa: E402
    import support  # noqa: E402
    import lds_run  # noqa: E402
except Exception:  # noqa: E722
    rdna2spirv = None  # type: ignore
    support = None  # type: ignore
    lds_run = None  # type: ignore

MASK32: int = 0xFFFFFFFF
SIGN_BIT32: int = 0x80000000
DEFAULT_LDS_DWORDS: int = 16384  # 64 KiB capacity in 32-bit dwords


# ---------------------------------------------------------------------------
# Reference Mathematical Oracle and Architectural Model
# ---------------------------------------------------------------------------

def ref_ds_or_b32(curr_val: int, or_mask: int) -> int:
    """Pure mathematical reference oracle for 32-bit bitwise OR:

    Computes:
      LDS[addr] = (LDS[addr] | or_mask) & 0xFFFFFFFF

    Properties:
      - Identity: x | 0 == x
      - Saturation: x | 0xFFFFFFFF == 0xFFFFFFFF
      - Idempotence: x | x == x
      - Monotonicity: bits only transition 0 -> 1, never 1 -> 0
    """
    return ((curr_val & MASK32) | (or_mask & MASK32)) & MASK32


def calc_lds_effective_address(vaddr: int, offset16: int = 0) -> int:
    """Calculates 32-bit effective byte address with 16-bit unsigned offset."""
    return (vaddr + (offset16 & 0xFFFF)) & MASK32


def calc_lds_dword_index(eff_addr: int, capacity_dwords: int = DEFAULT_LDS_DWORDS) -> int:
    """Calculates aligned dword index, dropping unaligned low 2 bits and wrapping modulo capacity."""
    return (eff_addr >> 2) & (capacity_dwords - 1)


class LDSMemory:
    """Architectural model of Local Data Share (LDS) scratchpad memory."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if size_dwords <= 0 or (size_dwords & (size_dwords - 1)) != 0:
            raise ValueError(f"size_dwords must be a power of two, got {size_dwords}")
        self.size_dwords: int = size_dwords
        self.data: list[int] = [0] * size_dwords

    def read_dword(self, eff_addr: int) -> int:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        return self.data[idx] & MASK32

    def write_dword(self, eff_addr: int, val: int) -> None:
        idx = calc_lds_dword_index(eff_addr, self.size_dwords)
        self.data[idx] = val & MASK32

    def atomic_or_b32(self, vaddr: int, or_mask: int, offset16: int = 0) -> int:
        """Executes non-returning 32-bit atomic OR; returns updated memory dword."""
        eff_addr = calc_lds_effective_address(vaddr, offset16)
        curr = self.read_dword(eff_addr)
        updated = ref_ds_or_b32(curr, or_mask)
        self.write_dword(eff_addr, updated)
        return updated


class TestLdsOrB32Oracle(unittest.TestCase):
    """Unit tests for the reference mathematical oracle and architectural LDS models."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)

    def test_oracle_basic_or(self) -> None:
        self.lds.write_dword(0x40, 0xF0F00000)
        updated = self.lds.atomic_or_b32(0x40, 0x0F0F0000)
        self.assertEqual(updated, 0xFFFF0000)
        self.assertEqual(self.lds.read_dword(0x40), 0xFFFF0000)

    def test_oracle_identity_zero_mask(self) -> None:
        cases = [0x0, 0x1, 0x12345678, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_ds_or_b32(val, 0), val)

    def test_oracle_saturation_all_ones(self) -> None:
        cases = [0x0, 0x1, 0x55555555, 0xAAAAAAAA, 0x80000000, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_ds_or_b32(val, MASK32), MASK32)

    def test_oracle_idempotence(self) -> None:
        cases = [0x0, 0x10203040, 0x55AA55AA, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                self.assertEqual(ref_ds_or_b32(val, val), val)

    def test_oracle_address_calculation_with_offset(self) -> None:
        self.lds.write_dword(0x44, 0x10)
        # vaddr=0x40, offset=4 -> eff_addr=0x44
        updated = self.lds.atomic_or_b32(0x40, 0x20, offset16=4)
        self.assertEqual(updated, 0x30)
        self.assertEqual(self.lds.read_dword(0x44), 0x30)

    def test_oracle_address_dword_alignment(self) -> None:
        # Byte addresses 0x40, 0x41, 0x42, 0x43 all map to dword index 16
        self.lds.write_dword(0x40, 0x100)
        self.lds.atomic_or_b32(0x41, 0x01)
        self.lds.atomic_or_b32(0x42, 0x02)
        self.lds.atomic_or_b32(0x43, 0x04)
        self.assertEqual(self.lds.read_dword(0x40), 0x107)

    def test_oracle_capacity_wrapping(self) -> None:
        # LDS with 1024 dwords (4096 bytes): address 4096 wraps to index 0
        self.lds.write_dword(0, 0x10)
        self.lds.atomic_or_b32(4096, 0x20)
        self.assertEqual(self.lds.read_dword(0), 0x30)


# ---------------------------------------------------------------------------
# Symbolic AST Opcode Definitions and Typed Expressions
# ---------------------------------------------------------------------------

class RegKind(enum.Enum):
    """Register kinds in RDNA2 architecture."""
    VGPR = "v"
    SGPR = "s"


@dataclasses.dataclass(frozen=True)
class Register:
    """Register operand with kind, index, and grouping."""
    kind: RegKind
    index: int
    count: int = 1

    def format_asm(self) -> str:
        if self.count == 1:
            return f"{self.kind.value}{self.index}"
        return f"{self.kind.value}[{self.index}:{self.index + self.count - 1}]"

    @property
    def is_vgpr(self) -> bool:
        return self.kind == RegKind.VGPR


@dataclasses.dataclass(frozen=True)
class LdsModifiers:
    """Modifiers and options for LDS instructions."""
    offset: int = 0

    def __post_init__(self) -> None:
        if self.offset < 0 or self.offset > 65535:
            raise ValueError(f"LDS offset must be in range [0, 65535]: {self.offset}")


class Opcode(enum.Enum):
    """LDS instruction opcodes."""
    DS_OR_B32 = "ds_or_b32"


@dataclasses.dataclass(frozen=True)
class Instruction:
    """Symbolic AST node representing a ds_or_b32 instruction."""
    opcode: Opcode
    vaddr: Register
    vdata: Register
    mods: LdsModifiers = dataclasses.field(default_factory=LdsModifiers)

    def __post_init__(self) -> None:
        if not self.vaddr.is_vgpr or self.vaddr.count != 1:
            raise ValueError(f"vaddr must be a single VGPR: {self.vaddr}")
        if not self.vdata.is_vgpr or self.vdata.count != 1:
            raise ValueError(f"vdata must be a single VGPR: {self.vdata}")

    def to_assembly(self) -> str:
        va_str = self.vaddr.format_asm()
        vd_str = self.vdata.format_asm()
        off_str = f" offset:{self.mods.offset}" if self.mods.offset != 0 else ""
        return f"{self.opcode.value} {va_str}, {vd_str}{off_str}"


@dataclasses.dataclass
class Program:
    """Symbolic sequence of LDS instructions."""
    instructions: list[Instruction] = dataclasses.field(default_factory=list)

    def append(self, inst: Instruction) -> None:
        self.instructions.append(inst)

    def to_assembly(self) -> str:
        return "\n".join(inst.to_assembly() for inst in self.instructions)


class TestLdsOrB32AST(unittest.TestCase):
    """Unit tests for AST data structures and assembly emission."""

    def test_ast_instruction_creation(self) -> None:
        inst = Instruction(
            opcode=Opcode.DS_OR_B32,
            vaddr=Register(RegKind.VGPR, 0),
            vdata=Register(RegKind.VGPR, 1),
        )
        self.assertEqual(inst.opcode, Opcode.DS_OR_B32)
        self.assertEqual(inst.vaddr.format_asm(), "v0")
        self.assertEqual(inst.vdata.format_asm(), "v1")
        self.assertEqual(inst.to_assembly(), "ds_or_b32 v0, v1")

    def test_ast_to_assembly_with_offset(self) -> None:
        inst = Instruction(
            opcode=Opcode.DS_OR_B32,
            vaddr=Register(RegKind.VGPR, 2),
            vdata=Register(RegKind.VGPR, 3),
            mods=LdsModifiers(offset=64),
        )
        self.assertEqual(inst.to_assembly(), "ds_or_b32 v2, v3 offset:64")

    def test_ast_program_assembly(self) -> None:
        prog = Program()
        prog.append(Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1)))
        prog.append(Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 2), LdsModifiers(offset=16)))
        lines = prog.to_assembly().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], "ds_or_b32 v0, v1")
        self.assertEqual(lines[1], "ds_or_b32 v0, v2 offset:16")


# ---------------------------------------------------------------------------
# Condition Codes and Status Flags
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class LdsConditionCodes:
    """Condition codes reflecting 32-bit bitwise operation result values."""
    zero: bool
    negative: bool
    positive: bool


def compute_lds_condition_codes(result: int) -> LdsConditionCodes:
    """Computes condition codes from a 32-bit bitwise operation result."""
    val32 = result & MASK32
    return LdsConditionCodes(
        zero=(val32 == 0),
        negative=bool(val32 & SIGN_BIT32),
        positive=(val32 > 0 and not bool(val32 & SIGN_BIT32)),
    )


# ---------------------------------------------------------------------------
# Deterministic Scalar Instruction Interpreter
# ---------------------------------------------------------------------------

class ScalarLdsInterpreter:
    """Deterministic scalar interpreter for ds_or_b32."""

    def __init__(self, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        self.vgprs: list[int] = [0] * 256
        self.vcc: int = 0
        self.scc: int = 0
        self.lds: LDSMemory = LDSMemory(size_dwords)
        self.trace: list[str] = []
        self.last_condition_codes: LdsConditionCodes | None = None

    def read_vgpr(self, idx: int) -> int:
        if idx < 0 or idx >= 256:
            raise IndexError(f"VGPR index out of bounds: v{idx}")
        return self.vgprs[idx] & MASK32

    def write_vgpr(self, idx: int, val: int) -> None:
        if idx < 0 or idx >= 256:
            raise IndexError(f"VGPR index out of bounds: v{idx}")
        self.vgprs[idx] = val & MASK32

    def step(self, inst: Instruction) -> int:
        """Executes a single ds_or_b32 instruction; returns updated memory value."""
        vaddr = self.read_vgpr(inst.vaddr.index)
        vdata = self.read_vgpr(inst.vdata.index)
        updated = self.lds.atomic_or_b32(vaddr, vdata, offset16=inst.mods.offset)

        # ds_or_b32 is non-returning: VGPRs remain unchanged!
        self.last_condition_codes = compute_lds_condition_codes(updated)
        self.trace.append(f"{inst.to_assembly()} => lds={updated:#010x}")
        return updated

    def run_program(self, prog: Program) -> list[int]:
        """Executes all instructions in a program sequentially."""
        return [self.step(inst) for inst in prog.instructions]


class TestLdsOrB32ScalarInterpreter(unittest.TestCase):
    """Unit tests for the deterministic scalar instruction interpreter."""

    def setUp(self) -> None:
        self.interp = ScalarLdsInterpreter(size_dwords=1024)

    def test_scalar_step_basic_or(self) -> None:
        self.interp.lds.write_dword(0x10, 0x1100)
        self.interp.write_vgpr(0, 0x10)    # vaddr
        self.interp.write_vgpr(1, 0x0022)  # vdata

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        updated = self.interp.step(inst)
        self.assertEqual(updated, 0x1122)
        self.assertEqual(self.interp.lds.read_dword(0x10), 0x1122)

    def test_scalar_step_with_immediate_offset(self) -> None:
        self.interp.lds.write_dword(0x20, 0x0F)
        self.interp.write_vgpr(2, 0x10)    # vaddr=0x10
        self.interp.write_vgpr(3, 0xF0)    # vdata

        # 0x10 + 0x10 offset = 0x20
        inst = Instruction(
            Opcode.DS_OR_B32,
            Register(RegKind.VGPR, 2),
            Register(RegKind.VGPR, 3),
            LdsModifiers(offset=0x10),
        )
        updated = self.interp.step(inst)
        self.assertEqual(updated, 0xFF)
        self.assertEqual(self.interp.lds.read_dword(0x20), 0xFF)

    def test_scalar_non_returning_preserves_vgprs(self) -> None:
        self.interp.write_vgpr(0, 0x40)
        self.interp.write_vgpr(1, 0xDEADBEEF)

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        self.interp.step(inst)

        # Neither vaddr nor vdata are modified
        self.assertEqual(self.interp.read_vgpr(0), 0x40)
        self.assertEqual(self.interp.read_vgpr(1), 0xDEADBEEF)


# ---------------------------------------------------------------------------
# SIMD Multi-Lane Wavefront Interpreter (Wave32 and Wave64)
# ---------------------------------------------------------------------------

class SIMDLdsWavefrontInterpreter:
    """SIMD wavefront interpreter supporting Wave32 and Wave64 execution modes."""

    def __init__(self, wave_size: int = 32, size_dwords: int = DEFAULT_LDS_DWORDS) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got: {wave_size}")
        self.wave_size: int = wave_size
        self.vgprs: list[list[int]] = [[0] * 256 for _ in range(wave_size)]
        self.exec_mask: int = (1 << wave_size) - 1
        self.lds: LDSMemory = LDSMemory(size_dwords)

    def read_lane_vgpr(self, lane: int, idx: int) -> int:
        return self.vgprs[lane][idx] & MASK32

    def write_lane_vgpr(self, lane: int, idx: int, val: int) -> None:
        self.vgprs[lane][idx] = val & MASK32

    def set_exec_mask(self, mask: int) -> None:
        limit = (1 << self.wave_size) - 1
        self.exec_mask = mask & limit

    def step(self, inst: Instruction) -> list[int]:
        """Executes ds_or_b32 across all active lanes; returns updated memory values."""
        results: list[int] = []
        for lane in range(self.wave_size):
            lane_active = bool((self.exec_mask >> lane) & 1)
            if not lane_active:
                results.append(0)
                continue

            vaddr = self.read_lane_vgpr(lane, inst.vaddr.index)
            vdata = self.read_lane_vgpr(lane, inst.vdata.index)
            updated = self.lds.atomic_or_b32(vaddr, vdata, offset16=inst.mods.offset)
            results.append(updated)
        return results


class TestLdsOrB32SIMD(unittest.TestCase):
    """Unit tests for SIMD execution divergence across Wave32 and Wave64 wavefronts."""

    def test_simd_wave32_all_active_independent_addresses(self) -> None:
        simd = SIMDLdsWavefrontInterpreter(wave_size=32, size_dwords=1024)
        for lane in range(32):
            addr = lane * 4
            simd.lds.write_dword(addr, 0x1000)
            simd.write_lane_vgpr(lane, 0, addr)       # vaddr
            simd.write_lane_vgpr(lane, 1, 1 << lane)  # vdata

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.step(inst)

        for lane in range(32):
            self.assertEqual(simd.lds.read_dword(lane * 4), 0x1000 | (1 << lane))

    def test_simd_wave32_divergent_alternating_mask(self) -> None:
        simd = SIMDLdsWavefrontInterpreter(wave_size=32, size_dwords=1024)
        simd.set_exec_mask(0x55555555)  # Even lanes active, odd inactive

        for lane in range(32):
            addr = lane * 4
            simd.lds.write_dword(addr, 0x00FF)
            simd.write_lane_vgpr(lane, 0, addr)
            simd.write_lane_vgpr(lane, 1, 0xFF00)

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.step(inst)

        for lane in range(32):
            if lane % 2 == 0:
                self.assertEqual(simd.lds.read_dword(lane * 4), 0xFFFF)
            else:
                self.assertEqual(simd.lds.read_dword(lane * 4), 0x00FF)

    def test_simd_wave64_divergent_half_mask(self) -> None:
        simd = SIMDLdsWavefrontInterpreter(wave_size=64, size_dwords=1024)
        simd.set_exec_mask(0x00000000FFFFFFFF)  # Lower 32 lanes active

        for lane in range(64):
            addr = lane * 4
            simd.lds.write_dword(addr, 0x1200)
            simd.write_lane_vgpr(lane, 0, addr)
            simd.write_lane_vgpr(lane, 1, 0x0034)

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.step(inst)

        for lane in range(64):
            if lane < 32:
                self.assertEqual(simd.lds.read_dword(lane * 4), 0x1234)
            else:
                self.assertEqual(simd.lds.read_dword(lane * 4), 0x1200)

    def test_simd_multi_lane_atomic_reduction_to_same_address(self) -> None:
        # Multiple active lanes targeting the exact same LDS dword atomically aggregate their bits
        simd = SIMDLdsWavefrontInterpreter(wave_size=32, size_dwords=1024)
        simd.lds.write_dword(0x80, 0x00000000)

        for lane in range(16):
            simd.write_lane_vgpr(lane, 0, 0x80)       # Same address
            simd.write_lane_vgpr(lane, 1, 1 << lane)  # Distinct bit
        simd.set_exec_mask(0x0000FFFF)  # First 16 lanes active

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.step(inst)

        # All 16 bits [0:15] must be atomically aggregated into memory
        self.assertEqual(simd.lds.read_dword(0x80), 0x0000FFFF)


# ---------------------------------------------------------------------------
# Register Aliasing and In-Place Operand Preservation
# ---------------------------------------------------------------------------

class TestLdsOrB32RegisterAliasing(unittest.TestCase):
    """Unit tests for overlapping register allocations and in-place operand preservation."""

    def setUp(self) -> None:
        self.interp = ScalarLdsInterpreter(size_dwords=1024)

    def test_aliasing_vaddr_same_as_vdata(self) -> None:
        # ds_or_b32 v0, v0: v0 is both address and data mask!
        # When v0 = 0x40, effective byte address is 0x40, and OR mask is 0x40.
        self.interp.lds.write_dword(0x40, 0x01)
        self.interp.write_vgpr(0, 0x40)

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 0))
        updated = self.interp.step(inst)
        self.assertEqual(updated, 0x41)
        self.assertEqual(self.interp.lds.read_dword(0x40), 0x41)
        # Non-returning: v0 remains 0x40
        self.assertEqual(self.interp.read_vgpr(0), 0x40)

    def test_preserve_all_vgprs_post_execution(self) -> None:
        # Verify that registers v0..v7 remain unmodified
        for reg in range(8):
            self.interp.write_vgpr(reg, 0x1000 + reg)
        self.interp.lds.write_dword(0x1000, 0)

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        self.interp.step(inst)

        for reg in range(8):
            self.assertEqual(self.interp.read_vgpr(reg), 0x1000 + reg)


# ---------------------------------------------------------------------------
# Condition Codes, Status Flags, and Carry/Borrow Isolation Tests
# ---------------------------------------------------------------------------

class TestLdsOrB32ConditionCodes(unittest.TestCase):
    """Unit tests for condition code tracking and VCC/SCC isolation."""

    def test_condition_codes_zero(self) -> None:
        cc = compute_lds_condition_codes(0)
        self.assertTrue(cc.zero)
        self.assertFalse(cc.negative)
        self.assertFalse(cc.positive)

    def test_condition_codes_negative(self) -> None:
        cc = compute_lds_condition_codes(SIGN_BIT32 | 0x42)
        self.assertFalse(cc.zero)
        self.assertTrue(cc.negative)
        self.assertFalse(cc.positive)

    def test_condition_codes_positive(self) -> None:
        cc = compute_lds_condition_codes(0x00001234)
        self.assertFalse(cc.zero)
        self.assertFalse(cc.negative)
        self.assertTrue(cc.positive)

    def test_vcc_and_scc_isolation(self) -> None:
        interp = ScalarLdsInterpreter(size_dwords=1024)
        interp.vcc = 0xDEADBEEFCAFEBABE
        interp.scc = 1
        interp.lds.write_dword(0, 0)
        interp.write_vgpr(0, 0)
        interp.write_vgpr(1, 0xFF)

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        interp.step(inst)

        # Neither VCC nor SCC can be modified by LDS instructions
        self.assertEqual(interp.vcc, 0xDEADBEEFCAFEBABE)
        self.assertEqual(interp.scc, 1)


# ---------------------------------------------------------------------------
# IEEE 754 Boundary Extremes, Subnormals, and NaN Handling
# ---------------------------------------------------------------------------

class TestLdsOrB32BoundaryExtremes(unittest.TestCase):
    """Unit tests for boundary extremes, integer wrapping, and IEEE 754 bit representations."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_dwords=1024)

    def test_32bit_integer_extremes(self) -> None:
        # INT32_MAX | INT32_MIN = UINT32_MAX
        self.lds.write_dword(0, 0x7FFFFFFF)
        updated = self.lds.atomic_or_b32(0, 0x80000000)
        self.assertEqual(updated, 0xFFFFFFFF)

    def test_float32_bit_pattern_preservation(self) -> None:
        # Untyped bitwise OR operates on raw bits without float quietening or subnormal flushing
        qnan_bits = 0x7FC00000
        subnormal_bits = 0x00000001
        self.lds.write_dword(0x10, qnan_bits)
        updated = self.lds.atomic_or_b32(0x10, subnormal_bits)
        self.assertEqual(updated, qnan_bits | subnormal_bits)

        # +0.0 (0x00000000) | -0.0 (0x80000000) = 0x80000000
        self.lds.write_dword(0x20, 0x00000000)
        updated_zero = self.lds.atomic_or_b32(0x20, 0x80000000)
        self.assertEqual(updated_zero, 0x80000000)

    def test_lds_offset_boundary_max_16bit(self) -> None:
        # Max unsigned 16-bit offset = 0xFFFF (65535)
        # 65535 bytes = dword index (65535 >> 2) & 1023 = 16383 & 1023 = 1023
        self.lds.write_dword(1023 * 4, 0x10)
        updated = self.lds.atomic_or_b32(0, 0x20, offset16=0xFFFF)
        self.assertEqual(updated, 0x30)
        self.assertEqual(self.lds.read_dword(1023 * 4), 0x30)


# ---------------------------------------------------------------------------
# Negative Controls, Syntax Rejection, and Mutation Detection
# ---------------------------------------------------------------------------

class TestLdsOrB32NegativeControls(unittest.TestCase):
    """Negative controls and syntax rejection tests for ds_or_b32."""

    def test_invalid_vaddr_kind_and_count(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(Opcode.DS_OR_B32, Register(RegKind.SGPR, 0), Register(RegKind.VGPR, 1))
        with self.assertRaises(ValueError):
            Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0, count=2), Register(RegKind.VGPR, 1))

    def test_invalid_vdata_kind_and_count(self) -> None:
        with self.assertRaises(ValueError):
            Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.SGPR, 1))
        with self.assertRaises(ValueError):
            Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1, count=2))

    def test_invalid_offset_range(self) -> None:
        with self.assertRaises(ValueError):
            LdsModifiers(offset=-1)
        with self.assertRaises(ValueError):
            LdsModifiers(offset=65536)

    def test_invalid_wave_size(self) -> None:
        with self.assertRaises(ValueError):
            SIMDLdsWavefrontInterpreter(wave_size=16)
        with self.assertRaises(ValueError):
            SIMDLdsWavefrontInterpreter(wave_size=128)

    def test_register_bounds_scalar_interpreter(self) -> None:
        interp = ScalarLdsInterpreter(size_dwords=1024)
        with self.assertRaises(IndexError):
            interp.read_vgpr(-1)
        with self.assertRaises(IndexError):
            interp.read_vgpr(256)
        with self.assertRaises(IndexError):
            interp.write_vgpr(-1, 0)
        with self.assertRaises(IndexError):
            interp.write_vgpr(256, 0)


class TestLdsOrB32MutationDetection(unittest.TestCase):
    """Tests ensuring mutation kill for common semantic implementation errors."""

    def test_mutant_and_instead_of_or_killed(self) -> None:
        # A mutant computing bitwise AND instead of OR must be rejected
        self.assertNotEqual(ref_ds_or_b32(0xF0, 0x0F), 0xF0 & 0x0F)
        self.assertEqual(ref_ds_or_b32(0xF0, 0x0F), 0xFF)

    def test_mutant_xor_instead_of_or_killed(self) -> None:
        # A mutant computing bitwise XOR instead of OR must be rejected
        self.assertNotEqual(ref_ds_or_b32(0xFF, 0xFF), 0xFF ^ 0xFF)
        self.assertEqual(ref_ds_or_b32(0xFF, 0xFF), 0xFF)

    def test_mutant_inactive_lane_leak_killed(self) -> None:
        simd = SIMDLdsWavefrontInterpreter(wave_size=32, size_dwords=1024)
        simd.lds.write_dword(0, 0x10)
        simd.lds.write_dword(4, 0x20)

        # Enable only lane 0; lane 1 is inactive
        simd.set_exec_mask(0x1)

        simd.write_lane_vgpr(0, 0, 0)     # vaddr lane 0 = 0
        simd.write_lane_vgpr(0, 1, 0x01)  # vdata lane 0 = 0x01
        simd.write_lane_vgpr(1, 0, 4)     # vaddr lane 1 = 4
        simd.write_lane_vgpr(1, 1, 0x02)  # vdata lane 1 = 0x02

        inst = Instruction(Opcode.DS_OR_B32, Register(RegKind.VGPR, 0), Register(RegKind.VGPR, 1))
        simd.step(inst)

        # Lane 0 active: 0x10 | 0x01 = 0x11
        self.assertEqual(simd.lds.read_dword(0), 0x11)
        # Lane 1 inactive: must remain untouched 0x20
        self.assertEqual(simd.lds.read_dword(4), 0x20)


class TestLdsOrB32TranslationAndExecution(unittest.TestCase):
    """End-to-end SPIR-V translation and execution under support and spirv_mini_interp."""

    def test_translator_ds_or_b32_basic(self) -> None:
        if support is None or rdna2spirv is None or lds_run is None:
            self.skipTest("support/rdna2spirv/lds_run not available in this Python environment")

        asm = [
            "v_mov_b32_e32 v0, 0x10",
            "v_mov_b32_e32 v1, 0x0F",
            "ds_or_b32 v0, v1",
            "s_endpgm",
        ]
        spvasm = support.translate_lines("cs_lds_or_b32_basic", asm, fssm=lds_run.fssm())
        self.assertIn("OpAtomicOr", spvasm)

    def test_execution_ds_or_b32_chain(self) -> None:
        if support is None or rdna2spirv is None or lds_run is None:
            self.skipTest("support/rdna2spirv/lds_run not available in this Python environment")

        # Initialize LDS byte 0x20 with 0xF0, then execute ds_or_b32 with 0x0F
        rets, mem = lds_run.run_chain([("ds_or_b32", 0x20, 0, [0x0F])], {0x20: 0xF0})
        self.assertEqual(mem[0x20], 0xFF)


if __name__ == "__main__":
    unittest.main()
