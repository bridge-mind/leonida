"""Semantic interpreter tests and translator verification for 32-bit unsigned LDS atomic increment with wrap (ds_inc_u32, ds_inc_rtn_u32).

Standard library only; no GPU, no LLVM, no external tools, no game files.

    py -I tests/semantic/test_lds_inc_u32_hw.py
    py -m unittest discover -s tests/semantic -p test_lds_inc_u32_hw.py
"""
from __future__ import annotations

import os
import re
import struct
import sys
import unittest
from typing import Dict, List, Optional, Tuple, Union

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS_DIR = os.path.join(ROOT, "tools")
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import support
import lds_run as L
import spirv_mini_interp as si
import rdna2spirv

MASK32: int = 0xFFFFFFFF
UINT32_MAX: int = 0xFFFFFFFF


def ref_ds_inc_u32(curr_mem: int, data_val: int) -> int:
    """Reference mathematical oracle for RDNA2 ds_inc_u32: M = (M >= DATA) ? 0 : M + 1 (unsigned)."""
    cur = curr_mem & MASK32
    data = data_val & MASK32
    return 0 if cur >= data else (cur + 1) & MASK32


def ref_ds_inc_rtn_u32(curr_mem: int, data_val: int) -> Tuple[int, int]:
    """Reference returning atomic increment with wrap: returns (old_mem, new_mem)."""
    old = curr_mem & MASK32
    new = ref_ds_inc_u32(old, data_val)
    return old, new


class LDSMemory:
    """Architectural model of RDNA2 Local Data Share (LDS) memory."""

    def __init__(self, size_bytes: int = 4096) -> None:
        if size_bytes <= 0 or (size_bytes & (size_bytes - 1)) != 0:
            raise ValueError(f"LDS size must be a positive power of two, got {size_bytes}")
        self.size_bytes = size_bytes
        self.num_dwords = size_bytes // 4
        self.storage = [0] * self.num_dwords

    def resolve_dword_index(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        dword_idx = eff_addr >> 2
        return dword_idx & (self.num_dwords - 1)

    def read_dword(self, byte_addr: int, offset: int = 0) -> int:
        idx = self.resolve_dword_index(byte_addr, offset)
        return self.storage[idx]

    def write_dword(self, byte_addr: int, value: int, offset: int = 0) -> None:
        idx = self.resolve_dword_index(byte_addr, offset)
        self.storage[idx] = value & MASK32

    def atomic_inc_u32(self, byte_addr: int, data_val: int, offset: int = 0, rtn: bool = False) -> Tuple[Optional[int], int]:
        idx = self.resolve_dword_index(byte_addr, offset)
        old_val = self.storage[idx]
        new_val = ref_ds_inc_u32(old_val, data_val)
        self.storage[idx] = new_val
        return (old_val if rtn else None), new_val

    def get_bank(self, byte_addr: int, offset: int = 0) -> int:
        eff_addr = (byte_addr + offset) & MASK32
        return (eff_addr >> 2) & 31

class TestDSIncU32OracleAndArchitecturalModel(unittest.TestCase):
    """Part 1: Reference mathematical oracle and architectural model tests."""

    def test_reference_oracle_basic_increments(self) -> None:
        # Standard increment: cur < data
        self.assertEqual(ref_ds_inc_u32(0, 10), 1)
        self.assertEqual(ref_ds_inc_u32(1, 10), 2)
        self.assertEqual(ref_ds_inc_u32(8, 10), 9)

    def test_reference_oracle_wrap_resets(self) -> None:
        # Wrap when cur == data
        self.assertEqual(ref_ds_inc_u32(10, 10), 0)

        # Wrap when cur > data
        self.assertEqual(ref_ds_inc_u32(15, 10), 0)

        # When data == 0: cur >= 0 is always True (unsigned), so always resets to 0
        self.assertEqual(ref_ds_inc_u32(0, 0), 0)
        self.assertEqual(ref_ds_inc_u32(5, 0), 0)

    def test_reference_oracle_returning_form(self) -> None:
        old, new = ref_ds_inc_rtn_u32(3, 5)
        self.assertEqual(old, 3)
        self.assertEqual(new, 4)

        old, new = ref_ds_inc_rtn_u32(5, 5)
        self.assertEqual(old, 5)
        self.assertEqual(new, 0)

    def test_lds_memory_model(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        addr = 0x40
        lds.write_dword(addr, 4)

        # Non-returning atomic inc with data=10: 4 < 10 -> 5
        old, new = lds.atomic_inc_u32(addr, 10, rtn=False)
        self.assertIsNone(old)
        self.assertEqual(new, 5)
        self.assertEqual(lds.read_dword(addr), 5)

        # Returning atomic inc with data=5: 5 >= 5 -> resets to 0
        old, new = lds.atomic_inc_u32(addr, 5, rtn=True)
        self.assertEqual(old, 5)
        self.assertEqual(new, 0)
        self.assertEqual(lds.read_dword(addr), 0)

class Register:
    def __init__(self, name: str, index: int) -> None:
        self.name = name
        self.index = index

    def __repr__(self) -> str:
        return f"{self.name}{self.index}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Register):
            return False
        return self.name == other.name and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.name, self.index))


class VGPR(Register):
    def __init__(self, index: int) -> None:
        if not (0 <= index <= 255):
            raise ValueError(f"VGPR index out of range [0, 255]: {index}")
        super().__init__("v", index)


class DSIncU32Instruction:
    """Symbolic AST node representing ds_inc_u32 and ds_inc_rtn_u32 instructions."""

    def __init__(
        self,
        mnemonic: str,
        vaddr: VGPR,
        vdata0: VGPR,
        vdst: Optional[VGPR] = None,
        offset: int = 0,
        gds: bool = False,
    ) -> None:
        if mnemonic not in ("ds_inc_u32", "ds_inc_rtn_u32"):
            raise ValueError(f"Invalid mnemonic for ds_inc_u32 AST: {mnemonic}")
        if mnemonic == "ds_inc_rtn_u32" and vdst is None:
            raise ValueError("ds_inc_rtn_u32 requires vdst operand")
        if mnemonic == "ds_inc_u32" and vdst is not None:
            raise ValueError("ds_inc_u32 does not accept vdst operand")
        if not (0 <= offset <= 65535):
            raise ValueError(f"Offset out of range [0, 65535]: {offset}")

        self.mnemonic = mnemonic
        self.vaddr = vaddr
        self.vdata0 = vdata0
        self.vdst = vdst
        self.offset = offset
        self.gds = gds

    @property
    def is_return(self) -> bool:
        return self.mnemonic == "ds_inc_rtn_u32"

    def __repr__(self) -> str:
        parts = [self.mnemonic]
        if self.vdst is not None:
            parts.append(str(self.vdst))
        parts.append(str(self.vaddr))
        parts.append(str(self.vdata0))
        text = f"{parts[0]} {', '.join(parts[1:])}"
        if self.offset:
            text += f" offset:{self.offset}"
        if self.gds:
            text += " gds"
        return text

    @classmethod
    def parse(cls, asm: str) -> "DSIncU32Instruction":
        line = asm.strip()
        if not line:
            raise ValueError("Empty assembly string")
        tokens = line.split()
        mnem = tokens[0]
        if mnem not in ("ds_inc_u32", "ds_inc_rtn_u32"):
            raise ValueError(f"Unsupported mnemonic: {mnem}")

        rest = " ".join(tokens[1:])
        mod_tokens = rest.split()
        clean_operands: list[str] = []
        offset = 0
        gds = False

        for tok in mod_tokens:
            tok_clean = tok.rstrip(",")
            if tok_clean.startswith("offset:"):
                try:
                    offset = int(tok_clean.split(":", 1)[1], 0)
                except ValueError:
                    raise ValueError(f"Invalid offset format: {tok_clean}")
            elif tok_clean == "gds":
                gds = True
            elif tok_clean:
                clean_operands.append(tok_clean)

        def parse_vgpr(reg_str: str) -> VGPR:
            m = re.fullmatch(r"v(\d+)", reg_str)
            if not m:
                raise ValueError(f"Expected VGPR (e.g. v1), got: {reg_str}")
            return VGPR(int(m.group(1)))

        if mnem == "ds_inc_u32":
            if len(clean_operands) != 2:
                raise ValueError(f"ds_inc_u32 requires 2 operands (vaddr, vdata0), got {len(clean_operands)}")
            vaddr = parse_vgpr(clean_operands[0])
            vdata0 = parse_vgpr(clean_operands[1])
            return cls(mnem, vaddr, vdata0, vdst=None, offset=offset, gds=gds)
        else:
            if len(clean_operands) != 3:
                raise ValueError(f"ds_inc_rtn_u32 requires 3 operands (vdst, vaddr, vdata0), got {len(clean_operands)}")
            vdst = parse_vgpr(clean_operands[0])
            vaddr = parse_vgpr(clean_operands[1])
            vdata0 = parse_vgpr(clean_operands[2])
            return cls(mnem, vaddr, vdata0, vdst=vdst, offset=offset, gds=gds)


class TestDSIncU32ASTAndOpcodeDefinitions(unittest.TestCase):
    """Part 2: Symbolic AST opcode definitions and typed expressions."""

    def test_ast_construction_and_properties(self) -> None:
        inst_non_rtn = DSIncU32Instruction("ds_inc_u32", VGPR(1), VGPR(2), offset=16)
        self.assertFalse(inst_non_rtn.is_return)
        self.assertEqual(inst_non_rtn.vaddr, VGPR(1))
        self.assertEqual(inst_non_rtn.vdata0, VGPR(2))
        self.assertIsNone(inst_non_rtn.vdst)
        self.assertEqual(inst_non_rtn.offset, 16)
        self.assertEqual(repr(inst_non_rtn), "ds_inc_u32 v1, v2 offset:16")

        inst_rtn = DSIncU32Instruction("ds_inc_rtn_u32", VGPR(3), VGPR(4), vdst=VGPR(5), offset=32, gds=True)
        self.assertTrue(inst_rtn.is_return)
        self.assertEqual(inst_rtn.vdst, VGPR(5))
        self.assertEqual(repr(inst_rtn), "ds_inc_rtn_u32 v5, v3, v4 offset:32 gds")

    def test_ast_parsing_valid(self) -> None:
        inst1 = DSIncU32Instruction.parse("ds_inc_u32 v10, v20")
        self.assertEqual(inst1.mnemonic, "ds_inc_u32")
        self.assertEqual(inst1.vaddr, VGPR(10))
        self.assertEqual(inst1.vdata0, VGPR(20))

        inst2 = DSIncU32Instruction.parse("ds_inc_rtn_u32 v5, v1, v2 offset:0x40")
        self.assertEqual(inst2.mnemonic, "ds_inc_rtn_u32")
        self.assertEqual(inst2.vdst, VGPR(5))
        self.assertEqual(inst2.offset, 64)

    def test_ast_parsing_invalid(self) -> None:
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("")
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("ds_inc_u32 v1")
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("ds_inc_rtn_u32 v1, v2")
        with self.assertRaises(ValueError):
            DSIncU32Instruction.parse("ds_inc_u32 s1, v2")

class ScalarThreadState:
    """State of a single hardware execution lane / thread."""

    def __init__(self, lds: LDSMemory) -> None:
        self.vgprs: list[int] = [0] * 256
        self.sgprs: list[int] = [0] * 106
        self.lds = lds
        self.lgkm_cnt = 0

    def get_vgpr(self, reg: VGPR) -> int:
        return self.vgprs[reg.index]

    def set_vgpr(self, reg: VGPR, val: int) -> None:
        self.vgprs[reg.index] = val & MASK32


class ScalarDSInterpreter:
    """Deterministic instruction interpreter for scalar LDS increment executions."""

    def __init__(self, lds: LDSMemory) -> None:
        self.lds = lds

    def execute(self, state: ScalarThreadState, inst: DSIncU32Instruction) -> Optional[int]:
        addr = state.get_vgpr(inst.vaddr)
        data = state.get_vgpr(inst.vdata0)

        old_val, new_val = self.lds.atomic_inc_u32(addr, data, offset=inst.offset, rtn=inst.is_return)

        if inst.is_return and inst.vdst is not None:
            assert old_val is not None
            state.set_vgpr(inst.vdst, old_val)

        state.lgkm_cnt += 1
        return old_val


class TestDSIncU32ScalarInterpreter(unittest.TestCase):
    """Part 3: Deterministic scalar instruction interpreter tests."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_bytes=4096)
        self.interp = ScalarDSInterpreter(self.lds)
        self.state = ScalarThreadState(self.lds)

    def test_scalar_execution_non_returning(self) -> None:
        self.lds.write_dword(0x10, 5)
        self.state.set_vgpr(VGPR(1), 0x10)
        self.state.set_vgpr(VGPR(2), 10)

        inst = DSIncU32Instruction.parse("ds_inc_u32 v1, v2")
        res = self.interp.execute(self.state, inst)

        self.assertIsNone(res)
        self.assertEqual(self.lds.read_dword(0x10), 6)

    def test_scalar_execution_returning(self) -> None:
        self.lds.write_dword(0x20, 10)
        self.state.set_vgpr(VGPR(10), 0x20)
        self.state.set_vgpr(VGPR(11), 10)
        self.state.set_vgpr(VGPR(12), 0xDEADBEEF)

        # 10 >= 10 -> resets to 0
        inst = DSIncU32Instruction.parse("ds_inc_rtn_u32 v12, v10, v11")
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 10)
        self.assertEqual(self.state.get_vgpr(VGPR(12)), 10)
        self.assertEqual(self.lds.read_dword(0x20), 0)

    def test_scalar_execution_with_offset(self) -> None:
        self.lds.write_dword(0x34, 0)
        self.state.set_vgpr(VGPR(1), 0x30)
        self.state.set_vgpr(VGPR(2), 5)

        inst = DSIncU32Instruction.parse("ds_inc_rtn_u32 v0, v1, v2 offset:4")
        res = self.interp.execute(self.state, inst)

        self.assertEqual(res, 0)
        self.assertEqual(self.state.get_vgpr(VGPR(0)), 0)
        self.assertEqual(self.lds.read_dword(0x34), 1)

class SIMDWaveState:
    """Architectural state for a SIMD wavefront (Wave32 or Wave64)."""

    def __init__(self, lds: LDSMemory, wave_size: int = 32) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Wave size must be 32 or 64, got {wave_size}")
        self.wave_size = wave_size
        self.lds = lds
        self.exec_mask = (1 << wave_size) - 1
        self.lanes = [ScalarThreadState(lds) for _ in range(wave_size)]

    def is_lane_active(self, lane_id: int) -> bool:
        return bool((self.exec_mask >> lane_id) & 1)


class SIMDDSInterpreter:
    """Multi-lane SIMD interpreter modeling divergence and atomic address clashes."""

    def __init__(self, lds: LDSMemory) -> None:
        self.lds = lds

    def execute_wave(
        self,
        wave: SIMDWaveState,
        inst: DSIncU32Instruction,
        arbitration_order: Optional[List[int]] = None,
    ) -> List[Optional[int]]:
        results: List[Optional[int]] = [None] * wave.wave_size
        active_lanes = [i for i in range(wave.wave_size) if wave.is_lane_active(i)]

        order = [i for i in arbitration_order if i in active_lanes] if arbitration_order is not None else active_lanes

        for lane_id in order:
            lane_state = wave.lanes[lane_id]
            addr = lane_state.get_vgpr(inst.vaddr)
            data = lane_state.get_vgpr(inst.vdata0)

            old_val, new_val = self.lds.atomic_inc_u32(addr, data, offset=inst.offset, rtn=inst.is_return)

            if inst.is_return and inst.vdst is not None:
                assert old_val is not None
                lane_state.set_vgpr(inst.vdst, old_val)
                results[lane_id] = old_val

        return results


class TestDSIncU32SIMDDivergenceAndExecMask(unittest.TestCase):
    """Part 4: SIMD multi-lane divergence and 32/64-bit EXEC masking."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_bytes=4096)
        self.interp = SIMDDSInterpreter(self.lds)

    def test_divergent_addresses_wave32(self) -> None:
        wave = SIMDWaveState(self.lds, wave_size=32)
        inst = DSIncU32Instruction.parse("ds_inc_rtn_u32 v5, v1, v2")

        for i in range(32):
            self.lds.write_dword(i * 4, i)
            wave.lanes[i].set_vgpr(VGPR(1), i * 4)
            wave.lanes[i].set_vgpr(VGPR(2), 20)  # limit 20

        results = self.interp.execute_wave(wave, inst)

        for i in range(32):
            expected_old = i
            expected_new = 0 if i >= 20 else (i + 1)
            self.assertEqual(results[i], expected_old)
            self.assertEqual(wave.lanes[i].get_vgpr(VGPR(5)), expected_old)
            self.assertEqual(self.lds.read_dword(i * 4), expected_new)

    def test_exec_masking_wave64(self) -> None:
        lds = LDSMemory(size_bytes=8192)
        interp = SIMDDSInterpreter(lds)
        wave = SIMDWaveState(lds, wave_size=64)
        mask = 0
        for i in range(0, 64, 2):
            mask |= (1 << i)
        wave.exec_mask = mask

        inst = DSIncU32Instruction.parse("ds_inc_rtn_u32 v3, v1, v2")
        for i in range(64):
            lds.write_dword(i * 4, 0)
            wave.lanes[i].set_vgpr(VGPR(1), i * 4)
            wave.lanes[i].set_vgpr(VGPR(2), 10)
            wave.lanes[i].set_vgpr(VGPR(3), 0xCAFE)

        results = interp.execute_wave(wave, inst)

        for i in range(64):
            if i % 2 == 0:
                self.assertEqual(results[i], 0)
                self.assertEqual(wave.lanes[i].get_vgpr(VGPR(3)), 0)
                self.assertEqual(lds.read_dword(i * 4), 1)
            else:
                self.assertIsNone(results[i])
                self.assertEqual(wave.lanes[i].get_vgpr(VGPR(3)), 0xCAFE)
                self.assertEqual(lds.read_dword(i * 4), 0)

    def test_address_clash_consecutive_increments(self) -> None:
        # All 32 lanes increment same counter with threshold 50.
        # Starting from 0, 32 increments yield 32.
        inst = DSIncU32Instruction.parse("ds_inc_u32 v1, v2")

        lds = LDSMemory(size_bytes=1024)
        interp = SIMDDSInterpreter(lds)
        wave = SIMDWaveState(lds, wave_size=32)

        lds.write_dword(0x40, 0)
        for i in range(32):
            wave.lanes[i].set_vgpr(VGPR(1), 0x40)
            wave.lanes[i].set_vgpr(VGPR(2), 50)

        interp.execute_wave(wave, inst)
        self.assertEqual(lds.read_dword(0x40), 32)

class TestDSIncU32RegisterAliasing(unittest.TestCase):
    """Part 5: Register aliasing and in-place operand preservation."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_bytes=4096)
        self.interp = ScalarDSInterpreter(self.lds)

    def test_aliasing_vdst_equals_vaddr(self) -> None:
        state = ScalarThreadState(self.lds)
        self.lds.write_dword(0x50, 7)
        state.set_vgpr(VGPR(1), 0x50)
        state.set_vgpr(VGPR(2), 20)

        inst = DSIncU32Instruction.parse("ds_inc_rtn_u32 v1, v1, v2")
        res = self.interp.execute(state, inst)

        self.assertEqual(res, 7)
        self.assertEqual(state.get_vgpr(VGPR(1)), 7)
        self.assertEqual(self.lds.read_dword(0x50), 8)

    def test_aliasing_vdst_equals_vdata0(self) -> None:
        state = ScalarThreadState(self.lds)
        self.lds.write_dword(0x60, 10)
        state.set_vgpr(VGPR(1), 0x60)
        state.set_vgpr(VGPR(2), 10)

        inst = DSIncU32Instruction.parse("ds_inc_rtn_u32 v2, v1, v2")
        res = self.interp.execute(state, inst)

        self.assertEqual(res, 10)
        self.assertEqual(state.get_vgpr(VGPR(2)), 10)
        self.assertEqual(self.lds.read_dword(0x60), 0)

    def test_aliasing_vaddr_equals_vdata0(self) -> None:
        state = ScalarThreadState(self.lds)
        self.lds.write_dword(0x40, 0)
        state.set_vgpr(VGPR(1), 0x40)

        inst = DSIncU32Instruction.parse("ds_inc_u32 v1, v1")
        self.interp.execute(state, inst)

        self.assertEqual(state.get_vgpr(VGPR(1)), 0x40)
        self.assertEqual(self.lds.read_dword(0x40), 1)

class TestDSIncU32ConditionCodesAndStatusFlags(unittest.TestCase):
    """Part 6: Condition codes, status flags, and lgkmcnt tracking."""

    def test_status_flags_invariance(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        state = ScalarThreadState(lds)
        state.sgprs[0] = 0xABCD1234
        lds.write_dword(0x10, 5)
        state.set_vgpr(VGPR(1), 0x10)
        state.set_vgpr(VGPR(2), 10)

        interp = ScalarDSInterpreter(lds)
        inst = DSIncU32Instruction.parse("ds_inc_rtn_u32 v5, v1, v2")
        interp.execute(state, inst)

        self.assertEqual(state.sgprs[0], 0xABCD1234)
        self.assertEqual(state.lgkm_cnt, 1)

    def test_lgkmcnt_counter_tracking(self) -> None:
        lds = LDSMemory(size_bytes=4096)
        state = ScalarThreadState(lds)
        interp = ScalarDSInterpreter(lds)

        for i in range(4):
            inst = DSIncU32Instruction.parse(f"ds_inc_u32 v1, v2 offset:{i*4}")
            interp.execute(state, inst)

        self.assertEqual(state.lgkm_cnt, 4)

class TestDSIncU32BoundaryExtremes(unittest.TestCase):
    """Part 7: Boundary extremes, wrap reset points, and address wrap."""

    def setUp(self) -> None:
        self.lds = LDSMemory(size_bytes=4096)
        self.interp = ScalarDSInterpreter(self.lds)
        self.state = ScalarThreadState(self.lds)

    def test_boundary_increments_and_resets(self) -> None:
        # DATA = 0: any non-negative unsigned integer >= 0 is True -> always resets to 0
        self.assertEqual(ref_ds_inc_u32(0, 0), 0)
        self.assertEqual(ref_ds_inc_u32(1, 0), 0)
        self.assertEqual(ref_ds_inc_u32(0xFFFFFFFF, 0), 0)

        # DATA = UINT32_MAX: cur < UINT32_MAX increments
        self.assertEqual(ref_ds_inc_u32(0, 0xFFFFFFFF), 1)
        self.assertEqual(ref_ds_inc_u32(0xFFFFFFFE, 0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(ref_ds_inc_u32(0xFFFFFFFF, 0xFFFFFFFF), 0)

        # 0x80000000 in unsigned is 2147483648
        self.assertEqual(ref_ds_inc_u32(0x7FFFFFFF, 0x80000000), 0x80000000)
        self.assertEqual(ref_ds_inc_u32(0x80000000, 0x80000000), 0)

    def test_memory_address_wrapping_boundaries(self) -> None:
        idx = self.lds.resolve_dword_index(0xFFFFFFFC, offset=4)
        self.assertEqual(idx, 0)
        idx2 = self.lds.resolve_dword_index(4096)
        self.assertEqual(idx2, 0)

class TestDSIncU32NegativeAndTranslator(unittest.TestCase):
    """Part 8: Negative controls, syntax rejection, mutation detection, and translator execution."""

    def test_syntax_rejection(self) -> None:
        invalid_assemblies = [
            "ds_inc_u32",
            "ds_inc_u32 v1",
            "ds_inc_u32 v1, v2, v3",
            "ds_inc_rtn_u32 v1, v2",
            "ds_inc_u32 s1, v2",
            "ds_inc_rtn_u32 s0, v1, v2",
            "ds_inc_u32 v1, v2 offset:-1",
            "ds_inc_u32 v1, v2 offset:70000",
            "ds_inc_u32 v1, v2 bad_mod",
        ]
        for asm in invalid_assemblies:
            with self.subTest(asm=asm):
                with self.assertRaises(ValueError):
                    DSIncU32Instruction.parse(asm)

    def test_mutation_detection_signed_comparison(self) -> None:
        # Mutant: treats operands as signed integers instead of unsigned
        def mutant_inc_signed(cur: int, data: int) -> int:
            s_cur = cur - (1 << 32) if (cur & 0x80000000) else cur
            s_data = data - (1 << 32) if (data & 0x80000000) else data
            return 0 if s_cur >= s_data else (cur + 1) & MASK32

        # cur = 0xFFFFFFFF (-1 signed), data = 10 (+10 signed)
        # Unsigned: cur (4294967295) >= data (10) -> resets to 0.
        # Signed mutant: s_cur (-1) < s_data (10) -> increments to 0!
        # Let's test with cur = 0x80000000 (-2147483648 signed), data = 10:
        # Unsigned: 2147483648 >= 10 -> resets to 0.
        # Signed: -2147483648 < 10 -> increments to 0x80000001!
        oracle_res = ref_ds_inc_u32(0x80000000, 10)
        mutant_res = mutant_inc_signed(0x80000000, 10)
        self.assertEqual(oracle_res, 0)
        self.assertEqual(mutant_res, 0x80000001)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_mutation_detection_strict_greater_instead_of_ge(self) -> None:
        def mutant_strict_gt(cur: int, data: int) -> int:
            return 0 if cur > data else (cur + 1) & MASK32

        oracle_res = ref_ds_inc_u32(10, 10)
        mutant_res = mutant_strict_gt(10, 10)
        self.assertEqual(oracle_res, 0)
        self.assertEqual(mutant_res, 11)
        self.assertNotEqual(oracle_res, mutant_res)

    def test_translator_emits_cas_loop_with_ugreater_than_equal(self) -> None:
        """Verifies tools/rdna2spirv.py emits OpAtomicCompareExchange and OpUGreaterThanEqual."""
        lines = [
            "v_mov_b32_e32 v1, 0x40",
            "v_mov_b32_e32 v2, 10",
            "ds_inc_u32 v1, v2",
            "ds_inc_rtn_u32 v5, v1, v2 offset:16",
            "s_waitcnt lgkmcnt(0)",
        ]
        spv = L.translate(lines)
        self.assertIn("OpAtomicCompareExchange", spv)
        self.assertIn("OpUGreaterThanEqual", spv)

    def test_translator_backed_execution_via_lds_run(self) -> None:
        """Executes translated ds_inc_u32 and ds_inc_rtn_u32 in spirv_mini_interp."""
        # Test 1: Increment when cur < data (cur=3, data=10 -> 4)
        init = {0x40: 3}
        ops = [("ds_inc_rtn_u32", 0x40, 0, (10,))]
        rtns, fin = L.run_chain(ops, init)
        self.assertEqual(rtns, [3])
        self.assertEqual(fin, {0x40: 4})

        # Test 2: Reset to 0 when cur >= data (cur=10, data=10 -> 0)
        init2 = {0x44: 10}
        ops2 = [("ds_inc_rtn_u32", 0x44, 0, (10,))]
        rtns2, fin2 = L.run_chain(ops2, init2)
        self.assertEqual(rtns2, [10])
        self.assertEqual(fin2, {0x44: 0})

        # Test 3: Non-returning ds_inc_u32 with offset
        init3 = {0x48: 5, 0x4C: 20}
        ops3 = [
            ("ds_inc_u32", 0x48, 4, (20,)),       # 0x48 + 4 = 0x4C: cur=20, data=20 -> resets to 0
            ("ds_inc_rtn_u32", 0x48, 0, (10,)),   # 0x48: cur=5, data=10 -> 6
        ]
        rtns3, fin3 = L.run_chain(ops3, init3)
        self.assertEqual(rtns3, [0xDEAD, 5])
        self.assertEqual(fin3, {0x48: 6, 0x4C: 0})


if __name__ == "__main__":
    unittest.main()
