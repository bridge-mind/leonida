from __future__ import annotations

"""Semantic interpreter tests for scalar memory write / store operations (s_store_dword).

Covers AMD RDNA2 / GFX10 SMEM (Scalar Memory) store instruction semantics:
  * s_store_dword: writes 1 dword (32 bits / 4 bytes) from SGPR into memory
  * s_store_dwordx2: writes 2 dwords (64 bits / 8 bytes) from SGPR pair into memory
  * s_store_dwordx4: writes 4 dwords (128 bits / 16 bytes) from SGPR quad into memory
  * s_buffer_store_dword, s_buffer_store_dwordx2, s_buffer_store_dwordx4:
      writes 1, 2, or 4 dwords into buffer memory described by a 128-bit Buffer Resource Descriptor (V#)
  * 64-bit base pointer reconstruction from SGPR pair s[p:p+1]:
      base_address = (s[p+1] << 32) | s[p]
  * Buffer Resource Descriptor (V#) 128-bit layout and base address extraction:
      - Dword 0: Base address low 32 bits [31:0]
      - Dword 1: Base address high 16 bits [47:32] in bits [15:0]; bits [31:16] are format/stride/flags
      - Proper masking (dword1 & 0xFFFF) preventing descriptor flags from polluting the 64-bit pointer
      - Dword 2: num_records defining buffer size bounds
  * Address calculations & arithmetic edge cases:
      - Zero offset, immediate byte offsets, and dynamic SGPR offsets
      - Combined register offset + immediate offset ("sOFF offset:IMM"): address = base + sOFF + IMM
      - 32-bit carry propagation across the 4GB boundary into upper 32 bits
      - 64-bit address wrapping modulo 2^64
      - 4-byte dword alignment enforcement (hardware drops low 2 bits)
  * Multi-dword sequential strides:
      - Writes sequential dwords at 4-byte offsets (addr + 0, addr + 4, addr + 8, addr + 12)
      - Non-interference with neighboring memory locations
      - Memory overwrites and latest-write persistence
  * Destination / Source register targeting:
      - Source data from SGPRs, SGPR ranges, and special registers (vcc_lo, vcc_hi, vcc pair, m0, null)
  * Shader IR parsing (tools/shader_ir.py):
      - Tokenization and AST validation for scalar store instructions and operand variants
      - Validation of invalid operand configurations (wrong register counts, vgpr base)
  * Translator and mini-interpreter contracts:
      - rdna2spirv.Gen dispatch validation (raises Unsupported until translator implements scalar store)
      - spirv_mini_interp strictness validation (rejection of unmodelled storage classes)
"""

import os
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

import rdna2spirv  # noqa: E402
from shader_ir import parse_line  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


def compute_smem_store_pointer_address(base_lo: int, base_hi: int, offset: int, imm: int = 0) -> int:
    """Compute 64-bit effective byte address for s_store_dword from 64-bit pointer and offsets.

    Per AMD RDNA2 ISA SMEM addressing:
      address = ((base_64 + offset + imm) & 0xFFFFFFFFFFFFFFFF) & ~3
    The low 2 bits are truncated to enforce 4-byte dword alignment.
    """
    base_64 = ((base_hi & MASK32) << 32) | (base_lo & MASK32)
    eff = (base_64 + (offset & MASK32) + (imm & MASK32)) & MASK64
    return eff & ~3


def compute_smem_store_descriptor_address(desc_dw0: int, desc_dw1: int, offset: int, imm: int = 0) -> int:
    """Compute 64-bit effective byte address for s_buffer_store_dword from buffer descriptor V# and offsets.

    Per AMD RDNA2 ISA:
      - dword0: Base address [31:0]
      - dword1: Base address [47:32] in bits [15:0]. Bits [31:16] are format/stride metadata.
    """
    base_hi = desc_dw1 & 0xFFFF
    base_64 = (base_hi << 32) | (desc_dw0 & MASK32)
    eff = (base_64 + (offset & MASK32) + (imm & MASK32)) & MASK64
    return eff & ~3


def check_buffer_store_bounds(offset: int, imm: int, count: int, num_records: int) -> bool:
    """Check if buffer store range is within descriptor num_records.

    Buffer store is dropped (no-op) if (offset + imm + count * 4) exceeds num_records.
    """
    total_offset = (offset & MASK32) + (imm & MASK32)
    store_bytes = count * 4
    return (total_offset + store_bytes) <= (num_records & MASK32)


class SmemStoreMemoryBank:
    """Deterministic simulated memory bank for SMEM store validation."""

    def __init__(self) -> None:
        self._mem: dict[int, int] = {}

    def write_dword(self, byte_addr: int, val: int) -> None:
        aligned_addr = (byte_addr & MASK64) & ~3
        self._mem[aligned_addr] = val & MASK32

    def read_dword(self, byte_addr: int) -> int:
        aligned_addr = (byte_addr & MASK64) & ~3
        return self._mem.get(aligned_addr, 0)

    def write_dwords(self, byte_addr: int, values: list[int]) -> None:
        aligned_addr = (byte_addr & MASK64) & ~3
        for i, val in enumerate(values):
            self._mem[aligned_addr + i * 4] = val & MASK32

    def read_dwords(self, byte_addr: int, count: int) -> list[int]:
        aligned_addr = (byte_addr & MASK64) & ~3
        return [self._mem.get(aligned_addr + i * 4, 0) for i in range(count)]

    def has_address(self, byte_addr: int) -> bool:
        aligned_addr = (byte_addr & MASK64) & ~3
        return aligned_addr in self._mem

    def dump(self) -> dict[int, int]:
        return dict(self._mem)


class SmemStoreReferenceInterpreter:
    """Reference execution engine for RDNA2 scalar memory stores."""

    def __init__(self, memory: SmemStoreMemoryBank) -> None:
        self.mem = memory
        self.sgpr: list[int] = [0] * 106
        self.vcc_lo: int = 0
        self.vcc_hi: int = 0
        self.m0: int = 0

    def load_register(self, name: str, reg_idx: int = 0) -> int:
        if name == "sgpr":
            return self.sgpr[reg_idx] & MASK32
        if name == "vcc_lo":
            return self.vcc_lo & MASK32
        if name == "vcc_hi":
            return self.vcc_hi & MASK32
        if name == "m0":
            return self.m0 & MASK32
        if name == "null":
            return 0
        raise ValueError(f"unknown register {name}")

    def store_register(self, name: str, reg_idx: int, val: int) -> None:
        val &= MASK32
        if name == "sgpr":
            self.sgpr[reg_idx] = val
        elif name == "vcc_lo":
            self.vcc_lo = val
        elif name == "vcc_hi":
            self.vcc_hi = val
        elif name == "m0":
            self.m0 = val
        else:
            raise ValueError(f"unknown register {name}")

    def get_source_dwords(self, src_reg: str, src_idx: int, count: int) -> list[int]:
        words = []
        for i in range(count):
            if src_reg == "sgpr":
                words.append(self.load_register("sgpr", src_idx + i))
            elif src_reg == "vcc":
                if i == 0:
                    words.append(self.load_register("vcc_lo", 0))
                elif i == 1:
                    words.append(self.load_register("vcc_hi", 0))
                else:
                    words.append(0)
            elif src_reg in ("vcc_lo", "vcc_hi", "m0", "null"):
                if i == 0:
                    words.append(self.load_register(src_reg, 0))
                else:
                    words.append(0)
            else:
                raise ValueError(f"unsupported source register {src_reg}")
        return words

    def execute_s_store(self, count: int, src_reg: str, src_idx: int,
                        base_lo_idx: int, offset: int, imm_offset: int = 0) -> int:
        lo = self.load_register("sgpr", base_lo_idx)
        hi = self.load_register("sgpr", base_lo_idx + 1)
        addr = compute_smem_store_pointer_address(lo, hi, offset, imm_offset)
        values = self.get_source_dwords(src_reg, src_idx, count)
        for i, val in enumerate(values):
            self.mem.write_dword(addr + i * 4, val)
        return addr

    def execute_s_buffer_store(self, count: int, src_reg: str, src_idx: int,
                               desc_dw0_idx: int, offset: int, imm_offset: int = 0) -> tuple[int, bool]:
        dw0 = self.load_register("sgpr", desc_dw0_idx)
        dw1 = self.load_register("sgpr", desc_dw0_idx + 1)
        num_records = self.load_register("sgpr", desc_dw0_idx + 2)
        addr = compute_smem_store_descriptor_address(dw0, dw1, offset, imm_offset)
        in_bounds = check_buffer_store_bounds(offset, imm_offset, count, num_records)
        if in_bounds:
            values = self.get_source_dwords(src_reg, src_idx, count)
            for i, val in enumerate(values):
                self.mem.write_dword(addr + i * 4, val)
        return addr, in_bounds


class SmemStoreAddressOracleTests(unittest.TestCase):
    """Verifies SMEM 64-bit store effective address calculations and V# descriptor masking."""

    def test_s_store_pointer_address_basic(self):
        cases = [
            (0x1000, 0x0, 0, 0, 0x1000),
            (0x1000, 0x0, 16, 0, 0x1010),
            (0x1000, 0x0, 0x40, 0, 0x1040),
            (0x2000, 0x1, 0x100, 0, 0x100002100),
            (0x00000000, 0x00000008, 0x200, 0, 0x800000200),
        ]
        for lo, hi, off, imm, expected in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), off=hex(off), imm=hex(imm)):
                self.assertEqual(compute_smem_store_pointer_address(lo, hi, off, imm), expected)

    def test_s_store_pointer_carry_into_upper_32_bits(self):
        cases = [
            (0xFFFFFFFC, 0x00000001, 8, 0, 0x0000000200000004),
            (0xFFFFFFFF, 0x00000000, 1, 0, 0x0000000100000000),
            (0xFFFFFFF0, 0x00000005, 0x20, 0, 0x0000000600000010),
            (0xFFFFFFFF, 0x0000000F, 0xFFFFFFFF, 0, 0x00000010FFFFFFFC),
        ]
        for lo, hi, off, imm, expected in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), off=hex(off)):
                self.assertEqual(compute_smem_store_pointer_address(lo, hi, off, imm), expected)

    def test_s_store_pointer_wrap_modulo_64(self):
        self.assertEqual(compute_smem_store_pointer_address(0xFFFFFFF0, 0xFFFFFFFF, 0x20, 0), 0x10)
        self.assertEqual(compute_smem_store_pointer_address(0xFFFFFFFF, 0xFFFFFFFF, 1, 0), 0)

    def test_s_store_combined_register_and_imm_offset(self):
        # address = base + soffset + imm
        base_lo, base_hi = 0x00004000, 0x00000003
        soffset = 0x20
        imm = 0x0C
        expected = 0x000000030000402C
        self.assertEqual(compute_smem_store_pointer_address(base_lo, base_hi, soffset, imm), expected)

    def test_s_store_dword_alignment_truncation(self):
        # Low 2 bits of address are cleared (multiple of 4)
        for offset in (0x1000, 0x1001, 0x1002, 0x1003):
            with self.subTest(offset=hex(offset)):
                self.assertEqual(compute_smem_store_pointer_address(offset, 0, 0, 0), 0x1000)

    def test_s_buffer_store_descriptor_masking(self):
        cases = [
            (0x00001000, 0x00000002, 0x10, 0, 0x0000000200001010),
            # dw1 has 0x8004 in bits [31:16] (format/stride/flags)
            (0x00001000, 0x80040002, 0x10, 0, 0x0000000200001010),
            # dw1 has all upper 16 bits set (0xFFFF0000)
            (0x12345678, 0xFFFF0007, 0x0, 0, 0x0000000712345678),
            # dw1 format bits set, address bits 0
            (0x00000040, 0x3F000000, 0x20, 0, 0x0000000000000060),
            # Carry from dw0 into masked dw1
            (0xFFFFFFF8, 0x80000001, 0x10, 0, 0x0000000200000008),
        ]
        for dw0, dw1, off, imm, expected in cases:
            with self.subTest(dw0=hex(dw0), dw1=hex(dw1), off=hex(off)):
                self.assertEqual(compute_smem_store_descriptor_address(dw0, dw1, off, imm), expected)

    def test_buffer_store_bounds_calculation(self):
        num_records = 0x1000
        # In bounds: offset 0x0, 1 dword (4 bytes) -> 4 <= 0x1000
        self.assertTrue(check_buffer_store_bounds(0, 0, 1, num_records))
        # Exact boundary: offset 0xFFC, 1 dword (4 bytes) -> 0x1000 <= 0x1000
        self.assertTrue(check_buffer_store_bounds(0xFFC, 0, 1, num_records))
        # Out of bounds: offset 0xFFD, 1 dword (4 bytes) -> 0x1001 > 0x1000
        self.assertFalse(check_buffer_store_bounds(0xFFD, 0, 1, num_records))
        # Exact boundary with 4 dwords (16 bytes): offset 0xFF0 -> 0xFF0 + 16 = 0x1000
        self.assertTrue(check_buffer_store_bounds(0xFF0, 0, 4, num_records))
        # 1 byte over with 4 dwords: offset 0xFF4 -> 0xFF4 + 16 = 0x1004 > 0x1000
        self.assertFalse(check_buffer_store_bounds(0xFF4, 0, 4, num_records))
        # Combined soffset + imm exceeding bounds
        self.assertFalse(check_buffer_store_bounds(0xF00, 0x110, 1, num_records))


class SmemStoreExecutionTests(unittest.TestCase):
    """Verifies single and multi-dword scalar store execution, strides, overwriting, and isolation."""

    def setUp(self) -> None:
        self.bank = SmemStoreMemoryBank()
        self.interp = SmemStoreReferenceInterpreter(self.bank)

    def test_s_store_dword_single(self):
        # Base s[2:3] = 0x1_00000000, offset = 0x20 -> address 0x1_00000020
        self.interp.store_register("sgpr", 2, 0x00000000)
        self.interp.store_register("sgpr", 3, 0x00000001)
        self.interp.store_register("sgpr", 4, 0xCAFEBABE)

        addr = self.interp.execute_s_store(1, "sgpr", 4, 2, 0x20)
        self.assertEqual(addr, 0x100000020)
        self.assertEqual(self.bank.read_dword(0x100000020), 0xCAFEBABE)

    def test_s_store_dwordx2_sequential_stride(self):
        self.interp.store_register("sgpr", 6, 0x00001000)
        self.interp.store_register("sgpr", 7, 0x00000000)
        self.interp.store_register("sgpr", 8, 0x11112222)
        self.interp.store_register("sgpr", 9, 0x33334444)

        addr = self.interp.execute_s_store(2, "sgpr", 8, 6, 0)
        self.assertEqual(addr, 0x1000)
        self.assertEqual(self.bank.read_dword(0x1000), 0x11112222)
        self.assertEqual(self.bank.read_dword(0x1004), 0x33334444)

    def test_s_store_dwordx4_sequential_stride(self):
        self.interp.store_register("sgpr", 0, 0x00002000)
        self.interp.store_register("sgpr", 1, 0x00000000)
        test_words = [0xA0000001, 0xA0000002, 0xA0000003, 0xA0000004]
        for i, val in enumerate(test_words):
            self.interp.store_register("sgpr", 12 + i, val)

        addr = self.interp.execute_s_store(4, "sgpr", 12, 0, 0x10)
        self.assertEqual(addr, 0x2010)
        for i, val in enumerate(test_words):
            self.assertEqual(self.bank.read_dword(0x2010 + i * 4), val)

    def test_s_store_overwriting_and_persistence(self):
        self.interp.store_register("sgpr", 0, 0x1000)
        self.interp.store_register("sgpr", 1, 0)
        self.interp.store_register("sgpr", 4, 0x11111111)

        # First store
        self.interp.execute_s_store(1, "sgpr", 4, 0, 0)
        self.assertEqual(self.bank.read_dword(0x1000), 0x11111111)

        # Overwrite with second store
        self.interp.store_register("sgpr", 4, 0x22222222)
        self.interp.execute_s_store(1, "sgpr", 4, 0, 0)
        self.assertEqual(self.bank.read_dword(0x1000), 0x22222222)

    def test_s_store_memory_isolation_and_adjacent_untouched(self):
        # Pre-seed adjacent memory locations
        self.bank.write_dword(0x2FFC, 0xAAAAAAAA)
        self.bank.write_dword(0x3010, 0xBBBBBBBB)

        self.interp.store_register("sgpr", 0, 0x3000)
        self.interp.store_register("sgpr", 1, 0)
        self.interp.store_register("sgpr", 4, 0xC001D00D)
        self.interp.store_register("sgpr", 5, 0xF00DCAFE)

        # Store 2 dwords to [0x3000, 0x3008)
        self.interp.execute_s_store(2, "sgpr", 4, 0, 0)

        # Target locations written
        self.assertEqual(self.bank.read_dword(0x3000), 0xC001D00D)
        self.assertEqual(self.bank.read_dword(0x3004), 0xF00DCAFE)
        # Adjacent locations remain uncorrupted
        self.assertEqual(self.bank.read_dword(0x2FFC), 0xAAAAAAAA)
        self.assertEqual(self.bank.read_dword(0x3010), 0xBBBBBBBB)

    def test_s_store_special_source_registers(self):
        self.interp.store_register("sgpr", 0, 0x1000)
        self.interp.store_register("sgpr", 1, 0)

        # Single dword from vcc_lo
        self.interp.store_register("vcc_lo", 0, 0x12345678)
        self.interp.execute_s_store(1, "vcc_lo", 0, 0, 0)
        self.assertEqual(self.bank.read_dword(0x1000), 0x12345678)

        # Single dword from vcc_hi
        self.interp.store_register("vcc_hi", 0, 0x9ABCDEF0)
        self.interp.execute_s_store(1, "vcc_hi", 0, 0, 4)
        self.assertEqual(self.bank.read_dword(0x1004), 0x9ABCDEF0)

        # Single dword from m0
        self.interp.store_register("m0", 0, 0xFEEDFACE)
        self.interp.execute_s_store(1, "m0", 0, 0, 8)
        self.assertEqual(self.bank.read_dword(0x1008), 0xFEEDFACE)

        # Single dword from null (writes 0)
        self.bank.write_dword(0x100C, 0xFFFFFFFF)
        self.interp.execute_s_store(1, "null", 0, 0, 12)
        self.assertEqual(self.bank.read_dword(0x100C), 0)

        # Two dwords from vcc pair (vcc_lo and vcc_hi)
        self.interp.store_register("vcc_lo", 0, 0xAAAA5555)
        self.interp.store_register("vcc_hi", 0, 0x5555AAAA)
        self.interp.execute_s_store(2, "vcc", 0, 0, 0x20)
        self.assertEqual(self.bank.read_dword(0x1020), 0xAAAA5555)
        self.assertEqual(self.bank.read_dword(0x1024), 0x5555AAAA)

    def test_s_buffer_store_in_bounds(self):
        # V# descriptor: dw0 = 0x5000, dw1 = 0x80040003 (stride=0x8004, hi_addr=3)
        self.interp.store_register("sgpr", 8, 0x00005000)
        self.interp.store_register("sgpr", 9, 0x80040003)
        self.interp.store_register("sgpr", 10, 0x00010000)  # num_records = 64KB
        self.interp.store_register("sgpr", 11, 0x00140000)  # flags

        self.interp.store_register("sgpr", 20, 0x99887766)
        expected_addr = 0x0000000300005040

        addr, written = self.interp.execute_s_buffer_store(1, "sgpr", 20, 8, 0x40)
        self.assertTrue(written)
        self.assertEqual(addr, expected_addr)
        self.assertEqual(self.bank.read_dword(expected_addr), 0x99887766)

    def test_s_buffer_store_out_of_bounds_dropped(self):
        # V# descriptor with small buffer: num_records = 0x40 (64 bytes)
        self.interp.store_register("sgpr", 8, 0x00005000)
        self.interp.store_register("sgpr", 9, 0x00000000)
        self.interp.store_register("sgpr", 10, 0x00000040)
        self.interp.store_register("sgpr", 11, 0x00000000)

        self.interp.store_register("sgpr", 20, 0xBADF00D)
        # Offset 0x40 is at boundary (offset 0x40 + 4 bytes = 0x44 > 0x40)
        addr, written = self.interp.execute_s_buffer_store(1, "sgpr", 20, 8, 0x40)
        self.assertFalse(written)
        self.assertFalse(self.bank.has_address(addr))

    def test_s_buffer_store_multi_dword_partial_out_of_bounds_dropped(self):
        # V# descriptor with num_records = 0x10 (16 bytes)
        self.interp.store_register("sgpr", 8, 0x00006000)
        self.interp.store_register("sgpr", 9, 0x00000000)
        self.interp.store_register("sgpr", 10, 0x00000010)
        self.interp.store_register("sgpr", 11, 0x00000000)

        for i in range(4):
            self.interp.store_register("sgpr", 12 + i, 0x1000 + i)

        # 4 dwords (16 bytes) starting at offset 4 -> 4 + 16 = 20 > 16 -> dropped
        addr, written = self.interp.execute_s_buffer_store(4, "sgpr", 12, 8, 4)
        self.assertFalse(written)
        for i in range(4):
            self.assertFalse(self.bank.has_address(addr + i * 4))


class SmemStoreAstParsingTests(unittest.TestCase):
    """Verifies shader_ir.parse_line correctly parses scalar memory store instructions."""

    def test_parse_s_store_dword_immediate_offset(self):
        ins = parse_line("s_store_dword s0, s[2:3], 0x10")
        self.assertEqual(ins["op"], "s_store_dword")
        self.assertEqual(ins["args"][0], {"k": "sgpr", "r": 0, "n": 1, "neg": False, "abs": False})
        self.assertEqual(ins["args"][1], {"k": "sgpr", "r": 2, "n": 2, "neg": False, "abs": False})
        self.assertEqual(ins["args"][2], {"k": "imm", "v": 16})
        self.assertEqual(ins["mods"], [])

    def test_parse_s_store_dwordx2_immediate_offset(self):
        ins = parse_line("s_store_dwordx2 s[0:1], s[4:5], 0x20")
        self.assertEqual(ins["op"], "s_store_dwordx2")
        self.assertEqual(ins["args"][0], {"k": "sgpr", "r": 0, "n": 2, "neg": False, "abs": False})
        self.assertEqual(ins["args"][1], {"k": "sgpr", "r": 4, "n": 2, "neg": False, "abs": False})
        self.assertEqual(ins["args"][2], {"k": "imm", "v": 32})
        self.assertEqual(ins["mods"], [])

    def test_parse_s_store_dwordx4_immediate_offset(self):
        ins = parse_line("s_store_dwordx4 s[4:7], s[8:9], 0x40")
        self.assertEqual(ins["op"], "s_store_dwordx4")
        self.assertEqual(ins["args"][0], {"k": "sgpr", "r": 4, "n": 4, "neg": False, "abs": False})
        self.assertEqual(ins["args"][1], {"k": "sgpr", "r": 8, "n": 2, "neg": False, "abs": False})
        self.assertEqual(ins["args"][2], {"k": "imm", "v": 64})
        self.assertEqual(ins["mods"], [])

    def test_parse_s_store_dword_register_offset(self):
        ins = parse_line("s_store_dword s0, s[2:3], s4")
        self.assertEqual(ins["op"], "s_store_dword")
        self.assertEqual(ins["args"][2], {"k": "sgpr", "r": 4, "n": 1, "neg": False, "abs": False})
        self.assertEqual(ins["mods"], [])

    def test_parse_s_store_dword_register_offset_with_imm_mod(self):
        ins = parse_line("s_store_dword s0, s[2:3], s4 offset:0x10")
        self.assertEqual(ins["op"], "s_store_dword")
        self.assertEqual(ins["args"][2], {"k": "sgpr", "r": 4, "n": 1, "neg": False, "abs": False})
        self.assertEqual(ins["mods"], ["offset:0x10"])

    def test_parse_s_buffer_store_instructions(self):
        cases = [
            ("s_buffer_store_dword s0, s[0:3], 0", "s_buffer_store_dword", 1, 4),
            ("s_buffer_store_dwordx2 s[0:1], s[4:7], 0x10", "s_buffer_store_dwordx2", 2, 4),
            ("s_buffer_store_dwordx4 s[0:3], s[8:11], s12 offset:0x20", "s_buffer_store_dwordx4", 4, 4),
        ]
        for line, op, src_n, base_n in cases:
            with self.subTest(line=line):
                ins = parse_line(line)
                self.assertEqual(ins["op"], op)
                self.assertEqual(ins["args"][0]["n"], src_n)
                self.assertEqual(ins["args"][1]["n"], base_n)

    def test_parse_special_source_registers(self):
        specials = [
            ("s_store_dword vcc_lo, s[0:1], 0", "vcc_lo"),
            ("s_store_dword vcc_hi, s[0:1], 4", "vcc_hi"),
            ("s_store_dword m0, s[0:1], 8", "m0"),
            ("s_store_dword null, s[0:1], 12", "null"),
            ("s_store_dwordx2 vcc, s[0:1], 0", "vcc"),
        ]
        for line, reg_name in specials:
            with self.subTest(line=line):
                ins = parse_line(line)
                self.assertEqual(ins["args"][0]["k"], "special")
                self.assertEqual(ins["args"][0]["t"], reg_name)


class SmemStoreValidationTests(unittest.TestCase):
    """Verifies semantic validation rules for scalar store operands."""

    @staticmethod
    def validate_smem_store_args(op: str, args: list[dict], mods: list[str]) -> None:
        """Validates SMEM store operand structure, mirroring RDNA2 hardware constraints."""
        n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
        src, base = args[0], args[1]
        special_src = (n == 1 and src.get("k") == "special" and src.get("t") in ("vcc_lo", "vcc_hi", "m0", "null", "vcc")) or \
                      (n == 2 and src.get("k") == "special" and src.get("t") == "vcc")

        if src.get("k") != "sgpr" and not special_src:
            raise ValueError(f"invalid source register for {op}: {src}")
        if src.get("k") == "sgpr" and src.get("n", 1) != n:
            raise ValueError(f"expected {n} source SGPRs, got {src.get('n', 1)}")

        if base.get("k") != "sgpr":
            raise ValueError(f"base must be SGPR, got {base.get('k')}")

        if op.startswith("s_buffer_store"):
            if base.get("n") != 4:
                raise ValueError(f"s_buffer_store requires 128-bit V# descriptor (4 SGPRs), got {base.get('n')}")
        else:
            if base.get("n") != 2:
                raise ValueError(f"s_store requires 64-bit pointer (2 SGPRs), got {base.get('n')}")

        # Validate modifiers: only offset:IMM is permitted
        for m in mods:
            if not m.startswith("offset:"):
                raise ValueError(f"unsupported modifier {m}")
            try:
                int(m.split(":", 1)[1], 0)
            except ValueError as e:
                raise ValueError(f"invalid offset modifier value: {m}") from e

    def test_valid_store_args_pass_validation(self):
        cases = [
            "s_store_dword s0, s[2:3], 0",
            "s_store_dwordx2 s[0:1], s[2:3], 0x10",
            "s_store_dwordx4 s[0:3], s[2:3], s4 offset:0x20",
            "s_buffer_store_dword s0, s[0:3], 0",
            "s_buffer_store_dwordx2 s[0:1], s[0:3], 0",
            "s_buffer_store_dwordx4 s[0:3], s[0:3], 0",
            "s_store_dword vcc_lo, s[2:3], 0",
            "s_store_dwordx2 vcc, s[2:3], 0",
        ]
        for line in cases:
            with self.subTest(line=line):
                ins = parse_line(line)
                self.validate_smem_store_args(ins["op"], ins["args"], ins["mods"])

    def test_invalid_source_count_rejected(self):
        # s_store_dwordx2 expects 2 SGPRs, given 1
        ins = parse_line("s_store_dwordx2 s0, s[2:3], 0")
        with self.assertRaises(ValueError) as ctx:
            self.validate_smem_store_args(ins["op"], ins["args"], ins["mods"])
        self.assertIn("expected 2 source SGPRs", str(ctx.exception))

        # s_store_dwordx4 expects 4 SGPRs, given 2
        ins = parse_line("s_store_dwordx4 s[0:1], s[2:3], 0")
        with self.assertRaises(ValueError) as ctx:
            self.validate_smem_store_args(ins["op"], ins["args"], ins["mods"])
        self.assertIn("expected 4 source SGPRs", str(ctx.exception))

    def test_invalid_base_count_rejected(self):
        # s_store_dword requires 2 SGPRs, given 4
        ins = parse_line("s_store_dword s0, s[0:3], 0")
        with self.assertRaises(ValueError) as ctx:
            self.validate_smem_store_args(ins["op"], ins["args"], ins["mods"])
        self.assertIn("s_store requires 64-bit pointer (2 SGPRs)", str(ctx.exception))

        # s_buffer_store requires 4 SGPRs, given 2
        ins = parse_line("s_buffer_store_dword s0, s[0:1], 0")
        with self.assertRaises(ValueError) as ctx:
            self.validate_smem_store_args(ins["op"], ins["args"], ins["mods"])
        self.assertIn("s_buffer_store requires 128-bit V# descriptor", str(ctx.exception))

    def test_vgpr_base_rejected(self):
        ins = parse_line("s_store_dword s0, v[0:1], 0")
        with self.assertRaises(ValueError) as ctx:
            self.validate_smem_store_args(ins["op"], ins["args"], ins["mods"])
        self.assertIn("base must be SGPR", str(ctx.exception))

    def test_invalid_modifier_rejected(self):
        ins = parse_line("s_store_dword s0, s[2:3], s4 offset:badval")
        with self.assertRaises(ValueError) as ctx:
            self.validate_smem_store_args(ins["op"], ins["args"], ins["mods"])
        self.assertIn("invalid offset modifier value", str(ctx.exception))


class SmemStoreEvaluator:
    """Straight-line evaluator executing a program sequence of SMEM stores against a memory bank."""

    def __init__(self, initial_sgpr: dict[int, int], memory: SmemStoreMemoryBank) -> None:
        self.interp = SmemStoreReferenceInterpreter(memory)
        for r, v in initial_sgpr.items():
            self.interp.store_register("sgpr", r, v)
        self.stores: list[tuple[str, int, int]] = []  # (op, address, dword_count)

    def execute_line(self, line: str) -> int:
        ins = parse_line(line)
        op = ins["op"]
        n = 1 if op.endswith("dword") else int(op.rsplit("x", 1)[1])
        src = ins["args"][0]
        base = ins["args"][1]
        off_arg = ins["args"][2]

        offset = off_arg["v"] if off_arg["k"] == "imm" else self.interp.load_register("sgpr", off_arg["r"])
        imm = 0
        for m in ins["mods"]:
            if m.startswith("offset:"):
                imm = int(m.split(":", 1)[1], 0)

        src_reg = src.get("t", "sgpr") if src.get("k") == "special" else "sgpr"
        src_idx = src.get("r", 0)

        if op.startswith("s_buffer_store"):
            addr, written = self.interp.execute_s_buffer_store(n, src_reg, src_idx, base["r"], offset, imm)
            if written:
                self.stores.append((op, addr, n))
            return addr
        else:
            addr = self.interp.execute_s_store(n, src_reg, src_idx, base["r"], offset, imm)
            self.stores.append((op, addr, n))
            return addr


class SmemStoreProgramEvaluatorTests(unittest.TestCase):
    """Verifies multi-instruction sequences of scalar stores and cross-instruction memory integrity."""

    def test_interleaved_store_sequence(self):
        bank = SmemStoreMemoryBank()
        sgpr = {
            0: 0x00004000, 1: 0x00000000,   # Base pointer: 0x4000
            4: 0x11111111,                  # Data 1
            6: 0x22222222, 7: 0x33333333,   # Data 2 (pair)
            8: 0x44444444, 9: 0x55555555,   # Data 4 (quad)
            10: 0x66666666, 11: 0x77777777,
            20: 0x10,                       # Register offset 16
        }
        ev = SmemStoreEvaluator(sgpr, bank)

        # Store 1 dword at 0x4000
        addr1 = ev.execute_line("s_store_dword s4, s[0:1], 0")
        self.assertEqual(addr1, 0x4000)
        self.assertEqual(bank.read_dword(0x4000), 0x11111111)

        # Store 2 dwords at 0x4010 (using register offset s20)
        addr2 = ev.execute_line("s_store_dwordx2 s[6:7], s[0:1], s20")
        self.assertEqual(addr2, 0x4010)
        self.assertEqual(bank.read_dword(0x4010), 0x22222222)
        self.assertEqual(bank.read_dword(0x4014), 0x33333333)

        # Store 4 dwords at 0x4020 (using register offset + immediate offset: 0x10 + 0x10 = 0x20)
        addr3 = ev.execute_line("s_store_dwordx4 s[8:11], s[0:1], s20 offset:0x10")
        self.assertEqual(addr3, 0x4020)
        self.assertEqual(bank.read_dword(0x4020), 0x44444444)
        self.assertEqual(bank.read_dword(0x4024), 0x55555555)
        self.assertEqual(bank.read_dword(0x4028), 0x66666666)
        self.assertEqual(bank.read_dword(0x402C), 0x77777777)

        # Confirm all prior stores remained undisturbed
        self.assertEqual(bank.read_dword(0x4000), 0x11111111)
        self.assertEqual(bank.read_dword(0x4010), 0x22222222)
        self.assertEqual(bank.read_dword(0x4014), 0x33333333)


class SmemStoreTranslatorContractTests(unittest.TestCase):
    """Verifies translator handling of scalar stores.

    tools/rdna2spirv.py Gen currently raises Unsupported on s_store_dword, protecting against
    silent miscompilations until SMEM store PhysicalStorageBuffer write emission is supported.
    """

    def test_translator_raises_unsupported_on_s_store(self):
        g = rdna2spirv.Gen("cs")
        for line in ("s_store_dword s0, s[2:3], 0",
                     "s_store_dwordx2 s[0:1], s[2:3], 0",
                     "s_store_dwordx4 s[0:3], s[4:5], 0",
                     "s_buffer_store_dword s0, s[0:3], 0"):
            with self.subTest(line=line):
                ins = parse_line(line)
                with self.assertRaises(rdna2spirv.Unsupported) as ctx:
                    g.insn(ins)
                self.assertIn(ins["op"], str(ctx.exception))


class SmemStoreMiniInterpStrictnessTests(unittest.TestCase):
    """Verifies that spirv_mini_interp strictly rejects unmodelled SPIR-V constructs for stores."""

    def test_rejects_unsupported_integer_widths(self):
        module = """\
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%ushort = OpTypeInt 16 0
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("integer type that is not 32 or 64 bits", str(ctx.exception))

    def test_rejects_physical_storage_buffer_storage_class(self):
        module = """\
OpCapability Shader
OpCapability PhysicalStorageBufferAddresses
OpMemoryModel PhysicalStorageBuffer64 GLSL450
OpEntryPoint Fragment %main "main"
%void = OpTypeVoid
%uint = OpTypeInt 32 0
%_ptr_PSB = OpTypePointer PhysicalStorageBuffer %uint
%var = OpVariable %_ptr_PSB PhysicalStorageBuffer
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(si.UnsupportedConstruct) as ctx:
            si.run_fragment(module)
        self.assertIn("storage class PhysicalStorageBuffer", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
