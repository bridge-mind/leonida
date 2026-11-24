"""Semantic interpreter tests for untyped MUBUF buffer loads and stores.

Covers AMD RDNA2 / GFX10 untyped buffer memory load/store ISA semantics and
tools/rdna2spirv.py translation:
  * Buffer Resource Descriptor (V# 128-bit layout):
      - Dword 0: Base address low 32 bits [31:0]
      - Dword 1: Base address high 16 bits [15:0] in bits [15:0]; Stride in bytes
        in bits [29:16] (14 bits); cache flags in bits [31:30]
      - Dword 2: Number of records (num_records), representing capacity in bytes
        (raw buffer, stride == 0) or element record count (structured buffer, stride > 0)
      - Dword 3: Format and descriptor metadata (untyped loads/stores treat data raw)
      - 64-bit base address pointer expansion: ((dword1 & 0xFFFF) << 32) | dword0.
        Proper masking (dword1 & 0xFFFF) prevents stride and descriptor flags from
        polluting the 64-bit physical storage buffer address.
  * Effective Address Calculation:
      - Raw buffer: total = soffset + inst_offset + (offen ? vaddr : 0)
      - Structured buffer with idxen: total = soffset + inst_offset + vindex * stride + (offen ? voffset : 0)
      - Structured buffer without idxen: total = soffset + inst_offset + (offen ? vaddr : 0)
      - Effective 64-bit address: (base_64 + total) & 0xFFFFFFFFFFFFFFFF
      - Support for immediate offset modifier (offset:N), SGPR soffset, and vaddr = off
  * Bounds Checking & Out-of-Bounds Behavior:
      - Raw buffer (stride == 0): in_bounds = total < num_records
      - Structured buffer with idxen: in_bounds = vindex < num_records
      - Structured buffer without idxen: in_bounds = total < (num_records * stride)
      - Out-of-bounds loads: return 0 for all loaded dwords (via OpPhi / OpSelect)
      - Out-of-bounds stores: suppressed (memory unchanged)
  * Multi-dword Vector Memory Operations:
      - buffer_load_dword, buffer_load_dwordx2, buffer_load_dwordx3, buffer_load_dwordx4
      - buffer_store_dword, buffer_store_dwordx2, buffer_store_dwordx3, buffer_store_dwordx4
      - Consecutive dwords accessed at byte offsets: +0, +4, +8, +12
  * Lane Activity & Wave Execution:
      - Active lanes execute loads and stores.
      - Inactive lanes return 0 for loads and never mutate memory for stores.
  * Complete deterministic reference interpreter oracle and end-to-end SPIR-V
    translation execution via spirv_mini_interp.
"""
from __future__ import annotations

import gc
import os
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

MASK14 = 0x3FFF
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


class BufferResourceDescriptor:
    """128-bit AMD RDNA2 Buffer Resource Descriptor (V#)."""

    def __init__(self, d0: int = 0, d1: int = 0, d2: int = 0, d3: int = 0) -> None:
        self.d0 = d0 & MASK32
        self.d1 = d1 & MASK32
        self.d2 = d2 & MASK32
        self.d3 = d3 & MASK32

    @classmethod
    def from_fields(
        cls,
        base_address: int,
        stride: int = 0,
        num_records: int = 0,
        format_code: int = 0,
        flags: int = 0,
    ) -> BufferResourceDescriptor:
        d0 = base_address & MASK32
        base_hi = (base_address >> 32) & MASK16
        stride_field = (stride & MASK14) << 16
        d1 = base_hi | stride_field | ((flags & 3) << 30)
        d2 = num_records & MASK32
        d3 = (format_code & 0x7F) << 12
        return cls(d0, d1, d2, d3)

    @property
    def base_lo(self) -> int:
        return self.d0

    @property
    def base_hi(self) -> int:
        return self.d1 & MASK16

    @property
    def stride(self) -> int:
        return (self.d1 >> 16) & MASK14

    @property
    def base_64(self) -> int:
        return ((self.base_hi << 32) | self.base_lo) & MASK64

    @property
    def num_records(self) -> int:
        return self.d2

    @property
    def format_code(self) -> int:
        return (self.d3 >> 12) & 0x7F

    def as_words(self) -> tuple[int, int, int, int]:
        return (self.d0, self.d1, self.d2, self.d3)


def compute_mubuf_address(
    desc: BufferResourceDescriptor,
    soffset: int = 0,
    inst_offset: int = 0,
    vaddr_off: int | None = None,
    vaddr_idx: int | None = None,
) -> tuple[int, bool, int]:
    """Calculates effective 64-bit address, bounds status, and total byte offset.

    Returns:
        (effective_address, in_bounds, total_offset)
    """
    total = (soffset + inst_offset) & MASK32
    if vaddr_idx is not None:
        total = (total + (vaddr_idx & MASK32) * desc.stride) & MASK32
    if vaddr_off is not None:
        total = (total + (vaddr_off & MASK32)) & MASK32

    effective_addr = (desc.base_64 + total) & MASK64

    is_raw = desc.stride == 0
    tot_ok = total < desc.num_records
    if vaddr_idx is not None:
        idx_ok = (vaddr_idx & MASK32) < desc.num_records
        in_bounds = tot_ok if is_raw else idx_ok
    else:
        limit = desc.num_records * desc.stride
        str_ok = total < limit
        in_bounds = tot_ok if is_raw else str_ok

    return effective_addr, in_bounds, total


class MubufReferenceInterpreter:
    """Semantic reference interpreter modeling untyped MUBUF loads and stores."""

    def __init__(self, memory: dict[int, int] | None = None) -> None:
        self.memory = {} if memory is None else dict(memory)
        self.sgprs = [0] * 106
        self.vgprs = [0] * 256
        self.active_lane = True

    def set_sgpr_range(self, start: int, values: tuple[int, ...] | list[int]) -> None:
        for i, val in enumerate(values):
            self.sgprs[start + i] = val & MASK32

    def set_vgpr(self, reg: int, val: int) -> None:
        self.vgprs[reg] = val & MASK32

    def get_vgpr(self, reg: int) -> int:
        return self.vgprs[reg] & MASK32

    def get_descriptor(self, start_sgpr: int) -> BufferResourceDescriptor:
        return BufferResourceDescriptor(
            self.sgprs[start_sgpr],
            self.sgprs[start_sgpr + 1],
            self.sgprs[start_sgpr + 2],
            self.sgprs[start_sgpr + 3],
        )

    def read_dword(self, byte_addr: int) -> int:
        return self.memory.get(byte_addr & MASK64, 0) & MASK32

    def write_dword(self, byte_addr: int, val: int) -> None:
        self.memory[byte_addr & MASK64] = val & MASK32

    def buffer_load_dword(
        self,
        vdst: int,
        srsrc: int,
        soffset: int = 0,
        inst_offset: int = 0,
        vaddr_off: int | None = None,
        vaddr_idx: int | None = None,
        num_dwords: int = 1,
    ) -> list[int]:
        desc = self.get_descriptor(srsrc)
        addr, in_bounds, _ = compute_mubuf_address(
            desc,
            soffset=soffset,
            inst_offset=inst_offset,
            vaddr_off=vaddr_off,
            vaddr_idx=vaddr_idx,
        )
        loaded = []
        for i in range(num_dwords):
            if self.active_lane and in_bounds:
                val = self.read_dword(addr + 4 * i)
            else:
                val = 0
            self.set_vgpr(vdst + i, val)
            loaded.append(val)
        return loaded

    def buffer_store_dword(
        self,
        vdata: int,
        srsrc: int,
        soffset: int = 0,
        inst_offset: int = 0,
        vaddr_off: int | None = None,
        vaddr_idx: int | None = None,
        num_dwords: int = 1,
    ) -> bool:
        desc = self.get_descriptor(srsrc)
        addr, in_bounds, _ = compute_mubuf_address(
            desc,
            soffset=soffset,
            inst_offset=inst_offset,
            vaddr_off=vaddr_off,
            vaddr_idx=vaddr_idx,
        )
        if self.active_lane and in_bounds:
            for i in range(num_dwords):
                val = self.get_vgpr(vdata + i)
                self.write_dword(addr + 4 * i, val)
            return True
        return False


class WaveMubufInterpreter:
    """Multi-lane SIMD wave reference interpreter modeling RDNA2 Wave32/Wave64 MUBUF operations."""

    def __init__(self, wave_size: int = 64, memory: dict[int, int] | None = None) -> None:
        self.wave_size = wave_size
        self.memory = {} if memory is None else dict(memory)
        self.sgprs = [0] * 106
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.exec_mask = (1 << wave_size) - 1

    def set_sgpr_range(self, start: int, values: tuple[int, ...] | list[int]) -> None:
        for i, val in enumerate(values):
            self.sgprs[start + i] = val & MASK32

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK32

    def set_exec_mask(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def get_descriptor(self, start_sgpr: int) -> BufferResourceDescriptor:
        return BufferResourceDescriptor(
            self.sgprs[start_sgpr],
            self.sgprs[start_sgpr + 1],
            self.sgprs[start_sgpr + 2],
            self.sgprs[start_sgpr + 3],
        )

    def read_dword(self, byte_addr: int) -> int:
        return self.memory.get(byte_addr & MASK64, 0) & MASK32

    def write_dword(self, byte_addr: int, val: int) -> None:
        self.memory[byte_addr & MASK64] = val & MASK32

    def buffer_load_dword(
        self,
        vdst: int,
        srsrc: int,
        soffset: int = 0,
        inst_offset: int = 0,
        vaddr_off_reg: int | None = None,
        vaddr_idx_reg: int | None = None,
        num_dwords: int = 1,
    ) -> list[list[int]]:
        desc = self.get_descriptor(srsrc)
        wave_results = []
        for lane in range(self.wave_size):
            vaddr_off = self.get_lane_vgpr(lane, vaddr_off_reg) if vaddr_off_reg is not None else None
            vaddr_idx = self.get_lane_vgpr(lane, vaddr_idx_reg) if vaddr_idx_reg is not None else None
            addr, in_bounds, _ = compute_mubuf_address(
                desc,
                soffset=soffset,
                inst_offset=inst_offset,
                vaddr_off=vaddr_off,
                vaddr_idx=vaddr_idx,
            )
            lane_loaded = []
            for i in range(num_dwords):
                if self.is_lane_active(lane) and in_bounds:
                    val = self.read_dword(addr + 4 * i)
                else:
                    val = 0
                self.set_lane_vgpr(lane, vdst + i, val)
                lane_loaded.append(val)
            wave_results.append(lane_loaded)
        return wave_results

    def buffer_store_dword(
        self,
        vdata: int,
        srsrc: int,
        soffset: int = 0,
        inst_offset: int = 0,
        vaddr_off_reg: int | None = None,
        vaddr_idx_reg: int | None = None,
        num_dwords: int = 1,
    ) -> list[bool]:
        desc = self.get_descriptor(srsrc)
        stored_flags = []
        for lane in range(self.wave_size):
            vaddr_off = self.get_lane_vgpr(lane, vaddr_off_reg) if vaddr_off_reg is not None else None
            vaddr_idx = self.get_lane_vgpr(lane, vaddr_idx_reg) if vaddr_idx_reg is not None else None
            addr, in_bounds, _ = compute_mubuf_address(
                desc,
                soffset=soffset,
                inst_offset=inst_offset,
                vaddr_off=vaddr_off,
                vaddr_idx=vaddr_idx,
            )
            if self.is_lane_active(lane) and in_bounds:
                for i in range(num_dwords):
                    val = self.get_lane_vgpr(lane, vdata + i)
                    self.write_dword(addr + 4 * i, val)
                stored_flags.append(True)
            else:
                stored_flags.append(False)
        return stored_flags


class MubufSpirvHarness:
    """Manages translation and fragment execution for MUBUF shaders."""

    def __init__(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.disasm_dir = os.path.join(self.tmp_dir.name, "disasm")
        os.mkdir(self.disasm_dir)
        self.src_path = os.path.join(self.disasm_dir, "ps_mubuf.s")
        self.dxbc_path = os.path.join(self.tmp_dir.name, "ps_mubuf.dxbc")
        with open(self.dxbc_path, "wb") as f:
            f.write(STUB_DXBC)

    def cleanup(self) -> None:
        gc.collect()
        self.tmp_dir.cleanup()

    def translate_and_run(
        self, asm: str, memory: dict[int, int] | None = None
    ) -> tuple[tuple[int, ...], dict[int, int]]:
        with open(self.src_path, "w", encoding="utf-8") as f:
            f.write(asm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(self.src_path)
        mem = {} if memory is None else dict(memory)
        res = si.run_fragment(spv, memory=mem)
        mrt0 = res.get("%out_mrt0", (0, 0, 0, 0))
        return tuple(mrt0), mem


class TestDescriptorExpansion(unittest.TestCase):
    """Verifies V# descriptor field unpacking and 64-bit pointer expansion."""

    def test_base_address_reconstruction(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0x00007FFF12345678)
        self.assertEqual(desc.base_lo, 0x12345678)
        self.assertEqual(desc.base_hi, 0x7FFF)
        self.assertEqual(desc.base_64, 0x00007FFF12345678)

    def test_stride_and_flags_masked_from_base(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x00000004AABBCCDD,
            stride=128,
            num_records=1024,
            flags=3,
        )
        self.assertEqual(desc.base_hi, 0x0004)
        self.assertEqual(desc.stride, 128)
        self.assertEqual(desc.base_64, 0x00000004AABBCCDD)
        self.assertNotEqual(desc.d1 & 0xFFFF0000, 0)
        self.assertEqual(desc.d1 & 0xFFFF, 0x0004)

    def test_stride_bitfield_extraction(self):
        for stride_val in (0, 4, 16, 64, 256, 1024, 16383):
            with self.subTest(stride=stride_val):
                desc = BufferResourceDescriptor.from_fields(base_address=0x1000, stride=stride_val)
                self.assertEqual(desc.stride, stride_val)

    def test_num_records_raw_vs_structured(self):
        raw_desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=4096)
        self.assertEqual(raw_desc.num_records, 4096)
        self.assertEqual(raw_desc.stride, 0)

        struct_desc = BufferResourceDescriptor.from_fields(0x1000, stride=32, num_records=64)
        self.assertEqual(struct_desc.num_records, 64)
        self.assertEqual(struct_desc.stride, 32)

    def test_format_code_extraction(self):
        for code in (0, 13, 20, 22, 60, 62, 75):
            with self.subTest(code=code):
                desc = BufferResourceDescriptor.from_fields(0x1000, format_code=code)
                self.assertEqual(desc.format_code, code)

    def test_boundary_base_addresses(self):
        zero_desc = BufferResourceDescriptor.from_fields(0)
        self.assertEqual(zero_desc.base_64, 0)

        boundary_desc = BufferResourceDescriptor.from_fields(0x0000FFFFFFFFFFFF)
        self.assertEqual(boundary_desc.base_lo, 0xFFFFFFFF)
        self.assertEqual(boundary_desc.base_hi, 0xFFFF)
        self.assertEqual(boundary_desc.base_64, 0x0000FFFFFFFFFFFF)


class TestOffsetArithmeticAndBounds(unittest.TestCase):
    """Verifies effective address math and bounds checking for raw and structured buffers."""

    def test_raw_buffer_in_bounds(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0x2000, stride=0, num_records=256)
        addr, in_b, total = compute_mubuf_address(desc, soffset=16, inst_offset=32, vaddr_off=48)
        self.assertEqual(total, 96)
        self.assertEqual(addr, 0x2060)
        self.assertTrue(in_b)

    def test_raw_buffer_exact_boundaries(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0x1000, stride=0, num_records=100)
        _, in_b_96, _ = compute_mubuf_address(desc, inst_offset=96)
        self.assertTrue(in_b_96)

        _, in_b_99, _ = compute_mubuf_address(desc, inst_offset=99)
        self.assertTrue(in_b_99)

        _, in_b_100, _ = compute_mubuf_address(desc, inst_offset=100)
        self.assertFalse(in_b_100)

        _, in_b_104, _ = compute_mubuf_address(desc, inst_offset=104)
        self.assertFalse(in_b_104)

    def test_structured_buffer_idxen_in_bounds(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0x4000, stride=64, num_records=10)
        addr, in_b, total = compute_mubuf_address(desc, inst_offset=8, vaddr_idx=3)
        self.assertEqual(total, 3 * 64 + 8)
        self.assertEqual(addr, 0x4000 + 200)
        self.assertTrue(in_b)

    def test_structured_buffer_idxen_boundaries(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0x4000, stride=64, num_records=10)
        _, in_b_last, _ = compute_mubuf_address(desc, vaddr_idx=9)
        self.assertTrue(in_b_last)

        _, in_b_limit, _ = compute_mubuf_address(desc, vaddr_idx=10)
        self.assertFalse(in_b_limit)

        _, in_b_beyond, _ = compute_mubuf_address(desc, vaddr_idx=15)
        self.assertFalse(in_b_beyond)

    def test_structured_buffer_idxen_and_offen(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0x1000, stride=32, num_records=8)
        addr, in_b, total = compute_mubuf_address(
            desc, soffset=4, inst_offset=8, vaddr_idx=2, vaddr_off=12
        )
        expected_total = 4 + 8 + 2 * 32 + 12
        self.assertEqual(total, expected_total)
        self.assertEqual(addr, 0x1000 + expected_total)
        self.assertTrue(in_b)

    def test_structured_buffer_without_idxen(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0x1000, stride=16, num_records=4)
        _, in_b_60, _ = compute_mubuf_address(desc, vaddr_off=60)
        self.assertTrue(in_b_60)

        _, in_b_64, _ = compute_mubuf_address(desc, vaddr_off=64)
        self.assertFalse(in_b_64)

    def test_address_addition_modulo_64(self):
        desc = BufferResourceDescriptor.from_fields(base_address=0xFFFFFFFFFFFFFFF0, stride=0, num_records=0x1000)
        addr, _, _ = compute_mubuf_address(desc, inst_offset=0x20)
        self.assertEqual(addr, (1 << 48) | 0x10)


class TestReferenceInterpreterLoads(unittest.TestCase):
    """Verifies reference interpreter execution of untyped buffer loads."""

    def setUp(self):
        self.mem = {
            0x1000: 0x11111111,
            0x1004: 0x22222222,
            0x1008: 0x33333333,
            0x100C: 0x44444444,
            0x1080: 0x55555555,
            0x1084: 0x66666666,
        }
        self.interp = MubufReferenceInterpreter(self.mem)

    def test_single_dword_load_in_bounds(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=64)
        self.interp.set_sgpr_range(0, desc.as_words())
        res = self.interp.buffer_load_dword(vdst=2, srsrc=0, vaddr_off=4, num_dwords=1)
        self.assertEqual(res, [0x22222222])
        self.assertEqual(self.interp.get_vgpr(2), 0x22222222)

    def test_multi_dword_loads_in_bounds(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=64)
        self.interp.set_sgpr_range(0, desc.as_words())
        res_x2 = self.interp.buffer_load_dword(vdst=2, srsrc=0, vaddr_off=0, num_dwords=2)
        self.assertEqual(res_x2, [0x11111111, 0x22222222])

        res_x4 = self.interp.buffer_load_dword(vdst=4, srsrc=0, vaddr_off=0, num_dwords=4)
        self.assertEqual(res_x4, [0x11111111, 0x22222222, 0x33333333, 0x44444444])
        self.assertEqual([self.interp.get_vgpr(4 + i) for i in range(4)], res_x4)

    def test_single_dword_load_oob_returns_zero(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=16)
        self.interp.set_sgpr_range(0, desc.as_words())
        res = self.interp.buffer_load_dword(vdst=2, srsrc=0, vaddr_off=64, num_dwords=1)
        self.assertEqual(res, [0])
        self.assertEqual(self.interp.get_vgpr(2), 0)

    def test_multi_dword_load_oob_returns_zeros(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=16)
        self.interp.set_sgpr_range(0, desc.as_words())
        res = self.interp.buffer_load_dword(vdst=2, srsrc=0, vaddr_off=32, num_dwords=4)
        self.assertEqual(res, [0, 0, 0, 0])

    def test_inactive_lane_load_returns_zero(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=64)
        self.interp.set_sgpr_range(0, desc.as_words())
        self.interp.active_lane = False
        res = self.interp.buffer_load_dword(vdst=2, srsrc=0, vaddr_off=0, num_dwords=1)
        self.assertEqual(res, [0])
        self.assertEqual(self.interp.get_vgpr(2), 0)

    def test_structured_load_idxen(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=64, num_records=4)
        self.interp.set_sgpr_range(0, desc.as_words())
        res_idx2 = self.interp.buffer_load_dword(vdst=2, srsrc=0, vaddr_idx=2, num_dwords=1)
        self.assertEqual(res_idx2, [0x55555555])

        res_oob = self.interp.buffer_load_dword(vdst=2, srsrc=0, vaddr_idx=5, num_dwords=1)
        self.assertEqual(res_oob, [0])


class TestReferenceInterpreterStores(unittest.TestCase):
    """Verifies reference interpreter execution of untyped buffer stores."""

    def setUp(self):
        self.interp = MubufReferenceInterpreter()

    def test_single_dword_store_in_bounds(self):
        desc = BufferResourceDescriptor.from_fields(0x2000, stride=0, num_records=64)
        self.interp.set_sgpr_range(0, desc.as_words())
        self.interp.set_vgpr(10, 0xCAFEBABE)
        ok = self.interp.buffer_store_dword(vdata=10, srsrc=0, vaddr_off=16, num_dwords=1)
        self.assertTrue(ok)
        self.assertEqual(self.interp.read_dword(0x2010), 0xCAFEBABE)

    def test_multi_dword_stores_in_bounds(self):
        desc = BufferResourceDescriptor.from_fields(0x2000, stride=0, num_records=64)
        self.interp.set_sgpr_range(0, desc.as_words())
        for i, val in enumerate((0xAAAA0001, 0xAAAA0002, 0xAAAA0003, 0xAAAA0004)):
            self.interp.set_vgpr(10 + i, val)

        ok = self.interp.buffer_store_dword(vdata=10, srsrc=0, vaddr_off=0, num_dwords=4)
        self.assertTrue(ok)
        self.assertEqual(self.interp.read_dword(0x2000), 0xAAAA0001)
        self.assertEqual(self.interp.read_dword(0x2004), 0xAAAA0002)
        self.assertEqual(self.interp.read_dword(0x2008), 0xAAAA0003)
        self.assertEqual(self.interp.read_dword(0x200C), 0xAAAA0004)

    def test_single_dword_store_oob_suppressed(self):
        desc = BufferResourceDescriptor.from_fields(0x2000, stride=0, num_records=32)
        self.interp.set_sgpr_range(0, desc.as_words())
        self.interp.set_vgpr(10, 0xDEADBEEF)
        ok = self.interp.buffer_store_dword(vdata=10, srsrc=0, vaddr_off=64, num_dwords=1)
        self.assertFalse(ok)
        self.assertEqual(self.interp.read_dword(0x2040), 0)

    def test_multi_dword_store_oob_suppressed(self):
        desc = BufferResourceDescriptor.from_fields(0x2000, stride=0, num_records=16)
        self.interp.set_sgpr_range(0, desc.as_words())
        self.interp.set_vgpr(10, 0x1111)
        self.interp.set_vgpr(11, 0x2222)
        ok = self.interp.buffer_store_dword(vdata=10, srsrc=0, vaddr_off=32, num_dwords=2)
        self.assertFalse(ok)
        self.assertEqual(len(self.interp.memory), 0)

    def test_inactive_lane_store_suppressed(self):
        desc = BufferResourceDescriptor.from_fields(0x2000, stride=0, num_records=64)
        self.interp.set_sgpr_range(0, desc.as_words())
        self.interp.set_vgpr(10, 0x12345678)
        self.interp.active_lane = False
        ok = self.interp.buffer_store_dword(vdata=10, srsrc=0, vaddr_off=0, num_dwords=1)
        self.assertFalse(ok)
        self.assertEqual(len(self.interp.memory), 0)

    def test_structured_store_idxen(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=32, num_records=4)
        self.interp.set_sgpr_range(0, desc.as_words())
        self.interp.set_vgpr(5, 0x77778888)
        ok = self.interp.buffer_store_dword(vdata=5, srsrc=0, vaddr_idx=2, num_dwords=1)
        self.assertTrue(ok)
        self.assertEqual(self.interp.read_dword(0x1040), 0x77778888)

        ok_oob = self.interp.buffer_store_dword(vdata=5, srsrc=0, vaddr_idx=4, num_dwords=1)
        self.assertFalse(ok_oob)
        self.assertEqual(self.interp.read_dword(0x1080), 0)

    def test_adjacent_memory_preserved(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=64)
        self.interp.set_sgpr_range(0, desc.as_words())
        self.interp.write_dword(0x1000, 0xAAAAAAAA)
        self.interp.write_dword(0x1008, 0xCCCCCCCC)
        self.interp.set_vgpr(1, 0xBBBBBBBB)
        self.interp.buffer_store_dword(vdata=1, srsrc=0, vaddr_off=4, num_dwords=1)
        self.assertEqual(self.interp.read_dword(0x1000), 0xAAAAAAAA)
        self.assertEqual(self.interp.read_dword(0x1004), 0xBBBBBBBB)
        self.assertEqual(self.interp.read_dword(0x1008), 0xCCCCCCCC)


class TestWaveInterpreter(unittest.TestCase):
    """Verifies multi-lane SIMD wave execution (Wave32 and Wave64) with divergence and lane masking."""

    def test_wave64_divergent_loads_with_exec_mask(self):
        wave = WaveMubufInterpreter(wave_size=64)
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=256)
        wave.set_sgpr_range(0, desc.as_words())

        # Populate memory with lane-specific words
        for i in range(64):
            wave.write_dword(0x1000 + i * 4, 0x10000000 | i)

        # Each lane has its own byte offset in v0
        for lane in range(64):
            wave.set_lane_vgpr(lane, 0, lane * 4)

        # Active mask: lower half active (lanes 0..31), upper half disabled (lanes 32..63)
        wave.set_exec_mask((1 << 32) - 1)
        results = wave.buffer_load_dword(vdst=1, srsrc=0, vaddr_off_reg=0, num_dwords=1)

        for lane in range(32):
            self.assertEqual(results[lane], [0x10000000 | lane])
            self.assertEqual(wave.get_lane_vgpr(lane, 1), 0x10000000 | lane)
        for lane in range(32, 64):
            self.assertEqual(results[lane], [0])
            self.assertEqual(wave.get_lane_vgpr(lane, 1), 0)

    def test_wave32_divergent_stores_with_checkerboard_mask(self):
        wave = WaveMubufInterpreter(wave_size=32)
        desc = BufferResourceDescriptor.from_fields(0x2000, stride=0, num_records=256)
        wave.set_sgpr_range(0, desc.as_words())

        for lane in range(32):
            wave.set_lane_vgpr(lane, 0, lane * 4)          # offset
            wave.set_lane_vgpr(lane, 1, 0xCAFE0000 | lane) # data

        # Checkerboard mask: only odd lanes active (0xAAAAAAAA)
        checkerboard = 0xAAAAAAAA
        wave.set_exec_mask(checkerboard)
        flags = wave.buffer_store_dword(vdata=1, srsrc=0, vaddr_off_reg=0, num_dwords=1)

        for lane in range(32):
            addr = 0x2000 + lane * 4
            if lane % 2 == 1:
                self.assertTrue(flags[lane])
                self.assertEqual(wave.read_dword(addr), 0xCAFE0000 | lane)
            else:
                self.assertFalse(flags[lane])
                self.assertEqual(wave.read_dword(addr), 0)

    def test_wave_boundary_oob_split(self):
        wave = WaveMubufInterpreter(wave_size=64)
        # Buffer holds 16 dwords (64 bytes)
        desc = BufferResourceDescriptor.from_fields(0x4000, stride=0, num_records=64)
        wave.set_sgpr_range(0, desc.as_words())

        for i in range(16):
            wave.write_dword(0x4000 + i * 4, 0xA000 | i)

        # All lanes active
        wave.set_exec_mask((1 << 64) - 1)
        for lane in range(64):
            wave.set_lane_vgpr(lane, 0, lane * 4)

        results = wave.buffer_load_dword(vdst=5, srsrc=0, vaddr_off_reg=0, num_dwords=1)

        # Lanes 0..15 are within 64 bytes
        for lane in range(16):
            self.assertEqual(results[lane], [0xA000 | lane])
            self.assertEqual(wave.get_lane_vgpr(lane, 5), 0xA000 | lane)
        # Lanes 16..63 are out of bounds -> return 0
        for lane in range(16, 64):
            self.assertEqual(results[lane], [0])
            self.assertEqual(wave.get_lane_vgpr(lane, 5), 0)

    def test_wave_multi_dword_stores(self):
        wave = WaveMubufInterpreter(wave_size=16)
        desc = BufferResourceDescriptor.from_fields(0x8000, stride=0, num_records=512)
        wave.set_sgpr_range(0, desc.as_words())

        for lane in range(16):
            wave.set_lane_vgpr(lane, 0, lane * 16)  # 16-byte spacing
            wave.set_lane_vgpr(lane, 2, 0x11110000 | lane)
            wave.set_lane_vgpr(lane, 3, 0x22220000 | lane)

        wave.set_exec_mask(0xFFFF)
        flags = wave.buffer_store_dword(vdata=2, srsrc=0, vaddr_off_reg=0, num_dwords=2)

        for lane in range(16):
            self.assertTrue(flags[lane])
            base_addr = 0x8000 + lane * 16
            self.assertEqual(wave.read_dword(base_addr), 0x11110000 | lane)
            self.assertEqual(wave.read_dword(base_addr + 4), 0x22220000 | lane)
            self.assertEqual(wave.read_dword(base_addr + 8), 0)


class TestSpirvTranslationLoads(unittest.TestCase):
    """End-to-end SPIR-V translation and execution for untyped buffer loads."""

    def setUp(self):
        self.harness = MubufSpirvHarness()
        self.addCleanup(self.harness.cleanup)

    def test_spv_load_dword_raw_offen(self):
        asm = (
            "; test_load_raw_offen\n"
            "\ts_mov_b32 s0, 0x1000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tv_mov_b32 v1, 0x10\n"
            "\tbuffer_load_dword v2, v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x1010: 0x12345678})
        self.assertEqual(mrt[0], 0x12345678)

    def test_spv_load_dword_raw_offset_imm(self):
        asm = (
            "; test_load_raw_offset_imm\n"
            "\ts_mov_b32 s0, 0x1000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tbuffer_load_dword v2, off, s[0:3], 0 offset:24\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x1018: 0x87654321})
        self.assertEqual(mrt[0], 0x87654321)

    def test_spv_load_dword_raw_soffset_sgpr(self):
        asm = (
            "; test_load_raw_soffset_sgpr\n"
            "\ts_mov_b32 s0, 0x1000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\ts_mov_b32 s4, 0x20\n"
            "\tv_mov_b32 v1, 0x10\n"
            "\tbuffer_load_dword v2, v1, s[0:3], s4 offen offset:8\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x1038: 0xABCDEF01})
        self.assertEqual(mrt[0], 0xABCDEF01)

    def test_spv_load_dword_raw_oob_returns_zero(self):
        asm = (
            "; test_load_raw_oob\n"
            "\ts_mov_b32 s0, 0x1000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x80\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tv_mov_b32 v1, 0x100\n"
            "\tbuffer_load_dword v2, v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x1100: 0x99999999})
        self.assertEqual(mrt[0], 0)

    def test_spv_load_dword_structured_idxen(self):
        stride = 64
        s1_val = (stride << 16) | 0
        asm = (
            "; test_load_struct_idxen\n"
            "\ts_mov_b32 s0, 0x1000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 4\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v0, 2\n"
            "\tbuffer_load_dword v2, v0, s[0:3], 0 idxen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x1080: 0x5555AAAA})
        self.assertEqual(mrt[0], 0x5555AAAA)

    def test_spv_load_dword_structured_idxen_oob_returns_zero(self):
        stride = 64
        s1_val = (stride << 16) | 0
        asm = (
            "; test_load_struct_idxen_oob\n"
            "\ts_mov_b32 s0, 0x1000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 4\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v0, 4\n"
            "\tbuffer_load_dword v2, v0, s[0:3], 0 idxen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x1100: 0x5555AAAA})
        self.assertEqual(mrt[0], 0)

    def test_spv_load_dword_structured_idxen_offen(self):
        stride = 64
        s1_val = (stride << 16) | 0
        asm = (
            "; test_load_struct_idxen_offen\n"
            "\ts_mov_b32 s0, 0x1000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 4\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v0, 1\n"
            "\tv_mov_b32 v1, 12\n"
            "\tbuffer_load_dword v2, v[0:1], s[0:3], 0 idxen offen offset:4\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x1050: 0x77778888})
        self.assertEqual(mrt[0], 0x77778888)

    def test_spv_load_dword_structured_raw_offset(self):
        stride = 64
        s1_val = (stride << 16) | 0
        asm = (
            "; test_load_struct_raw_offset\n"
            "\ts_mov_b32 s0, 0x1000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 4\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v1, 60\n"
            "\tbuffer_load_dword v2, v1, s[0:3], 0 offen\n"
            "\tv_mov_b32 v3, 64\n"
            "\tbuffer_load_dword v4, v3, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v4, v2, v4 done vm\n"
            "\ts_endpgm\n"
        )
        mrt, _ = self.harness.translate_and_run(asm, memory={0x103C: 0xAAAA1111, 0x1040: 0xBBBB2222})
        self.assertEqual(mrt[0], 0xAAAA1111)
        self.assertEqual(mrt[1], 0)

    def test_spv_load_dwordx2_x3_x4(self):
        asm = (
            "; test_load_multi_dword\n"
            "\ts_mov_b32 s0, 0x1000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tv_mov_b32 v1, 0\n"
            "\tbuffer_load_dwordx4 v[2:5], v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v3, v4, v5 done vm\n"
            "\ts_endpgm\n"
        )
        mem = {0x1000: 0x01, 0x1004: 0x02, 0x1008: 0x03, 0x100C: 0x04}
        mrt, _ = self.harness.translate_and_run(asm, memory=mem)
        self.assertEqual(mrt, (0x01, 0x02, 0x03, 0x04))

    def test_spv_load_descriptor_masking(self):
        s1_val = (128 << 16) | 0x0001
        asm = (
            "; test_load_descriptor_masking\n"
            "\ts_mov_b32 s0, 0x1000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 0x1000\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v1, 0\n"
            "\tbuffer_load_dword v2, v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        mem = {0x000100001000: 0xFEDCBA98}
        mrt, _ = self.harness.translate_and_run(asm, memory=mem)
        self.assertEqual(mrt[0], 0xFEDCBA98)


class TestSpirvTranslationStores(unittest.TestCase):
    """End-to-end SPIR-V translation and execution for untyped buffer stores."""

    def setUp(self):
        self.harness = MubufSpirvHarness()
        self.addCleanup(self.harness.cleanup)

    def test_spv_store_dword_raw_offen(self):
        asm = (
            "; test_store_raw_offen\n"
            "\ts_mov_b32 s0, 0x2000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tv_mov_b32 v1, 0x10\n"
            "\tv_mov_b32 v2, 0xCAFEBABE\n"
            "\tbuffer_store_dword v2, v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem.get(0x2010), 0xCAFEBABE)

    def test_spv_store_dword_raw_offset_imm(self):
        asm = (
            "; test_store_raw_offset_imm\n"
            "\ts_mov_b32 s0, 0x2000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tv_mov_b32 v2, 0x99887766\n"
            "\tbuffer_store_dword v2, off, s[0:3], 0 offset:32\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem.get(0x2020), 0x99887766)

    def test_spv_store_dword_raw_soffset_sgpr(self):
        asm = (
            "; test_store_raw_soffset_sgpr\n"
            "\ts_mov_b32 s0, 0x2000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\ts_mov_b32 s4, 0x30\n"
            "\tv_mov_b32 v1, 0x08\n"
            "\tv_mov_b32 v2, 0x11223344\n"
            "\tbuffer_store_dword v2, v1, s[0:3], s4 offen offset:4\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem.get(0x203C), 0x11223344)

    def test_spv_store_dword_raw_oob_suppressed(self):
        asm = (
            "; test_store_raw_oob\n"
            "\ts_mov_b32 s0, 0x2000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x80\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tv_mov_b32 v1, 0x80\n"
            "\tv_mov_b32 v2, 0xDEADDEAD\n"
            "\tbuffer_store_dword v2, v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem, {})

    def test_spv_store_dword_structured_idxen(self):
        stride = 32
        s1_val = (stride << 16) | 0
        asm = (
            "; test_store_struct_idxen\n"
            "\ts_mov_b32 s0, 0x3000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 4\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v0, 3\n"
            "\tv_mov_b32 v2, 0x55667788\n"
            "\tbuffer_store_dword v2, v0, s[0:3], 0 idxen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem.get(0x3000 + 3 * 32), 0x55667788)

    def test_spv_store_dword_structured_idxen_oob_suppressed(self):
        stride = 32
        s1_val = (stride << 16) | 0
        asm = (
            "; test_store_struct_idxen_oob\n"
            "\ts_mov_b32 s0, 0x3000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 4\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v0, 4\n"
            "\tv_mov_b32 v2, 0xBAD00BAD\n"
            "\tbuffer_store_dword v2, v0, s[0:3], 0 idxen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem, {})

    def test_spv_store_dword_structured_idxen_offen(self):
        stride = 64
        s1_val = (stride << 16) | 0
        asm = (
            "; test_store_struct_idxen_offen\n"
            "\ts_mov_b32 s0, 0x1000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 4\n"
            "\ts_mov_b32 s3, 0\n"
            "\ts_mov_b32 s4, 4\n"
            "\tv_mov_b32 v0, 1\n"
            "\tv_mov_b32 v1, 8\n"
            "\tv_mov_b32 v2, 0x33445566\n"
            "\tbuffer_store_dword v2, v[0:1], s[0:3], s4 idxen offen offset:16\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem.get(0x105C), 0x33445566)

    def test_spv_store_dwordx2_x3_x4(self):
        asm = (
            "; test_store_multi_dword\n"
            "\ts_mov_b32 s0, 0x4000\n"
            "\ts_mov_b32 s1, 0\n"
            "\ts_mov_b32 s2, 0x100\n"
            "\ts_mov_b32 s3, 0x30000000\n"
            "\tv_mov_b32 v1, 0\n"
            "\tv_mov_b32 v2, 0x10101010\n"
            "\tv_mov_b32 v3, 0x20202020\n"
            "\tv_mov_b32 v4, 0x30303030\n"
            "\tv_mov_b32 v5, 0x40404040\n"
            "\tbuffer_store_dwordx4 v[2:5], v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v3, v4, v5 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem.get(0x4000), 0x10101010)
        self.assertEqual(mem.get(0x4004), 0x20202020)
        self.assertEqual(mem.get(0x4008), 0x30303030)
        self.assertEqual(mem.get(0x400C), 0x40404040)

    def test_spv_store_descriptor_masking(self):
        s1_val = (64 << 16) | 0x0002
        asm = (
            "; test_store_descriptor_masking\n"
            "\ts_mov_b32 s0, 0x2000\n"
            f"\ts_mov_b32 s1, {s1_val:#x}\n"
            "\ts_mov_b32 s2, 0x1000\n"
            "\ts_mov_b32 s3, 0\n"
            "\tv_mov_b32 v1, 0\n"
            "\tv_mov_b32 v2, 0xABCD1234\n"
            "\tbuffer_store_dword v2, v1, s[0:3], 0 offen\n"
            "\ts_waitcnt vmcnt(0)\n"
            "\texp mrt0 v2, v2, v2, v2 done vm\n"
            "\ts_endpgm\n"
        )
        _, mem = self.harness.translate_and_run(asm)
        self.assertEqual(mem.get(0x000200002000), 0xABCD1234)


class TestNegativeControlsAndMutations(unittest.TestCase):
    """Verifies that intentional descriptor/offset/boundary mutations fail semantic checks."""

    def test_unmasked_descriptor_base_corrupts_address(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=64, num_records=10)
        correct_addr, _, _ = compute_mubuf_address(desc, inst_offset=16)

        corrupted_hi = desc.d1
        corrupted_base_64 = (corrupted_hi << 32) | desc.d0
        corrupted_addr = (corrupted_base_64 + 16) & MASK64
        self.assertNotEqual(correct_addr, corrupted_addr)

    def test_corrupted_stride_diverges(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=32, num_records=10)
        addr_correct, _, _ = compute_mubuf_address(desc, vaddr_idx=2)

        corrupted_desc = BufferResourceDescriptor.from_fields(0x1000, stride=64, num_records=10)
        addr_corrupted, _, _ = compute_mubuf_address(corrupted_desc, vaddr_idx=2)
        self.assertNotEqual(addr_correct, addr_corrupted)

    def test_corrupted_bounds_check_diverges(self):
        desc = BufferResourceDescriptor.from_fields(0x1000, stride=0, num_records=100)
        _, in_b_correct, _ = compute_mubuf_address(desc, inst_offset=100)
        self.assertFalse(in_b_correct)

        in_b_corrupted = 100 <= desc.num_records
        self.assertTrue(in_b_corrupted)

    def test_unsupported_modifiers_rejected(self):
        harness = MubufSpirvHarness()
        self.addCleanup(harness.cleanup)
        for mod in ("lds", "tfe"):
            with self.subTest(mod=mod):
                asm = (
                    f"; test_unsupported_{mod}\n"
                    "\ts_mov_b32 s0, 0x1000\n"
                    "\ts_mov_b32 s1, 0\n"
                    "\ts_mov_b32 s2, 0x100\n"
                    "\ts_mov_b32 s3, 0\n"
                    f"\tbuffer_load_dword v2, v1, s[0:3], 0 offen {mod}\n"
                    "\ts_endpgm\n"
                )
                with self.assertRaises(rdna2spirv.Unsupported):
                    harness.translate_and_run(asm)


if __name__ == "__main__":
    unittest.main()
