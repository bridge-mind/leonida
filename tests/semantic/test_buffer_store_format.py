"""Semantic interpreter tests for AMD RDNA2 / GFX10 typed buffer stores (MTBUF).

Covers ISA semantics, format conversion, addressing, bounds checking, and translation:
  * Typed buffer store instructions:
      - tbuffer_store_format_x: 1 component store
      - tbuffer_store_format_xy: 2 component store
      - tbuffer_store_format_xyz: 3 component store
      - tbuffer_store_format_xyzw: 4 component store
  * GFX10 format packings & numerical conversions:
      - BUF_DATA_FORMAT_8 / BUF_NUM_FORMAT_UNORM (code 1, 8-bit unorm in [0, 255])
      - BUF_DATA_FORMAT_16 / BUF_NUM_FORMAT_FLOAT (code 13/29/71, 16-bit IEEE 754 half)
      - BUF_DATA_FORMAT_32 / BUF_NUM_FORMAT_UINT (code 20/62/72/75, 32-bit unsigned int)
      - BUF_DATA_FORMAT_32 / BUF_NUM_FORMAT_FLOAT (code 22/64/74/77, 32-bit IEEE 754 single)
  * Buffer Resource Descriptor (V# 128-bit layout):
      - Dword 0: Base address low 32 bits [31:0]
      - Dword 1: Base address high 16 bits [15:0]; Stride in bytes [29:16]; flags [31:30]
      - Dword 2: Number of records (num_records capacity limit)
      - Dword 3: Format code [18:12]; DST_SEL [11:0]; OOB_SELECT [29:28]
  * Addressing calculation:
      - effective_offset = (voffset + (soffset & ~3) + inst_offset)
      - Structured buffers: effective_offset + vindex * stride
      - Effective 64-bit physical address: (base_64 + total_offset) & 0xFFFFFFFFFFFFFFFF
  * Bounds checking:
      - Stores beyond num_records are silently ignored / no-op (memory untouched).
"""
from __future__ import annotations

import math
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

import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402


def make_fssm(rsrc2: int = 0x8088, group: tuple[int, int, int] = (64, 1, 1)) -> bytes:
    """The compute-shader header words the translator reads: COMPUTE_PGM_RSRC2 and thread group size."""
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, rsrc2, *group)
    return bytes(header)


CS_FSSM = make_fssm(0x8088, (64, 1, 1))
OUT_VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def run_cs(spv: str, memory: dict[int, int], push_constants: list[int] | None = None) -> dict[str, tuple]:
    """Execute compute shader through mini-interpreter with specified push constants and memory."""
    pc = push_constants if push_constants is not None else (OUT_VSHARP + [0] * 28)
    return si.run_fragment(
        spv,
        inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
        push_constants=pc,
        memory=memory,
    )

MASK14 = 0x3FFF
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

BUF_DATA_FORMAT_8 = 1
BUF_DATA_FORMAT_16 = 2
BUF_DATA_FORMAT_32 = 4

BUF_NUM_FORMAT_UNORM = 1
BUF_NUM_FORMAT_FLOAT = 7
BUF_NUM_FORMAT_UINT = 4
BUF_NUM_FORMAT_SINT = 5

GFX10_FMT_8_UNORM = 1
GFX10_FMT_16_FLOAT = 13
GFX10_FMT_16_16_FLOAT = 29
GFX10_FMT_16_16_16_16_FLOAT = 71
GFX10_FMT_32_UINT = 20
GFX10_FMT_32_32_UINT = 62
GFX10_FMT_32_32_32_UINT = 72
GFX10_FMT_32_32_32_32_UINT = 75
GFX10_FMT_32_FLOAT = 22
GFX10_FMT_32_32_FLOAT = 64
GFX10_FMT_32_32_32_FLOAT = 74
GFX10_FMT_32_32_32_32_FLOAT = 77
GFX10_FMT_8_8_8_8_UNORM = 56

FORMAT_ELEMENT_SIZES: dict[int, int] = {
    GFX10_FMT_8_UNORM: 1,
    GFX10_FMT_16_FLOAT: 2,
    GFX10_FMT_16_16_FLOAT: 4,
    GFX10_FMT_16_16_16_16_FLOAT: 8,
    GFX10_FMT_32_UINT: 4,
    GFX10_FMT_32_32_UINT: 8,
    GFX10_FMT_32_32_32_UINT: 12,
    GFX10_FMT_32_32_32_32_UINT: 16,
    GFX10_FMT_32_FLOAT: 4,
    GFX10_FMT_32_32_FLOAT: 8,
    GFX10_FMT_32_32_32_FLOAT: 12,
    GFX10_FMT_32_32_32_32_FLOAT: 16,
    GFX10_FMT_8_8_8_8_UNORM: 4,
}


def pack_unorm8(value: float) -> int:
    """Pack single float value into 8-bit unsigned normalized [0, 255].

    NaN maps to 0.0 before clamping. Clamps to [0.0, 1.0] and rounds to nearest even.
    """
    if math.isnan(value):
        value = 0.0
    clamped = max(0.0, min(1.0, value))
    scaled = clamped * 255.0
    rounded = int(round(scaled))
    return max(0, min(255, rounded))


def pack_float16(value: float) -> int:
    """Pack single float value into 16-bit IEEE 754 half-precision float."""
    raw = struct.pack("<e", value)
    return struct.unpack("<H", raw)[0]


def pack_uint32(value: int) -> int:
    """Pack integer value into 32-bit unsigned integer."""
    return int(value) & MASK32


def pack_float32(value: float) -> int:
    """Pack single float value into 32-bit IEEE 754 single-precision bit pattern."""
    raw = struct.pack("<f", value)
    return struct.unpack("<I", raw)[0]


def pack_typed_format_bytes(values: tuple[float | int, ...] | list[float | int], format_code: int) -> bytes:
    """Pack component values into raw bytes for a given GFX10 format code."""
    if format_code == GFX10_FMT_8_UNORM:
        val = pack_unorm8(float(values[0]))
        return struct.pack("<B", val)
    elif format_code == GFX10_FMT_8_8_8_8_UNORM:
        padded = list(values[:4]) + [0.0] * max(0, 4 - len(values))
        unorms = [pack_unorm8(float(v)) for v in padded[:4]]
        return struct.pack("<4B", *unorms)
    elif format_code == GFX10_FMT_16_FLOAT:
        val = pack_float16(float(values[0]))
        return struct.pack("<H", val)
    elif format_code == GFX10_FMT_16_16_FLOAT:
        padded = list(values[:2]) + [0.0] * max(0, 2 - len(values))
        f16s = [pack_float16(float(v)) for v in padded[:2]]
        return struct.pack("<2H", *f16s)
    elif format_code == GFX10_FMT_16_16_16_16_FLOAT:
        padded = list(values[:4]) + [0.0] * max(0, 4 - len(values))
        f16s = [pack_float16(float(v)) for v in padded[:4]]
        return struct.pack("<4H", *f16s)
    elif format_code in (GFX10_FMT_32_UINT, GFX10_FMT_32_32_UINT, GFX10_FMT_32_32_32_UINT, GFX10_FMT_32_32_32_32_UINT):
        count = {GFX10_FMT_32_UINT: 1, GFX10_FMT_32_32_UINT: 2, GFX10_FMT_32_32_32_UINT: 3, GFX10_FMT_32_32_32_32_UINT: 4}[format_code]
        padded = list(values[:count]) + [0] * max(0, count - len(values))
        u32s = [pack_uint32(int(v)) for v in padded[:count]]
        return struct.pack(f"<{count}I", *u32s)
    elif format_code in (GFX10_FMT_32_FLOAT, GFX10_FMT_32_32_FLOAT, GFX10_FMT_32_32_32_FLOAT, GFX10_FMT_32_32_32_32_FLOAT):
        count = {GFX10_FMT_32_FLOAT: 1, GFX10_FMT_32_32_FLOAT: 2, GFX10_FMT_32_32_32_FLOAT: 3, GFX10_FMT_32_32_32_32_FLOAT: 4}[format_code]
        padded = list(values[:count]) + [0.0] * max(0, count - len(values))
        u32s = [pack_float32(float(v)) for v in padded[:count]]
        return struct.pack(f"<{count}I", *u32s)
    else:
        raise ValueError(f"Unsupported format code: {format_code}")


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
        dst_sel: int = 0,
        oob_select: int = 3,
        flags: int = 0,
    ) -> BufferResourceDescriptor:
        d0 = base_address & MASK32
        base_hi = (base_address >> 32) & MASK16
        stride_field = (stride & MASK14) << 16
        d1 = base_hi | stride_field | ((flags & 3) << 30)
        d2 = num_records & MASK32
        d3 = ((format_code & 0x7F) << 12) | (dst_sel & 0xFFF) | ((oob_select & 3) << 28)
        return cls(d0, d1, d2, d3)

    @property
    def base_lo(self) -> int:
        return self.d0

    @property
    def base_hi(self) -> int:
        return self.d1 & MASK16

    @property
    def base_address_64(self) -> int:
        return ((self.base_hi << 32) | self.base_lo) & MASK64

    @property
    def stride(self) -> int:
        return (self.d1 >> 16) & MASK14

    @property
    def num_records(self) -> int:
        return self.d2

    @property
    def format_code(self) -> int:
        return (self.d3 >> 12) & 0x7F

    @property
    def dst_sel(self) -> int:
        return self.d3 & 0xFFF

    @property
    def oob_select(self) -> int:
        return (self.d3 >> 28) & 0x3

    def as_words(self) -> tuple[int, int, int, int]:
        return (self.d0, self.d1, self.d2, self.d3)


def compute_typed_buffer_address(
    desc: BufferResourceDescriptor,
    soffset: int = 0,
    inst_offset: int = 0,
    voffset: int = 0,
    vindex: int | None = None,
    element_size: int = 4,
) -> tuple[int, bool, int]:
    """Calculates effective 64-bit physical address, in-bounds status, and total byte offset.

    Addressing calculation rule:
      effective_offset = (voffset + (soffset & ~3) + inst_offset)
    With structured buffers:
      total_offset = effective_offset + (vindex * stride if vindex is not None else 0)

    Returns:
      (effective_address, in_bounds, total_offset)
    """
    masked_soff = soffset & ~3
    effective_offset = (voffset + masked_soff + inst_offset) & MASK32

    if vindex is not None:
        total_offset = (effective_offset + (vindex & MASK32) * desc.stride) & MASK32
    else:
        total_offset = effective_offset

    effective_addr = (desc.base_address_64 + total_offset) & MASK64

    if desc.oob_select == 2:
        in_bounds = desc.num_records != 0
    elif desc.oob_select == 0:
        if vindex is not None:
            idx_ok = (vindex & MASK32) < desc.num_records
            off_ok = (effective_offset + element_size) <= desc.stride
            in_bounds = idx_ok and off_ok
        else:
            in_bounds = (total_offset + element_size) <= (desc.num_records * desc.stride)
    elif desc.oob_select == 1:
        if vindex is not None:
            in_bounds = (vindex & MASK32) < desc.num_records
        else:
            in_bounds = (total_offset + element_size) <= (desc.num_records * desc.stride)
    else:
        in_bounds = (total_offset + element_size) <= desc.num_records

    return effective_addr, in_bounds, total_offset


class TypedBufferReferenceInterpreter:
    """Semantic reference interpreter modeling RDNA2 typed buffer stores (MTBUF)."""

    def __init__(self, memory: dict[int, int] | None = None) -> None:
        self.memory: dict[int, int] = {} if memory is None else dict(memory)

    def read_dword(self, byte_addr: int) -> int:
        aligned = byte_addr & ~3
        return self.memory.get(aligned & MASK64, 0) & MASK32

    def write_bytes(self, byte_addr: int, data: bytes) -> None:
        for offset, byte_val in enumerate(data):
            curr_addr = byte_addr + offset
            aligned = curr_addr & ~3
            shift = (curr_addr & 3) * 8
            old_dword = self.read_dword(aligned)
            cleared = old_dword & ~(0xFF << shift)
            new_dword = cleared | ((byte_val & 0xFF) << shift)
            self.memory[aligned & MASK64] = new_dword & MASK32

    def store_format(
        self,
        vdata: tuple[float | int, ...] | list[float | int],
        desc: BufferResourceDescriptor,
        soffset: int = 0,
        inst_offset: int = 0,
        voffset: int = 0,
        vindex: int | None = None,
        format_code: int = GFX10_FMT_8_UNORM,
    ) -> bool:
        elem_size = FORMAT_ELEMENT_SIZES.get(format_code, 4)
        addr, in_bounds, _ = compute_typed_buffer_address(
            desc,
            soffset=soffset,
            inst_offset=inst_offset,
            voffset=voffset,
            vindex=vindex,
            element_size=elem_size,
        )
        if desc.format_code == 0:
            return False
        if not in_bounds:
            return False
        packed = pack_typed_format_bytes(vdata, format_code)
        self.write_bytes(addr, packed)
        return True


class TestBufferDescriptor(unittest.TestCase):
    def test_descriptor_field_extraction(self) -> None:
        base = 0x00007FFF80001000
        stride = 64
        num_records = 256
        format_code = GFX10_FMT_32_32_32_32_FLOAT
        dst_sel = 0xF2E
        oob_sel = 3
        desc = BufferResourceDescriptor.from_fields(
            base_address=base,
            stride=stride,
            num_records=num_records,
            format_code=format_code,
            dst_sel=dst_sel,
            oob_select=oob_sel,
        )
        self.assertEqual(desc.base_lo, 0x80001000)
        self.assertEqual(desc.base_hi, 0x7FFF)
        self.assertEqual(desc.base_address_64, base)
        self.assertEqual(desc.stride, 64)
        self.assertEqual(desc.num_records, 256)
        self.assertEqual(desc.format_code, GFX10_FMT_32_32_32_32_FLOAT)
        self.assertEqual(desc.dst_sel, 0xF2E)
        self.assertEqual(desc.oob_select, 3)

    def test_descriptor_raw_words_roundtrip(self) -> None:
        words = (0x12345678, 0x0040ABCD, 0x00000400, 0x3004D123)
        desc = BufferResourceDescriptor(*words)
        self.assertEqual(desc.as_words(), words)
        self.assertEqual(desc.base_lo, 0x12345678)
        self.assertEqual(desc.base_hi, 0xABCD)
        self.assertEqual(desc.stride, 0x40)
        self.assertEqual(desc.num_records, 0x400)
        self.assertEqual(desc.format_code, (0x3004D123 >> 12) & 0x7F)
        self.assertEqual(desc.oob_select, (0x3004D123 >> 28) & 0x3)


class TestAddressingCalculations(unittest.TestCase):
    def test_soffset_alignment_masking(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0x1000, num_records=1024, oob_select=3)
        for unaligned, expected_masked in [(0, 0), (1, 0), (2, 0), (3, 0), (4, 4), (7, 4), (15, 12), (0xFFFFFFFF, 0xFFFFFFFC)]:
            addr, in_bounds, total = compute_typed_buffer_address(desc, soffset=unaligned, inst_offset=0, voffset=0)
            self.assertEqual(total, expected_masked)
            self.assertEqual(addr, 0x1000 + expected_masked)

    def test_effective_offset_formula(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0x2000, num_records=2048, oob_select=3)
        voffset = 0x80
        soffset = 0x43
        inst_offset = 0x14
        addr, in_bounds, total = compute_typed_buffer_address(desc, soffset=soffset, inst_offset=inst_offset, voffset=voffset)
        self.assertEqual(total, 0x80 + 0x40 + 0x14)
        self.assertEqual(addr, 0x2000 + total)
        self.assertTrue(in_bounds)

    def test_structured_vindex_stride_addressing(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0x5000, stride=32, num_records=16, oob_select=1)
        addr, in_bounds, total = compute_typed_buffer_address(desc, soffset=4, inst_offset=8, voffset=0, vindex=5, element_size=16)
        expected_offset = (4 + 8) + (5 * 32)
        self.assertEqual(total, expected_offset)
        self.assertEqual(addr, 0x5000 + expected_offset)
        self.assertTrue(in_bounds)

    def test_bounds_checking_raw_exact(self) -> None:
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=100, oob_select=3)
        _, in_bounds_edge, _ = compute_typed_buffer_address(desc, inst_offset=96, element_size=4)
        self.assertTrue(in_bounds_edge)
        _, in_bounds_oob, _ = compute_typed_buffer_address(desc, inst_offset=97, element_size=4)
        self.assertFalse(in_bounds_oob)


class TestFormatPackingReference(unittest.TestCase):
    def test_unorm8_packing(self) -> None:
        self.assertEqual(pack_unorm8(0.0), 0)
        self.assertEqual(pack_unorm8(1.0), 255)
        self.assertEqual(pack_unorm8(0.5), 128)
        self.assertEqual(pack_unorm8(-1.0), 0)
        self.assertEqual(pack_unorm8(2.5), 255)
        self.assertEqual(pack_unorm8(float("nan")), 0)

    def test_float16_packing(self) -> None:
        self.assertEqual(pack_float16(0.0), 0x0000)
        self.assertEqual(pack_float16(1.0), 0x3C00)
        self.assertEqual(pack_float16(2.0), 0x4000)
        self.assertEqual(pack_float16(-1.0), 0xBC00)
        self.assertEqual(pack_float16(65504.0), 0x7BFF)

    def test_uint32_packing(self) -> None:
        self.assertEqual(pack_uint32(0), 0)
        self.assertEqual(pack_uint32(0xFFFFFFFF), 0xFFFFFFFF)
        self.assertEqual(pack_uint32(0x12345678), 0x12345678)
        self.assertEqual(pack_uint32(0x100000005), 5)

    def test_float32_packing(self) -> None:
        self.assertEqual(pack_float32(0.0), 0x00000000)
        self.assertEqual(pack_float32(1.0), 0x3F800000)
        self.assertEqual(pack_float32(2.0), 0x40000000)
        self.assertEqual(pack_float32(-1.0), 0xBF800000)

    def test_pack_typed_format_bytes(self) -> None:
        b_u8 = pack_typed_format_bytes((1.0,), GFX10_FMT_8_UNORM)
        self.assertEqual(b_u8, b"\xff")
        b_u8_4 = pack_typed_format_bytes((0.0, 0.5, 1.0, 1.0), GFX10_FMT_8_8_8_8_UNORM)
        self.assertEqual(b_u8_4, bytes([0, 128, 255, 255]))
        b_f16 = pack_typed_format_bytes((1.0, 2.0), GFX10_FMT_16_16_FLOAT)
        self.assertEqual(struct.unpack("<2H", b_f16), (0x3C00, 0x4000))
        b_u32 = pack_typed_format_bytes((0x12345678, 0x9ABCDEF0), GFX10_FMT_32_32_UINT)
        self.assertEqual(struct.unpack("<2I", b_u32), (0x12345678, 0x9ABCDEF0))
        b_f32 = pack_typed_format_bytes((1.0, 2.0, 3.0, 4.0), GFX10_FMT_32_32_32_32_FLOAT)
        self.assertEqual(struct.unpack("<4I", b_f32), (0x3F800000, 0x40000000, 0x40400000, 0x40800000))


class TestTypedStoreReferenceInterpreter(unittest.TestCase):
    def test_unorm8_sub_dword_store_preserves_neighbors(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x200: 0x11223344})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x1000, format_code=GFX10_FMT_8_UNORM)
        success = interp.store_format((1.0,), desc, voffset=0x200, format_code=GFX10_FMT_8_UNORM)
        self.assertTrue(success)
        self.assertEqual(interp.read_dword(0x200), 0x112233FF)

    def test_float16_sub_dword_store_preserves_high_half(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x300: 0xAAAA5555})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x1000, format_code=GFX10_FMT_16_FLOAT)
        success = interp.store_format((1.0,), desc, voffset=0x300, format_code=GFX10_FMT_16_FLOAT)
        self.assertTrue(success)
        self.assertEqual(interp.read_dword(0x300), 0xAAAA3C00)

    def test_xyzw_float32_multi_dword_store(self) -> None:
        interp = TypedBufferReferenceInterpreter()
        desc = BufferResourceDescriptor.from_fields(base_address=0x100, num_records=0x1000, format_code=GFX10_FMT_32_32_32_32_FLOAT)
        success = interp.store_format((1.0, 2.0, 3.0, 4.0), desc, voffset=0x10, format_code=GFX10_FMT_32_32_32_32_FLOAT)
        self.assertTrue(success)
        self.assertEqual(interp.read_dword(0x110), 0x3F800000)
        self.assertEqual(interp.read_dword(0x114), 0x40000000)
        self.assertEqual(interp.read_dword(0x118), 0x40400000)
        self.assertEqual(interp.read_dword(0x11C), 0x40800000)

    def test_out_of_bounds_suppressed_memory_unchanged(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x200: 0xDEADBEEF})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x100, format_code=GFX10_FMT_32_UINT)
        success = interp.store_format((0x12345678,), desc, voffset=0x200, format_code=GFX10_FMT_32_UINT)
        self.assertFalse(success)
        self.assertEqual(interp.read_dword(0x200), 0xDEADBEEF)

    def test_invalid_descriptor_format_suppressed(self) -> None:
        interp = TypedBufferReferenceInterpreter(memory={0x200: 0xDEADBEEF})
        desc = BufferResourceDescriptor.from_fields(base_address=0, num_records=0x1000, format_code=0)
        success = interp.store_format((1.0,), desc, voffset=0x200, format_code=GFX10_FMT_8_UNORM)
        self.assertFalse(success)
        self.assertEqual(interp.read_dword(0x200), 0xDEADBEEF)


class WaveTypedBufferInterpreter:
    """Multi-lane SIMD wave reference interpreter modeling RDNA2 Wave32/Wave64 typed buffer stores."""

    def __init__(self, wave_size: int = 32, memory: dict[int, int] | None = None) -> None:
        self.wave_size = wave_size
        self.memory: dict[int, int] = {} if memory is None else dict(memory)
        self.exec_mask: int = (1 << wave_size) - 1

    def set_exec_mask(self, mask: int) -> None:
        mask_limit = (1 << self.wave_size) - 1
        self.exec_mask = mask & mask_limit

    def read_dword(self, byte_addr: int) -> int:
        aligned = byte_addr & ~3
        return self.memory.get(aligned & MASK64, 0) & MASK32

    def write_bytes(self, byte_addr: int, data: bytes) -> None:
        for offset, byte_val in enumerate(data):
            curr_addr = byte_addr + offset
            aligned = curr_addr & ~3
            shift = (curr_addr & 3) * 8
            old_dword = self.read_dword(aligned)
            cleared = old_dword & ~(0xFF << shift)
            new_dword = cleared | ((byte_val & 0xFF) << shift)
            self.memory[aligned & MASK64] = new_dword & MASK32

    def execute_wave_store(
        self,
        vdatas: list[tuple[float | int, ...] | list[float | int]],
        voffsets: list[int],
        desc: BufferResourceDescriptor,
        soffset: int = 0,
        inst_offset: int = 0,
        vindices: list[int] | None = None,
        format_code: int = GFX10_FMT_32_32_32_32_FLOAT,
    ) -> dict[str, int | list[int]]:
        """Executes typed buffer store across all wave lanes under the current EXEC mask.

        Inactive lanes (EXEC bit == 0) never store to memory.
        Active lanes that are out-of-bounds are silently ignored.
        """
        elem_size = FORMAT_ELEMENT_SIZES.get(format_code, 4)
        active_count = 0
        stored_count = 0
        written_addrs: list[int] = []

        if desc.format_code == 0:
            return {"active": 0, "stored": 0, "written_addrs": []}

        for lane_id in range(self.wave_size):
            is_active = (self.exec_mask & (1 << lane_id)) != 0
            if not is_active:
                continue
            active_count += 1

            voff = voffsets[lane_id] if lane_id < len(voffsets) else 0
            vidx = vindices[lane_id] if vindices is not None and lane_id < len(vindices) else None
            vdata = vdatas[lane_id] if lane_id < len(vdatas) else ()

            addr, in_bounds, _ = compute_typed_buffer_address(
                desc,
                soffset=soffset,
                inst_offset=inst_offset,
                voffset=voff,
                vindex=vidx,
                element_size=elem_size,
            )

            if in_bounds:
                packed = pack_typed_format_bytes(vdata, format_code)
                self.write_bytes(addr, packed)
                stored_count += 1
                written_addrs.append(addr)

        return {"active": active_count, "stored": stored_count, "written_addrs": written_addrs}


class TestWaveBoundsAndSIMDDivergence(unittest.TestCase):
    def test_wave32_uniform_exec_in_bounds(self) -> None:
        wave = WaveTypedBufferInterpreter(wave_size=32)
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            num_records=32 * 16,
            format_code=GFX10_FMT_32_32_32_32_FLOAT,
            oob_select=3,
        )
        voffsets = [i * 16 for i in range(32)]
        vdatas = [(float(i), float(i * 2), float(i * 3), float(i * 4)) for i in range(32)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, format_code=GFX10_FMT_32_32_32_32_FLOAT)
        self.assertEqual(stats["active"], 32)
        self.assertEqual(stats["stored"], 32)
        for i in range(32):
            base_addr = 0x1000 + i * 16
            self.assertEqual(wave.read_dword(base_addr), pack_float32(float(i)))
            self.assertEqual(wave.read_dword(base_addr + 4), pack_float32(float(i * 2)))
            self.assertEqual(wave.read_dword(base_addr + 8), pack_float32(float(i * 3)))
            self.assertEqual(wave.read_dword(base_addr + 12), pack_float32(float(i * 4)))

    def test_wave64_uniform_exec_in_bounds(self) -> None:
        wave = WaveTypedBufferInterpreter(wave_size=64)
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x2000,
            num_records=64 * 16,
            format_code=GFX10_FMT_32_32_32_32_UINT,
            oob_select=3,
        )
        voffsets = [i * 16 for i in range(64)]
        vdatas = [(0x1000 + i, 0x2000 + i, 0x3000 + i, 0x4000 + i) for i in range(64)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, format_code=GFX10_FMT_32_32_32_32_UINT)
        self.assertEqual(stats["active"], 64)
        self.assertEqual(stats["stored"], 64)
        for i in range(64):
            base_addr = 0x2000 + i * 16
            self.assertEqual(wave.read_dword(base_addr), 0x1000 + i)
            self.assertEqual(wave.read_dword(base_addr + 4), 0x2000 + i)
            self.assertEqual(wave.read_dword(base_addr + 8), 0x3000 + i)
            self.assertEqual(wave.read_dword(base_addr + 12), 0x4000 + i)

    def test_divergent_exec_mask_inactive_lanes_suppression(self) -> None:
        memory = {0x1000 + 4 * k: 0xDEADBEEF for k in range(32 * 4)}
        wave = WaveTypedBufferInterpreter(wave_size=32, memory=memory)
        wave.set_exec_mask(0x55555555)  # Even lanes active (0, 2, 4...), odd lanes inactive
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            num_records=32 * 16,
            format_code=GFX10_FMT_32_32_32_32_UINT,
            oob_select=3,
        )
        voffsets = [i * 16 for i in range(32)]
        vdatas = [(0xA000 + i, 0xB000 + i, 0xC000 + i, 0xD000 + i) for i in range(32)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, format_code=GFX10_FMT_32_32_32_32_UINT)
        self.assertEqual(stats["active"], 16)
        self.assertEqual(stats["stored"], 16)

        for i in range(32):
            base_addr = 0x1000 + i * 16
            if i % 2 == 0:
                self.assertEqual(wave.read_dword(base_addr), 0xA000 + i)
                self.assertEqual(wave.read_dword(base_addr + 4), 0xB000 + i)
                self.assertEqual(wave.read_dword(base_addr + 8), 0xC000 + i)
                self.assertEqual(wave.read_dword(base_addr + 12), 0xD000 + i)
            else:
                for dword_idx in range(4):
                    self.assertEqual(wave.read_dword(base_addr + 4 * dword_idx), 0xDEADBEEF)

    def test_half_wave_divergence(self) -> None:
        memory = {0x4000 + 4 * k: 0xCAFEBABE for k in range(64 * 4)}
        wave = WaveTypedBufferInterpreter(wave_size=64, memory=memory)
        wave.set_exec_mask(0x00000000FFFFFFFF)  # Lower 32 lanes active, upper 32 inactive
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x4000,
            num_records=64 * 16,
            format_code=GFX10_FMT_32_32_32_32_UINT,
            oob_select=3,
        )
        voffsets = [i * 16 for i in range(64)]
        vdatas = [(0x1111 * (i + 1), 0x2222 * (i + 1), 0x3333 * (i + 1), 0x4444 * (i + 1)) for i in range(64)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, format_code=GFX10_FMT_32_32_32_32_UINT)
        self.assertEqual(stats["active"], 32)
        self.assertEqual(stats["stored"], 32)

        for i in range(32):
            base_addr = 0x4000 + i * 16
            self.assertEqual(wave.read_dword(base_addr), 0x1111 * (i + 1))
        for i in range(32, 64):
            base_addr = 0x4000 + i * 16
            for dword_idx in range(4):
                self.assertEqual(wave.read_dword(base_addr + 4 * dword_idx), 0xCAFEBABE)

    def test_single_lane_divergence(self) -> None:
        memory = {0x3000 + 4 * k: 0x12345678 for k in range(32 * 4)}
        wave = WaveTypedBufferInterpreter(wave_size=32, memory=memory)
        target_lane = 13
        wave.set_exec_mask(1 << target_lane)
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x3000,
            num_records=32 * 16,
            format_code=GFX10_FMT_32_32_32_32_UINT,
            oob_select=3,
        )
        voffsets = [i * 16 for i in range(32)]
        vdatas = [(0x7777, 0x8888, 0x9999, 0xAAAA) for _ in range(32)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, format_code=GFX10_FMT_32_32_32_32_UINT)
        self.assertEqual(stats["active"], 1)
        self.assertEqual(stats["stored"], 1)

        for i in range(32):
            base_addr = 0x3000 + i * 16
            if i == target_lane:
                self.assertEqual(wave.read_dword(base_addr), 0x7777)
                self.assertEqual(wave.read_dword(base_addr + 4), 0x8888)
                self.assertEqual(wave.read_dword(base_addr + 8), 0x9999)
                self.assertEqual(wave.read_dword(base_addr + 12), 0xAAAA)
            else:
                for dword_idx in range(4):
                    self.assertEqual(wave.read_dword(base_addr + 4 * dword_idx), 0x12345678)

    def test_simd_lane_oob_suppression_boundary(self) -> None:
        memory = {0x1000 + 4 * k: 0xBAADF00D for k in range(32 * 4)}
        wave = WaveTypedBufferInterpreter(wave_size=32, memory=memory)
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            num_records=16 * 16,  # Exactly 16 records of 16 bytes = 256 bytes
            format_code=GFX10_FMT_32_32_32_32_UINT,
            oob_select=3,
        )
        voffsets = [i * 16 for i in range(32)]
        vdatas = [(0x5000 + i, 0x6000 + i, 0x7000 + i, 0x8000 + i) for i in range(32)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, format_code=GFX10_FMT_32_32_32_32_UINT)
        self.assertEqual(stats["active"], 32)
        self.assertEqual(stats["stored"], 16)

        for i in range(16):
            base_addr = 0x1000 + i * 16
            self.assertEqual(wave.read_dword(base_addr), 0x5000 + i)
        for i in range(16, 32):
            base_addr = 0x1000 + i * 16
            for dword_idx in range(4):
                self.assertEqual(wave.read_dword(base_addr + 4 * dword_idx), 0xBAADF00D)

    def test_xyzw_element_size_straddle_boundary(self) -> None:
        memory = {0x5000 + 4 * k: 0x55555555 for k in range(32)}
        wave = WaveTypedBufferInterpreter(wave_size=2, memory=memory)
        # Limit is 100 bytes.
        # Lane 0: offset 80 -> 80 + 16 = 96 <= 100 (in-bounds)
        # Lane 1: offset 88 -> 88 + 16 = 104 > 100 (straddles boundary -> out of bounds)
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x5000,
            num_records=100,
            format_code=GFX10_FMT_32_32_32_32_UINT,
            oob_select=3,
        )
        voffsets = [80, 88]
        vdatas = [(1, 2, 3, 4), (5, 6, 7, 8)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, format_code=GFX10_FMT_32_32_32_32_UINT)
        self.assertEqual(stats["active"], 2)
        self.assertEqual(stats["stored"], 1)

        self.assertEqual(wave.read_dword(0x5000 + 80), 1)
        self.assertEqual(wave.read_dword(0x5000 + 84), 2)
        self.assertEqual(wave.read_dword(0x5000 + 88), 3)
        self.assertEqual(wave.read_dword(0x5000 + 92), 4)
        # Memory past 96 was never modified by lane 1
        self.assertEqual(wave.read_dword(0x5000 + 96), 0x55555555)

    def test_channel_variants_multi_lane(self) -> None:
        # Test x (1-comp), xy (2-comp), xyz (3-comp), xyzw (4-comp)
        wave = WaveTypedBufferInterpreter(wave_size=4)
        # x variant (4 bytes per element)
        desc_x = BufferResourceDescriptor.from_fields(0x1000, num_records=64, format_code=GFX10_FMT_32_UINT, oob_select=3)
        stats_x = wave.execute_wave_store([(10,), (20,), (30,), (40,)], [0, 4, 8, 12], desc_x, format_code=GFX10_FMT_32_UINT)
        self.assertEqual(stats_x["stored"], 4)
        for i, val in enumerate([10, 20, 30, 40]):
            self.assertEqual(wave.read_dword(0x1000 + 4 * i), val)

        # xy variant (8 bytes per element)
        desc_xy = BufferResourceDescriptor.from_fields(0x2000, num_records=64, format_code=GFX10_FMT_32_32_UINT, oob_select=3)
        stats_xy = wave.execute_wave_store([(1, 2), (3, 4), (5, 6), (7, 8)], [0, 8, 16, 24], desc_xy, format_code=GFX10_FMT_32_32_UINT)
        self.assertEqual(stats_xy["stored"], 4)
        self.assertEqual(wave.read_dword(0x2000 + 16), 5)
        self.assertEqual(wave.read_dword(0x2000 + 20), 6)

        # xyz variant (12 bytes per element)
        wave_xyz = WaveTypedBufferInterpreter(wave_size=2)
        desc_xyz = BufferResourceDescriptor.from_fields(0x3000, num_records=64, format_code=GFX10_FMT_32_32_32_UINT, oob_select=3)
        stats_xyz = wave_xyz.execute_wave_store([(1, 2, 3), (4, 5, 6)], [0, 12], desc_xyz, format_code=GFX10_FMT_32_32_32_UINT)
        self.assertEqual(stats_xyz["stored"], 2)
        self.assertEqual(wave_xyz.read_dword(0x3000 + 12), 4)
        self.assertEqual(wave_xyz.read_dword(0x3000 + 16), 5)
        self.assertEqual(wave_xyz.read_dword(0x3000 + 20), 6)

    def test_structured_stride_multi_lane(self) -> None:
        wave = WaveTypedBufferInterpreter(wave_size=8)
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x8000,
            stride=32,
            num_records=5,  # Only first 5 records valid
            format_code=GFX10_FMT_32_32_UINT,
            oob_select=1,
        )
        vindices = list(range(8))
        voffsets = [0] * 8
        vdatas = [(100 + i, 200 + i) for i in range(8)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, vindices=vindices, format_code=GFX10_FMT_32_32_UINT)
        self.assertEqual(stats["active"], 8)
        self.assertEqual(stats["stored"], 5)
        for i in range(5):
            self.assertEqual(wave.read_dword(0x8000 + i * 32), 100 + i)
            self.assertEqual(wave.read_dword(0x8000 + i * 32 + 4), 200 + i)

    def test_soffset_inst_offset_multi_lane(self) -> None:
        wave = WaveTypedBufferInterpreter(wave_size=4)
        desc = BufferResourceDescriptor.from_fields(base_address=0x6000, num_records=256, format_code=GFX10_FMT_32_UINT, oob_select=3)
        soffset = 0x23  # Will be masked to 0x20
        inst_offset = 0x10
        voffsets = [0, 4, 8, 12]
        vdatas = [(100,), (200,), (300,), (400,)]
        stats = wave.execute_wave_store(vdatas, voffsets, desc, soffset=soffset, inst_offset=inst_offset, format_code=GFX10_FMT_32_UINT)
        self.assertEqual(stats["stored"], 4)
        expected_base = 0x6000 + 0x20 + 0x10
        for i, val in enumerate([100, 200, 300, 400]):
            self.assertEqual(wave.read_dword(expected_base + 4 * i), val)


class TestSpirvEmission(unittest.TestCase):
    def test_tbuffer_store_emits_storage_buffer_ops_not_image_write(self) -> None:
        spv = support.translate_lines(
            "cs_tbuf_spv_emission",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v21, 2.0",
                "v_mov_b32_e32 v22, 3.0",
                "v_mov_b32_e32 v23, 4.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_xyzw v[20:23], v5, s[0:3], 0 format:[BUF_FMT_32_32_32_32_FLOAT] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        self.assertIn("OpConvertUToPtr", spv)
        self.assertIn("OpStore", spv)
        self.assertIn("OpBranchConditional", spv)
        self.assertIn("OpSelectionMerge", spv)
        self.assertNotIn("OpImageWrite", spv)

    def test_image_store_contrast_emits_op_image_write(self) -> None:
        g = rdna2spirv.Gen("cs")
        ins = shader_ir.parse_line("image_store v[1:4], v[10:11], s[0:7] dmask:0xf dim:SQ_RSRC_IMG_2D")
        g.image_store(ins["args"], ins["mods"], "dim:SQ_RSRC_IMG_2D")
        has_image_write = any("OpImageWrite" in ln for ln in g.body)
        self.assertTrue(has_image_write)

    def test_sub_dword_unorm8_emits_atomic_rmw(self) -> None:
        spv = support.translate_lines(
            "cs_tbuf_atomic_rmw",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_x v20, v5, s[0:3], 0 format:[BUF_FMT_8_UNORM] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        self.assertIn("OpAtomicAnd", spv)
        self.assertIn("OpAtomicOr", spv)


class TestSpirvNegativeControls(unittest.TestCase):
    def test_unsupported_format_raises_exception(self) -> None:
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines(
                "cs_tbuf_unsupported_fmt",
                [
                    "v_mov_b32_e32 v20, 1.0",
                    "v_mov_b32_e32 v5, 0x200",
                    "tbuffer_store_format_x v20, v5, s[0:3], 0 format:[BUF_FMT_INVALID] offen",
                    "s_endpgm",
                ],
                fssm=CS_FSSM,
            )

    def test_unsupported_modifier_lds_raises_exception(self) -> None:
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines(
                "cs_tbuf_unsupported_mod",
                [
                    "v_mov_b32_e32 v20, 1.0",
                    "v_mov_b32_e32 v5, 0x200",
                    "tbuffer_store_format_x v20, v5, s[0:3], 0 lds format:[BUF_FMT_8_UNORM] offen",
                    "s_endpgm",
                ],
                fssm=CS_FSSM,
            )

    def test_invalid_vsharp_sgpr_count_raises_exception(self) -> None:
        with self.assertRaises(rdna2spirv.Unsupported):
            support.translate_lines(
                "cs_tbuf_invalid_sgpr",
                [
                    "v_mov_b32_e32 v20, 1.0",
                    "v_mov_b32_e32 v5, 0x200",
                    "tbuffer_store_format_x v20, v5, s[0:1], 0 format:[BUF_FMT_8_UNORM] offen",
                    "s_endpgm",
                ],
                fssm=CS_FSSM,
            )


class TestSpirvEndToEndExecution(unittest.TestCase):
    def test_e2e_tbuffer_store_format_x_unorm8(self) -> None:
        spv = support.translate_lines(
            "cs_e2e_tbuf_u8",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_x v20, v5, s[0:3], 0 format:[BUF_FMT_8_UNORM] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        memory = {0x200: 0x11223344}
        run_cs(spv, memory)
        self.assertEqual(memory[0x200], 0x112233FF)

    def test_e2e_tbuffer_store_format_xy_f16(self) -> None:
        spv = support.translate_lines(
            "cs_e2e_tbuf_f16",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v21, 2.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_xy v[20:21], v5, s[0:3], 0 format:[BUF_FMT_16_16_FLOAT] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        memory = {0x200: 0xDEADBEEF}
        run_cs(spv, memory)
        self.assertEqual(memory[0x200], 0x40003C00)

    def test_e2e_tbuffer_store_format_xy_u32(self) -> None:
        spv = support.translate_lines(
            "cs_e2e_tbuf_u32",
            [
                "v_mov_b32_e32 v20, 0x12345678",
                "v_mov_b32_e32 v21, 0x9ABCDEF0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_xy v[20:21], v5, s[0:3], 0 format:[BUF_FMT_32_32_UINT] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        memory: dict[int, int] = {}
        run_cs(spv, memory)
        self.assertEqual(memory[0x200], 0x12345678)
        self.assertEqual(memory[0x204], 0x9ABCDEF0)

    def test_e2e_tbuffer_store_format_xyzw_f32(self) -> None:
        spv = support.translate_lines(
            "cs_e2e_tbuf_f32",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v21, 2.0",
                "v_mov_b32_e32 v22, 3.0",
                "v_mov_b32_e32 v23, 4.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_xyzw v[20:23], v5, s[0:3], 0 format:[BUF_FMT_32_32_32_32_FLOAT] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        memory: dict[int, int] = {}
        run_cs(spv, memory)
        self.assertEqual(memory[0x200], 0x3F800000)
        self.assertEqual(memory[0x204], 0x40000000)
        self.assertEqual(memory[0x208], 0x40400000)
        self.assertEqual(memory[0x20C], 0x40800000)

    def test_e2e_oob_zero_clamp_beyond_num_records(self) -> None:
        spv = support.translate_lines(
            "cs_e2e_tbuf_oob_clamp",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_x v20, v5, s[0:3], 0 format:[BUF_FMT_8_UNORM] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        memory = {0x200: 0xCAFEBABE}
        # num_records is 0x100 (256 bytes), store target is 0x200 (512 bytes) -> OOB
        oob_vsharp = [0, 0, 0x100, 0x31016FAC]
        run_cs(spv, memory, push_constants=oob_vsharp + [0] * 28)
        self.assertEqual(memory[0x200], 0xCAFEBABE)

    def test_e2e_oob_zero_clamp_element_straddle(self) -> None:
        spv = support.translate_lines(
            "cs_e2e_tbuf_straddle_clamp",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v21, 2.0",
                "v_mov_b32_e32 v22, 3.0",
                "v_mov_b32_e32 v23, 4.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_xyzw v[20:23], v5, s[0:3], 0 format:[BUF_FMT_32_32_32_32_FLOAT] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        memory = {
            0x200: 0xDEADBEEF,
            0x204: 0xDEADBEEF,
            0x208: 0xDEADBEEF,
            0x20C: 0xDEADBEEF,
        }
        # num_records is 0x204 (516 bytes). 16-byte store at 0x200 needs bytes up to 0x210 -> straddles end!
        straddle_vsharp = [0, 0, 0x204, 0x31016FAC]
        run_cs(spv, memory, push_constants=straddle_vsharp + [0] * 28)
        for dword_addr in (0x200, 0x204, 0x208, 0x20C):
            self.assertEqual(memory[dword_addr], 0xDEADBEEF)

    def test_e2e_invalid_vsharp_format_suppressed(self) -> None:
        spv = support.translate_lines(
            "cs_e2e_tbuf_invalid_vsharp_fmt",
            [
                "v_mov_b32_e32 v20, 1.0",
                "v_mov_b32_e32 v5, 0x200",
                "tbuffer_store_format_x v20, v5, s[0:3], 0 format:[BUF_FMT_8_UNORM] offen",
                "s_endpgm",
            ],
            fssm=CS_FSSM,
        )
        memory = {0x200: 0x55555555}
        # V# format field (bits 18:12) set to 0 (GFX10_FORMAT_INVALID)
        invalid_fmt_vsharp = [0, 0, 0xFFFFFFFF, 0x31000FAC]
        run_cs(spv, memory, push_constants=invalid_fmt_vsharp + [0] * 28)
        self.assertEqual(memory[0x200], 0x55555555)


if __name__ == "__main__":
    unittest.main()
