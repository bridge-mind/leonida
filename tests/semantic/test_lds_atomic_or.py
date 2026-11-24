"""Semantic interpreter tests for LDS atomic bitwise OR operations (ds_or_b32, ds_or_rtn_b32, ds_or_b64, ds_or_rtn_b64).

Standard library only; no GPU, no LLVM, no external tools, no game files.

Run directly:
    python -I tests/semantic/test_lds_atomic_or.py
Or via unittest discovery:
    python -m unittest discover -s tests/semantic -v

Background & Specification Rules:
  * AMD RDNA2 / GFX10 Local Data Share (LDS) Architecture:
    - LDS is an on-chip, software-managed scratchpad memory shared across
      all work-items (threads/lanes) in a workgroup.
    - Single 32-bit Atomic OR:
        - `ds_or_b32 vaddr, vdata [offset:N]`:
            Non-returning form:
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              LDS[dword_idx] = (LDS[dword_idx] | vdata) & 0xFFFFFFFF
            The pre-atomic memory contents are discarded.
        - `ds_or_rtn_b32 vdst, vaddr, vdata [offset:N]`:
            Returning form:
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              old_val = LDS[dword_idx]
              LDS[dword_idx] = (old_val | vdata) & 0xFFFFFFFF
              vdst = old_val
            Preserves the old value before the OR into destination VGPR vdst.
    - 64-bit Atomic OR:
        - `ds_or_b64 vaddr, v[data_lo:data_hi] [offset:N]`:
            Non-returning 64-bit bitwise OR on two consecutive 32-bit dwords (low dword first):
              eff_byte = (vaddr + offset) & 0xFFFFFFFF
              dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
              LDS[dword_idx]   |= vdata_lo
              LDS[dword_idx+1] |= vdata_hi
        - `ds_or_rtn_b64 v[dst_lo:dst_hi], vaddr, v[data_lo:data_hi] [offset:N]`:
            Returning form: writes prior 64 bits into destination VGPR pair v[dst_lo:dst_hi].
    - Active lane masking:
        - LDS instructions are vector operations governed by the execution mask (EXEC).
        - Inactive lanes do not mutate LDS storage and leave destination VGPRs unmodified.
    - Hardware wrapping & alignment:
        - Byte addresses are shifted right by 2, dropping unaligned low bits [1:0].
        - The resulting dword index is masked with (lds_dwords - 1), wrapping out-of-bounds
          offsets within the allocated power-of-two LDS capacity.
    - Mathematical properties of bitwise OR:
        - Identity: x | 0 == x.
        - Dominant / Saturation element: x | 0xFFFFFFFF == 0xFFFFFFFF.
        - Idempotence: x | x == x.
        - Commutative & Associative: (x | y) == (y | x), (x | y) | z == x | (y | z).
        - Monotonicity: (x | y) & x == x (bits only transition 0 -> 1, never 1 -> 0).
        - Bitmask union: fundamental for flag aggregation, multi-lane barrier signals,
          and sparse occupancy tracking.
"""
from __future__ import annotations

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
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF
DEFAULT_LDS_DWORDS = 16384  # 64 KiB default capacity in dwords


class UnsupportedConstruct(NotImplementedError):
    """The module uses an unmodelled SPIR-V opcode, type, or storage class."""


class WorkgroupBoundsError(ValueError):
    """Workgroup dimension or memory bounds violation."""


def reference_atomic_or_b32(current_val: int, or_mask: int) -> tuple[int, int]:
    """Reference 32-bit atomic OR: returns (old_val, updated_mem_val)."""
    old = current_val & MASK32
    new = (old | or_mask) & MASK32
    return old, new


def reference_atomic_or_b64(current_val: int, or_mask: int) -> tuple[int, int]:
    """Reference 64-bit atomic OR: returns (old_val, updated_mem_val)."""
    old = current_val & MASK64
    new = (old | or_mask) & MASK64
    return old, new


def split_u64_to_u32(val: int) -> tuple[int, int]:
    """Splits a 64-bit unsigned integer into (low_u32, high_u32)."""
    val = val & MASK64
    return val & MASK32, (val >> 32) & MASK32


def combine_u32_to_u64(lo: int, hi: int) -> int:
    """Combines two 32-bit unsigned integers (low_u32, high_u32) into a 64-bit unsigned integer."""
    return ((hi & MASK32) << 32) | (lo & MASK32)


def calc_lds_byte_offset(
    vaddr: int,
    byte_offset: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
) -> tuple[int, int, int]:
    """Reference ISA calculation for single LDS address: (effective_byte, dword_idx, byte_in_dword)."""
    vaddr = vaddr & MASK32
    byte_offset = byte_offset & 0xFFFF  # 16-bit unsigned offset in RDNA2 DS instructions
    eff_byte = (vaddr + byte_offset) & MASK32
    dword_idx = (eff_byte >> 2) & (lds_dwords - 1)
    byte_in_dword = eff_byte & 3
    return eff_byte, dword_idx, byte_in_dword


def calc_lds_dual_offsets(
    vaddr: int,
    offset0: int = 0,
    offset1: int = 0,
    lds_dwords: int = DEFAULT_LDS_DWORDS,
    st64: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Reference ISA calculation for dual LDS addresses.

    Returns:
        ((eff_byte0, dword_idx0), (eff_byte1, dword_idx1))
    """
    vaddr = vaddr & MASK32
    unit = 256 if st64 else 4
    off0_bytes = (offset0 & 0xFF) * unit
    off1_bytes = (offset1 & 0xFF) * unit
    eff_byte0 = (vaddr + off0_bytes) & MASK32
    eff_byte1 = (vaddr + off1_bytes) & MASK32
    dword_idx0 = (eff_byte0 >> 2) & (lds_dwords - 1)
    dword_idx1 = (eff_byte1 >> 2) & (lds_dwords - 1)
    return (eff_byte0, dword_idx0), (eff_byte1, dword_idx1)


class TestAtomicOrReferenceMath(unittest.TestCase):
    """Verifies reference bitwise OR properties, algebraic laws, and edge cases."""

    def test_identity_zero_mask_b32(self):
        cases = [0, 1, 0x12345678, 0x80000000, 0x7FFFFFFF, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                old, new = reference_atomic_or_b32(val, 0)
                self.assertEqual(old, val)
                self.assertEqual(new, val)

    def test_saturation_all_ones_b32(self):
        cases = [0, 1, 0x55555555, 0xAAAAAAAA, 0x80000000, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                old, new = reference_atomic_or_b32(val, MASK32)
                self.assertEqual(old, val)
                self.assertEqual(new, MASK32)

    def test_idempotence_b32(self):
        cases = [0, 0x10203040, 0x55AA55AA, 0xFFFFFFFF]
        for val in cases:
            with self.subTest(val=hex(val)):
                old, new = reference_atomic_or_b32(val, val)
                self.assertEqual(old, val)
                self.assertEqual(new, val)

    def test_commutativity_and_associativity_b32(self):
        a, b = 0x3F2A1C8E, 0x9D4E7B01
        _, ab = reference_atomic_or_b32(a, b)
        _, ba = reference_atomic_or_b32(b, a)
        self.assertEqual(ab, ba)

        c = 0x55AA33CC
        _, step_ab = reference_atomic_or_b32(a, b)
        _, ab_c = reference_atomic_or_b32(step_ab, c)

        _, step_bc = reference_atomic_or_b32(b, c)
        _, a_bc = reference_atomic_or_b32(a, step_bc)
        self.assertEqual(ab_c, a_bc)

    def test_monotonicity_b32(self):
        # Bitwise OR can only set bits from 0 to 1, never clears bits
        cases = [
            (0, 0x00000001),
            (0x10000000, 0x00000001),
            (0x0F0F0F0F, 0xF0F0F0F0),
            (0x7FFFFFFF, 0x80000000),
        ]
        for initial, mask in cases:
            with self.subTest(initial=hex(initial), mask=hex(mask)):
                old, new = reference_atomic_or_b32(initial, mask)
                self.assertEqual(old, initial)
                # All bits of initial must still be present in new
                self.assertEqual(new & initial, initial)
                # All bits of mask must be present in new
                self.assertEqual(new & mask, mask)
                self.assertGreaterEqual(bin(new).count("1"), bin(initial).count("1"))

    def test_disjoint_bit_patterns_b32(self):
        # Alternating bit patterns
        old, new = reference_atomic_or_b32(0x55555555, 0xAAAAAAAA)
        self.assertEqual(old, 0x55555555)
        self.assertEqual(new, MASK32)

        # Upper and lower half-words
        old, new = reference_atomic_or_b32(0xFFFF0000, 0x0000FFFF)
        self.assertEqual(old, 0xFFFF0000)
        self.assertEqual(new, MASK32)

        # Nibble interleaving
        old, new = reference_atomic_or_b32(0x0F0F0F0F, 0xF0F0F0F0)
        self.assertEqual(old, 0x0F0F0F0F)
        self.assertEqual(new, MASK32)

    def test_sequential_bit_accumulation_b32(self):
        # Accumulate 32 individual bit flags across 32 sequential atomic operations
        current = 0
        for bit in range(32):
            flag = 1 << bit
            old, current = reference_atomic_or_b32(current, flag)
            self.assertEqual(old & flag, 0)
            self.assertEqual(current & flag, flag)
        self.assertEqual(current, MASK32)

    def test_sparse_bit_combinations_b32(self):
        # Non-contiguous bits
        val = 0
        steps = [1 << 3, 1 << 7, 1 << 15, 1 << 31]
        expected = (1 << 3) | (1 << 7) | (1 << 15) | (1 << 31)
        for s in steps:
            _, val = reference_atomic_or_b32(val, s)
        self.assertEqual(val, expected)

    def test_reference_atomic_or_b64_properties(self):
        init = 0x0123456789ABCDEF
        mask = 0xFEDCBA9876543210
        old, mem = reference_atomic_or_b64(init, mask)
        self.assertEqual(old, init)
        self.assertEqual(mem, init | mask)

        # 64-bit zero identity
        old_id, mem_id = reference_atomic_or_b64(init, 0)
        self.assertEqual(old_id, init)
        self.assertEqual(mem_id, init)

        # 64-bit all-ones saturation
        old_sat, mem_sat = reference_atomic_or_b64(init, MASK64)
        self.assertEqual(old_sat, init)
        self.assertEqual(mem_sat, MASK64)

        # 64-bit idempotence
        old_idem, mem_idem = reference_atomic_or_b64(init, init)
        self.assertEqual(old_idem, init)
        self.assertEqual(mem_idem, init)

    def test_reference_atomic_or_b64_cross_dword_patterns(self):
        # Low dword only mutated
        init = 0xAAAAAAAA_00000000
        mask = 0x00000000_FFFFFFFF
        _, res = reference_atomic_or_b64(init, mask)
        self.assertEqual(res, 0xAAAAAAAA_FFFFFFFF)

        # High dword only mutated
        init = 0x00000000_55555555
        mask = 0xFFFFFFFF_00000000
        _, res = reference_atomic_or_b64(init, mask)
        self.assertEqual(res, 0xFFFFFFFF_55555555)

        # Cross dword boundaries
        lo_word = 0x00000000_80000000  # bit 31
        hi_word = 0x00000001_00000000  # bit 32
        _, res = reference_atomic_or_b64(lo_word, hi_word)
        self.assertEqual(res, 0x00000001_80000000)

    def test_split_and_combine_u64_u32(self):
        v64 = 0xFEDCBA9876543210
        lo, hi = split_u64_to_u32(v64)
        self.assertEqual(lo, 0x76543210)
        self.assertEqual(hi, 0xFEDCBA98)
        self.assertEqual(combine_u32_to_u64(lo, hi), v64)


class TestLdsAddressingAndOffsets(unittest.TestCase):
    """Verifies single and dual LDS byte offsets, dword truncation, st64 strides, and wrapping."""

    def test_aligned_dword_addressing(self):
        lds_dwords = 256
        for dword in (0, 1, 2, 16, 64, 127, 255):
            byte_addr = dword * 4
            eff, dword_idx, byte_pos = calc_lds_byte_offset(byte_addr, 0, lds_dwords)
            self.assertEqual(eff, byte_addr)
            self.assertEqual(dword_idx, dword)
            self.assertEqual(byte_pos, 0)

    def test_unaligned_byte_truncation(self):
        lds_dwords = 128
        base_aligned = 64  # dword 16
        for b in (0, 1, 2, 3):
            eff, dword_idx, byte_pos = calc_lds_byte_offset(base_aligned, b, lds_dwords)
            self.assertEqual(eff, base_aligned + b)
            self.assertEqual(dword_idx, 16)
            self.assertEqual(byte_pos, b)

    def test_offset_16bit_masking(self):
        lds_dwords = 512
        # RDNA2 DS instruction offset field is 16-bit unsigned (0x0000 - 0xFFFF)
        eff, dword_idx, byte_pos = calc_lds_byte_offset(0, 0x10008, lds_dwords)
        self.assertEqual(eff, 8)
        self.assertEqual(dword_idx, 2)
        self.assertEqual(byte_pos, 0)

    def test_dual_offset_unit_strides(self):
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 3, 10, lds_dwords=1024, st64=False)
        self.assertEqual(eff0, 12)
        self.assertEqual(idx0, 3)
        self.assertEqual(eff1, 40)
        self.assertEqual(idx1, 10)

    def test_dual_offset_st64_stride(self):
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 2, 5, lds_dwords=1024, st64=True)
        self.assertEqual(eff0, 512)
        self.assertEqual(idx0, 128)
        self.assertEqual(eff1, 1280)
        self.assertEqual(idx1, 320)

    def test_boundary_power_of_two_wrapping(self):
        lds_dwords = 64  # 256 bytes total
        # Byte address 256 wraps to dword 0
        _, idx0, _ = calc_lds_byte_offset(256, 0, lds_dwords)
        self.assertEqual(idx0, 0)
        # Byte address 260 wraps to dword 1
        _, idx1, _ = calc_lds_byte_offset(256, 4, lds_dwords)
        self.assertEqual(idx1, 1)
        # 32-bit unsigned overflow: 0xFFFFFFFC + 4 = 0x00000000 -> dword 0
        _, idx_wrap, _ = calc_lds_byte_offset(0xFFFFFFFC, 4, lds_dwords)
        self.assertEqual(idx_wrap, 0)


_IGNORED = frozenset({
    "OpCapability", "OpExtInstImport", "OpMemoryModel", "OpEntryPoint", "OpExecutionMode",
    "OpDecorate", "OpMemberDecorate", "OpTypeVoid", "OpTypeFunction", "OpLabel",
    "OpReturn", "OpFunctionEnd", "OpName", "OpSource", "OpSelectionMerge",
})

_LINE = re.compile(r"^(?:(%\S+)\s*=\s*)?(Op\w+)\s*(.*)$")


def translate_cs_listing(body: list[str]) -> str:
    """Translates a compute shader listing using rdna2spirv without external files."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        name = "cs_lds_or"
        with open(os.path.join(tmp, "disasm", name + ".s"), "w", encoding="utf-8") as f:
            f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body) + "\ts_endpgm\n")
        stub_dxbc = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(stub_dxbc)
        # FSSM header with 4 KiB LDS (0x40088 in RSRC2 at 0x138)
        fssm = bytearray(0x148)
        struct.pack_into("<4I", fssm, 0x138, 0x40088, 32, 1, 1)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(fssm)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(os.path.join(tmp, "disasm", name + ".s"))
        return spv


class LdsSemanticInterpreter:
    """Deterministic multi-lane SIMD semantic interpreter for RDNA2 LDS atomic instructions.

    Models LDS shared memory, per-lane VGPR register files, and execution masks.
    """

    def __init__(self, lds_dwords: int = DEFAULT_LDS_DWORDS, num_vgprs: int = 256, num_lanes: int = 64):
        self.lds_dwords = lds_dwords
        self.lds = [0] * lds_dwords
        self.num_vgprs = num_vgprs
        self.num_lanes = num_lanes
        self.vgprs = [[0] * num_vgprs for _ in range(num_lanes)]
        self.active_lanes = [True] * num_lanes

    def set_vgpr(self, reg_idx: int, value: int, lane: int = 0) -> None:
        self.vgprs[lane][reg_idx] = value & MASK32

    def get_vgpr(self, reg_idx: int, lane: int = 0) -> int:
        return self.vgprs[lane][reg_idx]

    def set_lds_dword(self, dword_idx: int, value: int) -> None:
        self.lds[dword_idx & (self.lds_dwords - 1)] = value & MASK32

    def get_lds_dword(self, dword_idx: int) -> int:
        return self.lds[dword_idx & (self.lds_dwords - 1)]

    def set_lane_active(self, lane: int, active: bool) -> None:
        self.active_lanes[lane] = bool(active)

    def set_exec_mask(self, mask: int) -> None:
        """Sets active lanes based on a 64-bit integer execution mask."""
        for lane in range(self.num_lanes):
            self.active_lanes[lane] = bool((mask >> lane) & 1)

    def ds_write_b32(self, vaddr_reg: int, vdata_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.lds[dword_idx] = vdata & MASK32
        return dword_idx

    def ds_read_b32(self, vdst_reg: int, vaddr_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.vgprs[lane][vdst_reg] = self.lds[dword_idx]
        return dword_idx

    def ds_or_b32(self, vaddr_reg: int, vdata_reg: int, offset: int = 0, lane: int = 0) -> int | None:
        """Non-returning 32-bit atomic OR: updates LDS[target], returns dword_idx or None."""
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        self.lds[dword_idx] = (self.lds[dword_idx] | vdata) & MASK32
        return dword_idx

    def ds_or_rtn_b32(
        self,
        vdst_reg: int,
        vaddr_reg: int,
        vdata_reg: int,
        offset: int = 0,
        lane: int = 0,
    ) -> tuple[int, int] | None:
        """Returning 32-bit atomic OR: stores prior value into vdst and updates LDS[target].

        Returns (dword_idx, old_val) if lane active, or None if inactive.
        Handles destination register aliasing: reads vaddr and vdata before overwriting vdst.
        """
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata = self.vgprs[lane][vdata_reg]
        _, dword_idx, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        old_val = self.lds[dword_idx]
        self.lds[dword_idx] = (old_val | vdata) & MASK32
        self.vgprs[lane][vdst_reg] = old_val
        return dword_idx, old_val

    def set_vgpr_u64(self, reg_idx: int, value: int, lane: int = 0) -> None:
        lo, hi = split_u64_to_u32(value)
        self.set_vgpr(reg_idx, lo, lane=lane)
        self.set_vgpr(reg_idx + 1, hi, lane=lane)

    def get_vgpr_u64(self, reg_idx: int, lane: int = 0) -> int:
        lo = self.get_vgpr(reg_idx, lane=lane)
        hi = self.get_vgpr(reg_idx + 1, lane=lane)
        return combine_u32_to_u64(lo, hi)

    def set_lds_u64(self, dword_idx: int, value: int) -> None:
        lo, hi = split_u64_to_u32(value)
        self.set_lds_dword(dword_idx, lo)
        self.set_lds_dword((dword_idx + 1) & (self.lds_dwords - 1), hi)

    def get_lds_u64(self, dword_idx: int) -> int:
        lo = self.get_lds_dword(dword_idx)
        hi = self.get_lds_dword((dword_idx + 1) & (self.lds_dwords - 1))
        return combine_u32_to_u64(lo, hi)

    def ds_write_b64(self, vaddr_reg: int, vdata_lo_reg: int, offset: int = 0, lane: int = 0) -> tuple[int, int] | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        self.lds[dword_idx0] = self.vgprs[lane][vdata_lo_reg] & MASK32
        self.lds[dword_idx1] = self.vgprs[lane][vdata_lo_reg + 1] & MASK32
        return dword_idx0, dword_idx1

    def ds_read_b64(self, vdst_lo_reg: int, vaddr_reg: int, offset: int = 0, lane: int = 0) -> tuple[int, int] | None:
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        val_lo = self.lds[dword_idx0]
        val_hi = self.lds[dword_idx1]
        self.vgprs[lane][vdst_lo_reg] = val_lo
        self.vgprs[lane][vdst_lo_reg + 1] = val_hi
        return dword_idx0, dword_idx1

    def ds_or_b64(self, vaddr_reg: int, vdata_lo_reg: int, offset: int = 0, lane: int = 0) -> tuple[int, int] | None:
        """Non-returning 64-bit atomic OR: updates LDS[idx] and LDS[idx+1], returns (idx0, idx1) or None."""
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata_lo = self.vgprs[lane][vdata_lo_reg]
        vdata_hi = self.vgprs[lane][vdata_lo_reg + 1]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        self.lds[dword_idx0] = (self.lds[dword_idx0] | vdata_lo) & MASK32
        self.lds[dword_idx1] = (self.lds[dword_idx1] | vdata_hi) & MASK32
        return dword_idx0, dword_idx1

    def ds_or_rtn_b64(
        self,
        vdst_lo_reg: int,
        vaddr_reg: int,
        vdata_lo_reg: int,
        offset: int = 0,
        lane: int = 0,
    ) -> tuple[int, int] | None:
        """Returning 64-bit atomic OR: stores prior 64-bit value into vdst[0:1] and updates LDS[idx:idx+1].

        Returns (dword_idx0, old_val_u64) if lane active, or None if inactive.
        Reads all sources before writing vdst to handle aliasing safely.
        """
        if not self.active_lanes[lane]:
            return None
        vaddr = self.vgprs[lane][vaddr_reg]
        vdata_lo = self.vgprs[lane][vdata_lo_reg]
        vdata_hi = self.vgprs[lane][vdata_lo_reg + 1]
        _, dword_idx0, _ = calc_lds_byte_offset(vaddr, offset, self.lds_dwords)
        dword_idx1 = (dword_idx0 + 1) & (self.lds_dwords - 1)
        old_lo = self.lds[dword_idx0]
        old_hi = self.lds[dword_idx1]
        self.lds[dword_idx0] = (old_lo | vdata_lo) & MASK32
        self.lds[dword_idx1] = (old_hi | vdata_hi) & MASK32
        self.vgprs[lane][vdst_lo_reg] = old_lo
        self.vgprs[lane][vdst_lo_reg + 1] = old_hi
        return dword_idx0, combine_u32_to_u64(old_lo, old_hi)

    def parse_and_execute(self, instruction_text: str, lane: int = 0) -> int | tuple[int, int] | None:
        line = instruction_text.strip()
        if not line or line.startswith(";"):
            return None

        offset = 0
        m_off = re.search(r"\boffset:(\S+)", line)
        if m_off:
            offset = int(m_off.group(1), 0)
            line = re.sub(r"\boffset:\S+", "", line).strip()

        parts = line.split(None, 1)
        op = parts[0]
        args_str = parts[1] if len(parts) > 1 else ""
        raw_args = [a.strip() for a in args_str.split(",") if a.strip()]

        def reg_num(tok: str) -> int:
            cleaned = tok.replace("v", "").replace("[", "").replace("]", "")
            if ":" in cleaned:
                cleaned = cleaned.split(":")[0]
            return int(cleaned)

        if op == "ds_or_b32":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            return self.ds_or_b32(vaddr, vdata, offset=offset, lane=lane)

        elif op == "ds_or_rtn_b32":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            vdata = reg_num(raw_args[2])
            res = self.ds_or_rtn_b32(vdst, vaddr, vdata, offset=offset, lane=lane)
            return res[0] if res else None

        elif op == "ds_write_b32":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            return self.ds_write_b32(vaddr, vdata, offset=offset, lane=lane)

        elif op == "ds_read_b32":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            return self.ds_read_b32(vdst, vaddr, offset=offset, lane=lane)

        elif op == "ds_or_b64":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            res_b64 = self.ds_or_b64(vaddr, vdata, offset=offset, lane=lane)
            return res_b64[0] if res_b64 else None

        elif op == "ds_or_rtn_b64":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            vdata = reg_num(raw_args[2])
            res_rtn_b64 = self.ds_or_rtn_b64(vdst, vaddr, vdata, offset=offset, lane=lane)
            return res_rtn_b64[0] if res_rtn_b64 else None

        elif op == "ds_write_b64":
            vaddr = reg_num(raw_args[0])
            vdata = reg_num(raw_args[1])
            res_w64 = self.ds_write_b64(vaddr, vdata, offset=offset, lane=lane)
            return res_w64[0] if res_w64 else None

        elif op == "ds_read_b64":
            vdst = reg_num(raw_args[0])
            vaddr = reg_num(raw_args[1])
            res_r64 = self.ds_read_b64(vdst, vaddr, offset=offset, lane=lane)
            return res_r64[0] if res_r64 else None

        else:
            raise UnsupportedConstruct(f"Opcode {op} not supported in interpreter")

    def run_simd_wave(self, instruction_text: str) -> list[int | tuple[int, int] | None]:
        """Runs the instruction across all lanes in the wavefront in lockstep."""
        return [self.parse_and_execute(instruction_text, lane=lane) for lane in range(self.num_lanes)]


class LdsAtomicSpirvInterp:
    """Strict semantic interpreter for compute SPIR-V modules exercising LDS atomic OR."""

    def __init__(self, spvasm: str):
        self.spvasm = spvasm
        self.types: dict[str, tuple] = {}
        self.constants: dict[str, int | bool] = {}
        self.ptr_types: dict[str, tuple[str, str]] = {}
        self.var_class: dict[str, str] = {}
        self.local_size: tuple[int, int, int] = (1, 1, 1)
        self.lds_var: str | None = None
        self.lds_size: int = 0
        self.body: list[tuple[str | None, str, list[str], str]] = []
        self._parse()

    def _parse(self) -> None:
        in_body = False
        for raw in self.spvasm.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = _LINE.match(line)
            if not m:
                raise UnsupportedConstruct(f"Cannot parse line: {line!r}")
            res, op, rest = m.groups()
            a = rest.split()
            if in_body:
                self.body.append((res, op, a, line))
                continue
            if op == "OpFunction":
                in_body = True
                continue
            if op == "OpExecutionMode" and len(a) >= 5 and a[1] == "LocalSize":
                lx, ly, lz = int(a[2]), int(a[3]), int(a[4])
                if lx < 1 or ly < 1 or lz < 1 or (lx * ly * lz) > 1024:
                    raise WorkgroupBoundsError(f"Invalid LocalSize {lx}x{ly}x{lz}")
                self.local_size = (lx, ly, lz)
                continue
            if op in _IGNORED:
                continue
            if op == "OpTypeInt":
                if int(a[0]) != 32:
                    raise UnsupportedConstruct("Only 32-bit int supported in interpreter")
                self.types[res] = ("int", int(a[0]), int(a[1]))
            elif op == "OpTypeBool":
                self.types[res] = ("bool",)
            elif op == "OpTypeVector":
                self.types[res] = ("vec", a[0], int(a[1]))
            elif op == "OpTypeArray":
                size = int(self.constants[a[1]])
                self.types[res] = ("arr", a[0], size)
            elif op == "OpTypePointer":
                self.ptr_types[res] = (a[0], a[1])
            elif op == "OpConstant":
                self.constants[res] = int(a[1], 0) & MASK32
            elif op == "OpConstantTrue":
                self.constants[res] = True
            elif op == "OpConstantFalse":
                self.constants[res] = False
            elif op == "OpVariable":
                cls, pointee = self.ptr_types[a[0]]
                if a[1] != cls:
                    raise UnsupportedConstruct("OpVariable storage class mismatch")
                self.var_class[res] = cls
                if cls == "Workgroup":
                    if self.lds_var is not None:
                        raise UnsupportedConstruct("Multiple Workgroup variables")
                    self.lds_var = res
                    arr_ty = self.types[pointee]
                    if arr_ty[0] != "arr":
                        raise UnsupportedConstruct("Workgroup variable must be an array")
                    self.lds_size = arr_ty[2]
                elif cls not in ("Output", "Function"):
                    raise UnsupportedConstruct(f"Unsupported storage class {cls}")
            else:
                raise UnsupportedConstruct(f"Header opcode {op}")

    def run(
        self,
        num_invocations: int | None = None,
        seed_lds: dict[int, int] | None = None,
        lane_inputs: list[dict[str, int]] | None = None,
    ) -> tuple[list[int], list[dict[str, int]]]:
        """Executes the workgroup and returns (final_lds, outputs_per_lane)."""
        lx, ly, lz = self.local_size
        total_invocations = lx * ly * lz
        count = total_invocations if num_invocations is None else min(num_invocations, total_invocations)

        lds = [0] * self.lds_size
        if seed_lds:
            for idx, val in seed_lds.items():
                if 0 <= idx < self.lds_size:
                    lds[idx] = val & MASK32

        lane_outputs: list[dict[str, int]] = []
        for lidx in range(count):
            inputs = lane_inputs[lidx] if lane_inputs and lidx < len(lane_inputs) else {}
            out = self._run_lane(lidx, lds, inputs)
            lane_outputs.append(out)

        return list(lds), lane_outputs

    def _run_lane(self, lidx: int, lds: list[int], inputs: dict[str, int]) -> dict[str, int]:
        values: dict[str, int | bool] = dict(self.constants)
        values.update(inputs)
        var_box: dict[str, list[int]] = {}
        pointers: dict[str, tuple[str, list[int], int | None]] = {}
        outputs: list[str] = []

        if self.lds_var:
            pointers[self.lds_var] = ("lds", lds, None)

        for res, cls in self.var_class.items():
            if cls == "Output":
                var_box[res] = [0]
                pointers[res] = ("out", var_box[res], 0)
                outputs.append(res)
            elif cls == "Function":
                var_box[res] = [0]
                pointers[res] = ("fn", var_box[res], 0)

        def val(x: str) -> int:
            if x in values:
                return values[x]  # type: ignore
            raise UnsupportedConstruct(f"Undefined identifier {x}")

        for res, op, a, _ in self.body:
            if op in _IGNORED or op in ("OpBranch", "OpBranchConditional"):
                continue
            if op == "OpVariable":
                cls, _ = self.ptr_types[a[0]]
                if cls != "Function":
                    raise UnsupportedConstruct("Non-Function variable in body")
                var_box[res] = [val(a[2]) if len(a) > 2 else 0]
                pointers[res] = ("fn", var_box[res], 0)
            elif op == "OpAccessChain":
                base = a[1]
                if base not in pointers:
                    raise UnsupportedConstruct(f"OpAccessChain on non-pointer {base}")
                ptag = pointers[base][0]
                if ptag == "lds":
                    idx = val(a[2])
                    if not (0 <= idx < len(lds)):
                        raise WorkgroupBoundsError(f"LDS access out of bounds: {idx} >= {len(lds)}")
                    pointers[res] = ("lds", lds, idx)
                elif ptag in ("out", "fn"):
                    pointers[res] = pointers[base]
                else:
                    raise UnsupportedConstruct(f"Access chain on {ptag}")
            elif op == "OpLoad":
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag in ("lds", "out", "fn"):
                    idx_val = ptr[2]
                    values[res] = ptr[1][idx_val] & MASK32
            elif op == "OpStore":
                ptr = pointers[a[0]]
                v = val(a[1]) & MASK32
                ptag = ptr[0]
                if ptag in ("lds", "out", "fn"):
                    idx_val = ptr[2]
                    ptr[1][idx_val] = v
            elif op == "OpAtomicOr":
                # %res = OpAtomicOr %type %ptr %scope %semantics %val
                ptr = pointers[a[1]]
                ptag = ptr[0]
                if ptag != "lds":
                    raise UnsupportedConstruct(f"OpAtomicOr on non-LDS pointer {ptag}")
                mask_val = val(a[4]) & MASK32
                idx_val = ptr[2]
                old = ptr[1][idx_val] & MASK32
                ptr[1][idx_val] = (old | mask_val) & MASK32
                values[res] = old
            elif op == "OpIAdd":
                values[res] = (val(a[1]) + val(a[2])) & MASK32
            elif op == "OpISub":
                values[res] = (val(a[1]) - val(a[2])) & MASK32
            elif op == "OpShiftRightLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) >> sh) & MASK32
            elif op == "OpShiftLeftLogical":
                sh = val(a[2])
                if sh >= 32:
                    raise UnsupportedConstruct("Shift >= 32")
                values[res] = (val(a[1]) << sh) & MASK32
            elif op == "OpBitwiseAnd":
                values[res] = val(a[1]) & val(a[2]) & MASK32
            elif op == "OpBitwiseOr":
                values[res] = (val(a[1]) | val(a[2])) & MASK32
            elif op == "OpBitwiseXor":
                values[res] = (val(a[1]) ^ val(a[2])) & MASK32
            elif op == "OpSelect":
                cond = val(a[1])
                values[res] = val(a[2]) if cond else val(a[3])
            elif op == "OpCopyObject":
                values[res] = val(a[1])
            elif op == "OpBitcast":
                values[res] = val(a[1]) & MASK32
            else:
                raise UnsupportedConstruct(f"Opcode {op} not modelled in body")

        return {name: var_box[name][0] & MASK32 for name in outputs}


class TestLdsSemanticInterpreterB32(unittest.TestCase):
    """Verifies 32-bit LDS atomic OR execution in multi-lane SIMD interpreter."""

    def setUp(self):
        self.interp = LdsSemanticInterpreter(lds_dwords=256, num_vgprs=64, num_lanes=32)

    def test_ds_or_b32_non_returning_mutates_lds_only(self):
        self.interp.set_lds_dword(5, 0x10002000)
        self.interp.set_vgpr(0, 20)           # byte 20 -> dword 5
        self.interp.set_vgpr(1, 0x00010002)   # bitmask
        self.interp.set_vgpr(2, 0xDEADBEEF)   # dummy register

        res = self.interp.ds_or_b32(vaddr_reg=0, vdata_reg=1, offset=0, lane=0)
        self.assertEqual(res, 5)
        # LDS[5] updated with bitwise OR
        self.assertEqual(self.interp.get_lds_dword(5), 0x10012002)
        # Registers preserved
        self.assertEqual(self.interp.get_vgpr(0), 20)
        self.assertEqual(self.interp.get_vgpr(1), 0x00010002)
        self.assertEqual(self.interp.get_vgpr(2), 0xDEADBEEF)

    def test_ds_or_rtn_b32_basic_returns_pre_atomic_value(self):
        self.interp.set_lds_dword(7, 0x12340000)
        self.interp.set_vgpr(0, 28)           # byte 28 -> dword 7
        self.interp.set_vgpr(1, 0x00005678)   # bitmask
        self.interp.set_vgpr(2, 0)            # vdst

        res = self.interp.ds_or_rtn_b32(vdst_reg=2, vaddr_reg=0, vdata_reg=1, offset=0, lane=0)
        self.assertEqual(res, (7, 0x12340000))
        # vdst receives old memory value
        self.assertEqual(self.interp.get_vgpr(2), 0x12340000)
        # LDS memory receives updated value
        self.assertEqual(self.interp.get_lds_dword(7), 0x12345678)

    def test_ds_or_rtn_b32_with_offset(self):
        self.interp.set_lds_dword(8, 0x0F0F0000)
        self.interp.set_vgpr(0, 16)           # byte 16 (dword 4)
        self.interp.set_vgpr(1, 0x0000F0F0)   # bitmask
        # offset = 16 bytes (+4 dwords) -> target is dword 8
        res = self.interp.ds_or_rtn_b32(vdst_reg=3, vaddr_reg=0, vdata_reg=1, offset=16, lane=0)
        self.assertEqual(res, (8, 0x0F0F0000))
        self.assertEqual(self.interp.get_vgpr(3), 0x0F0F0000)
        self.assertEqual(self.interp.get_lds_dword(8), 0x0F0FF0F0)

    def test_inactive_lane_under_exec_mask(self):
        self.interp.set_lds_dword(2, 0x5555)
        self.interp.set_vgpr(0, 8, lane=1)
        self.interp.set_vgpr(1, 0xFFFF, lane=1)
        self.interp.set_vgpr(5, 0xDEAD, lane=1)
        self.interp.set_lane_active(1, False)

        res = self.interp.ds_or_rtn_b32(vdst_reg=5, vaddr_reg=0, vdata_reg=1, lane=1)
        self.assertIsNone(res)
        # Inactive lane must preserve both LDS memory and destination VGPR
        self.assertEqual(self.interp.get_vgpr(5, lane=1), 0xDEAD)
        self.assertEqual(self.interp.get_lds_dword(2), 0x5555)

    def test_multi_lane_divergent_destinations(self):
        # 32 lanes, each lane targeting its own dword slot (lane 0 -> dword 0, etc.)
        for lane in range(32):
            self.interp.set_lds_dword(lane, 0x1000 + lane)
            self.interp.set_vgpr(0, lane * 4, lane=lane)
            self.interp.set_vgpr(1, 0x00FF, lane=lane)
            self.interp.set_vgpr(2, 0, lane=lane)

        for lane in range(32):
            self.interp.ds_or_rtn_b32(vdst_reg=2, vaddr_reg=0, vdata_reg=1, offset=0, lane=lane)

        for lane in range(32):
            self.assertEqual(self.interp.get_vgpr(2, lane=lane), 0x1000 + lane)
            self.assertEqual(self.interp.get_lds_dword(lane), (0x1000 + lane) | 0x00FF)

    def test_shared_address_clash_all_lanes_same_dword(self):
        # All 32 lanes target dword 10, each lane setting exactly 1 distinct bit
        self.interp.set_lds_dword(10, 0x00000000)
        for lane in range(32):
            self.interp.set_vgpr(0, 40, lane=lane)  # byte 40 -> dword 10
            self.interp.set_vgpr(1, 1 << lane, lane=lane)
            self.interp.set_vgpr(2, 0, lane=lane)

        old_values = []
        for lane in range(32):
            _, old_val = self.interp.ds_or_rtn_b32(vdst_reg=2, vaddr_reg=0, vdata_reg=1, offset=0, lane=lane)
            old_values.append(old_val)

        # Final memory has all 32 bits set
        self.assertEqual(self.interp.get_lds_dword(10), MASK32)

        # First lane saw 0
        self.assertEqual(old_values[0], 0)
        # Every return value is a subset of MASK32 and monotonically non-decreasing in bits
        for i in range(1, 32):
            self.assertGreaterEqual(bin(old_values[i]).count("1"), bin(old_values[i - 1]).count("1"))
            # Lane i should see bits from lanes 0..i-1
            self.assertEqual(old_values[i], (1 << i) - 1)

    def test_shared_address_clash_overlapping_bitmasks(self):
        self.interp.set_lds_dword(4, 0x00)
        # 3 lanes clashing on dword 4 with overlapping masks: 0x0F, 0x3C, 0xF0
        masks = [0x0F, 0x3C, 0xF0]
        for lane, m in enumerate(masks):
            self.interp.set_vgpr(0, 16, lane=lane)  # dword 4
            self.interp.set_vgpr(1, m, lane=lane)
            self.interp.set_vgpr(2, 0, lane=lane)

        for lane in range(3):
            self.interp.ds_or_rtn_b32(vdst_reg=2, vaddr_reg=0, vdata_reg=1, lane=lane)

        # 0x0F | 0x3C | 0xF0 = 0xFF
        self.assertEqual(self.interp.get_lds_dword(4), 0xFF)
        self.assertEqual(self.interp.get_vgpr(2, lane=0), 0x00)
        self.assertEqual(self.interp.get_vgpr(2, lane=1), 0x0F)
        self.assertEqual(self.interp.get_vgpr(2, lane=2), 0x3F)

    def test_vdst_aliasing_vdata(self):
        # vdst is register 1, which also holds vdata
        self.interp.set_lds_dword(4, 0x11110000)
        self.interp.set_vgpr(0, 16)           # dword 4
        self.interp.set_vgpr(1, 0x00002222)   # vdata in v1

        # vdst=1 aliases vdata=1
        self.interp.ds_or_rtn_b32(vdst_reg=1, vaddr_reg=0, vdata_reg=1, lane=0)
        # LDS gets old | initial vdata
        self.assertEqual(self.interp.get_lds_dword(4), 0x11112222)
        # v1 receives old value 0x11110000
        self.assertEqual(self.interp.get_vgpr(1), 0x11110000)

    def test_vdst_aliasing_vaddr(self):
        # vdst is register 0, which also holds vaddr
        self.interp.set_lds_dword(6, 0x88880000)
        self.interp.set_vgpr(0, 24)           # vaddr = byte 24 (dword 6)
        self.interp.set_vgpr(1, 0x00001111)

        self.interp.ds_or_rtn_b32(vdst_reg=0, vaddr_reg=0, vdata_reg=1, lane=0)
        self.assertEqual(self.interp.get_lds_dword(6), 0x88881111)
        self.assertEqual(self.interp.get_vgpr(0), 0x88880000)

    def test_assembly_parser_dispatch_b32(self):
        self.interp.set_lds_dword(3, 0xAAAA0000)
        self.interp.set_vgpr(0, 12)
        self.interp.set_vgpr(1, 0x00005555)

        self.interp.parse_and_execute("ds_or_b32 v0, v1 offset:0")
        self.assertEqual(self.interp.get_lds_dword(3), 0xAAAA5555)

        self.interp.set_vgpr(2, 0)
        self.interp.set_vgpr(1, 0x11110000)
        self.interp.parse_and_execute("ds_or_rtn_b32 v2, v0, v1 offset:0")
        self.assertEqual(self.interp.get_vgpr(2), 0xAAAA5555)
        self.assertEqual(self.interp.get_lds_dword(3), 0xBBBB5555)

    def test_run_simd_wave_with_exec_mask(self):
        # Lanes 0..3 active, lanes 4..7 inactive
        self.interp.set_exec_mask(0x0F)
        for lane in range(8):
            self.interp.set_vgpr(0, lane * 4, lane=lane)
            self.interp.set_vgpr(1, 0xFF, lane=lane)
            self.interp.set_vgpr(2, 0xAA, lane=lane)

        results = self.interp.run_simd_wave("ds_or_rtn_b32 v2, v0, v1")
        for lane in range(4):
            self.assertIsNotNone(results[lane])
            self.assertEqual(self.interp.get_vgpr(2, lane=lane), 0)
            self.assertEqual(self.interp.get_lds_dword(lane), 0xFF)
        for lane in range(4, 8):
            self.assertIsNone(results[lane])
            self.assertEqual(self.interp.get_vgpr(2, lane=lane), 0xAA)
            self.assertEqual(self.interp.get_lds_dword(lane), 0)

    def test_workgroup_barrier_bitmask_aggregation(self):
        # 32 lanes signal arrival by setting their lane bit in LDS[0]
        self.interp.set_lds_dword(0, 0)
        for lane in range(32):
            self.interp.set_vgpr(0, 0, lane=lane)
            self.interp.set_vgpr(1, 1 << lane, lane=lane)
            self.interp.set_vgpr(2, 0, lane=lane)

        for lane in range(32):
            self.interp.ds_or_rtn_b32(vdst_reg=2, vaddr_reg=0, vdata_reg=1, lane=lane)

        # All 32 lanes have arrived
        self.assertEqual(self.interp.get_lds_dword(0), MASK32)
        # Last lane to check in observes all preceding bits set
        self.assertEqual(self.interp.get_vgpr(2, lane=31), (1 << 31) - 1)


class TestSpirvAtomicOrInterpreterB32(unittest.TestCase):
    """Executes SPIR-V compute kernels with LDS OpAtomicOr operations."""

    def test_ds_or_b32_non_returning_semantic(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c3 = OpConstant %uint 3
%c15 = OpConstant %uint 15
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c3
%dummy = OpAtomicOr %uint %ptr %c2 %c0 %c15
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, _ = interp.run(seed_lds={3: 0x1000})
        self.assertEqual(final_lds[3], 0x100F)
        self.assertEqual(final_lds[0], 0)
        self.assertEqual(final_lds[2], 0)
        self.assertEqual(final_lds[4], 0)

    def test_ds_or_rtn_b32_returning_semantic(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c7 = OpConstant %uint 7
%c50 = OpConstant %uint 50
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c7
%old = OpAtomicOr %uint %ptr %c2 %c0 %c50
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        initial_val = 0x12340000
        final_lds, lane_out = interp.run(seed_lds={7: initial_val})
        self.assertEqual(lane_out[0]["%out"], initial_val)
        self.assertEqual(final_lds[7], initial_val | 50)

    def test_multi_lane_shared_bitmask_reduction(self):
        # 8 invocations in a workgroup each atomically OR their bit into LDS[0]
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 8 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c0_slot = OpConstant %uint 0
%c1 = OpConstant %uint 1
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c0_slot
%ticket = OpAtomicOr %uint %ptr %c2 %c0 %c1
OpStore %out %ticket
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_outs = interp.run(num_invocations=8, seed_lds={0: 0})
        self.assertEqual(final_lds[0], 1)
        # First invocation saw 0, others saw 1
        tickets = [o["%out"] for o in lane_outs]
        self.assertEqual(tickets[0], 0)
        self.assertEqual(tickets[1:], [1] * 7)

    def test_chained_atomic_ors_same_slot(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c4_slot = OpConstant %uint 4
%c1 = OpConstant %uint 1
%c2_val = OpConstant %uint 2
%c4_val = OpConstant %uint 4
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c4_slot
%r1 = OpAtomicOr %uint %ptr %c2 %c0 %c1
%r2 = OpAtomicOr %uint %ptr %c2 %c0 %c2_val
%r3 = OpAtomicOr %uint %ptr %c2 %c0 %c4_val
OpStore %out %r3
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={4: 0})
        # r1 got 0 (lds became 1)
        # r2 got 1 (lds became 3)
        # r3 got 3 (lds became 7)
        self.assertEqual(lane_out[0]["%out"], 3)
        self.assertEqual(final_lds[4], 7)

    def test_offset_and_wrapping_in_spirv(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%c16 = OpConstant %uint 16
%c15 = OpConstant %uint 15
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c_base = OpConstant %uint 60
%c_offset = OpConstant %uint 16
%c_val = OpConstant %uint 42
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%s1 = OpIAdd %uint %c_base %c_offset
%i = OpShiftRightLogical %uint %s1 %c2
%m = OpBitwiseAnd %uint %i %c15
%ptr = OpAccessChain %_ptr_WG_uint %lds %m
%old = OpAtomicOr %uint %ptr %c2 %c0 %c_val
OpStore %out %old
OpReturn
OpFunctionEnd
"""
        # Base 60 + Offset 16 = 76 -> 76 >> 2 = 19 -> 19 & 15 = 3
        interp = LdsAtomicSpirvInterp(spvasm)
        final_lds, lane_out = interp.run(seed_lds={3: 100})
        self.assertEqual(lane_out[0]["%out"], 100)
        self.assertEqual(final_lds[3], 100 | 42)


class TestTranslatorIntegrationB32(unittest.TestCase):
    """Verifies that tools/rdna2spirv.py emits conforming SPIR-V for ds_or_b32 and ds_or_rtn_b32."""

    def _make_mock_gen(self, stage="cs"):
        class MockGen:
            def __init__(self):
                self.stage = stage
                self.uses_lds = False
                self.lds_dwords = 256
                self.lds_bound = None       # not bounds-checked (SHADER-309)
                self.wave64 = True
                self._tmp = 0
                self.emitted = []
                self.stored_dsts = []

            def tmp(self):
                self._tmp += 1
                return f"%t{self._tmp}"

            def cu(self, v):
                return f"%c{v}"

            def active_b(self):
                return "%exec_active"

            def src_u(self, arg):
                return f"%v_{arg['r']}"

            def dst(self, d, v):
                self.stored_dsts.append((d, v))

            def emit(self, line):
                self.emitted.append(line)

            LDS_ATOMICS = rdna2spirv.Gen.LDS_ATOMICS
            mod_int = staticmethod(rdna2spirv.Gen.mod_int)
            lds_ptr = rdna2spirv.Gen.lds_ptr
            lds_access = rdna2spirv.Gen.lds_access
            lds_atomic = rdna2spirv.Gen.lds_atomic
            lds_op = rdna2spirv.Gen.lds_op

        return MockGen()

    def test_lds_atomics_table_contains_ds_or(self):
        self.assertIn("ds_or_b32", rdna2spirv.Gen.LDS_ATOMICS)
        self.assertEqual(rdna2spirv.Gen.LDS_ATOMICS["ds_or_b32"], "OpAtomicOr")

    def test_ds_or_b32_non_returning_emission(self):
        gen = self._make_mock_gen(stage="cs")
        gen.lds_op("ds_or_b32", [{"r": 0}, {"r": 1}], ["offset:8"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        self.assertIn("OpBranchConditional %exec_active", emitted_str)
        self.assertIn("OpIAdd %uint %v_0 %c8", emitted_str)
        self.assertIn("OpShiftRightLogical %uint", emitted_str)
        self.assertIn("OpBitwiseAnd %uint", emitted_str)
        self.assertIn("OpAccessChain %_ptr_WG_uint %lds", emitted_str)
        self.assertIn("OpAtomicOr %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 0)

    def test_ds_or_rtn_b32_returning_emission(self):
        gen = self._make_mock_gen(stage="cs")
        gen.lds_op("ds_or_rtn_b32", [{"r": 5}, {"r": 0}, {"r": 1}], ["offset:12"])
        self.assertTrue(gen.uses_lds)
        emitted_str = "\n".join(gen.emitted)

        self.assertIn("OpAtomicOr %uint", emitted_str)
        self.assertIn("%c2 %c0 %v_1", emitted_str)
        self.assertEqual(len(gen.stored_dsts), 1)
        dest_reg, val_id = gen.stored_dsts[0]
        self.assertEqual(dest_reg["r"], 5)
        self.assertTrue(val_id.startswith("%t"))

    def test_full_shader_translation_ds_or_b32(self):
        body = [
            "v_mov_b32_e32 v0, 16",
            "v_mov_b32_e32 v1, 0x1234",
            "ds_or_b32 v0, v1 offset:8",
            "s_waitcnt lgkmcnt(0)",
        ]
        spv = translate_cs_listing(body)
        self.assertIn("OpAtomicOr", spv)
        self.assertIn("%lds", spv)


class TestLdsSemanticInterpreterB64(unittest.TestCase):
    """Verifies 64-bit LDS atomic OR execution in multi-lane SIMD interpreter."""

    def setUp(self):
        self.interp = LdsSemanticInterpreter(lds_dwords=256, num_vgprs=64, num_lanes=32)

    def test_ds_or_b64_non_returning_mutates_both_dwords(self):
        self.interp.set_lds_dword(10, 0x12340000)
        self.interp.set_lds_dword(11, 0x9ABC0000)
        self.interp.set_vgpr(0, 40)           # byte 40 -> dword 10
        self.interp.set_vgpr(2, 0x00005678)   # data lo
        self.interp.set_vgpr(3, 0x0000DEF0)   # data hi
        self.interp.set_vgpr(4, 0xDEADBEEF)   # dummy

        res = self.interp.ds_or_b64(vaddr_reg=0, vdata_lo_reg=2, offset=0, lane=0)
        self.assertEqual(res, (10, 11))
        self.assertEqual(self.interp.get_lds_dword(10), 0x12345678)
        self.assertEqual(self.interp.get_lds_dword(11), 0x9ABCDEF0)
        self.assertEqual(self.interp.get_vgpr(0), 40)
        self.assertEqual(self.interp.get_vgpr(2), 0x00005678)
        self.assertEqual(self.interp.get_vgpr(3), 0x0000DEF0)
        self.assertEqual(self.interp.get_vgpr(4), 0xDEADBEEF)

    def test_ds_or_rtn_b64_returns_prior_64bit_value(self):
        init_val = 0xCAFEBABE_DEADBEEF
        mask_val = 0x00000000_11111111
        self.interp.set_lds_u64(12, init_val)
        self.interp.set_vgpr(0, 48)           # byte 48 -> dword 12
        self.interp.set_vgpr_u64(2, mask_val)
        self.interp.set_vgpr_u64(4, 0)        # vdst in v[4:5]

        res = self.interp.ds_or_rtn_b64(vdst_lo_reg=4, vaddr_reg=0, vdata_lo_reg=2, offset=0, lane=0)
        self.assertEqual(res, (12, init_val))
        self.assertEqual(self.interp.get_vgpr_u64(4), init_val)
        self.assertEqual(self.interp.get_lds_u64(12), init_val | mask_val)

    def test_ds_or_rtn_b64_with_offset(self):
        init_val = 0x11112222_33334444
        mask_val = 0x55555555_AAAAAAAA
        # vaddr = 32 (dword 8), offset = 24 bytes (6 dwords) -> target is dword 14
        self.interp.set_lds_u64(14, init_val)
        self.interp.set_vgpr(0, 32)
        self.interp.set_vgpr_u64(2, mask_val)
        self.interp.set_vgpr_u64(6, 0)

        res = self.interp.ds_or_rtn_b64(vdst_lo_reg=6, vaddr_reg=0, vdata_lo_reg=2, offset=24, lane=0)
        self.assertEqual(res, (14, init_val))
        self.assertEqual(self.interp.get_vgpr_u64(6), init_val)
        self.assertEqual(self.interp.get_lds_u64(14), init_val | mask_val)

    def test_ds_or_rtn_b64_saturation_chain(self):
        # Initial 0; OR with 0x55555555_55555555, then 0xAAAAAAAA_AAAAAAAA -> MASK64
        self.interp.set_lds_u64(20, 0)
        self.interp.set_vgpr(0, 80)
        mask1 = 0x55555555_55555555
        mask2 = 0xAAAAAAAA_AAAAAAAA
        self.interp.set_vgpr_u64(2, mask1)

        res1 = self.interp.ds_or_rtn_b64(vdst_lo_reg=4, vaddr_reg=0, vdata_lo_reg=2, offset=0, lane=0)
        self.assertEqual(res1, (20, 0))
        self.assertEqual(self.interp.get_vgpr_u64(4), 0)
        self.assertEqual(self.interp.get_lds_u64(20), mask1)

        self.interp.set_vgpr_u64(2, mask2)
        res2 = self.interp.ds_or_rtn_b64(vdst_lo_reg=6, vaddr_reg=0, vdata_lo_reg=2, offset=0, lane=0)
        self.assertEqual(res2, (20, mask1))
        self.assertEqual(self.interp.get_vgpr_u64(6), mask1)
        self.assertEqual(self.interp.get_lds_u64(20), MASK64)

    def test_ds_or_rtn_b64_vdst_aliasing_vdata(self):
        init_val = 0x88880000_66660000
        data_val = 0x00007777_00005555
        self.interp.set_lds_u64(16, init_val)
        self.interp.set_vgpr(0, 64)
        self.interp.set_vgpr_u64(2, data_val)

        # vdst is v[2:3], which aliases vdata in v[2:3]
        res = self.interp.ds_or_rtn_b64(vdst_lo_reg=2, vaddr_reg=0, vdata_lo_reg=2, lane=0)
        self.assertEqual(res, (16, init_val))
        self.assertEqual(self.interp.get_lds_u64(16), init_val | data_val)
        self.assertEqual(self.interp.get_vgpr_u64(2), init_val)

    def test_ds_or_rtn_b64_vdst_aliasing_vaddr(self):
        init_val = 0xAAAA0000_CCCC0000
        data_val = 0x0000BBBB_0000DDDD
        self.interp.set_lds_u64(18, init_val)
        self.interp.set_vgpr(0, 72)
        self.interp.set_vgpr_u64(2, data_val)

        # vdst is v[0:1], aliasing vaddr in v0
        res = self.interp.ds_or_rtn_b64(vdst_lo_reg=0, vaddr_reg=0, vdata_lo_reg=2, lane=0)
        self.assertEqual(res, (18, init_val))
        self.assertEqual(self.interp.get_lds_u64(18), init_val | data_val)
        self.assertEqual(self.interp.get_vgpr_u64(0), init_val)

    def test_ds_or_b64_inactive_lane_under_exec_mask(self):
        self.interp.set_lds_u64(4, 0x12345678_9ABCDEF0)
        self.interp.set_vgpr(0, 16, lane=2)
        self.interp.set_vgpr_u64(2, 0xFFFFFFFF_FFFFFFFF, lane=2)
        self.interp.set_vgpr_u64(4, 0xFEEDFACE_CAFEBABE, lane=2)
        self.interp.set_lane_active(2, False)

        res = self.interp.ds_or_rtn_b64(vdst_lo_reg=4, vaddr_reg=0, vdata_lo_reg=2, lane=2)
        self.assertIsNone(res)
        self.assertEqual(self.interp.get_vgpr_u64(4, lane=2), 0xFEEDFACE_CAFEBABE)
        self.assertEqual(self.interp.get_lds_u64(4), 0x12345678_9ABCDEF0)

    def test_ds_or_rtn_b64_simd_wave_divergent(self):
        # 32 lanes, each lane targeting its own 64-bit element (lane * 8 bytes = dword 2 * lane)
        for lane in range(32):
            val = (0x1000_0000_0000 + lane * 0x1_0000_0001) & MASK64
            self.interp.set_lds_u64(lane * 2, val)
            self.interp.set_vgpr(0, lane * 8, lane=lane)
            self.interp.set_vgpr_u64(2, 0x00FF00FF_00FF00FF, lane=lane)
            self.interp.set_vgpr_u64(4, 0, lane=lane)

        for lane in range(32):
            self.interp.ds_or_rtn_b64(vdst_lo_reg=4, vaddr_reg=0, vdata_lo_reg=2, offset=0, lane=lane)

        for lane in range(32):
            orig_val = (0x1000_0000_0000 + lane * 0x1_0000_0001) & MASK64
            self.assertEqual(self.interp.get_vgpr_u64(4, lane=lane), orig_val)
            self.assertEqual(self.interp.get_lds_u64(lane * 2), orig_val | 0x00FF00FF_00FF00FF)

    def test_ds_or_rtn_b64_simd_wave_convergent_shared_word(self):
        # 32 lanes all targeting dword 30 (64-bit), each setting bit lane and bit lane + 32
        self.interp.set_lds_u64(30, 0)
        for lane in range(32):
            self.interp.set_vgpr(0, 120, lane=lane)  # byte 120 -> dword 30
            mask_64 = (1 << lane) | (1 << (lane + 32))
            self.interp.set_vgpr_u64(2, mask_64, lane=lane)
            self.interp.set_vgpr_u64(4, 0, lane=lane)

        for lane in range(32):
            self.interp.ds_or_rtn_b64(vdst_lo_reg=4, vaddr_reg=0, vdata_lo_reg=2, offset=0, lane=lane)

        self.assertEqual(self.interp.get_lds_u64(30), MASK64)

    def test_assembly_parser_dispatch_b64(self):
        self.interp.set_lds_u64(8, 0xAAAAAAAA_00000000)
        self.interp.set_vgpr(0, 32)
        self.interp.set_vgpr_u64(2, 0x00000000_55555555)

        self.interp.parse_and_execute("ds_or_b64 v0, v[2:3] offset:0")
        self.assertEqual(self.interp.get_lds_u64(8), 0xAAAAAAAA_55555555)

        self.interp.set_vgpr_u64(4, 0)
        self.interp.set_vgpr_u64(2, 0x55555555_AAAAAAAA)
        self.interp.parse_and_execute("ds_or_rtn_b64 v[4:5], v0, v[2:3] offset:0")
        self.assertEqual(self.interp.get_vgpr_u64(4), 0xAAAAAAAA_55555555)
        self.assertEqual(self.interp.get_lds_u64(8), MASK64)

        # Test ds_write_b64 and ds_read_b64 through parser
        self.interp.set_vgpr_u64(6, 0xCAFEBABE_12345678)
        self.interp.parse_and_execute("ds_write_b64 v0, v[6:7] offset:8")
        self.assertEqual(self.interp.get_lds_u64(10), 0xCAFEBABE_12345678)

        self.interp.set_vgpr_u64(8, 0)
        self.interp.parse_and_execute("ds_read_b64 v[8:9], v0 offset:8")
        self.assertEqual(self.interp.get_vgpr_u64(8), 0xCAFEBABE_12345678)

    def test_run_simd_wave_b64_with_exec_mask(self):
        # Lanes 0..3 active (0x0F), lanes 4..7 inactive
        self.interp.set_exec_mask(0x0F)
        for lane in range(8):
            self.interp.set_vgpr(0, lane * 8, lane=lane)
            self.interp.set_vgpr_u64(2, 0xFFFFFFFF_00000000, lane=lane)
            self.interp.set_vgpr_u64(4, 0x55555555_55555555, lane=lane)

        results = self.interp.run_simd_wave("ds_or_rtn_b64 v[4:5], v0, v[2:3]")
        for lane in range(4):
            self.assertIsNotNone(results[lane])
            self.assertEqual(self.interp.get_vgpr_u64(4, lane=lane), 0)
            self.assertEqual(self.interp.get_lds_u64(lane * 2), 0xFFFFFFFF_00000000)
        for lane in range(4, 8):
            self.assertIsNone(results[lane])
            self.assertEqual(self.interp.get_vgpr_u64(4, lane=lane), 0x55555555_55555555)
            self.assertEqual(self.interp.get_lds_u64(lane * 2), 0)


class TestLdsOffsetWrappingAndStrides(unittest.TestCase):
    """Verifies edge-case address calculation, 64-bit element wrapping, and st64 dual offsets."""

    def test_64bit_boundary_crossing_wrap(self):
        interp = LdsSemanticInterpreter(lds_dwords=64, num_vgprs=16, num_lanes=1)
        # LDS has 64 dwords (indices 0..63).
        # A 64-bit element at dword 63 spans dword 63 and dword (63+1)%64 = 0.
        interp.set_lds_dword(63, 0x11110000)
        interp.set_lds_dword(0, 0x22220000)
        interp.set_vgpr(0, 63 * 4)  # byte address 252 -> dword 63
        interp.set_vgpr(2, 0x00003333)
        interp.set_vgpr(3, 0x00004444)
        interp.set_vgpr_u64(4, 0)

        res = interp.ds_or_rtn_b64(vdst_lo_reg=4, vaddr_reg=0, vdata_lo_reg=2, offset=0, lane=0)
        self.assertEqual(res, (63, combine_u32_to_u64(0x11110000, 0x22220000)))
        self.assertEqual(interp.get_vgpr(4), 0x11110000)
        self.assertEqual(interp.get_vgpr(5), 0x22220000)
        self.assertEqual(interp.get_lds_dword(63), 0x11113333)
        self.assertEqual(interp.get_lds_dword(0), 0x22224444)

    def test_large_offset_overflow_wrap(self):
        interp = LdsSemanticInterpreter(lds_dwords=128, num_vgprs=16, num_lanes=1)
        # 32-bit unsigned wrapping: 0xFFFFFFFC + 8 = 0x00000004 -> dword 1
        interp.set_lds_dword(1, 0x55550000)
        interp.set_lds_dword(2, 0xAAAA0000)
        interp.set_vgpr(0, 0xFFFFFFFC)
        interp.set_vgpr(2, 0x000000FF)
        interp.set_vgpr(3, 0x00FF0000)

        interp.ds_or_b64(vaddr_reg=0, vdata_lo_reg=2, offset=8, lane=0)
        self.assertEqual(interp.get_lds_dword(1), 0x555500FF)
        self.assertEqual(interp.get_lds_dword(2), 0xAAFF0000)

    def test_calc_dual_offsets_st64_math(self):
        # Non-st64: stride unit is 4 bytes for 32-bit (1 dword per offset unit)
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0x100, 4, 8, lds_dwords=512, st64=False)
        self.assertEqual(eff0, 0x100 + 16)
        self.assertEqual(idx0, (0x100 + 16) >> 2)
        self.assertEqual(eff1, 0x100 + 32)
        self.assertEqual(idx1, (0x100 + 32) >> 2)

        # st64=True: stride unit is 4 * 64 = 256 bytes per offset unit
        (eff0_st, idx0_st), (eff1_st, idx1_st) = calc_lds_dual_offsets(0x100, 1, 2, lds_dwords=512, st64=True)
        self.assertEqual(eff0_st, 0x100 + 256)
        self.assertEqual(idx0_st, (0x100 + 256) >> 2)
        self.assertEqual(eff1_st, 0x100 + 512)
        self.assertEqual(idx1_st, (0x100 + 512) >> 2)

    def test_dual_offset_wrapping_power_of_two(self):
        lds_dwords = 32  # 128 bytes total
        (eff0, idx0), (eff1, idx1) = calc_lds_dual_offsets(0, 32, 33, lds_dwords=lds_dwords, st64=False)
        # offset 32 * 4 = 128 bytes -> wraps to dword 0
        self.assertEqual(idx0, 0)
        # offset 33 * 4 = 132 bytes -> wraps to dword 1
        self.assertEqual(idx1, 1)


class TestSpirvAtomicOrB64Emission(unittest.TestCase):
    """Verifies that 64-bit LDS atomic OR translates to conforming SPIR-V and executes correctly."""

    def test_rdna2spirv_translate_b64_emission(self):
        spv = translate_cs_listing([
            "ds_or_b64 v0, v[2:3]",
            "ds_or_rtn_b64 v[4:5], v0, v[2:3] offset:16",
        ])
        lines = [ln.strip() for ln in spv.splitlines() if "OpAtomicOr" in ln]
        # Must emit 2 OpAtomicOr for ds_or_b64 + 2 OpAtomicOr for ds_or_rtn_b64
        self.assertEqual(len(lines), 4)
        for ln in lines:
            self.assertIn("OpAtomicOr", ln)
            self.assertIn("%uint", ln)
            self.assertIn("%c2", ln)  # Workgroup scope
            self.assertIn("%c0", ln)  # Relaxed semantics

    def test_spirv_sequential_atomics_emulating_b64(self):
        # Compute shader executing two sequential 32-bit OpAtomicOr operations simulating a 64-bit atomic
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main" %out_lo %out_hi
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c64 = OpConstant %uint 64
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c10 = OpConstant %uint 10
%c11 = OpConstant %uint 11
%c_mask_lo = OpConstant %uint 3735928559
%c_mask_hi = OpConstant %uint 3405691582
%lds_arr = OpTypeArray %uint %c64
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%_ptr_Out_uint = OpTypePointer Output %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%out_lo = OpVariable %_ptr_Out_uint Output
%out_hi = OpVariable %_ptr_Out_uint Output
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr_lo = OpAccessChain %_ptr_WG_uint %lds %c10
%ptr_hi = OpAccessChain %_ptr_WG_uint %lds %c11
%old_lo = OpAtomicOr %uint %ptr_lo %c2 %c0 %c_mask_lo
%old_hi = OpAtomicOr %uint %ptr_hi %c2 %c0 %c_mask_hi
OpStore %out_lo %old_lo
OpStore %out_hi %old_hi
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        init_lo = 0x12340000
        init_hi = 0x9ABC0000
        final_lds, lane_outputs = interp.run(seed_lds={10: init_lo, 11: init_hi})
        self.assertEqual(lane_outputs[0]["%out_lo"], init_lo)
        self.assertEqual(lane_outputs[0]["%out_hi"], init_hi)
        self.assertEqual(final_lds[10], init_lo | 3735928559)
        self.assertEqual(final_lds[11], init_hi | 3405691582)


class TestNegativeControlsAndBounds(unittest.TestCase):
    """Verifies that invalid operands, unmodelled opcodes, bad stages, and bound violations fail safely."""

    def test_rdna2spirv_invalid_operand_shape_rejected(self):
        # ds_or_b64 requires VGPR pair for data: passing scalar v1 must raise Unsupported
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_cs_listing(["ds_or_b64 v0, v1"])

    def test_rdna2spirv_invalid_rtn_destination_shape_rejected(self):
        # ds_or_rtn_b64 requires VGPR pair for destination: passing scalar v4 must raise Unsupported
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_cs_listing(["ds_or_rtn_b64 v4, v0, v[2:3]"])

    def test_rdna2spirv_non_compute_stage_rejected(self):
        class MockPixelGen:
            def __init__(self):
                self.stage = "ps"
                self.uses_lds = False

            lds_op = rdna2spirv.Gen.lds_op

        gen = MockPixelGen()
        with self.assertRaises(rdna2spirv.Unsupported):
            gen.lds_op("ds_or_b32", [{"r": 0}, {"r": 1}], [])
        with self.assertRaises(rdna2spirv.Unsupported):
            gen.lds_op("ds_or_b64", [{"r": 0}, {"r": 1, "k": "vgpr", "n": 2}], [])

    def test_spirv_unsupported_storage_class(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_UC = OpTypePointer UniformConstant %uint
%u = OpVariable %_ptr_UC UniformConstant
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
        with self.assertRaises(UnsupportedConstruct):
            LdsAtomicSpirvInterp(spvasm)

    def test_spirv_unsupported_opcode_in_body(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c10 = OpConstant %uint 10
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bad = OpSDiv %uint %c10 %c10
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        with self.assertRaises(UnsupportedConstruct):
            interp.run()

    def test_spirv_invalid_workgroup_local_sizes(self):
        for bad_dims in [(0, 1, 1), (1025, 1, 1), (33, 32, 1)]:
            with self.subTest(dims=bad_dims):
                spvasm = f"""
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize {bad_dims[0]} {bad_dims[1]} {bad_dims[2]}
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
OpReturn
OpFunctionEnd
"""
                with self.assertRaises(WorkgroupBoundsError):
                    LdsAtomicSpirvInterp(spvasm)

    def test_spirv_atomic_on_non_lds_pointer(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%_ptr_fn_uint = OpTypePointer Function %uint
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%c7 = OpConstant %uint 7
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%fn_var = OpVariable %_ptr_fn_uint Function
%bad = OpAtomicOr %uint %fn_var %c2 %c0 %c7
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        with self.assertRaises(UnsupportedConstruct):
            interp.run()

    def test_spirv_out_of_bounds_lds_access(self):
        spvasm = """
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c16 = OpConstant %uint 16
%c99 = OpConstant %uint 99
%c2 = OpConstant %uint 2
%c0 = OpConstant %uint 0
%lds_arr = OpTypeArray %uint %c16
%_ptr_WG_lds = OpTypePointer Workgroup %lds_arr
%_ptr_WG_uint = OpTypePointer Workgroup %uint
%lds = OpVariable %_ptr_WG_lds Workgroup
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%ptr = OpAccessChain %_ptr_WG_uint %lds %c99
%bad = OpAtomicOr %uint %ptr %c2 %c0 %c16
OpReturn
OpFunctionEnd
"""
        interp = LdsAtomicSpirvInterp(spvasm)
        with self.assertRaises(WorkgroupBoundsError):
            interp.run()

    def test_simd_interp_unknown_opcode_rejected(self):
        interp = LdsSemanticInterpreter()
        with self.assertRaises(UnsupportedConstruct):
            interp.parse_and_execute("ds_unknown_op v0, v1")

    def test_spirv_shift_bounds_rejected(self):
        for op in ("OpShiftRightLogical", "OpShiftLeftLogical"):
            with self.subTest(op=op):
                spvasm = f"""
OpCapability Shader
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %main "main"
OpExecutionMode %main LocalSize 1 1 1
%uint = OpTypeInt 32 0
%c1 = OpConstant %uint 1
%c32 = OpConstant %uint 32
%void = OpTypeVoid
%fn = OpTypeFunction %void
%main = OpFunction %void None %fn
%entry = OpLabel
%bad = {op} %uint %c1 %c32
OpReturn
OpFunctionEnd
"""
                interp = LdsAtomicSpirvInterp(spvasm)
                with self.assertRaises(UnsupportedConstruct):
                    interp.run()


if __name__ == "__main__":
    unittest.main()
