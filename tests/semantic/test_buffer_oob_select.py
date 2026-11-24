"""MUBUF bounds checking per V# OOB_SELECT (word3 bits [29:28]), as measured natively on RDNA2 (gfx1036,
docs/buffer_bounds.md; Mesa's GFX11 column, not its GFX10 one).

With idx = vindex (0 without idxen), off = inst_offset + vaddr offset (0 without offen) and offs = off + soffset, a
dword access is out of range when:

    0 STRUCTURED_WITH_OFFSET   idx >= NUM_RECORDS or off + 4 > STRIDE
    1 STRUCTURED               idx >= NUM_RECORDS
    2 DISABLED                 NUM_RECORDS == 0
    3 RAW                      offs + 4 > NUM_RECORDS      (idx * STRIDE is in the address only)

An out-of-range load returns 0 without reading, an out-of-range store is dropped. The table below is written out by
hand from those rules (not computed), and every row runs twice: with word3 a literal the translator folds, and with
word3 or-ed together at run time, which it selects with OpSelect. The multi-dword tail of RAW loads is in
test_value_semantics.py (measured on an RX 6800); mode 0's is by analogy.
"""
from __future__ import annotations

import os
import re
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

import support
import spirv_mini_interp as si
import test_value_semantics as tv         # module only: importing its classes would collect them a second time

MASK14 = 0x3FFF
MASK16 = 0xFFFF
MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF


class BufferResourceDescriptor:
    """AMD RDNA2 128-bit Buffer Resource Descriptor (V#)."""

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
        oob_select: int = 0,
        format_code: int = 22,
        resource_level: int = 1,
        swizzle_enable: bool = False,
        add_tid_enable: bool = False,
        index_stride: int = 0,
    ) -> BufferResourceDescriptor:
        d0 = base_address & MASK32
        base_hi = (base_address >> 32) & MASK16
        stride_field = (stride & MASK14) << 16
        swz_bit = (1 << 31) if swizzle_enable else 0
        d1 = base_hi | stride_field | swz_bit
        d2 = num_records & MASK32
        fmt_field = (format_code & 0x7F) << 12
        rsrc_level = (resource_level & 1) << 19
        idx_stride = (index_stride & 3) << 21
        add_tid = (1 << 23) if add_tid_enable else 0
        oob_sel = (oob_select & 3) << 28
        dst_sel = 0xFAC
        d3 = dst_sel | fmt_field | rsrc_level | idx_stride | add_tid | oob_sel
        return cls(d0, d1, d2, d3)

    @property
    def base_lo(self) -> int:
        return self.d0

    @property
    def base_hi(self) -> int:
        return self.d1 & MASK16

    @property
    def base_64(self) -> int:
        return ((self.base_hi << 32) | self.base_lo) & MASK64

    @property
    def stride(self) -> int:
        return (self.d1 >> 16) & MASK14

    @property
    def num_records(self) -> int:
        return self.d2

    @property
    def oob_select(self) -> int:
        return (self.d3 >> 28) & 3

    @property
    def format_code(self) -> int:
        return (self.d3 >> 12) & 0x7F

    @property
    def swizzle_enable(self) -> bool:
        return bool(self.d1 & (1 << 31))

    @property
    def add_tid_enable(self) -> bool:
        return bool(self.d3 & (1 << 23))

    @property
    def index_stride(self) -> int:
        return 3 + ((self.d3 >> 21) & 3)

    def as_words(self) -> tuple[int, int, int, int]:
        return (self.d0, self.d1, self.d2, self.d3)


class BufferOobOracle:
    """Reference oracle for RDNA2 MUBUF/MTBUF bounds checking and component clamping.

    Implements the native hardware rules measured on gfx1036 (docs/buffer_bounds.md)
    and gfx1150 proxy:
      Mode 0 (STRUCTURED_WITH_OFFSET):
        idx < num_records and off + 4*k + 4 <= stride (off = inst_offset + vaddr, without soffset).
        If no idxen: num_records != 0 and off + 4*k + 4 <= stride.
      Mode 1 (STRUCTURED):
        idx < num_records (or num_records != 0 without idxen). Shared by all dwords.
      Mode 2 (DISABLED):
        num_records != 0. Shared by all dwords.
      Mode 3 (RAW):
        offs + 4*k + 4 <= num_records (offs = inst_offset + vaddr + soffset).
        idx * stride is part of the address calculation, but not part of the bounds check.
    """

    @staticmethod
    def check_dword_bounds(
        desc: BufferResourceDescriptor,
        k: int = 0,
        idx: int | None = None,
        off: int | None = None,
        soffset: int = 0,
        inst_offset: int = 0,
        lane: int = 0,
    ) -> bool:
        mode = desc.oob_select
        nr = desc.num_records
        stride = desc.stride

        lane_add = lane if desc.add_tid_enable else 0
        eff_idx = None if idx is None else ((idx + lane_add) & MASK32)

        voff = 0 if off is None else (off & MASK32)
        total_off = (voff + inst_offset) & MASK32
        offs = (total_off + soffset) & MASK32

        end_bytes = 4 * k + 4

        sw3 = desc.swizzle_enable and stride != 0
        if mode == 0 or (mode == 3 and sw3):
            if eff_idx is not None:
                idx_ok = eff_idx < nr
            else:
                idx_ok = nr != 0
            end_ok = (total_off + end_bytes) <= stride
            return idx_ok and end_ok
        elif mode == 1:
            if eff_idx is not None:
                return eff_idx < nr
            return nr != 0
        elif mode == 2:
            return nr != 0
        elif mode == 3:
            return (offs + end_bytes) <= nr
        return False

    @staticmethod
    def check_format_bounds(
        desc: BufferResourceDescriptor,
        format_size: int,
        idx: int | None = None,
        off: int | None = None,
        soffset: int = 0,
        inst_offset: int = 0,
        lane: int = 0,
    ) -> bool:
        """Typed element access (MTBUF/format load/store): all-or-nothing check for full format_size."""
        mode = desc.oob_select
        nr = desc.num_records
        stride = desc.stride

        lane_add = lane if desc.add_tid_enable else 0
        eff_idx = None if idx is None else ((idx + lane_add) & MASK32)

        voff = 0 if off is None else (off & MASK32)
        total_off = (voff + inst_offset) & MASK32
        offs = (total_off + soffset) & MASK32

        sw3 = desc.swizzle_enable and stride != 0
        if mode == 0 or (mode == 3 and sw3):
            if eff_idx is not None:
                idx_ok = eff_idx < nr
            else:
                idx_ok = nr != 0
            end_ok = (total_off + format_size) <= stride
            return idx_ok and end_ok
        elif mode == 1:
            if eff_idx is not None:
                return eff_idx < nr
            return nr != 0
        elif mode == 2:
            return nr != 0
        elif mode == 3:
            return (offs + format_size) <= nr
        return False

    @staticmethod
    def compute_effective_address(
        desc: BufferResourceDescriptor,
        k: int = 0,
        idx: int | None = None,
        off: int | None = None,
        soffset: int = 0,
        inst_offset: int = 0,
        lane: int = 0,
    ) -> int:
        lane_add = lane if desc.add_tid_enable else 0
        eff_idx = 0 if idx is None else ((idx + lane_add) & MASK32)
        voff = 0 if off is None else (off & MASK32)

        if desc.swizzle_enable:
            Li = desc.index_stride
            IS = 1 << Li
            off_s = (voff + inst_offset) & MASK32
            om4 = off_s & 0xFFFFFFFC
            ol = off_s & 3
            msb = eff_idx >> Li
            lsb = eff_idx & (IS - 1)
            inner = (msb * desc.stride + om4) & MASK32
            sh = (inner << Li) & MASK32
            sh = (sh + (lsb << 2)) & MASK32
            b_off = (sh + ol) & MASK32
            total = (soffset + b_off) & MASK32
            step = 4 << Li
            return (desc.base_64 + total + k * step) & MASK64
        else:
            total = (soffset + inst_offset + voff) & MASK32
            if idx is not None:
                total = (total + eff_idx * desc.stride) & MASK32
            return (desc.base_64 + total + 4 * k) & MASK64

    @classmethod
    def load_dwords(
        cls,
        desc: BufferResourceDescriptor,
        num_dwords: int,
        idx: int | None = None,
        off: int | None = None,
        soffset: int = 0,
        inst_offset: int = 0,
        memory: dict[int, int] | None = None,
        lane: int = 0,
        active: bool = True,
    ) -> list[int]:
        if memory is None:
            memory = {}
        if not active:
            return [0] * num_dwords

        mode = desc.oob_select
        if mode in (1, 2):
            whole_in_bounds = cls.check_dword_bounds(
                desc, k=0, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
            )
            if not whole_in_bounds:
                return [0] * num_dwords
            res = []
            for k in range(num_dwords):
                addr = cls.compute_effective_address(
                    desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
                )
                res.append(memory.get(addr, 0) & MASK32)
            return res

        res = []
        for k in range(num_dwords):
            in_b = cls.check_dword_bounds(
                desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
            )
            if in_b:
                addr = cls.compute_effective_address(
                    desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
                )
                res.append(memory.get(addr, 0) & MASK32)
            else:
                res.append(0)
        return res

    @classmethod
    def store_dwords(
        cls,
        desc: BufferResourceDescriptor,
        values: list[int],
        idx: int | None = None,
        off: int | None = None,
        soffset: int = 0,
        inst_offset: int = 0,
        memory: dict[int, int] | None = None,
        lane: int = 0,
        active: bool = True,
    ) -> list[int]:
        if memory is None or not active:
            return []

        written = []
        mode = desc.oob_select

        if mode in (1, 2):
            whole_in_bounds = cls.check_dword_bounds(
                desc, k=0, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
            )
            if not whole_in_bounds:
                return []
            for k, val in enumerate(values):
                addr = cls.compute_effective_address(
                    desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
                )
                memory[addr] = val & MASK32
                written.append(addr)
            return written

        for k, val in enumerate(values):
            in_b = cls.check_dword_bounds(
                desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
            )
            if in_b:
                addr = cls.compute_effective_address(
                    desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
                )
                memory[addr] = val & MASK32
                written.append(addr)
        return written

    @classmethod
    def load_format(
        cls,
        desc: BufferResourceDescriptor,
        format_size: int,
        idx: int | None = None,
        off: int | None = None,
        soffset: int = 0,
        inst_offset: int = 0,
        memory: dict[int, int] | None = None,
        lane: int = 0,
        active: bool = True,
    ) -> list[int]:
        num_dwords = format_size // 4
        if memory is None or not active:
            return [0] * num_dwords
        in_b = cls.check_format_bounds(
            desc, format_size=format_size, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
        )
        if not in_b:
            return [0] * num_dwords
        res = []
        for k in range(num_dwords):
            addr = cls.compute_effective_address(
                desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
            )
            res.append(memory.get(addr, 0) & MASK32)
        return res

    @classmethod
    def store_format(
        cls,
        desc: BufferResourceDescriptor,
        values: list[int],
        format_size: int,
        idx: int | None = None,
        off: int | None = None,
        soffset: int = 0,
        inst_offset: int = 0,
        memory: dict[int, int] | None = None,
        lane: int = 0,
        active: bool = True,
    ) -> list[int]:
        if memory is None or not active:
            return []
        in_b = cls.check_format_bounds(
            desc, format_size=format_size, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
        )
        if not in_b:
            return []
        written = []
        for k, val in enumerate(values):
            addr = cls.compute_effective_address(
                desc, k=k, idx=idx, off=off, soffset=soffset, inst_offset=inst_offset, lane=lane
            )
            memory[addr] = val & MASK32
            written.append(addr)
        return written


class WaveBufferOobInterpreter:
    """Multi-lane SIMD wave reference interpreter modeling RDNA2 Wave32/Wave64 buffer operations."""

    def __init__(self, wave_size: int = 32, memory: dict[int, int] | None = None) -> None:
        if wave_size not in (32, 64):
            raise ValueError(f"Unsupported wave size: {wave_size}")
        self.wave_size = wave_size
        self.memory = {} if memory is None else dict(memory)
        self.sgprs = [0] * 106
        self.vgprs = [[0] * 256 for _ in range(wave_size)]
        self.exec_mask = (1 << wave_size) - 1

    def set_exec_mask(self, mask: int) -> None:
        self.exec_mask = mask & ((1 << self.wave_size) - 1)

    def is_lane_active(self, lane: int) -> bool:
        return bool((self.exec_mask >> lane) & 1)

    def set_sgpr_range(self, start: int, values: list[int] | tuple[int, ...]) -> None:
        for i, val in enumerate(values):
            self.sgprs[start + i] = val & MASK32

    def set_lane_vgpr(self, lane: int, reg: int, val: int) -> None:
        self.vgprs[lane][reg] = val & MASK32

    def get_lane_vgpr(self, lane: int, reg: int) -> int:
        return self.vgprs[lane][reg] & MASK32

    def get_descriptor(self, start_sgpr: int) -> BufferResourceDescriptor:
        return BufferResourceDescriptor(
            self.sgprs[start_sgpr],
            self.sgprs[start_sgpr + 1],
            self.sgprs[start_sgpr + 2],
            self.sgprs[start_sgpr + 3],
        )

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
            active = self.is_lane_active(lane)
            off = self.get_lane_vgpr(lane, vaddr_off_reg) if vaddr_off_reg is not None else None
            idx = self.get_lane_vgpr(lane, vaddr_idx_reg) if vaddr_idx_reg is not None else None
            loaded = BufferOobOracle.load_dwords(
                desc=desc,
                num_dwords=num_dwords,
                idx=idx,
                off=off,
                soffset=soffset,
                inst_offset=inst_offset,
                memory=self.memory,
                lane=lane,
                active=active,
            )
            for k in range(num_dwords):
                self.set_lane_vgpr(lane, vdst + k, loaded[k])
            wave_results.append(loaded)
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
    ) -> list[list[int]]:
        desc = self.get_descriptor(srsrc)
        wave_written = []
        for lane in range(self.wave_size):
            active = self.is_lane_active(lane)
            off = self.get_lane_vgpr(lane, vaddr_off_reg) if vaddr_off_reg is not None else None
            idx = self.get_lane_vgpr(lane, vaddr_idx_reg) if vaddr_idx_reg is not None else None
            vals = [self.get_lane_vgpr(lane, vdata + k) for k in range(num_dwords)]
            written = BufferOobOracle.store_dwords(
                desc=desc,
                values=vals,
                idx=idx,
                off=off,
                soffset=soffset,
                inst_offset=inst_offset,
                memory=self.memory,
                lane=lane,
                active=active,
            )
            wave_written.append(written)
        return wave_written


BASE = tv.VEC_REGION
OUT_V = 0x31016FAC                          # the output V# in s[0:3]: RAW, base 0
W3 = 0x01016FAC                             # word3 without OOB_SELECT: RESOURCE_LEVEL, FORMAT 32_FLOAT, DST_SEL xyzw
NONE = None

# (mode, stride, num_records, idx, off, soffset, inst_offset, in_range); idx/off NONE: no idxen / no offen
CASES = [
    # mode 0: idx < NR and off + 4 <= stride (off is vaddr + inst_offset, not soffset)
    (0, 8, 4, 1, 4, 0, 0, True),
    (0, 8, 4, 1, 8, 0, 0, False),      # off 8 + 4 > stride 8
    (0, 12, 4, 0, 8, 0, 0, True),      # the last dword of the record fits: 8 + 4 <= 12
    (0, 10, 4, 0, 8, 0, 0, False),     # ... and 8 + 4 > 10 does not
    (0, 8, 4, 4, 0, 0, 0, False),      # idx 4 >= NR 4
    (0, 8, 4, 3, 4, 0, 0, True),
    (0, 8, 4, 1, 0, 16, 0, True),      # soffset is not part of off
    (0, 8, 4, 1, 0, 0, 8, False),      # inst_offset is
    (0, 8, 4, NONE, 4, 0, 0, True),    # offen only: idx 0 < NR
    (0, 8, 4, NONE, 8, 0, 0, False),
    (0, 0, 4, 0, 0, 0, 0, False),      # stride 0: 0 + 4 > 0
    (0, 8, 0, 0, 0, 0, 0, False),      # NR 0
    # mode 1: idx < NR
    (1, 8, 4, 3, 12, 0, 0, True),      # off past the stride is not checked
    (1, 8, 4, 4, 0, 0, 0, False),
    (1, 8, 4, NONE, 16, 0, 0, True),   # no idxen: idx 0 < NR, however far off is
    (1, 8, 0, NONE, 0, 0, 0, False),
    (1, 0, 4, 3, 0, 0, 0, True),       # stride 0 still checks the index
    (1, 0, 4, 4, 0, 0, 0, False),
    # mode 2: only NR == 0 is out of range
    (2, 8, 4, 5, 0, 0, 0, True),
    (2, 0, 4, NONE, 0x20, 0, 0, True),
    (2, 8, 0, NONE, 0, 0, 0, False),
    # mode 3: offs + 4 <= NR; idx and stride do not take part
    (3, 0, 0x10, NONE, 0xC, 0, 0, True),
    (3, 0, 0x10, NONE, 0x10, 0, 0, False),
    (3, 0, 0xC, NONE, 8, 0, 0, True),    # the last dword fits: 8 + 4 <= 12
    (3, 0, 0xA, NONE, 8, 0, 0, False),   # a dword that only starts inside: 8 + 4 > 10
    (3, 8, 0x10, 7, 4, 0, 0, True),    # idx 7 >= NR 16 is irrelevant; address = 7 * 8 + 4
    (3, 8, 0x10, NONE, 0x10, 0, 0, False),
    (3, 8, 0x10, 0, 4, 8, 0, True),    # soffset counts: offs 12
    (3, 8, 0x10, 0, 4, 12, 0, False),  # offs 16
    (3, 0, 0x10, NONE, 0, 0, 8, True),
    (3, 0, 0x10, NONE, 0, 0, 0x10, False),
    (3, 8, 8, 3, 0, 0, 0, True),       # idx 3 >= NR 8 is not a check here
    (3, 0, 0, NONE, 0, 0, 0, False),
]


def operands(idx, off, soffset, inst):
    """vaddr operand text, modifiers and the v1/v2 setup for one case."""
    regs, mods = [], []
    if idx is not NONE:
        regs.append(idx)
        mods.append("idxen")
    if off is not NONE:
        regs.append(off)
        mods.append("offen")
    vaddr = {0: "off", 1: "v1", 2: "v[1:2]"}[len(regs)]
    setup = [f"v_mov_b32 v{1 + i}, {r:#x}" for i, r in enumerate(regs)]
    if soffset:
        setup.append(f"s_mov_b32 s8, {soffset:#x}")
    return vaddr, " ".join(mods + ([f"offset:{inst}"] if inst else [])), setup, ("s8" if soffset else "0")


def v_sharp(mode, stride, nr, runtime, fmt=22):
    """Lines that put the V# in s[4:7]; with runtime the mode is or-ed in, so the translator cannot see it."""
    w3 = 0x01000FAC | fmt << 12
    lines = [f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, " + hex(stride << 16), f"s_mov_b32 s6, {nr:#x}"]
    if runtime:
        lines += [f"s_mov_b32 s7, {w3:#x}", f"s_mov_b32 s9, {mode << 28:#x}", "s_or_b32 s7, s7, s9"]
    else:
        lines += [f"s_mov_b32 s7, {w3 | mode << 28:#x}"]
    return lines


MODE_EXTRACT = re.compile(r"OpBitFieldUExtract %uint %t\d+ %c1c %c2\b")     # word3 >> 28 & 3 at run time


class OobSelect(unittest.TestCase):
    def translate(self, lines):
        return support.translate_lines("cs_oob_select_t", ["v_cmpx_eq_u32_e32 0, v0", *lines, "s_endpgm"],
                                       fssm=tv.CS_BOUNDS_FSSM)

    def run_spv(self, spv, preload=True):
        memory = tv.ReadLog({BASE + k: tv._pattern_word(k) for k in range(-4, 0x80)} if preload else {})
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
                             push_constants=[0, 0, 0x100, OUT_V] + [0] * 28, memory=memory)
        return memory

    def load_case(self, case, runtime):
        mode, stride, nr, idx, off, soffset, inst, ok = case
        vaddr, mods, setup, soff = operands(idx, off, soffset, inst)
        lines = [*v_sharp(mode, stride, nr, runtime), *setup,
                 f"buffer_load_dword v2, {vaddr}, s[4:7], {soff} {mods}", "s_waitcnt vmcnt(0)",
                 "buffer_store_dword v2, off, s[0:3], 0"]
        spv = self.translate(lines)
        # the output store through s[0:3] reads its mode at run time; the probed V# only when it is not a literal
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), 1 + runtime, case)
        memory = self.run_spv(spv)
        addr = (0 if idx is NONE else idx * stride) + (off or 0) + soffset + inst
        want = tv._pattern_word(addr) if ok else 0
        self.assertEqual(f"{dict.get(memory, 0, 0):#010x}", f"{want:#010x}", (case, runtime))
        self.assertEqual([a - BASE for a in memory.reads], [addr] if ok else [], (case, runtime))

    def test_loads_literal_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.load_case(case, runtime=False)

    def test_loads_runtime_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.load_case(case, runtime=True)

    def store_case(self, case, runtime):
        mode, stride, nr, idx, off, soffset, inst, ok = case
        vaddr, mods, setup, soff = operands(idx, off, soffset, inst)
        lines = [*v_sharp(mode, stride, nr, runtime), *setup, "v_mov_b32 v3, 0x11111111",
                 f"buffer_store_dword v3, {vaddr}, s[4:7], {soff} {mods}"]
        spv = self.translate(lines)
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), int(runtime), case)
        memory = self.run_spv(spv, preload=False)
        addr = (0 if idx is NONE else idx * stride) + (off or 0) + soffset + inst
        self.assertEqual(dict(memory), {BASE + addr: 0x11111111} if ok else {}, (case, runtime))

    def test_stores_literal_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.store_case(case, runtime=False)

    def test_stores_runtime_v_sharp(self):
        for case in CASES:
            with self.subTest(case=case):
                self.store_case(case, runtime=True)

    def test_multi_dword_load_follows_the_mode(self):
        # mode 0 is per dword (by analogy with the measured RAW tail): x2 at idx 1, off 4. With stride 12 both dwords
        # end inside (8, 12); with stride 8 the second ends at 12 > 8 and reads 0
        for runtime in (False, True):
            for stride, want in ((12, [tv._pattern_word(16), tv._pattern_word(20)]), (8, [tv._pattern_word(12), 0])):
                lines = [*v_sharp(0, stride, 4, runtime), "v_mov_b32 v1, 1", "v_mov_b32 v2, 4",
                         "buffer_load_dwordx2 v[4:5], v[1:2], s[4:7], 0 idxen offen", "s_waitcnt vmcnt(0)",
                         "buffer_store_dwordx2 v[4:5], off, s[0:3], 0"]
                memory = self.run_spv(self.translate(lines))
                self.assertEqual([dict.get(memory, 4 * i, 0) for i in range(2)], want, (runtime, stride))
        # mode 1 checks the index once for the whole load
        for runtime in (False, True):
            lines = [*v_sharp(1, 8, 4, runtime), "v_mov_b32 v1, 1", "v_mov_b32 v2, 4",
                     "buffer_load_dwordx2 v[4:5], v[1:2], s[4:7], 0 idxen offen", "s_waitcnt vmcnt(0)",
                     "buffer_store_dwordx2 v[4:5], off, s[0:3], 0"]
            memory = self.run_spv(self.translate(lines))
            self.assertEqual([dict.get(memory, 4 * i, 0) for i in range(2)],
                             [tv._pattern_word(12), tv._pattern_word(16)], runtime)

    # measured on the Radeon 880M (gfx1150), a valid proxy for RDNA2 here (docs/buffer_bounds.md): a dword access
    # that straddles the edge is per dword for loads AND stores, a typed element is all or nothing
    STORE_CASES = [
        # (mode, stride, nr, idx, off, dwords, written dword count)
        (3, 0, 0x20, NONE, 0x14, 4, 3), (3, 0, 0x20, NONE, 0x1C, 4, 1), (3, 0, 0x20, NONE, 0x20, 4, 0),
        (3, 0, 0x20, NONE, 0x1C, 2, 1), (3, 0, 0x20, NONE, 0x10, 4, 4),
        (0, 16, 8, 1, 4, 4, 3), (0, 16, 8, 1, 12, 4, 1), (0, 16, 8, 1, 16, 4, 0), (0, 16, 8, 1, 8, 2, 2),
        (0, 16, 8, 1, 12, 2, 1),
        (1, 16, 8, 1, 16, 4, 4), (2, 16, 8, 1, 16, 4, 4),
    ]

    def test_multi_dword_stores_are_per_dword(self):
        for runtime in (False, True):
            for mode, stride, nr, idx, off, n, written in self.STORE_CASES:
                vaddr, mods, setup, soff = operands(idx, off, 0, 0)
                regs = f"v[3:{2 + n}]"
                lines = [*v_sharp(mode, stride, nr, runtime), *setup,
                         *[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(n)],
                         f"buffer_store_dwordx{n} {regs}, {vaddr}, s[4:7], 0 {mods}"]
                memory = self.run_spv(self.translate(lines), preload=False)
                base = BASE + (0 if idx is NONE else idx * stride) + off
                want = {base + 4 * k: 0x11111111 * (k + 1) for k in range(written)}
                self.assertEqual(dict(memory), want, (mode, stride, nr, idx, off, n, runtime))

    FORMAT_CASES = [(3, 0, 0x20, NONE, 0x10, True), (3, 0, 0x20, NONE, 0x14, False), (3, 0, 0x20, NONE, 0x20, False),
                    (0, 16, 8, 1, 0, True), (0, 16, 8, 1, 4, False), (1, 16, 8, 1, 16, True)]

    def test_typed_element_is_all_or_nothing(self):
        # 32_32_32_32_UINT (75), a 16-byte element: in only if all 16 bytes end inside (buffer_load_format_xyzw and
        # buffer_store_format_xyzw alike)
        for runtime in (False, True):
            for mode, stride, nr, idx, off, ok in self.FORMAT_CASES:
                vaddr, mods, setup, soff = operands(idx, off, 0, 0)
                addr = (0 if idx is NONE else idx * stride) + off
                lines = [*v_sharp(mode, stride, nr, runtime, fmt=75), *setup,
                         f"buffer_load_format_xyzw v[3:6], {vaddr}, s[4:7], 0 {mods}", "s_waitcnt vmcnt(0)",
                         "buffer_store_dwordx4 v[3:6], off, s[0:3], 0"]
                memory = self.run_spv(self.translate(lines))
                got = [dict.get(memory, 4 * k, 0) for k in range(4)]
                want = [tv._pattern_word(addr + 4 * k) for k in range(4)] if ok else [0] * 4
                self.assertEqual(got, want, ("load", mode, stride, nr, idx, off, runtime))
                lines = [*v_sharp(mode, stride, nr, runtime, fmt=75), *setup,
                         *[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(4)],
                         f"buffer_store_format_xyzw v[3:6], {vaddr}, s[4:7], 0 {mods}"]
                memory = self.run_spv(self.translate(lines), preload=False)
                want = {BASE + addr + 4 * k: 0x11111111 * (k + 1) for k in range(4)} if ok else {}
                self.assertEqual(dict(memory), want, ("store", mode, stride, nr, idx, off, runtime))

    def test_loads_literal_v_sharp_from_s_mov_b64(self):
        # s_mov_b64 s[6:7], imm folds word3 (OOB_SELECT mode) without run-time extraction.
        for mode in range(4):
            with self.subTest(mode=mode):
                w3 = 0x01000FAC | (mode << 28)
                val64 = (w3 << 32) | 0x10
                lines = [f"s_mov_b64 s[4:5], {BASE:#x}", f"s_mov_b64 s[6:7], {val64:#x}",
                         "buffer_load_dword v2, off, s[4:7], 0", "s_waitcnt vmcnt(0)",
                         "buffer_store_dword v2, off, s[0:3], 0"]
                spv = self.translate(lines)
                self.assertEqual(len(MODE_EXTRACT.findall(spv)), 1, f"mode {mode} did not fold from s_mov_b64")

    def test_loads_literal_v_sharp_from_s_movk_i32(self):
        # s_movk_i32 s7, imm sets word3 for modes where word3 fits in 16 bits sign-extended.
        lines = [f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 16",
                 "s_movk_i32 s7, 0x3000",
                 "buffer_load_dword v2, off, s[4:7], 0", "s_waitcnt vmcnt(0)",
                 "buffer_store_dword v2, off, s[0:3], 0"]
        spv = self.translate(lines)
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), 1, "s_movk_i32 did not fold into literal V#")

    def test_loads_literal_v_sharp_from_sgpr_copy(self):
        # Copy propagation: s_mov_b32 from a constant SGPR preserves sgpr_imm.
        w3 = 0x31000FAC
        lines = [f"s_mov_b32 s8, {w3:#x}", "s_mov_b32 s7, s8",
                 f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0", "s_mov_b32 s6, 16",
                 "buffer_load_dword v2, off, s[4:7], 0", "s_waitcnt vmcnt(0)",
                 "buffer_store_dword v2, off, s[0:3], 0"]
        spv = self.translate(lines)
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), 1, "s_mov_b32 copy did not fold into literal V#")

    def test_loads_literal_v_sharp_from_s_mov_b64_copy(self):
        # Copy propagation: s_mov_b64 64-bit pair copy preserves sgpr_imm.
        w3 = 0x31000FAC
        val64 = (w3 << 32) | 16
        lines = [f"s_mov_b64 s[8:9], {val64:#x}", "s_mov_b64 s[6:7], s[8:9]",
                 f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 0",
                 "buffer_load_dword v2, off, s[4:7], 0", "s_waitcnt vmcnt(0)",
                 "buffer_store_dword v2, off, s[0:3], 0"]
        spv = self.translate(lines)
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), 1, "s_mov_b64 pair copy did not fold into literal V#")

    def test_loads_literal_v_sharp_from_s_mov_b64_overlapping_copy(self):
        # Overlapping 64-bit copy s_mov_b64 s[6:7], s[5:6] where d == s + 1:
        # Both source registers must be evaluated before writing destination,
        # so s7 gets the old s6 literal rather than the newly written s6 (s5).
        w3 = 0x31000FAC
        lines = [f"s_mov_b32 s4, {BASE:#x}", "s_mov_b32 s5, 16",
                 f"s_mov_b32 s6, {w3:#x}",
                 "s_mov_b64 s[6:7], s[5:6]",
                 "buffer_load_dword v2, off, s[4:7], 0", "s_waitcnt vmcnt(0)",
                 "buffer_store_dword v2, off, s[0:3], 0"]
        spv = self.translate(lines)
        self.assertEqual(len(MODE_EXTRACT.findall(spv)), 1, "s_mov_b64 overlapping copy did not fold into literal V#")


class TestBufferOobOracle(unittest.TestCase):
    """Unit tests for the BufferResourceDescriptor reference oracle and OOB clamping."""

    def test_oracle_matches_handwritten_table(self):
        for case in CASES:
            mode, stride, nr, idx, off, soffset, inst_offset, ok = case
            with self.subTest(case=case):
                desc = BufferResourceDescriptor.from_fields(
                    base_address=BASE,
                    stride=stride,
                    num_records=nr,
                    oob_select=mode,
                )
                res = BufferOobOracle.check_dword_bounds(
                    desc,
                    k=0,
                    idx=idx,
                    off=off,
                    soffset=soffset,
                    inst_offset=inst_offset,
                )
                self.assertEqual(res, ok, f"Oracle mismatch for case {case}")

    def test_oracle_descriptor_properties(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x00007FFF12345678,
            stride=64,
            num_records=256,
            oob_select=3,
            format_code=22,
            swizzle_enable=True,
            add_tid_enable=True,
            index_stride=2,
        )
        self.assertEqual(desc.base_lo, 0x12345678)
        self.assertEqual(desc.base_hi, 0x7FFF)
        self.assertEqual(desc.base_64, 0x00007FFF12345678)
        self.assertEqual(desc.stride, 64)
        self.assertEqual(desc.num_records, 256)
        self.assertEqual(desc.oob_select, 3)
        self.assertEqual(desc.format_code, 22)
        self.assertTrue(desc.swizzle_enable)
        self.assertTrue(desc.add_tid_enable)
        self.assertEqual(desc.index_stride, 5)

    def test_oracle_mode0_structured_with_offset(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            stride=16,
            num_records=4,
            oob_select=0,
        )
        # In-bounds: idx 2 < 4, off 8 + 4 <= 16
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, idx=2, off=8))
        # Boundary: off 12 + 4 == 16 (in-bounds)
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, idx=2, off=12))
        # OOB: off 13 + 4 == 17 > 16
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, idx=2, off=13))
        # OOB: idx 4 >= num_records 4
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, idx=4, off=0))
        # Soffset does not participate in mode 0 offset check
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, idx=1, off=8, soffset=0x100))

    def test_oracle_mode1_structured(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            stride=16,
            num_records=4,
            oob_select=1,
        )
        # Mode 1 checks idx < num_records only. Offsets past stride are permitted
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, idx=3, off=1024))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, idx=4, off=0))
        # No idxen: idx 0 < 4 is True
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, idx=None, off=2048))
        # num_records == 0 causes OOB
        desc_zero = BufferResourceDescriptor.from_fields(base_address=0x1000, stride=16, num_records=0, oob_select=1)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_zero, k=0, idx=None, off=0))

    def test_oracle_mode2_disabled(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            stride=0,
            num_records=1,
            oob_select=2,
        )
        # Bounds check disabled: in-bounds for any offset or index as long as num_records != 0
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, idx=999, off=0x10000))
        desc_zero = BufferResourceDescriptor.from_fields(base_address=0x1000, stride=0, num_records=0, oob_select=2)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_zero, k=0, idx=0, off=0))

    def test_oracle_mode3_raw(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x2000,
            stride=0,
            num_records=32,
            oob_select=3,
        )
        # In-bounds: offs 24 + 4 <= 32
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, off=24, soffset=0))
        # Boundary: offs 28 + 4 == 32
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, off=28, soffset=0))
        # OOB: offs 29 + 4 == 33 > 32
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, off=29, soffset=0))
        # Soffset participates in mode 3
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, off=20, soffset=12))  # 32 + 4 > 32

    def test_oracle_multi_dword_partial_clamping(self):
        # In Mode 3 RAW buffer of 24 bytes:
        # buffer_load_dwordx4 at offset 12 loads dwords at offsets 12, 16, 20, 24.
        # dword 0 (offset 12, end 16 <= 24): in-bounds
        # dword 1 (offset 16, end 20 <= 24): in-bounds
        # dword 2 (offset 20, end 24 <= 24): in-bounds
        # dword 3 (offset 24, end 28 > 24): OOB -> zero fill
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            stride=0,
            num_records=24,
            oob_select=3,
        )
        memory = {
            0x1000 + 12: 0xAAAA0001,
            0x1000 + 16: 0xAAAA0002,
            0x1000 + 20: 0xAAAA0003,
            0x1000 + 24: 0xAAAA0004,
        }
        loaded = BufferOobOracle.load_dwords(desc, num_dwords=4, off=12, memory=memory)
        self.assertEqual(loaded, [0xAAAA0001, 0xAAAA0002, 0xAAAA0003, 0])

        # Mode 0 partial clamping at stride boundary
        # stride 16, num_records 4. load_dwordx4 at off 4:
        # k=0: 4 + 4 = 8 <= 16 (in)
        # k=1: 4 + 8 = 12 <= 16 (in)
        # k=2: 4 + 12 = 16 <= 16 (in)
        # k=3: 4 + 16 = 20 > 16 (OOB -> 0)
        desc_m0 = BufferResourceDescriptor.from_fields(
            base_address=0x2000,
            stride=16,
            num_records=4,
            oob_select=0,
        )
        mem_m0 = {
            0x2000 + 4: 0xBBBB0001,
            0x2000 + 8: 0xBBBB0002,
            0x2000 + 12: 0xBBBB0003,
            0x2000 + 16: 0xBBBB0004,
        }
        loaded_m0 = BufferOobOracle.load_dwords(desc_m0, num_dwords=4, idx=0, off=4, memory=mem_m0)
        self.assertEqual(loaded_m0, [0xBBBB0001, 0xBBBB0002, 0xBBBB0003, 0])

    def test_oracle_multi_dword_stores_suppression(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            stride=0,
            num_records=24,
            oob_select=3,
        )
        memory = {}
        values = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
        written = BufferOobOracle.store_dwords(desc, values, off=12, memory=memory)
        self.assertEqual(len(written), 3)
        self.assertEqual(memory, {
            0x1000 + 12: 0x11111111,
            0x1000 + 16: 0x22222222,
            0x1000 + 20: 0x33333333,
        })

    def test_oracle_typed_format_all_or_nothing(self):
        # 16-byte format (e.g. RGBA32) in Mode 3 with num_records 24
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            stride=0,
            num_records=24,
            oob_select=3,
        )
        memory = {0x1000 + 4 * k: 0x100 + k for k in range(8)}
        # off 8: 8 + 16 == 24 <= 24 -> in-bounds
        loaded = BufferOobOracle.load_format(desc, format_size=16, off=8, memory=memory)
        self.assertEqual(loaded, [0x102, 0x103, 0x104, 0x105])
        # off 12: 12 + 16 == 28 > 24 -> all-or-nothing OOB -> all zero
        loaded_oob = BufferOobOracle.load_format(desc, format_size=16, off=12, memory=memory)
        self.assertEqual(loaded_oob, [0, 0, 0, 0])

    def test_oracle_inactive_lanes(self):
        desc = BufferResourceDescriptor.from_fields(
            base_address=0x1000,
            stride=0,
            num_records=32,
            oob_select=3,
        )
        memory = {0x1000: 0xCAFEBABE}
        loaded = BufferOobOracle.load_dwords(desc, num_dwords=1, off=0, memory=memory, active=False)
        self.assertEqual(loaded, [0])
        store_mem = {}
        written = BufferOobOracle.store_dwords(desc, [0x123], off=0, memory=store_mem, active=False)
        self.assertEqual(written, [])
        self.assertEqual(store_mem, {})


class TestBufferOobSimdMultiLane(unittest.TestCase):
    """Multi-lane SIMD wave tests modeling partial component OOB zero fill and clamping."""

    def translate_helper(self, lines):
        return support.translate_lines(
            "cs_oob_simd_t",
            ["v_cmpx_eq_u32_e32 0, v0", *lines, "s_endpgm"],
            fssm=tv.CS_BOUNDS_FSSM,
        )

    def run_spv_helper(self, spv, num_records=24):
        memory = tv.ReadLog({BASE + k: tv._pattern_word(k) for k in range(-4, 0x80)})
        support.run_fragment(
            spv,
            inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 0x100, OUT_V] + [0] * 28,
            memory=memory,
        )
        return memory

    def test_wave32_divergent_offsets_raw_load_x4_partial_fill(self):
        wave = WaveBufferOobInterpreter(wave_size=32)
        base = 0x1000
        nr = 48
        desc = BufferResourceDescriptor.from_fields(
            base_address=base,
            stride=0,
            num_records=nr,
            oob_select=3,
        )
        wave.set_sgpr_range(4, desc.as_words())

        # Populate memory with distinctive pattern
        for byte_off in range(0, 80, 4):
            wave.memory[base + byte_off] = 0x50000000 | byte_off

        # Set divergent offsets for lanes: v1 = lane * 4
        for lane in range(32):
            wave.set_lane_vgpr(lane, 1, lane * 4)

        # Execute buffer_load_dwordx4 v[4:7], v1, s[4:7], 0 offen
        results = wave.buffer_load_dword(
            vdst=4,
            srsrc=4,
            soffset=0,
            inst_offset=0,
            vaddr_off_reg=1,
            num_dwords=4,
        )

        for lane in range(32):
            off = lane * 4
            expected = []
            for k in range(4):
                if off + 4 * k + 4 <= nr:
                    expected.append(0x50000000 | (off + 4 * k))
                else:
                    expected.append(0)
            self.assertEqual(results[lane], expected, f"Mismatch on lane {lane} with offset {off}")
            # Check register state
            for k in range(4):
                self.assertEqual(wave.get_lane_vgpr(lane, 4 + k), expected[k])

        # Verify specifically the boundary transition lanes:
        # Lane 8 (off 32): 32+16 = 48 <= 48 -> all 4 loaded
        self.assertEqual(results[8], [0x50000020, 0x50000024, 0x50000028, 0x5000002C])
        # Lane 9 (off 36): dword 3 is OOB (36+16=52 > 48) -> zero fill
        self.assertEqual(results[9], [0x50000024, 0x50000028, 0x5000002C, 0])
        # Lane 10 (off 40): dwords 2, 3 OOB -> zero fill
        self.assertEqual(results[10], [0x50000028, 0x5000002C, 0, 0])
        # Lane 11 (off 44): dwords 1, 2, 3 OOB -> zero fill
        self.assertEqual(results[11], [0x5000002C, 0, 0, 0])
        # Lane 12 (off 48): all dwords OOB -> all zero
        self.assertEqual(results[12], [0, 0, 0, 0])

    def test_wave32_divergent_indices_mode0_partial_fill(self):
        wave = WaveBufferOobInterpreter(wave_size=32)
        base = 0x2000
        stride = 16
        nr = 8
        desc = BufferResourceDescriptor.from_fields(
            base_address=base,
            stride=stride,
            num_records=nr,
            oob_select=0,
        )
        wave.set_sgpr_range(4, desc.as_words())

        # Seed memory
        for idx in range(nr):
            for off in range(0, stride, 4):
                wave.memory[base + idx * stride + off] = (idx << 16) | off

        # Lanes have varying idxen (v1) and offen (v2)
        for lane in range(32):
            wave.set_lane_vgpr(lane, 1, lane)
            wave.set_lane_vgpr(lane, 2, (lane % 5) * 4)

        results = wave.buffer_load_dword(
            vdst=4,
            srsrc=4,
            soffset=0,
            inst_offset=0,
            vaddr_off_reg=2,
            vaddr_idx_reg=1,
            num_dwords=4,
        )

        # Mode 0 rule: in-bounds if idx < nr AND off + 4*k + 4 <= stride
        # Lane 0: idx 0 < 8, off 0 -> all 4 dwords fit (0, 4, 8, 12, ends <= 16)
        self.assertEqual(results[0], [0x00000000, 0x00000004, 0x00000008, 0x0000000C])
        # Lane 1: idx 1 < 8, off 4 -> 3 dwords fit, 4th OOB
        self.assertEqual(results[1], [0x00010004, 0x00010008, 0x0001000C, 0])
        # Lane 2: idx 2 < 8, off 8 -> 2 dwords fit, 3rd and 4th OOB
        self.assertEqual(results[2], [0x00020008, 0x0002000C, 0, 0])
        # Lane 3: idx 3 < 8, off 12 -> 1 dword fits, others OOB
        self.assertEqual(results[3], [0x0003000C, 0, 0, 0])
        # Lane 4: idx 4 < 8, off 16 -> 16+4 > 16 -> all OOB
        self.assertEqual(results[4], [0, 0, 0, 0])
        # Lane 8..31: idx >= 8 -> all OOB
        for lane in range(8, 32):
            self.assertEqual(results[lane], [0, 0, 0, 0])

    def test_wave64_partial_exec_mask_and_inactive_lanes(self):
        wave = WaveBufferOobInterpreter(wave_size=64)
        base = 0x4000
        desc = BufferResourceDescriptor.from_fields(
            base_address=base,
            stride=4,
            num_records=64,
            oob_select=1,
        )
        wave.set_sgpr_range(4, desc.as_words())

        # Alternate active lanes: 0, 2, 4, ...
        exec_mask = 0x5555555555555555
        wave.set_exec_mask(exec_mask)

        for lane in range(64):
            wave.set_lane_vgpr(lane, 1, lane)
            wave.set_lane_vgpr(lane, 2, 0xDEAD0000 | lane)

        # Active lanes store their data; inactive lanes are masked
        written = wave.buffer_store_dword(
            vdata=2,
            srsrc=4,
            soffset=0,
            inst_offset=0,
            vaddr_idx_reg=1,
            num_dwords=1,
        )
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(len(written[lane]), 1)
                self.assertEqual(wave.memory[base + lane * 4], 0xDEAD0000 | lane)
            else:
                self.assertEqual(written[lane], [])
                self.assertNotIn(base + lane * 4, wave.memory)

        # Clear vgprs and load back
        for lane in range(64):
            wave.set_lane_vgpr(lane, 3, 0xFFFFFFFF)

        loaded = wave.buffer_load_dword(
            vdst=3,
            srsrc=4,
            soffset=0,
            inst_offset=0,
            vaddr_idx_reg=1,
            num_dwords=1,
        )
        for lane in range(64):
            if lane % 2 == 0:
                self.assertEqual(loaded[lane], [0xDEAD0000 | lane])
                self.assertEqual(wave.get_lane_vgpr(lane, 3), 0xDEAD0000 | lane)
            else:
                self.assertEqual(loaded[lane], [0])
                self.assertEqual(wave.get_lane_vgpr(lane, 3), 0)

    def test_wave32_divergent_partial_store_x4_boundary_integrity(self):
        wave = WaveBufferOobInterpreter(wave_size=32)
        base = 0x5000
        nr = 32
        desc = BufferResourceDescriptor.from_fields(
            base_address=base,
            stride=0,
            num_records=nr,
            oob_select=3,
        )
        wave.set_sgpr_range(4, desc.as_words())

        # Test partial component store suppression near boundary
        # Lane 0: off 16 -> 4 dwords fit (16..32)
        # Lane 1: off 20 -> 3 dwords fit (20..32)
        # Lane 2: off 24 -> 2 dwords fit (24..32)
        # Lane 3: off 28 -> 1 dword fits (28..32)
        # Lane 4: off 32 -> 0 dwords fit
        for lane in range(5):
            wave.set_lane_vgpr(lane, 1, 16 + lane * 4)
            for k in range(4):
                wave.set_lane_vgpr(lane, 2 + k, 0x11110000 * (k + 1) | lane)

        # Execute store for each lane independently to observe written addresses
        for lane in range(5):
            sub_wave = WaveBufferOobInterpreter(wave_size=32)
            sub_wave.set_sgpr_range(4, desc.as_words())
            sub_wave.set_lane_vgpr(lane, 1, 16 + lane * 4)
            for k in range(4):
                sub_wave.set_lane_vgpr(lane, 2 + k, 0x11110000 * (k + 1) | lane)

            written = sub_wave.buffer_store_dword(
                vdata=2,
                srsrc=4,
                soffset=0,
                inst_offset=0,
                vaddr_off_reg=1,
                num_dwords=4,
            )
            expected_count = max(0, 4 - lane)
            self.assertEqual(len(written[lane]), expected_count, f"Lane {lane} write count mismatch")
            # Confirm that no bytes >= base + 32 were ever written
            for addr in sub_wave.memory:
                self.assertLess(addr, base + nr, f"Out of bounds store leaked past limit: {addr:#x}")

    def test_wave_load_dwordx2_and_dwordx3_partial_fill(self):
        wave = WaveBufferOobInterpreter(wave_size=32)
        base = 0x6000
        nr = 24
        desc = BufferResourceDescriptor.from_fields(
            base_address=base,
            stride=0,
            num_records=nr,
            oob_select=3,
        )
        wave.set_sgpr_range(4, desc.as_words())
        for byte_off in range(0, 40, 4):
            wave.memory[base + byte_off] = 0x70000000 | byte_off

        # Test x2 loads
        wave.set_lane_vgpr(0, 1, 16)  # off 16: both dwords fit (16, 20)
        wave.set_lane_vgpr(1, 1, 20)  # off 20: dword 0 fits, dword 1 OOB
        wave.set_lane_vgpr(2, 1, 24)  # off 24: both OOB
        res_x2 = wave.buffer_load_dword(vdst=2, srsrc=4, vaddr_off_reg=1, num_dwords=2)
        self.assertEqual(res_x2[0], [0x70000010, 0x70000014])
        self.assertEqual(res_x2[1], [0x70000014, 0])
        self.assertEqual(res_x2[2], [0, 0])

        # Test x3 loads
        wave.set_lane_vgpr(0, 1, 12)  # off 12: all 3 fit (12, 16, 20)
        wave.set_lane_vgpr(1, 1, 16)  # off 16: 2 fit (16, 20), 1 OOB
        wave.set_lane_vgpr(2, 1, 20)  # off 20: 1 fits (20), 2 OOB
        wave.set_lane_vgpr(3, 1, 24)  # off 24: all 3 OOB
        res_x3 = wave.buffer_load_dword(vdst=4, srsrc=4, vaddr_off_reg=1, num_dwords=3)
        self.assertEqual(res_x3[0], [0x7000000C, 0x70000010, 0x70000014])
        self.assertEqual(res_x3[1], [0x70000010, 0x70000014, 0])
        self.assertEqual(res_x3[2], [0x70000014, 0, 0])
        self.assertEqual(res_x3[3], [0, 0, 0])

    def test_spirv_translated_wave_partial_oob_zero_fill(self):
        # End-to-end test translating buffer_load_dwordx4 with partial component zero fill
        # num_records = 24.
        # off 12: dwords at 12, 16, 20 fit, dword 24 is OOB -> [P(12), P(16), P(20), 0]
        # off 16: dwords at 16, 20 fit, dwords 24, 28 OOB -> [P(16), P(20), 0, 0]
        # off 20: dword at 20 fits, others OOB -> [P(20), 0, 0, 0]
        # off 24: all OOB -> [0, 0, 0, 0]
        cases = [
            (12, [tv._pattern_word(12), tv._pattern_word(16), tv._pattern_word(20), 0]),
            (16, [tv._pattern_word(16), tv._pattern_word(20), 0, 0]),
            (20, [tv._pattern_word(20), 0, 0, 0]),
            (24, [0, 0, 0, 0]),
        ]
        for off, want in cases:
            with self.subTest(offset=off):
                lines = [
                    f"s_mov_b32 s4, {BASE:#x}",
                    "s_mov_b32 s5, 0",
                    "s_mov_b32 s6, 24",
                    "s_mov_b32 s7, 0x31016FAC",  # Mode 3 RAW
                    f"v_mov_b32 v1, {off}",
                    "buffer_load_dwordx4 v[4:7], v1, s[4:7], 0 offen",
                    "s_waitcnt vmcnt(0)",
                    "buffer_store_dwordx4 v[4:7], off, s[0:3], 0",
                ]
                spv = self.translate_helper(lines)
                memory = self.run_spv_helper(spv)
                got = [dict.get(memory, 4 * k, 0) for k in range(4)]
                self.assertEqual(got, want, f"Offset {off} yielded {got} instead of {want}")


class TestBufferOobStrideNegativeAndControls(unittest.TestCase):
    """Tests for stride wrapping, negative offsets, and negative controls."""

    def translate_helper(self, lines):
        return support.translate_lines(
            "cs_oob_stride_neg_t",
            ["v_cmpx_eq_u32_e32 0, v0", *lines, "s_endpgm"],
            fssm=tv.CS_BOUNDS_FSSM,
        )

    def run_spv_helper(self, spv):
        memory = tv.ReadLog({BASE + k: tv._pattern_word(k) for k in range(-4, 0x80)})
        support.run_fragment(
            spv,
            inputs={"%wg_id": (0, 0, 0), "%local_id": (0, 0, 0)},
            push_constants=[0, 0, 0x100, OUT_V] + [0] * 28,
            memory=memory,
        )
        return memory

    def test_stride_zero_behavior_across_all_modes(self):
        # Mode 0 with stride 0: any dword access is out of bounds (off + 4 <= 0 is impossible)
        desc_m0 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=16, oob_select=0)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m0, k=0, idx=0, off=0))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m0, k=0, idx=None, off=0))

        # Mode 1 with stride 0: checks idx < num_records, ignoring stride
        desc_m1 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=16, oob_select=1)
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc_m1, k=0, idx=5, off=64))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m1, k=0, idx=16, off=0))

        # Mode 2 with stride 0: checks num_records != 0 only
        desc_m2 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=16, oob_select=2)
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc_m2, k=0, idx=100, off=1024))

        # Mode 3 with stride 0: raw byte addressing bounded by num_records
        desc_m3 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=16, oob_select=3)
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc_m3, k=0, off=12))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m3, k=0, off=13))

    def test_stride_14bit_mask_and_boundary(self):
        # Stride is 14 bits (0..16383)
        max_stride = 0x3FFF  # 16383
        desc = BufferResourceDescriptor.from_fields(
            base_address=BASE,
            stride=max_stride,
            num_records=4,
            oob_select=0,
        )
        self.assertEqual(desc.stride, 16383)
        # Exactly fits: 16379 + 4 = 16383 <= 16383
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc, k=0, idx=0, off=16379))
        # 1-byte overflow: 16380 + 4 = 16384 > 16383
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, idx=0, off=16380))

    def test_stride_wrapping_partial_vector_loads(self):
        # Stride 16, num_records 4. Multi-dword loads wrapping past stride
        desc = BufferResourceDescriptor.from_fields(
            base_address=BASE,
            stride=16,
            num_records=4,
            oob_select=0,
        )
        mem = {BASE + 4 * k: 0x1000 + k for k in range(16)}

        # off 8: dword 0 (8..12), dword 1 (12..16) fit; dwords 2, 3 wrap past stride
        loaded_8 = BufferOobOracle.load_dwords(desc, num_dwords=4, idx=0, off=8, memory=mem)
        self.assertEqual(loaded_8, [0x1002, 0x1003, 0, 0])

        # off 12: dword 0 (12..16) fits; dwords 1..3 wrap past stride
        loaded_12 = BufferOobOracle.load_dwords(desc, num_dwords=4, idx=0, off=12, memory=mem)
        self.assertEqual(loaded_12, [0x1003, 0, 0, 0])

        # off 14: unaligned start straddling stride edge: 14 + 4 = 18 > 16 -> OOB
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, idx=0, off=14))

    def test_negative_voffset_32bit_wrap(self):
        # A negative 32-bit voffset (e.g. -4 = 0xFFFFFFFC) wraps to high 32-bit address space
        # In Mode 0: off = 0xFFFFFFFC; 0xFFFFFFFC + 4 = 0x100000000 (overflow) > stride -> OOB
        desc_m0 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=16, num_records=4, oob_select=0)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m0, k=0, idx=0, off=0xFFFFFFFC))

        # In Mode 3: offs = 0xFFFFFFFC > num_records -> OOB
        desc_m3 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=32, oob_select=3)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m3, k=0, off=0xFFFFFFFC))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m3, k=0, off=0xFFFFFFF0))

        # Verify in translated SPIR-V for Mode 3 RAW with negative voffset
        lines = [
            f"s_mov_b32 s4, {BASE:#x}",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 32",
            "s_mov_b32 s7, 0x31016FAC",  # Mode 3 RAW
            "v_mov_b32 v1, -4",          # negative voffset (0xFFFFFFFC)
            "buffer_load_dword v2, v1, s[4:7], 0 offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dword v2, off, s[0:3], 0",
        ]
        spv = self.translate_helper(lines)
        memory = self.run_spv_helper(spv)
        self.assertEqual(dict.get(memory, 0, 0), 0)

    def test_negative_soffset_participation_by_mode(self):
        # Mode 0: soffset does not participate in the bounds check off = voff + inst_offset
        # A negative soffset (e.g. -4) does not invalidate an in-bounds off=0
        desc_m0 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=16, num_records=4, oob_select=0)
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc_m0, k=0, idx=0, off=0, soffset=0xFFFFFFFC))

        # Mode 3: soffset participates in offs = voff + inst_offset + soffset
        # A negative soffset that wraps into high 32 bits exceeds num_records and is rejected
        desc_m3 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=32, oob_select=3)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m3, k=0, off=0, soffset=0xFFFFFFFC))

    def test_negative_controls_zero_num_records(self):
        # When num_records == 0, ALL modes must reject loads and stores
        for mode in range(4):
            with self.subTest(mode=mode):
                desc = BufferResourceDescriptor.from_fields(base_address=BASE, stride=16, num_records=0, oob_select=mode)
                self.assertFalse(BufferOobOracle.check_dword_bounds(desc, k=0, idx=0, off=0))
                # Load returns 0
                loaded = BufferOobOracle.load_dwords(desc, num_dwords=1, idx=0, off=0)
                self.assertEqual(loaded, [0])
                # Store writes nothing
                mem = {}
                written = BufferOobOracle.store_dwords(desc, [0x12345678], idx=0, off=0, memory=mem)
                self.assertEqual(written, [])
                self.assertEqual(mem, {})

    def test_negative_controls_mesa_gfx10_vs_rdna2_hardware(self):
        # Discriminative tests documented in docs/buffer_bounds.md where Mesa GFX10 rule fails on RDNA2:
        # Case 1: Raw buffer with N = 3, read at offset 0
        # Mesa GFX10 column (offset >= N): 0 >= 3 is False -> would allow load (WRONG)
        # RDNA2 GFX11 column (offset + 4 > N): 0 + 4 = 4 > 3 is True -> OUT OF BOUNDS (CORRECT)
        desc_raw_3 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=3, oob_select=3)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_raw_3, k=0, off=0))

        # Case 2: Structured buffer with stride 10, inst_offset 8
        # Mesa GFX10 column (offset >= stride): 8 >= 10 is False -> would allow load (WRONG)
        # RDNA2 GFX11 column (off + 4 > stride): 8 + 4 = 12 > 10 is True -> OUT OF BOUNDS (CORRECT)
        desc_str_10 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=10, num_records=4, oob_select=0)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_str_10, k=0, idx=0, off=0, inst_offset=8))

        # Case 3: Raw read with soffset 8, offset 8, N = 18
        # Mesa GFX11 without soffset (offset + 4 > N): 8 + 4 = 12 <= 18 -> would allow load (WRONG)
        # RDNA2 measured hardware rule (offset + soffset + 4 > N): 8 + 8 + 4 = 20 > 18 -> OUT OF BOUNDS (CORRECT)
        desc_raw_18 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=0, num_records=18, oob_select=3)
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_raw_18, k=0, off=8, soffset=8))

        # End-to-end translation verification for Case 1 (num_records = 3, offset 0 -> must return 0)
        lines = [
            f"s_mov_b32 s4, {BASE:#x}",
            "s_mov_b32 s5, 0",
            "s_mov_b32 s6, 3",           # num_records = 3
            "s_mov_b32 s7, 0x31016FAC",  # Mode 3 RAW
            "buffer_load_dword v2, off, s[4:7], 0",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dword v2, off, s[0:3], 0",
        ]
        spv = self.translate_helper(lines)
        memory = self.run_spv_helper(spv)
        self.assertEqual(dict.get(memory, 0, 0), 0)

    def test_negative_controls_out_of_range_indices(self):
        # In Mode 0 and Mode 1: idx == num_records must be rejected
        desc_m0 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=16, num_records=4, oob_select=0)
        desc_m1 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=16, num_records=4, oob_select=1)

        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m0, k=0, idx=4, off=0))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m0, k=0, idx=5, off=0))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m1, k=0, idx=4, off=0))
        self.assertFalse(BufferOobOracle.check_dword_bounds(desc_m1, k=0, idx=10, off=0))

        # Mode 3 RAW: index is NOT part of bounds check (it is part of address only)
        desc_m3 = BufferResourceDescriptor.from_fields(base_address=BASE, stride=16, num_records=32, oob_select=3)
        self.assertTrue(BufferOobOracle.check_dword_bounds(desc_m3, k=0, idx=100, off=0))


if __name__ == "__main__":
    unittest.main()
