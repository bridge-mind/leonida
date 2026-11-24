"""Swizzled MUBUF addressing (V# SWIZZLE_ENABLE) and ADD_TID_ENABLE, as measured natively on RDNA2 (gfx1036, wave32
and wave64; docs/design.md, Swizzled buffer addressing; tools/buffer_swizzle_probe.py):

    SWIZZLE_ENABLE  word1 bit 31          STRIDE      word1 [29:16]
    INDEX_STRIDE    word3 [22:21]         IS = 8 << INDEX_STRIDE indices
    ADD_TID_ENABLE  word3 bit 23          element_size is fixed at 4 bytes (no ELEMENT_SIZE field since GFX9)

    index  = idxen ? vindex : (ADD_TID_ENABLE ? lane : 0)            (idxen ignores ADD_TID_ENABLE)
    offset = uint32((offen ? vaddr : 0) + inst_offset)               (32-bit wrap; soffset is not part of it)
    swz_active = SWIZZLE_ENABLE and (idxen or ADD_TID_ENABLE)        (no effect when both are 0)
    buffer_offset = (index / IS * stride + (offset & ~3)) * IS + (index % IS) * 4 + (offset & 3)
    address = base + soffset + buffer_offset

Two swizzle sub-modes exist on RDNA2:
  - Scratch swizzle (SWIZZLE_ENABLE=1, idxen=0, ADD_TID_ENABLE=1): raw multi-dword accesses step by 4 * IS bytes per
    dword, and typed formats > 4 bytes (plus 8-byte buffer_atomic_*_x2) are suppressed even in OOB_SELECT 2.
  - Indexed swizzle (SWIZZLE_ENABLE=1, idxen=1): raw multi-dword, typed, and _x2 atomic accesses step linearly by +4
    bytes per dword, and typed formats > 4 bytes execute normally.

OOB_SELECT 3 with active swizzle and stride != 0 checks (index + soffset) >= NUM_RECORDS or offset + payload > stride;
with stride 0 or inactive swizzle it keeps the raw check (uint64(offset) + uint64(soffset) + payload > NUM_RECORDS, no
32-bit wrap on offset + soffset). Every row runs twice: with literal V# words that the translator folds, and with
words 1 and 3 or-ed at run time.
"""
import re
import unittest

import support
import spirv_mini_interp as si
import test_value_semantics as tv         # module only: importing its classes would collect them a second time

BASE = tv.VEC_REGION
OUT_V = 0x31016FAC                          # the output V# in s[0:3]: RAW, base 0
W3 = 0x01000FAC                             # RESOURCE_LEVEL, DST_SEL xyzw; FORMAT, INDEX_STRIDE, ADD_TID, mode or-ed in
NONE = None
LANE = 5                                    # the lane of the single invocation in the ADD_TID rows
SWIZZLE_EXTRACT = re.compile(r"OpBitFieldUExtract %uint %t\d+ %c1f %c1\b")    # word1 >> 31 & 1 at run time


class PatternMemory(dict):
    """{byte_address: word}: byte n of the region at BASE holds 0x40 + n (tv._pattern_word), anything written is
    kept, and the address of every load is recorded."""

    def __init__(self):
        super().__init__()
        self.reads = []

    def get(self, key, default=None):
        self.reads.append(key)
        if key in self:
            return self[key]
        return tv._pattern_word(key - BASE) if BASE <= key < BASE + 0x10000 else default


def v_sharp(stride, nr, istride, mode, *, swizzle=True, tid=False, runtime=False, fmt=22):
    """Lines that put the V# in s[4:7]; with runtime words 1 and 3 are or-ed at run time (s_or_b32 is not folded)."""
    w1 = stride << 16 | swizzle << 31
    w3 = W3 | fmt << 12 | istride << 21 | tid << 23 | mode << 28
    lines = [f"s_mov_b32 s4, {BASE:#x}", f"s_mov_b32 s6, {nr:#x}"]
    if runtime:
        return lines + ["s_mov_b32 s5, 0", f"s_mov_b32 s9, {w1:#x}", "s_or_b32 s5, s5, s9",
                        "s_mov_b32 s7, 0", f"s_mov_b32 s10, {w3:#x}", "s_or_b32 s7, s7, s10"]
    return lines + [f"s_mov_b32 s5, {w1:#x}", f"s_mov_b32 s7, {w3:#x}"]


def operands(idx, off, soffset, inst):
    """vaddr operand text, modifiers, the v1/v2 setup and the soffset operand for one row."""
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


# (stride, num_records, INDEX_STRIDE field, mode, idx, off, soffset, inst_offset, byte offset read or NONE)
LOADS = [
    (8, 16, 0, 3, 1, 4, 0, 0, 0x24),                    # IS 8: (0 * 8 + 4) * 8 + 1 * 4
    (8, 16, 0, 3, 9, 0, 0, 0, 0x44),                    # (1 * 8 + 0) * 8 + 1 * 4
    (8, 8, 1, 0, 3, 4, 0, 0, 0x4C),                     # IS 16: (0 * 8 + 4) * 16 + 3 * 4
    (16, 100, 3, 1, 70, 8, 0, 0, 0x618),                # IS 64: (1 * 16 + 8) * 64 + 6 * 4
    (8, 16, 0, 3, 1, NONE, 0, 4, 0x24),                 # inst_offset is part of offset: as row 1
    (8, 16, 0, 0, 1, 4, 0x100, 0, 0x124),               # mode 0: soffset added after swizzle, not in index check
    (8, 16, 0, 3, 1, 4, 0x100, 0, NONE),                # mode 3 swizzled: index + soffset = 0x101 >= 16 -> OOB
    (8, 16, 0, 3, 3, 4, 12, 0, 0x38),                   # mode 3 swizzled: 3 + 12 = 15 < 16 -> IN, 0x20 + 0x0c + 0x0c
    (8, 16, 0, 3, 4, 4, 12, 0, NONE),                   # mode 3 swizzled: 4 + 12 = 16 >= 16 -> OOB
    (8, 4, 0, 1, NONE, 0x10, 0, 0, 0x10),               # no idxen, ADD_TID=0: swizzle inactive -> linear 0x10
    (8, 0x40, 0, 3, NONE, 0x30, 0, 0, 0x30),            # no idxen, ADD_TID=0, mode 3: stays RAW linear despite stride 8
    (8, 4, 0, 2, 5, 0, 0, 0, 0x14),                     # mode 2: 0 * 8 * 8 + 5 * 4
    (8, 16, 0, 3, 16, 0, 0, 0, NONE),                   # mode 3, swizzled, stride 8: index 16 >= 16
    (8, 16, 0, 3, 1, 8, 0, 0, NONE),                    # ... and 8 + 4 > stride 8
    (8, 16, 0, 0, 16, 0, 0, 0, NONE),                   # mode 0 the same: 16 >= 16
    (0, 0x40, 0, 3, NONE, 0x30, 0, 0, 0x30),            # no idxen, ADD_TID=0: linear RAW (0x34 <= 0x40)
    (0, 0x40, 0, 3, 2, 0x30, 0, 0, 0x188),              # idxen, stride 0: swizzled (0x30 * 8 + 8) with RAW check
    (0, 0x40, 0, 3, NONE, 0x40, 0, 0, NONE),            # 0x44 > 0x40
    (8, 4, 0, 1, 3, 0x20, 0, 0, 0x10C),                 # mode 1 checks only the index: (0 * 8 + 0x20) * 8 + 3 * 4
    (8, 16, 0, 3, 1, 0xFFFFFFFC, 0, 8, 0x24),           # uint32(voff + inst_offset) wraps to 4 -> IN at 0x24
    (0, 0xFFFFFFFF, 0, 3, NONE, 0x20, 0xFFFFFFF0, 0, NONE),  # RAW mode 3: uint64(off) + uint64(soff) does not wrap
    (0, 0xFFFFFFFF, 0, 3, 1, 0x20, 0xFFFFFFF0, 0, NONE),     # swizzled stride 0 RAW mode 3: no 32-bit wrap on soff
]

# ADD_TID_ENABLE on lane 5: (stride, nr, INDEX_STRIDE, mode, swizzle, idx, off, soffset, byte offset read or NONE)
TID_LOADS = [
    (4, 8, 0, 1, False, NONE, NONE, 0, 0x14),           # linear, idxen=0: index = lane 5 -> 5 * 4
    (4, 8, 0, 1, False, 2, NONE, 0, 0x08),              # linear, idxen=1 ignores ADD_TID -> index 2 -> 2 * 4
    (4, 5, 0, 1, False, 2, NONE, 0, 0x08),              # idxen=1 ignores ADD_TID in bounds too: index 2 < 5
    (8, 8, 0, 3, False, NONE, 4, 0, 0x2C),              # linear RAW: 5 * 8 + 4; RAW check 4 + 4 <= 8
    (8, 16, 0, 3, True, 4, NONE, 0, 0x10),              # swizzled, idxen=1 ignores ADD_TID: index 4 -> 4 * 4
    (8, 16, 1, 0, True, NONE, 4, 0, 0x54),              # scratch-swizzled IS 16: index 5, (0 * 8 + 4) * 16 + 5 * 4
    (8, 5, 0, 0, True, NONE, NONE, 0, NONE),            # index 5 >= 5
    (8, 10, 0, 3, True, NONE, 0, 4, 0x18),              # scratch-swizzled mode 3: 5 + 4 = 9 < 10 -> 5 * 4 + 4
    (8, 9, 0, 3, True, NONE, 0, 4, NONE),               # scratch-swizzled mode 3: 5 + 4 = 9 >= 9 -> OOB
]


class BufferSwizzle(unittest.TestCase):
    def translate(self, lines, prologue=("v_cmpx_eq_u32_e32 0, v0",)):
        return support.translate_lines("cs_buf_swizzle_t", [*prologue, *lines, "s_endpgm"], fssm=tv.CS_BOUNDS_FSSM)

    def run_spv(self, spv, lane=0):
        memory = PatternMemory()
        support.run_fragment(spv, inputs={"%wg_id": (0, 0, 0), "%local_id": (lane, 0, 0)},
                             push_constants=[0, 0, 0x100, OUT_V] + [0] * 28, memory=memory)
        return memory

    def check_selection(self, spv, runtime, what):
        # the output store through s[0:3] reads SWIZZLE_ENABLE at run time; the probed V# only when it is not a literal
        self.assertEqual(len(SWIZZLE_EXTRACT.findall(spv)), 1 + runtime, what)

    def test_dword_loads(self):
        for runtime in (False, True):
            for row in LOADS:
                stride, nr, ist, mode, idx, off, soffset, inst, want = row
                with self.subTest(row=row, runtime=runtime):
                    vaddr, mods, setup, soff = operands(idx, off, soffset, inst)
                    spv = self.translate([*v_sharp(stride, nr, ist, mode, runtime=runtime), *setup,
                                          f"buffer_load_dword v3, {vaddr}, s[4:7], {soff} {mods}",
                                          "s_waitcnt vmcnt(0)", "buffer_store_dword v3, off, s[0:3], 0"])
                    self.check_selection(spv, runtime, row)
                    memory = self.run_spv(spv)
                    self.assertEqual([a - BASE for a in memory.reads], [] if want is NONE else [want])
                    self.assertEqual(memory[0], 0 if want is NONE else tv._pattern_word(want))

    def test_dword_stores(self):
        for runtime in (False, True):
            for row in LOADS:
                stride, nr, ist, mode, idx, off, soffset, inst, want = row
                with self.subTest(row=row, runtime=runtime):
                    vaddr, mods, setup, soff = operands(idx, off, soffset, inst)
                    spv = self.translate([*v_sharp(stride, nr, ist, mode, runtime=runtime), *setup,
                                          "v_mov_b32 v3, 0x11111111",
                                          f"buffer_store_dword v3, {vaddr}, s[4:7], {soff} {mods}"])
                    memory = self.run_spv(spv)
                    self.assertEqual(dict(memory), {} if want is NONE else {BASE + want: 0x11111111})

    def test_scratch_swizzle_multi_dword_steps_by_four_index_strides(self):
        # Scratch swizzle (SWIZZLE_ENABLE=1, idxen=0, ADD_TID_ENABLE=1): dword k is 4 * IS bytes after dword k - 1,
        # each dword bounds-checked on its own in modes 0 and 3.
        cases = [
            # (stride, nr, IS field, mode, n, lane, off, byte offsets of dwords in range, dwords out of range)
            (16, 4, 0, 0, 4, 2, 0, [0x08, 0x28, 0x48, 0x68], 0),  # IS 8: 4k * 8 + 2 * 4
            (8, 4, 0, 0, 4, 2, 0, [0x08, 0x28], 2),               # dwords 2 and 3 end past stride (12, 16 > 8)
            (8, 16, 1, 3, 2, 5, 0, [0x14, 0x54], 0),              # IS 16: 4k * 16 + 5 * 4
            (8, 16, 1, 3, 2, 5, 4, [0x54], 1),                    # offset 4: dword 1 ends at 12 > 8
            (16, 32, 1, 1, 3, 17, 4, [0x144, 0x184, 0x1C4], 0),   # IS 16: (1 * 16 + 4 + 4k) * 16 + 1 * 4
        ]
        for runtime in (False, True):
            for case in cases:
                stride, nr, ist, mode, n, lane, off, inside, outside = case
                with self.subTest(case=case, runtime=runtime):
                    regs = f"v[3:{2 + n}]"
                    lines = [*v_sharp(stride, nr, ist, mode, tid=True, runtime=runtime), f"v_mov_b32 v1, {off}"]
                    spv = self.translate(lines + [f"buffer_load_dwordx{n} {regs}, v1, s[4:7], 0 offen",
                                                  "s_waitcnt vmcnt(0)", f"buffer_store_dwordx{n} {regs}, off, s[0:3], 0"],
                                         prologue=())
                    memory = self.run_spv(spv, lane=lane)
                    self.assertEqual([a - BASE for a in memory.reads], inside)
                    want = [tv._pattern_word(o) for o in inside] + [0] * outside
                    self.assertEqual([memory.get(4 * k) for k in range(n)], want)
                    spv = self.translate(lines + [*[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(n)],
                                                  f"buffer_store_dwordx{n} {regs}, v1, s[4:7], 0 offen"],
                                         prologue=())
                    memory = self.run_spv(spv, lane=lane)
                    want = {BASE + o: 0x11111111 * (k + 1) for k, o in enumerate(inside)}
                    self.assertEqual(dict(memory), want)

    def test_indexed_swizzle_multi_dword_steps_linearly(self):
        # Indexed swizzle (SWIZZLE_ENABLE=1, idxen=1, regardless of ADD_TID_ENABLE): base dword is swizzled, and
        # subsequent dwords step linearly by +4 bytes (even across IS boundaries), with per-dword bounds in modes 0/3.
        cases = [
            # (stride, nr, IS field, mode, tid, n, idx, off, byte offsets of dwords in range, dwords out of range)
            (16, 4, 0, 0, False, 4, 2, 0, [0x08, 0x0C, 0x10, 0x14], 0),   # IS 8: base 0x08, linear +4
            (8, 4, 0, 0, False, 4, 2, 0, [0x08, 0x0C], 2),                # dwords 2 and 3 end past stride (12, 16 > 8)
            (8, 16, 0, 3, True, 2, 7, 0, [0x1C, 0x20], 0),                # idx=7 crosses IS=8 boundary linearly; tid ignored
            (8, 16, 1, 3, False, 2, 5, 4, [0x54], 1),                     # offset 4: dword 1 ends at 12 > 8
            (16, 32, 1, 1, False, 3, 17, 4, [0x144, 0x148, 0x14C], 0),    # IS 16: (1 * 16 + 4) * 16 + 1 * 4 + 4k
        ]
        for runtime in (False, True):
            for case in cases:
                stride, nr, ist, mode, tid, n, idx, off, inside, outside = case
                with self.subTest(case=case, runtime=runtime):
                    regs = f"v[3:{2 + n}]"
                    lines = [*v_sharp(stride, nr, ist, mode, tid=tid, runtime=runtime), f"v_mov_b32 v1, {idx}",
                             f"v_mov_b32 v2, {off}"]
                    spv = self.translate(lines + [f"buffer_load_dwordx{n} {regs}, v[1:2], s[4:7], 0 idxen offen",
                                                  "s_waitcnt vmcnt(0)", f"buffer_store_dwordx{n} {regs}, off, s[0:3], 0"],
                                         prologue=())
                    memory = self.run_spv(spv, lane=LANE)
                    self.assertEqual([a - BASE for a in memory.reads], inside)
                    want = [tv._pattern_word(o) for o in inside] + [0] * outside
                    self.assertEqual([memory.get(4 * k) for k in range(n)], want)
                    spv = self.translate(lines + [*[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(n)],
                                                  f"buffer_store_dwordx{n} {regs}, v[1:2], s[4:7], 0 idxen offen"],
                                         prologue=())
                    memory = self.run_spv(spv, lane=LANE)
                    want = {BASE + o: 0x11111111 * (k + 1) for k, o in enumerate(inside)}
                    self.assertEqual(dict(memory), want)

    def test_add_tid_enable(self):
        for runtime in (False, True):
            for row in TID_LOADS:
                stride, nr, ist, mode, swizzle, idx, off, soffset, want = row
                with self.subTest(row=row, runtime=runtime):
                    vaddr, mods, setup, soff = operands(idx, off, soffset, 0)
                    spv = self.translate([*v_sharp(stride, nr, ist, mode, swizzle=swizzle, tid=True, runtime=runtime),
                                          *setup, f"buffer_load_dword v3, {vaddr}, s[4:7], {soff} {mods}",
                                          "s_waitcnt vmcnt(0)", f"v_mov_b32 v4, {4 * LANE}",
                                          "buffer_store_dword v3, v4, s[0:3], 0 offen"], prologue=())
                    memory = self.run_spv(spv, lane=LANE)
                    self.assertEqual([a - BASE for a in memory.reads], [] if want is NONE else [want])
                    self.assertEqual(memory[4 * LANE], 0 if want is NONE else tv._pattern_word(want))

    def test_indexed_swizzle_typed_element_steps_linearly(self):
        # Indexed swizzle (idxen=1): 32_32_32_32_UINT (75), 16 bytes, steps linearly from its swizzled base 0x04.
        inside = [0x04, 0x08, 0x0C, 0x10]
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                lines = [*v_sharp(16, 4, 0, 1, runtime=runtime, fmt=75), "v_mov_b32 v1, 1"]
                spv = self.translate(lines + ["buffer_load_format_xyzw v[3:6], v1, s[4:7], 0 idxen",
                                              "s_waitcnt vmcnt(0)", "buffer_store_dwordx4 v[3:6], off, s[0:3], 0"])
                memory = self.run_spv(spv)
                self.assertEqual(sorted(set(a - BASE for a in memory.reads)), inside)
                self.assertEqual([memory.get(4 * k) for k in range(4)], [tv._pattern_word(o) for o in inside])
                spv = self.translate(lines + [*[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(4)],
                                              "buffer_store_format_xyzw v[3:6], v1, s[4:7], 0 idxen"])
                memory = self.run_spv(spv)
                self.assertEqual(dict(memory), {BASE + o: 0x11111111 * (k + 1) for k, o in enumerate(inside)})

    def test_scratch_swizzle_suppresses_typed_over_4_bytes_and_atomic_x2(self):
        # Scratch swizzle (SWIZZLE_ENABLE=1, idxen=0, ADD_TID_ENABLE=1): typed formats > 4 bytes (MUBUF/MTBUF) and
        # 8-byte buffer_atomic_*_x2 are suppressed even in OOB_SELECT 2 (DISABLED), while 4-byte typed formats work.
        for runtime in (False, True):
            with self.subTest(runtime=runtime):
                lines16 = v_sharp(16, 16, 0, 2, tid=True, runtime=runtime, fmt=75)
                spv = self.translate(lines16 + ["buffer_load_format_xyzw v[3:6], off, s[4:7], 0",
                                                "s_waitcnt vmcnt(0)", "buffer_store_dwordx4 v[3:6], off, s[0:3], 0"],
                                     prologue=())
                memory = self.run_spv(spv, lane=LANE)
                self.assertEqual(memory.reads, [])
                self.assertEqual([memory.get(4 * k) for k in range(4)], [0, 0, 0, 0])

                spv = self.translate(lines16 + [*[f"v_mov_b32 v{3 + k}, {0x11111111 * (k + 1):#x}" for k in range(4)],
                                                "buffer_store_format_xyzw v[3:6], off, s[4:7], 0"], prologue=())
                memory = self.run_spv(spv, lane=LANE)
                self.assertEqual(dict(memory), {})

                lines_mt = v_sharp(16, 16, 0, 2, tid=True, runtime=runtime, fmt=20)
                spv = self.translate(lines_mt + [
                    "tbuffer_load_format_xy v[3:4], off, s[4:7], 0 format:[BUF_FMT_32_32_UINT]",
                    "s_waitcnt vmcnt(0)", "buffer_store_dwordx2 v[3:4], off, s[0:3], 0"], prologue=())
                memory = self.run_spv(spv, lane=LANE)
                self.assertEqual(memory.reads, [])
                self.assertEqual([memory.get(0), memory.get(4)], [0, 0])

                spv = self.translate(lines_mt + [
                    "v_mov_b32 v3, 0xaaaaaaaa", "v_mov_b32 v4, 0xbbbbbbbb",
                    "buffer_atomic_swap_x2 v[3:4], off, s[4:7], 0 glc",
                    "s_waitcnt vmcnt(0)", "buffer_store_dwordx2 v[3:4], off, s[0:3], 0"], prologue=())
                memory = self.run_spv(spv, lane=LANE)
                self.assertEqual(memory.reads, [])
                self.assertEqual([memory.get(0), memory.get(4)], [0, 0])
                self.assertNotIn(BASE + 4 * LANE, memory)

                # Control: 4-byte typed format (32_UINT, code 20) succeeds at lane * 4 = 0x14
                spv = self.translate(lines_mt + ["buffer_load_format_x v3, off, s[4:7], 0",
                                                 "s_waitcnt vmcnt(0)", "buffer_store_dword v3, off, s[0:3], 0"],
                                     prologue=())
                memory = self.run_spv(spv, lane=LANE)
                self.assertEqual(sorted(set(a - BASE for a in memory.reads)), [0x14])
                self.assertEqual(memory[0], tv._pattern_word(0x14))

    def test_linear_literal_v_sharp_is_unchanged(self):
        # a literal V# without SWIZZLE_ENABLE or ADD_TID_ENABLE folds both away: no lane, no swizzle arithmetic
        lines = [*v_sharp(8, 16, 0, 3, swizzle=False), "v_mov_b32 v1, 1",
                 "buffer_load_dwordx2 v[3:4], v1, s[4:7], 0 idxen", "s_waitcnt vmcnt(0)",
                 "buffer_store_dwordx2 v[3:4], off, s[0:3], 0"]
        spv = self.translate(lines)
        self.check_selection(spv, False, "linear")
        memory = self.run_spv(spv)
        self.assertEqual([a - BASE for a in memory.reads], [8, 12])


class SwizzleFixture(unittest.TestCase):
    """runtime/test/shaders/cs_buf_swizzle.asm in the interpreter: 64 threads (lanes 0 and 1 store through the
    ADD_TID V#), a 256-byte heap filled with 0xee as vk_compute_test's, the words derived in the .asm header."""

    WANT = [0x5EED0024, 0x5EED0044, 0x5EED004C, 0x5EED0008, 0x5EED000C, 0, 0xBEEF0000, 0xBEEF0001]

    def test_words(self):
        spv = support.translate_fixture("cs_buf_swizzle", fssm=tv.CS_BOUNDS_FSSM)
        memory = {a: 0xEEEEEEEE for a in range(0, 0x100, 4)}
        si.run_workgroup(spv, 64, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0),
                                                           "%sg_lane": i % 32},
                         push_constants=[0, 0, 0xFFFFFFFF, 0x31016FAC] + [0] * 28, memory=memory)
        self.assertEqual([f"{memory[4 * i]:#010x}" for i in range(8)], [f"{w:#010x}" for w in self.WANT])
        self.assertEqual(set(memory), set(range(0, 0x100, 4)), "nothing written outside the heap")


if __name__ == "__main__":
    unittest.main()
