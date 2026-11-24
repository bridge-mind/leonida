r"""SHADER-343: Native RDNA2 hardware probe for swizzled buffer addressing (V# SWIZZLE_ENABLE, ADD_TID_ENABLE).

Runs self-contained MUBUF/MTBUF buffer-swizzle kernels natively on an AMD GPU through tools/hw_oracle (wave32 and
wave64) and compares every output word with Leonida's live translator (tools/rdna2spirv.py ->
tests/semantic/spirv_mini_interp.py). Standard library only; no game files.

Groups covered (docs/design.md, Swizzled buffer addressing):
    swizzle_basic         IS=8/16/32/64 formula, index >= IS, non-4-mult stride, inst_offset, mode 0 soffset, stride 0
    idxen_ignores_tid     idxen=1 ignores ADD_TID_ENABLE (linear & swizzled, address & bounds, load/typed/atomic)
    swizzle_activation    SWIZZLE_ENABLE=1 with idxen=0 and ADD_TID_ENABLE=0 stays linear and RAW in mode 3
    multi_dword_stepping  scratch swizzle (+4*IS) vs indexed swizzle (+4 linear, crossing IS), loads/stores & per-dword OOB
    typed_and_atomic_x2   scratch swizzle suppresses typed >4B & atomic_*_x2 even in mode 2; indexed swizzle steps +4
    oob_mode3_soffset     OOB_SELECT 3 with active swizzle and stride != 0 checks (index + soffset) >= num_records
    offset_wrap           uint32(voff + inst_offset) wraps at 32 bits; RAW mode 3 uint64(off) + uint64(soff) does not
    add_tid_wave_lane     64-thread ADD_TID_ENABLE wraps at wave lane 32 on wave32 vs 64 on wave64

Usage:
    python -I tools/buffer_swizzle_probe.py                         # translator side only (CPU)
    python -I tools/buffer_swizzle_probe.py --oracle hw             # native AMD GPU only (wave32 + wave64)
    python -I tools/buffer_swizzle_probe.py --oracle both           # compare hardware vs translator
    python -I tools/buffer_swizzle_probe.py --oracle both --strict  # fail on known divergences too
    python -I tools/buffer_swizzle_probe.py --self-test

Exit codes:
    0   all unexpected comparisons match (and wave32 == wave64 on wave-invariant cases)
    1   at least one unexpected mismatch, unexpected wave32/64 diff, or known divergence under --strict
    2   bad CLI arguments or harness error
    77  hardware oracle requested but unavailable on this machine
"""
import argparse
import json
import os
import struct
import sys
import tempfile
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.dont_write_bytecode = True
for p in (HERE, os.path.join(ROOT, "tests", "semantic")):
    if p not in sys.path:
        sys.path.insert(0, p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp  # noqa: E402

# Buffer layout in the allocated buffer (s[0:3] RAW V# at offset 0):
#   0x000..0x0FF (64 dwords):  output words out[0..63]
#   0x100..0x7FF (448 dwords): data region DATA_BASE (s[16:17]), seeded with 0x5EED0000 + offset
OUT_WORDS = 512
DATA_OFF = 0x100
DATA_DWORDS = OUT_WORDS - (DATA_OFF // 4)
SEED_BASE = 0x5EED0000
OUT_V_W3 = 0x31016FAC
V_W3_BASE = 0x01000FAC

KNOWN_DIVERGENCES = {}


def _cs_fssm(threads):
    header = bytearray(0x148)
    struct.pack_into("<4I", header, 0x138, 0x8088, threads, 1, 1)
    return bytes(header)


def _vsharp_lines(stride, nr, istride, mode, *, swz=True, tid=False, fmt=22):
    """Build probed V# in s[4:7] anchored at DATA_BASE (s[16:17])."""
    w1_hi = (stride << 16) | (int(swz) << 31)
    w3 = V_W3_BASE | (fmt << 12) | (istride << 21) | (int(tid) << 23) | (mode << 28)
    return [
        "s_mov_b32 s4, s16",
        f"s_or_b32 s5, s17, {w1_hi:#x}",
        f"s_mov_b32 s6, {nr:#x}",
        f"s_mov_b32 s7, {w3:#x}",
    ]


class Case:
    """One self-contained buffer-swizzle probe kernel.

    `body` runs after s[0:3] is the RAW output V# (base 0) and s[16:17] is DATA_BASE (base + 0x100), with the data
    region seeded to `0x5EED0000 + byte_offset`. `body` stores `words` output dwords into out[0..words-1] via s[0:3].
    """

    __slots__ = ("group", "name", "body", "note", "threads", "words", "wave_dependent")

    def __init__(self, group, name, body, note="", *, threads=1, words=4, wave_dependent=False):
        self.group = group
        self.name = name
        self.body = tuple(body)
        self.note = note
        self.threads = threads
        self.words = words
        self.wave_dependent = wave_dependent


def _single_lane(lines):
    return ["v_cmpx_eq_u32_e32 0, v0", *lines]


def build_cases():
    cases = []

    # 1. Baseline swizzled addressing formula (IS=8/16/32/64, index>=IS, non-4-mult stride, inst_offset, mode 0 soffset)
    cases += [
        Case("swizzle_basic", "is8_idx_off", _single_lane([
            *_vsharp_lines(8, 16, 0, 3),
            "v_mov_b32 v1, 1", "v_mov_b32 v2, 4",
            "buffer_load_dword v8, v[1:2], s[4:7], 0 idxen offen",     # (0*8+4)*8 + 1*4 = 0x24
            "v_mov_b32 v1, 9", "v_mov_b32 v2, 0",
            "buffer_load_dword v9, v[1:2], s[4:7], 0 idxen offen",     # (1*8+0)*8 + 1*4 = 0x44
            "v_mov_b32 v1, 1",
            "buffer_load_dword v10, v1, s[4:7], 0 idxen offset:4",     # inst_offset=4 -> 0x24
            *_vsharp_lines(8, 4, 0, 1),
            "v_mov_b32 v1, 3", "v_mov_b32 v2, 0x20",
            "buffer_load_dword v11, v[1:2], s[4:7], 0 idxen offen",    # mode 1: (0*8+0x20)*8 + 3*4 = 0x10c
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "IS=8 idx/off, idx>=IS, inst_offset, and mode 1 large off"),

        Case("swizzle_basic", "is16_32_64_non4_stride", _single_lane([
            *_vsharp_lines(8, 8, 1, 0),                                # IS=16
            "v_mov_b32 v1, 3", "v_mov_b32 v2, 4",
            "buffer_load_dword v8, v[1:2], s[4:7], 0 idxen offen",     # (0*8+4)*16 + 3*4 = 0x4c
            *_vsharp_lines(16, 64, 2, 1),                              # IS=32
            "v_mov_b32 v1, 35", "v_mov_b32 v2, 4",
            "buffer_load_dword v9, v[1:2], s[4:7], 0 idxen offen",     # (1*16+4)*32 + 3*4 = 0x28c
            *_vsharp_lines(8, 100, 3, 1),                              # IS=64
            "v_mov_b32 v1, 65",
            "buffer_load_dword v10, v1, s[4:7], 0 idxen",              # (1*8)*64 + 1*4 = 0x204
            *_vsharp_lines(18, 16, 0, 0),                              # IS=8, non-4-multiple stride 18
            "v_mov_b32 v1, 9", "v_mov_b32 v2, 4",
            "buffer_load_dword v11, v[1:2], s[4:7], 0 idxen offen",    # (1*18+4)*8 + 1*4 = 0xb4
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "IS=16/32/64 and non-4-multiple stride 18"),

        Case("swizzle_basic", "stride0_and_mode0_soff", _single_lane([
            *_vsharp_lines(0, 0x40, 0, 3),                             # stride=0, idxen=1: swizzled addr + RAW check
            "v_mov_b32 v1, 2", "v_mov_b32 v2, 0x30",
            "buffer_load_dword v8, v[1:2], s[4:7], 0 idxen offen",     # (0*0+0x30)*8 + 2*4 = 0x188
            *_vsharp_lines(8, 16, 0, 0),                               # mode 0: soffset added after swizzle, not checked
            "s_mov_b32 s8, 0x100",
            "v_mov_b32 v1, 1", "v_mov_b32 v2, 4",
            "buffer_load_dword v9, v[1:2], s[4:7], s8 idxen offen",    # 0x24 + 0x100 = 0x124
            *_vsharp_lines(8, 4, 0, 2),                                # mode 2: disabled bounds
            "v_mov_b32 v1, 5",
            "buffer_load_dword v10, v1, s[4:7], 0 idxen",              # 5*4 = 0x14
            *_vsharp_lines(8, 16, 0, 0),                               # mode 0: idx 16 >= 16 -> OOB
            "v_mov_b32 v1, 16",
            "buffer_load_dword v11, v1, s[4:7], 0 idxen",              # 0
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "stride=0 RAW with idxen, mode 0 soffset, mode 2, and mode 0 OOB"),
    ]

    # 2. idxen == 1 ignores ADD_TID_ENABLE (linear & swizzled, address & bounds, load/typed/atomic)
    cases += [
        Case("idxen_ignores_tid", "linear_idxen_tid", [
            *_vsharp_lines(4, 5, 0, 1, swz=False, tid=True),
            "v_mov_b32 v1, 2",
            "buffer_load_dword v8, v1, s[4:7], 0 idxen",               # every lane reads idx=2 (< 5) -> 0x08
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v8, v2, s[0:3], 0 offen",
        ], "linear idxen=1 ignores ADD_TID_ENABLE for both address and bounds", threads=8, words=8),

        Case("idxen_ignores_tid", "swizzled_idxen_tid", [
            *_vsharp_lines(8, 16, 0, 3, swz=True, tid=True),
            "v_mov_b32 v1, 4",
            "buffer_load_dword v8, v1, s[4:7], 0 idxen",               # every lane reads idx=4 -> 0x10
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v8, v2, s[0:3], 0 offen",
        ], "swizzled idxen=1 ignores ADD_TID_ENABLE across lanes", threads=8, words=8),

        Case("idxen_ignores_tid", "typed_and_atomic_idxen_tid", [
            "v_cmpx_eq_u32_e32 3, v0",                                 # run on lane 3 so lane != 0
            *_vsharp_lines(8, 16, 0, 0, swz=True, tid=True, fmt=20),
            "v_mov_b32 v1, 1",
            "buffer_load_format_x v8, v1, s[4:7], 0 idxen",            # idx=1 -> 0x04
            "v_mov_b32 v1, 2",
            "tbuffer_load_format_x v9, v1, s[4:7], 0 format:[BUF_FMT_32_UINT] idxen",  # idx=2 -> 0x08
            "v_mov_b32 v1, 3", "v_mov_b32 v10, 0x77777777",
            "buffer_atomic_swap v10, v1, s[4:7], 0 idxen glc",         # idx=3 -> old 0x0c
            "s_waitcnt vmcnt(0)",
            "buffer_load_dword v11, off, s[0:3], 0 offset:0x10c",      # verify swap wrote 0x0c, not (3+3)*4=0x18
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ], "lane 3 typed MUBUF, MTBUF, and atomic with idxen=1 ignore ADD_TID_ENABLE", threads=4, words=4),
    ]

    # 3. SWIZZLE_ENABLE=1 with idxen=0 and ADD_TID_ENABLE=0 has zero effect (linear + mode 3 stays RAW)
    cases += [
        Case("swizzle_activation", "no_idx_no_tid_linear", _single_lane([
            *_vsharp_lines(8, 4, 0, 1, swz=True, tid=False),
            "v_mov_b32 v1, 0x10",
            "buffer_load_dword v8, v1, s[4:7], 0 offen",               # linear 0x10 (not 0x80)
            *_vsharp_lines(8, 0x40, 0, 3, swz=True, tid=False),
            "v_mov_b32 v1, 0x30",
            "buffer_load_dword v9, v1, s[4:7], 0 offen",               # stays RAW linear 0x30 despite stride 8
            *_vsharp_lines(0x40, 4, 0, 0, swz=True, tid=False),
            "v_mov_b32 v1, 0x10",
            "buffer_load_dword v10, v1, s[4:7], 0 offen",              # mode 0 linear 0x10
            *_vsharp_lines(8, 4, 0, 2, swz=True, tid=False),
            "v_mov_b32 v1, 0x10",
            "buffer_load_dword v11, v1, s[4:7], 0 offen",              # mode 2 linear 0x10
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "SWIZZLE_ENABLE=1 with idxen=0 and ADD_TID_ENABLE=0 stays linear across modes 0..3"),
    ]

    # 4. Multi-dword stepping: scratch swizzle (+4*IS) vs indexed swizzle (+4 linear)
    cases += [
        Case("multi_dword_stepping", "scratch_swz_load_x4", [
            "v_cmpx_eq_u32_e32 2, v0",
            *_vsharp_lines(16, 4, 0, 0, swz=True, tid=True),           # IS=8, lane=2 -> 0x08, 0x28, 0x48, 0x68
            "buffer_load_dwordx4 v[8:11], off, s[4:7], 0",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ], "scratch swizzle (idxen=0, ADD_TID=1) x4 load steps by +4*IS", threads=4, words=4),

        Case("multi_dword_stepping", "scratch_swz_load_x4_oob", [
            "v_cmpx_eq_u32_e32 2, v0",
            *_vsharp_lines(8, 4, 0, 0, swz=True, tid=True),            # stride=8: dwords 0..1 IN, 2..3 OOB
            "buffer_load_dwordx4 v[8:11], off, s[4:7], 0",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ], "scratch swizzle x4 load per-dword OOB at stride=8", threads=4, words=4),

        Case("multi_dword_stepping", "scratch_swz_store_x2", [
            "v_cmpx_eq_u32_e32 5, v0",
            *_vsharp_lines(8, 16, 1, 3, swz=True, tid=True),           # IS=16, lane=5 -> writes 0x14 and 0x54
            "v_mov_b32 v1, 0xaaaa0001", "v_mov_b32 v2, 0xaaaa0002",
            "buffer_store_dwordx2 v[1:2], off, s[4:7], 0",
            "s_waitcnt vmcnt(0)",
            "buffer_load_dword v8, off, s[0:3], 0 offset:0x114",
            "buffer_load_dword v9, off, s[0:3], 0 offset:0x118",
            "buffer_load_dword v10, off, s[0:3], 0 offset:0x154",
            "buffer_load_dword v11, off, s[0:3], 0 offset:0x158",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ], "scratch swizzle x2 store steps by +4*IS (IS=16 -> +0x40)", threads=8, words=4),

        Case("multi_dword_stepping", "indexed_swz_load_x4", _single_lane([
            *_vsharp_lines(16, 4, 0, 0, swz=True, tid=False),          # IS=8, idx=2 -> 0x08, 0x0c, 0x10, 0x14
            "v_mov_b32 v1, 2",
            "buffer_load_dwordx4 v[8:11], v1, s[4:7], 0 idxen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "indexed swizzle (idxen=1) x4 load steps linearly by +4"),

        Case("multi_dword_stepping", "indexed_swz_load_x2_cross_is_and_oob", _single_lane([
            *_vsharp_lines(8, 16, 0, 3, swz=True, tid=True),           # IS=8, idx=7 -> 0x1c, 0x20 (linear across IS)
            "v_mov_b32 v1, 7",
            "buffer_load_dwordx2 v[8:9], v1, s[4:7], 0 idxen",
            *_vsharp_lines(8, 16, 1, 3, swz=True, tid=False),          # IS=16, idx=5, off=4 -> 0x54, 0 (dword 1 OOB)
            "v_mov_b32 v1, 5", "v_mov_b32 v2, 4",
            "buffer_load_dwordx2 v[10:11], v[1:2], s[4:7], 0 idxen offen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "indexed swizzle x2 crosses IS=8 linearly and enforces per-dword stride OOB"),

        Case("multi_dword_stepping", "indexed_swz_store_x2", _single_lane([
            *_vsharp_lines(8, 16, 0, 3, swz=True, tid=False),          # IS=8, idx=2 -> writes 0x08 and 0x0c
            "v_mov_b32 v1, 2", "v_mov_b32 v2, 0xbbbb0001", "v_mov_b32 v3, 0xbbbb0002",
            "buffer_store_dwordx2 v[2:3], v1, s[4:7], 0 idxen",
            "s_waitcnt vmcnt(0)",
            "buffer_load_dword v8, off, s[0:3], 0 offset:0x108",
            "buffer_load_dword v9, off, s[0:3], 0 offset:0x10c",
            "buffer_load_dword v10, off, s[0:3], 0 offset:0x128",
            "buffer_load_dword v11, off, s[0:3], 0 offset:0x12c",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "indexed swizzle x2 store steps linearly by +4 (leaves +4*IS untouched)"),
    ]

    # 5. Typed formats and buffer_atomic_*_x2 in indexed vs scratch swizzle
    cases += [
        Case("typed_and_atomic_x2", "indexed_swz_fmt16_load", _single_lane([
            *_vsharp_lines(16, 4, 0, 1, swz=True, tid=False, fmt=75),  # 32_32_32_32_UINT, idx=1 -> 0x04..0x10
            "v_mov_b32 v1, 1",
            "buffer_load_format_xyzw v[8:11], v1, s[4:7], 0 idxen",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "indexed swizzle 16B typed load steps linearly by +4"),

        Case("typed_and_atomic_x2", "indexed_swz_fmt16_store", _single_lane([
            *_vsharp_lines(16, 4, 0, 1, swz=True, tid=False, fmt=75),
            "v_mov_b32 v1, 1",
            "v_mov_b32 v2, 0xcccc0001", "v_mov_b32 v3, 0xcccc0002",
            "v_mov_b32 v4, 0xcccc0003", "v_mov_b32 v5, 0xcccc0004",
            "buffer_store_format_xyzw v[2:5], v1, s[4:7], 0 idxen",
            "s_waitcnt vmcnt(0)",
            "buffer_load_dword v8, off, s[0:3], 0 offset:0x104",
            "buffer_load_dword v9, off, s[0:3], 0 offset:0x108",
            "buffer_load_dword v10, off, s[0:3], 0 offset:0x124",
            "buffer_load_dword v11, off, s[0:3], 0 offset:0x144",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "indexed swizzle 16B typed store writes contiguous dwords"),

        Case("typed_and_atomic_x2", "scratch_swz_fmt16_suppressed", [
            "v_cmpx_eq_u32_e32 5, v0",
            *_vsharp_lines(16, 16, 0, 2, swz=True, tid=True, fmt=75),  # mode 2 (DISABLED), yet suppressed -> all 0
            "buffer_load_format_xyzw v[8:11], off, s[4:7], 0",
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ], "scratch swizzle suppresses 16B typed MUBUF load even in OOB_SELECT 2", threads=8, words=4),

        Case("typed_and_atomic_x2", "scratch_swz_mtbuf_atomic_x2_and_4b_ctrl", [
            "v_cmpx_eq_u32_e32 4, v0",                                 # lane 4 -> addr +0x10 (8-byte aligned)
            *_vsharp_lines(16, 16, 0, 2, swz=True, tid=True, fmt=20),  # mode 2, 32_UINT (4B) in V#
            "tbuffer_load_format_xy v[2:3], off, s[4:7], 0 format:[BUF_FMT_32_32_UINT]",  # 8B -> suppressed (0, 0)
            "v_mov_b32 v4, 0xeeeeeeee", "v_mov_b32 v5, 0xffffffff",
            "buffer_atomic_swap_x2 v[4:5], off, s[4:7], 0 glc",        # 8B -> suppressed (0, 0, mem untouched)
            "buffer_load_format_x v10, off, s[4:7], 0",                # 4B control -> 0x5eed0010
            "s_waitcnt vmcnt(0)",
            "v_or_b32 v8, v2, v3",                                     # 0
            "v_or_b32 v9, v4, v5",                                     # 0
            "buffer_load_dword v11, off, s[0:3], 0 offset:0x110",      # mem untouched -> 0x5eed0010
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ], "scratch swizzle suppresses 8B MTBUF & atomic_swap_x2 while 4B typed control succeeds", threads=8, words=4),

        Case("typed_and_atomic_x2", "indexed_swz_atomic_x2", _single_lane([
            *_vsharp_lines(16, 16, 0, 0, swz=True, tid=False),
            "v_mov_b32 v1, 2",                                         # addr +0x08 (8-byte aligned)
            "v_mov_b32 v8, 0xdddd0001", "v_mov_b32 v9, 0xdddd0002",
            "buffer_atomic_swap_x2 v[8:9], v1, s[4:7], 0 idxen glc",   # old [0x5eed0008, 0x5eed000c]
            "s_waitcnt vmcnt(0)",
            "buffer_load_dword v10, off, s[0:3], 0 offset:0x108",      # new 0xdddd0001
            "buffer_load_dword v11, off, s[0:3], 0 offset:0x10c",      # new 0xdddd0002
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "indexed swizzle 8B buffer_atomic_swap_x2 executes linearly at aligned address"),
    ]

    # 6. OOB_SELECT 3 with active swizzle and stride != 0 checks (index + soffset) >= num_records
    cases += [
        Case("oob_mode3_soffset", "indexed_mode3_soff_edge", _single_lane([
            *_vsharp_lines(8, 16, 0, 3, swz=True, tid=False),
            "s_mov_b32 s8, 0x100",
            "v_mov_b32 v1, 1", "v_mov_b32 v2, 4",
            "buffer_load_dword v8, v[1:2], s[4:7], s8 idxen offen",    # 1 + 0x100 >= 16 -> 0 (OOB)
            "s_mov_b32 s8, 12",
            "v_mov_b32 v1, 3", "v_mov_b32 v2, 4",
            "buffer_load_dword v9, v[1:2], s[4:7], s8 idxen offen",    # 3 + 12 = 15 < 16 -> 0x38
            "v_mov_b32 v1, 4", "v_mov_b32 v2, 4",
            "buffer_load_dword v10, v[1:2], s[4:7], s8 idxen offen",   # 4 + 12 = 16 >= 16 -> 0 (OOB)
            *_vsharp_lines(8, 14, 0, 3, swz=True, tid=False),
            "s_mov_b32 s8, 8",
            "v_mov_b32 v1, 5",
            "buffer_load_dword v11, v1, s[4:7], s8 idxen",             # 5 + 8 = 13 < 14 -> 0x14 + 8 = 0x1c
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "mode 3 swizzled with stride != 0 checks (index + soffset) < num_records"),

        Case("oob_mode3_soffset", "scratch_mode3_soff_lanes", [
            *_vsharp_lines(8, 10, 0, 3, swz=True, tid=True),
            "s_mov_b32 s8, 4",
            "buffer_load_dword v8, off, s[4:7], s8",                   # lanes 0..5 IN (lane+4 < 10), 6..15 OOB
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v8, v2, s[0:3], 0 offen",
        ], "scratch swizzle mode 3 with soffset=4 and NR=10: lanes 0..5 IN, 6..15 OOB", threads=16, words=16),
    ]

    # 7. 32-bit wrap on uint32(voff + inst_offset) vs 64-bit no-wrap on RAW mode 3 (off + soffset)
    cases += [
        Case("offset_wrap", "voff_inst_wrap_and_soff_nowrap", _single_lane([
            *_vsharp_lines(8, 16, 0, 3, swz=True, tid=False),
            "v_mov_b32 v1, 1", "v_mov_b32 v2, 0xfffffffc",
            "buffer_load_dword v8, v[1:2], s[4:7], 0 idxen offen offset:8",   # off wraps to 4 -> 0x24
            *_vsharp_lines(0, 0x40, 0, 3, swz=False, tid=False),
            "v_mov_b32 v1, 0xfffffffc",
            "buffer_load_dword v9, v1, s[4:7], 0 offen offset:8",             # linear off wraps to 4 -> 0x04
            *_vsharp_lines(0, 0xffffffff, 0, 3, swz=False, tid=False),
            "s_mov_b32 s8, 0xfffffff0",
            "v_mov_b32 v1, 0x20",
            "buffer_load_dword v10, v1, s[4:7], s8 offen",                    # 0x20 + 0xfffffff0 = 0x100000010 -> OOB
            *_vsharp_lines(0, 0xffffffff, 0, 3, swz=True, tid=False),
            "v_mov_b32 v1, 1", "v_mov_b32 v2, 0x20",
            "buffer_load_dword v11, v[1:2], s[4:7], s8 idxen offen",          # swizzled stride=0 RAW -> OOB
            "s_waitcnt vmcnt(0)",
            "buffer_store_dwordx4 v[8:11], off, s[0:3], 0",
        ]), "uint32(voff + inst_offset) wraps at 32 bits; RAW mode 3 off + soffset does not"),
    ]

    # 8. 64-thread ADD_TID_ENABLE uses wave lane (0..31 in wave32, 0..63 in wave64), not workgroup thread ID
    cases += [
        Case("add_tid_wave_lane", "tid_wave32_vs_wave64", [
            *_vsharp_lines(8, 64, 0, 1, swz=True, tid=True),
            "buffer_load_dword v8, off, s[4:7], 0",
            "s_waitcnt vmcnt(0)",
            "v_lshlrev_b32 v2, 2, v0",
            "buffer_store_dword v8, v2, s[0:3], 0 offen",
        ], "64 threads with ADD_TID_ENABLE: wave32 wraps lanes 32..63 to 0..31, wave64 uses 0..63",
             threads=64, words=64, wave_dependent=True),
    ]

    return cases


def _hw_kernel_lines(case, idx, wave):
    """Wrap `case.body` for hw_oracle (raw=True, s[2:3] = output buffer base address)."""
    lbl = f".Lseed_{idx}"
    lbl_done = f".Lseed_done_{idx}"
    # Build RAW V# in s[0:3] from s[2:3], compute DATA_BASE = base + DATA_OFF in s[16:17], and seed DATA_BASE on
    # lane 0 (preserving EXEC with wave-sized save/restore so all threads participate in the body afterwards).
    if wave == 32:
        save_exec = ["s_mov_b32 s28, exec_lo", "v_cmpx_eq_u32_e32 0, v0"]
        restore_exec = ["s_mov_b32 exec_lo, s28"]
    else:
        save_exec = ["s_mov_b64 s[28:29], exec", "v_cmpx_eq_u32_e32 0, v0"]
        restore_exec = ["s_mov_b64 exec, s[28:29]"]
    lines = [
        "s_mov_b32 s0, s2",
        "s_and_b32 s1, s3, 0xffff",
        "s_mov_b32 s2, 0xffffffff",
        f"s_mov_b32 s3, {OUT_V_W3:#x}",
        f"s_add_u32 s16, s0, {DATA_OFF:#x}",
        "s_addc_u32 s17, s1, 0",
        "s_and_b32 s17, s17, 0xffff",
        *save_exec,
        "s_mov_b32 s20, 0",
        f"{lbl}:",
        f"s_cmp_ge_u32 s20, {DATA_DWORDS * 4:#x}",
        f"s_cbranch_scc1 {lbl_done}",
        f"s_add_u32 s21, s20, {SEED_BASE:#x}",
        "v_mov_b32 v30, s21",
        f"s_add_u32 s22, s20, {DATA_OFF:#x}",
        "buffer_store_dword v30, off, s[0:3], s22",
        "s_add_u32 s20, s20, 4",
        f"s_branch {lbl}",
        f"{lbl_done}:",
        "s_waitcnt vmcnt(0)",
        *restore_exec,
    ]
    if case.threads > 1:
        lines.append("s_barrier")
    lines.extend(case.body)
    lines.append("s_endpgm")
    return lines


def _translate_case(work, case, wave=32):
    os.makedirs(os.path.join(work, "disasm"), exist_ok=True)
    os.makedirs(os.path.join(work, "fssm"), exist_ok=True)
    name = f"cs_swz_probe_{case.name}_w{wave}"
    src = os.path.join(work, "disasm", name + ".s")
    wave_tag = "s_mov_b64 s[28:29], exec" if wave == 64 else "s_mov_b32 s28, exec_lo"
    body = [
        wave_tag,
        f"s_add_u32 s16, s0, {DATA_OFF:#x}",
        "s_addc_u32 s17, s1, 0",
        "s_and_b32 s17, s17, 0xffff",
        *case.body,
        "s_endpgm",
    ]
    with open(src, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"; {name}\n" + "".join(f"\t{ln}\n" for ln in body))
    with open(os.path.join(work, "fssm", name + ".fssm"), "wb") as f:
        f.write(_cs_fssm(case.threads))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        spv = rdna2spirv.translate(src)
    memory = {a: 0 for a in range(0, DATA_OFF, 4)}
    for off in range(0, DATA_DWORDS * 4, 4):
        memory[DATA_OFF + off] = SEED_BASE + off
    spirv_mini_interp.run_workgroup(
        spv, case.threads, wave,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % wave},
        push_constants=[0, 0, 0xFFFFFFFF, OUT_V_W3] + [0] * 28,
        memory=memory,
    )
    return [memory.get(4 * i, 0) for i in range(case.words)]


def run_translator(cases, waves=(32, 64)):
    per_wave = {}
    with tempfile.TemporaryDirectory(prefix="buf_swz_probe_") as work:
        for wave in waves:
            out = {}
            for case in cases:
                if wave != waves[0] and not case.wave_dependent:
                    out[case.name] = per_wave[waves[0]][case.name]
                    continue
                try:
                    out[case.name] = _translate_case(work, case, wave=wave)
                except Exception as e:
                    out[case.name] = f"error: {type(e).__name__}: {e}"
            per_wave[wave] = out
    return per_wave


def run_hw(cases, arch=None, waves=(32, 64)):
    from hw_oracle import asm, runner  # noqa: E402
    oracle = runner.open_oracle(arch=arch)
    try:
        device = oracle.device
        per_wave = {}
        for wave in waves:
            kernels = [
                asm.Kernel(_hw_kernel_lines(case, idx, wave), wave=wave, threads=case.threads,
                           words=OUT_WORDS, raw=True)
                for idx, case in enumerate(cases)
            ]
            res = oracle.run(kernels, words=OUT_WORDS)
            per_wave[wave] = {case.name: (r[:case.words] if isinstance(r, list) else r) for case, r in zip(cases, res)}
        return device, per_wave
    finally:
        oracle.close()


def _fmt_words(w):
    if isinstance(w, list):
        if len(w) <= 8:
            return " ".join(f"0x{x:08x}" for x in w)
        head = " ".join(f"0x{x:08x}" for x in w[:4])
        tail = " ".join(f"0x{x:08x}" for x in w[-4:])
        return f"[{len(w)} words: {head} ... {tail}]"
    return str(w)


def self_test():
    cases = build_cases()
    names = [c.name for c in cases]
    assert len(names) == len(set(names)), "duplicate case names"
    for name in KNOWN_DIVERGENCES:
        assert name in names, f"unknown KNOWN_DIVERGENCES key: {name}"
    res = run_translator(cases, waves=(32, 64))
    for w in (32, 64):
        errors = {k: v for k, v in res[w].items() if not isinstance(v, list)}
        assert not errors, f"translator wave{w} errors: {errors}"
    assert res[32]["tid_wave32_vs_wave64"] != res[64]["tid_wave32_vs_wave64"], "wave32/64 ADD_TID must differ"
    print(f"self-test: {len(cases)} cases across {len({c.group for c in cases})} groups translated and ran on CPU OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", choices=("translator", "hw", "both"), default="translator")
    ap.add_argument("--wave", choices=("32", "64", "both"), default="both")
    ap.add_argument("--group", action="append", default=[], help="run only named group(s)")
    ap.add_argument("--hw-arch", default=None, help="override GPU architecture for hw_oracle")
    ap.add_argument("--list-groups", action="store_true")
    ap.add_argument("--strict", action="store_true", help="treat known translator divergences as failures")
    ap.add_argument("--json", dest="json_out", default=None, help="write raw results to a JSON file")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    cases = build_cases()
    groups = []
    for c in cases:
        if c.group not in groups:
            groups.append(c.group)
    if args.list_groups:
        for g in groups:
            n = sum(1 for c in cases if c.group == g)
            print(f"{g:<22} ({n} cases)")
        return 0

    if args.group:
        unknown = [g for g in args.group if g not in groups]
        if unknown:
            print(f"unknown group(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        want = set(args.group)
        cases = [c for c in cases if c.group in want]

    waves = (32, 64) if args.wave == "both" else (int(args.wave),)
    tr_res = run_translator(cases, waves=waves) if args.oracle in ("translator", "both") else None
    hw_device, hw_res = None, None
    if args.oracle in ("hw", "both"):
        from hw_oracle import runner  # noqa: E402
        try:
            hw_device, hw_res = run_hw(cases, arch=args.hw_arch, waves=waves)
        except runner.Unavailable as e:
            print(f"hardware oracle unavailable: {e}", file=sys.stderr)
            return 77
        print(f"hardware oracle: {hw_device} (waves: {','.join(str(w) for w in waves)})")

    mismatches = 0
    known_diffs = 0
    wave_diffs = 0
    records = []
    for c in cases:
        t32 = tr_res[32][c.name] if tr_res and 32 in tr_res else None
        t64 = tr_res[64][c.name] if tr_res and 64 in tr_res else None
        h32 = hw_res[32][c.name] if hw_res and 32 in hw_res else None
        h64 = hw_res[64][c.name] if hw_res and 64 in hw_res else None
        status = "OK"
        tag = None
        if not c.wave_dependent and h32 is not None and h64 is not None and h32 != h64:
            status = "WAVE_DIFF"
            wave_diffs += 1
        elif c.wave_dependent and h32 is not None and h64 is not None and h32 == h64:
            status = "WAVE_SAME"
            mismatches += 1
        elif any(v is not None and not isinstance(v, list) for v in (t32, t64, h32, h64)):
            status = "ERROR"
            mismatches += 1
        else:
            bad = ((t32 is not None and h32 is not None and t32 != h32)
                   or (t64 is not None and h64 is not None and t64 != h64))
            if bad:
                hw_ref = h32 if h32 is not None else h64
                known = KNOWN_DIVERGENCES.get(c.name)
                if known and hw_ref == known[1]:
                    status = "KNOWN_DIFF"
                    tag = known[0]
                    known_diffs += 1
                else:
                    status = "MISMATCH"
                    mismatches += 1
        records.append({
            "group": c.group, "name": c.name, "note": c.note, "wave_dependent": c.wave_dependent,
            "status": status, "known_tag": tag,
            "translator_w32": t32, "translator_w64": t64, "hw_w32": h32, "hw_w64": h64,
        })
        if status != "OK" or args.oracle != "both":
            label = f"{status}:{tag}" if tag else status
            print(f"[{label:<22}] {c.group}/{c.name}: {c.note}")
            if h32 is not None:
                print(f"                         hw_w32: {_fmt_words(h32)}")
            if h64 is not None and h64 != h32:
                print(f"                         hw_w64: {_fmt_words(h64)}")
            if t32 is not None:
                print(f"                         tr_w32: {_fmt_words(t32)}")
            if t64 is not None and t64 != t32:
                print(f"                         tr_w64: {_fmt_words(t64)}")

    total = len(cases)
    matched = total - mismatches - known_diffs - wave_diffs
    print(f"summary: {total} cases: {matched} match, {known_diffs} known divergence, "
          f"{mismatches} unexpected mismatch, {wave_diffs} wave32/64 diff")

    if args.json_out:
        out_path = os.path.abspath(args.json_out)
        if os.path.isdir(out_path):
            print(f"ERROR: --json path is a directory: {out_path}", file=sys.stderr)
            return 2
        parent = os.path.dirname(out_path)
        try:
            os.makedirs(parent, exist_ok=True)
            with open(out_path, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"device": hw_device, "waves": list(waves), "cases": records}, f, indent=2)
                f.write("\n")
        except OSError as exc:
            print(f"ERROR: cannot write {out_path}: {exc.strerror or exc}", file=sys.stderr)
            return 2

    if mismatches or wave_diffs or (args.strict and known_diffs):
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
