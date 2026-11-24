"""Semantic, validation, and interpreter tests for global_load_dwordx3 and global_store_dwordx3.

RDNA2 ISA (FLAT/GLOBAL):
  global_load_dwordx3 v[D:D+2], vaddr, saddr|off [offset:N]
    Loads three consecutive 32-bit dwords starting at byte address A into v[D..D+2] (low address first)
    and leaves v[D+3] untouched. Inactive lanes issue no loads and preserve v[D..D+2].
  global_store_dwordx3 vaddr, v[D:D+2], saddr|off [offset:N]
    Stores three consecutive 32-bit dwords from v[D..D+2] starting at byte address A and leaves
    adjacent dwords untouched. Inactive lanes issue no stores.
  Address A = ((saddr: 64-bit SGPR pair + zext32(vaddr)) | (off: 64-bit VGPR pair)) + sext12(offset),
  translated to host memory by adding push-constant heap_base (dwords 32..33).

Covers both addressing modes, signed 12-bit offsets (-2048..2047 and hex spellings), 64-bit carry/borrow
across 4 GiB boundaries, heap_base translation, vaddr/vdst and vaddr/vdata register aliasing, strict 3-dword
boundaries, partial-EXEC wave32 and wave64 execution, fragment and compute stages, helper metadata
(read_before_write_vgprs / vgpr_dest_registers), rejected forms, a negative control, and spirv-val
validation (native wave32, native wave64, and --wave64-emulate). Standard library only; no game files.
"""
import os
import shutil
import struct
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402  (adds LEONIDA_TOOLS_DIR / tools to sys.path)
import rdna2spirv  # noqa: E402
import shader_ir  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")

FSSM_CS_W32 = bytearray(0x148)
struct.pack_into("<4I", FSSM_CS_W32, 0x138, 0x88, 32, 1, 1)
FSSM_CS_W32 = bytes(FSSM_CS_W32)

FSSM_CS_W64 = bytearray(0x148)
struct.pack_into("<4I", FSSM_CS_W64, 0x138, 0x00, 64, 1, 1)
FSSM_CS_W64 = bytes(FSSM_CS_W64)

BASE = 0x00000003_00004800
SENTINEL = 0xEEEEEEEE
GUARD_LO = 0xA5A50001
GUARD_HI = 0xA5A50002
V13_INIT = 0xCAFEBABE


def tagged(addr):
    """Distinct 32-bit word derived from the full 64-bit byte address."""
    return (0x50000000 ^ ((addr >> 20) & 0x0FFF0000) ^ (addr & 0xFFFF)) & M32


def seed_window(center, lo_off=-0x1010, hi_off=0x1010):
    return {a: tagged(a) for a in range(center + lo_off, center + hi_off, 4)}


def translate_cs(lines, *, wave64=False, emulate=False):
    fssm = FSSM_CS_W64 if wave64 else FSSM_CS_W32
    prev = rdna2spirv.WAVE64_EMULATE
    rdna2spirv.WAVE64_EMULATE = emulate
    try:
        return support.translate_lines("cs_global_x3", [*lines, "s_endpgm"], fssm=fssm)
    finally:
        rdna2spirv.WAVE64_EMULATE = prev


def translate_ps(lines):
    return support.translate_lines(
        "ps_global_x3",
        [*lines, "exp mrt0 v10, v11, v12, v13 done vm", "s_endpgm"],
    )


def run_cs(lines, memory, *, nlanes=1, wave64=False, push_constants=None):
    spv = translate_cs(lines, wave64=wave64)
    mem = dict(memory)
    pc = [0] * 36 if push_constants is None else list(push_constants)
    wsize = 64 if wave64 else 32
    si.run_workgroup(
        spv,
        nlanes,
        wsize,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc,
        memory=mem,
    )
    return mem


def run_ps(lines, memory, *, push_constants=None):
    spv = translate_ps(lines)
    mem = dict(memory)
    pc = [0] * 36 if push_constants is None else list(push_constants)
    words = support.mrt0_words(spv, push_constants=pc, memory=mem)
    return words, mem


def setup_base_regs(base):
    """Initialize s[20:21] + v1 and v[4:5] to `base` (split 0x800 + (base - 0x800) for saddr)."""
    saddr = base - 0x800
    return [
        f"s_mov_b32 s20, {saddr & M32:#x}",
        f"s_mov_b32 s21, {(saddr >> 32) & M32:#x}",
        "v_mov_b32_e32 v1, 0x800",
        f"v_mov_b32_e32 v4, {base & M32:#x}",
        f"v_mov_b32_e32 v5, {(base >> 32) & M32:#x}",
    ]


class GlobalLoadDwordx3Tests(unittest.TestCase):
    def test_load_both_modes_offsets_and_vgpr_boundary(self):
        mem = seed_window(BASE)
        for mode in ("v1, s[20:21]", "v[4:5], off"):
            for spelled, off in (("-4", -4), ("-2048", -2048), ("2044", 2044), ("0", 0), ("-0x4", -4), ("0x7fc", 2044)):
                mod = f" offset:{spelled}" if spelled != "0" else ""
                lines = [
                    *setup_base_regs(BASE),
                    "v_mov_b32_e32 v10, 0x11111111",
                    "v_mov_b32_e32 v11, 0x22222222",
                    "v_mov_b32_e32 v12, 0x33333333",
                    f"v_mov_b32_e32 v13, {V13_INIT:#x}",
                    f"global_load_dwordx3 v[10:12], {mode}{mod}",
                    "s_waitcnt vmcnt(0)",
                ]
                want = BASE + off
                expected = [tagged(want), tagged(want + 4), tagged(want + 8), V13_INIT]
                with self.subTest(mode=mode, spelled=spelled):
                    words, _ = run_ps(lines, mem)
                    self.assertEqual(words, expected)

    def test_load_offset_2047_aligned(self):
        # Base one past a dword boundary makes (base + 2047) dword-aligned, exercising the maximum positive 12-bit offset.
        base = BASE + 1
        want = base + 2047
        mem = {want + 4 * i: tagged(want + 4 * i) for i in range(4)}
        for mode in ("v1, s[20:21]", "v[4:5], off"):
            lines = [
                *setup_base_regs(base),
                f"v_mov_b32_e32 v13, {V13_INIT:#x}",
                f"global_load_dwordx3 v[10:12], {mode} offset:2047",
                "s_waitcnt vmcnt(0)",
            ]
            with self.subTest(mode=mode):
                words, _ = run_ps(lines, mem)
                self.assertEqual(words, [tagged(want), tagged(want + 4), tagged(want + 8), V13_INIT])

    def test_load_vaddr_aliasing_vdst(self):
        mem = seed_window(BASE)
        # `off` mode: 64-bit vaddr v[4:5] overlaps vdst v[4:6].
        want_off = BASE - 4
        words_off, _ = run_ps([
            *setup_base_regs(BASE),
            f"v_mov_b32_e32 v7, {V13_INIT:#x}",
            "global_load_dwordx3 v[4:6], v[4:5], off offset:-4",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32_e32 v10, v4",
            "v_mov_b32_e32 v11, v5",
            "v_mov_b32_e32 v12, v6",
            "v_mov_b32_e32 v13, v7",
        ], mem)
        self.assertEqual(words_off, [tagged(want_off), tagged(want_off + 4), tagged(want_off + 8), V13_INIT])

        # `saddr` mode: 32-bit vaddr v1 overlaps vdst v[1:3].
        want_saddr = BASE + 8
        words_saddr, _ = run_ps([
            *setup_base_regs(BASE),
            f"v_mov_b32_e32 v4, {V13_INIT:#x}",
            "global_load_dwordx3 v[1:3], v1, s[20:21] offset:8",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32_e32 v10, v1",
            "v_mov_b32_e32 v11, v2",
            "v_mov_b32_e32 v12, v3",
            "v_mov_b32_e32 v13, v4",
        ], mem)
        self.assertEqual(words_saddr, [tagged(want_saddr), tagged(want_saddr + 4), tagged(want_saddr + 8), V13_INIT])

    def test_load_64bit_carry_borrow_and_heap_base(self):
        heap_base = 0x00000042_00000000
        pc = [0] * 32 + [heap_base & M32, (heap_base >> 32) & M32, 0, 0]

        # Borrow across 4 GiB boundary: s[20:21] = 0x3_00000000, v1 = 0, offset:-4 -> 0x2_FFFFFFFC.
        va_borrow = 0x00000002_FFFFFFFC
        mem_borrow = {heap_base + va_borrow + 4 * i: 0xB000 + i for i in range(3)}
        words_borrow, _ = run_ps([
            "s_mov_b32 s20, 0x0",
            "s_mov_b32 s21, 0x3",
            "v_mov_b32_e32 v1, 0x0",
            f"v_mov_b32_e32 v13, {V13_INIT:#x}",
            "global_load_dwordx3 v[10:12], v1, s[20:21] offset:-4",
            "s_waitcnt vmcnt(0)",
        ], mem_borrow, push_constants=pc)
        self.assertEqual(words_borrow, [0xB000, 0xB001, 0xB002, V13_INIT])

        # Carry across 4 GiB boundary: v[4:5] = 0x3_FFFFFFF8, offset:8 -> 0x4_00000000.
        va_carry = 0x00000004_00000000
        mem_carry = {heap_base + va_carry + 4 * i: 0xC000 + i for i in range(3)}
        words_carry, _ = run_ps([
            "v_mov_b32_e32 v4, 0xfffffff8",
            "v_mov_b32_e32 v5, 0x3",
            f"v_mov_b32_e32 v13, {V13_INIT:#x}",
            "global_load_dwordx3 v[10:12], v[4:5], off offset:8",
            "s_waitcnt vmcnt(0)",
        ], mem_carry, push_constants=pc)
        self.assertEqual(words_carry, [0xC000, 0xC001, 0xC002, V13_INIT])

    def test_load_inactive_lanes_keep_vdst_wave32_and_wave64(self):
        for wave64, nlanes, exec_lines, active_lanes in (
            (False, 8, ["s_mov_b32 exec_lo, 0x25"], {0, 2, 5}),
            (True, 64, ["s_mov_b32 exec_lo, 0x00000005", "s_mov_b32 exec_hi, 0x00000002"], {0, 2, 33}),
        ):
            src_base = 0x10000
            out_base = 0x20000
            mem = {}
            for lane in range(nlanes):
                for k in range(3):
                    mem[src_base + lane * 16 + 4 * k] = (0x70000000 | (lane << 8) | k)
            lines = [
                f"s_mov_b32 s20, {src_base:#x}",
                "s_mov_b32 s21, 0",
                "v_mbcnt_lo_u32_b32 v0, -1, 0",
                *(["v_mbcnt_hi_u32_b32 v0, -1, v0"] if wave64 else []),
                "v_lshlrev_b32_e32 v1, 4, v0",
                "v_mov_b32_e32 v10, 0xdead0000",
                "v_mov_b32_e32 v11, 0xdead0001",
                "v_mov_b32_e32 v12, 0xdead0002",
                f"v_mov_b32_e32 v13, {V13_INIT:#x}",
                *exec_lines,
                "global_load_dwordx3 v[10:12], v1, s[20:21]",
                "s_waitcnt vmcnt(0)",
                *(["s_mov_b64 exec, -1"] if wave64 else ["s_mov_b32 exec_lo, -1"]),
                f"v_add_nc_u32_e32 v4, {out_base:#x}, v1",
                "v_mov_b32_e32 v5, 0",
                "global_store_dwordx4 v[4:5], v[10:13], off",
            ]
            with self.subTest(wave64=wave64):
                res = run_cs(lines, mem, nlanes=nlanes, wave64=wave64)
                for lane in range(nlanes):
                    got = [res.get(out_base + lane * 16 + 4 * k) for k in range(4)]
                    if lane in active_lanes:
                        want = [0x70000000 | (lane << 8) | k for k in range(3)] + [V13_INIT]
                    else:
                        want = [0xDEAD0000, 0xDEAD0001, 0xDEAD0002, V13_INIT]
                    self.assertEqual(got, want, f"lane {lane}")


class GlobalStoreDwordx3Tests(unittest.TestCase):
    def test_store_both_modes_offsets_and_strict_bounds(self):
        payload = (0xA1B2C3D4, 0x0BADC0DE, 0x13572468, 0x99999999)
        for mode in ("v1, v[6:8], s[20:21]", "v[4:5], v[6:8], off"):
            for spelled, off in (("-4", -4), ("-2048", -2048), ("2044", 2044), ("0", 0), ("-0x4", -4), ("0x7fc", 2044)):
                want = BASE + off
                mem = {
                    want - 4: GUARD_LO,
                    want: SENTINEL,
                    want + 4: SENTINEL,
                    want + 8: SENTINEL,
                    want + 12: GUARD_HI,
                }
                mod = f" offset:{spelled}" if spelled != "0" else ""
                lines = [
                    *setup_base_regs(BASE),
                    *(f"v_mov_b32_e32 v{6 + k}, {payload[k]:#x}" for k in range(4)),
                    f"global_store_dwordx3 {mode}{mod}",
                    "v_mov_b32_e32 v10, 1",
                    "v_mov_b32_e32 v11, 2",
                    "v_mov_b32_e32 v12, 3",
                    "v_mov_b32_e32 v13, 4",
                ]
                with self.subTest(mode=mode, spelled=spelled):
                    _, res = run_ps(lines, mem)
                    self.assertEqual(
                        [res[want - 4], res[want], res[want + 4], res[want + 8], res[want + 12]],
                        [GUARD_LO, payload[0], payload[1], payload[2], GUARD_HI],
                    )

    def test_store_offset_2047_aligned(self):
        base = BASE + 1
        want = base + 2047
        payload = (0x11112222, 0x33334444, 0x55556666)
        for mode in ("v1, v[6:8], s[20:21]", "v[4:5], v[6:8], off"):
            mem = {want - 4: GUARD_LO, want + 12: GUARD_HI}
            lines = [
                *setup_base_regs(base),
                *(f"v_mov_b32_e32 v{6 + k}, {payload[k]:#x}" for k in range(3)),
                f"global_store_dwordx3 {mode} offset:2047",
            ]
            with self.subTest(mode=mode):
                res = run_cs(lines, mem)
                self.assertEqual(
                    [res[want - 4], res[want], res[want + 4], res[want + 8], res[want + 12]],
                    [GUARD_LO, payload[0], payload[1], payload[2], GUARD_HI],
                )

    def test_store_vaddr_aliasing_vdata(self):
        want = BASE + 4
        mem = {want - 4: GUARD_LO, want + 12: GUARD_HI}
        res = run_cs([
            f"v_mov_b32_e32 v4, {BASE & M32:#x}",
            f"v_mov_b32_e32 v5, {(BASE >> 32) & M32:#x}",
            "v_mov_b32_e32 v6, 0x778899aa",
            "global_store_dwordx3 v[4:5], v[4:6], off offset:4",
        ], mem)
        self.assertEqual(
            [res[want - 4], res[want], res[want + 4], res[want + 8], res[want + 12]],
            [GUARD_LO, BASE & M32, (BASE >> 32) & M32, 0x778899AA, GUARD_HI],
        )

    def test_store_64bit_carry_borrow_heap_base_and_roundtrip(self):
        heap_base = 0x00000019_00000000
        pc = [0] * 32 + [heap_base & M32, (heap_base >> 32) & M32, 0, 0]
        # Store at (0x4_00000000 - 4) via offset:-4, then load back at (0x3_FFFFFFF4 + 8) via offset:8.
        va = 0x00000003_FFFFFFFC
        mem = {heap_base + va - 4: GUARD_LO, heap_base + va + 12: GUARD_HI}
        words, res = run_ps([
            "s_mov_b32 s20, 0x0",
            "s_mov_b32 s21, 0x4",
            "v_mov_b32_e32 v1, 0x0",
            "v_mov_b32_e32 v6, 0x10203040",
            "v_mov_b32_e32 v7, 0x50607080",
            "v_mov_b32_e32 v8, 0x90a0b0c0",
            "global_store_dwordx3 v1, v[6:8], s[20:21] offset:-4",
            "s_waitcnt vmcnt(0)",
            "v_mov_b32_e32 v4, 0xfffffff4",
            "v_mov_b32_e32 v5, 0x3",
            f"v_mov_b32_e32 v13, {V13_INIT:#x}",
            "global_load_dwordx3 v[10:12], v[4:5], off offset:8",
            "s_waitcnt vmcnt(0)",
        ], mem, push_constants=pc)
        self.assertEqual(words, [0x10203040, 0x50607080, 0x90A0B0C0, V13_INIT])
        self.assertEqual(res[heap_base + va - 4], GUARD_LO)
        self.assertEqual(res[heap_base + va + 12], GUARD_HI)

    def test_store_inactive_lanes_leave_memory_untouched_wave32_and_wave64(self):
        for wave64, nlanes, exec_lines, active_lanes in (
            (False, 8, ["s_mov_b32 exec_lo, 0x4a"], {1, 3, 6}),
            (True, 64, ["s_mov_b32 exec_lo, 0x0000000a", "s_mov_b32 exec_hi, 0x00000010"], {1, 3, 36}),
        ):
            out_base = 0x30000
            mem = {}
            for lane in range(nlanes):
                for k in range(4):
                    mem[out_base + lane * 16 + 4 * k] = 0xDEAD0000 | (lane << 4) | k
            lines = [
                "v_mbcnt_lo_u32_b32 v0, -1, 0",
                *(["v_mbcnt_hi_u32_b32 v0, -1, v0"] if wave64 else []),
                "v_lshlrev_b32_e32 v1, 4, v0",
                f"s_mov_b32 s20, {out_base:#x}",
                "s_mov_b32 s21, 0",
                "v_add_nc_u32_e32 v6, 0x100, v0",
                "v_add_nc_u32_e32 v7, 0x200, v0",
                "v_add_nc_u32_e32 v8, 0x300, v0",
                "v_mov_b32_e32 v9, 0x99999999",
                *exec_lines,
                "global_store_dwordx3 v1, v[6:8], s[20:21]",
            ]
            with self.subTest(wave64=wave64):
                res = run_cs(lines, mem, nlanes=nlanes, wave64=wave64)
                for lane in range(nlanes):
                    got = [res[out_base + lane * 16 + 4 * k] for k in range(4)]
                    if lane in active_lanes:
                        want = [0x100 + lane, 0x200 + lane, 0x300 + lane, 0xDEAD0000 | (lane << 4) | 3]
                    else:
                        want = [0xDEAD0000 | (lane << 4) | k for k in range(4)]
                    self.assertEqual(got, want, f"lane {lane}")


class GlobalDwordx3MetadataAndValidationTests(unittest.TestCase):
    def test_read_before_write_and_dest_registers(self):
        load_saddr = shader_ir.parse_line("\tglobal_load_dwordx3 v[10:12], v1, s[20:21] offset:-4")
        load_off = shader_ir.parse_line("\tglobal_load_dwordx3 v[10:12], v[4:5], off")
        store_saddr = shader_ir.parse_line("\tglobal_store_dwordx3 v1, v[6:8], s[20:21]")
        store_off = shader_ir.parse_line("\tglobal_store_dwordx3 v[4:5], v[6:8], off")

        self.assertEqual(list(rdna2spirv.vgpr_dest_registers(load_saddr)), [10, 11, 12])
        self.assertEqual(rdna2spirv.read_before_write_vgprs([load_saddr]), {1})
        self.assertEqual(rdna2spirv.read_before_write_vgprs([load_off]), {4, 5})
        self.assertEqual(rdna2spirv.read_before_write_vgprs([store_saddr]), {1, 6, 7, 8})
        self.assertEqual(rdna2spirv.read_before_write_vgprs([store_off]), {4, 5, 6, 7, 8})
        # Loading into v[10:12] defines v10..v12, leaving v13 read-before-write when read afterwards.
        read_after = [
            load_saddr,
            shader_ir.parse_line("\tv_add_nc_u32_e32 v20, v10, v12"),
            shader_ir.parse_line("\tv_mov_b32_e32 v21, v13"),
        ]
        self.assertEqual(rdna2spirv.read_before_write_vgprs(read_after), {1, 13})

    def test_rejected_forms_raise_unsupported(self):
        bad_lines = (
            "global_load_dwordx3 v[10:12], v1, s[20:21] lds",
            "global_load_dwordx3 v[10:12], v1, s[20:21] tfe",
            "global_load_dwordx3 v[10:12], v1, s[20:21] d16",
            "global_store_dwordx3 v[4:5], v[6:8], off tfe",
            "global_load_dwordx3 v[10:12], v[4:5], off offset:4096",
            "global_load_dwordx3 v[10:12], v[4:5], off offset:0x1000",
            "global_store_dwordx3 v[4:5], v[6:8], off offset:-2049",
            "global_load_dwordx3 v[10:12], v4, off",
            "global_store_dwordx3 v1, v[6:8], s20",
        )
        for line in bad_lines:
            with self.subTest(line=line):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_cs([line])

    @unittest.skipUnless(SPIRV_AS and SPIRV_VAL, "requires spirv-as and spirv-val")
    def test_spirv_val_wave32_wave64_and_emulate(self):
        cs_lines = [
            "s_mov_b32 s20, 0x1000",
            "s_mov_b32 s21, 0",
            "v_mbcnt_lo_u32_b32 v0, -1, 0",
            "v_lshlrev_b32_e32 v1, 4, v0",
            "v_cmp_lt_u32_e32 vcc, v0, 3",
            "s_and_saveexec_b64 s[6:7], vcc",
            "global_load_dwordx3 v[10:12], v1, s[20:21] offset:-4",
            "s_waitcnt vmcnt(0)",
            "v_add_nc_u32_e32 v4, 0x2000, v1",
            "v_mov_b32_e32 v5, 0",
            "global_store_dwordx3 v[4:5], v[10:12], off offset:16",
        ]
        modules = (
            ("ps", translate_ps([
                *setup_base_regs(BASE),
                "v_mov_b32_e32 v13, 0",
                "global_load_dwordx3 v[10:12], v1, s[20:21] offset:-4",
                "s_waitcnt vmcnt(0)",
                "global_store_dwordx3 v[4:5], v[10:12], off offset:2044",
            ])),
            ("cs_w32", translate_cs(cs_lines[:2] + cs_lines[2:4] + [
                "v_cmp_lt_u32_e32 vcc_lo, v0, 3",
                "s_and_saveexec_b32 s6, vcc_lo",
                *cs_lines[6:],
            ], wave64=False)),
            ("cs_w64", translate_cs(
                cs_lines[:3] + ["v_mbcnt_hi_u32_b32 v0, -1, v0"] + cs_lines[3:],
                wave64=True, emulate=False,
            )),
            ("cs_w64_emulate", translate_cs(
                cs_lines[:3] + ["v_mbcnt_hi_u32_b32 v0, -1, v0"] + cs_lines[3:],
                wave64=True, emulate=True,
            )),
        )
        for label, spv in modules:
            with self.subTest(mode=label):
                assembled = subprocess.run(
                    [SPIRV_AS, "--target-env", "vulkan1.2", "-", "-o", "-"],
                    input=spv.encode("utf-8"),
                    capture_output=True,
                )
                self.assertEqual(assembled.returncode, 0, assembled.stderr.decode("utf-8", "replace"))
                validated = subprocess.run(
                    [SPIRV_VAL, "--target-env", "vulkan1.2", "-"],
                    input=assembled.stdout,
                    capture_output=True,
                )
                self.assertEqual(validated.returncode, 0, validated.stderr.decode("utf-8", "replace"))

    def test_negative_control_truncating_width_fails(self):
        mem = seed_window(BASE)
        # Translating x2 instead of x3 cannot populate the third dword (v12 stays at its pre-seeded sentinel).
        words_load, _ = run_ps([
            *setup_base_regs(BASE),
            "v_mov_b32_e32 v12, 0xdeadbeef",
            f"v_mov_b32_e32 v13, {V13_INIT:#x}",
            "global_load_dwordx2 v[10:11], v1, s[20:21]",
            "s_waitcnt vmcnt(0)",
        ], mem)
        self.assertNotEqual(words_load[:3], [tagged(BASE), tagged(BASE + 4), tagged(BASE + 8)])

        _, mem_store = run_ps([
            *setup_base_regs(BASE),
            "v_mov_b32_e32 v6, 0x11",
            "v_mov_b32_e32 v7, 0x22",
            "v_mov_b32_e32 v8, 0x33",
            "global_store_dwordx2 v[4:5], v[6:7], off",
        ], {BASE: SENTINEL, BASE + 4: SENTINEL, BASE + 8: SENTINEL})
        self.assertNotEqual(mem_store[BASE + 8], 0x33)


if __name__ == "__main__":
    unittest.main()
