"""Semantic tests for RDNA2 s_cselect_b64 on 64-bit scalar data pairs vs lane masks.

Covers:
  * 64-bit SGPR data pairs with distinct low/high dwords for SCC=0 and SCC=1
  * Non-mask integer constants (positive, hex literals, small negative sign-extended,
    and mixed 0/-1 with a 64-bit SGPR data pair)
  * Destination aliasing (exact D==S0, D==S1, D==S0==S1, and overlapping pairs)
  * SCC preservation across s_cselect_b64
  * Overwriting a former lane-mask SGPR pair with 64-bit data clears stale mask state
  * Regression checks for lane-mask s_cselect_b64 (vcc, exec, lane-mask SGPR pairs)
    across wave32, wave64, compute workgroups, and --wave64-emulate translation

Run directly:
    python -I tests/semantic/test_scselect_b64_data.py -v
"""
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402
from support import STUB_DXBC, mrt0_words, rdna2spirv  # noqa: E402

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

_TMP_DIR = None
_PS_PATH = None
_CS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    dis = os.path.join(_TMP_DIR.name, "disasm")
    fssm = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(dis)
    os.mkdir(fssm)
    _PS_PATH = os.path.join(dis, "ps_cselect64.s")
    _CS_PATH = os.path.join(dis, "cs_cselect64.s")
    with open(os.path.join(_TMP_DIR.name, "ps_cselect64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "cs_cselect64.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    # Minimal FSSM header: 4 user SGPRs (V# in s[0:3]), TGID_X in s4, 64x1x1 threads.
    hdr = bytearray(0x148)
    struct.pack_into("<4I", hdr, 0x138, 0x8088, 64, 1, 1)
    with open(os.path.join(fssm, "cs_cselect64.fssm"), "wb") as f:
        f.write(hdr)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = "; ps_cselect64\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_PS_PATH)


def translate_cs(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = "; cs_cselect64\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_CS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_CS_PATH)


def run_ps(lines):
    spv = translate_ps(lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    return mrt0_words(spv), spv


def run_ps_wave(lines, wave):
    spv = translate_ps(lines + ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])
    res = wi.run_wave(spv, wave=wave)
    return [o[next(iter(o))] if o is not None else None for o in res["outputs"]], spv


def scc_line(scc):
    return "s_cmp_eq_u32 1, 1" if scc else "s_cmp_eq_u32 0, 1"


def load_pair_lines(reg, val64):
    lo = val64 & MASK32
    hi = (val64 >> 32) & MASK32
    return [f"s_mov_b32 s{reg}, {lo:#x}", f"s_mov_b32 s{reg + 1}, {hi:#x}"]


def ref_cselect_b64(s0, s1, scc):
    return (s0 if scc else s1) & MASK64


class DataPairSelectionTests(unittest.TestCase):
    def test_sgpr_data_pairs_distinct_dwords(self):
        pairs = [
            (0x55667788_11223344, 0xDDEEFF00_99AABBCC),
            (0x12345678_00000000, 0xFFFFFFFF_87654321),
            (0x00000002_80000001, 0x80000000_7FFFFFFE),
        ]
        for a64, b64 in pairs:
            for scc in (0, 1):
                with self.subTest(a=hex(a64), b=hex(b64), scc=scc):
                    want = ref_cselect_b64(a64, b64, scc)
                    lines = (
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [
                            scc_line(scc),
                            "s_cselect_b64 s[2:3], s[4:5], s[6:7]",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s2",
                            "v_mov_b32 v1, s3",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ]
                    )
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, scc])

    def test_non_mask_constants(self):
        cases = [
            ("42", "-2", 42, 0xFFFFFFFF_FFFFFFFE),
            ("0x10", "0x80000000", 0x10, 0x00000000_80000000),
            ("-16", "64", 0xFFFFFFFF_FFFFFFF0, 64),
            ("0x12345678", "0", 0x12345678, 0),
        ]
        for s0_asm, s1_asm, s0_val, s1_val in cases:
            for scc in (0, 1):
                with self.subTest(s0=s0_asm, s1=s1_asm, scc=scc):
                    want = ref_cselect_b64(s0_val, s1_val, scc)
                    words, _ = run_ps([
                        scc_line(scc),
                        f"s_cselect_b64 s[2:3], {s0_asm}, {s1_asm}",
                        "s_cselect_b32 s10, 1, 0",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s3",
                        "v_mov_b32 v2, s10",
                        "v_mov_b32 v3, 0",
                    ])
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, scc])

    def test_mixed_zero_minus_one_with_data_pair(self):
        data64 = 0x55667788_11223344
        for imm_asm, imm_val in (("0", 0), ("-1", MASK64)):
            for data_first in (True, False):
                for scc in (0, 1):
                    s0_asm, s1_asm = ("s[4:5]", imm_asm) if data_first else (imm_asm, "s[4:5]")
                    s0_val, s1_val = (data64, imm_val) if data_first else (imm_val, data64)
                    with self.subTest(s0=s0_asm, s1=s1_asm, scc=scc):
                        want = ref_cselect_b64(s0_val, s1_val, scc)
                        lines = load_pair_lines(4, data64) + [
                            scc_line(scc),
                            f"s_cselect_b64 s[2:3], {s0_asm}, {s1_asm}",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s2",
                            "v_mov_b32 v1, s3",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ]
                        words, _ = run_ps(lines)
                        self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, scc])

        # Mixed -1/0 via s_mov_b64 (lane_masks EITHER) and a plain 64-bit SGPR data pair.
        for imm_asm, imm_val in (("0", 0), ("-1", MASK64)):
            for scc in (0, 1):
                with self.subTest(mov_imm=imm_asm, scc=scc):
                    want = ref_cselect_b64(imm_val, data64, scc)
                    lines = load_pair_lines(4, data64) + [
                        f"s_mov_b64 s[8:9], {imm_asm}",
                        scc_line(scc),
                        "s_cselect_b64 s[2:3], s[8:9], s[4:5]",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s3",
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v3, 0",
                    ]
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

        # Pure 0 / -1 selection into an SGPR pair.
        for scc in (0, 1):
            want = ref_cselect_b64(MASK64, 0, scc)
            words, _ = run_ps([
                scc_line(scc),
                "s_cselect_b64 s[2:3], -1, 0",
                "v_mov_b32 v0, s2",
                "v_mov_b32 v1, s3",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ])
            self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_destination_aliasing(self):
        a64 = 0x55667788_11223344
        b64 = 0xDDEEFF00_99AABBCC
        for src0, src1, s0_val, s1_val in (
            ("s[4:5]", "s[6:7]", a64, b64),
            ("s[6:7]", "s[4:5]", b64, a64),
            ("s[4:5]", "s[4:5]", a64, a64),
        ):
            for scc in (0, 1):
                with self.subTest(src0=src0, src1=src1, scc=scc):
                    want = ref_cselect_b64(s0_val, s1_val, scc)
                    lines = (
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [
                            scc_line(scc),
                            f"s_cselect_b64 s[4:5], {src0}, {src1}",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s4",
                            "v_mov_b32 v1, s5",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ]
                    )
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, scc])

        # Partial register overlap: dst s[4:5] overlaps both s[3:4] and s[5:6].
        s3, s4, s5, s6 = 0x10203040, 0x50607080, 0x90A0B0C0, 0xD0E0F001
        p34 = (s4 << 32) | s3
        p56 = (s6 << 32) | s5
        for src0, src1, s0_val, s1_val in (
            ("s[3:4]", "s[5:6]", p34, p56),
            ("s[5:6]", "s[3:4]", p56, p34),
        ):
            for scc in (0, 1):
                with self.subTest(overlap=(src0, src1), scc=scc):
                    want = ref_cselect_b64(s0_val, s1_val, scc)
                    words, _ = run_ps([
                        f"s_mov_b32 s3, {s3:#x}",
                        f"s_mov_b32 s4, {s4:#x}",
                        f"s_mov_b32 s5, {s5:#x}",
                        f"s_mov_b32 s6, {s6:#x}",
                        scc_line(scc),
                        f"s_cselect_b64 s[4:5], {src0}, {src1}",
                        "v_mov_b32 v0, s4",
                        "v_mov_b32 v1, s5",
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v3, 0",
                    ])
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_scc_preservation(self):
        # Chained s_cselect_b64 + s_addc_u32 verify SCC is left untouched.
        a64 = 0x22222222_11111111
        b64 = 0x44444444_33333333
        for scc in (0, 1):
            with self.subTest(scc=scc):
                want = ref_cselect_b64(a64, b64, scc)
                lines = (
                    load_pair_lines(4, a64)
                    + load_pair_lines(6, b64)
                    + [
                        scc_line(scc),
                        "s_cselect_b64 s[2:3], s[4:5], s[6:7]",
                        "s_cselect_b64 s[8:9], s[6:7], s[4:5]",
                        "s_addc_u32 s10, 10, 20",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s3",
                        "v_mov_b32 v2, s8",
                        "v_mov_b32 v3, s10",
                    ]
                )
                words, _ = run_ps(lines)
                want_second = ref_cselect_b64(b64, a64, scc)
                self.assertEqual(
                    words,
                    [want & MASK32, (want >> 32) & MASK32, want_second & MASK32, 30 + scc],
                )

    def test_overwriting_stale_lane_mask_pair_clears_mask_state(self):
        # s[4:5] first holds EXEC (lane mask), then is overwritten by 64-bit data via s_cselect_b64.
        # Subsequent s_bcnt1_i32_b64 and s_cmp_eq_u64 must treat s[4:5] as 64-bit data, not a wave mask.
        a64 = 0x0F0F0000_00005555  # popcount: 8 + 8 = 16
        b64 = 0x00FF0000_00000007  # popcount: 8 + 3 = 11
        for wave, mbcnt in ((32, ["v_mbcnt_lo_u32_b32 v4, -1, 0"]),
                            (64, ["v_mbcnt_lo_u32_b32 v4, -1, 0", "v_mbcnt_hi_u32_b32 v4, -1, v4"])):
            for scc in (0, 1):
                with self.subTest(wave=wave, scc=scc):
                    want = ref_cselect_b64(a64, b64, scc)
                    want_cnt = want.bit_count()
                    lines = (
                        mbcnt
                        + [
                            "v_cmp_gt_u32_e32 vcc, 8, v4",
                            "s_mov_b64 s[4:5], vcc",  # s[4:5] is a lane mask before s_cselect_b64
                        ]
                        + load_pair_lines(6, a64)
                        + load_pair_lines(8, b64)
                        + [
                            scc_line(scc),
                            "s_cselect_b64 s[4:5], s[6:7], s[8:9]",
                            "s_bcnt1_i32_b64 s10, s[4:5]",
                            "s_cmp_eq_u64 s[4:5], s[6:7]",
                            "s_cselect_b32 s11, 1, 0",
                            "v_mov_b32 v0, s4",
                            "v_mov_b32 v1, s5",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, s11",
                        ]
                    )
                    outs, _ = run_ps_wave(lines, wave=wave)
                    expected = (want & MASK32, (want >> 32) & MASK32, want_cnt, scc)
                    for lane in range(wave):
                        self.assertEqual(outs[lane], expected, f"wave={wave} lane={lane}")

    def test_null_source_reads_zero(self):
        data64 = 0x55667788_11223344
        for s0_asm, s1_asm, s0_val, s1_val in (
            ("s[4:5]", "null", data64, 0),
            ("null", "s[4:5]", 0, data64),
        ):
            for scc in (0, 1):
                with self.subTest(s0=s0_asm, s1=s1_asm, scc=scc):
                    want = ref_cselect_b64(s0_val, s1_val, scc)
                    lines = load_pair_lines(4, data64) + [
                        scc_line(scc),
                        f"s_cselect_b64 s[2:3], {s0_asm}, {s1_asm}",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s3",
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v3, 0",
                    ]
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:2], [want & MASK32, (want >> 32) & MASK32])

    def test_unsupported_destination_or_source_rejected(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_ps(["s_cselect_b64 m0, s[4:5], s[6:7]", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_ps(["s_cselect_b64 s[2:3], s4, s[6:7]", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


class LaneMaskPreservationTests(unittest.TestCase):
    def test_wave32_vcc_exec_and_sgpr_masks(self):
        for scc in (0, 1):
            with self.subTest(scc=scc):
                # vcc = lanes 0..9 (10 lanes), s[2:3] = lanes 16..31 (16 lanes).
                lines = [
                    "v_mbcnt_lo_u32_b32 v4, -1, 0",
                    "v_cmp_gt_u32_e32 vcc, 10, v4",
                    "v_cmp_lt_u32_e64 s[2:3], 15, v4",
                    scc_line(scc),
                    "s_cselect_b64 s[4:5], vcc, s[2:3]",
                    "s_cselect_b32 s10, 1, 0",         # SCC preserved across mask s_cselect_b64
                    "s_bcnt1_i32_b64 s8, s[4:5]",
                    "v_mov_b32 v1, 0xAA",
                    "s_mov_b32 s12, 0",
                    "s_cmp_eq_u32 s10, 1",
                    "s_cselect_b64 exec, s[4:5], exec",
                    "v_mov_b32 v1, 0x55",
                    "s_cmp_eq_u32 1, 1",
                    "s_cselect_b64 exec, -1, exec",
                    "v_mov_b32 v0, s8",
                    "v_mov_b32 v2, s10",
                    "v_mov_b32 v3, s4",
                ]
                outs, _ = run_ps_wave(lines, wave=32)
                want_cnt = 10 if scc else 16
                for lane in range(32):
                    in_mask = (lane < 10) if scc else (lane >= 16)
                    want_v1 = 0x55 if (scc == 0 or in_mask) else 0xAA
                    want_s4 = MASK32 if in_mask else 0
                    self.assertEqual(outs[lane], (want_cnt, want_v1, scc, want_s4), f"lane={lane}")

    def test_wave64_vcc_exec_and_sgpr_masks(self):
        for scc in (0, 1):
            with self.subTest(scc=scc):
                # vcc = lanes 0..19 (20 lanes), s[2:3] = lanes 36..63 (28 lanes, upper half only).
                lines = [
                    "v_mbcnt_lo_u32_b32 v4, -1, 0",
                    "v_mbcnt_hi_u32_b32 v4, -1, v4",
                    "v_cmp_gt_u32_e32 vcc, 20, v4",
                    "v_cmp_lt_u32_e64 s[2:3], 35, v4",
                    scc_line(scc),
                    "s_cselect_b64 s[4:5], vcc, s[2:3]",
                    "s_cselect_b32 s10, 1, 0",         # SCC preserved
                    "s_bcnt1_i32_b64 s8, s[4:5]",
                    # Also test vcc as destination (with 0/-1 sources) and vcc as source into SGPR pair.
                    scc_line(scc),
                    "s_cselect_b64 vcc, -1, 0",
                    "s_cmp_eq_u32 1, 1",
                    "s_cselect_b64 s[6:7], vcc, 0",
                    "s_bcnt1_i32_b64 s9, s[6:7]",
                    "v_mov_b32 v1, 0x11",
                    "s_cmp_eq_u32 1, 1",
                    "s_cselect_b64 exec, s[4:5], 0",
                    "v_mov_b32 v1, 0x22",
                    "s_cmp_eq_u32 0, 1",
                    "s_cselect_b64 exec, 0, -1",       # scc=0 -> select -1 -> restore all 64 lanes
                    "v_mov_b32 v0, s8",
                    "v_mov_b32 v2, s9",
                    "v_mov_b32 v3, s10",
                ]
                outs, _ = run_ps_wave(lines, wave=64)
                want_cnt = 20 if scc else 28
                want_vcc_cnt = 64 if scc else 0
                for lane in range(64):
                    in_mask = (lane < 20) if scc else (lane >= 36)
                    want_v1 = 0x22 if in_mask else 0x11
                    self.assertEqual(outs[lane], (want_cnt, want_v1, want_vcc_cnt, scc), f"lane={lane}")

    def test_wave64_compute_and_emulate_translation(self):
        # Compute shader using both wave64 lane-mask s_cselect_b64 and 64-bit data-pair s_cselect_b64.
        lines = [
            "v_cmp_lt_u32_e32 vcc, 39, v0",            # lanes 40..63 (24 lanes)
            "s_mov_b32 s6, 0x11223344",
            "s_mov_b32 s7, 0x55667788",
            "s_mov_b32 s8, 0x99aabbcc",
            "s_mov_b32 s9, 0xddeeff00",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 s[10:11], s[6:7], s[8:9]",  # data select -> s10=0x11223344, s11=0x55667788
            "s_cselect_b64 s[12:13], vcc, 0",          # mask select -> 24 lanes set
            "s_bcnt1_i32_b64 s14, s[12:13]",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_cmp_eq_u32 1, 1",
            "s_cselect_b64 exec, vcc, 0",
            "v_mov_b32 v1, s10",
            "v_mov_b32 v2, s11",
            "v_mov_b32 v3, s14",
            "v_mov_b32 v4, 0",
            "buffer_store_dwordx4 v[1:4], v4, s[0:3], 0",
            "s_cselect_b64 exec, -1, exec",
            "s_endpgm",
        ]
        spv = translate_cs(lines)
        mem = si.run_workgroup(
            spv,
            64,
            64,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 28,
            memory={},
        )
        self.assertEqual([mem.get(0), mem.get(4), mem.get(8), mem.get(12)], [0x11223344, 0x55667788, 24, 0])

        # Also verify --wave64-emulate translation succeeds cleanly on the same wave64 compute shader.
        prev = rdna2spirv.WAVE64_EMULATE
        try:
            rdna2spirv.WAVE64_EMULATE = True
            spv_emu = translate_cs(lines)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        self.assertIn("OpStore %x_it", spv_emu)


class NegativeControlTests(unittest.TestCase):
    def test_sabotaging_high_dword_select_breaks_result(self):
        # The fix emits two OpSelect %uint instructions (lo and hi); replacing the second
        # with the first simulates the old mask_dst bug (duplicating lo into hi).
        lines = (
            load_pair_lines(4, 0x55667788_11223344)
            + load_pair_lines(6, 0xDDEEFF00_99AABBCC)
            + [
                scc_line(1),
                "s_cselect_b64 s[2:3], s[4:5], s[6:7]",
                "v_mov_b32 v0, s2",
                "v_mov_b32 v1, s3",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ]
        )
        words, spv = run_ps(lines)
        self.assertEqual(words[:2], [0x11223344, 0x55667788])
        m = re.search(
            r"(%\w+) = OpINotEqual %bool %\w+ %c0\s+"
            r"(%\w+) = OpSelect %uint \1 %\w+ %\w+\s+"
            r"(%\w+) = OpSelect %uint \1 %\w+ %\w+",
            spv,
        )
        self.assertIsNotNone(m)
        lo_id, hi_id = m.group(2), m.group(3)
        bad = re.sub(
            rf"{re.escape(hi_id)} = OpSelect %uint [^\n]+",
            f"{hi_id} = OpCopyObject %uint {lo_id}",
            spv,
            count=1,
        )
        self.assertNotEqual(bad, spv)
        self.assertEqual(mrt0_words(bad)[:2], [0x11223344, 0x11223344])


if __name__ == "__main__":
    unittest.main()
