"""Semantic tests for RDNA2 64-bit scalar bitwise ops on 64-bit data pairs vs lane masks.

Covers s_and_b64, s_or_b64, s_xor_b64, s_andn2_b64, s_orn2_b64, s_nand_b64, s_nor_b64, s_xnor_b64, and s_not_b64:
  * Distinct lo/hi 64-bit SGPR data pairs (verifying D.lo, D.hi, and SCC)
  * Positive/negative immediates (0xFE, -2, -16, 0x80000000, 0, -1) and null src/dst
  * SCC four-corner matrix: (lo==0, hi==0), (lo!=0, hi==0), (lo==0, hi!=0), (lo!=0, hi!=0)
  * Destination aliasing (exact D==S0, D==S1, D==S0==S1, and partial overlap s[4:5] vs s[3:4]/s[5:6])
  * Overwriting a former lane-mask SGPR pair with 64-bit data clears stale mask state
  * Lane-mask regression checks across wave32, wave64, compute workgroups, and --wave64-emulate
    (verifying wave-wide SCC voting and EXEC/VCC/SGPR/null mask behavior)

Run directly:
    python -I tests/semantic/test_salu64_bitwise_data.py -v
"""
import atexit
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
_EXIT_CLEANUP = False


def setUpModule():
    global _TMP_DIR, _PS_PATH, _CS_PATH, _EXIT_CLEANUP
    if _TMP_DIR is not None:
        # translate_ps() and the importers' helpers (test_interp_merge_check, test_mask_src_sgpr_const,
        # test_saveexec_variants) may have created it already; replacing it would leak the first one
        return
    _TMP_DIR = tempfile.TemporaryDirectory()
    # A module that imports a helper from here and runs after this module's teardown recreates the folder through
    # translate_ps(), and unittest never tears this module down again. Registered once, after the first
    # TemporaryDirectory, so it runs before tempfile's own exit finalizer (atexit is last in, first out) and that
    # finalizer stays quiet; tearDownModule() cleans whichever folder is live at exit.
    if not _EXIT_CLEANUP:
        atexit.register(tearDownModule)
        _EXIT_CLEANUP = True
    dis = os.path.join(_TMP_DIR.name, "disasm")
    fssm = os.path.join(_TMP_DIR.name, "fssm")
    os.mkdir(dis)
    os.mkdir(fssm)
    _PS_PATH = os.path.join(dis, "ps_salu64_bit.s")
    _CS_PATH = os.path.join(dis, "cs_salu64_bit.s")
    with open(os.path.join(_TMP_DIR.name, "ps_salu64_bit.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    with open(os.path.join(_TMP_DIR.name, "cs_salu64_bit.dxbc"), "wb") as f:
        f.write(STUB_DXBC)
    # Minimal FSSM header: 4 user SGPRs (V# in s[0:3]), TGID_X in s4, 64x1x1 threads.
    hdr = bytearray(0x148)
    struct.pack_into("<4I", hdr, 0x138, 0x8088, 64, 1, 1)
    with open(os.path.join(fssm, "cs_salu64_bit.fssm"), "wb") as f:
        f.write(hdr)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = "; ps_salu64_bit\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_PS_PATH)


def translate_cs(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = "; cs_salu64_bit\n" + "".join(f"\t{ln}\n" for ln in lines)
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


def load_pair_lines(reg, val64):
    lo = val64 & MASK32
    hi = (val64 >> 32) & MASK32
    return [f"s_mov_b32 s{reg}, {lo:#x}", f"s_mov_b32 s{reg + 1}, {hi:#x}"]


BIN_OPS = {
    "s_and_b64": lambda x, y: (x & y) & MASK64,
    "s_or_b64": lambda x, y: (x | y) & MASK64,
    "s_xor_b64": lambda x, y: (x ^ y) & MASK64,
    "s_andn2_b64": lambda x, y: (x & (~y & MASK64)) & MASK64,
    "s_orn2_b64": lambda x, y: (x | (~y & MASK64)) & MASK64,
    "s_nand_b64": lambda x, y: ~(x & y) & MASK64,
    "s_nor_b64": lambda x, y: ~(x | y) & MASK64,
    "s_xnor_b64": lambda x, y: ~(x ^ y) & MASK64,
}


class DataPairBitwiseTests(unittest.TestCase):
    def test_all_six_ops_distinct_dwords(self):
        pairs = [
            (0x55667788_11223344, 0x0F0F00FF_F0F0FF00),
            (0xA5A55A5A_01234567, 0x5A5AA5A5_89ABCDEF),
            (0x12345678_00000000, 0xFFFFFFFF_87654321),
        ]
        for op, fn in BIN_OPS.items():
            for a64, b64 in pairs:
                with self.subTest(op=op, a=hex(a64), b=hex(b64)):
                    want = fn(a64, b64)
                    want_scc = 1 if want != 0 else 0
                    lines = (
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [
                            f"{op} s[2:3], s[4:5], s[6:7]",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s2",
                            "v_mov_b32 v1, s3",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ]
                    )
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

        for a64 in (0x55667788_11223344, 0xFFFFFFFF_00000000, 0x00000000_FFFFFFFF, 0, MASK64):
            with self.subTest(op="s_not_b64", a=hex(a64)):
                want = (~a64) & MASK64
                want_scc = 1 if want != 0 else 0
                lines = load_pair_lines(4, a64) + [
                    "s_not_b64 s[2:3], s[4:5]",
                    "s_cselect_b32 s10, 1, 0",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, s10",
                    "v_mov_b32 v3, 0",
                ]
                words, _ = run_ps(lines)
                self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

    def test_scc_four_corners(self):
        # Verify SCC = ((rlo | rhi) != 0) across all four (lo, hi) zero/nonzero combinations.
        cases = {
            "s_and_b64": [
                (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF, 0),
                (0x00000000_00000008, 0xFFFFFFFF_0000000F, 1),
                (0x00000010_00000000, 0x000000F0_FFFFFFFF, 1),  # only hi non-zero!
                (0x80000000_00000001, 0x80000000_00000001, 1),
            ],
            "s_or_b64": [
                (0x00000000_00000000, 0x00000000_00000000, 0),
                (0x00000000_00000004, 0x00000000_00000000, 1),
                (0x00000000_00000000, 0x00000020_00000000, 1),  # only hi non-zero!
                (0x00000001_00000000, 0x00000000_00000001, 1),
            ],
            "s_xor_b64": [
                (0x12345678_9ABCDEF0, 0x12345678_9ABCDEF0, 0),
                (0x12345678_9ABCDEF0, 0x12345678_9ABCDEF1, 1),
                (0x12345678_9ABCDEF0, 0x12345679_9ABCDEF0, 1),  # only hi non-zero!
                (0x00000001_00000001, 0x00000000_00000000, 1),
            ],
            "s_andn2_b64": [
                (0x0000000F_0000000F, 0x000000FF_000000FF, 0),
                (0x0000000F_0000001F, 0x000000FF_0000000F, 1),
                (0x00000100_0000000F, 0x000000FF_0000000F, 1),  # only hi non-zero!
                (0x00000100_00000100, 0x00000000_00000000, 1),
            ],
            "s_orn2_b64": [
                (0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF, 0),
                (0x00000000_00000001, 0xFFFFFFFF_FFFFFFFF, 1),
                (0x00000001_00000000, 0xFFFFFFFF_FFFFFFFF, 1),  # only hi non-zero!
                (0x00000001_00000001, 0xFFFFFFFF_FFFFFFFF, 1),
            ],
            "s_nand_b64": [
                (MASK64, MASK64, 0),
                (0xFFFFFFFF_FFFFFFFE, MASK64, 1),
                (0xFFFFFFFE_FFFFFFFF, MASK64, 1),  # only hi non-zero!
                (0, 0, 1),
            ],
            "s_nor_b64": [
                (0xFFFFFFFF_00000000, 0x00000000_FFFFFFFF, 0),
                (0xFFFFFFFF_FFFFFFFE, 0, 1),
                (0xFFFFFFFE_FFFFFFFF, 0, 1),  # only hi non-zero!
                (0, 0, 1),
            ],
            "s_xnor_b64": [
                (0x12345678_9ABCDEF0, 0xEDCBA987_6543210F, 0),
                (0x12345678_9ABCDEF0, 0xEDCBA987_6543210E, 1),
                (0x12345678_9ABCDEF0, 0xEDCBA986_6543210F, 1),  # only hi non-zero!
                (0x12345678_9ABCDEF0, 0x12345678_9ABCDEF0, 1),
            ],
        }
        for op, items in cases.items():
            fn = BIN_OPS[op]
            for a64, b64, want_scc in items:
                with self.subTest(op=op, a=hex(a64), b=hex(b64)):
                    want = fn(a64, b64)
                    self.assertEqual(1 if want != 0 else 0, want_scc)
                    lines = (
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [
                            # Poison SCC with the opposite value beforehand to ensure the op overwrites it.
                            "s_cmp_eq_u32 0, 1" if want_scc else "s_cmp_eq_u32 1, 1",
                            f"{op} s[2:3], s[4:5], s[6:7]",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s2",
                            "v_mov_b32 v1, s3",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ]
                    )
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

        # s_not_b64 four corners: ~MASK64 == 0, ~0xFFFFFFFF_FFFFFFFE == lo-only, ~0xFFFFFFFE_FFFFFFFF == hi-only
        for a64, want_scc in (
            (0xFFFFFFFF_FFFFFFFF, 0),
            (0xFFFFFFFF_FFFFFFFE, 1),
            (0xFFFFFFFE_FFFFFFFF, 1),  # only hi non-zero!
            (0x12345678_9ABCDEF0, 1),
        ):
            with self.subTest(op="s_not_b64", a=hex(a64)):
                want = (~a64) & MASK64
                lines = load_pair_lines(4, a64) + [
                    "s_cmp_eq_u32 0, 1" if want_scc else "s_cmp_eq_u32 1, 1",
                    "s_not_b64 s[2:3], s[4:5]",
                    "s_cselect_b32 s10, 1, 0",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, s10",
                    "v_mov_b32 v3, 0",
                ]
                words, _ = run_ps(lines)
                self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

    def test_immediates_sign_extension_and_zero_minus_one(self):
        data64 = 0x55667788_112233FF
        imms = [
            ("0xfe", 0x00000000_000000FE),
            ("-2", 0xFFFFFFFF_FFFFFFFE),
            ("-16", 0xFFFFFFFF_FFFFFFF0),
            ("0x80000000", 0x00000000_80000000),
            ("0", 0),
            ("-1", MASK64),
        ]
        for op, fn in BIN_OPS.items():
            for imm_asm, imm_val in imms:
                for data_first in (True, False):
                    s0_asm, s1_asm = ("s[4:5]", imm_asm) if data_first else (imm_asm, "s[4:5]")
                    s0_val, s1_val = (data64, imm_val) if data_first else (imm_val, data64)
                    with self.subTest(op=op, s0=s0_asm, s1=s1_asm):
                        want = fn(s0_val, s1_val)
                        want_scc = 1 if want != 0 else 0
                        lines = load_pair_lines(4, data64) + [
                            f"{op} s[2:3], {s0_asm}, {s1_asm}",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s2",
                            "v_mov_b32 v1, s3",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ]
                        words, _ = run_ps(lines)
                        self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

        # s_not_b64 on immediates (positive, negative, 0, -1).
        for imm_asm, imm_val in imms:
            with self.subTest(op="s_not_b64", imm=imm_asm):
                want = (~imm_val) & MASK64
                want_scc = 1 if want != 0 else 0
                words, _ = run_ps([
                    f"s_not_b64 s[2:3], {imm_asm}",
                    "s_cselect_b32 s10, 1, 0",
                    "v_mov_b32 v0, s2",
                    "v_mov_b32 v1, s3",
                    "v_mov_b32 v2, s10",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

        # Pure 0 / -1 binary ops into an SGPR pair (lane_masks EITHER).
        for op, fn in BIN_OPS.items():
            for s0_asm, s0_val in (("0", 0), ("-1", MASK64)):
                for s1_asm, s1_val in (("0", 0), ("-1", MASK64)):
                    with self.subTest(op=op, pure=(s0_asm, s1_asm)):
                        want = fn(s0_val, s1_val)
                        want_scc = 1 if want != 0 else 0
                        words, _ = run_ps([
                            f"{op} s[2:3], {s0_asm}, {s1_asm}",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s2",
                            "v_mov_b32 v1, s3",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ])
                        self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

    def test_null_source_and_null_destination(self):
        data64 = 0x55667788_00000000  # hi-only non-zero!
        for op, fn in BIN_OPS.items():
            for s0_asm, s1_asm, s0_val, s1_val in (
                ("s[4:5]", "null", data64, 0),
                ("null", "s[4:5]", 0, data64),
            ):
                with self.subTest(op=op, s0=s0_asm, s1=s1_asm):
                    want = fn(s0_val, s1_val)
                    want_scc = 1 if want != 0 else 0
                    lines = load_pair_lines(4, data64) + [
                        f"{op} s[2:3], {s0_asm}, {s1_asm}",
                        "s_cselect_b32 s10, 1, 0",
                        "v_mov_b32 v0, s2",
                        "v_mov_b32 v1, s3",
                        "v_mov_b32 v2, s10",
                        "v_mov_b32 v3, 0",
                    ]
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

        # null destination discards the result while still updating SCC (including hi-only non-zero).
        for op, a64, b64, want_scc in (
            ("s_and_b64", 0x00000010_00000000, 0x00000010_00000000, 1),
            ("s_and_b64", 0x00000010_00000000, 0x00000001_FFFFFFFF, 0),
            ("s_xor_b64", 0x12345678_9ABCDEF0, 0x12345678_9ABCDEF0, 0),
            ("s_xor_b64", 0x12345678_9ABCDEF0, 0x12345679_9ABCDEF0, 1),
            ("s_andn2_b64", 0x00000010_00000000, 0x00000000_FFFFFFFF, 1),
            ("s_orn2_b64", 0x00000000_00000000, 0xFFFFFFFF_FFFFFFFF, 0),
        ):
            with self.subTest(null_dst=op, a=hex(a64), b=hex(b64)):
                lines = (
                    load_pair_lines(4, a64)
                    + load_pair_lines(6, b64)
                    + [
                        "s_cmp_eq_u32 0, 1" if want_scc else "s_cmp_eq_u32 1, 1",
                        f"{op} null, s[4:5], s[6:7]",
                        "s_cselect_b32 s10, 1, 0",
                        "v_mov_b32 v0, s10",
                        "v_mov_b32 v1, 0",
                        "v_mov_b32 v2, 0",
                        "v_mov_b32 v3, 0",
                    ]
                )
                words, _ = run_ps(lines)
                self.assertEqual(words[0], want_scc)

        for a64, want_scc in ((MASK64, 0), (0xFFFFFFFE_FFFFFFFF, 1)):
            with self.subTest(null_dst="s_not_b64", a=hex(a64)):
                lines = load_pair_lines(4, a64) + [
                    "s_cmp_eq_u32 0, 1" if want_scc else "s_cmp_eq_u32 1, 1",
                    "s_not_b64 null, s[4:5]",
                    "s_cselect_b32 s10, 1, 0",
                    "v_mov_b32 v0, s10",
                    "v_mov_b32 v1, 0",
                    "v_mov_b32 v2, 0",
                    "v_mov_b32 v3, 0",
                ]
                words, _ = run_ps(lines)
                self.assertEqual(words[0], want_scc)

        # s_not_b64 with null source (~0 == MASK64, SCC == 1).
        words, _ = run_ps([
            "s_not_b64 s[2:3], null",
            "s_cselect_b32 s10, 1, 0",
            "v_mov_b32 v0, s2",
            "v_mov_b32 v1, s3",
            "v_mov_b32 v2, s10",
            "v_mov_b32 v3, 0",
        ])
        self.assertEqual(words[:3], [MASK32, MASK32, 1])

    def test_destination_aliasing(self):
        a64 = 0x55667788_11223344
        b64 = 0xDDEEFF00_99AABBCC
        for op, fn in BIN_OPS.items():
            for src0, src1, s0_val, s1_val in (
                ("s[4:5]", "s[6:7]", a64, b64),
                ("s[6:7]", "s[4:5]", b64, a64),
                ("s[4:5]", "s[4:5]", a64, a64),
            ):
                with self.subTest(op=op, src0=src0, src1=src1):
                    want = fn(s0_val, s1_val)
                    want_scc = 1 if want != 0 else 0
                    lines = (
                        load_pair_lines(4, a64)
                        + load_pair_lines(6, b64)
                        + [
                            f"{op} s[4:5], {src0}, {src1}",
                            "s_cselect_b32 s10, 1, 0",
                            "v_mov_b32 v0, s4",
                            "v_mov_b32 v1, s5",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, 0",
                        ]
                    )
                    words, _ = run_ps(lines)
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

        # Partial register overlap: dst s[4:5] overlaps both s[3:4] and s[5:6].
        s3, s4, s5, s6 = 0x10203040, 0x50607080, 0x90A0B0C0, 0xD0E0F001
        p34 = (s4 << 32) | s3
        p56 = (s6 << 32) | s5
        for op, fn in BIN_OPS.items():
            for src0, src1, s0_val, s1_val in (
                ("s[3:4]", "s[5:6]", p34, p56),
                ("s[5:6]", "s[3:4]", p56, p34),
            ):
                with self.subTest(op=op, overlap=(src0, src1)):
                    want = fn(s0_val, s1_val)
                    want_scc = 1 if want != 0 else 0
                    words, _ = run_ps([
                        f"s_mov_b32 s3, {s3:#x}",
                        f"s_mov_b32 s4, {s4:#x}",
                        f"s_mov_b32 s5, {s5:#x}",
                        f"s_mov_b32 s6, {s6:#x}",
                        f"{op} s[4:5], {src0}, {src1}",
                        "s_cselect_b32 s10, 1, 0",
                        "v_mov_b32 v0, s4",
                        "v_mov_b32 v1, s5",
                        "v_mov_b32 v2, s10",
                        "v_mov_b32 v3, 0",
                    ])
                    self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

        # Unary s_not_b64 aliasing: exact (s[4:5] <- s[4:5]) and partial overlaps (s[3:4], s[5:6]).
        for src, s_val in (("s[4:5]", (s5 << 32) | s4), ("s[3:4]", p34), ("s[5:6]", p56)):
            with self.subTest(op="s_not_b64", overlap=src):
                want = (~s_val) & MASK64
                want_scc = 1 if want != 0 else 0
                words, _ = run_ps([
                    f"s_mov_b32 s3, {s3:#x}",
                    f"s_mov_b32 s4, {s4:#x}",
                    f"s_mov_b32 s5, {s5:#x}",
                    f"s_mov_b32 s6, {s6:#x}",
                    f"s_not_b64 s[4:5], {src}",
                    "s_cselect_b32 s10, 1, 0",
                    "v_mov_b32 v0, s4",
                    "v_mov_b32 v1, s5",
                    "v_mov_b32 v2, s10",
                    "v_mov_b32 v3, 0",
                ])
                self.assertEqual(words[:3], [want & MASK32, (want >> 32) & MASK32, want_scc])

    def test_overwriting_stale_lane_mask_pair_clears_mask_state(self):
        # s[4:5] first holds vcc (lane mask), then is overwritten with 64-bit data by s_and_b64 or s_not_b64.
        # Subsequent s_bcnt1_i32_b64 must count the 64-bit data popcount, not active wave lanes.
        a64 = 0x0F0F0000_00005555
        b64 = 0x00FF0000_000000FF
        cases = [
            ("s_and_b64 s[4:5], s[6:7], s[8:9]", a64 & b64),
            ("s_not_b64 s[4:5], s[6:7]", (~a64) & MASK64),
        ]
        for insn, want in cases:
            want_cnt = want.bit_count()
            for wave, mbcnt in ((32, ["v_mbcnt_lo_u32_b32 v4, -1, 0"]),
                                (64, ["v_mbcnt_lo_u32_b32 v4, -1, 0", "v_mbcnt_hi_u32_b32 v4, -1, v4"])):
                with self.subTest(insn=insn, wave=wave):
                    lines = (
                        mbcnt
                        + [
                            "v_cmp_gt_u32_e32 vcc, 8, v4",
                            "s_mov_b64 s[4:5], vcc",  # s[4:5] is a lane mask before overwrite
                        ]
                        + load_pair_lines(6, a64)
                        + load_pair_lines(8, b64)
                        + [
                            insn,
                            "s_cselect_b32 s11, 1, 0",
                            "s_bcnt1_i32_b64 s10, s[4:5]",
                            "v_mov_b32 v0, s4",
                            "v_mov_b32 v1, s5",
                            "v_mov_b32 v2, s10",
                            "v_mov_b32 v3, s11",
                        ]
                    )
                    outs, _ = run_ps_wave(lines, wave=wave)
                    expected = (want & MASK32, (want >> 32) & MASK32, want_cnt, 1 if want else 0)
                    for lane in range(wave):
                        self.assertEqual(outs[lane], expected, f"wave={wave} lane={lane}")

    def test_unsupported_destination_or_source_rejected(self):
        for bad in (
            "s_and_b64 m0, s[4:5], s[6:7]",
            "s_and_b64 s[2:3], s4, s[6:7]",
            "s_not_b64 m0, s[4:5]",
            "s_not_b64 s[2:3], s4",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_ps([bad, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"])


class LaneMaskPreservationTests(unittest.TestCase):
    def test_wave32_all_six_ops_and_scc_vote(self):
        # A = lanes 20..29 (10 lanes; lane 0 is OFF so per-lane SCC would differ from wave SCC!),
        # B = lanes 25..31 (7 lanes).
        #   and(A, B)   = 25..29 (5 lanes)  SCC=1
        #   or(A, B)    = 20..31 (12 lanes) SCC=1
        #   xor(A, B)   = 20..24, 30..31 (7 lanes) SCC=1
        #   andn2(A, B) = 20..24 (5 lanes)  SCC=1
        #   orn2(A, B)  = 0..29 (30 lanes)  SCC=1
        #   not(A)      = 0..19, 30..31 (22 lanes) SCC=1
        #   and(A, not(A)) = empty (0 lanes) SCC=0
        cases = [
            ("s_and_b64 s[6:7], s[2:3], s[4:5]", lambda l: 25 <= l <= 29, 5, 1),
            ("s_or_b64 s[6:7], s[2:3], s[4:5]", lambda l: 20 <= l <= 31, 12, 1),
            ("s_xor_b64 s[6:7], s[2:3], s[4:5]", lambda l: (20 <= l <= 24) or (30 <= l <= 31), 7, 1),
            ("s_andn2_b64 s[6:7], s[2:3], s[4:5]", lambda l: 20 <= l <= 24, 5, 1),
            ("s_orn2_b64 s[6:7], s[2:3], s[4:5]", lambda l: l <= 29, 30, 1),
            ("s_not_b64 s[6:7], s[2:3]", lambda l: not (20 <= l <= 29), 22, 1),
            ("s_nand_b64 s[6:7], s[2:3], s[4:5]", lambda l: not (25 <= l <= 29), 27, 1),
            ("s_nor_b64 s[6:7], s[2:3], s[4:5]", lambda l: l <= 19, 20, 1),
            ("s_xnor_b64 s[6:7], s[2:3], s[4:5]", lambda l: l <= 19 or 25 <= l <= 29, 25, 1),
        ]
        for insn, pred, want_cnt, want_scc in cases:
            with self.subTest(insn=insn):
                lines = [
                    "v_mbcnt_lo_u32_b32 v4, -1, 0",
                    "v_cmp_ge_u32_e64 s[2:3], v4, 20",
                    "v_cmp_lt_u32_e64 s[8:9], v4, 30",
                    "s_and_b64 s[2:3], s[2:3], s[8:9]",  # A = 20..29
                    "v_cmp_ge_u32_e64 s[4:5], v4, 25",   # B = 25..31
                    insn,
                    "s_cselect_b32 s10, 1, 0",
                    "s_bcnt1_i32_b64 s11, s[6:7]",
                    "v_mov_b32 v0, s11",
                    "v_mov_b32 v1, s10",
                    "v_mov_b32 v2, s6",
                    "v_mov_b32 v3, 0",
                ]
                outs, _ = run_ps_wave(lines, wave=32)
                for lane in range(32):
                    want_bit = MASK32 if pred(lane) else 0
                    self.assertEqual(outs[lane], (want_cnt, want_scc, want_bit, 0), f"lane={lane}")

        # Empty mask intersection -> SCC=0 across all lanes, and null dst on lane-mask path sets SCC.
        lines = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_cmp_eq_u32_e32 vcc, 27, v4",
            "s_and_b64 null, vcc, exec",
            "s_cselect_b32 s10, 1, 0",
            "s_not_b64 s[2:3], vcc",
            "s_and_b64 null, vcc, s[2:3]",
            "s_cselect_b32 s11, 1, 0",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, s11",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        outs, _ = run_ps_wave(lines, wave=32)
        for lane in range(32):
            self.assertEqual(outs[lane], (1, 0, 0, 0), f"lane={lane}")

    def test_wave64_all_six_ops_exec_vcc_and_scc_vote(self):
        # A = lanes 40..55 (16 lanes, strictly in upper half so lanes 0..31 are all OFF!),
        # B = lanes 48..63 (16 lanes).
        cases = [
            ("s_and_b64 s[6:7], vcc, s[4:5]", lambda l: 48 <= l <= 55, 8, 1),
            ("s_or_b64 s[6:7], vcc, s[4:5]", lambda l: 40 <= l <= 63, 24, 1),
            ("s_xor_b64 s[6:7], vcc, s[4:5]", lambda l: (40 <= l <= 47) or (56 <= l <= 63), 16, 1),
            ("s_andn2_b64 s[6:7], vcc, s[4:5]", lambda l: 40 <= l <= 47, 8, 1),
            ("s_orn2_b64 s[6:7], vcc, s[4:5]", lambda l: l <= 55, 56, 1),
            ("s_not_b64 s[6:7], vcc", lambda l: not (40 <= l <= 55), 48, 1),
            ("s_nand_b64 s[6:7], vcc, s[4:5]", lambda l: not (48 <= l <= 55), 56, 1),
            ("s_nor_b64 s[6:7], vcc, s[4:5]", lambda l: l <= 39, 40, 1),
            ("s_xnor_b64 s[6:7], vcc, s[4:5]", lambda l: l <= 39 or 48 <= l <= 55, 48, 1),
        ]
        for insn, pred, want_cnt, want_scc in cases:
            with self.subTest(insn=insn):
                lines = [
                    "v_mbcnt_lo_u32_b32 v4, -1, 0",
                    "v_mbcnt_hi_u32_b32 v4, -1, v4",
                    "v_cmp_ge_u32_e32 vcc, v4, 40",
                    "v_cmp_lt_u32_e64 s[8:9], v4, 56",
                    "s_and_b64 vcc, vcc, s[8:9]",        # vcc = 40..55
                    "v_cmp_ge_u32_e64 s[4:5], v4, 48",   # s[4:5] = 48..63
                    insn,
                    "s_cselect_b32 s10, 1, 0",
                    "s_bcnt1_i32_b64 s11, s[6:7]",
                    "v_mov_b32 v1, 0x11",
                    "s_and_b64 exec, exec, s[6:7]",
                    "v_mov_b32 v1, 0x22",
                    "s_or_b64 exec, exec, -1",           # restore all 64 lanes
                    "v_mov_b32 v0, s11",
                    "v_mov_b32 v2, s10",
                    "v_mov_b32 v3, 0",
                ]
                outs, _ = run_ps_wave(lines, wave=64)
                for lane in range(64):
                    want_v1 = 0x22 if pred(lane) else 0x11
                    self.assertEqual(outs[lane], (want_cnt, want_v1, want_scc, 0), f"lane={lane}")

    def test_wave64_compute_and_emulate_translation(self):
        lines = [
            "v_cmp_lt_u32_e32 vcc, 39, v0",           # lanes 40..63 (24 lanes)
            "s_mov_b32 s6, 0x11223344",
            "s_mov_b32 s7, 0x55667788",
            "s_mov_b32 s8, 0x0F0F00FF",
            "s_mov_b32 s9, 0xF0F0FF00",
            "s_xor_b64 s[10:11], s[6:7], s[8:9]",     # data xor: lo=0x1e2d33bb, hi=0xa5968888
            "s_not_b64 s[12:13], vcc",                # mask not: 40 lanes set (0..39)
            "s_bcnt1_i32_b64 s14, s[12:13]",
            "v_cmp_eq_u32_e32 vcc, 0, v0",
            "s_and_b64 exec, exec, vcc",
            "v_mov_b32 v1, s10",
            "v_mov_b32 v2, s11",
            "v_mov_b32 v3, s14",
            "v_mov_b32 v4, 0",
            "buffer_store_dwordx4 v[1:4], v4, s[0:3], 0",
            "s_or_b64 exec, exec, -1",
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
        self.assertEqual(
            [mem.get(0), mem.get(4), mem.get(8), mem.get(12)],
            [0x11223344 ^ 0x0F0F00FF, 0x55667788 ^ 0xF0F0FF00, 40, 0],
        )

        prev = rdna2spirv.WAVE64_EMULATE
        try:
            rdna2spirv.WAVE64_EMULATE = True
            spv_emu = translate_cs(lines)
        finally:
            rdna2spirv.WAVE64_EMULATE = prev
        self.assertIn("OpStore %x_it", spv_emu)


class NegativeControlTests(unittest.TestCase):
    def test_sabotaging_high_dword_bitwise_breaks_result(self):
        # Replacing the high-dword OpBitwiseXor with a copy of the low-dword result simulates
        # the old mask_dst bug (duplicating lo into hi).
        lines = (
            load_pair_lines(4, 0x55667788_11223344)
            + load_pair_lines(6, 0xF0F0FF00_0F0F00FF)
            + [
                "s_xor_b64 s[2:3], s[4:5], s[6:7]",
                "v_mov_b32 v0, s2",
                "v_mov_b32 v1, s3",
                "v_mov_b32 v2, 0",
                "v_mov_b32 v3, 0",
            ]
        )
        words, spv = run_ps(lines)
        want_lo = 0x11223344 ^ 0x0F0F00FF
        want_hi = 0x55667788 ^ 0xF0F0FF00
        self.assertEqual(words[:2], [want_lo, want_hi])
        m = re.search(
            r"(%\w+) = OpBitwiseXor %uint %\w+ %\w+\s+"
            r"(%\w+) = OpBitwiseXor %uint %\w+ %\w+",
            spv,
        )
        self.assertIsNotNone(m)
        lo_id, hi_id = m.group(1), m.group(2)
        bad = re.sub(
            rf"{re.escape(hi_id)} = OpBitwiseXor %uint [^\n]+",
            f"{hi_id} = OpCopyObject %uint {lo_id}",
            spv,
            count=1,
        )
        self.assertNotEqual(bad, spv)
        self.assertEqual(mrt0_words(bad)[:2], [want_lo, want_lo])


if __name__ == "__main__":
    unittest.main()
