"""Semantic interpreter tests for VOP3 op_sel on 16-bit vector compares:
  - v_cmp_*_f16_e64 / v_cmpx_*_f16_e64 (lt, le, gt, ge, eq, lg, neq, o, u, plus neg/abs on high half)
  - v_cmp_*_u16_e64 / v_cmpx_*_u16_e64 (eq, ne, lg, lt, le, gt, ge)
  - v_cmp_*_i16_e64 / v_cmpx_*_i16_e64 (eq, ne, lg, lt, le, gt, ge, including signed negative high halves)

VOP3 op_sel:[s0,s1] selects bit 15:0 (0) or bit 31:16 (1) of each 16-bit source operand before float
input modifiers (neg/abs) or integer zero/sign extension. VOPC writes a scalar lane mask (VCC/SGPR
or EXEC for v_cmpx), so there is no destination op_sel bit.

Run directly:
    python -I tests/semantic/test_vcmp16_opsel.py -v
"""
import gc
import math
import os
import shutil
import struct
import subprocess
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

VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]  # OOB_SELECT 3 (raw)
SENTINEL = 0xDEADBEEF

# Half-precision bit constants
H_ZERO = 0x0000      # +0.0
H_NZERO = 0x8000     # -0.0
H_HALF = 0x3800      # +0.5
H_ONE = 0x3C00       # +1.0
H_TWO = 0x4000       # +2.0
H_THREE = 0x4200     # +3.0
H_FOUR = 0x4400      # +4.0
H_MONE = 0xBC00      # -1.0
H_MTWO = 0xC000      # -2.0
H_MFOUR = 0xC400     # -4.0
H_INF = 0x7C00       # +inf
H_NINF = 0xFC00      # -inf
H_QNAN = 0x7E00      # qNaN


def pack16(lo: int, hi: int) -> int:
    return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)


def f16_half(word: int, hi: int) -> float:
    bits = (word >> 16) & 0xFFFF if hi else (word & 0xFFFF)
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def u16_half(word: int, hi: int) -> int:
    return (word >> 16) & 0xFFFF if hi else (word & 0xFFFF)


def i16_half(word: int, hi: int) -> int:
    v = u16_half(word, hi)
    return v - 0x10000 if v & 0x8000 else v


def f16_rel(cond: str, x: float, y: float) -> int:
    nan = math.isnan(x) or math.isnan(y)
    if cond == "o":
        return int(not nan)
    if cond == "u":
        return int(nan)
    if cond == "lt":
        return int(x < y)
    if cond == "le":
        return int(x <= y)
    if cond == "gt":
        return int(x > y)
    if cond == "ge":
        return int(x >= y)
    if cond == "eq":
        return int(x == y)
    if cond == "lg":
        return int((not nan) and x != y)
    if cond == "neq":
        return int(nan or x != y)
    raise ValueError(cond)


def int_rel(cond: str, x: int, y: int) -> int:
    return int({
        "eq": x == y,
        "ne": x != y,
        "lg": x != y,
        "lt": x < y,
        "le": x <= y,
        "gt": x > y,
        "ge": x >= y,
    }[cond])


_TMP_DIR = None
_PS_PATH = None


def setUpModule():
    global _TMP_DIR, _PS_PATH
    _TMP_DIR = tempfile.TemporaryDirectory()
    dis = os.path.join(_TMP_DIR.name, "disasm")
    os.mkdir(dis)
    _PS_PATH = os.path.join(dis, "ps_vcmp16_opsel.s")
    with open(os.path.join(_TMP_DIR.name, "ps_vcmp16_opsel.dxbc"), "wb") as f:
        f.write(STUB_DXBC)


def tearDownModule():
    global _TMP_DIR
    if _TMP_DIR is not None:
        _TMP_DIR.cleanup()
        _TMP_DIR = None


def translate_ps(lines):
    if _TMP_DIR is None:
        setUpModule()
    asm = "; ps_vcmp16_opsel\n" + "".join(f"\t{ln}\n" for ln in lines)
    with open(_PS_PATH, "w", encoding="utf-8", newline="") as f:
        f.write(asm)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ResourceWarning)
        return rdna2spirv.translate(_PS_PATH)


def run_ps_bit(x_word: int, y_word: int, cmp_line: str, mask: str = "s4") -> int:
    lines = [
        f"v_mov_b32 v1, {x_word:#x}",
        f"v_mov_b32 v2, {y_word:#x}",
        cmp_line,
        f"v_cndmask_b32_e64 v0, 0x40000000, 0x40000001, {mask}",
        "exp mrt0 v0, v0, v0, v0 done vm",
        "s_endpgm",
    ]
    w = mrt0_words(translate_ps(lines))[0]
    assert (w >> 1) == 0x20000000, hex(w)
    return w & 1


def run_cs(name: str, body_lines, wave_size: int, user_sgprs=None):
    """Translate a synthetic compute shader and run all lanes in spirv_mini_interp.run_workgroup."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"; {name}\n")
            for ln in body_lines:
                f.write(f"\t{ln}\n")
            f.write("\ts_endpgm\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        rsrc2 = 0x8090 if user_sgprs else 0x8088
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", rsrc2, wave_size, 1, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gc.collect()
        assert rdna2spirv.translate.last_gen.wave64 == (wave_size == 64)
    sgprs = list(user_sgprs) if user_sgprs else []
    pc = VSHARP + sgprs + [0] * (28 - len(sgprs))
    return si.run_workgroup(
        spv, wave_size, wave_size,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )


class F16CompareOpSelTests(unittest.TestCase):
    def test_f16_opsel_relations(self):
        # Packed pairs chosen so each (s0, s1) combination tests distinct ordering & equality cases:
        # Pair A: v1=(1.0, 4.0), v2=(2.0, 1.0) -> (lo,lo): 1<2; (hi,lo): 4>2; (lo,hi): 1==1; (hi,hi): 4>1
        # Pair B: v1=(3.0, 2.0), v2=(2.0, 2.0) -> (lo,lo): 3>2; (hi,lo): 2==2; (lo,hi): 3>2; (hi,hi): 2==2
        # Pair C: v1=(-2.0, 0.5), v2=(0.5, -1.0) -> signed negatives in both halves
        pairs = [
            (pack16(H_ONE, H_FOUR), pack16(H_TWO, H_ONE)),
            (pack16(H_THREE, H_TWO), pack16(H_TWO, H_TWO)),
            (pack16(H_MTWO, H_HALF), pack16(H_HALF, H_MONE)),
        ]
        for cond in ("lt", "le", "gt", "ge", "eq", "lg", "neq"):
            for xw, yw in pairs:
                for s0, s1 in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    want = f16_rel(cond, f16_half(xw, s0), f16_half(yw, s1))
                    with self.subTest(cond=cond, x=hex(xw), y=hex(yw), opsel=(s0, s1)):
                        got = run_ps_bit(
                            xw, yw,
                            f"v_cmp_{cond}_f16_e64 s4, v1, v2 op_sel:[{s0},{s1}]",
                            mask="s4",
                        )
                        self.assertEqual(got, want)

    def test_f16_opsel_ordered_unordered_nan(self):
        # NaN in high half of v1, finite in low half; NaN in low half of v2, finite in high half:
        # (0,0): finite vs NaN -> unordered; (1,0): NaN vs NaN -> unordered;
        # (0,1): finite vs finite -> ordered; (1,1): NaN vs finite -> unordered.
        xw = pack16(H_ONE, H_QNAN)
        yw = pack16(H_QNAN, H_TWO)
        # Also test finite-lo/NaN-hi vs finite-lo/finite-hi so [0,0] is ordered and [1,0] is unordered.
        cases = [
            (xw, yw),
            (pack16(H_ONE, H_QNAN), pack16(H_TWO, H_THREE)),
            (pack16(H_QNAN, H_ONE), pack16(H_TWO, H_QNAN)),
        ]
        for cond in ("o", "u", "lt", "eq", "neq"):
            for a, b in cases:
                for s0, s1 in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    want = f16_rel(cond, f16_half(a, s0), f16_half(b, s1))
                    with self.subTest(cond=cond, a=hex(a), b=hex(b), opsel=(s0, s1)):
                        got = run_ps_bit(
                            a, b,
                            f"v_cmp_{cond}_f16_e64 vcc_lo, v1, v2 op_sel:[{s0},{s1}]",
                            mask="vcc_lo",
                        )
                        self.assertEqual(got, want)

    def test_f16_opsel_neg_abs_modifiers(self):
        # v1 = (lo=+1.0, hi=-4.0), v2 = (lo=+2.0, hi=+4.0)
        # Selecting hi of v1 (-4.0):
        #   -v1 -> +4.0: eq v2.hi (+4.0) is 1, gt v2.lo (+2.0) is 1
        #   |v1| -> +4.0: eq v2.hi (+4.0) is 1, lt v2.lo (+2.0) is 0
        #   -|v1| -> -4.0: lt v2.lo (+2.0) is 1
        # Reading lo of v1 (+1.0) instead would flip every case below.
        xw = pack16(H_ONE, H_MFOUR)
        yw = pack16(H_TWO, H_FOUR)
        cases = [
            ("v_cmp_eq_f16_e64 s4, -v1, v2 op_sel:[1,1]", 1),
            ("v_cmp_gt_f16_e64 s4, -v1, v2 op_sel:[1,0]", 1),
            ("v_cmp_eq_f16_e64 s4, |v1|, v2 op_sel:[1,1]", 1),
            ("v_cmp_lt_f16_e64 s4, |v1|, v2 op_sel:[1,0]", 0),
            ("v_cmp_lt_f16_e64 s4, v1, -v2 op_sel:[0,1]", 0),   # +1.0 < -4.0 -> 0 (lo of v2 would give +1.0 < -2.0 -> 0, check gt below)
            ("v_cmp_lt_f16_e64 s4, -|v1|, v2 op_sel:[1,0]", 1), # -4.0 < +2.0 -> 1
            ("v_cmp_gt_f16_e64 s4, -|v1|, -v2 op_sel:[1,0]", 0), # -4.0 > -2.0 -> 0 (lo would be -1.0 > -2.0 -> 1!)
            ("v_cmp_eq_f16_e64 s4, v1, -v2 op_sel:[1,1]", 1),   # -4.0 == -4.0 -> 1
        ]
        for line, want in cases:
            with self.subTest(line=line):
                self.assertEqual(run_ps_bit(xw, yw, line, mask="s4"), want)


class Int16CompareOpSelTests(unittest.TestCase):
    def test_u16_opsel_all_combos(self):
        # Pair 1: v1=(lo=0x0001, hi=0xFFFF), v2=(lo=0x0002, hi=0x0001)
        #   (0,0): 1 < 2; (1,0): 65535 > 2; (0,1): 1 == 1; (1,1): 65535 > 1
        # Pair 2: v1=(lo=0x8000, hi=0x0005), v2=(lo=0x7FFF, hi=0x0005)
        #   (0,0): 32768 > 32767; (1,0): 5 < 32767; (0,1): 32768 > 5; (1,1): 5 == 5
        pairs = [
            (pack16(0x0001, 0xFFFF), pack16(0x0002, 0x0001)),
            (pack16(0x8000, 0x0005), pack16(0x7FFF, 0x0005)),
        ]
        for cond in ("eq", "ne", "lg", "lt", "le", "gt", "ge"):
            for xw, yw in pairs:
                for s0, s1 in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    want = int_rel(cond, u16_half(xw, s0), u16_half(yw, s1))
                    with self.subTest(cond=cond, x=hex(xw), y=hex(yw), opsel=(s0, s1)):
                        got = run_ps_bit(
                            xw, yw,
                            f"v_cmp_{cond}_u16_e64 s4, v1, v2 op_sel:[{s0},{s1}]",
                            mask="s4",
                        )
                        self.assertEqual(got, want)

    def test_i16_opsel_all_combos(self):
        # Negative signed i16 in high half vs positive in low half:
        # Pair 1: v1=(lo=+1, hi=-1=0xFFFF), v2=(lo=+2, hi=+1)
        #   (0,0): +1 < +2; (1,0): -1 < +2; (0,1): +1 == +1; (1,1): -1 < +1 (unsigned would say 0xFFFF > 1!)
        # Pair 2: v1=(lo=+32767=0x7FFF, hi=-32768=0x8000), v2=(lo=-1=0xFFFF, hi=+0)
        #   (0,0): +32767 > -1; (1,0): -32768 < -1; (0,1): +32767 > 0; (1,1): -32768 < 0
        # Pair 3: v1=(lo=+5, hi=-2=0xFFFE), v2=(lo=-2=0xFFFE, hi=-2=0xFFFE)
        #   (0,0): +5 > -2; (1,0): -2 == -2; (0,1): +5 > -2; (1,1): -2 == -2
        pairs = [
            (pack16(0x0001, 0xFFFF), pack16(0x0002, 0x0001)),
            (pack16(0x7FFF, 0x8000), pack16(0xFFFF, 0x0000)),
            (pack16(0x0005, 0xFFFE), pack16(0xFFFE, 0xFFFE)),
        ]
        for cond in ("eq", "ne", "lg", "lt", "le", "gt", "ge"):
            for xw, yw in pairs:
                for s0, s1 in ((0, 0), (1, 0), (0, 1), (1, 1)):
                    want = int_rel(cond, i16_half(xw, s0), i16_half(yw, s1))
                    with self.subTest(cond=cond, x=hex(xw), y=hex(yw), opsel=(s0, s1)):
                        got = run_ps_bit(
                            xw, yw,
                            f"v_cmp_{cond}_i16_e64 s4, v1, v2 op_sel:[{s0},{s1}]",
                            mask="s4",
                        )
                        self.assertEqual(got, want)


class CmpxWaveExecOpSelTests(unittest.TestCase):
    def test_cmpx_compute_wave32_and_wave64(self):
        # For each type (f16, u16, i16) and wave size (32, 64), give even lanes a packed value whose
        # high half PASSES the compare (low half FAILS), and odd lanes a packed value whose high half
        # FAILS (low half PASSES). v_cmpx_*_e64 with op_sel:[1,0] must keep even lanes active and
        # switch off odd lanes, writing lane|0x100 only for even lanes and leaving SENTINEL on odd lanes.
        cases = [
            # (tag, cmp_line, even_v1, odd_v1, v2)
            ("f16_lt_10", "v_cmpx_lt_f16_e64 v1, v2 op_sel:[1,0]",
             pack16(H_FOUR, H_ONE), pack16(H_ONE, H_FOUR), pack16(H_TWO, H_TWO)),
            ("f16_gt_01", "v_cmpx_gt_f16_e64 v1, v2 op_sel:[0,1]",
             pack16(H_FOUR, H_ONE), pack16(H_ONE, H_FOUR), pack16(H_FOUR, H_TWO)),
            ("u16_gt_10", "v_cmpx_gt_u16_e64 v1, v2 op_sel:[1,0]",
             pack16(0x0001, 0xFFFF), pack16(0xFFFF, 0x0001), pack16(0x0010, 0x0010)),
            ("i16_lt_10", "v_cmpx_lt_i16_e64 v1, v2 op_sel:[1,0]",
             pack16(0x0005, 0xFFFF), pack16(0xFFFF, 0x0005), pack16(0x0001, 0x0001)),
        ]
        for wave in (32, 64):
            vcc = "vcc" if wave == 64 else "vcc_lo"
            save_exec = "s_mov_b64 s[8:9], exec" if wave == 64 else "s_mov_b32 s8, exec_lo"
            for tag, cmp_line, even_v1, odd_v1, v2_word in cases:
                with self.subTest(wave=wave, case=tag):
                    body = [
                        save_exec,
                        # Pre-fill output[lane] = SENTINEL while all lanes are active.
                        "v_lshlrev_b32_e32 v4, 2, v0",
                        f"v_mov_b32 v5, {SENTINEL:#x}",
                        "buffer_store_dword v5, v4, s[0:3], 0 offen",
                        # v3 = lane & 1; even lanes get even_v1, odd lanes get odd_v1.
                        "v_and_b32_e32 v3, 1, v0",
                        f"v_cmp_eq_u32_e64 {vcc}, v3, 0",
                        f"v_mov_b32 v6, {even_v1:#x}",
                        f"v_mov_b32 v7, {odd_v1:#x}",
                        f"v_cndmask_b32_e64 v1, v7, v6, {vcc}",
                        f"v_mov_b32 v2, {v2_word:#x}",
                        cmp_line,
                        # Only lanes that survived v_cmpx execute this store.
                        "v_or_b32_e32 v5, 0x100, v0",
                        "buffer_store_dword v5, v4, s[0:3], 0 offen",
                    ]
                    mem = run_cs(f"cs_cmpx_{tag}_w{wave}", body, wave_size=wave)
                    for lane in range(wave):
                        want = (0x100 | lane) if (lane % 2 == 0) else SENTINEL
                        self.assertEqual(mem.get(lane * 4), want, f"lane {lane}")

    def test_cmpx_int16_spirv_wave_interp(self):
        # Run v_cmpx_*_u16_e64 and v_cmpx_*_i16_e64 with op_sel directly through spirv_wave_interp
        # across both wave32 and wave64, verifying lane-exact EXEC kill on pixel export.
        cases = [
            # (tag, cmp_line, even_v1, odd_v1, v2)
            ("u16_gt_10", "v_cmpx_gt_u16_e64 v1, v2 op_sel:[1,0]",
             pack16(0x0001, 0xFFFF), pack16(0xFFFF, 0x0001), pack16(0x0010, 0x0010)),
            ("u16_lt_01", "v_cmpx_lt_u16_e64 v1, v2 op_sel:[0,1]",
             pack16(0x0002, 0x0100), pack16(0x0100, 0x0002), pack16(0x0001, 0x0050)),
            ("i16_lt_10", "v_cmpx_lt_i16_e64 v1, v2 op_sel:[1,0]",
             pack16(0x0005, 0xFFFF), pack16(0xFFFF, 0x0005), pack16(0x0001, 0x0001)),
            ("i16_gt_01", "v_cmpx_gt_i16_e64 v1, v2 op_sel:[0,1]",
             pack16(0x0001, 0xFFFE), pack16(0xFFFE, 0x0001), pack16(0x0005, 0xFFFF)),
        ]
        for wave in (32, 64):
            vcc = "vcc" if wave == 64 else "vcc_lo"
            lane_setup = [
                "v_mbcnt_lo_u32_b32 v3, -1, 0",
            ]
            if wave == 64:
                lane_setup.append("v_mbcnt_hi_u32_b32 v3, -1, v3")
            for tag, cmp_line, even_v1, odd_v1, v2_word in cases:
                with self.subTest(wave=wave, case=tag):
                    lines = lane_setup + [
                        "v_and_b32_e32 v4, 1, v3",
                        f"v_cmp_eq_u32_e64 {vcc}, v4, 0",
                        f"v_mov_b32 v6, {even_v1:#x}",
                        f"v_mov_b32 v7, {odd_v1:#x}",
                        f"v_cndmask_b32_e64 v1, v7, v6, {vcc}",
                        f"v_mov_b32 v2, {v2_word:#x}",
                        cmp_line,
                        "v_or_b32_e32 v0, 0x40000000, v3",
                        "exp mrt0 v0, v0, v0, v0 done vm",
                        "s_endpgm",
                    ]
                    spv = translate_ps(lines)
                    self.assertEqual(rdna2spirv.translate.last_gen.wave64, wave == 64)
                    outputs = wi.run_wave(spv, wave=wave)["outputs"]
                    for lane in range(wave):
                        if lane % 2 == 0:
                            self.assertIsNotNone(outputs[lane], f"lane {lane} should stay active")
                            self.assertEqual(outputs[lane]["%out_mrt0"][0], 0x40000000 | lane)
                        else:
                            self.assertIsNone(outputs[lane], f"lane {lane} should be killed by v_cmpx")


class VcmpInactiveLanesSdstTests(unittest.TestCase):
    def test_vcmp_inactive_lanes_wave32(self):
        body = [
            "v_mov_b32 v1, 0x1234",
            "v_mov_b32 v2, 0x1234",
            "s_mov_b32 exec_lo, 0x55555555",
            "v_cmp_eq_u16_e64 s4, v1, v2 op_sel:[0,0]",
            "s_mov_b32 exec_lo, -1",
            "v_lshrrev_b32_e32 v3, v0, s4",
            "v_and_b32_e32 v3, 1, v3",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v3, v6, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_vcmp_inactive_w32", body, 32)
        for lane in range(32):
            self.assertEqual(mem.get(lane * 4), 1 if (lane % 2 == 0) else 0)

    def test_vcmp_inactive_lanes_wave64(self):
        body = [
            "v_mov_b32 v1, 0x1234",
            "v_mov_b32 v2, 0x1234",
            "s_mov_b64 exec, 0x5555555555555555",
            "v_cmp_eq_u16_e64 s[4:5], v1, v2 op_sel:[0,0]",
            "s_mov_b64 exec, -1",
            "v_cmp_ge_u32_e64 s6, v0, 32",
            "v_cndmask_b32_e64 v4, s4, s5, s6",
            "v_and_b32_e32 v5, 31, v0",
            "v_lshrrev_b32_e32 v3, v5, v4",
            "v_and_b32_e32 v3, 1, v3",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v3, v6, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_vcmp_inactive_w64", body, 64)
        for lane in range(64):
            self.assertEqual(mem.get(lane * 4), 1 if (lane % 2 == 0) else 0)


class ConstantsAndLiteralsOpSelTests(unittest.TestCase):
    def test_inline_float_constants_opsel(self):
        # 1.0 has 0x3C00 in low half, +0.0 in high half
        v1 = pack16(H_ONE, H_TWO)
        self.assertEqual(run_ps_bit(v1, 0, "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(v1, 0, "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[1,0]", mask="s4"), 0)
        self.assertEqual(run_ps_bit(pack16(H_ZERO, H_TWO), 0, "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[0,1]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(v1, 0, "v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[0,1]", mask="s4"), 0)
        self.assertEqual(run_ps_bit(pack16(H_MTWO, H_ONE), 0, "v_cmp_eq_f16_e64 s4, v1, -2.0 op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(H_ZERO, H_ONE), 0, "v_cmp_eq_f16_e64 s4, v1, -2.0 op_sel:[0,1]", mask="s4"), 1)

    def test_inline_integer_constants_opsel(self):
        # Inline integer constant 1: opsel=0 -> 1; opsel=1 -> 0
        self.assertEqual(run_ps_bit(pack16(1, 2), 0, "v_cmp_eq_u16_e64 s4, v1, 1 op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0, 2), 0, "v_cmp_eq_u16_e64 s4, v1, 1 op_sel:[0,1]", mask="s4"), 1)
        # Inline integer constant -1: opsel=0 -> 0xFFFF; opsel=1 -> 0xFFFF
        self.assertEqual(run_ps_bit(pack16(0xFFFF, 0), 0, "v_cmp_eq_u16_e64 s4, v1, -1 op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0xFFFF, 0), 0, "v_cmp_eq_u16_e64 s4, v1, -1 op_sel:[0,1]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0xFFFF, 0), 0, "v_cmp_eq_i16_e64 s4, v1, -1 op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0xFFFF, 0), 0, "v_cmp_eq_i16_e64 s4, v1, -1 op_sel:[0,1]", mask="s4"), 1)

    def test_32bit_literals_opsel(self):
        # Literal 0x80007FFF: lo=0x7FFF (+32767), hi=0x8000 (-32768 in i16, 32768 in u16)
        self.assertEqual(run_ps_bit(pack16(0x7FFF, 0), 0, "v_cmp_eq_u16_e64 s4, v1, 0x80007FFF op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0x8000, 0), 0, "v_cmp_eq_u16_e64 s4, v1, 0x80007FFF op_sel:[0,1]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0x7FFF, 0), 0, "v_cmp_eq_i16_e64 s4, v1, 0x80007FFF op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0, 0x8000), 0, "v_cmp_eq_i16_e64 s4, v1, 0x80007FFF op_sel:[1,1]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(0, 0), 0, "v_cmp_lt_i16_e64 s4, v1, 0x80007FFF op_sel:[0,1]", mask="s4"), 0)

        # f16 literal 0x40003C00 (lo = 1.0, hi = 2.0)
        self.assertEqual(run_ps_bit(pack16(H_ONE, H_FOUR), 0, "v_cmp_eq_f16_e64 s4, v1, 0x40003C00 op_sel:[0,0]", mask="s4"), 1)
        self.assertEqual(run_ps_bit(pack16(H_FOUR, H_TWO), 0, "v_cmp_eq_f16_e64 s4, v1, 0x40003C00 op_sel:[1,1]", mask="s4"), 1)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"),
                         "needs spirv-as and spirv-val on PATH")
    def test_spirv_val_compliance(self):
        cases = [
            ("f16_opsel_const", ["v_cmp_eq_f16_e64 s4, v1, 1.0 op_sel:[0,1]", "v_cndmask_b32_e64 v0, 0, 1, s4", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]),
            ("u16_opsel_lit", ["v_cmp_gt_u16_e64 s4, v1, 0x80007FFF op_sel:[1,0]", "v_cndmask_b32_e64 v0, 0, 1, s4", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]),
            ("i16_opsel_const", ["v_cmp_lt_i16_e64 s4, v1, -1 op_sel:[0,1]", "v_cndmask_b32_e64 v0, 0, 1, s4", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]),
            ("cmpx_f16_wave64", ["v_cmpx_lt_f16_e64 v1, v2 op_sel:[1,0]", "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for tag, lines in cases:
                spv_text = translate_ps(lines)
                spv_path = os.path.join(tmp, f"ps_{tag}.spv")
                sa_path = os.path.join(tmp, f"ps_{tag}.spvasm")
                with open(sa_path, "w", encoding="utf-8") as f:
                    f.write(spv_text)
                rdna2spirv.assemble_spirv(sa_path, spv_path)
                res = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", spv_path], capture_output=True, text=True)
                self.assertEqual(res.returncode, 0, f"spirv-val failed for {tag}: {res.stderr}")


class SgprHighHalvesOpSelTests(unittest.TestCase):
    def test_sgpr_high_halves_f16(self):
        user_s4 = pack16(H_ONE, H_FOUR)
        # s4 as src1: opsel:[0, 0] reads lo(s4)=1.0
        lines = [
            f"v_mov_b32 v1, {pack16(H_ONE, 0):#x}",
            "v_cmp_eq_f16_e64 s5, v1, s4 op_sel:[0,0]",
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_sgpr_f16_0", lines, 32, user_sgprs=[user_s4])
        self.assertEqual(mem.get(0), 1)

        # opsel:[0, 1] reads hi(s4)=4.0
        lines = [
            f"v_mov_b32 v1, {pack16(H_FOUR, 0):#x}",
            "v_cmp_eq_f16_e64 s5, v1, s4 op_sel:[0,1]",
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_sgpr_f16_1", lines, 32, user_sgprs=[user_s4])
        self.assertEqual(mem.get(0), 1)

        # s4 as src0:
        lines = [
            f"v_mov_b32 v1, {pack16(H_FOUR, 0):#x}",
            "v_cmp_eq_f16_e64 s5, s4, v1 op_sel:[1,0]",
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_sgpr_f16_src0", lines, 32, user_sgprs=[user_s4])
        self.assertEqual(mem.get(0), 1)

        # s4 with neg / abs on high half:
        lines = [
            f"v_mov_b32 v1, {pack16(H_MFOUR, 0):#x}",
            "v_cmp_eq_f16_e64 s5, v1, -s4 op_sel:[0,1]",
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_sgpr_f16_neg", lines, 32, user_sgprs=[user_s4])
        self.assertEqual(mem.get(0), 1)

    def test_sgpr_high_halves_int16(self):
        user_s4 = pack16(0x7FFF, 0xFFFF)
        lines = [
            f"v_mov_b32 v1, {pack16(0xFFFF, 0):#x}",
            "v_cmp_eq_u16_e64 s5, v1, s4 op_sel:[0,1]",
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_sgpr_u16_1", lines, 32, user_sgprs=[user_s4])
        self.assertEqual(mem.get(0), 1)

        lines = [
            "v_mov_b32 v1, 0",
            "v_cmp_lt_i16_e64 s5, s4, v1 op_sel:[1,0]",
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_sgpr_i16_lt", lines, 32, user_sgprs=[user_s4])
        self.assertEqual(mem.get(0), 1)

        lines = [
            "v_mov_b32 v1, 0",
            "v_cmp_lt_u16_e64 s5, s4, v1 op_sel:[1,0]",
            "v_and_b32_e32 v2, 1, s5",
            "v_lshlrev_b32_e32 v3, 2, v0",
            "buffer_store_dword v2, v3, s[0:3], 0 offen",
        ]
        mem = run_cs("cs_sgpr_u16_lt", lines, 32, user_sgprs=[user_s4])
        self.assertEqual(mem.get(0), 0)


if __name__ == "__main__":
    unittest.main()
