"""Semantic tests for Wave64/Wave32 s_cmp_* on opposite mask halves and mask-vs-scalar operands.

Translates synthetic .s listings (no llvm-mc or game data required) with tools/rdna2spirv.py and
executes all 64 (wave64) or 32 (wave32) lanes via tests/semantic/spirv_mini_interp.run_workgroup.

Covers:
  - Wave64 s_cmp_eq_u32 / s_cmp_lg_u32 on opposite halves: vcc_lo vs vcc_hi, exec_lo vs exec_hi,
    and MASK_LO SGPR vs MASK_HI SGPR (equal halves, unequal halves matching on lane 0 with equal
    popcount, and single-bit mismatches).
  - Wave64 and Wave32 s_cmp_eq_u32 / s_cmp_lt_u32 / s_cmp_gt_i32 comparing a lane-mask half
    against arbitrary 32-bit scalar SGPRs and immediates (e.g. 0x80000003).
  - Wave64 s_cmp_eq_u64 / s_cmp_lg_u64 comparing a lane mask against an arbitrary 64-bit SGPR
    pair or non-0/-1 immediate.
  - Preservation of the fast wave_equal vote path for same-half and mask-vs-0/-1 equality compares.
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
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# COMPUTE_PGM_RSRC2: 0x8090 -> 8 user SGPRs (s[0:3]=V#, s4=lo, s5=hi, s6=cmp_lo, s7=cmp_hi), TGID_X.
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def make_fssm(threads):
    return bytes(0x138) + struct.pack("<4I", 0x8090, threads, 1, 1)


def translate_cs(name, asm, threads=64):
    """Translate a synthetic compute .s listing into SPIR-V assembly."""
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8") as f:
            f.write(f"; {name} synthetic\n")
            for ln in asm.splitlines():
                st = ln.strip()
                if st and not st.startswith(";"):
                    f.write(f"\t{st}\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(make_fssm(threads))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(s_path)
            gc.collect()
        return spv


def run_cs(spv, threads, s4=0, s5=0, s6=0, s7=0):
    pc = VSHARP + [s4 & 0xFFFFFFFF, s5 & 0xFFFFFFFF, s6 & 0xFFFFFFFF, s7 & 0xFFFFFFFF] + [0] * 24
    return si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )


def emit_store_scc(slot):
    """Emit RDNA2 lines that store SCC (1 or 0) to buffer dword `slot`."""
    return (
        f"s_cselect_b32 s20, 1, 0\n"
        f"v_mov_b32_e32 v4, s20\n"
        f"v_mov_b32_e32 v5, {slot * 4}\n"
        f"buffer_store_dword v4, v5, s[0:3], 0 offen\n"
    )


# Builds a wave64 lane mask in `vcc` whose bit i is ((s4 if i < 32 else s5) >> (i & 31)) & 1,
# and copies it into s[12:13] (MASK_LO=s12, MASK_HI=s13).
WAVE64_BUILD_VCC_AND_SGPR = """\
v_mov_b32_e32 v11, s5
v_cmp_lt_u32_e64 vcc, v0, 32
v_cndmask_b32_e64 v1, v11, s4, vcc
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc, 0, v3
s_and_b64 s[12:13], vcc, exec
"""

# Same for wave32: builds `vcc` from s4 (lanes 0..31).
WAVE32_BUILD_VCC = """\
v_mov_b32_e32 v1, s4
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc_lo, 0, v3
"""


def _s32(u):
    u &= 0xFFFFFFFF
    return u - 0x100000000 if u >= 0x80000000 else u


def pack_u64(lo: int, hi: int) -> int:
    """Pack low and high 32-bit unsigned words into a 64-bit integer."""
    return ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)


def ref_s_cmp_u64(op: str, src0: int, src1: int) -> int:
    """Expected SCC (1 or 0) of a 64-bit scalar compare.

    GFX10 has only these two (llvm-mc rejects s_cmp_lt/le/gt/ge_u64 for gfx1013):
    - s_cmp_eq_u64: SCC = (src0[63:0] == src1[63:0])
    - s_cmp_lg_u64: SCC = (src0[63:0] != src1[63:0])
    """
    s0 = src0 & 0xFFFFFFFFFFFFFFFF
    s1 = src1 & 0xFFFFFFFFFFFFFFFF
    if op == "s_cmp_eq_u64":
        return int(s0 == s1)
    if op == "s_cmp_lg_u64":
        return int(s0 != s1)
    raise ValueError(f"no 64-bit scalar compare {op} on GFX10")


class Wave64OppositeHalvesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: vcc_lo == vcc_hi
            + "s_cmp_eq_u32 vcc_lo, vcc_hi\n" + emit_store_scc(0)
            # 1: vcc_lo != vcc_hi
            + "s_cmp_lg_u32 vcc_lo, vcc_hi\n" + emit_store_scc(1)
            # 2: MASK_LO (s12) == MASK_HI (s13)
            + "s_cmp_eq_u32 s12, s13\n" + emit_store_scc(2)
            # 3: MASK_LO (s12) != MASK_HI (s13)
            + "s_cmp_lg_u32 s12, s13\n" + emit_store_scc(3)
            # 4 & 5: exec_lo == exec_hi / exec_lo != exec_hi (save/restore exec so store runs)
            + "s_mov_b64 s[14:15], exec\n"
            + "s_and_b64 exec, exec, vcc\n"
            + "s_cmp_eq_u32 exec_lo, exec_hi\n"
            + "s_cselect_b32 s21, 1, 0\n"
            + "s_cmp_lg_u32 exec_lo, exec_hi\n"
            + "s_cselect_b32 s22, 1, 0\n"
            + "s_mov_b64 exec, s[14:15]\n"
            + "v_mov_b32_e32 v4, s21\nv_mov_b32_e32 v5, 16\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s22\nv_mov_b32_e32 v5, 20\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_opp_halves", asm, threads=64)

    def test_opposite_halves_equal_and_unequal(self):
        cases = [
            # Equal halves
            (0x80000003, 0x80000003),
            (0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF),
            # Unequal halves that match on lane 0 AND have identical popcount (2)
            (0x00000003, 0x00000005),
            # Unequal halves with lane 0 off in both and identical popcount (1)
            (0x00000002, 0x80000000),
            # Unequal halves where one is all-zero
            (0x00000000, 0x80000000),
            (0x00000001, 0x00000002),
        ]
        for lo, hi in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi)
                eq = int(lo == hi)
                ne = int(lo != hi)
                got = [mem[4 * k] for k in range(6)]
                self.assertEqual(got, [eq, ne, eq, ne, eq, ne])


class MaskVsScalarAndOrderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm_w64 = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: vcc_lo == 0x80000003 (arbitrary immediate)
            + "s_cmp_eq_u32 vcc_lo, 0x80000003\n" + emit_store_scc(0)
            # 1: vcc_hi == s6 (arbitrary scalar SGPR)
            + "s_cmp_eq_u32 vcc_hi, s6\n" + emit_store_scc(1)
            # 2: s6 == s12 (scalar on left, MASK_LO on right)
            + "s_cmp_eq_u32 s6, s12\n" + emit_store_scc(2)
            # 3: vcc_lo < s6 (unsigned ordering)
            + "s_cmp_lt_u32 vcc_lo, s6\n" + emit_store_scc(3)
            # 4: s13 < s7 (MASK_HI unsigned ordering)
            + "s_cmp_lt_u32 s13, s7\n" + emit_store_scc(4)
            # 5: vcc_lo > 0 (signed i32 ordering: 0x80000003 is negative!)
            + "s_cmp_gt_i32 vcc_lo, 0\n" + emit_store_scc(5)
            # 6: vcc_hi > s7 (signed i32 ordering)
            + "s_cmp_gt_i32 vcc_hi, s7\n" + emit_store_scc(6)
            # 7: exec_hi < s7 (exec_hi ordering against scalar)
            + "s_mov_b64 s[14:15], exec\n"
            + "s_and_b64 exec, exec, vcc\n"
            + "s_cmp_lt_u32 exec_hi, s7\n"
            + "s_cselect_b32 s21, 1, 0\n"
            + "s_mov_b64 exec, s[14:15]\n"
            + "v_mov_b32_e32 v4, s21\nv_mov_b32_e32 v5, 28\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "s_endpgm\n"
        )
        cls.spv_w64 = translate_cs("cs_w64_mask_scalar", asm_w64, threads=64)

        asm_w32 = (
            WAVE32_BUILD_VCC
            # 0: vcc_lo == 0x80000003
            + "s_cmp_eq_u32 vcc_lo, 0x80000003\n" + emit_store_scc(0)
            # 1: vcc_lo < s6 (u32)
            + "s_cmp_lt_u32 vcc_lo, s6\n" + emit_store_scc(1)
            # 2: vcc_lo > 0 (i32)
            + "s_cmp_gt_i32 vcc_lo, 0\n" + emit_store_scc(2)
            # 3: s6 == vcc_lo (scalar on left)
            + "s_cmp_eq_u32 s6, vcc_lo\n" + emit_store_scc(3)
            + "s_endpgm\n"
        )
        cls.spv_w32 = translate_cs("cs_w32_mask_scalar", asm_w32, threads=32)

    def test_wave64_mask_vs_scalar_and_ordering(self):
        cases = [
            # (lo, hi, cmp_lo=s6, cmp_hi=s7)
            (0x80000003, 0x00000005, 0x80000003, 0x00000006),
            (0x80000003, 0x80000001, 0x80000004, 0x00000000),
            (0x00000005, 0x00000005, 0x00000005, 0x00000005),
            (0x00000001, 0xFFFFFFFE, 0x00000000, 0xFFFFFFFF),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv_w64, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                want = [
                    int(lo == 0x80000003),
                    int(hi == s6),
                    int(s6 == lo),
                    int(lo < s6),
                    int(hi < s7),
                    int(_s32(lo) > 0),
                    int(_s32(hi) > _s32(s7)),
                    int(hi < s7),
                ]
                self.assertEqual([mem[4 * k] for k in range(8)], want)

    def test_wave32_mask_vs_scalar_and_ordering(self):
        cases = [
            (0x80000003, 0x80000003),
            (0x80000003, 0x80000004),
            (0x00000005, 0x00000004),
            (0x00000000, 0x00000001),
        ]
        for lo, s6 in cases:
            with self.subTest(lo=hex(lo), s6=hex(s6)):
                mem = run_cs(self.spv_w32, 32, s4=lo, s6=s6)
                want = [
                    int(lo == 0x80000003),
                    int(lo < s6),
                    int(_s32(lo) > 0),
                    int(s6 == lo),
                ]
                self.assertEqual([mem[4 * k] for k in range(4)], want)


class Wave64U64MaskCompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: s_cmp_eq_u64 vcc, s[6:7] (mask vs arbitrary 64-bit scalar pair)
            + "s_cmp_eq_u64 vcc, s[6:7]\n" + emit_store_scc(0)
            # 1: s_cmp_lg_u64 vcc, s[6:7]
            + "s_cmp_lg_u64 vcc, s[6:7]\n" + emit_store_scc(1)
            # 2: s_cmp_eq_u64 s[6:7], s[12:13] (scalar pair vs mask SGPR pair)
            + "s_cmp_eq_u64 s[6:7], s[12:13]\n" + emit_store_scc(2)
            # 3: s_cmp_eq_u64 vcc, 0x12345678 (mask vs non-0/-1 immediate)
            + "s_cmp_eq_u64 vcc, 0x12345678\n" + emit_store_scc(3)
            # 4: s_cmp_lg_u64 vcc, 0x12345678
            + "s_cmp_lg_u64 vcc, 0x12345678\n" + emit_store_scc(4)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_u64_mask", asm, threads=64)

    def test_u64_mask_vs_scalar_pair_and_imm(self):
        cases = [
            # Exact match with scalar pair
            (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            # lo matches, hi differs (same lane 0 bit)
            (0x12345679, 0x00000001, 0x12345679, 0x00000002),
            # hi matches, lo differs
            (0x00000003, 0x80000001, 0x00000005, 0x80000001),
            # Exact match with 0x00000000_12345678 immediate
            (0x12345678, 0x00000000, 0x12345678, 0x00000000),
            # lo matches 0x12345678, hi non-zero -> != 0x12345678
            (0x12345678, 0x00000001, 0x00000000, 0x00000000),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                eq_pair = int(lo == s6 and hi == s7)
                eq_imm = int(lo == 0x12345678 and hi == 0)
                want = [eq_pair, 1 - eq_pair, eq_pair, eq_imm, 1 - eq_imm]
                self.assertEqual([mem[4 * k] for k in range(5)], want)


class PartialWaveAllOnesTests(unittest.TestCase):
    """A group smaller than the wave: lanes that were never launched have 0 bits in VCC and EXEC but take no part
    in a subgroup vote, so `s_cmp` of a mask against -1 must compare the mask word (GPU: cs_*_partial_scmp.asm)."""

    def _run(self, name, prologue, cases, threads, wave):
        asm = prologue + "v_cmp_eq_u32_e32 {}, v1, v1\n".format("vcc" if wave == 64 else "vcc_lo")
        for k, (cmp_op, _) in enumerate(cases):
            asm += cmp_op + "\n" + emit_store_scc(k)
        spv = translate_cs(name, asm + "s_endpgm\n", threads=threads)
        mem = si.run_workgroup(
            spv, threads, wave,
            inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=VSHARP + [0] * 28, memory={},
        )
        self.assertEqual([mem[4 * k] for k in range(len(cases))], [want for _, want in cases])

    def test_wave64_40_threads(self):
        # vcc = exec = 0x000000ff_ffffffff
        self._run("cs_w64_partial_ones", "v_mbcnt_lo_u32_b32 v1, -1, 0\nv_mbcnt_hi_u32_b32 v1, -1, v1\n", [
            ("s_cmp_eq_u32 vcc_hi, -1", 0),
            ("s_cmp_lg_u32 vcc_hi, -1", 1),
            ("s_cmp_eq_u32 vcc_lo, -1", 1),
            ("s_cmp_eq_u64 vcc, -1", 0),
            ("s_cmp_lg_u64 vcc, -1", 1),
            ("s_cmp_eq_u64 exec, -1", 0),
            ("s_cmp_lg_u64 vcc, 0", 1),
        ], threads=40, wave=64)

    def test_wave32_20_threads(self):
        # vcc_lo = exec_lo = 0x000fffff
        self._run("cs_w32_partial_ones", "v_mbcnt_lo_u32_b32 v1, -1, 0\n", [
            ("s_cmp_eq_u32 vcc_lo, -1", 0),
            ("s_cmp_lg_u32 vcc_lo, -1", 1),
            ("s_cmp_eq_u32 exec_lo, -1", 0),
            ("s_cmp_lg_u32 vcc_lo, 0", 1),
        ], threads=20, wave=32)


class FastPathPreservationTests(unittest.TestCase):
    def test_same_half_and_zero_imm_keep_vote_fast_path(self):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            + "s_cmp_eq_u32 vcc_lo, 0\n"
            + "s_cmp_eq_u32 vcc_lo, exec_lo\n"
            + "s_cmp_lg_u32 vcc_hi, 0\n"
            + "s_cmp_eq_u64 vcc, exec\n"
            + "s_endpgm\n"
        )
        spv = translate_cs("cs_w64_fast_vote", asm, threads=64)
        self.assertIn("OpGroupNonUniformAll", spv)
        self.assertIn("OpGroupNonUniformAny", spv)
        self.assertNotIn("OpGroupNonUniformBitwiseOr", spv)

    def test_opposite_halves_and_scalar_use_wave_mask_not_vote(self):
        prologue = WAVE64_BUILD_VCC_AND_SGPR.replace("s_and_b64 s[12:13], vcc, exec", "s_mov_b64 s[12:13], vcc")
        for name, cmp_op in (
            ("opp", "s_cmp_eq_u32 vcc_lo, vcc_hi"),
            ("imm", "s_cmp_eq_u32 vcc_lo, 0x80000003"),
            ("ord", "s_cmp_lt_u32 vcc_lo, s6"),
            ("u64", "s_cmp_eq_u64 vcc, s[6:7]"),
            # -1 (all ones): a vote misses lanes that were never launched, whose bits are 0 in the mask
            ("ones32", "s_cmp_lg_u32 vcc_hi, -1"),
            ("ones64", "s_cmp_eq_u64 vcc, -1"),
        ):
            with self.subTest(case=name):
                spv = translate_cs(f"cs_w64_wm_{name}", prologue + cmp_op + "\ns_endpgm\n", threads=64)
                self.assertIn("OpGroupNonUniformBitwiseOr", spv)
                self.assertNotIn("OpGroupNonUniformAll", spv)
                self.assertNotIn("OpGroupNonUniformAny", spv)



class Wave64ReferenceOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: s_cmp_eq_u64 vcc, s[6:7] (mask vs scalar pair)
            + "s_cmp_eq_u64 vcc, s[6:7]\n" + emit_store_scc(0)
            # 1: s_cmp_lg_u64 vcc, s[6:7]
            + "s_cmp_lg_u64 vcc, s[6:7]\n" + emit_store_scc(1)
            # 2: s_cmp_eq_u64 s[6:7], vcc (scalar pair vs mask)
            + "s_cmp_eq_u64 s[6:7], vcc\n" + emit_store_scc(2)
            # 3: s_cmp_lg_u64 s[6:7], vcc
            + "s_cmp_lg_u64 s[6:7], vcc\n" + emit_store_scc(3)
            # 4: s_cmp_eq_u64 vcc, s[12:13] (mask vs mask SGPR copy)
            + "s_cmp_eq_u64 vcc, s[12:13]\n" + emit_store_scc(4)
            # 5: s_cmp_lg_u64 vcc, s[12:13]
            + "s_cmp_lg_u64 vcc, s[12:13]\n" + emit_store_scc(5)
            # 6: s_cmp_eq_u64 vcc, 0 (mask vs zero immediate)
            + "s_cmp_eq_u64 vcc, 0\n" + emit_store_scc(6)
            # 7: s_cmp_lg_u64 vcc, 0
            + "s_cmp_lg_u64 vcc, 0\n" + emit_store_scc(7)
            # 8: s_cmp_eq_u64 vcc, 0x55555555 (mask vs 32-bit literal immediate)
            + "s_cmp_eq_u64 vcc, 0x55555555\n" + emit_store_scc(8)
            # 9: s_cmp_lg_u64 vcc, 0x55555555
            + "s_cmp_lg_u64 vcc, 0x55555555\n" + emit_store_scc(9)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_oracle_u64", asm, threads=64)

    def test_wave64_mask_vs_oracle(self):
        """Cross-check simulated wave64 execution against reference oracle predictions."""
        cases = [
            # Exact matches
            (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x55555555, 0x00000000, 0x55555555, 0x00000000),
            # Low half equal, high half different
            (0xCAFEBABE, 0x11111111, 0xCAFEBABE, 0x22222222),
            # High half equal, low half different
            (0x11111111, 0xCAFEBABE, 0x22222222, 0xCAFEBABE),
            # Single-bit difference at boundaries
            (0x00000001, 0x00000000, 0x00000000, 0x00000000),
            (0x80000000, 0x00000000, 0x00000000, 0x00000000),
            (0x00000000, 0x00000001, 0x00000000, 0x00000000),
            (0x00000000, 0x80000000, 0x00000000, 0x00000000),
            # Inverted patterns
            (0xAAAAAAAA, 0x55555555, 0x55555555, 0xAAAAAAAA),
            (0x0F0F0F0F, 0xF0F0F0F0, 0xF0F0F0F0, 0x0F0F0F0F),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mask64 = pack_u64(lo, hi)
                scalar64 = pack_u64(s6, s7)
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)

                want = [
                    ref_s_cmp_u64("s_cmp_eq_u64", mask64, scalar64),
                    ref_s_cmp_u64("s_cmp_lg_u64", mask64, scalar64),
                    ref_s_cmp_u64("s_cmp_eq_u64", scalar64, mask64),
                    ref_s_cmp_u64("s_cmp_lg_u64", scalar64, mask64),
                    ref_s_cmp_u64("s_cmp_eq_u64", mask64, mask64),
                    ref_s_cmp_u64("s_cmp_lg_u64", mask64, mask64),
                    ref_s_cmp_u64("s_cmp_eq_u64", mask64, 0),
                    ref_s_cmp_u64("s_cmp_lg_u64", mask64, 0),
                    ref_s_cmp_u64("s_cmp_eq_u64", mask64, 0x55555555),
                    ref_s_cmp_u64("s_cmp_lg_u64", mask64, 0x55555555),
                ]
                got = [mem[4 * k] for k in range(10)]
                self.assertEqual(got, want)



class Wave64SccConditionUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # Step 1: Compare A (eq u64), preserve SCC across ALU movs
            + "s_cmp_eq_u64 vcc, s[6:7]\n"
            + "s_mov_b32 s16, 0x42\n"
            + "v_mov_b32_e32 v6, s16\n"
            + "s_cselect_b32 s21, 1, 0\n"
            # Step 2: Compare B (lg u64), unconditionally updates SCC
            + "s_cmp_lg_u64 vcc, s[6:7]\n"
            + "s_mov_b32 s17, 0x43\n"
            + "s_cselect_b32 s22, 1, 0\n"
            # Step 3: Compare C (tautology scalar eq), sets SCC to 1
            + "s_cmp_eq_u64 s[6:7], s[6:7]\n"
            + "s_cselect_b32 s23, 1, 0\n"
            # Step 4: Compare D (contradiction scalar lg), sets SCC to 0
            + "s_cmp_lg_u64 s[6:7], s[6:7]\n"
            + "s_cselect_b32 s24, 1, 0\n"
            # Step 5: Compare E (32-bit low half eq)
            + "s_cmp_eq_u32 vcc_lo, s6\n"
            + "s_cselect_b32 s25, 1, 0\n"
            # Step 6: Compare F (32-bit high half lg)
            + "s_cmp_lg_u32 vcc_hi, s7\n"
            + "s_cselect_b32 s26, 1, 0\n"
            # Store s21..s26 to slots 0..5
            + "v_mov_b32_e32 v4, s21\nv_mov_b32_e32 v5, 0\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s22\nv_mov_b32_e32 v5, 4\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s23\nv_mov_b32_e32 v5, 8\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s24\nv_mov_b32_e32 v5, 12\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s25\nv_mov_b32_e32 v5, 16\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s26\nv_mov_b32_e32 v5, 20\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_scc_updates", asm, threads=64)

    def test_scc_sequential_updates_and_preservation(self):
        cases = [
            # Equal mask & scalar
            (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            # Low half equal, high half different
            (0x12345678, 0x11111111, 0x12345678, 0x22222222),
            # High half equal, low half different
            (0x11111111, 0x9ABCDEF0, 0x22222222, 0x9ABCDEF0),
            # Completely different
            (0x00000000, 0x00000000, 0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                u64_eq = int(lo == s6 and hi == s7)
                u64_lg = int(lo != s6 or hi != s7)
                lo_eq = int(lo == s6)
                hi_lg = int(hi != s7)
                want = [u64_eq, u64_lg, 1, 0, lo_eq, hi_lg]
                self.assertEqual([mem[4 * k] for k in range(6)], want)


class Wave64MaskDecompositionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: direct s_cmp_eq_u64 vcc, s[6:7]
            + "s_cmp_eq_u64 vcc, s[6:7]\n"
            + "s_cselect_b32 s20, 1, 0\n"
            # 1: decomposed eq: vcc_lo == s6 AND vcc_hi == s7
            + "s_cmp_eq_u32 vcc_lo, s6\n"
            + "s_cselect_b32 s21, 1, 0\n"
            + "s_cmp_eq_u32 vcc_hi, s7\n"
            + "s_cselect_b32 s22, 1, 0\n"
            + "s_and_b32 s23, s21, s22\n"
            # 2: direct s_cmp_lg_u64 vcc, s[6:7]
            + "s_cmp_lg_u64 vcc, s[6:7]\n"
            + "s_cselect_b32 s24, 1, 0\n"
            # 3: decomposed lg: vcc_lo != s6 OR vcc_hi != s7
            + "s_cmp_lg_u32 vcc_lo, s6\n"
            + "s_cselect_b32 s25, 1, 0\n"
            + "s_cmp_lg_u32 vcc_hi, s7\n"
            + "s_cselect_b32 s26, 1, 0\n"
            + "s_or_b32 s27, s25, s26\n"
            # 4: De Morgan negation of decomposed equality: s23 ^ 1
            + "s_xor_b32 s28, s23, 1\n"
            # 5: direct s_cmp_eq_u64 s[6:7], vcc (swapped)
            + "s_cmp_eq_u64 s[6:7], vcc\n"
            + "s_cselect_b32 s29, 1, 0\n"
            # Store results to buffer: slots 0..5
            + "v_mov_b32_e32 v4, s20\nv_mov_b32_e32 v5, 0\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s23\nv_mov_b32_e32 v5, 4\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s24\nv_mov_b32_e32 v5, 8\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s27\nv_mov_b32_e32 v5, 12\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s28\nv_mov_b32_e32 v5, 16\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "v_mov_b32_e32 v4, s29\nv_mov_b32_e32 v5, 20\nbuffer_store_dword v4, v5, s[0:3], 0 offen\n"
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_decomp", asm, threads=64)

    def test_64bit_mask_decomposition_equivalence(self):
        cases = [
            (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            (0x12345678, 0x00000000, 0x12345678, 0xFFFFFFFF),
            (0x00000000, 0x9ABCDEF0, 0xFFFFFFFF, 0x9ABCDEF0),
            (0xAAAAAAAA, 0x55555555, 0x55555555, 0xAAAAAAAA),
            (0x00000001, 0x00000000, 0x00000000, 0x00000000),
            (0x00000000, 0x80000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                direct_eq = mem[0]
                decomp_eq = mem[4]
                direct_lg = mem[8]
                decomp_lg = mem[12]
                demorgan_lg = mem[16]
                swapped_eq = mem[20]

                self.assertEqual(direct_eq, decomp_eq)
                self.assertEqual(direct_lg, decomp_lg)
                self.assertEqual(decomp_lg, demorgan_lg)
                self.assertEqual(direct_eq, swapped_eq)


class Wave64BoundaryExtremesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: vcc == s[6:7]
            + "s_cmp_eq_u64 vcc, s[6:7]\n" + emit_store_scc(0)
            # 1: vcc != s[6:7]
            + "s_cmp_lg_u64 vcc, s[6:7]\n" + emit_store_scc(1)
            # 2: vcc == 0 (zero boundary)
            + "s_cmp_eq_u64 vcc, 0\n" + emit_store_scc(2)
            # 3: vcc != 0
            + "s_cmp_lg_u64 vcc, 0\n" + emit_store_scc(3)
            # 4: vcc == -1 (all ones boundary)
            + "s_cmp_eq_u64 vcc, -1\n" + emit_store_scc(4)
            # 5: vcc != -1
            + "s_cmp_lg_u64 vcc, -1\n" + emit_store_scc(5)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_boundary", asm, threads=64)

    def test_boundary_extremes(self):
        cases = [
            # All zeros
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            # All ones
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            # Half-word zeros and ones
            (0x00000000, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF),
            (0xFFFFFFFF, 0x00000000, 0xFFFFFFFF, 0x00000000),
            # INT64_MAX: 0x7FFFFFFF_FFFFFFFF
            (0xFFFFFFFF, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FFFFFFF),
            # INT64_MIN: 0x80000000_00000000
            (0x00000000, 0x80000000, 0x00000000, 0x80000000),
            # Single bit at lane 0
            (0x00000001, 0x00000000, 0x00000001, 0x00000000),
            # Single bit at lane 31 (MSB of lo)
            (0x80000000, 0x00000000, 0x80000000, 0x00000000),
            # Single bit at lane 32 (LSB of hi)
            (0x00000000, 0x00000001, 0x00000000, 0x00000001),
            # Single bit at lane 63 (MSB of hi)
            (0x00000000, 0x80000000, 0x00000000, 0x80000000),
            # Corner bits: lane 0 and lane 63 set
            (0x00000001, 0x80000000, 0x00000001, 0x80000000),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mask64 = pack_u64(lo, hi)
                scalar64 = pack_u64(s6, s7)
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)

                want = [
                    int(mask64 == scalar64),
                    int(mask64 != scalar64),
                    int(mask64 == 0),
                    int(mask64 != 0),
                    int(mask64 == 0xFFFFFFFFFFFFFFFF),
                    int(mask64 != 0xFFFFFFFFFFFFFFFF),
                ]
                self.assertEqual([mem[4 * k] for k in range(6)], want)


class Wave64SignedVsUnsignedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: unsigned lt: vcc_lo < s6
            + "s_cmp_lt_u32 vcc_lo, s6\n" + emit_store_scc(0)
            # 1: signed lt: vcc_lo < s6
            + "s_cmp_lt_i32 vcc_lo, s6\n" + emit_store_scc(1)
            # 2: unsigned gt: vcc_lo > s6
            + "s_cmp_gt_u32 vcc_lo, s6\n" + emit_store_scc(2)
            # 3: signed gt: vcc_lo > s6
            + "s_cmp_gt_i32 vcc_lo, s6\n" + emit_store_scc(3)
            # 4: unsigned lt: vcc_hi < s7
            + "s_cmp_lt_u32 vcc_hi, s7\n" + emit_store_scc(4)
            # 5: signed lt: vcc_hi < s7
            + "s_cmp_lt_i32 vcc_hi, s7\n" + emit_store_scc(5)
            # 6: 64-bit eq: vcc == s[6:7]
            + "s_cmp_eq_u64 vcc, s[6:7]\n" + emit_store_scc(6)
            # 7: 64-bit lg: vcc != s[6:7]
            + "s_cmp_lg_u64 vcc, s[6:7]\n" + emit_store_scc(7)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_signed_unsigned", asm, threads=64)

    def test_signed_vs_unsigned_divergence_and_equality_invariance(self):
        cases = [
            # MSB set in vcc_lo (negative signed, large unsigned) vs 0
            (0x80000000, 0x00000001, 0x00000000, 0x80000000),
            # All ones in vcc_lo (-1 signed, 0xFFFFFFFF unsigned) vs 1
            (0xFFFFFFFF, 0x7FFFFFFF, 0x00000001, 0x00000000),
            # 0x7FFFFFFF vs 0x80000000 (signed: pos > neg; unsigned: 0x7fffffff < 0x80000000)
            (0x7FFFFFFF, 0x7FFFFFFF, 0x80000000, 0x80000000),
            # Exact match
            (0x80000001, 0x80000002, 0x80000001, 0x80000002),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)

                u_lt_lo = int((lo & 0xFFFFFFFF) < (s6 & 0xFFFFFFFF))
                s_lt_lo = int(_s32(lo) < _s32(s6))
                u_gt_lo = int((lo & 0xFFFFFFFF) > (s6 & 0xFFFFFFFF))
                s_gt_lo = int(_s32(lo) > _s32(s6))

                u_lt_hi = int((hi & 0xFFFFFFFF) < (s7 & 0xFFFFFFFF))
                s_lt_hi = int(_s32(hi) < _s32(s7))

                eq_64 = int(lo == s6 and hi == s7)
                lg_64 = int(lo != s6 or hi != s7)

                want = [u_lt_lo, s_lt_lo, u_gt_lo, s_gt_lo, u_lt_hi, s_lt_hi, eq_64, lg_64]
                self.assertEqual([mem[4 * k] for k in range(8)], want)


class Wave64NegativeControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC_AND_SGPR
            # 0: vcc == s[6:7]
            + "s_cmp_eq_u64 vcc, s[6:7]\n" + emit_store_scc(0)
            # 1: vcc != s[6:7]
            + "s_cmp_lg_u64 vcc, s[6:7]\n" + emit_store_scc(1)
            # 2: s[12:13] == s[6:7] (mask copied to SGPR pair vs scalar pair)
            + "s_cmp_eq_u64 s[12:13], s[6:7]\n" + emit_store_scc(2)
            # 3: s[12:13] != s[6:7]
            + "s_cmp_lg_u64 s[12:13], s[6:7]\n" + emit_store_scc(3)
            # 4: s[6:7] == s[6:7] (non-mask SGPR pair tautology)
            + "s_cmp_eq_u64 s[6:7], s[6:7]\n" + emit_store_scc(4)
            # 5: s[6:7] != s[6:7] (non-mask SGPR pair contradiction)
            + "s_cmp_lg_u64 s[6:7], s[6:7]\n" + emit_store_scc(5)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_neg_ctrl", asm, threads=64)

    def test_single_bit_perturbations_break_equality(self):
        """Starting from exact equality, perturbing any single bit must yield eq=0, lg=1."""
        base_lo = 0x12345678
        base_hi = 0x9ABCDEF0
        bit_positions = [0, 1, 15, 31, 32, 47, 63]
        for bit in bit_positions:
            with self.subTest(bit=bit):
                if bit < 32:
                    p_s6 = base_lo ^ (1 << bit)
                    p_s7 = base_hi
                else:
                    p_s6 = base_lo
                    p_s7 = base_hi ^ (1 << (bit - 32))

                mem = run_cs(self.spv, 64, s4=base_lo, s5=base_hi, s6=p_s6, s7=p_s7)
                self.assertEqual(mem[0], 0)
                self.assertEqual(mem[4], 1)
                self.assertEqual(mem[8], 0)
                self.assertEqual(mem[12], 1)
                self.assertEqual(mem[16], 1)
                self.assertEqual(mem[20], 0)

    def test_mutual_exclusion_and_sgpr_orthogonality(self):
        cases = [
            (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            (0x12345678, 0x9ABCDEF0, 0x00000000, 0x00000000),
            (0xAAAAAAAA, 0x55555555, 0x55555555, 0xAAAAAAAA),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                vcc_eq, vcc_lg = mem[0], mem[4]
                sgpr_eq, sgpr_lg = mem[8], mem[12]
                tautology, contradiction = mem[16], mem[20]

                # Negative control 1: strict mutual exclusion
                self.assertEqual(vcc_eq ^ vcc_lg, 1)
                self.assertEqual(vcc_eq + vcc_lg, 1)
                self.assertEqual(sgpr_eq ^ sgpr_lg, 1)
                self.assertEqual(sgpr_eq + sgpr_lg, 1)

                # Negative control 2: VCC mask vs SGPR mask representation consistency
                self.assertEqual(vcc_eq, sgpr_eq)
                self.assertEqual(vcc_lg, sgpr_lg)

                # Negative control 3: Baseline invariants
                self.assertEqual(tautology, 1)
                self.assertEqual(contradiction, 0)


if __name__ == "__main__":
    unittest.main()
