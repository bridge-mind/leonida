"""Semantic regression tests for s_cmp_eq_u64 / s_cmp_lg_u64 on SGPR mask pairs.

Covers wave64 and wave32 SGPR pairs whose two 32-bit halves come from different sources:
  - half mask + 0 or -1: (vcc_lo, 0), (0, vcc_hi), (vcc_lo, -1) on full and partial waves
  - split (MASK_LO, MASK_HI) pair from two distinct masks: (vcc_lo, exec_hi)
  - swapped halves: (vcc_hi, vcc_lo)
  - mixed mask + plain scalar data: (vcc_lo, s7) and (s6, vcc_hi)
  - preservation of the fast wave_equal vote path for canonical (MASK_LO, MASK_HI) pairs
    and zero subgroup ops for plain-data pairs.
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
import spirv_asm  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# COMPUTE_PGM_RSRC2: 0x8090 -> 8 user SGPRs (s[0:3]=V#, s4=lo, s5=hi, s6=cmp_lo, s7=cmp_hi).
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]


def make_fssm(threads: int, wave32: bool = False) -> bytes:
    header = bytearray(0x138 + 16)
    if wave32:
        struct.pack_into("<I", header, 0xD4, 0x20)
    struct.pack_into("<4I", header, 0x138, 0x8090, threads, 1, 1)
    return bytes(header)


def translate_cs(name: str, asm: str, threads: int = 64, wave32: bool = False, emulate: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        s_path = os.path.join(tmp, "disasm", name + ".s")
        with open(s_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; {name} synthetic\n")
            for ln in asm.splitlines():
                st = ln.strip()
                if st and not st.startswith(";"):
                    f.write(f"\t{st}\n")
        with open(os.path.join(tmp, name + ".dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", name + ".fssm"), "wb") as f:
            f.write(make_fssm(threads, wave32=wave32))
        prev = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(s_path)
                gc.collect()
            return spv
        finally:
            rdna2spirv.WAVE64_EMULATE = prev


def run_cs(spv: str, threads: int, wave: int, s4: int = 0, s5: int = 0, s6: int = 0, s7: int = 0) -> dict[int, int]:
    pc = VSHARP + [s4 & 0xFFFFFFFF, s5 & 0xFFFFFFFF, s6 & 0xFFFFFFFF, s7 & 0xFFFFFFFF] + [0] * 24
    return si.run_workgroup(
        spv, threads, wave,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i % wave},
        push_constants=pc, memory={},
    )


def emit_store_scc(slot: int) -> str:
    return (
        "s_cselect_b32 s20, 1, 0\n"
        "v_mov_b32_e32 v4, s20\n"
        f"v_mov_b32_e32 v5, {slot * 4}\n"
        "buffer_store_dword v4, v5, s[0:3], 0 offen\n"
    )


# Builds wave64 vcc whose bit i is ((s4 if i < 32 else s5) >> (i & 31)) & 1.
WAVE64_BUILD_VCC = """\
v_mov_b32_e32 v11, s5
v_cmp_lt_u32_e64 vcc, v0, 32
v_cndmask_b32_e64 v1, v11, s4, vcc
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc, 0, v3
"""

# Builds wave32 vcc_lo whose bit i is (s4 >> i) & 1.
WAVE32_BUILD_VCC = """\
v_mov_b32_e32 v1, s4
v_and_b32_e32 v2, 31, v0
v_lshrrev_b32_e32 v3, v2, v1
v_and_b32_e32 v3, 1, v3
v_cmp_ne_u32_e64 vcc_lo, 0, v3
"""


class Wave64SplitMaskPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE64_BUILD_VCC
            # Pair A: s[10:11] = (vcc_lo, 0)
            + "s_mov_b32 s10, vcc_lo\n"
            + "s_mov_b32 s11, 0\n"
            + "s_cmp_eq_u64 s[10:11], 0\n" + emit_store_scc(0)
            + "s_cmp_lg_u64 s[10:11], 0\n" + emit_store_scc(1)
            + "s_cmp_eq_u64 s[10:11], vcc\n" + emit_store_scc(2)
            # Pair B: s[12:13] = (0, vcc_hi)
            + "s_mov_b32 s12, 0\n"
            + "s_mov_b32 s13, vcc_hi\n"
            + "s_cmp_eq_u64 s[12:13], 0\n" + emit_store_scc(3)
            + "s_cmp_eq_u64 s[12:13], vcc\n" + emit_store_scc(4)
            # Pair C: s[14:15] = (vcc_lo, -1)
            + "s_mov_b32 s14, vcc_lo\n"
            + "s_mov_b32 s15, -1\n"
            + "s_cmp_eq_u64 s[14:15], vcc\n" + emit_store_scc(5)
            + "s_cmp_eq_u64 s[14:15], 0\n" + emit_store_scc(6)
            # Pair D: s[16:17] = (vcc_lo, exec_hi) -> canonical (MASK_LO, MASK_HI) from two different sources
            + "s_mov_b32 s16, vcc_lo\n"
            + "s_mov_b32 s17, exec_hi\n"
            + "s_cmp_eq_u64 s[16:17], vcc\n" + emit_store_scc(7)
            + "s_cmp_eq_u64 s[16:17], exec\n" + emit_store_scc(8)
            # Pair E: s[18:19] = (vcc_hi, vcc_lo) -> swapped halves (MASK_HI, MASK_LO)
            + "s_mov_b32 s18, vcc_hi\n"
            + "s_mov_b32 s19, vcc_lo\n"
            + "s_cmp_eq_u64 s[18:19], s[6:7]\n" + emit_store_scc(9)
            + "s_cmp_eq_u64 s[18:19], vcc\n" + emit_store_scc(10)
            # Pair F: s[22:23] = (vcc_lo, s7) -> mixed mask + plain data
            + "s_mov_b32 s22, vcc_lo\n"
            + "s_mov_b32 s23, s7\n"
            + "s_cmp_eq_u64 s[22:23], s[6:7]\n" + emit_store_scc(11)
            # Pair G: s[24:25] = (s6, vcc_hi) -> mixed plain data + high mask
            + "s_mov_b32 s24, s6\n"
            + "s_mov_b32 s25, vcc_hi\n"
            + "s_cmp_eq_u64 s[24:25], s[6:7]\n" + emit_store_scc(12)
            # Pair H: s[26:27] = (vcc_lo, vcc_lo) -> duplicated low half
            + "s_mov_b32 s26, vcc_lo\n"
            + "s_mov_b32 s27, vcc_lo\n"
            + "s_cmp_eq_u64 s[26:27], vcc\n" + emit_store_scc(13)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w64_split_u64", asm, threads=64)

    def test_split_swapped_and_mixed_pairs_wave64(self):
        cases = [
            (0x00000000, 0x00000001, 0x00000001, 0x00000000),
            (0x00000001, 0x00000000, 0x00000000, 0x00000001),
            (0x00000005, 0xFFFFFFFF, 0xFFFFFFFF, 0x00000005),
            (0xFFFFFFFF, 0x00000005, 0xFFFFFFFF, 0x00000005),
            (0x00000003, 0x80000001, 0x00000003, 0x12345678),
            (0x12345678, 0x9ABCDEF0, 0x12345678, 0x9ABCDEF0),
            (0x00000000, 0x00000000, 0x00000000, 0x00000000),
            (0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF),
        ]
        for lo, hi, s6, s7 in cases:
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=hex(s6), s7=hex(s7)):
                mem = run_cs(self.spv, 64, 64, s4=lo, s5=hi, s6=s6, s7=s7)
                got = [mem[4 * k] for k in range(14)]
                want = [
                    int(lo == 0),                     # 0: (vcc_lo, 0) == 0
                    int(lo != 0),                     # 1: (vcc_lo, 0) != 0
                    int(hi == 0),                     # 2: (vcc_lo, 0) == vcc
                    int(hi == 0),                     # 3: (0, vcc_hi) == 0
                    int(lo == 0),                     # 4: (0, vcc_hi) == vcc
                    int(hi == 0xFFFFFFFF),            # 5: (vcc_lo, -1) == vcc
                    0,                                # 6: (vcc_lo, -1) == 0
                    int(hi == 0xFFFFFFFF),            # 7: (vcc_lo, exec_hi=all1) == vcc
                    int(lo == 0xFFFFFFFF),            # 8: (vcc_lo, exec_hi) == exec
                    int(hi == s6 and lo == s7),       # 9: (vcc_hi, vcc_lo) == s[6:7]
                    int(hi == lo),                    # 10: (vcc_hi, vcc_lo) == vcc
                    int(lo == s6),                    # 11: (vcc_lo, s7) == s[6:7]
                    int(hi == s7),                    # 12: (s6, vcc_hi) == s[6:7]
                    int(lo == hi),                    # 13: (vcc_lo, vcc_lo) == vcc
                ]
                self.assertEqual(got, want)

    def test_partial_wave64_either_minus_one_high_half(self):
        """On a 40-thread wave64, vcc_hi is 0x000000FF while s15 = -1 is 0xFFFFFFFF; they must not compare equal."""
        asm = (
            "v_cmp_eq_u32_e64 vcc, v0, v0\n"
            + "s_mov_b32 s14, vcc_lo\n"
            + "s_mov_b32 s15, -1\n"
            + "s_cmp_eq_u64 s[14:15], vcc\n" + emit_store_scc(0)
            + "s_cmp_lg_u64 s[14:15], vcc\n" + emit_store_scc(1)
            + "s_mov_b32 s16, vcc_lo\n"
            + "s_mov_b32 s17, exec_hi\n"
            + "s_cmp_eq_u64 s[16:17], vcc\n" + emit_store_scc(2)
            + "s_endpgm\n"
        )
        spv = translate_cs("cs_w64_partial_split", asm, threads=40)
        mem = run_cs(spv, threads=40, wave=64)
        self.assertEqual([mem[0], mem[4], mem[8]], [0, 1, 1])

    def test_wave64_emulate_split_pair_assembles_without_subgroup_ops(self):
        """Split mask pairs translate and assemble cleanly under --wave64-emulate without subgroup ops."""
        asm = (
            WAVE64_BUILD_VCC
            + "s_mov_b32 s10, vcc_lo\n"
            + "s_mov_b32 s11, 0\n"
            + "s_cmp_eq_u64 s[10:11], 0\n" + emit_store_scc(0)
            + "s_mov_b32 s16, vcc_lo\n"
            + "s_mov_b32 s17, exec_hi\n"
            + "s_cmp_eq_u64 s[16:17], vcc\n" + emit_store_scc(1)
            + "s_mov_b32 s18, vcc_hi\n"
            + "s_mov_b32 s19, vcc_lo\n"
            + "s_cmp_eq_u64 s[18:19], s[6:7]\n" + emit_store_scc(2)
            + "s_endpgm\n"
        )
        spv = translate_cs("cs_w64_emul_split", asm, threads=64, emulate=True)
        self.assertNotIn("OpGroupNonUniform", spv)
        self.assertIn("%x_buf", spv)
        binary = spirv_asm.assemble(spv, target_env="vulkan1.2")
        self.assertGreater(len(binary), 20)


class Wave32U64MaskPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        asm = (
            WAVE32_BUILD_VCC
            # 0, 1: (vcc_lo, 0) vs 0
            + "s_mov_b32 s10, vcc_lo\n"
            + "s_mov_b32 s11, 0\n"
            + "s_cmp_eq_u64 s[10:11], 0\n" + emit_store_scc(0)
            + "s_cmp_lg_u64 s[10:11], 0\n" + emit_store_scc(1)
            # 2: (vcc_lo, -1) vs 0 -> always 0
            + "s_mov_b32 s12, vcc_lo\n"
            + "s_mov_b32 s13, -1\n"
            + "s_cmp_eq_u64 s[12:13], 0\n" + emit_store_scc(2)
            # 3: (vcc_lo, s7) vs s[6:7] -> mixed mask + plain data
            + "s_mov_b32 s14, vcc_lo\n"
            + "s_mov_b32 s15, s7\n"
            + "s_cmp_eq_u64 s[14:15], s[6:7]\n" + emit_store_scc(3)
            + "s_endpgm\n"
        )
        cls.spv = translate_cs("cs_w32_u64_pair", asm, threads=32, wave32=True)

    def test_wave32_mask_pairs(self):
        for lo, s6, s7 in (
            (0x00000000, 0x00000000, 0x12345678),
            (0x00000003, 0x00000003, 0x12345678),
            (0x00000001, 0xFFFFFFFF, 0x12345678),
            (0xFFFFFFFF, 0xFFFFFFFF, 0x00000000),
        ):
            with self.subTest(lo=hex(lo), s6=hex(s6)):
                mem = run_cs(self.spv, threads=32, wave=32, s4=lo, s6=s6, s7=s7)
                want = [int(lo == 0), int(lo != 0), 0, int(lo == s6)]
                self.assertEqual([mem[4 * k] for k in range(4)], want)


class FastPathAndPlainDataTests(unittest.TestCase):
    def test_canonical_split_mask_pair_keeps_vote_fast_path(self):
        asm = (
            WAVE64_BUILD_VCC
            + "s_mov_b32 s16, vcc_lo\n"
            + "s_mov_b32 s17, exec_hi\n"
            + "s_cmp_eq_u64 s[16:17], vcc\n"
            + "s_cmp_lg_u64 s[16:17], 0\n"
            + "s_endpgm\n"
        )
        spv = translate_cs("cs_w64_split_fast", asm, threads=64)
        self.assertIn("OpGroupNonUniformAll", spv)
        self.assertNotIn("OpGroupNonUniformBitwiseOr", spv)

    def test_plain_data_pair_emits_no_subgroup_ops(self):
        asm = (
            "s_cmp_eq_u64 s[4:5], s[6:7]\n"
            + emit_store_scc(0)
            + "s_endpgm\n"
        )
        spv = translate_cs("cs_w64_plain_u64", asm, threads=64)
        self.assertNotIn("OpGroupNonUniform", spv)
        mem = run_cs(spv, threads=64, wave=64, s4=0x11, s5=0x22, s6=0x11, s7=0x22)
        self.assertEqual(mem[0], 1)


if __name__ == "__main__":
    unittest.main()
