"""Semantic tests for 32-bit bitwise SALU ops on lane masks (wave32 and wave64 half-masks).

Covers s_not_b32 and s_and/or/xor/andn2/orn2/nand/nor/xnor/cselect_b32 when operating on lane masks:
  - Wave32 s_not_b32 on a lane mask (vcc_lo / exec_lo / mask SGPR): inverted per-lane bits AND uniform
    wave-wide SCC (OpGroupNonUniformAny), not per-lane SCC.
  - Wave32 s_not_b32 / s_and_b32 / s_or_b32 / s_xor_b32 with immediate or scalar-SGPR operands writing
    exec_lo or vcc_lo: skips raw scalar constant folding and computes per-lane mask bits + wave-wide SCC.
  - Wave64 32-bit bitwise ops on exec_lo / exec_hi / vcc_lo / vcc_hi / MASK_LO / MASK_HI:
      * Writing exec_lo preserves exec_hi (lanes 32..63); writing exec_hi preserves exec_lo (lanes 0..31).
      * Writing vcc_lo preserves vcc_hi; writing vcc_hi preserves vcc_lo.
      * SCC votes over the written/read 32-lane half only (scc_any_half), unaffected by the other half.
      * Opposite-half source (e.g. exec_lo <- vcc_hi, exec_hi <- vcc_lo) collects the source half cleanly.
      * s_cselect_b32 into exec_lo / exec_hi preserves the untouched half.
  - Preserves scalar constant folding for plain SGPR destinations and spirv-val Vulkan 1.2 validation in
    both native and --wave64-emulate modes.

Standard library only; no GPU, no game files.
"""
from __future__ import annotations

import gc
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
for _p in (HERE, TOOLS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402

M32 = 0xFFFFFFFF
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)
# s[0:3] = raw V# (OOB_SELECT 3), s4 = lo mask, s5 = hi mask, s6 = aux scalar, s7 = aux scalar 2
VSHARP = [0, 0, 0xFFFFFFFF, 0x31016FAC]

# Builds wave32 VCC from s4 (bit i for lane i).
W32_BUILD_VCC = [
    "v_mov_b32_e32 v1, s4",
    "v_and_b32_e32 v2, 31, v0",
    "v_lshrrev_b32_e32 v3, v2, v1",
    "v_and_b32_e32 v3, 1, v3",
    "v_cmp_ne_u32_e64 vcc_lo, 0, v3",
]

# Builds wave64 VCC from (s4, s5) and copies into s[12:13] (MASK_LO=s12, MASK_HI=s13).
W64_BUILD_VCC = [
    "v_mov_b32_e32 v11, s5",
    "v_cmp_lt_u32_e64 vcc, v0, 32",
    "v_cndmask_b32_e64 v1, v11, s4, vcc",
    "v_and_b32_e32 v2, 31, v0",
    "v_lshrrev_b32_e32 v3, v2, v1",
    "v_and_b32_e32 v3, 1, v3",
    "v_cmp_ne_u32_e64 vcc, 0, v3",
    "s_and_b64 s[12:13], vcc, exec",
]

BIN_REFS = {
    "s_and_b32": lambda x, y: (x & y) & M32,
    "s_or_b32": lambda x, y: (x | y) & M32,
    "s_xor_b32": lambda x, y: (x ^ y) & M32,
    "s_andn2_b32": lambda x, y: (x & (~y & M32)) & M32,
    "s_orn2_b32": lambda x, y: (x | (~y & M32)) & M32,
    "s_nand_b32": lambda x, y: (~(x & y)) & M32,
    "s_nor_b32": lambda x, y: (~(x | y)) & M32,
    "s_xnor_b32": lambda x, y: (~(x ^ y)) & M32,
}


def translate_cs(name: str, lines: list[str], threads: int, emulate: bool = False) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        os.mkdir(os.path.join(tmp, "fssm"))
        src = os.path.join(tmp, "disasm", f"cs_{name}.s")
        with open(src, "w", encoding="utf-8", newline="\n") as f:
            f.write(f"; cs_{name}\n")
            for ln in lines:
                f.write(f"\t{ln}\n")
        with open(os.path.join(tmp, f"cs_{name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with open(os.path.join(tmp, "fssm", f"cs_{name}.fssm"), "wb") as f:
            f.write(bytes(0x138) + struct.pack("<4I", 0x8090, threads, 1, 1))
        old_emulate = rdna2spirv.WAVE64_EMULATE
        rdna2spirv.WAVE64_EMULATE = emulate
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                spv = rdna2spirv.translate(src)
                gc.collect()
        finally:
            rdna2spirv.WAVE64_EMULATE = old_emulate
        return spv


def run_cs(spv: str, threads: int, s4: int = 0, s5: int = 0, s6: int = 0, s7: int = 0) -> dict[int, int]:
    pc = VSHARP + [s4 & M32, s5 & M32, s6 & M32, s7 & M32] + [0] * 24
    return si.run_workgroup(
        spv, threads, threads,
        inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
        push_constants=pc, memory={},
    )


def bits_list(mask: int, width: int) -> list[int]:
    return [1 if (mask >> i) & 1 else 0 for i in range(width)]


class Wave32B32LaneMaskTests(unittest.TestCase):
    def test_s_not_b32_on_vcc_into_sgpr_and_scc(self):
        asm = W32_BUILD_VCC + [
            "s_not_b32 s10, vcc_lo",
            "s_cselect_b32 s11, 1, 0",
            "v_cndmask_b32_e64 v4, 0, 1, s10",
            "v_mov_b32_e32 v5, s11",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:128",
            "s_endpgm",
        ]
        spv = translate_cs("w32_not_sgpr", asm, 32)
        for mask in (0x00000001, 0x00000000, 0xFFFFFFFF, 0xA5A500FF, 0x80000000):
            with self.subTest(mask=hex(mask)):
                mem = run_cs(spv, 32, s4=mask)
                want_mask = (~mask) & M32
                want_scc = 1 if want_mask != 0 else 0
                got_bits = [mem[4 * i] for i in range(32)]
                got_scc = [mem[128 + 4 * i] for i in range(32)]
                self.assertEqual(got_bits, bits_list(want_mask, 32))
                self.assertEqual(got_scc, [want_scc] * 32)

    def test_s_not_b32_exec_lo_imm_and_sgpr(self):
        for src, c in (("1", 1), ("0x0f0f00ff", 0x0F0F00FF), ("-1", M32), ("0", 0), ("s4", 0x55550001)):
            with self.subTest(src=src, c=hex(c)):
                asm = [
                    "v_mov_b32_e32 v4, 10",
                    f"s_not_b32 exec_lo, {src}",
                    "s_cselect_b32 s11, 1, 0",
                    "v_mov_b32_e32 v4, 20",
                    "s_mov_b32 exec_lo, -1",
                    "v_mov_b32_e32 v5, s11",
                    "v_lshlrev_b32_e32 v6, 2, v0",
                    "buffer_store_dword v4, v6, s[0:3], 0 offen",
                    "buffer_store_dword v5, v6, s[0:3], 0 offen offset:128",
                    "s_endpgm",
                ]
                spv = translate_cs("w32_not_exec", asm, 32)
                mem = run_cs(spv, 32, s4=c)
                want_mask = (~c) & M32
                want_scc = 1 if want_mask != 0 else 0
                got_v4 = [mem[4 * i] for i in range(32)]
                got_scc = [mem[128 + 4 * i] for i in range(32)]
                self.assertEqual(got_v4, [20 if (want_mask >> i) & 1 else 10 for i in range(32)])
                self.assertEqual(got_scc, [want_scc] * 32)

    def test_s_not_and_or_xor_imm_into_vcc_lo(self):
        cases = [
            ("s_not_b32 vcc_lo, 1", (~1) & M32),
            ("s_not_b32 vcc_lo, -1", 0),
            ("s_and_b32 vcc_lo, 0x0f, 0x03", 0x03),
            ("s_or_b32 vcc_lo, 0x05, 0x10", 0x15),
            ("s_xor_b32 vcc_lo, 0x0f, 0x03", 0x0C),
            ("s_xor_b32 vcc_lo, 0x0f, 0x0f", 0),
        ]
        for insn, want_mask in cases:
            with self.subTest(insn=insn):
                asm = [
                    insn,
                    "s_cselect_b32 s11, 1, 0",
                    "s_bcnt1_i32_b32 s12, vcc_lo",
                    "v_cndmask_b32_e64 v4, 0, 1, vcc_lo",
                    "v_mov_b32_e32 v5, s11",
                    "v_mov_b32_e32 v7, s12",
                    "v_lshlrev_b32_e32 v6, 2, v0",
                    "buffer_store_dword v4, v6, s[0:3], 0 offen",
                    "buffer_store_dword v5, v6, s[0:3], 0 offen offset:128",
                    "buffer_store_dword v7, v6, s[0:3], 0 offen offset:256",
                    "s_endpgm",
                ]
                spv = translate_cs("w32_imm_vcc", asm, 32)
                mem = run_cs(spv, 32)
                want_scc = 1 if want_mask != 0 else 0
                want_cnt = bin(want_mask).count("1")
                self.assertEqual([mem[4 * i] for i in range(32)], bits_list(want_mask, 32))
                self.assertEqual([mem[128 + 4 * i] for i in range(32)], [want_scc] * 32)
                self.assertEqual([mem[256 + 4 * i] for i in range(32)], [want_cnt] * 32)


class Wave64HalfMaskBitwiseTests(unittest.TestCase):
    def test_exec_lo_and_exec_hi_preserve_other_half_and_vote_own_half_scc(self):
        # Test s_not_b32 and all 8 binary ops writing exec_lo or exec_hi.
        ops = ["s_not_b32"] + list(BIN_REFS.keys())
        # Pairs (lo_vcc, hi_vcc): include cases where lo==0 & hi!=0, lo!=0 & hi==0, single-bit, all-ones.
        pairs = [
            (0x00000001, 0x00000000),
            (0x00000000, 0x80000000),
            (0xFFFFFFFF, 0x00000000),
            (0x00000000, 0xFFFFFFFF),
            (0x0F0F00FF, 0x55AA33CC),
        ]
        for op in ops:
            for half in ("lo", "hi"):
                reg_e = f"exec_{half}"
                reg_v = f"vcc_{half}"
                # Start with EXEC = 0x55555555_AAAAAAAA (alternating) so non-trivial ops are distinguished.
                init_exec_lo, init_exec_hi = 0x55555555, 0xAAAAAAAA
                insn = f"{op} {reg_e}, {reg_v}" if op == "s_not_b32" else f"{op} {reg_e}, {reg_e}, {reg_v}"
                asm = W64_BUILD_VCC + [
                    "s_mov_b64 s[14:15], exec",
                    f"s_mov_b32 exec_lo, {init_exec_lo:#x}",
                    f"s_mov_b32 exec_hi, {init_exec_hi:#x}",
                    "v_mov_b32_e32 v4, 10",
                    "s_mov_b64 exec, s[14:15]",
                    "v_mov_b32_e32 v4, 10",
                    f"s_mov_b32 exec_lo, {init_exec_lo:#x}",
                    f"s_mov_b32 exec_hi, {init_exec_hi:#x}",
                    insn,
                    "s_cselect_b32 s11, 1, 0",
                    "v_mov_b32_e32 v4, 20",
                    "s_mov_b64 exec, s[14:15]",
                    "v_mov_b32_e32 v5, s11",
                    "v_lshlrev_b32_e32 v6, 2, v0",
                    "buffer_store_dword v4, v6, s[0:3], 0 offen",
                    "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
                    "s_endpgm",
                ]
                spv = translate_cs(f"w64_{op}_{half}", asm, 64)
                for lo, hi in pairs:
                    with self.subTest(op=op, half=half, lo=hex(lo), hi=hex(hi)):
                        mem = run_cs(spv, 64, s4=lo, s5=hi)
                        if half == "lo":
                            new_h = ((~lo) & M32) if op == "s_not_b32" else BIN_REFS[op](init_exec_lo, lo)
                            want_exec = new_h | (init_exec_hi << 32)
                        else:
                            new_h = ((~hi) & M32) if op == "s_not_b32" else BIN_REFS[op](init_exec_hi, hi)
                            want_exec = init_exec_lo | (new_h << 32)
                        want_scc = 1 if new_h != 0 else 0
                        got_v4 = [mem[4 * i] for i in range(64)]
                        got_scc = [mem[256 + 4 * i] for i in range(64)]
                        self.assertEqual(got_v4, [20 if (want_exec >> i) & 1 else 10 for i in range(64)])
                        self.assertEqual(got_scc, [want_scc] * 64)

    def test_wave64_sgpr_half_mask_scc(self):
        # Writing an SGPR from vcc_lo / vcc_hi (and MASK_LO s12 / MASK_HI s13) votes only over that half.
        asm = W64_BUILD_VCC + [
            "s_and_b32 s10, exec_lo, vcc_lo",
            "s_cselect_b32 s20, 1, 0",
            "s_and_b32 s11, exec_hi, vcc_hi",
            "s_cselect_b32 s21, 1, 0",
            "s_not_b32 s16, s12",
            "s_cselect_b32 s22, 1, 0",
            "s_not_b32 s17, s13",
            "s_cselect_b32 s23, 1, 0",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "v_mov_b32_e32 v4, s20",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "v_mov_b32_e32 v4, s21",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:256",
            "v_mov_b32_e32 v4, s22",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:512",
            "v_mov_b32_e32 v4, s23",
            "buffer_store_dword v4, v6, s[0:3], 0 offen offset:768",
            "s_endpgm",
        ]
        spv = translate_cs("w64_sgpr_half_scc", asm, 64)
        for lo, hi in ((0x00000001, 0x00000000), (0x00000000, 0x00000001),
                       (0xFFFFFFFF, 0x00000000), (0x00000000, 0xFFFFFFFF)):
            with self.subTest(lo=hex(lo), hi=hex(hi)):
                mem = run_cs(spv, 64, s4=lo, s5=hi)
                self.assertEqual([mem[4 * i] for i in range(64)], [int(lo != 0)] * 64)
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [int(hi != 0)] * 64)
                self.assertEqual([mem[512 + 4 * i] for i in range(64)], [int(((~lo) & M32) != 0)] * 64)
                self.assertEqual([mem[768 + 4 * i] for i in range(64)], [int(((~hi) & M32) != 0)] * 64)

    def test_wave64_vcc_half_const_and_cross_half_exec(self):
        # 1) s_and_b32 vcc_lo, vcc_lo, 0x0f0f00ff preserves vcc_hi and sets SCC from lo only.
        # 2) s_and_b32 exec_lo, exec_lo, vcc_hi applies vcc_hi bits to lanes 0..31 and preserves exec_hi.
        # 3) s_cselect_b32 exec_hi, vcc_hi, 0x0000000f preserves exec_lo.
        asm = W64_BUILD_VCC + [
            "s_and_b32 vcc_lo, vcc_lo, 0x0f0f00ff",
            "s_cselect_b32 s20, 1, 0",
            "v_cndmask_b32_e64 v4, 0, 1, vcc",
            "s_mov_b64 s[14:15], exec",
            "v_mov_b32_e32 v7, 10",
            "s_and_b32 exec_lo, exec_lo, vcc_hi",
            "s_cselect_b32 s21, 1, 0",
            "v_mov_b32_e32 v7, 20",
            "s_mov_b64 exec, s[14:15]",
            "v_mov_b32_e32 v8, 30",
            "s_cmp_eq_u32 s6, 1",
            "s_cselect_b32 exec_hi, vcc_hi, 0x0000000f",
            "v_mov_b32_e32 v8, 40",
            "s_mov_b64 exec, s[14:15]",
            "v_lshlrev_b32_e32 v6, 2, v0",
            "buffer_store_dword v4, v6, s[0:3], 0 offen",
            "v_mov_b32_e32 v5, s20",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
            "buffer_store_dword v7, v6, s[0:3], 0 offen offset:512",
            "v_mov_b32_e32 v5, s21",
            "buffer_store_dword v5, v6, s[0:3], 0 offen offset:768",
            "buffer_store_dword v8, v6, s[0:3], 0 offen offset:1024",
            "s_endpgm",
        ]
        spv = translate_cs("w64_vcc_const_cross", asm, 64)
        for lo, hi, s6 in ((0xF0F0FF00, 0x00010005, 0), (0xFFFFFFFF, 0x80000001, 1), (0x00000001, 0x00000000, 1)):
            with self.subTest(lo=hex(lo), hi=hex(hi), s6=s6):
                mem = run_cs(spv, 64, s4=lo, s5=hi, s6=s6)
                vcc_lo = lo & 0x0F0F00FF
                vcc64 = vcc_lo | (hi << 32)
                self.assertEqual([mem[4 * i] for i in range(64)], bits_list(vcc64, 64))
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [int(vcc_lo != 0)] * 64)
                exec_cross = hi | (M32 << 32)
                self.assertEqual([mem[512 + 4 * i] for i in range(64)],
                                 [20 if (exec_cross >> i) & 1 else 10 for i in range(64)])
                self.assertEqual([mem[768 + 4 * i] for i in range(64)], [int(hi != 0)] * 64)
                csel_hi = hi if s6 == 1 else 0x0000000F
                exec_csel = M32 | (csel_hi << 32)
                self.assertEqual([mem[1024 + 4 * i] for i in range(64)],
                                 [40 if (exec_csel >> i) & 1 else 30 for i in range(64)])

    def test_wave64_vcc_half_after_whole_vcc_const_write(self):
        # Regression for @LloydNicholson's RX 9060 XT finding: s_mov_b64 vcc, -1 (or 0) must keep vcc
        # in the lane-mask state so a subsequent 32-bit half op on vcc_lo / vcc_hi extracts per-lane bits
        # and preserves the other half.
        cases = [
            (["s_mov_b64 vcc, -1", "s_and_b32 vcc_lo, vcc_lo, 0xf0"], 0x000000F0 | (M32 << 32), 1),
            (["s_mov_b64 vcc, 0", "s_or_b32 vcc_hi, vcc_hi, 0x0f"], 0x0000000F << 32, 1),
            (["s_mov_b64 vcc, -1", "s_not_b32 vcc_hi, 0xfffffff0"], M32 | (0x0000000F << 32), 1),
            (["s_mov_b64 vcc, -1", "s_and_b32 vcc_lo, vcc_lo, 0"], M32 << 32, 0),
        ]
        for idx, (setup, want_vcc, want_scc) in enumerate(cases):
            with self.subTest(setup=setup):
                asm = setup + [
                    "s_cselect_b32 s20, 1, 0",
                    "v_cndmask_b32_e64 v4, 0, 1, vcc",
                    "v_mov_b32_e32 v5, s20",
                    "v_lshlrev_b32_e32 v6, 2, v0",
                    "buffer_store_dword v4, v6, s[0:3], 0 offen",
                    "buffer_store_dword v5, v6, s[0:3], 0 offen offset:256",
                    "s_endpgm",
                ]
                spv = translate_cs(f"w64_vcc_const_init_{idx}", asm, 64)
                mem = run_cs(spv, 64)
                self.assertEqual([mem[4 * i] for i in range(64)], bits_list(want_vcc, 64))
                self.assertEqual([mem[256 + 4 * i] for i in range(64)], [want_scc] * 64)

    def test_spirv_val_native_and_wave64_emulate(self):
        spirv_as = shutil.which("spirv-as")
        spirv_val = shutil.which("spirv-val")
        if not (spirv_as and spirv_val):
            self.skipTest("spirv-as / spirv-val not on PATH")
        asm_w64 = W64_BUILD_VCC + [
            "s_mov_b64 s[14:15], exec",
            "s_not_b32 exec_lo, vcc_lo",
            "s_and_b32 exec_hi, exec_hi, 0x0f0f00ff",
            "s_or_b32 exec_lo, exec_lo, vcc_hi",
            "s_xor_b32 vcc_hi, vcc_hi, 0x55555555",
            "s_cselect_b32 exec_hi, vcc_hi, 0x000000ff",
            "s_mov_b64 exec, s[14:15]",
            "s_endpgm",
        ]
        for emulate in (False, True):
            spv = translate_cs(f"w64_val_{int(emulate)}", asm_w64, 64, emulate=emulate)
            with tempfile.TemporaryDirectory() as tmp:
                spvasm = os.path.join(tmp, "m.spvasm")
                spvbin = os.path.join(tmp, "m.spv")
                with open(spvasm, "w", encoding="utf-8", newline="\n") as f:
                    f.write(spv)
                r1 = subprocess.run([spirv_as, "--target-env", "vulkan1.2", spvasm, "-o", spvbin],
                                    capture_output=True, text=True)
                self.assertEqual(r1.returncode, 0, r1.stderr)
                r2 = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spvbin],
                                    capture_output=True, text=True)
                self.assertEqual(r2.returncode, 0, r2.stderr)


if __name__ == "__main__":
    unittest.main()
