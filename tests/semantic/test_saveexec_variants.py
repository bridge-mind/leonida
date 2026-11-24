"""Lane-level tests for every RDNA2 s_<op>_saveexec_b32/b64 and s_andn1/andn2_wrexec, against tools/cf_ref.py.

D = EXEC; EXEC = f(S0, EXEC); SCC = (EXEC != 0) for saveexec, and EXEC = f(S0, EXEC); D = EXEC for wrexec. Each case
runs a translated pixel shader in tests/semantic/spirv_wave_interp.py as one wave whose EXEC starts partial (v_cmpx),
with S0 a different lane range, and checks every lane's saved bit, new EXEC bit and the wave's SCC against the ISA
model in tools/cf_ref.py (_SAVEEXEC, the table the FUZZ-8 differential fuzzer uses). Lanes that EXEC had off and that
f turns on (nand, nor, orn1, orn2, xnor, ...) become active, as on hardware. Standard library only; no game files.
"""
import os
import sys
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_salu64_bitwise_data import run_ps_wave  # noqa: E402  (also puts tools/ on sys.path)
import cf_ref  # noqa: E402

KINDS = ["and", "or", "xor", "andn1", "andn2", "orn1", "orn2", "nand", "nor", "xnor"]


def lanes(lo, hi, wave):
    return sum(1 << i for i in range(wave) if lo <= i < hi)


def case(op_line, wave, exec_hi, s0_lo, s0_hi):
    """EXEC = lanes [0, exec_hi), S0 = lanes [s0_lo, s0_hi); run `op_line` (D in s[4:5] / s4, S0 in s[2:3] / s2)."""
    w64 = wave == 64
    s0, d = ("s[2:3]", "s[4:5]") if w64 else ("s2", "s4")
    lines = ["v_mbcnt_lo_u32_b32 v4, -1, 0"] + (["v_mbcnt_hi_u32_b32 v4, -1, v4"] if w64 else []) + [
        f"v_cmp_ge_u32_e64 {s0}, v4, {s0_lo}",
        f"v_cmp_lt_u32_e64 {'s[6:7]' if w64 else 's6'}, v4, {s0_hi}",
        f"s_and_b{wave} {s0}, {s0}, {'s[6:7]' if w64 else 's6'}",
        "v_mov_b32 v1, 0x11",
        f"v_cmpx_gt_u32_e32 {exec_hi}, v4",
        op_line.format(d=d, s0=s0),
        "s_cselect_b32 s10, 1, 0",
        "v_mov_b32 v1, 0x22",
        f"s_mov_b{wave} exec, -1",
        f"v_cndmask_b32_e64 v2, 0, 1, {d}",
        "v_mov_b32 v0, s10",
        "v_mov_b32 v3, 0",
    ]
    outs, _ = run_ps_wave(lines, wave=wave)
    return outs


class SaveexecVariants(unittest.TestCase):
    def test_null_mask_source(self) -> None:
        fixture = Path(HERE) / "fixtures" / "null_mask_source.asm"
        lines = [line for line in fixture.read_text(encoding="utf-8").splitlines()
                 if line and not line.startswith(";")]
        outputs, _ = run_ps_wave(lines, wave=64)
        self.assertEqual(outputs, [(1, 64, 0xFFFFFFFF, 7)] * 64)

    def check(self, kind, wave, exec_hi, s0_lo, s0_hi, wrexec=False):
        m = (1 << wave) - 1
        e, s = lanes(0, exec_hi, wave), lanes(s0_lo, s0_hi, wave)
        new = cf_ref._SAVEEXEC[kind](s, e, m) & m
        saved = new if wrexec else e
        op = f"s_{kind}_{'wrexec' if wrexec else 'saveexec'}_b{wave} {{d}}, {{s0}}"
        outs = case(op, wave, exec_hi, s0_lo, s0_hi)
        for lane in range(wave):
            want = (int(new != 0), 0x22 if new >> lane & 1 else 0x11, saved >> lane & 1, 0)
            self.assertEqual(outs[lane], want, f"{op} lane {lane}")

    def test_saveexec_wave32(self):
        for kind in KINDS:
            for exec_hi, s0 in ((24, (10, 30)), (32, (0, 32)), (8, (8, 16))):
                with self.subTest(kind=kind, exec_hi=exec_hi, s0=s0):
                    self.check(kind, 32, exec_hi, *s0)

    def test_saveexec_wave64(self):
        for kind in KINDS:
            for exec_hi, s0 in ((40, (20, 50)), (64, (0, 64))):
                with self.subTest(kind=kind, exec_hi=exec_hi, s0=s0):
                    self.check(kind, 64, exec_hi, *s0)

    def test_wrexec(self):
        for kind in ("andn1", "andn2"):
            for wave, exec_hi, s0 in ((32, 24, (10, 30)), (64, 40, (20, 50)), (32, 16, (16, 32))):
                with self.subTest(kind=kind, wave=wave):
                    self.check(kind, wave, exec_hi, *s0, wrexec=True)

    def test_sdst_src0_aliasing(self):
        # 1. s_nand_saveexec_b32 s4, s4
        lines32 = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_cmp_ge_u32_e64 s4, v4, 8",
            "v_cmp_lt_u32_e64 s6, v4, 24",
            "s_and_b32 s4, s4, s6",
            "v_mov_b32 v1, 0x11",
            "v_cmpx_gt_u32_e32 16, v4",
            "s_nand_saveexec_b32 s4, s4",
            "s_cselect_b32 s10, 1, 0",
            "v_mov_b32 v1, 0x22",
            "s_mov_b32 exec, -1",
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v3, 0",
        ]
        outs32, _ = run_ps_wave(lines32, wave=32)
        for lane in range(32):
            want_scc = 1
            want_exec = 0x11 if 8 <= lane < 16 else 0x22
            want_saved = 1 if lane < 16 else 0
            self.assertEqual(outs32[lane], (want_scc, want_exec, want_saved, 0), f"lane {lane}")

        # 2. s_nand_saveexec_b64 s[4:5], s[4:5]
        lines64 = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_mbcnt_hi_u32_b32 v4, -1, v4",
            "v_cmp_ge_u32_e64 s[4:5], v4, 16",
            "v_cmp_lt_u32_e64 s[6:7], v4, 48",
            "s_and_b64 s[4:5], s[4:5], s[6:7]",
            "v_mov_b32 v1, 0x11",
            "v_cmpx_gt_u32_e32 32, v4",
            "s_nand_saveexec_b64 s[4:5], s[4:5]",
            "s_cselect_b32 s10, 1, 0",
            "v_mov_b32 v1, 0x22",
            "s_mov_b64 exec, -1",
            "v_cndmask_b32_e64 v2, 0, 1, s[4:5]",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v3, 0",
        ]
        outs64, _ = run_ps_wave(lines64, wave=64)
        for lane in range(64):
            want_scc = 1
            want_exec = 0x11 if 16 <= lane < 32 else 0x22
            want_saved = 1 if lane < 32 else 0
            self.assertEqual(outs64[lane], (want_scc, want_exec, want_saved, 0), f"lane {lane}")

        # 3. s_andn1_wrexec_b32 s4, s4
        lines_wr = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_cmp_ge_u32_e64 s4, v4, 8",
            "v_cmp_lt_u32_e64 s6, v4, 24",
            "s_and_b32 s4, s4, s6",
            "v_mov_b32 v1, 0x11",
            "v_cmpx_gt_u32_e32 16, v4",
            "s_andn1_wrexec_b32 s4, s4",
            "s_cselect_b32 s10, 1, 0",
            "v_mov_b32 v1, 0x22",
            "s_mov_b32 exec, -1",
            "v_cndmask_b32_e64 v2, 0, 1, s4",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v3, 0",
        ]
        outs_wr, _ = run_ps_wave(lines_wr, wave=32)
        for lane in range(32):
            want_scc = 1
            want_exec = 0x22 if lane < 8 else 0x11
            want_saved = 1 if lane < 8 else 0
            self.assertEqual(outs_wr[lane], (want_scc, want_exec, want_saved, 0), f"lane {lane}")

    def test_wave64_upper_dword_scc_boundary(self):
        # src0 = 0x00000001_00000000 (only lane 32 active)
        lines = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_mbcnt_hi_u32_b32 v4, -1, v4",
            "v_cmp_eq_u32_e64 s[2:3], v4, 32",
            "s_mov_b64 exec, -1",
            "s_and_saveexec_b64 s[4:5], s[2:3]",
            "s_cselect_b32 s10, 1, 0",
            "s_mov_b64 exec, -1",
            "v_cndmask_b32_e64 v2, 0, 1, s[4:5]",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v3, 0",
        ]
        outs, _ = run_ps_wave(lines, wave=64)
        for lane in range(64):
            self.assertEqual(outs[lane][0], 1, "SCC should be 1 when bit 32 is non-zero")
            self.assertEqual(outs[lane][2], 1, "saved exec should be all 1s")

        # paired with src0 = 0 -> SCC must be 0
        lines_zero = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_mbcnt_hi_u32_b32 v4, -1, v4",
            "s_mov_b64 exec, -1",
            "s_and_saveexec_b64 s[4:5], 0",
            "s_cselect_b32 s10, 1, 0",
            "s_mov_b64 exec, -1",
            "v_cndmask_b32_e64 v2, 0, 1, s[4:5]",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v3, 0",
        ]
        outs_zero, _ = run_ps_wave(lines_zero, wave=64)
        for lane in range(64):
            self.assertEqual(outs_zero[lane][0], 0, "SCC should be 0 when new EXEC is 0")

    def test_asymmetric_operand_order_guard(self):
        # exec = [0, 20), s0 = [10, 30) in wave32
        m = (1 << 32) - 1
        e, s = lanes(0, 20, 32), lanes(10, 30, 32)
        cases = [
            ("andn1", (~s & e) & m),
            ("andn2", (s & ~e) & m),
            ("orn1", (~s | e) & m),
            ("orn2", (s | ~e) & m),
        ]
        for kind, want_new in cases:
            with self.subTest(kind=kind):
                op = f"s_{kind}_saveexec_b32 {{d}}, {{s0}}"
                outs = case(op, 32, 20, 10, 30)
                for lane in range(32):
                    want_bit = (want_new >> lane) & 1
                    self.assertEqual(outs[lane][1], 0x22 if want_bit else 0x11, f"{kind} lane {lane}")

    def test_lane_reactivation(self):
        # Initial EXEC = [0, 16). s_nor_saveexec_b32 with s2 = [8, 24) reactivates lanes [24, 32).
        lines = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_cmp_ge_u32_e64 s2, v4, 8",
            "v_cmp_lt_u32_e64 s6, v4, 24",
            "s_and_b32 s2, s2, s6",
            "v_mov_b32 v1, 0x55",
            "v_cmpx_gt_u32_e32 16, v4",
            "s_nor_saveexec_b32 s4, s2",
            "v_mov_b32 v1, 0xAA",
            "s_mov_b32 exec, -1",
            "v_mov_b32 v0, v1",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        outs, _ = run_ps_wave(lines, wave=32)
        for lane in range(32):
            want = 0xAA if 24 <= lane < 32 else 0x55
            self.assertEqual(outs[lane][0], want, f"lane {lane}")

    def test_inline_constants_and_null_dst(self):
        # 1. inline constant -1 in wave64
        lines_imm = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "v_mbcnt_hi_u32_b32 v4, -1, v4",
            "s_mov_b64 exec, -1",
            "s_andn1_saveexec_b64 s[4:5], -1",
            "s_cselect_b32 s10, 1, 0",
            "s_mov_b64 exec, -1",
            "v_cndmask_b32_e64 v2, 0, 1, s[4:5]",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v3, 0",
        ]
        outs_imm, _ = run_ps_wave(lines_imm, wave=64)
        for lane in range(64):
            self.assertEqual(outs_imm[lane][0], 0, "SCC should be 0 because ~(-1) & EXEC == 0")
            self.assertEqual(outs_imm[lane][2], 1, "saved exec should be all 1s")

        # 2. null dst: s_nand_saveexec_b32 null, s4
        lines_null = [
            "v_mbcnt_lo_u32_b32 v4, -1, 0",
            "s_mov_b32 s4, 0x0F0F0F0F",
            "s_mov_b32 exec, -1",
            "s_nand_saveexec_b32 null, s4",
            "s_cselect_b32 s10, 1, 0",
            "s_mov_b32 exec, -1",
            "v_mov_b32 v0, s10",
            "v_mov_b32 v1, 0",
            "v_mov_b32 v2, 0",
            "v_mov_b32 v3, 0",
        ]
        outs_null, _ = run_ps_wave(lines_null, wave=32)
        for lane in range(32):
            self.assertEqual(outs_null[lane][0], 1, "SCC should be 1")


if __name__ == "__main__":
    unittest.main()
