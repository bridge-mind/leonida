"""Runs of lane-local VALU instructions under one `if (active)` (rdna2spirv.py Gen.insns), checked by value.

An if/else over EXEC (s_and_saveexec, s_andn2 exec, s_or exec) with integer VALU ops on both sides, run on every
lane of a wave32 and a wave64 pixel shader with spirv_wave_interp: inactive lanes keep their old values, active
ones take the new, and the results equal those of the per-write OpSelect translation (MIN_ACTIVE_RUN out of
reach). A shader that never writes EXEC translates as before, and an instruction wrongly judged lane-local stops
the translation instead of losing its EXEC behaviour.

    python -I tests/semantic/test_active_runs.py -v
"""
import os
import re
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import spirv_wave_interp as wi  # noqa: E402
from support import STUB_DXBC, rdna2spirv  # noqa: E402


def translate_ps(lines):
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        path = os.path.join(tmp, "disasm", "ps_runs.s")
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write("; ps_runs\n" + "".join(f"\t{ln}\n" for ln in lines))
        with open(os.path.join(tmp, "ps_runs.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            return rdna2spirv.translate(path)


def run_wave(spv, wave):
    res = wi.run_wave(spv, wave=wave)
    return [o[next(iter(o))] for o in res["outputs"]]


def if_else(wave):
    """Lanes 0-9 take the then side, the others the else side; v0..v3 are exported."""
    w64 = wave == 64
    s = "s[0:1]" if w64 else "s0"
    exec_ = "exec" if w64 else "exec_lo"
    return (["v_mbcnt_lo_u32_b32 v4, -1, 0"] + (["v_mbcnt_hi_u32_b32 v4, -1, v4"] if w64 else []) + [
        "v_mov_b32 v0, 0x100", "v_mov_b32 v1, 0x200", "v_mov_b32 v2, 0x300", "v_mov_b32 v3, 0x400",   # a run
        "v_cmp_gt_u32_e32 vcc, 10, v4",
        f"s_and_saveexec_b{wave} {s}, {'vcc' if w64 else 'vcc_lo'}",
        "v_add_nc_u32 v0, v0, v4",          # then: one run of three
        "v_xor_b32 v1, 0x55, v4",
        "v_lshlrev_b32 v2, 4, v4",
        f"s_andn2_b{wave} {exec_}, {s}, {exec_}",
        "v_or_b32 v3, v3, v4",              # else: a run of one (selects), the carry op, a run of two
        f"v_add_co_u32 v2, {'s[4:5]' if w64 else 's4'}, v2, 7",
        "v_mul_lo_u32 v3, v3, 3",
        "v_sub_nc_u32 v0, v0, 1",
        f"s_or_b{wave} {exec_}, {exec_}, {s}",
        "v_add_nc_u32 v1, v1, 1",           # all lanes again
        "v_add_nc_u32 v0, v0, 2",
        "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"])


def expected(lane):
    if lane < 10:
        return (0x100 + lane + 2, (0x55 ^ lane) + 1, lane << 4, 0x400)
    return (0x100 - 1 + 2, 0x200 + 1, 0x300 + 7, (0x400 | lane) * 3)


def runs(spv):
    """The SPIR-V lines inside each run (from its label to the branch out)."""
    return re.findall(r"%\w+_run = OpLabel\n(.*?)\n\s*OpBranch ", spv, re.S)


class ActiveRuns(unittest.TestCase):
    def test_inactive_lanes_keep_their_values(self):
        for wave in (32, 64):
            with self.subTest(wave=wave):
                spv = translate_ps(if_else(wave))
                self.assertEqual(len(runs(spv)), 4, "the four moves, then side, end of else side, after if/else")
                for body in runs(spv):
                    self.assertNotIn("%active", body)      # no per-write select inside a run
                outs = run_wave(spv, wave)
                for lane in range(wave):
                    self.assertEqual(outs[lane], expected(lane), f"lane {lane}")

    def test_same_values_as_per_write_selects(self):
        for wave in (32, 64):
            with self.subTest(wave=wave):
                with patch.object(rdna2spirv, "MIN_ACTIVE_RUN", 10 ** 9):
                    plain = translate_ps(if_else(wave))
                self.assertEqual(runs(plain), [])
                self.assertEqual(run_wave(translate_ps(if_else(wave)), wave), run_wave(plain, wave))

    def test_no_runs_without_exec_writes(self):
        lines = ["v_mov_b32 v0, 1", "v_add_nc_u32 v1, v0, 2", "v_xor_b32 v2, v1, 3", "v_mov_b32 v3, v2",
                 "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
        with patch.object(rdna2spirv, "MIN_ACTIVE_RUN", 10 ** 9):
            plain = translate_ps(lines)
        self.assertEqual(translate_ps(lines), plain)

    def test_an_op_wrongly_judged_lane_local_stops_the_translation(self):
        # Every VALU op judged lane-local: the v_cmp after the four moves joins their run and writes VCC there.
        with patch.object(rdna2spirv.Gen, "lane_local", lambda self, ins: ins["op"].startswith("v_")):
            with self.assertRaisesRegex(AssertionError, r"v_cmp_gt_u32_e32 is not lane-local .*OpStore %vcc"):
                translate_ps(if_else(32))


if __name__ == "__main__":
    unittest.main()
