"""The PS valid mask: a pixel is discarded by EXEC at its last executed `exp ... vm`, not by EXEC at the end, and a
lane that was off at the last export to the color target is not written either.

RDNA2 ISA reference guide (30-November-2020), 11.1 and 11.2.1: with VM set EXEC is the valid mask, the final one
sent is used, and an export to NULL may carry only the mask. Measured on RDNA2 (gfx1036, ps_oracle.py of #3061, 18
programs; docs/design.md, EXEC): that rule holds, and the color export has its own mask, EXEC at that export, which
is also what decides on a path with no `vm` export (class ColorExportMask). Hand-written programs, no tools needed.
"""
from pathlib import Path
import unittest

import support
import rdna2spirv
import shader_ir
from spirv_mini_interp import run_fragment
from spirv_wave_interp import run_wave

ONE = 0x3F800000
SEED = [f"v_mov_b32_e32 v2, 0x{ONE:x}", "v_mov_b32_e32 v1, 1"]
OFF = ["v_cmpx_eq_u32_e32 0, v1"]
ON = ["s_mov_b64 exec, -1"]
EXP = ["exp mrt0 v2, v2, v2, v2 done vm"]
EXP_NO_VM = ["exp mrt0 v2, v2, v2, v2"]
EXP_MRT1 = ["exp mrt1 v2, v2, v2, v2"]
NULL = ["exp null off, off, off, off done vm"]
END = ["s_endpgm"]
KILLED = {}
MRT0 = {"%out_mrt0": (ONE,) * 4}


def sized(lines):
    """The lines with a placeholder encoding of the right size: branch offsets are counted in dwords of it."""
    out = []
    for line in lines:
        size = 8 if line.startswith("exp ") or "0x3f800000" in line else 4
        out.append(f"{line} ; encoding: [" + ",".join(["0x00"] * size) + "]")
    return out


def translate(lines):
    return support.translate_lines("ps_valid_mask", sized(lines))


def dwords(lines):
    return sum(2 if line.startswith("exp ") or "0x3f800000" in line else 1 for line in lines)


class StraightLine(unittest.TestCase):
    def check(self, lines, want):
        self.assertEqual(run_fragment(translate(SEED + lines + END)), want)

    def test_disjoint_color_exports_accumulate_per_target(self) -> None:
        fixture = Path(__file__).with_name("fixtures") / "partial_color_exports.asm"
        shader = support.translate_lines("ps_partial_exports", fixture.read_text(encoding="ascii").splitlines())
        self.assertEqual(run_fragment(shader), {
            "%out_mrt0": (0x3F800000, 0x40000000, 0x40400000, 0x40800000),
            "%out_mrt1": (0x40A00000, 0x40C00000, 0x40E00000, 0x41000000),
        })

    def test_first_partial_color_export_has_zero_unwritten_channels(self) -> None:
        lines = ["exp mrt0 off, v2, off, v2 done vm"]
        shader = translate(SEED + lines + END)
        self.assertEqual(run_fragment(shader), {"%out_mrt0": (0, ONE, 0, ONE)})
        # The interpreter defaults unwritten outputs to zero. Vulkan needs an explicit initializer here.
        self.assertIn("%initial_mrt = OpConstantNull %v4float", shader)
        self.assertIn("%out_mrt0 = OpVariable %_ptr_Output_v4 Output %initial_mrt", shader)

    def test_partial_export_obeys_exec_and_replaces_enabled_channels(self) -> None:
        lines = ["v_mov_b32_e32 v3, 0x40000000", "exp mrt0 v2, v2, v2, v2", "s_waitcnt expcnt(0)"] + OFF
        lines += ["exp mrt0 off, v3, v3, off", "s_waitcnt expcnt(0)"] + ON
        lines += ["exp mrt0 off, off, v3, off done vm"]
        self.check(lines, {"%out_mrt0": (ONE, ONE, 0x40000000, ONE)})

    def test_exec_unchanged_after_the_export(self):
        self.check(EXP, MRT0)
        self.check(OFF + EXP, KILLED)

    def test_off_at_the_export_and_back_on_is_killed(self):
        self.check(OFF + EXP + ON, KILLED)

    def test_on_at_the_export_and_off_afterwards_is_kept(self):
        self.check(EXP + OFF, MRT0)
        self.check(EXP + ["s_mov_b64 exec, 0"], MRT0)

    def test_null_export_carries_the_mask_too(self):
        self.check(EXP_NO_VM + OFF + NULL + ON, KILLED)
        self.check(EXP_NO_VM + NULL + OFF, MRT0)

    def test_compressed_export_carries_the_mask_too(self):
        compr = ["exp mrt0 v2, v2, v2, v2 done compr vm"]
        self.check(OFF + compr + ON, KILLED)
        self.assertEqual(list(run_fragment(translate(SEED + compr + OFF + END))), ["%out_mrt0"])

    def test_depth_export_carries_the_mask_too(self):
        depth = ["exp mrtz v2, off, off, off done vm"]
        self.check(EXP_NO_VM + OFF + depth + ON, KILLED)
        self.assertEqual(run_fragment(translate(SEED + EXP_NO_VM + depth + OFF + END))["%out_mrt0"], (ONE,) * 4)

    def test_the_last_vm_export_wins(self):
        first = ["exp mrt0 v2, v2, v2, v2 vm"]
        self.check(first + OFF + NULL, KILLED)
        self.check(first + OFF + ON + NULL + OFF, MRT0)

    def test_a_later_export_without_vm_does_not_move_the_mask(self):
        self.check(EXP + OFF + EXP_MRT1, {**MRT0, "%out_mrt1": (0,) * 4})      # 0: the interpreter's unwritten output

    def test_without_a_vm_export_exec_at_the_color_export_decides(self):
        # Hardware: `exp mrt0 done` with every lane on, EXEC cut afterwards: written; off at the export and back on
        # afterwards: not written. Until 2026-11-23 EXEC at the end decided, the other way round for both.
        self.check(EXP_NO_VM + OFF, MRT0)
        self.check(EXP_NO_VM + OFF + ON, MRT0)
        self.check(OFF + EXP_NO_VM + ON, KILLED)
        self.check(OFF + EXP_NO_VM, KILLED)


class ColorExportMask(unittest.TestCase):
    """Measured on RDNA2 (gfx1036): the rows of the docs/design.md EXEC table where EXEC changes between a color
    export and the last `vm` export. A = 1.0 (v2), B = 2.0 (v3)."""
    B = ["v_mov_b32_e32 v3, 2.0"]
    EXP_B = ["exp mrt0 v3, v3, v3, v3"]
    TWO = {"%out_mrt0": (0x40000000,) * 4}

    def check(self, lines, want):
        self.assertEqual(run_fragment(translate(SEED + self.B + lines + END)), want)

    def test_off_at_the_color_export_is_not_written_even_with_a_later_full_vm(self):
        self.check(OFF + EXP_NO_VM + ON + NULL, KILLED)
        self.check(EXP_NO_VM + NULL, MRT0)

    def test_the_last_color_export_is_the_one_that_counts(self):
        self.check(EXP_NO_VM + OFF + self.EXP_B + ON + NULL, KILLED)      # A from every lane, then B without this one
        self.check(OFF + EXP_NO_VM + ON + self.EXP_B + NULL, self.TWO)    # off at A, on at B

    def test_both_masks_apply(self):
        first = ["exp mrt0 v2, v2, v2, v2 vm"]
        self.check(first + OFF + ["exp mrt0 v3, v3, v3, v3 done"] + ON, KILLED)   # vm: on; last color export: off
        self.check(OFF + first + ON + ["exp mrt0 v3, v3, v3, v3 done"], KILLED)   # vm: off; last color export: on
        self.check(first + ["exp mrt0 v3, v3, v3, v3 done"] + OFF, self.TWO)

    def test_another_target_keeps_the_pixel(self):
        # Not measured (one target on the hardware side): a lane that exported to one target stays, as before.
        self.check(EXP + OFF + EXP_MRT1, {**MRT0, "%out_mrt1": (0,) * 4})


class GpuFixture(unittest.TestCase):
    def test_ps_exp_color_mask(self):
        """runtime/test/shaders/ps_exp_color_mask.asm: the words of its header, from every lane."""
        out = run_fragment(support.translate_fixture("ps_exp_color_mask"))
        self.assertEqual(out, {"%out_mrt0": (0x40000000, 0x40800000, 0x40000000, 0x40800000)})


class Branches(unittest.TestCase):
    def skip_over(self, taken, skipped):
        """SCC = `taken`, then a branch over `skipped`."""
        return [f"s_cmp_eq_u32 0, {0 if taken else 1}", f"s_cbranch_scc1 {dwords(skipped)}"] + skipped

    def test_exec_write_in_a_later_block(self):
        for taken in (False, True):
            body = EXP + self.skip_over(taken, OFF)
            self.assertEqual(run_fragment(translate(SEED + body + END)), MRT0, taken)
            body = OFF + EXP + self.skip_over(taken, ON)
            self.assertEqual(run_fragment(translate(SEED + body + END)), KILLED, taken)

    def test_exec_write_several_blocks_later(self):
        far = self.skip_over(False, ["v_mov_b32_e32 v3, 0"])
        body = EXP + far + self.skip_over(False, OFF)
        self.assertEqual(run_fragment(translate(SEED + body + END)), MRT0)
        body = OFF + EXP + far + self.skip_over(False, ON)
        self.assertEqual(run_fragment(translate(SEED + body + END)), KILLED)

    def test_a_path_around_the_vm_export_uses_exec_at_the_color_export(self):
        # The branch skips the `vm` export: that path has no valid mask, so EXEC at its color export decides.
        body = EXP_NO_VM + self.skip_over(True, NULL) + OFF
        self.assertEqual(run_fragment(translate(SEED + body + END)), MRT0)
        body = OFF + EXP_NO_VM + ON + self.skip_over(True, NULL)
        self.assertEqual(run_fragment(translate(SEED + body + END)), KILLED)
        body = EXP_NO_VM + self.skip_over(False, NULL) + OFF
        self.assertEqual(run_fragment(translate(SEED + body + END)), MRT0)

    def test_exec_write_earlier_in_a_loop(self):
        # Two rounds. EXEC is written above the export, so only the back edge puts that write after it.
        head = SEED + EXP_NO_VM + ["s_mov_b32 s4, 0"]
        loop = ON + ["s_cmp_eq_u32 s4, 0", "s_cbranch_scc1 1"] + OFF + NULL + ["s_add_u32 s4, s4, 1",
                                                                             "s_cmp_lt_u32 s4, 2"]
        back = [f"s_cbranch_scc1 {65536 - dwords(loop) - 1}"]
        text = translate(head + loop + back + ON + END)
        self.assertEqual(run_wave(text, wave=32)["outputs"], [None] * 32)       # off at the second round's export
        loop = ON + ["s_cmp_eq_u32 s4, 1", "s_cbranch_scc1 1"] + OFF + NULL + ["s_add_u32 s4, s4, 1",
                                                                             "s_cmp_lt_u32 s4, 2"]
        text = translate(head + loop + back + OFF + END)
        self.assertNotIn(None, run_wave(text, wave=32)["outputs"])              # on at the second round's export


class OnlyWhereNeeded(unittest.TestCase):
    def test_no_latch_when_exec_cannot_change_after_the_export(self):
        for body in (EXP, OFF + EXP, OFF + ON + EXP, OFF + EXP_NO_VM, OFF + EXP + ["s_cmp_eq_u64 exec, 0"],
                     OFF + EXP + ["v_mov_b32_e32 v3, 0"]):
            text = translate(SEED + body + END)
            self.assertNotIn("%vm_mask", text, body)
            self.assertNotIn("%col_mask", text, body)
        self.assertIn("%col_mask_mrt0", translate(SEED + EXP_NO_VM + OFF + END))   # EXEC changes after a color export

    def test_code_after_endpgm_does_not_count(self):
        self.assertNotIn("%vm_mask", translate(SEED + OFF + EXP + END + ON))

    def test_every_form_that_writes_exec_is_seen(self):
        for line in ("v_cmpx_eq_u32_e32 0, v1", "v_cmpx_lt_f32_e64 v1, v2", "v_cmpx_class_f32_e32 v1, v2",
                     "v_cmpx_eq_u64 v[2:3], v[4:5]", "s_and_saveexec_b64 s[4:5], vcc",
                     "s_or_saveexec_b32 s4, vcc_lo", "s_andn2_wrexec_b64 s[4:5], s[6:7]", "s_mov_b64 exec, -1",
                     "s_mov_b32 exec_lo, -1", "s_mov_b32 exec_hi, 0", "s_wqm_b64 exec, exec",
                     "s_and_b64 exec, exec, s[4:5]", "v_cmp_eq_u32_e64 exec, 0, v1"):
            self.assertTrue(rdna2spirv.writes_exec(shader_ir.parse_line("\t" + line)), line)
        for line in ("s_cmp_eq_u64 exec, 0", "s_bitcmp1_b32 exec_lo, 3", "s_mov_b64 s[4:5], exec",
                     "exp mrt0 v0, v1, v2, v3 done vm", "v_cndmask_b32_e64 v0, v1, v2, exec"):
            self.assertFalse(rdna2spirv.writes_exec(shader_ir.parse_line("\t" + line)), line)


if __name__ == "__main__":
    unittest.main()
