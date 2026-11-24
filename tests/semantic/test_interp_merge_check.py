"""spirv_mini_interp and spirv_wave_interp reject a module that breaks spirv-val's merge or phi rule.

OpSelectionMerge must directly precede OpBranchConditional / OpSwitch, and OpLoopMerge directly precede OpBranch /
OpBranchConditional (SPIR-V 2.16.2). Both interpreters used to skip merges, so #1633's first version, with an
`OpLoad %active` between the merge and its branch, passed every semantic test while spirv-val rejected every module
it produced (found by @mokkymiah, confirmed by @Impre-visible). Each case takes a translated module, which the
interpreters run, and moves one instruction to sit between a merge and its branch. Standard library only.
"""
import os
import re
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, HERE)

from test_salu64_bitwise_data import translate_ps  # noqa: E402  (also puts tools/ on sys.path)
import spirv_mini_interp as si  # noqa: E402
import spirv_wave_interp as wi  # noqa: E402

LINES = ["v_mbcnt_lo_u32_b32 v4, -1, 0", "v_cmpx_gt_u32_e32 5, v4", "v_mov_b32 v0, 1.0",
         "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"]


def spv():
    text = translate_ps(LINES)
    assert text.count("OpSelectionMerge") >= 1
    return text


def break_merge(text):
    """Put the instruction just before the first OpSelectionMerge between it and its branch."""
    lines = text.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip().startswith("OpSelectionMerge"))
    assert re.match(r"\s*%\S+ = Op", lines[k - 1]), lines[k - 1]
    lines[k - 1], lines[k] = lines[k], lines[k - 1]
    return "\n".join(lines) + "\n"


def drop_branch(text):
    """Remove the branch after the first OpSelectionMerge, so the merge is followed by something else."""
    lines = text.splitlines()
    k = next(i for i, ln in enumerate(lines) if ln.strip().startswith("OpSelectionMerge"))
    return "\n".join(lines[:k + 1] + lines[k + 2:]) + "\n"


class MergeRule(unittest.TestCase):
    def test_translated_module_runs(self):
        text = spv()
        self.assertEqual(si.run_fragment(text)["%out_mrt0"][0], 0x3F800000)
        self.assertEqual(len(wi.run_wave(text, wave=32)["outputs"]), 32)

    def test_instruction_between_merge_and_branch(self):
        bad = break_merge(spv())
        with self.assertRaisesRegex(si.InvalidModule, "OpSelectionMerge"):
            si.run_fragment(bad)
        with self.assertRaisesRegex(wi.InvalidModule, "OpSelectionMerge"):
            wi.run_wave(bad, wave=32)

    def test_merge_without_its_branch(self):
        bad = drop_branch(spv())
        with self.assertRaises(si.InvalidModule):
            si.run_fragment(bad)
        with self.assertRaises(wi.InvalidModule):
            wi.run_wave(bad, wave=32)

    def test_loop_merge_rule(self):
        ok = [(None, "OpLoopMerge", [], "OpLoopMerge %m %c None"), (None, "OpBranch", [], "OpBranch %h")]
        si.check_structure(ok)
        with self.assertRaisesRegex(si.InvalidModule, "OpLoopMerge"):
            si.check_structure([ok[0], (None, "OpIAdd", [], "%x = OpIAdd %uint %a %b"), ok[1]])
        with self.assertRaisesRegex(si.InvalidModule, "end of the function"):
            si.check_structure([ok[0]])


CS_LINES = ["v_mov_b32_e32 v1, 0x40", "global_load_dword v2, v1, s[0:1]", "s_waitcnt vmcnt(0)",
            "v_mov_b32_e32 v3, 0", "buffer_store_dword v2, v3, s[0:3], 0 offen", "s_endpgm"]


def cs_spv():
    from test_salu64_bitwise_data import translate_cs
    text = translate_cs(CS_LINES)
    assert " = OpPhi " in text
    return text


def run_cs(text):
    mem = {0x40: 0x1234}
    si.run_workgroup(text, 1, 64, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
                     push_constants=[0, 0, 0x1000, 0x31016FAC] + [0] * 32, memory=mem)
    return mem


class PhiRule(unittest.TestCase):
    """OpPhi must be the first instructions of its block: only OpLabel or OpPhi before it (SPIR-V 2.16.2)."""

    def test_translated_module_runs(self):
        self.assertEqual(run_cs(cs_spv())[0], 0x1234)

    def test_phi_after_another_instruction(self):
        lines = cs_spv().splitlines()
        k = next(i for i, ln in enumerate(lines) if " = OpPhi " in ln)
        lines[k], lines[k + 1] = lines[k + 1], lines[k]     # the phi now follows a non-phi instruction
        bad = "\n".join(lines) + "\n"
        with self.assertRaisesRegex(si.InvalidModule, "OpPhi"):
            run_cs(bad)

    def test_rule_directly(self):
        lab = (None, "OpLabel", [], "%b = OpLabel")
        phi = ("%p", "OpPhi", [], "%p = OpPhi %uint %a %x %c %y")
        add = ("%q", "OpIAdd", [], "%q = OpIAdd %uint %a %a")
        si.check_structure([lab, phi, phi, add])
        with self.assertRaisesRegex(si.InvalidModule, "OpPhi"):
            si.check_structure([lab, add, phi])
        with self.assertRaisesRegex(wi.InvalidModule, "OpPhi"):
            wi.run_wave("OpFunction %void None %f\n%b = OpLabel\n%q = OpIAdd %uint %a %a\n"
                        "%p = OpPhi %uint %a %x %c %y\nOpReturn\nOpFunctionEnd\n", wave=32)


class FunctionVariableRule(unittest.TestCase):
    """A Function-storage OpVariable belongs in the function's first block (SPIR-V 2.16.1)."""

    def test_rule_directly(self):
        lab = (None, "OpLabel", [], "%b = OpLabel")
        var = ("%v", "OpVariable", ["%_ptr_Function_uint", "Function"], "%v = OpVariable %_ptr_Function_uint Function")
        br = (None, "OpBranch", ["%c"], "OpBranch %c")
        si.check_structure([lab, var, br, lab])
        with self.assertRaisesRegex(si.InvalidModule, "first block"):
            si.check_structure([lab, br, lab, var])
        # a new function starts counting again
        fn = ("%f", "OpFunction", [], "%f = OpFunction %void None %t")
        si.check_structure([fn, lab, br, lab, fn, lab, var])

    def test_translated_module_with_a_variable_moved(self):
        lines = cs_spv().splitlines()
        k = next(i for i, ln in enumerate(lines) if " OpVariable " in ln and " Function" in ln)
        var = lines.pop(k)
        last_label = max(i for i, ln in enumerate(lines) if ln.strip().endswith("= OpLabel"))
        lines.insert(last_label + 1, var)
        bad = "\n".join(lines) + "\n"
        with self.assertRaisesRegex(si.InvalidModule, "first block"):
            run_cs(bad)
        with self.assertRaisesRegex(wi.InvalidModule, "first block"):
            wi.run_wave(bad, wave=32)


if __name__ == "__main__":
    unittest.main()
