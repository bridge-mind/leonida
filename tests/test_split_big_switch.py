"""Unit tests and spirv-val validation for rdna2spirv.split_big_switch (SHADER-77 / #539).

Runs with the standard library only (spirv-as/spirv-val test skips if SPIRV-Tools are not on PATH):
    python3 -I tests/test_split_big_switch.py
    pytest tests/test_split_big_switch.py
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402

HAVE_SPIRV_TOOLS = bool(shutil.which("spirv-as") and shutil.which("spirv-val"))


def make_dispatch_module(n_cases):
    """Build a minimal valid Vulkan 1.2 SPIR-V compute module with an n_cases dispatch OpSwitch."""
    cases = " ".join(f"{i} %b{i}" for i in range(n_cases))
    lines = [
        "               OpCapability Shader",
        "               OpMemoryModel Logical GLSL450",
        '               OpEntryPoint GLCompute %main "main"',
        "               OpExecutionMode %main LocalSize 1 1 1",
        "       %void = OpTypeVoid",
        "         %fn = OpTypeFunction %void",
        "       %uint = OpTypeInt 32 0",
        "  %_ptr_uint = OpTypePointer Function %uint",
        "       %bool = OpTypeBool",
        "         %c0 = OpConstant %uint 0",
        "     %c_done = OpConstant %uint 4294967295",
        "       %main = OpFunction %void None %fn",
        "      %entry = OpLabel",
        "         %pc = OpVariable %_ptr_uint Function",
        "               OpStore %pc %c0",
        "               OpBranch %loop",
        "       %loop = OpLabel",
        "               OpLoopMerge %loop_exit %loop_cont None",
        "               OpBranch %loop_hdr",
        "   %loop_hdr = OpLabel",
        "        %pcv = OpLoad %uint %pc",
        "       %done = OpIEqual %bool %pcv %c_done",
        "               OpSelectionMerge %not_done None",
        "               OpBranchConditional %done %loop_exit %not_done",
        "   %not_done = OpLabel",
        "               OpSelectionMerge %sw_merge None",
        f"               OpSwitch %pcv %sw_default {cases}",
    ]
    for i in range(n_cases):
        lines.append(f"%b{i} = OpLabel")
        lines.append("               OpStore %pc %c_done")
        lines.append("               OpBranch %sw_merge")
    lines += [
        "%sw_default = OpLabel",
        "               OpStore %pc %c_done",
        "               OpBranch %sw_merge",
        "%sw_merge = OpLabel",
        "               OpBranch %loop_cont",
        "%loop_cont = OpLabel",
        "               OpBranch %loop",
        "%loop_exit = OpLabel",
        "               OpReturn",
        "               OpFunctionEnd",
        "",
    ]
    return "\n".join(lines)


class SplitBigSwitchTests(unittest.TestCase):
    def test_at_or_below_limit_is_unchanged(self):
        for n_cases in (10, 16383):
            with self.subTest(n_cases=n_cases):
                src = make_dispatch_module(n_cases)
                out = rdna2spirv.split_big_switch(src)
                self.assertEqual(out, src)
                self.assertEqual(len(re.findall(r"^\s*OpSwitch\b", out, re.M)), 1)
                self.assertNotIn("%c_swchunk", out)
                self.assertNotIn("OpUDiv", out)

    def test_above_case_limit_splits_into_two_chunks(self):
        n_cases = 16384
        src = make_dispatch_module(n_cases)
        out = rdna2spirv.split_big_switch(src)
        self.assertNotEqual(out, src)

        chunk = rdna2spirv.SWITCH_CHUNK
        self.assertEqual(chunk, 16000)
        self.assertIn(f"%c_swchunk = OpConstant %uint {chunk}", out)

        # %pcc = OpUDiv must sit immediately before OpSelectionMerge %sw_merge None.
        self.assertRegex(
            out,
            r"%pcc = OpUDiv %uint %pcv %c_swchunk\n\s*OpSelectionMerge %sw_merge None\n"
            r"\s*OpSwitch %pcc %sw_default 0 %swc0 1 %swc1\n",
        )

        # Two inner chunk switches + one outer switch = 3 OpSwitch instructions.
        switches = re.findall(r"^\s*OpSwitch\b.*$", out, re.M)
        self.assertEqual(len(switches), 3)
        for k in (0, 1):
            self.assertIn(f"%swc{k} = OpLabel", out)
            self.assertIn(f"OpSelectionMerge %swm{k} None", out)
            self.assertRegex(out, rf"(?m)^\s*OpSwitch %pcv %swm{k} ")

        # Boundary blocks route to their chunk's merge; merges sit before %sw_default.
        self.assertRegex(out, r"%b0 = OpLabel\n[^\n]+\n\s*OpBranch %swm0\n")
        self.assertRegex(out, r"%b15999 = OpLabel\n[^\n]+\n\s*OpBranch %swm0\n")
        self.assertRegex(out, r"%b16000 = OpLabel\n[^\n]+\n\s*OpBranch %swm1\n")
        self.assertRegex(out, r"%b16383 = OpLabel\n[^\n]+\n\s*OpBranch %swm1\n")
        self.assertRegex(
            out,
            r"%swm0 = OpLabel\n\s*OpBranch %sw_merge\n"
            r"%swm1 = OpLabel\n\s*OpBranch %sw_merge\n"
            r"%sw_default = OpLabel",
        )

    def test_37220_cases_structure_and_routing(self):
        # 37,220 cases => 3 + 2 * 37,220 = 74,443 words in the original OpSwitch (matching #539).
        n_cases = 37220
        self.assertEqual(3 + 2 * n_cases, 74443)
        src = make_dispatch_module(n_cases)
        out = rdna2spirv.split_big_switch(src)

        self.assertIn("%c_swchunk = OpConstant %uint 16000", out)
        self.assertRegex(
            out,
            r"%pcc = OpUDiv %uint %pcv %c_swchunk\n\s*OpSelectionMerge %sw_merge None\n"
            r"\s*OpSwitch %pcc %sw_default 0 %swc0 1 %swc1 2 %swc2\n",
        )

        # 1 outer + 3 inner switches, every inner switch well below both 16,383 cases and 65,535 words.
        switches = re.findall(r"^\s*OpSwitch\b.*$", out, re.M)
        self.assertEqual(len(switches), 4)
        for k, expected_count in ((0, 16000), (1, 16000), (2, 5220)):
            m = re.search(rf"^\s*OpSwitch %pcv %swm{k} (.+)$", out, re.M)
            self.assertIsNotNone(m, f"missing inner switch for chunk {k}")
            n_pairs = len(m.group(1).split()) // 2
            self.assertEqual(n_pairs, expected_count)
            self.assertLessEqual(n_pairs, 16383)
            self.assertLessEqual(3 + 2 * n_pairs, 65535)

        # Representative blocks across all 3 chunks route to %swm0 / %swm1 / %swm2.
        self.assertRegex(out, r"%b0 = OpLabel\n[^\n]+\n\s*OpBranch %swm0\n")
        self.assertRegex(out, r"%b15999 = OpLabel\n[^\n]+\n\s*OpBranch %swm0\n")
        self.assertRegex(out, r"%b16000 = OpLabel\n[^\n]+\n\s*OpBranch %swm1\n")
        self.assertRegex(out, r"%b31999 = OpLabel\n[^\n]+\n\s*OpBranch %swm1\n")
        self.assertRegex(out, r"%b32000 = OpLabel\n[^\n]+\n\s*OpBranch %swm2\n")
        self.assertRegex(out, r"%b37219 = OpLabel\n[^\n]+\n\s*OpBranch %swm2\n")
        self.assertRegex(
            out,
            r"%swm0 = OpLabel\n\s*OpBranch %sw_merge\n"
            r"%swm1 = OpLabel\n\s*OpBranch %sw_merge\n"
            r"%swm2 = OpLabel\n\s*OpBranch %sw_merge\n"
            r"%sw_default = OpLabel",
        )

    @unittest.skipUnless(HAVE_SPIRV_TOOLS, "needs spirv-as and spirv-val on PATH")
    def test_37220_cases_spirv_as_and_spirv_val(self):
        # Negative control: raw 37,220-case OpSwitch (74,443 words > 65,535 limit) fails spirv-as/spirv-val.
        # Positive control: split_big_switch output assembles and passes spirv-val --target-env vulkan1.2.
        raw = make_dispatch_module(37220)
        split = rdna2spirv.split_big_switch(raw)

        with tempfile.TemporaryDirectory() as td:
            raw_spvasm = os.path.join(td, "raw.spvasm")
            raw_spv = os.path.join(td, "raw.spv")
            with open(raw_spvasm, "w", encoding="utf-8") as f:
                f.write(raw)
            r_as_raw = subprocess.run(
                ["spirv-as", "--target-env", "vulkan1.2", raw_spvasm, "-o", raw_spv],
                capture_output=True, text=True,
            )
            if r_as_raw.returncode == 0:
                r_val_raw = subprocess.run(
                    ["spirv-val", "--target-env", "vulkan1.2", raw_spv],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(
                    r_val_raw.returncode, 0,
                    "expected unsplit 74,443-word OpSwitch to fail spirv-val",
                )

            split_spvasm = os.path.join(td, "split.spvasm")
            split_spv = os.path.join(td, "split.spv")
            with open(split_spvasm, "w", encoding="utf-8") as f:
                f.write(split)
            r_as = subprocess.run(
                ["spirv-as", "--target-env", "vulkan1.2", split_spvasm, "-o", split_spv],
                capture_output=True, text=True,
            )
            self.assertEqual(r_as.returncode, 0, f"spirv-as failed:\n{r_as.stderr}")
            r_val = subprocess.run(
                ["spirv-val", "--target-env", "vulkan1.2", split_spv],
                capture_output=True, text=True,
            )
            self.assertEqual(r_val.returncode, 0, f"spirv-val failed:\n{r_val.stderr}")


if __name__ == "__main__":
    unittest.main()
