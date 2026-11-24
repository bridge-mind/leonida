"""test_smem_clock.py: s_memtime / s_memrealtime as OpReadClockKHR (VK_KHR_shader_clock)."""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import translate_lines  # noqa: E402

EXIT = ["s_waitcnt lgkmcnt(0)", "v_mov_b32 v0, s4", "v_mov_b32 v1, s5", "v_mov_b32 v2, s6", "v_mov_b32 v3, s7",
        "exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]


def translate(ops, clock=True):
    old = rdna2spirv.SHADER_CLOCK
    rdna2spirv.SHADER_CLOCK = clock
    try:
        spv = translate_lines("ps_smem_clock", ops + EXIT)
    finally:
        rdna2spirv.SHADER_CLOCK = old
    return spv, rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)


def spirv_val(spv):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        return None
    with tempfile.TemporaryDirectory() as d:
        src, out = os.path.join(d, "s.spvasm"), os.path.join(d, "s.spv")
        with open(src, "w", encoding="utf-8") as f:
            f.write(spv)
        for cmd in (["spirv-as", "--target-env", "vulkan1.2", src, "-o", out],
                    ["spirv-val", "--target-env", "vulkan1.2", out]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                return r.stderr or r.stdout
    return ""


class SmemClockTests(unittest.TestCase):
    def test_memtime_is_subgroup_clock(self):
        spv, meta = translate(["s_memtime s[4:5]", "s_mov_b32 s6, 0", "s_mov_b32 s7, 0"])
        self.assertIn("OpReadClockKHR %v2uint_d %c3", spv)
        self.assertIn("OpCapability ShaderClockKHR", spv)
        self.assertIn('OpExtension "SPV_KHR_shader_clock"', spv)
        self.assertIn("shaderSubgroupClock", meta["features"])
        self.assertNotIn("shaderDeviceClock", meta["features"])

    def test_memrealtime_is_device_clock(self):
        spv, meta = translate(["s_memrealtime s[6:7]", "s_mov_b32 s4, 0", "s_mov_b32 s5, 0"])
        self.assertIn("OpReadClockKHR %v2uint_d %c1", spv)
        self.assertIn("shaderDeviceClock", meta["features"])
        self.assertNotIn("shaderSubgroupClock", meta["features"])

    def test_both_with_f64_share_one_vector_type(self):
        spv, meta = translate(["s_memtime s[4:5]", "s_memrealtime s[6:7]", "v_add_f64 v[8:9], v[8:9], v[8:9]"])
        self.assertEqual(spv.count("OpTypeVector %uint 2"), 1)
        self.assertEqual(spv.count("OpCapability ShaderClockKHR"), 1)
        self.assertEqual(sorted(f for f in meta["features"] if "Clock" in f),
                         ["shaderDeviceClock", "shaderSubgroupClock"])

    def test_no_clock_no_capability(self):
        spv, meta = translate(["s_mov_b32 s4, 0", "s_mov_b32 s5, 0", "s_mov_b32 s6, 0", "s_mov_b32 s7, 0"])
        self.assertNotIn("ShaderClockKHR", spv)
        self.assertNotIn("SPV_KHR_shader_clock", spv)
        self.assertFalse([f for f in meta["features"] if "Clock" in f])

    def test_single_sgpr_destination_is_unsupported(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate(["s_memtime s4"])

    def test_off_by_default(self):
        self.assertFalse(rdna2spirv.SHADER_CLOCK)
        for op in ("s_memtime s[4:5]", "s_memrealtime s[6:7]"):
            with self.subTest(op=op), self.assertRaises(rdna2spirv.Unsupported):
                translate([op], clock=False)

    def test_spirv_val(self):
        spv, _ = translate(["s_memtime s[4:5]", "s_memrealtime s[6:7]"])
        err = spirv_val(spv)
        if err is None:
            self.skipTest("spirv-as / spirv-val not on PATH")
        self.assertEqual(err, "")


if __name__ == "__main__":
    unittest.main()
