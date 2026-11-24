"""--fma khr on binary64: the f64 fused multiply-adds (v_div_fmas_f64) as OpFmaKHR on doubles.

GLSL.std.450 Fma may round twice on doubles as on floats, and NVIDIA's driver does (RTX 4090, 621.24, Windows:
runtime/test/shaders/cs_div_fmas_f64.asm gives the split 0 instead of 2^-60). Under FMA_KHR (--fma khr) the
translator emits OpFmaKHR %double, which SPV_KHR_fma defines as correctly rounded; the index then lists feature
shaderFmaFloat64 (VK_KHR_shader_fma) and, for a module without an f32 FMA, not shaderFmaFloat32. The default output
is unchanged.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS_DIR)
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from support import translate_lines  # noqa: E402

EXIT = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
# a = 1 + 2^-30, c = -(1 + 2^-29): fma(a, a, c) = 2^-60 with one rounding, 0 with two (cs_div_fmas_f64.asm).
DIV_FMAS_F64 = ["s_mov_b64 vcc, 0", "v_mov_b32 v4, 0x00400000", "v_mov_b32 v5, 0x3ff00000",
                "v_mov_b32 v6, 0x00800000", "v_mov_b32 v7, 0xbff00000",
                "v_div_fmas_f64 v[8:9], v[4:5], v[4:5], v[6:7]",
                "v_mov_b32 v0, v8", "v_mov_b32 v1, v9", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
F32_FMA = ["v_mov_b32 v8, 0x3f800800", "v_mul_f32 v9, v8, v8", "v_fma_f32 v0, v8, v8, -v9",
           "v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]


def translate(lines, khr):
    old = rdna2spirv.FMA_KHR
    rdna2spirv.FMA_KHR = khr
    try:
        spv = translate_lines("ps_fma_khr_f64", lines + EXIT)
        return spv, rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
    finally:
        rdna2spirv.FMA_KHR = old


class FmaKhrF64Tests(unittest.TestCase):
    def test_default_keeps_glsl_fma_on_doubles(self):
        spv, meta = translate(DIV_FMAS_F64, False)
        self.assertEqual(spv.count("OpExtInst %double %glsl Fma"), 3)
        self.assertNotIn("OpFmaKHR", spv)
        self.assertNotIn("FMAKHR", meta["capabilities"])
        self.assertFalse({"shaderFmaFloat32", "shaderFmaFloat64"} & set(meta["features"]))

    def test_khr_emits_opfmakhr_on_doubles(self):
        spv, meta = translate(DIV_FMAS_F64, True)
        self.assertNotIn("%double %glsl Fma", spv)
        self.assertEqual(spv.count("OpFmaKHR %double"), 3)   # plain, pre-scaled, and the exact product error
        self.assertIn("FMAKHR", meta["capabilities"])
        self.assertIn("SPV_KHR_fma", meta["extensions"])
        self.assertIn("shaderFmaFloat64", meta["features"])
        self.assertNotIn("shaderFmaFloat32", meta["features"])   # no f32 OpFmaKHR in this module

    def test_f32_only_module_still_needs_float32_only(self):
        _, meta = translate(F32_FMA, True)
        self.assertIn("shaderFmaFloat32", meta["features"])
        self.assertNotIn("shaderFmaFloat64", meta["features"])

    def test_both_widths_list_both_features(self):
        lines = DIV_FMAS_F64[:6] + ["v_mov_b32 v10, 0x3f800800", "v_fma_f32 v2, v10, v10, v10"] + DIV_FMAS_F64[6:8] + ["v_mov_b32 v3, 0"]
        _, meta = translate(lines, True)
        self.assertLessEqual({"shaderFmaFloat32", "shaderFmaFloat64"}, set(meta["features"]))

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "SPIRV-Tools not installed")
    def test_spirv_tools_accept_it(self):
        require_tools = os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS") == "1"
        spv, _ = translate(DIV_FMAS_F64, True)
        with tempfile.TemporaryDirectory() as tmp:
            asm, out = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
            with open(asm, "w", encoding="utf-8", newline="\n") as f:
                f.write(spv)
            made = subprocess.run(["spirv-as", asm, "-o", out, "--target-env", "vulkan1.2"], capture_output=True, text=True)
            if (not require_tools and made.returncode
                    and made.stderr.rstrip().endswith("Invalid capability 'FMAKHR'.")):
                self.skipTest("spirv-as lacks SPV_KHR_fma")
            self.assertEqual(made.returncode, 0, made.stderr)
            checked = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", out], capture_output=True, text=True)
            if (not require_tools and checked.returncode
                    and checked.stderr.rstrip().endswith("Invalid capability operand: 6030")):
                self.skipTest("spirv-val lacks SPV_KHR_fma")
            self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
