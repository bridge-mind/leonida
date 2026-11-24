"""SHADER-597: a module that uses Float64 also declares SignedZeroInfNanPreserve 64.

Without it Vulkan lets the driver assume no -0, NaN or inf in 64-bit float ops (shaderSignedZeroInfNanPreserveFloat64),
so f64 results such as -0 + -0 = -0 would rest on driver defaults. The 32-bit mode (SHADER-99, #664) is declared in
every module; the 64-bit one only where Float64 is, under the same --no-float-controls opt-out, and the index entry
names both properties so vk_pipeline_check skips the shader on a device that reports either one false.
"""

import sys
import tempfile
import unittest
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(TEST_DIR.parent / "tools"), str(TEST_DIR)]

import rdna2spirv
from listing import translate_listing

# -0 + -0 in doubles: v[0:1] = v[6:7] = -0.0 (low word first); the sum goes to v[4:5].
F64_BODY = """v_mov_b32 v0, 0
v_mov_b32 v1, 0x80000000
v_mov_b32 v6, 0
v_mov_b32 v7, 0x80000000
v_add_f64 v[4:5], v[0:1], v[6:7]"""
F32_BODY = """v_mov_b32 v2, 0x80000000
v_add_f32 v3, v2, v2"""

MODE_32 = "OpExecutionMode %main SignedZeroInfNanPreserve 32"
MODE_64 = "OpExecutionMode %main SignedZeroInfNanPreserve 64"


class F64FloatControlsTest(unittest.TestCase):
    def translate(self, body, name):
        with tempfile.TemporaryDirectory() as directory:
            spv = translate_listing(Path(directory), body, name=name)
        return spv, rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)

    def test_f64_module_declares_the_64_bit_mode(self):
        spv, meta = self.translate(F64_BODY, "ps_f64_float_controls")
        self.assertIn("OpCapability Float64", spv)
        self.assertEqual(spv.count("OpCapability SignedZeroInfNanPreserve\n"), 1)
        self.assertIn(MODE_32, spv)
        self.assertIn(MODE_64, spv)
        # Core since SPIR-V 1.4, as for the 32-bit mode: no SPV_KHR_float_controls at vulkan1.2.
        self.assertNotIn("SPV_KHR_float_controls", spv)
        # DenormPreserve 16 (SHADER-526) is declared in every module and its property comes after the two widths.
        self.assertEqual(meta["float_controls"],
                         ["shaderSignedZeroInfNanPreserveFloat32", "shaderSignedZeroInfNanPreserveFloat64",
                          "shaderDenormPreserveFloat16"])
        self.assertIn("shaderFloat64", meta["features"])

    def test_module_without_float64_keeps_only_the_32_bit_mode(self):
        spv, meta = self.translate(F32_BODY, "ps_f32_float_controls")
        self.assertNotIn("OpCapability Float64", spv)
        self.assertIn(MODE_32, spv)
        self.assertNotIn(MODE_64, spv)
        self.assertEqual(meta["float_controls"],
                         ["shaderSignedZeroInfNanPreserveFloat32", "shaderDenormPreserveFloat16"])

    def test_no_float_controls_leaves_out_both_widths(self):
        saved = rdna2spirv.FLOAT_CONTROLS
        rdna2spirv.FLOAT_CONTROLS = False
        try:
            spv, meta = self.translate(F64_BODY, "ps_f64_no_float_controls")
        finally:
            rdna2spirv.FLOAT_CONTROLS = saved
        self.assertIn("OpCapability Float64", spv)
        self.assertNotIn("SignedZeroInfNanPreserve", spv)
        self.assertNotIn("DenormPreserve", spv)
        self.assertNotIn("float_controls", meta)


if __name__ == "__main__":
    unittest.main()
