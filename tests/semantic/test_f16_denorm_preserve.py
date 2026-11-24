"""SHADER-526: every module declares DenormPreserve 16, so a driver keeps an f16 denormal instead of flushing it.

RDNA2 keeps f16 denormals. `v_mul_f16_sdwa v3, 2.0, v1 src0_sel:BYTE_1` reads the denormal 0x0040 (2^-18) and
multiplies it by 1024.0 to 0x1c00 (runtime/test/shaders/ps_sdwa_f16_const.asm, measured on gfx1036 with
tools/hw_oracle). The translator models f16 arithmetic in f32 with integer narrowing, so in the module that value is
an ordinary f32 and no f32 denormal is involved: the 16-bit mode is what governs the f16-to-f32 conversion the
driver emits for it. Without the mode NVIDIA flushed that conversion and the same test read lane y as 0x0000 on an
RTX 3090 (driver 610.88) and an RTX 4070 SUPER (610.88), 4096/4096 pixels; with it, 4096/4096 pixels exact.

SPV_KHR_float_controls is core since SPIR-V 1.4, so the mode needs no extension at `--target-env vulkan1.2` (the
header the assembler writes is 1.5). The index lists the device property it needs, shaderDenormPreserveFloat16, next
to shaderSignedZeroInfNanPreserveFloat32: `vk_pipeline_check` skips a shader where the device reports false, and
`--no-float-controls` leaves both modes out.
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
import support  # noqa: E402

EXIT = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
# A pixel shader with nothing but an ordinary value in it: the mode is a property of the module, not of the op.
PLAIN = ["v_mov_b32 v0, 0", "v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]

MODE = "OpExecutionMode %main DenormPreserve 16"
CAP = "OpCapability DenormPreserve"
# ps_sdwa_f16_const's four words (gfx1036): lane y is the denormal 0x0040 * 1024.0.
SDWA_WORDS = (0xAAAA3800, 0xAAAA1C00, 0xAAAA3E00, 0xAAAA73F0)


def translate(lines, float_controls=True):
    old = rdna2spirv.FLOAT_CONTROLS
    rdna2spirv.FLOAT_CONTROLS = float_controls
    try:
        return support.translate_lines("ps_f16_denorm", lines + EXIT)
    finally:
        rdna2spirv.FLOAT_CONTROLS = old


def meta(spv):
    return rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)


def without_modes(spv):
    """The module text with the float-controls capability and execution modes removed."""
    return "\n".join(ln for ln in spv.splitlines()
                     if "SignedZeroInfNanPreserve" not in ln and "DenormPreserve" not in ln)


class F16DenormPreserveTests(unittest.TestCase):
    def test_module_declares_denorm_preserve_16(self):
        spv = translate(PLAIN)
        self.assertIn(CAP, spv)
        self.assertIn(MODE, spv)
        # module layout: capabilities, then extensions, then imports
        self.assertLess(spv.index(CAP), spv.index("OpExtInstImport"))
        self.assertLess(spv.index("OpCapability SignedZeroInfNanPreserve"), spv.index(CAP))

    def test_index_metadata_names_the_property(self):
        self.assertEqual(meta(translate(PLAIN))["float_controls"],
                         ["shaderSignedZeroInfNanPreserveFloat32", "shaderDenormPreserveFloat16"])

    def test_no_float_controls_leaves_it_out(self):
        spv = translate(PLAIN, float_controls=False)
        self.assertNotIn("DenormPreserve", spv)
        self.assertNotIn("SignedZeroInfNanPreserve", spv)
        self.assertNotIn("float_controls", meta(spv))

    def test_the_mode_changes_only_the_header(self):
        # Nothing of the translated code depends on it: the driver is the one that has to honour it.
        self.assertEqual(without_modes(translate(PLAIN)), without_modes(translate(PLAIN, float_controls=False)))

    def test_the_f16_denormal_words_and_the_mode_together(self):
        spv = support.translate_fixture("ps_sdwa_f16_const")
        self.assertIn(MODE, spv)
        self.assertEqual(tuple(support.mrt0_words(spv)), SDWA_WORDS)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "SPIRV-Tools not installed")
    def test_spirv_tools_accept_it_at_vulkan1_2(self):
        with tempfile.TemporaryDirectory() as tmp:
            asm, out = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
            with open(asm, "w", encoding="utf-8", newline="\n") as f:
                f.write(support.translate_fixture("ps_sdwa_f16_const"))
            made = subprocess.run(["spirv-as", asm, "-o", out, "--target-env", "vulkan1.2"],
                                  capture_output=True, text=True)
            self.assertEqual(made.returncode, 0, made.stderr)
            checked = subprocess.run(["spirv-val", "--target-env", "vulkan1.2",
                                      "--allow-offset-texture-operand", out], capture_output=True, text=True)
            self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
