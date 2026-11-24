"""--fma khr: fused multiply-adds as OpFmaKHR (SPV_KHR_fma) instead of GLSL.std.450 Fma.

RDNA2's v_fma_f32, v_fmac_f32, v_fmaak_f32, v_fmamk_f32 and v_fma_mix* round once. The Vulkan environment gives GLSL
Fma only the precision of "OpFMul followed by OpFAdd", so a driver may round twice; OpFmaKHR is correctly rounded.
The translator keeps GLSL Fma by default (a device without VK_KHR_shader_fma still works) and emits OpFmaKHR, with
capability FMAKHR and extension SPV_KHR_fma, under FMA_KHR (the --fma khr option). The index lists the need as
capability FMAKHR and feature shaderFmaFloat32, which the runtime reads to enable VK_KHR_shader_fma.
"""
import os
import shutil
import struct
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
import spirv_mini_interp as si  # noqa: E402
from support import mrt0_words, translate_lines  # noqa: E402

EXIT = ["exp mrt0 v0, v1, v2, v3 done vm", "s_endpgm"]
F32_FMA_OPS = [
    ["v_mov_b32 v8, 0x3f800800", "v_mul_f32 v9, v8, v8", "v_fma_f32 v0, v8, v8, -v9",
     "v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"],
    ["v_mov_b32 v8, 0x3f800800", "v_mov_b32 v1, 0xbf801000", "v_fmaak_f32 v0, v8, v8, 0xbf801000",
     "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"],
    ["v_mov_b32 v8, 0x3f800800", "v_mov_b32 v0, 0xbf801000", "v_fmac_f32 v0, v8, v8",
     "v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"],
    ["v_mov_b32 v8, 0x3f800800", "v_mov_b32 v9, 0xbf801000", "v_fmamk_f32 v0, v8, 0x3f800800, v9",
     "v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"],
]
NO_FMA = ["v_mov_b32 v0, 0x3f800000", "v_add_f32 v1, v0, v0", "v_mul_f32 v2, v1, v1", "v_mov_b32 v3, 0"]


def translate(lines, khr):
    old = rdna2spirv.FMA_KHR
    rdna2spirv.FMA_KHR = khr
    try:
        return translate_lines("ps_fma_khr", lines + EXIT)
    finally:
        rdna2spirv.FMA_KHR = old


class FmaKhrTests(unittest.TestCase):
    def test_default_is_glsl_fma(self):
        self.assertFalse(rdna2spirv.FMA_KHR)
        for ops in F32_FMA_OPS:
            spv = translate(ops, False)
            self.assertIn("%glsl Fma", spv)
            self.assertNotIn("OpFmaKHR", spv)
            self.assertNotIn("FMAKHR", spv)
            self.assertNotIn("SPV_KHR_fma", spv)

    def test_khr_emits_opfmakhr_with_capability_and_extension(self):
        for ops in F32_FMA_OPS:
            spv = translate(ops, True)
            self.assertIn("OpFmaKHR %float", spv)
            self.assertNotIn("%glsl Fma", spv)
            self.assertIn("OpCapability FMAKHR", spv)
            self.assertIn('OpExtension "SPV_KHR_fma"', spv)
            # module layout: capabilities, then extensions, then imports
            self.assertLess(spv.index("OpCapability FMAKHR"), spv.index('OpExtension "SPV_KHR_fma"'))
            self.assertLess(spv.index('OpExtension "SPV_KHR_fma"'), spv.index("OpExtInstImport"))

    def test_mix_forms_use_it_too(self):
        mix = ["v_mov_b32 v8, 0x3f800000", "v_fma_mix_f32 v0, v8, v8, v8", "v_mov_b32 v1, 0", "v_mov_b32 v2, 0",
               "v_mov_b32 v3, 0"]
        self.assertIn("OpFmaKHR", translate(mix, True))
        self.assertIn("%glsl Fma", translate(mix, False))

    def test_both_forms_give_the_exact_fused_words(self):
        want = [0x33800000, 0x33800000, 0x33800000, 0x33800000]     # 2^-24, see runtime/test/shaders/cs_fma_fused.asm
        for ops, expect in zip(F32_FMA_OPS, want):
            for khr in (False, True):
                with self.subTest(op=ops[2], khr=khr):
                    self.assertEqual(mrt0_words(translate(ops, khr))[0], expect)

    def test_the_split_form_would_differ(self):
        # What the exact words above are worth: a multiply and an add, rounded separately, lose them.
        a, b, c = (struct.unpack("<f", struct.pack("<I", w))[0] for w in (0x3f800800, 0x3f800800, 0xbf801000))
        self.assertEqual(si.fma_f32(a, b, c), 2.0 ** -24)
        self.assertEqual(si.f32(si.f32(a * b) + c), 0.0)

    def test_shaders_without_fma_are_unchanged(self):
        self.assertEqual(translate(NO_FMA, False), translate(NO_FMA, True))
        self.assertNotIn("FMAKHR", translate(NO_FMA, True))

    def test_f16_fma_does_not_depend_on_it(self):
        # v_fma_f16 is round-to-odd arithmetic (Gen.fma_f16_odd), not Fma: the same text in both modes.
        f16 = ["v_mov_b32 v8, 0x6ce1", "v_mov_b32 v9, 0x4200", "v_mov_b32 v10, 0x8004", "v_fma_f16 v0, v8, v9, v10",
               "v_mov_b32 v1, 0", "v_mov_b32 v2, 0", "v_mov_b32 v3, 0"]
        self.assertEqual(translate(f16, False), translate(f16, True))
        self.assertEqual(mrt0_words(translate(f16, True))[0], 0x7351)

    def test_index_metadata_names_the_requirement(self):
        for khr, caps, feats in ((True, ["FMAKHR"], ["shaderFmaFloat32"]), (False, [], [])):
            old = rdna2spirv.FMA_KHR
            rdna2spirv.FMA_KHR = khr
            try:
                spv = translate_lines("ps_fma_khr", F32_FMA_OPS[0] + EXIT)
                meta = rdna2spirv.shader_meta(rdna2spirv.translate.last_gen, spv)
            finally:
                rdna2spirv.FMA_KHR = old
            self.assertEqual([c for c in meta["capabilities"] if c == "FMAKHR"], caps)
            self.assertEqual([f for f in meta["features"] if f == "shaderFmaFloat32"], feats)
            self.assertEqual("SPV_KHR_fma" in meta["extensions"], khr)

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "SPIRV-Tools not installed")
    def test_spirv_tools_accept_it(self):
        require_tools = os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS") == "1"
        spv = translate(F32_FMA_OPS[0], True)
        with tempfile.TemporaryDirectory() as tmp:
            asm, out = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
            with open(asm, "w", encoding="utf-8") as f:
                f.write(spv)
            made = subprocess.run(["spirv-as", asm, "-o", out, "--target-env", "vulkan1.2"], capture_output=True, text=True)
            if (not require_tools and made.returncode
                    and made.stderr.rstrip().endswith("Invalid capability 'FMAKHR'.")):
                self.skipTest("spirv-as lacks SPV_KHR_fma")
            self.assertEqual(made.returncode, 0, made.stderr)
            checked = subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand", out],
                                     capture_output=True, text=True)
            # A newer assembler can be paired with an older validator. 6030 is the FMAKHR capability operand.
            if (not require_tools and checked.returncode
                    and checked.stderr.rstrip().endswith("Invalid capability operand: 6030")):
                self.skipTest("spirv-val lacks SPV_KHR_fma")
            self.assertEqual(checked.returncode, 0, checked.stderr)


if __name__ == "__main__":
    unittest.main()
