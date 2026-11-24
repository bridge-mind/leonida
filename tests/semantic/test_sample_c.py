"""image_sample_c and image_sample_c_b (and their _o forms): depth-compare samples with implicit LOD / bias.

Standard library only (spirv-val is used when it is on PATH); no GPU, no LLVM, no game files.

The GPU fixtures are runtime/test/shaders/ps_sample_c.asm and ps_sample_c_b.asm (vk_texture_test dref_c, dref_c_b). Here:
the shape of the SPIR-V in a pixel shader (OpImageSampleDrefImplicitLod, a Bias operand for _c_b, the Offset for _o),
LOD 0 for _c and explicit bias LOD for _c_b outside the pixel stage, d16 and a16 forms, and that the module validates.
"""
import os
import re
import shutil
import struct
import subprocess
import tempfile
import unittest

import support

DESC = "s[8:15], s[16:19]"
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)
PRO = ["v_mov_b32 v4, 1.0", "v_mov_b32 v5, 0.5", "v_mov_b32 v6, 0.25", "v_mov_b32 v7, 0.75", "v_mov_b32 v8, 0"]


def translate(name, line):
    fssm = CS_FSSM if name.startswith("cs_") else None
    return support.translate_lines(name, PRO + [line, "exp mrt0 v9, v9, v9, v9 done vm", "s_endpgm"]
                                   if name.startswith("ps_") else PRO + [line, "s_endpgm"], fssm=fssm)


def samples(spv):
    return [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]


class SampleC(unittest.TestCase):
    def test_c_is_dref_implicit_lod_in_a_pixel_shader(self):
        spv = translate("ps_samplec", f"image_sample_c v9, v[4:6], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
        (line,) = samples(spv)
        self.assertRegex(line, r"= OpImageSampleDrefImplicitLod %float %\w+ %\w+ %\w+$")

    def test_c_b_has_a_bias_operand(self):
        spv = translate("ps_samplecb", f"image_sample_c_b v9, v[4:7], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
        (line,) = samples(spv)
        self.assertRegex(line, r"= OpImageSampleDrefImplicitLod %float %\w+ %\w+ %\w+ Bias %\w+$")

    def test_offset_forms_carry_the_offset(self):
        for op, regs in (("image_sample_c_o", "v[4:7]"), ("image_sample_c_b_o", "v[4:8]")):
            with self.subTest(op=op):
                spv = translate("ps_sampleco", f"{op} v9, {regs}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
                (line,) = samples(spv)
                self.assertIn("OpImageSampleDrefImplicitLod", line)
                self.assertIn("Offset", line)

    def test_d16_result_is_an_f16_in_the_low_half(self):
        spv = translate("ps_sampled16", f"image_sample_c v9, v[4:6], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D d16")
        self.assertIn("OpImageSampleDrefImplicitLod", spv)
        self.assertNotIn("PackHalf2x16", spv)

    def test_c_outside_a_pixel_shader_samples_lod_zero(self):
        for name, fssm in (("vs_samplec", None), ("cs_samplec", CS_FSSM)):
            with self.subTest(shader=name):
                spv = translate(name, f"image_sample_c v9, v[4:6], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
                (line,) = samples(spv)
                self.assertRegex(line, r"= OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod %f0$")

    def test_c_b_outside_a_pixel_shader_samples_bias_as_explicit_lod(self):
        for name in ("vs_samplecb", "cs_samplecb"):
            with self.subTest(shader=name):
                spv = translate(name, f"image_sample_c_b v9, v[4:7], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
                (line,) = samples(spv)
                self.assertRegex(line, r"= OpImageSampleDrefExplicitLod %float %\w+ %\w+ %\w+ Lod %\w+$")

    def test_the_bias_comes_before_the_compare(self):
        # GFX10 address order after the offset: bias, z-compare, coordinates (the compare in v5, the bias in v4).
        spv = translate("ps_sampleorder", f"image_sample_c_b v9, v[4:7], {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
        (line,) = samples(spv)
        ref, bias = re.search(r"%\w+ %\w+ (%\w+) Bias (%\w+)$", line).groups()

        def loaded_from(value):
            """The VGPR index a value was loaded from: the OpLoad of %vgpr[k] behind it, through bitcasts."""
            for _ in range(6):
                m = re.search(rf"^\s*{re.escape(value)} = (\w+) (?:%\w+ )?(%\w+)", spv, re.M)
                if not m:
                    return None
                op, src = m.groups()
                if op == "OpLoad":
                    m2 = re.search(rf"^\s*{re.escape(src)} = OpAccessChain %\S+ %vgpr (%\w+)", spv, re.M)
                    consts = {k: int(v) for k, v in re.findall(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}
                    return consts.get(m2.group(1)) if m2 else None
                value = src
            return None
        self.assertEqual((loaded_from(bias), loaded_from(ref)), (4, 5))

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_modules_validate(self):
        for op, regs in (("image_sample_c", "v[4:6]"), ("image_sample_c_b", "v[4:7]"), ("image_sample_c_o", "v[4:7]"),
                         ("image_sample_c_b_o", "v[4:8]")):
            with self.subTest(op=op):
                spv = translate("ps_sampleval", f"{op} v9, {regs}, {DESC} dmask:0x1 dim:SQ_RSRC_IMG_2D")
                with tempfile.TemporaryDirectory() as tmp:
                    a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
                    with open(a, "w", encoding="utf-8", newline="\n") as f:
                        f.write(spv)
                    subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
                    # a texel offset that comes from a register needs VK_KHR_maintenance8 (design.md, Texel offsets)
                    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand", b], check=True)


if __name__ == "__main__":
    unittest.main()