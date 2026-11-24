"""image_sample_cl, _b_cl, _c_cl, _c_b_cl (and their _o forms): implicit-LOD samples with a min-LOD clamp.

Standard library only (spirv-val is used when it is on PATH); no GPU, no LLVM, no game files.

GPU tests: ps_texture_cl and ps_texture_b_cl (two-mip texture, check.py GRADIENT_TESTS / CLAMP_TESTS). Here the shape
of the SPIR-V: the clamp is the component after the coordinates and becomes MinLod (with Bias, with the compare, after
Offset), outside the pixel stage the LOD is max(0, clamp), a16 packs the clamp with the coordinates, and the module
declares MinLod and validates.
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
D2 = "dim:SQ_RSRC_IMG_2D"
PRO = ["v_mov_b32 v4, 1.0", "v_mov_b32 v5, 0.5", "v_mov_b32 v6, 0.25", "v_mov_b32 v7, 0.75", "v_mov_b32 v8, 0.125",
       "v_mov_b32 v9, 0"]
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate(name, line):
    fssm = CS_FSSM if name.startswith("cs_") else None
    body = PRO + [line] + (["exp mrt0 v9, v9, v9, v9 done vm"] if name.startswith("ps_") else []) + ["s_endpgm"]
    return support.translate_lines(name, body, fssm=fssm)


def samples(spv):
    return [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]


def source_vgpr(spv, value):
    """The VGPR index the id `value` was loaded from (through bitcasts), or None."""
    consts = {k: int(v) for k, v in re.findall(r"^\s*(%\w+) = OpConstant %uint (\d+)", spv, re.M)}
    for _ in range(8):
        m = re.search(rf"^\s*{re.escape(value)} = (\w+) (?:%\w+ )?(%\w+)", spv, re.M)
        if not m:
            return None
        op, src = m.groups()
        if op == "OpLoad":
            m2 = re.search(rf"^\s*{re.escape(src)} = OpAccessChain %\S+ %vgpr (%\w+)", spv, re.M)
            return consts.get(m2.group(1)) if m2 else None
        value = src
    return None


class SampleCl(unittest.TestCase):
    def one(self, name, line):
        (s,) = samples(translate(name, line))
        return s

    def test_cl_is_implicit_lod_with_min_lod(self):
        s = self.one("ps_cl", f"image_sample_cl v[0:3], v[4:6], {DESC} dmask:0xf {D2}")
        self.assertRegex(s, r"= OpImageSampleImplicitLod %v4float %\w+ %\w+ MinLod %\w+$")

    def test_the_clamp_is_the_component_after_the_coordinates(self):
        spv = translate("ps_cl", f"image_sample_cl v[0:3], v[4:6], {DESC} dmask:0xf {D2}")
        (s,) = samples(spv)
        self.assertEqual(source_vgpr(spv, s.split()[-1]), 6)

    def test_b_cl_has_bias_and_min_lod(self):
        spv = translate("ps_bcl", f"image_sample_b_cl v[0:3], v[4:7], {DESC} dmask:0xf {D2}")
        (s,) = samples(spv)
        self.assertRegex(s, r"= OpImageSampleImplicitLod %v4float %\w+ %\w+ Bias\|MinLod %\w+ %\w+$")
        bias, minlod = s.split()[-2:]
        self.assertEqual((source_vgpr(spv, bias), source_vgpr(spv, minlod)), (4, 7))

    def test_dref_forms(self):
        s = self.one("ps_ccl", f"image_sample_c_cl v9, v[4:7], {DESC} dmask:0x1 {D2}")
        self.assertRegex(s, r"= OpImageSampleDrefImplicitLod %float %\w+ %\w+ %\w+ MinLod %\w+$")
        s = self.one("ps_cbcl", f"image_sample_c_b_cl v9, v[4:8], {DESC} dmask:0x1 {D2}")
        self.assertRegex(s, r"= OpImageSampleDrefImplicitLod %float %\w+ %\w+ %\w+ Bias\|MinLod %\w+ %\w+$")

    def test_offset_comes_before_the_min_lod(self):
        s = self.one("ps_clo", f"image_sample_cl_o v[0:3], v[4:7], {DESC} dmask:0xf {D2}")
        self.assertRegex(s, r"= OpImageSampleImplicitLod %v4float %\w+ %\w+ Offset\|MinLod %\w+ %\w+$")

    def test_a16_packs_the_clamp_with_the_coordinates(self):
        # u, v in one VGPR, the clamp in the low half of the next; no whole-VGPR extra for _cl itself
        s = self.one("ps_clo16", f"image_sample_cl v[0:3], v[4:5], {DESC} dmask:0xf {D2} a16")
        self.assertIn("MinLod", s)
        s = self.one("ps_bclo16", f"image_sample_b_cl v[0:3], v[4:6], {DESC} dmask:0xf {D2} a16")
        self.assertIn("Bias|MinLod", s)

    def test_without_derivatives_the_lod_is_the_clamp_or_zero(self):
        for name, fssm in (("vs_cl", None), ("cs_cl", CS_FSSM)):
            with self.subTest(shader=name):
                spv = translate(name, f"image_sample_cl v[0:3], v[4:6], {DESC} dmask:0xf {D2}")
                (s,) = samples(spv)
                self.assertRegex(s, r"= OpImageSampleExplicitLod %v4float %\w+ %\w+ Lod %\w+$")
                self.assertIn("FMax %f0", spv)
                self.assertNotIn("MinLod", s)

    def test_module_declares_min_lod(self):
        self.assertIn("OpCapability MinLod", translate("ps_cl", f"image_sample_cl v[0:3], v[4:6], {DESC} dmask:0xf {D2}"))
        self.assertNotIn("OpCapability MinLod", translate("ps_noc", f"image_sample v[0:3], v[4:5], {DESC} dmask:0xf {D2}"))

    def test_forms_with_no_clamp_variant_stay_refused(self):
        for op, regs in (("image_sample_l_cl", "v[4:7]"), ("image_sample_lz_cl", "v[4:6]")):
            with self.subTest(op=op), self.assertRaises(support.rdna2spirv.Unsupported):
                translate("ps_x", f"{op} v[0:3], {regs}, {DESC} dmask:0xf {D2}")

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"), "needs spirv-as and spirv-val on PATH")
    def test_modules_validate(self):
        cases = [("image_sample_cl", "v[0:3]", "v[4:6]", 0xf), ("image_sample_b_cl", "v[0:3]", "v[4:7]", 0xf),
                 ("image_sample_c_cl", "v9", "v[4:7]", 1), ("image_sample_c_b_cl", "v9", "v[4:8]", 1),
                 ("image_sample_cl_o", "v[0:3]", "v[4:7]", 0xf)]
        for op, dst, regs, dmask in cases:
            for name in ("ps_val", "vs_val"):
                with self.subTest(op=op, shader=name):
                    spv = translate(name, f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2}")
                    with tempfile.TemporaryDirectory() as tmp:
                        a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
                        with open(a, "w", encoding="utf-8", newline="\n") as f:
                            f.write(spv)
                        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
                        subprocess.run(["spirv-val", "--target-env", "vulkan1.2", "--allow-offset-texture-operand", b],
                                       check=True)


if __name__ == "__main__":
    unittest.main()