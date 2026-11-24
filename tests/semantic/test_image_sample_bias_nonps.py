"""Biased image_sample variants in non-PS stages (cs, vs, gs, ms).

Covers all 8 RDNA2 biased sample opcodes:
  - image_sample_b, image_sample_b_o, image_sample_b_cl, image_sample_b_cl_o
  - image_sample_c_b, image_sample_c_b_o, image_sample_c_b_cl, image_sample_c_b_cl_o

Outside a pixel shader there are no screen-space derivatives (implicit base LOD is 0), so
biased samples become explicit-LOD samples with `Lod = bias` (or `FMax(bias, minlod)` for
`_cl`). Pixel-shader behavior stays unchanged (`ImplicitLod` with `Bias` / `Bias|MinLod`).

Standard library only (spirv-as/spirv-val and llvm-mc are used when available); no GPU, no
game files.
"""
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

import support

TOOLS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools"))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import mc_command  # noqa: E402
import rdna2_decode  # noqa: E402

DESC = "s[8:15], s[16:19]"
D2 = "dim:SQ_RSRC_IMG_2D"
PRO = [
    "v_mov_b32 v4, 1.0",
    "v_mov_b32 v5, 0.5",
    "v_mov_b32 v6, 0.25",
    "v_mov_b32 v7, 0.75",
    "v_mov_b32 v8, 0.125",
    "v_mov_b32 v9, 0.0625",
    "v_mov_b32 v10, 0",
    "v_mov_b32 v11, 0",
]
CS_FSSM = bytearray(0x148)
struct.pack_into("<4I", CS_FSSM, 0x138, 0x8088, 32, 1, 1)
CS_FSSM = bytes(CS_FSSM)


def translate(name, line, *, texel_offset_fallback=False):
    fssm = CS_FSSM if name.startswith("cs_") else None
    body = PRO + [line] + (["exp mrt0 v0, v0, v0, v0 done vm"] if name.startswith("ps_") else []) + ["s_endpgm"]
    prev = support.rdna2spirv.TEXEL_OFFSET_FALLBACK
    support.rdna2spirv.TEXEL_OFFSET_FALLBACK = texel_offset_fallback
    try:
        return support.translate_lines(name, body, fssm=fssm)
    finally:
        support.rdna2spirv.TEXEL_OFFSET_FALLBACK = prev


def samples(spv):
    return [ln.strip() for ln in spv.splitlines() if "= OpImageSample" in ln]


def source_vgpr(spv, value):
    """The VGPR index `value` was loaded from (through bitcasts), or None."""
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


def fmax_args(spv, value):
    m = re.search(rf"^\s*{re.escape(value)} = OpExtInst %float %glsl FMax (%\w+) (%\w+)", spv, re.M)
    return m.groups() if m else None


def spirv_validate(spv, *, allow_offset=False):
    with tempfile.TemporaryDirectory() as tmp:
        a, b = os.path.join(tmp, "m.spvasm"), os.path.join(tmp, "m.spv")
        with open(a, "w", encoding="utf-8", newline="\n") as f:
            f.write(spv)
        subprocess.run(["spirv-as", "--target-env", "vulkan1.2", a, "-o", b], check=True)
        cmd = ["spirv-val", "--target-env", "vulkan1.2"]
        if allow_offset:
            cmd.append("--allow-offset-texture-operand")
        cmd.append(b)
        subprocess.run(cmd, check=True)


def _llvm_mc():
    cmd = mc_command.mc_command()
    try:
        r = subprocess.run(cmd + ["-triple=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
                           input="s_endpgm\n", capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return cmd if r.returncode == 0 and "encoding" in r.stdout else None


MC = _llvm_mc()


def assemble_gfx1030(lines):
    r = subprocess.run(MC + ["-triple=amdgcn", "-mcpu=gfx1030", "-show-encoding"],
                       input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(f"llvm-mc failed:\n{r.stderr}")
    blob = bytearray()
    for ln in r.stdout.splitlines():
        if "encoding: [" in ln:
            body = ln.split("encoding: [", 1)[1].split("]", 1)[0]
            blob.extend(int(b, 16) for b in body.split(","))
    return bytes(blob)


ALL_8_2D = [
    ("image_sample_b", "v[0:3]", "v[4:6]", 0xf, False, False, False),
    ("image_sample_b_o", "v[0:3]", "v[4:7]", 0xf, False, True, False),
    ("image_sample_b_cl", "v[0:3]", "v[4:7]", 0xf, False, False, True),
    ("image_sample_b_cl_o", "v[0:3]", "v[4:8]", 0xf, False, True, True),
    ("image_sample_c_b", "v0", "v[4:7]", 0x1, True, False, False),
    ("image_sample_c_b_o", "v0", "v[4:8]", 0x1, True, True, False),
    ("image_sample_c_b_cl", "v0", "v[4:8]", 0x1, True, False, True),
    ("image_sample_c_b_cl_o", "v0", "v[4:9]", 0x1, True, True, True),
]


class NonPsBiasedSampleTests(unittest.TestCase):
    def test_all_8_variants_in_cs_and_vs_emit_explicit_lod(self):
        for stage in ("cs", "vs"):
            for op, dst, regs, dmask, is_c, has_o, has_cl in ALL_8_2D:
                with self.subTest(stage=stage, op=op):
                    spv = translate(f"{stage}_{op}", f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2}")
                    (s,) = samples(spv)
                    expected_op = "OpImageSampleDrefExplicitLod %float" if is_c else "OpImageSampleExplicitLod %v4float"
                    expected_mask = "Lod|Offset" if has_o else "Lod"
                    self.assertIn(expected_op, s)
                    self.assertIn(f" {expected_mask} ", s)
                    self.assertNotIn("Bias", s)
                    self.assertNotIn("MinLod", s)

                    # Extract the Lod operand id (immediately after Lod or Lod|Offset)
                    m = re.search(rf" {re.escape(expected_mask)} (%\w+)", s)
                    self.assertIsNotNone(m, s)
                    lod_id = m.group(1)

                    bias_vgpr = 5 if has_o else 4
                    if has_cl:
                        args = fmax_args(spv, lod_id)
                        self.assertIsNotNone(args, f"expected FMax for {op}: {s}")
                        minlod_vgpr = bias_vgpr + (1 if is_c else 0) + 1 + 2
                        self.assertEqual((source_vgpr(spv, args[0]), source_vgpr(spv, args[1])),
                                         (bias_vgpr, minlod_vgpr))
                    else:
                        self.assertEqual(source_vgpr(spv, lod_id), bias_vgpr)

                    if is_c:
                        m_ref = re.search(r"OpImageSampleDrefExplicitLod %float %\w+ %\w+ (%\w+) ", s)
                        self.assertIsNotNone(m_ref, s)
                        self.assertEqual(source_vgpr(spv, m_ref.group(1)), bias_vgpr + 1)

    def test_ps_behavior_is_unchanged(self):
        for op, dst, regs, dmask, is_c, has_o, has_cl in ALL_8_2D:
            with self.subTest(op=op):
                spv = translate(f"ps_{op}", f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2}")
                (s,) = samples(spv)
                expected_op = "OpImageSampleDrefImplicitLod %float" if is_c else "OpImageSampleImplicitLod %v4float"
                masks = ["Bias"]
                if has_o:
                    masks.append("Offset")
                if has_cl:
                    masks.append("MinLod")
                self.assertIn(expected_op, s)
                self.assertIn(f" {'|'.join(masks)} ", s)

    def test_all_supported_dims(self):
        # (dim_mod, ncoord, supports_dref)
        dims = [
            ("dim:SQ_RSRC_IMG_1D", 1, True),
            ("dim:SQ_RSRC_IMG_2D", 2, True),
            ("dim:SQ_RSRC_IMG_2D_ARRAY", 3, True),
            ("dim:SQ_RSRC_IMG_CUBE", 3, True),
            ("dim:SQ_RSRC_IMG_3D", 3, False),
        ]
        for stage in ("cs", "vs"):
            for dim_mod, ncoord, supports_dref in dims:
                for is_c in (False, True):
                    if is_c and not supports_dref:
                        continue
                    for has_o in (False, True):
                        if has_o and "CUBE" in dim_mod:
                            continue
                        for has_cl in (False, True):
                            op = "image_sample" + ("_c" if is_c else "") + "_b" + ("_cl" if has_cl else "") + ("_o" if has_o else "")
                            naddr = (1 if has_o else 0) + 1 + (1 if is_c else 0) + ncoord + (1 if has_cl else 0)
                            regs = f"v[4:{4 + naddr - 1}]"
                            dst = "v0" if is_c else "v[0:3]"
                            dmask = 0x1 if is_c else 0xf
                            with self.subTest(stage=stage, op=op, dim=dim_mod):
                                spv = translate(f"{stage}_{op}", f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {dim_mod}")
                                (s,) = samples(spv)
                                self.assertIn("OpImageSampleDrefExplicitLod" if is_c else "OpImageSampleExplicitLod", s)
                                self.assertIn("Lod|Offset" if has_o else "Lod", s)

    def test_a16_and_d16_modifiers(self):
        # 2D address VGPR counts under a16: offset(1), bias(1, low f16), ref(1, f32), uv(1), clamp(1, low f16)
        a16_cases = [
            ("image_sample_b", "v[0:1]", "v[4:5]", 0xf, False, False, False),
            ("image_sample_b_o", "v[0:1]", "v[4:6]", 0xf, False, True, False),
            ("image_sample_b_cl", "v[0:1]", "v[4:6]", 0xf, False, False, True),
            ("image_sample_b_cl_o", "v[0:1]", "v[4:7]", 0xf, False, True, True),
            ("image_sample_c_b", "v0", "v[4:6]", 0x1, True, False, False),
            ("image_sample_c_b_o", "v0", "v[4:7]", 0x1, True, True, False),
            ("image_sample_c_b_cl", "v0", "v[4:7]", 0x1, True, False, True),
            ("image_sample_c_b_cl_o", "v0", "v[4:8]", 0x1, True, True, True),
        ]
        for stage in ("cs", "vs"):
            for op, dst, regs, dmask, is_c, has_o, has_cl in a16_cases:
                with self.subTest(stage=stage, op=op):
                    spv = translate(f"{stage}_{op}_a16d16",
                                    f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2} a16 d16")
                    (s,) = samples(spv)
                    self.assertIn("OpImageSampleDrefExplicitLod" if is_c else "OpImageSampleExplicitLod", s)
                    self.assertIn("UnpackHalf2x16", spv)
                    # d16 uses integer f16_bits / pack_f16_pair_int (never PackHalf2x16, #489)
                    self.assertNotIn("PackHalf2x16", spv)
                    if not is_c:
                        self.assertIn("OpShiftLeftLogical %uint", spv)

    def test_texel_offset_fallback_folds_at_bias_lod(self):
        cases = [c for c in ALL_8_2D if c[5]]  # all 4 _o variants
        for stage in ("cs", "vs"):
            for op, dst, regs, dmask, is_c, _has_o, has_cl in cases:
                with self.subTest(stage=stage, op=op):
                    spv = translate(f"{stage}_{op}_fb",
                                    f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2}",
                                    texel_offset_fallback=True)
                    (s,) = samples(spv)
                    self.assertIn("OpImageSampleDrefExplicitLod" if is_c else "OpImageSampleExplicitLod", s)
                    self.assertNotIn("Offset", s)
                    self.assertNotIn("OpImageQueryLod", spv)
                    self.assertNotIn("OpDPdx", spv)
                    self.assertNotIn("Exp2", spv)
                    self.assertIn("OpImageQuerySizeLod", spv)
                    # Floor() in texel_offset_fold must consume bias_lod (not constant 0)
                    m_lod = re.search(r" Lod (%\w+)$", s)
                    self.assertIsNotNone(m_lod, s)
                    lod_id = m_lod.group(1)
                    self.assertIn(f"Floor {lod_id}", spv)
                    if has_cl:
                        self.assertIsNotNone(fmax_args(spv, lod_id))

    @unittest.skipUnless(shutil.which("spirv-as") and shutil.which("spirv-val"),
                         "needs spirv-as and spirv-val on PATH")
    def test_spirv_val_all_variants_stages_dims_and_fallback(self):
        for stage in ("cs", "vs"):
            for op, dst, regs, dmask, is_c, has_o, has_cl in ALL_8_2D:
                with self.subTest(stage=stage, op=op, mode="direct"):
                    spv = translate(f"{stage}_val_{op}", f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2}")
                    spirv_validate(spv, allow_offset=has_o)
                if has_o:
                    with self.subTest(stage=stage, op=op, mode="fallback"):
                        spv = translate(f"{stage}_valfb_{op}", f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2}",
                                        texel_offset_fallback=True)
                        spirv_validate(spv, allow_offset=False)

            # Spot-check other dims + a16 + d16 in spirv-val
            extra = [
                ("image_sample_b", "v[0:3]", "v[4:5]", 0xf, "dim:SQ_RSRC_IMG_1D", False),
                ("image_sample_b_cl", "v[0:3]", "v[4:8]", 0xf, "dim:SQ_RSRC_IMG_2D_ARRAY", False),
                ("image_sample_b", "v[0:3]", "v[4:7]", 0xf, "dim:SQ_RSRC_IMG_CUBE", False),
                ("image_sample_b_cl_o", "v[0:3]", "v[4:9]", 0xf, "dim:SQ_RSRC_IMG_3D", True),
                ("image_sample_c_b", "v0", "v[4:6]", 0x1, "dim:SQ_RSRC_IMG_1D", False),
                ("image_sample_c_b_cl", "v0", "v[4:9]", 0x1, "dim:SQ_RSRC_IMG_2D_ARRAY", False),
                ("image_sample_c_b", "v0", "v[4:8]", 0x1, "dim:SQ_RSRC_IMG_CUBE", False),
                ("image_sample_b_cl_o", "v[0:1]", "v[4:7]", 0xf, "dim:SQ_RSRC_IMG_2D a16 d16", True),
                ("image_sample_c_b_cl_o", "v0", "v[4:8]", 0x1, "dim:SQ_RSRC_IMG_2D a16 d16", True),
            ]
            for op, dst, regs, dmask, mods, has_o in extra:
                with self.subTest(stage=stage, op=op, mods=mods):
                    spv = translate(f"{stage}_valx_{op}", f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {mods}")
                    spirv_validate(spv, allow_offset=has_o)

    @unittest.skipUnless(MC, "needs llvm-mc with AMDGPU gfx1030 target (set LLVM_MC)")
    def test_llvm_mc_gfx1030_binary_decode_roundtrip(self):
        for stage in ("cs", "vs"):
            for op, dst, regs, dmask, is_c, has_o, has_cl in ALL_8_2D:
                with self.subTest(stage=stage, op=op):
                    asm_lines = PRO + [
                        f"{op} {dst}, {regs}, {DESC} dmask:{dmask:#x} {D2}",
                        "s_endpgm",
                    ]
                    blob = assemble_gfx1030(asm_lines)
                    decoded, unknown = rdna2_decode.disasm_bytes(blob)
                    self.assertEqual(unknown, 0, decoded)
                    self.assertIn(op, decoded)
                    decoded_lines = [ln for ln in decoded.splitlines() if ln.strip() and not ln.strip().startswith(";")]
                    fssm = CS_FSSM if stage == "cs" else None
                    spv = support.translate_lines(f"{stage}_rt_{op}", decoded_lines, fssm=fssm)
                    (s,) = samples(spv)
                    self.assertIn("OpImageSampleDrefExplicitLod" if is_c else "OpImageSampleExplicitLod", s)
                    self.assertIn("Lod|Offset" if has_o else "Lod", s)
                    if has_cl:
                        self.assertIn("FMax", spv)
                    if shutil.which("spirv-as") and shutil.which("spirv-val"):
                        spirv_validate(spv, allow_offset=has_o)


if __name__ == "__main__":
    unittest.main()
