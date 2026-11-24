"""Translation and SPIR-V structure tests for image_get_resinfo and image_get_lod in tools/rdna2spirv.py.

spirv_mini_interp.py does not yet implement OpImageQuerySizeLod, OpImageQueryLevels, or OpImageQueryLod,
so this suite characterizes current translator output (capabilities, image/sampler types, query opcodes,
dmask VGPR compaction, and negative/mutation controls) rather than pinning hardware-measured rules
(such as 2D resinfo depth=0 in docs/shader_validation.md), using the Python standard library only:
    python -I tests/test_image_query_translate.py -v
"""
import gc
import os
import re
import struct
import sys
import tempfile
import unittest
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
import rdna2spirv  # noqa: E402

# Minimal valid DXBC container with zero parts (no declared PS inputs).
STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def translate_lines(name, lines, dxbc=STUB_DXBC):
    """Translate a hand-written RDNA2 listing via rdna2spirv.translate()."""
    text = f"; {name}  hand-written test program\n" + "".join(f"\t{ln.strip()}\n" for ln in lines)
    with tempfile.TemporaryDirectory() as tmp:
        os.mkdir(os.path.join(tmp, "disasm"))
        s_path = os.path.join(tmp, "disasm", f"{name}.s")
        with open(s_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        with open(os.path.join(tmp, f"{name}.dxbc"), "wb") as f:
            f.write(dxbc)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            result = rdna2spirv.translate(s_path)
            gc.collect()
        return result


def vgpr_stores(spvasm):
    """Return [(vgpr_index, stored_id), ...] in emission order for stores into %vgpr."""
    ptr_to_idx = {}
    const_u = {}
    stores = []
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = OpConstant %uint (\d+)", line)
        if m:
            const_u[m.group(1)] = int(m.group(2))
            continue
        m = re.match(r"\s*(%\w+) = OpAccessChain %_ptr_Function_uint %vgpr (%\w+)", line)
        if m and m.group(2) in const_u:
            ptr_to_idx[m.group(1)] = const_u[m.group(2)]
            continue
        m = re.match(r"\s*OpStore (%\w+) (%\w+)", line)
        if m and m.group(1) in ptr_to_idx:
            stores.append((ptr_to_idx[m.group(1)], m.group(2)))
    return stores


def parse_defs(spvasm):
    """Map %id -> (opcode, operands_list) for SSA instructions in SPIR-V assembly."""
    defs = {}
    for line in spvasm.splitlines():
        m = re.match(r"\s*(%\w+) = (Op\w+)(?: (.*))?$", line)
        if m:
            rest = m.group(3).split() if m.group(3) else []
            defs[m.group(1)] = (m.group(2), rest)
    return defs


def unwrap_dst_value(defs, sid):
    """Unwrap EXEC-masked OpSelect %uint <act> <new_val> <old_val> emitted by Gen.dst()."""
    op, args = defs[sid]
    if op == "OpSelect":
        return args[2]
    return sid


def through_select(defs, vid):
    """Past the OpSelects SHADER-388 puts on each size word (0 past the view's last level), to the queried value."""
    while defs[vid][0] == "OpSelect":
        vid = defs[vid][1][2]                     # [type, condition, in-range value, 0]
    return vid


def reaches(defs, vid, pred, seen=None):
    """True when some definition feeding `vid` satisfies pred(op, args)."""
    seen = set() if seen is None else seen
    if vid in seen or vid not in defs:
        return False
    seen.add(vid)
    op, args = defs[vid]
    return pred(op, args) or any(reaches(defs, a, pred, seen) for a in args if a.startswith("%"))


class ImageGetResinfoTranslateTests(unittest.TestCase):
    def test_2d_full_dmask_emits_image_query_and_all_four_channels(self):
        spv = translate_lines("ps_resinfo_2d", [
            "v_mov_b32 v0, 0",
            "image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpCapability ImageQuery", spv)
        self.assertIn("OpAccessChain %_ptr_UC_img %images ", spv)
        self.assertIn("OpLoad %img2d ", spv)
        self.assertRegex(spv, r"%\w+ = OpImageQuerySizeLod %v2int %\w+ %\w+")
        self.assertRegex(spv, r"%\w+ = OpImageQueryLevels %int %\w+")
        self.assertNotIn("%images_arr", spv)

        defs = parse_defs(spv)
        stores = [(idx, unwrap_dst_value(defs, sid)) for idx, sid in vgpr_stores(spv) if idx in (4, 5, 6, 7)]
        self.assertEqual([idx for idx, _ in stores], [4, 5, 6, 7])
        # v4 <- width (extract 0 of SizeLod), v5 <- height (extract 1 of SizeLod),
        # v6 <- constant 1 (current translator output; note: measured RDNA2 hardware rule is 0 per
        # docs/shader_validation.md and tests/test_image_query_probe.py — this test characterizes
        # current translator output and does not pin the hardware depth rule),
        # v7 <- levels (OpImageQueryLevels).
        src_ops = [defs[through_select(defs, defs[sid][1][1])][0] for _, sid in stores]
        self.assertEqual(src_ops[0], "OpCompositeExtract")
        self.assertEqual(src_ops[1], "OpCompositeExtract")
        self.assertEqual(src_ops[2], "OpBitcast")  # as_int(cu(1)) -> Bitcast %uint
        # a 2D image's depth word is 0 on RDNA2 (SHADER-388, measured with tools/image_query_probe.py)
        self.assertEqual(defs[defs[stores[2][1]][1][1]][1][1], "%c0")
        self.assertEqual(src_ops[3], "OpImageQueryLevels")

    def test_2d_array_uses_v3int_and_images_arr(self):
        spv = translate_lines("ps_resinfo_2da", [
            "v_mov_b32 v0, 1",
            "image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D_ARRAY",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpCapability ImageQuery", spv)
        self.assertIn("OpAccessChain %_ptr_UC_imga %images_arr ", spv)
        self.assertIn("OpLoad %img2da ", spv)
        self.assertRegex(spv, r"%\w+ = OpImageQuerySizeLod %v3int %\w+ %\w+")
        self.assertRegex(spv, r"%\w+ = OpImageQueryLevels %int %\w+")

        defs = parse_defs(spv)
        stores = [(idx, unwrap_dst_value(defs, sid)) for idx, sid in vgpr_stores(spv) if idx in (4, 5, 6, 7)]
        self.assertEqual([idx for idx, _ in stores], [4, 5, 6, 7])
        # For 2D_ARRAY, channel 2 (v6) comes from SizeLod %v3int axis 2 (the layer count, minified since SHADER-388).
        c2 = defs[stores[2][1]][1][1]
        self.assertTrue(reaches(defs, c2, lambda op, a: op == "OpCompositeExtract" and a[2] == "2"))

    def test_partial_dmask_compacts_written_vgprs(self):
        # dmask:0x8 (bit 3 only -> mip levels) must write ONLY v4 (v[4:4]), not v5..v7.
        spv_levels_only = translate_lines("ps_resinfo_dmask8", [
            "v_mov_b32 v0, 0",
            "image_get_resinfo v4, v0, s[8:15] dmask:0x8 dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ])
        defs = parse_defs(spv_levels_only)
        written = [(idx, unwrap_dst_value(defs, sid)) for idx, sid in vgpr_stores(spv_levels_only) if idx >= 4]
        self.assertEqual([idx for idx, _ in written], [4])
        self.assertEqual(defs[defs[written[0][1]][1][1]][0], "OpImageQueryLevels")

        # dmask:0x9 (bits 0 and 3 -> width and mip levels) must compact into v4 and v5 only.
        spv_w_lv = translate_lines("ps_resinfo_dmask9", [
            "v_mov_b32 v0, 0",
            "image_get_resinfo v[4:5], v0, s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ])
        defs_w_lv = parse_defs(spv_w_lv)
        written_w_lv = [(idx, unwrap_dst_value(defs_w_lv, sid)) for idx, sid in vgpr_stores(spv_w_lv) if idx >= 4]
        self.assertEqual([idx for idx, _ in written_w_lv], [4, 5])
        self.assertEqual(defs_w_lv[through_select(defs_w_lv, defs_w_lv[written_w_lv[0][1]][1][1])][0],
                         "OpCompositeExtract")
        self.assertEqual(defs_w_lv[defs_w_lv[written_w_lv[1][1]][1][1]][0], "OpImageQueryLevels")

    def test_non_ps_stage_supports_resinfo(self):
        spv = translate_lines("vs_resinfo", [
            "v_mov_b32 v0, 0",
            "image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D",
            "s_endpgm",
        ])
        self.assertIn("OpCapability ImageQuery", spv)
        self.assertIn("OpImageQuerySizeLod %v2int", spv)
        self.assertIn("OpImageQueryLevels %int", spv)

    def test_unsupported_dims_and_mods_raise(self):
        for dim in ("SQ_RSRC_IMG_1D", "SQ_RSRC_IMG_3D", "SQ_RSRC_IMG_CUBE", "SQ_RSRC_IMG_1D_ARRAY"):
            with self.subTest(dim=dim):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_resinfo_bad_dim", [
                        "v_mov_b32 v0, 0",
                        f"image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:{dim}",
                        "exp mrt0 v4, v5, v6, v7 done vm",
                        "s_endpgm",
                    ])
        for mod in ("a16", "d16"):
            with self.subTest(mod=mod):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_resinfo_bad_mod", [
                        "v_mov_b32 v0, 0",
                        f"image_get_resinfo v[4:7], v0, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D {mod}",
                        "exp mrt0 v4, v5, v6, v7 done vm",
                        "s_endpgm",
                    ])


class ImageGetLodTranslateTests(unittest.TestCase):
    def test_2d_and_2d_array_emit_image_query_lod(self):
        for dim, img_ty, arr_name, ptr_ty, simg_ty in (
            ("SQ_RSRC_IMG_2D", "%img2d", "%images", "%_ptr_UC_img", "%simg2d"),
            ("SQ_RSRC_IMG_2D_ARRAY", "%img2da", "%images_arr", "%_ptr_UC_imga", "%simg2da"),
        ):
            with self.subTest(dim=dim):
                spv = translate_lines("ps_get_lod", [
                    "v_mov_b32 v0, 0.25",
                    "v_mov_b32 v1, 0.5",
                    "image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:" + dim,
                    "exp mrt0 v4, v5, v4, v5 done vm",
                    "s_endpgm",
                ])
                self.assertIn("OpCapability ImageQuery", spv)
                self.assertIn(f"OpAccessChain {ptr_ty} {arr_name} ", spv)
                self.assertIn(f"OpLoad {img_ty} ", spv)
                self.assertIn("OpAccessChain %_ptr_UC_smp %samplers ", spv)
                self.assertIn("OpLoad %smp ", spv)
                self.assertIn(f"OpSampledImage {simg_ty} ", spv)
                self.assertRegex(spv, r"%\w+ = OpImageQueryLod %v2float %\w+ %\w+")

                defs = parse_defs(spv)
                written = [(idx, unwrap_dst_value(defs, sid)) for idx, sid in vgpr_stores(spv) if idx in (4, 5)]
                self.assertEqual([idx for idx, _ in written], [4, 5])
                # Both outputs are OpBitcast %uint of OpCompositeExtract %float from OpImageQueryLod.
                for comp_idx, (_, sid) in enumerate(written):
                    self.assertEqual(defs[sid][0], "OpBitcast")
                    ext_id = defs[sid][1][1]
                    self.assertEqual(defs[ext_id][0], "OpCompositeExtract")
                    self.assertEqual(defs[defs[ext_id][1][1]][0], "OpImageQueryLod")
                    self.assertEqual(defs[ext_id][1][2], str(comp_idx))

    def test_dmask_selects_clamped_or_unclamped_lod(self):
        # dmask:0x1 -> channel 0 (mipmap/clamped lod) written to v4 only.
        spv_c0 = translate_lines("ps_get_lod_d1", [
            "v_mov_b32 v0, 0.25",
            "v_mov_b32 v1, 0.5",
            "image_get_lod v4, v[0:1], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ])
        defs0 = parse_defs(spv_c0)
        w0 = [(idx, unwrap_dst_value(defs0, sid)) for idx, sid in vgpr_stores(spv_c0) if idx >= 4]
        self.assertEqual([idx for idx, _ in w0], [4])
        self.assertEqual(defs0[defs0[w0[0][1]][1][1]][1][2], "0")

        # dmask:0x2 -> channel 1 (relative/unclamped lod) compacted into v4 only.
        spv_c1 = translate_lines("ps_get_lod_d2", [
            "v_mov_b32 v0, 0.25",
            "v_mov_b32 v1, 0.5",
            "image_get_lod v4, v[0:1], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D",
            "exp mrt0 v4, v4, v4, v4 done vm",
            "s_endpgm",
        ])
        defs1 = parse_defs(spv_c1)
        w1 = [(idx, unwrap_dst_value(defs1, sid)) for idx, sid in vgpr_stores(spv_c1) if idx >= 4]
        self.assertEqual([idx for idx, _ in w1], [4])
        self.assertEqual(defs1[defs1[w1[0][1]][1][1]][1][2], "1")

    def test_a16_unpacks_half2x16_coordinates(self):
        spv = translate_lines("ps_get_lod_a16", [
            "v_mov_b32 v0, 0x38003400",
            "image_get_lod v[4:5], v0, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D a16",
            "exp mrt0 v4, v5, v4, v5 done vm",
            "s_endpgm",
        ])
        self.assertIn("OpExtInst %v2float %glsl UnpackHalf2x16", spv)
        self.assertRegex(spv, r"%\w+ = OpImageQueryLod %v2float %\w+ %\w+")

    def test_unsupported_stages_dims_and_a16_flags_raise(self):
        with self.assertRaises(rdna2spirv.Unsupported):
            translate_lines("vs_get_lod", [
                "v_mov_b32 v0, 0.25",
                "v_mov_b32 v1, 0.5",
                "image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D",
                "s_endpgm",
            ])
        for dim in ("SQ_RSRC_IMG_1D", "SQ_RSRC_IMG_3D", "SQ_RSRC_IMG_CUBE", "SQ_RSRC_IMG_1D_ARRAY"):
            with self.subTest(dim=dim):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_get_lod_bad_dim", [
                        "v_mov_b32 v0, 0.25",
                        "v_mov_b32 v1, 0.5",
                        f"image_get_lod v[4:5], v[0:1], s[8:15], s[16:19] dmask:0x3 dim:{dim}",
                        "exp mrt0 v4, v5, v4, v5 done vm",
                        "s_endpgm",
                    ])
        for flag in ("tfe", "lwe"):
            with self.subTest(flag=flag):
                with self.assertRaises(rdna2spirv.Unsupported):
                    translate_lines("ps_get_lod_a16_flag", [
                        "v_mov_b32 v0, 0x38003400",
                        f"image_get_lod v[4:5], v0, s[8:15], s[16:19] dmask:0x3 dim:SQ_RSRC_IMG_2D a16 {flag}",
                        "exp mrt0 v4, v5, v4, v5 done vm",
                        "s_endpgm",
                    ])


class ImageQueryNegativeControlTests(unittest.TestCase):
    def test_shader_without_image_query_omits_capability_and_opcodes(self):
        spv = translate_lines("ps_no_query", [
            "v_mov_b32 v4, 1",
            "v_mov_b32 v5, 2",
            "v_mov_b32 v6, 3",
            "v_mov_b32 v7, 4",
            "exp mrt0 v4, v5, v6, v7 done vm",
            "s_endpgm",
        ])
        self.assertNotIn("OpCapability ImageQuery", spv)
        self.assertNotIn("OpImageQuerySizeLod", spv)
        self.assertNotIn("OpImageQueryLevels", spv)
        self.assertNotIn("OpImageQueryLod", spv)


if __name__ == "__main__":
    unittest.main()
