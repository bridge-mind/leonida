"""Semantic and validation tests for RDNA2 image_store with d16 (packed 16-bit data VGPRs).

Covers FSPX/FSR4 blocker `image_store 2D a16 d16` and all supported STORE_DIMS (1D, 2D,
2D_ARRAY, 3D), with and without a16 / glc, across full and sparse dmasks (0x0..0xf):
  - Float/norm view: IEEE binary16 halves widened via UnpackHalf2x16 -> %v4float.
  - UINT view: 16-bit unsigned halves zero-extended via OpBitFieldUExtract -> %v4uint_st.
  - SINT view: 16-bit signed halves sign-extended via OpBitFieldSExtract -> %v4int_st.
  - Missing dmask components default to 0.
  - Executes the translated unpacking and coordinate extraction IR in spirv_mini_interp
    to assert bit-exact texel values and unpacked a16 coordinates, plus spirv-val Vulkan 1.2
    (native and --wave64-emulate).
"""
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TOOLS = os.path.join(ROOT, "tools")
SEMANTIC = os.path.dirname(os.path.abspath(__file__))
for _p in (TOOLS, SEMANTIC):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rdna2spirv  # noqa: E402
import spirv_mini_interp as si  # noqa: E402
import support  # noqa: E402

SPIRV_AS = shutil.which("spirv-as")
SPIRV_VAL = shutil.which("spirv-val")


def _f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def _f32_bits(val: float) -> int:
    return struct.unpack("<I", struct.pack("<f", val))[0]


def _pack_u16(lo: int, hi: int) -> int:
    return (lo & 0xFFFF) | ((hi & 0xFFFF) << 16)


def _validate(spvasm: str) -> None:
    if not (SPIRV_AS and SPIRV_VAL):
        return
    with tempfile.TemporaryDirectory() as d:
        sa, sv = os.path.join(d, "m.spvasm"), os.path.join(d, "m.spv")
        with open(sa, "w", encoding="utf-8", newline="\n") as f:
            f.write(spvasm)
        for cmd in (
            [SPIRV_AS, sa, "-o", sv, "--target-env", "vulkan1.2"],
            [SPIRV_VAL, "--target-env", "vulkan1.2", "--allow-offset-texture-operand", sv],
        ):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise AssertionError(f"{cmd[0]} failed:\n{r.stderr}")


def _eval_image_store_texels(setup_lines: list[str], store_line: str) -> dict[str, tuple[int, ...]]:
    """Translate setup + image_store via rdna2spirv, cut right after the three texel
    OpCompositeConstructs (and coordinate construct), and execute in spirv_mini_interp.

    Returns {"float": (u32, ...), "uint": (u32, ...), "sint": (u32, ...), "coords": (u32, ...)}.
    """
    spv = support.translate_lines(
        "ps_img_store_d16_eval",
        setup_lines + [store_line, "exp mrt0 v0, v0, v0, v0 done vm", "s_endpgm"],
    )
    _validate(spv)

    lines = spv.splitlines()
    # Find the OpCompositeConstruct lines for %v4float, %v4uint_st, %v4int_st and coords (%v2int/%v3int)
    # emitted by image_store, plus the first OpSelectionMerge (which starts the active/view branch).
    v4f = v4u = v4s = coord_id = None
    coord_ty = None
    cut_idx = None
    for idx, ln in enumerate(lines):
        m = re.match(r"\s*(%\w+)\s*=\s*OpCompositeConstruct\s+(%\w+)\s+(.*)$", ln)
        if m:
            res, ty, _ = m.groups()
            if ty in ("%v2int", "%v3int"):
                coord_id, coord_ty = res, ty
            elif ty == "%v4float":
                v4f = res
            elif ty == "%v4uint_st":
                v4u = res
            elif ty == "%v4int_st":
                v4s = res
                cut_idx = idx + 1
                break
    assert v4f and v4u and v4s and cut_idx is not None, "image_store texel constructs not found"

    # Build a small runnable fragment ending right after v4s, writing each vector to an Output.
    # For 1D coords (no %v2int/%v3int), capture the %int operand of the first OpImageWrite instead.
    if coord_id is None:
        for ln in lines[cut_idx:]:
            m = re.match(r"\s*OpImageWrite\s+%\w+\s+(%\w+)\s+", ln)
            if m:
                coord_id = m.group(1)
                coord_ty = "%int"
                break

    # Extract header up to %main = OpFunction, add four Output vars, and emit stores at cut_idx.
    out_decls = [
        "%_ptr_Out_f = OpTypePointer Output %v4float",
        "%_ptr_Out_u = OpTypePointer Output %v4uint_st",
        "%_ptr_Out_s = OpTypePointer Output %v4int_st",
        f"%_ptr_Out_c = OpTypePointer Output {coord_ty}",
        "%cap_f = OpVariable %_ptr_Out_f Output",
        "%cap_u = OpVariable %_ptr_Out_u Output",
        "%cap_s = OpVariable %_ptr_Out_s Output",
        "%cap_c = OpVariable %_ptr_Out_c Output",
    ]
    fn_idx = next(i for i, ln in enumerate(lines) if " = OpFunction " in ln)
    # Strip bindless/image OpVariables and OpDecorate/OpEntryPoint lines irrelevant to the mini-interp,
    # keeping all types/constants and the function prefix up to cut_idx.
    prefix = []
    skip_hdr = (
        "OpTypeImage", "OpTypeSampledImage", "OpTypeSampler", "OpTypeRuntimeArray", "UniformConstant",
        "StorageBuffer", "Private", "OpConstantComposite", "%stv", "%stimg", "%desc", "%t_desc",
        "%images", "%samplers", "%itx", "DescriptorSet", "Binding", "RuntimeDescriptorArray",
        "StorageImage", "NonUniform",
    )
    for ln in lines[:fn_idx]:
        if re.search(r"OpEntryPoint ", ln):
            prefix.append('OpEntryPoint Fragment %main "main" %cap_f %cap_u %cap_s %cap_c')
        elif any(tok in ln for tok in skip_hdr):
            continue
        else:
            prefix.append(ln)
    body = lines[fn_idx:cut_idx] + [
        f"OpStore %cap_f {v4f}",
        f"OpStore %cap_u {v4u}",
        f"OpStore %cap_s {v4s}",
        f"OpStore %cap_c {coord_id}",
        "OpReturn",
        "OpFunctionEnd",
    ]
    # Remove any OpLoad/OpAccessChain to %desc_tbl (only used for idx, never feeds v4f/v4u/v4s/coords)
    # by seeding a dummy memory in spirv_mini_interp? Actually %desc_tbl is UniformConstant/StorageBuffer;
    # filter out instructions in body whose results are not needed by v4f/v4u/v4s/coord_id.
    mini_spv = "\n".join(prefix + out_decls + body) + "\n"

    # Strip desc_lookup_sgpr instructions that read %desc_tbl (not needed for texel/coord evaluation).
    # Those live strictly between coords and v4f; remove lines referencing %desc_tbl or %t_desc or %user_data
    # that aren't dependencies of v4f/v4u/v4s/coord_id.
    def _prune_to_targets(src_lines: list[str], fn_start: int, targets: set[str]) -> str:
        header = src_lines[:fn_start]
        fn_lines = src_lines[fn_start:]
        # Keep OpFunction, OpLabel, OpVariable, OpStore to %vgpr/%sgpr/%cap_*, and backward deps of targets.
        needed = set(targets)
        keep_idx: set[int] = set()
        for i in range(len(fn_lines) - 1, -1, -1):
            s = fn_lines[i].split(";", 1)[0].strip()
            toks = s.split()
            if not toks:
                continue
            if len(toks) >= 3 and toks[1] == "=":
                res, op, args = toks[0], toks[2], toks[3:]
            else:
                res, op, args = None, toks[0], toks[1:]
            if op in ("OpFunction", "OpLabel", "OpVariable", "OpReturn", "OpFunctionEnd"):
                keep_idx.add(i)
                continue
            if op == "OpStore" and args and (
                args[0].startswith("%cap_") or "%vgpr" in fn_lines[i]
            ):
                keep_idx.add(i)
                needed.update(t for t in args if t.startswith("%"))
                continue
            # Also keep OpStore into %vgpr pointers
            if op == "OpStore":
                # Check if ptr traces back to %vgpr
                keep_idx.add(i)
                needed.update(t for t in args if t.startswith("%"))
                continue
            if res in needed:
                keep_idx.add(i)
                needed.update(t for t in args if t.startswith("%"))
        # Drop any OpStore that feeds %sgpr from %user_data (not needed here)
        pruned_fn = []
        for i, ln in enumerate(fn_lines):
            if i in keep_idx and "%desc_tbl" not in ln and "%user_data" not in ln and "%ud" not in ln:
                pruned_fn.append(ln)
        return "\n".join(header + pruned_fn) + "\n"

    pruned = _prune_to_targets(
        prefix + out_decls + body,
        len(prefix + out_decls),
        {v4f, v4u, v4s, coord_id},
    )
    outs = si.run_fragment(pruned)

    def _as_words(v):
        if isinstance(v, tuple):
            return tuple(x & 0xFFFFFFFF for x in v)
        return (v & 0xFFFFFFFF,)

    return {
        "float": _as_words(outs["%cap_f"]),
        "uint": _as_words(outs["%cap_u"]),
        "sint": _as_words(outs["%cap_s"]),
        "coords": _as_words(outs["%cap_c"]),
    }


class ImageStoreD16Tests(unittest.TestCase):
    def test_rgba_d16_a16_float_uint_sint_values_exact(self):
        # FSPX pattern: image_store 2D a16 d16 with full dmask:0xf (2 packed VGPRs: v[4:5], 1 coord VGPR: v2).
        # Test positive/negative normals, subnormal, negative integer (sign-extended in SINT, zero-extended in UINT).
        lo0 = _f16_bits(1.5)         # 0x3E00
        hi0 = _f16_bits(-2.25)       # 0xC080 -> u16 49280, i16 -16256 (0xFFFFC080)
        lo1 = 0xFFFF                 # f16 NaN (checked via u16/i16), u16 65535, i16 -1 (0xFFFFFFFF)
        hi1 = 0x8000                 # f16 -0.0 (0x80000000), u16 32768, i16 -32768 (0xFFFF8000)
        w0 = _pack_u16(lo0, hi0)
        w1 = _pack_u16(lo1, hi1)
        coord_packed = _pack_u16(17, 42)
        got = _eval_image_store_texels(
            [
                f"v_mov_b32_e32 v4, 0x{w0:08x}",
                f"v_mov_b32_e32 v5, 0x{w1:08x}",
                f"v_mov_b32_e32 v2, 0x{coord_packed:08x}",
            ],
            "image_store v[4:5], v2, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16 d16",
        )
        self.assertEqual(got["coords"], (17, 42))
        self.assertEqual(got["float"][0], _f32_bits(1.5))
        self.assertEqual(got["float"][1], _f32_bits(-2.25))
        self.assertEqual(got["float"][3], 0x80000000)  # -0.0
        self.assertEqual(got["uint"], (lo0, hi0, lo1, hi1))
        self.assertEqual(got["sint"], (0x00003E00, 0xFFFFC080, 0xFFFFFFFF, 0xFFFF8000))

    def test_sparse_dmasks_pack_enabled_components_consecutively(self):
        # dmask:0x9 (R and A): both halves packed in a SINGLE VGPR v4 (R = low 16, A = high 16); G=B=0.
        w_ra = _pack_u16(_f16_bits(0.5), _f16_bits(-4.0))
        got_ra = _eval_image_store_texels(
            [
                f"v_mov_b32_e32 v4, 0x{w_ra:08x}",
                "v_mov_b32_e32 v2, 3",
                "v_mov_b32_e32 v3, 9",
            ],
            "image_store v4, v[2:3], s[8:15] dmask:0x9 dim:SQ_RSRC_IMG_2D unorm d16",
        )
        self.assertEqual(got_ra["coords"], (3, 9))
        self.assertEqual(got_ra["float"], (_f32_bits(0.5), 0, 0, _f32_bits(-4.0)))
        self.assertEqual(got_ra["uint"], (_f16_bits(0.5), 0, 0, _f16_bits(-4.0)))
        self.assertEqual(got_ra["sint"], (0x00003800, 0, 0, 0xFFFFC400))

        # dmask:0x7 (R, G, B): 2 VGPRs v[4:5] -> R=v4.lo, G=v4.hi, B=v5.lo, A=0.
        w0 = _pack_u16(0x0007, 0xFFFE)   # +7, -2
        w1 = _pack_u16(0x8001, 0x1234)   # -32767 (high half 0x1234 unused!)
        got_rgb = _eval_image_store_texels(
            [
                f"v_mov_b32_e32 v4, 0x{w0:08x}",
                f"v_mov_b32_e32 v5, 0x{w1:08x}",
                "v_mov_b32_e32 v2, 1",
                "v_mov_b32_e32 v3, 2",
            ],
            "image_store v[4:5], v[2:3], s[8:15] dmask:0x7 dim:SQ_RSRC_IMG_2D unorm d16",
        )
        self.assertEqual(got_rgb["uint"], (0x0007, 0xFFFE, 0x8001, 0))
        self.assertEqual(got_rgb["sint"], (0x00000007, 0xFFFFFFFE, 0xFFFF8001, 0))

        # dmask:0x4 (B only): 1 VGPR v4 -> B=v4.lo, R=G=A=0.
        got_b = _eval_image_store_texels(
            [
                "v_mov_b32_e32 v4, 0x12343c00",
                "v_mov_b32_e32 v2, 5",
            ],
            "image_store v4, v2, s[8:15] dmask:0x4 dim:SQ_RSRC_IMG_1D unorm d16",
        )
        self.assertEqual(got_b["coords"], (5,))
        self.assertEqual(got_b["float"], (0, 0, _f32_bits(1.0), 0))
        self.assertEqual(got_b["uint"], (0, 0, 0x3C00, 0))

    def test_all_store_dims_a16_glc_and_wave64_emulate_validate(self):
        cases = [
            ("1D", "v2", "dmask:0x1", "d16"),
            ("2D", "v2", "dmask:0xf", "a16 d16"),
            ("2D", "v[2:3]", "dmask:0x3", "glc d16"),
            ("2D_ARRAY", "v[2:3]", "dmask:0xf", "a16 d16"),
            ("3D", "v[2:4]", "dmask:0x7", "glc d16"),
            ("2D", "v2", "dmask:0x0", "a16 d16"),
        ]
        for dim, addr, dmask, extra in cases:
            with self.subTest(dim=dim, dmask=dmask, extra=extra):
                spv = support.translate_lines(
                    f"ps_img_d16_{dim.lower()}",
                    [
                        "v_mov_b32_e32 v2, 0",
                        "v_mov_b32_e32 v3, 0",
                        "v_mov_b32_e32 v4, 0",
                        "v_mov_b32_e32 v5, 0",
                        f"image_store v[4:5], {addr}, s[8:15] {dmask} dim:SQ_RSRC_IMG_{dim} unorm {extra}",
                        "exp mrt0 v4, v4, v4, v4 done vm",
                        "s_endpgm",
                    ],
                )
                _validate(spv)
                self.assertIn("OpImageWrite", spv)
                if "glc" in extra:
                    self.assertIn("Coherent", spv)

        # Also validate a wave64 compute shader under --wave64-emulate (FSPX compute path).
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "disasm"))
            os.makedirs(os.path.join(d, "fspx"))
            s_path = os.path.join(d, "disasm", "cs_img_store_d16.s")
            with open(s_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("; cs_img_store_d16  code 0x0..0x40, gfx1013, 0 invalid words\n")
                for ln in (
                    "s_mov_b64 s[16:17], exec",
                    "v_mov_b32_e32 v2, 0x00020001",
                    "v_mov_b32_e32 v4, 0x40003c00",
                    "v_mov_b32_e32 v5, 0x44004200",
                    "image_store v[4:5], v2, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm a16 d16",
                    "s_endpgm",
                ):
                    f.write(f"\t{ln} ; encoding: [0x00,0x00,0x00,0x00]\n")
            with open(os.path.join(d, "fspx", "cs_img_store_d16.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rsrc2": (16 << 1), "threads": [64, 1, 1]}, f)
            old_w64 = rdna2spirv.WAVE64_EMULATE
            try:
                rdna2spirv.WAVE64_EMULATE = True
                spv = rdna2spirv.translate(s_path)
            finally:
                rdna2spirv.WAVE64_EMULATE = old_w64
        _validate(spv)

    def test_unsupported_dims_and_modifiers_still_refused(self):
        for bad in (
            "image_store v[4:5], v[2:4], s[8:15] dmask:0xf dim:SQ_RSRC_IMG_CUBE unorm d16",
            "image_store v[4:5], v2, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16 tfe",
            "image_store v[4:5], v2, s[8:15] dmask:0xf dim:SQ_RSRC_IMG_2D unorm d16 lwe",
        ):
            with self.subTest(bad=bad):
                with self.assertRaises(rdna2spirv.Unsupported):
                    support.translate_lines(
                        "ps_img_d16_bad",
                        ["v_mov_b32_e32 v2, 0", bad, "exp mrt0 v2, v2, v2, v2 done vm", "s_endpgm"],
                    )


if __name__ == "__main__":
    unittest.main()
