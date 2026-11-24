"""Semantic interpreter and translator tests for perspective interpolation parameter calculation.

Covers RDNA2 perspective interpolation instructions:
- v_interp_p1_f32: computes D = P10 * S + P0 (first axis / barycentric weight i)
- v_interp_p2_f32: computes D = P20 * S + D (second axis / barycentric weight j, accumulating p1)
Combined result: D = P0 + i * (P1 - P0) + j * (P2 - P0) = (1 - i - j) * P0 + i * P1 + j * P2.

Under perspective projection with clip-space depths W0, W1, W2 > 0 and screen-space
barycentric weights (l0, l1, l2), the hardware SPI computes perspective-corrected
barycentric coordinates:
  inv_W = l0 / W0 + l1 / W1 + l2 / W2
  i_persp = (l1 / W1) / inv_W
  j_persp = (l2 / W2) / inv_W
Feeding (i_persp, j_persp) into (v_interp_p1_f32, v_interp_p2_f32) yields the exact
perspective-correct attribute interpolation across the triangle.
"""
import gc
import math
import os
import struct
import sys
import tempfile
import unittest
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TOOLS_DIR = os.environ.get("LEONIDA_TOOLS_DIR") or os.path.join(ROOT, "tools")

sys.dont_write_bytecode = True
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import rdna2spirv
import spirv_mini_interp as si

STUB_DXBC = b"DXBC" + bytes(24) + struct.pack("<I", 0) + bytes(4)


def f32_round(val):
    try:
        return struct.unpack("<f", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return float("inf") if val > 0 else float("-inf")


def f32_to_bits(val):
    try:
        return struct.unpack("<I", struct.pack("<f", float(val)))[0]
    except OverflowError:
        return 0x7F800000 if val > 0 else 0xFF800000


def bits_to_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def v_interp_p1_f32_ref(i_coord, p10, p0):
    return f32_round(f32_round(p10 * i_coord) + p0)


def v_interp_p2_f32_ref(j_coord, p20, p1_accum):
    return f32_round(f32_round(p20 * j_coord) + p1_accum)


def v_interp_eval_ref(i_coord, j_coord, p0, p1, p2):
    p10 = f32_round(p1 - p0)
    p20 = f32_round(p2 - p0)
    p1_accum = v_interp_p1_f32_ref(i_coord, p10, p0)
    return v_interp_p2_f32_ref(j_coord, p20, p1_accum)


def compute_perspective_barycentrics(l0, l1, l2, w0, w1, w2):
    inv_w0 = 1.0 / w0
    inv_w1 = 1.0 / w1
    inv_w2 = 1.0 / w2
    inv_w_total = l0 * inv_w0 + l1 * inv_w1 + l2 * inv_w2
    i_persp = (l1 * inv_w1) / inv_w_total
    j_persp = (l2 * inv_w2) / inv_w_total
    return f32_round(i_persp), f32_round(j_persp)


def analytical_perspective_interpolate(p0, p1, p2, l0, l1, l2, w0, w1, w2):
    inv_w0 = 1.0 / w0
    inv_w1 = 1.0 / w1
    inv_w2 = 1.0 / w2
    inv_w_total = l0 * inv_w0 + l1 * inv_w1 + l2 * inv_w2
    numer = l0 * p0 * inv_w0 + l1 * p1 * inv_w1 + l2 * p2 * inv_w2
    return f32_round(numer / inv_w_total)


def translate_snippet(asm_body, shader_name="ps_test"):
    with tempfile.TemporaryDirectory() as tmp:
        disasm_dir = os.path.join(tmp, "disasm")
        os.mkdir(disasm_dir)
        src_path = os.path.join(disasm_dir, f"{shader_name}.s")
        with open(src_path, "w", encoding="utf-8") as f:
            f.write(f"; {shader_name}\n{asm_body}\n")
        with open(os.path.join(tmp, f"{shader_name}.dxbc"), "wb") as f:
            f.write(STUB_DXBC)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ResourceWarning)
            spv = rdna2spirv.translate(src_path)
            gc.collect()
        return spv


def make_spirv_interp_module(p0, p10, p20, i_val, j_val):
    return f"""\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%_ptr_Output_uint = OpTypePointer Output %uint
%out = OpVariable %_ptr_Output_uint Output
%p0 = OpConstant %float {p0}
%p10 = OpConstant %float {p10}
%p20 = OpConstant %float {p20}
%i = OpConstant %float {i_val}
%j = OpConstant %float {j_val}
%main = OpFunction %void None %fn
%entry = OpLabel
%p1 = OpExtInst %float %glsl Fma %p10 %i %p0
%p2 = OpExtInst %float %glsl Fma %p20 %j %p1
%res = OpBitcast %uint %p2
OpStore %out %res
OpReturn
OpFunctionEnd
"""


def make_spirv_interp_vec4_module(p0_vec, p10_vec, p20_vec, i_val, j_val):
    header = """\
OpCapability Shader
%glsl = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %out_mrt0
%void = OpTypeVoid
%fn = OpTypeFunction %void
%uint = OpTypeInt 32 0
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4 = OpTypePointer Output %v4float
%out_mrt0 = OpVariable %_ptr_Output_v4 Output
"""
    constants = f"""\
%i = OpConstant %float {i_val}
%j = OpConstant %float {j_val}
"""
    for ch_idx, ch_name in enumerate(("x", "y", "z", "w")):
        constants += f"%p0_{ch_name} = OpConstant %float {p0_vec[ch_idx]}\n"
        constants += f"%p10_{ch_name} = OpConstant %float {p10_vec[ch_idx]}\n"
        constants += f"%p20_{ch_name} = OpConstant %float {p20_vec[ch_idx]}\n"

    body = """\
%main = OpFunction %void None %fn
%entry = OpLabel
"""
    for ch_name in ("x", "y", "z", "w"):
        body += f"%p1_{ch_name} = OpExtInst %float %glsl Fma %p10_{ch_name} %i %p0_{ch_name}\n"
        body += f"%p2_{ch_name} = OpExtInst %float %glsl Fma %p20_{ch_name} %j %p1_{ch_name}\n"

    body += """\
%res_vec = OpCompositeConstruct %v4float %p2_x %p2_y %p2_z %p2_w
OpStore %out_mrt0 %res_vec
OpReturn
OpFunctionEnd
"""
    return header + constants + body


class PerspectiveMathReferenceTests(unittest.TestCase):
    """Mathematical reference verification for perspective interpolation parameters."""

    def test_vertex_0_boundary(self):
        result = v_interp_eval_ref(0.0, 0.0, 10.0, 20.0, 30.0)
        self.assertEqual(result, 10.0)

    def test_vertex_1_boundary(self):
        result = v_interp_eval_ref(1.0, 0.0, 10.0, 20.0, 30.0)
        self.assertEqual(result, 20.0)

    def test_vertex_2_boundary(self):
        result = v_interp_eval_ref(0.0, 1.0, 10.0, 20.0, 30.0)
        self.assertEqual(result, 30.0)

    def test_edge_midpoints(self):
        mid_01 = v_interp_eval_ref(0.5, 0.0, 10.0, 20.0, 30.0)
        self.assertEqual(mid_01, 15.0)

        mid_02 = v_interp_eval_ref(0.0, 0.5, 10.0, 20.0, 30.0)
        self.assertEqual(mid_02, 20.0)

        mid_12 = v_interp_eval_ref(0.5, 0.5, 10.0, 20.0, 30.0)
        self.assertEqual(mid_12, 25.0)

    def test_centroid_affine(self):
        third = f32_round(1.0 / 3.0)
        result = v_interp_eval_ref(third, third, 9.0, 18.0, 27.0)
        self.assertAlmostEqual(result, 18.0, delta=1e-4)

    def test_perspective_parameter_calculation_equal_depth(self):
        l0, l1, l2 = 0.2, 0.5, 0.3
        w0, w1, w2 = 4.0, 4.0, 4.0
        i_persp, j_persp = compute_perspective_barycentrics(l0, l1, l2, w0, w1, w2)
        self.assertAlmostEqual(i_persp, 0.5, places=5)
        self.assertAlmostEqual(j_persp, 0.3, places=5)

        interp_res = v_interp_eval_ref(i_persp, j_persp, 10.0, 50.0, 100.0)
        analytic = analytical_perspective_interpolate(10.0, 50.0, 100.0, l0, l1, l2, w0, w1, w2)
        self.assertAlmostEqual(interp_res, analytic, delta=1e-4)

    def test_perspective_parameter_calculation_varying_depth(self):
        test_cases = [
            (0.333333, 0.333333, 0.333334, 1.0, 2.0, 4.0, 100.0, 200.0, 400.0),
            (0.1, 0.7, 0.2, 0.5, 5.0, 2.5, 0.0, 1.0, 0.5),
            (0.6, 0.2, 0.2, 10.0, 1.0, 5.0, -10.0, 20.0, 50.0),
            (0.0, 0.5, 0.5, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0),
        ]
        for l0, l1, l2, w0, w1, w2, p0, p1, p2 in test_cases:
            with self.subTest(w=(w0, w1, w2), p=(p0, p1, p2)):
                i_p, j_p = compute_perspective_barycentrics(l0, l1, l2, w0, w1, w2)
                interp_val = v_interp_eval_ref(i_p, j_p, p0, p1, p2)
                analytic_val = analytical_perspective_interpolate(p0, p1, p2, l0, l1, l2, w0, w1, w2)
                self.assertAlmostEqual(interp_val, analytic_val, delta=1e-3)

    def test_flat_attribute_invariance(self):
        for i_coord, j_coord in [(0.0, 0.0), (0.2, 0.7), (1.0, 0.0), (0.5, 0.5), (1.5, -0.5)]:
            with self.subTest(coord=(i_coord, j_coord)):
                res = v_interp_eval_ref(i_coord, j_coord, 42.0, 42.0, 42.0)
                self.assertEqual(res, 42.0)

    def test_negative_attribute_slopes(self):
        res = v_interp_eval_ref(0.25, 0.5, 100.0, 60.0, 20.0)
        self.assertEqual(res, 50.0)


class InterpreterPerspectiveFmaTests(unittest.TestCase):
    """Direct execution of perspective interpolation FMA sequences in spirv_mini_interp."""

    def test_spirv_eval_corner_vertices(self):
        cases = [
            (5.0, 10.0, 15.0, 0.0, 0.0, 5.0),
            (5.0, 10.0, 15.0, 1.0, 0.0, 15.0),
            (5.0, 10.0, 15.0, 0.0, 1.0, 20.0),
        ]
        for p0, p10, p20, i_c, j_c, expected in cases:
            with self.subTest(expected=expected):
                mod = make_spirv_interp_module(p0, p10, p20, i_c, j_c)
                out = si.run_fragment(mod)
                res_bits = out["%out"][0]
                expected_bits = f32_to_bits(expected)
                self.assertEqual(f"{res_bits:#010x}", f"{expected_bits:#010x}")

    def test_spirv_eval_perspective_sample_point(self):
        p0 = 2.0
        p10 = 6.0
        p20 = 14.0
        i_c = 0.25
        j_c = 0.375
        expected = p0 + p10 * i_c + p20 * j_c
        mod = make_spirv_interp_module(p0, p10, p20, i_c, j_c)
        out = si.run_fragment(mod)
        res_bits = out["%out"][0]
        expected_bits = f32_to_bits(expected)
        self.assertEqual(f"{res_bits:#010x}", f"{expected_bits:#010x}")

    def test_spirv_eval_vec4_attribute(self):
        p0_vec = (1.0, 2.0, 3.0, 4.0)
        p10_vec = (4.0, 6.0, 8.0, 10.0)
        p20_vec = (12.0, 14.0, 16.0, 18.0)
        i_c = 0.5
        j_c = 0.25

        expected_vec = [
            p0 + p10 * i_c + p20 * j_c
            for p0, p10, p20 in zip(p0_vec, p10_vec, p20_vec)
        ]
        mod = make_spirv_interp_vec4_module(p0_vec, p10_vec, p20_vec, i_c, j_c)
        out = si.run_fragment(mod)
        self.assertIn("%out_mrt0", out)
        mrt0 = out["%out_mrt0"]
        self.assertEqual(len(mrt0), 4)
        for ch_idx, (got_bits, want_val) in enumerate(zip(mrt0, expected_vec)):
            want_bits = f32_to_bits(want_val)
            self.assertEqual(f"{got_bits:#010x}", f"{want_bits:#010x}")

    def test_spirv_eval_subnormal_and_zero(self):
        mod_zero = make_spirv_interp_module(0.0, 0.0, 0.0, 0.0, 0.0)
        out_zero = si.run_fragment(mod_zero)
        self.assertEqual(out_zero["%out"][0], 0)

        mod_subnormal = make_spirv_interp_module(1e-40, 0.0, 0.0, 0.5, 0.5)
        out_subnormal = si.run_fragment(mod_subnormal)
        self.assertEqual(out_subnormal["%out"][0], f32_to_bits(1e-40))


class GeometryInvariantTests(unittest.TestCase):
    """Geometric invariants and coordinate transformation guarantees."""

    def test_attribute_translation_invariance(self):
        i_coord, j_coord = 0.3, 0.4
        p0, p1, p2 = 5.0, 15.0, 25.0
        base = v_interp_eval_ref(i_coord, j_coord, p0, p1, p2)

        offset = 17.5
        shifted = v_interp_eval_ref(i_coord, j_coord, p0 + offset, p1 + offset, p2 + offset)
        self.assertAlmostEqual(shifted - base, offset, places=5)

    def test_attribute_scaling_invariance(self):
        i_coord, j_coord = 0.25, 0.35
        p0, p1, p2 = 8.0, 16.0, 24.0
        base = v_interp_eval_ref(i_coord, j_coord, p0, p1, p2)

        scale = 3.5
        scaled = v_interp_eval_ref(i_coord, j_coord, p0 * scale, p1 * scale, p2 * scale)
        self.assertAlmostEqual(scaled, base * scale, places=4)

    def test_vertex_permutation_symmetry(self):
        p0, p1, p2 = 1.0, 7.0, 13.0
        i_coord, j_coord = 0.2, 0.6
        val_normal = v_interp_eval_ref(i_coord, j_coord, p0, p1, p2)
        val_swapped = v_interp_eval_ref(j_coord, i_coord, p0, p2, p1)
        self.assertAlmostEqual(val_normal, val_swapped, places=5)

    def test_extrapolation_continuity(self):
        p0, p1, p2 = 0.0, 10.0, 20.0
        val_outside_neg = v_interp_eval_ref(-0.5, 0.0, p0, p1, p2)
        self.assertEqual(val_outside_neg, -5.0)

        val_outside_pos = v_interp_eval_ref(1.5, 0.0, p0, p1, p2)
        self.assertEqual(val_outside_pos, 15.0)


class TranslatorInterpSemanticsTests(unittest.TestCase):
    """Translator binding checks for v_interp_p1_f32 and v_interp_p2_f32."""

    def test_interp_p1_p2_marks_attribute_interpolated_not_flat(self):
        asm = """\
\tv_interp_p1_f32_e32 v2, v0, attr0.x
\tv_interp_p2_f32_e32 v2, v1, attr0.x
\texp mrt0 v2, v2, v2, v2 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_interp_test")
        lines = [line.strip() for line in spv.splitlines()]

        self.assertTrue(any("OpDecorate %in_attr0 Location 0" in line for line in lines))
        self.assertFalse(any("OpDecorate %in_attr0 Flat" in line for line in lines))
        self.assertTrue(any("OpAccessChain %_ptr_Input_float %in_attr0 %c0" in line for line in lines))

    def test_interp_mov_alone_marks_attribute_flat(self):
        asm = """\
\tv_interp_mov_f32_e32 v2, p0, attr0.x
\texp mrt0 v2, v2, v2, v2 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_flat_test")
        lines = [line.strip() for line in spv.splitlines()]

        self.assertTrue(any("OpDecorate %in_attr0 Location 0" in line for line in lines))
        self.assertTrue(any("OpDecorate %in_attr0 Flat" in line for line in lines))

    def test_interp_p1_p2_multi_attribute_channels(self):
        asm = """\
\tv_interp_p1_f32_e32 v2, v0, attr0.x
\tv_interp_p2_f32_e32 v2, v1, attr0.x
\tv_interp_p1_f32_e32 v3, v0, attr0.y
\tv_interp_p2_f32_e32 v3, v1, attr0.y
\tv_interp_p1_f32_e32 v4, v0, attr1.z
\tv_interp_p2_f32_e32 v4, v1, attr1.z
\tv_interp_p1_f32_e32 v5, v0, attr2.w
\tv_interp_p2_f32_e32 v5, v1, attr2.w
\texp mrt0 v2, v3, v4, v5 done vm
\ts_endpgm
"""
        spv = translate_snippet(asm, "ps_multi_attr")
        lines = [line.strip() for line in spv.splitlines()]

        self.assertTrue(any("OpDecorate %in_attr0 Location 0" in line for line in lines))
        self.assertTrue(any("OpDecorate %in_attr1 Location 1" in line for line in lines))
        self.assertTrue(any("OpDecorate %in_attr2 Location 2" in line for line in lines))

        self.assertFalse(any("OpDecorate %in_attr0 Flat" in line for line in lines))
        self.assertFalse(any("OpDecorate %in_attr1 Flat" in line for line in lines))
        self.assertFalse(any("OpDecorate %in_attr2 Flat" in line for line in lines))

        self.assertTrue(any("OpAccessChain %_ptr_Input_float %in_attr0 %c0" in line for line in lines))
        self.assertTrue(any("OpAccessChain %_ptr_Input_float %in_attr0 %c1" in line for line in lines))
        self.assertTrue(any("OpAccessChain %_ptr_Input_float %in_attr1 %c2" in line for line in lines))
        self.assertTrue(any("OpAccessChain %_ptr_Input_float %in_attr2 %c3" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
