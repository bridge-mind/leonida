"""Translator-driven semantic tests for pixel-shader interpolation (v_interp_p1_f32, v_interp_p2_f32, v_interp_mov_f32).

Translates hand-written RDNA2 pixel shaders through tools/rdna2spirv.py, validates the generated SPIR-V with
spirv-as/spirv-val when on PATH, and executes them in spirv_mini_interp via support.varyings / support.mrt0_words:
  - v_interp_p1_f32 + v_interp_p2_f32 across all 4 attribute channels (x/y/z/w) at vertex and interior barycentrics;
  - multiple attributes (attr0, attr1, attr2) mapped to sequential Vulkan input locations;
  - v_interp_mov_f32 p0 emitting Flat decoration and returning vertex 0's raw word regardless of barycentrics;
  - combined smooth and flat attributes in one shader;
  - inactive-lane preservation around v_interp_p1_f32 / v_interp_p2_f32.

    python -I tests/semantic/test_interp_p1_p2_f32.py -v
"""
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import support  # noqa: E402


def _f32_bits(val):
    return struct.unpack("<I", struct.pack("<f", val))[0]


def _bits_f32(word):
    return struct.unpack("<f", struct.pack("<I", word & 0xFFFFFFFF))[0]


def _validate_spv(spvasm):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    with tempfile.TemporaryDirectory() as tmp:
        spv = os.path.join(tmp, "out.spv")
        r = subprocess.run([spirv_as, "--target-env", "vulkan1.2", "-o", spv, "-"],
                           input=spvasm, text=True, capture_output=True)
        if r.returncode:
            raise AssertionError(f"spirv-as failed:\n{r.stderr}")
        r = subprocess.run([spirv_val, "--target-env", "vulkan1.2", spv],
                           capture_output=True, text=True)
        if r.returncode:
            raise AssertionError(f"spirv-val failed:\n{r.stderr}")


class InterpP1P2F32Tests(unittest.TestCase):
    def test_smooth_four_channels_at_vertices_and_interior_barycentrics(self):
        spvasm = support.translate_lines("ps_interp_4ch", [
            "s_mov_b32 m0, s0",
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "v_interp_p1_f32 v3, v0, attr0.y",
            "v_interp_p2_f32 v3, v1, attr0.y",
            "v_interp_p1_f32 v4, v0, attr0.z",
            "v_interp_p2_f32 v4, v1, attr0.z",
            "v_interp_p1_f32 v5, v0, attr0.w",
            "v_interp_p2_f32 v5, v1, attr0.w",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ])
        _validate_spv(spvasm)
        self.assertIn("OpDecorate %in_attr0 Location 0", spvasm)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spvasm)

        verts = [
            (1.0, 2.0, -4.0, 0.5),
            (5.0, -2.0, 0.0, 1.5),
            (-3.0, 6.0, 8.0, 3.5),
        ]
        attrs = {0: [tuple(_f32_bits(c) for c in v) for v in verts]}
        for bary in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                     (0.25, 0.5, 0.25), (0.2, 0.3, 0.5)):
            with self.subTest(bary=bary):
                words = support.mrt0_words(spvasm, inputs=support.varyings(spvasm, attrs, bary))
                got = [_bits_f32(w) for w in words]
                want = [sum(b * v[c] for b, v in zip(bary, verts)) for c in range(4)]
                for g, w in zip(got, want):
                    self.assertAlmostEqual(g, w, places=5)

    def test_multiple_attributes_and_flat_mov_p0(self):
        spvasm = support.translate_lines("ps_interp_multi_flat", [
            "s_mov_b32 m0, s0",
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "v_interp_p1_f32 v3, v0, attr2.w",
            "v_interp_p2_f32 v3, v1, attr2.w",
            "v_interp_mov_f32 v4, p0, attr1.y",
            "v_interp_mov_f32 v5, p0, attr1.z",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ])
        _validate_spv(spvasm)
        self.assertIn("OpDecorate %in_attr0 Location 0", spvasm)
        self.assertIn("OpDecorate %in_attr1 Location 1", spvasm)
        self.assertIn("OpDecorate %in_attr2 Location 2", spvasm)
        self.assertIn("OpDecorate %in_attr1 Flat", spvasm)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spvasm)
        self.assertNotIn("OpDecorate %in_attr2 Flat", spvasm)

        attrs = {
            0: [(_f32_bits(2.0), 0, 0, 0), (_f32_bits(6.0), 0, 0, 0), (_f32_bits(10.0), 0, 0, 0)],
            1: [(0, _f32_bits(7.25), _f32_bits(-12.5), 0),
                (0, _f32_bits(99.0), _f32_bits(88.0), 0),
                (0, _f32_bits(-5.0), _f32_bits(42.0), 0)],
            2: [(0, 0, 0, _f32_bits(10.0)), (0, 0, 0, _f32_bits(20.0)), (0, 0, 0, _f32_bits(40.0))],
        }
        bary = (0.25, 0.5, 0.25)
        words = support.mrt0_words(spvasm, inputs=support.varyings(spvasm, attrs, bary))
        got = [_bits_f32(w) for w in words]
        self.assertAlmostEqual(got[0], 0.25 * 2.0 + 0.5 * 6.0 + 0.25 * 10.0, places=5)
        self.assertAlmostEqual(got[1], 0.25 * 10.0 + 0.5 * 20.0 + 0.25 * 40.0, places=5)
        # Flat p0 reads vertex 0 regardless of barycentrics
        self.assertEqual(got[2], 7.25)
        self.assertEqual(got[3], -12.5)

    def test_inactive_lane_preserves_destination(self):
        spvasm = support.translate_lines("ps_interp_inactive", [
            "s_mov_b32 m0, s0",
            "v_mov_b32_e32 v2, 3.0",
            "v_mov_b32_e32 v3, 4.0",
            "s_mov_b32 exec_lo, 0",
            "v_interp_p1_f32 v2, v0, attr0.x",
            "v_interp_p2_f32 v2, v1, attr0.x",
            "v_interp_mov_f32 v3, p0, attr1.x",
            "s_mov_b32 exec_lo, -1",
            "v_mov_b32_e32 v4, 0.0",
            "v_mov_b32_e32 v5, 1.0",
            "exp mrt0 v2, v3, v4, v5 done vm",
            "s_endpgm",
        ])
        _validate_spv(spvasm)
        attrs = {
            0: [(_f32_bits(99.0), 0, 0, 0)] * 3,
            1: [(_f32_bits(-50.0), 0, 0, 0)] * 3,
        }
        words = support.mrt0_words(spvasm, inputs=support.varyings(spvasm, attrs, (1.0, 0.0, 0.0)))
        self.assertEqual([_bits_f32(w) for w in words], [3.0, 4.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
