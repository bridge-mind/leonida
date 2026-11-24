"""v_interp_mov_f32 with the p10 / p20 vertex selects: the attribute deltas P1 - P0 and P2 - P0.

RDNA2 ISA, VINTRP: v_interp_mov_f32 returns {P10, P20, P0}[S], the three values the hardware keeps per attribute
for the p1/p2 pair (P = P0 + i * P10 + j * P20). p0 is the provoking vertex (Flat); p10 and p20 read the location
PerVertexKHR (like the *_f16 interps) and subtract vertex 0's word from vertex 1's or 2's. Hand-written programs
only; no game files.

    python -m unittest discover -s tests/semantic -p test_interp_mov_delta.py
"""
import struct
import unittest

import support
import rdna2spirv

BARY = (0.25, 0.25, 0.5)
VERTS0 = ((1.0, 10.0, 100.0, -1.0), (3.0, 14.0, 107.5, 2.0), (6.0, 19.0, 115.25, -4.0))


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


WORDS0 = [tuple(f32_bits(x) for x in v) for v in VERTS0]

DELTAS = ["v_interp_mov_f32_e32 v2, p10, attr0.x",     # 3 - 1
          "v_interp_mov_f32_e32 v3, p20, attr0.y",     # 19 - 10
          "v_interp_mov_f32_e32 v4, p0, attr0.z",      # vertex 0
          "v_interp_mov_f32_e32 v5, p20, attr0.w",     # -4 - -1
          "exp mrt0 v2, v3, v4, v5 done vm",
          "s_endpgm"]


class InterpMovDelta(unittest.TestCase):
    def test_p10_p20_are_vertex_deltas(self):
        spv = support.translate_lines("ps_mov_delta", DELTAS)
        self.assertIn("OpDecorate %in_attr0 PerVertexKHR", spv)
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)
        words = support.mrt0_words(spv, inputs=support.varyings(spv, {0: WORDS0}, BARY))
        self.assertEqual(words, [f32_bits(2.0), f32_bits(9.0), f32_bits(100.0), f32_bits(-3.0)])

    def test_deltas_rebuild_the_p1_p2_interpolation(self):
        # P0 + i * P10 + j * P20 from the three movs equals the p1/p2 pair's result on the same location.
        spv = support.translate_lines("ps_mov_rebuild", [
            "v_interp_mov_f32_e32 v2, p0, attr0.z",
            "v_interp_mov_f32_e32 v3, p10, attr0.z",
            "v_interp_mov_f32_e32 v4, p20, attr0.z",
            "v_fmac_f32_e32 v2, v0, v3",
            "v_fmac_f32_e32 v2, v1, v4",
            "v_interp_p1_f32_e32 v5, v0, attr0.z",
            "v_interp_p2_f32_e32 v5, v1, attr0.z",
            "exp mrt0 v2, v5, v3, v4 done vm",
            "s_endpgm"])
        words = support.mrt0_words(spv, inputs=support.varyings(spv, {0: WORDS0}, BARY))
        expect = 100.0 + 0.25 * 7.5 + 0.5 * 15.25
        self.assertEqual(words, [f32_bits(expect), f32_bits(expect), f32_bits(7.5), f32_bits(15.25)])

    def test_p0_only_location_stays_flat(self):
        spv = support.translate_lines("ps_mov_p0", ["v_interp_mov_f32_e32 v2, p0, attr1.x",
                                                    "v_interp_mov_f32_e32 v3, p10, attr0.x",
                                                    "exp mrt0 v2, v3, v3, v3 done vm",
                                                    "s_endpgm"])
        self.assertIn("OpDecorate %in_attr1 Flat", spv)
        self.assertIn("OpDecorate %in_attr0 PerVertexKHR", spv)

    def test_bary_fallback_refuses(self):
        old = rdna2spirv.BARY_FALLBACK
        rdna2spirv.BARY_FALLBACK = True
        try:
            for sel in ("p10", "p20"):
                with self.subTest(sel=sel), self.assertRaises(rdna2spirv.Unsupported) as cm:
                    support.translate_lines("ps_mov_fb", [f"v_interp_mov_f32_e32 v2, {sel}, attr0.x",
                                                          "exp mrt0 v2, v2, v2, v2 done vm",
                                                          "s_endpgm"])
                self.assertIn(f"vertex select {sel} without per-vertex inputs", str(cm.exception))
        finally:
            rdna2spirv.BARY_FALLBACK = old


if __name__ == "__main__":
    unittest.main()
