"""SHADER-177: v_interp reads the barycentric pair through its VGPR source (I = v0, J = v1), and
v_interp_mov_f32 names the vertex with p0 (p10/p20 would be the other two vertices). The single-lane
model returns the attribute's own value, which is only that when the operand really holds the pair,
so anything else is refused instead of silently translating wrong (docs/design.md, PS system inputs):

* v_interp_p1/p2 (f32 and f16 forms) must read their OWN coordinate of the pair — p1 reads I = v0,
  p2 reads J = v1, per op — as the hardware register or a qualifying copy: anything else, including a
  swapped p1(v1)/p2(v0) pair or both-v0, -> Unsupported naming the register and the role (the corpus
  has none: 117,001 v0 and 116,997 v1 over 26,503 shaders);
* v_interp_mov_f32 with p10/p20 was refused here; it is now modelled as P1 - P0 / P2 - P0
  (test_interp_mov_delta.py; all 6,926 in the corpus are p0);
* a plain v_mov_b32 copy of the op's coordinate (rooted at v0 for p1, at v1 for p2) is still accepted,
  including a chain of copies: its single write in the whole shader is the mov, and its source v0/v1
  is itself unwritten up to the copy (a copy of an already-overwritten v0/v1 carries a computed value —
  the zero-before-copy measurement renders (0,0) on hardware — while zeroing after the copy is fine,
  the copy saved the value). One corpus shader reads its J through a copy of v1 and must keep
  translating (bit-identical .spv);
* a register written twice (the mov, then anything else) no longer qualifies — and since the
  computed-I/J model landed, a written-again register, a copy of an already-overwritten v0/v1, or a
  modified (dpp) mov is no longer refused but modelled: the operand is a value the program wrote,
  and the model weights the interpolation with the value loaded from %vgpr (Vonsek's zero-before-copy
  row renders P0, which is exactly those loads' result). A pristine copy in the wrong role stays
  refused: it carries no computed value to model;
* the standard forms (v0/v1 pairs, p0 mov) translate and compute exactly as before.

On main (before the guards) the refused cases translated silently and passed spirv-val: the operand
was ignored outright, so a second-pair I/J produced the very module of the standard pair.
"""
import struct
import unittest

import support
import rdna2spirv

ATTR0 = (0.5, 0.25, 0.0, 1.0)
FLAT_ATTR1_X = 0x101          # attr1.x read through v_interp_mov only: the provoking vertex's word
BARY = (0.25, 0.25, 0.5)       # (k, i, j): the weights the interpreter seeds v0/v1 from (.y/.z)
# Three DISTINCT vertices of attr0 (raw words), so a test's weights are visible in the result.
VERTS0 = ((0.0, 0.0, 1.0, 0.0), (2.0, 0.0, 3.0, 0.0), (0.0, 2.0, 5.0, 0.0))


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


WORDS0 = [tuple(f32_bits(x) for x in v) for v in VERTS0]


class InterpPairGuards(unittest.TestCase):
    def test_second_pair_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_i", ["v_interp_p1_f32_e32 v2, v4, attr0.x",
                                             "v_interp_p2_f32_e32 v2, v5, attr0.x",
                                             "exp mrt0 v2, v2, v2, v2 done vm",
                                             "s_endpgm"])
        self.assertIn("v_interp_p1_f32 I source v4", str(cm.exception))

    def test_second_pair_f16_refused(self):
        with self.assertRaises(rdna2spirv.Unsupported) as cm:
            support.translate_lines("ps_i16", ["v_interp_p1ll_f16 v4, v0, attr1.x",
                                               "v_interp_p2_f16 v4, v9, attr1.x, v4",
                                               "exp mrt0 v4, v4, v4, v4 done vm",
                                               "s_endpgm"])
        self.assertIn("v_interp_p2_f16 J source v9", str(cm.exception))

    def test_mov_vertex_select_translates(self):
        # p10/p20 are modelled as P1 - P0 / P2 - P0 (test_interp_mov_delta.py); only an unknown select is refused.
        for sel in ("p10", "p20"):
            with self.subTest(sel=sel):
                spv = support.translate_lines("ps_m", [f"v_interp_mov_f32_e32 v2, {sel}, attr1.x",
                                                       "exp mrt0 v2, v2, v2, v2 done vm",
                                                       "s_endpgm"])
                self.assertIn("OpDecorate %in_attr1 PerVertexKHR", spv)

    def test_copy_of_pair_translates_and_computes(self):
        for chain in (["v_mov_b32_e32 v6, v1"],                    # the corpus form: J read through a copy
                      ["v_mov_b32_e32 v6, v1", "v_mov_b32_e32 v7, v6"]):   # a chain of plain copies
            with self.subTest(chain=len(chain)):
                src = chain[-1].split()[1].rstrip(",")
                lines = chain + ["v_interp_p1_f32_e32 v2, v0, attr0.x",
                                 f"v_interp_p2_f32_e32 v2, {src}, attr0.x",
                                 "exp mrt0 v2, v2, v2, v2 done vm",
                                 "s_endpgm"]
                spv = support.translate_lines("ps_copy", lines)
                words = support.mrt0_words(spv, inputs={"%in_attr0": ATTR0})
                self.assertEqual(words, [f32_bits(ATTR0[0])] * 4)

    def test_copy_written_again_is_computed(self):
        # Two writes (the copy, then an add) mean the register no longer provably holds the pair, so
        # the guard would refuse it -- but the computed-I/J model represents it: the operand load
        # reads the value the add wrote. J' = I + I doubles the second weight.
        spv = support.translate_lines("ps_copy2", ["v_mov_b32_e32 v6, v1",
                                                   "v_add_f32_e32 v6, v0, v0",
                                                   "v_interp_p1_f32_e32 v2, v0, attr0.x",
                                                   "v_interp_p2_f32_e32 v2, v6, attr0.x",
                                                   "exp mrt0 v2, v2, v2, v2 done vm",
                                                   "s_endpgm"])
        words = support.mrt0_words(spv, inputs=support.varyings(spv, {0: WORDS0}, BARY))
        i, j = BARY[1], 2 * BARY[1]
        exp = VERTS0[0][0] + i * (VERTS0[1][0] - VERTS0[0][0]) + j * (VERTS0[2][0] - VERTS0[0][0])
        self.assertEqual(words, [f32_bits(exp)] * 4)
        self.assertIn("PerVertexKHR", spv)

    def test_copy_with_modifiers_is_computed(self):
        # A dpp mov is not a plain copy, so the single write does not establish the barycentric --
        # but it is still a write of the operand register, and the computed-I/J model loads whatever
        # value the program put there. The pair translates; the weights are the wave-wide dpp
        # results, so this asserts translation (and the per-vertex read), not single-lane words.
        spv = support.translate_lines("ps_copy3", ["v_mov_b32_dpp v6, v1 row_shl:1 row_mask:0xf bank_mask:0xf",
                                                   "v_interp_p1_f32_e32 v2, v0, attr0.x",
                                                   "v_interp_p2_f32_e32 v2, v6, attr0.x",
                                                   "exp mrt0 v2, v2, v2, v2 done vm",
                                                   "s_endpgm"])
        self.assertIn("OpEntryPoint Fragment", spv)
        self.assertIn("PerVertexKHR", spv)

    def test_swapped_and_same_register_roles_refused(self):
        # The roles are per op — p1 interpolates with I = v0, p2 with J = v1 — so a swapped pair
        # (p1(v1), p2(v0)) and a both-v0 pair are refused, not silently exchanged (review: kapitche).
        cases = {(1, 0): "v_interp_p1_f32 I source v1",
                 (1, 1): "v_interp_p1_f32 I source v1",
                 (0, 0): "v_interp_p2_f32 J source v0"}
        for (i, j), msg in cases.items():
            with self.subTest(i=i, j=j):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    support.translate_lines("ps_role", [f"v_interp_p1_f32_e32 v2, v{i}, attr0.x",
                                                       f"v_interp_p2_f32_e32 v2, v{j}, attr0.x",
                                                       "exp mrt0 v2, v2, v2, v2 done vm",
                                                       "s_endpgm"])
                self.assertIn(msg, str(cm.exception))

    def test_copy_root_role_enforced(self):
        # A copy only stands in for the coordinate it is rooted at: v6 = v0 cannot feed p2's J, and
        # v6 = v1 cannot feed p1's I (the other op of the pair reads the register it should).
        for chain, p1, p2, msg in ((["v_mov_b32_e32 v6, v0"], "v0", "v6", "v_interp_p2_f32 J source v6"),
                                   (["v_mov_b32_e32 v6, v1"], "v6", "v1", "v_interp_p1_f32 I source v6")):
            with self.subTest(root=chain[0]):
                with self.assertRaises(rdna2spirv.Unsupported) as cm:
                    support.translate_lines("ps_rolc", chain + [f"v_interp_p1_f32_e32 v2, {p1}, attr0.x",
                                                               f"v_interp_p2_f32_e32 v2, {p2}, attr0.x",
                                                               "exp mrt0 v2, v2, v2, v2 done vm",
                                                               "s_endpgm"])
                self.assertIn(msg, str(cm.exception))

    def test_copy_in_matching_roles_translates_and_computes(self):
        # Vonsek measurement row 1: copies of untouched v0/v1, one as a chain, feed p1's I and p2's
        # J -> (u, v).
        lines = ["v_mov_b32_e32 v6, v0", "v_mov_b32_e32 v7, v6", "v_mov_b32_e32 v8, v1",
                 "v_interp_p1_f32_e32 v2, v7, attr0.x",
                 "v_interp_p2_f32_e32 v2, v8, attr0.x",
                 "exp mrt0 v2, v2, v2, v2 done vm",
                 "s_endpgm"]
        spv = support.translate_lines("ps_rolk", lines)
        words = support.mrt0_words(spv, inputs={"%in_attr0": ATTR0})
        self.assertEqual(words, [f32_bits(ATTR0[0])] * 4)

    def test_copy_saves_pair_from_later_zero(self):
        # Vonsek measurement row 2: copy v0/v1 out first, then zero them, read through the copies
        # -> (u, v): the copy saved the value, a write to the source after the copy does not
        # disqualify it.
        lines = ["v_mov_b32_e32 v6, v0", "v_mov_b32_e32 v7, v1",
                 "v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                 "v_interp_p1_f32_e32 v2, v6, attr0.x",
                 "v_interp_p2_f32_e32 v2, v7, attr0.x",
                 "exp mrt0 v2, v2, v2, v2 done vm",
                 "s_endpgm"]
        spv = support.translate_lines("ps_rols", lines)
        words = support.mrt0_words(spv, inputs={"%in_attr0": ATTR0})
        self.assertEqual(words, [f32_bits(ATTR0[0])] * 4)

    def test_copy_of_overwritten_source_is_computed(self):
        # Vonsek measurement row 3: zeroing v0/v1 BEFORE the copy means the copy carries the
        # computed (0) value, not the barycentric pair — hardware renders (0,0) = P0 — so the copy
        # no longer qualifies for the guard. The computed-I/J model represents it exactly: the
        # operand loads read the zeros the program wrote, and the weights (0, 0) pick vertex 0.
        # (The corpus's one copy PS copies v1 while it is still unwritten, then overwrites it: the
        # saved-copy case above, which keeps translating untouched by this.)
        spv = support.translate_lines("ps_rolz", ["v_mov_b32_e32 v0, 0", "v_mov_b32_e32 v1, 0",
                                                  "v_mov_b32_e32 v6, v0", "v_mov_b32_e32 v7, v1",
                                                  "v_interp_p1_f32_e32 v2, v6, attr0.x",
                                                  "v_interp_p2_f32_e32 v2, v7, attr0.x",
                                                  "exp mrt0 v2, v2, v2, v2 done vm",
                                                  "s_endpgm"])
        words = support.mrt0_words(spv, inputs=support.varyings(spv, {0: WORDS0}, BARY))
        self.assertEqual(words, [f32_bits(VERTS0[0][0])] * 4)   # weights (0, 0): the provoking vertex
        self.assertIn("PerVertexKHR", spv)

    def test_standard_pairs_and_p0_mov_unchanged(self):
        spv = support.translate_lines("ps_std", ["v_interp_p1_f32_e32 v2, v0, attr0.x",
                                                 "v_interp_p2_f32_e32 v2, v1, attr0.x",
                                                 "v_interp_mov_f32_e32 v3, p0, attr1.x",
                                                 "exp mrt0 v2, v3, v3, v3 done vm",
                                                 "s_endpgm"])
        words = support.mrt0_words(spv, inputs={"%in_attr0": ATTR0, "%in_attr1": (FLAT_ATTR1_X,) * 4})
        self.assertEqual(words, [f32_bits(ATTR0[0]), FLAT_ATTR1_X, FLAT_ATTR1_X, FLAT_ATTR1_X])
        self.assertIn("OpDecorate %in_attr1 Flat", spv)      # mov-only input stays Flat
        self.assertNotIn("OpDecorate %in_attr0 Flat", spv)

    def test_f16_standard_forms_still_translate(self):
        spv = support.translate_lines("ps_f16", ["v_interp_p1ll_f16 v4, v0, attr1.x",
                                                 "v_interp_p2_f16 v4, v1, attr1.x, v4",
                                                 "exp mrt0 v4, v4, v4, v4 done vm",
                                                 "s_endpgm"])
        self.assertIn("OpEntryPoint Fragment", spv)

    def test_guard_uses_the_copy_analysis(self):
        # Mutation check: with interp_copy_sources returning nothing, the copy case no longer takes
        # the guard's centre-interpolated path; it falls through to the computed-I/J model
        # (PerVertexKHR + operand loads), proving the acceptance above really flows through the
        # tracking and not through a vacuous guard. Either path is exact for a real copy.
        lines = ["v_mov_b32_e32 v6, v1",
                 "v_interp_p1_f32_e32 v2, v0, attr0.x",
                 "v_interp_p2_f32_e32 v2, v6, attr0.x",
                 "exp mrt0 v2, v2, v2, v2 done vm",
                 "s_endpgm"]
        self.assertNotIn("PerVertexKHR", support.translate_lines("ps_vc", lines))
        orig = rdna2spirv.interp_copy_sources
        rdna2spirv.interp_copy_sources = lambda insns: {}
        try:
            self.assertIn("PerVertexKHR", support.translate_lines("ps_vc", lines))
        finally:
            rdna2spirv.interp_copy_sources = orig


if __name__ == "__main__":
    unittest.main()
