"""Regression tests for SHADER-314: v_readlane_b32 of a uniform VGPR with a tainted lane selector."""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, HERE)

import rdna2spirv  # noqa: E402
from test_wave64_halves import listing, ps, translate_listing  # noqa: E402


class ReadlaneUniformTaintedLaneTests(unittest.TestCase):
    def test_uniform_vgpr_with_tainted_lane_is_tier_b(self):
        asm = listing("ps_readlane_uniform_tainted_lane", ps([
            "v_mov_b32_e32 v5, s4",
            "s_ff1_i32_b64 s6, exec",
            "v_readlane_b32 s5, v5, s6",
            "v_mov_b32_e32 v6, s5",
        ]))
        translate_listing("ps_readlane_uniform_tainted_lane", asm)
        w = rdna2spirv.translate.last_gen.w64g
        self.assertEqual(w["tier"], "B")
        self.assertEqual(w["reasons"], [])
        self.assertIn("readlane-uniform", w["idioms"])

    def test_varying_vgpr_with_tainted_lane_stays_tier_c(self):
        asm = listing("ps_readlane_varying_tainted_lane", ps([
            "v_mbcnt_lo_u32_b32 v5, -1, 0",
            "s_ff1_i32_b64 s6, exec",
            "v_readlane_b32 s5, v5, s6",
            "v_mov_b32_e32 v6, s5",
        ]))
        translate_listing("ps_readlane_varying_tainted_lane", asm)
        w = rdna2spirv.translate.last_gen.w64g
        self.assertEqual(w["tier"], "C")
        self.assertEqual(w["reasons"], ["first-lane"])


if __name__ == "__main__":
    unittest.main()
