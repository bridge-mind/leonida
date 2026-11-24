"""Synthetic checks for tools/interp_hw_probe.py (SHADER-430): the CPU side of the v_interp probe.

The probe itself needs an AMD GPU (tools/hw_oracle). This file tests the exact-model side that the
GPU run is compared against: the Fraction-to-f32/f16 rounding helpers (nearest even, subnormals
kept, infinity on overflow), the bit round trips, the zero comparator, the f32 denormal flush rule,
the LDS attribute-word packing of the two kernel builders, and the CLI's exit-code contract when
no hardware oracle exists. Standard library only; no game files.

    python -I tests/test_interp_hw_probe.py
"""
import contextlib
import io
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction as F
from random import Random
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "interp_hw_probe.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import interp_hw_probe as ihp  # noqa: E402


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def f16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


class RoundingTest(unittest.TestCase):
    def test_exact_values_round_trip(self):
        self.assertEqual(ihp.r32(F(1)), 1.0)
        self.assertEqual(ihp.r32(F(1, 2)), 0.5)
        self.assertEqual(ihp.r16(F(1)), 1.0)

    def test_one_third_rounds_up_to_nearest(self):
        # 1/3 has no exact f32: the larger neighbour is nearer, sign kept.
        self.assertEqual(ihp.r32(F(1, 3)), f32(0x3EAAAAAB))
        self.assertEqual(ihp.r32(-F(1, 3)), -f32(0x3EAAAAAB))
        self.assertEqual(ihp.r16(F(1, 3)), f16(0x3555))

    def test_midpoints_round_to_even(self):
        # a = 1 + 1.5 ulp is the exact midpoint of the 1-ulp-up and 2-ulp-up neighbours, and the
        # 2-ulp neighbour has the even mantissa: that is where the tie lands.
        self.assertEqual(ihp.r32(F(1) + F(1, 2 ** 24)), 1.0)          # midpoint (1.0, 1+2**-23): even keeps 1.0
        self.assertEqual(ihp.r32(F(1) + F(3, 2 ** 24)), f32(0x3F800002))
        self.assertEqual(ihp.r16(F(1) + F(1, 2 ** 11)), 1.0)          # midpoint (1.0, 1+2**-10)

    def test_overflow_gives_signed_infinity(self):
        self.assertEqual(ihp.r32(F(2) ** 128), math.inf)
        self.assertEqual(ihp.r32(-(F(2) ** 128)), -math.inf)
        self.assertEqual(ihp.r16(F(2) ** 16), math.inf)

    def test_subnormals_are_kept(self):
        self.assertEqual(ihp.r32(F(2) ** -149), float.fromhex("0x1p-149"))
        # halfway from 2**-149 to 2**-148 rounds to even: the smaller value.
        self.assertEqual(ihp.r32(F(2) ** -150), 0.0)
        self.assertEqual(ihp.r16(F(1, 3) * F(2) ** -14), f16(0x0155))

    def test_zero(self):
        self.assertEqual(ihp.r32(F(0)), 0.0)


class ComparatorTest(unittest.TestCase):
    def test_same_takes_bit_patterns_with_sign_aware_zero(self):
        self.assertTrue(ihp.same(ihp.f32w(1.5), 1.5))
        self.assertFalse(ihp.same(ihp.f32w(1.5001), 1.5))
        self.assertTrue(ihp.same(ihp.f32w(0.0), 0.0))                 # +0.0 bits vs +0.0
        self.assertFalse(ihp.same(ihp.f32w(0.0), -0.0))               # +0.0 bits vs -0.0
        self.assertTrue(ihp.same(ihp.f16w(0), 0.0, bits=16))
        self.assertTrue(ihp.same(ihp.f16w(-0.0), -0.0, bits=16))   # -0.0 encodes as 0x8000

    def test_finite_helper(self):
        self.assertTrue(ihp.finite(1.0, -0.0, 2.0 ** -130))
        self.assertFalse(ihp.finite(1.0, math.inf))
        self.assertFalse(ihp.finite(0.0, float("nan")))


class FlushTest(unittest.TestCase):
    def test_flush32_only_tiny_nonzero_values_become_signed_zero(self):
        self.assertEqual(ihp.flush32(float.fromhex("0x1p-130"), True), 0.0)
        self.assertEqual(math.copysign(1, ihp.flush32(-float.fromhex("0x1p-130"), True)), -1.0)
        self.assertEqual(ihp.flush32(1.0, True), 1.0)                 # a normal value stays
        self.assertEqual(ihp.flush32(float.fromhex("0x1p-130"), False),
                         float.fromhex("0x1p-130"))                   # flush off keeps the value
        self.assertEqual(ihp.flush32(0.0, True), 0.0)


class KernelBuildersTest(unittest.TestCase):
    def test_gen_word_never_makes_f16_nan_or_inf_halves(self):
        rng = Random(11)
        for _ in range(500):
            word = ihp.gen_word(rng, packed=True)
            lo, hi = word & 0xFFFF, (word >> 16) & 0xFFFF
            self.assertNotEqual(lo & 0x7C00, 0x7C00, hex(word))
            self.assertNotEqual(hi & 0x7C00, 0x7C00, hex(word))

    def test_random_kernel_is_deterministic_per_seed(self):
        a = ihp.random_kernel(Random(7), {"denorm32": 3})[0].lines
        b = ihp.random_kernel(Random(7), {"denorm32": 3})[0].lines
        self.assertEqual(a, b)                                        # the instruction listing

    def test_random_kernel_layout(self):
        kern, info = ihp.random_kernel(Random(3), {"denorm32": 3})
        stores = [l for l in kern.lines if l.startswith("global_store_dword")]
        writes = [l for l in kern.lines if l.startswith("ds_write_b32")]
        self.assertEqual(len(stores), 224)                            # 8 attrs x 4 channels x 7 outputs
        self.assertEqual(len(writes), 160)                            # 3 attribute words each + I/J per lane
        self.assertIn("s_mov_b32 m0, 0", kern.lines)
        self.assertIn("v_interp_p1_f32 v2, v20, attr0.x", kern.lines)
        self.assertEqual(len(info), 4)                                # params, kinds, ij, per-channel tags

    def test_tie_kernel_layout_is_two_stores_per_form(self):
        kern, (groups, lanes, nout) = ihp.tie_kernel({"denorm32": 3})
        stores = [l for l in kern.lines if "global_store_dword" in l]
        p1ll = [l for l in kern.lines if "v_interp_p1ll_f16" in l]
        p2 = [l for l in kern.lines if "v_interp_p2_f16" in l]
        self.assertEqual(nout, 4 * len(groups))
        self.assertEqual(len(stores), nout)                           # p1ll and p2 stored per form per group
        self.assertEqual(len(p1ll), 2 * len(groups))                  # normal and ` high`
        self.assertEqual(len(p2), 2 * len(groups))


class TieCheckTest(unittest.TestCase):
    def test_a_value_that_rounds_differently_once_versus_twice_is_tallied(self):
        # q = 1 + 2**-11 + 2**-24: f16 directly rounds it UP off the tie, while the f32 step ties
        # to even and the f16 re-round then ties DOWN to 1.0.
        h20 = F(2) ** -14
        j = F(8) + F(1, 8192)                      # h20 * j = 2**-11 + 2**-24
        groups = [((1.0, 1.0, 2.0 ** -14), [(1.0, float(j))])]
        lanes = [(0, 1.0, float(j))]
        nout = 4
        q = F(1) + h20 * j
        got = 0x3C01                                # the direct-once f16 result
        res = [ihp.f32w(1.0), got, 0, 0]
        tally, bad = {}, []
        ihp.check_ties(res, (groups, lanes, nout), tally, bad)
        key = ("tie", "p2_f16 rounds once (not f32, then f16)")
        self.assertIn(key, tally)
        self.assertEqual(tally[key][0], 1)
        self.assertEqual(tally[key][1], 0)
        self.assertEqual(bad, [])
        # and the sanity of the case itself: direct f16 differs from the via-f32 double round
        self.assertNotEqual(ihp.f16w(ihp.r16(q)), ihp.f16w(ihp.r16(F(ihp.r32(q)))))

    def test_an_equal_result_is_not_tallied(self):
        h20 = F(1)
        groups = [((1.0, 1.0, 1.0), [(1.0, 0.5)])]
        lanes = [(0, 1.0, 0.5)]
        res = [ihp.f32w(1.0), 0x3C00, 0, 0]         # got == both roundings == 1.0's f16
        tally, bad = {}, []
        ihp.check_ties(res, (groups, lanes, 4), tally, bad)
        self.assertNotIn(("tie", "p2_f16 rounds once (not f32, then f16)"), tally)
        self.assertEqual(bad, [])

    def test_a_word_that_matches_the_twice_rounded_result_is_reported_bad(self):
        # Same q as the case above, but the stored word is the via-f32 one (1.0, 0x3C00) instead of
        # the direct f16 (0x3C01). The model expects the direct rounding, so this has to land in the
        # differ column and in bad, not in the ok column the previous test asserts.
        h20 = F(2) ** -14
        j = F(8) + F(1, 8192)
        groups = [((1.0, 1.0, 2.0 ** -14), [(1.0, float(j))])]
        lanes = [(0, 1.0, float(j))]
        q = F(1) + h20 * j
        once, twice = ihp.f16w(ihp.r16(q)), ihp.f16w(ihp.r16(F(ihp.r32(q))))
        self.assertEqual((once, twice), (0x3C01, 0x3C00))     # the two really do disagree here
        res = [ihp.f32w(1.0), twice, 0, 0]
        tally, bad = {}, []
        ihp.check_ties(res, (groups, lanes, 4), tally, bad)
        key = ("tie", "p2_f16 rounds once (not f32, then f16)")
        self.assertEqual(tally[key], [0, 1])
        self.assertEqual(len(bad), 1)
        self.assertIn("tie: lane 0", bad[0])
        self.assertIn(f"once {once:#06x} twice {twice:#06x}", bad[0])


class RandomCheckTest(unittest.TestCase):
    """check_random against hand-computed words: one (attr, component) pair, I = J = 1.0.

    p0 = p10 = p20 = the f32 1.0 (0x3F800000), and I = J = the f32 1.0 too, so every fma is exact.
    The halves of that word are f16 0.0000 (low) and f16 0x3F80 = 1.875 (high): the f32 exponent
    bits land in the f16 mantissa, which is exactly the kind of junk the split halves see.
      o[0] p1_f32      = 1*1 + 1 = 2.0                   -> 0x40000000
      o[1] p2_f32      = 1*1 + 2 = 3.0                   -> 0x40400000
      o[2] p1ll low    = 0*1 + 0 = 0.0                   -> 0x00000000
      o[3] p2_f16 low  = 0*1 + 0 = 0.0, canary on top    -> 0xDEAD0000
      o[4] p1ll high   = 1.875*1 + 1.875 = 3.75          -> 0x40700000
      o[5] p2_f16 high = 1.875*1 + 3.75 = 5.625 (f16 0x45A0) -> 0xDEAD45A0
      o[6] mov p0                                       -> 0x3F800000
    """

    WORDS = [0x40000000, 0x40400000, 0x00000000, 0xDEAD0000, 0x40700000, 0xDEAD45A0, 0x3F800000]

    def run_check(self, words):
        params = {(0, 0): (0x3F800000, 0x3F800000, 0x3F800000)}
        kinds = {(0, 0): "f32"}
        ij = [(ihp.f32w(1.0), ihp.f32w(1.0))]
        info = (params, kinds, ij, 4 * ihp.NOUT_PER)
        tally, bad = {}, []
        ihp.check_random(list(words), info, False, tally, bad)
        return tally, bad

    def test_hand_computed_words_all_match(self):
        tally, bad = self.run_check(self.WORDS)
        self.assertEqual(bad, [])
        self.assertEqual(len(tally), 9)          # mov, both fma pairs, both half pairs, both canaries
        for name, counts in tally.items():
            self.assertEqual(counts, [1, 0], name)

    def test_a_wrong_p1_f32_word_is_a_mismatch(self):
        words = list(self.WORDS)
        words[0] = 0x3FFFFFFF                       # one f32 ulp below 2.0
        tally, bad = self.run_check(words)
        self.assertEqual(tally[("f32", "p1_f32 = fma(P10, I, P0)")], [0, 1])
        self.assertEqual(len(bad), 1)
        self.assertIn("p1_f32 = fma(P10, I, P0)", bad[0])
        self.assertIn("lane 0 attr0.x", bad[0])

    def test_a_p2_f16_that_keeps_its_dead_high_half_is_a_mismatch(self):
        words = list(self.WORDS)
        words[5] = 0x000045A0                       # right low half, canary gone
        tally, bad = self.run_check(words)
        self.assertEqual(tally[("f32", "p2_f16 high keeps vdst[31:16]")], [0, 1])
        self.assertEqual(tally[("f32", "p2_f16 high = fma(h20, J, p1ll) in f16")], [1, 0])
        self.assertTrue(any("keeps vdst[31:16]" in b for b in bad))


class CliTest(unittest.TestCase):
    def run_tool(self, *argv):
        with mock.patch("builtins.print"):
            try:
                return ihp.main(list(argv))
            except SystemExit as e:
                return e.code

    def test_bad_seeds_are_a_usage_error(self):
        with self.assertRaises(SystemExit) as ctx:
            ihp.main(["--seeds", "1,x,3"])
        self.assertEqual(ctx.exception.code, 2)

    def test_denorm32_choice_is_enforced(self):
        with self.assertRaises(SystemExit) as ctx:
            ihp.main(["--denorm32", "1"])
        self.assertEqual(ctx.exception.code, 2)

    def test_no_oracle_is_77(self):
        # Patch the oracle open instead of hoping this machine has no AMD GPU: on a box with a
        # working gfx1036 target the unpatched call would run the real probe and return 0.
        err = io.StringIO()
        with mock.patch.object(ihp.runner, "open_oracle",
                               side_effect=ihp.runner.Unavailable("no AMD GPU on this box")):
            with contextlib.redirect_stderr(err):
                rc = ihp.main(["--arch", "gfx1036"])
        self.assertEqual(rc, 77)
        self.assertIn("no AMD GPU on this box", err.getvalue())


if __name__ == "__main__":
    unittest.main()
