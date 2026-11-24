"""Tests for tools/pc_checker.py. Run with: python -I tests/test_pc_checker.py"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

import pc_checker as pc  # noqa: E402

XBOX = "Xbox Series X"


class GpuScore(unittest.TestCase):
    def test_exact(self):
        self.assertEqual(pc.gpu_score("NVIDIA GeForce RTX 3070"), 100)

    def test_longest_match_wins(self):
        self.assertEqual(pc.gpu_score("GeForce RTX 3070 Ti"), 108)
        self.assertEqual(pc.gpu_score("AMD Radeon RX 6700 XT"), 100)

    def test_vram_suffix_ignored(self):
        self.assertEqual(pc.gpu_score("NVIDIA GeForce RTX 3070 (8 GB VRAM)"), 100)

    def test_laptop_penalty(self):
        self.assertAlmostEqual(pc.gpu_score("RTX 3070 Ti Laptop GPU"), 108 * 0.8)

    def test_integrated(self):
        self.assertEqual(pc.gpu_score("Intel(R) UHD Graphics 770"), 15)

    def test_unknown(self):
        self.assertIsNone(pc.gpu_score("Some Future Card"))
        self.assertIsNone(pc.gpu_score(""))

    def test_spacing_and_case_ignored(self):
        self.assertEqual(pc.gpu_lookup("rx9070xt"), (195, True))
        self.assertEqual(pc.gpu_lookup("RTX4070TiSuper"), (172, True))

    def test_unlisted_card_is_estimated(self):
        score, exact = pc.gpu_lookup("NVIDIA GeForce RTX 6070")   # newer than the table
        self.assertFalse(exact)
        self.assertAlmostEqual(score, 150 * pc.GEN_GROWTH)
        score, exact = pc.gpu_lookup("Radeon RX 9070 GRE")         # variant of a listed card
        self.assertFalse(exact)
        self.assertLess(score, pc.gpu_score("RX 9070"))

    def test_estimate_keeps_ordering(self):
        self.assertLess(pc.gpu_score("RTX 6060"), pc.gpu_score("RTX 6070"))
        self.assertLess(pc.gpu_score("RTX 6070"), pc.gpu_score("RTX 6080"))


class CpuScore(unittest.TestCase):
    def test_explicit(self):
        self.assertEqual(pc.cpu_score("AMD Ryzen 5 5600"), 100)
        self.assertEqual(pc.cpu_score("Intel(R) Core(TM) i5-12400F"), 102)
        self.assertEqual(pc.cpu_score("AMD Ryzen 7 5800X3D 8-Core"), 140)

    def test_estimated(self):
        self.assertAlmostEqual(pc.cpu_score("Intel Core i7-11700K"), 103 * 1.06)

    def test_ultra(self):
        self.assertEqual(pc.cpu_score("Intel Core Ultra 7 265K"), 178)

    def test_unknown(self):
        self.assertIsNone(pc.cpu_score("Mystery CPU"))
        self.assertIsNone(pc.cpu_score(""))


class Evaluate(unittest.TestCase):
    def test_in_range(self):
        head, _, rows = pc.evaluate(XBOX, "AMD Ryzen 5 5600", "NVIDIA GeForce RTX 3070", 32)
        self.assertEqual(head, "Looks good")
        self.assertTrue(all(r[1] == pc.IN_RANGE for r in rows))

    def test_below(self):
        head, sub, _ = pc.evaluate(XBOX, "Intel Core i3-10100F", "NVIDIA GeForce GTX 1650", 8)
        self.assertEqual(head, "Probably not enough")
        self.assertIn("The CPU, GPU and RAM are weaker than Xbox Series X", sub)

    def test_above(self):
        head, _, rows = pc.evaluate(XBOX, "AMD Ryzen 7 9800X3D", "NVIDIA GeForce RTX 4090", 64)
        self.assertEqual(head, "Looks good")
        self.assertTrue(all(r[1] == pc.ABOVE for r in rows))
        self.assertEqual({pc.STATUS_TEXT[r[1]] for r in rows}, {"Good"})

    def test_unknown_component(self):
        head, sub, rows = pc.evaluate(XBOX, "Mystery CPU", "NVIDIA GeForce RTX 3070", 32)
        self.assertEqual(head, "Can't tell yet")
        self.assertEqual(rows[0][1], pc.UNKNOWN)
        self.assertIn("Mystery CPU", sub)

    def test_ram_bounds(self):
        self.assertEqual(pc._judge_ram(16, 16), pc.IN_RANGE)
        self.assertEqual(pc._judge_ram(8, 16), pc.BELOW)
        self.assertEqual(pc._judge_ram(64, 16), pc.ABOVE)
        self.assertEqual(pc._judge_ram(None, 16), pc.UNKNOWN)

    def test_text_output(self):
        name = "Community 1080p Low estimate"
        text = pc.result_as_text(name, *pc.evaluate(name, "AMD Ryzen 5 5600",
                                                    "NVIDIA GeForce RTX 3060", 16))
        self.assertIn("CPU: Good", text)

    def test_estimated_flag(self):
        _, _, rows = pc.evaluate(XBOX, "AMD Ryzen 5 5600", "NVIDIA GeForce RTX 6070", 32)
        self.assertFalse(rows[0][4])
        self.assertTrue(rows[1][4])


class Api(unittest.TestCase):
    GOOD = ("AMD Ryzen 5 5600", "NVIDIA GeForce RTX 3070", 32)
    WEAK = ("Intel Core i3-10100F", "NVIDIA GeForce GTX 1650", 8)

    def setUp(self):
        self._detect = pc.detect_all
        pc.detect_all = lambda: self.fail("should not detect when everything is given")
        self._tk, self._env = pc.tk, os.environ.pop(pc.SKIP_ENV, None)

    def tearDown(self):
        pc.detect_all, pc.tk = self._detect, self._tk
        os.environ.pop(pc.SKIP_ENV, None)
        if self._env is not None:
            os.environ[pc.SKIP_ENV] = self._env

    def test_verdicts(self):
        c = pc.SpecChecker(XBOX)
        self.assertEqual(c.check(*self.GOOD).verdict, pc.GOOD)
        weak = c.check(*self.WEAK)
        self.assertEqual(weak.verdict, pc.WEAK)
        self.assertEqual(weak.weak_parts, ["CPU", "GPU", "RAM"])
        self.assertEqual(c.check("Mystery", *self.GOOD[1:]).verdict, pc.UNKNOWN)

    def test_as_dict_is_json(self):
        import json
        d = pc.SpecChecker().check(*self.WEAK).as_dict()
        self.assertEqual(json.loads(json.dumps(d))["verdict"], pc.WEAK)

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            pc.SpecChecker("nope")

    def test_warn_never_blocks_good_pc(self):
        c = pc.SpecChecker(XBOX)
        pc.tk = None
        self.assertTrue(c.warn_if_weak(c.check(*self.GOOD)))

    def test_warn_without_tkinter_prints_and_continues(self):
        c = pc.SpecChecker(XBOX)
        pc.tk = None
        self.assertTrue(c.warn_if_weak(c.check(*self.WEAK)))

    def test_skip_env(self):
        os.environ[pc.SKIP_ENV] = "1"
        c = pc.SpecChecker(XBOX)
        self.assertTrue(c.warn_if_weak(c.check(*self.WEAK)))


if __name__ == "__main__":
    unittest.main()