"""tools/fma_mix_probe.py: the inputs tell single from double f16 rounding of v_fma_mixlo/hi_f16, and each side is
judged right. Standard library only; no GPU, no LLVM, no game files.

The words below are worked out by hand, not taken from the tool's own model (f16: ulp 2^-10 on [1, 2), ties at the odd
multiples of 2^-11; f32: ulp 2^-23 on [1, 2)):

  A  exact 1 + 2^-11 + 2^-40: once -> above the tie -> 0x3c01; f32 drops 2^-40 (< 2^-24), the tie goes to even 0x3c00
  B  exact 1 + 3 * 2^-11 - 2^-40: once -> below the tie -> 0x3c01; f32 lands on the tie, even is 0x3c02
  C  A negated: 0xbc01 / 0xbc00
  D  exact 2^-25 + 2^-60: once -> above half the smallest subnormal -> 0x0001; f32 keeps 2^-25 exactly, the tie
     between 0 and 2^-24 goes to even 0x0000
  E  exact 65520 - 2^-20: once -> below the overflow tie -> 65504 0x7bff; f32 (ulp 2^-8) gives 65520, which rounds to
     inf 0x7c00
  F  (1 + 2^-12)^2 = 1 + 2^-11 + 2^-24: once -> 0x3c01; f32 ties (2^-24 is half its ulp) to even 1 + 2^-11 -> 0x3c00
  G  (1 + 2^-7)(1 - 2^-7) 2^-11 + (1 + 2^-10) = 1 + 3 * 2^-11 - 2^-25: once 0x3c01; f32 ties onto 1 + 3 * 2^-11 -> 0x3c02
  H  2.5 = 0x4100 either way
"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests", "semantic"))

import fma_mix_probe as probe  # noqa: E402
import spirv_mini_interp  # noqa: E402

BY_HAND = {   # name: (single, double)
    "A": (0x3C01, 0x3C00), "B": (0x3C01, 0x3C02), "C": (0xBC01, 0xBC00), "D": (0x0001, 0x0000),
    "E": (0x7BFF, 0x7C00), "F": (0x3C01, 0x3C00), "G": (0x3C01, 0x3C02), "H": (0x4100, 0x4100),
}
VSHARP = [0, 0, 0x1000, 0x31016FAC]       # s[0:3]: base 0, 4 KiB, as in tests/semantic/test_lds_atomics_values.py


def word(case, f16):
    """The destination word a correct op leaves: the f16 in its half, the sentinel in the other."""
    s = probe.SENTINEL
    return (s & 0xFFFF) | f16 << 16 if case.hi else (s & 0xFFFF0000) | f16


class Inputs(unittest.TestCase):
    def test_model_matches_hand_derivation(self):
        self.assertEqual(sorted(c.name for c in probe.CASES), sorted(BY_HAND))
        for case in probe.CASES:
            with self.subTest(case=case.name):
                want = probe.expected(case)
                self.assertEqual((want[probe.SINGLE], want[probe.DOUBLE]), BY_HAND[case.name])

    def test_every_case_but_the_control_discriminates(self):
        for case in probe.CASES:
            single, double = BY_HAND[case.name]
            self.assertEqual(single == double, case.what.startswith("control"), case.name)

    def test_both_ops_and_both_operand_kinds_are_covered(self):
        self.assertEqual({c.op for c in probe.CASES}, {"v_fma_mixlo_f16", "v_fma_mixhi_f16"})
        self.assertTrue(any(c.f16 for c in probe.CASES) and any(not c.f16 for c in probe.CASES))

    def test_operand_refuses_what_the_model_does_not_cover(self):
        for w, f16 in ((0x7F800000, False), (0x00000001, False), (0x7C00, True)):
            with self.assertRaises(ValueError):
                probe.operand(w, f16)


class Translator(unittest.TestCase):
    def test_translator_rounds_once(self):
        # docs/design.md, Known approximations: v_fma_mix* rounds once straight to f16 (measured on RDNA2 hardware).
        got = probe.run_translator()
        rule, _ = probe.verdict(list(zip(probe.CASES, got)))
        self.assertEqual(rule, probe.SINGLE)
        for case, w in zip(probe.CASES, got):
            self.assertEqual(w, word(case, BY_HAND[case.name][0]), case.name)

    def test_compute_listing_gives_the_pixel_shader_words(self):
        # The Vulkan side's shader (operands from user SGPRs, one store per case) on the CPU interpreter, with the
        # V# and the operand words as push constants the way vk_compute_test passes s0..s31.
        with tempfile.TemporaryDirectory() as work:
            spv = probe.translate_cs(work)
        pc = VSHARP + probe.sgpr_words()
        mem = spirv_mini_interp.run_workgroup(
            spv, 1, 32, inputs_for=lambda i: {"%wg_id": (0, 0, 0), "%local_id": (i, 0, 0), "%sg_lane": i},
            push_constants=pc + [0] * (32 - len(pc)), memory={})
        self.assertEqual([mem.get(4 * k) for k in range(len(probe.CASES))], probe.run_translator())

    def test_compute_header_declares_every_operand_sgpr(self):
        rsrc2 = probe.cs_rsrc2(probe.CASES)
        self.assertEqual((rsrc2 >> 1) & 31, 4 + 3 * len(probe.CASES))
        self.assertLessEqual(len(probe.sgpr_words()), 28)       # VKC_SGPRS takes s4..s31


class Verdict(unittest.TestCase):
    def results(self, rule, **override):
        out = []
        for case in probe.CASES:
            single, double = BY_HAND[case.name]
            f16 = single if rule == probe.SINGLE else double
            out.append((case, override.get(case.name, word(case, f16))))
        return out

    def test_each_rule_is_recognised(self):
        for rule in (probe.SINGLE, probe.DOUBLE):
            self.assertEqual(probe.verdict(self.results(rule))[0], rule)

    def test_a_mix_of_rules_is_neither(self):
        a = probe.CASES[0]
        self.assertIsNone(probe.verdict(self.results(probe.DOUBLE, A=word(a, BY_HAND["A"][0])))[0])

    def test_a_third_value_is_neither(self):
        self.assertIsNone(probe.verdict(self.results(probe.SINGLE, E=word(probe.CASES[4], 0x7BFE)))[0])

    def test_a_wrong_control_is_neither(self):
        self.assertIsNone(probe.verdict(self.results(probe.SINGLE, H=word(probe.CASES[7], 0x4101)))[0])

    def test_the_other_half_must_be_kept(self):
        b = probe.CASES[1]                                   # mixhi: the low half keeps the sentinel's
        self.assertIsNone(probe.verdict(self.results(probe.DOUBLE, B=BY_HAND["B"][1] << 16))[0])
        self.assertEqual(probe.half(b, word(b, 0x3C02)), 0x3C02)

    def test_a_missing_result_is_neither(self):
        rule, lines = probe.verdict(self.results(probe.SINGLE, D=None))
        self.assertIsNone(rule)
        self.assertIn("missing", lines[3])


class HardwareKernel(unittest.TestCase):
    def test_kernel_body(self):
        g = next(c for c in probe.CASES if c.f16)
        lines = probe.hw_lines(g)
        self.assertIn(f"{g.op} v8, v1, v2, v3 op_sel_hi:[1,1,1]", lines)
        self.assertEqual(lines[-2:], ["global_store_b32 v20, v8, s[2:3]", "s_endpgm"])

    def test_gfx10_kernel_text_sets_the_float_mode(self):
        from hw_oracle import asm
        text = asm.raw_kernel_text("k0", probe.hw_lines(probe.CASES[0]), "gfx1030", mode=probe.HW_MODE)
        self.assertIn("global_store_dword v20, v8, s[2:3]", text)        # GFX10 spelling
        for directive in (".amdhsa_float_denorm_mode_16_64 3", ".amdhsa_float_round_mode_16_64 0",
                          ".amdhsa_ieee_mode 0"):
            self.assertIn(directive, text)


class Command(unittest.TestCase):
    def test_translator_only_run_exits_0(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = probe.main([])
        self.assertEqual(rc, 0)
        self.assertIn("translator: follows single rounding", buf.getvalue())

    def test_unknown_side_and_missing_build_are_usage_errors(self):
        import contextlib
        import io
        for argv in (["--oracle", "gpu"], ["--oracle", "vulkan"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    probe.main(argv)
                self.assertEqual(cm.exception.code, 2)

    def test_vulkan_side_without_a_build_is_unavailable(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as empty, contextlib.redirect_stdout(io.StringIO()) as buf:
            rc = probe.main(["--oracle", "vulkan", "--build", empty])
        self.assertEqual(rc, 77)
        self.assertIn("vk_compute_test is not built", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
