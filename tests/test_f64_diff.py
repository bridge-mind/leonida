"""CLI and reference checks for tools/f64_diff.py; standard library only."""
import contextlib
import io
import os
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "f64_diff.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import f64_diff as fd


class F64DiffTests(unittest.TestCase):
    def command(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], cwd=ROOT,
                              capture_output=True, text=True, timeout=60)

    def test_cli_self_test(self):
        self.assertEqual(self.command("--self-test").returncode, 0)

    def test_lists_47_supported_forms_and_exclusions(self):
        result = self.command("--list-ops")
        listed = [line for line in result.stdout.splitlines() if line.startswith("v_")]
        self.assertEqual(result.returncode, 0)
        self.assertEqual(len(listed), 47)
        self.assertIn("v_div_scale_f64", result.stdout)

    def test_exit_2_for_invalid_cli(self):
        self.assertEqual(self.command("--ops", "v_not_real_f64").returncode, 2)
        self.assertEqual(self.command("--programs", "-1").returncode, 2)

    def test_reference_rounds_all_four_binary64_modes(self):
        half_ulp = Fraction(1, 1 << 53)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 0), 0x3FF0000000000000)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 1), 0x3FF0000000000001)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 2), 0x3FF0000000000000)
        self.assertEqual(fd.round_fraction(Fraction(1) + half_ulp, 3), 0x3FF0000000000000)
        self.assertEqual(fd.round_fraction(-(Fraction(1) + half_ulp), 1), 0xBFF0000000000000)
        self.assertEqual(fd.round_fraction(-(Fraction(1) + half_ulp), 2), 0xBFF0000000000001)

    def test_nan_tolerance_preserves_integer_low_word_after_partial_write(self):
        expected = [0, 0x7FF80000, 0, 0]
        actual = [0, 0xFFFFFFFF, 0, 0]
        self.assertTrue(fd.equivalent(expected, actual, ("int", "f64")))
        self.assertFalse(fd.equivalent([1, *expected[1:]], actual, ("int", "f64")))
        self.assertFalse(fd.equivalent(expected, actual, ("int", "int")))

    def test_exit_77_when_requested_oracle_is_unavailable(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=os.path.join(ROOT, "tools", "no-such-clang.exe"))
        hw = subprocess.run([sys.executable, "-I", TOOL, "--oracle", "hw", "--programs", "3"], cwd=ROOT,
                            capture_output=True, text=True, timeout=120, env=env)
        self.assertEqual(hw.returncode, 77, hw.stdout + hw.stderr)
        self.assertIn("hardware oracle unavailable", hw.stderr)
        self.assertEqual(self.command("--oracle", "gpu", "--exe", "missing.exe").returncode, 77)
        self.assertEqual(self.command("--oracle", "gpu", "--exe", "missing.exe", "--ops", "not_f64").returncode, 2)

    def test_skips_are_counted_with_reason(self):
        result = self.command("--ops", "v_cvt_f32_f64", "--programs", "0")
        self.assertEqual(result.returncode, 0)
        self.assertIn("1 total", result.stdout)
        self.assertIn("OpFConvert", result.stdout)

    def test_sqrt_preserves_negative_zero(self):
        self.assertEqual(fd.ref_program(["v_sqrt_f64 v[8:9], -0.0"]), [0, 0x80000000, 0, 0])

    def test_rsq_subnormal_is_an_explicit_skip(self):
        lines = ["v_mov_b32 v12, 0x00000001", "v_mov_b32 v13, 0x00000000",
                 "v_rsq_f64 v[8:9], v[12:13]"]
        with tempfile.TemporaryDirectory() as tmp:
            status, reason, _, _ = fd.run_one(lines, "ps_f64_subnormal", tmp)
        self.assertEqual(status, "skip")
        self.assertIn("subnormal", reason)
        self.assertIn("FTZ", reason)

    def test_gpu_rsq_precision_is_counted_as_unjudged(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, reason, _, actual = fd.run_one(["v_rsq_f64 v[8:9], 3.25"],
                                                   "ps_f64_gpu_rsq_precision", tmp, "gpu", {"exe": "unused"})
        self.assertEqual(status, "skip")
        self.assertIn("implementation-defined", reason)
        self.assertIsNone(actual)

    def test_div_fixup_nan_s0_is_skipped_by_default_and_runs_with_known(self):
        setup, body, _, _ = fd.make_case(random.Random(1), "v_div_fixup_f64", known_case="div-fixup-nan-s0")
        lines = setup + body
        with tempfile.TemporaryDirectory() as tmp:
            skipped = fd.run_one(lines, "ps_f64_fixup_default", tmp)
            self.assertEqual(skipped[0], "skip")
            self.assertIn("div-fixup-nan-s0", skipped[1])
            known = fd.run_one(lines, "ps_f64_fixup_known", tmp, include_known=True)
        # the translator now gives the ISA's signed infinity for NaN(S0) too (#4536 was written when it kept the NaN)
        self.assertEqual(known[0], "match")

    def test_compare_reference_does_not_write_v8_without_cndmask(self):
        for op in ("v_cmp_eq_f64_e64 s20, v[12:13], 0.0",
                   "v_cmp_class_f64_e64 s20, v[12:13], 0x3ff"):
            result = fd.ref_program(["v_mov_b32 v8, 0x12345678", "v_mov_b32 v12, 0x00000000",
                                     "v_mov_b32 v13, 0x3ff00000", op])
            self.assertEqual(result[0], 0x12345678)

    def test_shrink_reuses_prepared_gpu_context(self):
        context = {"exe": "fake"}
        observed = []
        def fake_run_one(lines, name, tmp, oracle, gpu_ctx, include_known=False):
            observed.append(gpu_ctx)
            return ("mismatch" if len(lines) > 1 else "match", "", [], [])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(fd, "prepare_gpu", side_effect=AssertionError("GPU setup repeated")), \
                mock.patch.object(fd, "run_one", side_effect=fake_run_one):
            reduced = fd.shrink(["first", "second"], [], "gpu", "fake.exe", context, tmp)
        self.assertEqual(reduced, ["first", "second"])
        self.assertTrue(observed)
        self.assertTrue(all(item is context for item in observed))

    def test_known_divergences_only_reinserted_with_known(self):
        normal = self.command("--ops", "v_add_f64", "--programs", "0")
        known = self.command("--ops", "v_add_f64", "--programs", "0", "--no-sweep", "--known", "--all-failures")
        self.assertNotIn("KNOWN_DIVERGENCE", normal.stdout)
        # the known NaN(S0) case runs and, since the translator follows the ISA there, matches
        self.assertEqual(known.returncode, 0, known.stdout)
        self.assertNotIn("MISMATCH:", known.stdout)

    def test_sabotaged_translator_mismatch_and_shrinking(self):
        lines = ["v_mov_b32 v12, 0x00000000", "v_mov_b32 v13, 0x3ff80000",
                 "v_trunc_f64 v[8:9], v[12:13]"]
        original = fd.translate
        def planted_bug(listing, name, temp):
            out = original(listing, name, temp)
            if any(line.startswith("v_trunc_f64 ") for line in listing): out[0] ^= 1
            return out
        try:
            fd.translate = planted_bug
            with tempfile.TemporaryDirectory() as tmp:
                status, _, expected, actual = fd.run_one(lines, "ps_f64_negative_control", tmp)
                self.assertEqual(status, "mismatch")
                self.assertNotEqual(expected, actual)
                reduced = fd.shrink(lines, expected)
                self.assertEqual(len(reduced), 1)
                self.assertTrue(reduced[0].startswith("v_trunc_f64 "))
        finally:
            fd.translate = original

    # ---- the reference functions, checked against hand-written tables (no shared code with the reference) ----

    ONE, TWO, THREE = 0x3FF0000000000000, 0x4000000000000000, 0x4008000000000000
    NEG1, NEG2, NEG3 = 0xBFF0000000000000, 0xC000000000000000, 0xC008000000000000
    PZ, NZ = 0, 0x8000000000000000
    PINF, NINF = 0x7FF0000000000000, 0xFFF0000000000000
    QNAN, SNAN = 0x7FF8000000000000, 0x7FF0000000000001

    def test_compare_table_all_16_conditions(self):
        # Which of the four relations (lt, eq, gt, un) make each condition true: the ISA's definitions, written out.
        true_for = {
            "f": set(), "lt": {"lt"}, "eq": {"eq"}, "le": {"lt", "eq"}, "gt": {"gt"}, "lg": {"lt", "gt"},
            "ge": {"gt", "eq"}, "o": {"lt", "eq", "gt"}, "u": {"un"}, "nge": {"lt", "un"}, "nlg": {"eq", "un"},
            "ngt": {"lt", "eq", "un"}, "nle": {"gt", "un"}, "neq": {"lt", "gt", "un"}, "nlt": {"gt", "eq", "un"},
            "tru": {"lt", "eq", "gt", "un"},
        }
        self.assertEqual(set(true_for), set(fd.CONDS))
        self.assertEqual(len(fd.CONDS), 16)
        pairs = [(self.ONE, self.TWO, "lt"), (self.TWO, self.ONE, "gt"), (self.ONE, self.ONE, "eq"),
                 (self.NEG1, self.ONE, "lt"), (self.ONE, self.NEG1, "gt"), (self.NEG2, self.NEG2, "eq"),
                 (self.PZ, self.NZ, "eq"), (self.NZ, self.PZ, "eq"), (self.PZ, self.PZ, "eq"), (self.NZ, self.NZ, "eq"),
                 (0x8000000000000001, self.PZ, "lt"), (self.PZ, 1, "lt"), (1, 0x8000000000000001, "gt"),
                 (self.PINF, self.PINF, "eq"), (self.NINF, self.NINF, "eq"), (self.NINF, self.PINF, "lt"),
                 (self.PINF, self.NINF, "gt"), (self.PINF, self.ONE, "gt"), (self.NINF, self.ONE, "lt"),
                 (self.ONE, self.PINF, "lt"), (self.QNAN, self.ONE, "un"), (self.ONE, self.QNAN, "un"),
                 (self.QNAN, self.QNAN, "un"), (self.SNAN, self.ONE, "un"), (self.ONE, self.SNAN, "un"),
                 (self.QNAN, self.PINF, "un"), (self.NINF, self.QNAN, "un"), (self.QNAN, self.PZ, "un")]
        for cond, relations in true_for.items():
            for a, b, relation in pairs:
                with self.subTest(cond=cond, a=hex(a), b=hex(b)):
                    self.assertEqual(fd.compare(cond, a, b), int(relation in relations))

    def test_class_table_all_ten_classes_and_boundaries(self):
        table = [
            (0x7FF0000000000001, 0), (0xFFF0000000000001, 0), (0x7FF7FFFFFFFFFFFF, 0),      # signalling NaN, any sign
            (0x7FF8000000000000, 1), (0xFFF8000000000000, 1), (0xFFFFFFFFFFFFFFFF, 1),      # quiet NaN, any payload
            (self.NINF, 2), (self.NEG1, 3), (0xFFEFFFFFFFFFFFFF, 3), (0x8010000000000000, 3),
            (0x800FFFFFFFFFFFFF, 4), (0x8000000000000001, 4), (self.NZ, 5), (self.PZ, 6),
            (1, 7), (0x000FFFFFFFFFFFFF, 7), (0x0010000000000000, 8), (self.ONE, 8),
            (0x7FEFFFFFFFFFFFFF, 8), (self.PINF, 9),
        ]
        for bits, want in table:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(fd.class_index(bits), want)
        self.assertEqual({want for _, want in table}, set(range(10)))

    def test_div_fixup_table(self):
        one, two, three = self.ONE, self.TWO, self.THREE
        e = lambda exp: exp << 52  # a biased exponent field with a zero fraction
        qn2 = 0x7FF8000000000012
        table = [
            ("normal", three, one, one, three),
            ("sign_out_negative_from_s1", three, self.NEG2, one, three | self.NZ),
            ("sign_out_positive_both_negative", self.NEG3, self.NEG2, self.NEG1, three),
            ("sign_out_clears_s0_sign", self.NEG3, one, one, three),
            ("nan_s2_quiets_s2", one, one, self.SNAN, 0x7FF8000000000001),
            ("nan_s2_beats_nan_s1", one, self.SNAN, qn2, qn2),
            ("nan_s1_quiets_s1", one, self.SNAN, one, 0x7FF8000000000001),
            ("nan_s2_beats_zero_over_zero", one, self.PZ, qn2, qn2),
            ("zero_over_zero", one, self.PZ, self.NZ, 0xFFF8000000000000),
            ("inf_over_inf", one, self.PINF, self.NINF, 0xFFF8000000000000),
            ("x_over_inf_negative_zero", one, self.NINF, one, self.NZ),
            ("x_over_inf_positive_zero", one, self.PINF, one, self.PZ),
            ("zero_over_y", one, self.NEG2, self.NZ, self.PZ),
            ("x_over_zero", one, self.NZ, one, self.NINF),
            ("inf_over_y", one, self.NEG2, self.NINF, self.PINF),
            ("nan_s0_to_signed_inf_isa_rule", self.QNAN, one, one, self.PINF),
            ("nan_s0_to_signed_inf_negative", self.QNAN, self.NEG2, one, self.NINF),
            ("underflow_edge_still_passes_s0", three, e(1075), e(1), three),
            ("underflow_to_zero", three, e(1076), e(1), self.PZ),
            ("underflow_beats_nan_s0", self.QNAN, e(1076), e(1), self.PZ),
            ("overflow_edge_still_passes_s0", three, e(1), e(1024), three),
            ("overflow_to_inf", three, e(1), e(1025), self.PINF),
        ]
        for name, s0, s1, s2, want in table:
            with self.subTest(case=name):
                self.assertEqual(fd.fixup(s0, s1, s2), want)

    def test_frexp_table(self):
        half = 0x3FE0000000000000
        table = [
            (self.ONE, half, 1), (half, half, 0), (self.THREE, 0x3FE8000000000000, 2), (self.NEG2, self.NZ | half, 2),
            (0xC020000000000000, self.NZ | half, 4),                                      # -8 = -0.5 * 2**4
            (1, half, -1073), (3, 0x3FE8000000000000, -1072),                              # smallest subnormals
            (0x000FFFFFFFFFFFFF, 0x3FEFFFFFFFFFFFFE, -1022),
            (0x0010000000000000, half, -1021), (0x7FEFFFFFFFFFFFFF, 0x3FEFFFFFFFFFFFFF, 1024),
            (self.PZ, self.PZ, 0), (self.NZ, self.NZ, 0), (self.PINF, self.PINF, 0), (self.NINF, self.NINF, 0),
            (self.QNAN, self.QNAN, 0), (self.SNAN, self.SNAN, 0),
        ]
        for bits, mant, exp in table:
            with self.subTest(bits=hex(bits)):
                self.assertEqual(fd.frexp(bits), (mant, exp))

    def test_add_bits_nan_inf_and_zero_table(self):
        for a, b in ((self.QNAN, self.ONE), (self.ONE, self.QNAN), (self.SNAN, self.ONE), (self.QNAN, self.QNAN),
                     (self.PINF, self.NINF), (self.NINF, self.PINF), (self.QNAN, self.PINF)):
            for mode in range(4):
                with self.subTest(a=hex(a), b=hex(b), mode=mode):
                    self.assertTrue(fd.is_nan(fd.add_bits(a, b, mode)))
        for a, b, want in ((self.PINF, self.ONE, self.PINF), (self.ONE, self.NINF, self.NINF),
                           (self.PINF, self.PINF, self.PINF), (self.NINF, self.NINF, self.NINF)):
            for mode in range(4):
                with self.subTest(a=hex(a), b=hex(b), mode=mode):
                    self.assertEqual(fd.add_bits(a, b, mode), want)

    def test_add_bits_exact_zero_sign_in_all_four_modes(self):
        # IEEE 754-2019 6.3. Modes: 0 nearest even, 1 toward +inf, 2 toward -inf, 3 toward zero.
        tiny = 1  # the smallest subnormal, so x + (-x) is an exact non-zero cancellation
        cases = [
            ("+0 + +0", self.PZ, self.PZ, (self.PZ, self.PZ, self.PZ, self.PZ)),
            ("-0 + -0", self.NZ, self.NZ, (self.NZ, self.NZ, self.NZ, self.NZ)),
            ("+0 + -0", self.PZ, self.NZ, (self.PZ, self.PZ, self.NZ, self.PZ)),
            ("-0 + +0", self.NZ, self.PZ, (self.PZ, self.PZ, self.NZ, self.PZ)),
            ("1 + -1", self.ONE, self.NEG1, (self.PZ, self.PZ, self.NZ, self.PZ)),
            ("-1 + 1", self.NEG1, self.ONE, (self.PZ, self.PZ, self.NZ, self.PZ)),
            ("tiny + -tiny", tiny, tiny | self.NZ, (self.PZ, self.PZ, self.NZ, self.PZ)),
        ]
        for name, a, b, wants in cases:
            for mode, want in enumerate(wants):
                with self.subTest(case=name, mode=mode):
                    self.assertEqual(fd.add_bits(a, b, mode), want)
        # A non-zero result keeps its own sign whatever the mode.
        self.assertEqual(fd.add_bits(self.ONE, self.ONE, 2), self.TWO)
        self.assertEqual(fd.add_bits(self.PZ, self.NEG1, 2), self.NEG1)

    # ---- --oracle hw: routed through tools/hw_oracle, compared with the reference (a fake stands in for the GPU) ----

    class FakeHW:
        """Answers each raw kernel with the reference's own result, as a hardware stand-in; flip corrupts v_trunc_f64."""
        device = "fake gfx1036"

        def __init__(self, flip=False, reason=None):
            self.flip, self.reason, self.calls, self.closed = flip, reason, [], False

        def run(self, kernels):
            self.calls.append(list(kernels))
            out = []
            for kernel in kernels:
                if self.reason:
                    out.append(self.reason)
                    continue
                try:
                    words = fd.ref_program(kernel.lines[:-3], True)   # drop the store epilogue
                except (fd.SkipCase, ArithmeticError, IndexError, KeyError, ValueError):
                    out.append("assembler rejected: fake")
                    continue
                broken = self.flip and any(line.startswith("v_trunc_f64 ") for line in kernel.lines)   # a planted hardware bug
                out.append([words[0] ^ 1] + list(words[1:]) if broken else list(words))
            return out

        def close(self):
            self.closed = True

    def run_main(self, fake, *args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(fd, "open_hw_oracle", return_value=fake) as opener, \
                mock.patch.object(fd, "translate", side_effect=AssertionError("the translator must not run")), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = fd.main(["--oracle", "hw", *args])
        return code, stdout.getvalue(), stderr.getvalue(), opener

    def test_hw_oracle_routes_programs_through_hw_oracle_and_uses_hw_arch(self):
        fake = self.FakeHW()
        code, out, _, opener = self.run_main(fake, "--hw-arch", "gfx1036", "--programs", "40", "--seed", "1")
        self.assertEqual(code, 0, out)
        opener.assert_called_once_with("gfx1036")
        self.assertIn("hw: ", out)
        self.assertIn(" 0 mismatch", out)
        self.assertIn("fake gfx1036", out)
        self.assertTrue(fake.closed)
        self.assertEqual(len(fake.calls), 1)                 # one launch for the whole campaign
        kernel = fake.calls[0][0]
        self.assertEqual(kernel.lines[-1], "s_endpgm")
        self.assertTrue(kernel.raw and kernel.threads == 1 and kernel.words == 4)
        self.assertEqual(kernel.mode, fd.HW_MODE)
        self.assertGreater(len(fake.calls[0]), 40)           # sweep + programs
        matches = int(out.split("hw: ")[1].split(" match")[0])
        self.assertGreater(matches, 60)

    def test_hw_oracle_with_known_runs_known_cases_against_the_reference(self):
        fake = self.FakeHW()
        code, out, _, _ = self.run_main(fake, "--known", "--programs", "20", "--seed", "2")
        self.assertEqual(code, 0, out)
        self.assertNotIn("KNOWN_DIVERGENCE", out)           # translator-only exclusions do not apply to hardware
        texts = [line for kernel in fake.calls[0] for line in kernel.lines]
        self.assertTrue(any(" mul:2" in line for line in texts))
        self.assertTrue(any(" clamp" in line for line in texts))

    def test_hw_oracle_mismatch_exits_1_and_shrinks_with_single_runs(self):
        fake = self.FakeHW(flip=True)
        code, out, _, _ = self.run_main(fake, "--ops", "v_trunc_f64", "--programs", "0")
        self.assertEqual(code, 1, out)
        self.assertIn("MISMATCH", out)
        self.assertIn("minimal program:", out)
        self.assertIn("v_trunc_f64", out)
        self.assertGreater(len(fake.calls), 1)               # the shrinker re-ran trial programs
        self.assertTrue(all(len(batch) == 1 for batch in fake.calls[1:]))
        self.assertTrue(fake.closed)

    def test_hw_oracle_skip_reason_from_runner_is_counted(self):
        fake = self.FakeHW(reason="MODE dx10_clamp, ieee cannot be set on gfx1170")
        code, out, _, _ = self.run_main(fake, "--ops", "v_trunc_f64", "--programs", "3")
        self.assertEqual(code, 0, out)
        self.assertIn("hardware: MODE dx10_clamp", out)
        self.assertIn("0 match, 0 mismatch", out)

    def test_hw_oracle_unavailable_exits_77_without_running(self):
        stderr = io.StringIO()
        with mock.patch.object(fd, "open_hw_oracle", side_effect=fd.HWUnavailable("no AMD clang")), \
                contextlib.redirect_stderr(stderr):
            code = fd.main(["--oracle", "hw", "--programs", "3"])
        self.assertEqual(code, 77)
        self.assertIn("hardware oracle unavailable: no AMD clang", stderr.getvalue())

    def test_hw_oracle_does_not_judge_approximate_transcendentals(self):
        fake = self.FakeHW()
        with tempfile.TemporaryDirectory() as tmp:
            for op in ("v_rcp_f64", "v_rsq_f64", "v_sqrt_f64"):
                status, reason, _, actual = fd.run_one([f"{op} v[8:9], 3.25"], "ps_f64_hw_approx", tmp, "hw", fake)
                self.assertEqual(status, "skip")
                self.assertIn("PROC-216", reason)
                self.assertIsNone(actual)
        self.assertEqual(fake.calls, [])

    def test_hw_oracle_open_failure_without_tools_is_hwunavailable(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=os.path.join(ROOT, "tools", "no-such-clang.exe"))
        with mock.patch.dict(os.environ, env), self.assertRaises(fd.HWUnavailable):
            fd.open_hw_oracle("gfx1036")

    def test_cli_exit_1_on_plantable_mismatch(self):
        original = fd.translate
        stdout = io.StringIO()
        try:
            fd.translate = lambda listing, name, temp: [0xDEADBEEF, 0, 0, 0]
            with contextlib.redirect_stdout(stdout):
                code = fd.main(["--ops", "v_trunc_f64", "--programs", "0"])
        finally:
            fd.translate = original
        self.assertEqual(code, 1)
        self.assertIn("minimal program:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
