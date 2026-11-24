"""Tests for tools/f32_fuzz.py, the FUZZ-3 float VALU differential fuzzer (no game data, no GPU, no llvm-mc).

    python -I tests/test_f32_fuzz.py
"""
import os
import random
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.dont_write_bytecode = True

import f32_fuzz as ff  # noqa: E402


class ModelTests(unittest.TestCase):
    def test_round_f32_ties_to_even_and_overflow(self):
        from fractions import Fraction
        self.assertEqual(ff.round_f32(Fraction(1) + Fraction(1, 1 << 24)), ff.bits(1.0))          # tie -> even
        self.assertEqual(ff.round_f32(Fraction(1) + Fraction(3, 1 << 24)), ff.bits(1.0) + 2)
        self.assertEqual(ff.round_f32(Fraction(2) ** 128), ff.PINF)
        self.assertEqual(ff.round_f32(Fraction(-1, 3)), 0xBEAAAAAB)

    def test_mad_and_fma_differ_where_rounding_matters(self):
        x, y, z = 0xBC500C59, 0x435A2099, 0x3E3DB015
        self.assertEqual(ff.add(ff.mul(x, y), z), 0xC0256A01)
        self.assertEqual(ff.fma(x, y, z), 0xC0256A00)

    def test_signed_zeros(self):
        self.assertEqual(ff.add(0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(ff.add(ff.bits(1.0), ff.bits(-1.0)), 0)
        self.assertEqual(ff.legacy_mul(0x80000000, ff.NINF), 0)
        self.assertEqual(ff.rounding(ff.bits(-0.25), "ceil"), 0x80000000)

    def test_rtz_f16(self):
        self.assertEqual(ff.round_rtz_f16(ff.bits(65519.0)), 0x7BFF)
        self.assertEqual(ff.round_rtz_f16(ff.NINF), 0xFC00)
        self.assertEqual(ff.round_rtz_f16(ff.bits(2.0 ** -24 * 1.5)), 0x0001)

    def test_cube_major_axis(self):
        # |z| largest and negative: face 5, sc = -x, tc = -y, ma = 2z.
        x, y, z = ff.bits(0.5), ff.bits(0.25), ff.bits(-2.0)
        self.assertEqual(ff.cube("v_cubeid_f32", x, y, z), ff.bits(5.0))
        self.assertEqual(ff.cube("v_cubesc_f32", x, y, z), ff.bits(-0.5))
        self.assertEqual(ff.cube("v_cubetc_f32", x, y, z), ff.bits(-0.25))
        self.assertEqual(ff.cube("v_cubema_f32", x, y, z), ff.bits(-4.0))
        # Measured on RDNA2 (gfx1036): v_cubema_f32 returns +0.0 (never -0.0) when all axes are ±0.0.
        self.assertEqual(ff.cube("v_cubema_f32", 0, 0, 0x80000000), 0)
        self.assertEqual(ff.cube("v_cubema_f32", 0x80000000, 0x80000000, 0x80000000), 0)


class HarnessTests(unittest.TestCase):
    def test_self_test(self):
        self.assertTrue(ff.self_test())

    def test_generation_is_deterministic(self):
        a = ff.gen(random.Random(5), "v_fma_f32")
        b = ff.gen(random.Random(5), "v_fma_f32")
        self.assertEqual(a, b)

    def test_seeded_run_has_no_mismatch(self):
        # A short seeded sweep over every op: the translator rules found by FUZZ-3 stay fixed. Undefined-in-SPIR-V
        # inputs and denormals are counted, never compared.
        counts, by_op, _ = ff.run_suite(400, 3, tuple(ff.OPS), quiet=True)
        self.assertEqual(by_op, {})
        self.assertEqual(counts["mismatch"], 0)
        self.assertGreater(counts["match"], 300)

    def test_v_mac_f32_is_not_fused(self):
        vals = [0xBC500C59, 0x435A2099] + [0] * 6 + [0x3E3DB015]
        status, detail = ff.check("v_mac_f32", vals, [(0, {}), (1, {})], False, [])
        self.assertEqual(status, "match", detail)

    def test_output_modifiers_on_conversions(self):
        vals = [0x00007800] + [0] * 8
        for op, mods in (("v_cvt_f32_ubyte1", ["clamp"]), ("v_cvt_f32_u32", ["mul:4"]),
                         ("v_cvt_f32_i32", ["div:2"])):
            with self.subTest(op=op, mods=mods):
                status, detail = ff.check(op, vals, [(0, {})], True, mods)
                self.assertEqual(status, "match", detail)

    def test_undefined_spirv_is_its_own_class(self):
        vals = [0] + [0] * 8                                 # v_rcp_f32 of +0: hardware +inf, OpFDiv by zero
        status, _ = ff.check("v_rcp_f32", vals, [(0, {})], False, [])
        self.assertEqual(status, "undefined")

    def test_denormal_operand_is_skipped(self):
        vals = [0x00000001, ff.bits(1.0)] + [0] * 7
        status, detail = ff.check("v_add_f32", vals, [(0, {}), (1, {})], False, [])
        self.assertEqual(status, "skip")
        self.assertIn("denormal", detail)

    def test_fmamk_madmk_operand_order_and_translator(self):
        # S0 * K + S2 = 2.0 * 3.0 + 4.0 = 10.0 (whereas S0 * S2 + K would be 11.0).
        vals = [0] * len(ff.SEED_REGS)
        vals[0], vals[1] = ff.bits(2.0), ff.bits(4.0)
        srcs = [(0, {}), ("k", ff.bits(3.0)), (1, {})]
        for op in ("v_fmamk_f32", "v_madmk_f32"):
            with self.subTest(op=op):
                got_model = ff.model(op, vals, srcs, [])
                self.assertEqual(got_model, [ff.bits(10.0), 0])
                self.assertEqual(ff.line(op, vals, srcs, False, []),
                                 f"{op} v8, v0, 0x{ff.bits(3.0):x}, v1")
                status, words = ff.run_translated(ff.listing(op, vals, srcs, False, [])[0])
                self.assertEqual((status, words[:2]), ("ok", got_model))


class CliTests(unittest.TestCase):
    TOOL = os.path.join(ROOT, "tools", "f32_fuzz.py")

    def test_no_arguments_prints_usage(self):
        r = subprocess.run([sys.executable, "-I", self.TOOL], capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("usage:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


class CliTests(unittest.TestCase):
    TOOL = os.path.join(ROOT, "tools", "f32_fuzz.py")

    def test_no_arguments_prints_usage(self):
        r = subprocess.run([sys.executable, "-I", self.TOOL], capture_output=True, text=True, timeout=5)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("usage:", r.stderr)
        self.assertNotIn("Traceback", r.stderr)


class HardwareOracle(unittest.TestCase):
    """--oracle hw: kernel text, verdicts, GFX11 skips and the exit-77 contract (no GPU needed), plus a run on the
    GPU when the oracle tools are present."""
    TOOL = os.path.join(ROOT, "tools", "f32_fuzz.py")

    def case(self, op, vals, srcs, mods=(), e64=False):
        return ff.Case(op, list(vals) + [0] * (9 - len(vals)), srcs, e64, list(mods))

    def test_kernel_body(self):
        c = self.case("v_add_f32", [ff.bits(1.0), ff.bits(2.0)], [(0, {}), (1, {})], ["mul:2", "clamp"], True)
        lines = ff.hw_lines(c, "gfx1150")
        self.assertIn("v_add_f32_e64 v8, v0, v1 clamp mul:2", lines)       # GFX11 wants clamp before omod
        self.assertEqual(lines[-3:], ["v_mov_b32 v20, 0", "global_store_b64 v20, v[8:9], s[2:3]", "s_endpgm"])
        self.assertEqual(ff.hw_lines(c, "gfx1030")[-2], "global_store_dwordx2 v20, v[8:9], s[2:3]")
        self.assertEqual(len(ff.hw_lines(c, "gfx1150", guard=False)), len(lines) - 1)

    def test_packed_u8_hardware_verdict_compares_raw_bytes(self) -> None:
        case = self.case("v_cvt_pk_u8_f32", [0x43000000, 0, 0x7FC00000],
                         [(0, {}), (1, {}), (2, {})], e64=True)
        self.assertEqual(ff.hw_verdict(case, [0x7FC00080, 0]), ("match", [0x7FC00080, 0]))
        self.assertEqual(ff.hw_verdict(case, [0x7FC00081, 0]), ("mismatch", [0x7FC00080, 0]))

    def test_removed_opcodes_are_skipped_not_substituted(self):
        from hw_oracle import asm
        for op in ("v_mad_f32", "v_mac_f32", "v_madak_f32", "v_madmk_f32", "v_mad_legacy_f32"):
            self.assertIn(op, asm.valu_skip_reason(op, "gfx1150"))
            self.assertIn(op, asm.valu_skip_reason(op, "gfx1030"))
            self.assertIsNone(asm.valu_skip_reason(op, "gfx1013"))
        self.assertIsNone(asm.valu_skip_reason("v_mul_legacy_f32", "gfx1150"))

    def test_mode_words(self):
        self.assertEqual(ff.mode_word(ff.HW_MODES["flush"]), 0x1C0)
        self.assertEqual(ff.mode_word(ff.HW_MODES["keep"]), 0x1F0)
        self.assertEqual(ff.mode_word(ff.HW_MODES["ieee"]), 0x3C0)
        self.assertFalse(ff.omod_active("keep", "v_add_f32"))
        self.assertTrue(ff.omod_active("keep", "v_rcp_f32"))

    def test_verdicts(self):
        a = self.case("v_add_f32", [ff.bits(1.0), ff.bits(2.0)], [(0, {}), (1, {})])
        self.assertEqual(ff.hw_verdict(a, [ff.bits(3.0), 0])[0], "match")
        self.assertEqual(ff.hw_verdict(a, [ff.bits(3.0) ^ 1, 0])[0], "mismatch")
        d = self.case("v_ceil_f32", [ff.bits(-0.75)], [(0, {})], ["mul:2"], True)
        self.assertEqual(ff.hw_verdict(d, [0, 0])[0], "deviation:omod_neg_zero")     # omod: -0 becomes +0
        self.assertEqual(ff.hw_verdict(d, [0x80000000, 0])[0], "match")
        self.assertEqual(ff.hw_verdict(d, [ff.bits(1.0), 0])[0], "mismatch")
        dn = self.case("v_add_f32", [1, ff.bits(1.0)], [(0, {}), (1, {})])
        self.assertEqual(ff.hw_verdict(dn, [ff.bits(1.0), 0])[0], "skip")

    def test_omod_deviation_needs_an_active_omod(self):
        d = self.case("v_ceil_f32", [ff.bits(-0.75)], [(0, {})], ["mul:2"], True)
        want = [0x80000000, 0]
        self.assertEqual(ff.dev_omod_neg_zero(d, want), [0, 0])
        self.assertIsNone(ff.dev_omod_neg_zero(d, want, "ieee"))
        self.assertIsNone(ff.dev_omod_neg_zero(d, want, "keep"))
        self.assertIsNone(ff.dev_omod_neg_zero(self.case("v_ceil_f32", [0], [(0, {})], [], True), want))

    def test_omod_label_does_not_count_a_kept_minus_zero_as_applied(self):
        # ceil(-0.75) = -0: with and without the omod the model says -0, so a -0 from the hardware shows neither
        # that the omod was applied nor that it was ignored (SHADER-353: 65 such programs under MODE.IEEE = 1 on
        # gfx1031 were counted as "applied" while every unambiguous one was ignored).
        d = self.case("v_ceil_f32", [ff.bits(-0.75)], [(0, {})], ["mul:2"], True)
        self.assertEqual(ff.omod_label(d, [0x80000000, 0]), "-0 result kept (applied or ignored, same word)")
        self.assertEqual(ff.omod_label(d, [0, 0]), "applied, but a -0 result becomes +0")
        a = self.case("v_ceil_f32", [ff.bits(1.5)], [(0, {})], ["mul:2"], True)
        self.assertEqual(ff.omod_label(a, [ff.bits(4.0), 0]), "applied")
        self.assertEqual(ff.omod_label(a, [ff.bits(2.0), 0]), "ignored")
        self.assertEqual(ff.omod_label(a, [ff.bits(3.0), 0]), "other")

    def test_zero_inf_probes_catch_a_plain_multiply(self):
        import random
        cases = [ff.gen_probe(random.Random(i), "v_mul_legacy_f32", "zero_inf") for i in range(8)]
        self.assertTrue(all(c is not None for c in cases))
        with ff.perturbed("legacy_is_ieee"):
            bad = sum(ff.hw_verdict(c, [0, 0])[0] == "match" for c in cases)
        self.assertLess(bad, len(cases))

    def test_f16_nan_halves_compare_as_nan(self):
        self.assertTrue(ff.hw_same([0x7E00, 0], [0x7E06, 0], "v_cvt_f16_f32"))
        self.assertFalse(ff.hw_same([0x7E00, 0], [0x3C00, 0], "v_cvt_f16_f32"))
        self.assertFalse(ff.hw_same([0x00007E00, 0], [0x00017E00, 0], "v_cvt_f16_f32"))   # high half is kept exactly

    def test_negative_control_detects_each_perturbation(self):
        self.assertTrue(ff.hw_self_test())

    def test_denormal_variants_distinguish_flush_from_keep(self):
        m = self.case("v_mul_f32", [1, ff.bits(1048576.0)], [(0, {}), (1, {})])
        self.assertEqual(ff.denorm_label(m, [0, 0]), ("operand", "ftz|ftz_in|ftz_out"))
        self.assertEqual(ff.denorm_label(m, [ff.bits(2.0 ** -129), 0]), ("operand", "keep"))

    def test_hardware_only_ops_stay_out_of_the_translator_run(self):
        self.assertTrue(set(ff.HW_ONLY_OPS).isdisjoint(ff.OPS))
        r = subprocess.run([sys.executable, "-I", self.TOOL, "--ops", "v_min_f32"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    def test_exit_77_without_the_oracle_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=os.path.join(ROOT, "tools", "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", self.TOOL, "--oracle", "hw", "--programs", "2"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_unknown_oracle_is_a_usage_error(self):
        r = subprocess.run([sys.executable, "-I", self.TOOL, "--oracle", "gpu"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)

    def test_on_hardware(self):
        from hw_oracle import runner
        try:
            oracle = runner.open_oracle()
        except runner.Unavailable as e:
            self.skipTest(f"hardware oracle unavailable: {e}")
        from hw_oracle import asm
        why = asm.rdna23_skip_reason(oracle.arch)
        if why:
            oracle.close()
            self.skipTest(why)
        try:
            ff.hw_probe_modes(oracle)
            cases = ff.gen_cases(300, 21, tuple(ff.ALL_OPS)) + ff.gen_probes(2, 21, tuple(ff.ALL_OPS))
            rep = ff.hw_suite(oracle, cases, info_modes=("keep",))
        finally:
            oracle.close()
        self.assertEqual([(c.op, c.index) for c, _, _ in rep.mismatches], [])
        self.assertGreater(sum(r["match"] for r in rep.per_op.values()), 150)
        for name, (n, bad) in ff.negative_control(rep).items():
            if n >= 20 and name in ("add_lsb", "mul_rtz", "rndne_floor", "min_ieee_nan"):
                self.assertGreater(bad, 0, name)


if __name__ == "__main__":
    unittest.main()
