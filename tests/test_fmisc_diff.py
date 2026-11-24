"""tools/fmisc_diff.py: the f32 misc-op and float-compare reference against hand-worked values, a short seeded fuzz
run, and canaries. Standard library only; no GPU."""
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import unittest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "fmisc_diff.py"
sys.path.insert(0, str(TOOL.parent))
import fmisc_diff as d  # noqa: E402

M = d.M32
SNAN, QNAN = 0x7F800001, 0x7FC00000


def prefix(v=None, s=None):
    return ([f"v_mov_b32_e32 v{r}, 0x{(v or {}).get(r, 0):x}" for r in d.VREGS]
            + [f"s_mov_b32 s{r}, 0x{(s or {}).get(r, 0):x}" for r in d.SREGS])


def run(*body, v=None, s=None, regs=(1, 2, 3, 4)):
    """The reference's four exported words; v / s map register -> seed value (others are 0)."""
    return d.reference(prefix(v, s) + list(body) + d.observe(list(regs)))


def one(line, **kw):
    return run(line, **kw)[0]


def cmp(line, **kw):
    return run(line, "v_cndmask_b32_e64 v1, 0, 1.0, " + line.split()[1].rstrip(","), **kw)[0] == 0x3F800000


class Reference(unittest.TestCase):
    def test_fract(self):
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: 0xBFC00000}), 0x3F000000)     # -1.5 -> 0.5
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: 0xB3800000}), 0x3F7FFFFF)     # -2^-24 -> below 1.0
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: 0x80000000}), 0)              # -0 -> +0
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: 0x4B7FFFFF}), 0)              # an integer
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: d.NINF}), 0xFFC00000)       # gfx1036: exactly this
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: d.PINF}), 0xFFC00000)
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: 0x7F800001}), 0x7F800001)     # a signalling NaN stays
        self.assertEqual(one("v_fract_f32_e32 v1, v2", v={2: 0xFFC00123}), 0xFFC00123)     # a NaN comes back as is
        self.assertEqual(one("v_fract_f32_e64 v1, -v2 mul:2", v={2: 0x3F000000}), 0x3F800000)  # 0.5 * 2

    def test_frexp(self):
        self.assertEqual(one("v_frexp_mant_f32_e32 v1, v2", v={2: 0xC1200000}), 0xBF200000)  # -10 = -0.625 * 2^4
        self.assertEqual(one("v_frexp_exp_i32_f32_e32 v1, v2", v={2: 0xC1200000}), 4)
        self.assertEqual(one("v_frexp_exp_i32_f32_e32 v1, v2", v={2: 0x3E800000}), M)      # 0.25 = 0.5 * 2^-1
        self.assertEqual(one("v_frexp_mant_f32_e32 v1, v2", v={2: d.NINF}), d.NINF)
        self.assertEqual(one("v_frexp_exp_i32_f32_e32 v1, v2", v={2: d.PINF}), 0)
        self.assertEqual(one("v_frexp_mant_f32_e32 v1, v2", v={2: 0x80000000}), 0x80000000)

    def test_ldexp(self):
        self.assertEqual(one("v_ldexp_f32 v1, v2, -3", v={2: 0x41000000}), 0x3F800000)      # 8 * 2^-3
        self.assertEqual(one("v_ldexp_f32 v1, -v2, v3", v={2: 0x3F800000, 3: 128}), d.NINF)
        self.assertEqual(one("v_ldexp_f32 v1, v2, v3", v={2: 0x3F800000, 3: 0x80000000}), 0)
        self.assertEqual(one("v_ldexp_f32 v1, v2, v3", v={2: 0xBF800000, 3: 0x80000000}), 0x80000000)
        self.assertEqual(one("v_ldexp_f32 v1, v2, 64", v={2: d.NINF}), d.NINF)
        with self.assertRaises(d.Uncertain):
            one("v_ldexp_f32 v1, v2, -127", v={2: 0x3F800000})                             # a denormal result

    def test_conversions(self):
        self.assertEqual(one("v_cvt_u32_f32_e32 v1, v2", v={2: 0x4F800000}), M)            # 2^32 saturates
        self.assertEqual(one("v_cvt_u32_f32_e32 v1, v2", v={2: 0xBFC00000}), 0)
        self.assertEqual(one("v_cvt_i32_f32_e32 v1, v2", v={2: 0xCF000001}), 0x80000000)
        self.assertEqual(one("v_cvt_i32_f32_e32 v1, v2", v={2: 0xC0200000}), M - 1)        # -2.5 -> -2
        self.assertEqual(one("v_cvt_i32_f32_e64 v1, -v2", v={2: QNAN}), 0)
        self.assertEqual(one("v_cvt_off_f32_i4_e32 v1, v2", v={2: 0x18}), 0xBF000000)      # -8 / 16
        self.assertEqual(one("v_cvt_off_f32_i4_e64 v1, 7 mul:4", v={}), 0x3FE00000)        # 7/16 * 4

    def test_fmamk_rounds_once(self):
        # (1 + 2^-23) * (1 - 2^-24) - 1 is 2^-24 - 2^-47, exact in f32 (0x337ffffe); a rounded product would give
        # 1 + 2^-24 -> 1.0 (ties to even) and then 0
        self.assertEqual(one("v_fmamk_f32 v1, v2, 0x3f7fffff, v3", v={2: 0x3F800001, 3: 0xBF800000}), 0x337FFFFE)
        self.assertEqual(one("v_fmamk_f32 v1, v2, 0x0, v3", v={2: d.PINF, 3: 0}), QNAN)
        self.assertEqual(one("v_fmamk_f32 v1, -v2, 0x3f800000, v3", v={2: 0, 3: 0x80000000}), 0x80000000)

    def test_output_modifiers(self):
        self.assertEqual(one("v_ldexp_f32 v1, v2, 1 clamp", v={2: 0x3F400000}), 0x3F800000)  # 1.5 -> 1.0
        self.assertEqual(one("v_frexp_mant_f32_e64 v1, v2 clamp div:2", v={2: 0xC1200000}), 0)
        with self.assertRaises(d.Uncertain):
            one("v_fract_f32_e64 v1, v2 mul:2", v={2: 0x40000000})                         # omod on a zero
        with self.assertRaises(d.Uncertain):
            one("v_fract_f32_e64 v1, v2 clamp", v={2: d.PINF})                             # clamp of a NaN

    def test_compares(self):
        v = {2: 0x3F800000, 3: QNAN, 4: 0xBF800000}
        self.assertTrue(cmp("v_cmp_lt_f32_e64 s20, v4, v2", v=v))
        self.assertFalse(cmp("v_cmp_lt_f32_e64 s20, -v4, v2", v=v))                         # 1 < 1
        self.assertTrue(cmp("v_cmp_le_f32_e64 s21, -v4, |v2|", v=v))
        self.assertTrue(cmp("v_cmp_eq_f32_e64 vcc_lo, |v4|, v2", v=v))
        self.assertTrue(cmp("v_cmp_eq_f32_e32 vcc_lo, 0, v5", v={5: 0x80000000}))           # +0 == -0
        for c, want in (("lt", False), ("nge", True), ("u", True), ("o", False), ("lg", False), ("neq", True),
                        ("tru", True), ("f", False)):
            self.assertEqual(cmp(f"v_cmp_{c}_f32_e64 s20, v3, v2", v=v), want, c)

    def test_class(self):
        v = {2: SNAN, 3: QNAN, 4: 0xBF800000, 5: 0x80000000, 6: d.PINF}
        for reg, bit in ((2, 0), (3, 1), (4, 3), (5, 5), (6, 9)):
            self.assertTrue(cmp(f"v_cmp_class_f32_e64 s20, v{reg}, 0x{1 << bit:x}", v=v), reg)
            self.assertFalse(cmp(f"v_cmp_class_f32_e64 s20, v{reg}, 0x{0x3FF ^ (1 << bit):x}", v=v), reg)
        # measured on gfx1036: neg / abs move the sign of a signalling NaN and keep it signalling
        self.assertTrue(cmp("v_cmp_class_f32_e64 vcc_lo, -|v2|, 1", v=v))
        self.assertTrue(cmp("v_cmp_class_f32_e64 vcc_lo, |v4|, 0x100", v=v))                # |-1| is +normal

    def test_what_is_not_judged(self):
        with self.assertRaises(d.Uncertain):
            one("v_fract_f32_e32 v1, v2", v={2: 1})                                         # denormal operand
        with self.assertRaises(d.Uncertain):                                                # a NaN's payload
            run("v_fract_f32_e32 v1, v2", "v_cvt_off_f32_i4_e32 v3, v1", v={2: QNAN})
        # a computed NaN read as a float, or copied, is fine; any NaN matches any NaN
        self.assertTrue(d.is_nan(run("v_fract_f32_e32 v1, v2", "v_fract_f32_e32 v3, v1", v={2: QNAN})[2]))
        self.assertIsNone(d._diff(["", "", "", "v_mov_b32_e32 v8, v1"] * 1, [QNAN] + [0] * 3,
                                  [0xFFC00001] + [0] * 3, "x"))


class Generator(unittest.TestCase):
    def lines(self, known=False, literals=True, n=3000, seed=1):
        real = d.KNOWN, d.LITERALS
        d.KNOWN, d.LITERALS = known, literals
        try:
            gen = d.Gen(d.random.Random(seed))
            return [line for _ in range(n) for line in gen.op(d.ALL_OPS)]
        finally:
            d.KNOWN, d.LITERALS = real

    def test_every_op_is_generated_in_every_form(self):
        lines = self.lines()
        forms = {line.split()[0] for line in lines}
        for name, (form, _, fout, _) in d.OPS.items():
            if form == "vop1":
                self.assertTrue({name + "_e32", name + "_e64"} <= forms, name)
            else:
                self.assertIn(name, forms)
        for c in d.FCMP:
            self.assertTrue({f"v_cmp_{c}_f32_e32", f"v_cmp_{c}_f32_e64"} <= forms, c)
        self.assertIn("v_cmp_class_f32_e64", forms)
        for word in (" mul:2", " mul:4", " div:2", " clamp", "|", "-v", "-s"):
            self.assertTrue(any(word in line for line in lines), word)

    def test_sources_respect_the_encoding_rules(self):
        for line in self.lines():
            op, _, rest = line.partition(" ")
            srcs, mods = d.parse(rest)
            srcs = srcs[1:]
            if op == "v_cndmask_b32_e64":
                continue
            bare = [re.sub(r"^-?\|?([vs]\d+)\|?$", r"\1", x) for x in srcs]
            lits = [x for x in bare if x.startswith("0x")]
            scalars = set(x for x in bare if x.startswith("s")) | set(lits)
            self.assertLessEqual(len(lits), 1, line)
            self.assertLessEqual(len(scalars), 2, line)
            if op.endswith("_e32") or op == "v_fmamk_f32":
                self.assertFalse(any(x != y for x, y in zip(srcs, bare)), line)             # no modifiers
                self.assertTrue(all(x.startswith("v") for x in bare[1:]) or op == "v_fmamk_f32", line)
            if " clamp" in line and any(m in line for m in d._OMOD):
                self.assertLess(line.index(" clamp"), max(line.find(m) for m in d._OMOD), line)
        self.assertFalse(any("0x" in line for line in self.lines(literals=False)
                             if not line.startswith("v_fmamk_f32")))

    def test_known_divergences_are_left_out_by_default(self):
        def class_mods(lines):
            return any(line.startswith("v_cmp_class") and re.match(r"\s*(-|\|)[vs|]", line.split(",")[1])
                       for line in lines)
        self.assertFalse(class_mods(self.lines()))
        self.assertTrue(class_mods(self.lines(known=True)))


class Translated(unittest.TestCase):
    def test_harness_canaries(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(d.self_test())

    def test_short_seeded_run_matches(self):
        counts, failures = d.run_suite(80, 5, 4, d.ALL_OPS)
        self.assertEqual(failures, [], failures[:1])
        self.assertTrue(all(k.startswith("uncertain: ") for k in counts["skip"]), counts["skip"])
        self.assertGreaterEqual(counts["pass"], 60)

    def test_hand_cases_match_the_translator(self):
        seeds = {2: 0xBFC00000, 3: 0xB3800000, 4: 0xC1200000, 5: 0x4F800000, 6: QNAN, 7: 0x3F800001, 12: 0x18,
                 13: 0xBF800000}
        cases = ["v_fract_f32_e32 v1, v2", "v_fract_f32_e64 v1, -v3", "v_frexp_mant_f32_e64 v1, |v4| mul:2",
                 "v_frexp_exp_i32_f32_e32 v1, v4", "v_ldexp_f32 v1, -v4, s4 clamp", "v_ldexp_f32 v1, v13, 0x80",
                 "v_cvt_u32_f32_e32 v1, v5", "v_cvt_i32_f32_e64 v1, -|v4|", "v_cvt_off_f32_i4_e64 v1, v12 mul:4",
                 "v_fmamk_f32 v1, v7, 0x3f7fffff, v13", "v_fmamk_f32 v1, s5, 0x7f800000, v2"]
        for case in cases:
            with self.subTest(case=case):
                self.assertIsNone(d.check(prefix(seeds, {4: 3, 5: 0x3F000000}) + [case] + d.observe([1, 2, 3, 4])))
        for c in d.FCMP:
            for case in (f"v_cmp_{c}_f32_e64 s20, -v2, |v13|", f"v_cmp_{c}_f32_e32 vcc_lo, 1.0, v6"):
                with self.subTest(case=case):
                    m = case.split()[1].rstrip(",")
                    self.assertIsNone(d.check(prefix(seeds) + [case, f"v_cndmask_b32_e64 v1, 0, 1.0, {m}"]
                                              + d.observe([1, 2, 3, 4])))

    def test_a_wrong_translation_is_caught(self):
        lines = prefix({2: 0xBFC00000}) + ["v_fract_f32_e32 v1, v2"] + d.observe([1, 2, 3, 4])
        self.assertIsNone(d.check(lines))
        want = d.reference(lines)
        real = d.translated
        try:
            d.translated = lambda _lines: [want[0] ^ 1] + want[1:]
            self.assertIn("word 0 (v1)", d.check(lines))
            d.translated = lambda _lines: [QNAN ^ 0x1234] + want[1:]          # a NaN where a number is wanted
            self.assertIn("word 0 (v1)", d.check(lines))
        finally:
            d.translated = real

    def test_shrinking_keeps_the_instruction_that_matters(self):
        lines = d.generate(d.random.Random(2), 5, ["v_frexp_mant_f32", "v_cvt_off_f32_i4", "cmp"])
        body = lines[d.N_SEED:-d.N_OBS]
        target = body[len(body) // 2]
        if target.startswith("v_cmp"):
            target = body[len(body) // 2 + 1]
        real = d.translated
        try:
            d.translated = lambda l: [w ^ (1 if target in l else 0) for w in d.reference(l)]
            small = d.shrink(lines)
        finally:
            d.translated = real
        kept = [line for line in small if not line.startswith(("v_mov_b32_e32", "s_mov_b32"))]
        self.assertIn(target, kept)
        self.assertLessEqual(len(kept), 2)                       # the target (and its compare, for a v_cndmask)


class Cli(unittest.TestCase):
    def main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            try:
                rc = d.main(list(argv))
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue()

    def test_exit_codes(self):
        rc, out = self.main("--list-ops")
        self.assertEqual(rc, 0)
        self.assertIn("v_ldexp_f32", out)
        self.assertEqual(self.main("--ops", "v_nope")[0], 2)
        self.assertEqual(self.main("--oracle", "gpu")[0], 2)
        rc, out = self.main("--programs", "5", "--seed", "1")
        self.assertEqual(rc, 0, out)
        real = d.translated
        try:
            d.translated = lambda l: [w ^ 1 for w in d.reference(l)]
            rc, out = self.main("--programs", "5", "--seed", "1", "--ops", "v_cvt_u32_f32")
        finally:
            d.translated = real
        self.assertEqual(rc, 1)
        self.assertIn("MISMATCH", out)

    def test_a_run_where_every_program_was_skipped_is_not_a_success(self):
        def refuse(lines, *args):
            raise d.Skip("unsupported form")
        real = d.translated
        try:
            d.translated = refuse
            rc, out = self.main("--programs", "3", "--seed", "1", "--ops", "v_cvt_u32_f32")
        finally:
            d.translated = real
        self.assertEqual(rc, 2, out)
        self.assertIn("no program ran: every one was skipped", out)

    def test_hardware_oracle_without_the_tools_exits_77(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOL.parent / "no-such-clang.exe"))
        r = subprocess.run([sys.executable, "-I", str(TOOL), "--oracle", "hw", "--programs", "3"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)


class VulkanOracle(unittest.TestCase):
    """--oracle hw,vk: the compute program the GPU runs, its wave64 spelling, and the verdict against the hardware."""

    def setUp(self):
        import vk_fuzz_run
        self.vk = vk_fuzz_run

    def test_program_text_wave32(self):
        lines = ["v_mov_b32_e32 v1, 0x3f800000", "v_cmp_lt_f32_e64 s20, v1, v2", "v_cndmask_b32_e64 v8, 0, 1, s20"]
        text = self.vk.program_text("cs_x", lines, 4, 32).splitlines()
        self.assertIn("; group: 32,1,1", text)
        self.assertEqual(text[3:5], ["s_mov_b32 exec_lo, -1", "v_mov_b32 v20, v0"])
        self.assertIn("v_lshlrev_b32 v21, 4, v20", text)
        self.assertIn("buffer_store_dword v8, v21, s[0:3], 0 offen", text)
        self.assertIn("buffer_store_dword v11, v21, s[0:3], 0 offen offset:12", text)
        self.assertEqual(text[-1], "s_endpgm")
        self.assertIn("v_cmp_lt_f32_e64 s20, v1, v2", text)

    def test_wave64_spells_lane_masks_as_pairs(self):
        w = self.vk.wave64_text
        self.assertEqual(w("v_cmp_lt_f32_e64 s20, v1, v2"), "v_cmp_lt_f32_e64 s[20:21], v1, v2")
        self.assertEqual(w("v_cndmask_b32_e64 v8, 0, 1, s21"), "v_cndmask_b32_e64 v8, 0, 1, s[22:23]")
        self.assertEqual(w("v_cmp_lt_f32_e32 vcc_lo, v1, v2"), "v_cmp_lt_f32_e32 vcc, v1, v2")
        self.assertEqual(w("s_mov_b32 s4, 0x20"), "s_mov_b32 s4, 0x20")           # data SGPRs stay
        self.assertEqual(w("s_mov_b32 s2, 0x20"), "s_mov_b32 s2, 0x20")
        text = self.vk.program_text("cs_x", ["v_cmp_lt_f32_e64 s21, v1, v2"], 4, 64).splitlines()
        self.assertIn("; mattr: +wavefrontsize64,-wavefrontsize32", text)
        self.assertIn("s_mov_b64 exec, -1", text)
        self.assertIn("v_cmp_lt_f32_e64 s[22:23], v1, v2", text)

    def test_kernel_for_wave64(self):
        k = d.hw_kernel(["v_cmp_lt_f32_e64 s20, v1, v2"], 64)
        self.assertEqual(k.wave, 64)
        self.assertIn("v_cmp_lt_f32_e64 s[20:21], v1, v2", k.lines)

    def test_verdict_classes(self):
        lines = prefix() + ["v_fract_f32 v1, v2"] + d.observe([1, 2, 3, 4])
        same = [0x3F800000, 0, 0, 0]
        self.assertEqual(d.vk_verdict(lines, same, list(same))[0], "match")
        nan = [QNAN, 0, 0, 0]
        self.assertEqual(d.vk_verdict(lines, nan, [0xFFC00001, 0, 0, 0])[0], "match")     # any NaN
        klass, why = d.vk_verdict(lines, same, [0x3F800001, 0, 0, 0])
        self.assertEqual(klass, "mismatch")
        self.assertIn("word 0", why)
        # a denormal seed makes the reference decline: a difference there is the denormal gap, not a mismatch
        den = prefix(v={2: 0x00000001}) + ["v_fract_f32 v1, v2"] + d.observe([1, 2, 3, 4])
        self.assertEqual(d.vk_verdict(den, [0, 1, 0, 0], [0x3F800000, 1, 0, 0])[0], "denormal")

    def test_vk_needs_the_hardware(self):
        r = subprocess.run([sys.executable, "-I", str(TOOL), "--oracle", "vk", "--programs", "1"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
        self.assertIn("--oracle hw,vk", r.stderr)


if __name__ == "__main__":
    unittest.main()
