"""tools/d16_format_probe.py: the candidate rules, the GFX10 descriptor, what each kernel stages and loads, the
comparison and its exit codes (through a fake oracle that plays the GPU), and the exit-77 contract. The last class
runs on the hardware when the oracle tools are present and the GPU is GFX10, and is skipped otherwise. Standard
library only."""
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
import random
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
import d16_format_probe as pr  # noqa: E402
from hw_oracle import asm, runner  # noqa: E402


class Rules(unittest.TestCase):
    def test_self_test(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(pr.self_test(), 0)

    def test_the_489_counts_in_docs_design(self):
        # docs/design.md ("d16 and f16 format narrowing"): with RTZ narrowing 123 of 256 UNORM8, 31,742 of 65,536
        # UNORM16 and 110 of 256 SNORM8 values came out 1 ulp low
        self.assertEqual(pr.rte_rtz_disagreements("unorm", 8), 123)
        self.assertEqual(pr.rte_rtz_disagreements("snorm", 8), 110)
        self.assertEqual(pr.rte_rtz_disagreements("unorm", 16), 31742)

    def test_only_four_16_bit_values_separate_one_rounding_from_two(self):
        split = {(k, v) for k in ("unorm", "snorm") for v in range(65536)
                 if pr.models16(k, 16, v)["rte"] != pr.models16(k, 16, v)["f32_rte"]}
        self.assertEqual(split, {("unorm", 0xFFAF), ("unorm", 0xFFEF), ("snorm", 0x7FF7), ("snorm", 0x8009)})

    def test_8_and_10_bit_formats_only_tell_rte_from_rtz(self):
        for kind, n in (("unorm", 8), ("snorm", 8), ("unorm", 10)):
            for v in range(1 << n):
                m = pr.models16(kind, n, v)
                self.assertEqual(len({m["f32_rte"], m["rte"], m["mul_rte"], m["f32_rte_flush"]}), 1, (kind, v))

    def test_snorm_minimum_is_minus_one_and_f32_decode_is_exact(self):
        self.assertEqual(pr.models16("snorm", 8, 0x80)["f32_rte"], 0xBC00)
        self.assertEqual(pr.models16("snorm", 8, 0x81)["f32_rte"], 0xBC00)
        self.assertEqual(pr.models32("unorm", 16, 0xFFFF)["f32_rte"], 0x3F800000)
        self.assertEqual(pr.models32("unorm", 8, 0x80)["f32_rte"], 0x3F008081)          # 128 / 255

    def test_round_frac_matches_the_fraction_form(self):
        """round_frac works in integers; the Fraction form it replaced is the reference here, for f16 and f32 and
        both modes: 16-bit format quotients, the first and last significands of a binade exact, at a tie and just
        under one (a tie at the last one carries into the next binade), the same for denormals, powers of two down
        past the smallest denormal, random fractions in [-1, 1], and all of them negated."""
        def reference(x, prec, emin, rtz=False):
            if x == 0:
                return Fraction(0)
            sign, a = (-1 if x < 0 else 1), abs(x)
            e = a.numerator.bit_length() - a.denominator.bit_length()
            if Fraction(2) ** e > a:
                e -= 1
            if Fraction(2) ** (e + 1) <= a:
                e += 1
            ulp = Fraction(2) ** (max(e, emin) - prec + 1)
            q = a / ulp
            return sign * (q.numerator // q.denominator if rtz else round(q)) * ulp

        rng = random.Random(161)
        for prec, emin in ((11, -14), (24, -126)):
            cases = {Fraction(v, 65535) for v in range(0, 65536, 97)}
            cases |= {Fraction(v, 32767) for v in range(0, 32768, 89)}
            for e in (-1, -2, -7, emin):                     # the binade [2^e, 2^(e+1)), ulp 2^(e - prec + 1)
                ulp = Fraction(2) ** (e - prec + 1)
                for m in (2 ** (prec - 1), 2 ** (prec - 1) + 1, 2 ** prec - 2, 2 ** prec - 1):   # first and last
                    cases |= {m * ulp, (m + Fraction(1, 2)) * ulp, (m + Fraction(1, 3)) * ulp}   # exact, tie, below
            ulp = Fraction(2) ** (emin - prec + 1)          # denormals: 0 < x < 2^emin, the same ulp
            for m in (0, 1, 2, 3, 2 ** (prec - 1) - 1):
                cases |= {m * ulp, (m + Fraction(1, 2)) * ulp, (m + Fraction(1, 4)) * ulp}
            cases |= {Fraction(1, 2 ** j) for j in range(prec - emin + 3)}   # down to under the smallest denormal
            cases |= {Fraction(rng.randrange(-(1 << 40), 1 << 40), rng.randrange(1 << 40, 1 << 41))
                      for _ in range(400)}
            cases |= {-c for c in cases}
            for x in sorted(cases):
                for rtz in (False, True):
                    self.assertEqual(pr.round_frac(x, prec, emin, rtz), reference(x, prec, emin, rtz),
                                     (x, prec, emin, rtz))


class Kernels(unittest.TestCase):
    def test_descriptor_word3(self):
        w = pr.descriptor_word3(8)
        self.assertEqual([w & 7, w >> 3 & 7, w >> 6 & 7, w >> 9 & 7], [4, 5, 6, 7])
        self.assertEqual((w >> 12 & 0x7F, w >> 24 & 1, w >> 28 & 3, w >> 30), (8, 1, 3, 0))

    def test_every_value_is_staged_once(self):
        for fmt, (_, _, n) in pr.FORMATS.items():
            seen = []
            for inst, base in pr.jobs(fmt):
                if inst != "buffer_d16":
                    continue
                self.assertEqual(pr.kernel_lines(inst, fmt, base)[0], f"v_add_nc_u32 v1, 0x{base:x}, v0")
                seen += [(base + t) & ((1 << n) - 1) for t in range(min(pr.THREADS, (1 << n) - base))]
            self.assertEqual(seen, list(range(1 << n)), fmt)

    def test_mtbuf_names_the_format_and_mubuf_takes_it_from_the_descriptor(self):
        lines = pr.kernel_lines("tbuffer_d16", "snorm16", 0)
        self.assertIn("format:8 offen glc dlc", lines[10])
        self.assertTrue(pr.kernel_lines("buffer_d16", "snorm16", 0)[10].startswith("buffer_load_format_d16_x"))
        self.assertIn(f"s_mov_b32 s11, 0x{pr.descriptor_word3(8):x}", lines)

    def test_kernels_assemble_for_rdna2(self):
        try:
            clang = runner.find_clang()
        except runner.Unavailable as e:
            self.skipTest(str(e))
        kernels = [asm.Kernel(pr.kernel_lines(inst, fmt, 0), 32, pr.THREADS, pr.WORDS, raw=True, vgprs=8, sgprs=16)
                   for fmt in pr.FORMATS for inst in pr.INSTS]
        text, names, _ = asm.module_text(kernels, "gfx1031", wave=32)
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "k.s"
            s.write_text(text, encoding="utf-8")
            r = subprocess.run([clang, "--target=amdgcn-amd-amdhsa", "-mcpu=gfx1031", "-c", str(s), "-o",
                                str(Path(d) / "k.o")], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr[:800])
        self.assertIn("global_store_dword v4, v3, s[2:3]", text)
        self.assertIn("s_waitcnt_vscnt null, 0x0", text)


class FakeOracle:
    """Plays the GPU: reads each kernel's base, format code and load, and answers with a rule's bits."""
    arch = "gfx1031"
    device = "fake gfx1031"

    def __init__(self, rule="rte", clobber_hi=False, wave64_off=None, drop=0, none_at=None):
        self.rule, self.clobber_hi, self.wave64_off = rule, clobber_hi, wave64_off
        self.drop, self.none_at = drop, none_at   # answer `drop` fewer kernels; no result for kernel `none_at`

    def run(self, kernels, words=4):
        code_of = {c: (f, k, n) for f, (c, k, n) in pr.FORMATS.items()}
        out = []
        for k in kernels:
            base = int(re.search(r"0x([0-9a-f]+)", k.lines[0]).group(1), 16)
            w3 = int(next(ln for ln in k.lines if ln.startswith("s_mov_b32 s11")).split("0x")[1], 16)
            fmt, kind, n = code_of[w3 >> 12 & 0x7F]
            load = k.lines[10]
            staged = [(base + t) & ((1 << n) - 1) for t in range(pr.THREADS)]
            res = []
            for v in staged:
                if kind == "f16":
                    bits = v
                elif load.startswith("buffer_load_format_x"):
                    res.append(pr.models32(kind, n, v)["f32_rte"])
                    continue
                else:
                    bits = pr.models16(kind, n, v)[self.rule]
                    if self.wave64_off is not None and k.wave == 64 and v == self.wave64_off:
                        bits ^= 1
                hi = 0 if self.clobber_hi else pr.HI_SEED
                res.append(hi | bits)
            out.append(staged + res)
        if self.none_at is not None:
            out[self.none_at] = None
        return out[:len(out) - self.drop] if self.drop else out

    def close(self):
        pass


def run_main(oracle, *args):
    buf = io.StringIO()
    with mock.patch.object(runner, "open_oracle", return_value=oracle), redirect_stdout(buf):
        rc = pr.main(["--formats", "unorm8,float16", *args])
    return rc, buf.getvalue()


class Compare(unittest.TestCase):
    def test_the_translators_rule_matches_everywhere(self):
        rc, out = run_main(FakeOracle())
        self.assertEqual(rc, 0, out)
        self.assertIn("off the rule 0", out)
        self.assertIn("own bits 65536/65536", out)

    def test_double_rounding_hardware_shows_the_four_halfway_values(self):
        buf = io.StringIO()
        with mock.patch.object(runner, "open_oracle", return_value=FakeOracle("f32_rte")), redirect_stdout(buf):
            rc = pr.main(["--formats", "unorm16,snorm16", "--waves", "32"])
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertEqual(buf.getvalue().count("off the rule 2"), 4)          # MUBUF and MTBUF, two formats
        self.assertIn("value 0xffef: hw 0x3c00, rule 0x3bff", buf.getvalue())

    def test_rtz_hardware_is_a_finding_not_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            j = os.path.join(d, "r.json")
            rc, out = run_main(FakeOracle("f32_rtz"), "--json", j)
            with open(j, encoding="utf-8") as f:
                rep = json.load(f)
        self.assertEqual(rc, 0, out)
        d16 = rep["formats"]["unorm8"]["insts"]["buffer_d16"]
        self.assertEqual(d16["off_rule"], 123)
        self.assertEqual(d16["match"]["f32_rtz"], 256)
        self.assertIn("f32_rtz", d16["first_off"][0]["fits"])

    def test_a_wave_difference_exits_1(self):
        rc, out = run_main(FakeOracle(wave64_off=9))
        self.assertEqual(rc, 1, out)

    def test_a_short_result_list_is_a_harness_error(self):
        # BUILD-733: fewer results than kernels used to be zipped away, leaving values unchecked and exit 0.
        for drop in (1, 10_000):   # one kernel missing, then every result (an empty list)
            with self.subTest(drop=drop):
                rc, out = run_main(FakeOracle(drop=drop))
                self.assertEqual(rc, 2)
                self.assertRegex(out, r"harness error: \w+ wave32: the oracle returned \d+ results for \d+ kernels")

    def test_a_missing_result_is_a_harness_error(self):
        rc, out = run_main(FakeOracle(none_at=0))
        self.assertEqual(rc, 2)
        self.assertIn("base 0: no result", out)

    def test_a_clobbered_high_half_is_a_harness_error(self):
        rc, out = run_main(FakeOracle(clobber_hi=True))
        self.assertEqual(rc, 2, out)
        self.assertIn("high half", out)

    def test_not_gfx10_exits_77(self):
        o = FakeOracle()
        o.arch = "gfx1150"
        rc, out = run_main(o)
        self.assertEqual(rc, 77, out)


class Cli(unittest.TestCase):
    def test_exits_77_without_the_tools(self):
        env = dict(os.environ, LEONIDA_AMD_CLANG=str(TOOLS / "no-such-clang.exe"), LEONIDA_HW_DRM_NODE="")
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "d16_format_probe.py"), "--formats", "unorm8"],
                           capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 77, r.stdout + r.stderr)
        self.assertIn("unavailable", r.stdout)

    def test_unknown_format_is_a_usage_error(self):
        r = subprocess.run([sys.executable, "-I", str(TOOLS / "d16_format_probe.py"), "--formats", "unorm12"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("unorm12", r.stderr)


class OnHardware(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.oracle = runner.open_oracle()
        except runner.Unavailable as e:
            raise unittest.SkipTest(f"hardware oracle unavailable: {e}") from None
        if asm.family(cls.oracle.arch) != "gfx10":
            cls.oracle.close()
            raise unittest.SkipTest(f"{cls.oracle.arch} is not GFX10")

    @classmethod
    def tearDownClass(cls):
        cls.oracle.close()

    def test_unorm8_controls_hold_and_both_wave_sizes_agree(self):
        w32 = pr.sweep(self.oracle, "unorm8", 32)
        w64 = pr.sweep(self.oracle, "unorm8", 64)
        self.assertEqual(w32, w64)
        res = pr.classify("unorm8", w32)
        self.assertEqual(res["control"], [])
        self.assertEqual(w32["buffer_d16"][255] & 0xFFFF, 0x3C00)
        self.assertEqual(w32["buffer_d16"][0] & 0xFFFF, 0)
        self.assertEqual(w32["buffer_f32"][255], 0x3F800000)


class ReportPathTests(unittest.TestCase):
    """--json names a file; its folder is created and a write failure is one line, not a traceback."""

    def test_json_creates_a_missing_folder(self):
        with tempfile.TemporaryDirectory() as d:
            report = os.path.join(d, "reports", "deeper", "r.json")
            rc, out = run_main(FakeOracle(), "--json", report)
            self.assertEqual(rc, 0, out)
            self.assertTrue(os.path.isfile(report))

    def test_json_folder_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(SystemExit):
                run_main(FakeOracle(), "--json", d)

    def test_a_failed_write_is_one_line(self):
        with tempfile.TemporaryDirectory() as d:
            blocker = os.path.join(d, "not_a_folder")
            with open(blocker, "wb") as f:
                f.write(b"keep me")
            err = io.StringIO()
            with redirect_stderr(err):
                rc, _ = run_main(FakeOracle(), "--json", os.path.join(blocker, "r.json"))
            self.assertEqual(rc, 2)
            self.assertIn("ERROR: cannot write", err.getvalue())
            self.assertNotIn("Traceback", err.getvalue())
            with open(blocker, "rb") as f:
                self.assertEqual(f.read(), b"keep me")


if __name__ == "__main__":
    unittest.main()
