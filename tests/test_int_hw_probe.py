"""Unit tests for tools/int_hw_probe.py (SHADER-137). Standard library only; no GPU or game files needed."""
import io
import json
import os
import sys
import tempfile
import unittest
import unittest.mock
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
sys.dont_write_bytecode = True
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import int_hw_probe as probe  # noqa: E402
from hw_oracle import runner as hw_runner  # noqa: E402


class IntHwProbeTests(unittest.TestCase):
    def test_cases_unique_and_nonempty(self):
        cases = probe.build_cases()
        self.assertGreaterEqual(len(cases), 95)
        names = [c.name for c in cases]
        self.assertEqual(len(names), len(set(names)))
        groups = {c.group for c in cases}
        self.assertGreaterEqual(len(groups), 23)
        for key in probe.KNOWN_DIVERGENCES:
            self.assertIn(key, names)

    def test_wave32_and_wave64_placeholder_expansion(self):
        by_name = {c.name: c for c in probe.build_cases()}
        c = by_name["v_mad_u64_u32_inactive_lane"]
        w32 = "\n".join(c.lines(wave=32))
        w64 = "\n".join(c.lines(wave=64))
        self.assertIn("s_mov_b32 s14, exec_lo", w32)
        self.assertIn("v_mad_u64_u32 v[8:9], vcc_lo", w32)
        self.assertIn("s_mov_b64 s[14:15], exec", w64)
        self.assertIn("v_mad_u64_u32 v[8:9], vcc,", w64)
        self.assertNotIn("{", w32)
        self.assertNotIn("{", w64)

    def test_self_test_and_translator_groups(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = probe.main(["--self-test"])
        self.assertEqual(rc, 0)
        self.assertIn("self-test:", buf.getvalue())

    def test_list_groups_and_unknown_group(self):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out):
            rc = probe.main(["--list-groups"])
        self.assertEqual(rc, 0)
        self.assertIn("valu_cvt_pknorm", out.getvalue())

        with redirect_stderr(err):
            rc = probe.main(["--group", "does_not_exist"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown group", err.getvalue())

    def test_hw_unavailable_returns_77(self):
        err = io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", side_effect=hw_runner.Unavailable("no GPU")):
            with redirect_stderr(err):
                rc = probe.main(["--oracle", "hw", "--group", "salu_bfe_past31"])
        self.assertEqual(rc, 77)
        self.assertIn("no GPU", err.getvalue())

    def test_known_divergence_vs_strict_and_json_output(self):
        # Mock hardware oracle returning the measured gfx1035 words for valu_cvt_flr_rpi.
        measured = {
            "v_cvt_rpi_precision": [0x00000000, 0x00800001, 0x00FFFFFF, 0x00000000],
            "v_cvt_flr_rpi_nan_inf": [0x7FFFFFFF, 0x80000000, 0x7FFFFFFF, 0x80000000],
            "v_cvt_i32_f32_nan_control": [0x00000000, 0x00000000, 0x00000000, 0x00000000],
            "v_cvt_flr_edges": [0xFFFFFFFE, 0x7FFFFFFF, 0x80000000, 0x7FFFFFFF],
        }
        fake_hw = ("device=mock arch=gfx1035", {32: measured, 64: measured})
        with tempfile.TemporaryDirectory() as tmp:
            jpath = os.path.join(tmp, "out.json")
            out = io.StringIO()
            # The table is empty since #1831/#1856 fixed the measured divergences, so give the reporting path one: the
            # "hardware" returns other words for v_cvt_flr_edges, and the table lists them as a known divergence.
            fake_hw[1][32]["v_cvt_flr_edges"] = fake_hw[1][64]["v_cvt_flr_edges"] = [1, 2, 3, 4]
            known = {"v_cvt_flr_edges": ("synthetic", [1, 2, 3, 4], "test-only divergence")}
            patch_known = unittest.mock.patch.dict(probe.KNOWN_DIVERGENCES, known)
            patch_known.start()
            self.addCleanup(patch_known.stop)
            with unittest.mock.patch.object(probe, "run_hw", return_value=fake_hw):
                with redirect_stdout(out):
                    rc = probe.main(["--oracle", "both", "--group", "valu_cvt_flr_rpi", "--json", jpath])
            self.assertEqual(rc, 0)
            self.assertIn("3 match, 1 known divergence, 0 unexpected mismatch, 0 wave32/64 diff", out.getvalue())
            data = json.loads(Path(jpath).read_text(encoding="utf-8"))
            self.assertEqual(data["device"], "device=mock arch=gfx1035")
            self.assertEqual(len(data["cases"]), 4)

            out2 = io.StringIO()
            with unittest.mock.patch.object(probe, "run_hw", return_value=fake_hw):
                with redirect_stdout(out2):
                    rc_strict = probe.main(["--oracle", "both", "--group", "valu_cvt_flr_rpi", "--strict"])
            self.assertEqual(rc_strict, 1)

            err = io.StringIO()
            with unittest.mock.patch.object(probe, "run_hw", return_value=fake_hw):
                with redirect_stderr(err):
                    rc_dir = probe.main(["--oracle", "both", "--group", "valu_cvt_flr_rpi", "--json", tmp])
            self.assertEqual(rc_dir, 2)

    def test_unexpected_mismatch_and_wave_diff_fail(self):
        bad_hw = ("device=mock arch=gfx1035", {
            32: {"s_bfe_i32_past31": [1, 2, 3, 4], "s_bfe_u32_past31": [0, 0, 0, 0], "s_bfe_i32_width0": [0, 0, 0, 0]},
            64: {"s_bfe_i32_past31": [1, 2, 3, 5], "s_bfe_u32_past31": [0, 0, 0, 0], "s_bfe_i32_width0": [0, 0, 0, 0]},
        })
        out = io.StringIO()
        with unittest.mock.patch.object(probe, "run_hw", return_value=bad_hw):
            with redirect_stdout(out):
                rc = probe.main(["--oracle", "both", "--group", "salu_bfe_past31"])
        self.assertEqual(rc, 1)
        self.assertIn("WAVE_DIFF", out.getvalue())

    def test_json_preflight_rejects_bad_paths_before_hw_or_translator(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing_file = os.path.join(tmp, "not_a_dir")
            Path(existing_file).write_text("x", encoding="utf-8")
            bad_paths = [
                "",
                "   ",
                tmp,
                os.path.join(existing_file, "out.json"),
                os.path.join(existing_file, "nested", "out.json"),
            ]
            for bad in bad_paths:
                with self.subTest(json_path=bad):
                    err = io.StringIO()
                    with unittest.mock.patch.object(probe, "run_translator", side_effect=AssertionError("ran translator")), \
                         unittest.mock.patch.object(probe, "run_hw", side_effect=AssertionError("opened hw_oracle")), \
                         unittest.mock.patch.object(hw_runner, "open_oracle", side_effect=AssertionError("opened oracle")), \
                         redirect_stderr(err):
                        rc = probe.main(["--oracle", "both", "--group", "salu_bfe_past31", "--json", bad])
                    self.assertEqual(rc, 2)
                    self.assertIn("ERROR:", err.getvalue())

    def test_json_write_oserror_returns_2_without_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            jpath = os.path.join(tmp, "out.json")
            out, err = io.StringIO(), io.StringIO()
            with unittest.mock.patch.object(probe, "run_translator", return_value={"s_bfe_i32_past31": [0], "s_bfe_u32_past31": [0], "s_bfe_i32_width0": [0]}), \
                 unittest.mock.patch("builtins.open", side_effect=OSError("disk full")), \
                 redirect_stdout(out), redirect_stderr(err):
                rc = probe.main(["--oracle", "translator", "--group", "salu_bfe_past31", "--json", jpath])
            self.assertEqual(rc, 2)
            self.assertIn("ERROR:", err.getvalue())
            self.assertIn("disk full", err.getvalue())


# v_bfe_u32 / v_bfe_i32 words measured natively on an AMD Radeon 610M (gfx1036, RDNA2), wave32 and wave64 identical:
# `python -I tools/int_hw_probe.py --oracle both --group valu_bfe_past31`. A field that would run past bit 31 stops
# there and v_bfe_i32 sign-extends from bit 31 (0xd2f90159, offset 28, width 16 gives 0xfffffffd, not 0xd); offset
# and width are read from bits [4:0] only (offset 36 is 4, width 32 is 0).
VALU_BFE_GFX1036 = {
    "v_bfe_past31_vgpr": [0xFFFFFFFD, 0x0000000D, 0xFFFFFFFD, 0x0000000D],
    "v_bfe_past31_sign": [0x00000001, 0xFFFFFFFF, 0x00000000, 0xFFFFFFFF],
    "v_bfe_high_bits": [0x00000015, 0x00000000, 0xFFFFFFFD, 0x00000000],
    "v_bfe_past31_const": [0xFFFFFFFD, 0x0000000D, 0xFFFFFFFD, 0x00000000],
}


class ValuBfeMeasuredOnRdna2(unittest.TestCase):
    def test_translator_gives_the_gfx1036_words(self):
        cases = [c for c in probe.build_cases() if c.group == "valu_bfe_past31"]
        self.assertEqual(sorted(c.name for c in cases), sorted(VALU_BFE_GFX1036))
        got = probe.run_translator(cases)
        for name, want in VALU_BFE_GFX1036.items():
            with self.subTest(case=name):
                self.assertEqual(got[name], want)


if __name__ == "__main__":
    unittest.main()
