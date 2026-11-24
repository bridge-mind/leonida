"""Unit tests for tools/buffer_swizzle_probe.py (SHADER-343). Standard library only; no GPU or game files needed."""
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

import buffer_swizzle_probe as probe  # noqa: E402
from hw_oracle import runner as hw_runner  # noqa: E402


class BufferSwizzleProbeTests(unittest.TestCase):
    def test_cases_unique_and_nonempty(self):
        cases = probe.build_cases()
        self.assertGreaterEqual(len(cases), 20)
        names = [c.name for c in cases]
        self.assertEqual(len(names), len(set(names)))
        groups = {c.group for c in cases}
        self.assertEqual(groups, {
            "swizzle_basic", "idxen_ignores_tid", "swizzle_activation",
            "multi_dword_stepping", "typed_and_atomic_x2", "oob_mode3_soffset",
            "offset_wrap", "add_tid_wave_lane",
        })
        for key in probe.KNOWN_DIVERGENCES:
            self.assertIn(key, names)

    def test_hw_kernel_lines_wave32_and_wave64(self):
        by_name = {c.name: c for c in probe.build_cases()}
        c = by_name["tid_wave32_vs_wave64"]
        w32 = "\n".join(probe._hw_kernel_lines(c, 0, wave=32))
        w64 = "\n".join(probe._hw_kernel_lines(c, 1, wave=64))
        self.assertIn("s_mov_b32 s28, exec_lo", w32)
        self.assertIn("s_mov_b64 s[28:29], exec", w64)
        self.assertIn("s_barrier", w32)
        self.assertIn(".Lseed_0:", w32)
        self.assertIn(".Lseed_1:", w64)

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
        self.assertIn("oob_mode3_soffset", out.getvalue())

        with redirect_stderr(err):
            rc = probe.main(["--group", "does_not_exist"])
        self.assertEqual(rc, 2)
        self.assertIn("unknown group", err.getvalue())

    def test_hw_unavailable_returns_77(self):
        err = io.StringIO()
        with unittest.mock.patch.object(hw_runner, "open_oracle", side_effect=hw_runner.Unavailable("no GPU")):
            with redirect_stderr(err):
                rc = probe.main(["--oracle", "hw", "--group", "offset_wrap"])
        self.assertEqual(rc, 77)
        self.assertIn("no GPU", err.getvalue())

    def test_both_match_known_divergence_strict_and_json(self):
        cases = [c for c in probe.build_cases() if c.group == "oob_mode3_soffset"]
        tr = probe.run_translator(cases, waves=(32, 64))
        fake_hw = ("device=mock arch=gfx1036", {32: dict(tr[32]), 64: dict(tr[64])})
        with tempfile.TemporaryDirectory() as tmp:
            jpath = os.path.join(tmp, "out.json")
            out = io.StringIO()
            with unittest.mock.patch.object(probe, "run_hw", return_value=fake_hw):
                with redirect_stdout(out):
                    rc = probe.main(["--oracle", "both", "--group", "oob_mode3_soffset", "--json", jpath])
            self.assertEqual(rc, 0)
            self.assertIn("2 match, 0 known divergence, 0 unexpected mismatch, 0 wave32/64 diff", out.getvalue())
            data = json.loads(Path(jpath).read_text(encoding="utf-8"))
            self.assertEqual(data["device"], "device=mock arch=gfx1036")
            self.assertEqual(len(data["cases"]), 2)

            # Synthetic known divergence vs --strict
            fake_hw[1][32]["indexed_mode3_soff_edge"] = fake_hw[1][64]["indexed_mode3_soff_edge"] = [1, 2, 3, 4]
            known = {"indexed_mode3_soff_edge": ("synthetic", [1, 2, 3, 4], "test-only divergence")}
            with unittest.mock.patch.dict(probe.KNOWN_DIVERGENCES, known):
                out2 = io.StringIO()
                with unittest.mock.patch.object(probe, "run_hw", return_value=fake_hw):
                    with redirect_stdout(out2):
                        rc_ok = probe.main(["--oracle", "both", "--group", "oob_mode3_soffset"])
                self.assertEqual(rc_ok, 0)
                self.assertIn("1 match, 1 known divergence, 0 unexpected mismatch", out2.getvalue())

                out3 = io.StringIO()
                with unittest.mock.patch.object(probe, "run_hw", return_value=fake_hw):
                    with redirect_stdout(out3):
                        rc_strict = probe.main(["--oracle", "both", "--group", "oob_mode3_soffset", "--strict"])
                self.assertEqual(rc_strict, 1)

            err = io.StringIO()
            with unittest.mock.patch.object(probe, "run_hw", return_value=fake_hw):
                with redirect_stdout(io.StringIO()), redirect_stderr(err):
                    rc_dir = probe.main(["--oracle", "both", "--group", "oob_mode3_soffset", "--json", tmp])
            self.assertEqual(rc_dir, 2)

    def test_report_failure_names_the_output(self):
        group = "offset_wrap"
        result = {32: {case.name: [1, 2, 3, 4] for case in probe.build_cases() if case.group == group}}
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_bytes(b"keep")
            report = str(blocker / "report.json")
            with unittest.mock.patch.object(probe, "run_translator", return_value=result), \
                    redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as err:
                rc = probe.main(["--group", group, "--wave", "32", "--json", report])
            self.assertEqual(rc, 2)
            self.assertIn(f"cannot write {report}", err.getvalue())
            self.assertEqual(blocker.read_bytes(), b"keep")

    def test_unexpected_mismatch_and_wave_diff_fail(self):
        bad_hw = ("device=mock arch=gfx1036", {
            32: {"voff_inst_wrap_and_soff_nowrap": [1, 2, 3, 4]},
            64: {"voff_inst_wrap_and_soff_nowrap": [1, 2, 3, 5]},
        })
        out = io.StringIO()
        with unittest.mock.patch.object(probe, "run_hw", return_value=bad_hw):
            with redirect_stdout(out):
                rc = probe.main(["--oracle", "both", "--group", "offset_wrap"])
        self.assertEqual(rc, 1)
        self.assertIn("WAVE_DIFF", out.getvalue())


if __name__ == "__main__":
    unittest.main()
