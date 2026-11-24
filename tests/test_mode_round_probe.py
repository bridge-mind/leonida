"""CPU-only unit tests for tools/mode_round_probe.py (SHADER-132)."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(REPO, "tools")
PROBE = os.path.join(TOOLS, "mode_round_probe.py")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import mode_round_probe as probe  # noqa: E402
from hw_oracle import asm as hw_asm, runner  # noqa: E402


class ModeRoundProbeTests(unittest.TestCase):
    def test_self_test(self):
        proc = subprocess.run([sys.executable, "-I", PROBE, "--self-test"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("self-test OK", proc.stdout)

    def test_missing_oracle_exits_77(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        proc = subprocess.run([sys.executable, "-I", PROBE], env=env,
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 77, proc.stderr or proc.stdout)

    def test_invalid_waves_fail_before_opening_oracle(self):
        for bad_waves in ("", "   ", "abc", "16", "32,16", "32,foo"):
            with self.subTest(waves=bad_waves):
                err = io.StringIO()
                with mock.patch.object(runner, "open_oracle") as mock_open:
                    with mock.patch.object(sys, "stderr", err):
                        rc = probe.main(["--waves", bad_waves])
                self.assertEqual(rc, 2)
                mock_open.assert_not_called()
                self.assertIn("error:", err.getvalue())

    def test_invalid_json_paths_fail_before_opening_oracle(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            existing_dir = root / "outdir"
            existing_dir.mkdir()
            regular_file = root / "notadir.txt"
            regular_file.write_text("x", encoding="utf-8")

            bad_paths = (
                "",
                "   ",
                "bad\x00name.json",
                str(existing_dir),
                str(root / "trailing") + os.sep,
                str(regular_file / "nested" / "report.json"),
            )
            for bad in bad_paths:
                with self.subTest(json_out=bad):
                    err = io.StringIO()
                    with mock.patch.object(runner, "open_oracle") as mock_open:
                        with mock.patch.object(sys, "stderr", err):
                            rc = probe.main(["--waves", "32", "--json", bad])
                    self.assertEqual(rc, 2)
                    mock_open.assert_not_called()
                    self.assertIn("error:", err.getvalue())

    def test_valid_nested_json_does_not_create_dirs_when_oracle_unavailable(self):
        with tempfile.TemporaryDirectory() as d:
            nested_dir = Path(d) / "does_not_exist" / "sub"
            out_json = nested_dir / "report.json"
            err = io.StringIO()
            with mock.patch.object(runner, "open_oracle", side_effect=runner.Unavailable("no gpu")):
                with mock.patch.object(sys, "stderr", err):
                    rc = probe.main(["--waves", "32", "--json", str(out_json)])
            self.assertEqual(rc, 77)
            self.assertFalse((Path(d) / "does_not_exist").exists())

    def test_write_json_report_creates_parents_and_writes_lf_json(self):
        with tempfile.TemporaryDirectory() as d:
            out_json = Path(d) / "nested" / "sub" / "report.json"
            err = probe.write_json_report(str(out_json), {"arch": "gfx1030", "waves": {"32": {"ok": 1}}})
            self.assertIsNone(err)
            self.assertTrue(out_json.is_file())
            raw = out_json.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            self.assertEqual(json.loads(raw.decode("utf-8")), {"arch": "gfx1030", "waves": {"32": {"ok": 1}}})

    def test_json_write_oserror_returns_exit_1_without_traceback(self):
        fake_oracle = mock.MagicMock()
        fake_oracle.arch = "gfx1030"
        with tempfile.TemporaryDirectory() as d:
            out_json = Path(d) / "report.json"
            err = io.StringIO()
            with mock.patch.object(runner, "open_oracle", return_value=fake_oracle), \
                 mock.patch.object(probe, "probe_cases", return_value={"total": 1, "ok": 1, "mismatches": []}), \
                 mock.patch.object(probe, "probe_transcendentals", return_value={"all_ok": True, "ops": {}}), \
                 mock.patch("builtins.open", side_effect=PermissionError("denied")), \
                 mock.patch.object(sys, "stderr", err):
                rc = probe.main(["--waves", "32", "--json", str(out_json)])
            self.assertEqual(rc, 1)
            self.assertIn("denied", err.getvalue())


class FakeModeOracle:
    """Answers the probe's kernels from the tables in tools/mode_round_probe.py, like a GFX10 GPU that matches them,
    except that `bad_op` gets `diffs` nonzero RNE/RTZ XOR words in its first 1024-input chunk."""
    arch = "gfx1030"

    def __init__(self, bad_op=None, diffs=0):
        self.bad_op, self.diffs = bad_op, diffs

    def run(self, programs, words=4):
        out = []
        for k in programs:
            if k.words == 1024:
                op = next(t[0] for t in probe.TRANSCENDENTALS if f"{t[0]} v2, v1" in k.lines)
                chunk = [0] * 1024
                if op == self.bad_op and k.lines[0] == "v_add_nc_u32 v1, 0x0, v0":
                    chunk[:self.diffs] = [1] * self.diffs
                out.append(chunk)
                continue
            rtz = probe.RTZ in k.lines
            case = next((c for c in probe.CASES
                         if k.lines[:len(c[1]) + 3] == [*c[1], probe.RTZ if rtz else probe.RNE, "s_nop 0", c[2]]), None)
            if case is not None:
                want = case[5] if rtz else case[4]
                out.append(list(want) if case[3] else [want, 0])
                continue
            spot = next(t for t in probe.TRANSCENDENTALS if f"{t[0]} v3, v1" in k.lines)
            out.append([spot[2], 0])
        return out


class SummaryLineTests(unittest.TestCase):
    """The per-wave summary line reports what the oracle returned, not a fixed '7/7 ... 0/65536'."""

    def run_main(self, oracle):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(runner, "open_oracle", return_value=oracle),              mock.patch.object(hw_asm, "family", return_value="gfx10"),              mock.patch.object(sys, "stdout", out), mock.patch.object(sys, "stderr", err):
            rc = probe.main(["--waves", "32"])
        return rc, out.getvalue(), err.getvalue()

    def test_all_matching_reports_every_transcendental(self):
        rc, out, err = self.run_main(FakeModeOracle())
        self.assertEqual(rc, 0, err)
        self.assertIn(f"{len(probe.CASES)}/{len(probe.CASES)} f16/f64 RNE/RTZ cases exact", out)
        self.assertIn("7/7 f16 transcendentals 0/65536 RNE/RTZ diffs", out)

    def test_mode_dependent_transcendental_is_reported(self):
        rc, out, err = self.run_main(FakeModeOracle(bad_op="v_sin_f16_e32", diffs=3))
        self.assertEqual(rc, 1, err)
        self.assertNotIn("7/7 f16 transcendentals", out)
        self.assertIn("6/7 f16 transcendentals 0/65536 RNE/RTZ diffs", out)
        self.assertIn("v_sin_f16_e32: 3/65536 RNE/RTZ diffs", out)


if __name__ == "__main__":
    unittest.main()
