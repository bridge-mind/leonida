"""run_gpu_tests.py's cs_wavemask, with a fake vk_compute_test: it asks for a 64-wide subgroup only for the native
build. With LEONIDA_W64_EMULATE=1 it runs on any subgroup size, as check.py does (W64_EMULATED)."""
import contextlib
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests

PASS32 = "device: fake (subgroup 32..32)\n1 groups x 64: 0 mismatches"


class WavemaskCheck(unittest.TestCase):
    def run_check(self, emulated, rc, msg):
        calls = []

        def fake(cmd, env=None):
            calls.append(cmd)
            return rc, msg

        with patch.object(run_gpu_tests, "run_cmd", side_effect=fake):
            result = run_gpu_tests.wavemask_check("vk_compute_test", "cs_wavemask.spv", None, emulated)
        self.assertEqual(len(calls), 1)
        return result, calls[0]

    def test_native_build_requires_wave64_and_skips_without_it(self):
        result, cmd = self.run_check(False, 77, "device: fake (subgroup 32..32)")
        self.assertIn("--require-wave64", cmd)
        self.assertIsNone(result[0])
        self.assertIn("SKIP", result[1])

    def test_native_build_needs_the_subgroup_confirmed(self):
        result, _ = self.run_check(False, 0, PASS32)
        self.assertIs(result[0], False)

    def test_native_build_passes_on_a_64_wide_subgroup(self):
        result, _ = self.run_check(False, 0, "compute subgroup size: 64\n1 groups x 64: 0 mismatches")
        self.assertIs(result[0], True)

    def test_emulated_build_runs_on_a_32_wide_subgroup(self):
        result, cmd = self.run_check(True, 0, PASS32)
        self.assertNotIn("--require-wave64", cmd)
        self.assertEqual(result, (True, "1 groups x 64: 0 mismatches"))

    def test_emulated_build_reports_mismatches(self):
        result, _ = self.run_check(True, 1, "device: fake (subgroup 32..32)\nlane 40: got 0, want 1")
        self.assertIs(result[0], False)



class WavemaskInTheSuite(unittest.TestCase):
    """run_suite runs cs_wavemask once and reports it once (no assembler or GPU needed)."""

    def test_one_run_and_one_row(self):
        compute_runs = []

        def fake_run_cmd(cmd, env=None, gpu=True):
            if cmd[0] == sys.executable:                    # the translator: leave the .spv it would write
                Path(cmd[cmd.index("--out") + 1], "cs_wavemask.spv").touch()
                return 0, ""
            compute_runs.append(cmd)
            return 77, "device: fake (subgroup 32..32)"

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(filter="cs_wavemask", llvm_mc="llvm-mc", build=tmp)
            with (patch.object(run_gpu_tests, "assemble_to_s",   # run_suite copies the .s it writes (#769)
                               side_effect=lambda _mc, _src, dst, _name: Path(dst).touch()),
                  patch.object(run_gpu_tests, "spirv_as"),
                  patch.object(run_gpu_tests, "w64_emulated", return_value=False),
                  patch.object(run_gpu_tests, "run_cmd", side_effect=fake_run_cmd), contextlib.redirect_stdout(out)):
                self.assertEqual(run_gpu_tests.run_suite(args, tmp), 0)
        self.assertEqual(len(compute_runs), 1, compute_runs)
        rows = [line for line in out.getvalue().splitlines() if "cs_wavemask" in line]
        self.assertEqual(len(rows), 1, out.getvalue())
        self.assertIn("SKIP (no 64-wide compute subgroup on this device)", rows[0])
        self.assertIn("0/1 passed, 1 skipped", out.getvalue())

if __name__ == "__main__":
    unittest.main()
