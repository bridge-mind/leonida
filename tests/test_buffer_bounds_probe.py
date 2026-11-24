"""CPU-only unit tests for tools/buffer_bounds_probe.py (SHADER-341)."""
import os
import subprocess
import sys
import tempfile
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(REPO, "tools", "buffer_bounds_probe.py")


class BufferBoundsProbeTests(unittest.TestCase):
    def test_self_test(self):
        proc = subprocess.run([sys.executable, "-I", PROBE, "--self-test"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        self.assertIn("self-test OK", proc.stdout)

    def test_non_gfx10_or_missing_oracle_exits_77(self):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        proc = subprocess.run([sys.executable, "-I", PROBE], env=env,
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 77, proc.stderr or proc.stdout)

    def test_bad_waves(self):
        proc = subprocess.run([sys.executable, "-I", PROBE, "--waves", "16"],
                              capture_output=True, text=True, check=False)
        self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)

    def test_json_folder_is_refused_before_the_gpu_run(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = subprocess.run([sys.executable, "-I", PROBE, "--json", folder],
                                  env=self.no_oracle_env(), capture_output=True, text=True,
                                  check=False)
            self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)
            self.assertIn("ERROR: --json", proc.stderr)
            self.assertIn("is a directory", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)

    def test_json_in_a_missing_folder_is_refused_before_the_gpu_run(self):
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "missing", "report.json")
            proc = subprocess.run([sys.executable, "-I", PROBE, "--json", target],
                                  env=self.no_oracle_env(), capture_output=True, text=True,
                                  check=False)
            self.assertEqual(proc.returncode, 2, proc.stderr or proc.stdout)
            self.assertIn("ERROR: --json", proc.stderr)
            self.assertIn("does not exist", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse(os.path.exists(target))

    def no_oracle_env(self):
        """An env with no hardware oracle, so a missing guard shows up as the 77 skip rather than a sweep."""
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("LEONIDA_HW_") and k != "LEONIDA_AMD_CLANG"}
        env["LEONIDA_AMD_CLANG"] = os.path.join(REPO, "does_not_exist_clang")
        return env


if __name__ == "__main__":
    unittest.main()
