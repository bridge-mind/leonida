"""A test program Windows refuses to start (Smart App Control blocks fresh unsigned builds: WinError 4551) is one
failed test with the reason, not a traceback that ends the run. The error is faked, so this runs on any OS."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
sys.path.insert(0, str(ROOT / "scripts"))
import check
import run_gpu_tests


def blocked(winerror=4551):
    """The OSError Python raises on Windows when Application Control refuses to start a program."""
    e = PermissionError(13, "An Application Control policy has blocked this file")
    e.winerror = winerror
    return e


class CheckPy(unittest.TestCase):
    def test_blocked_program_is_a_failed_test(self):
        with patch.object(check.subprocess, "run", side_effect=blocked()):
            code, out = check.run("/build/vk_compute_test.exe", "cs_lds.spv", "4", "mirror64")
        self.assertNotEqual(code, 0)
        self.assertNotEqual(code, check.UNSUPPORTED)          # a blocked program is not a device limit: no SKIP
        self.assertIn("could not start vk_compute_test.exe", out)
        self.assertIn("Application Control", out)

    def test_blocked_mesh_harness_is_a_failed_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("ms_w64x_ballot", "fs_passthru"):
                (Path(tmp) / f"{name}.spv").touch()
            with patch.object(check.subprocess, "run", side_effect=blocked()):
                problem = check.mesh_w64x_test(tmp, tmp, "ms_w64x_ballot")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("could not start vk_harness", problem)
        self.assertIn("Application Control", problem)

    def test_blocked_pixel_harness_gives_the_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "ps_ops.spv").touch()
            with patch.object(check.subprocess, "run", side_effect=blocked()):
                problem = check.pixel_test(tmp, tmp, "ps_ops")
        self.assertNotIsInstance(problem, check.Skip)
        self.assertIn("could not start vk_harness", problem)
        self.assertIn("Application Control", problem)

    def test_other_start_errors_quote_the_error(self):
        with patch.object(check.subprocess, "run", side_effect=OSError(8, "Exec format error")):
            code, out = check.run("/build/vk_harness", "a.spv")
        self.assertNotEqual(code, 0)
        self.assertIn("could not start vk_harness: [Errno 8] Exec format error", out)
        self.assertNotIn("Application Control", out)

    def test_missing_programs_still_say_how_to_build_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(check.subprocess, "run", side_effect=FileNotFoundError(2, "not found")):
                with self.assertRaises(SystemExit) as stop:
                    check.run(str(Path(tmp) / "vk_compute_test"), "cs_lds.spv")
        self.assertIn("Test programs not found", str(stop.exception))

    def test_each_windows_refusal_names_who_refused(self):
        for code, who in ((4551, "Application Control"), (1260, "software restriction policy"),
                          (225, "Microsoft Defender")):
            self.assertIn(who, check.start_failure("vk_store_test.exe", blocked(code)))


class RunGpuTestsPy(unittest.TestCase):
    def test_blocked_program_is_a_failed_test(self):
        with patch.object(run_gpu_tests.subprocess, "run", side_effect=blocked()):
            code, msg = run_gpu_tests.run_cmd(["C:/build/vk_compute_test.exe", "cs_bvh.spv", "1", "bvh"])
        self.assertNotEqual(code, 0)
        self.assertNotEqual(code, run_gpu_tests.UNSUPPORTED)
        self.assertIn("could not start vk_compute_test.exe", msg)
        self.assertIn("Application Control", msg)
        self.assertEqual(len(msg.splitlines()), 1)            # one result line in the summary table


if __name__ == "__main__":
    unittest.main()
