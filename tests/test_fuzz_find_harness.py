"""The fuzzers find vk_compute_test in a Visual Studio build, and say so in one line when it isn't built."""
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "test" / "shaders"))
from compute_fixture import find_test_program  # noqa: E402

EXE = "vk_compute_test" + (".exe" if os.name == "nt" else "")


class FindTestProgram(unittest.TestCase):
    def test_single_config_build(self):
        with tempfile.TemporaryDirectory() as build:
            (Path(build) / EXE).touch()
            self.assertEqual(find_test_program(build, "vk_compute_test"), build)

    def test_visual_studio_config_folders(self):
        for config in ("RelWithDebInfo", "Release"):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as build:
                (Path(build) / config).mkdir()
                (Path(build) / config / EXE).touch()
                self.assertEqual(find_test_program(build, "vk_compute_test"), os.path.join(build, config))

    def test_not_built(self):
        with tempfile.TemporaryDirectory() as build:
            self.assertIsNone(find_test_program(build, "vk_compute_test"))


class FuzzerWithoutHarness(unittest.TestCase):
    def test_one_line_instead_of_a_traceback(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import lds_fuzz
        import valu_fuzz
        for module in (valu_fuzz, lds_fuzz):
            with self.subTest(tool=module.__name__), tempfile.TemporaryDirectory() as build:
                with self.assertRaises(SystemExit) as stop:
                    module.run_gpu(None, "unused.spv", build, {}, False, "unused.bin")
                self.assertIn("vk_compute_test not found", str(stop.exception.code))


if __name__ == "__main__":
    unittest.main()
