"""run_gpu_tests.py: a filtered run still builds the vertex shader a selected test draws with, and relative --build
/--work paths keep working for test programs that run in another working directory (vk_texture_test)."""
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gpu_tests


class DrawShaders(unittest.TestCase):
    def test_texture_sample_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_texture_sample.asm"]),
                         ["ps_texture_sample.asm", "vs_ids.asm"])

    def test_cube_sample_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_cube_sample.asm"]),
                         ["ps_cube_sample.asm", "vs_ids.asm"])

    def test_flat_still_brings_vs_ids(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_flat.asm"]), ["ps_flat.asm", "vs_ids.asm"])

    def test_each_filtered_clip_case_brings_ps_clip_color(self):
        for name in run_gpu_tests.CLIP_CASES:
            with self.subTest(name=name):
                selected = f"{name}.asm"
                self.assertEqual(run_gpu_tests.with_draw_shaders([selected]), [selected, "ps_clip_color.asm"])

    def test_vs_ids_is_not_added_twice(self):
        files = ["ps_flat.asm", "ps_texture_sample.asm", "vs_ids.asm"]
        self.assertEqual(run_gpu_tests.with_draw_shaders(files), files)

    def test_other_shaders_add_nothing(self):
        self.assertEqual(run_gpu_tests.with_draw_shaders(["ps_ops.asm"]), ["ps_ops.asm"])

    def test_every_shader_named_exists(self):
        shaders = Path(run_gpu_tests.SHADER_DIR)
        for name, extra in run_gpu_tests.DRAWN_WITH.items():
            # the clip cases (#193) are variants of vs_clip.asm with different signatures, not files of their own
            source = "vs_clip" if name in run_gpu_tests.CLIP_CASES else name
            self.assertTrue((shaders / f"{source}.asm").is_file(), name)
            self.assertTrue((shaders / extra).is_file(), extra)


class FilterSelection(unittest.TestCase):
    def test_reused_work_translates_only_selected_shaders_and_dependencies(self):
        passes = []

        def write_disassembly(_llvm, _asm, path, name):
            Path(path).write_text(name, encoding="utf-8")

        def translate(cmd, gpu=True):
            inputs = Path(cmd[3])
            files = sorted(inputs.glob("*.s"))
            passes.append([path.name for path in files])
            for path in files:
                self.assertEqual(path.read_text(encoding="utf-8"), path.stem)
                self.assertTrue((inputs.parent / f"{path.stem}.dxbc").is_file())
            return 1, "stop before running GPU tests"

        with tempfile.TemporaryDirectory() as tmp, \
                patch("sys.stdout", io.StringIO()), \
                patch.object(run_gpu_tests, "assemble_to_s", side_effect=write_disassembly), \
                patch.object(run_gpu_tests, "run_cmd", side_effect=translate):
            for shader_filter in ("ps_bary", "ps_flat"):
                args = SimpleNamespace(filter=shader_filter, llvm_mc="unused")
                self.assertEqual(run_gpu_tests.run_suite(args, tmp), 1)

            self.assertEqual(passes, [["ps_bary.s"], ["ps_flat.s", "vs_ids.s"]])
            disasm = Path(tmp) / "disasm"
            self.assertEqual((disasm / "ps_bary.s").read_text(encoding="utf-8"), "ps_bary")
            self.assertTrue((disasm / "ps_flat.s").is_file())
            self.assertTrue((disasm / "vs_ids.s").is_file())

    def test_unmatched_filter_fails_before_running_shader_tools(self):
        args = SimpleNamespace(filter="no_matching_shader", build="unused")
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, \
                patch("sys.stderr", err), \
                patch.object(run_gpu_tests, "run_cmd", return_value=(0, "")) as run, \
                patch.object(run_gpu_tests, "spirv_as") as assemble:
            work = Path(tmp) / "work"
            self.assertEqual(run_gpu_tests.run_suite(args, str(work)), 2)
            self.assertIn(args.filter, err.getvalue())
            run.assert_not_called()
            assemble.assert_not_called()
            self.assertFalse(work.exists())


class RelativePaths(unittest.TestCase):
    def test_build_and_work_become_absolute(self):
        ext = ".exe" if os.name == "nt" else ""
        seen = {}

        def suite(a, work):
            seen["build"], seen["work"] = a.build, work
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build"
            build.mkdir()
            for tool in ("vk_compute_test", "vk_store_test", "vk_harness"):
                (build / (tool + ext)).touch()
            old = os.getcwd()
            os.chdir(tmp)
            try:
                with (patch.object(sys, "argv", ["run_gpu_tests.py", "--build", "build", "--work", "work"]),
                      patch.object(run_gpu_tests, "run_suite", side_effect=suite)):
                    self.assertEqual(run_gpu_tests.main(), 0)
            finally:
                os.chdir(old)
            # Compare resolved paths: Windows can spell the temp directory with an 8.3 name (RUNNER~1).
            for got, want in ((seen["build"], build), (seen["work"], Path(tmp) / "work")):
                self.assertTrue(os.path.isabs(got), got)
                self.assertEqual(Path(got).resolve(), want.resolve())


class WorkDirectory(unittest.TestCase):
    def test_file_blocking_work_directory_is_a_cli_error(self):
        ext = ".exe" if os.name == "nt" else ""
        for blocked_child in ("", "spirv"):
            with self.subTest(blocked_child=blocked_child), tempfile.TemporaryDirectory() as tmp:
                build = Path(tmp) / "build"
                build.mkdir()
                for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                    (build / (tool + ext)).touch()
                work = Path(tmp) / "work"
                blocker = work / blocked_child
                blocker.parent.mkdir(exist_ok=True)
                blocker.write_text("keep me", encoding="utf-8")

                result = subprocess.run(
                    [sys.executable, "-I", run_gpu_tests.__file__, "--build", str(build),
                     "--work", str(work), "--filter", "cs_lds"],
                    capture_output=True, text=True, timeout=10,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("error: cannot prepare GPU test directory", result.stderr)
                self.assertIn(str(blocker), result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(blocker.read_text(encoding="utf-8"), "keep me")


class BuildConfigurations(unittest.TestCase):
    def test_single_configuration_directory_is_preferred(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for directory in (root, root / "RelWithDebInfo"):
                directory.mkdir(exist_ok=True)
                for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                    (directory / (tool + ".exe")).touch()
            self.assertEqual(run_gpu_tests.find_test_programs(str(root), ".exe"), str(root))

    def test_visual_studio_relwithdebinfo_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            config = build / "RelWithDebInfo"
            config.mkdir()
            for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                (config / (tool + ".exe")).touch()
            self.assertEqual(run_gpu_tests.find_test_programs(str(build), ".exe"), str(config))

    def test_visual_studio_release_is_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            config = build / "Release"
            config.mkdir()
            for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                (config / (tool + ".exe")).touch()
            self.assertEqual(run_gpu_tests.find_test_programs(str(build), ".exe"), str(config))

    def test_incomplete_configuration_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            incomplete = build / "RelWithDebInfo"
            release = build / "Release"
            incomplete.mkdir()
            release.mkdir()
            for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS[:-1]:
                (incomplete / (tool + ".exe")).touch()
            for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                (release / (tool + ".exe")).touch()
            self.assertEqual(run_gpu_tests.find_test_programs(str(build), ".exe"), str(release))

    def test_missing_programs_returns_no_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(run_gpu_tests.find_test_programs(tmp, ".exe"))


    def test_relwithdebinfo_is_preferred_over_release(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            for config_name in run_gpu_tests.BUILD_CONFIGS:
                config = build / config_name
                config.mkdir()
                for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                    (config / (tool + ".exe")).touch()
            self.assertEqual(run_gpu_tests.find_test_programs(str(build), ".exe"),
                             str(build / "RelWithDebInfo"))

    def test_extensionless_executables_are_supported(self):
        with tempfile.TemporaryDirectory() as tmp:
            for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                (Path(tmp) / tool).touch()
            self.assertEqual(run_gpu_tests.find_test_programs(tmp, ""), tmp)

    def test_programs_are_not_combined_across_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp)
            (build / "RelWithDebInfo").mkdir()
            (build / "Release").mkdir()
            for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS[:2]:
                (build / "RelWithDebInfo" / (tool + ".exe")).touch()
            for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS[2:]:
                (build / "Release" / (tool + ".exe")).touch()
            self.assertIsNone(run_gpu_tests.find_test_programs(str(build), ".exe"))

    def test_missing_program_error_lists_per_directory_files(self):
        ext = ".exe" if os.name == "nt" else ""
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / "build output"
            rel = build / "RelWithDebInfo"
            rel.mkdir(parents=True)
            (rel / ("vk_compute_test" + ext)).touch()
            (rel / ("vk_store_test" + ext)).touch()
            with (patch.object(sys, "argv", ["run_gpu_tests.py", "--build", str(build)]),
                  self.assertRaises(SystemExit) as caught):
                run_gpu_tests.main()
            message = str(caught.exception)
            # names the build folder, says the configuration subfolders were searched, and lists what is missing
            self.assertIn(str(build), message)
            self.assertIn("RelWithDebInfo", message)
            self.assertIn("vk_harness" + ext, message)

    def test_main_passes_visual_studio_config_directory_to_suite(self):
        ext = ".exe" if os.name == "nt" else ""
        for config_name in run_gpu_tests.BUILD_CONFIGS:
            with self.subTest(config=config_name), tempfile.TemporaryDirectory() as tmp:
                build = Path(tmp) / "build"
                config = build / config_name
                config.mkdir(parents=True)
                for tool in run_gpu_tests.REQUIRED_TEST_PROGRAMS:
                    (config / (tool + ext)).touch()
                seen = {}

                def suite(a, work):
                    seen["build"] = a.build
                    return 0

                with (patch.object(sys, "argv", [
                          "run_gpu_tests.py", "--build", str(build), "--work", str(Path(tmp) / "work")]),
                      patch.object(run_gpu_tests, "run_suite", side_effect=suite)):
                    self.assertEqual(run_gpu_tests.main(), 0)
                self.assertEqual(Path(seen["build"]).resolve(), config.resolve())


if __name__ == "__main__":
    unittest.main()
