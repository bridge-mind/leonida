"""make_game.py: a missing vk_pipeline_check is reported in preflight, before a long shader build.

    python -I tests/test_make_game_pipeline_preflight.py

Standard library only. Every path is synthetic and every tool is mocked: no game files, GPU, Vulkan SDK or
build output is needed.
"""
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402


class PreflightEnv:
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = tmp.name
        self.work = os.path.join(self.root, "work")
        os.makedirs(self.work)
        # Satisfy every other prerequisite so only the checker under test can make preflight fail.
        for patcher in (mock.patch("sys.version_info", (3, 12, 0)),
                        mock.patch.dict(os.environ, {"LLVM_MC": "llvm-mc"}),
                        mock.patch("shutil.which", return_value="/bin/tool"),
                        mock.patch("shutil.disk_usage", return_value=mock.Mock(free=500 * 2**30))):
            patcher.start()
            self.addCleanup(patcher.stop)

    def write_shader_index(self):
        spirv = os.path.join(self.work, "shaders", "spirv")
        os.makedirs(spirv)
        with open(os.path.join(spirv, "index.json"), "w", encoding="utf-8") as f:
            f.write('{"shader_0": {}}')   # a finished translation is never empty

    def assert_preflight_fails(self, stages, match):
        err = io.StringIO()
        with mock.patch("sys.stderr", err):
            with self.assertRaises(SystemExit) as ctx:
                make_game.preflight(stages, self.work)
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn(match, err.getvalue())
        return err.getvalue()


class PipelinePreflightTest(PreflightEnv, unittest.TestCase):
    def test_pipelines_without_checker_fails_in_preflight(self):
        with mock.patch("os.name", "nt"), mock.patch.object(make_game, "pipeline_check_exe", return_value=None):
            err = self.assert_preflight_fails(["shaders", "pipelines"], "vk_pipeline_check")
        self.assertIn("Install these first, then run the build again:", err)

    def test_main_does_not_start_shaders_when_checker_is_missing(self):
        game = os.path.join(self.root, "game")
        out = os.path.join(self.root, "out")
        os.makedirs(game)
        argv = ["make_game.py", "--game", game, "--work", self.work, "--out", out,
                "--stages", "shaders,pipelines", "--no-tui"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("os.name", "nt"), \
                mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "game.exe"}), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value=None), \
                mock.patch.object(make_game, "shaders") as shaders, \
                mock.patch.object(make_game, "pipelines") as pipelines, \
                mock.patch.object(make_game, "log"), \
                mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                make_game.main()
        self.assertEqual(ctx.exception.code, 1)
        shaders.assert_not_called()
        pipelines.assert_not_called()

    def test_checker_already_built_still_requires_shaders(self):
        with mock.patch("os.name", "nt"), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value="vk_pipeline_check.exe"):
            self.assert_preflight_fails(["pipelines"], "No translated shaders yet")

    def test_checker_already_built_passes_preflight(self):
        self.write_shader_index()
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="/bin/vk_pipeline_check"):
            make_game.preflight(["pipelines"], self.work)

    @unittest.skipUnless(os.name == "posix", "requires POSIX execute permissions")
    def test_non_executable_linux_checker_without_cmake_fails_preflight(self):
        self.write_shader_index()
        checker_dir = os.path.join(self.root, "runtime", "build", "linux")
        os.makedirs(checker_dir)
        checker = os.path.join(checker_dir, "vk_pipeline_check")
        with open(checker, "wb"):
            pass
        os.chmod(checker, 0o644)
        with mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch("shutil.which", return_value=None), \
                mock.patch.object(make_game, "build_pipeline_check") as build:
            self.assert_preflight_fails(["pipelines"], "CMake 3.24 or newer")
        build.assert_not_called()

    def test_runtime_stage_builds_the_checker_so_it_is_not_required(self):
        # On Windows runtime produces vk_pipeline_check.exe when the Vulkan SDK is installed.
        self.write_shader_index()
        with mock.patch("os.name", "nt"), \
                mock.patch.dict(os.environ, {"VULKAN_SDK": r"C:\Vulkan"}), \
                mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=True), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value=None):
            make_game.preflight(["runtime", "pipelines"], self.work)

    def test_runtime_without_vulkan_sdk_does_not_build_the_checker(self):
        # scripts/build_runtime.ps1 builds vk_pipeline_check.exe only 'if defined VULKAN_SDK': a combined
        # Windows run without it would still spend hours in the shaders stage and fail in pipelines.
        for sdk in (None, ""):
            with self.subTest(sdk=sdk), mock.patch("os.name", "nt"), \
                    mock.patch.dict(os.environ), \
                    mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=True), \
                    mock.patch.object(make_game, "pipeline_check_exe", return_value=None):
                if sdk is None:
                    os.environ.pop("VULKAN_SDK", None)
                else:
                    os.environ["VULKAN_SDK"] = sdk
                err = self.assert_preflight_fails(["runtime", "pipelines"], "VULKAN_SDK")
                self.assertIn("vk_pipeline_check", err)

    def test_missing_checker_is_listed_with_other_missing_prerequisites(self):
        with mock.patch("os.name", "nt"), \
                mock.patch("shutil.which", side_effect=lambda t: None if t == "llvm-mc" else "/bin/tool"), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value=None):
            err = self.assert_preflight_fails(["shaders", "pipelines"], "vk_pipeline_check")
        self.assertIn("llvm-mc (LLVM 21+ with the AMDGPU target)", err)
        self.assertEqual(err.count("vk_pipeline_check"), 1)



class LinuxPipelineCheckTest(PreflightEnv, unittest.TestCase):
    """Off Windows preflight builds the checker itself once every other prerequisite is there."""

    def setUp(self):
        super().setUp()
        for patcher in (mock.patch("os.name", "posix"),
                        mock.patch.object(make_game, "pipeline_check_exe", return_value=None)):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_missing_shader_index_stops_before_building_checker(self):
        with mock.patch.object(make_game, "pipeline_check_exe", side_effect=[None, None, "/bin/vk_pipeline_check"]), \
                mock.patch.object(make_game, "build_pipeline_check") as build:
            self.assert_preflight_fails(["pipelines"], "No translated shaders yet")
        build.assert_not_called()

    def test_builds_the_checker_after_the_checks(self):
        self.write_shader_index()
        with mock.patch.object(make_game, "pipeline_check_exe", side_effect=[None, None, "/bin/vk_pipeline_check"]), \
                mock.patch.object(make_game, "build_pipeline_check") as build:
            make_game.preflight(["pipelines"], self.work, jobs=3)
        build.assert_called_once_with(3)

    def test_a_checker_built_where_it_isnt_looked_for_fails_in_preflight(self):
        with mock.patch.object(make_game, "build_pipeline_check") as build:
            self.assert_preflight_fails(["shaders", "pipelines"], "delete runtime/build/linux")
        build.assert_called_once()

    def test_missing_built_checker_does_not_initialize_asset_submodule(self):
        with mock.patch.object(make_game, "dotnet_sdk_ok", return_value=True), \
                mock.patch.object(make_game, "build_pipeline_check") as build, \
                mock.patch.object(make_game, "ensure_submodule") as initialize:
            self.assert_preflight_fails(["shaders", "assets", "pipelines"], "delete runtime/build/linux")
        build.assert_called_once()
        initialize.assert_not_called()

    def test_without_cmake_nothing_is_built(self):
        with mock.patch("shutil.which", side_effect=lambda t: None if t == "cmake" else "/bin/tool"), \
                mock.patch.object(make_game, "build_pipeline_check") as build:
            err = self.assert_preflight_fails(["pipelines"], "CMake 3.24 or newer")
        self.assertIn("vk_pipeline_check", err)
        build.assert_not_called()

    def test_another_missing_prerequisite_stops_before_the_build(self):
        with mock.patch("shutil.which", side_effect=lambda t: None if t == "llvm-mc" else "/bin/tool"), \
                mock.patch.object(make_game, "build_pipeline_check") as build:
            self.assert_preflight_fails(["shaders", "pipelines"], "llvm-mc")
        build.assert_not_called()

    def test_a_built_checker_is_not_rebuilt(self):
        self.write_shader_index()
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="/bin/vk_pipeline_check"), \
                mock.patch.object(make_game, "build_pipeline_check") as build:
            make_game.preflight(["pipelines"], self.work)
        build.assert_not_called()

    def test_the_build_uses_the_readme_folder(self):
        with mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch.object(make_game, "log"), \
                mock.patch.object(make_game, "run") as run:
            make_game.build_pipeline_check(2)
        configure, build = (c.args[0] for c in run.call_args_list)
        folder = os.path.join(self.root, "runtime", "build", "linux")
        self.assertEqual(configure[:5], ["cmake", "-S", os.path.join(self.root, "runtime"), "-B", folder])
        self.assertEqual(build, ["cmake", "--build", folder, "--parallel", "2", "--target", "vk_pipeline_check"])
        self.assertIn("Vulkan headers", run.call_args_list[1].kwargs["hint"])


if __name__ == "__main__":
    unittest.main()
