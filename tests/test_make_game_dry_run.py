from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game


class DryRunFlagsTest(unittest.TestCase):
    """Test dry run flags, discovery modes, and flag validation in make_game.py."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game_dir = os.path.join(self.tmp.name, "game")
        self.out_dir = os.path.join(self.tmp.name, "out")
        self.work_dir = os.path.join(self.tmp.name, "work")
        os.makedirs(self.game_dir, exist_ok=True)

    def test_dry_run_check_flag_exits_cleanly_without_invoking_build_stages(self) -> None:
        dummy_version = {
            "title": "Grand Theft Auto VI",
            "version": "1.0.2.1",
            "files": {},
            "expected_exe": "game_scarlett_master.exe",
        }
        argv = ["make_game.py", "--game", self.game_dir, "--check"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("make_game.below_normal"), \
                mock.patch("make_game.check", return_value=dummy_version) as mock_check, \
                mock.patch("make_game.preflight") as mock_preflight, \
                mock.patch("make_game.shaders") as mock_shaders, \
                mock.patch("make_game.assets") as mock_assets, \
                mock.patch("make_game.runtime") as mock_runtime, \
                mock.patch("make_game.pipelines") as mock_pipelines, \
                mock.patch("make_game.log"):
            ret = make_game.main()
            self.assertEqual(ret, 0)
            mock_check.assert_called_once_with(os.path.abspath(self.game_dir), False)
            mock_preflight.assert_not_called()
            mock_shaders.assert_not_called()
            mock_assets.assert_not_called()
            mock_runtime.assert_not_called()
            mock_pipelines.assert_not_called()
            self.assertFalse(os.path.exists(self.out_dir))
            self.assertFalse(os.path.exists(self.work_dir))

    def test_dry_run_full_check_passes_full_flag_and_halts_before_build(self) -> None:
        dummy_version = {
            "title": "Grand Theft Auto VI",
            "version": "1.0.2.1",
            "files": {},
            "expected_exe": "game_scarlett_master.exe",
        }
        argv = ["make_game.py", "--game", self.game_dir, "--check", "--full-check"]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("make_game.below_normal"), \
                mock.patch("make_game.check", return_value=dummy_version) as mock_check, \
                mock.patch("make_game.preflight") as mock_preflight, \
                mock.patch("make_game.shaders") as mock_shaders, \
                mock.patch("make_game.assets") as mock_assets, \
                mock.patch("make_game.runtime") as mock_runtime, \
                mock.patch("make_game.pipelines") as mock_pipelines, \
                mock.patch("make_game.log"):
            ret = make_game.main()
            self.assertEqual(ret, 0)
            mock_check.assert_called_once_with(os.path.abspath(self.game_dir), True)
            mock_preflight.assert_not_called()
            mock_shaders.assert_not_called()
            mock_assets.assert_not_called()
            mock_runtime.assert_not_called()
            mock_pipelines.assert_not_called()

    def test_dry_run_check_failure_aborts_immediately_without_side_effects(self) -> None:
        argv = ["make_game.py", "--game", self.game_dir, "--check"]
        stderr_buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("make_game.below_normal"), \
                mock.patch("make_game.load_version", side_effect=SystemExit(1)), \
                mock.patch("sys.stderr", stderr_buf), \
                mock.patch("make_game.preflight") as mock_preflight, \
                mock.patch("make_game.shaders") as mock_shaders:
            with self.assertRaises(SystemExit) as ctx:
                make_game.main()
            self.assertEqual(ctx.exception.code, 1)
            mock_preflight.assert_not_called()
            mock_shaders.assert_not_called()

    def test_dry_run_find_dumps_mode_scans_without_invoking_check_or_build(self) -> None:
        found_sample = [
            (os.path.join(self.tmp.name, "dump1"), {"title": "Title 1", "version": "1.0.2.1"}),
            (os.path.join(self.tmp.name, "dump2"), {"title": "Title 2", "version": "1.0.2.1"}),
        ]
        argv = ["make_game.py", "--find"]
        stdout_buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("make_game.below_normal"), \
                mock.patch("make_game.find_dumps", return_value=found_sample) as mock_find, \
                mock.patch("sys.stdout", stdout_buf), \
                mock.patch("make_game.check") as mock_check, \
                mock.patch("make_game.preflight") as mock_preflight:
            ret = make_game.main()
            self.assertEqual(ret, 0)
            mock_find.assert_called_once()
            mock_check.assert_not_called()
            mock_preflight.assert_not_called()
            output = stdout_buf.getvalue()
            self.assertIn("Title 1 1.0.2.1", output)
            self.assertIn("Title 2 1.0.2.1", output)
            self.assertIn("2 found", output)

    def test_explicit_dry_run_flag_validation(self) -> None:
        argv = ["make_game.py", "--dry-run"]
        stderr_buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("sys.stderr", stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                make_game.main()
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("unrecognized arguments: --dry-run", stderr_buf.getvalue())

    def test_simulated_dry_run_wrapper_invokes_check_only(self) -> None:
        def invoke_with_dry_run(args: list[str]) -> tuple[int, bool]:
            clean_args = []
            is_dry = False
            for a in args:
                if a == "--dry-run":
                    is_dry = True
                    clean_args.append("--check")
                else:
                    clean_args.append(a)
            with mock.patch.object(sys, "argv", ["make_game.py", *clean_args]), \
                    mock.patch("make_game.below_normal"), \
                    mock.patch("make_game.check", return_value={"expected_exe": "g.exe"}), \
                    mock.patch("make_game.preflight") as mock_pf, \
                    mock.patch("make_game.log"):
                code = make_game.main()
                return code, mock_pf.called

        ret_code, ran_preflight = invoke_with_dry_run(["--game", self.game_dir, "--dry-run"])
        self.assertEqual(ret_code, 0)
        self.assertFalse(ran_preflight)


class SkipAssetsAndStageIsolationTest(unittest.TestCase):
    """Test skipping asset processing and isolating pipeline stages."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game_dir = os.path.join(self.tmp.name, "game")
        self.out_dir = os.path.join(self.tmp.name, "out")
        self.work_dir = os.path.join(self.tmp.name, "work")
        os.makedirs(self.game_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)

    def test_stages_omitting_assets_completely_skips_asset_stage(self) -> None:
        argv = [
            "make_game.py",
            "--game", self.game_dir,
            "--out", self.out_dir,
            "--work", self.work_dir,
            "--stages", "shaders,runtime,pipelines",
        ]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("make_game.below_normal"), \
                mock.patch("make_game.check", return_value={"expected_exe": "game.exe"}), \
                mock.patch("make_game.preflight"), \
                mock.patch("make_game.shaders") as mock_shaders, \
                mock.patch("make_game.assets") as mock_assets, \
                mock.patch("make_game.runtime") as mock_runtime, \
                mock.patch("make_game.pipelines") as mock_pipelines, \
                mock.patch("make_game.log"):
            ret = make_game.main()
            self.assertEqual(ret, 0)
            mock_shaders.assert_called_once()
            mock_assets.assert_not_called()
            mock_runtime.assert_called_once()
            mock_pipelines.assert_called_once()

    def test_skip_assets_preflight_bypasses_dotnet_sdk_and_submodule_checks(self) -> None:
        # The runtime stage is Windows-only, and preflight refuses it elsewhere before any check this test is
        # about; with no Windows-only stages the same checks run on every system.
        with mock.patch("sys.version_info", (3, 12, 0)), \
                mock.patch.object(make_game, "WINDOWS_ONLY", ()), \
                mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=True), \
                mock.patch.object(make_game, "dotnet_sdk_ok") as mock_dotnet, \
                mock.patch.object(make_game, "ensure_submodule") as mock_submodule, \
                mock.patch.object(make_game, "tool_available", return_value=True), \
                mock.patch.object(make_game, "pipeline_check_exe", return_value="/bin/vk_pipeline_check"), \
                mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                mock.patch("shutil.disk_usage", return_value=mock.Mock(free=100 * (2**30))):
            make_game.preflight(["shaders", "pipelines"], self.work_dir, wav=True)  # no runtime: Windows-only
            mock_dotnet.assert_not_called()
            mock_submodule.assert_not_called()

    def test_skip_assets_preflight_avoids_ffmpeg_requirement_even_when_wav_requested(self) -> None:
        with mock.patch("sys.version_info", (3, 12, 0)), \
                mock.patch.object(make_game, "dotnet_sdk_ok", return_value=True), \
                mock.patch.object(make_game, "tool_available", return_value=True), \
                mock.patch("shutil.which", side_effect=lambda cmd: None if cmd == "ffmpeg" else "/bin/ok"), \
                mock.patch("shutil.disk_usage", return_value=mock.Mock(free=50 * (2**30))):
            make_game.preflight(["shaders"], self.work_dir, wav=True)

    def test_skip_assets_preflight_reduces_disk_space_requirement(self) -> None:
        stderr_buf = io.StringIO()
        with mock.patch("sys.version_info", (3, 12, 0)), \
                mock.patch.object(make_game, "dotnet_sdk_ok", return_value=True), \
                mock.patch.object(make_game, "ensure_submodule"), \
                mock.patch.object(make_game, "tool_available", return_value=True), \
                mock.patch("shutil.which", return_value="/bin/ok"), \
                mock.patch("sys.stderr", stderr_buf), \
                mock.patch("shutil.disk_usage", return_value=mock.Mock(free=50 * (2**30))):
            with self.assertRaises(SystemExit) as ctx:
                make_game.preflight(["shaders", "assets"], self.work_dir, wav=False)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("400 GB is needed", stderr_buf.getvalue())

        with mock.patch("sys.version_info", (3, 12, 0)), \
                mock.patch.object(make_game, "tool_available", return_value=True), \
                mock.patch("shutil.which", return_value="/bin/ok"), \
                mock.patch("shutil.disk_usage", return_value=mock.Mock(free=50 * (2**30))):
            make_game.preflight(["shaders"], self.work_dir, wav=False)

    def test_skip_assets_cli_argument_rejection(self) -> None:
        argv = ["make_game.py", "--skip-assets"]
        stderr_buf = io.StringIO()
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("sys.stderr", stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                make_game.main()
            self.assertEqual(ctx.exception.code, 2)
            self.assertIn("unrecognized arguments: --skip-assets", stderr_buf.getvalue())

    def test_skip_assets_filter_logic_generates_isolated_stage_list(self) -> None:
        def filter_stages(all_stages: list[str], skip_assets: bool) -> list[str]:
            if skip_assets:
                return [s for s in all_stages if s != "assets"]
            return list(all_stages)

        stages_with_skip = filter_stages(make_game.STAGES, skip_assets=True)
        self.assertNotIn("assets", stages_with_skip)
        self.assertEqual(stages_with_skip, ["check", "shaders", "runtime", "pipelines"])

        stages_without_skip = filter_stages(make_game.STAGES, skip_assets=False)
        self.assertIn("assets", stages_without_skip)
        self.assertEqual(stages_without_skip, make_game.STAGES)


class PipelineStageIsolationTest(unittest.TestCase):
    """Test pipelines stage isolation, driver cache creation, and error boundaries."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.work_dir = os.path.join(self.tmp.name, "work")
        self.out_dir = os.path.join(self.tmp.name, "out")
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def test_pipelines_fails_when_spirv_index_is_missing(self) -> None:
        dummy_exe = os.path.join(self.tmp.name, "vk_pipeline_check.exe")
        with open(dummy_exe, "w", encoding="utf-8") as f:
            f.write("dummy")
        with mock.patch("make_game.pipeline_check_exe", return_value=dummy_exe):
            stderr_buf = io.StringIO()
            with mock.patch("sys.stderr", stderr_buf):
                with self.assertRaises(SystemExit) as ctx:
                    make_game.pipelines(self.work_dir, self.out_dir)
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("No translated shaders yet: run the shaders stage first.", stderr_buf.getvalue())

    def test_pipelines_fails_when_pipeline_check_exe_not_built_on_windows(self) -> None:
        spirv_dir = os.path.join(self.work_dir, "shaders", "spirv")
        os.makedirs(spirv_dir, exist_ok=True)
        with open(os.path.join(spirv_dir, "index.json"), "w", encoding="utf-8") as f:
            f.write("{}")

        with mock.patch("make_game.pipeline_check_exe", return_value=None), \
                mock.patch.object(make_game.os, "name", "nt"):
            stderr_buf = io.StringIO()
            with mock.patch("sys.stderr", stderr_buf):
                with self.assertRaises(SystemExit) as ctx:
                    make_game.pipelines(self.work_dir, self.out_dir)
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("vk_pipeline_check.exe wasn't built", stderr_buf.getvalue())

    def test_pipelines_fails_when_pipeline_check_exe_not_built_on_posix(self) -> None:
        spirv_dir = os.path.join(self.work_dir, "shaders", "spirv")
        os.makedirs(spirv_dir, exist_ok=True)
        with open(os.path.join(spirv_dir, "index.json"), "w", encoding="utf-8") as f:
            f.write("{}")

        with mock.patch("make_game.pipeline_check_exe", return_value=None), \
                mock.patch.object(make_game.os, "name", "posix"):
            stderr_buf = io.StringIO()
            with mock.patch("sys.stderr", stderr_buf):
                with self.assertRaises(SystemExit) as ctx:
                    make_game.pipelines(self.work_dir, self.out_dir)
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("vk_pipeline_check wasn't built", stderr_buf.getvalue())

    def test_pipelines_executes_compiler_isolated_with_expected_command_line(self) -> None:
        spirv_dir = os.path.join(self.work_dir, "shaders", "spirv")
        os.makedirs(spirv_dir, exist_ok=True)
        index_path = os.path.join(spirv_dir, "index.json")
        with open(index_path, "w", encoding="utf-8") as f:
            f.write('{"shader_0": {}}')   # a finished translation is never empty

        dummy_exe = os.path.join(self.tmp.name, "vk_pipeline_check.exe")
        with open(dummy_exe, "w", encoding="utf-8") as f:
            f.write("binary")

        expected_cache = os.path.join(self.out_dir, "pipeline_cache.bin")
        executed_cmds = []

        def fake_run(args: list[str], **kwargs: object) -> int:
            executed_cmds.append(args)
            return 0

        # The usage probe (checker_supports) really launches the checker, so it is stubbed too: CreateProcess on a
        # text file named .exe makes native Windows show a modal "Unsupported 16-Bit Application" dialog that blocks
        # the run. A checker without --inflight-file is what the old dummy answered.
        with mock.patch("make_game.pipeline_check_exe", return_value=dummy_exe), \
                mock.patch("make_game.checker_supports", return_value=False) as supports, \
                mock.patch("make_game.run", side_effect=fake_run), \
                mock.patch.object(make_game.subprocess, "run", side_effect=AssertionError("launched the dummy")), \
                mock.patch("make_game.log"):
            make_game.pipelines(self.work_dir, self.out_dir)

        supports.assert_called_once()
        self.assertEqual(len(executed_cmds), 1)
        self.assertEqual(executed_cmds[0], [dummy_exe, spirv_dir, expected_cache])

    def test_pipelines_stage_alone_invokes_no_asset_or_shader_compilers(self) -> None:
        game_dir = os.path.join(self.tmp.name, "game")
        os.makedirs(game_dir, exist_ok=True)
        argv = [
            "make_game.py",
            "--game", game_dir,
            "--out", self.out_dir,
            "--work", self.work_dir,
            "--stages", "pipelines",
        ]
        with mock.patch.object(sys, "argv", argv), \
                mock.patch("make_game.below_normal"), \
                mock.patch("make_game.check", return_value={"expected_exe": "game.exe"}), \
                mock.patch("make_game.preflight"), \
                mock.patch("make_game.shaders") as mock_shaders, \
                mock.patch("make_game.assets") as mock_assets, \
                mock.patch("make_game.runtime") as mock_runtime, \
                mock.patch("make_game.pipelines") as mock_pipelines, \
                mock.patch("make_game.log"):
            ret = make_game.main()
            self.assertEqual(ret, 0)
            mock_shaders.assert_not_called()
            mock_assets.assert_not_called()
            mock_runtime.assert_not_called()
            mock_pipelines.assert_called_once_with(os.path.abspath(self.work_dir), os.path.abspath(self.out_dir),
                                                   mock.ANY, runtime_build=None)


class StageResumptionAndStepIsolationTest(unittest.TestCase):
    """Test step resumption markers and stage decoupling without invoking external tools."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game_dir = os.path.join(self.tmp.name, "game")
        self.work_dir = os.path.join(self.tmp.name, "work")
        self.out_dir = os.path.join(self.tmp.name, "out")
        os.makedirs(self.game_dir, exist_ok=True)
        os.makedirs(self.work_dir, exist_ok=True)
        os.makedirs(self.out_dir, exist_ok=True)

    def test_shaders_stage_skips_finished_steps_based_on_markers(self) -> None:
        dxbc_dir = os.path.join(self.work_dir, "shaders", "dxbc")
        spirv_dir = os.path.join(self.work_dir, "shaders", "spirv")
        os.makedirs(os.path.join(dxbc_dir, "disasm"))
        os.makedirs(spirv_dir, exist_ok=True)

        with open(os.path.join(dxbc_dir, "index.json"), "w", encoding="utf-8") as f:
            f.write('[{"file": "shader.dxbc"}]')
        with open(os.path.join(dxbc_dir, "fssm_index.json"), "w", encoding="utf-8") as f:
            f.write('[{"name": "shader"}]')
        with open(os.path.join(dxbc_dir, "disasm", "shader.s"), "w", encoding="utf-8", newline="\n") as f:
            f.write("synthetic listing\n")
        with open(os.path.join(dxbc_dir, "disasm.done"), "w", encoding="utf-8") as f:
            f.write("test llvm-mc 21\n")
        with open(os.path.join(spirv_dir, "index.json"), "w", encoding="utf-8") as f:
            f.write('{"shader": {}}')   # a bare {} is an empty merge and is translated again (translated_index_is_empty)
        with open(os.path.join(spirv_dir, "wave64_emulate.txt"), "w", encoding="utf-8") as f:
            f.write("off\n")
        with open(os.path.join(spirv_dir, "translation_sources.sha256"), "w", encoding="utf-8") as f:
            f.write(make_game.translation_sources_digest() + "\n")

        with mock.patch("make_game.py") as mock_py, \
                mock.patch("make_game.llvm_mc_signature", return_value="test llvm-mc 21"), \
                mock.patch("make_game.translate") as mock_translate, \
                mock.patch("make_game.log"):
            make_game.shaders(self.game_dir, self.work_dir, jobs=2, emulate=False, keep_emulated=True)
            mock_py.assert_not_called()
            mock_translate.assert_not_called()

    def test_shaders_step_invalidation_when_early_marker_missing(self) -> None:
        dxbc_dir = os.path.join(self.work_dir, "shaders", "dxbc")
        spirv_dir = os.path.join(self.work_dir, "shaders", "spirv")
        os.makedirs(dxbc_dir, exist_ok=True)
        os.makedirs(spirv_dir, exist_ok=True)

        disasm_done = os.path.join(dxbc_dir, "disasm.done")
        spirv_index = os.path.join(spirv_dir, "index.json")
        with open(disasm_done, "w", encoding="utf-8") as f:
            f.write("done\n")
        with open(spirv_index, "w", encoding="utf-8") as f:
            f.write("{}\n")

        fake_py_calls = []

        def fake_py(tool: str, *args: str) -> None:
            fake_py_calls.append(tool)
            if tool == "shader_carve.py":
                with open(os.path.join(dxbc_dir, "index.json"), "w", encoding="utf-8") as f:
                    f.write('[{"file": "shader.dxbc"}]')
            elif tool == "fssm_unpack.py":
                with open(os.path.join(dxbc_dir, "fssm_index.json"), "w", encoding="utf-8") as f:
                    f.write('[{"name": "shader"}]')
            elif tool == "shader_disasm.py":
                disasm = os.path.join(dxbc_dir, "disasm")
                os.makedirs(disasm, exist_ok=True)
                with open(os.path.join(disasm, "shader.s"), "w", encoding="utf-8", newline="\n") as f:
                    f.write("synthetic listing\n")

        with mock.patch("make_game.py", side_effect=fake_py), \
                mock.patch("make_game.llvm_mc_signature", return_value="test llvm-mc 21"), \
                mock.patch("make_game.translate") as mock_translate, \
                mock.patch("make_game.need"), \
                mock.patch("make_game.log"):
            make_game.shaders(self.game_dir, self.work_dir, jobs=1, emulate=False, keep_emulated=False)

        self.assertIn("shader_carve.py", fake_py_calls)
        self.assertIn("fssm_unpack.py", fake_py_calls)

    def test_assets_stage_skips_existing_markers(self) -> None:
        assets_dir = os.path.join(self.work_dir, "assets")
        audio_dir = os.path.join(assets_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)

        for rel in make_game.MODEL_ARCHIVES:
            name = os.path.splitext(os.path.basename(rel))[0]
            for mode in ("models", "textures"):
                dest = os.path.join(assets_dir, name, mode)
                os.makedirs(dest, exist_ok=True)
                with open(os.path.join(dest, "manifest.json"), "w", encoding="utf-8") as f:
                    f.write("{}")
                with open(os.path.join(dest, make_game.DECODER_RECORD), "w", encoding="utf-8") as f:
                    f.write(make_game.decoder_digest())

        with open(os.path.join(audio_dir, "audio_manifest.json"), "w", encoding="utf-8") as f:
            f.write("{}")
        with open(os.path.join(audio_dir, "ogg.done"), "w", encoding="utf-8") as f:
            f.write("done\n")
        with open(os.path.join(audio_dir, "xma.done"), "w", encoding="utf-8") as f:
            f.write("done\n")

        with mock.patch("make_game.rpf9dump") as exporter, \
                mock.patch("make_game.run") as mock_run, \
                mock.patch("make_game.py") as mock_py, \
                mock.patch("shutil.which", return_value=None), \
                mock.patch("make_game.log"):
            make_game.assets(self.game_dir, self.work_dir, wav=False, jobs=1)
            exporter.assert_not_called()
            mock_run.assert_not_called()
            mock_py.assert_not_called()

    def test_texture_pools_resume_only_when_their_listed_dds_files_exist(self) -> None:
        assets = Path(self.work_dir, "assets")
        audio = assets / "audio"
        audio.mkdir(parents=True)
        for marker in ("audio_manifest.json", "ogg.done", "xma.done"):
            (audio / marker).touch()

        # Identically named pools in the base game and a pack have distinct completion markers.
        game = Path(self.game_dir)
        base_source = game / "data" / "ttdb" / "sample.ttpf"
        pack_source = game / "pack" / "story" / "data" / "ttdb" / "sample.ttpf"
        base_dest = assets / "ttpf"
        pack_dest = assets / "ttpf_pack" / "story"
        for src, dest in ((base_source, base_dest), (pack_source, pack_dest)):
            src.parent.mkdir(parents=True)
            src.touch()
            (dest / "sample").mkdir(parents=True)
            (dest / "sample" / "first.dds").write_bytes(b"completed sibling")
            (dest / "sample" / make_game.DECODER_RECORD).write_text(make_game.decoder_digest(), encoding="utf-8")
        (pack_dest / "sample" / "second.dds").mkdir()

        valid = json.dumps({"Count": 2, "Textures": [{"File": "first.dds"}, {"File": None}]})
        legacy = json.dumps({"Count": 2, "Textures": [{"File": "first.dds"}, {"File": "second.dds"}]})
        base_call = mock.call(["rpf9dump", "ttpf", str(base_source), str(base_dest)])
        pack_call = mock.call(["rpf9dump", "ttpf", str(pack_source), str(pack_dest)])
        def publish_catalog(cmd):
            (Path(cmd[3]) / "sample" / "manifest.json").write_text(valid, encoding="utf-8")

        for case, base_manifest, pack_manifest, expected in (
            ("complete including skipped rows", valid, valid, []),
            ("listed DDS is missing", legacy, valid, [base_call]),
            ("listed DDS is a directory", valid, legacy, [pack_call]),
            ("missing texture list", "{}", valid, [base_call]),
            ("incomplete row list", '{"Count": 2, "Textures": [{"File": "first.dds"}]}', valid, [base_call]),
            ("missing File field", valid, '{"Count": 1, "Textures": [{}]}', [pack_call]),
            ("incomplete JSON", valid, "{", [pack_call]),
        ):
            with self.subTest(case=case):
                (base_dest / "sample" / "manifest.json").write_text(base_manifest, encoding="utf-8")
                (pack_dest / "sample" / "manifest.json").write_text(pack_manifest, encoding="utf-8")
                with mock.patch.object(make_game, "MODEL_ARCHIVES", ()), \
                        mock.patch.object(make_game, "rpf9dump", return_value="rpf9dump"), \
                        mock.patch.object(make_game, "run", side_effect=publish_catalog) as run, \
                        mock.patch.object(make_game, "log"), mock.patch("shutil.which", return_value=None):
                    make_game.assets(self.game_dir, self.work_dir, wav=False, jobs=1)
                self.assertEqual(run.call_args_list, expected)

    def test_pending_exports_refresh_the_existing_helper_once(self) -> None:
        # Built under the repo (not --work) so `dotnet -o` is not pointed at a locked remote path.
        output = os.path.join(self.tmp.name, "tools", "rpf9dump", "bin", "Release", "net8.0")
        os.makedirs(output)
        exe = os.path.join(output, "rpf9dump.exe" if os.name == "nt" else "rpf9dump")
        with open(exe, "wb") as f:
            f.write(b"previous build")
        os.makedirs(os.path.join(self.tmp.name, "third_party", "rpf9-explorer", "Core"))
        os.makedirs(os.path.join(self.work_dir, "assets", "audio"))

        with mock.patch.object(make_game, "ROOT", self.tmp.name), \
                mock.patch.object(make_game, "MODEL_ARCHIVES", ["scene.rpf"]), \
                mock.patch.object(make_game, "need"), mock.patch.object(make_game, "py"), \
                mock.patch.object(make_game, "run") as run, mock.patch("shutil.which", return_value=None):
            make_game.assets(self.game_dir, self.work_dir, wav=False, jobs=1)

        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(commands[0][:2], ["dotnet", "build"])
        self.assertIn("-o", commands[0])
        self.assertEqual(commands[0][commands[0].index("-o") + 1], output)
        self.assertEqual([(cmd[0], cmd[1]) for cmd in commands[1:]],
                         [(exe, mode) for mode in ("models", "textures", "audio", "xma")])

    def test_drop_removes_existing_marker_files(self) -> None:
        p1 = os.path.join(self.work_dir, "m1.txt")
        p2 = os.path.join(self.work_dir, "m2.txt")
        p3 = os.path.join(self.work_dir, "nonexistent.txt")
        with open(p1, "w", encoding="utf-8") as f:
            f.write("1")
        with open(p2, "w", encoding="utf-8") as f:
            f.write("2")
        self.assertTrue(os.path.isfile(p1))
        self.assertTrue(os.path.isfile(p2))

        make_game.drop(p1, p2, p3)
        self.assertFalse(os.path.isfile(p1))
        self.assertFalse(os.path.isfile(p2))

    def test_mark_creates_expected_asset_marker(self) -> None:
        marker = os.path.join(self.work_dir, "step.done")
        make_game.mark(marker, "custom_step")

        with open(marker, "r", encoding="utf-8") as f:
            self.assertEqual(f.read().strip(), "custom_step finished")


if __name__ == "__main__":
    unittest.main()
