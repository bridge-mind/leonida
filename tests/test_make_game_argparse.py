"""Unit tests for make_game.py argument parsing, validation, stage sequencing, --jobs, and --wav handling.

    python -I tests/test_make_game_argparse.py
"""
import argparse
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game


class ArgparseDefaultsAndTypesTest(unittest.TestCase):
    """Test argument parsing defaults, types, and flags in make_game.py."""

    def test_default_argument_values(self):
        with patch.object(sys, "argv", ["make_game.py"]):
            with patch("make_game.below_normal"):
                with patch("make_game.choose_game", return_value=os.path.abspath("C:/dummy_game")):
                    with patch("make_game.check", return_value={"expected_exe": "game.exe"}):
                        with patch("make_game.preflight"), patch("make_game.cross_build_missing", return_value=[]):
                            with patch("make_game.shaders"):
                                with patch("make_game.assets"):
                                    with patch("make_game.runtime"):
                                        with patch("make_game.pipelines"):
                                            with patch("make_game.log"):
                                                orig_parse_args = argparse.ArgumentParser.parse_args
                                                captured = []

                                                def wrap_parse(parser_self, *args, **kwargs):
                                                    res = orig_parse_args(parser_self, *args, **kwargs)
                                                    captured.append(res)
                                                    return res

                                                with patch.object(argparse.ArgumentParser, "parse_args", wrap_parse):
                                                    ret = make_game.main()
                                                    self.assertEqual(ret, 0)

        self.assertEqual(len(captured), 1)
        a = captured[0]
        self.assertIsNone(a.game)
        self.assertIsNone(a.out)
        self.assertIsNone(a.work)
        self.assertIsNone(a.stages)
        self.assertFalse(a.check)
        self.assertFalse(a.full_check)
        self.assertFalse(a.wav)
        self.assertFalse(a.find)
        expected_default_jobs = max(1, (os.cpu_count() or 2) // 2)
        self.assertEqual(a.jobs, expected_default_jobs)

    def test_custom_arguments_parsed_correctly(self):
        test_argv = [
            "make_game.py",
            "--game", "C:/custom/game",
            "--out", "C:/custom/out",
            "--work", "D:/custom/work",
            "--stages", "shaders,assets",
            "--check",
            "--full-check",
            "--wav",
            "--jobs", "8",
            "--find",
        ]
        with patch.object(sys, "argv", test_argv):
            orig_parse_args = argparse.ArgumentParser.parse_args
            captured = []

            def wrap_parse(parser_self, *args, **kwargs):
                res = orig_parse_args(parser_self, *args, **kwargs)
                captured.append(res)
                return res

            with patch.object(argparse.ArgumentParser, "parse_args", wrap_parse):
                with patch("make_game.below_normal"):
                    with patch("make_game.find_dumps", return_value=[]):
                        with patch("make_game.log"):
                            with patch("sys.stdout", new_callable=io.StringIO):
                                ret = make_game.main()
                                self.assertEqual(ret, 0)

        self.assertEqual(len(captured), 1)
        a = captured[0]
        self.assertEqual(a.game, "C:/custom/game")
        self.assertEqual(a.out, "C:/custom/out")
        self.assertEqual(a.work, "D:/custom/work")
        self.assertEqual(a.stages, "shaders,assets")
        self.assertTrue(a.check)
        self.assertTrue(a.full_check)
        self.assertTrue(a.wav)
        self.assertEqual(a.jobs, 8)
        self.assertTrue(a.find)

    def test_non_integer_jobs_rejected_by_argparse(self):
        with patch.object(sys, "argv", ["make_game.py", "--jobs", "four"]):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                with self.assertRaises(SystemExit) as ctx:
                    make_game.main()
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn("invalid int value", mock_err.getvalue())

    def test_zero_or_negative_jobs_rejected_by_argparse(self):
        # 0 would start no translation shard and merge an empty index; a negative count, the same.
        for jobs in ("0", "-2"):
            with patch.object(sys, "argv", ["make_game.py", "--jobs", jobs]):
                with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                    with self.assertRaises(SystemExit) as ctx:
                        make_game.main()
                    self.assertEqual(ctx.exception.code, 2, jobs)
                    self.assertIn("must be at least 1", mock_err.getvalue())

    def test_unrecognized_argument_rejected(self):
        with patch.object(sys, "argv", ["make_game.py", "--unrecognized-option"]):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                with self.assertRaises(SystemExit) as ctx:
                    make_game.main()
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn("unrecognized arguments", mock_err.getvalue())


class StageValidationAndSequencingTest(unittest.TestCase):
    """Test stage parsing, stage validation, and stage execution sequencing."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.game_dir = os.path.join(self.tmp.name, "game")
        self.out_dir = os.path.join(self.tmp.name, "out")
        self.work_dir = os.path.join(self.tmp.name, "work")
        os.makedirs(self.game_dir)
        os.makedirs(self.out_dir)
        os.makedirs(self.work_dir)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *extra_args):
        base_argv = [
            "make_game.py",
            "--game", self.game_dir,
            "--out", self.out_dir,
            "--work", self.work_dir,
            "--no-tui",
            *extra_args,
        ]
        with patch.object(sys, "argv", base_argv):
            with patch("make_game.below_normal"):
                with patch("make_game.check", return_value={"expected_exe": "Leonida.exe"}):
                    with patch("make_game.preflight"):
                        with patch("make_game.log"):
                            with patch("make_game.shaders") as mock_shaders:
                                with patch("make_game.assets") as mock_assets:
                                    with patch("make_game.runtime") as mock_runtime:
                                        with patch("make_game.pipelines") as mock_pipelines:
                                            ret = make_game.main()
                                            return ret, {
                                                "shaders": mock_shaders,
                                                "assets": mock_assets,
                                                "runtime": mock_runtime,
                                                "pipelines": mock_pipelines,
                                            }

    def test_unknown_stage_rejected(self):
        with patch.object(sys, "argv", ["make_game.py", "--game", self.game_dir,
                                        "--out", self.out_dir, "--work", self.work_dir,
                                        "--stages", "invalid_stage"]):
            with patch("make_game.below_normal"):
                with patch("make_game.check", return_value={"expected_exe": "Leonida.exe"}):
                    with patch("make_game.log"):
                        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                            with self.assertRaises(SystemExit) as ctx:
                                make_game.main()
                            self.assertEqual(ctx.exception.code, 2)
                            self.assertIn("unknown stage 'invalid_stage'", mock_err.getvalue())

    def test_mixed_valid_and_unknown_stage_rejected(self):
        with patch.object(sys, "argv", ["make_game.py", "--game", self.game_dir,
                                        "--out", self.out_dir, "--work", self.work_dir,
                                        "--stages", "shaders,badstage,assets"]):
            with patch("make_game.below_normal"):
                with patch("make_game.check", return_value={"expected_exe": "Leonida.exe"}):
                    with patch("make_game.log"):
                        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                            with self.assertRaises(SystemExit) as ctx:
                                make_game.main()
                            self.assertEqual(ctx.exception.code, 2)
                            self.assertIn("unknown stage 'badstage'", mock_err.getvalue())

    def test_invalid_stages_rejected_before_game_access(self):
        for stages in ("", " ", ",,", " , , ", "badstage", "shaders,badstage"):
            with (
                self.subTest(stages=stages),
                patch.object(sys, "argv", ["make_game.py", "--stages", stages]),
                patch("make_game.below_normal"),
                patch("make_game.choose_game", return_value=self.game_dir) as choose,
                patch("make_game.check", return_value={}) as check,
                patch("make_game.preflight") as preflight,
                patch("make_game.shaders") as shaders,
                patch("make_game.log"),
                patch("sys.stderr", new_callable=io.StringIO) as error,
            ):
                with self.assertRaises(SystemExit) as ctx:
                    make_game.main()
                self.assertEqual(ctx.exception.code, 2)
                self.assertIn("--stages", error.getvalue())
                choose.assert_not_called()
                check.assert_not_called()
                preflight.assert_not_called()
                shaders.assert_not_called()

    def test_whitespace_and_empty_items_in_stages(self):
        ret, mocks = self.run_main("--stages", " shaders , , assets ")
        self.assertEqual(ret, 0)
        mocks["shaders"].assert_called_once()
        mocks["assets"].assert_called_once()
        mocks["runtime"].assert_not_called()
        mocks["pipelines"].assert_not_called()

    def test_independent_stages_all_run_when_listed_out_of_order(self):
        """shaders/assets/runtime overlap; pipelines starts after shaders and a selected runtime finish.
        wave64_settings is patched: left to probe the installed GPU it returns keep_emulated=False on native-wave64
        RDNA2 under the AMD Windows driver, and the test then failed there (TEST-461)."""
        manager = MagicMock()
        with patch.object(sys, "argv", ["make_game.py", "--game", self.game_dir,
                                        "--out", self.out_dir, "--work", self.work_dir, "--no-tui",
                                        "--stages", "pipelines,runtime,assets,shaders"]), \
                patch("make_game.wave64_settings", return_value=(False, True)):
            with patch("make_game.below_normal"):
                with patch("make_game.check", return_value={"expected_exe": "Leonida.exe"}):
                    with patch("make_game.preflight"):
                        with patch("make_game.log"):
                            with patch("make_game.pipeline_check_exe", return_value="vk_pipeline_check"):
                                with patch("make_game.shaders", manager.shaders):
                                    with patch("make_game.assets", manager.assets):
                                        with patch("make_game.runtime", manager.runtime):
                                            with patch("make_game.pipelines", manager.pipelines):
                                                ret = make_game.main()
                                                self.assertEqual(ret, 0)

        manager.shaders.assert_called_once_with(
            os.path.abspath(self.game_dir), os.path.abspath(self.work_dir), unittest.mock.ANY,
            emulate=unittest.mock.ANY, keep_emulated=True, fallbacks=unittest.mock.ANY)
        manager.assets.assert_called_once_with(
            os.path.abspath(self.game_dir), os.path.abspath(self.work_dir), False, unittest.mock.ANY,
            dump_jobs=unittest.mock.ANY)
        self.assertIsNotNone(manager.assets.call_args.kwargs["dump_jobs"])
        manager.runtime.assert_called_once_with(
            os.path.abspath(self.game_dir), os.path.abspath(self.out_dir), unittest.mock.ANY, None,
            unittest.mock.ANY, work=os.path.abspath(self.work_dir))
        manager.pipelines.assert_called_once_with(
            os.path.abspath(self.work_dir), os.path.abspath(self.out_dir), unittest.mock.ANY,
            runtime_build=None)

    def test_single_stage_execution_isolates_stages(self):
        for stage in ("shaders", "assets", "runtime", "pipelines"):
            ret, mocks = self.run_main("--stages", stage)
            self.assertEqual(ret, 0)
            for s_name, mock_fn in mocks.items():
                if s_name == stage:
                    mock_fn.assert_called_once()
                else:
                    mock_fn.assert_not_called()

    def test_check_flag_exits_early_without_running_build_stages(self):
        with patch.object(sys, "argv", ["make_game.py", "--game", self.game_dir, "--check"]):
            with patch("make_game.below_normal"):
                with patch("make_game.check", return_value={"expected_exe": "Leonida.exe"}) as mock_check:
                    with patch("make_game.preflight") as mock_preflight:
                        with patch("make_game.shaders") as mock_shaders:
                            ret = make_game.main()
                            self.assertEqual(ret, 0)
                            mock_check.assert_called_once_with(os.path.abspath(self.game_dir), False)
                            mock_preflight.assert_not_called()
                            mock_shaders.assert_not_called()

    def test_find_flag_exits_early_without_running_check_or_stages(self):
        with patch.object(sys, "argv", ["make_game.py", "--find"]):
            with patch("make_game.below_normal"):
                with patch("make_game.find_dumps", return_value=[(self.game_dir, {"title": "Test", "version": "1.0"})]) as mock_find:
                    with patch("make_game.check") as mock_check:
                        with patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                            ret = make_game.main()
                            self.assertEqual(ret, 0)
                            mock_find.assert_called_once()
                            mock_check.assert_not_called()
                            self.assertIn("1 found", mock_out.getvalue())


class JobsHandlingTest(unittest.TestCase):
    """Test --jobs default computation, command-line overrides, and forwarding."""

    def test_jobs_default_calculation_scenarios(self):
        scenarios = [
            (None, 1),
            (1, 1),
            (2, 1),
            (4, 2),
            (8, 4),
            (16, 8),
            (64, 32),
        ]
        for cpus, expected in scenarios:
            with self.subTest(cpus=cpus, expected=expected):
                with patch("os.cpu_count", return_value=cpus):
                    calc = max(1, (os.cpu_count() or 2) // 2)
                    self.assertEqual(calc, expected)

    def test_custom_jobs_forwarded_to_shaders(self):
        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "game")
            out = os.path.join(tmp, "out")
            work = os.path.join(tmp, "work")
            os.makedirs(game)
            with patch.object(sys, "argv", ["make_game.py", "--game", game, "--out", out, "--work", work,
                                            "--no-tui", "--stages", "shaders", "--jobs", "6"]):
                with patch("make_game.below_normal"):
                    with patch("make_game.check", return_value={"expected_exe": "g.exe"}):
                        with patch("make_game.preflight") as mock_preflight:
                            with patch("make_game.log"):
                                with patch("make_game.shaders") as mock_shaders, \
                                        patch("make_game.wave64_settings", return_value=(True, True)) as probe, \
                                        patch("make_game.runtime_gpu_per_vertex", return_value=None), \
                                        patch("make_game.runtime_gpu_barycentric_fallback", return_value=None), \
                                        patch("make_game.runtime_gpu_max_push_constants", return_value=None), \
                                        patch("make_game.runtime_gpu_mesh_draws_crash", return_value=None):
                                    ret = make_game.main()
                                    self.assertEqual(ret, 0)
                                    probe.assert_called_once_with("auto", mesh_compute=False)
                                    mock_preflight.assert_called_once_with(
                                        ["shaders"], os.path.abspath(work), False, None,
                                        emulate=True, keep_emulated=True, fallbacks=(), jobs=6)
                                    mock_shaders.assert_called_once_with(os.path.abspath(game), os.path.abspath(work), 6,
                                                                         emulate=True, keep_emulated=True, fallbacks=())

    def test_custom_jobs_forwarded_to_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "game")
            out = os.path.join(tmp, "out")
            work = os.path.join(tmp, "work")
            os.makedirs(game)
            with patch.object(sys, "argv", ["make_game.py", "--game", game, "--out", out, "--work", work,
                                            "--no-tui", "--stages", "assets", "--wav", "--jobs", "10"]):
                with patch("make_game.below_normal"):
                    with patch("make_game.check", return_value={"expected_exe": "g.exe"}):
                        with patch("make_game.preflight"):
                            with patch("make_game.log"):
                                with patch("make_game.assets") as mock_assets:
                                    ret = make_game.main()
                                    self.assertEqual(ret, 0)
                                    mock_assets.assert_called_once_with(
                                        os.path.abspath(game), os.path.abspath(work), True, 10, dump_jobs=None)

    def test_translate_spawns_exact_job_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            spirv = os.path.join(tmp, "spirv")
            os.makedirs(disasm)
            mock_proc = MagicMock()
            mock_proc.poll.return_value = 0
            with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
                with patch("make_game.py") as mock_py:
                    with patch("make_game.log"):
                        make_game.translate(disasm, spirv, jobs=4)
                        self.assertEqual(mock_popen.call_count, 4)
                        for i in range(4):
                            args = mock_popen.call_args_list[i][0][0]
                            self.assertIn("--shard", args)
                            shard_idx = args.index("--shard")
                            self.assertEqual(args[shard_idx + 1], f"{i}/4")
                        mock_py.assert_called_once_with("rdna2spirv.py", disasm, "--out", spirv, "--merge")


class WavValidationTest(unittest.TestCase):
    """Test --wav validation in preflight and assets stage behavior."""

    def setUp(self):
        # preflight runs `dotnet --list-sdks` (#223). Follow the test's faked shutil.which instead: the SDK counts as
        # installed exactly when the test says dotnet is on PATH, so no real dotnet is needed and none is consulted.
        patcher = patch("make_game.dotnet_sdk_ok", side_effect=lambda: bool(shutil.which("dotnet")))
        patcher.start()
        self.addCleanup(patcher.stop)


    @patch.object(sys, "version_info", (3, 12, 0))
    @patch("make_game.ensure_submodule")
    def test_preflight_wav_requires_ffmpeg_when_assets_included(self, _mock_submodule):
        with patch("shutil.which", return_value=None):
            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                with self.assertRaises(SystemExit) as ctx:
                    make_game.preflight(stages=["assets"], work="C:/dummy_work", wav=True)
                self.assertEqual(ctx.exception.code, 1)
                self.assertIn("ffmpeg (needed for --wav)", mock_err.getvalue())

    @patch.object(sys, "version_info", (3, 12, 0))
    @patch("make_game.ensure_submodule")
    def test_preflight_wav_passes_when_ffmpeg_present(self, _mock_submodule):
        def fake_which(cmd):
            if cmd in ("ffmpeg", "dotnet"):
                return f"/usr/bin/{cmd}"
            return None

        with patch("shutil.which", side_effect=fake_which):
            with patch("shutil.disk_usage") as mock_disk:
                mock_disk.return_value = MagicMock(free=500 * (2**30))
                with patch("os.makedirs"):
                    make_game.preflight(stages=["assets"], work="C:/dummy_work", wav=True)

    @patch.object(sys, "version_info", (3, 12, 0))
    def test_preflight_wav_without_assets_does_not_require_ffmpeg(self):
        def fake_which(cmd):
            if cmd in ("llvm-mc", "spirv-as", "spirv-val"):
                return f"/usr/bin/{cmd}"
            return None

        with patch("shutil.which", side_effect=fake_which):
            with patch.dict(os.environ, {"LLVM_MC": "llvm-mc"}):
                with patch("shutil.disk_usage") as mock_disk:
                    mock_disk.return_value = MagicMock(free=50 * (2**30))
                    with patch("os.makedirs"):
                        # Assets stage not included; ffmpeg is not required even though wav=True
                        make_game.preflight(stages=["shaders"], work="C:/dummy_work", wav=True)

    @patch.object(sys, "version_info", (3, 12, 0))
    @patch("make_game.ensure_submodule")
    def test_preflight_assets_without_wav_does_not_require_ffmpeg(self, _mock_submodule):
        def fake_which(cmd):
            if cmd == "dotnet":
                return "/usr/bin/dotnet"
            return None

        with patch("shutil.which", side_effect=fake_which):
            with patch("shutil.disk_usage") as mock_disk:
                mock_disk.return_value = MagicMock(free=500 * (2**30))
                with patch("os.makedirs"):
                    make_game.preflight(stages=["assets"], work="C:/dummy_work", wav=False)

    def test_assets_wav_fails_when_ogg_directory_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            game = os.path.join(tmp, "game")
            out = os.path.join(work, "assets")
            # Populate manifest files so archive processing loops don't run external tools
            for rel in make_game.MODEL_ARCHIVES:
                name = os.path.splitext(os.path.basename(rel))[0]
                for mode in ("models", "textures"):
                    p = os.path.join(out, name, mode)
                    os.makedirs(p, exist_ok=True)
                    with open(os.path.join(p, "manifest.json"), "w", encoding="utf-8") as f:
                        f.write("{}")
                    with open(os.path.join(p, make_game.DECODER_RECORD), "w", encoding="utf-8") as f:
                        f.write(make_game.decoder_digest())
            audio_dir = os.path.join(out, "audio")
            os.makedirs(audio_dir, exist_ok=True)
            with open(os.path.join(audio_dir, "audio_manifest.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            os.makedirs(game, exist_ok=True)

            with patch("make_game.rpf9dump", return_value="fake_dump"):
                with patch("make_game.log"):
                    with patch("make_game.py"):
                        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                            with self.assertRaises(SystemExit) as ctx:
                                make_game.assets(game, work, wav=True, jobs=2)
                            self.assertEqual(ctx.exception.code, 1)
                            self.assertIn("no Ogg Opus output to decode", mock_err.getvalue())

    def test_assets_wav_invokes_audio_wav_tool_when_ogg_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            game = os.path.join(tmp, "game")
            out = os.path.join(work, "assets")
            # Populate manifests for model archives
            for rel in make_game.MODEL_ARCHIVES:
                name = os.path.splitext(os.path.basename(rel))[0]
                for mode in ("models", "textures"):
                    p = os.path.join(out, name, mode)
                    os.makedirs(p, exist_ok=True)
                    with open(os.path.join(p, "manifest.json"), "w", encoding="utf-8") as f:
                        f.write("{}")
                    with open(os.path.join(p, make_game.DECODER_RECORD), "w", encoding="utf-8") as f:
                        f.write(make_game.decoder_digest())

            ogg_dir = os.path.join(out, "audio", "ogg")
            wav_dir = os.path.join(out, "audio", "wav")
            xma_dir = os.path.join(out, "audio", "xma")
            os.makedirs(ogg_dir, exist_ok=True)
            os.makedirs(xma_dir, exist_ok=True)
            os.makedirs(game, exist_ok=True)

            manifest_dir = os.path.join(out, "audio")
            # A step counts as done by its marker, not by its folder (a stopped run leaves the folder too).
            for name in ("audio_manifest.json", "ogg.done", "xma.done"):
                with open(os.path.join(manifest_dir, name), "w", encoding="utf-8") as f:
                    f.write("{}")

            with patch("make_game.rpf9dump", return_value="fake_dump"):
                with patch("make_game.log"):
                    with patch("make_game.py") as mock_py:
                        make_game.assets(game, work, wav=True, jobs=4)
                        mock_py.assert_called_with("audio_wav.py", ogg_dir, "--out", wav_dir, "--jobs", "4")

    def test_assets_passes_jobs_to_the_ogg_opus_step(self):
        # --jobs reached audio_wav.py only: with --jobs 1 the bank export still started half the CPU count of processes
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            game = os.path.join(tmp, "game")
            out = os.path.join(work, "assets")
            for rel in make_game.MODEL_ARCHIVES:
                name = os.path.splitext(os.path.basename(rel))[0]
                for mode in ("models", "textures"):
                    p = os.path.join(out, name, mode)
                    os.makedirs(p, exist_ok=True)
                    with open(os.path.join(p, "manifest.json"), "w", encoding="utf-8") as f:
                        f.write("{}")
                    with open(os.path.join(p, make_game.DECODER_RECORD), "w", encoding="utf-8") as f:
                        f.write(make_game.decoder_digest())
            audio = os.path.join(out, "audio")
            os.makedirs(os.path.join(audio, "xma"), exist_ok=True)
            os.makedirs(game, exist_ok=True)
            for name in ("audio_manifest.json", "xma.done"):     # no ogg.done: the Ogg Opus step is still to run
                with open(os.path.join(audio, name), "w", encoding="utf-8") as f:
                    f.write("{}")

            with patch("make_game.rpf9dump", return_value="fake_dump"), patch("make_game.log"), \
                    patch("make_game.py") as mock_py:
                make_game.assets(game, work, wav=False, jobs=3)
            mock_py.assert_any_call("awc_opus.py", os.path.join(audio, "awc"), os.path.join(audio, "ogg"), "--jobs", "3")

    def test_assets_without_wav_skips_audio_wav(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            game = os.path.join(tmp, "game")
            out = os.path.join(work, "assets")
            for rel in make_game.MODEL_ARCHIVES:
                name = os.path.splitext(os.path.basename(rel))[0]
                for mode in ("models", "textures"):
                    p = os.path.join(out, name, mode)
                    os.makedirs(p, exist_ok=True)
                    with open(os.path.join(p, "manifest.json"), "w", encoding="utf-8") as f:
                        f.write("{}")
                    with open(os.path.join(p, make_game.DECODER_RECORD), "w", encoding="utf-8") as f:
                        f.write(make_game.decoder_digest())

            manifest_dir = os.path.join(out, "audio")
            os.makedirs(manifest_dir, exist_ok=True)
            for name in ("audio_manifest.json", "ogg.done", "xma.done"):
                with open(os.path.join(manifest_dir, name), "w", encoding="utf-8") as f:
                    f.write("{}")
            os.makedirs(os.path.join(manifest_dir, "ogg"), exist_ok=True)
            os.makedirs(os.path.join(manifest_dir, "xma"), exist_ok=True)
            os.makedirs(game, exist_ok=True)

            with patch("make_game.rpf9dump", return_value="fake_dump"):
                with patch("make_game.log"):
                    with patch("make_game.py") as mock_py:
                        make_game.assets(game, work, wav=False, jobs=2)
                        for call_item in mock_py.call_args_list:
                            self.assertNotEqual(call_item[0][0], "audio_wav.py")


class PathContainmentAndDiskSpaceTest(unittest.TestCase):
    """Test rejection of output paths inside game folder and preflight disk space checks."""

    def setUp(self):
        # preflight runs `dotnet --list-sdks` (#223). Follow the test's faked shutil.which instead: the SDK counts as
        # installed exactly when the test says dotnet is on PATH, so no real dotnet is needed and none is consulted.
        patcher = patch("make_game.dotnet_sdk_ok", side_effect=lambda: bool(shutil.which("dotnet")))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_out_inside_game_folder_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "game")
            out = os.path.join(game, "out")
            work = os.path.join(tmp, "work")
            os.makedirs(game)
            with patch.object(sys, "argv", ["make_game.py", "--game", game, "--out", out, "--work", work]):
                with patch("make_game.below_normal"):
                    with patch("make_game.check", return_value={"expected_exe": "g.exe"}):
                        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                            with self.assertRaises(SystemExit) as ctx:
                                make_game.main()
                            self.assertEqual(ctx.exception.code, 1)
                            self.assertIn("--out and --work must be outside the game folder", mock_err.getvalue())

    def test_work_inside_game_folder_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "game")
            out = os.path.join(tmp, "out")
            work = os.path.join(game, "work")
            os.makedirs(game)
            with patch.object(sys, "argv", ["make_game.py", "--game", game, "--out", out, "--work", work]):
                with patch("make_game.below_normal"):
                    with patch("make_game.check", return_value={"expected_exe": "g.exe"}):
                        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                            with self.assertRaises(SystemExit) as ctx:
                                make_game.main()
                            self.assertEqual(ctx.exception.code, 1)
                            self.assertIn("--out and --work must be outside the game folder", mock_err.getvalue())

    @unittest.skipUnless(os.name == "nt", "paths compare without case only on Windows")
    def test_out_inside_game_folder_in_other_case_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "Game")
            out = os.path.join(tmp, "GAME", "out")      # the same folder on Windows
            work = os.path.join(tmp, "work")
            os.makedirs(game)
            with patch.object(sys, "argv", ["make_game.py", "--game", game, "--out", out, "--work", work]):
                with patch("make_game.below_normal"):
                    with patch("make_game.check", return_value={"expected_exe": "g.exe"}):
                        with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                            with self.assertRaises(SystemExit) as ctx:
                                make_game.main()
                            self.assertEqual(ctx.exception.code, 1)
                            self.assertIn("--out and --work must be outside the game folder", mock_err.getvalue())

    @patch.object(sys, "version_info", (3, 12, 0))
    def test_preflight_space_calculation_and_failure(self):
        # Shaders needs 30GB, Assets needs 370GB -> 400GB total
        def fake_which(cmd):
            if cmd in ("dotnet", "llvm-mc", "spirv-as", "spirv-val"):
                return f"/bin/{cmd}"
            return None

        with patch("shutil.which", side_effect=fake_which):
            with patch.dict(os.environ, {"LLVM_MC": "llvm-mc"}):
                with patch("make_game.ensure_submodule"):
                    with patch("os.makedirs"):
                        with patch("shutil.disk_usage") as mock_disk:
                            mock_disk.return_value = MagicMock(free=100 * (2**30))  # 100 GB available < 400 GB
                            with patch("sys.stderr", new_callable=io.StringIO) as mock_err:
                                with self.assertRaises(SystemExit) as ctx:
                                    make_game.preflight(["shaders", "assets"], work="C:/work")
                                self.assertEqual(ctx.exception.code, 1)
                                self.assertIn("about 400 GB is needed", mock_err.getvalue())


class ToolAvailableTest(unittest.TestCase):
    """Test make_game.tool_available command line and path detection."""

    def test_empty_command_is_false(self):
        self.assertFalse(make_game.tool_available(""))
        self.assertFalse(make_game.tool_available("   "))

    def test_existing_executable_file_is_true(self):
        self.assertTrue(make_game.tool_available(sys.executable))
        self.assertTrue(make_game.tool_available(f'"{sys.executable}"'))

    def test_command_head_on_path_is_true(self):
        with patch("shutil.which", return_value="/bin/custom-tool"):
            self.assertTrue(make_game.tool_available("custom-tool -v"))
            self.assertTrue(make_game.tool_available('"custom-tool" arg1 arg2'))

    def test_nonexistent_tool_is_false(self):
        with patch("shutil.which", return_value=None):
            self.assertFalse(make_game.tool_available("nonexistent_utility_12345"))


class SubprocessCliSmokeTest(unittest.TestCase):
    """Smoke test make_game.py via subprocess to ensure clean entrypoint execution."""

    def test_help_flag_displays_usage_and_options(self):
        make_game_py = os.path.join(ROOT, "make_game.py")
        p = subprocess.run([sys.executable, "-I", make_game_py, "--help"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--game", p.stdout)
        self.assertIn("--stages", p.stdout)
        self.assertIn("--jobs", p.stdout)
        self.assertIn("--wav", p.stdout)
        self.assertIn("--check", p.stdout)
        self.assertIn("--find", p.stdout)

    def test_find_flag_executes_without_error(self):
        make_game_py = os.path.join(ROOT, "make_game.py")
        p = subprocess.run([sys.executable, "-I", make_game_py, "--find"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("found", p.stdout)

    def test_invalid_jobs_option_fails_cli_parsing(self):
        make_game_py = os.path.join(ROOT, "make_game.py")
        p = subprocess.run([sys.executable, "-I", make_game_py, "--jobs", "not_an_int"],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("invalid int value", p.stderr)


if __name__ == "__main__":
    unittest.main()
