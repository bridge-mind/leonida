"""Unit test suite covering make_game.py prerequisite checking, version string parsing, and toolchain detection.

    python -I tests/test_make_game_preflight.py
    python -m unittest discover -s tests -v

Tests use synthetic manifests, mock environments, and isolated temporary directories so they are deterministic,
fast, and contain zero game data.
"""
from collections import namedtuple
import functools
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402

Usage = namedtuple("Usage", ["total", "used", "free"])


class PreflightTestBase(unittest.TestCase):
    def assert_fails(self, fn, *args, match=None, **kwargs):
        stderr_buf = io.StringIO()
        with mock.patch("sys.stderr", stderr_buf):
            with self.assertRaises(SystemExit) as ctx:
                fn(*args, **kwargs)
            self.assertEqual(ctx.exception.code, 1)
        err = stderr_buf.getvalue()
        if match:
            self.assertIn(match, err)
        return err


class VersionParsingAndMatchingTest(PreflightTestBase):
    def test_versions_loads_valid_manifest(self):
        manifests = make_game.versions()
        self.assertGreaterEqual(len(manifests), 1)
        v = manifests[0]
        self.assertIn("title", v)
        self.assertIn("version", v)
        self.assertIn("files", v)
        self.assertIn("expected_exe", v)
        self.assertEqual(v["version"], "1.0.2.1")
        self.assertEqual(v["expected_exe"], "game_scarlett_master.exe")

    def test_version_string_parsing_and_ordering(self):
        manifests = make_game.versions()
        parsed = [tuple(map(int, v["version"].split("."))) for v in manifests]
        self.assertIn((1, 0, 2, 1), parsed)
        self.assertGreater((1, 0, 2, 1), (1, 0, 2, 0))
        self.assertGreater((1, 0, 2, 0), (1, 0, 1, 9))
        self.assertGreater((2, 0, 0, 0), (1, 9, 9, 9))

    def test_versions_custom_directory_and_json_filtering(self):
        with tempfile.TemporaryDirectory() as tmp:
            v1_path = os.path.join(tmp, "b_1.0.0.json")
            v2_path = os.path.join(tmp, "a_2.0.0.json")
            txt_path = os.path.join(tmp, "readme.txt")

            with open(v1_path, "w", encoding="utf-8") as f:
                json.dump({"title": "Custom B", "version": "1.0.0", "files": {}, "expected_exe": "b.exe"}, f)
            with open(v2_path, "w", encoding="utf-8") as f:
                json.dump({"title": "Custom A", "version": "2.0.0", "files": {}, "expected_exe": "a.exe"}, f)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write("ignored")

            with mock.patch.object(make_game, "VERSIONS", tmp):
                loaded = make_game.versions()
                self.assertEqual(len(loaded), 2)
                # sorted by filename in os.listdir
                self.assertEqual(loaded[0]["version"], "2.0.0")
                self.assertEqual(loaded[1]["version"], "1.0.0")

    def test_match_version_success_on_probes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            for probe_name in make_game.PROBES:
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size)

            matched = make_game.match_version(tmp)
            self.assertIsNotNone(matched)
            self.assertEqual(matched["version"], manifest["version"])

    def test_match_version_missing_probe_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            # Create 2 probes, skip appdata0.rpf
            for probe_name in ("MicrosoftGame.Config", "Catalog000.bin"):
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size)

            self.assertIsNone(make_game.match_version(tmp))

    def test_match_version_wrong_probe_size_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            for probe_name in make_game.PROBES:
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size + 1 if probe_name == "MicrosoftGame.Config" else size)

            self.assertIsNone(make_game.match_version(tmp))

    def test_match_version_empty_directory_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(make_game.match_version(tmp))

    def test_load_version_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = make_game.versions()[0]
            for probe_name in make_game.PROBES:
                size = manifest["files"][probe_name]["size"]
                probe_path = os.path.join(tmp, probe_name)
                with open(probe_path, "wb") as f:
                    f.truncate(size)

            loaded = make_game.load_version(tmp)
            self.assertEqual(loaded["version"], manifest["version"])

    def test_load_version_failure_reports_supported_versions(self):
        with tempfile.TemporaryDirectory() as tmp:
            err = self.assert_fails(make_game.load_version, tmp,
                                    match="This doesn't look like a supported version of the game")
            self.assertIn("Supported:", err)
            self.assertIn("1.0.2.1", err)
            for name in make_game.PROBES:
                self.assertIn(f"Missing {name}.", err)


class GameCheckTest(PreflightTestBase):
    def test_check_nonexistent_directory_fails(self):
        self.assert_fails(make_game.check, "nonexistent_dir_0xdeadbeef", full=False,
                          match="Game folder not found: nonexistent_dir_0xdeadbeef")

    def test_check_option_reports_each_version_probe_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "MicrosoftGame.Config"), "wb") as config:
                config.write(b"test")
            os.mkdir(os.path.join(tmp, "Catalog000.bin"))
            manifests = [
                {"title": "Synthetic", "version": version, "files": {
                    "MicrosoftGame.Config": {"size": size},
                    "Catalog000.bin": {"size": 8},
                    "appdata0.rpf": {"size": 16},
                }}
                for version, size in [("1.0", 6), ("2.0", 10)]
            ]
            with mock.patch.object(make_game, "versions", return_value=manifests), \
                 mock.patch.object(sys, "argv", ["make_game.py", "--check", "--game", tmp]):
                err = self.assert_fails(make_game.main)

        self.assertIn("Synthetic 1.0\n    Wrong size for MicrosoftGame.Config (4 bytes, expected 6).", err)
        self.assertIn("Synthetic 2.0\n    Wrong size for MicrosoftGame.Config (4 bytes, expected 10).", err)
        self.assertEqual(err.count("Catalog000.bin is not a regular file."), 2)
        self.assertEqual(err.count("Missing appdata0.rpf."), 2)

    def test_check_synthetic_game_all_files_present_with_exe(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic Port Test",
                "version": "9.9.9.9",
                "expected_exe": "test_game.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                    "sub/data.bin": {"size": 128},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel, info in v_info["files"].items():
                p = os.path.join(game_dir, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f:
                    f.truncate(info["size"])

            exe_p = os.path.join(game_dir, v_info["expected_exe"])
            with open(exe_p, "wb") as f:
                f.truncate(256)

            stdout_buf = io.StringIO()
            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                with mock.patch("sys.stdout", stdout_buf):
                    res = make_game.check(game_dir, full=False)

            self.assertEqual(res["version"], "9.9.9.9")
            self.assertIn("all files present", stdout_buf.getvalue())
            self.assertIn("Executable found.", stdout_buf.getvalue())

    def test_check_missing_files_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic Missing Test",
                "version": "9.9.9.9",
                "expected_exe": "test_game.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                    "missing.bin": {"size": 100},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel in ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf"):
                p = os.path.join(game_dir, rel)
                with open(p, "wb") as f:
                    f.truncate(v_info["files"][rel]["size"])

            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                self.assert_fails(make_game.check, game_dir, full=False,
                                  match="1 file(s) missing or different")

    def test_check_wrong_size_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic Size Test",
                "version": "9.9.9.9",
                "expected_exe": "test_game.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                    "extra.bin": {"size": 100},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel, info in v_info["files"].items():
                p = os.path.join(game_dir, rel)
                with open(p, "wb") as f:
                    f.truncate(info["size"] + 5 if rel == "extra.bin" else info["size"])

            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                self.assert_fails(make_game.check, game_dir, full=False,
                                  match="1 file(s) missing or different")

    def test_check_missing_exe_logs_warning_and_proceeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_dir = os.path.join(tmp, "manifests")
            game_dir = os.path.join(tmp, "game")
            os.makedirs(manifest_dir)
            os.makedirs(game_dir)

            v_info = {
                "title": "Synthetic No Exe",
                "version": "9.9.9.9",
                "expected_exe": "custom_loader.exe",
                "files": {
                    "MicrosoftGame.Config": {"size": 16},
                    "Catalog000.bin": {"size": 32},
                    "appdata0.rpf": {"size": 64},
                },
            }
            with open(os.path.join(manifest_dir, "custom.json"), "w", encoding="utf-8") as f:
                json.dump(v_info, f)

            for rel, info in v_info["files"].items():
                p = os.path.join(game_dir, rel)
                with open(p, "wb") as f:
                    f.truncate(info["size"])

            stdout_buf = io.StringIO()
            with mock.patch.object(make_game, "VERSIONS", manifest_dir):
                with mock.patch("sys.stdout", stdout_buf):
                    res = make_game.check(game_dir, full=False)

            self.assertEqual(res["version"], "9.9.9.9")
            self.assertIn("custom_loader.exe is not in the folder", stdout_buf.getvalue())


class ToolchainDetectionTest(PreflightTestBase):
    def test_tool_available_direct_file(self):
        self.assertTrue(make_game.tool_available(sys.executable))

    def test_tool_available_quoted_string(self):
        self.assertTrue(make_game.tool_available(f'"{sys.executable}"'))

    def test_tool_available_command_with_arguments(self):
        self.assertTrue(make_game.tool_available(f'"{sys.executable}" -I'))

    def test_tool_available_shutil_which_lookup(self):
        with mock.patch("shutil.which", return_value=r"C:\bin\custom_tool.exe"):
            self.assertTrue(make_game.tool_available("custom_tool"))

    def test_tool_available_nonexistent_command(self):
        self.assertFalse(make_game.tool_available("nonexistent_compiler_xyz_789"))

    def test_tool_available_empty_and_whitespace(self):
        self.assertFalse(make_game.tool_available(""))
        self.assertFalse(make_game.tool_available("   "))
        self.assertFalse(make_game.tool_available('""'))

    APPDATA_ENV = {"LOCALAPPDATA": r"C:\Users\u\AppData\Local", "APPDATA": r"C:\Users\u\AppData\Roaming"}

    def test_appdata_hint_names_the_folder_llvm_mc_is_in(self):
        with mock.patch("os.name", "nt"), mock.patch.dict(os.environ, self.APPDATA_ENV):
            for cmd, var in ((r"C:\Users\u\AppData\Local\Leonida\llvm-mc\llvm-mc.exe", "LOCALAPPDATA"),
                             (r'"c:\users\U\appdata\roaming\llvm\llvm-mc.exe"', "APPDATA")):
                with self.subTest(cmd=cmd):
                    hint = make_game.appdata_hint(cmd)
                    self.assertIn(f"LLVM_MC points into %{var}%", hint)
                    self.assertIn(r"%USERPROFILE%\.leonida\llvm-mc", hint)

    def test_appdata_hint_is_empty_outside_appdata(self):
        with mock.patch("os.name", "nt"), mock.patch.dict(os.environ, self.APPDATA_ENV):
            for cmd in ("llvm-mc", r"D:\Tools\llvm-mc\llvm-mc.exe", r"C:\Users\u\.leonida\llvm-mc\llvm-mc.exe",
                        r"C:\Users\u\AppData\LocalLow\llvm-mc.exe", "wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc"):
                with self.subTest(cmd=cmd):
                    self.assertEqual(make_game.appdata_hint(cmd), "")

    def test_appdata_hint_is_empty_off_windows(self):
        with mock.patch("os.name", "posix"), mock.patch.dict(os.environ, self.APPDATA_ENV):
            self.assertEqual(make_game.appdata_hint(r"C:\Users\u\AppData\Local\llvm-mc.exe"), "")

    def test_need_success_returns_tool(self):
        self.assertEqual(make_game.need(sys.executable, "Install python"), sys.executable)

    def test_need_failure_exits_with_hint(self):
        self.assert_fails(make_game.need, "ghost_compiler", "Please install it.",
                          match="'ghost_compiler' was not found on PATH. Please install it.")


class PreflightPrerequisiteTest(PreflightTestBase):
    def setUp(self):
        super().setUp()
        # preflight runs `dotnet --list-sdks` (#223). Follow the test's faked shutil.which instead: the SDK counts as
        # installed exactly when the test says dotnet is on PATH, so no real dotnet is needed and none is consulted.
        patcher = mock.patch("make_game.dotnet_sdk_ok", side_effect=lambda: bool(shutil.which("dotnet")))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_preflight_success_all_tools_present_and_space_sufficient(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None) as ensure_submodule, \
                 mock.patch("make_game.visual_cpp_tools_ok", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 200 * 2**30, 800 * 2**30)):
                # off Windows runtime checks the cross-build instead (tests/test_make_game_platform.py)
                stages = ["shaders", "assets"] + (["runtime"] if os.name == "nt" else [])
                make_game.preflight(stages, tmp_work, wav=True)
                ensure_submodule.assert_called_once_with()

    def test_preflight_aggregates_multiple_missing_tools_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", return_value=None), \
                 mock.patch("make_game.tool_available", return_value=False), \
                 mock.patch.object(make_game, "ROOT", tmp_work), \
                 mock.patch("os.path.isfile", return_value=False), \
                 mock.patch("os.name", "nt"):
                err = self.assert_fails(make_game.preflight, ["shaders", "assets", "runtime"], tmp_work, wav=True,
                                        match="Install these first, then run the build again:")
                self.assertIn("ffmpeg (needed for --wav)", err)
                self.assertIn("llvm-mc (LLVM 21+ with the AMDGPU target)", err)
                self.assertIn("spirv-as:", err)
                self.assertIn("spirv-val:", err)
                self.assertIn(".NET 8 SDK", err)
                self.assertIn("Visual Studio 2022", err)

    def test_preflight_accepts_spaced_llvm_mc_resolved_by_which(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            mc = os.path.join(tmp_work, "LLVM Tools", "llvm-mc")
            available = {mc: mc + ".exe", "spirv-as": "spirv-as", "spirv-val": "spirv-val"}
            with mock.patch.dict(os.environ, {"LLVM_MC": mc}), \
                 mock.patch("shutil.which", side_effect=available.get), \
                 mock.patch("shutil.disk_usage", return_value=Usage(100 * 2**30, 0, 100 * 2**30)):
                make_game.preflight(["shaders"], tmp_work)

    def test_preflight_explains_llvm_mc_hidden_in_appdata(self):
        env = {"LOCALAPPDATA": r"C:\Users\u\AppData\Local",
               "LLVM_MC": r"C:\Users\u\AppData\Local\Leonida\llvm-mc\llvm-mc.exe"}
        available = {"spirv-as": "spirv-as", "spirv-val": "spirv-val"}
        with tempfile.TemporaryDirectory() as tmp_work, \
             mock.patch.dict(os.environ, env), \
             mock.patch("os.name", "nt"), \
             mock.patch("sys.version_info", (3, 12, 0)), \
             mock.patch("shutil.which", side_effect=available.get), \
             mock.patch("os.path.isfile", return_value=False):
            err = self.assert_fails(make_game.preflight, ["shaders"], tmp_work,
                                    match="llvm-mc (LLVM 21+ with the AMDGPU target)")
        self.assertIn("LLVM_MC points into %LOCALAPPDATA%", err)

    def test_preflight_rejects_visual_studio_installer_without_cpp_tools(self):
        with tempfile.TemporaryDirectory() as tmp_work, \
             mock.patch("os.name", "nt"), \
             mock.patch("os.path.isfile", return_value=True), \
             mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout=b"\r\n")):
            self.assert_fails(make_game.preflight, ["runtime"], tmp_work, match="Visual Studio 2022")

    def test_visual_cpp_tools_accepts_matching_installation(self):
        with mock.patch("os.path.isfile", return_value=True), \
             mock.patch("subprocess.run", return_value=mock.Mock(returncode=0, stdout=b"C:\\VS\r\n")) as run:
            self.assertTrue(make_game.visual_cpp_tools_ok())
        self.assertEqual(run.call_args.args[0][1:], ["-latest", "-products", "*", "-requires",
                                                   "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                                   "-property", "installationPath"])

    def test_preflight_python_version_check(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 11, 9)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                self.assert_fails(make_game.preflight, ["shaders"], tmp_work, wav=False,
                                  match="Python 3.12 or newer:")

    def test_preflight_wav_without_assets_does_not_require_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            def mock_which(cmd):
                return None if cmd == "ffmpeg" else f"/bin/{cmd}"

            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", side_effect=mock_which), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                # wav=True with ["shaders"] stage only: ffmpeg not required
                make_game.preflight(["shaders"], tmp_work, wav=True)

    def test_preflight_assets_without_wav_does_not_require_ffmpeg(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            def mock_which(cmd):
                return None if cmd == "ffmpeg" else f"/bin/{cmd}"

            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", side_effect=mock_which), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("shutil.disk_usage", return_value=Usage(1000 * 2**30, 100 * 2**30, 900 * 2**30)):
                # wav=False with ["assets"] stage: ffmpeg not required
                make_game.preflight(["assets"], tmp_work, wav=False)

    def test_preflight_disk_space_insufficient(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(500 * 2**30, 485 * 2**30, 15 * 2**30)):
                # Shaders needs 30 GB, only 15 GB free
                self.assert_fails(make_game.preflight, ["shaders"], tmp_work, wav=False,
                                  match="15 GB free on the drive of")

    def test_preflight_charges_space_only_when_shaders_need_to_be_rebuilt(self):
        cases = (
            ("current translation", {}, {}, False, True),
            ("matching emulation", {"spirv/wave64_emulate.txt": "on"}, {"emulate": True}, False, True),
            ("auto keeps emulation", {"spirv/wave64_emulate.txt": "on"}, {"keep_emulated": True}, False, True),
            ("missing disassembly marker", {"dxbc/disasm.done": None}, {}, False, False),
            ("old disassembly marker", {"dxbc/disasm.done": "shader_disasm.py finished"}, {}, False, False),
            ("changed LLVM", {"dxbc/disasm.done": "old llvm-mc"}, {}, False, False),
            ("stale translator", {"spirv/translation_sources.sha256": "stale"}, {}, False, False),
            ("old translation without fingerprint", {"spirv/translation_sources.sha256": None}, {}, False, False),
            ("needs emulation", {}, {"emulate": True}, False, False),
            ("explicit native mode", {"spirv/wave64_emulate.txt": "on"}, {}, False, False),
            ("interrupted unpack", {"dxbc/fssm_index.json": None}, {}, False, False),
            ("interrupted translation", {"spirv/index.json": None}, {}, False, False),
            ("old disassembly layout", {}, {}, True, False),
        )
        fingerprint = make_game.translation_sources_digest()
        usage = Usage(500 * 2**30, 485 * 2**30, 15 * 2**30)
        for name, changed, settings, legacy_layout, reusable in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as work:
                files = {
                    "dxbc/index.json": '[{"file": "shader.dxbc"}]',
                    "dxbc/fssm_index.json": '[{"name": "shader"}]',
                    "dxbc/disasm.done": "test llvm-mc 21",
                    "spirv/index.json": "{}",
                    "spirv/translation_sources.sha256": fingerprint,
                    "spirv/wave64_emulate.txt": "off",
                    **changed,
                }
                shader_root = os.path.join(work, "shaders")
                for name, content in files.items():
                    if content is not None:
                        path = os.path.join(shader_root, name)
                        os.makedirs(os.path.dirname(path), exist_ok=True)
                        with open(path, "w", encoding="utf-8") as handle:
                            handle.write(content + "\n")
                disasm_parts = ("disasm",) if legacy_layout else ("dxbc", "disasm")
                disasm = os.path.join(shader_root, *disasm_parts)
                os.makedirs(disasm)
                with open(os.path.join(disasm, "shader.s"), "w", encoding="utf-8", newline="\n") as f:
                    f.write("synthetic listing\n")
                with mock.patch("sys.version_info", (3, 12, 0)), \
                     mock.patch("make_game.llvm_mc_signature", return_value="test llvm-mc 21"), \
                     mock.patch("make_game.tool_available", return_value=True), \
                     mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                     mock.patch("shutil.disk_usage", return_value=usage), \
                     mock.patch("sys.stdout", io.StringIO()):
                    if reusable:
                        make_game.preflight(["shaders"], work, **settings)
                    else:
                        self.assert_fails(make_game.preflight, ["shaders"], work,
                                          match="about 30 GB is needed", **settings)


    def test_preflight_disk_space_stage_accumulation(self):
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None) as ensure_submodule, \
                 mock.patch("shutil.disk_usage", return_value=Usage(500 * 2**30, 250 * 2**30, 250 * 2**30)):
                # shaders (30) + assets (370) = 400 GB needed, only 250 GB free
                err = self.assert_fails(make_game.preflight, ["shaders", "assets"], tmp_work, wav=False,
                                        match="about 400 GB is needed")
                self.assertIn("250 GB free", err)
                ensure_submodule.assert_not_called()

    def test_preflight_suggests_loader_only_when_dropping_assets_makes_it_fit(self):
        gib = 1 << 30
        loader_build = ["check", "shaders", "runtime"]
        full_build = [*loader_build, "assets"]
        cases = (
            ("reusable shader outputs", full_build, 10 * gib, True, False, True),
            ("asset export only", ["assets"], 250 * gib, False, False, False),
            ("shader and asset exports only", ["shaders", "assets"], 250 * gib, False, False, False),
            ("WAV export requested", full_build, 250 * gib, False, True, False),
        )
        with mock.patch.multiple(
            make_game,
            mc_command=mock.Mock(return_value=["llvm-mc"]),
            dotnet_sdk_ok=mock.Mock(return_value=True),
            ensure_submodule=mock.Mock(),
            visual_cpp_tools_ok=mock.Mock(return_value=True),
            cross_build_missing=mock.Mock(return_value=[]),
            assets_exported=mock.Mock(return_value=False),
        ), mock.patch("shutil.which", return_value="/usr/bin/tool"):
            for name, stages, free, reusable, wav, expect_hint in cases:
                usage = Usage(500 * gib, 500 * gib - free, free)
                with self.subTest(name=name), tempfile.TemporaryDirectory() as work, \
                     mock.patch("shutil.disk_usage", return_value=usage), \
                     mock.patch.object(make_game, "shader_outputs_reusable", return_value=reusable):
                    err = self.assert_fails(make_game.preflight, stages, work, wav=wav,
                                            match="is needed")
                    self.assertEqual("--stages check,shaders,runtime" in err, expect_hint)

    def test_low_disk_cli_suggests_a_working_stage_selection(self):
        gib = 1 << 30
        with tempfile.TemporaryDirectory() as tmp, mock.patch.multiple(
            make_game,
            below_normal=mock.DEFAULT,
            check=mock.Mock(return_value={"expected_exe": "synthetic.exe"}),
            mc_command=mock.Mock(return_value=["llvm-mc"]),
            dotnet_sdk_ok=mock.Mock(return_value=True),
            ensure_submodule=mock.DEFAULT,
            visual_cpp_tools_ok=mock.Mock(return_value=True),
            cross_build_missing=mock.Mock(return_value=[]),
            cross_configure=mock.DEFAULT,
            pipeline_check_exe=mock.Mock(return_value="vk_pipeline_check"),
            assets_exported=mock.Mock(return_value=False),
            shader_outputs_reusable=mock.Mock(return_value=False),
            wave64_settings=mock.Mock(return_value=(False, False)),
            runtime_gpu_per_vertex=mock.Mock(return_value=True),
            log=mock.DEFAULT,
            shaders=mock.DEFAULT,
            assets=mock.DEFAULT,
            runtime=mock.DEFAULT,
            pipelines=mock.DEFAULT,
        ) as stages, mock.patch("shutil.which", return_value="available-tool"), \
                mock.patch("shutil.disk_usage") as disk:
            argv = ["make_game.py", "--game", os.path.join(tmp, "input"), "--out", os.path.join(tmp, "out")]
            with mock.patch.object(sys, "argv", argv):
                disk.return_value = Usage(500 * gib, 470 * gib + 1, 30 * gib - 1)
                error = self.assert_fails(make_game.main, match="about 400 GB is needed")
                self.assertNotIn("--stages", error)

                disk.return_value = Usage(500 * gib, 470 * gib, 30 * gib)
                error = self.assert_fails(make_game.main, match="about 400 GB is needed")
                suggestion = re.search(r"--stages ([a-z,]+)", error)
                self.assertIsNotNone(suggestion, error)
                for stage in ("shaders", "assets", "runtime", "pipelines"):
                    stages[stage].assert_not_called()

            with mock.patch.object(sys, "argv", [*argv, "--stages", suggestion.group(1)]):
                self.assertEqual(make_game.main(), 0)
            stages["shaders"].assert_called_once()
            stages["runtime"].assert_called_once()
            stages["assets"].assert_not_called()
            stages["pipelines"].assert_not_called()

    def test_preflight_counts_assets_again_with_wav_even_when_exported(self):
        # --wav decodes again over the exported audio (assets_exported says it is finished), so the 370 GB stays.
        with tempfile.TemporaryDirectory() as tmp_work:
            with mock.patch("sys.version_info", (3, 12, 0)), \
                 mock.patch("make_game.assets_exported", return_value=True), \
                 mock.patch("make_game.tool_available", return_value=True), \
                 mock.patch("make_game.ensure_submodule", return_value=None), \
                 mock.patch("shutil.which", return_value="/usr/bin/tool"), \
                 mock.patch("os.path.isfile", return_value=True), \
                 mock.patch("shutil.disk_usage", return_value=Usage(500 * 2**30, 250 * 2**30, 250 * 2**30)):
                # Without --wav the exported assets are credited, so 250 GB free is enough for the asset tools.
                make_game.preflight(["assets"], tmp_work, wav=False)
                err = self.assert_fails(make_game.preflight, ["assets"], tmp_work, wav=True,
                                        match="about 370 GB is needed")
                # c506d00c: the loader-only hint is offered only when shaders and runtime are selected and not --wav
                self.assertNotIn("without exporting assets", err)

    def test_ensure_submodule_present_is_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            core_dir = os.path.join(tmp, "third_party", "rpf9-explorer", "Core")
            os.makedirs(core_dir)
            with open(os.path.join(core_dir, "Core.cs"), "w") as f:
                f.write("// dummy")

            with mock.patch.object(make_game, "ROOT", tmp):
                make_game.ensure_submodule()

    def test_ensure_submodule_missing_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(make_game, "ROOT", tmp), \
                 mock.patch("shutil.which", return_value=None):
                self.assert_fails(make_game.ensure_submodule,
                                  match="git clone --recursive https://github.com/ItsNotRexed/Leonida")

    def test_ensure_submodule_zip_with_git_does_not_say_install_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            not_a_repo = subprocess.CompletedProcess([], 128)
            with mock.patch.object(make_game, "ROOT", tmp), \
                 mock.patch("shutil.which", return_value="/usr/bin/git"), \
                 mock.patch("subprocess.run", return_value=not_a_repo):
                err = self.assert_fails(make_game.ensure_submodule,
                                        match="git clone --recursive https://github.com/ItsNotRexed/Leonida")
                self.assertNotIn("Install Git", err)

    def test_ensure_submodule_failed_fetch_in_clone_names_the_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = [subprocess.CompletedProcess([], 0), subprocess.CompletedProcess([], 1)]
            with mock.patch.object(make_game, "ROOT", tmp), \
                 mock.patch("shutil.which", return_value="/usr/bin/git"), \
                 mock.patch("subprocess.run", side_effect=results), \
                 mock.patch.object(make_game, "log"):
                err = self.assert_fails(make_game.ensure_submodule,
                                        match="fetching third_party/rpf9-explorer failed")
                self.assertIn("git submodule update --init --recursive", err)
                self.assertNotIn("Download ZIP", err)


class GameDiscoveryTest(unittest.TestCase):
    def setUp(self):
        self.home = self.enterContext(tempfile.TemporaryDirectory())
        self.desktop = os.path.join(self.home, "Desktop")
        self.downloads = os.path.join(self.home, "Downloads")
        self.version = {"title": "Synthetic", "version": "1.0", "files": {
            "MicrosoftGame.Config": {"size": 3},
            "Catalog000.bin": {"size": 4},
            "appdata0.rpf": {"size": 5},
        }}

    def write_probes(self, folder):
        os.makedirs(folder)
        for name, info in self.version["files"].items():
            with open(os.path.join(folder, name), "wb") as file:
                file.truncate(info["size"])

    def discover(self, known=()):
        """find_dumps() over the synthetic home folder only; `known` is what the shell reports for Downloads,
        Desktop and Documents."""
        roots = (self.home, self.desktop, self.downloads, *known)
        isdir = os.path.isdir
        with mock.patch("os.path.expanduser", return_value=self.home), \
             mock.patch("os.path.isdir", side_effect=lambda path: path in roots and isdir(path)), \
             mock.patch.object(make_game, "known_folders", return_value=list(known)), \
             mock.patch.object(make_game, "versions", return_value=[self.version]):
            return make_game.find_dumps()

    def test_find_searches_known_folders_moved_into_onedrive(self):
        # OneDrive's Known Folder Move: Explorer's Desktop and Documents are ~\OneDrive\Desktop and
        # ~\OneDrive\Documents, two levels below the home folder, so the home root alone never reaches inside them.
        desktop = os.path.join(self.home, "OneDrive", "Desktop")
        documents = os.path.join(self.home, "OneDrive", "Documents")
        on_desktop = os.path.join(desktop, "GTA VI")
        in_documents = os.path.join(documents, "Games", "GTA VI")
        self.write_probes(on_desktop)
        self.write_probes(in_documents)

        self.assertEqual(self.discover(), [])
        self.assertEqual(self.discover(known=(desktop, documents)),
                         [(on_desktop, self.version), (in_documents, self.version)])

    def test_find_reports_a_copy_once_when_known_folders_are_not_moved(self):
        package = os.path.join(self.desktop, "Package")
        self.write_probes(package)
        self.assertEqual(self.discover(known=(self.downloads, self.desktop)), [(package, self.version)])

    def test_known_folders_from_the_shell(self):
        folders = make_game.known_folders()
        if os.name != "nt":
            self.assertEqual(folders, [])
            return
        self.assertEqual(len(folders), len(make_game.KNOWN_FOLDERS))
        for folder in folders:
            self.assertTrue(os.path.isabs(folder), folder)
        self.assertEqual(make_game.known_folders(), folders)   # the shell's buffers are freed, calls repeat cleanly

    def test_find_searches_below_rejected_folder_with_config(self):
        outer = os.path.join(self.desktop, "Backup")
        package = os.path.join(outer, "Package")
        self.write_probes(package)
        with open(os.path.join(outer, "MicrosoftGame.Config"), "wb") as config:
            config.write(b"abc")
        self.write_probes(os.path.join(outer, "Other", "TooDeep"))

        self.assertEqual(self.discover(), [(package, self.version)])

    def test_find_revisits_aliased_root_with_more_depth(self):
        target = os.path.join(self.downloads, "Outer", "Inner")
        package = os.path.join(self.desktop, "Package")
        self.write_probes(os.path.join(target, "Package"))
        self.write_probes(package)
        realpath = os.path.realpath

        def resolve(path):
            if path == self.desktop or path.startswith(self.desktop + os.sep):
                path = os.path.join(target, os.path.relpath(path, self.desktop))
            return realpath(path)

        with mock.patch("os.path.realpath", side_effect=resolve):
            self.assertEqual(self.discover(), [(package, self.version)])


    def test_discovery_roots_follow_the_platform(self):
        # Elsewhere than Windows a copy on another disk or on a share shows up under a mount point, so those are
        # searched too: /Volumes is where macOS mounts drives and mount_smbfs shares, /run/media is Fedora's.
        with mock.patch.object(make_game.os, "name", "posix"), \
             mock.patch.object(make_game, "known_folders", return_value=[]):
            roots = make_game.discovery_roots(self.home)
        self.assertIn(os.path.join(self.home, "Downloads"), roots)
        for root in ("/mnt", "/media", "/run/media", "/Volumes"):
            self.assertIn(root, roots)
        self.assertEqual([r for r in roots if r.endswith(":\\")], [])

    def test_volume_roots_add_the_users_mount_folders(self):
        # udisks2 mounts a drive at /media/<user>/<label> (Ubuntu, Debian) or /run/media/<user>/<label> (Fedora,
        # openSUSE): the per-user folder is a root, so a folder on that drive is in reach like D:\Games is.
        with mock.patch("getpass.getuser", return_value="alice"):
            roots = make_game.volume_roots()
        self.assertEqual(roots[:len(make_game.VOLUME_ROOTS)], list(make_game.VOLUME_ROOTS))
        self.assertIn(os.path.join("/media", "alice"), roots)
        self.assertIn(os.path.join("/run/media", "alice"), roots)
        with mock.patch.object(make_game.os, "name", "posix"), \
             mock.patch.object(make_game, "known_folders", return_value=[]), \
             mock.patch("getpass.getuser", return_value="alice"):
            self.assertIn(os.path.join("/run/media", "alice"), make_game.discovery_roots(self.home))

    def test_volume_roots_without_a_user_name(self):
        # No USER/LOGNAME and no passwd entry (a bare container): getuser() raises; the shared roots still work.
        for failure in (KeyError("getpwuid(): uid not found"), OSError("No username set in the environment")):
            with self.subTest(failure=type(failure).__name__), mock.patch("getpass.getuser", side_effect=failure):
                self.assertEqual(make_game.volume_roots(), list(make_game.VOLUME_ROOTS))
        for name in ("", ".", "..", os.path.join("a", "b")):
            with self.subTest(name=name), mock.patch("getpass.getuser", return_value=name):
                self.assertEqual(make_game.volume_roots(), list(make_game.VOLUME_ROOTS))

    def test_find_reaches_a_folder_on_a_udisks_drive(self):
        # <media>/alice/USB DRIVE/GTA VI: three levels below <media>, two below the user's mount folder.
        media = os.path.join(self.home, "media")
        package = os.path.join(media, "alice", "USB DRIVE", "GTA VI")
        self.write_probes(package)
        elsewhere = os.path.join(self.home, "nohome")

        def find(user_roots):
            with mock.patch.object(make_game.os, "name", "posix"), \
                 mock.patch.object(make_game, "VOLUME_ROOTS", (media,)), \
                 mock.patch.object(make_game, "USER_MOUNT_ROOTS", user_roots), \
                 mock.patch.object(make_game, "known_folders", return_value=[]), \
                 mock.patch("getpass.getuser", return_value="alice"), \
                 mock.patch("os.path.expanduser", return_value=elsewhere), \
                 mock.patch.object(make_game, "versions", return_value=[self.version]):
                return make_game.find_dumps()

        self.assertEqual(find(()), [])   # what the shared root alone reaches: not the folder on the drive
        self.assertEqual(find((media,)), [(package, self.version)])

    def test_windows_roots_are_the_drives_that_exist(self):
        with mock.patch.object(make_game.os, "name", "nt"), \
             mock.patch.object(make_game, "known_folders", return_value=[]), \
             mock.patch("os.path.isdir", side_effect=lambda path: path.startswith("C:")):
            roots = make_game.discovery_roots(self.home)
        self.assertIn("C:\\", roots)
        self.assertNotIn("D:\\", roots)
        self.assertNotIn("/Volumes", roots)

    def test_find_searches_the_platform_roots(self):
        package = os.path.join(self.home, "Volumes", "share", "GTA VI")
        self.write_probes(package)
        root = os.path.dirname(os.path.dirname(package))
        with mock.patch.object(make_game, "discovery_roots", return_value=[root]), \
             mock.patch.object(make_game, "versions", return_value=[self.version]):
            self.assertEqual(make_game.find_dumps(), [(package, self.version)])


class PathAndGameSelectionTest(PreflightTestBase):
    def test_choose_game_single_found(self):
        dummy_entry = ("D:\\TestGames\\Leonida", {"title": "Xbox Package", "version": "1.0.2.1"})
        with mock.patch("make_game.find_dumps", return_value=[dummy_entry]):
            chosen = make_game.choose_game()
            self.assertEqual(chosen, "D:\\TestGames\\Leonida")

    def test_choose_game_none_found_non_interactive(self):
        with mock.patch("make_game.find_dumps", return_value=[]), \
             mock.patch("sys.stdin.isatty", return_value=False):
            self.assert_fails(make_game.choose_game,
                              match="Your game files weren't found. Give the folder with --game.")

    def test_choose_game_multiple_found_non_interactive(self):
        dumps = [
            ("D:\\Games\\Copy1", {"title": "Copy 1", "version": "1.0.2.1"}),
            ("D:\\Games\\Copy2", {"title": "Copy 2", "version": "1.0.2.1"}),
        ]
        with mock.patch("make_game.find_dumps", return_value=dumps), \
             mock.patch("sys.stdin.isatty", return_value=False):
            self.assert_fails(make_game.choose_game,
                              match="More than one copy found: choose one with --game.")

    def default_out_in(self, platform, icloud_documents, earlier_build=False, call=None):
        """default_out() (or `call`) with a temporary home folder; icloud_documents makes the folder macOS keeps in
        iCloud Drive when Desktop & Documents sync is on (~/Library/Mobile Documents/com~apple~CloudDocs/Documents),
        earlier_build a ~/Documents/Leonida from before the change."""
        with tempfile.TemporaryDirectory() as home:
            if icloud_documents:
                os.makedirs(os.path.join(home, "Library", "Mobile Documents", "com~apple~CloudDocs", "Documents"))
            if earlier_build:
                os.makedirs(os.path.join(home, "Documents", "Leonida", "work"))
            with mock.patch.object(make_game.os.path, "expanduser", lambda p: home + p[1:] if p.startswith("~") else p), \
                    mock.patch.object(make_game.sys, "platform", platform):
                if call:
                    return [note.replace(home, "~") for note in call()]
                return os.path.relpath(make_game.default_out(), home)

    def test_icloud_note_names_an_earlier_build_in_documents(self):
        # A Mac that built into ~/Documents/Leonida before the default moved: say where that resumable build is.
        notes = self.default_out_in("darwin", True, earlier_build=True, call=make_game.icloud_default_notes)
        self.assertEqual(len(notes), 2, notes)
        self.assertIn("off iCloud", notes[0])
        self.assertIn(os.path.join("~", "Documents", "Leonida"), notes[1])
        self.assertIn(os.path.join("~", "Leonida") + " to resume", notes[1])
        self.assertIn("--out", notes[1])
        self.assertEqual(len(self.default_out_in("darwin", True, call=make_game.icloud_default_notes)), 1)

    def test_default_out_matches_documents_folder(self):
        for platform in ("win32", "linux", "darwin"):
            self.assertEqual(self.default_out_in(platform, False), os.path.join("Documents", "Leonida"), platform)

    def test_default_out_leaves_icloud_synced_documents_on_a_mac(self):
        # Before: with iCloud Desktop & Documents sync on, ~/Documents/Leonida (and its work folder, hundreds of GB
        # made from the game) went into iCloud Drive, which uploads it and can evict the work files from the disk.
        self.assertEqual(self.default_out_in("darwin", True), "Leonida")
        # Only macOS keeps Documents in iCloud Drive this way; elsewhere the folder means nothing.
        self.assertEqual(self.default_out_in("linux", True), os.path.join("Documents", "Leonida"))

    def test_constants_definitions(self):
        self.assertEqual(make_game.STAGES, ["check", "shaders", "assets", "runtime", "pipelines"])
        self.assertEqual(make_game.PROBES, ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf"))
        self.assertEqual(make_game.MIN_FREE_GB, {"shaders": 30, "assets": 370})

    def test_below_normal_priority_does_not_crash(self):
        make_game.below_normal()


SUBMODULE_READY = make_game.submodule_ready   # the real one, before a test patches it


class PreflightExportedAssetsSpaceTest(PreflightTestBase):
    """A rerun over finished asset exports doesn't need room for a second copy of them."""

    OGG = "awc_opus.py finished\n"
    XMA = "rpf9dump xma finished\n"
    DECODER = "current-decoder"

    def setUp(self):
        super().setUp()
        # The fingerprint assets() compares each model and texture folder with, without asking git for it.
        patcher = mock.patch("make_game.compute_decoder_digest", return_value=self.DECODER)
        patcher.start()
        self.addCleanup(patcher.stop)
        ready = mock.patch("make_game.submodule_ready", return_value=True)   # Core checked out, as in a source zip
        ready.start()
        self.addCleanup(ready.stop)

    def stamp(self, work, rel, fingerprint=None):
        """The decoder.sha256 assets() writes before it exports a model or texture folder; returns its size."""
        text = (self.DECODER if fingerprint is None else fingerprint) + "\n"
        path = os.path.join(work, "assets", *rel.split("/"), make_game.DECODER_RECORD)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return len(text)

    def run_preflight(self, work, free_gb, wav=False, ffmpeg=True):
        usage = Usage(1000 * 2**30, (1000 - free_gb) * 2**30, free_gb * 2**30)

        def which(cmd):
            return None if cmd == "ffmpeg" and not ffmpeg else "/usr/bin/tool"

        # The other prerequisites are faked, so the result doesn't depend on what this machine has installed (a .NET
        # SDK, ffmpeg) and preflight always reaches the space check.
        with mock.patch("sys.version_info", (3, 12, 0)), \
             mock.patch("shutil.which", side_effect=which), \
             mock.patch("make_game.tool_available", return_value=True), \
             mock.patch("make_game.dotnet_sdk_ok", return_value=True), \
             mock.patch("make_game.ensure_submodule", return_value=None), \
             mock.patch("sys.stdout", io.StringIO()), \
             mock.patch("shutil.disk_usage", return_value=usage):
            make_game.preflight(["shaders", "assets"], work, wav=wav)

    def write_xma(self, work, decoded):
        """One XMA2 stream as the assets stage leaves it, with or without the .wav that ffmpeg decodes it to."""
        bank = os.path.join(work, "assets", "audio", "xma", "bank0")
        os.makedirs(bank, exist_ok=True)
        for name in ("123.xma.wav",) + (("123.wav",) if decoded else ()):
            with open(os.path.join(bank, name), "wb") as f:
                f.write(b"RIFF")

    def write_markers(self, work, ogg=OGG, xma=XMA):
        audio = os.path.join(work, "assets", "audio")
        os.makedirs(audio, exist_ok=True)
        for name, text in (("ogg.done", ogg), ("xma.done", xma)):
            if text is not None:
                with open(os.path.join(audio, name), "w", encoding="utf-8") as f:
                    f.write(text)

    def test_markers_written_by_assets_are_read_back(self):
        with tempfile.TemporaryDirectory() as work:
            audio = os.path.join(work, "assets", "audio")
            os.makedirs(audio)
            make_game.mark(os.path.join(audio, "ogg.done"), "awc_opus.py")
            make_game.mark(os.path.join(audio, "xma.done"), "rpf9dump xma")
            self.assertTrue(make_game.assets_exported(work))

    def test_exported_assets_need_only_the_shader_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.run_preflight(work, free_gb=220)  # 30 GB for the shaders, not 400
            err = self.assert_fails(self.run_preflight, work, free_gb=20, match="about 30 GB is needed")
            self.assertIn("20 GB free", err)

    def test_fresh_work_folder_still_needs_the_asset_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertFalse(make_game.assets_exported(work))
            self.assert_fails(self.run_preflight, work, free_gb=220, match="about 400 GB is needed")

    def test_stopped_export_still_needs_the_asset_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)  # stopped before the XMA2 step
            self.assertFalse(make_game.assets_exported(work))
            self.assert_fails(self.run_preflight, work, free_gb=220, match="about 400 GB is needed")

    def test_finished_output_is_credited_once(self):
        # One finished step (its manifest is on disk) holds 150 GB, the only thing written: 150 of the 300 GB estimate
        # is still to write, so 30 + 150 GB must be free. Crediting it twice accepted 60 GB. The size is faked where
        # folders are listed, so every walk over the folder sees the same file and nothing big is written.
        with tempfile.TemporaryDirectory() as work:
            name = os.path.splitext(os.path.basename(make_game.MODEL_ARCHIVES[0]))[0]
            folder = os.path.join(work, "assets", name, "models")
            os.makedirs(folder)
            with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as f:
                f.write("{}")
            self.stamp(work, f"{name}/models")
            real_scandir = os.scandir

            class Big:
                name, path = "big.bin", os.path.join(folder, "big.bin")
                is_symlink = is_junction = staticmethod(lambda: False)
                is_dir = staticmethod(lambda follow_symlinks=True: False)
                is_file = staticmethod(lambda follow_symlinks=True: True)
                stat = staticmethod(lambda follow_symlinks=True: mock.Mock(st_size=150 * 2**30))

            def scandir(path="."):
                if os.path.normcase(os.fspath(path)) != os.path.normcase(folder):
                    return real_scandir(path)
                listing = mock.MagicMock()
                listing.__enter__.return_value = iter([Big])
                return listing

            with mock.patch("os.scandir", scandir):
                self.run_preflight(work, free_gb=250)
                self.assert_fails(self.run_preflight, work, free_gb=60, match="about 250 GB is needed")

    def test_marker_with_other_contents_does_not_count(self):
        for ogg, xma in (("", self.XMA), (self.OGG, "something else\n"), (self.XMA, self.OGG)):
            with self.subTest(ogg=ogg, xma=xma), tempfile.TemporaryDirectory() as work:
                self.write_markers(work, ogg=ogg, xma=xma)
                self.assertFalse(make_game.assets_exported(work))

    def test_marker_that_is_a_folder_does_not_count(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)
            os.makedirs(os.path.join(work, "assets", "audio", "xma.done"))
            self.assertFalse(make_game.assets_exported(work))

    def test_xma_streams_ffmpeg_will_decode_still_need_the_asset_space(self):
        # xma.done is written without ffmpeg too; once ffmpeg is on PATH every run decodes the streams with no .wav.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.write_xma(work, decoded=False)
            self.assert_fails(self.run_preflight, work, free_gb=220, match="about 400 GB is needed")

    def test_xma_streams_left_undecoded_without_ffmpeg_are_not_counted(self):
        # Without ffmpeg nothing decodes them, so the assets stage writes nothing more.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.write_xma(work, decoded=False)
            self.run_preflight(work, free_gb=220, ffmpeg=False)

    def test_decoded_xma_streams_need_only_the_shader_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.write_xma(work, decoded=True)
            self.run_preflight(work, free_gb=220)

    def test_pending_xma_lists_only_streams_without_a_wav(self):
        with tempfile.TemporaryDirectory() as work:
            xma = os.path.join(work, "assets", "audio", "xma")
            self.assertEqual(make_game.pending_xma(xma), [])  # no folder yet
            self.write_xma(work, decoded=True)
            self.assertEqual(make_game.pending_xma(xma), [])
            os.remove(os.path.join(xma, "bank0", "123.wav"))
            self.assertEqual(make_game.pending_xma(xma), [os.path.join(xma, "bank0", "123.xma.wav")])

    def test_wav_still_counts_the_asset_space(self):
        # The WAV decode writes several times the Ogg Opus size and has no marker of its own.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work)
            self.assert_fails(self.run_preflight, work, free_gb=220, wav=True, match="about 400 GB is needed")

    # A stopped export: models, textures and most pools on disk, no xma.done yet.
    def run_partial(self, work, free_gb, on_disk_gb, **kwargs):
        with mock.patch("make_game.assets_bytes_on_disk", return_value=on_disk_gb * 2**30) as walk:
            self.run_preflight(work, free_gb, **kwargs)
        return walk

    def test_stopped_export_counts_what_is_already_on_disk(self):
        # The case this was found on: 270 GB already exported, 205 GB free, refused for 400 GB.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)
            self.run_partial(work, free_gb=205, on_disk_gb=270)  # 30 + 370 - 270 = 130 GB

    def test_stopped_export_still_needs_what_is_left(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)
            err = self.assert_fails(self.run_partial, work, free_gb=205, on_disk_gb=50,
                                    match="about 350 GB is needed")
            self.assertIn("205 GB free", err)

    def test_more_on_disk_than_the_estimate_leaves_the_shader_space(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)
            self.assert_fails(self.run_partial, work, free_gb=20, on_disk_gb=400, match="about 30 GB is needed")

    def test_assets_folder_is_not_walked_when_there_is_room(self):
        with tempfile.TemporaryDirectory() as work:
            walk = self.run_partial(work, free_gb=500, on_disk_gb=270)
            walk.assert_not_called()

    def test_wav_still_counts_the_whole_asset_space_for_a_stopped_export(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)
            self.assert_fails(self.run_partial, work, free_gb=205, on_disk_gb=270, wav=True,
                              match="about 400 GB is needed")

    # The walk itself: only the output of steps assets() would skip on a rerun is counted.
    def put(self, work, rel, size):
        path = os.path.join(work, "assets", *rel.split("/"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"\0" * size)

    def put_pool(self, work, rel, files, promised=None):
        for name, size in files.items():
            self.put(work, f"{rel}/{name}", size)
        rows = [{"File": n} for n in (promised if promised is not None else files)]
        path = os.path.join(work, "assets", *rel.split("/"), "manifest.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"Count": len(rows), "Textures": rows}, f)
        return os.path.getsize(path)

    def test_assets_bytes_on_disk_counts_only_finished_steps(self):
        with tempfile.TemporaryDirectory() as work:
            self.assertEqual(make_game.assets_bytes_on_disk(work), 0)  # no assets folder yet
            self.put(work, "levels/models/a.obj", 100)
            self.put(work, "levels/models/manifest.json", 7)
            r1 = self.stamp(work, "levels/models")
            self.put(work, "levels/textures/b.dds", 50)                 # no fingerprint: exported again
            self.put(work, "common/models/c.obj", 30)
            self.put(work, "common/models/manifest.json", 3)
            r2 = self.stamp(work, "common/models")
            pool = self.put_pool(work, "ttpf/pool", {"d.dds": 4096})
            hashed = self.put_pool(work, "ttpf_pack/core/pool~0123456789ab/pool", {"e.dds": 200})
            self.put_pool(work, "ttpf/half", {"f.dds": 900}, promised=["f.dds", "g.dds"])  # g.dds missing
            self.put(work, "ttpf/stray/h.dds", 800)                      # no manifest at all
            self.put(work, "audio/audio_manifest.json", 9)
            self.put(work, "audio/awc/i.awc", 60)
            self.put(work, "audio/ogg/j.ogg", 70)                        # no ogg.done yet
            self.put(work, "audio/xma/k.xma", 80)
            self.put(work, "audio/xma.done", 1)
            self.put(work, "audio/wav/l.wav", 90)                        # --wav output: not in the estimate
            self.put(work, "unrelated.bin", 5000)
            with open(os.path.join(work, "outside.bin"), "wb") as f:     # not under assets
                f.write(b"\0" * 999)
            expected = (100 + 7 + r1) + (30 + 3 + r2) + (4096 + pool) + (200 + hashed) + (9 + 60 + 1) + 80
            self.assertEqual(make_game.assets_bytes_on_disk(work), expected)

    def test_unrelated_files_under_assets_are_not_credited(self):
        # A large file the exporter didn't write must not let a short drive through.
        with tempfile.TemporaryDirectory() as work:
            self.write_markers(work, xma=None)
            self.put(work, "big.bin", 4096)
            self.put(work, "levels/textures/abandoned.dds", 4096)    # no fingerprint: exported again
            self.assertEqual(make_game.assets_bytes_on_disk(work), 0)
            self.assert_fails(self.run_preflight, work, free_gb=10, match="about 400 GB is needed")

    # A model step finished with 96 GB and a texture step stopped with 54 GB of final .dds files before its manifest.
    # Sizes are faked where folders are listed; nothing big is written.
    def mixed_resume(self, work, texture_fingerprint=None, model_fingerprint=None, with_partial=False):
        models = os.path.join(work, "assets", "levels", "models")
        textures = os.path.join(work, "assets", "levels", "textures")
        self.put(work, "levels/models/manifest.json", 1)
        self.stamp(work, "levels/models", model_fingerprint)
        self.put(work, "levels/textures/a.dds", 1)                       # no manifest: the step stopped
        self.stamp(work, "levels/textures", texture_fingerprint)
        if with_partial:
            self.put(work, "levels/textures/b.dds.partial", 1)           # being written when it was stopped
        sizes = {os.path.join(models, "manifest.json"): 96 * 2**30 - 1,
                 os.path.join(textures, "a.dds"): 54 * 2**30,
                 os.path.join(textures, "b.dds.partial"): 40 * 2**30}
        real_scandir = os.scandir

        class Entry:
            def __init__(self, real):
                self.real, self.name, self.path = real, real.name, real.path
                self.is_symlink, self.is_junction = real.is_symlink, real.is_junction
                self.is_dir, self.is_file = real.is_dir, real.is_file

            def stat(self, follow_symlinks=True):
                st = self.real.stat(follow_symlinks=follow_symlinks)
                return mock.Mock(st_size=sizes.get(self.path, st.st_size))

        def scandir(path="."):
            listing = real_scandir(path)
            if os.path.normcase(os.fspath(path)) not in (os.path.normcase(models), os.path.normcase(textures)):
                return listing
            wrapped = mock.MagicMock()
            wrapped.__enter__.return_value = [Entry(e) for e in listing]
            listing.close()
            return wrapped
        return mock.patch("os.scandir", scandir)

    def test_mixed_resume_credits_the_finished_and_the_reusable_unfinished_output_once(self):
        # 30 + 370 - 96 - 54 = 250 GB; the record files are a few bytes. Finished-only credit asked for 304 GB and a
        # double credit for the finished step accepted 154 GB.
        with tempfile.TemporaryDirectory() as work, self.mixed_resume(work, with_partial=True):
            self.assertEqual(make_game.assets_bytes_on_disk(work) // 2**30, 150)
            self.run_preflight(work, free_gb=250)
            self.assert_fails(self.run_preflight, work, free_gb=249, match="about 250 GB is needed")

    def test_unfinished_output_of_another_decoder_is_not_credited(self):
        # assets() deletes a folder whose fingerprint differs and exports it again, so only the finished step counts.
        with tempfile.TemporaryDirectory() as work, self.mixed_resume(work, texture_fingerprint="older-decoder"):
            self.run_preflight(work, free_gb=304)
            self.assert_fails(self.run_preflight, work, free_gb=303, match="about 304 GB is needed")

    def test_a_finished_step_of_another_decoder_is_not_credited(self):
        with tempfile.TemporaryDirectory() as work, self.mixed_resume(work, model_fingerprint="older-decoder"):
            self.run_preflight(work, free_gb=346)       # 30 + 370 - 54
            self.assert_fails(self.run_preflight, work, free_gb=345, match="about 346 GB is needed")

    def test_the_credit_does_not_cache_a_decoder_digest_before_the_submodule_is_fetched(self):
        # preflight credits the exports before ensure_submodule(): a digest cached then (made from a missing or
        # changed checkout) would be the one assets() compares with afterwards.
        # A cached stand-in that would also match the records, so only the cache itself can tell the call happened.
        cached = functools.cache(lambda: self.DECODER)
        with mock.patch("make_game.decoder_digest", cached),                 tempfile.TemporaryDirectory() as work, self.mixed_resume(work):
            self.run_preflight(work, free_gb=250)
        self.assertEqual(cached.cache_info().currsize, 0)

    def test_nothing_is_credited_before_the_submodule_is_checked_out(self):
        # Records that match the digest of a missing checkout (no git pin) would not match the one assets() computes
        # after ensure_submodule() has fetched it, and assets() would delete those folders.
        with tempfile.TemporaryDirectory() as work, self.mixed_resume(work), \
                mock.patch("make_game.submodule_ready", return_value=False), \
                mock.patch("make_game.compute_decoder_digest") as digest:
            self.assertEqual(make_game.assets_bytes_on_disk(work), 0)
            self.assert_fails(self.run_preflight, work, free_gb=399, match="about 400 GB is needed")
            digest.assert_not_called()

    def test_submodule_ready_needs_a_non_empty_core(self):
        with tempfile.TemporaryDirectory() as root, mock.patch("make_game.ROOT", root):
            core = os.path.join(root, "third_party", "rpf9-explorer", "Core")
            for ready in (False, False, True):
                if ready:
                    with open(os.path.join(core, "x.cs"), "w", encoding="utf-8") as f:
                        f.write("")
                elif not os.path.isdir(core):
                    self.assertFalse(SUBMODULE_READY())   # missing
                    os.makedirs(core)
                    continue
                self.assertEqual(SUBMODULE_READY(), ready)   # empty, then non-empty (no .git needed)

    def test_a_decoder_that_cannot_be_hashed_credits_nothing_and_does_not_stop_preflight(self):
        with tempfile.TemporaryDirectory() as work, self.mixed_resume(work), \
                mock.patch("make_game.compute_decoder_digest", side_effect=FileNotFoundError("Program.cs")):
            self.assertEqual(make_game.assets_bytes_on_disk(work), 0)
            self.assert_fails(self.run_preflight, work, free_gb=399, match="about 400 GB is needed")

    def test_the_decoder_is_not_hashed_when_there_is_room_or_nothing_is_exported(self):
        with tempfile.TemporaryDirectory() as work, mock.patch("make_game.compute_decoder_digest") as digest:
            self.run_preflight(work, free_gb=500)
            self.assertEqual(make_game.assets_bytes_on_disk(work), 0)
            digest.assert_not_called()

    def test_wav_ignores_the_unfinished_output_too(self):
        with tempfile.TemporaryDirectory() as work, self.mixed_resume(work):
            self.assert_fails(self.run_preflight, work, free_gb=399, wav=True, match="about 400 GB is needed")

    def test_assets_bytes_on_disk_does_not_follow_a_linked_folder(self):
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as elsewhere:
            with open(os.path.join(elsewhere, "big.bin"), "wb") as f:
                f.write(b"\0" * 5000)
            self.put(work, "levels/models/manifest.json", 7)
            record = self.stamp(work, "levels/models")
            try:
                os.symlink(elsewhere, os.path.join(work, "assets", "levels", "models", "link"),
                           target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symbolic links are not available here")
            self.assertEqual(make_game.assets_bytes_on_disk(work), 7 + record)

    def test_assets_bytes_on_disk_does_not_follow_a_junction(self):
        if os.name != "nt":
            self.skipTest("NTFS junctions are Windows only")
        import _winapi
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as elsewhere:
            with open(os.path.join(elsewhere, "big.bin"), "wb") as f:
                f.write(b"\0" * 5000)
            self.put(work, "levels/models/manifest.json", 7)
            record = self.stamp(work, "levels/models")
            junction = os.path.join(work, "assets", "levels", "models", "junction")
            _winapi.CreateJunction(elsewhere, junction)
            try:
                self.assertEqual(make_game.assets_bytes_on_disk(work), 7 + record)
            finally:
                os.rmdir(junction)          # removes the junction, not its target

    def test_a_linked_assets_folder_counts_nothing(self):
        # <work>/assets pointing at another drive: what is there takes no room on the drive preflight measures.
        with tempfile.TemporaryDirectory() as work, tempfile.TemporaryDirectory() as elsewhere:
            os.makedirs(os.path.join(elsewhere, "levels", "models"))
            with open(os.path.join(elsewhere, "levels", "models", "manifest.json"), "wb") as f:
                f.write(b"\0" * 5000)
            link = os.path.join(work, "assets")
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(elsewhere, link)
            else:
                os.symlink(elsewhere, link, target_is_directory=True)
            try:
                self.assertEqual(make_game.assets_bytes_on_disk(work), 0)
            finally:
                if os.name == "nt":
                    os.rmdir(link)


if __name__ == "__main__":
    unittest.main()
