"""Platform-specific build stages, runtime packaging and vk_pipeline_check discovery.

    python -I tests/test_make_game_platform.py

No game data: stage selection uses mocks, and the packaging and discovery tests create synthetic files.
"""
import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import make_game  # noqa: E402

STAGE_FUNCS = ("shaders", "assets", "runtime", "pipelines")


class DefaultStagesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, os_name, *args, lacking=("clang-cl",)):
        argv = ["make_game.py", "--game", self.game, "--out", os.path.join(self.tmp.name, "out"), "--no-tui", *args]
        mocks = {n: mock.MagicMock() for n in STAGE_FUNCS}
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(make_game.os, "name", os_name), \
                mock.patch.object(make_game, "cross_build_missing", return_value=list(lacking)), \
                mock.patch.object(make_game, "below_normal"), \
                mock.patch.object(make_game, "check", return_value={"expected_exe": "game.exe"}), \
                mock.patch.object(make_game, "preflight") as preflight, \
                mock.patch.object(make_game, "log"), \
                mock.patch.multiple(make_game, **mocks):
            self.assertEqual(make_game.main(), 0)
        return preflight.call_args.args[0], {n: m.called for n, m in mocks.items()}

    def test_default_stages_leave_runtime_out_off_windows_without_the_cross_build(self):
        stages, ran = self.run_main("posix")
        self.assertEqual(stages, ["check", "shaders", "assets", "pipelines"])
        self.assertEqual(ran, {"shaders": True, "assets": True, "runtime": False, "pipelines": True})

    def test_default_stages_keep_runtime_off_windows_with_the_cross_build(self):
        stages, ran = self.run_main("posix", lacking=())
        self.assertEqual(stages, make_game.STAGES)
        self.assertTrue(ran["runtime"])

    def test_default_stages_keep_runtime_on_windows(self):
        stages, ran = self.run_main("nt")
        self.assertEqual(stages, make_game.STAGES)
        self.assertTrue(ran["runtime"])

    def test_explicit_full_stage_list_reaches_preflight_off_windows(self):
        stages, _ = self.run_main("posix", "--stages", ",".join(make_game.STAGES), "--wave64-emulate", "off")
        self.assertEqual(stages, make_game.STAGES)

    def test_explicit_stages_are_kept_as_given(self):
        stages, _ = self.run_main("posix", "--stages", "shaders,runtime")
        self.assertEqual(stages, ["shaders", "runtime"])

    def test_default_stages_keep_runtime_off_windows_with_a_runtime_build(self):
        stages, ran = self.run_main("posix", "--runtime-build", self.tmp.name)
        self.assertEqual(stages, make_game.STAGES)
        self.assertTrue(ran["runtime"])


class PreflightTest(unittest.TestCase):
    def preflight(self, lacking, work):
        with mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch.object(make_game.sys, "version_info", (3, 12, 0)), \
                mock.patch.object(make_game, "cross_build_missing", return_value=lacking), \
                mock.patch.object(make_game, "cross_configure") as configure, \
                mock.patch.object(make_game, "visual_cpp_tools_ok") as vs:
            make_game.preflight(["runtime"], work)
        vs.assert_not_called()
        return configure

    def test_runtime_off_windows_without_the_cross_build_fails_before_any_work(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as work, \
                mock.patch("sys.stderr", err), \
                mock.patch.object(make_game.shutil, "disk_usage") as usage:
            with self.assertRaises(SystemExit) as ctx:
                self.preflight(["clang-cl", "the xwin SDK"], os.path.join(work, "w"))
        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("clang-cl (for the runtime stage, or pass --runtime-build)", err.getvalue())
        self.assertIn("the xwin SDK (for the runtime stage", err.getvalue())
        usage.assert_not_called()
        self.assertFalse(os.path.exists(os.path.join(work, "w")))

    def test_runtime_off_windows_configures_the_cross_build_before_any_work(self):
        with tempfile.TemporaryDirectory() as work:
            self.preflight([], os.path.join(work, "w")).assert_called_once_with()


class CrossBuildMissingTest(unittest.TestCase):
    def setUp(self):
        self.sdk = self.enterContext(tempfile.TemporaryDirectory())
        os.makedirs(os.path.join(self.sdk, "VC", "Tools", "MSVC"))
        os.makedirs(os.path.join(self.sdk, "Windows Kits", "10", "Lib", "10.0.26100", "um", "x64"))

    def missing(self, have=("cmake", *make_game.CROSS_TOOLS), major=23, sdk=None, configured=None):
        def which(t):
            return t if t.startswith("/") and t in have else f"/usr/bin/{t}" if t in have else None
        majors = major if callable(major) else (lambda exe="clang-cl": major)
        with mock.patch.object(make_game.shutil, "which", side_effect=which), \
                mock.patch.object(make_game, "clang_cl_major", side_effect=majors) as self.major, \
                mock.patch.object(make_game, "configured_cross_tools", return_value=configured or {}), \
                mock.patch.dict(os.environ, {"XWIN_DIR": sdk if sdk is not None else self.sdk}):
            return make_game.cross_build_missing()

    def test_a_full_versioned_set_counts(self):
        self.assertEqual(self.missing(have=("cmake", *(f"{t}-21" for t in make_game.CROSS_TOOLS))), [])
        self.major.assert_called_once_with("clang-cl-21")

    def test_an_old_plain_set_does_not_hide_a_newer_versioned_one(self):
        have = ("cmake", *make_game.CROSS_TOOLS, *(f"{t}-21" for t in make_game.CROSS_TOOLS))
        self.assertEqual(self.missing(have=have, major=lambda exe="clang-cl": 18 if exe == "clang-cl" else 21), [])

    def test_versions_are_not_mixed_and_the_missing_one_is_named(self):
        have = ("cmake", "clang-cl-21", "lld-link-21", "llvm-lib-21", "llvm-rc-21", "llvm-mt-20")
        (item,) = self.missing(have=have)
        self.assertTrue(item.startswith("llvm-mt-21: LLVM 19 or newer"), item)

    def test_a_configured_build_folder_needs_only_its_own_tools(self):
        tools = {"CMAKE_CXX_COMPILER": "/opt/llvm-21/bin/clang-cl", "CMAKE_LINKER": "/opt/llvm-21/bin/lld-link",
                 "CMAKE_AR": "llvm-lib", "CMAKE_MT": "/opt/llvm-21/bin/llvm-mt", "CMAKE_RC_COMPILER": "llvm-rc"}
        have = ("cmake", "llvm-lib", "llvm-rc", "/opt/llvm-21/bin/clang-cl", "/opt/llvm-21/bin/lld-link",
                "/opt/llvm-21/bin/llvm-mt")
        self.assertEqual(self.missing(have=have, configured=tools), [])
        self.major.assert_not_called()
        (item,) = self.missing(have=have[:-1], configured=tools)
        self.assertTrue(item.startswith("/opt/llvm-21/bin/llvm-mt, which runtime/build/windows was configured with"))
        self.assertIn("delete runtime/build/windows", item)

    def test_everything_there(self):
        self.assertEqual(self.missing(), [])

    def test_missing_llvm_tools_are_named_together(self):
        (item,) = self.missing(have=("cmake", "clang-cl", "llvm-lib", "llvm-rc", "llvm-mt"))
        self.assertTrue(item.startswith("lld-link: LLVM 19 or newer"))
        (item,) = self.missing(have=("cmake",))
        self.assertTrue(item.startswith(", ".join(make_game.CROSS_TOOLS) + ":"))

    def test_clang_older_than_19_is_refused(self):
        (item,) = self.missing(major=18)
        self.assertIn("this one is 18", item)
        self.assertEqual(self.missing(major=None), [])

    def test_cmake_and_the_sdk(self):
        missing = self.missing(have=make_game.CROSS_TOOLS, sdk=os.path.join(self.sdk, "nothing"))
        self.assertEqual(len(missing), 2)
        self.assertEqual(missing[0], "CMake 3.24 or newer")
        self.assertIn("xwin", missing[1])

    def test_xwin_dir_defaults_to_the_documented_folder(self):
        home = self.enterContext(tempfile.TemporaryDirectory())
        env = {k: v for k, v in os.environ.items() if k != "XWIN_DIR"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(make_game.os.path, "expanduser", side_effect=lambda p: p.replace("~", home, 1)):
            self.assertIsNone(make_game.xwin_dir())
            sdk = os.path.join(home, ".local", "share", "xwin-sdk")
            os.makedirs(os.path.join(sdk, "VC", "Tools", "MSVC"))
            self.assertIsNone(make_game.xwin_dir())
            os.makedirs(os.path.join(sdk, "Windows Kits", "10", "Lib", "10.0.26100", "um", "x64"))
            self.assertEqual(make_game.xwin_dir(), sdk)


class RuntimeBuildPreflightTest(unittest.TestCase):
    def preflight(self, build, files, err=None):
        for name in files:
            open(os.path.join(build, name), "wb").close()
        err = io.StringIO() if err is None else err
        with mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch("sys.stderr", err), \
                mock.patch.object(make_game.sys, "version_info", (3, 12, 0)), \
                mock.patch.object(make_game, "visual_cpp_tools_ok", return_value=False) as vs:
            make_game.preflight(["runtime"], os.path.join(build, "w"), runtime_build=build)
        vs.assert_not_called()
        return err.getvalue()

    def test_runtime_off_windows_passes_with_a_complete_build(self):
        with tempfile.TemporaryDirectory() as build:
            self.preflight(build, make_game.RUNTIME_FILES)

    def test_runtime_build_without_the_tls_host_fails_before_any_work(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as build:
            with self.assertRaises(SystemExit) as ctx:
                self.preflight(build, ["gta6rt.exe"], err=err)
            self.assertEqual(ctx.exception.code, 1)
            self.assertIn("gta6rt_tls_host.dll", err.getvalue())
            self.assertFalse(os.path.exists(os.path.join(build, "w")))


class RuntimePackagingTest(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(tempfile.TemporaryDirectory())
        self.build = os.path.join(self.root, "runtime", "build")
        self.game = os.path.join(self.root, "game")
        self.out = os.path.join(self.root, "out")
        os.makedirs(self.build)
        os.makedirs(self.game)
        self.write_artifact("gta6rt.exe", b"runtime")

    def write_artifact(self, name, contents):
        with open(os.path.join(self.build, name), "wb") as f:
            f.write(contents)

    def test_first_load_hint(self):
        self.write_artifact("gta6rt_tls_host.dll", b"TLS host")
        image = "Player's Test.exe"
        out = os.path.join(self.root, "Port Files")
        for platform in ("nt", "posix"):
            with self.subTest(platform=platform), mock.patch.object(make_game.os, "name", platform), \
                    mock.patch.object(make_game, "log") as log:
                path = os.path.join(self.game, image)
                with open(path, "wb"):
                    pass
                make_game.runtime(self.game, out, {"expected_exe": image}, self.build)
                command = log.call_args.args[0].splitlines()[-1].strip()
                if platform == "nt":
                    self.assertIn("--dry-run", command)
                    self.assertIn("gta6rt.exe'", command)
                    self.assertIn("Player''s Test.exe'", command)
                else:
                    self.assertEqual(shlex.split(command), [os.path.join(out, "gta6rt.sh"), "--dry-run"])
                os.remove(path)
                make_game.runtime(self.game, out, {"expected_exe": image}, self.build)
                self.assertIn("is missing", log.call_args.args[0])
                self.assertNotIn("--dry-run", log.call_args.args[0])

    @unittest.skipIf(os.name == "nt", "executes the generated POSIX launcher")
    def test_wine_first_boot_uses_the_packaged_game_and_shader_paths(self):
        image = "Player's test.exe"
        work = os.path.join(self.root, "Shader cache '$HOME'")
        self.write_artifact("gta6rt_tls_host.dll", b"")
        self.write_artifact("gta6rt.exe", b"import os, sys, json\n"
                            b"print(json.dumps([os.getcwd(), sys.argv[1:], os.getenv('LEONIDA_SHADERS'), "
                            b"os.getenv('LEONIDA_D3D12')]))\n")
        make_game.runtime(self.game, self.out, {"expected_exe": image}, self.build, work=work)
        launcher = os.path.join(self.out, "gta6rt.sh")
        env = dict(os.environ, WINE=sys.executable)
        env.pop("LEONIDA_D3D12", None)
        env.pop("LEONIDA_SHADERS", None)
        missing = subprocess.run([launcher, "--run"], cwd=self.root, env=env, capture_output=True, text=True)
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn(image, missing.stderr)
        with open(os.path.join(self.game, image), "wb"):
            pass
        for mode in ("--dry-run", "--init"):
            with self.subTest(mode=mode):
                result = subprocess.check_output([launcher, mode], cwd=self.root, env=env)
                self.assertEqual(json.loads(result), [
                    self.game, [mode, "./" + image, "--overlay", "Z:" + self.out + "/overlay"],
                    None, "vulkan",
                ])
        env.update(LEONIDA_SHADERS="D:/custom", LEONIDA_D3D12="trace")
        result = subprocess.check_output([launcher, "--run"], cwd=self.root, env=env)
        self.assertEqual(json.loads(result)[2:], ["D:/custom", "trace"])
        result = subprocess.check_output([launcher, "--run", "other.exe", "--", "two words"], cwd=self.root, env=env)
        self.assertEqual(json.loads(result)[:2], [self.root, ["--run", "other.exe", "--", "two words"]])

        work = os.path.join(self.out, "Shader cache '$HOME'")
        make_game.runtime(self.game, self.out, {"expected_exe": image}, self.build, work=work)
        moved = self.out + " moved"
        os.rename(self.out, moved)
        env.pop("LEONIDA_SHADERS")
        result = subprocess.check_output([os.path.join(moved, "gta6rt.sh"), "--dry-run"], cwd=self.root, env=env)
        self.assertEqual(json.loads(result)[2], "Z:" + moved + "/Shader cache '$HOME'/shaders/spirv")

    def test_runtime_records_shader_work_directory(self):
        self.write_artifact("gta6rt_tls_host.dll", b"TLS host")
        work = os.path.join(self.root, "Shader Work \u03bb")
        with mock.patch.object(make_game, "log"):
            make_game.runtime(self.game, self.out, {"expected_exe": "game.exe"}, self.build, work=work)
        with open(os.path.join(self.out, "shader_path.txt"), "rb") as f:
            recorded = f.read()
        self.assertEqual(recorded, os.fsencode(os.path.join(work, "shaders", "spirv")))

    def package(self):
        with mock.patch.object(make_game.os, "name", "nt"), \
                mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch.object(make_game, "run"), \
                mock.patch.object(make_game, "log"):
            make_game.runtime(self.game, self.out, {"expected_exe": "game.exe"})

    def test_runtime_copies_executable_and_tls_host_together(self):
        self.write_artifact("gta6rt_tls_host.dll", b"TLS host")

        self.package()

        with open(os.path.join(self.out, "gta6rt.exe"), "rb") as f:
            self.assertEqual(f.read(), b"runtime")
        with open(os.path.join(self.out, "gta6rt_tls_host.dll"), "rb") as f:
            self.assertEqual(f.read(), b"TLS host")
        with open(os.path.join(self.out, "game_path.txt"), encoding="utf-8") as f:
            self.assertEqual(f.read(), self.game)

    def test_runtime_rejects_missing_tls_host_before_copying(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err), self.assertRaises(SystemExit) as ctx:
            self.package()

        self.assertEqual(ctx.exception.code, 1)
        self.assertIn("gta6rt_tls_host.dll", err.getvalue())
        self.assertFalse(os.path.exists(self.out))

    def test_runtime_build_starts_powershell_by_its_full_path(self):
        # The stage started "powershell" by name. CreateProcess searches System32 but not
        # System32\WindowsPowerShell\v1.0, so on a PC whose PATH had lost that folder the stage ended in a
        # FileNotFoundError traceback after the earlier stages had run.
        self.write_artifact("gta6rt_tls_host.dll", b"TLS host")
        system_root = os.path.join(self.root, "Windows")
        folder = os.path.join(system_root, "System32", "WindowsPowerShell", "v1.0")
        os.makedirs(folder)
        exe = os.path.join(folder, "powershell.exe")
        with open(exe, "wb"):
            pass
        with mock.patch.object(make_game.os, "name", "nt"), \
                mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch.dict(os.environ, {"SystemRoot": system_root, "PATH": ""}), \
                mock.patch.object(make_game, "run") as run, \
                mock.patch.object(make_game, "log"):
            make_game.runtime(self.game, self.out, {"expected_exe": "game.exe"})

        command = run.call_args.args[0]
        self.assertEqual(command[0], exe)
        self.assertIn(os.path.join(self.root, "scripts", "build_runtime.ps1"), command)

    def test_powershell_is_the_bare_name_where_the_file_is_not_at_its_usual_place(self):
        with mock.patch.dict(os.environ, {"SystemRoot": os.path.join(self.root, "nowhere")}):
            self.assertEqual(make_game.windows_powershell(), "powershell")


class ConfiguredCrossToolsTest(unittest.TestCase):
    def test_reads_cmakes_compiler_files_newest_last(self):
        root = self.enterContext(tempfile.TemporaryDirectory())
        files = os.path.join(root, "runtime", "build", "windows", "CMakeFiles")
        with mock.patch.object(make_game, "ROOT", root):
            self.assertEqual(make_game.configured_cross_tools(), {})
        for version, compiler in (("3.31.0", "/old/clang-cl"), ("4.5.1", "/usr/bin/clang-cl")):
            os.makedirs(os.path.join(files, version))
            with open(os.path.join(files, version, "CMakeCXXCompiler.cmake"), "w", encoding="utf-8") as f:
                f.write(f'set(CMAKE_CXX_COMPILER "{compiler}")\nset(CMAKE_CXX_COMPILER_ARG1 "")\n'
                        'set(CMAKE_AR "llvm-lib")\nset(CMAKE_LINKER "/usr/bin/lld-link")\n'
                        'set(CMAKE_MT "/usr/bin/llvm-mt")\n')
            os.utime(os.path.join(files, version, "CMakeCXXCompiler.cmake"), (1, 1 if version == "3.31.0" else 2))
        with open(os.path.join(files, "4.5.1", "CMakeRCCompiler.cmake"), "w", encoding="utf-8") as f:
            f.write('set(CMAKE_RC_COMPILER "llvm-rc")\n')
        with mock.patch.object(make_game, "ROOT", root):
            self.assertEqual(make_game.configured_cross_tools(), {
                "CMAKE_CXX_COMPILER": "/usr/bin/clang-cl", "CMAKE_AR": "llvm-lib", "CMAKE_LINKER": "/usr/bin/lld-link",
                "CMAKE_MT": "/usr/bin/llvm-mt", "CMAKE_RC_COMPILER": "llvm-rc"})


class CrossBuildTest(unittest.TestCase):
    def setUp(self):
        self.root = self.enterContext(tempfile.TemporaryDirectory())
        self.build = os.path.join(self.root, "runtime", "build", "windows")
        self.game = os.path.join(self.root, "game")
        self.out = os.path.join(self.root, "out")
        os.makedirs(self.game)

    def runtime(self, xwin="/sdk", ninja=True, suffix="", configured=None):
        def which(t):
            if t == "ninja":
                return "/usr/bin/ninja" if ninja else None
            return f"/usr/bin/{t}" if suffix else None
        def fake_run(args, hint="", **kw):
            if args[1] == "--build":
                os.makedirs(self.build, exist_ok=True)
                for name in make_game.RUNTIME_FILES:
                    with open(os.path.join(self.build, name), "wb") as f:
                        f.write(name.encode())
        with mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch.object(make_game, "ROOT", self.root), \
                mock.patch.object(make_game, "xwin_dir", return_value=xwin), \
                mock.patch.object(make_game.shutil, "which", side_effect=which), \
                mock.patch.object(make_game, "cross_tool_suffix", return_value=suffix), \
                mock.patch.object(make_game, "configured_cross_tools", return_value=configured or {}), \
                mock.patch.object(make_game, "run", side_effect=fake_run) as self.run_mock, \
                mock.patch.object(make_game, "log"):
            make_game.runtime(self.game, self.out, {"expected_exe": "game.exe"}, None, 4)
        return [c.args[0] for c in self.run_mock.call_args_list]

    def test_cross_builds_the_loader_and_packages_it_for_wine(self):
        configure, build = self.runtime()
        toolchain = os.path.join(self.root, "runtime", "cmake", "clang-cl-xwin.cmake")
        self.assertEqual(configure, ["cmake", "-S", os.path.join(self.root, "runtime"), "-B", self.build,
                                     "-G", "Ninja", "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                                     "--toolchain", toolchain, "-DXWIN_DIR=/sdk"])
        self.assertEqual(build, ["cmake", "--build", self.build, "--parallel", "4",
                                 "--target", "gta6rt", "gta6rt_tls_host"])
        for name in make_game.RUNTIME_FILES:
            with open(os.path.join(self.out, name), "rb") as f:
                self.assertEqual(f.read(), name.encode())
        self.assertTrue(os.path.isfile(os.path.join(self.out, "gta6rt.sh")))

    def test_versioned_tools_are_passed_until_the_folder_records_its_tools(self):
        for cache in (False, True):
            if cache:
                open(os.path.join(self.build, "CMakeCache.txt"), "w").close()
            configure, _ = self.runtime(suffix="-21")
            for var, name in make_game.CROSS_VARS.items():
                self.assertIn(f"-DCMAKE_{var}=/usr/bin/{name}-21", configure)
        configure, _ = self.runtime(suffix="-21", configured={"CMAKE_CXX_COMPILER": "/usr/bin/clang-cl-21"})
        self.assertFalse(any(a.startswith("-DCMAKE_CXX_COMPILER") for a in configure))

    def test_without_ninja_cmake_picks_the_generator(self):
        configure, _ = self.runtime(ninja=False)
        self.assertNotIn("-G", configure)
        self.assertIn("-DCMAKE_BUILD_TYPE=RelWithDebInfo", configure)

    def test_an_existing_build_folder_keeps_its_generator_and_build_type(self):
        os.makedirs(self.build)
        open(os.path.join(self.build, "CMakeCache.txt"), "w").close()
        configure, _ = self.runtime()
        self.assertNotIn("-G", configure)
        self.assertFalse(any(a.startswith("-DCMAKE_BUILD_TYPE") for a in configure))
        self.assertIn("-DXWIN_DIR=/sdk", configure)

    def test_without_an_sdk_nothing_runs(self):
        err = io.StringIO()
        with mock.patch("sys.stderr", err), self.assertRaises(SystemExit):
            self.runtime(xwin=None)
        self.run_mock.assert_not_called()
        self.assertIn("xwin", err.getvalue())
        self.assertFalse(os.path.exists(self.out))


class RuntimeBuildPackagingTest(unittest.TestCase):
    def test_windows_native_build_receives_the_requested_jobs(self):
        with tempfile.TemporaryDirectory() as root:
            build = os.path.join(root, "runtime", "build")
            os.makedirs(build)
            for name in make_game.RUNTIME_FILES:
                open(os.path.join(build, name), "wb").close()
            with mock.patch.object(make_game.os, "name", "nt"), \
                    mock.patch.object(make_game, "ROOT", root), \
                    mock.patch.object(make_game, "run") as run, mock.patch.object(make_game, "log"):
                make_game.runtime(root, os.path.join(root, "out"), {"expected_exe": "game.exe"}, jobs=1)
            command = run.call_args.args[0]
            self.assertEqual(command[command.index("-File") + 1:],
                             [os.path.join(root, "scripts", "build_runtime.ps1"), "-Jobs", "1"])

    def test_off_windows_copies_the_build_and_writes_a_wine_launcher(self):
        with tempfile.TemporaryDirectory() as root:
            build, game, out = (os.path.join(root, n) for n in ("build", "game", "out"))
            os.makedirs(build)
            os.makedirs(game)
            for name in make_game.RUNTIME_FILES:
                with open(os.path.join(build, name), "wb") as f:
                    f.write(name.encode())
            with mock.patch.object(make_game.os, "name", "posix"), \
                    mock.patch.object(make_game, "run") as run, \
                    mock.patch.object(make_game, "log"):
                make_game.runtime(game, out, {"expected_exe": "game.exe"}, build)
            run.assert_not_called()
            for name in make_game.RUNTIME_FILES:
                with open(os.path.join(out, name), "rb") as f:
                    self.assertEqual(f.read(), name.encode())
            launcher = os.path.join(out, "gta6rt.sh")
            with open(launcher, encoding="utf-8") as f:
                text = f.read()
            self.assertTrue(text.startswith("#!/bin/sh\n"))
            self.assertIn('"${WINE:-wine}" "$here/gta6rt.exe" "$@"', text)
            if os.name != "nt":
                self.assertTrue(os.access(launcher, os.X_OK))


class PipelineCheckExeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.build = os.path.join(self.tmp.name, "runtime", "build")
        os.makedirs(os.path.join(self.build, "linux"))
        self.root = mock.patch.object(make_game, "ROOT", self.tmp.name)
        self.root.start()

    def tearDown(self):
        self.root.stop()
        self.tmp.cleanup()

    def touch(self, *parts):
        p = os.path.join(self.build, *parts)
        open(p, "w").close()
        return p

    def exe(self, os_name, runtime_build=None):
        with mock.patch.object(make_game.os, "name", os_name):
            return make_game.pipeline_check_exe(runtime_build)

    def test_linux_prefers_the_readme_build_folder(self):
        self.assertIsNone(self.exe("posix"))
        plain = self.touch("vk_pipeline_check")
        os.chmod(plain, 0o755)
        self.assertEqual(self.exe("posix"), plain)
        readme = self.touch("linux", "vk_pipeline_check")
        os.chmod(readme, 0o755)
        self.assertEqual(self.exe("posix"), readme)

    @unittest.skipUnless(os.name == "posix", "requires POSIX execute permissions")
    def test_non_executable_linux_checker_does_not_hide_the_fallback(self):
        preferred = self.touch("linux", "vk_pipeline_check")
        os.chmod(preferred, 0o644)
        fallback = self.touch("vk_pipeline_check")
        os.chmod(fallback, 0o755)
        self.assertEqual(self.exe("posix"), fallback)

    def test_windows_finds_only_the_exe(self):
        self.touch("linux", "vk_pipeline_check")
        self.assertIsNone(self.exe("nt"))
        exe = self.touch("vk_pipeline_check.exe")
        self.assertEqual(self.exe("nt"), exe)

    def test_windows_prefers_supplied_checker_and_falls_back_to_local_build(self):
        local = self.touch("vk_pipeline_check.exe")
        with tempfile.TemporaryDirectory() as supplied:
            self.assertEqual(self.exe("nt", supplied), local)
            bundled = os.path.join(supplied, "vk_pipeline_check.exe")
            with open(bundled, "wb"):
                pass
            self.assertEqual(self.exe("nt", supplied), bundled)

    def test_pipelines_off_windows_runs_the_linux_build(self):
        exe = self.touch("linux", "vk_pipeline_check")
        os.chmod(exe, 0o755)
        self.touch("vk_pipeline_check.exe")
        work = os.path.join(self.tmp.name, "work")
        os.makedirs(os.path.join(work, "shaders", "spirv"))
        with open(os.path.join(work, "shaders", "spirv", "index.json"), "w") as f:
            f.write('{"shader_0": {}}')   # a finished translation is never empty
        out = os.path.join(self.tmp.name, "out")
        with mock.patch.object(make_game.os, "name", "posix"), \
                mock.patch.object(make_game, "log"), \
                mock.patch.object(make_game, "run", return_value=0) as run:
            make_game.pipelines(work, out, runtime_build=self.build)
        self.assertEqual(run.call_args.args[0][0], exe)

    def test_pipelines_off_windows_without_the_build_says_how_to_build_it(self):
        err = io.StringIO()
        with mock.patch.object(make_game.os, "name", "posix"), mock.patch("sys.stderr", err):
            with self.assertRaises(SystemExit):
                make_game.pipelines(os.path.join(self.tmp.name, "work"), os.path.join(self.tmp.name, "out"))
        self.assertIn("cmake -S runtime -B runtime/build/linux", err.getvalue())


class RuntimeStageCopyTest(unittest.TestCase):
    """runtime() with --runtime-build: the build's files go to the output folder, which may be the build folder."""

    def run_stage(self, build, out, *, work=None):
        for name in make_game.RUNTIME_FILES:
            with open(os.path.join(build, name), "wb") as f:
                f.write(name.encode())
        with mock.patch.object(make_game, "log"), mock.patch.object(make_game.os, "name", "posix"):
            make_game.runtime(os.path.join(build, "game"), out, {"expected_exe": "game.exe"},
                              runtime_build=build, work=work)

    def test_directory_at_a_generated_path_leaves_the_previous_runtime_untouched(self):
        outputs = (*make_game.RUNTIME_FILES, "game_path.txt", "shader_path.txt", "gta6rt.sh")
        for blocked in outputs:
            with self.subTest(blocked=blocked), tempfile.TemporaryDirectory() as build, \
                    tempfile.TemporaryDirectory() as out:
                directory = os.path.join(out, blocked)
                os.mkdir(directory)
                marker = os.path.join(directory, "marker")
                with open(marker, "wb") as f:
                    f.write(b"existing directory content")
                previous = {name: b"previous " + name.encode() for name in outputs if name != blocked}
                for name, contents in previous.items():
                    with open(os.path.join(out, name), "wb") as f:
                        f.write(contents)

                error = io.StringIO()
                with mock.patch.object(sys, "stderr", error), self.assertRaises((SystemExit, OSError)) as failure:
                    self.run_stage(build, out, work=os.path.join(build, "work"))

                for name, contents in previous.items():
                    with open(os.path.join(out, name), "rb") as f:
                        self.assertEqual(f.read(), contents, name)
                with open(marker, "rb") as f:
                    self.assertEqual(f.read(), b"existing directory content")
                self.assertIsInstance(failure.exception, SystemExit)
                self.assertIn(directory, error.getvalue())

    def test_directory_collision_is_checked_before_building(self):
        with tempfile.TemporaryDirectory() as out:
            os.mkdir(os.path.join(out, "game_path.txt"))
            with mock.patch.object(make_game, "run") as build, \
                    mock.patch.object(make_game, "cross_build_runtime") as cross_build, \
                    mock.patch.object(sys, "stderr"), self.assertRaises(SystemExit):
                make_game.runtime(out, out, {"expected_exe": "game.exe"})
            build.assert_not_called()
            cross_build.assert_not_called()

    def test_the_build_is_copied_to_another_folder(self):
        with tempfile.TemporaryDirectory() as build, tempfile.TemporaryDirectory() as out:
            self.run_stage(build, out)
            for name in make_game.RUNTIME_FILES:
                with open(os.path.join(out, name), "rb") as f:
                    self.assertEqual(f.read(), name.encode())
            self.assertTrue(os.path.isfile(os.path.join(out, "game_path.txt")))

    def test_the_build_folder_as_the_output_folder(self):
        # shutil.copy2 of a file onto itself raises SameFileError, and the stage stopped before game_path.txt
        with tempfile.TemporaryDirectory() as build:
            self.run_stage(build, build)
            for name in make_game.RUNTIME_FILES:
                with open(os.path.join(build, name), "rb") as f:
                    self.assertEqual(f.read(), name.encode())
            self.assertTrue(os.path.isfile(os.path.join(build, "game_path.txt")))

    def test_an_older_copy_in_the_output_folder_is_replaced(self):
        with tempfile.TemporaryDirectory() as build, tempfile.TemporaryDirectory() as out:
            with open(os.path.join(out, make_game.RUNTIME_FILES[0]), "wb") as f:
                f.write(b"old")
            self.run_stage(build, out)
            with open(os.path.join(out, make_game.RUNTIME_FILES[0]), "rb") as f:
                self.assertEqual(f.read(), make_game.RUNTIME_FILES[0].encode())


class RedirectedConsoleEncodingTest(unittest.TestCase):
    """`--find` must survive a dump path no single-byte code page can write.

    Windows encodes a redirected stdout (make_game.py > build.log) with the console's ANSI code page, and
    `python -I` ignores PYTHONUTF8, so a path with a character outside that code page -- a Cyrillic, Greek or
    CJK user name in the profile a dump sits under -- reached print() as UnicodeEncodeError and ended the
    listing with a traceback (stderr escapes such characters; stdout was strict). The dump below sits under a
    private-use character that no ANSI code page maps, so before the fix the child dies the same way on every
    Windows locale; on UTF-8 systems the character prints and this passes either way.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_find_lists_a_dump_whose_path_the_console_code_page_cannot_write(self):
        home = os.path.join(self.tmp.name, "home")
        dump = os.path.join(home, "Downloads", "GTA6-\ue9c3")
        os.makedirs(dump)
        with open(os.path.join(ROOT, "game", "xbox-1.0.2.1.json"), encoding="utf-8") as f:
            sizes = json.load(f)["files"]
        for name in make_game.PROBES:      # match_version() reads only names and sizes
            with open(os.path.join(dump, name), "wb") as f:
                f.truncate(sizes[name]["size"])
        env = dict(os.environ, USERPROFILE=home, HOME=home)
        r = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "make_game.py"), "--find"],
                           capture_output=True, env=env, cwd=ROOT, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("UnicodeEncodeError", r.stderr)
        self.assertTrue(any(f"GTA6-{c}" in r.stdout for c in ("\ue9c3", "?")),
                        f"the synthetic dump was not listed: {r.stdout!r}")


if __name__ == "__main__":
    unittest.main()
