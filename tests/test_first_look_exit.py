"""Exercise the Windows diagnostics script with a synthetic inspector and loader."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "first_look.ps1"
POWERSHELL = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]


@unittest.skipUnless(os.name == "nt", "Windows PowerShell diagnostics script")
class FirstLookExitTest(unittest.TestCase):
    def test_reports_are_written_and_dry_run_exit_status_is_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            tools = root / "tools"
            build = root / "runtime" / "build"
            for folder in (scripts, tools, build):
                folder.mkdir(parents=True)
            wrapper = Path(shutil.copyfile(SCRIPT, scripts / SCRIPT.name))
            (tools / "pe_inspect.py").write_text(
                "from pathlib import Path\n"
                "import sys\n"
                "report = Path(sys.argv[-1]) / (Path(sys.argv[1]).name + '.json')\n"
                "report.write_text('{\"imports\":[{\"where\":\"shim\",\"dll\":\"stub.dll\","
                "\"functions\":[\"Example\"]}],\"sections\":[]}', encoding='utf-8', newline='\\n')\n",
                encoding="utf-8", newline="\n",
            )
            compile_loader = (
                "$ErrorActionPreference = 'Stop'; "
                "Add-Type -OutputAssembly gta6rt.exe -OutputType ConsoleApplication -TypeDefinition @'\n"
                "public class Loader { public static int Main() {\n"
                "    int code = int.Parse(System.Environment.GetEnvironmentVariable(\"LEONIDA_TEST_LOADER_EXIT\"));\n"
                "    var dir = new System.IO.DirectoryInfo(System.AppDomain.CurrentDomain.BaseDirectory);\n"
                "    (code == 0 ? System.Console.Out : System.Console.Error).WriteLine(\"synthetic dry run from \" + dir.Name);\n"
                "    return code;\n"
                "} }\n'@"
            )
            env = {**os.environ, "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]}
            # Keep Core's modules out of Windows PowerShell's search path when Python was started from pwsh.
            env["PSModulePath"] = str(Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/Modules")
            subprocess.run([*POWERSHELL, "-Command", compile_loader], cwd=scripts, env=env, check=True, timeout=30)
            fixture = scripts / "gta6rt.exe"
            (scripts / "build_runtime.ps1").write_text(
                'Copy-Item "$PSScriptRoot/gta6rt.exe" "$PSScriptRoot/../runtime/build/gta6rt.exe"\n'
                'New-Item -ItemType File -Force "$PSScriptRoot/../runtime/build/gta6rt_tls_host.dll" | Out-Null\n'
                'Set-Content "$PSScriptRoot/built.txt" built\n', encoding="utf-8", newline="\n",
            )
            source = root / "runtime" / "src" / "loader.cpp"
            source.parent.mkdir()
            source.touch()
            marker = scripts / "built.txt"
            cmake = scripts / "cmake.cmd"
            cmake.write_text(
                '@echo off\nset "cfg=%~6"\nif not defined cfg set "cfg=Debug"\n'
                'if not exist "%~2\\%cfg%" mkdir "%~2\\%cfg%"\n'
                'copy /y "%~dp0gta6rt.exe" "%~2\\%cfg%\\gta6rt.exe" >nul\n'
                'type nul > "%~2\\%cfg%\\gta6rt_tls_host.dll"\n'
                'echo %cfg% > "%~dp0built.txt"\n', encoding="ascii",
            )
            vs_configs = {
                "vs-missing-dll": "Debug;Release;RelWithDebInfo",
                "vs-first": "Debug;Release;MinSizeRel;RelWithDebInfo",
                "vs-release-only": "Release",
                "vs-stale-Release": "Debug;Release;RelWithDebInfo",
            }
            cases = (
                ("Debug", ["Debug"], "Debug", False, 0),
                ("MinSizeRel", ["MinSizeRel", "Debug"], "MinSizeRel", False, 0),
                ("Release", ["Release", "MinSizeRel"], "Release", False, 0),
                ("RelWithDebInfo", ["RelWithDebInfo", "Release"], "RelWithDebInfo", False, 0),
                ("flat", ["", "RelWithDebInfo"], "build", False, 17),
                ("stale-Release", ["Release"], "build", True, 0),
                ("missing", [], "build", True, 0),
                ("vs-missing-dll", ["Release"], "Release", True, 0),
                ("vs-first", [], "RelWithDebInfo", True, 0),
                ("vs-release-only", [], "Release", True, 0),
                ("vs-stale-Release", ["Release"], "Release", True, 0),
            )
            for layout, folders, expected_dir, should_build, code in cases:
                with self.subTest(layout=layout):
                    shutil.rmtree(build)
                    build.mkdir()
                    marker.unlink(missing_ok=True)
                    if layout in vs_configs:
                        (build / "CMakeCache.txt").write_text(
                            "CMAKE_GENERATOR:INTERNAL=Visual Studio 17 2022\n"
                            f"CMAKE_COMMAND:INTERNAL={cmake}\n"
                            f"CMAKE_CONFIGURATION_TYPES:STRING={vs_configs[layout]}\n", encoding="utf-8",
                        )
                    for folder in folders:
                        loader = build / folder / "gta6rt.exe"
                        loader.parent.mkdir(exist_ok=True)
                        shutil.copyfile(fixture, loader)
                        if layout != "vs-missing-dll":
                            loader.with_name("gta6rt_tls_host.dll").touch()
                        if layout in ("stale-Release", "vs-stale-Release"):
                            older = source.stat().st_mtime - 3600
                            os.utime(loader, (older, older))
                    image = root / f"fixture-{layout}.exe"
                    image.write_bytes(b"MZ")
                    result = subprocess.run(
                        [*POWERSHELL, "-File", str(wrapper), "-Exe", str(image)],
                        env={**env, "LEONIDA_TEST_LOADER_EXIT": str(code)}, capture_output=True, text=True, timeout=30,
                    )
                    report = root / "reports" / "exe" / image.stem
                    self.assertEqual(marker.exists(), should_build, result.stdout + result.stderr)
                    if layout in vs_configs:
                        self.assertEqual(marker.read_text(encoding="ascii").strip(), expected_dir)
                    self.assertTrue((report / "shim-todo.txt").is_file(), result.stdout + result.stderr)
                    self.assertIn("synthetic dry run from " + expected_dir,
                                  (report / "dryrun.txt").read_text(encoding="utf-16"))
                    self.assertEqual(result.returncode, code, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
