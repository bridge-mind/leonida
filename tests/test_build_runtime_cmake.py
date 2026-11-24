"""Run the runtime wrapper's real CMake batch branch with a native argument recorder.

Windows only; needs the .NET Framework C# compiler, not MSVC, CMake, a GPU or game files.
The recorder validates configure paths and lets either stage fail independently.
"""
import base64
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_runtime.ps1"
CSC = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Microsoft.NET/Framework64/v4.0.30319/csc.exe"
RECORDER = r'''
using System;
using System.IO;
class Recorder {
    static int Main(string[] args) {
        bool build = args.Length > 0 && args[0] == "--build";
        string stage = build ? "build" : "configure";
        File.WriteAllLines(Path.Combine(Environment.GetEnvironmentVariable("LEONIDA_CMAKE_LOG"), stage + ".txt"), args);
        if (!build) {
            for (int i = 0; i + 1 < args.Length; ++i)
                if (args[i] == "-S" && !Directory.Exists(args[i + 1])) return 9;
        }
        return Environment.GetEnvironmentVariable("LEONIDA_CMAKE_FAIL") == stage ? 7 : 0;
    }
}
'''


@unittest.skipUnless(os.name == "nt" and CSC.is_file(), "requires Windows PowerShell and .NET Framework csc")
class CMakeBatchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="leonida_cmake_")
        cls.addClassCleanup(cls.temp.cleanup)
        cls.base = Path(cls.temp.name)
        source = cls.base / "recorder.cs"
        source.write_text(RECORDER, encoding="utf-8", newline="\n")
        cls.recorder = cls.base / "recorder.exe"
        subprocess.run([str(CSC), "/nologo", "/target:exe", "/out:" + str(cls.recorder), str(source)],
                       check=True, capture_output=True, timeout=30)

    def run_build(self, folder, extension, fail=""):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index('$cfg = if ($Debug)')
        end = text.index('} else {', text.index('if ($cmake -and $ninja)', start))
        branch = text[start:end] + "}\n"
        writer = text[text.index('$bat = Join-Path $out "_build.cmd"'):]
        root = folder / "checkout"
        src, out = root / "runtime", root / "runtime/build"
        out.mkdir(parents=True)
        tools = folder / "tools ^ & 東京"
        tools.mkdir()
        backend = tools / "recorder.exe"
        shutil.copyfile(self.recorder, backend)
        cmake = tools / ("cmake" + extension)
        if extension == ".exe":
            shutil.copyfile(self.recorder, cmake)
        else:
            cmake.write_text('@echo off\nchcp 65001 >nul\n"' + str(backend) + '" %*\nexit /b %errorlevel%\n',
                             encoding="utf-8", newline="\r\n")
        # The real vcvars lives in the VS installation, outside the checkout.
        vcvars = self.base / "vcvars64.bat"
        vcvars.write_text('@exit /b 0\n', encoding="ascii", newline="\r\n")
        ninja = tools / "ninja.exe"
        command = ("$ErrorActionPreference = 'Stop'\n"
                   "$root = $env:LEONIDA_CMAKE_ROOT\n"
                   "$src = Join-Path $root 'runtime'\n"
                   "$out = Join-Path $src 'build'\n"
                   "$installer = $env:LEONIDA_CMAKE_INSTALLER\n"
                   "$vcvars = Join-Path $installer 'vcvars64.bat'\n"
                   "$cmake = $env:LEONIDA_CMAKE_EXE\n"
                   "$ninja = $env:LEONIDA_CMAKE_NINJA\n"
                   "$Debug = $false\n$FetchSdl3 = $true\n$jobs = 3\n"
                   + branch + writer)
        encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            env=dict(os.environ, LEONIDA_CMAKE_ROOT=str(root), LEONIDA_CMAKE_LOG=str(folder),
                     LEONIDA_CMAKE_INSTALLER=str(self.base),
                     LEONIDA_CMAKE_EXE=str(cmake), LEONIDA_CMAKE_NINJA=str(ninja),
                     LEONIDA_CMAKE_FAIL=fail, VULKAN_SDK=str(folder / "vulkan")),
            capture_output=True, timeout=45,
        )
        return result, src, out, ninja

    def test_native_and_batch_cmake_paths_and_exit_codes(self):
        for name in ("plain", "caret^dir", "пробел ^ & 東京"):
            for extension in (".exe", ".cmd", ".bat"):
                for fail in ("", "configure", "build"):
                    with self.subTest(folder=name, extension=extension, fail=fail):
                        with tempfile.TemporaryDirectory(dir=self.base) as tmp:
                            folder = Path(tmp) / name
                            folder.mkdir()
                            result, src, out, ninja = self.run_build(folder, extension, fail)
                            batch = (out / "_build.cmd").read_bytes()
                            self.assertNotIn(b"\n", batch.replace(b"\r\n", b""))
                            self.assertEqual(result.returncode, 1 if fail else 0,
                                             (result.stdout + result.stderr).decode(errors="replace"))
                            configure = (folder / "configure.txt").read_text(encoding="utf-8-sig").splitlines()
                            self.assertEqual(configure, ["-S", str(src), "-B", str(out), "-G", "Ninja",
                                                        "-DCMAKE_MAKE_PROGRAM=" + str(ninja),
                                                        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
                                                        "-DLEONIDA_SDL3_FETCH=ON", "-DLEONIDA_REQUIRE_VULKAN=ON"])
                            built = folder / "build.txt"
                            if fail == "configure":
                                self.assertFalse(built.exists(), "configure failure must stop before build")
                            else:
                                self.assertEqual(built.read_text(encoding="utf-8-sig").splitlines(),
                                                 ["--build", str(out), "-j", "3"])


if __name__ == "__main__":
    unittest.main()
