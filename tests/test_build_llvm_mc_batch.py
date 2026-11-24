"""Exercise the native LLVM builder's batch writer without compiling LLVM."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt", "requires Windows PowerShell and cmd.exe")
class BuildBatchTest(unittest.TestCase):
    def test_unicode_paths_exit_status_and_console_code_page(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index('$bat = Join-Path $Temp "_build.cmd"')
        writer = text[start:text.index("$p = Start-Process", start)]

        with tempfile.TemporaryDirectory(prefix="leonida & llvm ") as directory:
            work = Path(directory) / "LLVM 東京"
            work.mkdir()
            (work / "vcvars64.bat").write_bytes(b"@exit /b 0\r\n")
            driver = Path(directory) / "write-build.ps1"
            driver.write_text(
                "param([string]$Temp, [int]$Code)\n"
                "$ErrorActionPreference = 'Stop'\n"
                "$installer = $Temp\n"
                "$vcvars = Join-Path $Temp 'vcvars64.bat'\n"
                "$body = 'cd /d \"' + $Temp + '\" || exit /b 1' + \"`r`nexit /b $Code\"\n"
                + writer,
                encoding="utf-8", newline="\n",
            )
            console_check = work / "check-console.cmd"
            console_check.write_bytes(
                b"@echo off\r\nchcp 437 >nul || exit /b 99\r\nchcp >before.txt\r\n"
                b'call "%~dp0_build.cmd"\r\nset "build_exit=%errorlevel%"\r\nchcp >after.txt\r\nexit /b %build_exit%\r\n'
            )
            startup = subprocess.STARTUPINFO()
            startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startup.wShowWindow = subprocess.SW_HIDE
            for code in (0, 17):
                with self.subTest(exit_code=code):
                    subprocess.run(
                        ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                         "-File", str(driver), "-Temp", str(work), "-Code", str(code)],
                        check=True, timeout=30,
                    )
                    batch = (work / "_build.cmd").read_bytes()
                    self.assertTrue(batch.startswith(b"@echo off\r\n"))
                    self.assertIn(str(work).encode("utf-8"), batch)
                    result = subprocess.run(
                        ["cmd.exe", "/d", "/c", r".\check-console.cmd"], cwd=work, timeout=30,
                        env={**os.environ, "NoDefaultCurrentDirectoryInExePath": "1"},
                        creationflags=subprocess.CREATE_NEW_CONSOLE, startupinfo=startup, capture_output=True,
                    )
                    self.assertEqual(result.returncode, code, result.stdout + result.stderr)
                    before = (work / "before.txt").read_bytes()
                    self.assertIn(b"437", before)
                    self.assertEqual((work / "after.txt").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
