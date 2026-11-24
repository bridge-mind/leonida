"""Check the runtime build wrapper's batch encoding without compiling the runtime."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_runtime.ps1"


@unittest.skipUnless(os.name == "nt", "requires Windows PowerShell and cmd.exe")
class BuildRuntimeBatchTest(unittest.TestCase):
    def test_build_cmd_preserves_unicode_paths(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index('$bat = Join-Path $out "_build.cmd"')
        writer = text[start:text.index("$p = Start-Process", start)]

        with tempfile.TemporaryDirectory(prefix="leonida & batch ") as directory:
            checkout = Path(directory) / "Leonida 東京"
            checkout.mkdir()
            (checkout / "vcvars64.bat").write_bytes(b"@exit /b 0\r\n")
            driver = Path(directory) / "write-build.ps1"
            driver.write_text(
                "param([string]$out, [string]$installer)\n"
                "$ErrorActionPreference = 'Stop'\n"
                "$vcvars = Join-Path $installer 'vcvars64.bat'\n"
                "$body = 'cd /d \"' + $installer + '\" || exit /b 1'\n"
                + writer,
                encoding="utf-8", newline="\n",
            )
            subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(driver), "-out", directory, "-installer", str(checkout)],
                check=True, timeout=30,
            )
            batch_path = Path(directory) / "_build.cmd"
            batch = batch_path.read_bytes()
            self.assertTrue(batch.startswith(b"@echo off\r\n"))
            self.assertIn(str(checkout).encode("utf-8"), batch)
            subprocess.run(
                ["cmd.exe", "/d", "/c", r".\_build.cmd"], cwd=directory,
                env={**os.environ, "NoDefaultCurrentDirectoryInExePath": "1"}, check=True, timeout=30,
            )


if __name__ == "__main__":
    unittest.main()
