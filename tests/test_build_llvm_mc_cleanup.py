"""Exercise the native LLVM builder's robust directory cleanup handling read-only attributes."""
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt", "requires Windows PowerShell")
class BuildCleanupTest(unittest.TestCase):
    def test_cleanup_removes_readonly_nested_files_and_directories(self):
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index("function Remove-DirectoryRobust")
        func_def = text[start:text.index("if (-not (Test-Path (Join-Path $src", start)].strip()

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "test_temp_cleanup"
            target.mkdir()
            sub = target / ".git" / "objects" / "pack"
            sub.mkdir(parents=True)
            ro_file = sub / "sample_pack.idx"
            ro_file.write_bytes(b"DATA")
            # Mark file and directory read-only on Windows
            os.chmod(ro_file, stat.S_IREAD)
            os.chmod(sub, stat.S_IREAD)

            driver = Path(directory) / "test-clean.ps1"
            driver.write_text(
                "param([string]$Path)\n"
                "$ErrorActionPreference = 'Stop'\n"
                + func_def + "\n"
                "Remove-DirectoryRobust $Path\n",
                encoding="utf-8", newline="\n",
            )
            res = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                 "-File", str(driver), "-Path", str(target)],
                capture_output=True, text=True, check=True, timeout=30,
            )
            self.assertFalse(target.exists(), f"Target directory should have been removed: {res.stderr}")


if __name__ == "__main__":
    unittest.main()
