import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt", "Windows LLVM installer")
class InstallDestinationTest(unittest.TestCase):
    def test_cleanup_boundary_before_build(self):
        cases = [
            ("work", False, True),
            ("WORK/bin", False, True),
            ("other/../work/bin", False, True),
            ("work-installed", False, False),
            ("work/bin", True, False),
        ]
        with tempfile.TemporaryDirectory() as cwd:
            for dest, keep, rejected in cases:
                with self.subTest(dest=dest, keep=keep):
                    cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(SCRIPT), "-Temp", "work", "-Dest", dest, "-NoEnv"]
                    if keep:
                        cmd.append("-Keep")
                    # An accepted path reaches the prerequisite check. Stop there.
                    result = subprocess.run(cmd, cwd=cwd, env={**os.environ, "PATH": ""},
                                            capture_output=True, text=True,
                                            encoding="utf-8", errors="replace")  # localized PowerShell output is not UTF-8
                    self.assertNotEqual(result.returncode, 0)
                    expected = "Cleanup would delete" if rejected else "git wasn't found"
                    self.assertIn(expected, result.stderr)
            self.assertEqual(list(Path(cwd).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
