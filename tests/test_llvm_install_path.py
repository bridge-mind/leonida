"""Exercise the real install tail without building LLVM or changing user environment variables."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt", "Windows LLVM installer")
class InstallPathTest(unittest.TestCase):
    def install(self, absolute=False, no_env=False):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cwd = root / "install from"
            other = root / "run from"
            cwd.mkdir()
            other.mkdir()
            source = root / "synthetic.exe"
            source.write_bytes(b"synthetic placeholder, never executed")
            destination = cwd / "tools with spaces" / "llvm-mc"
            dest_arg = str(destination) if absolute else r".\tools with spaces\llvm-mc"
            text = SCRIPT.read_text(encoding="utf-8-sig")
            tail = text[text.index('New-Item -ItemType Directory -Force $Dest'):]
            # Capture exactly what the production tail would persist, without touching the registry.
            setter = '[Environment]::SetEnvironmentVariable("LLVM_MC", $final, "User")'
            self.assertEqual(tail.count(setter), 1)
            tail = tail.replace(setter, '$script:saved = $final')
            driver = root / "install.ps1"
            driver.write_text(
                'param([string]$Dest, [string]$exe, [switch]$NoEnv)\n'
                '$ErrorActionPreference = "Stop"\n$Keep = $true\n$script:saved = $null\n'
                + tail + '\nWrite-Output ("RESULT:" + (@{final=$final; saved=$script:saved} | ConvertTo-Json -Compress))\n',
                encoding="utf-8", newline="\n")
            command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                       "-File", str(driver), "-Dest", dest_arg, "-exe", str(source)]
            if no_env:
                command.append("-NoEnv")
            result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=20)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = json.loads(next(line[7:] for line in result.stdout.splitlines()
                                   if line.startswith("RESULT:")))
            installed = destination / "llvm-mc.exe"
            self.assertEqual(installed.read_bytes(), source.read_bytes())
            self.assertTrue(Path(data["final"]).is_absolute(), data)
            self.assertEqual(Path(data["final"]).resolve(), installed.resolve())
            if no_env:
                self.assertIsNone(data["saved"])
            else:
                self.assertEqual(data["saved"], data["final"])
                # Resolve the saved path from a different process working directory.
                check = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
                     "if (-not (Test-Path -LiteralPath $env:LLVM_MC)) { exit 1 }"],
                    cwd=other, env={**os.environ, "LLVM_MC": data["saved"]},
                    capture_output=True, timeout=20)
                self.assertEqual(check.returncode, 0, check.stderr)

    def test_relative_destination_survives_directory_change(self):
        self.install()

    def test_absolute_destination_survives_directory_change(self):
        self.install(absolute=True)

    def test_no_env_still_installs_without_persisting(self):
        self.install(no_env=True)


if __name__ == "__main__":
    unittest.main()
