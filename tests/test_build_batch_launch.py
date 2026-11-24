"""The Windows build scripts start their generated batch file from any folder name, `&` and `^` included.

    python -I tests/test_build_batch_launch.py

`scripts/build_runtime.ps1` and `scripts/build_llvm_mc.ps1` write `_build.cmd` into the build folder and start it
with `Start-Process cmd.exe -ArgumentList ...`. With `/c "path"`, cmd.exe drops the quotes when the path holds a
special character such as `&` or `^` and then runs only the part before it. This test takes each script's launch
line as written and runs it on a stub batch in such folders: the stub must run, and its exit code must come back.
Windows only (PowerShell and cmd.exe); no compiler, no game files.
"""
import base64
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("build_runtime.ps1", "build_llvm_mc.ps1")
LAUNCH = re.compile(r"^\$p = Start-Process cmd\.exe -ArgumentList (.+?) -NoNewWindow -PassThru\s*$", re.M)
FOLDERS = ("plain", "a & b", "a&b", "x^y", "sp (x)", "東京 & co")


def launch_arguments(script):
    """The -ArgumentList expression of the script's one `$p = Start-Process cmd.exe` line."""
    found = LAUNCH.findall((ROOT / "scripts" / script).read_text(encoding="utf-8-sig"))
    if len(found) != 1:
        raise AssertionError(f"{script}: expected one `$p = Start-Process cmd.exe -ArgumentList ...` line, found "
                             f"{len(found)}; update tests/test_build_batch_launch.py together with the script")
    return found[0]


def start(arguments, bat):
    """Run `$p = Start-Process cmd.exe -ArgumentList <arguments>` with $bat set, as the script does; its exit code."""
    command = ("$ErrorActionPreference = 'Stop'\n"
               "$bat = $env:LEONIDA_LAUNCH_BAT\n"
               f"$p = Start-Process cmd.exe -ArgumentList {arguments} -NoNewWindow -PassThru\n"
               "$null = $p.Handle\n"
               "$p.WaitForExit()\n"
               "exit $p.ExitCode\n")
    encoded = base64.b64encode(command.encode("utf-16-le")).decode("ascii")
    return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                          env=dict(os.environ, LEONIDA_LAUNCH_BAT=str(bat)), capture_output=True, timeout=60)


@unittest.skipUnless(sys.platform == "win32", "requires Windows PowerShell and cmd.exe")
class BatchLaunchTest(unittest.TestCase):
    def check(self, arguments):
        """Folders the launch line fails in, as `name/exit code` strings."""
        broken = []
        with tempfile.TemporaryDirectory() as directory:
            for name in FOLDERS:
                folder = Path(directory) / name
                folder.mkdir()
                bat = folder / "_build.cmd"
                ran = folder / "ran.txt"
                for code in (0, 7):
                    bat.write_bytes(f'@echo off\r\necho ran> "%~dp0ran.txt"\r\nexit /b {code}\r\n'.encode("ascii"))
                    if ran.exists():
                        ran.unlink()
                    result = start(arguments, bat)
                    if not ran.exists() or result.returncode != code:
                        broken.append(f"{name}/{code}->{result.returncode}")
        return broken

    def test_each_script_starts_its_batch_in_any_folder(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                self.assertEqual(self.check(launch_arguments(script)), [],
                                 f"{script}'s launch line does not run the batch in these folders")

    def test_the_check_catches_a_launch_line_that_drops_the_quotes(self):
        # The form both scripts used before: cmd.exe /c "path" loses the quotes around a path holding & or ^.
        broken = self.check('"/c", "`"$bat`""')
        self.assertIn("a & b/0->1", broken)
        self.assertIn("x^y/7->1", broken)
        self.assertFalse([b for b in broken if b.startswith(("plain/", "sp (x)/"))], broken)


if __name__ == "__main__":
    unittest.main()
