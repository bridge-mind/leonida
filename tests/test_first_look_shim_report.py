"""Exercise the on-disk first-look report while a missing dependency is supplied."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pe_inspect import build_image


@unittest.skipUnless(os.name == "nt", "Windows PowerShell diagnostics")
class FirstLookShimReportTest(unittest.TestCase):
    def test_retry_with_last_missing_dll_supplied(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Every tool first_look.ps1 runs: a missing one fails its step, and the test's stand-in loader (an alias)
            # leaves that exit code in $LASTEXITCODE for the script's exit. (gdk_runtime_calls since round 43,
            # find_hash_lookup for mods, exe_checklist for the notes to confirm.)
            for name in ("scripts/first_look.ps1", "tools/pe_inspect.py", "tools/gdk_runtime_calls.py",
                         "tools/gdk_runtime_classes.json", "tools/find_hash_lookup.py", "tools/exe_checklist.py"):
                (root / name).parent.mkdir(exist_ok=True)
                shutil.copyfile(source / name, root / name)
            loader = root / "runtime/build/gta6rt.exe"
            loader.parent.mkdir(parents=True)
            loader.touch()
            loader.with_name("gta6rt_tls_host.dll").touch()
            (root / "fixture.exe").write_bytes(build_image().replace(b"xmem.dll", b"demo.dll"))
            # The report depends on the real PE inspection. Echo the dry-run arguments
            # in place of a native loader so the test needs no runtime build.
            script = r"""
$ErrorActionPreference = 'Stop'
Set-Alias -Name (Join-Path $PWD 'runtime\build\gta6rt.exe') -Value Write-Output
& .\scripts\first_look.ps1 -Exe (Join-Path $PWD 'fixture.exe')
"""
            env = {k: v for k, v in os.environ.items() if k.upper() != "PSMODULEPATH"}
            env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env["PATH"]
            reports = root / "reports/exe/fixture"
            todo = reports / "shim-todo.txt"

            def run_first_look():
                run = subprocess.run(
                    ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                     "-Command", script], cwd=root, env=env, capture_output=True, timeout=30,
                )
                self.assertEqual(run.returncode, 0, run.stderr)
                # A failed find_hash_lookup step is only a warning in the script and later steps reset
                # $LASTEXITCODE, so the exit status doesn't show it: check that each step wrote its report.
                for report in ("fixture.exe.gdk_calls.json", "hash_lookup.json"):
                    self.assertTrue((reports / report).is_file(), (report, run.stdout, run.stderr))
                imports = json.loads((reports / "fixture.exe.json").read_text("utf-8"))["imports"]
                return {entry["dll"].lower() for entry in imports if entry["where"] == "shim"}

            self.assertEqual(run_first_look(), {"demo.dll"})
            self.assertIn("demo.dll", todo.read_text().lower())
            (root / "demo.dll").touch()
            self.assertEqual(run_first_look(), set())
            self.assertEqual(todo.read_bytes(), b"")
            todo.unlink()
            self.assertEqual(run_first_look(), set())
            self.assertEqual(todo.read_bytes(), b"")


if __name__ == "__main__":
    unittest.main()
