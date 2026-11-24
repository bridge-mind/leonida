"""Exercise the day-one report from a PowerShell location different from the process working directory."""
import hashlib
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


@unittest.skipUnless(os.name == "nt", "Windows PowerShell first-look diagnostics")
class FirstLookPathTests(unittest.TestCase):
    def test_literal_image_loader_and_report_paths(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp, "checkout [1]")
            for folder in ("scripts", "tools", "runtime/build/Release"):
                (repo / folder).mkdir(parents=True)
            for relative in ("scripts/first_look.ps1", "tools/pe_inspect.py", "tools/gdk_runtime_calls.py",
                             "tools/gdk_runtime_classes.json", "tools/find_hash_lookup.py", "tools/exe_checklist.py"):
                shutil.copyfile(source / relative, repo / relative)
            game = Path(tmp, "package [1]")
            game.mkdir()
            image = game / "fixture[1].exe"
            # Keep a bundled import and a missing import so both reports have evidence.
            image.write_bytes(build_image().replace(b"kernel32.dll", b"fake_lib.dll"))
            (game / "fixture1.exe").write_bytes(b"wrong file")
            (game / "xmem.dll").touch()
            env = {
                **os.environ,
                "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                "PSModulePath": str(Path(os.environ["SystemRoot"], "System32/WindowsPowerShell/v1.0/Modules")),
                "FIRST_LOOK_SCRIPT": str(repo / "scripts/first_look.ps1"),
                "FIRST_LOOK_PACKAGE": str(game),
            }
            ps = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command"]
            # Stand-in for gta6rt: record the actual arguments in the normal dry-run report.
            subprocess.run([*ps, """$ErrorActionPreference = 'Stop'
Add-Type -OutputAssembly gta6rt.exe -OutputType ConsoleApplication -TypeDefinition '
public class Loader { public static void Main(string[] args) { System.Console.WriteLine(string.Join("\\n", args)); } }'
"""], cwd=tmp, env=env, check=True, timeout=30)
            loader = repo / "runtime/build/Release/gta6rt.exe"
            shutil.move(Path(tmp, "gta6rt.exe"), loader)
            loader.with_name("gta6rt_tls_host.dll").touch()
            command = [*ps, "Set-Location -LiteralPath $env:FIRST_LOOK_PACKAGE; "
                            "& $env:FIRST_LOOK_SCRIPT -Exe 'fixture[1].exe'"]
            result = subprocess.run(command, cwd=tmp, env=env, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            out = repo / "reports/exe" / image.stem
            self.assertIn(hashlib.sha256(image.read_bytes()).hexdigest(), (out / "hash.txt").read_text("utf-16"))
            self.assertEqual((out / "dryrun.txt").read_text("utf-16").splitlines(),
                             ["--dry-run", str(image), "--game-dir", str(game)])
            imports = json.loads((out / (image.name + ".json")).read_text("utf-8"))["imports"]
            self.assertTrue(any(d["dll"].lower() == "xmem.dll" and d["where"] == "bundled" for d in imports))
            self.assertIn("fake_lib.dll", (out / "shim-todo.txt").read_text())
            self.assertIn("the called ones come first", (out / "checklist.md").read_text("utf-8"))

            # A source update in the bracketed checkout must still trigger the rebuild.
            source_dir = repo / "runtime/src"
            source_dir.mkdir()
            (source_dir / "main.cpp").touch()
            older = (source_dir / "main.cpp").stat().st_mtime - 3600
            os.utime(loader, (older, older))
            (repo / "scripts/build_runtime.ps1").write_text(
                'Copy-Item -LiteralPath "$PSScriptRoot/../runtime/build/Release/gta6rt.exe" '
                '-Destination "$PSScriptRoot/../runtime/build/gta6rt.exe"\n'
                'Copy-Item -LiteralPath "$PSScriptRoot/../runtime/build/Release/gta6rt_tls_host.dll" '
                '-Destination "$PSScriptRoot/../runtime/build/gta6rt_tls_host.dll"\n'
                'Set-Content -LiteralPath (Join-Path $PSScriptRoot "built.txt") -Value built\n',
                encoding="utf-8", newline="\n",
            )
            rebuilt = subprocess.run(command, cwd=tmp, env=env, capture_output=True, timeout=30)
            self.assertEqual(rebuilt.returncode, 0, rebuilt.stdout + rebuilt.stderr)
            self.assertTrue((repo / "scripts/built.txt").is_file())

            image.unlink()  # fixture1.exe still exists and would match the wildcard.
            missing = subprocess.run(command, cwd=tmp, env=env, capture_output=True, timeout=30)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn(b"fixture[1].exe not found", missing.stderr)


if __name__ == "__main__":
    unittest.main()
