import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "meta_to_xml.ps1"
POWERSHELL_EXE = Path(os.environ.get("SystemRoot", r"C:\Windows")).joinpath(
    "System32", "WindowsPowerShell", "v1.0", "powershell.exe"
)
POWERSHELL = [str(POWERSHELL_EXE), "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass"]
BUILD_STUB = """
$ErrorActionPreference = 'Stop'
Add-Type -TypeDefinition 'public class SharpDXPlaceholder {}' -OutputAssembly SharpDX.dll
Add-Type -TypeDefinition 'public class MathematicsPlaceholder {}' -OutputAssembly SharpDX.Mathematics.dll
Add-Type -OutputAssembly CodeWalker.Core.dll -TypeDefinition @'
namespace CodeWalker.GameFiles {
    public class PsoFile {
        public void Load(byte[] bytes) {
            if (bytes.Length == 4)
                throw new System.InvalidOperationException("synthetic parser failure");
        }
    }
    public static class PsoXml {
        public static string GetXml(PsoFile file) { return "<synthetic />"; }
    }
}
'@
"""


@unittest.skipUnless(os.name == "nt", "Windows PowerShell 5.1 script")
class MetaToXmlPathsTest(unittest.TestCase):
    def test_plain_bracketed_and_unicode_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            deps = root / "deps"
            deps.mkdir()
            # Use PowerShell's built-in compiler to stand in for the three DLLs.
            build = subprocess.run(
                [*POWERSHELL, "-Command", "-"], cwd=deps, input=BUILD_STUB + "\n",
                capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)

            cases = (
                ("carve", "archive", "out"),
                ("carve [1]", "archive", "out"),
                ("carve", "archive [2]", "out"),
                ("carve", "archive", "out [3]"),
                ("carve", "archive 東京🧪", "out"),
            )
            for i, (carve_name, archive_name, out_name) in enumerate(cases):
                with self.subTest(carve=carve_name, archive=archive_name, out=out_name):
                    case = root / str(i)
                    carve, out = case / carve_name, case / out_name
                    archive = carve / archive_name
                    archive.mkdir(parents=True)
                    (archive / "index.json").write_text(json.dumps({"files": [
                        {"head": "5053494e", "off": 8},  # Indexed blob is missing.
                        {"head": "5053494e00", "off": 0},
                        {"head": "5053494e", "off": 16},
                    ]}), encoding="utf-8")
                    (archive / "000000000000.bin").write_bytes(b"PSIN\x00")
                    (archive / "000000000010.bin").write_bytes(b"PSIN")

                    result = subprocess.run(
                        [*POWERSHELL, "-File", str(SCRIPT), "-Carve", str(carve),
                         "-Out", str(out), "-CodeWalker", str(deps)],
                        capture_output=True, text=True, timeout=30,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertIn("pso ok 1 failed 2; rbf ok 0 failed 0", result.stdout)
                    self.assertEqual((out / archive_name / "000000000000.pso.xml").read_text(),
                                     "<synthetic />")
                    failures = (out / "failures.txt").read_text(encoding="utf-8-sig").splitlines()
                    self.assertEqual(len(failures), 3)
                    self.assertTrue(failures[1].startswith(f"{archive_name}\\000000000008 pso : "), failures[1])
                    self.assertEqual(failures[2], f"{archive_name}\\000000000010 pso : synthetic parser failure")


if __name__ == "__main__":
    unittest.main()
