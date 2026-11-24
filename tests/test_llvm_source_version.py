"""Exercise source selection with tiny local Git repos; no LLVM download or build."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_llvm_mc.ps1"


@unittest.skipUnless(os.name == "nt" and shutil.which("git"), "Windows and Git required")
class SourceVersionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.work = self.root / "retained work"
        self.src = self.work / "src"
        (self.src / "llvm").mkdir(parents=True)
        (self.src / "llvm" / "CMakeLists.txt").write_text(
            "# synthetic source\n", encoding="utf-8", newline="\n")
        (self.work / "build").mkdir()
        (self.work / "build" / "sentinel").write_bytes(b"keep existing build")
        self.git("init", "-q")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "synthetic release")
        self.git("tag", "llvmorg-21.1.8")
        text = SCRIPT.read_text(encoding="utf-8-sig")
        start = text.index('$src = Join-Path $Temp "src"')
        block = text[start:text.index("# DIA SDK off:", start)]
        self.driver = self.root / "select.ps1"
        self.driver.write_text(
            'param([string]$Temp, [string]$Version)\n'
            '$ErrorActionPreference = "Stop"\n'
            'function Remove-Item { throw "Unexpected deletion" }\n'
            + block + '\nWrite-Output "SOURCE_ACCEPTED"\n',
            encoding="utf-8", newline="\n")

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.src), *args], check=True,
                              capture_output=True, text=True, timeout=15).stdout.strip()

    def select(self, version, accepted):
        before = {str(p.relative_to(self.work)): p.read_bytes()
                  for p in self.work.rglob("*") if p.is_file()}
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
             "-File", str(self.driver), "-Temp", str(self.work), "-Version", version],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=20)
        self.assertEqual(result.returncode == 0, accepted, result.stdout + result.stderr)
        if accepted:
            self.assertIn("SOURCE_ACCEPTED", result.stdout)
        else:
            self.assertIn("Choose a fresh -Temp", result.stderr)
            self.assertNotIn("SOURCE_ACCEPTED", result.stdout)
        after = {str(p.relative_to(self.work)): p.read_bytes()
                 for p in self.work.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_same_release_reused(self):
        self.select("21.1.8", True)

    def test_different_release_missing_tag_rejected(self):
        self.select("23.1.5", False)

    def test_unknown_release_rejected(self):
        self.select("not-a-real-release", False)

    def test_existing_tag_at_different_commit_rejected(self):
        (self.src / "llvm" / "CMakeLists.txt").write_text(
            "# another synthetic release\n", encoding="utf-8", newline="\n")
        self.git("add", ".")
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "commit", "-qm", "second release")
        self.git("tag", "llvmorg-23.1.5")
        self.select("21.1.8", False)
        self.select("23.1.5", True)

    def test_annotated_release_tag_reused(self):
        self.git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                 "tag", "-am", "release", "llvmorg-23.1.5")
        self.select("23.1.5", True)

    def test_sources_without_git_metadata_rejected(self):
        # Hide the fixture's metadata without deleting it. Do not use an enclosing repo's HEAD.
        (self.src / ".git").rename(self.root / "saved-git")
        self.select("21.1.8", False)


if __name__ == "__main__":
    unittest.main()
