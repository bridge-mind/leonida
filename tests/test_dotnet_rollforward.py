"""Every .NET program in the repository may run on a newer runtime than the one it targets.

The projects target net8.0, but a machine with only a newer .NET (the 10 SDK and runtime that current distributions
ship, for instance) builds them and then cannot start them: the host stops with "You must install or update .NET
to run this application ... Framework 'Microsoft.NETCore.App', version '8.0.0'" (exit code 150). make_game.py's
preflight and global.json already accept a newer SDK, so the built program has to accept a newer runtime too: an
`Exe` project sets <RollForward> to Major or LatestMajor, which goes into its runtimeconfig.json. A machine that has
the 8 runtime keeps using it. Standard library only; no .NET needed.
"""

import os
import unittest
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {".git", "bin", "obj", "node_modules", "third_party", "build"}
POLICIES = {"Major", "LatestMajor"}


def project_files():
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith("build")]
        for name in files:
            if name.endswith((".csproj", ".fsproj", ".vbproj")):
                yield os.path.join(folder, name)


def property_values(path, name):
    """The values of <name> in every PropertyGroup of an SDK-style project file."""
    root = ET.parse(path).getroot()
    return [(el.text or "").strip() for group in root.iter("PropertyGroup") for el in group.findall(name)]


class RollForwardTest(unittest.TestCase):
    def test_the_projects_are_found(self):
        names = {os.path.basename(p) for p in project_files()}
        self.assertIn("rpf9dump.csproj", names)
        self.assertIn("Rpf9DumpDiagnostics.csproj", names)

    def test_every_executable_project_rolls_forward(self):
        checked = 0
        for path in project_files():
            if "Exe" not in property_values(path, "OutputType"):
                continue
            checked += 1
            rel = os.path.relpath(path, ROOT)
            with self.subTest(project=rel):
                self.assertTrue(
                    set(property_values(path, "RollForward")) & POLICIES,
                    f"{rel} is an Exe without <RollForward>Major</RollForward>: with only a newer .NET runtime "
                    "installed it builds but does not start (host exit 150)")
        self.assertGreaterEqual(checked, 2)


if __name__ == "__main__":
    unittest.main()
