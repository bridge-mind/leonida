"""CPU-only checks for the first-light harness lookup."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import first_light


class FirstLightHarnessPathsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(first_light, "ROOT", str(self.root)))
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("LEONIDA_VK_HARNESS", None)
        self.executable = "vk_harness.exe" if os.name == "nt" else "vk_harness"

    def put_harness(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts, self.executable)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        return path

    def test_finds_windows_release_configuration(self) -> None:
        release = self.put_harness("runtime", "build", "windows", "Release")
        self.assertEqual(first_light.find_harness(None), str(release))

    def test_finds_relwithdebinfo_in_the_standard_build_tree(self) -> None:
        self.put_harness("runtime", "build", "Release")
        optimized = self.put_harness("runtime", "build", "RelWithDebInfo")
        self.assertEqual(first_light.find_harness(None), str(optimized))

    def test_standalone_harness_keeps_priority_over_configured_harness(self) -> None:
        self.put_harness("runtime", "build", "windows", "Release")
        standalone = self.put_harness("runtime", "build")
        self.assertEqual(first_light.find_harness(None), str(standalone))

    def test_explicit_and_environment_paths_keep_priority(self) -> None:
        configured = self.put_harness("runtime", "build", "windows", "Release")
        explicit = self.put_harness("explicit")
        environment = self.put_harness("environment")
        os.environ["LEONIDA_VK_HARNESS"] = str(environment)
        self.assertEqual(first_light.find_harness(str(explicit)), str(explicit))
        self.assertEqual(first_light.find_harness(str(explicit) + ".missing"), str(environment))
        environment.unlink()
        self.assertEqual(first_light.find_harness(None), str(configured))


if __name__ == "__main__":
    unittest.main()
