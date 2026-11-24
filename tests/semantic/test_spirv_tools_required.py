"""Where CI promises SPIRV-Tools, a missing spirv-as / spirv-val is a failure, not a quiet skip (TEST-604).

Several semantic tests assemble and validate the translator's output and skip when `spirv-as` / `spirv-val` are not on
PATH: test_fma_khr, test_bary_fallback, test_image_coherent and others. A GitHub runner has neither tool, so without
an install step those tests report "skipped" and the job stays green. The `semantic-tests` job in
.github/workflows/ci.yml installs SPIRV-Tools on its Linux leg and sets LEONIDA_REQUIRE_SPIRV_TOOLS=1 there; this test
turns a missing or broken tool into a red job. Without the variable (a developer's machine, the Windows leg) it skips.
tests/test_ci_registration.py checks that ci.yml keeps doing both.
"""
import os
import shutil
import subprocess
import unittest


@unittest.skipUnless(os.environ.get("LEONIDA_REQUIRE_SPIRV_TOOLS") == "1",
                     "LEONIDA_REQUIRE_SPIRV_TOOLS is not 1 (CI sets it where it installs SPIRV-Tools)")
class SpirvToolsRequiredTest(unittest.TestCase):
    def test_spirv_as_and_spirv_val_are_installed_and_run(self):
        for tool in ("spirv-as", "spirv-val"):
            path = shutil.which(tool)
            self.assertIsNotNone(path, f"{tool} is not on PATH: the tests gated on SPIRV-Tools would all skip")
            result = subprocess.run([path, "--version"], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{tool} --version failed: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
