"""BUILD-1046: a missing input path is a usage error (exit 2), not a failed run (exit 1).

Each tool is run with a nonexistent positional input. All of them build their inputs themselves in this test;
no game file or asset is used. `image_query_probe.py --help` is bounded so its argument parsing cannot grow the
hardware-oracle startup cost back in.
"""
import os
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")

MISSING_INPUT_TOOLS = (
    "analyze_4240_full",
    "audio_config",
    "bary_fallback_gs",
    "content_definitions",
    "cpu_ref",
    "gamedata_maps",
    "gamedata_sps1",
    "navmesh_index",
    "pe_inspect",
    "rtti_scan",
    "shader_disasm",
    "tile_probe",
    "d3d12_guid_sites",
    "gamedata_blobs",
    "navmesh_cell_fit",
    "psin_psch",
    "root_spill_census",
)

# One process per tool; a hung import is a failure, not an infinite test.
TIMEOUT = 120


def run_tool(tool, *args, timeout=TIMEOUT):
    return subprocess.run([sys.executable, "-I", os.path.join(TOOLS, tool + ".py"), *args],
                          capture_output=True, text=True, timeout=timeout, errors="replace")


class MissingInputExitCodeTests(unittest.TestCase):
    def test_missing_input_is_exit_2_with_one_error_line(self):
        missing = os.path.join(HERE, "no-such-input-for-build-1046")
        self.assertFalse(os.path.exists(missing), "the fixture path must not exist")
        for tool in MISSING_INPUT_TOOLS:
            with self.subTest(tool=tool):
                p = run_tool(tool, missing)
                self.assertEqual(p.returncode, 2, p.stderr)
                self.assertNotIn("Traceback", p.stderr)
                lines = p.stderr.strip().splitlines()
                self.assertEqual(len(lines), 1, p.stderr)
                self.assertTrue(lines[0].strip())
                self.assertTrue(lines[0].startswith("ERROR:"), p.stderr)
                self.assertIn(missing, p.stderr)


class ImageQueryProbeHelpTests(unittest.TestCase):
    def test_help_returns_promptly_without_the_hardware_oracle(self):
        p = run_tool("image_query_probe", "--help", timeout=2)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("--self-test", p.stdout)
        self.assertNotIn("Traceback", p.stderr)


if __name__ == "__main__":
    unittest.main()
