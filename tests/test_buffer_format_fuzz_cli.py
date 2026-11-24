"""CPU-only tests for tools/buffer_format_fuzz.py's command line: an explicit --formats is checked before the GPU.

    python -I tests/test_buffer_format_fuzz_cli.py

A typo in --formats used to be found only after runner.open_oracle(), so a machine without the oracle answered
"hardware oracle unavailable" (77) and one with it raised SystemExit (1). It is a usage error (2) before the oracle.
"""
import importlib.util
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(REPO, "tools", "buffer_format_fuzz.py")


def run(*args):
    return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, check=False)


def load_tool():
    spec = importlib.util.spec_from_file_location("buffer_format_fuzz", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BufferFormatFuzzFormatsTests(unittest.TestCase):
    def test_unknown_format_is_a_usage_error_before_the_oracle(self):
        proc = run("--oracle", "hw", "--formats", "8_8_8_8_UNORM_typo")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unknown format '8_8_8_8_UNORM_typo'", proc.stderr)
        self.assertNotIn("hardware oracle", proc.stdout + proc.stderr)

    def test_a_code_outside_the_table_is_refused_too(self):
        proc = run("--oracle", "hw", "--formats", "0,8_8_8_8_UNORM")
        self.assertEqual(proc.returncode, 2, proc.stdout + proc.stderr)
        self.assertIn("unknown format '0'", proc.stderr)

    def test_known_names_and_codes_map_to_the_table(self):
        tool = load_tool()
        pairs = tool.select_formats("8_8_8_8_UNORM,56")
        self.assertEqual([n for _, n in pairs], ["8_8_8_8_UNORM", tool.GFX10_NAMES[56]])
        with self.assertRaises(ValueError):
            tool.select_formats("not_a_format")


if __name__ == "__main__":
    unittest.main()
