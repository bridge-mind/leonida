"""spirv_metal_set0.py on the command line: a SPIRV-Tools program that is missing or fails is one ERROR line.

    python -I tests/test_spirv_metal_set0_cli_errors.py

`spirv_metal_set0.py in.spv out.spv` ended in a traceback when spirv-dis or spirv-as could not be started
(FileNotFoundError) or rejected its input (CalledProcessError). No SPIRV-Tools are needed here: SPIRV_DIS and
SPIRV_AS point at stand-ins. The Python interpreter is the one that works on every platform: as "spirv-dis" it runs
the input file, so an input that is a script printing two disassembly lines is a spirv-dis that succeeds, and a
missing input is one that fails; as "spirv-as" it rejects `--target-env` and fails.
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "spirv_metal_set0.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import spirv_metal_set0 as m  # noqa: E402

PRINTS_A_DISASSEMBLY = 'print("; Version: 1.5")\nprint("               OpCapability Shader")\n'
HINT = "(SPIRV-Tools must be on PATH, or set SPIRV_DIS / SPIRV_AS)"


class CliErrorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "in.spv")
        self.out = os.path.join(self.tmp.name, "out.spv")
        self.missing = os.path.join(self.tmp.name, "no-such-spirv-tool")
        with open(self.src, "w", encoding="utf-8", newline="\n") as f:
            f.write(PRINTS_A_DISASSEMBLY)

    def run_tool(self, dis, asm, src=None):
        env = dict(os.environ, SPIRV_DIS=dis, SPIRV_AS=asm)
        # The tool is a Python child: piped, it writes in the locale encoding (the ANSI code page on Windows), so read
        # it back in that encoding. UTF-8 would turn a non-ASCII temp path into replacement characters.
        return subprocess.run([sys.executable, "-I", TOOL, src or self.src, self.out], env=env, capture_output=True,
                              text=True, errors="replace")

    def assert_failed(self, result):
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(os.path.exists(self.out))
        return result.stderr.splitlines()

    def test_spirv_dis_that_cannot_be_started(self):
        lines = self.assert_failed(self.run_tool(self.missing, sys.executable))
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(f"ERROR: {self.missing}: "), lines)
        self.assertTrue(lines[0].endswith(HINT), lines)

    def test_spirv_dis_that_rejects_the_input(self):
        absent = os.path.join(self.tmp.name, "absent.spv")
        lines = self.assert_failed(self.run_tool(sys.executable, sys.executable, src=absent))
        self.assertEqual(len(lines), 1, lines)
        # The stand-in's own complaint about the file follows the exit status.
        self.assertTrue(lines[0].startswith(f"ERROR: {sys.executable} failed with exit status 2: "), lines)
        self.assertIn("absent.spv", lines[0])

    def test_spirv_as_that_cannot_be_started(self):
        lines = self.assert_failed(self.run_tool(sys.executable, self.missing))
        self.assertEqual(len(lines), 1, lines)
        self.assertTrue(lines[0].startswith(f"ERROR: {self.missing}: "), lines)
        self.assertTrue(lines[0].endswith(HINT), lines)

    def test_spirv_as_that_fails(self):
        lines = self.assert_failed(self.run_tool(sys.executable, sys.executable))
        # spirv-as writes to the terminal itself; the tool's own line comes last.
        self.assertEqual(lines[-1], f"ERROR: {sys.executable} failed with exit status 2")
        self.assertEqual([line for line in lines if line.startswith("ERROR: ")], lines[-1:])

    def test_usage_and_layout_still_work_through_cli(self):
        usage = "Usage: spirv_metal_set0.py in.spv out.spv"
        for args, code, expected in ((["--print-layout"], 0, "binding  0:  4096 descriptors"), (["--help"], 0, usage),
                                     ([], 2, usage), (["only.spv"], 2, usage)):
            with self.subTest(args=args):
                out, err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    self.assertEqual(m.cli(["spirv_metal_set0.py", *args]), code)
                self.assertIn(expected, (err if code else out).getvalue())
                self.assertEqual((out if code else err).getvalue(), "")

    def test_cli_reports_the_first_line_the_program_wrote(self):
        failure = subprocess.CalledProcessError(1, ["spirv-dis", "in.spv"], stderr="\n  error: 1: bad\nmore\n")
        err = io.StringIO()
        with mock.patch.object(m.subprocess, "run", side_effect=failure), contextlib.redirect_stderr(err):
            self.assertEqual(m.cli(["spirv_metal_set0.py", "in.spv", "out.spv"]), 1)
        self.assertEqual(err.getvalue(), "ERROR: spirv-dis failed with exit status 1: error: 1: bad\n")

    def test_an_unreadable_file_of_the_tool_itself_is_not_blamed_on_spirv_tools(self):
        absent = os.path.join(self.tmp.name, "vulkan_bindless.h")
        capacities, err = m.capacities, io.StringIO()
        with mock.patch.object(m, "capacities", side_effect=lambda: capacities(absent)), \
                contextlib.redirect_stderr(err):
            self.assertEqual(m.cli(["spirv_metal_set0.py", "in.spv", "out.spv"]), 1)
        self.assertTrue(err.getvalue().startswith(f"ERROR: {absent}: "), err.getvalue())
        self.assertNotIn("SPIRV-Tools", err.getvalue())


if __name__ == "__main__":
    unittest.main()
