"""Tests for scripts/ci_expected_failures.py, which judges the macOS CI job's test runs.

    python -I tests/test_ci_expected_failures.py
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import ci_expected_failures as cef  # noqa: E402


class Judge(unittest.TestCase):
    def test_clean_run_passes(self):
        self.assertEqual(cef.judge([], 0, []), (0, []))

    def test_only_expected_failures_pass_with_a_warning(self):
        code, lines = cef.judge(["ps_ssharp"], 1, ["ps_ssharp"])
        self.assertEqual(code, 0)
        self.assertEqual(lines, ["::warning::ps_ssharp failed, as expected until its fix is merged"])

    def test_unexpected_failure_fails(self):
        code, lines = cef.judge(["ps_ssharp", "cs_lds"], 1, ["ps_ssharp"])
        self.assertEqual(code, 1)
        self.assertIn("::error::cs_lds failed", lines)

    def test_runner_failure_without_names_fails(self):
        code, lines = cef.judge([], 2, ["ps_ssharp"])
        self.assertEqual(code, 1)
        self.assertIn("::error::the test runner exited with 2 without naming a failed test", lines)

    def test_interrupted_run_cannot_pass_on_a_partial_failure_list(self):
        for status in (-15, 143):
            with self.subTest(status=status):
                code, lines = cef.judge(["known"], status, ["known", "not_run"])
                self.assertEqual(code, 1)
                self.assertTrue(any(str(status) in line and line.startswith("::error::") for line in lines))
                self.assertFalse(any("not_run passed" in line for line in lines))

    def test_expected_failure_that_passes_gets_a_notice(self):
        code, lines = cef.judge([], 0, ["xdxr_build_check"])
        self.assertEqual(code, 0)
        self.assertEqual(lines, ["::notice::xdxr_build_check passed: drop it from the expected failures"])


class Main(unittest.TestCase):
    def run_main(self, failed: str, *args: str) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.txt"
            path.write_text(failed, encoding="utf-8")
            # Swallowed: a "::error::" line in a CI log would become a real annotation.
            with contextlib.redirect_stdout(io.StringIO()):
                return cef.main([str(path), *args])

    def test_reads_names_one_per_line_and_ignores_blank_lines(self):
        self.assertEqual(self.run_main("\nxdxr_build_check\n\n", "8", "xdxr_build_check"), 0)
        self.assertEqual(self.run_main("a\nb\n", "8", "a"), 1)

    def test_an_expected_argument_can_hold_several_names(self):
        self.assertEqual(self.run_main("a\nb\n", "8", " a  b\n"), 0)
        self.assertEqual(self.run_main("a\nb\n", "8", ""), 1)

    def test_missing_arguments_is_a_usage_error(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cef.main([]), 2)

    def test_help_prints_the_usage_and_succeeds(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(cef.main(["--help"]), 0)
        self.assertIn("FAILED_LIST EXIT_STATUS", out.getvalue())

    def test_a_list_that_cannot_be_read_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "absent.txt"
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cef.main([str(missing), "8"]), 2)
        self.assertEqual(err.getvalue().count("\n"), 1)
        self.assertIn(str(missing), err.getvalue())

    def test_a_non_integer_status_is_one_error_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "failed.txt"
            path.write_text("a\n", encoding="utf-8")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(cef.main([str(path), "x"]), 2)
        self.assertEqual(err.getvalue().count("\n"), 1)
        self.assertIn("'x'", err.getvalue())


if __name__ == "__main__":
    unittest.main()
