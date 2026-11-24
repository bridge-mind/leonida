"""shader_diff.py exit codes: a harness error is 2, not a mismatch (no GPU, no game data)."""
import contextlib
import io
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import shader_diff as sd  # noqa: E402
from spirv_mini_interp import UnsupportedConstruct  # noqa: E402


class ReplayIndex(unittest.TestCase):
    def test_replays_exact_case_without_running_predecessors(self):
        # Compare actual generated instructions, their last writers and the case name.
        args = ["--seed", "7", "--max-ops", "4"]
        with mock.patch.object(sd, "run_one", wraps=sd.run_one) as execute, mock.patch("sys.stdout"):
            self.assertEqual(sd.main([*args, "--programs", "4"]), 0)
            full_run = execute.call_args_list.copy()
            for index in (0, 3):
                execute.reset_mock()
                self.assertEqual(sd.main([*args, "--program-index", str(index)]), 0)
                self.assertEqual(execute.call_args_list, [full_run[index]])

    def test_rejects_invalid_or_ambiguous_selection(self):
        for args in (["--program-index", "-1"], ["--program-index", "2", "--limit", "1"],
                     ["--program-index", "2", "--programs", "3"]):
            with self.subTest(args=args), mock.patch("sys.stderr"), self.assertRaises(SystemExit) as caught:
                sd.main(args)
            self.assertEqual(caught.exception.code, 2)


class HarnessErrorExit(unittest.TestCase):
    def test_harness_error_alone_exits_2(self):
        def boom(*_a):
            raise RuntimeError("broken harness")
        with mock.patch.object(sd, "run_one", boom), mock.patch("sys.stdout"):
            self.assertEqual(sd.main(["--programs", "2"]), 2)

    def test_unexpected_translator_failures_exit_2(self):
        for error_type in (IndexError, KeyError, ValueError):
            with self.subTest(error=error_type.__name__):
                out = io.StringIO()
                with (
                    mock.patch.object(sd.rdna2spirv, "translate", side_effect=error_type("broken translator")),
                    contextlib.redirect_stdout(out),
                ):
                    self.assertEqual(sd.main(["--programs", "1"]), 2)
                self.assertIn("0 skipped, 1 harness errors", out.getvalue())
                self.assertIn(f"{error_type.__name__}:", out.getvalue())
                self.assertIn("broken translator", out.getvalue())

    def test_unsupported_translation_is_a_skip(self):
        out = io.StringIO()
        with (
            mock.patch.object(sd.rdna2spirv, "translate", side_effect=sd.rdna2spirv.Unsupported("test opcode")),
            contextlib.redirect_stdout(out),
        ):
            # exit 2: a run where every program was skipped is not a success (#4448)
            self.assertEqual(sd.main(["--programs", "1"]), 2)
        self.assertIn("1 skipped, 0 harness errors", out.getvalue())
        self.assertIn("translator unsupported: test opcode", out.getvalue())

    def test_unexpected_interpreter_failure_exits_2(self):
        out = io.StringIO()
        with mock.patch.object(sd, "run_fragment", side_effect=RuntimeError("broken interpreter")):
            with contextlib.redirect_stdout(out):
                self.assertEqual(sd.main(["--programs", "1"]), 2)
        self.assertIn("0 skipped, 1 harness errors", out.getvalue())
        self.assertIn("RuntimeError: broken interpreter", out.getvalue())

    def test_unsupported_interpreter_construct_is_a_skip(self):
        out = io.StringIO()
        with mock.patch.object(sd, "run_fragment", side_effect=UnsupportedConstruct("test construct")):
            with contextlib.redirect_stdout(out):
                sd.main(["--programs", "1"])
        self.assertIn("1 skipped, 0 harness errors", out.getvalue())
        self.assertIn("test construct", out.getvalue())

    def test_a_run_where_every_program_was_skipped_is_not_a_success(self):
        out = io.StringIO()
        with mock.patch.object(sd, "run_fragment", side_effect=UnsupportedConstruct("test construct")):
            with contextlib.redirect_stdout(out):
                self.assertEqual(sd.main(["--programs", "2"]), 2)
        self.assertIn("no program ran: every one was skipped", out.getvalue())

    def test_some_skips_next_to_passes_still_exit_0(self):
        out = io.StringIO()
        calls = iter([("match", ""), ("skip", "unsupported form")])
        with mock.patch.object(sd, "run_one", lambda *_a: next(calls)):
            with contextlib.redirect_stdout(out):
                self.assertEqual(sd.main(["--programs", "2"]), 0)
        self.assertIn("1 pass, 0 mismatch, 1 skipped", out.getvalue())

    def test_mismatch_still_exits_1(self):
        with mock.patch.object(sd, "run_one", lambda *_a: ("mismatch", "lane word 0")), mock.patch("sys.stdout"):
            self.assertEqual(sd.main(["--programs", "1"]), 1)


class ArgumentValidation(unittest.TestCase):
    """--programs, --limit and --max-ops must be at least 1.

    Before this, `--limit 0` and `--limit -1` ran no program and printed
    "0 programs (seed 1): 0 pass, 0 mismatch ..." with exit 0, which reads as a
    pass, and `--max-ops 0` died with `ValueError: empty range in randrange(1, 1)`.
    """

    def _rejected(self, argv, option):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as caught:
                sd.main(argv)
        self.assertEqual(caught.exception.code, 2)
        self.assertIn(option, err.getvalue())
        self.assertIn("at least 1", err.getvalue())

    def test_limit_zero_is_rejected(self):
        self._rejected(["--limit", "0"], "--limit")

    def test_limit_negative_is_rejected(self):
        self._rejected(["--limit", "-5"], "--limit")

    def test_programs_zero_is_rejected(self):
        self._rejected(["--programs", "0"], "--programs")

    def test_programs_negative_is_rejected(self):
        self._rejected(["--programs", "-1"], "--programs")

    def test_max_ops_zero_is_rejected(self):
        self._rejected(["--max-ops", "0"], "--max-ops")

    def test_max_ops_negative_is_rejected(self):
        self._rejected(["--max-ops", "-2"], "--max-ops")

    def test_a_valid_limit_still_runs_that_many_programs(self):
        out = io.StringIO()
        with mock.patch.object(sd, "run_one", lambda *_a: ("match", "")), contextlib.redirect_stdout(out):
            self.assertEqual(sd.main(["--programs", "5", "--limit", "1"]), 0)
        self.assertIn("1 programs", out.getvalue())

    def test_the_defaults_are_unchanged(self):
        a = sd.main(["--programs", "1", "--limit", "1"])
        self.assertIn(a, (0, 1))  # 1 pass or 1 mismatch; never an argparse exit


if __name__ == "__main__":
    unittest.main()
