"""make_game.py: end of input at a prompt and Ctrl+C give one line, not a traceback (no game files needed).

    python -I tests/test_make_game_interrupt.py
"""
import contextlib
import io
import os
import sys
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game  # noqa: E402

COPY = {"title": "Xbox Series package", "version": "1.0.2.1"}


class PromptWithoutInput(unittest.TestCase):
    def choose(self, found):
        err = io.StringIO()
        with (patch("make_game.find_dumps", return_value=found), patch("make_game.log"),
              patch.object(sys, "stdin", **{"isatty.return_value": True}),
              patch("builtins.input", side_effect=EOFError), contextlib.redirect_stdout(io.StringIO()),
              contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as stop):
            make_game.choose_game()
        return stop.exception.code, err.getvalue()

    def test_game_folder_prompt(self):
        code, err = self.choose([])
        self.assertEqual(code, 1)
        self.assertIn("Give the folder with --game", err)

    def test_which_copy_prompt(self):
        code, err = self.choose([("D:/a", COPY), ("D:/b", COPY)])
        self.assertEqual(code, 1)
        self.assertIn("choose one with --game", err)


class CtrlC(unittest.TestCase):
    def test_one_line_and_exit_130(self):
        err = io.StringIO()
        with patch("make_game.main", side_effect=KeyboardInterrupt), contextlib.redirect_stderr(err):
            self.assertEqual(make_game.cli(), 130)
        self.assertEqual(err.getvalue().strip(), "Stopped. Run it again to continue where it left off.")

    def test_other_exit_codes_pass_through(self):
        with patch("make_game.main", return_value=0):
            self.assertEqual(make_game.cli(), 0)


if __name__ == "__main__":
    unittest.main()
