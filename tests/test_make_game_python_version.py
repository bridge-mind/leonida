"""make_game.py stops with a clear message on a Python older than 3.12, before code that version can't run.

    python -I tests/test_make_game_python_version.py
"""
import os
import subprocess
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKE_GAME = os.path.join(ROOT, "make_game.py")
RUN_AS = ("import runpy, sys; sys.version_info = {!r}; sys.argv = [{!r}, '--help']; "
          "runpy.run_path({!r}, run_name='__main__')")


def run_as(version: tuple) -> subprocess.CompletedProcess:
    code = RUN_AS.format(version, MAKE_GAME, MAKE_GAME)
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)


class PythonVersionGuardTest(unittest.TestCase):
    def test_older_python_gets_the_requirement_not_a_traceback(self):
        for version in ((3, 9, 25), (3, 11, 9)):
            with self.subTest(version=version):
                r = run_as(version)
                self.assertEqual(r.returncode, 1, r.stderr)
                self.assertEqual(r.stdout, "")
                self.assertIn("ERROR: Python 3.12 or newer is needed", r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_supported_python_runs(self):
        r = run_as((3, 12, 0))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("usage:", r.stdout)

    def test_import_is_not_blocked(self):
        code = f"import sys; sys.version_info = (3, 11, 9); sys.path.insert(0, {ROOT!r}); import make_game"
        r = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
