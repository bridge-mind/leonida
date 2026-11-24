"""Exercise the Unicode console paths CMake check with a synthetic loader."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECK = Path(__file__).resolve().parents[1] / "runtime/test/check_console_paths.cmake"
CMAKE = shutil.which("cmake")


@unittest.skipUnless(CMAKE, "CMake is required to exercise the console paths check")
class ConsolePathsTests(unittest.TestCase):
    def test_requires_the_guest_to_finish_successfully(self):
        cases = (
            ("successful guest", "tiny_game: OK\ngame exited with code 0\n", 0),
            ("paths only", "", 1),
            ("guest failure", "tiny_game: FAILED\ngame exited with code 1\n", 1),
            ("guest OK without loader exit", "tiny_game: OK\n", 1),
            ("loader exit without guest OK", "game exited with code 0\n", 1),
        )
        with tempfile.TemporaryDirectory() as tmp:
            loader = Path(tmp) / "loader.py"
            for name, verdict, expected in cases:
                with self.subTest(name=name):
                    output = "console paths: GAME: -> game-Игра, writes -> overlay-保存\n" + verdict
                    loader.write_text(
                        f"import sys\nsys.stdout.buffer.write({output.encode('utf-8')!r})\n",
                        encoding="utf-8", newline="\n",
                    )
                    result = subprocess.run(
                        [CMAKE, f"-DRUNNER={sys.executable}", f"-DLOADER={loader}",
                         "-DGUEST=synthetic.exe", "-P", str(CHECK)],
                        cwd=tmp, capture_output=True, encoding="utf-8", errors="replace",
                        timeout=10, check=False,
                    )
                    self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
