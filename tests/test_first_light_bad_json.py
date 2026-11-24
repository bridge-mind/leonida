"""first_light.py on a broken manifest.json (--check) or carve index.json (--list): one line naming the file.

    python -I tests/test_first_light_bad_json.py

Fully synthetic: the folders and their JSON files are written by hand in a temporary directory, so this needs no
package, no built harness and no LEONIDA_FIRST_LIGHT_DIR.
"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CliBadJsonTests(unittest.TestCase):
    """--check and --list on a broken user JSON: one clean line naming the file, never a traceback.

    Fully synthetic: the folders and their manifest.json/index.json are written by hand here, so
    these run without a package, a built harness or LEONIDA_FIRST_LIGHT_DIR.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="first-light-bad-json-")
        self.addCleanup(tmp.cleanup)
        self.out = os.path.join(tmp.name, "out")
        self.work = os.path.join(tmp.name, "work")
        self.game = os.path.join(tmp.name, "game")
        os.makedirs(self.out)

    def write(self, path, content):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb" if isinstance(content, bytes) else "w") as f:
            f.write(content)
        return path

    def bad_manifest(self, content):
        return self.write(os.path.join(self.out, "manifest.json"), content)

    def bad_index(self, content):
        return self.write(os.path.join(self.work, "shaders", "dxbc", "index.json"), content)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "first_light.py"), *args],
            capture_output=True,
            text=True,
        )

    def assert_one_clean_line(self, proc, path):
        self.assertNotEqual(proc.returncode, 0, f"{path} should fail the run")
        self.assertIn(path, proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(len([line for line in proc.stderr.splitlines() if line.strip()]), 1)

    def check_cli(self):
        return self.run_cli("--check", "--out", self.out, "--work", self.work)

    def list_cli(self):
        return self.run_cli("--list", "--game", self.game, "--work", self.work)

    def test_check_reports_a_truncated_manifest(self):
        path = self.bad_manifest('{"vs": ')
        self.assert_one_clean_line(self.check_cli(), path)

    def test_check_reports_a_non_utf8_manifest(self):
        path = self.bad_manifest(b'{"width": 4, "height": 4, "x": \xff}')
        self.assert_one_clean_line(self.check_cli(), path)

    def test_check_reports_a_manifest_of_the_wrong_shape(self):
        path = self.bad_manifest("[1, 2]")  # an array where run_draw indexes by name
        self.assert_one_clean_line(self.check_cli(), path)

    def test_check_reports_a_manifest_without_its_required_keys(self):
        path = self.bad_manifest('{"foo": 1}')  # run_draw reads width/height first
        self.assert_one_clean_line(self.check_cli(), path)

    def test_list_reports_a_truncated_index(self):
        path = self.bad_index('[{"name": "x", ')
        self.assert_one_clean_line(self.list_cli(), path)

    def test_list_reports_a_non_utf8_index(self):
        path = self.bad_index(b'[{"name": \xff}]')
        self.assert_one_clean_line(self.list_cli(), path)

    def test_list_reports_an_index_of_the_wrong_shape(self):
        path = self.bad_index('{"a": 1}')  # iterating the dict hands e["stage"] a str
        self.assert_one_clean_line(self.list_cli(), path)

    def test_list_reports_an_index_entry_without_its_required_keys(self):
        path = self.bad_index('[{"stage": "vs", "name": "x"}, {"stage": "ps"}]')
        self.assert_one_clean_line(self.list_cli(), path)

    def test_list_prints_the_candidates_from_a_hand_written_index(self):
        self.bad_index(
            '[{"stage": "vs", "name": "a"}, {"stage": "ps", "name": "b"}, {"stage": "cs", "name": "c"}]'
        )
        proc = self.list_cli()
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(proc.stdout.splitlines(), ["vs a", "ps b"])


if __name__ == "__main__":
    unittest.main()
