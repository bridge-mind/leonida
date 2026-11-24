"""tools/file_mode.py and its callers never change the process umask; no game data.

    python -I tests/test_file_mode.py -v
"""
import contextlib
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import file_mode  # noqa: E402
import awc_manifest  # noqa: E402
import classify_carve  # noqa: E402
import glb_writer  # noqa: E402
import rbf0_tree  # noqa: E402
import rsc8_audio  # noqa: E402
import rtti_scan  # noqa: E402

PRIMS = glb_writer.parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", is_path=False)


def no_umask(*_):
    raise AssertionError("os.umask was called: it sets the mask for every thread in the process")


class NoUmaskFlipTests(unittest.TestCase):
    # The earlier code read the mask with os.umask(0) then os.umask(mask); a thread creating a file between
    # the two calls got mask 0, a world-writable file (#750 and #728 review by kapitche).

    def writers(self, directory):
        return [
            ("write_glb", Path(directory) / "mesh.glb", lambda p: glb_writer.write_glb(p, PRIMS)),
            ("write_report", os.path.join(directory, "x.rtti.json"), lambda p: rtti_scan.write_report(p, "{}")),
            ("awc_manifest", os.path.join(directory, "audio.csv"),
             lambda p: awc_manifest.write_manifest([], 0, 0, "csv", p)),
            ("classify_carve", os.path.join(directory, "types.json"), self.classify_carve),
            ("rbf0_tree", os.path.join(directory, "tree.json"), lambda p: rbf0_tree.write_json(p, [], {})),
            ("rsc8_audio", os.path.join(directory, "rsc8.json"), lambda p: rsc8_audio.write_report(p, [], {})),
        ]

    @staticmethod
    def classify_carve(path):
        with mock.patch.object(sys, "argv", ["classify_carve.py", os.path.dirname(path)]), \
                contextlib.redirect_stdout(io.StringIO()):
            assert classify_carve.main() == 0

    def test_writes_never_call_umask(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("os.umask", side_effect=no_umask):
            for name, path, write in self.writers(directory):
                with self.subTest(name):
                    write(path)  # a new file: the case that used to read the umask
                    write(path)  # a replaced file
                    self.assertTrue(os.path.isfile(path))
            self.assertEqual(sorted(os.listdir(directory)), sorted(os.path.basename(p) for _, p, _ in
                                                                   self.writers(directory)))

    def test_new_file_mode_leaves_no_probe_behind(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch("os.umask", side_effect=no_umask):
            mode = file_mode.new_file_mode(directory)
            self.assertTrue(mode & 0o600 == 0o600, oct(mode))
            self.assertEqual(os.listdir(directory), [])

    def test_probe_name_collision_is_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            taken = os.path.join(directory, ".mode-aaaa.probe")
            Path(taken).write_bytes(b"someone else's")
            with mock.patch.object(file_mode.secrets, "token_hex", side_effect=["aaaa", "bbbb"]):
                file_mode.new_file_mode(directory)
            self.assertEqual(os.listdir(directory), [".mode-aaaa.probe"])
            self.assertEqual(Path(taken).read_bytes(), b"someone else's")

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_new_file_mode_follows_the_current_umask(self):
        with tempfile.TemporaryDirectory() as directory:
            old = os.umask(0o027)
            try:
                self.assertEqual(file_mode.new_file_mode(directory), 0o640)
                os.umask(0o002)
                self.assertEqual(file_mode.new_file_mode(directory), 0o664)
                for name, path, write in self.writers(directory):
                    with self.subTest(name), mock.patch("os.umask", side_effect=no_umask):
                        write(path)
                    self.assertEqual(os.stat(path).st_mode & 0o777, 0o664)
            finally:
                os.umask(old)

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_published_mode_keeps_the_replaced_files_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "kept")
            Path(path).write_bytes(b"")
            os.chmod(path, 0o604)
            self.assertEqual(file_mode.published_mode(path), 0o604)


if __name__ == "__main__":
    unittest.main()
