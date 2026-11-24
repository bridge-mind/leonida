"""Unit tests for make_game.py's packaged-container message: a store .xvc (an XVD file) given as --game, found by
--find, or sitting alone in the folder given, is named as an encrypted container instead of "not found".

The headers are synthetic: 4 KiB with only the fields make_game reads. No game file is needed.

    python -I tests/test_make_game_container.py
"""
import io
import os
import struct
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game


def xvd_bytes(*, flags=0x41, content_type=1, xvc_data_length=0x1000, version=(1, 2, 0, 1), magic=b"msft-xvd",
              size=0x1000):
    """A synthetic XVD header: zero signature, then the fields make_game reads at their file offsets."""
    b = bytearray(max(size, 0x1000))
    b[0x200:0x208] = magic
    struct.pack_into("<I", b, 0x208, flags)
    struct.pack_into("<I", b, 0x20C, 3)                  # FormatVersion
    struct.pack_into("<I", b, 0x284, content_type)
    struct.pack_into("<I", b, 0x290, xvc_data_length)
    struct.pack_into("<4H", b, 0x3BC, *version)
    return bytes(b[:size])


class XvdHeaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name, data):
        p = os.path.join(self.tmp.name, name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def test_reads_kind_version_and_encryption(self):
        h = make_game.xvd_header(self.write("game.xvc", xvd_bytes()))
        self.assertEqual(h, {"kind": "XVC", "encrypted": True, "content_type": 1, "version": "1.0.2.1"})

    def test_no_xvc_data_is_a_plain_xvd(self):
        h = make_game.xvd_header(self.write("os.xvd", xvd_bytes(xvc_data_length=0, content_type=3)))
        self.assertEqual(h["kind"], "XVD")
        self.assertEqual(h["content_type"], 3)

    def test_encryption_disabled_flag(self):
        h = make_game.xvd_header(self.write("open.xvc", xvd_bytes(flags=0x41 | 0x2)))
        self.assertFalse(h["encrypted"])
        self.assertIn("packed inside it", make_game.container_message("open.xvc", h))

    def test_not_an_xvd(self):
        self.assertIsNone(make_game.xvd_header(self.write("x.xvc", xvd_bytes(magic=b"notanxvd"))))
        self.assertIsNone(make_game.xvd_header(self.write("short.xvc", xvd_bytes(size=0x800))))
        self.assertIsNone(make_game.xvd_header(os.path.join(self.tmp.name, "missing.xvc")))
        self.assertIsNone(make_game.xvd_header(self.tmp.name))      # a folder

    def test_reads_only_the_header(self):
        p = self.write("big.xvc", xvd_bytes() + b"\xff" * 0x10000)
        real_open = open
        reads = []

        class Tracking:
            def __init__(self, f):
                self.f = f

            def read(self, n=-1):
                reads.append(n)
                return self.f.read(n)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.f.close()

        with patch("builtins.open", lambda path, mode="r", *a, **k: Tracking(real_open(path, mode, *a, **k))):
            make_game.xvd_header(p)
        self.assertEqual(reads, [0x1000])

    def test_containers_in_reads_only_container_suffixes(self):
        self.write("a.xvc", xvd_bytes())
        self.write("b.msixvc", xvd_bytes(version=(2, 3, 0, 1)))
        self.write("c.XVD", xvd_bytes(xvc_data_length=0))
        self.write("fake.xvc", b"not an xvd at all")
        self.write("other.bin", xvd_bytes())            # right header, wrong suffix: not looked at
        found = make_game.containers_in(self.tmp.name)
        self.assertEqual([os.path.basename(p) for p, _ in found], ["a.xvc", "b.msixvc", "c.XVD"])
        self.assertEqual(found[1][1]["version"], "1.0.3.2")
        self.assertEqual(make_game.containers_in(os.path.join(self.tmp.name, "missing")), [])


class GameArgumentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.xvc = os.path.join(self.tmp.name, "game.xvc")
        with open(self.xvc, "wb") as f:
            f.write(xvd_bytes())

    def failure(self, fn, *args):
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            fn(*args)
        return err.getvalue()

    def test_check_on_a_container_file_names_it(self):
        msg = self.failure(make_game.check, self.xvc, False)
        self.assertIn("packaged XVC container (package version 1.0.2.1)", msg)
        self.assertIn("encrypted", msg)
        self.assertIn("MicrosoftGame.Config", msg)
        self.assertNotIn("Game folder not found", msg)

    def test_check_on_another_file_is_still_not_found(self):
        other = os.path.join(self.tmp.name, "notes.txt")
        with open(other, "w", encoding="utf-8") as f:
            f.write("x")
        self.assertIn("Game folder not found", self.failure(make_game.check, other, False))

    def test_folder_with_only_a_container_mentions_it(self):
        msg = self.failure(make_game.check, self.tmp.name, False)
        self.assertIn("doesn't look like a supported version", msg)
        self.assertIn(f"{self.xvc} is a packaged XVC container", msg)

    def test_unsupported_folder_without_container_is_unchanged(self):
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        msg = self.failure(make_game.check, empty, False)
        self.assertIn("doesn't look like a supported version", msg)
        self.assertNotIn("container", msg)


class FindTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.downloads = os.path.join(self.tmp.name, "Downloads")
        os.makedirs(os.path.join(self.downloads, "deeper"))
        self.xvc = os.path.join(self.downloads, "game.xvc")
        with open(self.xvc, "wb") as f:
            f.write(xvd_bytes())
        with open(os.path.join(self.downloads, "deeper", "notes.xvc"), "wb") as f:
            f.write(b"text, not a container")

    def find(self, containers):
        with patch("os.path.expanduser", return_value=self.tmp.name), \
                patch("make_game.known_folders", return_value=[self.downloads]), \
                patch("os.name", "posix"), patch("os.path.isdir", side_effect=lambda p: p.startswith(self.tmp.name)
                                                   and os.path.exists(p) and not os.path.isfile(p)):
            return make_game.find_dumps(containers)

    def test_find_dumps_collects_containers_once(self):
        containers = []
        self.assertEqual(self.find(containers), [])
        self.assertEqual([p for p, _ in containers], [self.xvc])     # reached from two roots, listed once
        self.assertEqual(containers[0][1]["version"], "1.0.2.1")

    def test_find_dumps_without_a_list_returns_as_before(self):
        self.assertEqual(self.find(None), [])

    def test_find_flag_lists_containers_as_not_usable(self):
        def fake_find(containers=None):
            containers.append((self.xvc, make_game.xvd_header(self.xvc)))
            return []
        out = io.StringIO()
        with patch("make_game.find_dumps", side_effect=fake_find), patch("make_game.below_normal"), \
                patch.object(sys, "argv", ["make_game.py", "--find"]), redirect_stdout(out):
            self.assertEqual(make_game.main(), 0)
        text = out.getvalue()
        self.assertIn("0 found", text)
        self.assertIn(f"not usable: {self.xvc}  (packaged XVC container, package version 1.0.2.1", text)

    def test_not_found_names_the_container(self):
        def fake_find(containers=None):
            containers.append((self.xvc, make_game.xvd_header(self.xvc)))
            return []
        err = io.StringIO()
        with patch("make_game.find_dumps", side_effect=fake_find), patch("make_game.log"), \
                patch.object(sys, "stdin", None), redirect_stderr(err), self.assertRaises(SystemExit):
            make_game.choose_game()
        self.assertIn("weren't found", err.getvalue())
        self.assertIn(f"{self.xvc} is a packaged XVC container", err.getvalue())


if __name__ == "__main__":
    unittest.main()
