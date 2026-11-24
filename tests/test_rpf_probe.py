"""Tests for tools/rpf_probe.py, built from synthetic archive headers (no game data).

    python -I tests/test_rpf_probe.py
"""
import contextlib
import errno
import importlib.util
import io
import json
import os
import struct
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_module():
    path = os.path.join(ROOT, "tools", "rpf_probe.py")
    spec = importlib.util.spec_from_file_location("rpf_probe_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = load_module()


def archive_bytes(version=b"4FPR", words=(0x10, 0x20, 0x30, 0x40), pad_to=None, tail=b""):
    """Little-endian 'RPF4' magic, four header words, then optional padding and payload."""
    data = version + struct.pack("<4I", *words)
    if pad_to is not None and len(data) < pad_to:
        data += b"\x00" * (pad_to - len(data))
    return data + tail


class EntropyTests(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(probe.entropy(b""), 0.0)

    def test_single_symbol_is_zero(self):
        self.assertEqual(probe.entropy(b"\x00" * 16), 0.0)

    def test_two_equally_frequent_symbols(self):
        self.assertEqual(probe.entropy(b"\x00\x01" * 8), 1.0)

    def test_all_byte_values_once(self):
        self.assertEqual(probe.entropy(bytes(range(256))), 8.0)


class ProbeTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = tmp.name

    def write(self, data, name="a.rpf"):
        path = os.path.join(self.dir, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_version_and_header_words(self):
        row = probe.probe(self.write(archive_bytes(words=(0x0, 0xFFFFFFFF, 0x1234, 0x10))))
        self.assertEqual(row["version"], "RPF4")
        self.assertEqual(row["header_words"], ["0x0", "0xffffffff", "0x1234", "0x10"])

    def test_natural_order_magic_is_not_a_version(self):
        self.assertEqual(probe.probe(self.write(b"RPF4" + b"\x00" * 16))["version"], None)

    def test_other_magic_has_no_version(self):
        self.assertEqual(probe.probe(self.write(b"DDS " + b"\x00" * 30))["version"], None)

    def test_short_file_has_no_words(self):
        row = probe.probe(self.write(b"4FPR" + b"\x00" * 4))
        self.assertEqual(row["version"], "RPF4")
        self.assertEqual(row["header_words"], [])
        self.assertEqual(row["head_entropy"], 0.0)
        self.assertFalse(row["looks_encrypted"])

    def test_zero_filled_body_is_not_flagged(self):
        row = probe.probe(self.write(archive_bytes(words=(0, 0, 0, 0), pad_to=1024)))
        self.assertEqual(row["head_entropy"], 0.0)
        self.assertFalse(row["looks_encrypted"])

    def test_dense_body_is_flagged(self):
        row = probe.probe(self.write(archive_bytes(tail=bytes(range(256)) * 16)))
        self.assertGreater(row["head_entropy"], 7.9)
        self.assertTrue(row["looks_encrypted"])

    def test_only_the_first_64_kib_is_read(self):
        path = self.write(archive_bytes(pad_to=0x10000, tail=bytes(range(256)) * 16))
        row = probe.probe(path)
        self.assertEqual(row["size"], 0x10000 + 4096)
        self.assertEqual(row["head_entropy"], 0.0)
        self.assertFalse(row["looks_encrypted"])

    def test_size_is_the_full_file(self):
        self.assertEqual(probe.probe(self.write(archive_bytes(pad_to=0x10000)))["size"], 0x10000)


class MainTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.game = os.path.join(self.tmp, "game")
        os.makedirs(self.game)
        self.out = os.path.join(self.tmp, "reports", "rpf.json")

    def write(self, rel, data):
        path = os.path.join(self.game, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def run_main(self):
        out = io.StringIO()
        with mock.patch("sys.argv", ["rpf_probe.py", self.game, "--out", self.out]), contextlib.redirect_stdout(out):
            self.assertEqual(probe.main(), 0)
        return out.getvalue()

    def rows(self):
        with open(self.out, encoding="utf-8") as f:
            return json.load(f)

    def test_collects_rpf_files_recursively_and_sorts_by_size(self):
        small = self.write("a.rpf", archive_bytes())
        large = self.write("sub/b.RPF", archive_bytes(pad_to=4096))
        self.write("sub/notes.txt", b"ignored")
        out = self.run_main()
        rows = self.rows()
        self.assertEqual([os.path.basename(r["file"]) for r in rows], ["b.RPF", "a.rpf"])
        self.assertEqual(os.path.normpath(rows[0]["file"]), os.path.normpath(large))
        self.assertEqual(os.path.normpath(rows[1]["file"]), os.path.normpath(small))
        self.assertIn("2 archives", out)
        self.assertIn("'RPF4': 2", out)

    def test_equal_sized_archives_are_sorted_by_path(self):
        self.write("z.rpf", archive_bytes())
        self.write("a.rpf", archive_bytes())
        with mock.patch.object(probe.os, "walk", return_value=[(self.game, [], ["z.rpf", "a.rpf"])]):
            self.run_main()

        self.assertEqual([os.path.basename(row["file"]) for row in self.rows()], ["a.rpf", "z.rpf"])

    def test_creates_the_output_directory(self):
        self.write("a.rpf", archive_bytes())
        self.run_main()
        self.assertTrue(os.path.isfile(self.out))

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_refresh_preserves_private_report_permissions(self):
        self.write("a.rpf", archive_bytes())
        self.run_main()
        os.chmod(self.out, 0o600)
        self.write("b.rpf", archive_bytes())
        self.run_main()
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual(stat.S_IMODE(os.stat(self.out).st_mode), 0o600)

    def test_empty_game_directory(self):
        out = self.run_main()
        self.assertEqual(self.rows(), [])
        self.assertIn("0 archives", out)

    def assert_input_refused(self, destination, archive):
        with open(archive, "rb") as f:
            before = f.read()
        script = os.path.join(ROOT, "tools", "rpf_probe.py")
        result = subprocess.run(
            [sys.executable, "-I", script, self.game, "--out", destination],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
        self.assertIn("would overwrite an input archive", result.stderr)
        with open(archive, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_report_cannot_replace_an_input_archive(self):
        archive = self.write("a.rpf", archive_bytes())
        self.assert_input_refused(archive, archive)

    def test_report_cannot_replace_a_nested_archive(self):
        archive = self.write("sub/a.RPF", archive_bytes())
        self.assert_input_refused(archive, archive)

    def test_report_cannot_replace_a_normalized_input_path(self):
        archive = self.write("a.rpf", archive_bytes())
        os.mkdir(os.path.join(self.game, "sub"))
        destination = os.path.join(self.game, "sub", "..", "a.rpf")
        self.assert_input_refused(destination, archive)

    def test_report_cannot_replace_a_hard_link_to_an_archive(self):
        archive = self.write("a.rpf", archive_bytes())
        destination = os.path.join(self.tmp, "alias.json")
        try:
            os.link(archive, destination)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"hard links unavailable: {e}")
        self.assert_input_refused(destination, archive)
        self.assertTrue(os.path.samefile(archive, destination))

    def test_report_cannot_replace_a_symlink_to_an_archive(self):
        archive = self.write("a.rpf", archive_bytes())
        destination = os.path.join(self.tmp, "alias.json")
        try:
            os.symlink(archive, destination)
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"symlinks unavailable: {e}")
        self.assert_input_refused(destination, archive)
        self.assertTrue(os.path.islink(destination))

    def test_report_cannot_replace_the_target_of_a_scanned_symlink(self):
        archive = os.path.join(self.tmp, "external.rpf")
        with open(archive, "wb") as f:
            f.write(archive_bytes())
        try:
            os.symlink(archive, os.path.join(self.game, "linked.rpf"))
        except (OSError, NotImplementedError) as e:
            self.skipTest(f"symlinks unavailable: {e}")
        self.assert_input_refused(archive, archive)

    def test_existing_unrelated_report_can_be_replaced(self):
        original = archive_bytes()
        archive = self.write("a.rpf", original)
        os.makedirs(os.path.dirname(self.out))
        with open(self.out, "w", encoding="utf-8") as f:
            f.write("previous report\n")
        self.run_main()
        self.assertEqual([r["file"] for r in self.rows()], [archive])
        with open(archive, "rb") as f:
            self.assertEqual(f.read(), original)

    @unittest.skipUnless(os.name == "posix", "requires a POSIX file size limit")
    def test_failed_write_preserves_the_existing_report(self):
        self.write("demo.rpf", b"9FPR" + struct.pack("<I", 16) + bytes(32))
        previous = b'{"previous": true}'
        out_dir = os.path.dirname(self.out)
        os.makedirs(out_dir)
        with open(self.out, "wb") as f:
            f.write(previous)

        limited_probe = """\
import resource
import runpy
import signal
import sys

signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
resource.setrlimit(resource.RLIMIT_FSIZE, (32, 32))
runpy.run_path(sys.argv.pop(1), run_name="__main__")
"""
        script = os.path.join(ROOT, "tools", "rpf_probe.py")
        result = subprocess.run(
            [sys.executable, "-I", "-c", limited_probe, script, self.game, "--out", self.out],
            capture_output=True, text=True,
        )

        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), previous)
        self.assertEqual(os.listdir(out_dir), [os.path.basename(self.out)])
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(len(result.stderr.splitlines()), 1, result.stderr)
        self.assertIn(self.out, result.stderr)

    def test_failed_scan_preserves_the_existing_report(self):
        archive = self.write("a.rpf", archive_bytes())
        self.run_main()
        before = self.rows()
        missing = os.path.join(self.tmp, "missing")
        for source in (missing, archive):
            with self.subTest(source=source):
                err = io.StringIO()
                argv = ["rpf_probe.py", source, "--out", self.out]
                with mock.patch("sys.argv", argv), contextlib.redirect_stderr(err):
                    self.assertEqual(probe.main(), 2)
                self.assertIn(source, err.getvalue())
                self.assertEqual(self.rows(), before)

    def test_unreadable_subdirectory_preserves_the_existing_report(self):
        self.write("a.rpf", archive_bytes())
        blocked = os.path.dirname(self.write("blocked/b.rpf", archive_bytes()))
        self.run_main()
        before = self.rows()
        scandir = os.scandir

        def deny_subdirectory(path):
            if path == blocked:
                raise PermissionError(errno.EACCES, "Permission denied", path)
            return scandir(path)

        err = io.StringIO()
        argv = ["rpf_probe.py", self.game, "--out", self.out]
        with (mock.patch("sys.argv", argv), mock.patch.object(os, "scandir", side_effect=deny_subdirectory),
              contextlib.redirect_stderr(err)):
            self.assertEqual(probe.main(), 2)
        self.assertIn(blocked, err.getvalue())
        self.assertEqual(self.rows(), before)


if __name__ == "__main__":
    unittest.main()
