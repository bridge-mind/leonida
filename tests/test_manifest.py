"""Tests for tools/manifest.py covering directory hashing, manifest creation, verification, and corruption detection.

    python -I tests/test_manifest.py
"""
import contextlib
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unicodedata
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "manifest.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))
import manifest  # noqa: E402
import report_paths  # noqa: E402


class ManifestSha256Test(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def test_sha256_computes_expected_hash(self):
        payload = b"synthetic manifest test payload\nwith multiple lines and deterministic data."
        path = os.path.join(self.tmp.name, "sample.bin")
        with open(path, "wb") as f:
            f.write(payload)

        expected = hashlib.sha256(payload).hexdigest()
        self.assertEqual(manifest.sha256(path), expected)

    def test_sha256_empty_file(self):
        path = os.path.join(self.tmp.name, "empty.bin")
        with open(path, "wb") as f:
            pass

        expected = hashlib.sha256(b"").hexdigest()
        self.assertEqual(manifest.sha256(path), expected)


class ManifestWalkTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "pkg")
        os.makedirs(os.path.join(self.pkg, "sub", "deep"))

        with open(os.path.join(self.pkg, "root.dat"), "wb") as f:
            f.write(b"root file")
        with open(os.path.join(self.pkg, "sub", "mid.dat"), "wb") as f:
            f.write(b"mid level file")
        with open(os.path.join(self.pkg, "sub", "deep", "leaf.dat"), "wb") as f:
            f.write(b"leaf file")

    def test_walk_normalizes_to_forward_slashes_and_yields_all_files(self):
        entries = list(manifest.walk(self.pkg))
        rel_paths = [rel for rel, _ in entries]

        self.assertIn("root.dat", rel_paths)
        self.assertIn("sub/mid.dat", rel_paths)
        self.assertIn("sub/deep/leaf.dat", rel_paths)
        for rel in rel_paths:
            self.assertNotIn("\\", rel)

        for rel, abs_path in entries:
            self.assertTrue(os.path.isabs(abs_path))
            self.assertTrue(os.path.exists(abs_path))

    def test_walk_reports_an_unreadable_subdirectory(self):
        blocked = os.path.join(self.pkg, "sub", "deep")
        scandir = os.scandir

        def scan(path):
            if path == blocked:
                raise PermissionError(f"cannot read {path}")
            return scandir(path)

        with patch.object(os, "scandir", side_effect=scan):
            with self.assertRaises(PermissionError):
                list(manifest.walk(self.pkg))


class ManifestCreateVerifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "game_pkg")
        os.makedirs(os.path.join(self.pkg, "content", "data"))

        self.files = {
            "alpha.bin": b"alpha payload bytes " * 100,
            "bravo.bin": b"bravo payload bytes " * 50,
            "content/asset1.dat": b"asset 1 synthetic binary payload " * 20,
            "content/data/table.bin": b"table data entries and records " * 10,
        }
        for rel, data in self.files.items():
            full = os.path.join(self.pkg, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(full), exist_ok=True)
            with open(full, "wb") as f:
                f.write(data)

        self.out_manifest = os.path.join(self.tmp.name, "reports", "test_manifest.json")

    def test_create_manifest_format_and_hashes(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.create(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(self.out_manifest))

        with open(self.out_manifest, encoding="utf-8") as f:
            doc = json.load(f)

        self.assertIn("created", doc)
        self.assertIn("files", doc)
        entries = doc["files"]
        self.assertEqual(len(entries), len(self.files))

        for rel, data in self.files.items():
            self.assertIn(rel, entries)
            self.assertEqual(entries[rel]["size"], len(data))
            self.assertEqual(entries[rel]["sha256"], hashlib.sha256(data).hexdigest())

    def test_create_is_identical_for_serial_and_parallel(self):
        out_serial = os.path.join(self.tmp.name, "serial.json")
        out_parallel = os.path.join(self.tmp.name, "parallel.json")

        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, out_serial, jobs=1)
            manifest.create(self.pkg, out_parallel, jobs=4)

        with open(out_serial, encoding="utf-8") as f:
            serial_doc = json.load(f)
        with open(out_parallel, encoding="utf-8") as f:
            parallel_doc = json.load(f)

        self.assertEqual(serial_doc["files"], parallel_doc["files"])
        self.assertEqual(json.dumps(serial_doc["files"]), json.dumps(parallel_doc["files"]))

    def test_create_cannot_overwrite_a_file_it_hashes(self):
        target = os.path.join(self.pkg, "alpha.bin")
        with open(target, "rb") as f:
            before = f.read()
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(manifest.ManifestError):
                manifest.create(self.pkg, target, jobs=1)
        with open(target, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_create_cannot_overwrite_a_hard_link_to_a_hashed_file(self):
        alias = os.path.join(self.tmp.name, "alias.json")
        try:
            os.link(os.path.join(self.pkg, "alpha.bin"), alias)
        except OSError:
            self.skipTest("hard links are not available")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(manifest.ManifestError):
            manifest.create(self.pkg, alias, jobs=1)

    def test_create_can_still_write_a_new_manifest_inside_the_package(self):
        out = os.path.join(self.pkg, "manifest.json")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(manifest.create(self.pkg, out, jobs=1), 0)
        self.assertTrue(os.path.isfile(out))

    def test_create_rejects_a_folder_as_out(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(manifest.ManifestError):
            manifest.create(self.pkg, self.pkg, jobs=1)

    def test_verify_intact_directory_returns_zero(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 0)
        self.assertIn("OK", out.getvalue())

    def test_verify_rejects_resize_between_size_check_and_hashing(self) -> None:
        rel = "alpha.bin"
        path = os.path.join(self.pkg, rel)
        original = self.files[rel]
        grown = original + b" appended after the size check"
        os.makedirs(os.path.dirname(self.out_manifest), exist_ok=True)
        with open(self.out_manifest, "w", encoding="utf-8") as f:
            json.dump({"files": {rel: {"size": len(original), "sha256": hashlib.sha256(grown).hexdigest()}}}, f)

        getsize = os.path.getsize

        def getsize_then_grow(name: str) -> int:
            size = getsize(name)
            if name == path:
                with open(path, "ab") as f:
                    f.write(grown[len(original):])
            return size

        for jobs in (1, 4):
            with self.subTest(jobs=jobs), contextlib.redirect_stdout(io.StringIO()):
                with open(path, "wb") as f:
                    f.write(original)
                with patch.object(os.path, "getsize", side_effect=getsize_then_grow):
                    with self.assertRaisesRegex(manifest.ManifestError, r"alpha\.bin.*size changed before hashing"):
                        manifest.verify(self.pkg, self.out_manifest, jobs=jobs)

    def test_verify_detects_missing_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        missing_target = os.path.join(self.pkg, "content", "asset1.dat")
        os.remove(missing_target)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("MISSING   content/asset1.dat", output)
        self.assertIn("1 problem(s)", output)

    def test_verify_detects_size_mismatch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        target = os.path.join(self.pkg, "content", "asset1.dat")
        with open(target, "ab") as f:
            f.write(b"appended extra bytes")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("SIZE      content/asset1.dat", output)
        self.assertIn("1 problem(s)", output)

    def test_verify_detects_changed_content_same_size(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        target = os.path.join(self.pkg, "content", "asset1.dat")
        with open(target, "r+b") as f:
            original = f.read()
            mutated = bytes([original[0] ^ 0xFF]) + original[1:]
            f.seek(0)
            f.write(mutated)

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("CHANGED   content/asset1.dat", output)
        self.assertIn("1 problem(s)", output)

    def test_verify_reports_new_untracked_file(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        extra = os.path.join(self.pkg, "content", "unexpected_new.dat")
        with open(extra, "wb") as f:
            f.write(b"new rogue asset")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 0)
        output = out.getvalue()
        self.assertIn("NEW       content/unexpected_new.dat", output)
        self.assertIn("OK", output)

    def test_verify_accumulates_multiple_corruptions(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        os.remove(os.path.join(self.pkg, "alpha.bin"))

        with open(os.path.join(self.pkg, "bravo.bin"), "ab") as f:
            f.write(b"extra")

        tbl = os.path.join(self.pkg, "content", "data", "table.bin")
        with open(tbl, "r+b") as f:
            b = f.read()
            f.seek(0)
            f.write(bytes([b[0] ^ 0xFF]) + b[1:])

        with open(os.path.join(self.pkg, "new_file.dat"), "wb") as f:
            f.write(b"extra")

        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        self.assertEqual(rc, 1)
        output = out.getvalue()
        self.assertIn("MISSING   alpha.bin", output)
        self.assertIn("SIZE      bravo.bin", output)
        self.assertIn("CHANGED   content/data/table.bin", output)
        self.assertIn("NEW       new_file.dat", output)
        self.assertIn("3 problem(s)", output)

    def test_verify_messages_do_not_depend_on_jobs(self):
        with contextlib.redirect_stdout(io.StringIO()):
            manifest.create(self.pkg, self.out_manifest, jobs=1)

        target = os.path.join(self.pkg, "content", "asset1.dat")
        with open(target, "r+b") as f:
            orig = f.read()
            f.seek(0)
            f.write(bytes([orig[0] ^ 0xFF]) + orig[1:])

        out_serial = io.StringIO()
        with contextlib.redirect_stdout(out_serial):
            rc_serial = manifest.verify(self.pkg, self.out_manifest, jobs=1)

        out_parallel = io.StringIO()
        with contextlib.redirect_stdout(out_parallel):
            rc_parallel = manifest.verify(self.pkg, self.out_manifest, jobs=4)

        self.assertEqual(rc_serial, rc_parallel)
        self.assertEqual(out_serial.getvalue(), out_parallel.getvalue())


class ManifestCliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "pkg")
        os.makedirs(self.pkg)
        with open(os.path.join(self.pkg, "sample.dat"), "wb") as f:
            f.write(b"cli test payload bytes")
        self.out_manifest = os.path.join(self.tmp.name, "manifest.json")

    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True, errors="replace")

    def test_cli_create_and_verify(self):
        p_create = self.run_cli("create", self.pkg, "--out", self.out_manifest, "--jobs", "2")
        self.assertEqual(p_create.returncode, 0, p_create.stderr)
        self.assertIn("1 files", p_create.stdout)

        p_verify = self.run_cli("verify", self.pkg, "--manifest", self.out_manifest, "--jobs", "2")
        self.assertEqual(p_verify.returncode, 0, p_verify.stderr)
        self.assertIn("OK", p_verify.stdout)

    def test_in_tree_manifest_is_ignored_on_create_and_verify(self):
        manifest_path = os.path.join(self.pkg, "reports", "manifest.json")
        for _ in range(2):
            result = self.run_cli("create", self.pkg, "--out", manifest_path)
            self.assertEqual(result.returncode, 0, result.stderr)
        with open(manifest_path, encoding="utf-8") as f:
            self.assertEqual(set(json.load(f)["files"]), {"sample.dat"})
        result = self.run_cli("verify", self.pkg, "--manifest", manifest_path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "OK")

    def test_cli_verify_refuses_an_empty_manifest_for_a_folder_with_files(self):
        # A create that was cut short, or a mistyped --manifest, used to print OK without checking a file.
        empty = os.path.join(self.tmp.name, "empty.json")
        with open(empty, "w", encoding="utf-8", newline="\n") as f:
            f.write('{"files": {}}\n')
        result = self.run_cli("verify", self.pkg, "--manifest", empty)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("lists no files", result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_verify_failure_exit_code(self):
        self.assertEqual(self.run_cli("create", self.pkg, "--out", self.out_manifest, "--jobs", "1").returncode, 0)

        with open(os.path.join(self.pkg, "sample.dat"), "wb") as f:
            f.write(b"corrupted bytes!")

        p_verify = self.run_cli("verify", self.pkg, "--manifest", self.out_manifest, "--jobs", "1")
        self.assertEqual(p_verify.returncode, 1)
        self.assertIn("problem(s)", p_verify.stdout)

    def test_cli_rejects_invalid_game_directories(self):
        # Start with a valid empty manifest, which would otherwise verify even if the scan never ran.
        empty = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty)
        self.assertEqual(self.run_cli("create", empty, "--out", self.out_manifest).returncode, 0)
        self.assertEqual(self.run_cli("verify", empty, "--manifest", self.out_manifest).returncode, 0)
        with open(self.out_manifest, "rb") as f:
            original = f.read()

        for game_dir in (os.path.join(self.tmp.name, "missing"), os.path.join(self.pkg, "sample.dat")):
            for command, option in (("create", "--out"), ("verify", "--manifest")):
                with self.subTest(game_dir=game_dir, command=command):
                    result = self.run_cli(command, game_dir, option, self.out_manifest)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(os.path.basename(game_dir), result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
                    self.assertEqual(result.stdout, "")
                    with open(self.out_manifest, "rb") as f:
                        self.assertEqual(f.read(), original)

    def test_cli_verify_reports_a_damaged_manifest_without_a_traceback(self):
        good_hash = hashlib.sha256(b"cli test payload bytes").hexdigest()
        damaged = {
            "truncated": b'{"created": "x", "files": {"sample.dat": {"si',
            "not_utf8": b'\xff\xfe{"files": {}}',
            "not_an_object": b"[]",
            "files_is_a_list": b'{"files": []}',
            "no_files": b'{"created": "x"}',
            "entry_is_a_number": b'{"files": {"sample.dat": 5}}',
            "no_sha256": b'{"files": {"sample.dat": {"size": 22}}}',
            "size_is_a_string": b'{"files": {"sample.dat": {"size": "22", "sha256": "00"}}}',
            "dup_sha256": (
                f'{{"files": {{"sample.dat": {{"size": 22, "sha256": "{"0" * 64}", "sha256": "{good_hash}"}}}}}}'
            ).encode(),
            "dup_file_entry": (
                f'{{"files": {{"sample.dat": {{"size": 22, "sha256": "{"0" * 64}"}}, '
                f'"sample.dat": {{"size": 22, "sha256": "{good_hash}"}}}}}}'
            ).encode(),
            "dup_top_files": (
                f'{{"files": {{"sample.dat": {{"size": 22, "sha256": "{"0" * 64}"}}}}, '
                f'"files": {{"sample.dat": {{"size": 22, "sha256": "{good_hash}"}}}}}}'
            ).encode(),
        }
        for name, data in damaged.items():
            with self.subTest(name=name):
                path = os.path.join(self.tmp.name, f"{name}.json")
                with open(path, "wb") as f:
                    f.write(data)
                result = self.run_cli("verify", self.pkg, "--manifest", path, "--jobs", "1")
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("ERROR:", result.stderr)
                self.assertIn(f"{name}.json", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                self.assertEqual(result.stdout, "")


class ManifestAtomicCreateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "pkg")
        os.makedirs(self.pkg)
        with open(os.path.join(self.pkg, "sample.dat"), "wb") as f:
            f.write(b"atomic create payload")
        self.out = os.path.join(self.tmp.name, "manifest.json")

    def create(self, jobs=1):
        with contextlib.redirect_stdout(io.StringIO()):
            return manifest.create(self.pkg, self.out, jobs=jobs)

    def test_create_preserves_an_existing_partial_sibling(self):
        sibling = self.out + ".partial"
        with open(sibling, "wb") as f:
            f.write(b"existing report")

        self.assertEqual(self.create(), 0)
        with open(sibling, "rb") as f:
            self.assertEqual(f.read(), b"existing report")

    def test_overlapping_creates_each_publish_their_own_manifest(self):
        empty_pkg = os.path.join(self.tmp.name, "empty")
        os.mkdir(empty_pkg)
        replace = os.replace

        def publish_inner_first(staged, destination):
            with patch.object(manifest.os, "replace", replace):
                self.assertEqual(manifest.create(empty_pkg, self.out), 0)
            self.assertEqual(manifest.load_manifest(self.out), {})
            replace(staged, destination)

        with patch.object(manifest.os, "replace", side_effect=publish_inner_first):
            self.assertEqual(self.create(), 0)
        self.assertEqual(set(manifest.load_manifest(self.out)), {"sample.dat"})
        self.assertEqual(sorted(os.listdir(self.tmp.name)), ["empty", "manifest.json", "pkg"])

    def test_an_interrupted_create_keeps_the_previous_manifest(self):
        self.assertEqual(self.create(), 0)
        with open(self.out, "rb") as f:
            original = f.read()
        with open(os.path.join(self.pkg, "added.dat"), "wb") as f:
            f.write(b"new file")

        for error in (OSError, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                def fail(obj, f, **kwargs):
                    f.write('{"created": "')
                    raise error("stopped")
                with patch.object(manifest.json, "dump", side_effect=fail):
                    with self.assertRaises(error):
                        self.create()
                with open(self.out, "rb") as f:
                    self.assertEqual(f.read(), original)
                self.assertEqual(sorted(os.listdir(self.tmp.name)), ["manifest.json", "pkg"])

        self.assertEqual(self.create(), 0)
        self.assertEqual(sorted(manifest.load_manifest(self.out)), ["added.dat", "sample.dat"])

    def test_size_change_before_open_keeps_the_previous_manifest(self):
        self.assertEqual(self.create(), 0)
        with open(self.out, "rb") as f:
            previous = f.read()
        source = os.path.join(self.pkg, "sample.dat")

        def open_after_growth(path, mode, **kwargs):
            if path == source and mode == "rb":
                with open(source, "ab") as f:
                    f.write(b"more bytes")
            return open(path, mode, **kwargs)

        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                with patch.object(manifest, "open", side_effect=open_after_growth, create=True):
                    with self.assertRaisesRegex(manifest.ManifestError, r"sample\.dat.*changed"):
                        self.create(jobs=jobs)
                with open(self.out, "rb") as f:
                    self.assertEqual(f.read(), previous)

    def test_same_size_change_during_read_keeps_the_previous_manifest(self):
        self.assertEqual(self.create(), 0)
        with open(self.out, "rb") as f:
            previous = f.read()
        source = os.path.join(self.pkg, "sample.dat")

        def open_with_edit(path, mode, **kwargs):
            stream = open(path, mode, **kwargs)
            if path == source and mode == "rb":
                original_read = stream.read

                def read_then_edit(size):
                    chunk = original_read(size)
                    stream.read = original_read
                    stamp = os.stat(source)
                    with open(source, "r+b") as f:
                        f.write(b"X")
                    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 2_000_000_000))
                    return chunk

                stream.read = read_then_edit
            return stream

        with patch.object(manifest, "open", side_effect=open_with_edit, create=True):
            with self.assertRaisesRegex(manifest.ManifestError, r"sample\.dat.*changed"):
                self.create()
        with open(self.out, "rb") as f:
            self.assertEqual(f.read(), previous)


class ManifestDigestCaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.pkg = os.path.join(self.tmp.name, "pkg")
        os.makedirs(self.pkg)
        self.payloads = {
            "sample.dat": b"synthetic digest case fixture\n",
            "second.dat": b"a second file for parallel verification\n",
        }
        for rel, payload in self.payloads.items():
            with open(os.path.join(self.pkg, rel), "wb") as f:
                f.write(payload)
        self.entries = {
            rel: {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            for rel, payload in self.payloads.items()
        }
        self.out = os.path.join(self.tmp.name, "manifest.json")

    def write_manifest(self, case="lower"):
        entries = {}
        for rel, info in self.entries.items():
            original = info["sha256"]
            if case == "upper":
                digest = original.upper()
            elif case == "mixed":
                digest = "".join(c.upper() if i % 2 else c for i, c in enumerate(original))
                self.assertNotEqual(digest, original.upper())
            else:
                digest = original
            if case != "lower":
                self.assertNotEqual(digest, original)
            entries[rel] = {"size": info["size"], "sha256": digest}
        with open(self.out, "w", encoding="utf-8") as f:
            json.dump({"files": entries}, f)
        with open(self.out, "rb") as f:
            return f.read()

    def assert_files_unchanged(self, expected):
        for rel, payload in expected.items():
            with open(os.path.join(self.pkg, rel), "rb") as f:
                self.assertEqual(f.read(), payload)

    def run_cli(self, jobs):
        return subprocess.run(
            [sys.executable, "-I", "-B", TOOL, "verify", self.pkg,
             "--manifest", self.out, "--jobs", str(jobs)],
            capture_output=True, text=True, errors="replace")

    def test_verify_accepts_hex_digest_case(self):
        for case in ("lower", "upper", "mixed"):
            for jobs in (1, 4):
                with self.subTest(case=case, jobs=jobs):
                    before = self.write_manifest(case)
                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        self.assertEqual(manifest.verify(self.pkg, self.out, jobs=jobs), 0)
                    self.assertEqual(output.getvalue(), "OK\n")
                    self.assert_files_unchanged(self.payloads)
                    with open(self.out, "rb") as f:
                        self.assertEqual(f.read(), before)

    def test_cli_verify_accepts_hex_digest_case(self):
        for case in ("upper", "mixed"):
            for jobs in (1, 2):
                with self.subTest(case=case, jobs=jobs):
                    before = self.write_manifest(case)
                    result = self.run_cli(jobs)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, "OK\n")
                    self.assertEqual(result.stderr, "")
                    self.assert_files_unchanged(self.payloads)
                    with open(self.out, "rb") as f:
                        self.assertEqual(f.read(), before)

    def test_uppercase_digest_still_detects_changed_content(self):
        self.entries["sample.dat"]["sha256"] = self.entries["sample.dat"]["sha256"].upper()
        before = self.write_manifest()
        original = self.payloads["sample.dat"]
        changed = bytes([original[0] ^ 0xFF]) + original[1:]
        with open(os.path.join(self.pkg, "sample.dat"), "wb") as f:
            f.write(changed)
        for jobs in (1, 4):
            with self.subTest(jobs=jobs):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(manifest.verify(self.pkg, self.out, jobs=jobs), 1)
                self.assertEqual(output.getvalue(), "CHANGED   sample.dat\n1 problem(s)\n")
                self.assert_files_unchanged({**self.payloads, "sample.dat": changed})
                with open(self.out, "rb") as f:
                    self.assertEqual(f.read(), before)

    def test_uppercase_digests_still_report_missing_and_size(self):
        before = self.write_manifest("upper")
        os.remove(os.path.join(self.pkg, "sample.dat"))
        longer = self.payloads["second.dat"] + b"extra"
        with open(os.path.join(self.pkg, "second.dat"), "wb") as f:
            f.write(longer)
        for jobs in (1, 4):
            with self.subTest(jobs=jobs):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(manifest.verify(self.pkg, self.out, jobs=jobs), 1)
                self.assertEqual(output.getvalue(), "MISSING   sample.dat\nSIZE      second.dat\n2 problem(s)\n")
                self.assertFalse(os.path.exists(os.path.join(self.pkg, "sample.dat")))
                self.assert_files_unchanged({"second.dat": longer})
                with open(self.out, "rb") as f:
                    self.assertEqual(f.read(), before)

    def test_cli_verify_does_not_accept_malformed_digests(self):
        original = self.entries["sample.dat"]["sha256"]
        invalid = (" " + original.upper() + " ", " ".join(original[i:i + 2] for i in range(0, 64, 2)),
                   "F" * 64, "Z" * 64, "", original[:-1])
        for digest in invalid:
            for jobs in (1, 2):
                with self.subTest(digest=digest, jobs=jobs):
                    self.entries["sample.dat"]["sha256"] = digest
                    before = self.write_manifest()
                    result = self.run_cli(jobs)
                    self.assertEqual(result.returncode, 1, result.stderr)
                    self.assertEqual(result.stdout, "CHANGED   sample.dat\n1 problem(s)\n")
                    self.assertEqual(result.stderr, "")
                    self.assert_files_unchanged(self.payloads)
                    with open(self.out, "rb") as f:
                        self.assertEqual(f.read(), before)


class ManifestPosixBackslashTest(unittest.TestCase):
    @unittest.skipIf(os.sep == "\\", "backslashes are path separators on Windows")
    def test_posix_literal_backslash_does_not_collide_with_subdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            pkg = os.path.join(tmp, "pkg")
            os.makedirs(os.path.join(pkg, "sub"))
            with open(os.path.join(pkg, "sub", "a.dat"), "wb") as f:
                f.write(b"nested")
            with open(os.path.join(pkg, "sub\\a.dat"), "wb") as f:
                f.write(b"literal-backslash")
            out = os.path.join(tmp, "manifest.json")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(manifest.create(pkg, out, jobs=1), 0)
            entries = manifest.load_manifest(out)
            self.assertEqual(sorted(entries), ["sub/a.dat", "sub\\a.dat"])
            self.assertEqual(entries["sub/a.dat"]["size"], len(b"nested"))
            self.assertEqual(entries["sub\\a.dat"]["size"], len(b"literal-backslash"))
            verify_out = io.StringIO()
            with contextlib.redirect_stdout(verify_out):
                self.assertEqual(manifest.verify(pkg, out, jobs=1), 0)
            self.assertEqual(verify_out.getvalue(), "OK\n")


class ManifestOutGuardTest(unittest.TestCase):
    def test_out_guard_follows_the_folding_volume(self):
        """--out asks report_paths, so a case-insensitive APFS volume refuses an --out that is one of the hashed
        files; the private realpath+normcase copy answered "two files" there and let the manifest replace it."""
        with tempfile.TemporaryDirectory() as tmp:
            composed = os.path.join(tmp, "café.json")
            decomposed = os.path.join(tmp, unicodedata.normalize("NFD", "café.json"))
            folding = [True]
            with patch.object(report_paths, "sys", Mock(platform="darwin")), \
                    patch.object(report_paths, "case_insensitive_volume", lambda path: folding[0]):
                self.assertTrue(manifest.same_report_path(composed, decomposed))
                folding[0] = False
                self.assertFalse(manifest.same_report_path(composed, decomposed))


class ManifestBelowNormalTest(unittest.TestCase):
    def test_below_normal_callable(self):
        manifest.below_normal()


if __name__ == "__main__":
    unittest.main(verbosity=2)
