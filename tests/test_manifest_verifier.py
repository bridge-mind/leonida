"""Unit tests for tools/manifest_verifier.py using synthetic file fixtures.

Verifies SHA-256 manifest integrity in parallel with live progress reporting.
Zero game data used.

    python -I tests/test_manifest_verifier.py
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import posixpath
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "manifest_verifier.py")

# Add root directory to sys.path so manifest_verifier can be imported directly
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import tools.manifest_verifier as mv

SYNTHETIC_SIZES = {
    "alpha.bin": 256 * 1024,
    "bravo.bin": 64 * 1024,
    "charlie.bin": 128 * 1024,
    "delta.bin": 32 * 1024,
    "nested/echo.bin": 16 * 1024,
    "nested/deep/foxtrot.bin": 8 * 1024,
}


def make_payload(size: int, seed: str) -> bytes:
    block = hashlib.sha256(seed.encode("utf-8")).digest()
    return (block * (size // len(block) + 1))[:size]


class TestManifestVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.game_dir = os.path.join(self.root, "game")
        os.makedirs(self.game_dir, exist_ok=True)
        self.payloads: dict[str, bytes] = {}

        for rel, size in SYNTHETIC_SIZES.items():
            abs_p = os.path.join(self.game_dir, *rel.split("/"))
            os.makedirs(os.path.dirname(abs_p), exist_ok=True)
            data = make_payload(size, rel)
            with open(abs_p, "wb") as f:
                f.write(data)
            self.payloads[rel] = data

        self.manifest_path = os.path.join(self.root, "manifest.json")
        self.write_manifest(self.manifest_path, self.payloads)

    def write_manifest(
        self,
        dest: str,
        files_map: dict[str, bytes],
        wrapper: bool = True,
        upper_hash: bool = False,
    ) -> str:
        entries = {}
        for rel, data in files_map.items():
            h = hashlib.sha256(data).hexdigest()
            if upper_hash:
                h = h.upper()
            entries[rel] = {"size": len(data), "sha256": h}
        doc = {"created": "2026-11-22 00:00:00", "files": entries} if wrapper else entries
        with open(dest, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        return dest

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-I", TOOL, *args],
            cwd=self.root,
            capture_output=True,
            text=True,
            errors="replace",
        )

    def test_sha256_file(self) -> None:
        sample_path = os.path.join(self.game_dir, "alpha.bin")
        expected = hashlib.sha256(self.payloads["alpha.bin"]).hexdigest()
        self.assertEqual(mv.sha256_file(sample_path), expected)

    def test_sha256_file_reuses_buffer_across_short_reads(self) -> None:
        payload = bytes(range(25))
        sample_path = os.path.join(self.root, "short-reads.bin")
        with open(sample_path, "wb") as f:
            f.write(payload)

        buffers = []
        with open(sample_path, "rb", buffering=0) as f:
            real_readinto = f.readinto

            def read_some(buffer):
                view = memoryview(buffer)
                buffers.append(view.obj)
                return real_readinto(view[:3])

            with patch.object(f, "readinto", read_some), patch.object(mv, "open", return_value=f, create=True):
                self.assertEqual(
                    mv.sha256_file(sample_path, chunk_size=8),
                    hashlib.sha256(payload).hexdigest(),
                )

        self.assertGreater(len(buffers), 1)
        self.assertEqual(len(buffers[0]), 8)
        self.assertTrue(all(buf is buffers[0] for buf in buffers))

        # Small files cap the reusable buffer to the file size rather than allocating CHUNK.
        small_buffers = []
        with open(sample_path, "rb", buffering=0) as f:
            real_readinto = f.readinto

            def record_buffer(buffer):
                view = memoryview(buffer)
                small_buffers.append(view.obj)
                return real_readinto(view)

            with patch.object(f, "readinto", record_buffer), patch.object(mv, "open", return_value=f, create=True):
                self.assertEqual(mv.sha256_file(sample_path), hashlib.sha256(payload).hexdigest())

        self.assertEqual(len(small_buffers), 1)
        self.assertEqual(len(small_buffers[0]), len(payload))

    def test_sha256_file_rejects_nonpositive_chunk_size(self) -> None:
        sample_path = os.path.join(self.game_dir, "alpha.bin")
        for bad in (0, -1):
            with self.subTest(chunk_size=bad):
                with self.assertRaises(ValueError):
                    mv.sha256_file(sample_path, chunk_size=bad)

    def test_walk_normalization(self) -> None:
        items = dict(mv.walk(self.game_dir))
        self.assertEqual(sorted(items.keys()), sorted(SYNTHETIC_SIZES.keys()))
        for rel in items:
            self.assertNotIn("\\", rel)

    def test_posix_backslash_filename_cannot_match_a_nested_path(self) -> None:
        payload = b"same bytes as the expected file"
        nested = "nested/echo.bin"
        literal_name = r"nested\echo.bin"
        self.write_manifest(self.manifest_path, {nested: payload})

        # Simulate POSIX directory entries even on Windows, where a filename cannot contain a literal backslash.
        target = "/target"
        posix_os = Mock(sep="/", path=Mock(wraps=posixpath))
        posix_os.path.isdir.return_value = True
        posix_os.stat.side_effect = os.stat
        posix_os.path.getsize.return_value = len(payload)
        posix_os.walk.return_value = [(target, [], [literal_name])]
        posix_os.stat.side_effect = lambda path: os.stat(self.game_dir if path == target else path)
        with (patch.object(mv, "os", posix_os),
              patch.object(mv, "sha256_file", return_value=hashlib.sha256(payload).hexdigest()) as hash_file):
            report = mv.verify_manifest(target, self.manifest_path)
        self.assertFalse(report.passed)
        self.assertEqual(report.missing, [nested])
        self.assertEqual(report.untracked, [literal_name])
        hash_file.assert_not_called()

    def test_load_manifest_valid_and_case_insensitive(self) -> None:
        manifest_upper = os.path.join(self.root, "manifest_upper.json")
        self.write_manifest(manifest_upper, self.payloads, upper_hash=True)
        entries = mv.load_manifest(manifest_upper)
        self.assertEqual(len(entries), len(SYNTHETIC_SIZES))
        for rel, info in entries.items():
            self.assertEqual(info["sha256"], info["sha256"].lower())
            self.assertEqual(info["size"], SYNTHETIC_SIZES[rel])

    def test_load_manifest_flat_dictionary(self) -> None:
        flat_path = os.path.join(self.root, "manifest_flat.json")
        self.write_manifest(flat_path, self.payloads, wrapper=False)
        entries = mv.load_manifest(flat_path)
        self.assertEqual(len(entries), len(SYNTHETIC_SIZES))

    def test_load_manifest_flat_skips_metadata_keys(self) -> None:
        flat_path = os.path.join(self.root, "manifest_flat_meta.json")
        doc = {
            "created": "2026-11-22 00:00:00",
            "title": "synthetic",
            "version": "1.0",
            "note": "top-level metadata",
            "alpha.bin": {
                "size": len(self.payloads["alpha.bin"]),
                "sha256": hashlib.sha256(self.payloads["alpha.bin"]).hexdigest(),
            },
        }
        with open(flat_path, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        entries = mv.load_manifest(flat_path)
        self.assertEqual(list(entries.keys()), ["alpha.bin"])

    def test_flat_manifest_checks_files_named_like_metadata(self) -> None:
        files = {"alpha.bin": b"keep", "version": b"old"}
        directory = os.path.join(self.root, "flat-game")
        os.mkdir(directory)
        for name, data in files.items():
            with open(os.path.join(directory, name), "wb") as f:
                f.write(b"new" if name == "version" else data)
        path = self.write_manifest(os.path.join(self.root, "flat.json"), files, wrapper=False)
        report = mv.verify_manifest(directory, path, jobs=1)
        self.assertFalse(report.passed)
        self.assertEqual(report.problems, {"version": "CHANGED"})
        self.assertEqual(report.untracked, [])

    def test_load_manifest_duplicate_normalized_paths_raise(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        for wrapper in (True, False):
            with self.subTest(wrapper=wrapper):
                entry = {
                    "nested/a.bin": {"size": 1, "sha256": digest},
                    "nested\\a.bin": {"size": 1, "sha256": digest},
                }
                doc = {"files": entry} if wrapper else entry
                bad_path = os.path.join(self.root, f"dup_wrapper_{wrapper}.json")
                with open(bad_path, "w", encoding="utf-8") as f:
                    json.dump(doc, f)
                with self.assertRaises(mv.ManifestError) as cm:
                    mv.load_manifest(bad_path)
                self.assertIn(bad_path, str(cm.exception))
                self.assertIn("nested/a.bin", str(cm.exception))

    def test_cli_rejects_duplicate_json_keys(self) -> None:
        digest = hashlib.sha256(self.payloads["alpha.bin"]).hexdigest()
        entries = f'{{"alpha.bin": "wrong", "alpha.bin": "{digest}"}}'
        cases = (
            ("alpha.bin", entries),
            ("alpha.bin", f'{{"files": {entries}}}'),
            ("files", '{"files": {"missing.bin": "wrong"}, "files": {}}'),
            ("sha256", f'{{"files": {{"alpha.bin": {{"sha256": "wrong", "sha256": "{digest}"}}}}}}'),
        )
        for key, document in cases:
            with self.subTest(document=document):
                with open(self.manifest_path, "w", encoding="utf-8") as f:
                    f.write(document)
                result = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--quiet")
                self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                self.assertIn("duplicate JSON key", result.stderr)
                self.assertIn(repr(key), result.stderr)
                self.assertIn(self.manifest_path, result.stderr)

    def test_load_manifest_single_spelling_control(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        for wrapper in (True, False):
            for spelling in ("nested/a.bin", "nested\\a.bin"):
                with self.subTest(wrapper=wrapper, spelling=spelling):
                    entry = {spelling: {"size": 1, "sha256": digest}}
                    doc = {"files": entry} if wrapper else entry
                    p = os.path.join(self.root, f"one_{wrapper}_{'fwd' if '/' in spelling else 'bwd'}.json")
                    with open(p, "w", encoding="utf-8") as f:
                        json.dump(doc, f)
                    entries = mv.load_manifest(p)
                    self.assertEqual(list(entries.keys()), ["nested/a.bin"])

    def test_nested_manifest_verifies_files_named_like_metadata_keys(self) -> None:
        meta_dir = os.path.join(self.root, "meta_game")
        os.makedirs(meta_dir, exist_ok=True)
        meta_files = {
            "created": b"created-payload",
            "title": b"title-payload!!",
            "version": b"version-payload",
            "note": b"note-payload",
        }
        for name, data in meta_files.items():
            with open(os.path.join(meta_dir, name), "wb") as f:
                f.write(data)
        manifest_p = self.write_manifest(os.path.join(self.root, "meta_manifest.json"), meta_files, wrapper=True)

        entries = mv.load_manifest(manifest_p)
        self.assertEqual(sorted(entries.keys()), sorted(meta_files.keys()))

        ok_report = mv.verify_manifest(meta_dir, manifest_p, jobs=2, strict=True)
        self.assertTrue(ok_report.passed)
        self.assertEqual(sorted(ok_report.matched), sorted(meta_files.keys()))
        self.assertEqual(ok_report.untracked, [])

        # Tamper (same size -> CHANGED), truncate (-> SIZE), remove (-> MISSING)
        with open(os.path.join(meta_dir, "version"), "wb") as f:
            f.write(b"VERSION-PAYLOAD")
        with open(os.path.join(meta_dir, "title"), "wb") as f:
            f.write(b"short")
        os.remove(os.path.join(meta_dir, "note"))

        bad_report = mv.verify_manifest(meta_dir, manifest_p, jobs=1, strict=True)
        self.assertFalse(bad_report.passed)
        self.assertEqual(bad_report.problems, {
            "note": "MISSING",
            "title": "SIZE",
            "version": "CHANGED",
        })
        self.assertEqual(bad_report.matched, ["created"])
        self.assertEqual(bad_report.untracked, [])

    def test_load_manifest_non_dict_files_raises(self) -> None:
        for bad_value in ("abc", [], 123, None):
            with self.subTest(files_value=bad_value):
                bad_path = os.path.join(self.root, f"bad_files_{type(bad_value).__name__}.json")
                with open(bad_path, "w", encoding="utf-8") as f:
                    json.dump({"created": "2026-11-22", "files": bad_value}, f)
                with self.assertRaisesRegex(mv.ManifestError, "'files' must be a JSON object"):
                    mv.load_manifest(bad_path)

    def test_load_manifest_boolean_size_raises(self) -> None:
        digest = hashlib.sha256(b"x").hexdigest()
        for bool_size in (True, False):
            for wrapper in (True, False):
                with self.subTest(size=bool_size, wrapper=wrapper):
                    bad_path = os.path.join(self.root, f"bool_size_{bool_size}_{wrapper}.json")
                    entry = {"a.bin": {"size": bool_size, "sha256": digest}}
                    doc = {"files": entry} if wrapper else entry
                    with open(bad_path, "w", encoding="utf-8") as f:
                        json.dump(doc, f)
                    with self.assertRaises(mv.ManifestError):
                        mv.load_manifest(bad_path)

    def test_walk_unreadable_directory_raises_and_cli_exits_2(self) -> None:
        def failing_walk(top: str, *args, onerror=None, **kwargs):
            err = PermissionError(13, "Permission denied", os.path.join(top, "nested"))
            if onerror is not None:
                onerror(err)
            return
            yield  # pragma: no cover

        with patch("tools.manifest_verifier.os.walk", side_effect=failing_walk):
            with self.assertRaises(PermissionError):
                list(mv.walk(self.game_dir))
            with self.assertRaises(PermissionError):
                mv.verify_manifest(self.game_dir, self.manifest_path, jobs=1)

            stderr_buf = io.StringIO()
            with contextlib.redirect_stderr(stderr_buf):
                rc = mv.main([self.game_dir, "--manifest", self.manifest_path, "--quiet"])
            self.assertEqual(rc, 2)
            self.assertIn("ERROR:", stderr_buf.getvalue())
            self.assertIn("Permission denied", stderr_buf.getvalue())

    def test_load_manifest_invalid_json(self) -> None:
        bad_json = os.path.join(self.root, "bad.json")
        with open(bad_json, "w", encoding="utf-8") as f:
            f.write("not json at all {")
        with self.assertRaises(mv.ManifestError):
            mv.load_manifest(bad_json)

    def test_load_manifest_missing_file(self) -> None:
        missing_path = os.path.join(self.root, "non_existent.json")
        with self.assertRaises(mv.ManifestError):
            mv.load_manifest(missing_path)

    def test_cli_distinguishes_manifest_path_errors(self) -> None:
        stat_path = os.stat

        def stat_with_denied_manifest(path, *args, **kwargs):
            if path == self.manifest_path:
                raise PermissionError("Simulated permission denied")
            return stat_path(path, *args, **kwargs)

        for flags in ([], ["--json"]):
            with self.subTest(flags=flags):
                with (
                    patch.object(mv.os, "stat", side_effect=stat_with_denied_manifest),
                    patch("sys.stdout", new_callable=io.StringIO) as stdout,
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    rc = mv.main([self.game_dir, "--manifest", self.manifest_path, "--quiet", *flags])
                self.assertEqual(rc, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("cannot read manifest", stderr.getvalue())
                self.assertIn("Simulated permission denied", stderr.getvalue())

    def test_cli_reports_non_regular_manifest_path(self) -> None:
        for flags in ([], ["--json"]):
            with self.subTest(flags=flags):
                res = self.run_cli(self.game_dir, "--manifest", self.root, "--quiet", *flags)
                self.assertEqual(res.returncode, 2)
                self.assertEqual(res.stdout, "")
                self.assertIn("manifest path is not a regular file", res.stderr)

    def test_load_manifest_malformed_entries(self) -> None:
        invalid_entries = os.path.join(self.root, "invalid_entries.json")
        with open(invalid_entries, "w", encoding="utf-8") as f:
            json.dump({"files": {"bad.bin": {"size": "not-an-int", "sha256": "abc"}}}, f)
        with self.assertRaises(mv.ManifestError):
            mv.load_manifest(invalid_entries)

    def test_cli_rejects_malformed_sha256_digests(self) -> None:
        for digest in ("a" * 63, "a" * 65, "g" * 64):
            for entry in (digest, {"size": len(self.payloads["alpha.bin"]), "sha256": digest}):
                with self.subTest(digest=digest, entry=entry):
                    with open(self.manifest_path, "w", encoding="utf-8", newline="\n") as f:
                        json.dump({"files": {"alpha.bin": entry}}, f)
                    result = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--json")
                    self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(self.manifest_path, result.stderr)
                    self.assertIn("alpha.bin", result.stderr)
                    self.assertIn("64 hexadecimal characters", result.stderr)

    def test_verify_target_dir_missing(self) -> None:
        with self.assertRaises(mv.ManifestError):
            mv.verify_manifest(os.path.join(self.root, "missing_dir"), self.manifest_path)

    def test_cli_distinguishes_target_path_errors(self) -> None:
        stat_path = os.stat

        def stat_with_denied_target(path, *args, **kwargs):
            if path == self.game_dir:
                raise PermissionError("Simulated permission denied")
            return stat_path(path, *args, **kwargs)

        for flags in ([], ["--json"]):
            with self.subTest(flags=flags, case="denied"):
                with (
                    patch.object(mv.os, "stat", side_effect=stat_with_denied_target),
                    patch("sys.stdout", new_callable=io.StringIO) as stdout,
                    patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    rc = mv.main([self.game_dir, "--manifest", self.manifest_path, "--quiet", *flags])
                self.assertEqual(rc, 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertIn("cannot access target directory", stderr.getvalue())
                self.assertIn("Simulated permission denied", stderr.getvalue())

            for target, message in (
                (self.manifest_path, "target path is not a directory"),
                (os.path.join(self.root, "absent"), "target directory does not exist"),
            ):
                with self.subTest(flags=flags, target=target):
                    res = self.run_cli(target, "--manifest", self.manifest_path, "--quiet", *flags)
                    self.assertEqual(res.returncode, 2)
                    self.assertEqual(res.stdout, "")
                    self.assertIn(message, res.stderr)

    def test_clean_verification_serial_and_parallel(self) -> None:
        serial_report = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=1)
        parallel_report = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=4)

        self.assertTrue(serial_report.passed)
        self.assertTrue(parallel_report.passed)
        self.assertEqual(serial_report.problem_count, 0)
        self.assertEqual(parallel_report.problem_count, 0)
        self.assertEqual(sorted(serial_report.matched), sorted(SYNTHETIC_SIZES.keys()))
        self.assertEqual(serial_report.matched, parallel_report.matched)
        self.assertEqual(serial_report.bytes_verified, sum(SYNTHETIC_SIZES.values()))
        self.assertEqual(parallel_report.bytes_verified, sum(SYNTHETIC_SIZES.values()))

    def test_problem_detection_and_deterministic_order(self) -> None:
        # 1. CHANGED (same size, modified byte)
        alpha_path = os.path.join(self.game_dir, "alpha.bin")
        with open(alpha_path, "r+b") as f:
            b = f.read(1)
            f.seek(0)
            f.write(bytes([b[0] ^ 0xFF]))

        # 2. SIZE (truncated)
        bravo_path = os.path.join(self.game_dir, "bravo.bin")
        with open(bravo_path, "r+b") as f:
            f.truncate(1024)

        # 3. MISSING
        charlie_path = os.path.join(self.game_dir, "charlie.bin")
        os.remove(charlie_path)

        # 4. NEW (untracked)
        new_path = os.path.join(self.game_dir, "extra.bin")
        with open(new_path, "wb") as f:
            f.write(b"untracked extra file")

        serial = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=1)
        parallel = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=4)

        self.assertFalse(serial.passed)
        self.assertFalse(parallel.passed)
        self.assertEqual(serial.problem_count, 3)
        self.assertEqual(parallel.problem_count, 3)
        self.assertEqual(serial.problems, parallel.problems)
        self.assertEqual(serial.problems["alpha.bin"], "CHANGED")
        self.assertEqual(serial.problems["bravo.bin"], "SIZE")
        self.assertEqual(serial.problems["charlie.bin"], "MISSING")
        self.assertEqual(serial.untracked, ["extra.bin"])
        self.assertEqual(parallel.untracked, ["extra.bin"])

        # Check stdout formatted printing
        out = io.StringIO()
        mv.print_report(serial, out=out)
        lines = out.getvalue().splitlines()
        expected = [
            "CHANGED   alpha.bin",
            "SIZE      bravo.bin",
            "MISSING   charlie.bin",
            "NEW       extra.bin",
            "3 problem(s)",
        ]
        self.assertEqual(lines, expected)

    def test_untracked_file_not_failing_without_strict(self) -> None:
        new_path = os.path.join(self.game_dir, "bonus.bin")
        with open(new_path, "wb") as f:
            f.write(b"bonus")

        report_default = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=2, strict=False)
        self.assertTrue(report_default.passed)
        self.assertEqual(report_default.problem_count, 0)
        self.assertEqual(report_default.untracked, ["bonus.bin"])

        report_strict = mv.verify_manifest(self.game_dir, self.manifest_path, jobs=2, strict=True)
        self.assertFalse(report_strict.passed)
        self.assertEqual(report_strict.untracked, ["bonus.bin"])
        self.assertEqual(report_strict.problem_count, 1)
        self.assertEqual(report_strict.to_dict()["problem_count"], 1)
        out = io.StringIO()
        mv.print_report(report_strict, out=out)
        self.assertIn("1 problem(s)", out.getvalue())

    def test_progress_reporting_callback_and_stream(self) -> None:
        progress_events: list[mv.ProgressInfo] = []
        stream = io.StringIO()

        report = mv.verify_manifest(
            self.game_dir,
            self.manifest_path,
            jobs=2,
            progress_callback=progress_events.append,
            progress_stream=stream,
        )

        self.assertTrue(report.passed)
        self.assertEqual(len(progress_events), len(SYNTHETIC_SIZES))

        # Check progress monotonicity
        prev_files = 0
        prev_bytes = 0
        for ev in progress_events:
            self.assertGreaterEqual(ev.completed_files, prev_files)
            self.assertGreaterEqual(ev.completed_bytes, prev_bytes)
            self.assertGreaterEqual(ev.pct, 0.0)
            self.assertLessEqual(ev.pct, 1.0001)
            prev_files = ev.completed_files
            prev_bytes = ev.completed_bytes

        self.assertEqual(progress_events[-1].completed_files, len(SYNTHETIC_SIZES))
        self.assertEqual(progress_events[-1].completed_bytes, sum(SYNTHETIC_SIZES.values()))
        self.assertAlmostEqual(progress_events[-1].pct, 1.0, places=4)

        stream_output = stream.getvalue()
        self.assertTrue(any("[" in line for line in stream_output.splitlines()))

    def test_unreadable_file_graceful_handling(self) -> None:
        hash_file = mv.sha256_file

        def mock_sha(path: str, *, expected_size: int | None = None) -> str:
            if os.path.basename(path) == "alpha.bin":
                raise OSError("Simulated permission denied")
            return hash_file(path, expected_size=expected_size)

        # Change exactly one byte so bravo reaches hashing and reports CHANGED.
        with open(os.path.join(self.game_dir, "bravo.bin"), "r+b") as f:
            f.write(bytes([self.payloads["bravo.bin"][0] ^ 1]))

        scheduled_bytes = sum(SYNTHETIC_SIZES.values())
        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                events = []
                with patch.object(mv, "sha256_file", side_effect=mock_sha):
                    report = mv.verify_manifest(
                        self.game_dir, self.manifest_path, jobs=jobs, progress_callback=events.append
                    )
                self.assertFalse(report.passed)
                self.assertEqual(report.unreadable, [("alpha.bin", "Simulated permission denied")])
                self.assertEqual(report.hash_mismatch, ["bravo.bin"])
                self.assertEqual(set(report.matched), set(SYNTHETIC_SIZES) - {"alpha.bin", "bravo.bin"})
                self.assertEqual(report.problem_count, 2)
                self.assertEqual(events[-1].completed_files, len(SYNTHETIC_SIZES))
                self.assertEqual(events[-1].completed_bytes, scheduled_bytes)
                self.assertEqual(events[-1].pct, 1.0)
                verified_bytes = scheduled_bytes - SYNTHETIC_SIZES["alpha.bin"]
                self.assertEqual(report.bytes_verified, verified_bytes)
                self.assertEqual(report.to_dict()["bytes_verified"], verified_bytes)

    def test_empty_manifest_and_zero_byte_file(self) -> None:
        zero_p = os.path.join(self.game_dir, "empty.bin")
        with open(zero_p, "wb"):
            pass
        empty_manifest = os.path.join(self.root, "empty.json")
        with open(empty_manifest, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "files": {
                        "empty.bin": {
                            "size": 0,
                            "sha256": hashlib.sha256(b"").hexdigest(),
                        }
                    }
                },
                f,
            )
        report = mv.verify_manifest(self.game_dir, empty_manifest, jobs=1)
        self.assertTrue(report.passed)
        self.assertIn("empty.bin", report.matched)

    def test_cli_manifest_without_entries_is_an_error_not_ok(self) -> None:
        # Every file would otherwise be NEW, and the run would print OK and exit 0 having verified nothing.
        empty = os.path.join(self.root, "no_entries.json")
        for doc in ({"files": {}}, {}, {"created": "2026-11-22 00:00:00"}):
            with self.subTest(doc=doc):
                with open(empty, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(doc, f)
                res = self.run_cli(self.game_dir, "--manifest", empty, "--quiet")
                self.assertEqual(res.returncode, 2, res.stdout + res.stderr)
                self.assertIn(f"lists no files, but {self.game_dir} holds {len(SYNTHETIC_SIZES)}", res.stderr)
                self.assertNotIn("OK", res.stdout)

    def test_cli_clean_verification(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--jobs", "2")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(res.stdout.strip(), "OK")

    def test_cli_verify_subcommand_syntax(self) -> None:
        res = self.run_cli("verify", self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(res.stdout.strip(), "OK")

    def test_cli_direct_syntax_rejects_extra_path(self) -> None:
        res = self.run_cli(self.game_dir, "extra", "--manifest", self.manifest_path)
        self.assertEqual(res.returncode, 2, res.stdout + res.stderr)
        self.assertIn("unrecognized arguments: extra", res.stderr)

    def test_cli_verbose_output(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--verbose")
        self.assertEqual(res.returncode, 0)
        self.assertIn("OK (", res.stdout)
        self.assertIn("GiB in", res.stdout)

    def test_cli_json_output(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--json")
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        data = json.loads(res.stdout)
        self.assertTrue(data["passed"])
        self.assertEqual(data["problem_count"], 0)
        self.assertEqual(data["matched_count"], len(SYNTHETIC_SIZES))

    def test_text_report_includes_unreadable_reason(self) -> None:
        hash_file = mv.sha256_file

        def read_file(path: str, *, expected_size: int | None = None) -> str:
            if os.path.basename(path) == "alpha.bin":
                raise OSError("Simulated permission denied")
            return hash_file(path, expected_size=expected_size)

        with open(os.path.join(self.game_dir, "delta.bin"), "r+b") as f:
            f.write(bytes([self.payloads["delta.bin"][0] ^ 1]))

        with patch.object(mv, "sha256_file", side_effect=read_file):
            report = mv.verify_manifest(self.game_dir, self.manifest_path)
        self.assertFalse(report.passed)
        self.assertEqual(report.to_dict()["unreadable"], [("alpha.bin", "Simulated permission denied")])

        output = io.StringIO()
        mv.print_report(report, out=output)
        self.assertEqual(output.getvalue().splitlines(), [
            "UNREADABLE alpha.bin: Simulated permission denied",
            "CHANGED   delta.bin",
            "2 problem(s)",
        ])

    def test_cli_tampered_returns_exit_code_1(self) -> None:
        delta_p = os.path.join(self.game_dir, "delta.bin")
        with open(delta_p, "r+b") as f:
            f.write(b"tampered")
        res = self.run_cli(self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res.returncode, 1)
        self.assertIn("CHANGED   delta.bin", res.stdout)
        self.assertIn("1 problem(s)", res.stdout)

    def test_cli_strict_with_untracked_fails(self) -> None:
        extra_p = os.path.join(self.game_dir, "new_file.dat")
        with open(extra_p, "wb") as f:
            f.write(b"extra")

        res_lax = self.run_cli(self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res_lax.returncode, 0)
        self.assertIn("NEW       new_file.dat", res_lax.stdout)
        self.assertIn("OK", res_lax.stdout)

        res_strict = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--strict")
        self.assertEqual(res_strict.returncode, 1)
        self.assertIn("NEW       new_file.dat", res_strict.stdout)
        self.assertIn("1 problem(s)", res_strict.stdout)

        res_json = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--strict", "--json")
        self.assertEqual(res_json.returncode, 1)
        self.assertEqual(json.loads(res_json.stdout)["problem_count"], 1)

    def test_cli_strict_ignores_only_selected_in_tree_manifest(self) -> None:
        in_tree = self.write_manifest(os.path.join(self.game_dir, "nested", "manifest.json"), self.payloads)
        args = (self.game_dir, "--manifest", in_tree, "--strict", "--json")
        res = self.run_cli(*args)
        self.assertEqual(res.returncode, 0, res.stdout + res.stderr)
        self.assertEqual(json.loads(res.stdout)["untracked"], [])

        self.write_manifest(os.path.join(self.game_dir, "manifest.json"), self.payloads)
        res = self.run_cli(*args)
        self.assertEqual(res.returncode, 1, res.stdout + res.stderr)
        self.assertEqual(json.loads(res.stdout)["untracked"], ["manifest.json"])

    def test_cli_missing_manifest_returns_exit_code_2(self) -> None:
        res = self.run_cli(self.game_dir, "--manifest", os.path.join(self.root, "none.json"))
        self.assertEqual(res.returncode, 2)
        self.assertIn("ERROR:", res.stderr)

    def test_cli_missing_target_dir_returns_exit_code_2(self) -> None:
        res = self.run_cli(os.path.join(self.root, "none_dir"), "--manifest", self.manifest_path)
        self.assertEqual(res.returncode, 2)
        self.assertIn("ERROR:", res.stderr)

    def test_cli_quiet_suppresses_progress(self) -> None:
        res_default = self.run_cli(self.game_dir, "--manifest", self.manifest_path)
        self.assertEqual(res_default.returncode, 0)
        # default outputs progress to stderr
        self.assertTrue(len(res_default.stderr) > 0)

        res_quiet = self.run_cli(self.game_dir, "--manifest", self.manifest_path, "--quiet")
        self.assertEqual(res_quiet.returncode, 0)
        self.assertEqual(res_quiet.stderr.strip(), "")


class TestStableHashing(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game = os.path.join(self.tmp.name, "game")
        os.mkdir(self.game)
        self.path = os.path.join(self.game, "sample.bin")
        self.manifest = os.path.join(self.tmp.name, "manifest.json")
        self.original = b"abcdefgh"
        self.write_file(self.original)

    def write_file(self, data: bytes) -> None:
        with open(self.path, "wb") as f:
            f.write(data)

    def write_manifest(self, size: int, data: bytes) -> None:
        entry = {"size": size, "sha256": hashlib.sha256(data).hexdigest()}
        with open(self.manifest, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"files": {"sample.bin": entry}}, f)

    def run_with_read_mutation(self, mutate, action):
        real_open = open
        changed = False

        @contextlib.contextmanager
        def hooked_open(*args, **kwargs):
            with real_open(*args, **kwargs) as f:
                if os.fspath(args[0]) != self.path:
                    yield f
                    return
                reader = Mock(wraps=f)

                def readinto(buf):
                    nonlocal changed
                    count = f.readinto(buf)
                    if count and not changed:
                        changed = True
                        mutate()
                    return count

                reader.readinto.side_effect = readinto
                yield reader

        with patch.object(mv, "open", hooked_open, create=True):
            return action()

    def test_resize_between_scheduling_and_hashing_cannot_pass(self) -> None:
        replacement = b"x"
        self.write_manifest(len(self.original), replacement)
        real_hash = mv.sha256_file

        def resize_before_open(path, *args, **kwargs):
            self.write_file(replacement)
            return real_hash(path, *args, **kwargs)

        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                self.write_file(self.original)
                with patch.object(mv, "sha256_file", side_effect=resize_before_open):
                    report = mv.verify_manifest(self.game, self.manifest, jobs=jobs)
                self.assertFalse(report.passed)
                self.assertEqual(report.problems, {"sample.bin": "UNREADABLE"})
                self.assertIn("size changed before hashing", report.unreadable[0][1])

    def test_truncation_during_read_cannot_pass_a_valid_manifest(self) -> None:
        self.write_manifest(len(self.original), self.original)
        for jobs in (1, 2):
            with self.subTest(jobs=jobs):
                self.write_file(self.original)
                report = self.run_with_read_mutation(
                    lambda: self.write_file(b"x"),
                    lambda: mv.verify_manifest(self.game, self.manifest, jobs=jobs),
                )
                self.assertFalse(report.passed)
                self.assertEqual(report.problems, {"sample.bin": "UNREADABLE"})
                self.assertIn("changed while hashing", report.unreadable[0][1])

    def test_growth_shrinkage_and_rewrite_during_hashing_are_rejected(self) -> None:
        for replacement in (b"abcdefghijklmnop", b"ab", b"abcdWXYZ"):
            with self.subTest(replacement=replacement):
                self.write_file(self.original)

                def change():
                    before = os.stat(self.path)
                    self.write_file(replacement)
                    os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))

                with self.assertRaisesRegex(OSError, "changed while hashing"):
                    self.run_with_read_mutation(change, lambda: mv.sha256_file(self.path, chunk_size=4))

    def test_replaced_path_during_hashing_is_rejected(self) -> None:
        def replace():
            replacement = os.path.join(self.tmp.name, "replacement.bin")
            with open(replacement, "wb") as f:
                f.write(b"replaced")
            try:
                os.replace(replacement, self.path)
            except PermissionError:
                self.skipTest("filesystem does not allow replacing an open file")

        with self.assertRaisesRegex(OSError, "changed while hashing"):
            self.run_with_read_mutation(replace, lambda: mv.sha256_file(self.path, chunk_size=4))

    def test_stable_files_and_custom_chunk_sizes_still_verify(self) -> None:
        for payload in (self.original, b""):
            self.write_file(payload)
            expected = hashlib.sha256(payload).hexdigest()
            for chunk_size in (1, 3, 32):
                with self.subTest(size=len(payload), chunk_size=chunk_size):
                    self.assertEqual(mv.sha256_file(self.path, chunk_size=chunk_size), expected)
            self.write_manifest(len(payload), payload)
            for jobs in (1, 2):
                with self.subTest(size=len(payload), jobs=jobs):
                    report = mv.verify_manifest(self.game, self.manifest, jobs=jobs)
                    self.assertTrue(report.passed)
                    self.assertEqual(report.matched, ["sample.bin"])
                    self.assertEqual(report.bytes_verified, len(payload))

    def test_nonempty_file_with_reported_size_zero_is_rejected(self) -> None:
        path = "/proc/self/cmdline"
        if not os.path.isfile(path):
            self.skipTest("requires a procfs file whose reported size is zero")
        self.assertEqual(os.path.getsize(path), 0)
        with open(path, "rb") as f:
            self.assertTrue(f.read(1))
        with self.assertRaisesRegex(OSError, "changed while hashing"):
            mv.sha256_file(path)


class TestVerifyCache(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.game = os.path.join(self.tmp.name, "game")
        os.makedirs(self.game)
        self.data = {"a.bin": make_payload(4096, "a"), "b.bin": make_payload(512, "b")}
        for rel, data in self.data.items():
            self.write(rel, data)
        self.manifest = os.path.join(self.tmp.name, "manifest.json")
        self.write_manifest(self.data)
        self.cache = os.path.join(self.tmp.name, "cache.json")

    def write_manifest(self, files: dict[str, bytes]) -> None:
        with open(self.manifest, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"files": {r: {"size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for r, d in files.items()}}, f)

    def write(self, rel: str, data: bytes, age_s: int = 60) -> None:
        path = os.path.join(self.game, rel)
        with open(path, "wb") as f:
            f.write(data)
        os.utime(path, (time.time() - age_s,) * 2)  # older than the racy window

    def verify(self, **kw) -> mv.ManifestReport:
        return mv.verify_manifest(self.game, self.manifest, cache_path=self.cache, **kw)

    def cached_names(self) -> set[str]:
        with open(self.cache, encoding="utf-8") as f:
            return {os.path.basename(k) for k in json.load(f)["files"]}

    def test_second_run_hashes_nothing(self) -> None:
        for jobs in (1, 4):
            with self.subTest(jobs=jobs):
                self.assertEqual(self.verify(jobs=jobs).passed, True)
                with patch.object(mv, "sha256_file", side_effect=AssertionError("hashed again")):
                    report = self.verify(jobs=jobs)
                self.assertTrue(report.passed)
                self.assertEqual((report.cached, report.matched), (2, ["a.bin", "b.bin"]))

    def test_rewrite_with_old_mtime_is_caught(self) -> None:
        for jobs in (1, 4):
            with self.subTest(jobs=jobs):
                self.write("a.bin", self.data["a.bin"])
                self.verify(jobs=jobs)
                path = os.path.join(self.game, "a.bin")
                before = os.stat(path)
                time.sleep(0.05)
                with open(path, "r+b") as f:  # same size, one byte off, old mtime put back
                    f.write(b"\xff")
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                report = self.verify(jobs=jobs)
                self.assertEqual((report.hash_mismatch, report.cached), (["a.bin"], 1))

    def test_changed_manifest_hash_misses(self) -> None:
        self.verify()
        self.write_manifest({**self.data, "a.bin": b"y" * 4096})
        report = self.verify()
        self.assertEqual((report.hash_mismatch, report.cached), (["a.bin"], 1))

    def test_what_never_gets_cached(self) -> None:
        self.write("a.bin", b"x" * 4096)  # bad file
        self.verify()
        self.assertEqual(self.cached_names(), {"b.bin"})
        self.write("a.bin", self.data["a.bin"], age_s=0)  # too new
        self.verify()
        self.assertEqual(self.cached_names(), {"b.bin"})

        os.unlink(self.cache)
        self.write("a.bin", self.data["a.bin"])
        real_hash = mv.sha256_file

        def touch(path: str, **kw) -> str:  # stamp moves while hashing
            digest = real_hash(path, **kw)
            os.utime(path, (time.time() - 120,) * 2)
            return digest

        with patch.object(mv, "sha256_file", side_effect=touch):
            self.assertTrue(self.verify().passed)
        self.assertFalse(os.path.exists(self.cache))

    def test_junk_cache_is_ignored(self) -> None:
        for text in ("not json", "[]", '{"version": 999, "files": {}}', '{"version": 1, "files": {"k": 3}}'):
            with self.subTest(text=text):
                with open(self.cache, "w", encoding="utf-8", newline="\n") as f:
                    f.write(text)
                report = self.verify()
                self.assertEqual((report.passed, report.cached), (True, 0))

    def test_cache_inside_target_and_strict(self) -> None:
        self.cache = os.path.join(self.game, "cache.json")
        for _ in range(2):  # cold and warm
            self.assertEqual(self.verify(strict=True).untracked, [])
        self.write("extra.bin", b"new")
        report = self.verify(strict=True)
        self.assertEqual((report.passed, report.untracked), (False, ["extra.bin"]))

    def test_stale_entries_get_pruned(self) -> None:
        self.verify()
        self.write_manifest({"a.bin": self.data["a.bin"]})
        self.verify()
        self.assertEqual(self.cached_names(), {"a.bin"})

    def test_cli_cache_flag(self) -> None:
        cmd = [sys.executable, "-I", TOOL, self.game, "--manifest", self.manifest, "--cache", self.cache, "--json"]
        subprocess.run(cmd, capture_output=True, check=True)
        doc = json.loads(subprocess.run(cmd, capture_output=True, check=True, text=True).stdout)
        self.assertEqual((doc["cached_count"], doc["bytes_verified"]), (2, 0))
        with open(self.cache, "rb") as f:
            self.assertNotIn(b"\r\n", f.read())


if __name__ == "__main__":
    unittest.main()
