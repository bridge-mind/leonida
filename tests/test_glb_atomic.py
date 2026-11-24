"""GLB export file ownership and overlapping-write regressions; no game data.

    python -I tests/test_glb_atomic.py -v
"""
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import glb_writer


class GlbAtomicTests(unittest.TestCase):
    @staticmethod
    def primitives(size=1):
        return glb_writer.parse_obj(
            f"v 0 0 0\nv {size} 0 0\nv 0 {size} 0\nf 1 2 3\n",
            is_path=False,
        )

    def test_success_preserves_existing_sibling_and_output_bytes(self):
        prims = self.primitives()
        expected = glb_writer.build_glb(prims)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            sibling = Path(str(target) + ".tmp")
            sibling.write_bytes(b"unrelated file")
            target.write_bytes(b"previous export")

            stats = glb_writer.write_glb(target, prims)

            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(stats["bytes"], len(expected))
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_failed_replace_preserves_destination_and_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            sibling = Path(str(target) + ".tmp")
            target.write_bytes(b"previous export")
            sibling.write_bytes(b"unrelated file")
            with mock.patch.object(glb_writer.os, "replace", side_effect=PermissionError("busy")):
                with self.assertRaisesRegex(PermissionError, "busy"):
                    glb_writer.write_glb(target, self.primitives())

            self.assertEqual(target.read_bytes(), b"previous export")
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_overlapping_writes_each_publish_their_own_complete_bytes(self):
        first, second = self.primitives(1), self.primitives(2)
        first_bytes, second_bytes = map(glb_writer.build_glb, (first, second))
        replace = os.replace
        staging_paths = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"

            def interleave(source, destination):
                staging_paths.append(os.fspath(source))
                self.assertEqual(Path(source).parent.resolve(), target.parent.resolve())
                if len(staging_paths) == 1:
                    # The first writer has closed its file but has not published it.
                    glb_writer.write_glb(target, second)
                    self.assertEqual(target.read_bytes(), second_bytes)
                replace(source, destination)

            with mock.patch.object(glb_writer.os, "replace", side_effect=interleave):
                glb_writer.write_glb(target, first)

            self.assertEqual(target.read_bytes(), first_bytes)
            self.assertEqual(len(set(staging_paths)), 2)
            self.assertEqual(os.listdir(directory), [target.name])

    def test_partial_write_failure_cleans_only_its_own_staging_file(self):
        create_temp = tempfile.NamedTemporaryFile

        def failing_temp(*args, **kwargs):
            handle = create_temp(*args, **kwargs)
            write = handle.write

            def fail_write(payload):
                write(payload[:12])
                raise OSError("simulated full disk")

            handle.write = fail_write
            return handle

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            sibling = Path(str(target) + ".tmp")
            target.write_bytes(b"previous export")
            sibling.write_bytes(b"unrelated file")
            with mock.patch.object(glb_writer.tempfile, "NamedTemporaryFile", side_effect=failing_temp):
                with self.assertRaisesRegex(OSError, "simulated full disk"):
                    glb_writer.write_glb(target, self.primitives())

            self.assertEqual(target.read_bytes(), b"previous export")
            self.assertEqual(sibling.read_bytes(), b"unrelated file")
            self.assertEqual(set(os.listdir(directory)), {target.name, sibling.name})

    def test_transient_sharing_violation_is_retried(self):
        prims = self.primitives()
        replace = os.replace
        calls = []

        def busy_twice(source, destination):
            calls.append(source)
            if len(calls) <= 2:
                raise PermissionError(13, "sharing violation")
            replace(source, destination)

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            target.write_bytes(b"previous export")
            with mock.patch.object(glb_writer, "_REPLACE_RETRY", True), \
                    mock.patch.object(glb_writer.time, "sleep") as sleep, \
                    mock.patch.object(glb_writer.os, "replace", side_effect=busy_twice):
                glb_writer.write_glb(target, prims)

            self.assertEqual(target.read_bytes(), glb_writer.build_glb(prims))
            self.assertEqual(len(calls), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertEqual(os.listdir(directory), [target.name])

    def test_lasting_sharing_violation_still_raises_after_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            target.write_bytes(b"previous export")
            replace = mock.Mock(side_effect=PermissionError("busy"))
            with mock.patch.object(glb_writer, "_REPLACE_RETRY", True), \
                    mock.patch.object(glb_writer.time, "sleep"), \
                    mock.patch.object(glb_writer.os, "replace", replace):
                with self.assertRaisesRegex(PermissionError, "busy"):
                    glb_writer.write_glb(target, self.primitives())

            self.assertEqual(replace.call_count, len(glb_writer._REPLACE_DELAYS) + 1)
            self.assertEqual(target.read_bytes(), b"previous export")
            self.assertEqual(os.listdir(directory), [target.name])

    def test_other_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            replace = mock.Mock(side_effect=FileNotFoundError("gone"))
            with mock.patch.object(glb_writer, "_REPLACE_RETRY", True), \
                    mock.patch.object(glb_writer.time, "sleep") as sleep, \
                    mock.patch.object(glb_writer.os, "replace", replace):
                with self.assertRaises(FileNotFoundError):
                    glb_writer.write_glb(target, self.primitives())
            self.assertEqual(replace.call_count, 1)
            sleep.assert_not_called()

    @unittest.skipUnless(os.name == "nt", "sharing violations on replace are Windows-only")
    def test_threads_writing_one_file_all_publish(self):
        # Before the retry, 8 threads x 40 writes to one path failed 43-152 of 320
        # calls with PermissionError on Windows 11 (#425 review).
        payloads = [self.primitives(size) for size in range(1, 9)]
        allowed = {glb_writer.build_glb(p) for p in payloads}
        errors = []
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"

            def worker(prims):
                for _ in range(40):
                    try:
                        glb_writer.write_glb(target, prims)
                    except OSError as exc:
                        errors.append(exc)

            threads = [threading.Thread(target=worker, args=(p,)) for p in payloads]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            self.assertIn(target.read_bytes(), allowed)
            self.assertEqual(os.listdir(directory), [target.name])

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits")
    def test_published_glb_keeps_permission_bits(self):
        # The staging file is created 0600; the published GLB must not lose group/other read.
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            old = os.umask(0o022)
            try:
                glb_writer.write_glb(target, self.primitives())
                self.assertEqual(target.stat().st_mode & 0o777, 0o644)
                os.chmod(target, 0o640)
                glb_writer.write_glb(target, self.primitives(2))
                self.assertEqual(target.stat().st_mode & 0o777, 0o640)
            finally:
                os.umask(old)

    def test_concurrent_new_files_leave_the_umask_alone(self):
        # A new file's mode comes from a probe file (file_mode.published_mode), never from setting the umask to 0
        # and back, which could leave a threaded process at umask 0.
        import file_mode
        modes = []
        with tempfile.TemporaryDirectory() as directory,                 mock.patch.object(file_mode.os, "umask", side_effect=AssertionError("umask touched")) as umask:
            threads = [threading.Thread(target=lambda i=i: modes.append(
                file_mode.published_mode(os.path.join(directory, f"new{i}.glb")))) for i in range(2)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        umask.assert_not_called()
        self.assertEqual(len(modes), 2)
        self.assertEqual(modes[0], modes[1])

    def test_published_glb_is_writable(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "mesh.glb"
            glb_writer.write_glb(target, self.primitives())
            self.assertTrue(os.access(target, os.W_OK))
            glb_writer.write_glb(target, self.primitives(2))
            self.assertEqual(target.read_bytes(), glb_writer.build_glb(self.primitives(2)))

    def test_creates_missing_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "mesh.glb"
            prims = self.primitives()
            glb_writer.write_glb(target, prims)
            self.assertEqual(target.read_bytes(), glb_writer.build_glb(prims))
            self.assertEqual(os.listdir(target.parent), [target.name])


if __name__ == "__main__":
    unittest.main()
