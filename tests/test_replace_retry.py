"""tools/replace_retry.py: os.replace that retries the transient Windows PermissionErrors, and its call sites.

    python -I tests/test_replace_retry.py -v

Synthetic files only. Tests that exercise native file sharing are Windows-only.
"""
import contextlib
import errno
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import replace_retry  # noqa: E402
import fssm_unpack  # noqa: E402
from test_fssm_unpack import make_dxbc, make_fssm, make_xbch, make_xdat  # noqa: E402

REAL_REPLACE = os.replace


def win_error(code):
    """A PermissionError as Windows raises it: errno EACCES with the Win32 code in .winerror."""
    e = PermissionError(errno.EACCES, "The process cannot access the file because it is being used by another process")
    e.winerror = code    # a plain attribute on Linux, where OSError has no such field unless given five arguments
    return e


def flaky(failures, error):
    """An os.replace stand-in that raises `error` `failures` times (None: always), then renames for real."""
    calls = []

    def fake(src, dst):
        calls.append((src, dst))
        if failures is None or len(calls) <= failures:
            raise error
        return REAL_REPLACE(src, dst)
    fake.calls = calls
    return fake


class ReplaceRetryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src = os.path.join(self.tmp.name, "a.tmp")
        self.dst = os.path.join(self.tmp.name, "a.json")
        with open(self.src, "w") as f:
            f.write("new")
        with open(self.dst, "w") as f:
            f.write("old")
        patcher = mock.patch.object(replace_retry.time, "sleep")
        self.sleep = patcher.start()
        self.addCleanup(patcher.stop)

    def read_dst(self):
        with open(self.dst) as f:
            return f.read()

    def test_plain_success_renames_once(self):
        fake = flaky(0, None)
        with mock.patch.object(replace_retry.os, "replace", fake):
            replace_retry.replace(self.src, self.dst)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(self.read_dst(), "new")
        self.sleep.assert_not_called()

    def test_sharing_violation_then_success(self):
        for code in (32, 5):
            with self.subTest(winerror=code):
                with open(self.src, "w") as f:
                    f.write("new")
                fake = flaky(2, win_error(code))
                with mock.patch.object(replace_retry.os, "replace", fake):
                    replace_retry.replace(self.src, self.dst)
                self.assertEqual(len(fake.calls), 3)
                self.assertEqual(self.read_dst(), "new")

    def test_persistent_failure_is_reraised_after_the_time_is_up(self):
        error = win_error(32)
        fake = flaky(None, error)
        # Fake clock: each sleep advances it, so the 5 s budget runs out without waiting.
        now = [0.0]
        self.sleep.side_effect = lambda s: now.__setitem__(0, now[0] + s)
        with mock.patch.object(replace_retry.os, "replace", fake), \
                mock.patch.object(replace_retry.time, "monotonic", lambda: now[0]):
            with self.assertRaises(PermissionError) as cm:
                replace_retry.replace(self.src, self.dst)
        self.assertIs(cm.exception, error)
        self.assertGreater(len(fake.calls), 2)
        self.assertLess(len(fake.calls), 100)       # bounded
        self.assertLessEqual(now[0], replace_retry.RETRY_SECONDS + 1e-9)
        self.assertEqual(self.read_dst(), "old")    # nothing half-replaced

    def test_permission_error_without_a_transient_code_is_not_retried(self):
        for error in (PermissionError(errno.EACCES, "denied"), win_error(1314), win_error(2)):
            with self.subTest(error=repr(error)):
                fake = flaky(None, error)
                with mock.patch.object(replace_retry.os, "replace", fake):
                    with self.assertRaises(PermissionError):
                        replace_retry.replace(self.src, self.dst)
                self.assertEqual(len(fake.calls), 1)

    def test_other_errors_are_not_retried(self):
        for error in (FileNotFoundError(errno.ENOENT, "gone"), OSError(errno.EXDEV, "cross-device")):
            with self.subTest(error=repr(error)):
                fake = flaky(None, error)
                with mock.patch.object(replace_retry.os, "replace", fake):
                    with self.assertRaises(type(error)):
                        replace_retry.replace(self.src, self.dst)
                self.assertEqual(len(fake.calls), 1)

    def check_native_lock(self, locked_path):
        # A normal Python file handle on Windows allows readers but denies deletion.
        with open(locked_path, "rb") as held:
            with self.assertRaises(PermissionError) as caught:
                REAL_REPLACE(self.src, self.dst)
            self.assertIn(caught.exception.winerror, (5, 32))
            self.assertEqual(self.read_dst(), "old")

            # Release the handle after the retry helper hits the real sharing error.
            self.sleep.side_effect = lambda _: held.close()
            replace_retry.replace(self.src, self.dst)

        self.sleep.assert_called()
        self.assertEqual(self.read_dst(), "new")
        self.assertFalse(os.path.exists(self.src))

    @unittest.skipUnless(os.name == "nt", "requires Windows file sharing")
    def test_real_source_lock(self):
        self.check_native_lock(self.src)

    @unittest.skipUnless(os.name == "nt", "requires Windows file sharing")
    def test_real_destination_lock(self):
        self.check_native_lock(self.dst)

    def test_real_replace_overwrites(self):
        replace_retry.replace(self.src, self.dst)
        self.assertEqual(self.read_dst(), "new")
        self.assertFalse(os.path.exists(self.src))


class CallSiteTests(unittest.TestCase):
    def test_fssm_unpack_survives_a_transient_rename_failure(self):
        # The crash seen on Windows: os.replace(fssm_index.json.tmp, fssm_index.json) raising WinError 32.
        with tempfile.TemporaryDirectory() as src:
            with open(os.path.join(src, "good.dxbc"), "wb") as f:
                f.write(make_dxbc([(b"DXIL", make_xdat({0x0D: make_xbch(make_fssm())}))]))
            fake = flaky(2, win_error(32))
            with mock.patch.object(replace_retry.os, "replace", fake), \
                    mock.patch.object(replace_retry.time, "sleep"), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fssm_unpack.main([src]), 0)
            self.assertEqual(len(fake.calls), 3)
            with open(os.path.join(src, "fssm_index.json"), encoding="utf-8") as f:
                self.assertEqual([r["name"] for r in json.load(f)], ["good"])
            self.assertFalse(os.path.exists(os.path.join(src, "fssm_index.json.tmp")))

    def test_rdna2spirv_write_index_survives_a_transient_rename_failure(self):
        import rdna2spirv
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "index.json")
            fake = flaky(1, win_error(5))
            with mock.patch.object(replace_retry.os, "replace", fake), mock.patch.object(replace_retry.time, "sleep"):
                rdna2spirv.write_index(path, [{"name": "x"}])
            self.assertEqual(len(fake.calls), 2)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(json.load(f), [{"name": "x"}])


if __name__ == "__main__":
    unittest.main()
