"""Concurrent-output regression tests for tools/world_build.py (no game data).

A world build stages chunks in the output directory, publishes them one at a time,
then prunes stale chunks.  Two builds on one output must not interleave: the
second one must fail while the first holds the lock, and the next build must
work again after the lock is released.
"""
from contextlib import contextmanager
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOOLS = os.path.join(ROOT, "tools")
for path in (TOOLS, HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

import test_world_build as twb  # noqa: E402
import world_build as wb  # noqa: E402

CHILD_HOLD_LOCK = """
import sys, time
sys.path.insert(0, sys.argv[1])
import world_build as wb
with wb._exclusive_output(sys.argv[2]):
    with open(sys.argv[3], 'w', encoding='ascii') as fh:
        fh.write('ready')
    time.sleep(30)
"""


@contextmanager
def _lock_held_by_child(tmp, out):
    """Hold ``_exclusive_output(out)`` in a child process for the body of the ``with``."""
    ready = os.path.join(tmp, "ready")
    proc = subprocess.Popen(
        [sys.executable, "-I", "-c", CHILD_HOLD_LOCK, TOOLS, out, ready],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.time() + 10
        while not os.path.isfile(ready):
            if proc.poll() is not None:
                raise AssertionError("lock holder exited before becoming ready")
            if time.time() >= deadline:
                raise AssertionError("lock holder never became ready")
            time.sleep(0.05)
        yield
    finally:
        proc.terminate()
        try:
            proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate(timeout=5)


def _short_path(path):
    """The 8.3 spelling of an existing ``path``, or None when the volume has none."""
    import ctypes
    get_short = ctypes.windll.kernel32.GetShortPathNameW
    get_short.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    get_short.restype = ctypes.c_uint
    buf = ctypes.create_unicode_buffer(1024)
    if not get_short(path, buf, len(buf)):
        return None
    short = buf.value
    return None if os.path.normcase(short) == os.path.normcase(path) else short


class WorldBuildOutputLockTests(unittest.TestCase):
    def test_second_process_refuses_while_first_holds_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = os.path.join(tmp, "assets")
            out = os.path.join(tmp, "world")
            twb._write_synthetic_assets(assets, origin_overlap=False)
            with _lock_held_by_child(tmp, out):
                with self.assertRaisesRegex(RuntimeError, "another world_build is using"):
                    wb.build_world(assets, out, textures=False)
                self.assertFalse(os.path.exists(os.path.join(out, "chunks")))

    def _assert_other_spelling_refused(self, tmp, parent, other_parent):
        # The child locks <parent>/World; a build through <other_parent>/World is the same folder
        # and must be refused, both before the folder exists (two first runs) and after.
        assets = os.path.join(tmp, "assets")
        twb._write_synthetic_assets(assets, origin_overlap=False)
        out = os.path.join(parent, "World")
        other = os.path.join(other_parent, "World")
        with _lock_held_by_child(tmp, out):
            for state in ("missing", "existing"):
                if state == "existing":
                    os.makedirs(out, exist_ok=True)
                self.assertTrue(state == "missing" or os.path.samefile(out, other))
                with self.subTest(output=state):
                    with self.assertRaisesRegex(RuntimeError, "another world_build is using"):
                        wb.build_world(assets, other, textures=False)
                    self.assertFalse(os.path.exists(os.path.join(out, "world.json")))
                    self.assertFalse(os.path.exists(os.path.join(out, "chunks")))

    def test_short_name_spelling_of_a_held_output_is_refused(self):
        if os.name != "nt":
            self.skipTest("8.3 short names are Windows only")
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "LongOutputFolderName")
            os.makedirs(parent)
            short = _short_path(parent)
            if short is None:
                self.skipTest("8.3 short names are unavailable on this volume")
            self._assert_other_spelling_refused(tmp, parent, short)

    def test_linked_parent_spelling_of_a_held_output_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = os.path.join(tmp, "real")
            os.makedirs(parent)
            link = os.path.join(tmp, "link")
            if os.name == "nt":
                import _winapi
                _winapi.CreateJunction(parent, link)
            else:
                try:
                    os.symlink(parent, link, target_is_directory=True)
                except (OSError, NotImplementedError):
                    self.skipTest("symbolic links are not available here")
            try:
                self._assert_other_spelling_refused(tmp, parent, link)
            finally:
                if os.name == "nt":
                    os.rmdir(link)  # removes the junction, not its target

    def test_build_world_holds_the_output_lock_and_releases_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = os.path.join(tmp, "assets")
            out = os.path.join(tmp, "world")
            twb._write_synthetic_assets(assets, origin_overlap=False)
            calls = []

            @contextmanager
            def fake_lock(path):
                calls.append(path)
                yield

            with mock.patch.object(wb, "_exclusive_output", fake_lock):
                wb.build_world(assets, out, textures=False)
                wb.build_world(assets, out, textures=False)
            self.assertEqual(calls, [os.path.abspath(out)] * 2)


if __name__ == "__main__":
    unittest.main()
