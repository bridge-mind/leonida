"""make_game.py: translate() terminates and reaps shard processes on Ctrl+C or launch failure.

python -I tests/test_make_game_translate_cleanup.py
"""

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import make_game  # noqa: E402


class FakeProc:
    def __init__(
        self, *, running=True, wait_side_effect=None, poll_side_effect=None,
        timeout_on_terminate=False
    ):
        self.running = running
        self.wait_side_effect = wait_side_effect
        self.poll_side_effect = poll_side_effect
        self.timeout_on_terminate = timeout_on_terminate
        self.terminated = False
        self.killed = False
        self.waits = 0

    def poll(self):
        if self.poll_side_effect is not None:
            exc = self.poll_side_effect
            self.poll_side_effect = None
            raise exc
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        if not self.timeout_on_terminate:
            self.running = False

    def kill(self):
        self.killed = True
        self.running = False

    def wait(self, timeout=None):
        self.waits += 1
        if self.wait_side_effect is not None and self.waits == 1:
            raise self.wait_side_effect
        if self.running and self.timeout_on_terminate and not self.killed:
            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0)
        return 0


class TranslateCleanupTests(unittest.TestCase):
    def test_keyboard_interrupt_during_wait_terminates_and_reaps_all_running_shards(
        self,
    ):
        # translate() polls shards and sleeps; Ctrl+C must arrive from poll() (or sleep), not wait().
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            spirv = os.path.join(tmp, "spirv")
            os.makedirs(disasm)
            p0 = FakeProc(running=False)
            p1 = FakeProc(running=True, poll_side_effect=KeyboardInterrupt())
            p2 = FakeProc(running=True)
            procs = [p0, p1, p2]
            spawned = iter(procs)

            with (
                patch("make_game.log"),
                patch(
                    "make_game.subprocess.Popen",
                    side_effect=lambda *a, **kw: next(spawned),
                ),
                patch("make_game.py") as merge_call,
                patch("make_game.time.sleep"),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    make_game.translate(disasm, spirv, jobs=3)

            self.assertFalse(p0.terminated)  # already finished before the interrupt
            self.assertTrue(p1.terminated)
            self.assertTrue(p2.terminated)
            self.assertGreaterEqual(p1.waits, 1)
            self.assertGreaterEqual(p2.waits, 1)
            merge_call.assert_not_called()

    def test_mid_launch_oserror_terminates_already_started_shards(self):
        with tempfile.TemporaryDirectory() as tmp:
            disasm = os.path.join(tmp, "disasm")
            spirv = os.path.join(tmp, "spirv")
            os.makedirs(disasm)
            real_procs = []
            real_popen = subprocess.Popen

            def fake_popen(*_args, **_kwargs):
                if len(real_procs) == 2:
                    raise OSError("too many open files")
                p = real_popen([sys.executable, "-c", "import time; time.sleep(30)"])
                real_procs.append(p)
                return p

            with (
                patch("make_game.log"),
                patch("make_game.subprocess.Popen", side_effect=fake_popen),
                patch("make_game.py") as merge_call,
            ):
                with self.assertRaises(OSError):
                    make_game.translate(disasm, spirv, jobs=4)

            self.assertEqual(len(real_procs), 2)
            for p in real_procs:
                self.assertIsNotNone(
                    p.poll(), "shard process was left running after mid-launch failure"
                )
            merge_call.assert_not_called()

    def test_stop_procs_kills_unresponsive_process_after_timeout(self):
        stubborn = FakeProc(running=True, timeout_on_terminate=True)
        make_game._stop_procs([stubborn])
        self.assertTrue(stubborn.terminated)
        self.assertTrue(stubborn.killed)
        self.assertIsNotNone(stubborn.poll())


if __name__ == "__main__":
    unittest.main()
