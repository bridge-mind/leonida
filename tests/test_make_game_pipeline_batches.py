"""Pipeline batching and memory-crash recovery through the real make_game.pipelines orchestration.

The child is a small scripted checker speaking the production CLI/file protocol. Each invocation is a real
process, with synthetic index entries and controlled exits; no GPU, compiler, game data or packages are needed.
"""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import make_game  # noqa: E402


CHECKER = r'''
import json
import os
from pathlib import Path
import signal
import sys

spirv, cache, *args = sys.argv[1:]
spirv, cache = Path(spirv), Path(cache)
config = json.loads((spirv / "control.json").read_text(encoding="utf-8"))
names = sorted(json.loads((spirv / "index.json").read_text(encoding="utf-8")), key=lambda n: n.encode("utf-8"))
def option(flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default
start = int(option("--start", 0))
count = int(option("--count", len(names)))
selected = names[start:start + count]
skip_path = option("--skip-file")
skipped = Path(skip_path).read_text(encoding="utf-8").splitlines() if skip_path else []
active = [name for name in selected if name not in skipped]
inflight = option("--inflight-file")
def record(event):
    with (spirv / "calls.jsonl").open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(dict(event=event, pid=os.getpid(), selected=selected, active=active, args=args)) + "\n")
record("started")
if config.get("setup_error") or config.get("probe_setup_error") and count == 1 and option("--jobs") == "1":
    print("no Vulkan physical devices found", file=sys.stderr)
    sys.exit(config.get("setup_exit", 2))
if config.get("interrupt") or config.get("interrupt_probe") and count == 1 and option("--jobs") == "1":
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.ExitProcess(0xC000013A)
    else:
        os.kill(os.getpid(), signal.SIGTERM)
if config.get("crash_before_work"):
    sys.exit(70)
if config.get("probe_crash_before_work") and count == 1 and option("--jobs") == "1":
    report = config.get("probe_report", "missing")
    if inflight and report != "missing":
        Path(inflight).write_text("unrelated\tpipeline\n" if report == "other" else "",
                                  encoding="utf-8", newline="\n")
    sys.exit(70)
crasher = config.get("crasher")
if crasher in active or len(active) > config.get("max_batch", 1000000):
    if inflight:
        suspects = [crasher] if crasher in active else active[:config.get("suspects", 2)]
        Path(inflight).write_text("".join(n + "\tpipeline\n" for n in suspects), encoding="utf-8", newline="\n")
    sys.exit(70)
# This stand-in's cache contains completed names so the test can detect lost checkpoints and repeated work.
completed = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else []
cache.write_text(json.dumps(completed + active), encoding="utf-8", newline="\n")
record("completed")
if inflight:
    Path(inflight).write_text("", encoding="utf-8", newline="\n")
sys.exit(1 if skipped and set(selected).intersection(skipped) or config.get("invalid") in active else 0)
'''


class PipelineBatchesTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="pipeline batches ")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.work = self.root / "work"
        self.spirv = self.work / "shaders" / "spirv"
        self.spirv.mkdir(parents=True)
        self.out = self.root / "out"
        self.names = [f"m_{i:02d}" for i in range(7)]
        self.write_index(self.names)
        self.child = self.root / "checker.py"
        self.child.write_text(CHECKER, encoding="utf-8", newline="\n")
        self.commands = []

    def write_index(self, names):
        (self.spirv / "index.json").write_text(json.dumps({n: {} for n in reversed(names)}),
                                                encoding="utf-8", newline="\n")

    def invoke(self, control=None, range_support=True):
        (self.spirv / "control.json").write_text(json.dumps(control or {}), encoding="utf-8", newline="\n")
        def run(args, hint="", check=True, **kwargs):
            args = [str(a) for a in args]
            self.commands.append(args)
            done = subprocess.run([sys.executable, "-I", str(self.child), *args[1:]],
                                  capture_output=True, text=True, timeout=15)
            if check and done.returncode:
                raise SystemExit(1)
            return done.returncode
        def supports(exe, *options):
            return range_support or "--start" not in options
        logs, err = [], io.StringIO()
        # create=True keeps the regression runnable on the old implementation, which lacks this constant.
        with mock.patch.object(make_game, "pipeline_check_exe", return_value="synthetic-checker"), \
                mock.patch.object(make_game, "checker_supports", side_effect=supports), \
                mock.patch.object(make_game, "PIPELINE_BATCH_SIZE", 3, create=True), \
                mock.patch.object(make_game, "run", side_effect=run), \
                mock.patch.object(make_game, "log", side_effect=logs.append), contextlib.redirect_stderr(err):
            try:
                make_game.pipelines(str(self.work), str(self.out), jobs=2)
            except SystemExit as exc:
                return exc.code, logs, err.getvalue()
        return 0, logs, err.getvalue()

    def calls(self, event=None):
        path = self.spirv / "calls.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
        return [row for row in rows if event is None or row["event"] == event]

    def test_bounded_processes_cover_the_sorted_index_and_tail(self):
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        completed = self.calls("completed")
        self.assertEqual([r["selected"] for r in completed], [self.names[:3], self.names[3:6], self.names[6:]])
        self.assertEqual(len({r["pid"] for r in completed}), 3)
        self.assertEqual(json.loads((self.out / "pipeline_cache.bin").read_text(encoding="utf-8")), self.names)
        self.assertTrue(all(r["args"][r["args"].index("--jobs") + 1] == "2" for r in completed))

    def test_collective_memory_crash_splits_without_skipping_healthy_modules(self):
        code, logs, err = self.invoke({"max_batch": 2})
        self.assertEqual(code, 0, err)
        # Diagnostic one-module probes are also successful; the final bounded work must still cover every name.
        checked = {n for row in self.calls("completed") for n in row["active"]}
        self.assertEqual(checked, set(self.names))
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertTrue(any("smaller" in line for line in logs), logs)

    def test_a_real_crasher_stays_failed_when_later_batches_pass(self):
        code, _, err = self.invoke({"crasher": "m_01"})
        self.assertNotEqual(code, 0)
        self.assertEqual((self.out / "pipeline_crashes.txt").read_text(encoding="utf-8"), "m_01\n")
        self.assertIn("m_01", err)
        checked = {n for row in self.calls("completed") for n in row["active"]}
        self.assertEqual(checked, set(self.names) - {"m_01"})

    def test_one_inflight_module_is_not_blacklisted_when_it_survives_alone(self):
        self.write_index(self.names[:2])   # small corpora need the same protection as multi-batch runs
        code, _, err = self.invoke({"max_batch": 1, "suspects": 1})
        self.assertEqual(code, 0, err)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertEqual({n for row in self.calls("completed") for n in row["active"]}, set(self.names[:2]))

    def test_interrupting_a_diagnostic_probe_stops_before_more_batches(self):
        code, _, _ = self.invoke({"max_batch": 2, "interrupt_probe": True})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)   # the failed batch and the interrupted diagnostic probe
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def test_a_validation_failure_does_not_hide_the_remaining_batches(self):
        code, _, _ = self.invoke({"invalid": "m_01"})
        self.assertNotEqual(code, 0)
        self.assertEqual([n for row in self.calls("completed") for n in row["active"]], self.names)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def test_setup_errors_and_interrupts_stop_immediately(self):
        for control in ({"setup_error": True}, {"interrupt": True}):
            with self.subTest(control=control):
                trace = self.spirv / "calls.jsonl"
                if trace.exists():
                    trace.unlink()
                self.commands.clear()
                code, _, _ = self.invoke(control)
                self.assertNotEqual(code, 0)
                self.assertEqual(len(self.calls()), 1)

    def test_native_setup_exit_one_stops_instead_of_checking_every_range(self):
        code, _, _ = self.invoke({"setup_error": True, "setup_exit": 1})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 1)

    def test_native_setup_exit_one_in_a_probe_stops_before_splitting(self):
        code, _, _ = self.invoke({"max_batch": 2, "probe_setup_error": True, "setup_exit": 1})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())

    def assert_inconclusive_probe_stops(self, report):
        code, logs, err = self.invoke({"max_batch": 2, "probe_crash_before_work": True,
                                      "probe_report": report})
        self.assertNotEqual(code, 0)
        self.assertEqual(len(self.calls()), 2)   # failed batch, then the first inconclusive probe
        self.assertEqual(self.calls()[-1]["selected"], [self.names[0]])
        self.assertFalse((self.out / "pipeline_crashes.txt").exists())
        self.assertFalse(any("took vk_pipeline_check down" in line for line in logs), logs)
        self.assertIn("inconclusive", err)

    def test_abnormal_probe_without_a_report_does_not_blacklist_a_module(self):
        self.assert_inconclusive_probe_stops("missing")

    def test_abnormal_probe_with_an_empty_report_does_not_blacklist_a_module(self):
        self.assert_inconclusive_probe_stops("empty")

    def test_abnormal_probe_reporting_another_module_does_not_blacklist_the_selected_one(self):
        self.assert_inconclusive_probe_stops("other")

    def test_a_crash_before_work_cannot_retry_forever_or_claim_success(self):
        code, _, err = self.invoke({"crash_before_work": True})
        self.assertNotEqual(code, 0)
        self.assertLessEqual(len(self.calls()), 4)
        self.assertTrue(err)

    def test_an_older_checker_gets_no_unsupported_range_arguments(self):
        code, _, err = self.invoke(range_support=False)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.commands), 1)
        self.assertNotIn("--start", self.commands[0])
        self.assertNotIn("--count", self.commands[0])

    def test_small_indices_keep_the_single_run(self):
        self.write_index(self.names[:2])
        code, _, err = self.invoke()
        self.assertEqual(code, 0, err)
        self.assertEqual(len(self.commands), 1)


if __name__ == "__main__":
    unittest.main()
