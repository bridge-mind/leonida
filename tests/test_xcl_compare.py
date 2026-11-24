"""tools/xcl_compare.py: parsing of the test title's lines and the report it builds.

    python -I tests/test_xcl_compare.py

Nothing here starts XCL or gta6rt, and no game files are read. The run tests start a small Python script that
prints the test title's line format in place of a title.
"""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
import xcl_compare as xc  # noqa: E402


class ParseResult(unittest.TestCase):
    KNOWN = ["kernelx: GetConsoleType / XSystemGetDeviceType", "files: T:\\ title scratch storage", "argv"]

    def test_pass(self):
        self.assertEqual(xc.parse_result("[PASS] argv", None, self.KNOWN), ("PASS", "argv", ""))

    def test_fail_with_colon_in_name_uses_the_running_check(self):
        name = "kernelx: GetConsoleType / XSystemGetDeviceType"
        got = xc.parse_result("[FAIL] " + name + ": GetConsoleType = 0", name, self.KNOWN)
        self.assertEqual(got, ("FAIL", name, "GetConsoleType = 0"))

    def test_fail_with_colon_in_name_uses_known_names(self):
        name = "kernelx: GetConsoleType / XSystemGetDeviceType"
        got = xc.parse_result("[FAIL] " + name + ": GetConsoleType = 0", None, self.KNOWN)
        self.assertEqual(got, ("FAIL", name, "GetConsoleType = 0"))

    def test_fail_unknown_name_splits_at_first_colon(self):
        self.assertEqual(xc.parse_result("[FAIL] new check: why", None, []), ("FAIL", "new check", "why"))

    def test_skip(self):
        self.assertEqual(xc.parse_result("[SKIP] argv", None, self.KNOWN), ("SKIP", "argv", ""))

    def test_other_lines(self):
        for line in ("[RUN] argv", "[rt] unimplemented: kernelx.dll!GetConsoleType", "xcl_testtitle: start", ""):
            self.assertIsNone(xc.parse_result(line, None, self.KNOWN))


class Report(unittest.TestCase):
    def runs(self):
        x, l = xc.Run("XCL"), xc.Run("Leonida")
        for n in ("a", "b", "c"):
            x.set(n, status="PASS")
        l.set("a", status="PASS")
        l.set("b", status="FAIL", reason="null", unimplemented=["kernelx.dll!XMemAlloc"])
        l.set("c", status="HANG", reason="no output for 15 s", stack=["main: XTaskQueueTerminate"])
        l.set("d", status="FAIL", reason="only here")
        return x, l

    def test_rows_follow_xcl_order_then_extra_checks(self):
        x, l = self.runs()
        r = xc.make_report(x, l, list(x.order), "gta6rt.exe", ".")
        self.assertEqual([row["check"] for row in r["checks"]], ["a", "b", "c", "d"])

    def test_counts_and_differences(self):
        x, l = self.runs()
        r = xc.make_report(x, l, list(x.order), "gta6rt.exe", ".")
        self.assertEqual(r["summary"]["xcl"]["PASS"], 3)
        self.assertEqual(r["summary"]["xcl"]["NOT RUN"], 1)
        self.assertEqual(r["summary"]["leonida"], {"PASS": 1, "FAIL": 2, "SKIP": 0, "HANG": 1, "NOT RUN": 0})
        self.assertEqual(r["differences"], 3)
        b = r["checks"][1]
        self.assertEqual((b["leonida"], b["unimplemented"]), ("FAIL", ["kernelx.dll!XMemAlloc"]))
        self.assertEqual(r["checks"][2]["stack"], ["main: XTaskQueueTerminate"])

    def test_same_status_different_reason_differs(self):
        x, l = xc.Run("XCL"), xc.Run("Leonida")
        x.set("a", status="FAIL", reason="E_NOTIMPL")
        l.set("a", status="FAIL", reason="null pointer")
        x.set("b", status="FAIL", reason="same")
        l.set("b", status="FAIL", reason="same")
        r = xc.make_report(x, l, list(x.order), "gta6rt.exe", ".")
        self.assertEqual([row["same"] for row in r["checks"]], [False, True])
        self.assertEqual(r["differences"], 1)


class Transcript(unittest.TestCase):
    """run_title streams the title's output to the transcript file and keeps none of it on the Run."""

    def test_output_goes_to_the_file(self):
        class Sink:
            def put(self, event):
                pass

        script = "print('[RUN] a'); print('x' * 256); print('[PASS] a')"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.txt")
            run = xc.Run("synthetic")
            hung = xc.run_title([sys.executable, "-I", "-c", script], dict(os.environ), tmp, run, ["a"], 15, None,
                                Sink(), [], path)
            with open(path, encoding="utf-8") as f:
                lines = f.read().split("\n")
        self.assertIsNone(hung)
        self.assertEqual(run.checks["a"]["status"], "PASS")
        self.assertTrue(lines[0].startswith("--- "))
        self.assertEqual(lines[1:4], ["[RUN] a", "x" * 256, "[PASS] a"])
        self.assertFalse(hasattr(run, "log"))


class Sink:
    def put(self, event):
        pass


class OutputEOF(unittest.TestCase):
    def run_closed_pipe(self, after_close):
        script = ("import os, time\n"
                  "print('[RUN] probe', flush=True)\n"
                  "os.close(1)\n"
                  "os.close(2)\n" + after_close)
        run = xc.Run("synthetic")
        with tempfile.TemporaryDirectory() as tmp:
            hung = xc.run_title([sys.executable, "-I", "-c", script], os.environ, tmp, run,
                                ["probe"], 1, None, Sink(), [])
        return hung, run

    def test_live_title_after_eof_returns_hung_check_for_restart(self):
        hung, run = self.run_closed_pipe("time.sleep(3)")
        self.assertEqual(hung, "probe")
        self.assertEqual(run.checks["probe"]["status"], "HANG")
        self.assertEqual(run.checks["probe"]["reason"], "no output for 1 s")

    def test_exit_after_eof_keeps_exit_code(self):
        hung, run = self.run_closed_pipe("raise SystemExit(7)")
        self.assertIsNone(hung)
        self.assertEqual(run.exit_code, 7)
        self.assertEqual(run.checks["probe"]["status"], "FAIL")


# A stand-in title: checks a, b and c; b hangs unless XCL_TESTTITLE_SKIP names it (or HONOUR_SKIP is 0).
FAKE_TITLE = """
import os, sys, time
skip = os.environ.get("XCL_TESTTITLE_SKIP", "").split(";")
honour = os.environ.get("HONOUR_SKIP", "1") == "1"
for name in ("a", "b", "c"):
    if honour and name in skip:
        print("[SKIP] " + name, flush=True)
        continue
    print("[RUN] " + name, flush=True)
    if name == "b":
        time.sleep(60)
    print("[PASS] " + name, flush=True)
"""


class Restart(unittest.TestCase):
    """run_with_restarts: a hung check is recorded as HANG and the title is run again with it skipped."""

    def run_fake(self, honour):
        env = dict(os.environ, HONOUR_SKIP="1" if honour else "0")
        cmd = [sys.executable, "-I", "-c", FAKE_TITLE]
        attempts = []

        def cmd_for(attempt):
            attempts.append(attempt)
            return cmd

        run = xc.Run("synthetic")
        with tempfile.TemporaryDirectory() as tmp:
            skip = xc.run_with_restarts(cmd_for, env, tmp, run, ["a", "b", "c"], 2, None, Sink(), [], 3,
                                        os.path.join(tmp, "out.txt"))
        return run, skip, attempts

    def test_hang_then_restart_with_the_check_skipped(self):
        run, skip, attempts = self.run_fake(honour=True)
        self.assertEqual(skip, ["b"])
        self.assertEqual(attempts, [0, 1])
        self.assertEqual(run.checks["b"]["status"], "HANG")
        self.assertEqual(run.checks["b"]["reason"], "no output for 2 s")
        self.assertEqual((run.checks["a"]["status"], run.checks["c"]["status"]), ("PASS", "PASS"))

    def test_stops_when_a_skipped_check_hangs_again(self):
        run, skip, attempts = self.run_fake(honour=False)
        self.assertEqual(skip, ["b"])
        self.assertEqual(attempts, [0, 1])
        self.assertEqual(run.checks["b"]["status"], "HANG")
        self.assertNotIn("c", run.checks)  # run_all marks it NOT RUN


class RestartTraces(unittest.TestCase):
    """Hang restarts reset game/ and overlay/, while preserving earlier trace_N.jsonl files."""

    def test_hang_restart_preserves_previous_trace_and_resets_game_and_overlay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xcl_dir = root / "xcl"
            rel = xcl_dir / "target" / "release"
            rel.mkdir(parents=True)
            for name in ("xcl.exe", "xcl_testtitle.exe", "testgame_mw.dll"):
                (rel / name).write_bytes(b"stub")
            testgame = xcl_dir / "tests" / "data" / "testgame"
            testgame.mkdir(parents=True)
            (testgame / "marker.txt").write_text("clean", encoding="utf-8", newline="\n")

            gta6rt = root / "gta6rt.exe"
            gta6rt.write_bytes(b"stub")

            work = root / "work"
            stale_dir = work / "leonida"
            stale_dir.mkdir(parents=True)
            stale_trace = stale_dir / "trace_stale.jsonl"
            stale_trace.write_text("old\n", encoding="utf-8", newline="\n")

            seen = []

            def fake_run_title(cmd, env, cwd, run, known, timeout, cdb, events, skip, transcript=None):
                if run.label == "XCL":
                    for name in ("a", "b", "c"):
                        run.set(name, status="PASS")
                    return None
                trace_path = Path(cmd[cmd.index("--trace") + 1])
                overlay_path = Path(cmd[cmd.index("--overlay") + 1])
                game_path = Path(cmd[cmd.index("--game-dir") + 1])
                attempt = len(seen)
                seen.append({
                    "attempt": attempt,
                    "trace": trace_path,
                    "overlay_existed": overlay_path.exists(),
                    "marker": (game_path / "marker.txt").read_text(encoding="utf-8"),
                })
                overlay_path.mkdir(parents=True, exist_ok=True)
                (overlay_path / "dirty.txt").write_text(f"attempt {attempt}\n", encoding="utf-8", newline="\n")
                (game_path / "marker.txt").write_text(f"dirty {attempt}", encoding="utf-8", newline="\n")
                trace_path.write_text(f'{{"attempt": {attempt}}}\n', encoding="utf-8", newline="\n")
                if attempt == 0:
                    run.set("a", status="PASS")
                    run.set("b", status="HANG", reason="no output for 15 s")
                    return "b"
                run.set("c", status="PASS")
                return None

            args = argparse.Namespace(
                xcl=str(xcl_dir),
                gta6rt=str(gta6rt),
                work=str(work),
                timeout=15.0,
                max_restarts=3,
                skip=[],
                cdb=None,
                out=None,
            )
            with mock.patch.object(xc, "run_title", side_effect=fake_run_title):
                xc.run_all(args, Sink())

            self.assertEqual(len(seen), 2)
            self.assertFalse(stale_trace.exists())
            self.assertFalse(seen[1]["overlay_existed"])
            self.assertEqual(seen[1]["marker"], "clean")

            trace0 = work / "leonida" / "trace_0.jsonl"
            trace1 = work / "leonida" / "trace_1.jsonl"
            self.assertTrue(trace0.is_file(), "attempt 0 trace was deleted by attempt 1 restart")
            self.assertTrue(trace1.is_file())
            self.assertEqual(trace0.read_text(encoding="utf-8"), '{"attempt": 0}\n')
            self.assertEqual(trace1.read_text(encoding="utf-8"), '{"attempt": 1}\n')


if __name__ == "__main__":
    unittest.main()
