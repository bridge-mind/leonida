"""Stdlib LEONIDA TUI helpers (no TTY interaction, no game data).

    python -I tests/test_leonida_tui.py
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import leonida_tui  # noqa: E402
import make_game  # noqa: E402


class EligibleTest(unittest.TestCase):
    def test_no_tui_flag_disables(self):
        self.assertFalse(leonida_tui.tui_eligible(True))


class JobsBoundsTest(unittest.TestCase):
    def test_clamp_and_defaults(self):
        with mock.patch.object(leonida_tui.os, "cpu_count", return_value=8):
            lo, half, hi = leonida_tui.cpu_jobs_bounds()
            self.assertEqual((lo, half, hi), (1, 4, 8))
            self.assertEqual(leonida_tui.clamp_jobs(0), 1)
            self.assertEqual(leonida_tui.clamp_jobs(1), 1)
            self.assertEqual(leonida_tui.clamp_jobs(4), 4)
            self.assertEqual(leonida_tui.clamp_jobs(99), 8)

    def test_read_key_timeout_zero_polls_once(self):
        # Regression: timeout 0.0 used to skip the kbhit loop entirely on Windows.
        calls = {"hit": 0}

        class FakeMsvcrt:
            def kbhit(self):
                calls["hit"] += 1
                return False

            def getwch(self):
                raise AssertionError("should not read when kbhit is false")

        with mock.patch.object(leonida_tui, "os") as fake_os:
            fake_os.name = "nt"
            with mock.patch.dict(sys.modules, {"msvcrt": FakeMsvcrt()}):
                # Re-import path: _read_key imports msvcrt inside the nt branch.
                self.assertIsNone(leonida_tui._read_key(0.0))
        self.assertEqual(calls["hit"], 1)

    def test_nt_arrow_scan_codes(self):
        seq = iter(["\xe0", "H", "\xe0", "P", "\xe0", "K", "\xe0", "M",
                    "\xe0", "I", "\xe0", "Q", "\xe0", "G", "\xe0", "O"])

        class FakeMsvcrt:
            def getwch(self):
                return next(seq)

        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "Up")
        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "Down")
        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "Left")
        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "Right")
        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "PageUp")
        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "PageDown")
        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "Home")
        self.assertEqual(leonida_tui._nt_consume(FakeMsvcrt()), "End")

    def test_wheel_delta_to_keys(self):
        self.assertEqual(leonida_tui._nt_wheel_delta_to_keys(120), ["Up"])
        self.assertEqual(leonida_tui._nt_wheel_delta_to_keys(-240), ["Down", "Down"])
        self.assertEqual(leonida_tui._nt_wheel_delta_to_keys(0), [])
        # Horizontal is inverted vs keyboard: tilt right → previous tab (Left).
        self.assertEqual(leonida_tui._nt_wheel_delta_to_keys(120, horizontal=True), ["Left"])
        self.assertEqual(leonida_tui._nt_wheel_delta_to_keys(-120, horizontal=True), ["Right"])

    def test_sgr_mouse_wheel(self):
        self.assertEqual(leonida_tui._parse_sgr_mouse("64;10;5", "M"), "Up")
        self.assertEqual(leonida_tui._parse_sgr_mouse("65;1;1", "m"), "Down")
        self.assertEqual(leonida_tui._parse_sgr_mouse("66;1;1", "M"), "Left")
        self.assertEqual(leonida_tui._parse_sgr_mouse("67;1;1", "M"), "Right")
        self.assertIsNone(leonida_tui._parse_sgr_mouse("0;1;1", "M"))  # click ignored


class StageConsoleTest(unittest.TestCase):
    def test_write_and_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shaders.log")
            c = leonida_tui.StageConsole("shaders", path, maxlen=3)
            c.write_line("a")
            c.write_line("b")
            c.write_line("c")
            c.write_line("d")
            self.assertEqual(c.snapshot(10), ["b", "c", "d"])
            c.close()
            with open(path, encoding="utf-8") as f:
                self.assertEqual(f.read().splitlines(), ["a", "b", "c", "d"])

    def test_wrap_log_lines_breaks_and_keeps_newest(self):
        rows = leonida_tui.wrap_log_lines(["abcdefghij", "xyz"], width=4, max_rows=3)
        self.assertEqual(rows, ["efgh", "ij", "xyz"])


class CapturePopenTest(unittest.TestCase):
    def test_pipes_stdio_and_detaches_console_on_windows(self):
        kw = leonida_tui.capture_popen_kwargs()
        self.assertIs(kw["stdout"], __import__("subprocess").PIPE)
        self.assertIs(kw["stderr"], __import__("subprocess").STDOUT)
        if os.name == "nt":
            self.assertEqual(kw["creationflags"], __import__("subprocess").CREATE_NO_WINDOW)
        else:
            self.assertNotIn("creationflags", kw)


class InterruptKeyTest(unittest.TestCase):
    def test_ctrl_c_char_force_exits(self):
        calls = {"exit": 0}

        class FakeMsvcrt:
            def getwch(self):
                return "\x03"

        with mock.patch.object(leonida_tui, "force_exit_interrupt",
                               side_effect=lambda: calls.__setitem__("exit", 1)):
            self.assertIsNone(leonida_tui._nt_consume(FakeMsvcrt()))
        self.assertEqual(calls["exit"], 1)


class RestoreConsoleTest(unittest.TestCase):
    def test_leave_alt_clears_and_restores_modes(self):
        chunks = []

        class Fake:
            def write(self, data):
                chunks.append(data)
                return len(data)

            def flush(self):
                pass

        leonida_tui.restore_console(Fake(), ansi=True, leave_alt=True)
        text = "".join(chunks)
        self.assertIn("\033[?1049l", text)
        self.assertIn("\033[?7h", text)
        self.assertIn("\033[?25h", text)
        self.assertIn("\033[H", text)
        self.assertIn("\033[J", text)


class BuildRuntimeCaptureTest(unittest.TestCase):
    def test_script_no_longer_shares_the_parent_console(self):
        path = os.path.join(ROOT, "scripts", "build_runtime.ps1")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # The old Start-Process line shared the console; the new one must hide + redirect.
        self.assertNotRegex(text, r"Start-Process[^\n]*-NoNewWindow")
        self.assertIn("-WindowStyle Hidden", text)
        self.assertIn("-RedirectStandardOutput", text)
        self.assertIn("Write-Output", text)
        self.assertNotIn("Write-Host", text)


class BoardSelectTest(unittest.TestCase):
    def test_tab_cycles_and_digit_selects(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = leonida_tui.LeonidaBoard(["shaders", "assets", "runtime"], tmp)
            self.assertEqual(board.stages[board.selected], "shaders")
            board._handle_key("Tab")
            self.assertEqual(board.stages[board.selected], "assets")
            board._handle_key("3")
            self.assertEqual(board.stages[board.selected], "runtime")
            board._handle_key("Left")
            self.assertEqual(board.stages[board.selected], "assets")
            board.write("assets", "hello")
            self.assertIn("hello", board.consoles["assets"].snapshot(5))
            board.stop()
            for c in board.consoles.values():
                c.close()

    def test_other_stage_lines_do_not_dirty_visible_pane(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = leonida_tui.LeonidaBoard(["shaders", "assets"], tmp)
            board._dirty = False
            board.write("assets", "hidden")
            self.assertFalse(board._dirty)
            board.write("shaders", "visible")
            self.assertTrue(board._dirty)
            board.stop()
            for c in board.consoles.values():
                c.close()

    def test_up_scrolls_log_and_end_returns_to_live(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = leonida_tui.LeonidaBoard(["shaders"], tmp)
            for i in range(20):
                board.write("shaders", f"line-{i}")
            self.assertEqual(board.scroll, 0)
            board._handle_key("Up")
            board._handle_key("Up")
            self.assertEqual(board.scroll, 2)
            board._handle_key("End")
            self.assertEqual(board.scroll, 0)
            board._handle_key("Up")
            board._handle_key("Tab")  # only one stage; still resets scroll
            self.assertEqual(board.scroll, 0)
            board.stop()
            for c in board.consoles.values():
                c.close()

    def test_wheel_keys_scroll_like_arrows(self):
        with tempfile.TemporaryDirectory() as tmp:
            board = leonida_tui.LeonidaBoard(["shaders"], tmp)
            for i in range(20):
                board.write("shaders", f"line-{i}")
            board._handle_key("Up")  # wheel-up maps to Up
            self.assertEqual(board.scroll, 1)
            board._handle_key("Down")  # wheel-down maps to Down
            self.assertEqual(board.scroll, 0)
            board.stop()
            for c in board.consoles.values():
                c.close()


class StageScopeTest(unittest.TestCase):
    def test_routes_log_into_stage_console(self):
        with tempfile.TemporaryDirectory() as tmp:
            with leonida_tui.LeonidaBoard(["shaders", "assets"], tmp) as board:
                with leonida_tui.stage_scope("shaders"):
                    make_game.log("from shaders")
                self.assertTrue(any("from shaders" in line for line in board.consoles["shaders"].snapshot(20)))
                self.assertEqual(board.consoles["shaders"].status, "done")

    def test_stray_stdout_lands_in_active_pane(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(leonida_tui, "enable_ansi", return_value=False), \
                    mock.patch.object(leonida_tui, "term_size", return_value=(80, 24)):
                with leonida_tui.LeonidaBoard(["shaders", "assets"], tmp) as board:
                    with leonida_tui.stage_scope("assets"):
                        print("leaked line from library")
                    # Flush any buffered proxy content.
                    sys.stdout.flush()
                    self.assertTrue(
                        any("leaked line from library" in line
                            for line in board.consoles["assets"].snapshot(20)))

    def test_redraw_row_count_matches_terminal_height(self):
        with tempfile.TemporaryDirectory() as tmp:
            captured = []

            class FakeOut:
                encoding = "utf-8"

                def write(self, data):
                    captured.append(data)
                    return len(data)

                def flush(self):
                    pass

                def isatty(self):
                    return False

                def fileno(self):
                    return 1

            fake = FakeOut()
            with mock.patch.object(leonida_tui, "enable_ansi", return_value=False), \
                    mock.patch.object(leonida_tui, "term_size", return_value=(80, 24)), \
                    mock.patch.object(sys, "stdout", fake), \
                    mock.patch.object(sys, "stderr", fake):
                board = leonida_tui.LeonidaBoard(["shaders"], tmp)
                board._real_out = fake
                board._real_err = fake
                board._ansi = False
                board.write("shaders", "line one")
                board._redraw(force=True)
                board.stop()
                for c in board.consoles.values():
                    c.close()
            text = "".join(captured)
            # Non-ANSI path joins exactly height rows with no trailing newline → no scroll.
            self.assertEqual(text.count("\n"), 23)  # 24 rows → 23 separators
            self.assertIn("line one", text)


class SerialOverlapTest(unittest.TestCase):
    def test_overlap_false_runs_in_order(self):
        order = []

        def record(name):
            def fn(*a, **k):
                order.append(name)
            return fn

        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "g")
            out = os.path.join(tmp, "o")
            work = os.path.join(tmp, "w")
            os.makedirs(game)
            with mock.patch.object(make_game, "shaders", side_effect=record("shaders")), \
                    mock.patch.object(make_game, "assets", side_effect=record("assets")), \
                    mock.patch.object(make_game, "runtime", side_effect=record("runtime")), \
                    mock.patch.object(make_game, "pipelines", side_effect=record("pipelines")), \
                    mock.patch.object(make_game, "log"), \
                    mock.patch.object(make_game, "pipeline_check_exe", return_value="x"):
                make_game.run_build_stages(
                    ["shaders", "assets", "runtime", "pipelines"],
                    game=game, work=work, out=out, v={}, wav=False, jobs=2,
                    runtime_build=None, emulate=False, keep_emulated=True, fallbacks=(),
                    overlap=False)
        self.assertEqual(order, ["shaders", "assets", "runtime", "pipelines"])

    def test_overlap_waits_for_runtime_before_pipelines_even_if_checker_exists(self):
        """Flyander/#1797: a stale vk_pipeline_check on disk must not start pipelines mid-rebuild."""
        import threading
        import time

        order = []
        lock = threading.Lock()

        def record(name, hold=0.0):
            def fn(*a, **k):
                with lock:
                    order.append(f"start-{name}")
                if hold:
                    time.sleep(hold)
                with lock:
                    order.append(f"end-{name}")
            return fn

        with tempfile.TemporaryDirectory() as tmp:
            game = os.path.join(tmp, "g")
            out = os.path.join(tmp, "o")
            work = os.path.join(tmp, "w")
            os.makedirs(game)
            with mock.patch.object(make_game, "shaders", side_effect=record("shaders")), \
                    mock.patch.object(make_game, "assets", side_effect=record("assets")), \
                    mock.patch.object(make_game, "runtime", side_effect=record("runtime", hold=0.15)), \
                    mock.patch.object(make_game, "pipelines", side_effect=record("pipelines")), \
                    mock.patch.object(make_game, "log"), \
                    mock.patch.object(make_game, "pipeline_check_exe", return_value="stale_checker"):
                make_game.run_build_stages(
                    ["shaders", "assets", "runtime", "pipelines"],
                    game=game, work=work, out=out, v={}, wav=False, jobs=2,
                    runtime_build=None, emulate=False, keep_emulated=True, fallbacks=(),
                    overlap=True)
        self.assertIn("end-runtime", order)
        self.assertIn("start-pipelines", order)
        self.assertLess(order.index("end-runtime"), order.index("start-pipelines"))


class SetupDefaultsTest(unittest.TestCase):
    def test_esc_keeps_clamped_defaults(self):
        keys = iter(["Esc"])

        def fake_key(timeout_s=0.05):
            try:
                return next(keys)
            except StopIteration:
                return "Enter"

        with mock.patch.object(leonida_tui.os, "cpu_count", return_value=8), \
                mock.patch.object(leonida_tui, "_read_key", side_effect=fake_key), \
                mock.patch.object(leonida_tui, "enable_ansi", return_value=False), \
                mock.patch.object(leonida_tui, "term_size", return_value=(80, 24)), \
                mock.patch("sys.stdout"):
            cfg = leonida_tui.run_setup(7, default_overlap=False)
        self.assertEqual(cfg.jobs, 7)
        self.assertFalse(cfg.overlap)

    def test_jobs_cannot_exceed_cpu_count(self):
        keys = iter(["Right", "Right", "Right", "Enter"])

        def fake_key(timeout_s=0.05):
            try:
                return next(keys)
            except StopIteration:
                return None

        with mock.patch.object(leonida_tui.os, "cpu_count", return_value=4), \
                mock.patch.object(leonida_tui, "_read_key", side_effect=fake_key), \
                mock.patch.object(leonida_tui, "enable_ansi", return_value=False), \
                mock.patch.object(leonida_tui, "term_size", return_value=(80, 24)), \
                mock.patch("sys.stdout"):
            cfg = leonida_tui.run_setup(4, default_overlap=True)
        self.assertEqual(cfg.jobs, 4)

    def test_arrows_select_row_and_change_values(self):
        # Down to Overlap, Right toggles off, Up to Jobs, Left lowers jobs, Enter.
        keys = iter(["Down", "Right", "Up", "Left", "Enter"])

        def fake_key(timeout_s=0.05):
            try:
                return next(keys)
            except StopIteration:
                return None

        with mock.patch.object(leonida_tui.os, "cpu_count", return_value=8), \
                mock.patch.object(leonida_tui, "_read_key", side_effect=fake_key), \
                mock.patch.object(leonida_tui, "enable_ansi", return_value=False), \
                mock.patch.object(leonida_tui, "term_size", return_value=(80, 24)), \
                mock.patch("sys.stdout"):
            cfg = leonida_tui.run_setup(4, default_overlap=True)
        self.assertEqual(cfg.jobs, 3)
        self.assertFalse(cfg.overlap)


if __name__ == "__main__":
    unittest.main()
