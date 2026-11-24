"""Stdlib LEONIDA build console: setup form + switchable per-stage log panes.

No third-party TUI libraries. When stdin/stdout are not TTYs, callers skip this module.
"""
from __future__ import annotations

import contextvars
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass

STAGE_ORDER = ("shaders", "assets", "runtime", "pipelines")

# Figlet-style wordmark (single block, ASCII only).
LEONIDA_BANNER = [
    r"  _     _____ ___  _   _ ___ ____    _    ",
    r" | |   | ____/ _ \| \ | |_ _|  _ \  / \   ",
    r" | |   |  _|| | | |  \| || || | | |/ _ \  ",
    r" | |___| |__| |_| | |\  || || |_| / ___ \ ",
    r" |_____|_____\___/|_| \_|___|____/_/   \_" + "\\",
]

# ANSI (used only when enable_ansi() succeeded).
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_WHITE = "\033[37m"
_REVERSE = "\033[7m"  # selection: invert, not a blue fill

_current_stage: contextvars.ContextVar[str | None] = contextvars.ContextVar("leonida_stage", default=None)
_board_ref: contextvars.ContextVar[object | None] = contextvars.ContextVar("leonida_board", default=None)
# Process-wide fallback: ContextVar is not visible in every worker on every Python/thread setup,
# but make_game.run() must still find the board to pipe cmake/dotnet away from the console.
_active_board: object | None = None
_live_procs: set = set()
_live_procs_lock = threading.Lock()
_interrupt_lock = threading.Lock()
_interrupt_fired = False
_ctrl_handler_installed = False
# Windows console input mode while the board is live (mouse wheel → scroll).
_nt_mouse_armed: tuple | None = None  # (handle, old_mode) or None
_nt_key_queue: deque = deque()

# Console input flags (Windows).
_ENABLE_PROCESSED_INPUT = 0x0001
_ENABLE_LINE_INPUT = 0x0002
_ENABLE_ECHO_INPUT = 0x0004
_ENABLE_WINDOW_INPUT = 0x0008
_ENABLE_MOUSE_INPUT = 0x0010
_ENABLE_EXTENDED_FLAGS = 0x0080
_ENABLE_QUICK_EDIT_MODE = 0x0040
_KEY_EVENT = 0x0001
_MOUSE_EVENT = 0x0002
_MOUSE_WHEELED = 0x0004
_MOUSE_HWHEELED = 0x0008


@dataclass
class SetupConfig:
    jobs: int
    overlap: bool


def cpu_jobs_bounds() -> tuple[int, int, int]:
    """(minimum, default, maximum) job counts from the machine's CPU count."""
    cpus = max(1, os.cpu_count() or 2)
    return 1, max(1, cpus // 2), cpus


def clamp_jobs(jobs: int) -> int:
    lo, _, hi = cpu_jobs_bounds()
    return max(lo, min(hi, int(jobs)))


def tui_eligible(no_tui: bool) -> bool:
    if no_tui:
        return False
    try:
        return bool(sys.stdin and sys.stdin.isatty() and sys.stdout and sys.stdout.isatty())
    except Exception:
        return False


def current_stage() -> str | None:
    return _current_stage.get()


def current_board():
    return _board_ref.get() or _active_board


def stage_scope(name: str):
    """Set the active stage for log()/subprocess capture in this context/thread."""
    return _StageScope(name)


class _StageScope:
    def __init__(self, name: str):
        self.name = name
        self._token = None

    def __enter__(self):
        self._token = _current_stage.set(self.name)
        board = current_board()
        if board is not None:
            board.set_status(self.name, "running")
        return self

    def __exit__(self, exc_type, exc, tb):
        board = current_board()
        if board is not None:
            board.set_status(self.name, "failed" if exc_type else "done")
        if self._token is not None:
            _current_stage.reset(self._token)
        return False


def enable_ansi() -> bool:
    """Turn on VT processing on Windows consoles; True when clear/home sequences are usable."""
    if os.name != "nt":
        return True
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def term_size() -> tuple[int, int]:
    size = shutil.get_terminal_size(fallback=(100, 30))
    return max(60, size.columns), max(20, size.lines)


def restore_console(stream=None, *, ansi: bool, leave_alt: bool = False) -> None:
    """Undo TUI modes and clear the visible buffer so the shell prompt is not painted over leftovers."""
    out = stream if stream is not None else sys.stdout
    try:
        if not ansi:
            try:
                out.write("\n")
                out.flush()
            except OSError:
                pass
            return
        parts = [
            "\033[?1007l",  # alternate scroll (wheel → arrows) off
            "\033[?1000l",  # mouse tracking off
            "\033[?1006l",  # SGR mouse off
            "\033[?7h",     # wrap on
            "\033[?25h",    # cursor on
            "\033[0m",      # attributes off
        ]
        if leave_alt:
            parts.append("\033[?1049l")
        # Clear the buffer we are about to show the user (main screen after leaving alt).
        parts.extend(["\033[H", "\033[J"])
        try:
            out.write("".join(parts))
            out.flush()
        except OSError:
            pass
    finally:
        _nt_disarm_mouse_input()


def _c(text: str, *codes: str, on: bool) -> str:
    if not on or not codes:
        return text
    return "".join(codes) + text + _RESET


def wrap_log_lines(lines: list[str], width: int, max_rows: int) -> list[str]:
    """Hard-wrap log lines to width; keep the newest max_rows rows (terminal-like)."""
    cols = max(1, width)
    wrapped: list[str] = []
    for ln in lines:
        text = ln.replace("\t", " ")
        if text == "":
            wrapped.append("")
            continue
        while text:
            wrapped.append(text[:cols])
            text = text[cols:]
    if max_rows <= 0:
        return wrapped
    return wrapped[-max_rows:]


def _nt_arm_mouse_input() -> None:
    """Enable console mouse-wheel reports; disable Quick Edit so the OS delivers them."""
    global _nt_mouse_armed
    if os.name != "nt" or _nt_mouse_armed is not None:
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-10)  # STD_INPUT_HANDLE
        old = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(old)):
            return
        # Keep processed input; drop line/echo/quick-edit so keys and wheel are readable.
        new_mode = (
            (old.value | _ENABLE_PROCESSED_INPUT | _ENABLE_MOUSE_INPUT
             | _ENABLE_EXTENDED_FLAGS | _ENABLE_WINDOW_INPUT)
            & ~(_ENABLE_LINE_INPUT | _ENABLE_ECHO_INPUT | _ENABLE_QUICK_EDIT_MODE)
        )
        if not kernel32.SetConsoleMode(handle, new_mode):
            return
        _nt_mouse_armed = (handle, old.value)
        _nt_key_queue.clear()
    except Exception:
        _nt_mouse_armed = None


def _nt_disarm_mouse_input() -> None:
    global _nt_mouse_armed
    if _nt_mouse_armed is None:
        return
    handle, old_mode = _nt_mouse_armed
    _nt_mouse_armed = None
    _nt_key_queue.clear()
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleMode(handle, old_mode)
    except Exception:
        pass


def _nt_wheel_delta_to_keys(delta: int, *, horizontal: bool = False) -> list[str]:
    """Map a Win32 wheel delta to keys (120 units ≈ one notch).

    Vertical → Up/Down (log scroll). Horizontal → Left/Right (stage tabs), inverted so a
    right tilt/swipe selects the previous tab (opposite of the keyboard arrows).
    """
    if delta == 0:
        return []
    notches = max(1, abs(int(delta)) // 120)
    if horizontal:
        key = "Left" if delta > 0 else "Right"
    else:
        key = "Up" if delta > 0 else "Down"
    return [key] * notches


def _nt_decode_key_event(key_down: bool, vk: int, ch: str) -> str | None:
    if not key_down:
        return None
    if ch in ("\x03", "\x1a"):
        request_interrupt()
        return None
    # Virtual-key map for arrows / paging (UnicodeChar is often NUL for these).
    vk_map = {
        0x26: "Up", 0x28: "Down", 0x25: "Left", 0x27: "Right",
        0x21: "PageUp", 0x22: "PageDown", 0x24: "Home", 0x23: "End",
        0x09: "Tab", 0x0D: "Enter", 0x1B: "Esc", 0x20: "Space",
    }
    if vk in vk_map:
        return vk_map[vk]
    if ch == "\r":
        return "Enter"
    if ch == "\t":
        return "Tab"
    if ch == "\x1b":
        return "Esc"
    if ch == " ":
        return "Space"
    if ch and ch.isprintable():
        return ch
    return None


def _nt_read_console_key(timeout_s: float) -> str | None:
    """Read one key or wheel step via ReadConsoleInput (mouse mode armed)."""
    if _nt_key_queue:
        return _nt_key_queue.popleft()
    if _nt_mouse_armed is None:
        return None
    handle, _ = _nt_mouse_armed
    try:
        import ctypes
        from ctypes import wintypes
    except ImportError:
        return None

    class COORD(ctypes.Structure):
        _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

    class KEY_EVENT_RECORD(ctypes.Structure):
        _fields_ = [
            ("bKeyDown", wintypes.BOOL),
            ("wRepeatCount", wintypes.WORD),
            ("wVirtualKeyCode", wintypes.WORD),
            ("wVirtualScanCode", wintypes.WORD),
            ("UnicodeChar", wintypes.WCHAR),
            ("dwControlKeyState", wintypes.DWORD),
        ]

    class MOUSE_EVENT_RECORD(ctypes.Structure):
        _fields_ = [
            ("dwMousePosition", COORD),
            ("dwButtonState", wintypes.DWORD),
            ("dwControlKeyState", wintypes.DWORD),
            ("dwEventFlags", wintypes.DWORD),
        ]

    class INPUT_UNION(ctypes.Union):
        _fields_ = [("KeyEvent", KEY_EVENT_RECORD), ("MouseEvent", MOUSE_EVENT_RECORD)]

    class INPUT_RECORD(ctypes.Structure):
        _fields_ = [("EventType", wintypes.WORD), ("Event", INPUT_UNION)]

    kernel32 = ctypes.windll.kernel32
    timeout_ms = 0 if timeout_s <= 0 else max(1, int(timeout_s * 1000))
    deadline = time.monotonic() + max(0.0, timeout_s)

    while True:
        if _nt_key_queue:
            return _nt_key_queue.popleft()
        wait = kernel32.WaitForSingleObject(handle, timeout_ms if timeout_s > 0 else 0)
        if wait != 0:  # WAIT_OBJECT_0 == 0
            if timeout_s <= 0 or time.monotonic() >= deadline:
                return None
            remaining = deadline - time.monotonic()
            timeout_ms = max(1, int(remaining * 1000))
            continue
        rec = INPUT_RECORD()
        nread = wintypes.DWORD()
        if not kernel32.ReadConsoleInputW(handle, ctypes.byref(rec), 1, ctypes.byref(nread)):
            return None
        if nread.value == 0:
            return None
        if rec.EventType == _MOUSE_EVENT:
            flags = rec.Event.MouseEvent.dwEventFlags
            if flags & (_MOUSE_WHEELED | _MOUSE_HWHEELED):
                # High word of ButtonState is a signed wheel delta.
                raw = rec.Event.MouseEvent.dwButtonState
                delta = ctypes.c_short((raw >> 16) & 0xFFFF).value
                keys = _nt_wheel_delta_to_keys(delta, horizontal=bool(flags & _MOUSE_HWHEELED))
                if not keys:
                    continue
                _nt_key_queue.extend(keys[1:])
                return keys[0]
            # Clicks / moves: consume so they do not clog the buffer.
            continue
        if rec.EventType != _KEY_EVENT:
            continue
        ke = rec.Event.KeyEvent
        mapped = _nt_decode_key_event(bool(ke.bKeyDown), int(ke.wVirtualKeyCode), ke.UnicodeChar or "")
        if mapped is not None:
            # RepeatCount > 1 (held key): queue extras.
            for _ in range(max(0, int(ke.wRepeatCount) - 1)):
                _nt_key_queue.append(mapped)
            return mapped
        if timeout_s <= 0:
            # Non-blocking: keep draining until a useful event or empty.
            wait = kernel32.WaitForSingleObject(handle, 0)
            if wait != 0:
                return None
            continue
        if time.monotonic() >= deadline:
            return None


def _parse_sgr_mouse(params: str, trail: str) -> str | None:
    """CSI < btn ; col ; row M/m → wheel buttons (64/65 vertical, 66/67 horizontal)."""
    if trail not in ("M", "m"):
        return None
    parts = params.split(";")
    if not parts:
        return None
    try:
        btn = int(parts[0])
    except ValueError:
        return None
    # 64/65 = vertical wheel; 66/67 = horizontal (inverted: 66 → Left, 67 → Right).
    if btn == 64:
        return "Up"
    if btn == 65:
        return "Down"
    if btn == 66:
        return "Left"
    if btn == 67:
        return "Right"
    return None


def _read_key(timeout_s: float = 0.05) -> str | None:
    """One key from stdin, or None. Returns Tab, Left, Right, Enter, Esc, Space, or a character."""
    if os.name == "nt":
        if _nt_mouse_armed is not None:
            return _nt_read_console_key(timeout_s)
        try:
            import msvcrt
        except ImportError:
            return None
        # timeout <= 0: single non-blocking poll (the board loop used 0.0 and never read keys).
        if timeout_s <= 0:
            if not msvcrt.kbhit():
                return None
            return _nt_consume(msvcrt)
        deadline = time.monotonic() + timeout_s
        while True:
            if msvcrt.kbhit():
                return _nt_consume(msvcrt)
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.02)

    import select

    try:
        import termios
        import tty
    except ImportError:
        return None
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        ready, _, _ = select.select([sys.stdin], [], [], max(0.0, timeout_s))
        if not ready:
            return None
        ch = sys.stdin.read(1)
        if ch in ("\x03", "\x1a"):
            request_interrupt()
            return None
        if ch == "\x1b":
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                return "Esc"
            rest = sys.stdin.read(1)
            if rest != "[":
                return "Esc"
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not ready:
                return "Esc"
            code = sys.stdin.read(1)
            if code == "<":
                # SGR mouse: CSI < btn ; col ; row M/m (wheel → Up/Down).
                buf = ""
                while len(buf) < 32:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.02)
                    if not ready:
                        break
                    c = sys.stdin.read(1)
                    if c in ("M", "m"):
                        return _parse_sgr_mouse(buf, c)
                    buf += c
                return None
            if code == "A":
                return "Up"
            if code == "B":
                return "Down"
            if code == "D":
                return "Left"
            if code == "C":
                return "Right"
            if code == "H":
                return "Home"
            if code == "F":
                return "End"
            # CSI n~ : PageUp=5, PageDown=6, Home=1, End=4
            ready, _, _ = select.select([sys.stdin], [], [], 0.02)
            tilde = sys.stdin.read(1) if ready else ""
            if tilde == "~":
                return {"5": "PageUp", "6": "PageDown", "1": "Home", "4": "End"}.get(code)
            return "Esc"
        if ch == "\n":
            return "Enter"
        if ch == "\t":
            return "Tab"
        if ch == " ":
            return "Space"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _nt_consume(msvcrt) -> str | None:
    ch = msvcrt.getwch()
    # With msvcrt polling, Ctrl+C often arrives as \x03 instead of a console CTRL_C_EVENT, so the
    # main thread never sees KeyboardInterrupt and the build looks wedged.
    if ch in ("\x03", "\x1a"):  # Ctrl+C, Ctrl+Z
        request_interrupt()
        return None
    if ch in ("\x00", "\xe0"):
        code = msvcrt.getwch()
        return {"H": "Up", "P": "Down", "K": "Left", "M": "Right",
                "I": "PageUp", "Q": "PageDown", "G": "Home", "O": "End"}.get(code)
    if ch == "\r":
        return "Enter"
    if ch == "\x1b":
        return "Esc"
    if ch == "\t":
        return "Tab"
    if ch == " ":
        return "Space"
    return ch


def install_interrupt_handlers() -> None:
    """Make Ctrl+C reliable under the TUI (msvcrt often eats it; children can wedge joins)."""
    global _ctrl_handler_installed
    if _ctrl_handler_installed:
        return
    _ctrl_handler_installed = True
    try:
        import signal
        signal.signal(signal.SIGINT, lambda *_: force_exit_interrupt())
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, lambda *_: force_exit_interrupt())
    except (ValueError, OSError, AttributeError):
        pass
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            Handler = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

            def _handler(ctrl_type):
                # 0=CTRL_C, 1=CTRL_BREAK — return True so the default handler does not race us.
                if ctrl_type in (0, 1):
                    force_exit_interrupt()
                    return True
                return False

            # Keep a reference so the callback is not GC'd.
            install_interrupt_handlers._handler = Handler(_handler)  # type: ignore[attr-defined]
            ctypes.windll.kernel32.SetConsoleCtrlHandler(install_interrupt_handlers._handler, True)
        except Exception:
            pass


def force_exit_interrupt() -> None:
    """Restore the console, kill children, and exit 130. Does not return.

    Graceful KeyboardInterrupt paths still hang when a worker is blocked in readline() or
    ThreadPoolExecutor shutdown waits — this is the reliable Ctrl+C path for the TUI.
    """
    global _interrupt_fired, _active_board
    with _interrupt_lock:
        if _interrupt_fired:
            os._exit(130)
        _interrupt_fired = True
    board = current_board()
    if board is not None:
        try:
            board._stop.set()
        except Exception:
            pass
    kill_live_processes()
    out = sys.__stdout__
    err = sys.__stderr__
    ansi = True
    leave_alt = False
    if board is not None:
        out = getattr(board, "_real_out", out) or out
        err = getattr(board, "_real_err", err) or err
        ansi = bool(getattr(board, "_ansi", True))
        leave_alt = bool(getattr(board, "_alt_screen", False))
        try:
            sys.stdout = out
            sys.stderr = err
        except Exception:
            pass
    try:
        restore_console(out, ansi=ansi, leave_alt=leave_alt)
    except Exception:
        pass
    try:
        err.write("Stopped. Run it again to continue where it left off.\n")
        err.flush()
    except Exception:
        pass
    _active_board = None
    os._exit(130)


def request_interrupt() -> None:
    """Ctrl+C from the key reader — hard-exit so the build cannot stay wedged."""
    force_exit_interrupt()


def register_live_process(proc) -> None:
    with _live_procs_lock:
        _live_procs.add(proc)


def unregister_live_process(proc) -> None:
    with _live_procs_lock:
        _live_procs.discard(proc)


def kill_live_processes() -> None:
    """Terminate every subprocess started for board capture (including Windows child trees)."""
    with _live_procs_lock:
        procs = list(_live_procs)
        _live_procs.clear()
    for proc in procs:
        try:
            if proc.poll() is not None:
                continue
            if os.name == "nt" and proc.pid:
                # PowerShell → cmd → cmake is a tree; terminate() alone leaves grandchildren.
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                proc.terminate()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except (OSError, ProcessLookupError, AttributeError):
            try:
                proc.kill()
            except (OSError, ProcessLookupError, AttributeError):
                pass


def run_setup(default_jobs: int | None = None, default_overlap: bool = True) -> SetupConfig:
    """Interactive LEONIDA build setup. Esc keeps the defaults (clamped).

    Up/Down select a row; Left/Right change its value; Enter starts; Esc keeps defaults.
    Selection paints only the label+value, not the full terminal width.
    """
    install_interrupt_handlers()
    lo, half, hi = cpu_jobs_bounds()
    if default_jobs is None:
        default_jobs = half
    default_jobs = clamp_jobs(default_jobs)
    jobs = default_jobs
    overlap = bool(default_overlap)
    # 0 = Jobs, 1 = Overlap
    cursor = 0
    fields = ("jobs", "overlap")
    ansi = enable_ansi()
    result: SetupConfig | None = None

    def _row(label: str, value: str, selected: bool, bar_width: int) -> str:
        # Highlight covers label+value only (not full terminal width), but Jobs and Overlap
        # share one bar width so a short jobs number does not leave a shorter white strip.
        body = f"  {label:<12} {value}".ljust(bar_width)
        if selected:
            return _c(body, _BOLD, _REVERSE, on=ansi)
        return body

    def draw() -> None:
        width, height = term_size()
        rule = _c("=" * width, _DIM, _CYAN, on=ansi)
        thin = _c("-" * width, _DIM, on=ansi)
        jobs_text = f"  {'Jobs':<12} {jobs}"
        overlap_text = f"  {'Overlap':<12} {'yes' if overlap else 'no'}"
        bar_width = max(len(jobs_text), len(overlap_text))
        rows = [""] * height
        rows[0] = rule
        rows[1] = ""
        for i, line in enumerate(LEONIDA_BANNER):
            rows[2 + i] = _c(line[:width].ljust(width), _BOLD, _CYAN, on=ansi)
        rows[7] = ""
        rows[8] = _c("  Build setup", _BOLD, _WHITE, on=ansi)
        rows[9] = thin
        rows[10] = ""
        # Hint stays outside the selection highlight so the bar does not run to the edge.
        rows[11] = (
            _row("Jobs", str(jobs), cursor == 0, bar_width)
            + _c(f"    min {lo} · default {half} · max {hi}", _DIM, on=ansi)
        )
        rows[12] = ""
        rows[13] = _row("Overlap", "yes" if overlap else "no", cursor == 1, bar_width)
        rows[height - 3] = thin
        rows[height - 2] = _c(
            "  Up/Down select   Left/Right change   Enter start   Esc defaults",
            _DIM, on=ansi)
        rows[height - 1] = rule

        if ansi:
            parts = ["\033[?25l\033[?7l\033[H\033[J"]
            for r, line in enumerate(rows, start=1):
                parts.append(f"\033[{r};1H\033[2K{line}")
            parts.append("\033[1;1H")
            sys.stdout.write("".join(parts))
        else:
            sys.stdout.write("\n".join(rows))
        sys.stdout.flush()

    try:
        draw()
        while True:
            key = _read_key(0.1)
            if key is None:
                continue
            if key == "Esc":
                result = SetupConfig(jobs=default_jobs, overlap=bool(default_overlap))
                break
            if key == "Enter":
                result = SetupConfig(jobs=jobs, overlap=overlap)
                break
            if key in ("Up", "Down", "Tab"):
                cursor = (cursor + (1 if key in ("Down", "Tab") else -1)) % len(fields)
                draw()
            elif key in ("Left", "["):
                if cursor == 0:
                    jobs = max(lo, jobs - 1)
                else:
                    overlap = not overlap
                draw()
            elif key in ("Right", "]", "/", "Space"):
                if cursor == 0:
                    jobs = min(hi, jobs + 1)
                else:
                    overlap = not overlap
                draw()
    finally:
        # Setup draws on the main buffer (not alt screen). Clear it so the board / shell
        # does not come back to a full-screen leftover form.
        restore_console(ansi=ansi)
    return result if result is not None else SetupConfig(jobs=default_jobs, overlap=bool(default_overlap))


class StageConsole:
    """Ring buffer + on-disk log for one build stage."""

    def __init__(self, name: str, log_path: str, maxlen: int = 2000):
        self.name = name
        self.log_path = log_path
        self.lines: deque[str] = deque(maxlen=maxlen)
        self.status = "idle"  # idle | running | done | failed
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        self._file = open(log_path, "a", encoding="utf-8", errors="replace")

    def close(self) -> None:
        with self._lock:
            try:
                self._file.close()
            except Exception:
                pass

    def write_line(self, line: str) -> None:
        line = line.rstrip("\r\n")
        with self._lock:
            self.lines.append(line)
            try:
                self._file.write(line + "\n")
                self._file.flush()
            except OSError:
                pass

    def snapshot(self, n: int) -> list[str]:
        with self._lock:
            if n <= 0:
                return list(self.lines)
            return list(self.lines)[-n:]

    def line_count(self) -> int:
        with self._lock:
            return len(self.lines)


class _BoardStream:
    """Catch stray print()/stdout writes and fold them into the active stage pane."""

    def __init__(self, board: "LeonidaBoard", real):
        self._board = board
        self._real = real
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, data):
        if not isinstance(data, str):
            data = str(data)
        if self._board._is_drawing():
            return self._real.write(data)
        with self._lock:
            self._buf += data
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                line = line.rstrip("\r")
                if line:
                    self._board.write(current_stage(), line)
        return len(data)

    def flush(self):
        with self._lock:
            leftover = self._buf
            self._buf = ""
        if leftover and not self._board._is_drawing():
            self._board.write(current_stage(), leftover.rstrip("\r"))
        return self._real.flush()

    def fileno(self):
        return self._real.fileno()

    def isatty(self):
        return False

    @property
    def encoding(self):
        return getattr(self._real, "encoding", "utf-8")


class LeonidaBoard:
    """LEONIDA-branded live view with key-switchable stage panes."""

    # Fixed chrome (1-based): rows 1–13 header (rule, banner, summary, tabs, pane-top rule);
    # PANE_TOP is the first log row; last 3 rows are thin / help / rule.
    PANE_TOP = 14

    def __init__(self, stages: list[str], log_dir: str):
        self.stages = [s for s in STAGE_ORDER if s in stages] or list(stages)
        self.log_dir = log_dir
        self.consoles = {
            name: StageConsole(name, os.path.join(log_dir, f"{name}.log"))
            for name in self.stages
        }
        self.selected = 0
        self.summary = "build starting"
        # Lines scrolled up from the live tail (0 = follow newest output).
        self.scroll = 0
        self._ansi = enable_ansi()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._dirty = True
        self._token = None
        self._alt_screen = False
        self._tls = threading.local()
        self._real_out = sys.stdout
        self._real_err = sys.stderr
        self._proxy_out = _BoardStream(self, self._real_out)
        self._proxy_err = _BoardStream(self, self._real_err)

    def _is_drawing(self) -> bool:
        return bool(getattr(self._tls, "drawing", False))

    def __enter__(self):
        global _active_board
        install_interrupt_handlers()
        self._token = _board_ref.set(self)
        _active_board = self
        sys.stdout = self._proxy_out
        sys.stderr = self._proxy_err
        if self._ansi:
            try:
                # Alt screen + hide cursor + disable autowrap (wrap is what pushes logs "below").
                # 1007: wheel → Up/Down in alt screen; 1000/1006: SGR mouse wheel reports.
                self._raw("\033[?1049h\033[?25l\033[?7l\033[?1007h\033[?1000h\033[?1006h")
                self._alt_screen = True
            except OSError:
                self._alt_screen = False
        _nt_arm_mouse_input()
        self._thread = threading.Thread(target=self._loop, name="leonida-board", daemon=True)
        self._thread.start()
        self._redraw(force=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        global _active_board
        self.stop()
        sys.stdout = self._real_out
        sys.stderr = self._real_err
        # Leave alt screen (if we entered it) and clear the main buffer. Without the clear,
        # Windows Terminal shows the pre-board main buffer (setup form / leaked cmake lines).
        restore_console(self._real_out, ansi=self._ansi, leave_alt=self._alt_screen)
        self._alt_screen = False
        if self._token is not None:
            _board_ref.reset(self._token)
        if _active_board is self:
            _active_board = None
        for c in self.consoles.values():
            c.close()
        print(f"LEONIDA: stage logs in {self.log_dir}", flush=True)
        return False

    def _raw(self, text: str) -> None:
        self._tls.drawing = True
        try:
            self._real_out.write(text)
            self._real_out.flush()
        finally:
            self._tls.drawing = False

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def set_status(self, name: str, status: str) -> None:
        c = self.consoles.get(name)
        if c is None:
            return
        with self._lock:
            c.status = status
            self._dirty = True

    def set_summary(self, text: str) -> None:
        with self._lock:
            self.summary = text
            self._dirty = True

    def write(self, stage: str | None, line: str) -> None:
        if stage and stage in self.consoles:
            self.consoles[stage].write_line(line)
            visible = stage
        elif self.stages:
            self.consoles[self.stages[self.selected]].write_line(line)
            visible = self.stages[self.selected]
        else:
            return
        with self._lock:
            if visible == self.stages[self.selected]:
                self._dirty = True

    def note(self, line: str) -> None:
        """Build-level message: update summary; do not spam the log pane."""
        with self._lock:
            self.summary = line
            self._dirty = True

    def _loop(self) -> None:
        while not self._stop.is_set():
            while True:
                key = _read_key(0.0)
                if not key:
                    break
                self._handle_key(key)
            # Always repaint: child processes that inherit the console can scroll the
            # alt buffer between ticks; a dirty-only redraw never heals that.
            self._redraw(force=True)
            self._stop.wait(0.08)

    def _handle_key(self, key: str) -> None:
        n = len(self.stages)
        if n == 0:
            return
        with self._lock:
            if key in ("Tab", "Right"):
                self.selected = (self.selected + 1) % n
                self.scroll = 0
                self._dirty = True
            elif key == "Left":
                self.selected = (self.selected - 1) % n
                self.scroll = 0
                self._dirty = True
            elif key in "1234":
                idx = int(key) - 1
                if idx < n:
                    self.selected = idx
                    self.scroll = 0
                    self._dirty = True
            elif key in ("Up", "Down", "PageUp", "PageDown", "Home", "End"):
                # Pane height is measured at redraw; use a conservative step here.
                _, height = term_size()
                pane_h = max(1, height - self.PANE_TOP - 2)
                if key == "Up":
                    self.scroll += 1
                elif key == "Down":
                    self.scroll = max(0, self.scroll - 1)
                elif key == "PageUp":
                    self.scroll += pane_h
                elif key == "PageDown":
                    self.scroll = max(0, self.scroll - pane_h)
                elif key == "Home":
                    self.scroll = 10**9  # clamped in _redraw against buffer length
                elif key == "End":
                    self.scroll = 0
                self._dirty = True

    def _status_style(self, status: str) -> tuple[str, tuple[str, ...]]:
        if status == "running":
            return "[~]", (_YELLOW,)
        if status == "done":
            return "[+]", (_GREEN,)
        if status == "failed":
            return "[x]", (_RED,)
        return "[ ]", (_DIM,)

    def _redraw(self, force: bool = False) -> None:
        with self._lock:
            if not force and not self._dirty:
                return
            self._dirty = False
            selected = self.selected
            summary = self.summary
            scroll = self.scroll
            stages = list(self.stages)
            statuses = {n: self.consoles[n].status for n in stages}
        if not stages:
            return

        width, height = term_size()
        # Last 3 rows: thin, help, rule. Pane is everything from PANE_TOP through the row above.
        footer_thin = height - 2  # 1-based row of the footer thin rule
        pane_bottom = footer_thin - 1
        pane_top = min(self.PANE_TOP, pane_bottom)
        pane_h = max(1, pane_bottom - pane_top + 1)

        name = stages[selected]
        log_lines = self.consoles[name].snapshot(0)
        if not log_lines:
            log_lines = ["(no output yet)"]
        wrapped = wrap_log_lines(log_lines, max(1, width - 2), max_rows=0)
        max_scroll = max(0, len(wrapped) - pane_h)
        if scroll > max_scroll:
            scroll = max_scroll
            with self._lock:
                self.scroll = scroll
        end = len(wrapped) - scroll
        start = max(0, end - pane_h)
        body = wrapped[start:end]
        body = [""] * (pane_h - len(body)) + body

        ansi = self._ansi
        rule = _c("=" * width, _DIM, _CYAN, on=ansi)
        thin = _c("-" * width, _DIM, on=ansi)
        tabs = []
        for i, s in enumerate(stages):
            mark, style = self._status_style(statuses[s])
            label = f" {mark} {s} "
            if i == selected:
                tabs.append(_c(label, _BOLD, _REVERSE, on=ansi))
            else:
                tabs.append(_c(label, *style, on=ansi))

        banner = [_c(line[:width].ljust(width), _BOLD, _CYAN, on=ansi) for line in LEONIDA_BANNER]
        scroll_hint = "  Up/Down/wheel scroll" if scroll else ""
        help_line = _c(
            (f"  Left/Right / Tab / 1-{len(stages)}  stage"
             f"{scroll_hint}    End live    logs: {self.log_dir}")[:width].ljust(width),
            _DIM, on=ansi)

        # Exact height rows — never emit more lines than the terminal has (avoids scroll).
        rows = [""] * height
        rows[0] = rule
        if height > 1:
            rows[1] = ""
        for i, line in enumerate(banner):
            idx = 2 + i
            if idx < height:
                rows[idx] = line
        if 7 < height:
            rows[7] = ""
        if 8 < height:
            rows[8] = _c(f"  {summary}"[:width].ljust(width), _BOLD, _WHITE, on=ansi)
        if 9 < height:
            rows[9] = ""
        if 10 < height:
            # Join tabs as-is (ANSI codes); wrap is disabled so overflow cannot scroll.
            rows[10] = "  " + "  ".join(tabs)
        if 11 < height:
            rows[11] = ""
        if 12 < height and pane_top > 13:
            rows[12] = thin
        for i, ln in enumerate(body):
            row = pane_top - 1 + i  # 0-based
            if 0 <= row < footer_thin - 1:
                rows[row] = ("  " + ln) if ln else ""
        if 0 <= footer_thin - 1 < height:
            rows[footer_thin - 1] = thin
        if footer_thin < height:
            rows[footer_thin] = help_line
        if footer_thin + 1 < height:
            rows[footer_thin + 1] = rule

        try:
            if ansi:
                parts = ["\033[H\033[J"]
                for r, line in enumerate(rows[:height], start=1):
                    # Clear the row then write — no newlines, so the terminal cannot scroll.
                    parts.append(f"\033[{r};1H\033[2K{line}")
                # Park at home (cursor hidden). Parking in the pane let leaked console
                # writes advance downward until they scrolled past the frame.
                parts.append("\033[1;1H")
                self._raw("".join(parts))
            else:
                # No trailing newline: height lines already fill the viewport.
                self._raw("\n".join(rows[:height]))
        except OSError:
            pass


def capture_popen_kwargs() -> dict:
    """Popen kwargs that fold a child's output into the board instead of the console.

    On Windows, cmake/ninja/MSBuild often WriteConsole past a piped stdout when they still
    have a console handle — that is the brief flash on top of the LEONIDA frame. CREATE_NO_WINDOW
    removes the console handle so those tools fall back to the pipe.
    """
    kw: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "errors": "replace",
    }
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kw


def pump_process(proc, stage: str | None, board: LeonidaBoard | None) -> int:
    """Read proc.stdout line-by-line into the stage console; return exit code."""
    register_live_process(proc)
    try:
        stream = proc.stdout
        if stream is not None:
            while True:
                if board is not None and getattr(board, "_stop", None) is not None and board._stop.is_set():
                    kill_live_processes()
                    break
                raw = stream.readline()
                if raw == "" or raw == b"":
                    break
                if isinstance(raw, bytes):
                    line = raw.decode("utf-8", errors="replace")
                else:
                    line = raw
                # Ninja status uses \\r; treat each chunk as a line so progress is visible.
                line = line.replace("\r", "\n").rstrip("\n")
                if not line:
                    continue
                for piece in line.split("\n"):
                    piece = piece.strip()
                    if not piece:
                        continue
                    if board is not None:
                        board.write(stage, piece)
                    else:
                        print(piece, flush=True)
            try:
                stream.close()
            except OSError:
                pass
        try:
            return proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            kill_live_processes()
            try:
                return proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                return proc.poll() if proc.poll() is not None else 1
    finally:
        unregister_live_process(proc)
