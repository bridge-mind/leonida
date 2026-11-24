r"""Build the PC port from your own copy of the game.

    Build.bat                                                   # double-click: finds your copy and builds
    python -I make_game.py                                      # same, from a terminal
    python -I make_game.py --find                               # list the copies found on this PC
    python -I make_game.py --pick                               # choose the game folder in a folder dialog
    python -I make_game.py --game "D:\path\to\your\package" --out "D:\Games\Leonida"
    python -I make_game.py --game "..." --check                 # only check the copy, build nothing
    python -I make_game.py --stages shaders,assets              # only some stages

The game folder is your own Xbox Series package, version 1.0.2.1, with its files already readable on disk
(see game/xbox-1.0.2.1.json for the file list). It is found automatically in Downloads, Desktop, Documents
(also where Windows keeps them, such as in OneDrive), Games, your home folder, the top of each drive on Windows
and the usual mount points elsewhere (/mnt, /media, /run/media, /Volumes, and the per-user /media/<user> and
/run/media/<user> where Linux desktops mount drives), two folder levels down. A share path
(\\server\share\...) can't be listed, so give it to --game. This script never decrypts or works around anything:
it reads plain files and writes
everything it makes under --out. The only thing it downloads is the third_party/rpf9-explorer submodule, when
it is missing. Never share what it makes.

Stages (each skips work that is already done, so a stopped build continues where it left off):
    check     always first: file names and sizes against game/<version>.json (add --full-check for SHA-256, slow)
    shaders   carve shaders.rpf, unpack the native RDNA2 code, disassemble, translate to SPIR-V; then the same for the
              compute programs inside amd_fidelityfx_upscaler_xbox.dll and XboxBvhBuilder.dll
    assets    models (OBJ), texture dictionaries and texture pools (DDS), audio banks (AWC; --wav adds PCM WAV)
              BCPack uses strict decoding by default. For local calibration use <work>/bcpack-calibration.json.
    runtime   build the loader (needs the game's executable to run, not to build). On Linux it is cross-built with
              clang-cl and an xwin SDK (docs/testing.md) and gta6rt.sh starts it with Wine; --runtime-build DIR
              takes a Windows build of runtime/ instead (the runtime-windows artifact of the Runtime workflow)
    pipelines create every translated shader on your GPU's driver as a pipeline, report anything the driver
              rejects or that has to be skipped, and save the pipeline cache next to the port (the runtime doesn't
              load that cache yet)

shaders, assets and runtime do not depend on each other and run together after check. pipelines starts once the
translated shaders are ready and, when this run includes runtime, that stage has finished rebuilding
vk_pipeline_check (a checker already on disk is enough only when runtime is not selected), so it can overlap
the rest of assets. When shaders and assets share the machine, --jobs is split between them and passed to
rpf9dump; a solo assets run still lets rpf9dump use every core.

Long steps log percent and a rough ETA (shader translation shards, asset exports, XMA decode, and overall
weighted stage completion while they overlap).

On an interactive terminal, a LEONIDA setup screen asks for --jobs and whether to overlap stages, then a
key-switchable per-stage log view (1-4 / Tab) keeps parallel output demuxed; full logs land in <work>/logs/.
Pass --no-tui (or redirect stdout) for the plain interleaved console.
"""
import argparse
import functools
import glob
import hashlib
import json
import ntpath
import os
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__" and sys.version_info < (3, 12):   # before code an older Python can't run
    sys.exit("ERROR: Python 3.12 or newer is needed: https://www.python.org/downloads/")

ROOT = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(ROOT, "tools")
sys.path.insert(0, TOOLS)
from mc_command import mc_command  # noqa: E402
import leonida_tui  # noqa: E402
from shader_stage_manifest import shader_stems  # noqa: E402
from replace_retry import replace as replace_retry  # noqa: E402

VERSIONS = os.path.join(ROOT, "game")
STAGES = ["check", "shaders", "assets", "runtime", "pipelines"]
WINDOWS_ONLY = ("runtime",)   # off Windows: the clang-cl + xwin cross-build, or --runtime-build
CROSS_TOOLS = ("clang-cl", "lld-link", "llvm-lib", "llvm-rc", "llvm-mt")
CROSS_VARS = {"C_COMPILER": "clang-cl", "CXX_COMPILER": "clang-cl", "LINKER": "lld-link", "AR": "llvm-lib",
              "RC_COMPILER": "llvm-rc", "MT": "llvm-mt"}   # the toolchain's CMAKE_<var> defaults
XWIN_DEFAULT = os.path.join("~", ".local", "share", "xwin-sdk")
RUNTIME_FILES = ("gta6rt.exe", "gta6rt_tls_host.dll")
# Archives scanned for models and texture dictionaries, small to large.
MODEL_ARCHIVES = [
    "common.rpf", "textures.rpf", "data.rpf",
    "pack/story000_restofworld/story000_restofworld.rpf", "pack/shared_minigame000/shared_minigame000.rpf",
    "pack/core_restofworld/core_restofworld.rpf", "pack/core_restofworld001/core_restofworld001.rpf",
    "packs.rpf", "pack/story000/story000.rpf", "levels.rpf",
]


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    board = leonida_tui.current_board()
    if board is not None:
        board.write(leonida_tui.current_stage(), line)
        return
    print(line, flush=True)


def fail(msg: str) -> "None":
    board = leonida_tui.current_board()
    if board is not None:
        stage = leonida_tui.current_stage()
        board.write(stage, f"ERROR: {msg}")
        if stage:
            board.set_status(stage, "failed")
        # Do not print through the board's stderr proxy (would duplicate into the pane).
        sys.exit(1)
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def fmt_duration(seconds: float) -> str:
    """Short wall-clock remaining/elapsed text for progress lines."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m" if sec == 0 else f"{minutes}m {sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


# Rough relative cost of each stage when run alone (docs: translate ~80 min, assets largest on disk,
# pipelines ~2.8 h wall). Used only for overall percent while stages overlap; not a schedule.
STAGE_WEIGHTS = {"shaders": 40, "assets": 45, "runtime": 2, "pipelines": 55}


class Progress:
    """Thread-safe counter that logs percent and ETA. total 0 logs once and stays quiet."""

    def __init__(self, label: str, total: int, *, every_s: float = 5.0):
        self.label = label
        self.total = max(0, int(total))
        self.done = 0
        self.every_s = every_s
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        self._last_log = 0.0
        if self.total == 0:
            log(f"{label}: nothing to do")
        else:
            self._emit(force=True)

    def tick(self, n: int = 1, *, force: bool = False) -> None:
        with self._lock:
            self.done = min(self.total, self.done + max(0, n))
            self._emit(force=force or self.done >= self.total)

    def set(self, done: int, *, force: bool = False) -> None:
        with self._lock:
            self.done = min(self.total, max(0, int(done)))
            self._emit(force=force or self.done >= self.total)

    def _emit(self, *, force: bool) -> None:
        if self.total <= 0:
            return
        now = time.monotonic()
        if not force and (now - self._last_log) < self.every_s and self.done < self.total:
            return
        self._last_log = now
        pct = 100 * self.done // self.total
        elapsed = now - self._t0
        if 0 < self.done < self.total:
            left = elapsed * (self.total - self.done) / self.done
            eta = f", ~{fmt_duration(left)} left"
        elif self.done >= self.total and elapsed >= 1:
            eta = f", {fmt_duration(elapsed)} elapsed"
        else:
            eta = ""
        log(f"{self.label}: {self.done}/{self.total} ({pct}%){eta}")


class StageBoard:
    """Overall build percent from weighted stage completion (stages may finish out of order)."""

    def __init__(self, stages: list[str]):
        self.weights = {s: STAGE_WEIGHTS[s] for s in stages if s in STAGE_WEIGHTS}
        self._done: set[str] = set()
        self._t0 = time.monotonic()
        self._lock = threading.Lock()
        if self.weights:
            self._announce(f"build: 0% · waiting on {', '.join(self.weights)}")

    def _announce(self, text: str) -> None:
        board = leonida_tui.current_board()
        if board is not None:
            board.set_summary(text)
            board.note(text)
        else:
            log(text)

    def complete(self, name: str) -> None:
        with self._lock:
            if name not in self.weights or name in self._done:
                return
            self._done.add(name)
            finished = sum(self.weights[s] for s in self._done)
            total = sum(self.weights.values())
            pct = 100 * finished // total if total else 100
            left = [s for s in self.weights if s not in self._done]
            elapsed = time.monotonic() - self._t0
            if left and finished:
                eta = f", ~{fmt_duration(elapsed * (total - finished) / finished)} left"
            elif not left and elapsed >= 1:
                eta = f", {fmt_duration(elapsed)} elapsed"
            else:
                eta = ""
            rest = f" · still {', '.join(left)}" if left else ""
            self._announce(f"build: {pct}% · {name} done{rest}{eta}")

def require_folder(flag: str, path: str) -> None:
    """Fail before any work when a destination, or its nearest existing parent, is a file: os.makedirs would
    raise a traceback later, and with a separate --work the build reported success without creating --out."""
    probe = path
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return
        probe = parent
    if not os.path.isdir(probe):
        fail(f"{flag} is not a folder: {path}")


def below_normal() -> None:
    """Keep the PC usable while a long build runs."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.argtypes = ()
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)


def run(args: list, hint: str = "", check: bool = True, **kw) -> int:
    """Run a command, logging it. A non-zero exit stops the build unless check is False; returns the exit code."""
    log("  " + " ".join(os.path.basename(str(a)) if i == 0 else str(a) for i, a in enumerate(args)))
    board = leonida_tui.current_board()
    # Capture into the stage pane when the LEONIDA board is up and the caller did not set its own pipes.
    if (board is not None and kw.get("stdout") is None and kw.get("stderr") is None
            and not kw.get("capture_output")):
        cmd = [str(a) for a in args]
        capture = leonida_tui.capture_popen_kwargs()
        # Caller flags (cwd, env, …) merge in; capture owns the stdio / console-detach bits.
        extra = {k: v for k, v in kw.items()
                 if k not in ("stdout", "stderr", "capture_output", "text",
                              "universal_newlines", "encoding", "errors", "creationflags")}
        if "creationflags" in kw and os.name == "nt":
            capture["creationflags"] = capture.get("creationflags", 0) | int(kw["creationflags"])
        proc = subprocess.Popen(cmd, **capture, **extra)
        code = leonida_tui.pump_process(proc, leonida_tui.current_stage(), board)
        if check and code != 0:
            fail(f"{os.path.basename(str(args[0]))} exited with {code}" + (f". {hint}" if hint else ""))
        return code
    r = subprocess.run([str(a) for a in args], **kw)
    if check and r.returncode != 0:
        fail(f"{os.path.basename(str(args[0]))} exited with {r.returncode}" + (f". {hint}" if hint else ""))
    return r.returncode


def py(tool: str, *args) -> None:
    run([sys.executable, "-I", os.path.join(TOOLS, tool), *args])


def tool_available(cmd: str) -> bool:
    """A PATH name, a path, or a whole command line such as 'wsl -e /usr/lib/llvm-21/bin/llvm-mc'."""
    cmd = cmd.strip().strip('"')
    if os.path.isfile(cmd):
        return True
    head = cmd.split()[0].strip('"') if cmd.split() else ""
    return bool(head and (shutil.which(head) or os.path.isfile(head)))


def appdata_hint(cmd: str) -> str:
    """Why a tool under AppData may look missing on Windows, or "".

    `python` from the Python install manager (the app execution alias in WindowsApps) doesn't see folders that other
    programs create under %LOCALAPPDATA% or %APPDATA%; `py`, or the same python.exe started by its full path, does.
    scripts\\build_llvm_mc.ps1 used to install llvm-mc.exe to %LOCALAPPDATA%\\Leonida\\llvm-mc."""
    if os.name != "nt":
        return ""
    path = ntpath.normcase(cmd.strip().strip('"'))
    for var in ("LOCALAPPDATA", "APPDATA"):
        base = os.environ.get(var, "").strip()
        if base and path.startswith(ntpath.normcase(ntpath.join(base, ""))):
            return (f" LLVM_MC points into %{var}%, which `python` from the Python install manager can't see: copy "
                    "llvm-mc.exe out of AppData (scripts\\build_llvm_mc.ps1 now installs it to "
                    "%USERPROFILE%\\.leonida\\llvm-mc), set LLVM_MC to the new path and reopen the terminal, or start "
                    "make_game.py with `py`.")
    return ""


def need(exe: str, hint: str) -> str:
    if not tool_available(exe):
        fail(f"'{exe}' was not found on PATH. {hint}")
    return exe


# ---------------------------------------------------------------- check
def versions() -> list:
    out = []
    for name in sorted(os.listdir(VERSIONS)):
        if name.endswith(".json"):
            with open(os.path.join(VERSIONS, name), encoding="utf-8") as f:
                out.append(json.load(f))
    return out


# Small files checked by size to recognise a version without reading anything large.
PROBES = ("MicrosoftGame.Config", "Catalog000.bin", "appdata0.rpf")


def probe_mismatches(game: str, version: dict) -> list[str]:
    """Describe which probe files prevent a version match, using only filesystem metadata."""
    files = version["files"]
    probes = [name for name in PROBES if name in files]
    if not probes:
        return ["No version probe files are listed in the manifest."]

    mismatches = []
    for name in probes:
        expected_size = files[name]["size"]
        try:
            info = os.stat(os.path.join(game, name))
        except (FileNotFoundError, NotADirectoryError):
            mismatches.append(f"Missing {name}.")
            continue
        except OSError as error:
            mismatches.append(f"Cannot read file metadata for {name} ({error.strerror}).")
            continue

        if not stat.S_ISREG(info.st_mode):
            mismatches.append(f"{name} is not a regular file.")
        elif info.st_size != expected_size:
            mismatches.append(f"Wrong size for {name} ({info.st_size} bytes, expected {expected_size}).")
    return mismatches


def match_version(game: str):
    """The supported version this folder is, or None. Only names and sizes are read."""
    return next((v for v in versions() if not probe_mismatches(game, v)), None)


def load_version(game: str) -> dict:
    reports = []
    for version in versions():
        mismatches = probe_mismatches(game, version)
        if not mismatches:
            return version
        reports.append(f"  {version['title']} {version['version']}\n    " + "\n    ".join(mismatches))
    notes = "".join(f"\n{container_message(p, h)}" for p, h in containers_in(game))
    fail("This doesn't look like a supported version of the game. Point --game at the folder that holds "
         "MicrosoftGame.Config and the .rpf archives.\nSupported:\n"
         + "\n".join(reports or ["  No version manifests found."]) + notes)


# ---------------------------------------------------------------- packaged containers
# The store keeps the game as one packaged container (.xvc), an XVD file whose contents are encrypted. Nothing here
# reads keys or decrypts (docs/design.md, "Ground rules"), so the build can't use it. Pointed at one, or at a folder
# that only holds one, the build reads the plain 4 KiB header to say what it is instead of "not found".
# Offsets are from the start of the file: xvdtool's XvdHeader + 0x200, the signature in front of it.
XVD_MAGIC = b"msft-xvd"
XVD_HEADER_SIZE = 0x1000
XVD_ENCRYPTION_DISABLED = 0x2          # VolumeFlags bit
CONTAINER_SUFFIXES = (".xvc", ".msixvc", ".xvd")
VOLUME_ROOTS = ("/mnt", "/media", "/run/media", "/Volumes")   # POSIX: where drives and shares are mounted
# udisks2, which mounts removable and internal drives on Linux desktops, puts a drive at <root>/<user>/<label>: one
# level more than a drive letter, so these roots are also searched from the user's own folder.
USER_MOUNT_ROOTS = ("/media", "/run/media")


def volume_roots() -> list:
    """POSIX mount points find_dumps() starts from: VOLUME_ROOTS, then /media/<user> and /run/media/<user>, so a copy
    in a folder on a udisks2-mounted drive (/media/<user>/<label>/GTA VI) is two levels down, as D:\\Games\\GTA VI
    is on Windows."""
    roots = list(VOLUME_ROOTS)
    try:
        import getpass
        user = getpass.getuser()
    except Exception:   # no USER/LOGNAME and no passwd entry (a bare container): the shared roots still work
        user = ""
    if user and os.sep not in user and user not in (".", ".."):
        roots += [os.path.join(root, user) for root in USER_MOUNT_ROOTS]
    return roots


def xvd_header(path: str):
    """The few XVD header fields the build reports, or None when the file isn't an XVD. Reads 4 KiB, nothing else."""
    try:
        with open(path, "rb") as f:
            b = f.read(XVD_HEADER_SIZE)
    except OSError:
        return None
    if len(b) < XVD_HEADER_SIZE or b[0x200:0x208] != XVD_MAGIC:
        return None
    flags, = struct.unpack_from("<I", b, 0x208)
    content_type, = struct.unpack_from("<I", b, 0x284)
    xvc_data_length, = struct.unpack_from("<I", b, 0x290)
    version = struct.unpack_from("<4H", b, 0x3BC)        # stored lowest part first: 1,2,0,1 is 1.0.2.1
    return {"kind": "XVC" if xvc_data_length else "XVD",
            "encrypted": not flags & XVD_ENCRYPTION_DISABLED,
            "content_type": content_type,
            "version": ".".join(str(p) for p in reversed(version))}


def container_message(path: str, h: dict) -> str:
    """Why the build can't use a packaged container, and what to give it instead."""
    state = "Its files are encrypted inside it" if h["encrypted"] else "Its files are packed inside it"
    return (f"{path} is a packaged {h['kind']} container (package version {h['version']}), not a game folder. "
            f"{state}, and this project doesn't read keys, decrypt or unpack containers, so the build can't use "
            "it. Give --game the folder that holds MicrosoftGame.Config and the .rpf archives.")


def containers_in(folder: str) -> list:
    """(path, header) for each packaged container directly in folder; only files with a container suffix are read."""
    try:
        with os.scandir(folder) as it:
            names = sorted(e.path for e in it if e.name.lower().endswith(CONTAINER_SUFFIXES)
                           and e.is_file(follow_symlinks=False))
    except OSError:
        return []
    return [(p, h) for p in names if (h := xvd_header(p))]


# Windows known-folder IDs (KnownFolders.h): Downloads, Desktop, Documents. OneDrive's Known Folder Move puts
# Desktop and Documents in ~\OneDrive, and each can be moved anywhere from its Properties > Location tab, so the
# folders Explorer shows aren't always the ones under the home folder.
KNOWN_FOLDERS = ("{374DE290-123F-4565-9164-39C4925E467B}", "{B4BFCC3A-DB2C-424C-B029-7FE99A87C641}",
                 "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}")


def known_folders() -> list:
    """Windows: where Downloads, Desktop and Documents really are (SHGetKnownFolderPath). Empty elsewhere, and
    without the ones the shell can't report."""
    if os.name != "nt":
        return []
    import ctypes
    import uuid
    from ctypes import wintypes
    try:
        shell32, ole32 = ctypes.WinDLL("shell32"), ctypes.WinDLL("ole32")
    except OSError:
        return []
    get = shell32.SHGetKnownFolderPath
    get.argtypes = [ctypes.c_char_p, wintypes.DWORD, wintypes.HANDLE, ctypes.POINTER(ctypes.c_void_p)]
    get.restype = ctypes.c_long
    free = ole32.CoTaskMemFree
    free.argtypes = [ctypes.c_void_p]
    free.restype = None
    found = []
    for folder in KNOWN_FOLDERS:
        path = ctypes.c_void_p()
        result = get(uuid.UUID(folder).bytes_le, 0, None, ctypes.byref(path))
        try:
            if result == 0 and path.value:
                found.append(ctypes.wstring_at(path.value))
        finally:
            free(path)   # the shell's buffer is freed whether or not the call succeeded
    return found


def discovery_roots(home: str) -> list:
    r"""Where find_dumps() starts looking. Windows: the known folders, the home folder and every drive letter. Else:
    the same folders plus the usual mount points, because that is where a copy on another disk or on a network share
    shows up (/Volumes is where macOS mounts both; Linux desktops mount a drive at /media/<user>/<label> or, on Fedora
    and openSUSE, /run/media/<user>/<label>, so the per-user folders are roots too).
    A Windows share path (\\server\share\...) can't be listed at all; --game takes it directly."""
    roots = [os.path.join(home, d) for d in ("Downloads", "Desktop", "Documents", "Games")] + known_folders() + [home]
    if os.name == "nt":
        import string
        roots += [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
    else:
        roots += volume_roots()
    return roots


def find_dumps(containers: list | None = None) -> list:
    """Folders on this PC that hold a supported version: the folders discovery_roots() names (Downloads, Desktop,
    Documents under the home folder and where Windows really keeps them, the home folder, each drive on Windows, the
    usual mount points elsewhere), up to two folder levels down.
    Only folder names and a few file sizes are read. With a containers list, also appends (path, header) for each
    packaged container (.xvc/.msixvc/.xvd) in the folders searched, which the build can't use: only the 4 KiB header
    of a file with that suffix is read."""
    home = os.path.expanduser("~")
    roots = discovery_roots(home)
    skip = {"windows", "program files", "program files (x86)", "programdata", "appdata", "$recycle.bin",
            "system volume information", "node_modules", ".git"}
    visited_depths, found, seen_containers = {}, {}, set()

    def visit(d: str, depth: int) -> None:
        try:
            real = os.path.normcase(os.path.realpath(d))
        except OSError:
            return
        previous_depth = visited_depths.get(real, -1)
        if real in found or previous_depth >= depth:
            return
        # A later root can reach the same folder with more levels left to search.
        visited_depths[real] = depth
        if previous_depth < 0 and os.path.isfile(os.path.join(d, "MicrosoftGame.Config")):
            v = match_version(d)
            if v:
                found[real] = (d, v)
                return
        if depth == 0:
            return
        try:
            with os.scandir(d) as it:
                entries = list(it)
        except OSError:
            return
        subs = []
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    if e.name.lower() not in skip and not e.name.startswith((".", "$")):
                        subs.append(e.path)
                elif containers is not None and e.name.lower().endswith(CONTAINER_SUFFIXES) \
                        and e.is_file(follow_symlinks=False):
                    key = os.path.normcase(os.path.realpath(e.path))
                    if key not in seen_containers and (h := xvd_header(e.path)):
                        seen_containers.add(key)
                        containers.append((e.path, h))
            except OSError:
                continue
        for s in subs:
            visit(s, depth - 1)

    for r in roots:
        if os.path.isdir(r):
            visit(r, 2)
    return list(found.values())


PICK_TITLE = "Choose the folder that holds MicrosoftGame.Config"
PICK_UNSUPPORTED = ("That folder isn't a supported version of the game. Choose the folder that holds "
                    "MicrosoftGame.Config and the .rpf archives.")


def native_picker():
    """The desktop's own folder dialog on Linux and the BSDs, where Tk's is a bare one of its own: kdialog on
    KDE, zenity elsewhere (GNOME and the rest), either one when only that one is installed. Returns
    (tool, path) or None: on Windows and macOS Tk already opens the system dialog, and with no display there
    is nothing to open."""
    if os.name == "nt" or sys.platform == "darwin":
        return None
    if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        return None
    kde = "KDE" in os.environ.get("XDG_CURRENT_DESKTOP", "").upper()
    for tool in (("kdialog", "zenity") if kde else ("zenity", "kdialog")):
        exe = shutil.which(tool)
        if exe:
            return tool, exe
    return None


def native_ask(tool: str, exe: str, start: str) -> str:
    """One folder dialog from `tool`: the chosen folder, or "" on cancel. OSError when the tool itself fails."""
    if tool == "kdialog":
        cmd = [exe, "--title", "Leonida", "--getexistingdirectory", start]
    else:
        cmd = [exe, "--file-selection", "--directory", "--title=" + PICK_TITLE,
               "--filename=" + os.path.join(start, "")]
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    if r.returncode == 1:       # both tools: the dialog was cancelled or closed
        return ""
    if r.returncode != 0:
        raise OSError(f"{tool} exited with {r.returncode}")
    return r.stdout.strip("\r\n")


def native_warn(tool: str, exe: str, msg: str) -> None:
    cmd = ([exe, "--title", "Leonida", "--sorry", msg] if tool == "kdialog"
           else [exe, "--warning", "--title=Leonida", "--text=" + msg])
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def pick_game() -> str:
    """--pick: choose the game folder in the system's folder dialog. Asks again until the folder is a supported
    version; cancelling stops the build."""
    native = native_picker()
    if native:
        start = os.path.expanduser("~")
        try:
            while True:
                d = native_ask(*native, start)
                if not d:
                    fail("No folder chosen: choose one with --pick or --game.")
                d = os.path.normpath(d)
                v = match_version(d)
                if v:
                    log(f"chose {v['title']} {v['version']} in {d}")
                    return d
                notes = "".join(f"\n\n{container_message(p, h)}" for p, h in containers_in(d))
                native_warn(*native, PICK_UNSUPPORTED + notes)
                start = d           # open the next dialog where the last one left off
        except OSError as e:        # the tool is there but won't run (no portal, broken install): try Tk
            log(f"the {native[0]} folder dialog couldn't open ({e}); using Tk's")
    try:
        import tkinter
        from tkinter import filedialog, messagebox
        root = tkinter.Tk()
    except Exception as e:      # no tkinter, or no display (SSH, headless Linux)
        fail(f"The folder dialog couldn't open ({e}). Give the folder with --game.")
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        while True:
            d = filedialog.askdirectory(parent=root, mustexist=True, title=PICK_TITLE)
            if not d:
                fail("No folder chosen: choose one with --pick or --game.")
            d = os.path.normpath(d)
            v = match_version(d)
            if v:
                log(f"chose {v['title']} {v['version']} in {d}")
                return d
            notes = "".join(f"\n\n{container_message(p, h)}" for p, h in containers_in(d))
            messagebox.showwarning("Leonida", PICK_UNSUPPORTED + notes, parent=root)
    finally:
        root.destroy()


def choose_game() -> str:
    """No --game given: find the dump, or ask for it."""
    log("looking for your game files...")
    containers = []
    found = find_dumps(containers)
    interactive = sys.stdin is not None and sys.stdin.isatty()
    if len(found) == 1:
        log(f"found {found[0][1]['title']} {found[0][1]['version']} in {found[0][0]}")
        return found[0][0]
    if found:
        for i, (d, v) in enumerate(found, 1):
            print(f"  {i}. {d}  ({v['title']} {v['version']})")
        if not interactive:
            fail("More than one copy found: choose one with --game.")
        while True:
            try:
                pick = input("Which one? ").strip()
            except EOFError:   # Ctrl+Z / Ctrl+D, or a console that gives no input
                fail("No copy chosen: choose one with --game.")
            if pick.isdigit() and 1 <= int(pick) <= len(found):
                return found[int(pick) - 1][0]
    notes = "".join(f"\n{container_message(p, h)}" for p, h in containers)
    if not interactive:
        fail("Your game files weren't found. Give the folder with --game." + notes)
    print("Your game files weren't found automatically. Paste the folder that holds MicrosoftGame.Config "
          "and the .rpf archives." + notes)
    while True:
        try:
            d = input("Game folder: ").strip().strip('"')
        except EOFError:
            fail("No folder given. Give the folder with --game.")
        if d and match_version(d):
            return d
        h = xvd_header(d) if d and os.path.isfile(d) else None
        pasted = [(d, h)] if h else containers_in(d) if d else []
        for p, ph in pasted:
            print(f"  {container_message(p, ph)}")
        print("  That folder isn't a supported version of the game. Try again.")


def icloud_documents() -> bool:
    """macOS: whether iCloud Drive holds Desktop & Documents. With that sync on, ~/Documents lives in iCloud Drive,
    which uploads what goes in it and may evict files from the disk ("Optimize Mac Storage"). The folder can stay in
    iCloud Drive after the sync is turned off; leaving Documents alone then costs nothing."""
    return sys.platform == "darwin" and os.path.isdir(
        os.path.join(os.path.expanduser("~"), "Library", "Mobile Documents", "com~apple~CloudDocs", "Documents"))


def default_out() -> str:
    """Documents/Leonida, except on a Mac whose Documents iCloud syncs: the port and its work folder (hundreds of
    GB made from the game) must stay on this computer, so they go to ~/Leonida there."""
    if icloud_documents():
        return os.path.join(os.path.expanduser("~"), "Leonida")
    return os.path.join(os.path.expanduser("~"), "Documents", "Leonida")


def icloud_default_notes() -> list:
    """The log lines when default_out() left iCloud-synced Documents: why, and where a build made before this change
    is, so its resumable work folder isn't silently left behind."""
    notes = ["iCloud syncs Documents on this Mac: building into ~/Leonida instead, so the game's files stay off iCloud"]
    earlier = os.path.join(os.path.expanduser("~"), "Documents", "Leonida")
    if os.path.isdir(earlier):
        notes.append(f"an earlier build is in {earlier}: move it to {default_out()} to resume it there, "
                     f'or pass --out "{earlier}" to keep using it')
    return notes


# ---------------------------------------------------------------- preflight
# GiB (free bytes / 2**30), the unit Windows Explorer shows as "GB"; 370 GiB is about 397 GB in decimal units.
MIN_FREE_GB = {"shaders": 30, "assets": 370}


def assets_exported(work: str) -> bool:
    """Whether an earlier run finished the assets stage's exports, so its space is already in use. xma.done is the
    last marker assets() writes, after every model, texture and audio export before it succeeded; ogg.done is read
    too. A marker counts only when it holds what mark() writes, not because a file of that name exists.
    xma.done is written with or without ffmpeg, and every run decodes the XMA2 streams that have no .wav yet once
    ffmpeg is on PATH. So with ffmpeg there, a stream still waiting for its .wav means the stage will write again.
    Models, textures and pools made by another decoder (DECODER_RECORD) don't change the answer: assets() removes each
    such folder before it exports it again, so the new export goes into the space the old one held."""
    audio = os.path.join(work, "assets", "audio")
    for name, what in (("ogg.done", "awc_opus.py"), ("xma.done", "rpf9dump xma")):
        try:
            with open(os.path.join(audio, name), encoding="utf-8") as f:
                if f.read().strip() != f"{what} finished":
                    return False
        except (OSError, UnicodeError):
            return False
    return not (shutil.which("ffmpeg") and pending_xma(os.path.join(audio, "xma")))


def _linked(path: str) -> bool:
    """A symbolic link or an NTFS junction: what it points at may be on another drive, so it is never counted."""
    return os.path.islink(path) or os.path.isjunction(path)


def _tree_bytes(root: str, skip: tuple = (), ignore_suffix: str = "") -> int:
    """Bytes of the regular files under root, without following links or junctions (root's own names in skip and
    files ending in ignore_suffix are left out). One os.scandir walk; on Windows the sizes come with the directory listing, so no file is opened."""
    if _linked(root):
        return 0
    total, stack = 0, [(root, True)]
    while stack:
        folder, top = stack.pop()
        try:
            with os.scandir(folder) as it:
                for e in it:
                    try:
                        if e.is_symlink() or e.is_junction() or (top and e.name in skip):
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append((e.path, False))
                        elif e.is_file(follow_symlinks=False) and not (ignore_suffix and e.name.endswith(ignore_suffix)):
                            total += e.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:   # missing, or can't be listed: count nothing for it
            continue
    return total


def _finished_pool_dirs(top: str) -> list:
    """Texture pool folders under assets/ttpf or assets/ttpf_pack whose manifest ttpf_manifest() accepts, the check
    assets() uses to skip a pool. A folder with a manifest is a pool: its subfolders are not searched."""
    found, stack = [], [top]
    while stack:
        folder = stack.pop()
        if _linked(folder):
            continue
        if os.path.isfile(os.path.join(folder, "manifest.json")):
            if ttpf_manifest(folder) is not None:
                found.append(folder)
            continue
        try:
            with os.scandir(folder) as it:
                stack += [e.path for e in it if e.is_dir(follow_symlinks=False)]
        except OSError:
            continue
    return found


def assets_bytes_on_disk(work: str) -> int:
    """Bytes a stopped assets stage has already written and a rerun will keep, so they are not needed free again.
    A model or texture folder counts, finished or not, when its decoder.sha256 matches the current decoder:
    assets() deletes a folder with another or no fingerprint and exports it again, but for a matching one rpf9dump
    skips every output that already exists (Util.WriteAtomic only renames complete files into place; a leftover
    *.partial is deleted and written again, so it is not counted). Pools, audio and the other steps count only when
    finished (a pool whose manifest ttpf_manifest() accepts, audio_manifest.json, ogg.done, xma.done); the unfinished
    part of those is not counted, which is conservative."""
    out = os.path.join(work, "assets")
    if _linked(out):
        return 0
    total = 0
    # Before ensure_submodule() has fetched the submodule, the digest has no git pin and differs from the one assets()
    # will compute (a source zip with a non-empty Core has no pin either, and is ready). Credit none until then.
    decoder = None if submodule_ready() else ""
    for rel in MODEL_ARCHIVES:
        name = os.path.splitext(os.path.basename(rel))[0]
        for mode in ("models", "textures"):
            folder = os.path.join(out, name, mode)
            record = os.path.join(folder, DECODER_RECORD)
            if not os.path.isfile(record):
                continue   # no fingerprint: assets() exports this folder again, so none of it is kept
            if decoder is None:   # read lazily: a work folder without exports never hashes the decoder
                try:   # not the cached decoder_digest(): the submodule may not be fetched yet (ensure_submodule)
                    decoder = compute_decoder_digest()
                except OSError:
                    decoder = ""   # can't tell what the exports came from: credit none of them
            if decoder and read_shader_setting(record) == decoder:
                total += _tree_bytes(folder, ignore_suffix=".partial")
    for top in ("ttpf", "ttpf_pack"):
        total += sum(_tree_bytes(d) for d in _finished_pool_dirs(os.path.join(out, top)))
    audio = os.path.join(out, "audio")
    if os.path.isfile(os.path.join(audio, "audio_manifest.json")):
        total += _tree_bytes(audio, skip=("ogg", "xma", "wav"))
    for step in ("ogg", "xma"):
        if os.path.isfile(os.path.join(audio, f"{step}.done")):
            total += _tree_bytes(os.path.join(audio, step))
    return total


def pipeline_check_prerequisite(runtime_selected: bool = False) -> str:
    """Why the pipelines stage can't run yet. Kept here so preflight can fail before a long shader build;
    pipelines() prints its own, more detailed message when the stage is reached. On Windows the runtime stage
    only builds the checker when VULKAN_SDK is set (scripts/build_runtime.ps1)."""
    if runtime_selected:
        return ("VULKAN_SDK (the runtime stage builds vk_pipeline_check only when it is set, and the pipelines "
                "stage needs that executable)")
    return ("vk_pipeline_check (needed by the pipelines stage; run the runtime stage on Windows, or build it "
            "with CMake as described in README.md)")


def preflight(stages: list, work: str, wav: bool = False, runtime_build: str | None = None, *,
              emulate: bool = False, keep_emulated: bool = False,
              fallbacks: tuple[str, ...] = (), jobs: int = 1) -> None:
    """Find every missing prerequisite before a long build starts, and list them all at once."""
    if "runtime" in stages and runtime_build:
        absent = [n for n in RUNTIME_FILES if not os.path.isfile(os.path.join(runtime_build, n))]
        if absent:
            fail(f"--runtime-build: {runtime_build} has no {' or '.join(absent)}")
    missing = []
    if wav and "assets" in stages and not shutil.which("ffmpeg"):
        missing.append("ffmpeg (needed for --wav): https://ffmpeg.org/download.html, then reopen the terminal")
    if sys.version_info < (3, 12):
        missing.append("Python 3.12 or newer: https://www.python.org/downloads/")
    if "shaders" in stages:
        mc = mc_command()[0]
        if not (os.path.isfile(mc) or shutil.which(mc)):
            missing.append(
                "llvm-mc (LLVM 21+ with the AMDGPU target). The official Windows LLVM build has no AMDGPU "
                "target: build a native one with scripts\\build_llvm_mc.ps1, or install llvm-21 in WSL and set "
                "LLVM_MC to the whole command, e.g. "
                "LLVM_MC=\"wsl -d Ubuntu -e /usr/lib/llvm-21/bin/llvm-mc\" (docs/design.md)." + appdata_hint(mc))
        for t in ("spirv-as", "spirv-val"):
            if not shutil.which(t):
                missing.append(f"{t}: install the Vulkan SDK, https://vulkan.lunarg.com/ (then reopen the terminal)")
    if "assets" in stages:
        if not dotnet_sdk_ok():
            missing.append(".NET 8 SDK (the .NET runtime alone can't build rpf9dump): "
                           "https://dotnet.microsoft.com/download/dotnet/8.0")
    if os.name == "nt" and "runtime" in stages and not runtime_build and not visual_cpp_tools_ok():
        missing.append("Visual Studio 2022 with 'Desktop development with C++'")
    if os.name != "nt" and not runtime_build and any(s in WINDOWS_ONLY for s in stages):
        missing.extend(f"{s} (for the runtime stage, or pass --runtime-build)" for s in cross_build_missing())
    # The pipelines stage needs a checker the runtime stage builds (on Windows only with VULKAN_SDK set).
    # Without this check a build with shaders and pipelines spends hours before failing at the very end.
    if "pipelines" in stages and not pipeline_check_exe(runtime_build):
        if os.name != "nt":
            if not shutil.which("cmake"):
                missing.append("CMake 3.24 or newer, a C++20 compiler and the Vulkan headers, to build "
                               "vk_pipeline_check for the pipelines stage")
        else:
            runtime_building = "runtime" in stages and not runtime_build
            if not runtime_building or not os.environ.get("VULKAN_SDK"):
                missing.append(pipeline_check_prerequisite(runtime_building))
    if missing:
        fail("Install these first, then run the build again:\n  - " + "\n  - ".join(dict.fromkeys(missing)))
    if "pipelines" in stages and "shaders" not in stages:
        require_translated_shaders(work)
    need = {s: MIN_FREE_GB.get(s, 0) for s in stages}
    # A rerun that only refreshes the shaders shouldn't need room for a second copy of exports already on disk.
    # --wav still counts it: the WAV decode writes several times the Ogg Opus size and has no marker.
    if need.get("assets") and not wav and assets_exported(work):
        log(f"preflight: the assets are already exported in {work}; not counting their {need['assets']} GB")
        need["assets"] = 0
    os.makedirs(work, exist_ok=True)
    free_gb = shutil.disk_usage(work).free / 2**30
    need_gb = sum(need.values())
    if free_gb < need_gb and need.get("shaders") and shader_outputs_reusable(work, emulate, keep_emulated, fallbacks):
        log(f"preflight: the shaders in {work} can be reused; not counting their {need['shaders']} GB")
        need_gb -= need["shaders"]
    # An export stopped part-way (no xma.done yet) has already put some of the 370 GB on the drive, and the rerun
    # skips the steps it finished. Only walked when the space is short, so a build with room to spare reads no
    # folder.
    # --wav still counts the whole estimate, as above: the WAV decode writes several times the Ogg Opus size and has
    # no marker, so what is on disk says nothing about what it will still write.
    if free_gb < need_gb and need.get("assets") and not wav:
        done_gb = min(assets_bytes_on_disk(work) // 2**30, need["assets"])
        if done_gb:
            log(f"preflight: {done_gb} GB of a stopped asset export is already in {work}; not counting it again")
            need_gb -= done_gb
    if free_gb < need_gb:
        hint = ""
        assets_gb = need.get("assets", 0)
        loader_stages_selected = "shaders" in stages and "runtime" in stages
        if loader_stages_selected and not wav and assets_gb and free_gb >= need_gb - assets_gb:
            hint = " To prepare the loader without exporting assets, use --stages check,shaders,runtime."
        fail(f"{free_gb:.0f} GB free on the drive of {work}, about {need_gb} GB is needed. "
             "Free some space or choose another drive with --work." + hint)
    if os.name != "nt" and not runtime_build and any(s in WINDOWS_ONLY for s in stages):
        cross_configure()
    if "pipelines" in stages and os.name != "nt" and not pipeline_check_exe():
        build_pipeline_check(jobs)
        if not pipeline_check_exe():
            fail("vk_pipeline_check built, but not as runtime/build/linux/vk_pipeline_check (a multi-config "
                 "generator puts it in a subfolder): unset CMAKE_GENERATOR if it names one, delete "
                 "runtime/build/linux and run the build again.")
    if "assets" in stages:
        ensure_submodule()


def visual_cpp_tools_ok() -> bool:
    """Look for the x64 C++ tools required by scripts/build_runtime.ps1."""
    vswhere = os.path.join(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
                           "Microsoft Visual Studio", "Installer", "vswhere.exe")
    if not os.path.isfile(vswhere):
        return False
    try:
        result = subprocess.run([vswhere, "-latest", "-products", "*", "-requires",
                                 "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                 "-property", "installationPath"], capture_output=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def windows_powershell() -> str:
    """Windows PowerShell 5.1 by its full path, to start scripts/build_runtime.ps1.

    CreateProcess searches System32, but powershell.exe is in System32\\WindowsPowerShell\\v1.0, so the bare name is
    found only while that folder is on PATH. On a PC whose PATH had lost it the runtime stage ended in a
    FileNotFoundError traceback, after the shader and asset stages had already run. The bare name is still the
    answer where the file is not at its usual place.
    """
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    exe = os.path.join(root, "System32", "WindowsPowerShell", "v1.0", "powershell.exe")
    return exe if os.path.isfile(exe) else "powershell"


def xwin_dir() -> "str | None":
    """The xwin /winsysroot splat for the cross-build: XWIN_DIR, else the folder docs/testing.md uses."""
    d = os.path.expanduser(os.environ.get("XWIN_DIR") or XWIN_DEFAULT)
    sdk = glob.glob(os.path.join(glob.escape(d), "Windows Kits", "10", "Lib", "*", "um", "x64"))
    return d if os.path.isdir(os.path.join(d, "VC", "Tools", "MSVC")) and sdk else None


CROSS_SUFFIXES = ("", *(f"-{n}" for n in range(40, 18, -1)))


def cross_tool_suffix() -> "str | None":
    """"" for the plain LLVM tool names, "-N" for a set named like clang-cl-21: the first complete set whose
    clang-cl is 19 or newer, plain names first. None if there is none."""
    for suffix in CROSS_SUFFIXES:
        if all(shutil.which(t + suffix) for t in CROSS_TOOLS):
            major = clang_cl_major("clang-cl" + suffix)
            if major is None or major >= 19:
                return suffix
    return None


def cross_build_dir() -> str:
    return os.path.join(ROOT, "runtime", "build", "windows")


def configured_cross_tools() -> dict:
    """The LLVM tools an existing runtime/build/windows was configured with, from CMake's compiler files."""
    found = {}
    pattern = os.path.join(glob.escape(cross_build_dir()), "CMakeFiles", "*", "CMake*Compiler.cmake")
    for path in sorted(glob.glob(pattern), key=os.path.getmtime):
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        for m in re.finditer(r'^set\((CMAKE_(?:CXX_COMPILER|LINKER|AR|MT|RC_COMPILER)) "([^"]+)"\)', text, re.M):
            found[m.group(1)] = m.group(2)
    return found


def clang_cl_major(exe: str = "clang-cl") -> "int | None":
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"clang version (\d+)", r.stdout)
    return int(m.group(1)) if m else None


def cross_build_missing() -> list:
    """What the clang-cl + xwin cross-build of runtime/ (runtime/cmake/clang-cl-xwin.cmake) lacks here."""
    missing = []
    if not shutil.which("cmake"):
        missing.append("CMake 3.24 or newer")
    configured = configured_cross_tools()
    if configured:
        gone = sorted({t for t in configured.values() if not shutil.which(t)})
        if gone:
            missing.append(f"{', '.join(gone)}, which runtime/build/windows was configured with: put it back on "
                           "PATH, or delete runtime/build/windows to configure it with the tools on PATH now")
    elif cross_tool_suffix() is None:
        major = clang_cl_major() if all(shutil.which(t) for t in CROSS_TOOLS) else None
        partial = next((s for s in CROSS_SUFFIXES[1:] if shutil.which("clang-cl" + s)), "")
        absent = [t + partial for t in CROSS_TOOLS if not shutil.which(t + partial)]
        if major is not None and not absent:
            missing.append(f"clang-cl 19 or newer (this one is {major}; the MSVC STL from xwin rejects older Clang)")
        else:
            missing.append(f"{', '.join(absent)}: LLVM 19 or newer (clang, lld and llvm), all five by their plain "
                           "names or with one version suffix such as clang-cl-21")
    if not xwin_dir():
        missing.append("an xwin splat made with `xwin splat --use-winsysroot-style --preserve-ms-arch-notation` "
                       "(VC/Tools/MSVC and Windows Kits/10/Lib/<version>/um/x64) in XWIN_DIR or "
                       "~/.local/share/xwin-sdk, see docs/testing.md, \"The Windows runtime on Linux\"")
    return missing


def dotnet_sdk_ok() -> bool:
    """True if a .NET SDK 8 or newer is installed. `dotnet` is also on PATH with only a runtime, and then
    `dotnet build` fails with "No .NET SDKs were found"."""
    if not shutil.which("dotnet"):
        return False
    try:
        r = subprocess.run(["dotnet", "--list-sdks"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return any(int(m) >= 8 for m in re.findall(r"^(\d+)\.", r.stdout, re.M))


def submodule_ready() -> bool:
    """Whether third_party/rpf9-explorer is checked out (a non-empty Core): what ensure_submodule() requires and
    what decoder_digest() is only meaningful after."""
    core = os.path.join(ROOT, "third_party", "rpf9-explorer", "Core")
    return os.path.isdir(core) and bool(os.listdir(core))


def ensure_submodule() -> None:
    core = os.path.join(ROOT, "third_party", "rpf9-explorer", "Core")
    if submodule_ready():
        return
    clone = "git clone --recursive https://github.com/ItsNotRexed/Leonida"
    git = shutil.which("git")
    if not git:
        fail("third_party/rpf9-explorer is missing (a \"Download ZIP\" from GitHub leaves it out). Install Git "
             f"(https://git-scm.com/downloads) and download the project with: {clone}")
    if subprocess.run([git, "-C", ROOT, "rev-parse", "--resolve-git-dir", ".git"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode != 0:
        fail("third_party/rpf9-explorer is missing (a \"Download ZIP\" from GitHub leaves it out). "
             f"Download the project with: {clone}")
    log("fetching third_party/rpf9-explorer")
    r = subprocess.run([git, "-C", ROOT, "submodule", "update", "--init", "--recursive"])
    if r.returncode == 0 and os.path.isdir(core):
        return
    fail("fetching third_party/rpf9-explorer failed. Check your network connection, then run "
         "\"git submodule update --init --recursive\" in this folder and build again.")


def check(game: str, full: bool) -> dict:
    if not os.path.isdir(game):
        h = xvd_header(game) if os.path.isfile(game) else None
        fail(container_message(game, h) if h else f"Game folder not found: {game}")
    v = load_version(game)
    log(f"check: {v['title']} {v['version']}, {len(v['files'])} files")
    bad = 0
    for rel, info in v["files"].items():
        p = os.path.join(game, rel)
        if not os.path.isfile(p):
            print(f"  missing  {rel}"); bad += 1
        elif os.path.getsize(p) != info["size"]:
            print(f"  wrong size  {rel}"); bad += 1
    if bad:
        fail(f"{bad} file(s) missing or different. Use a complete, unmodified copy of version {v['version']}.")
    if full:
        py("manifest.py", "verify", game, "--manifest", os.path.join(VERSIONS, f"xbox-{v['version']}.json"))
    exe = os.path.join(game, v["expected_exe"])
    log(f"check: all files present{' and hashes match' if full else ''}. "
        + ("Executable found." if os.path.isfile(exe) else
           f"{v['expected_exe']} is not in the folder: the runtime stage can be built but not run."))
    return v


# ---------------------------------------------------------------- shaders
def read_shader_setting(path: str, default: str = "") -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read().strip()
    except OSError:
        return default


def requested_shader_mode(had: str, emulate: bool, keep_emulated: bool) -> str:
    return "on" if emulate or (had == "on" and keep_emulated) else "off"


def translator_flags_setting(fallbacks: tuple[str, ...]) -> str:
    """The recorded form of the translator fallback flags in spirv/translator_flags.txt (empty means none)."""
    return " ".join(fallbacks)


def llvm_mc_signature() -> str:
    command = mc_command()
    command[0] = shutil.which(command[0]) or command[0]
    try:
        result = subprocess.run([*command, "--version"], capture_output=True, text=True, errors="replace", timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        fail(f"Could not read the llvm-mc version. Check LLVM_MC: {error}")
    version = result.stdout.strip()
    if result.returncode != 0 or not version:
        fallback = f"exit code {result.returncode}" if result.returncode else "no version printed"
        detail = (result.stderr.strip().splitlines() or [fallback])[0]
        fail(f"Could not read the llvm-mc version. Check LLVM_MC: {detail}")
    return json.dumps([command, version])


def unpacked_shader_stems(dxbc: str) -> set[str] | None:
    """Return current unpacked names, allowing the unpacker to skip non-FSSM containers."""
    try:
        carved = shader_stems(os.path.join(dxbc, "index.json"), "file", ".dxbc")
        unpacked = shader_stems(os.path.join(dxbc, "fssm_index.json"), "name")
    except (OSError, ValueError):
        return None
    return unpacked if unpacked and unpacked <= carved else None


def disassembly_listings(dxbc: str) -> set[str] | None:
    try:
        with os.scandir(os.path.join(dxbc, "disasm")) as entries:
            listings = set()
            for entry in entries:
                if not entry.name.endswith(".s"):
                    continue
                if entry.is_symlink():
                    return None
                if entry.is_file() and entry.stat().st_size > 0:
                    listings.add(entry.name[:-2])
            return listings
    except OSError:
        return None


def disassembly_status(dxbc: str, signature: str) -> tuple[bool, bool, set[str]]:
    """Report readiness, whether existing listings can be reused, and any repair inventory."""
    done = read_shader_setting(os.path.join(dxbc, "disasm.done"))
    marker = done or read_shader_setting(os.path.join(dxbc, "disasm.started"))
    inactive = (False, False, set())
    if not marker:
        return inactive
    active = unpacked_shader_stems(dxbc)
    listings = disassembly_listings(dxbc)
    if active is None or listings is None or not listings <= active:
        return inactive
    if marker == signature:
        # Older completion markers recorded only LLVM. Require a listing for every input.
        return (listings == active, False, set()) if done else (False, True, set())
    try:
        metadata = json.loads(marker)
    except ValueError:
        return inactive
    if not isinstance(metadata, dict) or metadata.get("llvm") != signature:
        return inactive
    recorded = metadata.get("listings")
    if not isinstance(recorded, list) or not recorded or not all(isinstance(name, str) for name in recorded):
        return inactive
    expected = set(recorded)
    if not listings <= expected <= active:
        return inactive
    return bool(done) and listings == expected, True, expected


def shader_outputs_reusable(work: str, emulate: bool, keep_emulated: bool,
                            fallbacks: tuple[str, ...] = ()) -> bool:
    """Check whether the shader stage can resume without carving, unpacking or translating again."""
    sh = os.path.join(work, "shaders")
    dxbc = os.path.join(sh, "dxbc")
    spirv = os.path.join(sh, "spirv")
    markers = (os.path.join(dxbc, "index.json"), os.path.join(dxbc, "fssm_index.json"),
               os.path.join(spirv, "index.json"))
    if not all(os.path.isfile(path) for path in markers):
        return False
    had = read_shader_setting(os.path.join(spirv, "wave64_emulate.txt"), default="off")
    return (had == requested_shader_mode(had, emulate, keep_emulated)
            and read_shader_setting(os.path.join(spirv, "translator_flags.txt")) == translator_flags_setting(fallbacks)
            and read_shader_setting(os.path.join(spirv, "translation_sources.sha256")) == translation_sources_digest()
            and os.path.isfile(os.path.join(dxbc, "disasm.done"))
            and disassembly_status(dxbc, llvm_mc_signature())[0])


def prepare_shader_disassembly(work: str, disasm: str, resume: bool) -> None:
    """Check the output path and remove old listings before starting a new run."""
    expected = os.path.join(os.path.realpath(work), "shaders", "dxbc", "disasm")
    if os.path.normcase(os.path.realpath(disasm)) != os.path.normcase(expected):
        fail(f"The shader disassembly folder must be inside the work folder: {disasm}")
    if not resume and os.path.isdir(disasm):
        with os.scandir(disasm) as entries:
            for entry in entries:
                if entry.name.endswith(".s") and (entry.is_file(follow_symlinks=False) or entry.is_symlink()):
                    os.unlink(entry.path)


def translated_index_is_empty(index: str) -> bool:
    """Whether a saved translated index is the `{}` that rdna2spirv.py --merge publishes when it translated
    nothing, so the next run must not keep it as done. A missing, unreadable or non-mapping file is a different
    problem and is not this: only a whole-file empty mapping came from a merge that ran and produced nothing."""
    try:
        with open(index, encoding="utf-8") as f:
            return json.load(f) == {}
    except (OSError, ValueError):
        return False


def shaders(game: str, work: str, jobs: int, emulate: bool = False, keep_emulated: bool = False,
            fallbacks: tuple[str, ...] = ()) -> None:
    src = os.path.join(game, "shaders.rpf")
    sh = os.path.join(work, "shaders")
    dxbc = os.path.join(sh, "dxbc")
    spirv = os.path.join(sh, "spirv")
    # Each stage counts as finished only when the file it writes last is there, never because its folder exists
    # or holds many files: a run stopped part-way leaves both. Re-running a stage drops the later stages' proof.
    carved = os.path.join(dxbc, "index.json")             # shader_carve.py, renamed into place at the end
    unpacked = os.path.join(dxbc, "fssm_index.json")      # fssm_unpack.py, the same
    disassembly_started = os.path.join(dxbc, "disasm.started")
    disassembled = os.path.join(dxbc, "disasm.done")      # records LLVM and the listings produced
    translated = os.path.join(spirv, "index.json")        # rdna2spirv.py --merge, renamed into place
    shard_plan = os.path.join(spirv, SHARD_PLAN)          # lets translate() keep finished shards; see there
    signature = llvm_mc_signature()
    if not os.path.isfile(carved):
        log("shaders: carving shaders.rpf")
        drop(translated, shard_plan, disassembled, disassembly_started, unpacked)
        py("shader_carve.py", src, dxbc)
    if unpacked_shader_stems(dxbc) is None:
        log("shaders: unpacking native code")
        drop(translated, shard_plan, disassembled, disassembly_started)
        py("fssm_unpack.py", dxbc, "--carve-index", carved)
    # rdna2spirv.py finds index.json, the .dxbc files and fssm/ in the parent of the .s files' folder, so the
    # disassembly must sit inside the carve folder: next to it, every stage reads as "?" and is rejected.
    disasm = os.path.join(dxbc, "disasm")
    complete, resume, expected = disassembly_status(dxbc, signature)
    if not complete:
        if os.path.isfile(disassembled) or os.path.isfile(translated):
            log("shaders: llvm-mc changed or the saved disassembly could not be verified")
        log("shaders: resuming disassembly (gfx1013)" if resume else "shaders: disassembling (gfx1013)")
        drop(translated, shard_plan)
        if not resume:
            drop(disassembly_started, disassembled)
        prepare_shader_disassembly(work, disasm, resume)
        if not os.path.isfile(disassembly_started) or os.path.isfile(disassembled):
            marker = disassembly_started + ".tmp"
            with open(marker, "w", encoding="utf-8", newline="\n") as f:
                start = json.dumps({"llvm": signature, "listings": sorted(expected)}) if expected else signature
                f.write(start + "\n")
            os.replace(marker, disassembly_started)
        drop(disassembled)
        resume_args = ("--skip-existing",) if resume else ()
        py("shader_disasm.py", os.path.join(dxbc, "fssm"), "--unpack-index", unpacked,
           "--out", disasm, "--mcpu", "gfx1013", "--jobs", jobs, *resume_args)
        listings = disassembly_listings(dxbc)
        if not listings or not expected <= listings:
            fail("Shader disassembly did not restore all expected listings")
        marker = disassembled + ".tmp"
        with open(marker, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"llvm": signature, "listings": sorted(listings)}, f)
            f.write("\n")
        os.replace(marker, disassembled)
        drop(disassembly_started)
    # The translation also depends on the translator's own files, so their digest is kept beside it. A folder without
    # it (from before the file existed), with an unreadable one or with another one is translated again; carving, unpacking and
    # disassembly stay current.
    sources_file = os.path.join(spirv, "translation_sources.sha256")
    sources = translation_sources_digest()
    if os.path.isfile(translated) and read_shader_setting(sources_file) != sources:
        log("shaders: no matching translator fingerprint for the existing shaders: translating again")
        drop(translated)
    # The translation depends on --wave64-emulate, so the mode it was made with is kept beside it. A folder from
    # before the file existed was translated without emulation.
    mode_file = os.path.join(spirv, "wave64_emulate.txt")
    had = read_shader_setting(mode_file, default="off")
    want = requested_shader_mode(had, emulate, keep_emulated)
    if want == "on" and not emulate:
        # auto answers differently from run to run (a GPU asleep or hidden, vulkaninfo missing); emulated shaders are
        # exact on every GPU, so they are kept rather than translating the whole corpus again. When they are
        # translated again for another reason, or a stopped translation is retried, they stay emulated.
        log("shaders: the shaders were translated with wave64 emulation and it stays on (exact on every GPU); "
            "--wave64-emulate off translates them natively")
        emulate = True
    elif had != want and os.path.isfile(translated):
        log(f"shaders: wave64 emulation is now {want} (the shaders were translated with it {had}): "
            "translating again")
        drop(translated)
    # The translation also depends on the fallback flags (--no-float-controls, --bary-fallback,
    # --texel-offset-fallback, --mesh-compute), so they are kept beside it like the wave64 mode. A folder from
    # before the file existed was translated without them.
    flags_file = os.path.join(spirv, "translator_flags.txt")
    want_flags = translator_flags_setting(fallbacks)
    had_flags = read_shader_setting(flags_file)
    if had_flags != want_flags and os.path.isfile(translated):
        log(f"shaders: the fallback flags are now {want_flags or 'none'} (the shaders were translated with "
            f"{had_flags or 'none'}): translating again")
        drop(translated)
    if os.path.isfile(translated) and translated_index_is_empty(translated):
        # An empty merge, or an index that cannot be read, is not a finished translation. The shard plan goes with it:
        # its parts are what the empty merge read, so a retry has to translate them again rather than reuse them.
        log("shaders: the saved translated index names no shaders: translating again")
        drop(translated, shard_plan)
    if not os.path.isfile(translated):
        need("spirv-as", "Install the Vulkan SDK (it includes spirv-as and spirv-val) and reopen the terminal.")
        need("spirv-val", "Install the Vulkan SDK and reopen the terminal.")
        log(f"shaders: translating RDNA2 to SPIR-V on {jobs} processes (the longest step)")
        # The mode and the translator's digest go down before the translation starts, while there is no index.json:
        # a stop at any point then leaves either no index (translated again) or an index with the right mode and
        # digest, never a guessed one. rdna2spirv.py --merge writes index.json last, after the stubs.
        os.makedirs(spirv, exist_ok=True)
        with open(mode_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(want + "\n")
        with open(flags_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(want_flags + "\n")
        with open(sources_file, "w", encoding="utf-8", newline="\n") as f:
            f.write(sources + "\n")
        translate(disasm, spirv, jobs, emulate=emulate, fallbacks=fallbacks)
    verify_translation_completeness(dxbc, spirv)
    fspx_shaders(game, work, signature, emulate=emulate, fallbacks=fallbacks)
    bvh_shaders(game, work, signature, jobs, emulate=emulate, fallbacks=fallbacks)
    log("shaders: done")


FSPX_DLL = "amd_fidelityfx_upscaler_xbox.dll"
FSPX_TOOLS = ("fspx_unpack.py", "shader_disasm.py")   # with TRANSLATION_SOURCES, what the FSPX outputs come from


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fspx_settings(dll: str, mc_signature: str, emulate: bool, fallbacks: tuple[str, ...]) -> str:
    """What the FSPX outputs depend on, as one line: the DLL, the unpacking and translating tools, llvm-mc, the
    wave64 mode and the fallback flags."""
    tools = hashlib.sha256()
    for name in FSPX_TOOLS:
        tools.update(f"{name}\0".encode() + bytes.fromhex(file_sha256(os.path.join(TOOLS, name))))
    return json.dumps({"dll": file_sha256(dll), "tools": tools.hexdigest(), "translator": translation_sources_digest(),
                       "llvm_mc": mc_signature, "wave64_emulate": bool(emulate), "fallbacks": list(fallbacks)},
                      sort_keys=True)


def fspx_cache_complete(spirv: str) -> bool:
    """Check that the FSPX translation index names at least one program and that its named modules are present.
    A translation that produced no modules writes a bare `{}`, which must not be cached as a finished step."""
    try:
        index = json.loads(read_shader_setting(os.path.join(spirv, "index.json")))
    except ValueError:
        return False
    return isinstance(index, dict) and bool(index) and all(os.path.isfile(os.path.join(spirv, name + ".spv")) for name in index)


def fspx_shaders(game: str, work: str, mc_signature: str, emulate: bool = False,
                 fallbacks: tuple[str, ...] = ()) -> None:
    """The FidelityFX DLL's compute programs (docs/fspx.md): tools/fspx_unpack.py unpacks and disassembles them into
    <work>/shaders/fspx, rdna2spirv.py translates them into <work>/shaders/spirv/fspx, where the runtime's shader
    library finds them by fspx_code_sha256. The game's own shaders neither feed nor depend on this step.

    The 29 programs take seconds, so the step runs whole. Reuse requires the translation index, its named modules
    and a matching fspx_build.json (fspx_settings)."""
    dll = os.path.join(game, FSPX_DLL)
    unpacked = os.path.join(work, "shaders", "fspx")
    spirv = os.path.join(work, "shaders", "spirv", "fspx")
    translated = os.path.join(spirv, "index.json")
    settings_file = os.path.join(spirv, "fspx_build.json")
    if not os.path.isfile(dll):
        log(f"shaders: no {FSPX_DLL} in the game folder, so no FSPX programs to translate")
        return
    settings = fspx_settings(dll, mc_signature, emulate, fallbacks)
    if read_shader_setting(settings_file) == settings and fspx_cache_complete(spirv):
        return
    log(f"shaders: unpacking and translating the FSPX programs of {FSPX_DLL}")
    drop(translated, settings_file)
    # A program that no longer translates must not keep a module from the run before.
    for folder, suffixes in ((os.path.join(unpacked, "disasm"), (".s",)), (spirv, (".spv", ".spvasm"))):
        if os.path.isdir(folder):
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.name.endswith(suffixes) and entry.is_file(follow_symlinks=False):
                        os.unlink(entry.path)
    py("fspx_unpack.py", dll, "--out", unpacked, "--disasm")
    extra = (["--wave64-emulate"] if emulate else []) + list(fallbacks)
    py("rdna2spirv.py", os.path.join(unpacked, "disasm"), "--out", spirv, "--no-asm", *extra)
    clash = fspx_key_clash(spirv)
    if clash:
        drop(translated)   # the runtime must not load an index whose keys pick between different modules
        fail(f"shaders: FSPX programs {clash[0]} and {clash[1]} have the same kind-6 and kind-0 bytes "
             "(fspx_code_sha256 and fspx_state_sha256) but different translated modules, so the runtime could not "
             "tell which one a pipeline means (docs/fspx.md, \"Build and run-time lookup\")")
    with open(settings_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(settings + "\n")


def fspx_key_clash(spirv: str):
    """Two entries of <spirv>/index.json with the same fspx_code_sha256 and fspx_state_sha256 whose .spv files
    differ, as (name, name), or None. The same kind-6 bytes with different kind-0 entries is allowed: the run time
    picks by both keys when the bytecode carries kind 0, and refuses a bare kind-6 program whose modules differ."""
    with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
        index = json.load(f)
    first = {}
    for name in sorted(index):
        meta = index[name]
        if not isinstance(meta, dict) or "fspx_code_sha256" not in meta:
            continue
        key = (meta["fspx_code_sha256"], meta.get("fspx_state_sha256"))
        if key not in first:
            first[key] = name
            continue
        modules = []
        for n in (first[key], name):
            try:
                with open(os.path.join(spirv, n + ".spv"), "rb") as f:
                    modules.append(f.read())
            except OSError:
                modules.append(None)
        if modules[0] is None or modules[0] != modules[1]:
            return first[key], name
    return None


BVH_DLL = "XboxBvhBuilder.dll"
BVH_TOOLS = ("pe_shader_containers.py", "fssm_unpack.py", "shader_disasm.py")   # as FSPX_TOOLS, for the builder
# rdna2spirv.py's --max-shared-memory for the builder's programs: 32768, the maxComputeSharedMemorySize of the AMD
# Windows and Intel drivers (docs/design.md, "Workgroup memory limit"). The constructor creates a pipeline for every
# program and gives up at the first one that fails, so one module over the device's limit costs the whole builder.
# Measured on 1.0.2.1 (docs/embedded_shaders.md section 7): with --wave64-emulate RadixAlloc declares 37,084 B under
# the translator's 65536 default and 22,236 B under this cap; no other module changes, natively none does, and the
# same 51 programs translate either way.
BVH_MAX_SHARED_MEMORY = 32768


def bvh_settings(dll: str, mc_signature: str, emulate: bool, fallbacks: tuple[str, ...]) -> str:
    """What the builder's outputs depend on, as one line: the DLL, the extracting, unpacking and translating tools,
    llvm-mc, the wave64 mode, the fallback flags and the workgroup-memory cap."""
    return json.dumps({"dll": file_sha256(dll), "tools": sources_digest(TOOLS, BVH_TOOLS),
                       "translator": translation_sources_digest(), "llvm_mc": mc_signature,
                       "wave64_emulate": bool(emulate), "fallbacks": list(fallbacks),
                       "max_shared_memory": BVH_MAX_SHARED_MEMORY}, sort_keys=True)


def bvh_programs(unpacked: str, spirv: str) -> "tuple[list[str], list[str]] | None":
    """The programs pe_shader_containers.py extracted, by file stem, and the ones among them the runtime cannot
    find: no entry with an fssm_sha256 in <spirv>/index.json, or no module beside it. None when either index is
    missing or unreadable, or names a module that is not there."""
    try:
        extracted = sorted(shader_stems(os.path.join(unpacked, "index.json"), "file", ".dxbc"))
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(index, dict) or not all(os.path.isfile(os.path.join(spirv, name + ".spv")) for name in index):
        return None
    found = {name for name, meta in index.items()
             if isinstance(meta, dict) and isinstance(meta.get("fssm_sha256"), str)}
    return extracted, [name for name in extracted if name not in found]


def bvh_report(programs: "tuple[list[str], list[str]]") -> None:
    """One line with the counts, and one that names every program left without a module. It is printed on every
    run, also when the step itself is reused: a program the translator has no rule for is never dropped quietly."""
    extracted, missing = programs
    log(f"shaders: {len(extracted) - len(missing)} of {len(extracted)} {BVH_DLL} programs translated")
    if missing:
        log(f"shaders: NOT translated: {', '.join(missing)}. The runtime has no module for "
            f"{'it' if len(missing) == 1 else 'them'}, so that compute pipeline is refused (E_INVALIDARG) and the "
            "builder's constructor, which creates all of them, gives up there. When the step runs, rdna2spirv.py "
            "prints the instruction each one stops at (docs/embedded_shaders.md, section 4)")


def bvh_shaders(game: str, work: str, mc_signature: str, jobs: int = 1, emulate: bool = False,
                fallbacks: tuple[str, ...] = ()) -> None:
    """XboxBvhBuilder.dll's compute programs (docs/embedded_shaders.md): pe_shader_containers.py extracts the DLL's
    DXBC containers into <work>/shaders/bvh, fssm_unpack.py and shader_disasm.py unpack and disassemble their FSSM
    programs there, as for a shader from shaders.rpf, and rdna2spirv.py translates them into
    <work>/shaders/spirv/bvh, where the runtime's shader library finds them by fssm_sha256, the key of the game's
    own shaders. Those neither feed nor depend on this step.

    The 52 programs take seconds, so the step runs whole. Reuse requires both indexes, the translated modules and a
    matching bvh_build.json (bvh_settings)."""
    dll = os.path.join(game, BVH_DLL)
    unpacked = os.path.join(work, "shaders", "bvh")
    spirv = os.path.join(work, "shaders", "spirv", "bvh")
    extracted = os.path.join(unpacked, "index.json")
    unpack_index = os.path.join(unpacked, "fssm_index.json")
    translated = os.path.join(spirv, "index.json")
    settings_file = os.path.join(spirv, "bvh_build.json")
    if not os.path.isfile(dll):
        # A work folder used before with a package that had the DLL still holds that run's index, and the runtime
        # loads <shaders>/bvh/index.json whenever it is there. Without the index the old modules are not found.
        drop(translated, settings_file)
        log(f"shaders: no {BVH_DLL} in the game folder, so no BVH builder programs to translate")
        return
    settings = bvh_settings(dll, mc_signature, emulate, fallbacks)
    programs = bvh_programs(unpacked, spirv) if read_shader_setting(settings_file) == settings else None
    if programs:
        bvh_report(programs)
        return
    log(f"shaders: extracting and translating the compute programs of {BVH_DLL}")
    drop(translated, settings_file, extracted, unpack_index)
    # pe_shader_containers.py never overwrites a container (a second run would write <module>_NN.dxbc beside the
    # first), and a program that no longer translates must not keep a module from the run before.
    for folder, suffixes in ((unpacked, (".dxbc",)), (os.path.join(unpacked, "fssm"), (".fssm",)),
                             (os.path.join(unpacked, "disasm"), (".s",)), (spirv, (".spv", ".spvasm"))):
        if os.path.isdir(folder):
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.name.endswith(suffixes) and entry.is_file(follow_symlinks=False):
                        os.unlink(entry.path)
    py("pe_shader_containers.py", dll, "--extract", unpacked)
    py("fssm_unpack.py", unpacked, "--carve-index", extracted)
    # rdna2spirv.py reads each program's stage from index.json and its FSSM blob from fssm/, both in the parent of
    # the listings' folder, so the disassembly sits inside the extraction folder as it does for shaders.rpf.
    disasm = os.path.join(unpacked, "disasm")
    py("shader_disasm.py", os.path.join(unpacked, "fssm"), "--unpack-index", unpack_index, "--out", disasm,
       "--mcpu", "gfx1013", "--jobs", jobs)
    extra = (["--wave64-emulate"] if emulate else []) + list(fallbacks)
    py("rdna2spirv.py", disasm, "--out", spirv, "--no-asm", "--max-shared-memory", BVH_MAX_SHARED_MEMORY, *extra)
    programs = bvh_programs(unpacked, spirv)
    if not programs or len(programs[1]) == len(programs[0]):
        drop(translated)   # an empty or unreadable index would read as a finished step
        fail(f"shaders: none of the {BVH_DLL} programs translated, so their index is not kept. Run the shaders "
             "stage again; if it happens again, the tool output above is why.")
    bvh_report(programs)
    with open(settings_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(settings + "\n")


WAVE64_MODES = ("auto", "on", "off")

# The translator's fallback flags make_game.py can forward, and the argparse attribute each one reads.
FALLBACK_FLAGS = (("--no-float-controls", "no_float_controls"), ("--bary-fallback", "bary_fallback"),
                  ("--texel-offset-fallback", "texel_offset_fallback"), ("--mesh-compute", "mesh_compute"),
                  ("--f16-interp-fallback", "f16_interp_fallback"), ("--pc-tail-ubo", "pc_tail_ubo"))


def fallback_flags(a) -> tuple[str, ...]:
    """The fallback flags the parsed arguments ask for, in a stable order (the order they are recorded in)."""
    return tuple(flag for flag, attr in FALLBACK_FLAGS if getattr(a, attr, False))


def recorded_fallback_flags(work: str) -> tuple[str, ...]:
    """The fallback flags the translation in `work` was made with (spirv/translator_flags.txt), () when there is none."""
    return tuple(read_shader_setting(os.path.join(work, "shaders", "spirv", "translator_flags.txt")).split())


def keep_recorded_fallback(flag: str, work: str) -> bool:
    """Whether an automatic fallback flag stays on when the GPU probe can't tell if the GPU needs it.

    The probe answers differently from run to run (a GPU asleep or hidden, a missing Vulkan loader, its 60 s
    timeout). Dropping the flag then would translate the whole corpus without it, and the next run that can ask
    would translate it a third time. A flag the probe turned on was needed by the GPU it last reached, so a
    translation made with it is kept, as wave64 emulation is; a probe that answers still decides."""
    if flag not in recorded_fallback_flags(work):
        return False
    log(f"shaders: couldn't ask the GPU the runtime would use whether it needs {flag}; the shaders were translated "
        "with it and it stays on instead of translating them all again (a run that can ask the GPU decides again)")
    return True


def smallest_max_subgroup(vulkaninfo_text: str):
    """The smallest maxSubgroupSize vulkaninfo reports over all devices (text or --json output), or None."""
    size = r'maxSubgroupSize"?\s*[=:]\s*(\d+)'
    # A device's own name is the least indented one; deeper ones (VK_KHR_maintenance7's layered APIs, e.g.
    # MoltenVK's Metal device, in both the text and the --json output) belong to the device above them.
    names = list(re.finditer(r'(?m)^([ \t]*)"?deviceName"?\s*[=:]', vulkaninfo_text))
    depth = min((len(m.group(1)) for m in names), default=0)
    starts = [m.start() for m in names if len(m.group(1)) == depth]
    if not starts:
        sizes = [int(n) for n in re.findall(size, vulkaninfo_text)]
        return min(sizes) if sizes else None
    # One section per device (a 1.3 device can print the size twice, in two property structs): one value each,
    # and a device that prints none makes the answer unknown rather than being left out of the smallest.
    per_device = []
    for a, b in zip(starts, starts[1:] + [len(vulkaninfo_text)]):
        sizes = [int(n) for n in re.findall(size, vulkaninfo_text[a:b])]
        if not sizes:
            return None
        per_device.append(min(sizes))
    return min(per_device)


def wave64_settings(mode: str, *, mesh_compute: bool = False) -> tuple[bool, bool]:
    """Return whether to translate with emulation and whether to reuse emulated shaders. A wave64 compute or mesh
    shader is only exact without it when the GPU gives it 64-wide subgroups; with it, it is exact on any subgroup
    size, at the cost of a workgroup barrier per segment. "auto" keeps emulation off on native-wave64 RDNA1/2
    under the AMD Windows driver. Elsewhere it turns emulation on unless every Vulkan device reports
    maxSubgroupSize >= 64 and the runtime probe does not rule out native wave64."""
    if mode == "on":
        return True, False
    if mode == "off":
        return False, False
    gpu = runtime_gpu_info()
    rdna1 = gpu.get("rdna1_amd_windows", False)
    # On RDNA1 the native mesh modules cannot run. Both the default build's compute pipelines and
    # any mesh shaders translated with --mesh-compute use the GPU's compute subgroup capabilities.
    compute_only = mesh_compute or (rdna1 and gpu.get("mesh_in_supported_stages") is False)
    native64 = gpu.get("native64")
    if compute_only:
        native64 = gpu.get("native64_compute", native64)
    if native64 and (rdna1 or gpu.get("rdna2_amd_windows")):
        # #857: AMD's Windows compiler for RDNA1/2 crashes on many emulated LDS compute pipelines.
        # The runtime probe can identify that GPU even when vulkaninfo is unavailable.
        log(f"shaders: the runtime's GPU ({gpu.get('device', '?')}) is {'RDNA1' if rdna1 else 'RDNA2'} "
            "on the AMD Windows driver and runs wave64 natively. That driver can crash on emulated wave64 code "
            "(#857), so auto uses native wave64 and rebuilds emulated outputs (--wave64-emulate on to force emulation)")
        return False, False
    exe = shutil.which("vulkaninfo")
    size = None
    if exe:
        try:
            r = subprocess.run([exe], capture_output=True, text=True, errors="replace", timeout=120)
            size = smallest_max_subgroup(r.stdout)
        except (OSError, subprocess.SubprocessError):
            size = None
    if size is None:
        log("shaders: couldn't read the GPU's subgroup size from vulkaninfo: wave64 emulation on "
            "(exact on every GPU; --wave64-emulate off to skip it on a GPU with 64-wide subgroups)")
        return True, True
    on = size < 64
    if not on and native64 is False:
        stages = "compute" if compute_only else "both compute and mesh"
        log(f"shaders: smallest maximum subgroup size is {size}, but the Vulkan query found no 64-wide path for "
            f"{stages} (fixed default size or required-size control): wave64 emulation on")
        return True, True
    log(f"shaders: smallest maximum subgroup size is {size}: wave64 emulation {'on' if on else 'off'}")
    return on, True


def wave64_emulate(mode: str, *, mesh_compute: bool = False) -> bool:
    """Whether a new translation uses wave64 emulation."""
    return wave64_settings(mode, mesh_compute=mesh_compute)[0]


def runtime_gpu_info() -> dict:
    """tools/vk_subgroup_info.py's answer for the GPU the runtime would use; {} when it can't tell."""
    try:
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "vk_subgroup_info.py")],
                           capture_output=True, text=True, timeout=60)
        info = json.loads(r.stdout.strip().splitlines()[-1])
        return info if isinstance(info, dict) and "error" not in info else {}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return {}


def runtime_gpu_per_vertex():
    """per_vertex_inputs as tools/vk_subgroup_info.py answers for the GPU the runtime would use: False where the
    driver can't compile PerVertexKHR (MoltenVK) or lacks fragmentShaderBarycentric, None when it can't tell."""
    try:
        r = subprocess.run([sys.executable, "-I", os.path.join(TOOLS, "vk_subgroup_info.py")],
                           capture_output=True, text=True, timeout=60)
        v = json.loads(r.stdout.strip().splitlines()[-1]).get("per_vertex_inputs")
        return None if v is None else bool(v)
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def runtime_gpu_barycentric_fallback():
    """Whether the selected Vulkan GPU's driver has a known BaryCoordKHR bug (Mesa ANV, AMD Windows on gfx90c), or None
    when it can't be told."""
    try:
        v = runtime_gpu_info().get("barycentric_fallback")
        return v if isinstance(v, bool) else None
    except (AttributeError, TypeError):
        return None


def barycentric_fallback_override():
    """LEONIDA_BARY_FALLBACK=1/0 forces the same barycentric path as the GPU test runner; None leaves it automatic."""
    v = (os.environ.get("LEONIDA_BARY_FALLBACK") or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    return None


# VkDriverId of Intel's Windows Vulkan driver (igvk64.dll).
VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS = 5


def runtime_gpu_mesh_draws_crash():
    """True where the GPU the runtime would use is on Intel's Windows Vulkan driver with VK_EXT_mesh_shader: there
    vkCmdDrawMeshTasksEXT crashes (0xC0000409 in igvk64.dll) for a mesh shader with a live push-constant load at
    byte offset 32 or more, which every corpus mesh shader has (BUILD-2202, measured on an Arc A770 with driver
    32.0.101.9175; --mesh-compute is exact there). No fixed driver version is known, so every one counts. False
    for any other GPU or one without mesh shaders, None when tools/vk_subgroup_info.py can't tell."""
    info = runtime_gpu_info()
    if not info or "driver_id" not in info:
        return None
    return info.get("driver_id") == VK_DRIVER_ID_INTEL_PROPRIETARY_WINDOWS and \
        info.get("mesh_in_supported_stages") is True


def runtime_gpu_max_push_constants():
    """max_push_constants_size as tools/vk_subgroup_info.py answers for the GPU the runtime would use, or None."""
    v = runtime_gpu_info().get("max_push_constants_size")
    return int(v) if isinstance(v, int) and not isinstance(v, bool) and v > 0 else None


# Source and data files that produce the translated modules, metadata and partner stubs.
TRANSLATION_SOURCES = ("rdna2spirv.py", "shader_ir.py", "cfg_structure.py", "fssm_header.py", "lane_masks.py", "wave64_halves.py",
                       "wave64_emulate_opt.py", "shader_meta.py", "spirv_stubs.py", "spirv_asm.py", "spirv_asm_tables.py",
                       "spirv_opt.py", "report_paths.py", "replace_retry.py", "gfx10_formats.json")


def sources_digest(folder: str, names, lead: str = "") -> str:
    """SHA-256 over lead, then the name and exact bytes of each named file in folder."""
    h = hashlib.sha256(lead.encode())
    for name in names:
        with open(os.path.join(folder, name), "rb") as f:
            data = f.read()
        h.update(f"{name}\0{len(data)}\0".encode())
        h.update(data)
    return h.hexdigest()


def translation_sources_digest() -> str:
    """SHA-256 over the name and exact bytes of each TRANSLATION_SOURCES file."""
    return sources_digest(TOOLS, TRANSLATION_SOURCES)


def drop(*paths: str) -> None:
    """Remove the files that mark later stages finished, so they run again on this stage's new output."""
    for p in paths:
        if os.path.isfile(p):
            os.remove(p)


SHARD_PLAN = "shards.json"   # beside the index parts: the --jobs, mode and translator source that made them


def drop_index(index: str) -> str:
    """Remove a translated index the next run must not take as finished; "" once it is gone.

    A remove that fails returns a sentence for the stage's failure message instead of raising: on Windows the
    lock that stopped the index being read or merged (an antivirus scan, the search indexer) usually stops this
    too, and the PermissionError traceback would hide why the stage failed. The reason is the strerror, not
    str(error), which quotes the path a second time through repr() (every Windows backslash doubled)."""
    try:
        os.remove(index)
    except FileNotFoundError:
        pass
    except OSError as error:
        return f" {index} could not be removed ({error.strerror or error}): delete it before running the shaders stage again."
    return ""


def translator_digest() -> str:
    """SHA-256 over every file directly in tools/: rdna2spirv.py, the modules it imports and the data it reads
    (gfx10_formats.json). Any change there counts, so parts made by other code are never merged with new ones."""
    import hashlib
    h = hashlib.sha256()
    for name in sorted(os.listdir(TOOLS)):
        path = os.path.join(TOOLS, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                h.update(name.encode("utf-8") + b"\0" + f.read() + b"\0")
    return h.hexdigest()


def _stop_procs(procs: list) -> None:
    """Terminate and reap any child processes still running after an interrupted or failed launch."""
    for p in procs:
        if p.poll() is None:
            try:
                p.terminate()
            except OSError:
                pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
            except OSError:
                pass
            try:
                p.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                pass
        except OSError:
            pass


def translate(disasm: str, spirv: str, jobs: int, emulate: bool = False,
              fallbacks: tuple[str, ...] = ()) -> None:
    """rdna2spirv.py runs on one core: start one --shard per job side by side, then merge their index parts.

    A shard publishes shard<i>.json and then index.part<i>.json. Both are needed to reuse it. A run stopped
    part-way, or during the merge, keeps those parts when it resumes with the same --jobs, wave64 mode, fallback
    flags and translator source, and starts only the shards that have none. shaders() drops the plan whenever an
    earlier step reruns."""
    os.makedirs(spirv, exist_ok=True)
    plan_path = os.path.join(spirv, SHARD_PLAN)
    plan = {"jobs": jobs, "wave64_emulate": bool(emulate), "fallbacks": list(fallbacks),
            "translator": translator_digest()}
    try:
        with open(plan_path, encoding="utf-8") as f:
            same = json.load(f) == plan
    except (OSError, ValueError):
        same = False
    if not same:
        # Parts from another --jobs would be merged in too, and parts from another mode, flags or translator are
        # wrong. They go before the new plan is written: a stop in between must not leave old parts under a new plan.
        for f in os.listdir(spirv):
            if (f.startswith("index.part") and f.endswith(".json")) or re.fullmatch(r"shard\d+\.json", f):
                os.remove(os.path.join(spirv, f))
        with open(plan_path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(plan, f)
    todo = [
        i for i in range(jobs)
        if not (os.path.isfile(os.path.join(spirv, f"index.part{i}.json"))
                and os.path.isfile(os.path.join(spirv, f"shard{i}.json")))
    ]
    if len(todo) < jobs:
        log(f"  keeping {jobs - len(todo)} of {jobs} shards finished by the run before")
    tool = os.path.join(TOOLS, "rdna2spirv.py")
    extra = ["--wave64-emulate"] if emulate else []
    extra.extend(fallbacks)
    if todo:
        which = f"0..{jobs - 1}" if len(todo) == jobs else ",".join(str(i) for i in todo)
        log(f"  rdna2spirv.py {disasm} --out {spirv} --shard {which}/{jobs}" + "".join(" " + e for e in extra))
    # --no-asm: keep only the .spv modules; the .spvasm text would add ~35 GB for the corpus
    board = leonida_tui.current_board()
    stage = leonida_tui.current_stage()
    popen_kw = leonida_tui.capture_popen_kwargs() if board is not None else {}
    procs: dict[int, object] = {}
    pumps: dict[int, threading.Thread] = {}
    bad = []
    try:
        for i in todo:
            procs[i] = subprocess.Popen(
                [sys.executable, "-I", tool, disasm, "--out", spirv, "--shard", f"{i}/{jobs}",
                 "--no-asm", *extra], **popen_kw)
        if board is not None:
            for i, p in procs.items():
                t = threading.Thread(target=leonida_tui.pump_process, args=(p, stage, board), daemon=True)
                pumps[i] = t
                t.start()
        progress = Progress("shaders: translate", jobs, every_s=10.0)
        progress.set(jobs - len(todo), force=True)
        while procs:
            for i, p in list(procs.items()):
                code = p.poll()
                if code is None:
                    continue
                del procs[i]
                if i in pumps:
                    pumps[i].join(timeout=30)
                    del pumps[i]
                if code != 0:
                    bad.append(f"shard {i} exited with {code}")
                progress.tick(force=True)
            if procs:
                time.sleep(1.0)
    except BaseException:
        _stop_procs(list(procs.values()))
        raise
    if bad:
        # An older index describes the corpus from the run before: keeping it would make the next run take the
        # failed translation as finished. The next successful --merge writes it again.
        fail("rdna2spirv.py: " + ", ".join(bad) + "." + drop_index(os.path.join(spirv, "index.json")))
    py("rdna2spirv.py", disasm, "--out", spirv, "--merge")
    index = os.path.join(spirv, "index.json")
    if os.path.isfile(index):
        # Present but empty or unreadable: the merge published nothing, and the next run would read that as finished.
        try:
            with open(index, encoding="utf-8") as f:
                merged = json.load(f)
        except (OSError, ValueError) as e:
            fail(f"rdna2spirv.py --merge wrote an unreadable {index}: {getattr(e, 'strerror', None) or e}."
                 + drop_index(index))
        if not merged:
            # A marker left behind would look like a finished translation to the next run; drop it with the failure.
            left = drop_index(index)
            # The plan would make the retry reuse the same empty parts even after the disassembly was repaired.
            try:
                os.remove(plan_path)
            except OSError:
                pass
            fail("rdna2spirv.py translated 0 shaders, so the empty index is not kept. Run the shaders stage again; "
                 "if it happens again, the disassembly logged above is why." + left)


def verify_translation_completeness(dxbc: str, spirv: str) -> None:
    """Every carved shader must be in the merged translation index before the stage may call itself done.

    rdna2spirv.py drops a shader that is refused or whose module fails spirv-val (the module never reaches the
    final path) and still exits 0, and nothing later in the build compares the counts: the merge already refuses
    an empty index, and a translation that did not run fails the stage before this point, so those cases are left
    to the paths that own them. What this adds is the silent one, a readable index naming fewer shaders than the
    carve: a port assembled from it would lack the difference."""
    try:
        with open(os.path.join(dxbc, "index.json"), encoding="utf-8") as f:
            carved = json.load(f)
        with open(os.path.join(spirv, "index.json"), encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError):
        return
    if not isinstance(carved, list) or not isinstance(index, dict):
        fail("shaders: the carve index is not a list or the translation index is not an object; "
             "run the shaders stage again")
    if not index:
        return   # an empty index is #3328's to refuse or translate again, not a partial one to compare
    if len(index) != len(carved):
        fail(f"shaders: the translation index holds {len(index)} of the {len(carved)} carved shaders "
             f"({len(carved) - len(index)} were refused or failed spirv-val, listed in this stage's output): "
             "a port built from it would lack them. Fix the translator for that class and translate again")

# ---------------------------------------------------------------- assets
# Beside each model, texture and pool manifest.json: the decoder_digest() that made it.
DECODER_RECORD = "decoder.sha256"


def compute_decoder_digest() -> str:
    """SHA-256 over the checked-out third_party/rpf9-explorer commit and the name and exact bytes of each
    tools/rpf9dump source file: what the model and texture exports come from."""
    sub = os.path.join(ROOT, "third_party", "rpf9-explorer")
    pin = ""
    # Asked only inside the submodule's own checkout: in an empty folder git answers with this repository's HEAD.
    # shortcut: a copy without git metadata (source zips) has no pin to read, so only tools/rpf9dump is tracked
    # there; hash third_party/rpf9-explorer/Core as well if such copies need it.
    git = shutil.which("git")
    if git and os.path.exists(os.path.join(sub, ".git")):
        r = subprocess.run([git, "-C", sub, "rev-parse", "HEAD"], capture_output=True, text=True)
        pin = r.stdout.strip() if r.returncode == 0 else ""
        # An edit to a tracked submodule file that is not committed changes the decoder but not HEAD.
        # shortcut: new untracked files in the submodule are not seen; hash them too if that comes up.
        r = subprocess.run([git, "-C", sub, "diff", "HEAD"], capture_output=True)
        if r.returncode == 0 and r.stdout:
            pin += "+" + hashlib.sha256(r.stdout).hexdigest()
    folder = os.path.join(TOOLS, "rpf9dump")
    names = sorted(n for n in os.listdir(folder) if n.endswith((".cs", ".csproj", ".json")))
    return sources_digest(folder, names, pin)


# assets() asks for it once, after preflight's ensure_submodule() has fetched the submodule. Anything earlier (the
# preflight credit) calls compute_decoder_digest() instead, so a digest made from a missing checkout is never cached.
decoder_digest = functools.cache(compute_decoder_digest)


def rpf9dump(work: str) -> str:
    # Build under the repo, not --work: a remote or locked -o path can make dotnet fail copying
    # RpfPreview.Core.dll (MSB3027).
    out_dir = os.path.join(ROOT, "tools", "rpf9dump", "bin", "Release", "net8.0")
    exe = os.path.join(out_dir, "rpf9dump.exe" if os.name == "nt" else "rpf9dump")
    core = os.path.join(ROOT, "third_party", "rpf9-explorer", "Core")
    if not os.path.isdir(core):
        fail("third_party/rpf9-explorer is empty. Run: git submodule update --init")
    need("dotnet", "Install the .NET 8 SDK.")
    log("assets: building rpf9dump")
    run(["dotnet", "build", os.path.join(TOOLS, "rpf9dump", "rpf9dump.csproj"), "-c", "Release",
         "-o", out_dir, "--nologo", "-v", "q"])
    return exe


def ttpf_manifest(pool_dir: str) -> dict | None:
    """An older exporter could publish the manifest after a failed DDS write. Check its promised files on resume."""
    try:
        with open(os.path.join(pool_dir, "manifest.json"), encoding="utf-8") as f:
            doc = json.load(f)
        with os.scandir(pool_dir) as it:
            have = {e.name for e in it if e.is_file()}
    except (OSError, ValueError, RecursionError):
        return None
    if not isinstance(doc, dict):
        return None
    rows, count = doc.get("Textures"), doc.get("Count")
    if not isinstance(rows, list) or type(count) is not int or count != len(rows):
        return None
    for row in rows:
        if not isinstance(row, dict) or "File" not in row:
            return None
        name = row["File"]
        if name is None:  # The catalog marked this texture unsupported.
            continue
        if not isinstance(name, str):
            return None
        if name not in have:
            # An existing file may use another spelling on a case-insensitive volume.
            if name != os.path.basename(name) or not os.path.isfile(os.path.join(pool_dir, name)):
                return None
    return doc


def assets(game: str, work: str, wav: bool, jobs: int, dump_jobs: int | None = None) -> None:
    # make_game.py also runs under python -I, which omits the script directory.
    if TOOLS not in sys.path:
        sys.path.insert(0, TOOLS)
    import ttpf_assets

    get_exporter = functools.cache(rpf9dump)
    out = os.path.join(work, "assets")

    def dump(*parts: str) -> None:
        """Run rpf9dump; pass --jobs only when the build overlaps another CPU-heavy stage."""
        cmd = [get_exporter(work), *parts]
        if dump_jobs is not None:
            cmd.extend(["--jobs", str(dump_jobs)])
        run(cmd)

    # An export also depends on the decoder, so its digest is kept beside the manifest, written before the export
    # starts. A folder without it (from before the file existed) or with another one is exported again, as the
    # shaders stage does for the translator.
    decoder = decoder_digest()
    archive_todo = []
    for rel in MODEL_ARCHIVES:
        src = os.path.join(game, rel)
        name = os.path.splitext(os.path.basename(rel))[0]
        for mode in ("models", "textures"):
            folder = os.path.join(out, name, mode)
            record = os.path.join(folder, DECODER_RECORD)
            if os.path.isdir(folder) and read_shader_setting(record) != decoder:
                # rpf9dump keeps the files it finds, so another decoder's output is removed first. The exporter is
                # built before that: a build that fails leaves the old export in place.
                get_exporter(work)
                log(f"assets: no matching decoder fingerprint for {mode} from {rel}: exporting again")
                shutil.rmtree(folder)
            elif os.path.isfile(os.path.join(folder, "manifest.json")):
                continue
            archive_todo.append((rel, mode, folder, record, src))

    # Texture pools: data/ttdb/** and each pack's own pack/<name>/data/ttdb/** (pool names repeat across packs,
    # so pack pools go under ttpf_pack/<pack>/).
    pools = []
    for root, _, files in os.walk(game):
        pools += [os.path.join(root, f) for f in files if f.lower().endswith(".ttpf")]
    if pools:
        try:
            settings = ttpf_assets.load_profile(work)
        except ValueError as exc:   # a hand-edited calibration profile is one build error, not a traceback
            fail(str(exc))
    else:
        settings = None
    # rpf9dump names each pool's output folder after the source basename. Pools with the same basename
    # in one destination namespace would share that folder and its manifest, so the second pool would
    # look already exported and lose all its DDS. Give every member of a colliding group its own
    # export root, keyed off the relative source path; unique pools keep their current folder.
    info = []
    for p in pools:
        name = os.path.splitext(os.path.basename(p))[0]
        parts = os.path.relpath(p, game).replace("\\", "/").split("/")
        dest = os.path.join(out, "ttpf_pack", parts[1]) if parts[0] == "pack" else os.path.join(out, "ttpf")
        info.append((p, name, dest, "/".join(parts)))
    counts = {}
    for _, name, dest, _ in info:
        key = (os.path.normcase(dest), os.path.normcase(name))
        counts[key] = counts.get(key, 0) + 1
    pool_jobs = []
    for p, name, dest, rel in sorted(info, key=lambda x: os.path.getsize(x[0])):
        root = dest
        if counts[(os.path.normcase(dest), os.path.normcase(name))] > 1:
            root = os.path.join(dest, f"{name}~{hashlib.sha256(rel.encode('utf-8')).hexdigest()[:12]}")
        pool_dir = os.path.join(root, name)
        # rpf9dump ttpf decodes with the same rpf9-explorer Core and keeps the DDS it finds, so a pool takes the
        # same decoder record as the model and texture exports above.
        record = os.path.join(pool_dir, DECODER_RECORD)
        if os.path.isdir(pool_dir) and read_shader_setting(record) != decoder:
            log(f"assets: no matching decoder fingerprint for texture pool {os.path.relpath(p, game)}: exporting again")
            shutil.rmtree(pool_dir)
        pool_jobs.append((p, name, root, pool_dir, record))

    pool_todo = [(p, name, root, pool_dir, record) for p, name, root, pool_dir, record in pool_jobs
                 if ttpf_manifest(pool_dir) is None]
    pool_follow = pool_jobs
    need_audio = not os.path.isfile(os.path.join(out, "audio", "audio_manifest.json"))
    ogg = os.path.join(out, "audio", "ogg")
    ogg_done = os.path.join(out, "audio", "ogg.done")
    need_ogg = not os.path.isfile(ogg_done)
    need_wav = bool(wav)
    xma_dir = os.path.join(out, "audio", "xma")
    xma_done = os.path.join(out, "audio", "xma.done")
    need_xma = not os.path.isfile(xma_done)
    ff = shutil.which("ffmpeg")
    need_xma_decode = bool(ff)  # decode_xma no-ops when nothing is pending
    total = (len(archive_todo) + len(pool_todo) + len(pool_follow) + int(need_audio) + int(need_ogg)
             + int(need_wav) + int(need_xma) + int(need_xma_decode))
    progress = Progress("assets", total, every_s=2.0)

    for rel, mode, folder, record, src in archive_todo:
        log(f"assets: {mode} from {rel}")
        os.makedirs(folder, exist_ok=True)
        with open(record, "w", encoding="utf-8", newline="\n") as f:
            f.write(decoder + "\n")
        dump(mode, src, out)
        progress.tick(force=True)
    for p, name, root, pool_dir, record in pool_todo:
        log(f"assets: texture pool {os.path.relpath(p, game)}")
        os.makedirs(pool_dir, exist_ok=True)
        with open(record, "w", encoding="utf-8", newline="\n") as f:
            f.write(decoder + "\n")
        dump("ttpf", p, root)
        if ttpf_manifest(pool_dir) is None:
            fail(f"Texture exporter did not publish a complete manifest in {pool_dir}")
        progress.tick(force=True)
    for p, name, root, pool_dir, record in pool_follow:
        manifest = ttpf_manifest(pool_dir)
        if manifest is None:
            fail(f"Texture exporter did not publish a complete manifest in {pool_dir}")
        written, blocked = ttpf_assets.complete_pool(p, pool_dir, manifest, settings)
        if written or blocked:
            log(f"assets: {name}: {written} full BCPack DDS written, {blocked} limited (see manifest.json)")
        progress.tick(force=True)
    if need_audio:
        log("assets: audio banks")
        dump("audio", os.path.join(game, "audio.rpf"), out)
        progress.tick(force=True)
    # The two audio steps below count as finished only by a marker written after they succeed: their folders are
    # created first and exist after a stopped run too.
    if need_ogg:
        # Lossless: the banks' Opus packets repackaged as standard Ogg Opus files (PCM streams as .wav)
        log("assets: audio streams -> Ogg Opus")
        py("awc_opus.py", os.path.join(out, "audio", "awc"), ogg, "--jobs", str(jobs))
        mark(ogg_done, "awc_opus.py")
        progress.tick(force=True)
    if need_wav:
        if not os.path.isdir(ogg):
            fail("assets: no Ogg Opus output to decode. The audio bank step must finish first (run the assets stage).")
        # Opt-in: lossless Ogg Opus -> 16-bit PCM WAV, several times larger. Existing WAVs are skipped.
        log("assets: audio Ogg Opus -> PCM WAV")
        py("audio_wav.py", ogg, "--out", os.path.join(out, "audio", "wav"), "--jobs", str(jobs))
        progress.tick(force=True)
    if need_xma:
        log("assets: XMA2 streams")
        dump("xma", os.path.join(game, "audio.rpf"), out)
        mark(xma_done, "rpf9dump xma")
        progress.tick(force=True)
    # Every run: decodes what a stopped or failed run left, and everything once ffmpeg is installed later.
    if ff:
        decode_xma(ff, xma_dir, jobs)
        progress.tick(force=True)
    else:
        log("assets: ffmpeg not on PATH: XMA2 streams kept as .xma.wav (ffmpeg decodes them)")
    log("assets: done")


def mark(path: str, what: str) -> None:
    """Write a step's completion marker; call it only after the step succeeded."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"{what} finished\n")


def pending_xma(xma_dir: str) -> list:
    """The XMA2 streams (.xma.wav) under xma_dir that have no decoded .wav next to them yet."""
    todo = []
    for root, _, files in os.walk(xma_dir):
        for f in files:
            if f.endswith(".xma.wav") and not os.path.isfile(os.path.join(root, f[:-8] + ".wav")):
                todo.append(os.path.join(root, f))
    return todo


def decode_xma(ff: str, xma_dir: str, jobs: int = 1) -> None:
    """XMA2 (.xma.wav) -> 16-bit PCM .wav with ffmpeg, for each stream that has no .wav yet. ffmpeg writes to a
    temporary name that is renamed when it succeeds, so a stopped or failed decode never leaves a short .wav."""
    todo = pending_xma(xma_dir)
    if not todo:
        return
    log(f"assets: XMA2 -> PCM WAV (ffmpeg), {len(todo)} stream(s)")
    progress = Progress("assets: XMA2 decode", len(todo), every_s=5.0)

    def one(src: str) -> bool:
        dst = src[:-8] + ".wav"
        tmp = dst + ".part"
        # No stdin: ffmpeg reads it for key commands, which stops a build started in the background of a terminal.
        r = subprocess.run([ff, "-v", "error", "-y", "-i", src, "-c:a", "pcm_s16le", "-f", "wav", tmp],
                           stdin=subprocess.DEVNULL)
        progress.tick()
        if r.returncode == 0 and os.path.isfile(tmp):
            replace_retry(tmp, dst)
            return True
        if os.path.isfile(tmp):
            os.remove(tmp)
        return False

    todo = sorted(todo)
    # One ffmpeg per stream, `jobs` at a time; the threads only wait on their process. map keeps the order, so the
    # first failure reported is the same as one at a time.
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        bad = [src for src, good in zip(todo, ex.map(one, todo)) if not good]
    if bad:
        log(f"assets: ffmpeg could not decode {len(bad)} XMA2 stream(s), first {os.path.relpath(bad[0], xma_dir)}. "
            "They stay as .xma.wav; the next run tries them again.")


# ---------------------------------------------------------------- runtime
def wine_launcher(out: str, work: str, image: str) -> str:
    shaders = os.path.abspath(os.path.join(work, "shaders", "spirv"))
    # The runtime reads shader_path.txt. Its absolute path needs updating only if the whole port moved.
    moved_catalog = ""
    if os.path.commonpath([os.path.abspath(out), shaders]) == os.path.abspath(out):
        relative = shlex.quote(os.path.relpath(shaders, out))
        moved_catalog = f"""    if [ "$here" != {shlex.quote(os.path.realpath(out))} ]; then
        shaders="$here"/{relative}
        export LEONIDA_SHADERS="${{LEONIDA_SHADERS:-Z:$shaders}}"
    fi
"""
    return f"""#!/bin/sh
# Starts the Windows build of the loader with Wine. WINE picks another one, e.g. a Proton build's bin/wine.
here=$(dirname "$(readlink -f "$0")")
if [ "$#" -eq 1 ] && {{ [ "$1" = --dry-run ] || [ "$1" = --init ] || [ "$1" = --run ]; }}; then
    game=$(cat "$here/game_path.txt" && printf .) || exit 1
    cd "${{game%.}}" || exit 1
    image={shlex.quote(image)}
    if [ ! -f "./$image" ]; then
        printf 'Missing game executable: %s/%s\\n' "$PWD" "$image" >&2
        exit 1
    fi
    # Wine maps the Unix filesystem root to Z:. The game reads from its package and writes to the port's overlay.
{moved_catalog}    export LEONIDA_D3D12="${{LEONIDA_D3D12:-vulkan}}"
    set -- "$1" "./$image" --overlay "Z:$here/overlay"
fi
exec "${{WINE:-wine}}" "$here/gta6rt.exe" "$@"
"""


def cmake_configure(build: str, *config: str, hint: str = "") -> None:
    """Configure runtime/ in build; the generator and build type are chosen only for a new build folder."""
    fresh = [] if os.path.isfile(os.path.join(build, "CMakeCache.txt")) else [
        *(["-G", "Ninja"] if shutil.which("ninja") else []), "-DCMAKE_BUILD_TYPE=RelWithDebInfo"]
    run(["cmake", "-S", os.path.join(ROOT, "runtime"), "-B", build, *fresh, *config], hint=hint)


def cmake_build(build: str, targets: list, jobs: int, hint: str) -> None:
    run(["cmake", "--build", build, "--parallel", str(jobs), "--target", *targets], hint=hint)


def cross_configure() -> str:
    xwin = xwin_dir()
    if not xwin:
        fail("runtime: no xwin SDK in XWIN_DIR or ~/.local/share/xwin-sdk (docs/testing.md, "
             "\"The Windows runtime on Linux\"), and no --runtime-build.")
    build = cross_build_dir()
    suffix = "" if configured_cross_tools() else cross_tool_suffix() or ""
    tools = [f"-DCMAKE_{var}={shutil.which(name + suffix) or name + suffix}" for var, name in CROSS_VARS.items()
             ] if suffix else []
    cmake_configure(build, "--toolchain", os.path.join(ROOT, "runtime", "cmake", "clang-cl-xwin.cmake"),
                    f"-DXWIN_DIR={xwin}", *tools, hint="CMake's error above says what the cross-build lacks "
                    "(docs/testing.md, \"The Windows runtime on Linux\"). Or pass --runtime-build, or drop "
                    "runtime from --stages (the default without it: check,shaders,assets,pipelines).")
    return build


def cross_build_runtime(jobs: int) -> str:
    log("runtime: cross-building with clang-cl and xwin")
    build = cross_configure()
    cmake_build(build, ["gta6rt", "gta6rt_tls_host"], jobs,
                "See the compiler output above; docs/testing.md has the cross-build's requirements.")
    return build


def build_pipeline_check(jobs: int) -> None:
    build = os.path.join(ROOT, "runtime", "build", "linux")
    log("pipelines: building vk_pipeline_check")
    cmake_configure(build)
    cmake_build(build, ["vk_pipeline_check"], jobs,
                "It needs a C++20 compiler and the Vulkan headers (your distribution's Vulkan headers package, "
                "or VULKAN_SDK); the CMake output above says which is missing.")


def runtime(game: str, out: str, v: dict, runtime_build: str | None = None, jobs: int = 1,
            *, work: str | None = None) -> None:
    destinations = [*RUNTIME_FILES, "game_path.txt"]
    if work is not None:
        destinations.append("shader_path.txt")
    if os.name != "nt":
        destinations.append("gta6rt.sh")
    # Check the complete set before compiling or copying, so a later conflict leaves the old runtime usable.
    for name in destinations:
        destination = os.path.join(out, name)
        if os.path.isdir(destination):
            fail(f"runtime: {destination} is a directory and the runtime needs it as a file. "
                 "Use another --out folder or move the directory before retrying.")

    if runtime_build:
        log(f"runtime: using the build in {runtime_build}")
        build = runtime_build
    elif os.name != "nt":
        build = cross_build_runtime(jobs)
    else:
        log("runtime: building")
        run([windows_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
             os.path.join(ROOT, "scripts", "build_runtime.ps1"), "-Jobs", str(jobs)])
        build = os.path.join(ROOT, "runtime", "build")
    artifacts = [os.path.join(build, name) for name in RUNTIME_FILES]
    for built in artifacts:
        if not os.path.isfile(built):
            fail(f"runtime: {'build did not produce' if not runtime_build else 'no'} {built}")
    os.makedirs(out, exist_ok=True)
    for built in artifacts:
        # --runtime-build may be the output folder itself (a runtime build unpacked where the port goes): the
        # files are in place then, and copying a file onto itself raises SameFileError.
        if not os.path.exists(os.path.join(out, os.path.basename(built))) or \
                not os.path.samefile(built, os.path.join(out, os.path.basename(built))):
            shutil.copy2(built, out)
    with open(os.path.join(out, "game_path.txt"), "w", encoding="utf-8", newline="\n") as f:
        f.write(os.path.abspath(game))
    if work is not None:
        shader_dir = os.path.join(os.path.abspath(work), "shaders", "spirv")
        # Wine's path converter takes the original Unix filename bytes. The Windows reader expects UTF-8.
        encoded = shader_dir.encode("utf-8") if os.name == "nt" else os.fsencode(shader_dir)
        pending = tempfile.NamedTemporaryFile(dir=out, delete=False)
        try:
            with pending:
                pending.write(encoded)
            os.replace(pending.name, os.path.join(out, "shader_path.txt"))
        finally:
            if os.path.exists(pending.name):
                os.unlink(pending.name)
    start = os.path.join(out, "gta6rt.exe")
    if os.name != "nt":
        start = os.path.join(out, "gta6rt.sh")
        with open(start, "w", encoding="utf-8", newline="\n") as f:
            f.write(wine_launcher(out, work or os.path.join(out, "work"), v["expected_exe"]))
        os.chmod(start, os.stat(start).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    image = os.path.join(game, v["expected_exe"])
    if not os.path.isfile(image):
        log(f"runtime: built, but {v['expected_exe']} is missing from your copy, so there is nothing to run yet.")
    elif os.name == "nt":
        def ps_literal(path: str) -> str:
            # PowerShell also treats typographic single quotes as string delimiters.
            return "'" + re.sub(r"['\u2018-\u201b]", r"\g<0>\g<0>", path) + "'"

        log("runtime: built. Inspect the game's imports in PowerShell:\n"
            f"  & {ps_literal(start)} --dry-run {ps_literal(image)}")
    else:
        log(f"runtime: built. Inspect the game's imports with Wine:\n  {shlex.join([start, '--dry-run'])}")


def positive_int(text: str) -> int:
    """--jobs: 0 or less would start no translation shard and merge an empty index as if the step had finished."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def pipeline_check_exe(runtime_build: str | None = None) -> str | None:
    """Find the checker for this host, using a supplied Windows build before the local Windows build."""
    build = os.path.join(ROOT, "runtime", "build")
    windows = os.name == "nt"
    if windows:
        windows_builds = (runtime_build, build) if runtime_build else (build,)
        found = [os.path.join(d, "vk_pipeline_check.exe") for d in windows_builds]
    else:
        found = [os.path.join(build, "linux", "vk_pipeline_check"), os.path.join(build, "vk_pipeline_check")]
    return next((p for p in found if os.path.isfile(p) and (windows or os.access(p, os.X_OK))), None)


def require_translated_shaders(work: str) -> str:
    spirv = os.path.join(work, "shaders", "spirv")
    index = os.path.join(spirv, "index.json")
    if not os.path.isfile(index):
        fail("No translated shaders yet: run the shaders stage first.")
    try:
        with open(index, encoding="utf-8") as f:
            translated = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"The translated shader index {index} is unreadable ({getattr(e, 'strerror', None) or e}): "
             "run the shaders stage again.")
    if not translated:
        # A build from before the shaders stage refused an empty merge can carry this marker.
        fail(f"The translated shader index {index} is empty, so there is nothing to check. Run the shaders stage again.")
    return spirv


def pipelines(work: str, out: str, jobs: "int | None" = None, *, runtime_build: str | None = None) -> None:
    exe = pipeline_check_exe(runtime_build)
    if not exe and os.name == "nt":
        fail("vk_pipeline_check.exe wasn't built: run the runtime stage with the Vulkan SDK installed.")
    if not exe:
        fail("vk_pipeline_check wasn't built. With CMake 3.24+, a C++20 compiler and the Vulkan headers and "
             "loader installed, run: cmake -S runtime -B runtime/build/linux && "
             "cmake --build runtime/build/linux --parallel")
    spirv = require_translated_shaders(work)
    log("pipelines: creating every shader module and pipeline (compute and graphics) on this GPU")
    os.makedirs(out, exist_ok=True)
    cache = os.path.join(out, "pipeline_cache.bin")
    inflight = os.path.join(out, "pipeline_inflight.txt")
    skip = os.path.join(out, "pipeline_crashes.txt")
    drop(inflight, skip)   # every module gets a fresh chance on each run (a new translator may have fixed it)
    # --jobs: make_game.py's own; without it vk_pipeline_check uses half the hardware threads whatever was asked
    base = [exe, spirv, cache, *(["--jobs", str(jobs)] if jobs else [])]
    if not checker_supports(exe, "--inflight-file", "--skip-file"):
        # An older build takes any argument it doesn't know as the cache path, so it would write the cache into
        # pipeline_inflight.txt and exit 0. Run it as before: a module that kills it stops the stage.
        log("pipelines: this vk_pipeline_check has no --inflight-file; rebuild the runtime stage so a module that "
            "crashes the driver is skipped instead of stopping the stage")
        run(base)
        log("pipelines: done")
        return
    base += ["--inflight-file", inflight]
    # Bound each checker invocation to a range that recovery can split further when concurrent heavy compiles
    # exceed available memory, without starting the entire corpus again (BUILD-1030).
    names = pipeline_names(spirv) if checker_supports(exe, "--start", "--count") else []
    crashed: "list[str]" = []

    def check_range(start: int = 0, count: "int | None" = None) -> int:
        selection = ["--start", str(start), "--count", str(count)] if count is not None else []
        allowed = set(names[start:start + count]) if count is not None else None
        for _ in range(PIPELINE_CRASH_RETRIES + 1):
            drop(inflight)   # a crash during setup must not inherit another process's in-flight module names
            code = run(base + selection + (["--skip-file", skip] if crashed else []), check=False)
            # Native setup failures also return 1. A finished pass writes/clears this report even when shaders
            # fail validation; setup failures return before writing it. Do not repeat setup for every batch.
            if code == 1 and not os.path.isfile(inflight):
                fail("vk_pipeline_check exited with 1 before writing its work report; check Vulkan initialization "
                     "and report-file access in the output above")
            if code in (0, 1, 2) or interrupted(code):
                return code
            suspects = list(dict.fromkeys(n for n in read_names(inflight)
                                          if n not in crashed and (allowed is None or n in allowed)))
            # Even the last in-flight module can be innocent after other heavy compiles in the batch.
            if count is not None or len(suspects) > 1:
                suspects = [n for n in suspects if crashes_alone(exe, spirv, cache, n, inflight)]
            if not suspects:
                if count is not None and count > 1:
                    # All suspects survived in fresh processes: do not blacklist them. Re-check the entire failed
                    # range in smaller processes; completed earlier batches retain their cache checkpoints.
                    log(f"pipelines: shaders {start + 1}-{start + count} exited with {exit_label(code)} without "
                        "a reproducible module crash; retrying this range in smaller batches")
                    half = count // 2
                    first = check_range(start, half)
                    if first == 2 or interrupted(first):
                        return first
                    second = check_range(start + half, count - half)
                    return second or first
                fail(f"vk_pipeline_check exited with {exit_label(code)} while no module could be named as the cause; "
                     "the checker could not finish even this range in a fresh process; check available memory "
                     "and report it if it keeps happening")
            for n in suspects:
                log(f"pipelines: {n} took vk_pipeline_check down ({exit_label(code)}); checking the rest without it")
            crashed.extend(suspects)
            with open(skip, "w", encoding="utf-8", newline="\n") as f:
                f.write("".join(n + "\n" for n in crashed))
        fail(f"vk_pipeline_check still crashed after {PIPELINE_CRASH_RETRIES} retries; modules that took it down: "
             + ", ".join(crashed))

    code = 0
    ranges = [(s, min(PIPELINE_BATCH_SIZE, len(names) - s))
              for s in range(0, len(names), PIPELINE_BATCH_SIZE)] if names else [(0, None)]
    for start, count in ranges:
        if count is not None:
            log(f"pipelines: checking shaders {start + 1}-{start + count} of {len(names)} in a fresh process")
        result = check_range(start, count)
        if result:
            code = result   # a later successful batch must not erase an earlier validation failure
        if result == 2 or interrupted(result):
            break
    if crashed and not code:
        code = 1
    drop(inflight)
    if code != 0:
        why = f"; {len(crashed)} module(s) took the driver down and were counted as failed: " + ", ".join(crashed) \
            if crashed else ""
        fail(f"vk_pipeline_check exited with {exit_label(code)}{why}")
    log("pipelines: done")


# How many times the pipelines stage restarts vk_pipeline_check after a module takes it down (each costs at most the
# 10 minutes since the last cache checkpoint, plus loading the cache). Looked up when pipelines() runs.
PIPELINE_CRASH_RETRIES = 5
PIPELINE_BATCH_SIZE = 500   # fresh compiler process and cache checkpoint after at most this many index entries


def interrupted(code: int) -> bool:
    """Ctrl+C or a kill, not a crash: SIGINT/SIGTERM off Windows, STATUS_CONTROL_C_EXIT on it."""
    return code in (-2, -15) or (code & 0xFFFFFFFF) == 0xC000013A


def exit_label(code: int) -> str:
    """An exit code as Windows writes it (0x80000003 for an LLVM abort in the driver), or the signal off Windows."""
    if code < 0:
        return f"signal {-code}"
    return f"{code} ({code:#010x})" if code > 255 else str(code)


def checker_supports(exe: str, *options: str) -> bool:
    """Whether this vk_pipeline_check build knows the options: run without arguments, it prints its usage (and exits
    2) before touching Vulkan. Builds before --inflight-file don't reject unknown arguments, so asking is the only way
    to tell."""
    try:
        r = subprocess.run([exe], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    usage = (r.stdout or "") + (r.stderr or "")
    return all(o in usage for o in options)


def read_names(path: str) -> "list[str]":
    """The module names in vk_pipeline_check's --inflight-file ("name<TAB>phase" lines)."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [n for n in (line.split("\t", 1)[0].strip() for line in f) if n]
    except OSError:
        return []


def pipeline_names(spirv: str) -> "list[str]":
    """The checker's name-sorted index (byte order, as its std::map); ranges and diagnostic probes share it."""
    path = os.path.join(spirv, "index.json")
    try:
        with open(path, encoding="utf-8") as f:
            index = json.load(f)
    except (OSError, ValueError) as e:
        fail(f"pipelines: cannot read {path}: {getattr(e, 'strerror', None) or e}")
    if not isinstance(index, dict):
        fail(f"pipelines: {path} must contain a JSON object")
    return sorted(index, key=lambda n: n.encode("utf-8"))


def crashes_alone(exe: str, spirv: str, cache: str, name: str, inflight: str | None = None) -> bool:
    """Confirm a crash only when this probe's fresh report names the selected module as in flight."""
    names = pipeline_names(spirv)
    if name not in names:
        return False
    args = [exe, spirv, cache, "--jobs", "1", "--start", str(names.index(name)), "--count", "1"]
    if inflight:
        drop(inflight)
        args += ["--inflight-file", inflight]
    code = run(args, check=False)
    if code == 1 and inflight and not os.path.isfile(inflight):
        fail(f"pipelines: checking {name} alone exited with 1 before writing its work report; "
             "check Vulkan initialization and report-file access in the output above")
    if interrupted(code) or code == 2:
        fail(f"pipelines: checking {name} alone stopped with {exit_label(code)}")
    if code in (0, 1):
        return False
    # The process may have died during setup or after finishing, rather than inside this module. The native
    # report is periodic for small modules, so a fast real crash can also leave no evidence: stop conservatively
    # in that case instead of blaming an innocent module or retrying more ranges with an unreliable checker.
    if not inflight or name not in read_names(inflight):
        fail(f"pipelines: checking {name} alone exited with {exit_label(code)} without a fresh report naming it "
             "as in flight; the diagnostic result is inconclusive, so no crash is attributed to this module. "
             "Check the checker output, available memory and report-file access before retrying")
    return True


def path_is_within(path: str, directory: str) -> bool:
    try:
        return os.path.commonpath([path, directory]) == directory
    except ValueError:  # Different Windows drives or UNC shares have no common path.
        return False


def _invoke_stage(name: str, fn) -> None:
    """Run a stage (optionally on a worker thread). fail() uses sys.exit; turn that into an exception the pool
    can collect so siblings can finish before the process stops."""
    with leonida_tui.stage_scope(name):
        try:
            fn()
        except SystemExit as e:
            if e.code in (0, None):
                return
            raise RuntimeError(f"stage exited with {e.code}") from e


def run_build_stages(stages: list, *, game: str, work: str, out: str, v: dict, wav: bool, jobs: int,
                     runtime_build: str | None, emulate: bool, keep_emulated: bool,
                     fallbacks: tuple[str, ...], overlap: bool = True) -> None:
    """Run build stages after check. With overlap=True (default), shaders/assets/runtime share the machine and
    pipelines starts once shaders (and a selected runtime rebuild) are ready. With overlap=False, stages run in order."""
    cpu_heavy = [s for s in ("shaders", "assets") if s in stages]
    split = max(1, jobs // len(cpu_heavy)) if (overlap and len(cpu_heavy) > 1) else jobs
    dump_jobs = split if (overlap and len(cpu_heavy) > 1 and "assets" in stages) else None
    shader_jobs = split if (overlap and len(cpu_heavy) > 1 and "shaders" in stages) else jobs
    progress = StageBoard([s for s in ("shaders", "assets", "runtime", "pipelines") if s in stages])

    def fn_shaders():
        shaders(game, work, shader_jobs, emulate=emulate, keep_emulated=keep_emulated, fallbacks=fallbacks)

    def fn_assets():
        assets(game, work, wav, jobs, dump_jobs=dump_jobs)

    def fn_runtime():
        runtime(game, out, v, runtime_build, jobs, work=work)

    def fn_pipelines():
        pipelines(work, out, jobs, runtime_build=runtime_build)

    builders = {
        "shaders": fn_shaders,
        "assets": fn_assets,
        "runtime": fn_runtime,
        "pipelines": fn_pipelines,
    }

    if not overlap:
        order = [s for s in ("shaders", "assets", "runtime", "pipelines") if s in stages]
        if not order:
            return
        log(f"running {', '.join(order)} in series")
        for name in order:
            try:
                _invoke_stage(name, builders[name])
                progress.complete(name)
            except Exception:
                sys.exit(1)
        return

    work_items: list[tuple[str, object]] = [(n, builders[n]) for n in ("shaders", "assets", "runtime")
                                            if n in stages]

    if len(work_items) > 1:
        if dump_jobs is not None:
            log(f"running {', '.join(n for n, _ in work_items)} in parallel "
                f"(jobs split to {split} for shaders/assets)")
        else:
            log(f"running {', '.join(n for n, _ in work_items)} in parallel")

    if not work_items and "pipelines" not in stages:
        return

    failed = False
    ex = ThreadPoolExecutor(max_workers=max(1, len(work_items) + (1 if "pipelines" in stages else 0)))
    try:
        futures = {name: ex.submit(_invoke_stage, name, fn) for name, fn in work_items}

        def wait(name: str) -> bool:
            """True when the stage finished cleanly. Marks the build failed on error."""
            nonlocal failed
            try:
                futures[name].result()
                progress.complete(name)
                return True
            except Exception:
                failed = True
                return False

        if "pipelines" in stages:
            shaders_ok = wait("shaders") if "shaders" in futures else True
            if shaders_ok:
                # When runtime is in this run it rebuilds vk_pipeline_check at the same path: wait for it so
                # pipelines does not validate with a stale binary or race the Windows linker. A checker already
                # on disk is enough only when this run is not rebuilding it.
                runtime_ok = wait("runtime") if "runtime" in futures else True
                if runtime_ok:
                    futures["pipelines"] = ex.submit(_invoke_stage, "pipelines", fn_pipelines)

        for name, fut in list(futures.items()):
            try:
                fut.result()
                progress.complete(name)
            except Exception:
                failed = True
        ex.shutdown(wait=True)
    except KeyboardInterrupt:
        # Default ThreadPoolExecutor.__exit__ waits for workers; a worker blocked in
        # readline()/cmake would ignore Ctrl+C forever. Kill children and abandon workers.
        leonida_tui.kill_live_processes()
        try:
            ex.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            ex.shutdown(wait=False)
        raise

    if failed:
        # The failing stage already printed through fail(); stop without a second message.
        sys.exit(1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the PC port from your own copy of the game.")
    ap.add_argument("--game", help="your package folder (found automatically if left out; a share path "
                                    "\\\\server\\share\\... goes here)")
    ap.add_argument("--pick", action="store_true", help="choose your package folder in a folder dialog")
    ap.add_argument("--out", help="folder for the finished port (default Documents/Leonida; ~/Leonida on a Mac "
                                  "whose Documents iCloud syncs)")
    ap.add_argument("--work", help="folder for intermediate files (hundreds of GB; default <out>/work)")
    ap.add_argument("--stages", help="comma list from: " + ",".join(STAGES))
    ap.add_argument("--check", action="store_true", help="only check the game files")
    ap.add_argument("--full-check", action="store_true", help="also compare SHA-256 of every file (slow)")
    ap.add_argument("--wav", action="store_true",
                    help="also decode the Ogg Opus audio to 16-bit PCM WAV (much larger; needs ffmpeg)")
    ap.add_argument("--jobs", type=positive_int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--find", action="store_true", help="only list the game copies found on this PC")
    ap.add_argument("--runtime-build", metavar="DIR",
                    help="a Windows build of runtime/ (gta6rt.exe and gta6rt_tls_host.dll) for the runtime stage "
                         "instead of building it (off Windows: instead of the clang-cl + xwin cross-build); "
                         "on Windows, also use vk_pipeline_check.exe from DIR if present")
    ap.add_argument("--wave64-emulate", choices=WAVE64_MODES, default="auto",
                    help="translate wave64 compute and mesh shaders to be exact on any subgroup size (needed on "
                         "32-wide GPUs: NVIDIA, Intel). auto (default) checks GPU subgroup support and prefers "
                         "native wave64 on affected AMD Windows drivers")
    ap.add_argument("--no-float-controls", action="store_true",
                    help="translate without SignedZeroInfNanPreserve and DenormPreserve (Depends on: #664), for "
                         "devices whose vk_pipeline_check skips a shader without "
                         "shaderSignedZeroInfNanPreserveFloat32 or shaderDenormPreserveFloat16")
    ap.add_argument("--bary-fallback", action="store_true",
                    help="translate raw barycentric reads from an interpolated vec3 input (with a pass-through "
                         "geometry shader the runtime draws), for devices without "
                         "VK_KHR_fragment_shader_barycentric")
    ap.add_argument("--texel-offset-fallback", action="store_true",
                    help="fold the run-time texel offset of image_sample*_o into the coordinate, for devices "
                         "without VK_KHR_maintenance8")
    ap.add_argument("--f16-interp-fallback", action="store_true",
                    help="16-bit interpolation from decoded halves the vertex/mesh shader exports instead of "
                         "PerVertexKHR inputs; on by itself when the runtime's GPU can't use them (MoltenVK)")
    ap.add_argument("--mesh-compute", action="store_true",
                    help="translate mesh shaders to compute shaders writing an output buffer, for devices "
                         "without VK_EXT_mesh_shader; on by itself on Intel's Windows Vulkan driver, whose mesh "
                         "draws crash on the corpus' mesh shaders (BUILD-2202)")
    ap.add_argument("--native-mesh", action="store_true",
                    help="keep native mesh shaders on Intel's Windows Vulkan driver instead of turning on "
                         "--mesh-compute (for a driver that fixes BUILD-2202's crash)")
    ap.add_argument("--pc-tail-ubo", action="store_true",
                    help="keep s0..s31 in 128 push-constant bytes and move heap/code base to a set-1 uniform "
                         "block; on by itself when the runtime's GPU reports maxPushConstantsSize below what "
                         "the shaders need (144, or 152 with --mesh-compute)")
    ap.add_argument("--no-tui", action="store_true",
                    help="plain console logs (default when stdout is not a TTY); skip the LEONIDA setup screen "
                         "and per-stage log view")
    a = ap.parse_args()
    stages = STAGES
    if a.stages is not None:
        stages = [s.strip() for s in a.stages.split(",") if s.strip()]
        if not stages:
            ap.error("--stages must select at least one stage from: " + ",".join(STAGES))
        for s in stages:
            if s not in STAGES:
                ap.error(f"--stages: unknown stage '{s}' (choose from: {','.join(STAGES)})")
    if a.pick and a.game:
        ap.error("--pick and --game both choose the game folder: give one")
    below_normal()
    if a.find:
        containers = []
        found = find_dumps(containers)
        for d, v in found:
            print(f"{d}  ({v['title']} {v['version']})")
        print(f"{len(found)} found")
        for p, h in containers:
            print(f"not usable: {p}  (packaged {h['kind']} container, package version {h['version']}; "
                  "the build needs the folder that holds MicrosoftGame.Config)")
        return 0
    game = os.path.abspath(a.game or (pick_game() if a.pick else choose_game()))
    out = os.path.abspath(a.out or default_out())
    work = os.path.abspath(a.work or os.path.join(out, "work"))
    require_folder("--out", out)
    require_folder("--work", work)
    v = check(game, a.full_check)
    if a.check:
        return 0
    # Resolve existing parents too: a not-yet-created destination can be under a junction or symlink.
    real_game = os.path.normcase(os.path.realpath(game))
    is_game_drive_root = os.path.splitdrive(real_game)[1] in ("\\", "/")
    game_entries = []
    if is_game_drive_root:
        # The dump's own top-level names, from the version manifest: os.listdir(real_game) would also count an
        # unrelated --out or --work folder on the drive, so the rerun that resumes the build was rejected.
        top_level = {rel.replace("\\", "/").split("/", 1)[0] for rel in [v["expected_exe"], *v["files"]]}
        game_entries = [os.path.normcase(os.path.realpath(os.path.join(real_game, e)))
                        for e in sorted(top_level) if e]
    for p in (out, work):
        # normcase: Windows paths compare without case, so C:\Games\X and c:\games\x\out are inside.
        real_destination = os.path.normcase(os.path.realpath(p))
        if is_game_drive_root:
            # If the game is at drive root (e.g. D:\), reject the drive root itself and anything
            # inside the dump's own top-level entries, but allow separate top-level folders.
            inside = (real_destination == real_game) or any(
                path_is_within(real_destination, entry) for entry in game_entries
            )
        else:
            inside = path_is_within(real_destination, real_game)
        if inside:
            fail("--out and --work must be outside the game folder.")
    runtime_build = os.path.abspath(a.runtime_build) if a.runtime_build else None
    if a.stages is None and os.name != "nt" and not runtime_build:
        lacking = cross_build_missing()
        if lacking:
            stages = [s for s in stages if s not in WINDOWS_ONLY]
            log(f"{', '.join(WINDOWS_ONLY)}: left out, the cross-build needs {'; '.join(lacking)} "
                "(or pass --runtime-build)")
    if "shaders" in stages and not a.mesh_compute and not a.native_mesh:
        crash = runtime_gpu_mesh_draws_crash()
        if crash:
            log("shaders: the GPU the runtime would use is on Intel's Windows Vulkan driver, whose mesh draws crash on "
                "these mesh shaders (BUILD-2202): translating with --mesh-compute (--native-mesh keeps them native)")
            a.mesh_compute = True
        elif crash is None:
            a.mesh_compute = keep_recorded_fallback("--mesh-compute", work)
    emulate, keep_emulated = (wave64_settings(a.wave64_emulate, mesh_compute=a.mesh_compute) if "shaders" in stages
                              else (False, a.wave64_emulate == "auto"))
    if "shaders" in stages and not a.f16_interp_fallback:
        per_vertex = runtime_gpu_per_vertex()
        if per_vertex is False:
            log("shaders: the GPU the runtime would use can't take PerVertexKHR inputs (MoltenVK, or no "
                "fragmentShaderBarycentric): translating with --f16-interp-fallback")
            a.f16_interp_fallback = True
        elif per_vertex is None:
            a.f16_interp_fallback = keep_recorded_fallback("--f16-interp-fallback", work)
    if "shaders" in stages:
        if a.bary_fallback:
            # Command-line precedence: --bary-fallback wins over LEONIDA_BARY_FALLBACK=0.
            pass
        else:
            override = barycentric_fallback_override()
            if override is True:
                log("shaders: LEONIDA_BARY_FALLBACK=1 forces --bary-fallback")
                a.bary_fallback = True
            elif override is False:
                log("shaders: LEONIDA_BARY_FALLBACK=0 forces native barycentrics")
                a.bary_fallback = False
            else:
                bary_fallback = runtime_gpu_barycentric_fallback()
                if bary_fallback is True:
                    log("shaders: this GPU's driver has a known BaryCoordKHR bug (Mesa ANV, or AMD's Windows driver "
                        "on gfx90c): translating with --bary-fallback")
                    a.bary_fallback = True
                elif bary_fallback is None:
                    a.bary_fallback = keep_recorded_fallback("--bary-fallback", work)
    if "shaders" in stages and not a.pc_tail_ubo:
        need_pc = 152 if a.mesh_compute else 144
        max_pc = runtime_gpu_max_push_constants()
        if max_pc is None:
            a.pc_tail_ubo = keep_recorded_fallback("--pc-tail-ubo", work)
        elif max_pc < need_pc:
            log(f"shaders: the GPU the runtime would use reports maxPushConstantsSize = {max_pc} (< {need_pc}): "
                "translating with --pc-tail-ubo")
            a.pc_tail_ubo = True
    fallbacks = fallback_flags(a) if "shaders" in stages else ()
    overlap = True
    a.jobs = leonida_tui.clamp_jobs(a.jobs)
    use_tui = leonida_tui.tui_eligible(a.no_tui)
    if use_tui:
        cfg = leonida_tui.run_setup(a.jobs, default_overlap=True)
        a.jobs = leonida_tui.clamp_jobs(cfg.jobs)
        overlap = cfg.overlap
    preflight(stages, work, a.wav, runtime_build, emulate=emulate, keep_emulated=keep_emulated,
              fallbacks=fallbacks, jobs=a.jobs)
    if not a.out and icloud_documents():
        for note in icloud_default_notes():
            log(note)
    log(f"building into {out} (work files in {work})")
    t0 = time.time()
    build_stages = [s for s in ("shaders", "assets", "runtime", "pipelines") if s in stages]
    if use_tui and build_stages:
        log_dir = os.path.join(work, "logs")
        with leonida_tui.LeonidaBoard(build_stages, log_dir) as tui_board:
            tui_board.set_summary(f"build 0% · {', '.join(build_stages)}")
            run_build_stages(stages, game=game, work=work, out=out, v=v, wav=a.wav, jobs=a.jobs,
                             runtime_build=runtime_build, emulate=emulate, keep_emulated=keep_emulated,
                             fallbacks=fallbacks, overlap=overlap)
            tui_board.set_summary(f"finished in {(time.time() - t0) / 60:.0f} min")
    else:
        run_build_stages(stages, game=game, work=work, out=out, v=v, wav=a.wav, jobs=a.jobs,
                         runtime_build=runtime_build, emulate=emulate, keep_emulated=keep_emulated,
                         fallbacks=fallbacks, overlap=overlap)
    log(f"finished in {(time.time() - t0) / 60:.0f} min")
    return 0


def cli() -> int:
    """main(), with Ctrl+C reported in one line instead of a traceback: every stage resumes on the next run."""
    # A redirected stdout (make_game.py > build.log) is encoded with the console's ANSI code page on Windows, and
    # -I ignores PYTHONUTF8, so a character outside that code page -- a Cyrillic, Greek or CJK user name in a
    # found dump's path or in --out -- raises UnicodeEncodeError and ends the run. stderr escapes such
    # characters already (backslashreplace); stdout is strict, so write '?' where the code page has no glyph.
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    try:
        return main()
    except KeyboardInterrupt:
        leonida_tui.kill_live_processes()
        # Board __exit__ usually restored the console; if Ctrl+C hit outside it, clear leftovers.
        try:
            leonida_tui.restore_console(ansi=leonida_tui.enable_ansi(), leave_alt=True)
        except Exception:
            pass
        print("Stopped. Run it again to continue where it left off.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(cli())
