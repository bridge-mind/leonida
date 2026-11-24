#!/usr/bin/env python3
"""Run the 600-frame synthetic Win32 guest scene headless in full, save and resume mode and check the results.

  python -I scripts/run_guest_scene.py --loader L --guest G --shaders ROOT [--out DIR] [--wine WINE]
                                       [--play-audio] [--negative-controls]

Each run renders offscreen in a fresh process and write overlay. Without --out the runs go to a temporary folder that is removed
afterwards. The summary is printed first, and on failure the last run's log tails are printed too. Audio stays offline
unless --play-audio sets LEONIDA_SCENE_AUDIO=winmm. The guest then plays the pulse synchronously with PlaySoundW.
Run records distinguish a child exit, a deadline expiry and a launch failure.
--negative-controls also runs a corrupted scene.bin and a corrupted checkpoint.bin, and each one must exit with 1.
Exit code 0 means every check passed, 1 means a check failed and 2 means bad arguments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

FRAMES, HALF, RECORD_SIZE = 600, 300, 32  # frames.bin: eight little-endian uint32 words per frame
PAGES, PAGE_SIZE, PAGE_FRAMES = 10, 16, 60
MODES = {"full": (0, FRAMES), "save": (0, HALF), "resume": (HALF, FRAMES)}  # first frame, end frame
A_FRAMES = (137, 422)
SHOT_FRAMES = (0, 137, 299, 300, 422, 599)
DIRECTIONS = ((1, 0), (1, 1), (0, 1), (0, 0), (-1, 0), (-1, -1), (0, -1), (1, 0), (0, 0), (0, 1), (-1, 0), (1, -1))
VERSION, STATE_WORDS, AUDIO_WORDS = 1, 20, 10
AUDIO_RATE, AUDIO_SAMPLES, AUDIO_RAMP, AUDIO_HALF_PERIOD = 44100, 11025, 441, 50  # 441 Hz square, 10 ms ramps
PLAYBACK = {0: "offline", 1: "played", 3: "device error"}
TIMEOUT = 180
MARKER = b"GUEST_SCENE_PASS"
NEGATIVE = {"bad-scene": ("full", b"scene.bin: not the 10 expected pages"),
            "bad-checkpoint": ("resume", b"checkpoint.bin: bad header or checksum")}


class Fail(Exception):
    pass


def check(cond: object, msg: str) -> None:
    if not cond:
        raise Fail(msg)


def fnv1a(data: bytes, bits: int) -> int:
    h, prime = (0x811C9DC5, 0x01000193) if bits == 32 else (0xCBF29CE484222325, 0x100000001B3)
    mask = (1 << bits) - 1
    for byte in data:
        h = ((h ^ byte) * prime) & mask
    return h


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tree_files(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): sha256(p.read_bytes()) for p in sorted(root.rglob("*")) if p.is_file()}


def tree_hash(files: dict[str, str]) -> str:
    return sha256(b"".join(rel.encode() + b"\0" + files[rel].encode() + b"\n" for rel in sorted(files)))


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as f:
        f.write(data)


def guest_path(path: Path, wine: bool) -> str:
    """The path as the guest sees it: native on Windows, through the Z: drive under Wine."""
    full = str(Path(path).resolve())
    return "Z:" + full.replace("/", "\\") if wine else full


def scene_bytes() -> bytes:
    return bytes((page * 29 + i * 7 + 11) & 255 for page in range(PAGES) for i in range(PAGE_SIZE))


def trace() -> list[tuple[int, int, int, int, int]]:
    """(pad buttons, x, y, scene byte, bias index) after each frame, derived from the input trace alone."""
    scene, steps, x, y = scene_bytes(), [], 8, 8
    for f in range(FRAMES):
        dx, dy = DIRECTIONS[f // 50]
        x, y = (x + dx) % 64, (y + dy) % 64
        a = f in A_FRAMES
        buttons = {-1: 256, 0: 0, 1: 512}[dx] | {-1: 64, 0: 0, 1: 128}[dy] | 4 * a
        value = scene[(f // PAGE_FRAMES) * PAGE_SIZE + (x + 3 * y) % PAGE_SIZE]
        steps.append((buttons, x, y, value, ((value >> 5) + bool(dx or dy) + 2 * a) % 4))
    return steps


def predicted_image(offset: int, bias: int) -> tuple[int, bytes]:
    """FNV-1a 64 of the RGBA32F readback and its P6 screenshot, computed in floats from the draw description."""
    palette = ((0.5, 0.125, 0.25, 0.5), (0.125, 1.0, 0.75, 1.0), (0.25, 1.125, 0.875, 1.0), (0.75, 0.25, 0.75, 1.0))
    colors = [tuple(c + (bias / 8 if i and ch < 3 else 0) for ch, c in enumerate(px)) for i, px in enumerate(palette)]
    binary = [struct.pack("<4f", *color) for color in colors]
    rgb = [bytes(min(255, int(v * 255 + 0.5)) for v in color[:3]) for color in colors]
    pixels = [int(offset <= col < offset + 32) if row < 32 else 2 if col >= 32 else 3 if row > col + 32 else 0
              for row in range(64) for col in range(64)]
    return fnv1a(b"".join(binary[p] for p in pixels), 64), b"P6\n64 64\n255\n" + b"".join(rgb[p] for p in pixels)


def check_reference(frames: bytes, steps: list[tuple[int, ...]], out: Path) -> int:
    images: dict[tuple[int, int], tuple[int, bytes]] = {}
    for f, record in enumerate(struct.iter_unpack("<8I", frames)):
        buttons, x, y, value, bias = steps[f]
        if (x % 8, bias) not in images:
            images[x % 8, bias] = predicted_image(x % 8, bias)
        digest, shot = images[x % 8, bias]
        check(record == (f, digest & 0xFFFFFFFF, digest >> 32, buttons, x, y, value, bias),
              f"independent scene/pixel prediction failed at frame {f}: got {record}")
        if f in SHOT_FRAMES:
            for mode in ("full", "save" if f < HALF else "resume"):
                check((out / mode / f"shot_{f:03d}.ppm").read_bytes() == shot,
                      f"{mode}: shot_{f:03d}.ppm differs from the pixel oracle")
    return len(images)


def check_state(path: Path, magic: bytes, frames: bytes, steps: list[tuple[int, ...]]) -> None:
    """Every word of a state record. The trace decides all but the chain, which is FNV-1a 64 over frames.bin."""
    data, what = path.read_bytes(), f"{path.parent.name}/{path.name}"
    check(len(data) == 4 * STATE_WORDS, f"{what}: {len(data)} bytes, want {4 * STATE_WORDS}")
    n = len(frames) // RECORD_SIZE
    buttons, x, y, value, bias = steps[n - 1]
    page, chain = (n - 1) // PAGE_FRAMES, fnv1a(frames, 64)
    want = (int.from_bytes(magic, "little"), VERSION, STATE_WORDS, n, x, y, buttons, sum(f < n for f in A_FRAMES),
            1, A_FRAMES[0], page, *struct.unpack_from("<4I", scene_bytes(), page * PAGE_SIZE), value, bias,
            chain & 0xFFFFFFFF, chain >> 32, fnv1a(data[:-4], 32))
    for i, (got, exp) in enumerate(zip(struct.unpack(f"<{STATE_WORDS}I", data), want)):
        check(got == exp, f"{what}: word {i} is {got:#x}, want {exp:#x}")


def expected_pcm(peak: int) -> bytes:
    """The guest's pulse: a 441 Hz square wave with 10 ms linear ramps at both ends, mono 16-bit."""
    samples = []
    for i in range(AUDIO_SAMPLES):
        amp = peak * min(i, AUDIO_SAMPLES - 1 - i, AUDIO_RAMP) // AUDIO_RAMP
        samples.append(-amp if (i // AUDIO_HALF_PERIOD) & 1 else amp)
    return struct.pack(f"<{AUDIO_SAMPLES}h", *samples)


def check_audio(mode_dir: Path, has_event: bool, play_audio: bool) -> dict:
    name, wav_path, bin_path = mode_dir.name, mode_dir / "event.wav", mode_dir / "audio.bin"
    if not has_event:
        check(not wav_path.exists() and not bin_path.exists(), f"{name}: replayed an earlier sound")
        return {"event": None}
    with wave.open(str(wav_path), "rb") as w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes(), w.getcomptype())
        pcm = w.readframes(w.getnframes())
    check(fmt == (1, 2, AUDIO_RATE, AUDIO_SAMPLES, "NONE"), f"{name}: event.wav is {fmt}, want mono 16-bit 44.1 kHz")
    check(wav_path.stat().st_size == 44 + 2 * AUDIO_SAMPLES, f"{name}: event.wav has extra chunks or bytes")
    peak = 9000
    check(pcm == expected_pcm(peak), f"{name}: event.wav samples differ from the generated pulse")
    raw = bin_path.read_bytes()
    check(len(raw) == 4 * AUDIO_WORDS, f"{name}: audio.bin is {len(raw)} bytes, want {4 * AUDIO_WORDS}")
    words = struct.unpack(f"<{AUDIO_WORDS}I", raw)
    want = (int.from_bytes(b"LSAU", "little"), VERSION, AUDIO_WORDS, A_FRAMES[0], AUDIO_RATE, AUDIO_SAMPLES,
            fnv1a(pcm, 32))
    check(words[:7] == want, f"{name}: audio.bin starts {words[:7]}, want {want}")
    check(words[9] == fnv1a(raw[:-4], 32), f"{name}: audio.bin checksum mismatch")
    playback, mmresult = words[7], words[8]
    check(playback in PLAYBACK, f"{name}: invalid playback result {playback}")
    if play_audio:
        check(playback in (1, 3), f"{name}: WinMM playback was not attempted")
    else:
        check(playback == 0 and mmresult == 0, f"{name}: offline run reports playback {playback}, MMRESULT {mmresult}")
    return {"event": "event.wav", "frame": words[3], "peak": peak, "pcm_fnv32": f"{words[6]:08x}",
            "wav_sha256": sha256(wav_path.read_bytes()), "playback": PLAYBACK[playback], "mmresult": mmresult}


class Session:
    """Starts the loader once per run and keeps each run's streams and record under <out>/logs."""

    def __init__(self, args: argparse.Namespace, out: Path) -> None:
        self.args, self.out, self.logs = args, out, out / "logs"
        self.runs: dict[str, dict] = {}
        self.last: list[Path] = []

    def run(self, name: str, mode: str, package: Path, overlay: Path) -> tuple[int, list[bytes], bytes]:
        a, wine = self.args, bool(self.args.wine)
        cmd = ([a.wine] if wine else []) + [str(a.loader), "--run", guest_path(a.guest, wine), "--game-dir",
                                            guest_path(package, wine), "--overlay", guest_path(overlay, wine),
                                            "--", guest_path(a.shaders, wine)]
        env_set = {"LEONIDA_SCENE_MODE": mode,
                   "LEONIDA_SCENE_AUDIO": "winmm" if a.play_audio else "offline",
                   "LEONIDA_KEYBOARD_PAD": "always", "LEONIDA_GAMEINPUT_THREAD": "0", "LEONIDA_D3D12": "vulkan",
                   "LEONIDA_SHADERS": guest_path(a.shaders / "spirv", wine), "LEONIDA_UNIMPLEMENTED": "stop"}
        env = dict(os.environ, **env_set)
        self.last = [self.logs / f"{name}.stdout.log", self.logs / f"{name}.stderr.log"]
        start = time.monotonic()
        rc, launch_error = None, None
        with self.last[0].open("xb") as so, self.last[1].open("xb") as se:
            try:
                rc = subprocess.run(cmd, cwd=overlay, env=env, stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                    timeout=TIMEOUT).returncode
            except subprocess.TimeoutExpired:
                outcome = "timed_out"
            except OSError as e:
                outcome = "launch_failed"
                launch_error = str(e)
            else:
                outcome = "exited"
        stdout, stderr = (p.read_bytes() for p in self.last)
        rec = {"cmd": cmd, "cwd": str(overlay), "env": env_set, "outcome": outcome, "returncode": rc, "timeout_s": TIMEOUT,
               "seconds": round(time.monotonic() - start, 3), "stdout_sha256": sha256(stdout),
               "stderr_sha256": sha256(stderr)}
        if launch_error is not None:
            rec["launch_error"] = launch_error
        write_new(self.logs / f"{name}.json", json.dumps(rec, indent=2).encode())
        self.runs[name] = rec
        if launch_error is not None:
            raise Fail(f"{name}: could not launch {cmd[0]}: {launch_error}")
        check(rc is not None, f"{name}: timed out after {TIMEOUT}s")
        return rc, [line.rstrip(b"\r") for line in stdout.splitlines()], stdout + stderr

    def expect_pass(self, mode: str, package: Path) -> None:
        first, end = MODES[mode]
        rc, lines, _ = self.run(mode, mode, package, self.out / mode)
        check(rc == 0, f"{mode}: exit code {rc}")
        want, got = MARKER + b" frames=%d" % (end - first), [line for line in lines if line.startswith(MARKER)]
        check(got == [want], f"{mode}: want one {want.decode()!r} line, got {[g.decode('ascii', 'replace') for g in got]}")

    def expect_fail(self, name: str, package: Path, overlay: Path) -> dict:
        mode, diagnostic = NEGATIVE[name]
        rc, lines, text = self.run(name, mode, package, overlay)
        check(rc == 1, f"{name}: exit code {rc}, want exactly 1")
        check(not any(line.startswith(MARKER) for line in lines), f"{name}: printed {MARKER.decode()}")
        check(diagnostic in text, f"{name}: no diagnostic naming {diagnostic.decode()}")
        for output in ("frames.bin", "state.bin"):
            check(not (overlay / output).exists(), f"{name}: published {output} after failing")
        return {"returncode": rc, "diagnostic": diagnostic.decode()}


def negative_controls(session: Session, out: Path) -> dict:
    bad_package, scene = out / "neg-scene-package", bytearray(scene_bytes())
    bad_package.mkdir()
    scene[0] ^= 0xFF
    write_new(bad_package / "scene.bin", bytes(scene))
    (out / "neg-bad-scene").mkdir()
    results = {"bad-scene": session.expect_fail("bad-scene", bad_package, out / "neg-bad-scene")}
    overlay, ckpt = out / "neg-bad-checkpoint", bytearray((out / "save" / "checkpoint.bin").read_bytes())
    overlay.mkdir()
    ckpt[20] ^= 0xFF
    write_new(overlay / "checkpoint.bin", bytes(ckpt))
    results["bad-checkpoint"] = session.expect_fail("bad-checkpoint", out / "package", overlay)
    return results


def execute(args: argparse.Namespace, out: Path, temporary: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)  # parse_args has checked that it is new or empty
    session = Session(args, out)
    report: dict = {"status": "fail", "frames": FRAMES, "headless": True, "wine": args.wine, "temporary": temporary,
                    "audio_mode": "winmm" if args.play_audio else "offline", "report": str(out / "report.json")}
    try:
        shader_files = tree_files(args.shaders)
        report["inputs"] = {
            "loader": {"path": str(args.loader), "sha256": sha256(args.loader.read_bytes())},
            "guest": {"path": str(args.guest), "sha256": sha256(args.guest.read_bytes())},
            "shaders": {"path": str(args.shaders), "files": len(shader_files), "tree_sha256": tree_hash(shader_files)},
            "scene.bin": sha256(scene_bytes())}
        for sub in ("logs", "package", *MODES):
            (out / sub).mkdir()
        package = out / "package"
        write_new(package / "scene.bin", scene_bytes())
        report["runs"] = session.runs
        for mode in MODES:
            if mode == "resume":
                write_new(out / mode / "checkpoint.bin", (out / "save" / "checkpoint.bin").read_bytes())
            session.expect_pass(mode, package)
        frames = {m: (out / m / "frames.bin").read_bytes() for m in MODES}
        for m, (first, end) in MODES.items():
            check(len(frames[m]) == (end - first) * RECORD_SIZE, f"{m}: frames.bin is {len(frames[m])} bytes")
            check(not list((out / m).glob("*.tmp")), f"{m}: an unpublished .tmp file was left behind")
        check(frames["save"] + frames["resume"] == frames["full"], "save+resume frames.bin differ from full")
        steps = trace()
        report["predicted_images"] = check_reference(frames["full"], steps, out)
        report["frame_chain"] = f"{fnv1a(frames['full'], 64):016x}"
        check_state(out / "save" / "state.bin", b"LSST", frames["save"], steps)
        for m in ("full", "resume"):
            check_state(out / m / "state.bin", b"LSST", frames["full"], steps)
        check_state(out / "save" / "checkpoint.bin", b"LSCK", frames["save"], steps)
        for m in ("full", "resume"):
            check((out / m / "checkpoint.bin").read_bytes() == (out / "save" / "checkpoint.bin").read_bytes(),
                  f"{m}: checkpoint.bin differs from the save run's")
        report["audio"] = {m: check_audio(out / m, m != "resume", args.play_audio) for m in MODES}
        check((out / "full" / "event.wav").read_bytes() == (out / "save" / "event.wav").read_bytes(),
              "event.wav differs between full and save")
        if args.negative_controls:
            report["negative_controls"] = negative_controls(session, out)
        check(tree_files(package) == {"scene.bin": sha256(scene_bytes())}, "a guest modified the synthetic package")
        report["status"] = "pass"
    except (Fail, OSError, ValueError, struct.error, wave.Error, EOFError) as e:
        report["error"] = str(e) if isinstance(e, Fail) else f"{type(e).__name__}: {e}"
        for log in session.last:
            if log.is_file():
                tail = log.read_bytes().decode("utf-8", "replace").splitlines()[-40:]
                print(f"--- last {len(tail)} lines of {log} ---", *tail, sep="\n", file=sys.stderr)
    finally:
        report["outputs"] = {d.name: {"tree_sha256": tree_hash(files), "files": files} for d in sorted(out.iterdir())
                             if d.is_dir() and d.name != "logs" for files in [tree_files(d)]}
        write_new(out / "report.json", json.dumps(report, indent=2, sort_keys=True).encode())
    print(f"{report['status'].upper()}: {report.get('error', 'all guest scene checks passed')}")
    if not temporary:
        print(out / "report.json")
    return 0 if report["status"] == "pass" else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--loader", type=Path, required=True)
    ap.add_argument("--guest", type=Path, required=True)
    ap.add_argument("--shaders", type=Path, required=True, help="shader root containing spirv/")
    ap.add_argument("--out", type=Path, help="new or empty folder (default: a temporary folder, removed afterwards)")
    ap.add_argument("--wine", help="run the loader through this Wine binary")
    ap.add_argument("--play-audio", action="store_true", help="LEONIDA_SCENE_AUDIO=winmm instead of offline")
    ap.add_argument("--negative-controls", action="store_true", help="also run a bad scene.bin and checkpoint")
    args = ap.parse_args()
    for name in ("loader", "guest"):
        if not getattr(args, name).is_file():
            ap.error(f"--{name}: no such file")
    if not (args.shaders / "spirv").is_dir():
        ap.error("--shaders: no spirv/ folder under it")
    if args.wine:
        found = args.wine if Path(args.wine).is_file() else shutil.which(args.wine)
        if not found:
            ap.error("--wine: not found")
        args.wine = str(Path(found).resolve())
    elif os.name != "nt":
        ap.error("--wine is required outside Windows")
    if args.out and args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        ap.error("--out must be new or an empty folder")
    args.loader, args.guest, args.shaders = args.loader.resolve(), args.guest.resolve(), args.shaders.resolve()
    return args


def main() -> int:
    args = parse_args()
    if args.out:
        return execute(args, args.out.resolve(), temporary=False)
    with tempfile.TemporaryDirectory(prefix="guest-scene-", ignore_cleanup_errors=True) as tmp:
        return execute(args, Path(tmp) / "run", temporary=True)


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    # Redirected Windows streams may use a code page that cannot encode a path in the report.
    sys.stdout.reconfigure(errors="backslashreplace")
    sys.stderr.reconfigure(errors="backslashreplace")
    sys.exit(main())
