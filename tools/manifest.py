"""SHA-256 manifest of a game folder, so later copies (and the exe, when it arrives) can be checked.

Usage:
    python -I manifest.py create <game_dir> [--out FILE] [--jobs N]
    python -I manifest.py verify <game_dir> [--manifest FILE] [--jobs N]

Files are hashed with up to --jobs threads (default: half the CPU count; 1 = serial).
The manifest itself is excluded when its path is inside the scanned folder.
On Windows it lowers its own priority to BelowNormal so the PC stays usable.
"""
import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

CHUNK = 8 * 1024 * 1024
DEFAULT = os.path.join(os.path.dirname(__file__), "..", "reports", "manifest.json")


class ManifestError(Exception):
    """An input or manifest error that the CLI reports without a traceback."""


def default_jobs() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


def positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def below_normal() -> None:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k32 = ctypes.windll.kernel32
        k32.GetCurrentProcess.argtypes = ()
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k32.SetPriorityClass.restype = wintypes.BOOL
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS


def _file_stamp(st: os.stat_result) -> tuple[int, int, int]:
    return st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _hash_file(path: str, *, expected_size: int | None = None) -> tuple[int, str]:
    """Return the size and digest of the same open file, or fail if it changes."""
    h = hashlib.sha256()
    with open(path, "rb", buffering=0) as f:
        before = os.fstat(f.fileno())
        size = before.st_size
        if expected_size is not None and size != expected_size:
            raise ManifestError(f"{path}: file size changed before hashing")
        remaining = size
        while remaining and (chunk := f.read(min(CHUNK, remaining))):
            h.update(chunk)
            remaining -= len(chunk)
        # Compare timestamps from the same handle. Windows can report different times by path.
        # Compare the path by identity so a replacement of the open file is detected too.
        if (remaining or _file_stamp(os.fstat(f.fileno())) != _file_stamp(before)
                or not os.path.samestat(before, os.stat(path))):
            raise ManifestError(f"{path}: file changed while hashing")
    return size, h.hexdigest()


def sha256(path: str, *, expected_size: int | None = None) -> str:
    return _hash_file(path, expected_size=expected_size)[1]


def walk(game_dir: str, *, exclude=None):
    def scan_failed(error: OSError) -> None:
        raise error

    excluded_dir = excluded_name = None
    if exclude is not None:
        directory, name = os.path.split(os.path.abspath(exclude))
        excluded_dir = os.path.normcase(os.path.realpath(directory))
        excluded_name = os.path.normcase(name)

    # os.walk otherwise treats missing or unreadable directories as empty.
    for root, _, files in os.walk(game_dir, onerror=scan_failed):
        contains_manifest = excluded_dir is not None and os.path.normcase(os.path.realpath(root)) == excluded_dir
        for name in sorted(files):
            if contains_manifest and os.path.normcase(name) == excluded_name:
                continue
            p = os.path.join(root, name)
            rel = os.path.relpath(p, game_dir)
            if os.sep != "/":
                rel = rel.replace(os.sep, "/")
            if os.altsep and os.altsep != "/":
                rel = rel.replace(os.altsep, "/")
            yield rel, p


def create(game_dir: str, out: str, jobs: int = 1) -> int:
    # walk() excludes the manifest path itself, so an existing game file named like --out would otherwise be
    # hashed first and replaced by the manifest (os.replace below), leaving the recorded hash wrong. Allow only an
    # existing file that already reads as a manifest (re-creating one in place); refuse any other existing target
    # inside the folder and every hard link to a file that will be hashed.
    if os.path.isdir(out):
        raise ManifestError(f"--out is a folder, not a manifest file: {out}")
    if os.path.isfile(out):
        try:
            inside = os.path.normcase(os.path.commonpath(
                [os.path.realpath(game_dir), os.path.realpath(out)])) == os.path.normcase(os.path.realpath(game_dir))
        except ValueError:
            inside = False
        if inside:
            try:
                load_manifest(out)
            except (ManifestError, OSError):
                raise ManifestError(f"--out would overwrite the input {out}; "
                                    "choose a manifest path that is not one of the hashed files") from None
    files = sorted(walk(game_dir, exclude=out))
    for _rel, path in files:
        if same_report_path(out, path):
            raise ManifestError(f"--out would overwrite the input {path}; "
                                "choose a manifest path that is not one of the hashed files")
    sizes = {path: os.path.getsize(path) for _, path in files}
    total = sum(sizes.values())
    entries = {}
    done = count = 0
    t0 = time.time()

    def finish(rel: str, size: int, digest: str) -> None:
        nonlocal done, count
        entries[rel] = {"size": size, "sha256": digest}
        done += size
        count += 1
        rate = done / max(time.time() - t0, 1e-6) / 2**20
        pct = done / total if total else 1.0
        # One line per completed file, printed only from the main thread. Completion order is not
        # stable with threads, so this reports counts and bytes rather than coupling output to a path.
        print(f"[{pct:6.1%}] {count}/{len(files)} files {rate:7.0f} MiB/s", flush=True)

    def work(rel: str, path: str) -> tuple[str, int, str]:
        size, digest = _hash_file(path, expected_size=sizes[path])
        return rel, size, digest

    if jobs == 1:
        for rel, p in files:
            finish(*work(rel, p))
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = [pool.submit(work, rel, p) for rel, p in files]
            for future in as_completed(futures):
                finish(*future.result())

    # Re-insert in walk order so the JSON is independent of which worker finished first.
    entries = {rel: entries[rel] for rel, _ in files}
    out_dir = os.path.dirname(os.path.abspath(out))
    os.makedirs(out_dir, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_dir) as staging:
        staged = os.path.join(staging, "manifest.json")
        with open(staged, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": entries}, f, indent=1)
        os.replace(staged, out)
    print(f"{len(entries)} files, {total / 2**30:.1f} GiB -> {out}")
    return 0


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate key {key!r}")
        out[key] = value
    return out


def load_manifest(manifest: str) -> dict:
    """The manifest's "files" table, checked entry by entry, so a damaged file is reported by name rather than
    as a JSONDecodeError, KeyError or TypeError traceback halfway through verify."""
    with open(manifest, encoding="utf-8") as f:
        try:
            doc = json.load(f, object_pairs_hook=_reject_duplicate_keys)
        except ValueError as error:  # JSONDecodeError, duplicate keys, or UnicodeDecodeError for non-UTF-8
            raise ManifestError(f"{manifest} is not valid JSON ({error}); create it again") from None
    files = doc.get("files") if isinstance(doc, dict) else None
    if not isinstance(files, dict):
        raise ManifestError(f"{manifest} has no \"files\" table; create it again")
    for rel, info in files.items():
        if not (isinstance(info, dict) and type(info.get("size")) is int and isinstance(info.get("sha256"), str)):
            raise ManifestError(f"{manifest}: entry {rel!r} needs an integer \"size\" and a \"sha256\" string")
    return files


def verify(game_dir: str, manifest: str, jobs: int = 1) -> int:
    want = load_manifest(manifest)
    have = dict(walk(game_dir, exclude=manifest))
    if not want and have:
        # A create that was cut short, or a mistyped --manifest, would otherwise print OK without checking a file.
        raise ManifestError(f"{manifest} lists no files, but {game_dir} holds {len(have)}; create it again")
    problems = {}
    pending = []
    for rel, info in want.items():
        p = have.get(rel)
        if p is None:
            problems[rel] = "MISSING"
        elif os.path.getsize(p) != info["size"]:
            problems[rel] = "SIZE"
        else:
            pending.append((rel, p, info["sha256"].lower()))

    def hash_pending(rel: str, path: str) -> str:
        return sha256(path, expected_size=want[rel]["size"])

    if jobs == 1:
        for rel, p, digest in pending:
            if hash_pending(rel, p) != digest:
                problems[rel] = "CHANGED"
    else:
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            futures = {pool.submit(hash_pending, rel, p): (rel, digest) for rel, p, digest in pending}
            for future in as_completed(futures):
                rel, digest = futures[future]
                if future.result() != digest:
                    problems[rel] = "CHANGED"

    # Print in manifest order, not completion order, so the report is identical for any --jobs.
    bad = 0
    for rel in want:
        kind = problems.get(rel)
        if kind:
            print(f"{kind:<10}{rel}")
            bad += 1
    for rel in sorted(set(have) - set(want)):
        print(f"{'NEW':<10}{rel}")
    print("OK" if not bad else f"{bad} problem(s)")
    return 1 if bad else 0


def add_jobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--jobs", type=positive_int, default=default_jobs(), metavar="N",
        help="hash up to N files concurrently with threads; 1 hashes serially "
             "(default: half the CPU count = %(default)s)")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("game_dir"); c.add_argument("--out", default=DEFAULT)
    add_jobs(c)
    v = sub.add_parser("verify"); v.add_argument("game_dir"); v.add_argument("--manifest", default=DEFAULT)
    add_jobs(v)
    a = ap.parse_args()
    below_normal()
    try:
        if a.cmd == "create":
            return create(a.game_dir, a.out, a.jobs)
        return verify(a.game_dir, a.manifest, a.jobs)
    except (OSError, ManifestError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
