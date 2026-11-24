"""Survey the .rpf archives in a game folder: version magic, header words, and whether the table looks encrypted.

Usage:
    python -I rpf_probe.py <game_dir> [--out FILE]

Reads the first 64 KiB of each archive only. It doesn't decrypt or extract anything.
A directory scan or archive read error stops the survey before it writes the report.
The report destination must not name an input archive or an alias of one.
"""
import argparse
import json
import math
import os
import struct
import sys
import tempfile
from collections import Counter

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402
from report_paths import same_report_path  # noqa: E402


def entropy(b: bytes) -> float:
    if not b:
        return 0.0
    n = len(b)
    return -sum(c / n * math.log2(c / n) for c in Counter(b).values())


def probe(path: str) -> dict:
    with open(path, "rb") as f:
        head = f.read(0x10000)
    magic = head[:4]
    # Little-endian 'RPFn' is stored as bytes 'nFPR'.
    version = magic[::-1].decode("latin-1") if magic[1:] == b"FPR" else None
    words = list(struct.unpack_from("<4I", head, 4)) if len(head) >= 20 else []
    body = head[16:]
    return {
        "file": path,
        "size": os.path.getsize(path),
        "version": version,
        "header_words": [hex(w) for w in words],
        "head_entropy": round(entropy(body), 3),
        "looks_encrypted": entropy(body) > 7.9,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game_dir")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "reports", "rpf_probe.json"))
    a = ap.parse_args()
    rows = []

    def scan_failed(error: OSError) -> None:
        raise error

    try:
        archives = []
        for root, _, files in os.walk(a.game_dir, onerror=scan_failed):
            for name in files:
                if name.lower().endswith(".rpf"):
                    archives.append(os.path.join(root, name))
        for path in archives:
            if same_report_path(a.out, path):
                print(f"ERROR: --out would overwrite an input archive: {path}", file=sys.stderr)
                return 2
        for path in archives:
            rows.append(probe(path))
    except OSError as error:
        path = error.filename or a.game_dir
        reason = error.strerror or str(error)
        print(f"ERROR: {path}: {reason}", file=sys.stderr)
        return 2
    rows.sort(key=lambda r: (-r["size"], r["file"]))
    try:
        out_dir = os.path.dirname(os.path.abspath(a.out))
        os.makedirs(out_dir, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_dir) as staging:
            staged = os.path.join(staging, "rpf_probe.json")
            with open(staged, "w", encoding="utf-8", newline="\n") as f:
                json.dump(rows, f, indent=1)
            os.chmod(staged, published_mode(a.out))
            os.replace(staged, a.out)
    except OSError as error:
        reason = error.strerror or str(error)
        print(f"ERROR: {a.out}: {reason}", file=sys.stderr)
        return 2
    versions = Counter(r["version"] for r in rows)
    print(f"{len(rows)} archives, {sum(r['size'] for r in rows) / 2**30:.1f} GiB, versions {dict(versions)}")
    for r in rows:
        rel = os.path.relpath(r["file"], a.game_dir)
        print(f"{rel:<52} {r['size'] / 2**30:7.2f} GiB  {r['version']}  {' '.join(r['header_words'])}  "
              f"H={r['head_entropy']}{'  packed/encrypted' if r['looks_encrypted'] else ''}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
