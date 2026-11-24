"""Group every carved file by signature, using only the per-archive index.json (no file reads).

Classes:
  bink2        Bink 2 video (KB2x)
  rbf0         Rockstar binary metadata ('RBF0', carries its own field names)
  respage      resource page ('42 40 00 00': tagged-pointer structures for textures/meshes/etc.)
  gpu-zerohdr  starts with 16+ zero bytes (GPU data: textures, vertex/index buffers)
  other:<hex>  anything else, keyed by its first 4 bytes

Usage:
    python -I tools/classify_carve.py <work>\\carve
Writes types.json (per archive and overall counts/bytes per class, plus size stats) and prints a summary.
Every per-class entry, overall and per archive, is {"files": N, "bytes": B, "min": ..., "max": ..., "median": ...}:
min, max and median are the smallest, largest and median file size in bytes among the files of that class in that
scope. For an even count the median is the lower of the two middle sizes (statistics.median_low), so every value
in the report is an integer. A class only appears when it has at least one file, so the stats keys are never null.
The tool keeps one size (a plain Python int) per file per class to compute these.
"""
import json
import os
import sys
import tempfile
from collections import defaultdict
from statistics import median_low

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402


def classify(r):
    if r.get("type") == "bink2":
        return "bink2"
    h = r["head"]
    if h.startswith("52424630"):
        return "rbf0"
    if h.startswith("42400000"):
        return "respage"
    if h.startswith("00000000000000000000000000000000"):
        return "gpu-zerohdr"
    return "other:" + h[:8]


def index_rows(path):
    with open(path, encoding="utf-8") as f:
        try:
            document = json.load(f)
        except ValueError:
            raise ValueError("invalid JSON") from None
    rows = document.get("files") if isinstance(document, dict) else None
    if not isinstance(rows, list):
        raise ValueError('expected a "files" list')

    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict) or type(row.get("size")) is not int or row["size"] < 0:
            raise ValueError(f"row {number} needs a nonnegative integer size")
        if row.get("type") != "bink2" and not isinstance(row.get("head"), str):
            raise ValueError(f"row {number} needs a string head")
        yield row


def entry(v):
    sizes = v[2]
    return {"files": v[0], "bytes": v[1], "min": min(sizes), "max": max(sizes), "median": median_low(sizes)}


USAGE = "usage: python -I tools/classify_carve.py <work>\\carve"


def main():
    if sys.argv[1:] in (["-h"], ["--help"]):
        print(USAGE)
        return 0
    if len(sys.argv) != 2:
        print(USAGE, file=sys.stderr)
        return 2
    root = sys.argv[1]
    if not os.path.isdir(root):
        what = "is not a folder" if os.path.exists(root) else "does not exist"
        print(f"{root}: {what}; give the carve folder (the one with an index.json per archive)", file=sys.stderr)
        return 2
    report = os.path.join(root, "types.json")
    if os.path.isdir(report):
        print(f"{report}: is a directory; remove or rename that directory before rerun", file=sys.stderr)
        return 2
    per = {}
    total = defaultdict(lambda: [0, 0, []])
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "index.json")
        if not os.path.isfile(p):
            continue
        c = defaultdict(lambda: [0, 0, []])
        try:
            for r in index_rows(p):
                k = classify(r)
                c[k][0] += 1
                c[k][1] += r["size"]
                c[k][2].append(r["size"])
                total[k][0] += 1
                total[k][1] += r["size"]
                total[k][2].append(r["size"])
        except (OSError, ValueError) as error:
            print(f"{p}: {error}", file=sys.stderr)
            return 2
        per[name] = {k: entry(v) for k, v in sorted(c.items(), key=lambda kv: -kv[1][1])}
    top = sorted(total.items(), key=lambda kv: -kv[1][1])
    out = {"overall": {k: entry(v) for k, v in top}, "per_archive": per}
    fd, temp = tempfile.mkstemp(dir=root, prefix=".types-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(out, f, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp, published_mode(report))                             # mkstemp creates 0600
        os.replace(temp, report)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise
    print(f"{'class':24} {'files':>10} {'GiB':>8}")
    for k, v in top[:25]:
        print(f"{k:24} {v[0]:>10} {v[1] / 2**30:8.2f}")
    print(f"{len(top)} classes in total")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
