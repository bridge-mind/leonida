"""Name the Bink 2 streams of an RPF archive from the game's own mount lists.

The RPF9 table of contents is encrypted, so ``bink_index.py`` can only report offsets. The names are still in the
package: every pack registers its videos in a ``CDataFileMgr__ContentChangeSetDef`` mount list (``core_binks``,
``story000_binks``, ...) as ``<filename>$(common)/movies/<folder>/<name>.bk2</filename>``. The archive stores the
streams of a pack in the order of that list once it is sorted by lowercase path (folder, then file name without
the extension, in plain byte order). A list can name more files than its archive holds, because some entries live
in other packs, so the list and the streams are aligned rather than zipped.

The alignment is a monotone dynamic programme over (list entry, stream in offset order) that may skip list
entries, never streams. It scores:

* name hints against the header: ``_720``/``_1080``/``_1440``/``_4k`` against the height, ``snapmatic/`` against a
  portrait frame, ``tv/sprt/`` against 1280x720;
* siblings: consecutive matches of one family (the name without trailing numbers or side/size words) should share
  width, height and frame count, and unrelated neighbours get a small bonus when they do;
* optional pins: a CSV of ``offset,path`` from someone who watched the stream (``path`` may list alternatives
  separated by ``|``).

Each name gets a confidence:

``verified``   a pin names exactly this path;
``anchored``   a family neighbour agrees on the header, a name hint matched, or a pin allows this path;
``bracketed``  between the nearest anchored matches (or the ends) the number of names equals the number of
               streams, so no skipped entry can hide there;
``positional`` only the order supports it.

Usage:
    python -I bink_names.py <movies.rpf | dir | index.csv | index.json> --mount <LIST.xml | common.rpf | dir> [...]
                            [--pins pins.csv] [--max-skip N] [--out names.csv|.json]

The source is read with ``bink_index.py`` (an archive or directory is scanned; a CSV or JSON written by
``bink_index.py --out`` is reused). ``--mount`` takes list files, an archive (``common.rpf``), an ``rpf9_carve.py``
directory, or any directory of files; only payloads that mention ``$(common)/movies/`` are used. A list longer than
one zlib stream is rebuilt whole by ``gamedata_index.py``. The output names game files: it is a local work product and must not be
committed or attached to an issue. The tests use synthetic headers and lists only.
Usage: `python -I tools/bink_names.py --help` lists the options.
"""
import argparse
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import bink_index  # noqa: E402
import gamedata_index  # noqa: E402
import gamedata_xml  # noqa: E402

LIST_ENTRY = re.compile(r"<filename>\s*\$\(common\)/movies/([^<]+?\.bk2)\s*</filename>", re.I)
FAMILY_TAIL = re.compile(r"(_?\d+|_?(4k|left|right|top|topleft|topright|side|high|low|medium|light|dark|close|far))+$",
                         re.I)
LIST_MARKER = b"$(common)/movies/"
MAX_LIST_BYTES = 16 << 20
HEIGHT_HINTS = (("_720", 720), ("_1080", 1080), ("_1440", 1440), ("_4k", 2160))
CONFIDENCES = ("verified", "anchored", "bracketed", "positional")
FIELDS = ["source", "offset", "size", "path", "confidence", "width", "height", "frames"]
REPORT_FIELDS = ["source", "offset", "size", "width", "height", "frames"]

PIN_SCORE = 50.0
HINT_SCORE = 3.0
SNAPMATIC_SCORE = 2.0
TV_SCORE = 1.0
FAMILY_SCORE = 2.0
NEIGHBOUR_SCORE = 0.3
SKIP_COST = 0.6
MAX_SKIP = 30


def sort_key(path):
    """Archive order: folder, then file name without the extension, both lowercase and compared as bytes."""
    folder, _, name = path.lower().rpartition("/")
    return folder.encode(), name.rsplit(".", 1)[0].encode()


def _archive_documents(archive):
    """XML payloads of an archive, streams joined by gamedata_index (nothing written to disk)."""
    source = gamedata_index.ArchiveSource(archive)
    try:
        by_offset = {row["off"]: k for k, row in enumerate(source.rows)}
        for row in gamedata_index.build_rows(source.rows, source):
            if row["type"] == "xml":
                start = by_offset[row["offset"]]
                yield b"".join(source.read(source.rows[k]) for k in range(start, start + row["chunks"]))
    finally:
        source.close()


def mount_documents(paths):
    """Byte payloads that may hold mount lists.

    A carve directory (``index.json`` from rpf9_carve.py) and an ``.rpf`` are read as whole XML payloads, so a list
    split over several zlib streams stays in one piece. Any other directory is searched file by file.
    """
    def scan_failed(error):
        raise error

    for path in paths:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "index.json")):
            for doc in gamedata_xml.carve_documents(path):
                yield doc["data"]
        elif os.path.isfile(path) and path.lower().endswith(".rpf"):
            yield from _archive_documents(path)
        elif os.path.isdir(path):
            for root, _dirs, files in os.walk(path, onerror=scan_failed):
                for name in sorted(files):
                    full = os.path.join(root, name)
                    if os.path.getsize(full) <= MAX_LIST_BYTES:
                        with open(full, "rb") as fh:
                            yield fh.read()
        else:
            with open(path, "rb") as fh:
                yield fh.read()


def read_mount_lists(paths):
    """Every distinct movie path the lists mention, in archive order."""
    seen, names = set(), []
    for data in mount_documents(paths):
        if LIST_MARKER not in data.lower():
            continue
        text = data.decode("utf-8", errors="replace")
        for name in LIST_ENTRY.findall(text):
            name = name.strip().replace("\\", "/")
            if name.lower() not in seen:
                seen.add(name.lower())
                names.append(name)
    return sorted(names, key=sort_key)


def _int(value):
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError(f"expected an integer, got {type(value).__name__}")
    return int(value, 0) if isinstance(value, str) else value


def load_streams(source, skip_report=None):
    """Stream rows (source, offset, size, width, height, frames) in archive order."""
    lower = source.lower()
    if os.path.isfile(source) and lower.endswith(".json"):
        with open(source, encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict) or not isinstance(doc.get("streams"), list):
            raise ValueError("expected a JSON object with a 'streams' list")
        rows = doc["streams"]
        label, start = "stream {i}", 1
    elif os.path.isfile(source) and lower.endswith(".csv"):
        with open(source, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            header = reader.fieldnames or []
            missing = [f for f in REPORT_FIELDS if f not in header]
            if missing:
                raise ValueError(f"CSV header is missing required fields: {', '.join(missing)}")
            rows = list(reader)
        label, start = "row {i}", 2
    else:
        rows, _summary = bink_index.scan_target(source, skip_report=skip_report)
        label, start = "stream {i}", 1
    out = []
    for j, row in enumerate(rows):
        try:
            if not isinstance(row, dict):
                raise TypeError("row is not an object")
            offset = _int(row["offset"])
            size = _int(row["size"])
            width = _int(row["width"])
            height = _int(row["height"])
            frames = _int(row["frames"])
            if offset < 0 or size <= 0 or width <= 0 or height <= 0 or frames <= 0:
                raise ValueError("negative offset or non-positive stream field")
            out.append({"source": row["source"], "offset": offset, "size": size,
                        "width": width, "height": height, "frames": frames})
        except (KeyError, ValueError, TypeError, OverflowError) as error:
            raise ValueError(f"{label.format(i=start + j)} is missing or unusable: {error}") from None
        if not isinstance(out[-1]["source"], str):
            raise ValueError(f"{label.format(i=start + j)} has a non-string source")
        out[-1]["source"] = out[-1]["source"].replace("\\", "/")  # an older Windows index wrote "\"
    # A directory of carved files sorts by file name; carves name files by archive offset.
    out.sort(key=lambda r: (r["source"], r["offset"]))
    return out


def _normalize_pin_path(raw):
    path = raw.strip().replace("\\", "/").lower()
    if path.startswith("$(common)/movies/"):
        path = path[len("$(common)/movies/"):]
    return path


def load_pins(path):
    """{(source or "", offset): set of lowercase paths} from a CSV with offset,path[,source]."""
    pins = {}
    if not path:
        return pins
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        missing = [f for f in ("offset", "path") if f not in header]
        if missing:
            raise ValueError(f"CSV header is missing required fields: {', '.join(missing)}")
        for j, row in enumerate(reader, start=2):
            try:
                offset = _int(row["offset"])
                if offset < 0:
                    raise ValueError("offset must be non-negative")
                key = ((row.get("source") or "").strip().replace("\\", "/"), offset)  # pins from before "/" too
                raw_path = row["path"]
                if not isinstance(raw_path, str):
                    raise TypeError("path is missing")
                parsed = [item for item in (_normalize_pin_path(p) for p in raw_path.split("|")) if item]
                if not parsed:
                    raise ValueError("empty path")
            except (KeyError, ValueError, TypeError, OverflowError) as error:
                raise ValueError(f"row {j} is missing or unusable: {error}") from None
            pins.setdefault(key, set()).update(parsed)
    return pins


def family(name):
    return FAMILY_TAIL.sub("", name.rsplit(".", 1)[0].lower())


def _shape(stream):
    return stream["width"], stream["height"], stream["frames"]


class Scorer:
    def __init__(self, pins):
        self.pins = pins
        self.pinned = {next(iter(p)) for p in pins.values() if len(p) == 1}

    def pin(self, stream):
        return self.pins.get((stream["source"], stream["offset"]), self.pins.get(("", stream["offset"])))

    def unary(self, name, stream):
        n = name.lower()
        score = 0.0
        pin = self.pin(stream)
        if pin is not None:
            score += PIN_SCORE if n in pin else -PIN_SCORE
        elif n in self.pinned:
            score -= PIN_SCORE
        for tag, height in HEIGHT_HINTS:
            if tag in n:
                score += HINT_SCORE if stream["height"] == height else -HINT_SCORE
        if n.startswith("snapmatic/"):
            score += SNAPMATIC_SCORE if stream["height"] > stream["width"] else -SNAPMATIC_SCORE
        if n.startswith("tv/sprt/"):
            score += TV_SCORE if (stream["width"], stream["height"]) == (1280, 720) else -TV_SCORE
        return score


def pair(name_a, stream_a, name_b, stream_b):
    same = _shape(stream_a) == _shape(stream_b)
    if family(name_a) == family(name_b):
        return FAMILY_SCORE if same else -FAMILY_SCORE
    return NEIGHBOUR_SCORE if same else 0.0


def align(names, streams, scorer, skip_cost=SKIP_COST, max_skip=MAX_SKIP):
    """Best monotone assignment of one list entry per stream: [(name index, stream index)]."""
    n_names, n_streams = len(names), len(streams)
    if n_streams == 0:
        return []
    if n_names < n_streams:
        raise ValueError(f"the lists name {n_names} movies but the source has {n_streams} streams")
    slack = n_names - n_streams          # entries that can be skipped in total
    neg = float("-inf")
    score = [dict() for _ in range(n_streams)]
    back = [dict() for _ in range(n_streams)]
    for i in range(min(slack, max_skip) + 1):
        score[0][i] = scorer.unary(names[i], streams[0]) - skip_cost * i
    for j in range(1, n_streams):
        prev = score[j - 1]
        for i in range(j, j + slack + 1):
            best, arg = neg, None
            for p in range(max(j - 1, i - 1 - max_skip), i):
                if p not in prev:
                    continue
                v = prev[p] - skip_cost * (i - p - 1) + pair(names[p], streams[j - 1], names[i], streams[j])
                if v > best:
                    best, arg = v, p
            if arg is not None:
                score[j][i] = best + scorer.unary(names[i], streams[j])
                back[j][i] = arg
    last = {i: value for i, value in score[-1].items() if n_names - 1 - i <= max_skip}
    if not last:
        raise ValueError("no alignment within --max-skip; the lists do not match this source")
    i = max(last, key=lambda k: last[k] - skip_cost * (n_names - 1 - k))
    path = []
    for j in range(n_streams - 1, -1, -1):
        path.append((i, j))
        i = back[j].get(i)
    return path[::-1]


def classify(names, streams, path, scorer):
    """Confidence per aligned pair, in path order."""
    unaries = [scorer.unary(names[i], streams[j]) for i, j in path]
    anchored = []
    for k, (i, j) in enumerate(path):
        neighbours = [(k + d, *path[k + d]) for d in (-1, 1) if 0 <= k + d < len(path)]
        fam = any(unaries[k2] >= 0 and family(names[i2]) == family(names[i]) and _shape(streams[j2]) == _shape(streams[j])
                  for k2, i2, j2 in neighbours)
        anchored.append(unaries[k] > 0 or (fam and unaries[k] >= 0))
    out = []
    for k, (i, j) in enumerate(path):
        pin = scorer.pin(streams[j]) or set()
        name = names[i].lower()
        if pin == {name}:
            out.append("verified")
        elif name in pin or anchored[k]:
            out.append("anchored")
        elif unaries[k] < 0:
            out.append("positional")
        else:
            a = max((x for x in range(k) if anchored[x]), default=None)
            b = min((x for x in range(k + 1, len(path)) if anchored[x]), default=None)
            ia, ja = path[a] if a is not None else (-1, -1)
            ib, jb = path[b] if b is not None else (len(names), len(streams))
            out.append("bracketed" if ib - ia == jb - ja else "positional")
    return out


def name_streams(names, streams, pins=None, max_skip=MAX_SKIP):
    """Rows (one per stream) and the list entries that matched no stream."""
    scorer = Scorer(pins or {})
    path = align(names, streams, scorer, max_skip=max_skip)
    conf = classify(names, streams, path, scorer)
    rows = []
    for (i, j), c in zip(path, conf):
        s = streams[j]
        rows.append({"source": s["source"], "offset": s["offset"], "size": s["size"], "path": names[i],
                     "confidence": c, "width": s["width"], "height": s["height"], "frames": s["frames"]})
    used = {i for i, _ in path}
    unmatched = [n for k, n in enumerate(names) if k not in used]
    return rows, unmatched


def write(out, rows, unmatched, summary):
    if out.lower().endswith(".json"):
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump({"summary": summary, "streams": rows, "unmatched_list_entries": unmatched}, fh, indent=1)
    else:
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive, a directory, or a bink_index.py CSV/JSON")
    ap.add_argument("--mount", action="append", required=True,
                    help="a mount-list XML, an archive or carve directory holding the lists, or a directory to "
                         "search (repeatable)")
    ap.add_argument("--pins", help="CSV with offset,path[,source] of streams identified by viewing them")
    ap.add_argument("--max-skip", type=int, default=MAX_SKIP,
                    help="most list entries skipped before, between or after stream matches "
                         "(an empty source leaves all entries unmatched)")
    ap.add_argument("--out", help="names.csv or names.json; index.json in a source or mount directory is reserved "
                                  "for the carve manifest (default: a summary only)")
    a = ap.parse_args(argv)
    if a.max_skip < 0:
        ap.error("--max-skip must be 0 or more")
    inputs = [a.source, *a.mount] + ([a.pins] if a.pins else [])
    for path in inputs:
        if not os.path.exists(path):
            print(f"bink_names: error: {path}: no such file or directory", file=sys.stderr)
            return 2
    if a.out and any(bink_index.same_output_path(p, a.out) for p in inputs):
        ap.error("--out must not overwrite an input")
    if a.out and os.path.isdir(a.out):
        ap.error(f"--out is a folder, not a report file: {a.out}")
    if a.out:
        for folder in [p for p in [a.source, *a.mount] if os.path.isdir(p)]:
            manifest = os.path.join(folder, "index.json")
            if bink_index.same_output_path(a.out, manifest):
                ap.error(f"--out is reserved for the carve manifest: {a.out}")
            if os.path.isfile(a.out):
                for root, _dirs, names in os.walk(folder):
                    for name in names:
                        if name.lower().endswith((".json", ".csv")):
                            continue
                        if bink_index.same_output_path(os.path.join(root, name), a.out):
                            ap.error(f"--out would overwrite an input file: {a.out}")

    try:
        names = read_mount_lists(a.mount)
    except (OSError, ValueError) as error:
        print(f"bink_names: error: {error}", file=sys.stderr)
        return 1
    try:
        streams = load_streams(a.source, skip_report=a.out if a.out and os.path.isfile(a.out) else None)
    except (OSError, csv.Error, ValueError) as error:
        print(f"bink_names: error: {a.source}: {error}", file=sys.stderr)
        return 1
    try:
        pins = load_pins(a.pins)
    except (OSError, csv.Error, ValueError) as error:
        print(f"bink_names: error: {a.pins}: {error}", file=sys.stderr)
        return 1
    try:
        rows, unmatched = name_streams(names, streams, pins, a.max_skip)
    except ValueError as error:
        print(f"bink_names: error: {error}", file=sys.stderr)
        return 1
    counts = {c: sum(r["confidence"] == c for r in rows) for c in CONFIDENCES}
    summary = {"list_entries": len(names), "streams": len(streams), "named": len(rows),
               "unmatched_list_entries": len(unmatched), **counts}
    if a.out:
        try:
            write(a.out, rows, unmatched, summary)
        except OSError as error:
            print(f"bink_names: error: cannot write {a.out}: {error}", file=sys.stderr)
            return 2
    print(f"{len(names)} list entries, {len(streams)} streams: "
          + ", ".join(f"{counts[c]} {c}" for c in CONFIDENCES)
          + f"; {len(unmatched)} list entries not in this source")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
