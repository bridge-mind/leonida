"""Index RBF0 resources in an RPF9 data area or a folder of carved streams.

RBF0 is RAGE's binary form of an XML data file. This tool walks the same zlib streams as rpf9_carve.py, joins the
64 KiB continuation chunks of one logical resource, and parses the record stream far enough to report the root
schema name, the inline descriptor table, element/value/text counts, scalar counts and readable strings. It never
decrypts, never writes payloads, and does not turn RBF0 back into XML.

Format facts and the measured corpus composition are in docs/rbf0.md. Two layouts occur: a 4-byte prologue
`FB FF 01 00` followed by 64-bit scalar values, and no prologue with 32-bit scalar values. A record is one
descriptor-index byte and one type byte; a new descriptor adds a u16 name length and its UTF-8 name. Types:
0x00 opens an element (three u16 words, the third is its attribute count), 0x10 an integer, 0x20 true, 0x30 false,
0x40 a float, 0x50 three floats, 0x60 a string (u16 length), 0xFD text (u32 length), 0xFF closes the open element.
The stream must end where the root closes. No section table or name-hash table is present.

Usage:
    python -I rbf0_index.py <archive.rpf> <out.csv|out.json> [--start N] [--limit BYTES] [--strings N]
    python -I rbf0_index.py <carved-dir> <out.csv|out.json> [--strings N]

The output contains names taken from the archive, so treat it as game data: keep it out of git and do not attach
it to an issue or a pull request.
"""
import argparse
import collections
import csv
import json
import mmap
import os
import re
import struct
import sys
import zlib

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from zero_span import gap_split  # noqa: E402

MAGIC = b"RBF0"
PROLOGUE = b"\xfb\xff\x01\x00"
# zlib stream start, or a Bink 2 video ('KB2' + revision letter). Kept in step with rpf9_carve.py.
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
MAX_CHUNK = 65536
MAX_CHUNKS = 256
MAX_BYTES = 64 * 1024 * 1024
KEYWORDS = ("anim/", "pack:/", "skel_", "rbf0", "xml", "nav", "map", "ipl", "handling", "meta", "hud",
            "radio", "vehicle", "ped/", "weapon", "credits", "phone", "rage__", "fwui")
IDENT = re.compile(r"[A-Za-z0-9_.:@+-]+")
COLUMNS = ["offset", "stored", "size", "chunks", "layout", "root", "descriptors", "elements", "values",
           "texts", "strings", "ints", "floats", "depth", "status", "sample"]
TYPE_NAMES = {0x00: "open", 0x10: "int", 0x20: "true", 0x30: "false", 0x40: "float",
              0x50: "vec3", 0x60: "string", 0xFD: "text"}


class RbfError(ValueError):
    """A malformed or truncated RBF0 record stream."""

    def __init__(self, message, offset, truncated=False):
        super().__init__(f"{message} at 0x{offset:x}")
        self.detail = message
        self.offset = offset
        self.truncated = truncated


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if len(data) >= at + 4 else None


def parse_rbf0(data, string_limit=None):
    """Parse one complete RBF0 resource. Returns a dict; a truncated or bad stream keeps its partial counts.

    `string_limit` caps the returned `strings` list only; every count is still complete.
    """
    if data[:4] != MAGIC:
        raise RbfError("not an RBF0 stream", 0)
    wide = data[4:8] == PROLOGUE
    p = 8 if wide else 4
    scalar = 8 if wide else 4
    descs = []
    stack = []                 # [attribute_count, attributes_seen, content_started]
    strings = []
    ints = []
    floats = []
    types = collections.Counter()
    root = ""
    root_end = None
    elements = values = texts = 0
    max_depth = 0
    attr_mismatches = 0
    status = "ok"
    error = ""

    def take(n, what):
        nonlocal p
        if p + n > len(data):
            raise RbfError(f"{what} truncated", p, truncated=True)
        start = p
        p += n
        return data[start:p]

    try:
        while p < len(data):
            at = p
            index = take(1, "record index")[0]
            kind = take(1, "record type")[0]
            if index == 0xFF:
                if kind != 0xFF:
                    raise RbfError("close tag is not FF FF", at)
                if not stack:
                    raise RbfError("close without an open element", at)
                attrs, seen, content_started = stack.pop()
                if not content_started and seen < attrs:
                    attr_mismatches += 1
                if not stack:
                    root_end = p
                    break
                continue
            if index == 0xFD:
                if kind != 0xFF:
                    raise RbfError("text tag is not FD FF", at)
                n = struct.unpack("<I", take(4, "text length"))[0]
                raw = take(n, "text")
                if stack and not stack[-1][2]:
                    if stack[-1][1] < stack[-1][0]:
                        attr_mismatches += 1
                    stack[-1][2] = True
                text = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
                if string_limit is None or len(strings) < string_limit:
                    strings.append(text)
                texts += 1
                types["text"] += 1
                continue
            if index == len(descs):
                n = struct.unpack("<H", take(2, "descriptor name length"))[0]
                name = take(n, "descriptor name").decode("utf-8", "replace")
                descs.append((name, kind))
            elif index > len(descs):
                raise RbfError(f"descriptor index {index} of {len(descs)}", at)
            name = descs[index][0]
            if kind == 0x00:
                raw = take(6, "element header")
                word0, word1, attrs = struct.unpack("<HHH", raw)
                if word0 or word1:
                    raise RbfError(f"element header word is not zero ({word0}, {word1})", at)
                if stack and not stack[-1][2]:
                    if stack[-1][1] < stack[-1][0]:
                        attr_mismatches += 1
                    stack[-1][2] = True
                stack.append([attrs, 0, False])
                elements += 1
                max_depth = max(max_depth, len(stack))
                if not root:
                    root = name
                types["open"] += 1
                continue
            if kind == 0x10:
                raw = take(scalar, "integer")
                ints.append(struct.unpack("<Q" if wide else "<I", raw)[0])
            elif kind == 0x20 or kind == 0x30:
                pass
            elif kind == 0x40:
                raw = take(scalar, "float")
                floats.append(struct.unpack("<d" if wide else "<f", raw)[0])
            elif kind == 0x50:
                raw = take(3 * scalar, "vector")
                floats.extend(struct.unpack("<ddd" if wide else "<fff", raw))
            elif kind == 0x60:
                n = struct.unpack("<H", take(2, "string length"))[0]
                raw = take(n, "string")
                text = raw.split(b"\x00", 1)[0].decode("utf-8", "replace")
                if string_limit is None or len(strings) < string_limit:
                    strings.append(text)
            else:
                raise RbfError(f"unknown record type {kind:#04x}", at)
            values += 1
            types[TYPE_NAMES.get(kind, f"{kind:#04x}")] += 1
            if stack and stack[-1][1] < stack[-1][0]:
                stack[-1][1] += 1
    except RbfError as exc:
        status = "truncated" if exc.truncated else "bad"
        error = str(exc)
    if status == "ok":
        if stack or root_end is None:
            status = "truncated"
            error = f"truncated: {len(stack)} unclosed element(s) at 0x{p:x}"
        elif root_end != len(data):
            status = "trailing"
            error = f"{len(data) - root_end} bytes after the root closed"
    return {
        "layout": "wide-64" if wide else "narrow-32",
        "prologue": PROLOGUE.hex() if wide else "",
        "root": root,
        "descriptors": len(descs),
        "descriptor_names": [d[0] for d in descs],
        "elements": elements,
        "values": values,
        "texts": texts,
        "strings": strings,
        "ints": ints,
        "floats": floats,
        "depth": max_depth,
        "attributes": attr_mismatches,
        "types": dict(types),
        "status": status,
        "error": error,
        "end": root_end if root_end is not None else p,
        "complete": status == "ok",
    }


class ArchiveReader:
    """Inflate a carved row of an RPF9 data area on demand."""

    def __init__(self, path):
        self.fh = open(path, "rb")

    def read(self, row):
        self.fh.seek(row["offset"])
        return zlib.decompress(self.fh.read(row["stored"]))

    def close(self):
        self.fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _row(data, offset, stored):
    return {"offset": offset, "stored": stored, "size": len(data), "head": data[:8]}


def join_resource(rows, index, read, check_gap=True):
    """Parse rows[index] and, when it ends on a 64 KiB boundary, its continuation streams.

    Returns (blob, chunks, parsed). Continuation chunks are capped at 64 KiB and start with neither RBF0 nor PSIN.
    """
    row = rows[index]
    blob = read(row)
    chunks = 1
    parsed = parse_rbf0(blob, string_limit=None)
    if parsed["complete"] or row["size"] != MAX_CHUNK:
        return blob, chunks, parsed
    j = index + 1
    while j < len(rows) and chunks < MAX_CHUNKS and len(blob) < MAX_BYTES:
        nxt = rows[j]
        if rows[j - 1]["size"] != MAX_CHUNK or nxt["size"] > MAX_CHUNK or nxt["head"][:4] in (MAGIC, b"PSIN"):
            break
        if check_gap and nxt["offset"] > rows[j - 1]["offset"] + rows[j - 1]["stored"] + 16:
            break
        blob += read(nxt)
        chunks += 1
        parsed = parse_rbf0(blob, string_limit=None)
        if parsed["status"] != "truncated":
            break
        j += 1
    return blob, chunks, parsed


def resources_from_rows(rows, read, check_gap=True, string_limit=16):
    """Parse every RBF0 start row, join its continuations, and build the reported records."""
    out = []
    for i, row in enumerate(rows):
        if row["head"][:4] != MAGIC:
            continue
        blob, chunks, parsed = join_resource(rows, i, read, check_gap=check_gap)
        sample = parsed["strings"][:string_limit] if string_limit is not None else list(parsed["strings"])
        out.append({
            "offset": row["offset"],
            "stored": sum(rows[j]["stored"] for j in range(i, i + chunks)),
            "size": len(blob),
            "chunks": chunks,
            "layout": parsed["layout"],
            "prologue": parsed["prologue"],
            "root": parsed["root"],
            "descriptors": parsed["descriptors"],
            "elements": parsed["elements"],
            "values": parsed["values"],
            "texts": parsed["texts"],
            "strings": len(parsed["strings"]),
            "ints": len(parsed["ints"]),
            "floats": len(parsed["floats"]),
            "depth": parsed["depth"],
            "attributes": parsed["attributes"],
            "types": parsed["types"],
            "status": parsed["status"],
            "error": parsed["error"],
            "descriptor_names": parsed["descriptor_names"],
            "sample": sample,
            "_all_strings": parsed["strings"],
        })
    return out


def _bucket(size):
    if size < 1024:
        return "<1K"
    if size < 4096:
        return "1-4K"
    if size < 16384:
        return "4-16K"
    if size < 65536:
        return "16-64K"
    if size < 1048576:
        return "64K-1M"
    return ">=1M"


def summarise(rows, records, span, zero, unread, mode):
    rbf0_stored = sum(r["stored"] for r in records)
    rbf0_inflated = sum(r["size"] for r in records)
    all_strings = [s for r in records for s in r["_all_strings"]]
    keywords = collections.Counter()
    for word in KEYWORDS:
        keywords[word] = sum(1 for s in all_strings if word in s.lower())
    distinct_strings = {s for s in all_strings}
    return {
        "mode": mode,
        "streams": len(rows),
        "compressed_bytes": sum(r["stored"] for r in rows),
        "inflated_bytes": sum(r["size"] for r in rows),
        "span": span,
        "zero_padding": zero,
        "unread_bytes": unread,
        "readable_fraction": round(sum(r["stored"] for r in rows) / span, 6) if span else 0.0,
        "rbf0": {
            "resources": len(records),
            "single_stream": sum(1 for r in records if r["chunks"] == 1),
            "multi_chunk": sum(1 for r in records if r["chunks"] > 1),
            "max_chunks": max((r["chunks"] for r in records), default=0),
            "stored_bytes": rbf0_stored,
            "inflated_bytes": rbf0_inflated,
            "layouts": dict(collections.Counter(r["layout"] for r in records)),
            "status": dict(collections.Counter(r["status"] for r in records)),
            "by_root": dict(collections.Counter(r["root"] for r in records)),
            "size_buckets": dict(collections.Counter(_bucket(r["size"]) for r in records)),
            "descriptors": sum(r["descriptors"] for r in records),
            "elements": sum(r["elements"] for r in records),
            "values": sum(r["values"] for r in records),
            "texts": sum(r["texts"] for r in records),
            "strings": len(all_strings),
            "distinct_strings": len(distinct_strings),
            "path_occurrences": sum(1 for s in all_strings if "/" in s),
            "path_strings": sum(1 for s in distinct_strings if "/" in s),
            "identifier_strings": sum(1 for s in all_strings if IDENT.fullmatch(s) is not None),
            "ints": sum(r["ints"] for r in records),
            "floats": sum(r["floats"] for r in records),
            "max_depth": max((r["depth"] for r in records), default=0),
            "attribute_mismatches": sum(r["attributes"] for r in records),
            "keyword_occurrences": dict(keywords),
        },
    }


def scan_rpf(path, start=None, limit=None, progress=None, string_limit=16):
    """Walk an RPF9 data area and return (rows, records, summary). No table of contents is read or decrypted."""
    rows = []
    if os.path.getsize(path) == 0:
        raise ValueError("empty file")
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        size = len(mm)
        first = struct.unpack_from("<I", mm, 4)[0] if size >= 8 else 0
        pos = start if start is not None else first
        if not 0 <= pos <= size:
            raise ValueError("start offset is outside the archive")
        end = min(size, pos + limit) if limit else size
        span_start = pos
        last, zero, unread = pos, 0, 0
        while pos < end:
            match = ZHDR.search(mm, pos, end)
            if not match:
                break
            s = match.start()
            if mm[s:s + 3] == b"KB2":
                blen = (_u32(mm, s + 4) or 0) + 8
                w, h = _u32(mm, s + 0x14), _u32(mm, s + 0x18)
                if not (64 <= blen <= end - s and w and h and 16 <= w <= 8192 and 16 <= h <= 8192):
                    pos = s + 1
                    continue
                gap_zero, gap_unread = gap_split(mm, last, s)
                zero += gap_zero
                unread += gap_unread
                rows.append({"offset": s, "stored": blen, "size": blen, "head": mm[s:s + 8]})
                last = pos = s + blen
                continue
            d = zlib.decompressobj()
            chunks, p = [], s
            try:
                while not d.eof and p < end:   # inside the --limit window, as in story_index (#712)
                    block = mm[p:min(p + (1 << 20), end)]
                    chunks.append(d.decompress(block))
                    p += len(block)
                    if d.unused_data or d.eof:
                        break
                out = b"".join(chunks) + d.flush()
            except zlib.error:
                pos = s + 1
                continue
            if not d.eof or len(out) < 64:
                pos = s + 1
                continue
            used = p - s - len(d.unused_data)
            gap_zero, gap_unread = gap_split(mm, last, s)
            zero += gap_zero
            unread += gap_unread
            rows.append(_row(out, s, used))
            last = pos = s + used
            if progress and len(rows) % 2000 == 0:
                progress(len(rows), pos, end)
        gap_zero, gap_unread = gap_split(mm, last, end)
        zero += gap_zero
        unread += gap_unread
        mspan = end - span_start
    with ArchiveReader(path) as reader:
        records = resources_from_rows(rows, reader.read, check_gap=True, string_limit=string_limit)
    summary = summarise(rows, records, mspan, zero, unread, "rpf9")
    summary["archive"] = path
    return rows, records, summary


def carved_streams(path):
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        # rpf9_carve.py's index.json and any .csv report sit next to the streams; they are not payloads.
        if not name.lower().endswith((".json", ".csv")) and os.path.isfile(full):
            yield name, full


def scan_dir(path, string_limit=16):
    """Index already-carved files. Sorting by a leading hexadecimal file name keeps archive order."""
    entries = []
    unnamed = []
    for name, full in carved_streams(path):
        with open(full, "rb") as f:
            data = f.read()
        match = re.match(r"([0-9a-fA-F]{6,})", name)
        if match:
            entries.append((int(match.group(1), 16), data))
        else:
            unnamed.append(data)
    next_offset = max((e[0] for e in entries), default=-1) + 1
    for data in unnamed:
        entries.append((next_offset, data))
        next_offset += 1
    entries.sort(key=lambda e: e[0])
    rows = [_row(data, offset, len(data)) for offset, data in entries]
    by_row = {id(row): data for row, (_, data) in zip(rows, entries)}
    records = resources_from_rows(rows, lambda row: by_row[id(row)], check_gap=False,
                                  string_limit=string_limit)
    summary = summarise(rows, records, sum(len(d) for _, d in entries), 0, 0, "directory")
    summary["directory"] = path
    return rows, records, summary


def write_output(path, records, summary, all_strings=False):
    """Write CSV or JSON, chosen by the extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        docs = []
        for r in records:
            doc = {k: v for k, v in r.items() if k not in ("_all_strings", "sample", "strings")}
            doc["string_count"] = r["strings"]
            doc["sample_strings"] = r["_all_strings"] if all_strings else r["sample"]
            doc["sample_truncated"] = not all_strings and r["strings"] > len(r["sample"])
            docs.append(doc)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "files": docs}, f, indent=1)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        for r in records:
            sample = "; ".join(r["sample"])
            if r["strings"] > len(r["sample"]):
                sample += f" (+{r['strings'] - len(r['sample'])} more)"
            writer.writerow([
                f"0x{r['offset']:x}", r["stored"], r["size"], r["chunks"], r["layout"], r["root"],
                r["descriptors"], r["elements"], r["values"], r["texts"], r["strings"], r["ints"],
                r["floats"], r["depth"], r["status"], sample])


def offset(text):
    """--start: 0 or more, decimal or 0x hex."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be 0 or more") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def byte_count(text):
    """--limit: at least 1; 0 used to mean no limit and a negative value silently scanned less."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError("must be at least 1") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def same_output_path(first, second):
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive or a directory of carved streams")
    ap.add_argument("out", help="output path ending in .csv or .json")
    ap.add_argument("--start", type=offset, help="first byte to scan (default: the RPF data start)")
    ap.add_argument("--limit", type=byte_count, help="number of bytes to scan")
    ap.add_argument("--strings", type=int, default=16, help="readable strings kept per resource (default 16)")
    ap.add_argument("--all-strings", action="store_true", help="write every readable string to JSON (game data)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if same_output_path(a.source, a.out):
        ap.error("output must not overwrite the source")
    if a.strings < 0:
        ap.exit(2, f"{ap.prog}: error: argument --strings: must be 0 or more\n")
    if os.path.isdir(a.source) and (a.start is not None or a.limit is not None):
        ap.error("--start and --limit apply to a single archive, not a directory")
    if os.path.isdir(a.out):
        ap.error(f"{a.out} is a folder, not a .csv or .json file")
    folder = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(folder):
        ap.error(f"{folder} is not a folder, so {a.out} cannot be written")
    limit = None if a.all_strings else a.strings
    try:
        if os.path.isdir(a.source):
            manifest = os.path.join(a.source, "index.json")
            if same_output_path(manifest, a.out) or (os.path.isfile(a.out) and any(
                    same_output_path(path, a.out) for _, path in carved_streams(a.source))):
                ap.error(f"output would overwrite a carve input: {a.out}")
            rows, records, summary = scan_dir(a.source, string_limit=limit)
        else:
            def progress(n, pos, end):
                if not a.quiet:
                    print(f"[{pos / max(1, end):6.1%}] {n} streams", flush=True)
            rows, records, summary = scan_rpf(a.source, a.start, a.limit, progress, string_limit=limit)
    except ValueError as e:
        sys.exit(f"{a.source}: {e}")
    except OSError as e:
        sys.exit(f"{a.source}: {e.strerror or e}")
    write_output(a.out, records, summary, all_strings=a.all_strings)
    if not a.quiet:
        s = summary["rbf0"]
        print(f"{s['resources']} RBF0 resources, {s['multi_chunk']} multi-chunk, "
              f"{s['inflated_bytes']} inflated bytes -> {a.out}")
        print(f"roots: {len(s['by_root'])} classes; status: {s['status']}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
