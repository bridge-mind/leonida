"""Index RPF9 `0DRP` page directories without the encrypted table of contents.

The RPF9 data area is a sequence of zlib streams. A directory stream starts with eight zero bytes, a
`0x50000000`-based self-pointer, `0DRP` at +0x10, version 122 at +0x14, table pointers, and four u16 counts
at +0x48. The fixed header is 0x60 bytes; the first resource payload starts at +0x60.

The table at +0x38 is the useful one: `u16@0x4c` records of 16 bytes, each `{u32 hash, u32 size, u64 pointer}`.
The pointer is a virtual offset into the logical resource after subtracting 0x50000000, not an archive offset
and not necessarily an offset inside this zlib stream. If the table or a payload crosses the inflated stream
end, the next zlib stream in archive order continues the same resource. This tool concatenates only as many
continuation streams as the declared table and entry ranges require, then reports exact virtual boundaries.

The header has no key: what earlier reports called a 64-bit key at +0x60 is the first payload byte on
29,297 of 29,298 measured pages. This tool reports `payload_at_0x60` instead.

The +0x28 table (32-byte records) is reported raw as `q28_records`: the hash in its first u32 matches a +0x38
hash on most pages, its +0x10 pointer resolves in the resource or is null (`target` null), and its +0x1c word
behaves as flags. The +0x08 and +0x18 words are not decoded and are exported as `field_08` and `count_or_size`.
The +0x30 and +0x40 tables are reported by offset only; their internal layout is not decoded here.

Each page also carries a `resource`: the directory stream plus the continuations that reach the q38 table and its
last record, their byte total, and the +0x20 count next to the 64 KiB units the group occupies. The count does not
predict the boundary (its best byte reading matches 56.7% of `story000` pages), so the table extent is used and the
mismatch stays visible as `declared_matches_units`.

Usage:
    python -I page_index.py <archive.rpf> <out.json|out.csv> [--start N] [--limit BYTES] [--quiet]
    python -I page_index.py <carved-dir> <out.json|out.csv> [--quiet]

--start and --limit apply to a single archive only; with a directory they are an error (exit 2).
"""
import argparse
from collections import Counter
import csv
import json
import mmap
import os
import re
import struct
import sys
import zlib

# -I omits the script directory from sys.path.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

# zlib stream start. Kept in step with rpf9_carve.py.
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
BASE = 0x50000000
HEADER_SIZE = 0x60
PAGE_UNIT = 0x10000   # 64 KiB: the unit a resource's declared +0x20 count is compared against
MAX_CONTINUATIONS = 16
Q28_UNREAD = "q28 table ends past the streams read for q38"
COLUMNS = [
    "page_offset", "stored", "inflated", "version", "page_count",
    "q28_count", "q30_count", "q38_count", "span", "complete",
    "record_index", "hash", "size", "target", "target_stream",
]


def _u16(data, at):
    return struct.unpack_from("<H", data, at)[0] if len(data) >= at + 2 else None


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if len(data) >= at + 4 else None


def _u64(data, at):
    return struct.unpack_from("<Q", data, at)[0] if len(data) >= at + 8 else None


def offset_of(value):
    """Decode a 0x50000000-based pointer. All measured values are zero-extended 32-bit."""
    if value is None:
        return None
    if value >> 32 == 0 and BASE <= value < BASE + 0x10000000:
        return value - BASE
    if value >> 32 == BASE:  # tolerate a full-width form of the same base
        return value & 0xFFFFFFFF
    return None


def is_page(data):
    return len(data) >= HEADER_SIZE and data[0x10:0x14] == b"0DRP"


def scan_rpf(path, start=None, limit=None):
    """Yield (archive_offset, stored_bytes, inflated_bytes) for each zlib stream."""
    with open(path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size < 8:   # no header word; mmap also refuses an empty file
            return
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        size = len(m)
        first = struct.unpack_from("<I", m, 4)[0]
        pos = start if start is not None else first
        if not 0 <= pos <= size:
            m.close()
            raise ValueError("start offset is outside the archive")
        end = min(size, pos + limit) if limit else size
        try:
            while pos < end:
                match = ZHDR.search(m, pos, end)
                if not match:
                    break
                s = match.start()
                d = zlib.decompressobj()
                chunks, p = [], s
                try:
                    while not d.eof and p < end:
                        block = m[p:min(p + (1 << 20), end)]
                        if not block:
                            break
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
                yield s, used, out
                pos = s + used
        finally:
            m.close()


def scan_dir(path):
    """Yield (offset, None, bytes) from carved `<offset>.bin` files, sorted by offset."""
    entries = []
    for name in os.listdir(path):
        full = os.path.join(path, name)
        if not os.path.isfile(full) or name.lower().endswith((".json", ".csv")):
            continue
        match = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(match.group(1), 16) if match else None
        entries.append((offset, name, full))
    fallback = 0
    for offset, name, full in sorted(entries, key=lambda e: (e[0] is None, e[0] or 0, e[1])):
        if offset is None:
            offset = fallback
        fallback = max(fallback, offset + 1)
        with open(full, "rb") as f:
            data = f.read()
        yield offset, None, data


def _q38(data, q38, count):
    """Parse a q38 table from `data`. Returns (records, complete, error)."""
    if q38 is None:
        return [], True, "q38 is not a 0x50000000 pointer"
    if count == 0:
        return [], True, ""
    if q38 + 16 * count > len(data):
        return [], False, ""
    records = []
    for i in range(count):
        h, size, pointer = struct.unpack_from("<IIQ", data, q38 + 16 * i)
        target = offset_of(pointer)
        if target is None:
            return [], True, "q38 entry %d has a non-0x50000000 pointer" % i
        records.append({"hash": h, "size": size, "target": target})
    need = max([q38 + 16 * count] + [r["target"] + r["size"] for r in records])
    if need > len(data):
        return [], False, ""
    return records, True, ""


def _q28(data, q28, count):
    """Parse the +0x28 table raw. Unresolved words keep their measured names.

    A record whose pointer is null is kept with `target` None: its other words are set, and the records after
    it are ordinary ones (3,672 such records on 3,511 of 22,823 `anim.rpf` pages). A table that ends past the
    streams read for q38 is not read, and says so instead of looking like a page without one.
    """
    if not count:
        return [], ""
    if q28 is None:   # none on data.rpf, navmeshes.rpf or anim.rpf; named so declared records cannot go missing
        return [], "q28 is not a 0x50000000 pointer"
    if q28 + 32 * count > len(data):
        return [], Q28_UNREAD
    out = []
    for i in range(count):
        hash_a, hash_b, field_08, pointer, count_or_size, flags = struct.unpack_from(
            "<IIQQII", data, q28 + 32 * i)
        target = offset_of(pointer)
        if target is None and pointer:
            return out, "q28 entry %d has a non-0x50000000 pointer" % i
        out.append({
            "hash_a": hash_a, "hash_b": hash_b, "field_08": field_08,
            "target": target, "count_or_size": count_or_size, "flags": flags,
        })
    return out, ""


def parse_page(base, continuations):
    """Parse one 0DRP page plus its continuation stream payloads.

    Returns (page_dict, complete). A page is complete when the q38 table and every q38 entry fit in the
    concatenated resource. An unparsable pointer is reported and does not keep the page waiting.
    """
    if not is_page(base):
        return None, True
    data = base + b"".join(s[2] for s in continuations)
    page = {
        "offset": None,
        "stored": None,
        "inflated": len(base),
        "version": _u32(base, 0x14),
        "page_count": _u32(base, 0x20),
        "counts": [_u16(base, 0x48), _u16(base, 0x4A), _u16(base, 0x4C), _u16(base, 0x4E)],
        "tables": {
            "q28": offset_of(_u64(base, 0x28)),
            "q30": offset_of(_u64(base, 0x30)),
            "q38": offset_of(_u64(base, 0x38)),
            "q40": offset_of(_u64(base, 0x40)),
        },
        "q38_records": [],
        "q28_records": [],
        "continuations": [{"offset": s[0], "stored": s[1], "inflated": len(s[2])} for s in continuations],
        "span": 0,
        "payload_at_0x60": False,
        "complete": True,
        "error": "",
    }
    q38 = page["tables"]["q38"]
    count = page["counts"][2] or 0
    records, complete, error = _q38(data, q38, count)
    page["error"] = error
    if not complete:
        page["complete"] = False
        return page, False
    page["span"] = max([r["target"] + r["size"] for r in records], default=0)
    page["payload_at_0x60"] = any(r["target"] == HEADER_SIZE for r in records)
    # Attach a stream index to each target. Index 0 is the directory stream itself.
    lengths = [len(base)] + [len(s[2]) for s in continuations]
    starts = []
    total = 0
    for n in lengths:
        starts.append(total)
        total += n
    enriched = []
    for r in records:
        stream = 0
        for i in range(1, len(starts)):
            if r["target"] >= starts[i]:
                stream = i
        item = dict(r)
        item["target_stream"] = stream
        item["spans_streams"] = stream + 1 < len(starts) and r["target"] + r["size"] > starts[stream + 1]
        enriched.append(item)
    page["q38_records"] = enriched
    q28 = page["tables"]["q28"]
    page["q28_records"], page["q28_error"] = _q28(data, q28, page["counts"][0] or 0)
    return page, True


def resource_of(page):
    """The streams that make up one logical resource, and how its declared count compares.

    The boundary is the q38 table and its last record: the resource is the directory stream plus the
    continuation streams needed to reach them. The +0x20 count is reported next to the 64 KiB units
    the group actually occupies, so a caller can see a disagreement instead of trusting the field.
    """
    streams = [{"offset": page["offset"], "inflated": page["inflated"]}]
    streams += [{"offset": c["offset"], "inflated": c["inflated"]} for c in page["continuations"]]
    total = page["inflated"] + sum(c["inflated"] for c in page["continuations"])
    units = max(1, -(-total // PAGE_UNIT))
    declared = page["page_count"]
    return {
        "streams": streams,
        "bytes": total,
        "page_units": units,
        "declared_pages": declared,
        "declared_matches_units": None if declared is None else declared == units,
    }


def index_streams(streams, progress=None):
    """Consume a stream sequence and return page records. At most one page waits for a continuation."""
    pages = []
    waiting = None
    streams_seen = 0

    def add(page):
        page["resource"] = resource_of(page)
        pages.append(page)

    def start(offset, stored, data):
        page, complete = parse_page(data, [])
        page["offset"] = offset
        page["stored"] = stored
        if complete:
            add(page)
            return None
        return (page, data, [])

    for offset, stored, data in streams:
        streams_seen += 1
        if progress and streams_seen % 2000 == 0:
            progress(streams_seen, len(pages))
        if waiting is not None:
            page, base, cont = waiting
            cont.append((offset, stored, data))
            page, complete = parse_page(base, cont)
            page["offset"] = waiting[0]["offset"]
            page["stored"] = waiting[0]["stored"]
            if complete:
                add(page)
                waiting = None
                continue
            if is_page(data):
                # The previous resource ended before its declared table/entries did. Report it, then
                # treat this stream as the new directory page instead of swallowing it as a continuation.
                # Report the page re-parsed with this stream: waiting[0] is the parse before the last
                # continuation was appended and would lose it from the reported continuations/records.
                page["complete"] = False
                page["error"] = page["error"] or "resource ended before q38 was complete"
                add(page)
                waiting = start(offset, stored, data)
                continue
            if len(cont) >= MAX_CONTINUATIONS:
                page["complete"] = False
                page["error"] = page["error"] or "too many continuation streams"
                add(page)
                waiting = None
            else:
                waiting = (page, base, cont)
            continue
        if is_page(data):
            waiting = start(offset, stored, data)
    if waiting is not None:
        page = waiting[0]
        page["complete"] = False
        page["error"] = page["error"] or "archive ended before q38 was complete"
        add(page)
    return pages, streams_seen


def summarise(pages, mode, source):
    complete = sum(1 for p in pages if p["complete"])
    return {
        "mode": mode,
        "source": source,
        "pages": len(pages),
        "complete_pages": complete,
        "incomplete_pages": len(pages) - complete,
        "q38_records": sum(len(p["q38_records"]) for p in pages),
        "q28_records": sum(len(p["q28_records"]) for p in pages),
        "q28_declared": sum(p["counts"][0] or 0 for p in pages),
        "q28_null_targets": sum(1 for p in pages for r in p["q28_records"] if r["target"] is None),
        "q28_tables_unread": sum(1 for p in pages if p.get("q28_error") == Q28_UNREAD),
        "pages_with_continuations": sum(1 for p in pages if p["continuations"]),
        "span_bytes": sum(p["span"] for p in pages),
        "payload_at_0x60": sum(1 for p in pages if p["payload_at_0x60"]),
        "resource_streams": {str(n): sum(1 for p in pages if len(p["resource"]["streams"]) == n)
                             for n in sorted({len(p["resource"]["streams"]) for p in pages})},
        "page_units": sum(p["resource"]["page_units"] for p in pages),
        "declared_matches_units": sum(1 for p in pages if p["resource"]["declared_matches_units"]),
        "declared_differs_from_units": sum(1 for p in pages
                                           if p["resource"]["declared_matches_units"] is False),
        "versions": {str(v): sum(1 for p in pages if p["version"] == v)
                     for v in sorted({p["version"] for p in pages})},
        "page_counts": {str(v): n for v, n in sorted(Counter(
            p["page_count"] for p in pages if p["page_count"] is not None).items())},
    }


def write_json(path, pages, summary):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "pages": pages}, f, indent=1)


def write_csv(path, pages):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for page in pages:
            records = page["q38_records"]
            base = {
                "page_offset": "0x%x" % page["offset"],
                "stored": "" if page["stored"] is None else page["stored"],
                "inflated": page["inflated"],
                "version": page["version"],
                "page_count": page["page_count"],
                "q28_count": page["counts"][0],
                "q30_count": page["counts"][1],
                "q38_count": page["counts"][2],
                "span": page["span"],
                "complete": int(page["complete"]),
            }
            if not records:
                w.writerow(dict(base, record_index="", hash="", size="", target="", target_stream=""))
                continue
            for i, r in enumerate(records):
                w.writerow(dict(base, record_index=i, hash="0x%08x" % r["hash"], size=r["size"],
                                target="0x%x" % r["target"], target_stream=r["target_stream"]))


def byte_count(text):
    """--limit: 0 or less used to scan nothing (negative) or everything (0 is falsy) without a word."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def offset(text):
    """--start: a negative offset would slice the archive from its end."""
    try:
        value = int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return value


def main(argv=None):
    ap = argparse.ArgumentParser(description="Index RPF9 0DRP page directories (read-only, no TOC decryption)")
    ap.add_argument("source", help="archive.rpf or a directory of carved <offset>.bin files")
    ap.add_argument("out", help="output .json or .csv")
    ap.add_argument("--start", type=offset, help="first archive offset (default: header u32@4)")
    ap.add_argument("--limit", type=byte_count, help="stop after this many bytes")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    if os.path.isdir(a.source) and (a.start is not None or a.limit is not None):
        ap.error("--start and --limit apply to a single archive, not a directory")
    if os.path.isdir(a.out):
        ap.error(f"{a.out} is a folder, not a .json or .csv file")
    ext = os.path.splitext(a.out)[1].lower()
    if ext not in (".json", ".csv"):   # checked before the scan, which can take minutes on a large archive
        ap.error("output must end in .json or .csv")
    if same_report_path(a.source, a.out):
        ap.error("output must not overwrite the source")
    if os.path.isdir(a.source):
        manifest = os.path.join(a.source, "index.json")
        if same_report_path(manifest, a.out):
            ap.error(f"output would overwrite the carve manifest: {a.out}")
        if os.path.isfile(a.out):
            for name in os.listdir(a.source):
                inp = os.path.join(a.source, name)
                if os.path.isfile(inp) and not name.lower().endswith((".json", ".csv")):
                    if same_report_path(inp, a.out):
                        ap.error(f"output would overwrite an input file: {a.out}")
    folder = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(folder):
        ap.error(f"{folder} is not a folder, so {a.out} cannot be written")

    def progress(n, pages):
        print("streams=%d pages=%d" % (n, pages), file=sys.stderr, flush=True)

    if os.path.isdir(a.source):
        streams = scan_dir(a.source)
        mode = "directory"
    elif os.path.isfile(a.source):
        streams = scan_rpf(a.source, a.start, a.limit)
        mode = "rpf9"
    else:
        print("not a file or directory: %s" % a.source, file=sys.stderr)
        return 2
    try:
        pages, streams_seen = index_streams(streams, None if a.quiet else progress)
    except (OSError, ValueError) as e:
        print("%s: %s" % (a.source, e), file=sys.stderr)
        return 2
    summary = summarise(pages, mode, a.source)
    summary["streams"] = streams_seen
    try:
        if ext == ".json":
            write_json(a.out, pages, summary)
        else:
            write_csv(a.out, pages)
    except OSError as e:
        print(f"{a.out}: {e.strerror or e}", file=sys.stderr)
        return 2
    if not a.quiet:
        print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
