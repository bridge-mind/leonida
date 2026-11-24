"""Index the animation containers in an RPF9 data area or a folder of carved streams.

RPF9's table of contents is encrypted, so this walks the data area the same way rpf9_carve.py does: find every
zlib stream that inflates cleanly to its end, then classify the inflated bytes. It never decrypts, never writes
game data, and does not decode animation.

Signatures are only the byte patterns measured in docs/animation.md. Anything else is reported as `raw:<hex>`
instead of being guessed. A `0DRP` page is reported with its version and page count; the tagged auxiliary table that starts a
`0DRP` second system page (historical label `clip-stream`) with its entry count, tags and payload end, and only
when its directory and payloads fit in the stream; `MoVE` with its version and size hint; `RBF0` with its first schema name; name tables with their first path and number of paths.

Usage:
    python -I anim_index.py <archive.rpf> <out.csv|out.json> [--start N] [--limit BYTES] [--quiet]
    python -I anim_index.py <carved-dir> <out.csv|out.json> [--quiet]
    python -I anim_index.py <archive.rpf> <out.json> --rigs <rigs.json>

`--rigs` additionally joins the RBF0 stream with its 64 KiB continuation blocks and writes the rig-set/rig entries
and plain `SKEL_` bone names to its own JSON. It stops at the next known container signature or a gap.
Usage: `python -I tools/anim_index.py --help` lists the options.
"""
import argparse
import bisect
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
from report_paths import same_report_path  # noqa: E402
from zero_span import gap_split  # noqa: E402

# zlib stream start, or a Bink 2 video ('KB2' + revision letter). Kept in step with rpf9_carve.py.
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")
PACK_NAMES = re.compile(rb"pack:/[^\x00]{1,200}")
SKEL_NAMES = re.compile(rb"(?:SKEL_|PH_)[A-Za-z0-9_]{1,100}")
COLUMNS = ["offset", "stored", "size", "signature", "version", "name", "count", "detail"]
RBF0_SOURCES = ("canonical@source", "generated@source", "pipelined@source", "canonical@legacy")
STOP_SIGNATURES = {"rbf0", "page-directory", "name-table", "move", "psin", "typehash", "clip-stream",
                   "skeleton-names", "bink2"}


def _u16(data, at):
    return struct.unpack_from("<H", data, at)[0] if len(data) >= at + 2 else None


def _u32(data, at):
    return struct.unpack_from("<I", data, at)[0] if len(data) >= at + 4 else None


def _u64(data, at):
    return struct.unpack_from("<Q", data, at)[0] if len(data) >= at + 8 else None


def _first_text(data, prefix):
    start = data.find(prefix)
    if start < 0:
        return ""
    end = data.find(b"\x00", start)
    if end < 0:
        end = min(len(data), start + 200)
    return data[start:end].decode("ascii", "replace")


def _printable(data):
    if not data:
        return False
    sample = data[:512]
    good = sum(1 for b in sample if 32 <= b < 127 or b in (9, 10, 13))
    return len(sample) >= 64 and good / len(sample) > 0.9


AUX_FIRST_TAG = 0x101
AUX_COUNTS = (2, 3)


def aux_table(data):
    """The tagged auxiliary table at the start of a `0DRP` resource's second system page (docs/animation.md):
    a u32 count (2 or 3), then `count` pairs {u32 tag, u32 payloadBytes}, then the payloads in order; the first
    tag is 0x101. Bytes after the payloads are legitimate (schema tables follow them). Returns
    ([(tag, size), ...], payload_end) when the directory and every payload fit in `data`, else None, so a
    matching prefix on truncated or unrelated bytes keeps its generic classification. Needs the whole
    inflated stream, not a head slice."""
    count = _u32(data, 0)
    if count not in AUX_COUNTS:
        return None
    directory_end = 4 + 8 * count
    if len(data) < directory_end:
        return None
    entries = [struct.unpack_from("<II", data, 4 + 8 * i) for i in range(count)]
    if entries[0][0] != AUX_FIRST_TAG:
        return None
    payload_end = directory_end + sum(size for _, size in entries)
    if payload_end > len(data):
        return None
    return entries, payload_end


def classify(data, offset, stored):
    """Return one record for an inflated stream. `offset` is its archive offset, `stored` its compressed size."""
    record = {"offset": offset, "stored": stored, "size": len(data), "signature": "raw",
              "version": None, "name": "", "count": None, "detail": ""}
    if data[:4] == b"MoVE":
        record.update(signature="move", version=_u32(data, 4))
        hint = _u32(data, 0x14)
        if hint is not None:
            record["detail"] = f"size_hint={hint}"
    elif data[:4] == b"PSIN":
        record.update(signature="psin", detail="metadata container")
    elif data[:4] == b"RBF0":
        schema = re.search(rb"[\x20-\x7e]{4,120}", data[8:160])
        record.update(signature="rbf0", name=schema.group().decode("ascii", "replace") if schema else "",
                      detail="rig metadata")
    elif data[:6] == b"pack:/":
        names = PACK_NAMES.findall(data)
        record.update(signature="name-table", name=_first_text(data, b"pack:/"), count=len(names),
                      detail="pack:/ paths")
    elif data[:5] == b"SKEL_":
        names = SKEL_NAMES.findall(data)
        record.update(signature="skeleton-names", name=_first_text(data, b"SKEL_"), count=len(names),
                      detail="bone names")
    elif len(data) >= 0x30 and data[16:20] == b"0DRP":
        record.update(signature="page-directory", version=_u32(data, 0x14), count=_u64(data, 0x20),
                      detail="0DRP at +16")
    elif (len(data) >= 16 and _u32(data, 4) == 8 and _u32(data, 12) == 0
          and 0 < (_u32(data, 8) or 0) <= 4096):
        record.update(signature="typehash", version=8, count=_u32(data, 8),
                      detail=f"type=0x{_u32(data, 0):08x}")
    elif (table := aux_table(data)) is not None:
        # Historical label kept so existing consumers group these pages as before; the count is the measured
        # entry count, not a version.
        entries, payload_end = table
        record.update(signature="clip-stream", count=len(entries),
                      detail=f"aux-table tags={','.join(f'0x{tag:x}' for tag, _ in entries)} "
                             f"payload_end={payload_end}")
    elif data[:3] == b"KB2" and len(data) >= 0x20:
        record.update(signature="bink2", version=chr(data[3]), count=_u32(data, 8))
        record["detail"] = f"{_u32(data, 0x14)}x{_u32(data, 0x18)}"
    elif data[:8] == b"\x00" * 8:
        record["signature"] = "zero-prefix"
    elif _printable(data):
        record["signature"] = "text"
    else:
        record["signature"] = "raw:" + data[:4].hex()
    return record


def _u16_string(data, at):
    """Short length-prefixed ASCII string, as used by RBF0 rig-set keys and schema names."""
    n = _u16(data, at)
    if n is None or not 1 <= n <= 300:
        return None
    raw = data[at + 2:at + 2 + n]
    if len(raw) != n or not raw or not all(32 <= c < 127 for c in raw):
        return None
    return raw.decode("ascii")


def _fd_string(data, at):
    """`fd ff` + u32 length (including NUL) + ASCII value, as used by RBF0 rig fields."""
    if data[at:at + 2] != b"\xfd\xff":
        return None
    n = _u32(data, at + 2)
    if n is None or not 1 <= n <= 300:
        return None
    raw = data[at + 6:at + 6 + n]
    if raw.endswith(b"\x00"):
        raw = raw[:-1]
    if not raw or not all(32 <= c < 127 for c in raw):
        return None
    return raw.decode("ascii")


def joaat32(text):
    """Jenkins one-at-a-time, lowercased, as used for animation path and bone name hashes."""
    h = 0
    for c in text.lower().encode("latin-1", "replace"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


def parse_skel_names(data, offset=0):
    """Read the full bone names out of a plain `SKEL_` stream. Front-coded tables are not decoded."""
    names = [m.group().decode("ascii", "replace") for m in SKEL_NAMES.finditer(data)]
    distinct = sorted(set(names))
    hashes = sum(1 for n in distinct if struct.pack("<I", joaat32(n)) in data)
    return {"offset": offset, "occurrences": len(names), "distinct": len(distinct),
            "hashes_present": hashes, "names": distinct}


def parse_rbf0(data, offset=0):
    """Parse the RBF0 rig-set manager without decoding its schema.

    Rig-set keys are `03 60` + u16 length + ASCII. Entries are `fd ff` + u32 length + a source tag, then the
    `ff ff 06 00` field marker, a rig key (`fd ff` + u32 length), `ff ff 07 00`, and a group string. Each source
    is followed by exactly one rig key; rig-set blocks end with six `ff` bytes or (when empty) four `ff` bytes.
    """
    if data[:4] != b"RBF0":
        raise ValueError("not an RBF0 blob")
    rigsets = sorted({(i, s) for i in range(len(data) - 4)
                      if (s := _u16_string(data, i)) and s.startswith("anim/rigset/")})
    sources, rigs = [], []
    for i in range(len(data) - 6):
        s = _fd_string(data, i)
        if s in RBF0_SOURCES:
            sources.append((i, s))
        elif s and s.startswith("anim/rig/"):
            rigs.append((i, s))
    if not rigsets:
        raise ValueError("no anim/rigset/ keys found")
    rig_offsets = [i for i, _ in rigs]
    rigset_offsets = [i for i, _ in rigsets]
    rows = [{"key": key, "offset": pos, "entries": []} for pos, key in rigsets]
    for n, (source_at, source) in enumerate(sources):
        j = bisect.bisect_left(rig_offsets, source_at)
        if j >= len(rigs):
            continue
        rig_at, rig = rigs[j]
        k = bisect.bisect_right(rigset_offsets, source_at) - 1
        if k < 0:
            continue
        end = sources[n + 1][0] if n + 1 < len(sources) else len(data)
        group = ""
        p = rig_at + 6 + len(rig) + 1
        while p < end:
            candidate = _fd_string(data, p)
            if candidate is not None:
                if not candidate.startswith("anim/"):
                    group = candidate
                break
            p += 1
        rows[k]["entries"].append({"source": source, "rig": rig, "group": group,
                                   "source_offset": source_at, "rig_offset": rig_at})
    for row in rows:
        row["distinct_rigs"] = len({e["rig"] for e in row["entries"]})
    rig_names = {e["rig"] for row in rows for e in row["entries"]}
    names = {row["key"] for row in rows} | rig_names
    # Little-endian u32 at every byte offset (all four alignments), checked against the wanted hashes only.
    wanted = {joaat32(n) for n in names}
    found = set()
    view = memoryview(data)
    for a in range(4):
        if found == wanted:
            break
        whole = a + (len(data) - a) // 4 * 4
        found.update(h for (h,) in struct.iter_unpack("<I", view[a:whole]) if h in wanted)
    stored_hashes = sum(1 for n in names if joaat32(n) in found)
    entries = [e for row in rows for e in row["entries"]]
    by_source = {}
    for source in RBF0_SOURCES:
        selected = [e for e in entries if e["source"] == source]
        by_source[source] = {"entries": len(selected), "distinct_rigs": len({e["rig"] for e in selected})}
    groups = collections.Counter(e["group"] for e in entries if e["group"])
    return {
        "header": {"magic": "RBF0", "word_0x04": _u16(data, 4), "schema": _u16_string(data, 0x0A),
                   "bytes": len(data), "offset": offset},
        "summary": {
            "rigsets": len(rows),
            "entries": len(entries),
            "distinct_rigs": len(rig_names),
            "distinct_names": len(names),
            "empty_rigsets": sum(1 for row in rows if not row["entries"]),
            "entries_with_group": sum(1 for e in entries if e["group"]),
            "rigset_categories": dict(collections.Counter(row["key"].split("/")[2] for row in rows)),
            "sources": by_source,
            "groups": dict(groups),
            "stored_name_hashes": stored_hashes,
            "max_entries_per_rigset": max((len(row["entries"]) for row in rows), default=0),
            "max_distinct_rigs_per_rigset": max((row["distinct_rigs"] for row in rows), default=0),
        },
        "rigsets": rows,
    }


def _inflate_row(archive, row):
    with open(archive, "rb") as f:
        f.seek(row["offset"])
        return zlib.decompress(f.read(row["stored"]))


MAX_CHUNK = 65536   # a resource is split into 64 KiB streams; only a full one is followed by a continuation


def collect_rbf0(archive, rows):
    """Join the RBF0 stream with its 64 KiB continuation blocks. Stops at a known signature, a gap, or after
    a stream shorter than 64 KiB: that one ends the resource, so the next stream belongs to something else."""
    start = next((i for i, r in enumerate(rows) if r["signature"] == "rbf0"), None)
    if start is None:
        return None
    blob = bytearray(_inflate_row(archive, rows[start]))
    previous_end = rows[start]["offset"] + rows[start]["stored"]
    previous_size = rows[start]["size"]
    for row in rows[start + 1:]:
        if len(blob) >= 8 * 1024 * 1024 or row["offset"] > previous_end + 16 or previous_size != MAX_CHUNK:
            break
        if row["signature"] in STOP_SIGNATURES or row["size"] > MAX_CHUNK:
            break
        blob += _inflate_row(archive, row)
        previous_end = row["offset"] + row["stored"]
        previous_size = row["size"]
    return bytes(blob)


def collect_skeletons(archive, rows):
    """Standalone `SKEL_` streams plus small resources (motion matchers) that embed bone tables."""
    out = []
    for row in rows:
        if row["signature"] == "skeleton-names" or (row["size"] <= 16384 and row["signature"].startswith("raw:")):
            data = _inflate_row(archive, row)
            if b"SKEL_" in data or b"PH_" in data:
                record = parse_skel_names(data, row["offset"])
                record["kind"] = ("plain-bone-table" if row["signature"] == "skeleton-names"
                                  else ("motion-matcher" if b"motionmatcher" in data else "bone-table"))
                if record["distinct"] or record["hashes_present"]:
                    out.append(record)
    return out


def collect_skeletons_dir(path):
    out = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full) or name.lower().endswith((".json", ".csv")):
            continue
        size = os.path.getsize(full)
        with open(full, "rb") as f:
            head = f.read(5)
            if size > 16384 and head != b"SKEL_":
                continue
            data = head + f.read()
        m = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(m.group(1), 16) if m else 0
        sig = classify(data, offset, len(data))["signature"]
        if not (sig == "skeleton-names" or (size <= 16384 and sig.startswith("raw:"))):
            continue
        if b"SKEL_" in data or b"PH_" in data:
            record = parse_skel_names(data, offset)
            record["kind"] = ("plain-bone-table" if sig == "skeleton-names"
                              else ("motion-matcher" if b"motionmatcher" in data else "bone-table"))
            if record["distinct"] or record["hashes_present"]:
                out.append(record)
    return out


def collect_rbf0_dir(path):
    """The same continuation rule for a carved directory sorted by offset."""
    entries = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if not os.path.isfile(full) or name.lower().endswith((".json", ".csv")):
            continue
        m = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(m.group(1), 16) if m else len(entries)
        with open(full, "rb") as f:
            entries.append((offset, name, f.read()))
    entries.sort(key=lambda e: e[0])
    first = next((i for i, (_, _, data) in enumerate(entries) if data[:4] == b"RBF0"), None)
    if first is None:
        return None
    blob = bytearray(entries[first][2])
    previous_end = entries[first][0] + len(entries[first][2])
    previous_size = len(entries[first][2])
    for offset, _, data in entries[first + 1:]:
        if previous_size != MAX_CHUNK:
            break
        sign = classify(data, offset, len(data))["signature"]
        if len(blob) >= 8 * 1024 * 1024 or offset > previous_end + 16 or sign in STOP_SIGNATURES or len(data) > MAX_CHUNK:
            break
        blob += data
        previous_end = offset + len(data)
        previous_size = len(data)
    return bytes(blob)



def scan_rpf(path, start=None, limit=None, progress=None):
    """Walk an RPF9 data area. Returns (records, summary). No tables of contents are read or decrypted."""
    rows = []
    if os.path.getsize(path) == 0:
        raise ValueError("empty file")
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        size = len(m)
        first = struct.unpack_from("<I", m, 4)[0] if size >= 8 else 0
        pos = start if start is not None else first
        if not 0 <= pos <= size:
            raise ValueError("start offset is outside the archive")
        end = min(size, pos + limit) if limit else size
        span_start = pos
        last, zero, unread = pos, 0, 0
        while pos < end:
            match = ZHDR.search(m, pos, end)
            if not match:
                break
            s = match.start()
            if m[s:s + 3] == b"KB2":
                blen = (_u32(m, s + 4) or 0) + 8
                w, h = _u32(m, s + 0x14), _u32(m, s + 0x18)
                if not (64 <= blen <= end - s and w and h and 16 <= w <= 8192 and 16 <= h <= 8192):
                    pos = s + 1
                    continue
                z, u = gap_split(m, last, s)
                zero += z
                unread += u
                head = m[s:s + min(blen, 0x20)]
                record = classify(head, s, blen)
                record["size"] = blen
                rows.append(record)
                last = pos = s + blen
                continue
            d = zlib.decompressobj()
            chunks, p = [], s
            try:
                while not d.eof and p < end:   # inside the --limit window, as in story_index (#712)
                    block = m[p:min(p + (1 << 20), end)]
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
            z, u = gap_split(m, last, s)
            zero += z
            unread += u
            rows.append(classify(out, s, used))
            last = pos = s + used
            if progress and len(rows) % 2000 == 0:
                progress(len(rows), pos, end)
        z, u = gap_split(m, last, end)
        zero += z
        unread += u
    summary = summarise(rows, end - span_start, zero, unread)
    summary["mode"] = "rpf9"
    summary["archive"] = path
    return rows, summary


def scan_dir(path, progress=None):
    """Index already-carved files. An offset is taken from a leading hexadecimal file name when possible."""
    rows, span = [], 0
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        # rpf9_carve.py's index.json and any .csv report sit next to the streams; they are not payloads.
        if not os.path.isfile(full) or name.lower().endswith((".json", ".csv")):
            continue
        with open(full, "rb") as f:
            data = f.read()
        m = re.match(r"([0-9a-fA-F]{6,})", name)
        offset = int(m.group(1), 16) if m else len(rows)
        span += len(data)
        rows.append(classify(data, offset, len(data)))
    summary = summarise(rows, span, 0, 0)
    summary["mode"] = "directory"
    summary["directory"] = path
    if progress:
        progress(len(rows), len(rows), len(rows))
    return rows, summary


def summarise(rows, span, zero, unread):
    by_sig = {}
    for r in rows:
        a = by_sig.setdefault(r["signature"], {"files": 0, "stored": 0, "inflated": 0})
        a["files"] += 1
        a["stored"] += r["stored"]
        a["inflated"] += r["size"]
    stored = sum(r["stored"] for r in rows)
    return {
        "files": len(rows),
        "compressed_bytes": stored,
        "inflated_bytes": sum(r["size"] for r in rows),
        "bink2_files": sum(1 for r in rows if r["signature"] == "bink2"),
        "bink2_bytes": sum(r["stored"] for r in rows if r["signature"] == "bink2"),
        "zero_padding": zero,
        "unread_bytes": unread,
        "span": span,
        "readable_fraction": round(stored / span, 6) if span else 0.0,
        "signatures": by_sig,
    }


def write_output(path, rows, summary):
    """Write CSV or JSON from the file extension. CSV writes one summary line to stderr."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "files": rows}, f, indent=1)
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for r in rows:
            w.writerow([f"0x{r['offset']:x}" if i == 0 else ("" if r[c] is None else r[c])
                        for i, c in enumerate(COLUMNS)])


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


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", help="an .rpf archive or a directory of carved .bin files")
    ap.add_argument("out", help="output path ending in .csv or .json")
    ap.add_argument("--start", type=offset, help="first byte to scan (default: the RPF data start)")
    ap.add_argument("--limit", type=byte_count, help="number of bytes to scan")
    ap.add_argument("--rigs", metavar="PATH", help="write parsed RBF0 rig-set/rig and SKEL_ JSON here")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if same_report_path(a.source, a.out):
        ap.error("output must not overwrite the source")
    if a.rigs:
        if same_report_path(a.source, a.rigs):
            ap.error("--rigs must not overwrite the source")
        if same_report_path(a.out, a.rigs):
            ap.error("index and --rigs must use different paths")
    if os.path.isdir(a.out):
        ap.error(f"{a.out} is a folder, not a .csv or .json file")
    if a.rigs and os.path.isdir(a.rigs):
        ap.error(f"{a.rigs} is a folder, not a JSON file")
    if os.path.isdir(a.source):
        manifest = os.path.join(a.source, "index.json")
        for flag, path in (("output", a.out), ("--rigs", a.rigs)):
            if not path:
                continue
            # Reserved even before the carve writes it: other tools read <source>/index.json as the carve manifest.
            if same_report_path(manifest, path):
                ap.error(f"{flag} would overwrite the carve manifest: {path}")
            if os.path.isfile(path):
                for name in os.listdir(a.source):
                    inp = os.path.join(a.source, name)
                    if os.path.isfile(inp) and not name.lower().endswith((".json", ".csv")):
                        if same_report_path(inp, path):
                            ap.error(f"{flag} would overwrite an input file: {path}")
    if os.path.splitext(a.out)[1].lower() not in (".json", ".csv"):
        ap.error("output must end in .json or .csv")
    if a.rigs and os.path.splitext(a.rigs)[1].lower() != ".json":
        ap.error("--rigs must end in .json")
    for flag, path in (("output", a.out), ("--rigs", a.rigs)):
        if not path:
            continue
        folder = os.path.dirname(os.path.abspath(path))
        if not os.path.isdir(folder):
            sys.exit(f"{path}: cannot write: folder does not exist: {folder}")
    try:
        if os.path.isdir(a.source):
            if a.start is not None or a.limit is not None:
                print("--start and --limit are only supported for archive files", file=sys.stderr)
                return 2
            rows, summary = scan_dir(a.source)
        else:
            def progress(n, pos, end):
                if not a.quiet:
                    print(f"[{pos / max(1, end):6.1%}] {n} streams", flush=True)
            rows, summary = scan_rpf(a.source, a.start, a.limit, progress)
    except ValueError as e:
        sys.exit(f"{a.source}: {e}")
    except OSError as e:        # missing, a folder that can't be listed, a file that can't be read
        sys.exit(f"{e.filename or a.source}: {e.strerror or e}")
    try:
        write_output(a.out, rows, summary)
    except OSError as e:
        sys.exit(f"{a.out}: cannot write: {e.strerror or e}")
    if a.rigs:
        if os.path.isdir(a.source):
            blob = collect_rbf0_dir(a.source)
            skeletons = collect_skeletons_dir(a.source)
        else:
            blob = collect_rbf0(a.source, rows)
            skeletons = collect_skeletons(a.source, rows)
        if blob is None:
            print("no RBF0 stream found; --rigs needs the archive or carved folder that contains it", file=sys.stderr)
            return 2
        try:
            rbf0 = parse_rbf0(blob)
        except ValueError as e:
            # Parse before opening --rigs, so a failed parse leaves an existing file as it was.
            print(f"{a.source}: RBF0 stream: {e}", file=sys.stderr)
            return 2
        try:
            with open(a.rigs, "w", encoding="utf-8", newline="\n") as f:
                json.dump({"rbf0": rbf0, "skeletons": skeletons}, f, indent=1)
        except OSError as e:
            sys.exit(f"{a.rigs}: cannot write: {e.strerror or e}")
    top = dict(sorted(summary["signatures"].items(), key=lambda kv: -kv[1]["files"])[:15])
    print(json.dumps({k: summary[k] for k in
                      ("files", "compressed_bytes", "inflated_bytes", "zero_padding", "unread_bytes",
                       "span", "readable_fraction", "bink2_files")}, indent=1))
    print(f"{len(summary['signatures'])} signature classes; top 15:")
    for name, counts in top.items():
        print(f"  {name:24} {counts['files']:6} files {counts['stored']:12} stored {counts['inflated']:12} inflated")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
