"""Index the payloads of an RPF9 archive (or a directory of payloads) for BVH4 data.

The RPF9 table of contents in this package is encrypted, so file names cannot be recovered without keys; this tool
does not try. It walks the data area, decompresses every zlib stream that decodes cleanly to its end, and
classifies the inflated bytes from their headers. Nothing it writes contains game payload bytes: the rows hold
offsets, sizes, counts, a header tag and a bounding box.

The one signature it relies on was measured over the whole `*_bvh4` pack: a BVH stream has `flags = 0x81020006`
at offset 0x100, `totalSizeIn64BNodes` at 0x104, and wrapper words at 0x20/0x28 such that

    0x140 + 64 * totalSizeIn64BNodes == u32@0x20  and  u32@0x28 == u32@0x20 - 0x100

A stream is reported as `bvh4` only when all of that holds. The row also carries the exact XboxBvh4Header fields:
`crc32`, `numElements`, scene bounds, `offsetToNodes`, `offsetToPrimRemap`, `rootNodePtr`, `offsetToElements`,
`offsetToParents`, `compressionParams`, and the decoded root node index/type. A 208-byte instance record is
reported separately: it has the constant descriptor words listed in INSTANCE_WORDS and a 3x4 float matrix at 0x50.

Usage:
    python -I tools/bvh_index.py <archive.rpf> [--out index.json] [--csv index.csv]
                                 [--start N] [--limit BYTES] [--max-inflate BYTES]
    python -I tools/bvh_index.py <dir-with-rpf-or-bin> [--out index.json] [--csv index.csv]
                                 [--max-inflate BYTES]

--start and --limit apply to a single archive only; with a directory they are an error (exit 2).
--max-inflate applies to every .rpf in the directory.
Archive rows include a fill histogram only when the complete node range was kept.

Exit status is 0 even when nothing is found; the summary says how many streams were read.
"""
import argparse
import csv
import fnmatch
import json
import mmap
import os
import re
import struct
import sys
import tempfile
import time
import unicodedata
import zlib
from contextlib import suppress

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from zero_span import gap_split  # noqa: E402

ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
TAG = 0x81020006
HEAD = 0x140
TAG_OFF = 0x100
N_OFF = 0x104
END_OFF = 0x20
TAIL_OFF = 0x28
AABB_OFF = 0x110
INVALID_NODE = 0xFFFFFFFF
NODE_TYPE_NAMES = {0: "tri0", 1: "tri1", 2: "tri2", 3: "tri3",
                    4: "box16", 5: "box32", 6: "instance", 7: "aabb_geometry"}
INSTANCE_WORDS = ((0x08, 0x50000080), (0x10, 1), (0x18, 0x500000A0), (0x20, 0x60000000),
                  (0x28, 0x500000B0), (0x30, 0x500000C0), (0x38, 0x50000050))
FILL_BYTES = (0x00, 0x81, 0x94, 0x29)


def below_normal():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        k = ctypes.windll.kernel32
        # GetCurrentProcess returns a pointer-sized pseudo-handle.
        k.GetCurrentProcess.argtypes = ()
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetPriorityClass.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        k.SetPriorityClass.restype = wintypes.BOOL
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)


def decode_node_pointer(value):
    """Decode the low three type bits and the high 29-bit 64-byte node index."""
    typ = value & 7
    invalid = value == INVALID_NODE
    return {"ptr": value, "index": value >> 3, "type": typ,
            "name": "invalid" if invalid else NODE_TYPE_NAMES[typ], "invalid": invalid}


def parse_header(head):
    """Parse the 64-byte XboxBvh4Header at payload offset 0x100.

    The field names, types and offsets match the public XboxBvh4Header used by the Xbox DXR traversal
    shaders. The offsets in the header are relative to 0x100, the start of this header. The source
    declares offsetToPrimRemap as int32_t and offsetToParents as uint32_t, so they are unpacked differently.
    """
    if len(head) < HEAD or struct.unpack_from("<I", head, TAG_OFF)[0] != TAG:
        return None
    flags = struct.unpack_from("<I", head, 0x100)[0]
    root = struct.unpack_from("<I", head, 0x130)[0]
    root_ptr = decode_node_pointer(root)
    return {
        "flags": flags,
        "version": flags & 0xFFFF,
        "bottom_level": bool(flags & 0x80000000),
        "offline": bool(flags & 0x00020000),
        "dehydrated": bool(flags & 0x01000000),
        "compressed": bool(flags & 0x00010000),
        "compressed_for_gpu": bool(flags & 0x00080000),
        "node_count": struct.unpack_from("<I", head, N_OFF)[0],
        "crc32": struct.unpack_from("<I", head, 0x108)[0],
        "num_elements": struct.unpack_from("<I", head, 0x10C)[0],
        "bounds_min": list(struct.unpack_from("<3f", head, AABB_OFF)),
        "bounds_max": list(struct.unpack_from("<3f", head, AABB_OFF + 12)),
        "offset_to_nodes": struct.unpack_from("<I", head, 0x128)[0],
        "offset_to_prim_remap": struct.unpack_from("<i", head, 0x12C)[0],
        "root_node_ptr": root,
        "root_node_index": root_ptr["index"],
        "root_node_type": root_ptr["type"],
        "root_node_type_name": root_ptr["name"],
        "root_node_invalid": root_ptr["invalid"],
        "offset_to_elements": struct.unpack_from("<I", head, 0x134)[0],
        "offset_to_parents": struct.unpack_from("<I", head, 0x138)[0],
        "compression_params": struct.unpack_from("<I", head, 0x13C)[0],
    }


def classify(head, size, tail=None):
    """Return a dict describing one payload.

    Only the first 0x140 bytes are inspected, so this works for a carved file or for bytes streaming out of a
    decompressor. `size` is the inflated size.
    """
    row = {"kind": "raw", "signature": "", "version": ""}
    body = head[3:] if head[:3] == b"\xef\xbb\xbf" else head
    if body.startswith(b"<?xml"):
        row.update(kind="xml", signature="xml", fragment=False)
        rest = body
        close = rest.find(b"?>")
        if close >= 0:
            rest = rest[close + 2:]
        start = rest.find(b"<")
        end = rest.find(b">", start + 1)
        if 0 <= start < end:
            name = rest[start + 1:end].strip().split(b" ")[0].split(b"/")[0].lstrip(b"!-")
            row["root_element"] = name.decode("ascii", "replace")[:64]
        return row
    probe = head[:256]
    if size >= 64 and len(probe) >= 64:
        printable = sum(1 for c in probe if c in (9, 10, 13) or 32 <= c < 127)
        if printable >= len(probe) * 0.9 and b"<" in probe and b">" in probe:
            row.update(kind="xml", signature="xml", fragment=True)
            return row
    if size == 208 and all(struct.unpack_from("<Q", head, off)[0] == val for off, val in INSTANCE_WORDS):
        h = struct.unpack_from("<Q", head, 0x40)[0]
        w = struct.unpack_from("<I", head, 0xA4)[0]
        row.update(kind="instance208", signature="inst208:0x50000080", hash="%016x" % h,
                   matrix=[round(v, 4) for v in struct.unpack_from("<12f", head, 0x50)],
                   off_a0=struct.unpack_from("<I", head, 0xA0)[0], hi16=w >> 16, lo16=w & 0xFFFF)
        return row
    if 0x60 <= size <= 0x100000 and len(head) >= 0x30 and head[:8] == bytes(8) \
            and struct.unpack_from("<I", head, 8)[0] & 0xFF000000 == 0x50000000 \
            and struct.unpack_from("<I", head, 0xC)[0] == 0:
        row.update(kind="descriptor", signature="desc:0x5000xxxx",
                   desc_word="0x%08x" % struct.unpack_from("<I", head, 8)[0])
        return row
    if size >= HEAD and len(head) >= HEAD and struct.unpack_from("<I", head, TAG_OFF)[0] == TAG:
        n = struct.unpack_from("<I", head, N_OFF)[0]
        end = struct.unpack_from("<I", head, END_OFF)[0]
        tail = struct.unpack_from("<I", head, TAIL_OFF)[0]
        hdr = parse_header(head)
        if hdr is not None and 0x140 + n * 64 == end and tail == end - 0x100:
            common = dict(signature="bvh4:0x81020006", version=hdr["version"],
                          node_count=n, node_bytes=n * 64, tail_offset=tail, declared_bytes=end,
                          bounds_min=[round(v, 4) for v in hdr["bounds_min"]],
                          bounds_max=[round(v, 4) for v in hdr["bounds_max"]],
                          flags=hdr["flags"], crc32=hdr["crc32"], num_elements=hdr["num_elements"],
                          offset_to_nodes=hdr["offset_to_nodes"],
                          offset_to_prim_remap=hdr["offset_to_prim_remap"],
                          root_node_ptr=hdr["root_node_ptr"],
                          root_node_index=hdr["root_node_index"],
                          root_node_type=hdr["root_node_type"],
                          root_node_type_name=hdr["root_node_type_name"],
                          root_node_invalid=hdr["root_node_invalid"],
                          offset_to_elements=hdr["offset_to_elements"],
                          offset_to_parents=hdr["offset_to_parents"],
                          compression_params=hdr["compression_params"],
                          bottom_level=hdr["bottom_level"], offline=hdr["offline"],
                          dehydrated=hdr["dehydrated"], compressed=hdr["compressed"])
            if end <= size:
                row.update(kind="bvh4", chunked=False, **common)
            elif size % 65536 == 0:
                # A payload larger than 8 MiB is split into 64 KiB-multiple RPF chunks; the header's node range
                # continues in the following chunks, so the declared end is past this chunk's inflated size.
                row.update(kind="bvh4-chunk", chunked=True, **common)
    return row


def inflate_stream(m, s, size, max_inflate):
    """Inflate the zlib stream at `s`. Returns (head, kept, total, used) or None if it does not decode fully.

    `kept` is the payload capped at max_inflate bytes; `total` is the real inflated size.
    """
    d = zlib.decompressobj()
    keep, kept, total, p = [], 0, 0, s

    def take(b):
        nonlocal kept
        if b and kept < max_inflate:
            piece = b[:max_inflate - kept]
            keep.append(piece)
            kept += len(piece)

    try:
        while not d.eof and p < size:
            chunk = m[p:min(p + (1 << 20), size)]   # size is the scan window's end
            if not chunk:
                break
            take(d.decompress(chunk))
            p += len(chunk)
            if d.unused_data:
                break
        take(d.flush())
    except zlib.error:
        return None
    total = kept if kept < max_inflate else _count_stream(m, s, size)
    if not d.eof or total < 64:
        return None
    used = p - s - len(d.unused_data)
    out = b"".join(keep)
    return out[:HEAD], out, total, used


def _count_stream(m, s, size):
    """Inflated length of a stream larger than the kept cap (second cheap pass, no output kept)."""
    d = zlib.decompressobj()
    total, p = 0, s
    while not d.eof and p < size:
        chunk = m[p:min(p + (1 << 20), size)]
        if not chunk:
            break
        total += len(d.decompress(chunk))
        p += len(chunk)
        if d.unused_data:
            break
    total += len(d.flush())
    return total


def scan_rpf(path, start=None, limit=None, max_inflate=1 << 28):
    rows = []
    t0 = time.time()
    with open(path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size == 0:   # mmap refuses an empty file; it holds no streams
            if start:
                raise ValueError("start offset is outside the archive")
            return rows, [summarize(rows, path, 0, 0, 0, time.time() - t0)]
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            if start is not None:
                pos = start
            else:   # too short for the header word: nothing to scan
                pos = struct.unpack_from("<I", m, 4)[0] if size >= 8 else size
            if not 0 <= pos <= size:
                raise ValueError("start offset is outside the archive")
            end = min(size, pos + limit) if limit else size
            span = end - pos
            last, zero, unread = pos, 0, 0
            while pos < end:
                mt = ZHDR.search(m, pos, end)
                if not mt:
                    break
                s = mt.start()
                # Bounded by the scan window (the archive without --limit): a stream that runs past it is left out,
                # so readable + zero + unread add up to the span, as in story_index (#712).
                got = inflate_stream(m, s, end, max_inflate)
                if got is None:
                    pos = s + 1
                    continue
                head, kept, total, used = got
                gap_zero, gap_unread = gap_split(m, last, s)
                zero += gap_zero
                unread += gap_unread
                row = classify(head, total, kept[-HEAD:] if len(kept) >= HEAD else kept)
                row.update(source=path, offset=s, stored=used, size=total)
                if row["kind"] in ("bvh4", "bvh4-chunk") and row["declared_bytes"] <= len(kept):
                    body = kept[HEAD:HEAD + row["node_count"] * 64]
                    row["fill"] = {("%02x" % b): body.count(bytes([b])) for b in FILL_BYTES if body.count(bytes([b]))}
                    row["fill"]["total"] = len(body)
                rows.append(row)
                last = pos = s + used
            gap_zero, gap_unread = gap_split(m, last, end)
            zero += gap_zero
            unread += gap_unread
        finally:
            m.close()
    return rows, [summarize(rows, path, span, zero, unread, time.time() - t0)]


def summarize(rows, source, span, zero, unread, elapsed):
    kinds = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    stored = sum(r["stored"] for r in rows)
    inflated = sum(r["size"] for r in rows)
    bvh = [r for r in rows if r["kind"] in ("bvh4", "bvh4-chunk")]
    total_nodes = sum(r["node_count"] for r in bvh)
    return {"source": str(source), "streams": len(rows), "kinds": kinds, "stored": stored,
            "inflated": inflated, "span": span, "zero_padding": zero, "unread": unread,
            "readable_fraction": round(stored / span, 6) if span else None,
            "bvh4_streams": len(bvh), "bvh4_complete": sum(1 for r in bvh if r["kind"] == "bvh4"),
            "bvh4_chunked": sum(1 for r in bvh if r["kind"] == "bvh4-chunk"),
            "bvh4_nodes": total_nodes, "bvh4_node_bytes": total_nodes * 64,
            "elapsed_s": round(elapsed, 2)}


def find_payloads(root, patterns, skipped=None):
    """The files under root matching each glob pattern, as a sorted list per pattern. Hidden names are left out and
    links are followed, as glob.glob("**", recursive=True) did. Unlike glob, a folder that cannot be listed is not
    dropped silently: it is named on stderr and added to skipped, so the run reports itself incomplete."""
    found = {p: [] for p in patterns}

    def unlistable(error):
        print(f"{error.filename or root}: skipped ({error.strerror or error})", file=sys.stderr)
        if skipped is not None:
            skipped.append(error.filename or root)

    for folder, dirs, names in os.walk(root, onerror=unlistable, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            if not name.startswith("."):
                for p in patterns:
                    if fnmatch.fnmatch(name, p):
                        found[p].append(os.path.join(folder, name))
    return [sorted(found[p]) for p in patterns]


def scan_dir(path, max_inflate=1 << 28, skipped=None):
    rpfs, bins = find_payloads(os.fspath(path), ("*.rpf", "*.bin"), skipped)
    rows, summaries = [], []
    if rpfs:
        for p in rpfs:
            try:
                r, s = scan_rpf(p, max_inflate=max_inflate)
            except (OSError, ValueError) as e:   # one unreadable archive should not lose the others
                print(f"{p}: skipped ({e})", file=sys.stderr)
                if skipped is not None:
                    skipped.append(p)     # the caller still reports the run as incomplete
                continue
            rows += r
            summaries.extend(s)
        return rows, summaries
    for p in bins:
        size = os.path.getsize(p)
        with open(p, "rb") as f:
            head = f.read(HEAD)
            if size > HEAD:
                f.seek(-HEAD, os.SEEK_END)
                tail = f.read(HEAD)
            else:
                tail = head
        row = classify(head, size, tail)
        row.update(source=p, offset=0, stored=size, size=size)
        # A chunk exposes the header but has only a prefix of the declared node buffer.
        if row["kind"] == "bvh4":
            with open(p, "rb") as f:
                body = f.read(HEAD + row["node_count"] * 64)[HEAD:]
            row["fill"] = {("%02x" % b): body.count(bytes([b])) for b in FILL_BYTES}
            row["fill"]["total"] = len(body)
        rows.append(row)
    summary = summarize(rows, path, 0, 0, 0, 0.0)
    summary["span"] = None
    summary["readable_fraction"] = None
    return rows, [summary]


def _make_parent(path):
    """Create the folder a report lives in: the scan is over by the time we write, so a missing
    folder used to end the run in a FileNotFoundError and lose the result."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _publish(dest, write, newline):
    """Write a report next to `dest` and swap it in, so a run that fails or is interrupted while writing (a
    serialization error, a full disk, Ctrl+C) leaves the previous report as it was instead of a truncated one.
    The result keeps the previous report's permission bits, or gets the usual ones for a new file."""
    _make_parent(dest)
    directory = os.path.dirname(os.path.abspath(dest))
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline=newline, dir=directory,
                                         prefix=".bvh_index-", suffix=".tmp", delete=False) as f:
            temp = f.name
            write(f)
        os.chmod(temp, published_mode(dest))
        os.replace(temp, dest)
    except BaseException:
        if temp is not None:
            with suppress(OSError):
                # The staging file may have inherited a read-only destination's mode; make it deletable
                # first, or Windows strands it beside the report when the replace fails.
                os.chmod(temp, 0o600)
                os.unlink(temp)
        raise


def report_path_problem(path):
    """Why a report can't be written at `path`, or None: its folder can't be created (a file in the way, a
    missing drive) or can't be written to. Checked before the scan, so a mistyped destination doesn't cost a run."""
    parent = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        return f"cannot create its folder: {e}"
    if not os.access(parent, os.W_OK):
        return f"its folder is not writable: {parent}"
    if os.path.exists(path) and not os.access(path, os.W_OK):
        return "the existing report is read-only"
    return None


def write_outputs(rows, summaries, out, csv_path):
    if out:
        payload = summaries if len(summaries) > 1 else (summaries[0] if summaries else {})
        _publish(out, lambda f: json.dump({"summary": payload, "streams": rows}, f, indent=0), "\n")
    if csv_path:
        cols = ["source", "offset", "stored", "size", "kind", "signature", "version", "node_count",
                "node_bytes", "declared_bytes", "tail_offset", "flags", "crc32", "num_elements",
                "offset_to_nodes", "offset_to_prim_remap", "root_node_ptr", "root_node_index",
                "root_node_type", "root_node_type_name", "root_node_invalid", "offset_to_elements",
                "offset_to_parents",
                "compression_params", "dehydrated", "chunked", "bounds_min", "bounds_max",
                "hash", "off_a0", "hi16", "lo16", "desc_word", "fragment"]

        def write_csv(f):
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for r in rows:
                r = dict(r)
                for k in ("bounds_min", "bounds_max", "matrix"):
                    if k in r:
                        r[k] = " ".join(str(v) for v in r[k])
                w.writerow(r)

        _publish(csv_path, write_csv, "")


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


_PC_CASE_SENSITIVE = 11  # <sys/unistd.h> on macOS; os.pathconf_names doesn't list it


def _case_insensitive_volume(path):
    """True when the volume that will hold `path` ignores letter case, as macOS's default APFS volume does.
    os.path.normcase folds case only on Windows, so on macOS `R.json` and `r.json` would pass as two reports."""
    if sys.platform != "darwin":
        return False
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    try:
        return os.pathconf(path, _PC_CASE_SENSITIVE) == 0
    except (OSError, ValueError):
        return True  # the macOS default; folding can only refuse a pair, never overwrite one


def _path_key(path):
    path = os.path.normcase(os.path.realpath(path))
    if _case_insensitive_volume(path):
        # APFS and HFS+ also treat composed and decomposed accents as one name.
        path = unicodedata.normalize("NFD", path.casefold())
    return path


def same_output_path(first, second):
    if _path_key(first) == _path_key(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target", help="RPF9 archive or a directory of .rpf/.bin payloads")
    ap.add_argument("--out", help="write JSON (summary + per-stream rows)")
    ap.add_argument("--csv", help="write CSV (offset, size, signature, readable fields)")
    ap.add_argument("--start", type=offset, help="data-area start (default: u32 at file offset 4)")
    ap.add_argument("--limit", type=byte_count, help="stop after this many bytes of the data area")
    ap.add_argument("--max-inflate", type=byte_count, default=1 << 28,
                    help="keep at most this many bytes of one stream")
    a = ap.parse_args(argv)
    if a.out and a.csv and same_output_path(a.out, a.csv):
        ap.error("--out and --csv must use different paths")
    if a.out and same_output_path(a.target, a.out):
        ap.error("--out must not overwrite the source")
    if a.csv and same_output_path(a.target, a.csv):
        ap.error("--csv must not overwrite the source")
    for flag, path in (("--out", a.out), ("--csv", a.csv)):
        if path and os.path.isdir(path):
            ap.error(f"{flag} is a folder, not a report file: {path}")
    if os.path.isdir(a.target):
        manifest = os.path.join(a.target, "index.json")
        for flag, path in (("--out", a.out), ("--csv", a.csv)):
            if not path:
                continue
            if os.path.isfile(manifest) and same_output_path(path, manifest):
                ap.error(f"{flag} would overwrite the carve manifest: {path}")
            if os.path.isfile(path):
                for root, _dirs, names in os.walk(a.target):
                    for name in names:
                        if name.lower().endswith((".json", ".csv")):
                            continue
                        if same_output_path(os.path.join(root, name), path):
                            ap.error(f"{flag} would overwrite an input file: {path}")
    if os.path.isdir(a.target) and (a.start is not None or a.limit is not None):
        ap.error("--start and --limit apply to a single archive, not a directory")
    for flag, path in (("--out", a.out), ("--csv", a.csv)):
        problem = report_path_problem(path) if path else None
        if problem:
            ap.error(f"{flag} cannot be written to {path}: {problem}")
    below_normal()
    if os.path.isdir(a.target):
        skipped = []
        rows, summaries = scan_dir(a.target, a.max_inflate, skipped)
    else:
        skipped = []
        try:
            rows, summaries = scan_rpf(a.target, a.start, a.limit, a.max_inflate)
        except (OSError, ValueError) as e:
            # An OSError repeats the target we already print, and reprs it: name the reason once.
            reason = e.strerror if isinstance(e, OSError) and e.filename and e.strerror else e
            print(f"{a.target}: {reason}", file=sys.stderr)
            return 2
    for s in summaries:
        print(json.dumps(s, indent=1))
    try:
        write_outputs(rows, summaries, a.out, a.csv)
    except OSError as e:   # a full disk, a folder removed or locked since the preflight: a line, not a traceback
        print(f"cannot write the report: {e}", file=sys.stderr)
        return 2
    if not rows:
        print("no zlib streams found", file=sys.stderr)
    if skipped:
        print(f"{len(skipped)} archive(s) or folder(s) could not be scanned; the report is incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
