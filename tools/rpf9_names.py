"""Index named RPF9 resources without the encrypted table of contents.

The RPF9 data area is mostly zlib streams laid end to end. This walks it the same way
``rpf9_carve.py`` does, but inflates only a bounded prefix of each stream and records the
measured resource-name spelling (``name.#dr``, ``name.#ft``, ``name.#dd``, ...). The result is
a lightweight lookup index for targeted model/animation extraction; it does not decode a model
and it never writes game data.

Usage:
    python -I rpf9_names.py <archive.rpf> <out.json> [--start N] [--limit BYTES]
                            [--max-inflate BYTES] [--types dr,ft,dd] [--quiet]

The output contains archive offsets, compressed sizes, inflated sizes and the names found in
each stream. It is local scratch; do not commit the names or the index.
"""
import argparse
import json
import mmap
import os
import re
import struct
import sys
import tempfile
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from file_mode import published_mode  # noqa: E402
from zero_span import gap_split  # noqa: E402

# zlib stream start, or a Bink 2 video ('KB2' + revision letter, then u32 size-8).
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]|KB2[a-z]")

# Cap on the transient output a single decompress() call may produce, and the largest
# input feed read from the map. ``max_inflate`` remains the bound on
# how much of the inflated stream is kept for name extraction.
DECOMPRESS_MAX = 1 << 20
# Most short streams end in the first read. zlib copies any bytes beyond EOF
# into unused_data, so reading a full MiB up front copies the next streams too.
INITIAL_FEED = 4096
NAME = re.compile(rb"(?<![A-Za-z0-9_\-])([A-Za-z0-9_\-]{2,120})\.#([A-Za-z0-9]{1,8})(?![A-Za-z0-9_\-])")


def parse_types(text):
    """Parse and validate a comma-separated suffix list such as ``dr,ft,dd``."""
    types = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if not re.fullmatch(r"[a-z0-9]{1,8}", part):
            raise ValueError(f"invalid resource suffix: {part!r}")
        types.append(part)
    if not types:
        raise ValueError("at least one resource suffix is required")
    return sorted(set(types))


def extract_names(prefix, types):
    """Return the sorted unique full names in ``prefix`` whose suffix is in ``types``."""
    found = {}
    wanted = set(types)
    for match in NAME.finditer(prefix):
        suffix = match.group(2).decode("ascii").lower()
        if suffix not in wanted:
            continue
        found[match.group(0).decode("ascii")] = suffix
    return [(name, found[name]) for name in sorted(found)]


def scan_archive(archive, start=None, limit=None, max_inflate=1 << 20, types=("dr", "ft", "dd"),
                 progress=None):
    """Scan an RPF9 data area and return ``(rows, summary)``.

    ``rows`` only contains streams with at least one matching name. ``summary`` counts every
    valid zlib stream so the coverage of the pass is explicit.
    """
    with open(archive, "rb") as fh:
        if os.fstat(fh.fileno()).st_size < 8:
            raise ValueError("not an RPF9 archive")
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            size = len(m)
            if size < 8 or m[:4] not in (b"RPF9", b"9FPR"):
                raise ValueError("not an RPF9 archive")
            data_start = struct.unpack_from("<I", m, 4)[0]
            if not 8 <= data_start <= size:
                raise ValueError("RPF9 data start is outside the archive")
            pos = data_start if start is None else start
            if not 0 <= pos <= size:
                raise ValueError("--start is outside the archive")
            end = min(size, pos + limit) if limit is not None else size
            rows = []
            last = pos
            zero = 0
            unread = 0
            streams = 0
            bink2_files = 0
            bink2_bytes = 0
            name_hits = 0
            unique = {}
            t0 = time.time()
            while pos < end:
                mt = ZHDR.search(m, pos, end)
                if not mt:
                    break
                s = mt.start()
                if m[s:s + 3] == b"KB2":
                    # Same header test as rpf9_carve.py: u32 size-8 at +4, width/height at +0x14.
                    if s + 0x1C > end:   # header cut off by the end of the window: not a video
                        pos = s + 1
                        continue
                    blen = struct.unpack_from("<I", m, s + 4)[0] + 8
                    w, h = struct.unpack_from("<II", m, s + 0x14)
                    if not (64 <= blen <= end - s and 16 <= w <= 8192 and 16 <= h <= 8192):
                        pos = s + 1
                        continue
                    gap_zero, gap_unread = gap_split(m, last, s)
                    zero += gap_zero
                    unread += gap_unread
                    bink2_files += 1
                    bink2_bytes += blen
                    last = pos = s + blen
                    continue
                d = zlib.decompressobj()
                prefix = bytearray()
                total = 0
                p = s
                pending = b""
                feed_size = min(INITIAL_FEED, DECOMPRESS_MAX)
                try:
                    # Bound each decompress() call's transient output with max_length and
                    # re-feed unconsumed_tail before pulling more input, so a stream cannot
                    # balloon memory regardless of how large it inflates.
                    while not d.eof and (pending or p < end):
                        if not pending:
                            pending = m[p:p + min(feed_size, end - p)]
                            p += len(pending)
                            feed_size = min(feed_size * 2, DECOMPRESS_MAX)
                        data = d.decompress(pending, DECOMPRESS_MAX)
                        total += len(data)
                        if len(prefix) < max_inflate:
                            prefix.extend(data[:max_inflate - len(prefix)])
                        pending = d.unconsumed_tail
                except zlib.error:
                    pos = s + 1
                    continue
                if not d.eof or total < 64:
                    pos = s + 1
                    continue
                used = p - s - len(d.unused_data)
                if used <= 0:
                    pos = s + 1
                    continue
                gap_zero, gap_unread = gap_split(m, last, s)
                zero += gap_zero
                unread += gap_unread
                streams += 1
                names = extract_names(prefix, types) if b".#" in prefix else []
                if names:
                    name_hits += len(names)
                    for name, suffix in names:
                        unique.setdefault(name, suffix)
                    rows.append({"offset": s, "stored": used, "size": total,
                                 "names": [name for name, _ in names]})
                last = pos = s + used
                if progress and streams % 2000 == 0:
                    span = max(1, end - (data_start if start is None else start))
                    progress(f"{streams} streams, {len(rows)} with names, "
                             f"{(pos - (data_start if start is None else start)) / span:.1%}, "
                             f"{time.time() - t0:.0f}s")
            gap_zero, gap_unread = gap_split(m, last, end)
            zero += gap_zero
            unread += gap_unread
            span = end - (data_start if start is None else start)
            by_type = {}
            for suffix in unique.values():
                by_type[suffix] = by_type.get(suffix, 0) + 1
            summary = {
                "archive": os.path.abspath(archive),
                "start": data_start if start is None else start,
                "span": span,
                "streams": streams,
                "streams_with_names": len(rows),
                "name_hits": name_hits,
                "unique_names": len(unique),
                "by_type": dict(sorted(by_type.items())),
                "max_inflate": max_inflate,
                "types": list(types),
                "zero_padding": zero,
                "unread_bytes": unread,
                "bink2_files": bink2_files,
                "bink2_bytes": bink2_bytes,
            }
            return rows, summary


def publish_report(path, rows, summary):
    """Write the report at ``path`` atomically, creating missing parent directories.

    The full JSON is staged next to the destination so an interrupted run leaves only the old report or no report,
    never a truncated one. Raised OSError carries a clear description of the staging/destination problem for
    ``main`` to turn into a named nonzero diagnostic.
    """
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory, prefix=".rpf9_names-") as stage_dir:
        temp = os.path.join(stage_dir, "report.json")
        with open(temp, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "files": rows}, f, indent=1)
            f.write("\n")
        os.chmod(temp, published_mode(path))
        os.replace(temp, path)


def _int_arg(text):
    try:
        return int(text, 0)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive", help="an .rpf archive")
    ap.add_argument("out", help="output JSON path")
    ap.add_argument("--start", type=_int_arg, help="first byte to scan (default: RPF data start)")
    ap.add_argument("--limit", type=_int_arg, help="number of bytes to scan")
    ap.add_argument("--max-inflate", type=_int_arg, default=1 << 20,
                    help="maximum inflated prefix to inspect per stream (default: 1 MiB)")
    ap.add_argument("--types", default="dr,ft,dd", help="comma-separated resource suffixes")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        types = parse_types(a.types)
        if a.start is not None and a.start < 0:
            raise ValueError("--start must be >= 0")
        if a.limit is not None and a.limit <= 0:
            raise ValueError("--limit must be > 0")
        if a.max_inflate <= 0:
            raise ValueError("--max-inflate must be > 0")
        if not os.path.isfile(a.archive):
            raise ValueError("archive does not exist")
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    out_abs = os.path.abspath(a.out)
    try:
        if os.path.exists(a.out) and os.path.samefile(a.archive, a.out):
            raise ValueError(f"output {a.out} is the input archive")
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    progress = None
    if not a.quiet:
        progress = lambda text: print(f"  {text}", file=sys.stderr, flush=True)
    try:
        rows, summary = scan_archive(a.archive, a.start, a.limit, a.max_inflate, types, progress)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        publish_report(out_abs, rows, summary)
    except OSError as exc:
        print(f"ERROR: cannot write report {a.out}: {exc.strerror or exc}", file=sys.stderr)
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
