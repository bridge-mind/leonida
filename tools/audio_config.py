"""Read the music/radio record table from the game's inflated audio-config blob.

The audio config is the zlib-compressed section at the start of `audio.rpf`, before the first AWC bank.
It is not part of the table of contents and it is not a script resource. The music records sit in a
few streams inside it and share a fixed 46-byte layout, keyed by a marker that is stable across every
record:

    record - 4   u32   duration in milliseconds
    record + 0   0c 01              marker
    record + 2   u16   config id
    record + 4   02 00 0c 00
    record + 8   u32   hash A        (unresolved)
    record + 12  u32   hash B        joaat(normalized name)
    record + 16  00 00 02 23
    record + 20  22 bytes            fixed flags/parameters

`hash B` is Jenkins one-at-a-time over the lowercase name with spaces and punctuation mapped to `_`.
The AWC stream id is `hash B & 0x1FFFFFFF`. The config and the current AWC manifest report
per-channel duration, which is the actual song/ad length. In the original comparison, all
1,589 records resolved to a stream id and 1,584 durations matched within 0.01 s after
halving the legacy manifest values to account for interleaved stereo samples.

The config stores no per-track start offsets. It gives the station grouping as contiguous config-id
ranges and station-local content-id lists, but the mix order is not stored; a continuous radio mix
must be matched in the audio. This tool therefore outputs only the record table.

Usage:
    python -I audio_config.py <carve dir or blob> [--json] [--names FILE] [--join]
        [--min-ms N] [--max-ms N]

`--names` is an optional NUL-separated string pool (the package's debug/name stream); each string is
hashed with the same function and used to fill `name` for the matching hash B. The pool is a local
game file and is never copied or committed.

The input is the inflated output of `rpf9_carve.py`, either the per-stream `.bin` files or a single
concatenated config blob. Records are found by their marker and duration range, not by a hard-coded
stream offset, so an incomplete carve simply yields fewer records.

`--join` streams the directory's `.bin` parts in name order as one input, so a record split across
a zlib block boundary is still found. JSON `offset` is then the joined byte position and
`file_offset` the position inside `file`, the part holding the record's `0c` marker byte; text
output prints `file:file_offset`. The joined scan holds at most two read blocks of input
(`JOIN_BLOCK_BYTES`, 1 MiB) plus a 24-byte carry, one name and start per part, and the records.
"""
import argparse
from bisect import bisect_right
import json
import os
import struct
import sys

MARKER_1 = b"\x0c\x01"
MARKER_2 = b"\x02\x00\x0c\x00"
MARKER_3 = b"\x00\x00\x02\x23"
RECORD_BYTES = 46
MIN_MS = 20_000
MAX_MS = 2_000_000
STREAM_MASK = 0x1FFFFFFF
JOIN_BLOCK_BYTES = 1 << 20   # --join read size; a typical carved part is one read


def joaat(text):
    """Jenkins one-at-a-time over the lowercased latin-1 bytes, as the name hash is defined.

    The same rule as ``tools/hashdict.py`` (the pinned name dictionary) and ``tools/awc_names.py``, which
    recovers stream names: a name has to hash alike in every tool or the id never matches. Lowercased
    UTF-8 gave a different hash for any non-ASCII character (``é`` is one byte 0xE9, not 0xC3 0xA9), so a
    title with an accent produced a stream id no bank stores.
    """
    h = 0
    for c in text.lower().encode("latin-1", "replace"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    h = (h + (h << 15)) & 0xFFFFFFFF
    return h & 0xFFFFFFFF


def title_stream_id(title: str) -> int:
    """AWC stream id for a config title.

    The config's ``hash B`` is ``joaat(normalized_name(title))`` and the
    stream id is its low 29 bits. This is the lookup used to resolve untitled
    config records when the title is known from credits or a playlist.
    """
    return joaat(normalized_name(title)) & STREAM_MASK


def normalized_name(text):
    """Lowercase, and map every non-alphanumeric run to a single underscore."""
    out = []
    last = False
    for ch in text.lower():
        ok = ch.isalnum()
        if ok:
            out.append(ch)
        elif not last:
            out.append("_")
        last = not ok
    return "".join(out).strip("_")


def _record_at(blob, i, offset, min_ms, max_ms):
    """Return the record whose `0c 01` marker is at blob[i], reported at `offset`, or None."""
    if blob[i + 4:i + 8] != MARKER_2 or blob[i + 16:i + 20] != MARKER_3:
        return None
    duration = struct.unpack_from("<I", blob, i - 4)[0]
    if not (min_ms <= duration <= max_ms):
        return None
    record = {
        "offset": offset,
        "id": struct.unpack_from("<H", blob, i + 2)[0],
        "duration_ms": duration,
        "duration_s": round(duration / 1000.0, 3),
        "hash_a": struct.unpack_from("<I", blob, i + 8)[0],
        "hash_b": struct.unpack_from("<I", blob, i + 12)[0],
    }
    record["stream_id"] = record["hash_b"] & STREAM_MASK
    return record


def scan(blob, min_ms=MIN_MS, max_ms=MAX_MS):
    """Return the record table found in one inflated config blob."""
    out = []
    end = len(blob) - 20
    pos = 4
    while pos <= end:  # a marker with exactly 20 bytes after it is complete
        record = blob[pos:pos + 2] == MARKER_1 and _record_at(blob, pos, pos, min_ms, max_ms)
        if not record:
            pos += 1
            continue
        out.append(record)
        pos += RECORD_BYTES
    return out


def scan_stream(chunks, min_ms=MIN_MS, max_ms=MAX_MS):
    """Yield the records scan() finds in b"".join(chunks), without joining the chunks.

    `offset` is the joined position. A position is decided only once 20 bytes follow it, so the
    carry between chunks is at most 24 bytes: the undecided tail plus the duration before it.
    """
    buf = b""
    base = 0          # joined position of buf[0]
    pos = 4           # next undecided joined position
    for chunk in chunks:
        buf += chunk
        end = base + len(buf) - 20
        while pos <= end:
            i = buf.find(MARKER_1, pos - base, end - base + 2)  # +2 for the two marker bytes
            if i < 0:
                pos = end
                break
            record = _record_at(buf, i, base + i, min_ms, max_ms)
            if record:
                yield record
            pos = base + i + (RECORD_BYTES if record else 1)
        cut = min(pos - 4, base + len(buf))
        buf, base = buf[cut - base:], cut


def paths_in(src):
    """Yield (name, path) for a blob or every `.bin` file in a directory, in name order."""
    if os.path.isfile(src):
        yield os.path.basename(src), src
        return
    for name in sorted(os.listdir(src)):
        path = os.path.join(src, name)
        if name.endswith(".bin") and os.path.isfile(path):
            yield name, path


def files_in(src):
    """Yield (name, bytes) for a blob or every `.bin` file in a directory, in name order."""
    for name, path in paths_in(src):
        with open(path, "rb") as f:
            yield name, f.read()


def load_names(path):
    """Map joaat -> display string for a NUL-separated pool."""
    with open(path, "rb") as f:
        pool = f.read()
    names = {}
    for item in pool.split(b"\x00"):
        if not item:
            continue
        text = item.decode("latin-1")    # a name's bytes, as hashdict.py and awc_names.py read them
        key = joaat(normalized_name(text))
        if key not in names:
            names[key] = text
    return names


def collect(src, min_ms=MIN_MS, max_ms=MAX_MS, names=None, join=False):
    """Scan `src` and add `file` and `file_offset`, the record's position inside `file`.

    Without `join`, `offset` equals `file_offset`. With `join`, the parts are streamed in name
    order, `offset` is the joined position and `file` is the part holding the `0c` marker byte.
    """
    records = []
    if join:
        files, starts = [], []

        def chunks():
            seen = 0
            for name, path in paths_in(src):
                files.append(name)
                starts.append(seen)
                with open(path, "rb") as f:
                    while chunk := f.read(JOIN_BLOCK_BYTES):
                        seen += len(chunk)
                        yield chunk

        for record in scan_stream(chunks(), min_ms, max_ms):
            index = bisect_right(starts, record["offset"]) - 1
            record["file"] = files[index]
            record["file_offset"] = record["offset"] - starts[index]
            records.append(record)
    else:
        for name, blob in files_in(src):
            for record in scan(blob, min_ms, max_ms):
                record["file"] = name
                record["file_offset"] = record["offset"]
                records.append(record)
    if names:
        for record in records:
            record["name"] = names.get(record["hash_b"], "")
    return records


def summarize(records):
    ids = [r["id"] for r in records]
    counts = {}
    for i in ids:
        counts[i] = counts.get(i, 0) + 1
    return {
        "records": len(records),
        "distinct_ids": len(counts),
        "id_min": min(ids) if ids else None,
        "id_max": max(ids) if ids else None,
        "duration_s": round(sum(r["duration_ms"] for r in records) / 1000.0, 3),
        "named": sum(1 for r in records if r.get("name")),
        "duplicate_ids": sorted(i for i, n in counts.items() if n > 1),
    }


def main():
    ap = argparse.ArgumentParser(description="Read the audio-config music/radio record table.")
    ap.add_argument("src", help="inflated config blob or directory of carved .bin streams")
    ap.add_argument("--json", action="store_true", help="write records and summary as JSON")
    ap.add_argument("--names", help="optional NUL-separated name pool for hash B")
    ap.add_argument("--join", action="store_true",
                    help="scan the directory's .bin streams in name order as one stream "
                         "(records can straddle a zlib block boundary); text shows the offset in "
                         "the file, JSON keeps the joined offset and adds file_offset")
    ap.add_argument("--min-ms", type=int, default=MIN_MS, help="minimum duration (default 20000)")
    ap.add_argument("--max-ms", type=int, default=MAX_MS, help="maximum duration (default 2000000)")
    a = ap.parse_args()
    if a.min_ms < 1 or a.max_ms < a.min_ms:
        ap.error("require 1 <= min-ms <= max-ms")

    if not os.path.exists(a.src):
        print(f"ERROR: {a.src}: does not exist", file=sys.stderr)
        return 2
    if a.join and not os.path.isdir(a.src):
        ap.error("--join applies to a directory, not a single file")
    try:
        names = load_names(a.names) if a.names else None
    except OSError as e:
        sys.exit(f"{a.names}: cannot read the name pool: {e.strerror or e}")
    try:
        records = collect(a.src, a.min_ms, a.max_ms, names, a.join)
    except OSError as e:
        sys.exit(f"{e.filename or a.src}: cannot read: {e.strerror or e}")
    if not records:
        sys.exit(f"{a.src}: no audio-config records found")
    summary = summarize(records)

    if a.json:
        json.dump({"records": records, "summary": summary}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    for r in records:
        label = r.get("name") or f"0x{r['hash_b']:08X}"
        print(f"{r['file']}:{r['file_offset']:6d}  id=0x{r['id']:04X}  "
              f"dur={r['duration_s']:9.3f}s  stream=0x{r['stream_id']:08X}  {label}")
    print(f"{summary['records']} records, {summary['distinct_ids']} distinct ids, "
          f"ids 0x{summary['id_min']:04X}-0x{summary['id_max']:04X}, "
          f"{summary['duration_s']:.1f} s total"
          + (f", {summary['named']} named" if names else ""))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
