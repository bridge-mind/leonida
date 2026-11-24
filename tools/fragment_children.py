"""Census the child drawables of the fragments in an RPF9 archive (ASSET-261).

docs/character_models.md records that every one of the 2,865 `packs.rpf` fragments has a single drawable and
no child arrays. This tool measures the same thing for any archive, `levels.rpf` included, from the raw
system pages: no manifest, no table of contents and no decoder is needed. Only archives whose resources are
plain page-aligned zlib streams are read, the way tools/skin_streams.py reads them.

A fragment is the resource shape rpf9-explorer's `RageModelDecoder` calls one: not a drawable, not a drawable
dictionary, and a header whose `+0x08`, `+0x18` and `+0xA8` fields are system pointers, whose `+0x10`, `+0x20`
and `+0x28` fields are null or system pointers, and whose u16 fields at `+0x30..+0x36` read (a, b, c, d) with
0 < a <= 4096, b <= 4096 and c <= d <= 4096. `+0x18` points at the fragment's main drawable.

Layout measured here (counts only; nothing the archive holds is written out):

    drawable array   `a` (u16 at `+0x30`) is the number of drawables. They are 0x110-byte records at a fixed
                     stride starting at the main drawable `+0x18` points at, so record i is at main + 0x110 * i.
                     Records 1..a-1 are the child drawables; their `+0x00` and `+0x08` fields are zero.
    drawable body    what the tool accepts as a drawable at an offset: `+0x00` and `+0x08` zero, `+0x10` (shader
                     group) and `+0xA8` (name) null or system pointers, finite ordered bounds, a finite
                     radius in [0, 1e6), and a LOD list behind `+0x50` (u16 model count 1..256 at +0x08,
                     repeated at +0x0A, an array of system pointers to models whose geometry count at
                     +0x10 equals the copies at +0x12 and +0x36).
    child drawable   a body at main + 0x110 * i for 1 <= i < a (the declared children).
    stray drawable   a body at any other 16-byte-aligned offset of the system pages outside the main record,
                     found by scanning; a count of what the array walk would miss.
    child array      a fragment whose array holds at least one child drawable.
    part element     the 0x50-byte records of the array `+0x28` points at (c of them, c = u16 at `+0x34`);
                     the fragments with more than one are counted too.

A fragment with no child drawable and no stray drawable has nothing the main-drawable export misses.

Usage:
    python -I tools/fragment_children.py <archive.rpf> [--jobs N] [--start BYTES] [--end BYTES]
                                         [--records out.jsonl] [--json out.json]

`--start`/`--end` bound the byte range scanned (resume a long run, or split it); `--jobs` splits the range
over worker processes. `--records` writes one JSON line per fragment with its stream offset and the measured
numbers, for a re-count without a re-scan; it is a local working file, not something to publish.
"""
import argparse
import array as array_module
import json
import mmap
import multiprocessing
import os
import struct
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I keeps the script directory off sys.path
import skin_streams as ss  # noqa: E402

SYSTEM = 0x50000000
PAGE = 4096
MAX_SYSTEM_BYTES = 64 * 1024 * 1024
MAX_PAGES = 32
MAX_ELEMENTS = 4096
ELEMENT_STRIDE = 0x50
RECORD = 0x110                # stride of the drawable records
HEADER = 192                  # a fragment header; nothing embedded starts inside it
ZHDR = ss.ZHDR


def u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def system_pointer(v):
    return SYSTEM <= v < SYSTEM + ss.MAX_SYS


def drawable_header(b, at, embedded):
    """The decoder's drawable-header test on b[at:at + 176]."""
    if at < 0 or at + 176 > len(b) or u64(b, at) != 0:
        return False
    pages, shader, high, name = u64(b, at + 8), u64(b, at + 16), u64(b, at + 80), u64(b, at + 168)
    if (pages != 0) if embedded else (not system_pointer(pages)):
        return False
    if not (system_pointer(shader) and system_pointer(high)):
        return False
    if not (system_pointer(name) or (embedded and name == 0)):
        return False
    radius = struct.unpack_from("<f", b, at + 44)[0]
    return 0 <= radius < 1e6 and ss.finite_bounds(b, at + 48)


def drawable_body(b, at):
    """The drawable-body test described in the module docstring, for the record at `at`."""
    n = len(b)
    if at < 0 or at + 176 > n or u64(b, at) != 0 or u64(b, at + 8) != 0:
        return False
    shader, name, lod = u64(b, at + 16), u64(b, at + 168), u64(b, at + 80)
    if (shader and not system_pointer(shader)) or (name and not system_pointer(name)):
        return False
    radius = struct.unpack_from("<f", b, at + 44)[0]
    if not 0 <= radius < 1e6 or not ss.finite_bounds(b, at + 48):
        return False
    lod -= SYSTEM
    if not 0 <= lod <= n - 16:
        return False
    models = u16(b, lod + 8)
    array_at = u64(b, lod) - SYSTEM
    if not (0 < models <= 256 and models == u16(b, lod + 10) and 0 <= array_at <= n - models * 8):
        return False
    for i in range(models):
        model = u64(b, array_at + 8 * i) - SYSTEM
        if not 0 <= model <= n - 56:
            return False
        geometries = u16(b, model + 16)
        if not (0 < geometries <= 256 and geometries == u16(b, model + 18) == u16(b, model + 54)):
            return False
    return True


def fragment_header(b):
    """The decoder's fragment test on the first 176+ bytes of a system page."""
    if len(b) < 176 or u64(b, 0) != 0:
        return False
    if not all(system_pointer(u64(b, at)) for at in (8, 24, 168)):
        return False
    if not all(u64(b, at) == 0 or system_pointer(u64(b, at)) for at in (16, 32, 40)):
        return False
    a, bb, c, d = struct.unpack_from("<4H", b, 48)
    return 0 < a <= 4096 and bb <= 4096 and c <= d <= 4096


def classify(prefix):
    """'fragment', or None when the first bytes are a drawable, a dictionary or nothing the tool counts."""
    if len(prefix) < 192 or drawable_header(prefix, 0, False):
        return None
    if ss.resource_kind(prefix) == "dictionary":
        return None
    return "fragment" if fragment_header(prefix) else None


def read_system(m, at, size):
    """The system pages of the resource whose first zlib stream starts at `at`, joined; None if unreadable.

    The page directory (`+0x08`) can sit past the first page, so pages are read until it is reachable.
    """
    data, used = ss.inflate_at(m, at, size)
    if data is None or len(data) < 192:
        return None
    directory = u64(data, 8) - SYSTEM
    if not 0 <= directory < MAX_SYSTEM_BYTES:
        return None
    parts, total, fp, pages = [data], len(data), at + used, 1
    while directory + 10 > total or pages < 1:
        nxt = ss.next_stream(m, fp, size)
        if nxt is None or pages >= MAX_PAGES:
            return None
        parts.append(nxt[0])
        total += len(nxt[0])
        fp = nxt[1] + nxt[2]
        pages += 1
    joined = b"".join(parts)
    system_pages = joined[directory + 8]
    if not 0 < system_pages <= MAX_PAGES:
        return None
    while pages < system_pages:
        nxt = ss.next_stream(m, fp, size)
        if nxt is None:
            return None
        parts.append(nxt[0])
        fp = nxt[1] + nxt[2]
        pages += 1
    joined = b"".join(parts)
    return joined, fp


def stray_bodies(system, array):
    """Offsets of drawable bodies outside the record array: `array` is the (start, end) the records span."""
    n = len(system)
    words = array_module.array("Q")
    words.frombytes(system[:n // 8 * 8])
    if sys.byteorder != "little":
        words.byteswap()
    found, low, high = [], SYSTEM, SYSTEM + n
    for at in range(HEADER, n - 176 + 1, 16):
        if array[0] <= at < array[1]:
            continue
        i = at >> 3
        if words[i] or words[i + 1] or not low <= words[i + 10] < high:   # +0x00, +0x08, the LOD pointer
            continue
        if drawable_body(system, at):
            found.append(at)
    return found


def walk(system):
    """Measure one fragment's system pages; returns a dict of counts, or None if the header is unusable."""
    if not fragment_header(system):
        return None
    main = u64(system, 24) - SYSTEM
    a, b, c, d = struct.unpack_from("<4H", system, 48)
    rec = {"main": main, "abcd": [a, b, c, d],
           "optional_null": sum(1 << i for i, at in enumerate((16, 32, 40)) if u64(system, at) == 0)}
    rec["main_ok"] = 0 <= main <= len(system) - 176 and drawable_header(system, main, True)
    found = [i for i in range(1, a) if drawable_body(system, main + RECORD * i)]
    rec["children"] = len(found)
    rec["declared"] = a - 1
    # the main record is excluded from the scan through the array span, so a missing child stays "stray"
    rec["stray"] = len(stray_bodies(system, (main, main + RECORD * a))) if main >= HEADER else -1
    elements = 0
    table = u64(system, 40) - SYSTEM
    if c and u64(system, 40) and 0 <= table <= len(system) - c * ELEMENT_STRIDE and c <= MAX_ELEMENTS:
        elements = c
    rec["elements"] = elements
    return rec


def empty_stats():
    return {"fragments": 0, "unreadable": 0, "main_drawable_invalid": 0, "optional_pointers_null": 0,
            "multi_drawable_fragments": 0, "declared_child_drawables": 0, "child_drawables": 0,
            "child_arrays": 0, "stray_drawables": 0, "fragments_with_stray": 0, "max_children": 0,
            "multi_part_fragments": 0, "part_elements": 0}


def count(stats, rec):
    stats["fragments"] += 1
    if not rec["main_ok"]:
        stats["main_drawable_invalid"] += 1
    if rec["optional_null"]:
        stats["optional_pointers_null"] += 1
    if rec["declared"]:
        stats["multi_drawable_fragments"] += 1
        stats["declared_child_drawables"] += rec["declared"]
    if rec["children"]:
        stats["child_arrays"] += 1
        stats["child_drawables"] += rec["children"]
        stats["max_children"] = max(stats["max_children"], rec["children"])
    if rec["stray"] > 0:
        stats["fragments_with_stray"] += 1
        stats["stray_drawables"] += rec["stray"]
    stats["part_elements"] += rec["elements"]
    if rec["elements"] > 1:
        stats["multi_part_fragments"] += 1


def merge(into, other):
    for key, value in other.items():
        into[key] = max(into[key], value) if key == "max_children" else into[key] + value


def scan_range(path, start, end, records, log=None):
    """Scan [start, end) of one archive; returns the stats. `records` is a writable handle or None."""
    stats = empty_stats()
    size = os.path.getsize(path)
    end = min(end, size)
    if size == 0 or start >= end:
        return stats
    t0 = time.time()
    with open(path, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        pos = start
        while pos < end:
            page = (pos + PAGE - 1) & ~(PAGE - 1)
            at = page + 16
            if at + 2 > size or page >= end:
                break
            pos = page + PAGE
            if m[at - 16:at] != b"\0" * 16 or not ZHDR.match(m, at):
                continue
            try:
                prefix = zlib.decompressobj().decompress(m[at:at + 4 * PAGE], 192)
            except zlib.error:
                continue
            if classify(prefix) is None:
                continue
            got = read_system(m, at, size)
            rec = None
            if got is not None:
                rec = walk(got[0])
            if rec is None:
                stats["unreadable"] += 1
                continue
            count(stats, rec)
            pos = max(pos, got[1])   # resume after the pages this resource consumed
            if records is not None:
                records.write(json.dumps(dict(rec, at=at), separators=(",", ":")) + "\n")
                records.flush()
            if log and stats["fragments"] % 5000 == 0:
                log(f"  {os.path.basename(path)} @{page:#x}: {stats['fragments']} fragments, "
                    f"{time.time() - t0:.0f}s")
    return stats


def worker(args):
    path, start, end, records_path = args
    handle = open(records_path, "w", encoding="utf-8", newline="\n") if records_path else None
    try:
        return scan_range(path, start, end, handle)
    finally:
        if handle:
            handle.close()


def split_ranges(start, end, jobs):
    step = max(PAGE, ((end - start) // jobs + PAGE - 1) // PAGE * PAGE)
    return [(lo, min(lo + step, end)) for lo in range(start, end, step)]


def scan_archive(path, start=0, end=None, jobs=1, records=None, log=None):
    size = os.path.getsize(path)
    end = size if end is None else min(end, size)
    total = empty_stats()
    if jobs <= 1 or end - start < 2 * PAGE:
        handle = open(records, "w", encoding="utf-8", newline="\n") if records else None
        try:
            merge(total, scan_range(path, start, end, handle, log))
        finally:
            if handle:
                handle.close()
        return total
    ranges = split_ranges(start, end, jobs)
    work = [(path, lo, hi, f"{records}.{i}" if records else None) for i, (lo, hi) in enumerate(ranges)]
    with multiprocessing.Pool(len(work)) as pool:
        for part in pool.imap_unordered(worker, work):
            merge(total, part)
    if records:   # join the per-range files in range order
        with open(records, "w", encoding="utf-8", newline="\n") as out:
            for i in range(len(work)):
                part = f"{records}.{i}"
                with open(part, encoding="utf-8") as fh:
                    out.write(fh.read())
                os.remove(part)
    return total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive", help="RPF9 archive to scan")
    ap.add_argument("--start", type=int, default=0, help="first byte offset to scan")
    ap.add_argument("--end", type=int, help="byte offset to stop at (default: end of file)")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (the range is split evenly)")
    ap.add_argument("--records", help="write one JSON line per fragment here (a local working file)")
    ap.add_argument("--json", help="write the totals to this file")
    a = ap.parse_args(argv)
    if a.jobs < 1 or a.start < 0 or (a.end is not None and a.end < a.start):
        print("error: bad --jobs, --start or --end", file=sys.stderr)
        return 2
    if not os.path.isfile(a.archive):
        print(f"error: {a.archive} is not a file", file=sys.stderr)
        return 2
    for out in (a.records, a.json):
        if not out:
            continue
        if os.path.isdir(out) or ss.same_output_path(out, a.archive):
            print(f"error: {out} is a directory or the input", file=sys.stderr)
            return 2
        parent = os.path.dirname(os.path.abspath(out))
        if not os.path.isdir(parent):
            # checked before the scan: the scan can run for hours and would raise only at write time
            print(f"error: {out} is in a folder that does not exist: {parent}", file=sys.stderr)
            return 2
    t0 = time.time()
    stats = scan_archive(a.archive, a.start, a.end, a.jobs, a.records, lambda text: print(text, flush=True))
    print(f"{os.path.basename(a.archive)}: {time.time() - t0:.0f}s")
    for key, value in stats.items():
        print(f"  {key:26s} {value:10d}")
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(stats, fh, indent=1)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
