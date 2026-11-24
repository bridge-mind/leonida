"""Decode the `0x5000xxxx` descriptor streams of a `*_bvh4.rpf` archive and tie them to their BVH buffers.

docs/bvh.md §3.2 has the layouts this checks. Every descriptor stream starts with a zero u64 and a `0x5000xxxx` word at
0x08 that points at a 32-byte block whose u32 at +8 (the "page word") is 0x101 or 0x1. Two templates exist:

* **instance** (page word 0x101, `0x60000000` at 0x20): u64 N at 0x10, then pointers to N u64 entries (0x18), two
  N x u64 arrays (0x28, 0x30) and N 3x4 float matrices (0x38, always 0x50). The 208-byte `instance208` record is the
  N = 1 case. The next stream is the BVH buffer: the BLAS of the non-zero entries back to back, each starting where
  the previous non-zero entry's low 32 bits end, and each entry's low 32 bits are that BLAS's declared end rounded up
  to 256 bytes. An entry of 0 has no BLAS.
* **table** (page word 0x1): u64 0 at 0x10, u64 1 at 0x18, a pointer to K u32 values (0x20) with `u16 count,
  u16 capacity` at 0x28, and a pointer to K pointers (0x30) with the same count at 0x38. Each pointer addresses a
  32-byte item whose only non-zero bytes are a u64 at +0x10. The u32 values are sorted ascending and distinct.

The tool walks the archive with bvh_index's own decompressor and classifier, parses every descriptor, checks an
instance descriptor against the stream that follows it, and prints counts. It writes no payload bytes, keys or
values: rows hold offsets, sizes, counts and check results.

Usage:
    python -I tools/bvh_groups.py <archive.rpf> [--json report.json] [--start N] [--limit BYTES]

Exit status: 0 after a scan (whatever it found), 2 for a bad argument or an unreadable archive.
"""
import argparse
import collections
import json
import mmap
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_index  # noqa: E402

VBASE = 0x50000000
PBASE = 0x60000000
ITEM = 32
MAP = 32


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def _vptr(b, o):
    """Offset a `0x5000xxxx` word at `o` points to (the high u32 must be 0), or None."""
    if o + 8 > len(b):
        return None
    lo, hi = struct.unpack_from("<II", b, o)
    if hi or lo & 0xFF000000 != VBASE:
        return None
    return lo & 0x00FFFFFF


def _a16(n):
    return (n + 15) & ~15


def _tiles(regions, start, size):
    """True when the (offset, length) regions, each padded to 16 bytes, cover [start, size) without gap or overlap."""
    pos = start
    for off, length in sorted(regions):
        if off != pos:
            return False
        pos = off + _a16(length)
    return pos == size


def round256(n):
    return (n + 255) & ~255


def parse_descriptor(buf):
    """Parse one descriptor stream. Returns a dict, or None when the bytes do not fit either template.

    The dict never carries keys or values from the stream, only the template, counts, sizes and the check results.
    """
    size = len(buf)
    if size < 0x40 or _u64(buf, 0) != 0:
        return None
    m = _vptr(buf, 0x08)
    if m is None or m + MAP > size:
        return None
    page_word = _u32(buf, m + 8)
    if _u32(buf, 0x20) == PBASE and _u32(buf, 0x24) == 0:
        return _parse_instance(buf, m, page_word)
    return _parse_table(buf, m, page_word)


def _parse_instance(buf, m, page_word):
    size = len(buf)
    n = _u64(buf, 0x10)
    ptrs = [_vptr(buf, o) for o in (0x18, 0x28, 0x30, 0x38)]
    if None in ptrs or not 1 <= n <= (size - 0x50) // 48 or size < 0x50:
        return None
    ent, a28, a30, xf = ptrs
    if ent + 8 * n > size or a28 + 8 * n > size or a30 + 8 * n > size or xf + 48 * n > size:
        return None
    entries = [_u64(buf, ent + 8 * i) for i in range(n)]
    sizes = [e & 0xFFFFFFFF for e in entries]
    return {
        "family": "instance", "size": size, "page_word": page_word, "count": n,
        "entry_sizes": sizes, "zero_entries": sizes.count(0),
        "entry_high_low16_mod64": all((e >> 32) & 0xFFFF == 0 or ((e >> 32) & 0xFFFF) % 64 == 0
                                      for e in entries if e & 0xFFFFFFFF),
        "matrix_offset": xf,
        "zero_0x48": _u64(buf, 0x48) == 0,
        "arrays_zero": all(_u64(buf, a + 8 * i) == 0 for a in (a28, a30) for i in range(n)),
        "tiles": _tiles([(xf, 48 * n), (ent, 8 * n), (a28, 8 * n), (a30, 8 * n), (m, MAP)], 0x50, size),
    }


def _parse_table(buf, m, page_word):
    size = len(buf)
    if _u64(buf, 0x10) != 0 or _u64(buf, 0x18) != 1:
        return None
    keys_at, items_at = _vptr(buf, 0x20), _vptr(buf, 0x30)
    if keys_at is None or items_at is None or _u32(buf, 0x2C) or _u32(buf, 0x3C):
        return None
    k, cap = _u32(buf, 0x28) & 0xFFFF, _u32(buf, 0x28) >> 16
    k2, cap2 = _u32(buf, 0x38) & 0xFFFF, _u32(buf, 0x38) >> 16
    if k == 0 or keys_at + 4 * k > size or items_at + 8 * k > size:
        return None
    values = [_u32(buf, keys_at + 4 * i) for i in range(k)]
    items = [_vptr(buf, items_at + 8 * i) for i in range(k)]
    items_ok = None not in items and all(p + ITEM <= size for p in items)
    item_words_ok = items_ok and all(
        _u64(buf, p) == 0 and _u64(buf, p + 8) == 0 and _u64(buf, p + 0x18) == 0 and _u64(buf, p + 0x10) != 0
        for p in items)
    contiguous = items_ok and all(items[i + 1] - items[i] == ITEM for i in range(k - 1))
    regions = [(m, MAP), (keys_at, 4 * k), (items_at, 8 * k)]
    if contiguous:
        regions.append((items[0], ITEM * k))
    return {
        "family": "table", "size": size, "page_word": page_word, "count": k,
        "count_fields_equal": (k, cap) == (k2, cap2) and k == cap,
        "values_sorted_distinct": all(values[i] < values[i + 1] for i in range(k - 1)),
        "items_contiguous": bool(contiguous), "item_words": bool(item_words_ok),
        "tiles": bool(contiguous) and _tiles(regions, 0x40, size),
    }


def check_buffer(desc, payload):
    """Check an instance descriptor against the stream that follows it.

    `payload` is the following stream (only its first 0x140 bytes are needed when the descriptor has one non-zero
    entry). Returns {"ok": bool, "blas": [(offset, declared end), ...], "reason": str}.
    """
    blas, off = [], 0
    nonzero = [s for s in desc["entry_sizes"] if s]
    for i, s in enumerate(nonzero):
        if off + bvh_index.HEAD > len(payload):
            return {"ok": False, "blas": blas, "reason": "BLAS %d past the end of the kept bytes" % i}
        head = payload[off:off + bvh_index.HEAD]
        tag, n = struct.unpack_from("<II", head, bvh_index.TAG_OFF)
        end, tail = _u32(head, bvh_index.END_OFF), _u32(head, bvh_index.TAIL_OFF)
        if tag != bvh_index.TAG or end != 0x140 + 64 * n or tail != end - 0x100:
            return {"ok": False, "blas": blas, "reason": "no BLAS header at offset %d" % off}
        if s != round256(end):
            return {"ok": False, "blas": blas, "reason": "entry %d is not the BLAS end rounded to 256" % i}
        blas.append((off, end))
        off += s
    return {"ok": bool(nonzero), "blas": blas, "reason": "" if nonzero else "no non-zero entry"}


def scan(path, start=None, limit=None, max_inflate=1 << 28):
    """Walk an RPF9 archive and return (rows, summary)."""
    rows = []
    with open(path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size == 0:
            return rows, summarize(rows)
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            pos = start if start is not None else (_u32(m, 4) if size >= 8 else size)
            if not 0 <= pos <= size:
                raise ValueError("start offset is outside the archive")
            end = min(size, pos + limit) if limit else size
            pending = None   # the previous stream's descriptor row, waiting for its following stream
            while pos < end:
                mt = bvh_index.ZHDR.search(m, pos, end)
                if not mt:
                    break
                s = mt.start()
                got = bvh_index.inflate_stream(m, s, end, max_inflate)
                if got is None:
                    pos = s + 1
                    continue
                head, kept, total, used = got
                kind = bvh_index.classify(head, total)["kind"]
                if pending is not None:
                    pending["next_kind"] = kind
                    if pending["family"] == "instance":
                        if kind in ("bvh4", "bvh4-chunk"):
                            res = check_buffer(pending, kept)
                            pending["buffer_ok"], pending["blas_count"] = res["ok"], len(res["blas"])
                            if not res["ok"]:
                                pending["reason"] = res["reason"]
                        else:
                            pending["buffer_ok"] = False
                            pending["reason"] = "followed by %s" % kind
                    pending = None
                row = {"offset": s, "size": total, "kind": kind}
                if kind in ("instance208", "descriptor"):
                    desc = parse_descriptor(bytes(kept)) if total == len(kept) else None
                    if desc is None:
                        row["family"] = "unparsed"
                    else:
                        row.update(desc)
                        pending = row
                rows.append(row)
                pos = s + used
            if pending is not None:
                pending["next_kind"] = None
                if pending["family"] == "instance":
                    pending["buffer_ok"] = False
                    pending["reason"] = "last stream of the scan"
        finally:
            m.close()
    return rows, summarize(rows)


def summarize(rows):
    c = collections.Counter()
    sizes = collections.defaultdict(collections.Counter)
    counts = collections.defaultdict(collections.Counter)
    bvh_after = collections.Counter()
    for i, r in enumerate(rows):
        if r["kind"] in ("bvh4", "bvh4-chunk") and i:
            p = rows[i - 1]
            bvh_after[p.get("family", p["kind"]) + ("/" + p["kind"] if "family" in p else "")] += 1
        fam = r.get("family")
        if fam is None:
            continue
        key = "%s/%s" % (fam, r["kind"])
        c[key] += 1
        sizes[key][r["size"]] += 1
        if fam == "unparsed":
            continue
        counts[key][r["count"]] += 1
        c["%s page_word=0x%x" % (key, r["page_word"])] += 1
        c["%s next=%s" % (key, r.get("next_kind"))] += 1
        c["%s tiles=%s" % (key, r["tiles"])] += 1
        if fam == "instance":
            c["%s buffer_ok=%s" % (key, r.get("buffer_ok"))] += 1
            c["%s arrays_zero=%s" % (key, r["arrays_zero"])] += 1
            c["%s zero_0x48=%s" % (key, r["zero_0x48"])] += 1
            c["%s entry_high_low16_mod64=%s" % (key, r["entry_high_low16_mod64"])] += 1
            c["%s matrix_offset=0x%x" % (key, r["matrix_offset"])] += 1
            if r["zero_entries"]:
                c["%s with zero entries" % key] += 1
                c["%s zero entries" % key] += r["zero_entries"]
            c["%s blas tied" % key] += r.get("blas_count", 0) if r.get("buffer_ok") else 0
        else:
            for f in ("count_fields_equal", "values_sorted_distinct", "items_contiguous", "item_words"):
                c["%s %s=%s" % (key, f, r[f])] += 1
            c["%s items" % key] += r["count"]
    return {"streams": len(rows), "counts": dict(sorted(c.items())),
            "bvh_preceded_by": dict(sorted(bvh_after.items())),
            "sizes": {k: dict(sorted(v.items())) for k, v in sorted(sizes.items())},
            "entry_or_item_counts": {k: dict(sorted(v.items())) for k, v in sorted(counts.items())}}


def _nonneg(text):
    v = int(text, 0)
    if v < 0:
        raise argparse.ArgumentTypeError("must be 0 or more")
    return v


def _positive(text):
    v = int(text, 0)
    if v < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("--json", help="write rows and summary here (offsets, sizes, counts and check results only)")
    ap.add_argument("--start", type=_nonneg)
    ap.add_argument("--limit", type=_positive)
    a = ap.parse_args(argv)
    if a.json and os.path.exists(a.json) and os.path.exists(a.archive) and os.path.samefile(a.json, a.archive):
        print("bvh_groups: --json would replace the archive", file=sys.stderr)
        return 2
    try:
        rows, summary = scan(a.archive, a.start, a.limit)
    except (OSError, ValueError) as e:
        print("bvh_groups: %s" % e, file=sys.stderr)
        return 2
    for k, v in summary["counts"].items():
        print("%8d  %s" % (v, k))
    for k, v in summary["bvh_preceded_by"].items():
        print("%8d  bvh stream preceded by %s" % (v, k))
    if a.json:
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"source": a.archive, "summary": summary, "rows": rows}, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
