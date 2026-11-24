"""Check the dehydrated BVH4 packing of docs/bvh.md §4a on every stream of a pack, without geometry.

`tools/bvh_rehydrate.py` reads a dehydrated BLAS with the packing recorded in docs/bvh.md §4a, and it matches
`XboxBvhBuilder.dll` byte for byte on 8 samples. This tool tests the same reading on every stream of a pack. A
stream passes when its bytes form exactly one tree under that reading:

* header: the dehydrated flag is set, and the leaf records and the box groups fit in the declared structure;
* level table: the last word is the level count; before it, the level ends are in order, a multiple of the node
  size apart, and end before the table;
* groups: every group's node words, child pointers and stored values fit in the bytes its nodes occupy;
* nodes: 1 to 4 children, and no override bit for a slot past the child count;
* child pointers: a box pointer names a box node of its own width (BOX16 or BOX32) at a lower level; a TRI0 pointer
  names a leaf slot, `offsetToElements + 64 * k` with `k < numElements`;
* tree: the root is a box node and nothing points to it, every other box node has exactly one parent, and every
  leaf is named by exactly one pointer;
* leaves: every vertex-order byte puts the three vertices in three different slots;
* stored values: each override lies inside `sceneBounds` mapped through the leaf frame (BOX16 widened to the half
  grid; an infinite BOX16 value only where the bound is outside the half range). Where every slot of the root stores
  a component, the root's union of that component must equal the bound.

None of this reads geometry, so it runs on the whole pack. Output holds offsets and counts only.

Standard library only.

Usage:
    python -I tools/bvh_packing_check.py <archive.rpf> [--workers N] [--window BYTES] [--start N] [--limit BYTES]
                                         [--json report.json]
    python -I tools/bvh_packing_check.py <stream.bin> [<stream.bin> ...] [--header-offset N] [--json report.json]

An .rpf is scanned the way tools/bvh_index.py scans it (every zlib stream that inflates to its end), split into
windows that run in parallel; a stream belongs to the window it starts in. A `bvh4-chunk` is joined with the streams
after it up to its declared end (docs/bvh.md §6). A .bin is one stream; --header-offset is where its header is (256
for a whole `bvh4` payload, 0 for a buffer that starts at the header).

Exit status 0 when every checked stream passes, 1 when one fails, 2 on an input error.
"""
import argparse
import json
import math
import mmap
import multiprocessing
import os
import struct
import sys
import time
import zlib
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_index as I  # noqa: E402
import bvh_rehydrate as R  # noqa: E402
from report_paths import same_report_path  # noqa: E402

BOX_TYPES = (R.BOX16, R.BOX32)
HALF_MAX = 65504.0
FLOAT_TOL = 4 * 2.0 ** -23          # relative, as tools/bvh_geometry_check.py allows for stored overrides
HALF_TOL = 2.0 ** -10               # one half ulp, relative
EXAMPLES = 20                       # offsets kept per failure reason


def _valid_order_table():
    table = bytearray(256)
    for order in range(256):
        a, b = order & 3, (order >> 2) & 3
        c = (a ^ 3) - b
        table[order] = 1 if sorted((a, b, c)) == [0, 1, 2] else 0
    return bytes(table)


VALID_ORDER = _valid_order_table()


def mapped_bounds(hdr):
    """sceneBounds mapped through the leaf frame, as tools/bvh_geometry_check.py maps it: (lo xyz, hi xyz)."""
    frame = R.LeafFrame(hdr)
    lo, hi = hdr.bounds[:3], hdr.bounds[3:]
    corners = [frame.apply((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    return [min(c[a] for c in corners) for a in range(3)] + [max(c[a] for c in corners) for a in range(3)]


def _inside(value, comp, box, wide):
    """True when a stored override lies inside the mapped bounds on its axis."""
    lo, hi = box[comp % 3], box[comp % 3 + 3]
    if value != value:
        return False
    if math.isinf(value):
        if wide:
            return False
        return (value < 0 and lo < -HALF_MAX) or (value > 0 and hi > HALF_MAX)
    tol = (FLOAT_TOL if wide else HALF_TOL) * max(abs(lo), abs(hi), 1.0)
    if not wide:
        tol += 2.0 ** -14
    return lo - tol <= value <= hi + tol


def _root_equal(value, bound, comp, wide):
    if wide:
        return abs(value - bound) <= FLOAT_TOL * max(abs(bound), 1e-30)
    want = R.half_down(bound) if comp < 3 else R.half_up(bound)
    return R.float_to_half(value) == want


class Result:
    """Per-stream outcome: failure reasons (counted once each) and summed statistics."""

    def __init__(self):
        self.fails = Counter()
        self.stats = Counter()

    def fail(self, reason, n=1):
        self.fails[reason] += n


def check_stream(buf, at=0):
    """Check one dehydrated BVH4 structure whose 64-byte header starts at buf[at]. Returns a Result."""
    res = Result()
    st = res.stats
    hdr = R.Header(buf, at)
    total = hdr.total_bytes
    src = bytes(buf[at:at + total])
    if len(src) < total:
        res.fail("short_input")
        return res
    if not hdr.flags & R.FLAG_DEHYDRATED:
        res.fail("not_dehydrated")
        return res
    n = hdr.num_elements
    elements = hdr.offset_to_elements
    nodes = hdr.offset_to_nodes
    st["leaves"] += n

    # Leaf records: chunks of 2,048 every 128 KiB; in the rehydrated layout leaf k sits at elements + 64 k.
    if elements + 64 * n > total:
        res.fail("leaves_past_end")
        return res
    if n and elements + 64 * n > nodes:
        res.fail("leaves_overlap_nodes")
    remaining, chunk = n, 0
    while remaining > 0:
        c = min(R.LEAVES_PER_CHUNK, remaining)
        base = elements + chunk * R.CHUNK_STRIDE
        orders = src[base + 10 * c:base + 11 * c]
        bad = orders.translate(VALID_ORDER).count(0)
        if bad:
            res.fail("leaf_vertex_order", bad)
        remaining -= c
        chunk += 1

    if hdr.root == R.INVALID:
        st["root_invalid"] += 1
        if n:
            res.fail("invalid_root_with_leaves")
        return res
    root_type = hdr.root & 7
    st["root_type_%d" % root_type] += 1
    if root_type not in BOX_TYPES:
        # A structure whose root is a single leaf has no box groups or level table to check.
        if root_type == R.TRI0 and n == 1 and ((hdr.root >> 3) << 6) == elements:
            st["root_leaf"] += 1
        else:
            res.fail("root_not_a_box")
        return res

    try:
        lv = R.levels(src, hdr)
    except (ValueError, struct.error):
        res.fail("level_table")
        return res
    table_start = total - 4 - 8 * len(lv)
    st["levels"] += len(lv)
    st["levels_%d" % len(lv)] += 1
    if lv and nodes + lv[-1][2] + 64 * lv[-1][3] > table_start:
        res.fail("levels_past_table")
        return res

    # Every box node: pointer (unit << 3 | type) -> (node id, level).
    node_of = {}
    runs = []          # (level, wide, start offset from header, count, first id)
    nid = 0
    for level, (b32_start, b32_count, b16_start, b16_count) in enumerate(lv):
        for start, count, wide in ((b32_start, b32_count, True), (b16_start, b16_count, False)):
            size = 128 if wide else 64
            typ = R.BOX32 if wide else R.BOX16
            first = nodes + start
            for j in range(count):
                node_of[((first + j * size) >> 6) << 3 | typ] = (nid + j, level)
            runs.append((level, wide, first, count, nid))
            nid += count
    st["box_nodes"] += nid
    if hdr.root not in node_of:
        res.fail("root_not_a_node")
        return res
    root_id, root_level = node_of[hdr.root]
    if root_level == len(lv) - 1:
        st["root_in_top_level"] += 1
    if root_id == nid - 1:
        st["root_is_last_node"] += 1
    top = lv[-1]
    if top[1] + top[3] == 1:
        st["top_level_single_node"] += 1

    box_refs = bytearray(nid)
    leaf_refs = bytearray(n)
    box = mapped_bounds(hdr)
    root_comps = None
    for level, wide, first, count, first_id in runs:
        size = 128 if wide else 64
        group_bytes = size * R.GROUP
        for g in range((count + R.GROUP - 1) // R.GROUP):
            k = min(R.GROUP, count - g * R.GROUP)
            base = first + g * group_bytes
            words = struct.unpack_from("<%dI" % k, src, base)
            kids_total = sum(w >> 24 for w in words)
            n_values = sum(bin(w & 0xFFFFFF).count("1") for w in words)
            used = 4 * (k + kids_total) + (4 if wide else 2) * n_values
            if used > k * size:
                res.fail("group_overflow")
                continue
            tail = src[base + used:base + k * size]
            if tail.count(0) != len(tail):
                st["group_tail_nonzero_bytes"] += len(tail) - tail.count(0)
            children = struct.unpack_from("<%dI" % kids_total, src, base + 4 * k)
            values = struct.unpack_from(("<%df" if wide else "<%dH") % n_values, src, base + 4 * (k + kids_total))
            if not wide:
                values = [R.half_to_float(h) for h in values]
            ci = vi = 0
            for t, word in enumerate(words):
                kids, mask = word >> 24, word & 0xFFFFFF
                this_id = first_id + g * R.GROUP + t
                st["children_%d" % kids] += 1
                if kids == 0:
                    res.fail("no_children")
                elif kids > 4:
                    res.fail("too_many_children")
                if mask >> (6 * min(kids, 4)):
                    res.fail("mask_past_children")
                for ptr in children[ci:ci + kids]:
                    typ = ptr & 7
                    st["child_type_%d" % typ] += 1
                    if ptr == R.INVALID:
                        res.fail("invalid_child")
                    elif typ in BOX_TYPES:
                        hit = node_of.get(ptr)
                        if hit is None:
                            other = ptr ^ (R.BOX16 ^ R.BOX32)
                            res.fail("child_wrong_width" if other in node_of else "child_not_a_node")
                        else:
                            if hit[1] >= level:
                                res.fail("child_not_below")
                            if box_refs[hit[0]] < 255:
                                box_refs[hit[0]] += 1
                    elif typ == R.TRI0:
                        off = ((ptr >> 3) << 6) - elements
                        if off < 0 or off % 64 or off // 64 >= n:
                            res.fail("child_not_a_leaf")
                        elif leaf_refs[off // 64] < 255:
                            leaf_refs[off // 64] += 1
                    else:
                        res.fail("child_type_%d" % typ)
                ci += kids
                comps = {} if this_id == root_id else None
                for bit in range(24):
                    if mask >> bit & 1:
                        v = values[vi]
                        vi += 1
                        st["overrides"] += 1
                        if not _inside(v, bit % 6, box, wide):
                            res.fail("override_outside_bounds")
                        if comps is not None and bit // 6 < kids:
                            comps.setdefault(bit % 6, []).append((bit // 6, v))
                if comps is not None:
                    root_comps = (kids, comps, wide)

    if root_comps is not None:
        kids, comps, wide = root_comps
        for comp in range(6):
            got = comps.get(comp, [])
            if len(got) == kids and kids:
                vals = [v for _, v in got]
                union = min(vals) if comp < 3 else max(vals)
                st["root_components_stored"] += 1
                if _root_equal(union, box[comp], comp, wide):
                    st["root_components_equal"] += 1
                else:
                    res.fail("root_union_not_bounds")

    if box_refs[root_id]:
        res.fail("root_referenced")
    unreferenced = sum(1 for i, r in enumerate(box_refs) if r == 0 and i != root_id)
    shared = sum(1 for r in box_refs if r > 1)
    if unreferenced:
        res.fail("node_unreferenced", unreferenced)
    if shared:
        res.fail("node_shared", shared)
    leaf_missing = leaf_refs.count(0)
    leaf_shared = n - leaf_missing - leaf_refs.count(1)
    if leaf_missing:
        res.fail("leaf_unreferenced", leaf_missing)
    if leaf_shared:
        res.fail("leaf_shared", leaf_shared)
    return res


# --- archive scan -----------------------------------------------------------------------------------------------

def _inflate(m, s, end):
    """Inflate the zlib stream at s (reading at most up to end). Returns (data, used) or None."""
    d = zlib.decompressobj()
    out, p = [], s
    try:
        while not d.eof and p < end:
            chunk = m[p:min(p + (1 << 20), end)]
            if not chunk:
                break
            out.append(d.decompress(chunk))
            p += len(chunk)
            if d.unused_data:
                break
        out.append(d.flush())
    except zlib.error:
        return None
    if not d.eof:
        return None
    data = b"".join(out)
    if len(data) < 64:
        return None
    return data, p - s - len(d.unused_data)


class Tally:
    def __init__(self):
        self.kinds = Counter()
        self.checked = self.passed = 0
        self.fail_streams = Counter()      # reason -> streams
        self.fail_items = Counter()        # reason -> occurrences
        self.stats = Counter()
        self.examples = {}
        self.raw = set()            # offsets of streams no classifier recognised
        self.consumed = set()       # offsets of continuation streams joined to a bvh4-chunk

    def add(self, offset, res):
        self.checked += 1
        self.stats.update(res.stats)
        if not res.fails:
            self.passed += 1
        for reason, k in res.fails.items():
            self.fail_streams[reason] += 1
            self.fail_items[reason] += k
            ex = self.examples.setdefault(reason, [])
            if len(ex) < EXAMPLES:
                ex.append(offset)

    def merge(self, other):
        self.kinds.update(other.kinds)
        self.checked += other.checked
        self.passed += other.passed
        self.fail_streams.update(other.fail_streams)
        self.fail_items.update(other.fail_items)
        self.stats.update(other.stats)
        self.raw |= other.raw
        self.consumed |= other.consumed
        for reason, ex in other.examples.items():
            mine = self.examples.setdefault(reason, [])
            mine.extend(ex)
            mine.sort()
            del mine[EXAMPLES:]

    def report(self):
        # A continuation that starts in a later window is seen there as an unrecognised stream; count it once.
        kinds = Counter(self.kinds)
        raw = len(self.raw - self.consumed)
        if raw:
            kinds["raw"] = raw
        return {"kinds": dict(sorted(kinds.items())), "checked": self.checked, "passed": self.passed,
                "failed": self.checked - self.passed,
                "fail_streams": dict(sorted(self.fail_streams.items())),
                "fail_items": dict(sorted(self.fail_items.items())),
                "stats": dict(sorted(self.stats.items())),
                "fail_examples": {k: sorted(v) for k, v in sorted(self.examples.items())}}


def scan_window(args):
    """Check every stream that starts in [start, end) of the archive; reads past end to finish a stream."""
    path, start, end = args
    tally = Tally()
    with open(path, "rb") as fh:
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            pos = start
            while pos < end:
                # ZHDR is 2 bytes; end + 1 lets a header starting at end - 1 match without entering [end, ...).
                mt = I.ZHDR.search(m, pos, min(end + 1, size))
                if not mt:
                    break
                s = mt.start()
                got = _inflate(m, s, size)
                if got is None:
                    pos = s + 1
                    continue
                data, used = got
                pos = s + used
                row = I.classify(data[:I.HEAD], len(data))
                kind = row["kind"]
                if kind == "raw":
                    tally.raw.add(s)
                else:
                    tally.kinds[kind] += 1
                if kind == "bvh4-chunk":
                    # Join the continuation streams that follow, up to the declared end.
                    parts, have, q = [data], len(data), pos
                    while have < row["declared_bytes"]:
                        nxt = I.ZHDR.search(m, q, size)
                        more = _inflate(m, nxt.start(), size) if nxt else None
                        if more is None:
                            break
                        parts.append(more[0])
                        have += len(more[0])
                        tally.kinds["continuation"] += 1
                        tally.consumed.add(nxt.start())
                        q = nxt.start() + more[1]
                    data = b"".join(parts)
                    if have < row["declared_bytes"]:
                        tally.add(s, _short())
                        continue
                    tally.kinds["bvh4-chunk-joined"] += 1
                    pos = q
                    kind = "bvh4"
                if kind == "bvh4":
                    try:
                        res = check_stream(data, I.TAG_OFF)
                    except (ValueError, struct.error) as e:
                        res = Result()
                        res.fail("error:" + type(e).__name__)
                    tally.add(s, res)
        finally:
            m.close()
    return tally


def _short():
    r = Result()
    r.fail("short_input")
    return r


def scan_archive(path, workers, window, start=None, limit=None, progress=True):
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        first = struct.unpack("<I", fh.read(8)[4:8])[0] if size >= 8 else size
    lo = first if start is None else start
    if not 0 <= lo <= size:
        raise ValueError("start offset is outside the archive")
    hi = min(size, lo + limit) if limit else size
    jobs = [(path, a, min(a + window, hi)) for a in range(lo, hi, window)]
    total = Tally()
    t0 = time.time()
    if workers <= 1 or len(jobs) <= 1:
        results = map(scan_window, jobs)
        pool = None
    else:
        pool = multiprocessing.Pool(workers, initializer=I.below_normal)
        results = pool.imap_unordered(scan_window, jobs)
    try:
        for done, t in enumerate(results, 1):
            total.merge(t)
            if progress:
                print("  %d/%d windows, %d streams checked, %d failed (%.0fs)"
                      % (done, len(jobs), total.checked, total.checked - total.passed, time.time() - t0),
                      file=sys.stderr, flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return total, round(time.time() - t0, 1)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check the dehydrated BVH4 packing (docs/bvh.md §4a) without geometry.")
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--window", type=int, default=64 << 20, help="archive bytes per parallel job (default 64 MiB)")
    ap.add_argument("--start", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--header-offset", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    if a.header_offset < 0:
        ap.error("--header-offset must be non-negative")
    if a.header_offset and any(p.lower().endswith(".rpf") for p in a.inputs):
        ap.error("--header-offset only applies to a stream file")
    bad_limit = a.limit is not None and a.limit <= 0
    if a.window <= 0 or a.workers <= 0 or bad_limit or (a.start is not None and a.start < 0):
        print("error: --window, --workers and --limit must be positive, --start not negative", file=sys.stderr)
        return 2
    if a.json:
        if os.path.isdir(a.json):
            print("error: --json is a directory: %s" % a.json, file=sys.stderr)
            return 2
        parent = os.path.dirname(os.path.abspath(a.json))
        if not os.path.isdir(parent):
            print("error: --json parent folder does not exist: %s" % parent, file=sys.stderr)
            return 2
        for p in a.inputs:
            if same_report_path(a.json, p):
                print("error: --json must not overwrite the input %s" % p, file=sys.stderr)
                return 2
    rpfs = [p for p in a.inputs if p.lower().endswith(".rpf")]
    if rpfs and len(a.inputs) > 1:
        print("error: give one .rpf, or one or more stream files", file=sys.stderr)
        return 2
    try:
        if rpfs:
            tally, elapsed = scan_archive(rpfs[0], a.workers, a.window, a.start, a.limit)
        else:
            if a.start is not None or a.limit is not None:
                print("error: --start and --limit apply to an .rpf only", file=sys.stderr)
                return 2
            t0 = time.time()
            tally = Tally()
            for p in a.inputs:
                with open(p, "rb") as f:
                    data = f.read()
                tally.kinds["file"] += 1
                tally.add(p, check_stream(data, a.header_offset))
            elapsed = round(time.time() - t0, 1)
    except (OSError, ValueError, struct.error) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    rep = tally.report()
    rep["elapsed_s"] = elapsed
    print("checked %d streams: %d passed, %d failed (%.1fs)" % (rep["checked"], rep["passed"], rep["failed"], elapsed))
    for reason, k in rep["fail_streams"].items():
        print("  %s: %d streams, %d items" % (reason, k, rep["fail_items"][reason]))
    if a.json:
        try:
            I._publish(a.json, lambda f: json.dump(rep, f, indent=1), "\n")
        except OSError as e:
            print("error: cannot write %s: %s" % (a.json, e.strerror or e), file=sys.stderr)
            return 2
    return 0 if rep["failed"] == 0 else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
