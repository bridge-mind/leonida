"""Decode the resource classes in a navmesh archive (``navmeshes.rpf``) without the encrypted TOC.

The archive's data area is a run of plain zlib streams. Three resource classes start a stream there. All three
begin with the same 16 bytes: eight zero bytes and a ``0x50000000``-based pointer to a block map whose u64 at +8
is the number of zlib streams the resource occupies (1 or 2 on the measured archive).

``0DRP``
    A self-describing ``0x7a`` resource (``0DRP`` at +0x10, version 122): struct, enum and instance-block tables
    as decoded for ``levels.rpf`` in ``docs/world.md``. In this archive the schemas are cover-point data, not
    navmesh geometry. The eight u16 words at +0x48 are the struct, enum and block counts, a 4 KiB page count
    (0 or ``ceil(data_end / 4096)``), and four zero words. +0x60 is the start of the first instance block, not a
    header field.
``0x00010015``
    The navmesh tile: a fixed 0xF0-byte header (``u32 0x00010015`` at +0x14) with vertex, polygon and
    special-link counts, four 16 KiB-blocked split arrays, a quadtree root and the tile's 512 x 512 sector id.
``grid``
    A 0x60-byte header followed by a 16-slot open-addressing hash table (272-byte groups) and a ``u32`` array.
    It pairs one-to-one with the static navmesh tiles by sector id. Its subsystem is not identified.

``PSIN`` containers are counted and section-walked. Everything else is reported as ``other``.

A resource continues into the next zlib stream when its block map says so. Some continuations inflate to fewer
than 64 bytes; the carver walk skips those, so this tool also accepts them, but only as continuations. Stream
counts in the summary follow the carver rule (64 bytes or more) so they can be compared with
``tools/navmesh_index.py``.

The tool reads headers, tables and counts. It writes aggregate metadata only: no payload bytes, names or keys.
It never touches the encrypted table of contents.

Usage:
    python -I navmesh_decode.py <archive.rpf|dir> [--out FILE] [--csv FILE]
                                [--start N] [--limit BYTES] [--deep] [--quiet]

    --out     JSON summary (default: <input name>.navmesh-decode.json in the current directory)
    --csv     optional CSV, one row of numeric header fields per resource
    --start   first byte to scan in an archive (default: the archive's data start, u32 at offset 4)
    --limit   bytes to scan from --start; a resource that starts inside the window is read to its end, but a
              stream past the window's end is reported under past_window, not in the carver counts
    --deep    also walk every navmesh polygon, quadtree node and grid hash-table slot (slower)
    --quiet   do not print the summary
"""
import argparse
import csv
import json
import math
import mmap
import os
import re
import struct
import sys
import time
import unicodedata
import zlib
from collections import Counter
from collections.abc import Iterator

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from zero_span import gap_split  # noqa: E402

BASE = 0x50000000
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")
HEX_PREFIX = re.compile(r"^(?:0[xX])?([0-9a-fA-F]{6,})(?:[^0-9a-zA-Z]|$)")
CHUNK = 1 << 20
MIN_STREAM = 64          # the carver's threshold; shorter streams are only accepted as continuations
MAX_STREAMS = 16
MAX_INFLATE = 256 << 20  # refuse to inflate a single stream or assembled resource past this size
MAX_QT_DEPTH = 32
MAX_QT_NODES = 1 << 20
VERSION_7A = 122
DATA15 = 0x00010015
HEADER_7A = 0x60
HEADER_NAV = 0xF0
GRID_GROUP = 272         # 16 control bytes + 16 slots of {u64 key, u64 value}
SPLIT_BLOCK = 16384      # byte budget of one split-array block
NODE_SIZE = 160          # CCoverPointNode
REGION_SIZE = 304        # CCoverPointRegion

# Case-sensitive joaat. Every name below was typed by hand and kept only because its hash equals a struct, enum,
# member or enum-value hash in the archive's own 0x7a schema tables. Nothing is read from the game to build it.
NAMES = (
    "CCoverPointRegion", "CCoverPointNode", "CCoverNodeId", "CCoverPointUsageInfo", "CCoverSpatialTree",
    "CCoverSpatialTreeNode", "CCoverPointRegionManifest", "CCoverPointRegionDef", "rage__spdAABB",
    "eCoverFlags", "eCoverHeight", "eCoverUsage", "eCoverRegionFlags", "Uid", "SpatialTreeNodes", "Entries",
    "RegionIndex", "COVHEIGHT_PRONE", "COVUSE_WALLTOLEFT", "COVUSE_WALLTORIGHT", "COVUSE_WALLTOBOTH",
    "COVUSE_WALLTONEITHER",
)
REGION = 0x2283B2CB
NODE = 0x0BCB00E4
DYNAMIC_COVER = 0x4712787B

COLUMNS = [
    "offset", "class", "streams", "stored", "inflated", "complete", "checks_failed",
    "version", "root_index", "n_structs", "n_enums", "n_blocks", "pages_4k", "root_hash", "cover_nodes",
    "flags", "sector", "vertices", "polygons", "pool", "special_links", "volumes", "data_size",
    "groups", "entries", "extent",
]


def joaat_cs(text):
    """Jenkins one-at-a-time over the exact bytes (no lower-casing), as the 0x7a schema hashes use."""
    h = 0
    for c in text.encode("ascii"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


KNOWN = {joaat_cs(n): n for n in NAMES}


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0] if 0 <= o and o + 2 <= len(b) else None


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0] if 0 <= o and o + 4 <= len(b) else None


def _i32(b, o):
    return struct.unpack_from("<i", b, o)[0] if 0 <= o and o + 4 <= len(b) else None


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0] if 0 <= o and o + 8 <= len(b) else None


def _f32s(b, o, n):
    return struct.unpack_from("<%df" % n, b, o) if 0 <= o and o + 4 * n <= len(b) else None


def _zero(b, lo, hi):
    return 0 <= lo <= hi <= len(b) and b[lo:hi].count(0) == hi - lo


def offset_of(value):
    """Decode a pointer. All measured pointers are zero-extended 32-bit values in [0x50000000, 0x60000000)."""
    if value is None or value >> 32 or not BASE <= value < BASE + 0x10000000:
        return None
    return value - BASE


def hx(value):
    return "0x%08x" % value


def label(value):
    return KNOWN.get(value, hx(value))


# ---------------------------------------------------------------------------------------------------- streams

def inflate_at(m, start, size, cap=MAX_INFLATE):
    """Inflate one zlib stream within [start, size). Returns (data, stored_bytes) or None."""
    d = zlib.decompressobj()
    parts, total, pos, pending = [], 0, start, b""
    try:
        while not d.eof:
            if not pending:
                if pos >= size:
                    return None
                pending = m[pos:min(pos + CHUNK, size)]
                pos += len(pending)
            out = d.decompress(pending, cap + 1 - total)
            pending = d.unconsumed_tail
            total += len(out)
            if total > cap:
                return None
            parts.append(out)
    except zlib.error:
        return None
    return b"".join(parts), pos - start - len(d.unused_data) - len(pending)


def walk(m, start, size):
    """Yield (offset, stored, data) for every zlib stream from `start`, advancing by the bytes it consumed."""
    pos = start
    while pos < size:
        found = ZHDR.search(m, pos, size)
        if not found:
            return
        s = found.start()
        got = inflate_at(m, s, size)
        if got is None or got[1] <= 0:
            pos = s + 1
            continue
        data, used = got
        yield s, used, data
        pos = s + used


def carved_files(root: str) -> Iterator[os.DirEntry[str]]:
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.name.lower().endswith((".json", ".csv")) and entry.is_file():
                yield entry


def walk_dir(root):
    """Yield (offset, None, data) from carved files sorted by leading hex offset, then name."""
    entries = []
    for entry in carved_files(root):
        name, full = entry.name, entry.path
        if os.path.getsize(full) > MAX_INFLATE:
            raise ValueError("%s exceeds %d-byte limit" % (full, MAX_INFLATE))
        mt = HEX_PREFIX.match(name)
        off = int(mt.group(1), 16) if mt else None
        entries.append((off, name, full))
    fallback = 0
    for off, _, full in sorted(entries, key=lambda e: (e[0] is None, e[0] or 0, e[1])):
        if off is None:
            off = fallback
        fallback = max(fallback, off + 1)
        with open(full, "rb") as f:
            yield off, None, f.read()


# ------------------------------------------------------------------------------------------------ classifying

def classify(b):
    if len(b) >= 4 and b[:4] == b"PSIN":
        return "psin"
    if len(b) >= HEADER_7A and b[0x10:0x14] == b"0DRP":
        return "0drp"
    if len(b) < 0x60 or not _zero(b, 0, 8) or offset_of(_u64(b, 8)) is None:
        return "other"
    if len(b) >= HEADER_NAV and _u32(b, 0x14) == DATA15:
        return "navmesh"
    groups = _u32(b, 0x10)
    if (_u64(b, 0x18) == BASE + 0x60 and groups and groups & (groups - 1) == 0
            and _u32(b, 0x14) == _u32(b, 0x50)):
        return "grid"
    return "other"


def _psin_state(b):
    """Return (complete, sections, needs_more) for a PSIN container."""
    if len(b) < 16:
        return False, 0, True
    at = struct.unpack_from(">I", b, 4)[0]
    if at < 16:
        return False, 0, False
    count = 0
    while at + 8 <= len(b):
        size = struct.unpack_from(">I", b, at + 4)[0]
        if size < 8:
            return False, count, False
        at += size
        count += 1
    if at == len(b):
        # Ends exactly here: complete with at least one section; a bare header is incomplete, not continued.
        return count > 0, count, False
    return False, count, True


def wants_more(cls, b, nstreams):
    """True when the resource in `b` (made of `nstreams` streams) should consume the next stream."""
    if nstreams >= MAX_STREAMS or len(b) >= MAX_INFLATE:
        return False
    if cls == "psin":
        return _psin_state(b)[2]
    if cls not in ("0drp", "navmesh", "grid"):
        return False
    bm = offset_of(_u64(b, 8))
    if bm is None:
        return False
    if bm + 16 > len(b):
        return True
    declared = _u64(b, bm + 8)
    return declared is not None and nstreams < min(declared, MAX_STREAMS)


def is_complete(cls, b, nstreams):
    """True when the resource in `b` has all streams declared by its header/block map (ignoring MAX_STREAMS)."""
    if cls == "psin":
        return _psin_state(b)[0]
    if cls not in ("0drp", "navmesh", "grid"):
        return True
    bm = offset_of(_u64(b, 8))
    if bm is None or bm + 16 > len(b):
        return False
    declared = _u64(b, bm + 8)
    return declared == nstreams


def block_map_ok(b, nstreams, size=24):
    """Verify the block map at u64@0x08: {u64 0, u64 stream_count, u64 0[, u64 0]}."""
    bm = offset_of(_u64(b, 8))
    if bm is None or bm + size > len(b):
        return False
    if _u64(b, bm) != 0 or _u64(b, bm + 8) != nstreams or _u64(b, bm + 16) != 0:
        return False
    return size <= 24 or _u64(b, bm + 24) == 0


# ----------------------------------------------------------------------------------------------------- 0DRP

def decode_7a(b, len0, nstreams, stats):
    """Decode a 0x7a resource. Returns (row, checks)."""
    chk = {}
    n28, n30, n38, n4e = (_u16(b, 0x48 + 2 * i) for i in range(4))
    version = _u32(b, 0x14)
    root_index = _u32(b, 0x20)
    q28, q30, q38, q40 = (offset_of(_u64(b, o)) for o in (0x28, 0x30, 0x38, 0x40))
    chk["prefix_zero"] = _zero(b, 0, 8)
    chk["version_122"] = version == VERSION_7A
    chk["reserved_zero"] = _zero(b, 0x18, 0x20) and _zero(b, 0x24, 0x28) and _zero(b, 0x50, 0x60)
    chk["block_map"] = block_map_ok(b, nstreams, size=24)
    row = {"version": version, "root_index": root_index, "n_structs": n28, "n_enums": n30, "n_blocks": n38,
           "pages_4k": n4e, "root_hash": None, "cover_nodes": None}

    # Instance blocks: {u32 struct hash, u32 byte size, u64 pointer}.
    blocks = []
    ok = q38 is not None and q38 + 16 * n38 <= len(b)
    if ok:
        for i in range(n38):
            h, size, p = struct.unpack_from("<IIQ", b, q38 + 16 * i)
            t = offset_of(p)
            if t is None or t + size > len(b):
                ok = False
                blocks = []
                break
            blocks.append((h, size, t))
    chk["blocks_in_bounds"] = ok
    spans = sorted((t, t + size) for _, size, t in blocks)
    chk["blocks_disjoint"] = ok and all(a[1] <= c[0] for a, c in zip(spans, spans[1:]))
    chk["payload_at_0x60"] = ok and any(t == HEADER_7A for _, _, t in blocks)
    chk["root_index_valid"] = ok and root_index is not None and 1 <= root_index <= len(blocks)
    root = blocks[root_index - 1] if chk["root_index_valid"] else None

    # Struct table: 32-byte records; member table: 16-byte records.
    structs = {}
    ok = q28 is not None and q28 + 32 * n28 <= len(b)
    if ok:
        for i in range(n28):
            h, _, _, mp, size, _, count = struct.unpack_from("<IIQQIHH", b, q28 + 32 * i)
            mt = offset_of(mp) if count else 0
            if mt is None or mt + 16 * count > len(b):
                ok = False
                structs = {}
                break
            structs[h] = (size, count, mt)
    chk["structs_in_bounds"] = ok
    chk["root_is_struct"] = root is not None and root[0] in structs

    # Enum table: 24-byte records {u32 hash, u32 signature, u64 -> values, u64 count}; values are 16 bytes.
    ok = n30 == 0 or (q30 is not None and q30 + 24 * n30 <= len(b))
    dense = ok
    checked_enums = {}
    if ok:
        for i in range(n30):
            h, _, vp, count = struct.unpack_from("<IIQQ", b, q30 + 24 * i)
            vt = offset_of(vp) if count else 0
            if vt is None or count > 0x10000 or vt + 16 * count > len(b):
                ok = dense = False
                break
            key = (vt, count)
            if key not in checked_enums:
                values = [struct.unpack_from("<q", b, vt + 16 * k)[0] for k in range(count)]
                checked_enums[key] = values == list(range(count))
            dense = dense and checked_enums[key]
            stats["enums"][label(h)] += 1
    chk["enums_in_bounds"] = ok
    chk["enum_values_dense"] = dense

    # q40: u32 count, then count x {u32 tag, u32 size}, then the payloads.
    if q40 is None:
        chk["q40_in_bounds"] = False
    else:
        count = _u32(b, q40)
        end = None
        if count is not None and count <= 0x10000 and q40 + 4 + 8 * count <= len(b):
            items = [struct.unpack_from("<II", b, q40 + 4 + 8 * i) for i in range(count)]
            end = q40 + 4 + 8 * count + sum(size for _, size in items)
            stats["q40_shape"][str([(hex(tag), size) for tag, size in items]) if count <= 2 else "count>2"] += 1
        chk["q40_in_bounds"] = end is not None and end <= len(b)

    # u16@0x4e: always 0 or ceil(data_end / 4096). Equals pages whenever pages*4096 < stream_end, and 0 when
    # pages*4096 > stream_end or on single-stream pages hitting the stream end; on two-stream pages where
    # data_end lands in the final 4 KiB of stream 0 (pages*4096 == len0), both 0 and pages occur.
    bm = offset_of(_u64(b, 8))
    if chk["blocks_in_bounds"] and blocks and bm is not None and q38 is not None:
        data_end = max([t + size for _, size, t in blocks] + [q38 + 16 * n38, bm + 24])
        pages = (data_end + 4095) // 4096
        stream_end = len0 if data_end <= len0 else len(b)
        if pages * 4096 < stream_end:
            chk["pages_4k_rule"] = n4e == pages
        elif data_end <= len0 and pages * 4096 == len0 and nstreams > 1:
            chk["pages_4k_rule"] = n4e in (0, pages)
            if n4e == pages:
                stats["pages_4k_boundary"]["nonzero"] += 1
            elif n4e == 0:
                stats["pages_4k_boundary"]["zero"] += 1
        else:
            chk["pages_4k_rule"] = n4e == 0
    else:
        chk["pages_4k_rule"] = False

    sig = tuple(sorted(structs))
    stats["schemas"][(label(root[0]) if root else "none", len(sig))] += 1
    stats["root"][label(root[0]) if root else "none"] += 1
    first = next((h for h, _, t in blocks if t == HEADER_7A), None)
    stats["block_at_0x60"][label(first) if first is not None else "none"] += 1
    for h, _, _ in blocks:
        stats["block_hash"][label(h)] += 1
    stats["u16"]["n_structs"][n28] += 1
    stats["u16"]["n_enums"][n30] += 1
    stats["u16"]["n_blocks"][n38] += 1
    stats["pages_4k_zero"] += n4e == 0
    stats["root_index"][root_index] += 1
    if root:
        row["root_hash"] = hx(root[0])

    # Cover payload: the node array, and the three 16-byte node ids at the start of every node. The first node
    # usually sits at +0x60, which is what the old "+0x60 key / +0x68 u32 / +0x70 repeats" description saw.
    nodes = [(size, t) for h, size, t in blocks if h == NODE]
    if nodes and structs.get(NODE, (0,))[0] == NODE_SIZE:
        size, t = nodes[0]
        chk["cover_nodes_whole"] = size % NODE_SIZE == 0
        count = size // NODE_SIZE
        row["cover_nodes"] = count
        stats["cover_nodes"] += count
        if root and root[0] == REGION:
            stats["region_cover_nodes"] += count
        elif root and root[0] == DYNAMIC_COVER:
            stats["dynamic_cover_nodes"] += count
        if t == HEADER_7A and count:
            _, _, local0, _, _, local1 = struct.unpack_from("<QIIQII", b, t)
            stats["first_node"]["id0_index_zero"] += local0 == 0
            stats["first_node"]["id1_index_ffffffff"] += local1 == 0xFFFFFFFF
            stats["first_node"]["pages"] += 1
        if root and root[0] == REGION and root[1] == REGION_SIZE:
            arr_count = _u16(b, root[2] + 0xA0)
            points = _i32(b, root[2] + 4)
            chk["region_array_count"] = arr_count == count
            stats["region_points"] += points or 0
            stats["region_inumpoints"]["match" if points == count else "mismatch"] += 1
            lo, hi = _f32s(b, root[2] + 0x10, 3), _f32s(b, root[2] + 0x20, 3)
            _extend(stats["region_bounds"], lo, hi)
    return row, chk


def _extend(box, lo, hi):
    if lo is None or hi is None:
        return
    if not (all(math.isfinite(v) for v in lo) and all(math.isfinite(v) for v in hi)):
        return
    for i in range(3):
        box[0][i] = min(box[0][i], lo[i])
        box[1][i] = max(box[1][i], hi[i])


# --------------------------------------------------------------------------------------------- navmesh tiles

SPLITS = ((0x70, 6, 0xC0), (0x80, 2, 0x90), (0x88, 12, 0x90), (0x98, 44, 0xC4))
NAV_ZERO = ((0x18, 0x20), (0x6C, 0x70), (0x78, 0x80), (0x94, 0x98), (0xD0, 0xD4), (0xE0, 0xE8), (0xEC, 0xF0))
IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def split_array(b, at, stride, total):
    """Validate a split array descriptor. Returns the list of (offset, count) blocks, or None."""
    d = offset_of(_u64(b, at))
    if d is None or d + 48 > len(b):
        return None
    _, declared, bp, sp, nblocks, tail = struct.unpack_from("<QQQQQQ", b, d)
    bp, sp = offset_of(bp), offset_of(sp)
    if declared != total or tail or nblocks > 0x10000 or bp is None or sp is None:
        return None
    if bp + 16 * nblocks > len(b) or sp + 4 * nblocks > len(b):
        return None
    per = SPLIT_BLOCK // stride
    out, acc = [], 0
    for i in range(nblocks):
        p, count = struct.unpack_from("<QQ", b, bp + 16 * i)
        o = offset_of(p)
        if o is None or o + stride * count > len(b) or _u32(b, sp + 4 * i) != acc:
            return None
        if i + 1 < nblocks and count != per:
            return None
        out.append((o, count))
        acc += count
    return out if acc == total else None


def _u16_max(b, o, n):
    if n == 0:
        return -1
    view = memoryview(b)[o:o + 2 * n]
    try:
        if sys.byteorder == "little":
            return max(view.cast("H"))
        return max(v for (v,) in struct.iter_unpack("<H", view))
    finally:
        view.release()


def _walk_quadtree(b, root, n_polys):
    """Validate the 96-byte node / 32-byte leaf quadtree. Returns (ok, nodes, leaves, max_depth)."""
    if root is None:
        return False, 0, 0, 0
    stack = [(root, 0)]
    seen = set()
    seen_leaves = set()
    nodes = leaves = max_depth = 0
    while stack:
        o, depth = stack.pop()
        if o in seen or depth > MAX_QT_DEPTH or nodes >= MAX_QT_NODES or o + 0x60 > len(b):
            return False, nodes, leaves, max_depth
        seen.add(o)
        nodes += 1
        if depth > max_depth:
            max_depth = depth
        if not _zero(b, o + 0x54, o + 0x60):
            return False, nodes, leaves, max_depth
        leaf_raw, c0, c1, c2, c3 = struct.unpack_from("<5Q", b, o + 0x2C)
        if leaf_raw:
            lo = offset_of(leaf_raw)
            if lo is None or lo + 0x20 > len(b):
                return False, nodes, leaves, max_depth
            if lo not in seen_leaves:
                z0, pp, z1, np = struct.unpack_from("<4Q", b, lo)
                if z0 != 0 or z1 != 0 or np > n_polys:
                    return False, nodes, leaves, max_depth
                if np:
                    po = offset_of(pp)
                    if po is None or po + 2 * np > len(b) or _u16_max(b, po, np) >= n_polys:
                        return False, nodes, leaves, max_depth
                seen_leaves.add(lo)
            leaves += 1
        for cp in (c0, c1, c2, c3):
            if cp:
                co = offset_of(cp)
                if co is None:
                    return False, nodes, leaves, max_depth
                stack.append((co, depth + 1))
    return True, nodes, leaves, max_depth


def decode_navmesh(b, nstreams, stats, deep):
    chk = {}
    flags = _u32(b, 0x10)
    c0, c4, c8, cc, d4, d8, dc, e8 = (_u32(b, o) for o in (0xC0, 0xC4, 0xC8, 0xCC, 0xD4, 0xD8, 0xDC, 0xE8))
    pool = _u32(b, 0x90)
    a8, b0, b8 = (_u64(b, o) for o in (0xA8, 0xB0, 0xB8))
    chk["prefix_zero"] = _zero(b, 0, 8)
    chk["block_map"] = block_map_ok(b, nstreams, size=32)
    chk["reserved_zero"] = all(_zero(b, lo, hi) for lo, hi in NAV_ZERO)
    chk["identity_matrix"] = _f32s(b, 0x20, 16) == IDENTITY
    chk["data_size_identity"] = cc == 6 * c0 + (2 + 12) * pool + 44 * c4 + 32 * d4
    chk["flag0_set"] = bool(flags & 1)
    links = bool(a8 and b0 and d4 and d8)
    no_links = not (a8 or b0 or d4 or d8)
    chk["flag1_links"] = links if flags & 2 else no_links
    chk["flag2_no_sector"] = (c8 == 0 and e8 == 0) == bool(flags & 4)
    blocks = []
    for at, stride, count_at in SPLITS:
        got = split_array(b, at, stride, _u32(b, count_at))
        chk["split_%02x" % at] = got is not None
        blocks.append(got)
        if got is not None:
            stats["split_blocks_max"][hex(at)] = max(stats["split_blocks_max"].get(hex(at), 0), len(got))
    if blocks[1] is not None:
        chk["indices_below_vertices"] = all(_u16_max(b, o, n) < c0 for o, n in blocks[1])
    ok = True
    if flags & 2:
        la, lb = offset_of(a8), offset_of(b0)
        ok = (la is not None and lb is not None and la + 32 * d4 <= len(b) and lb + 2 * d8 <= len(b)
              and _u16_max(b, lb, d8) < d4)
    chk["links_in_bounds"] = ok
    vol = offset_of(b8)
    chk["volumes_in_bounds"] = (b8 == 0 and dc == 0) or (vol is not None and vol + 96 * dc <= len(b))

    root = offset_of(_u64(b, 0xA0))
    chk["quadtree_root"] = root is not None and root + 0x60 <= len(b)
    lo = _f32s(b, root, 3) if chk["quadtree_root"] else None
    hi = _f32s(b, root + 0x10, 3) if chk["quadtree_root"] else None
    ext = _f32s(b, 0x60, 3)
    chk["extents_match_root"] = (lo is not None and hi is not None and ext is not None
                                 and all(math.isfinite(v) for v in (*lo, *hi, *ext))
                                 and all(abs((hi[i] - lo[i]) - ext[i]) < 0.02 for i in range(3)))
    if not flags & 4:
        sx, sy = c8 & 0x1FF, c8 >> 9
        x0, y0 = (sx - 256) * 128.0, (sy - 256) * 128.0
        chk["sector_origin"] = (lo is not None and hi is not None and (lo[0], lo[1]) == (x0, y0)
                                and (hi[0], hi[1]) == (x0 + 128.0, y0 + 128.0))
        chk["tile_128m"] = ext is not None and (ext[0], ext[1]) == (128.0, 128.0)
        stats["sectors"].add(c8)
        stats["sector_x"][0] = min(stats["sector_x"][0], sx)
        stats["sector_x"][1] = max(stats["sector_x"][1], sx)
        stats["sector_y"][0] = min(stats["sector_y"][0], sy)
        stats["sector_y"][1] = max(stats["sector_y"][1], sy)
        _extend(stats["tile_bounds"], lo, hi)
    if deep:
        if blocks[3] is not None:
            chain_ok, has_water = _poly_chain(b, blocks[3], c8, pool)
            chk["polygons_chain"] = chain_ok
            stats["water_bit3"]["b3_%d_w_%d" % (bool(flags & 8), has_water)] += 1
        qt_ok, qn, ql, qd = _walk_quadtree(b, root, c4)
        chk["quadtree_valid"] = qt_ok
        stats["quadtree"]["nodes"] += qn
        stats["quadtree"]["leaves"] += ql
        stats["quadtree"]["max_depth"] = max(stats["quadtree"]["max_depth"], qd)
        chk["volumes_valid"] = _walk_volumes(b, vol, dc, stats)

    stats["flags"][flags] += 1
    stats["build"][hx(e8)] += 1
    for key, value in (("vertices", c0), ("polygons", c4), ("pool", pool), ("special_links", d4),
                       ("link_indices", d8), ("volumes", dc)):
        stats["totals"][key] += value
        stats["max"][key] = max(stats["max"].get(key, 0), value)
    row = {"flags": flags, "sector": c8, "vertices": c0, "polygons": c4, "pool": pool, "special_links": d4,
           "volumes": dc, "data_size": cc}
    return row, chk


def _poly_chain(b, blocks, owner, pool):
    """Every polygon's +0 is the tile id; +4 is (first index << 4) | count and the runs tile the index pool."""
    nxt = 0
    has_water = False
    for o, n in blocks:
        for mesh, packed, pf18 in struct.iter_unpack("<II16xI16x", b[o:o + 44 * n]):
            if mesh != owner or packed >> 4 != nxt:
                return False, has_water
            if pf18 & 0x80000:
                has_water = True
            nxt += packed & 0xF
    return nxt == pool, has_water


def _walk_volumes(b, vol, dc, stats):
    """Paired 96-byte obstacle footprint records at +0xB8 (count u32@0xDC, ASSET-281)."""
    if dc == 0:
        return vol is None
    if vol is None or dc % 2 != 0 or vol + 96 * dc > len(b):
        return False
    flags = Counter()
    vertex_counts = Counter()
    for p in range(dc // 2):
        o0 = vol + (2 * p) * 96
        o1 = o0 + 96
        if b[o0:o0 + 20] != b[o1:o1 + 20]:
            return False
        xyz = _f32s(b, o0 + 8, 3)
        if xyz is None or not all(math.isfinite(v) for v in xyz):
            return False
        f0, vc0 = b[o0 + 0x14], b[o0 + 0x15]
        f1, vc1 = b[o1 + 0x14], b[o1 + 0x15]
        if f0 not in (0, 4) or f1 != 8 or vc0 not in (4, 8) or vc1 != 4:
            return False
        if not (_zero(b, o0 + 0x16 + 4 * vc0, o0 + 0x56) and _zero(b, o0 + 0x5A, o0 + 0x60)):
            return False
        if not (_zero(b, o1 + 0x16 + 4 * vc1, o1 + 0x56) and _zero(b, o1 + 0x5A, o1 + 0x60)):
            return False
        flags[f0] += 1
        flags[f1] += 1
        vertex_counts[vc0] += 1
        vertex_counts[vc1] += 1
    vstats = stats["volumes"]
    vstats["pairs"] += dc // 2
    vstats["flags"].update(flags)
    vstats["vertex_counts"].update(vertex_counts)
    return True


# ------------------------------------------------------------------------------------------------------ grid

def decode_grid(b, nstreams, stats, deep):
    chk = {}
    groups, entries = _u32(b, 0x10), _u32(b, 0x14)
    payload = offset_of(_u64(b, 0x20))
    ex = struct.unpack_from("<3I", b, 0x28)
    sector = _u32(b, 0x34)
    origin = _f32s(b, 0x40, 3)
    chk["prefix_zero"] = _zero(b, 0, 8)
    chk["block_map"] = block_map_ok(b, nstreams, size=32)
    chk["reserved_zero"] = _zero(b, 0x38, 0x40) and _zero(b, 0x4C, 0x50) and _zero(b, 0x54, 0x60)
    chk["table_in_bounds"] = 0x60 + GRID_GROUP * groups <= len(b)
    chk["payload_in_bounds"] = payload is not None and payload + 4 * entries <= len(b)
    chk["cubic_extent"] = ex[0] == ex[1] == ex[2]
    sx, sy = sector & 0x1FF, sector >> 9
    chk["sector_origin"] = origin is not None and (origin[0], origin[1]) == ((sx - 256) * 128.0, (sy - 256) * 128.0)
    if chk["table_in_bounds"]:
        ctrl = b"".join(b[0x60 + GRID_GROUP * g:0x60 + GRID_GROUP * g + 16] for g in range(groups))
        empty = ctrl.count(0x80)
        valid_ctrl = max(ctrl) <= 0x80 if ctrl else True
        used = len(ctrl) - empty if valid_ctrl else -1
        chk["control_bytes"] = valid_ctrl
        chk["occupied_equals_entries"] = valid_ctrl and used == entries
        if deep:
            chk["values_bijective"] = chk["occupied_equals_entries"] and _grid_values(b, groups, entries)
            if chk["values_bijective"] and chk["payload_in_bounds"]:
                keys = _grid_slots(b, groups)
                chk["keys_octree"] = len(keys) == entries and _grid_octree(keys)
                if chk["keys_octree"]:
                    chk["inner_payload_ff00"] = _grid_payloads(b, payload, keys, stats)
    stats["groups"][groups] += 1
    stats["extent"][ex[0]] += 1
    oz = origin[2] if (origin and math.isfinite(origin[2])) else None
    stats["oz"][oz] += 1
    stats["sectors"].add(sector)
    stats["entries_total"] += entries
    stats["entries_max"] = max(stats["entries_max"], entries)
    return {"sector": sector, "groups": groups, "entries": entries, "extent": ex[0]}, chk


def _grid_values(b, groups, entries):
    seen = bytearray(entries)
    for g in range(groups):
        base = 0x60 + GRID_GROUP * g
        for k, c in enumerate(b[base:base + 16]):
            if c < 0x80:
                value = _u64(b, base + 16 + 16 * k + 8)
                if value is None or value >= entries or seen[value]:
                    return False
                seen[value] = 1
    return all(seen)


def _grid_slots(b, groups):
    """{key: value} of every occupied slot (call after _grid_values has accepted the table)."""
    keys = {}
    for g in range(groups):
        base = 0x60 + GRID_GROUP * g
        for k, c in enumerate(b[base:base + 16]):
            if c < 0x80:
                keys[_u64(b, base + 16 + 16 * k)] = _u64(b, base + 16 + 16 * k + 8)
    return keys


def _grid_octree(keys):
    """Keys are sentinel-bit linear octree keys: 1 is the root, a child is (parent << 3) | octant."""
    if 1 not in keys:
        return False
    for key in keys:
        if key == 0 or (key.bit_length() - 1) % 3 or (key != 1 and key >> 3 not in keys):
            return False
    return True


GRID_INNER = 0x0000FF00  # payload of every node that has children
GRID_LEAF_KIND = {0x01: "low_byte_01", 0x03: "low_byte_03"}


def _grid_payloads(b, payload, keys, stats):
    """Inner nodes carry 0x0000ff00; count leaf payloads by low byte and every key by depth."""
    inner = {key >> 3 for key in keys if key != 1}
    words = {key: _u32(b, payload + 4 * value) for key, value in keys.items()}
    if any(words[key] != GRID_INNER for key in inner):
        return False
    stats["inner_nodes"] += len(inner)
    for key, word in words.items():
        stats["key_depth"][(key.bit_length() - 1) // 3] += 1
        if key in inner:
            continue
        kind = "ff00" if word == GRID_INNER else GRID_LEAF_KIND.get(word & 0xFF, "other")
        stats["leaf_payload"][kind] += 1
    return True


# ------------------------------------------------------------------------------------------------------ scan

def new_stats():
    inf = float("inf")
    return {
        "0drp": {"schemas": Counter(), "root": Counter(), "block_at_0x60": Counter(), "block_hash": Counter(),
                 "enums": Counter(), "q40_shape": Counter(), "root_index": Counter(), "pages_4k_zero": 0,
                 "pages_4k_boundary": Counter(),
                 "u16": {"n_structs": Counter(), "n_enums": Counter(), "n_blocks": Counter()},
                 "cover_nodes": 0, "region_cover_nodes": 0, "dynamic_cover_nodes": 0,
                 "region_points": 0, "region_inumpoints": Counter(), "first_node": Counter(),
                 "region_bounds": [[inf] * 3, [-inf] * 3]},
        "navmesh": {"flags": Counter(), "build": Counter(), "totals": Counter(), "max": {}, "sectors": set(),
                    "sector_x": [inf, -inf], "sector_y": [inf, -inf], "split_blocks_max": {},
                    "tile_bounds": [[inf] * 3, [-inf] * 3], "water_bit3": Counter(),
                    "quadtree": {"nodes": 0, "leaves": 0, "max_depth": 0},
                    "volumes": {"pairs": 0, "flags": Counter(), "vertex_counts": Counter()}},
        "grid": {"groups": Counter(), "extent": Counter(), "oz": Counter(), "sectors": set(), "entries_total": 0,
                 "entries_max": 0, "key_depth": Counter(), "inner_nodes": 0, "leaf_payload": Counter()},
        "psin": {"version": Counter(), "complete": 0},
    }


def scan(path, start=None, limit=None, deep=False, rows=None):
    """Decode every resource in an RPF9 archive or carved directory. Returns the summary dict."""
    if os.path.isdir(path):
        return _scan_dir(path, deep, rows)
    return _scan_rpf(path, start, limit, deep, rows)


def _scan_dir(path, deep, rows):
    t0 = time.time()
    stats = new_stats()
    classes, checks, carver = {}, {}, Counter()
    stream = walk_dir(path)
    pending = next(stream, None)
    while pending is not None:
        s, used, head = pending
        pending = next(stream, None)
        if len(head) < MIN_STREAM:
            carver["tiny_orphans"] += 1
            continue
        carver["streams"] += 1
        carver["inflated"] += len(head)
        cls = classify(head)
        parts = [(s, used, head)]
        buf = bytearray(head)
        while pending is not None and wants_more(cls, buf, len(parts)):
            if len(buf) + len(pending[2]) > MAX_INFLATE:
                break
            parts.append(pending)
            _, _, cdata = pending
            buf.extend(cdata)
            if len(cdata) < MIN_STREAM:
                carver["tiny_continuations"] += 1
            else:
                carver["streams"] += 1
                carver["inflated"] += len(cdata)
            pending = next(stream, None)
        data = bytes(buf)
        complete = is_complete(cls, data, len(parts))
        row, chk = _decode(cls, data, len(head), len(parts), stats, deep)
        _record(classes, checks, cls, parts, data, complete, chk)
        if rows is not None:
            row.update({"offset": s, "class": cls, "streams": len(parts), "stored": None,
                        "inflated": len(data), "complete": complete,
                        "checks_failed": " ".join(k for k, v in sorted(chk.items()) if not v)})
            rows.append(row)
    return _summary(path, "directory", None, None, carver, classes, checks, stats, time.time() - t0)


def _scan_rpf(path, start, limit, deep, rows):
    t0 = time.time()
    stats = new_stats()
    classes, checks, carver = {}, {}, Counter()
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if size < 8:
            raise ValueError("%s is %d bytes, too small for an RPF header" % (path, size))
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if m[1:4] != b"FPR" or m[0:1] not in (b"7", b"8", b"9"):
                raise ValueError("%s: not an RPF7/8/9 archive" % path)
            data_start = struct.unpack_from("<I", m, 4)[0]
            if start is None:
                start = data_start
            if not 0 <= start < size:
                raise ValueError("start %d is outside %s (%d bytes)" % (start, path, size))
            end = size if limit is None else min(size, start + limit)
            stream = walk(m, start, size)
            last = start
            pending = next(stream, None)
            while pending is not None and pending[0] < end:
                s, used, head = pending
                pending = next(stream, None)
                if len(head) < MIN_STREAM:
                    carver["tiny_orphans"] += 1
                    continue
                last = _account(m, carver, last, s, used, len(head), end)
                cls = classify(head)
                parts = [(s, used, head)]
                buf = bytearray(head)
                while pending is not None and wants_more(cls, buf, len(parts)):
                    if len(buf) + len(pending[2]) > MAX_INFLATE:
                        break
                    parts.append(pending)
                    cs, cused, cdata = pending
                    buf.extend(cdata)
                    if len(cdata) < MIN_STREAM:
                        carver["tiny_continuations"] += 1
                    else:
                        last = _account(m, carver, last, cs, cused, len(cdata), end)
                    pending = next(stream, None)
                data = bytes(buf)
                complete = is_complete(cls, data, len(parts))
                row, chk = _decode(cls, data, len(head), len(parts), stats, deep)
                _record(classes, checks, cls, parts, data, complete, chk)
                if rows is not None:
                    row.update({"offset": s, "class": cls, "streams": len(parts),
                                "stored": sum(p[1] for p in parts), "inflated": len(data), "complete": complete,
                                "checks_failed": " ".join(k for k, v in sorted(chk.items()) if not v)})
                    rows.append(row)
            if last < end:
                gap_zero, gap_unread = gap_split(m, last, end)
                carver["zero_padding" if not gap_unread else "unread"] += gap_zero + gap_unread
    return _summary(path, "rpf", start, end, carver, classes, checks, stats, time.time() - t0)


def _account(m, carver, last, s, used, inflated, end):
    """Carver-compatible accounting of one stream of 64 bytes or more, plus the gap before it.

    A stream that runs past the --limit window is still decoded (a resource is read to its end) but, as in
    navmesh_index.py (#766), it is not counted as carved: its bytes inside the window fall into the tail, so
    streams + zero_padding + unread add up to the span. It is reported under past_window instead."""
    if s + used > end:
        carver["past_window"] += 1
        carver["past_window_stored"] += used
        return last
    if s > last and last < end:
        gap_zero, gap_unread = gap_split(m, last, min(s, end))
        carver["zero_padding" if not gap_unread else "unread"] += gap_zero + gap_unread
    carver["streams"] += 1
    carver["stored"] += used
    carver["inflated"] += inflated
    return max(last, s + used)


def _decode(cls, buf, len0, nstreams, stats, deep):
    if cls == "0drp":
        return decode_7a(buf, len0, nstreams, stats["0drp"])
    if cls == "navmesh":
        return decode_navmesh(buf, nstreams, stats["navmesh"], deep)
    if cls == "grid":
        return decode_grid(buf, nstreams, stats["grid"], deep)
    if cls == "psin":
        complete, _, _ = _psin_state(buf)
        stats["psin"]["version"][_u32(buf, 8)] += 1
        stats["psin"]["complete"] += complete
        return {}, {}
    return {}, {}


def _record(classes, checks, cls, parts, buf, complete, chk):
    c = classes.setdefault(cls, {"resources": 0, "streams": Counter(), "stored": 0, "inflated": 0,
                                 "incomplete": 0})
    c["resources"] += 1
    c["streams"][len(parts)] += 1
    if parts[0][1] is not None:
        c["stored"] += sum(p[1] for p in parts)
    c["inflated"] += len(buf)
    c["incomplete"] += not complete
    table = checks.setdefault(cls, {})
    for name, value in chk.items():
        slot = table.setdefault(name, [0, 0])
        slot[0 if value else 1] += 1


def _counter(c, key=str):
    return {key(k): v for k, v in sorted(c.items(), key=lambda kv: (-kv[1], str(kv[0])))}


def _box(box):
    if box[0][0] == float("inf"):
        return None
    return {"min": [round(v, 3) for v in box[0]], "max": [round(v, 3) for v in box[1]]}


def _summary(path, mode, start, end, carver, classes, checks, stats, elapsed):
    d, n, g = stats["0drp"], stats["navmesh"], stats["grid"]
    static = n["sectors"]
    return {
        "input": os.path.basename(os.path.abspath(path)),
        "mode": mode,
        "start": start,
        "end": end,
        "span": (end - start) if (start is not None and end is not None) else None,
        "carver_streams": {k: carver.get(k, 0) for k in ("streams", "stored", "inflated", "zero_padding",
                                                         "unread", "tiny_continuations", "tiny_orphans",
                                                         "past_window", "past_window_stored")},
        "resources": {cls: dict(v, streams={str(k): c for k, c in sorted(v["streams"].items())})
                      for cls, v in sorted(classes.items())},
        "checks": {cls: {name: {"pass": p, "fail": f} for name, (p, f) in sorted(t.items())}
                   for cls, t in sorted(checks.items())},
        "0drp": {
            "schemas": {"%s/%d structs" % k: v for k, v in sorted(d["schemas"].items(), key=lambda kv: -kv[1])},
            "root": _counter(d["root"]),
            "block_at_0x60": _counter(d["block_at_0x60"]),
            "block_hash": _counter(d["block_hash"]),
            "enums": _counter(d["enums"]),
            "q40_shape": _counter(d["q40_shape"]),
            "root_index": _counter(d["root_index"]),
            "u16_0x48": {k: _counter(v) for k, v in d["u16"].items()},
            "pages_4k_zero": d["pages_4k_zero"],
            "pages_4k_boundary": dict(d["pages_4k_boundary"]),
            "cover_nodes": d["cover_nodes"],
            "region_cover_nodes": d["region_cover_nodes"],
            "dynamic_cover_nodes": d["dynamic_cover_nodes"],
            "region_points": d["region_points"],
            "region_inumpoints": dict(d["region_inumpoints"]),
            "first_node": dict(d["first_node"]),
            "region_bounds": _box(d["region_bounds"]),
        },
        "navmesh": {
            "flags": _counter(n["flags"]),
            "build_ids": len(n["build"]),
            "totals": dict(n["totals"]),
            "max": n["max"],
            "static_sectors": len(static),
            "sector_x": n["sector_x"] if static else None,
            "sector_y": n["sector_y"] if static else None,
            "split_blocks_max": n["split_blocks_max"],
            "tile_bounds": _box(n["tile_bounds"]),
            "water_bit3": dict(n["water_bit3"]),
            "quadtree": dict(n["quadtree"]),
            "volumes": {"pairs": n["volumes"]["pairs"], "flags": _counter(n["volumes"]["flags"]),
                        "vertex_counts": _counter(n["volumes"]["vertex_counts"])},
        },
        "grid": {
            "groups": _counter(g["groups"]),
            "extent": _counter(g["extent"]),
            "origin_z": _counter(g["oz"]),
            "sectors": len(g["sectors"]),
            "sectors_not_in_navmesh": len(g["sectors"] - static),
            "navmesh_sectors_not_in_grid": len(static - g["sectors"]),
            "entries_total": g["entries_total"],
            "entries_max": g["entries_max"],
            "key_depth": _counter(g["key_depth"]),
            "inner_nodes": g["inner_nodes"],
            "leaf_payload": _counter(g["leaf_payload"]),
        },
        "psin": {"version": _counter(stats["psin"]["version"]), "complete": stats["psin"]["complete"]},
        "elapsed_s": round(elapsed, 1),
    }


def _make_parent(path):
    """Create the folder an output path lives in: the scan is over by the time we write, so a missing
    folder used to end the run in a FileNotFoundError and lose the result."""
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path, rows):
    _make_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in COLUMNS})


def _positive(text):
    value = int(text, 0)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def _offset(text):
    value = int(text, 0)
    if value < 0:
        raise argparse.ArgumentTypeError("must not be negative")
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


def same_report_path(first, second):
    if _path_key(first) == _path_key(second):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="RPF9 archive (navmeshes.rpf) or directory of carved streams")
    ap.add_argument("--out", help="JSON summary (default: <input name>.navmesh-decode.json)")
    ap.add_argument("--csv", help="optional CSV, one row per resource")
    ap.add_argument("--start", type=_offset, help="first byte to scan in an archive")
    ap.add_argument("--limit", type=_positive, help="bytes to scan from --start")
    ap.add_argument("--deep", action="store_true", help="walk every polygon, quadtree node and grid slot")
    ap.add_argument("--quiet", action="store_true", help="do not print the summary")
    a = ap.parse_args(argv)
    if not os.path.exists(a.input):
        ap.error("%s does not exist" % a.input)
    if os.path.isdir(a.input) and (a.start is not None or a.limit is not None):
        ap.error("--start and --limit apply to archive inputs only")
    base = os.path.basename(os.path.abspath(a.input)) or "navmesh"
    out = a.out or os.path.splitext(base)[0] + ".navmesh-decode.json"
    if a.csv and same_report_path(out, a.csv):
        ap.error("--out and --csv must use different paths")
    for flag, path in (("--out", out), ("--csv", a.csv)):
        if not path:
            continue
        if os.path.isdir(path):              # open() on a folder ends the run in a PermissionError traceback
            ap.error(f"{flag} is a directory: {path}")
        if same_report_path(path, a.input):
            ap.error(f"{flag} would overwrite the input {a.input}")
        if os.path.isfile(path) and os.path.isdir(a.input):
            manifest = os.path.join(a.input, "index.json")
            if same_report_path(path, manifest):
                ap.error(f"{flag} would overwrite the carve manifest {manifest}")
            for entry in carved_files(a.input):
                if same_report_path(path, entry.path):
                    ap.error(f"{flag} would overwrite the input {entry.path}")
    rows = [] if a.csv else None
    try:
        summary = scan(a.input, a.start, a.limit, a.deep, rows)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    report_path = out
    try:
        _make_parent(out)
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(summary, f, indent=1, allow_nan=False)
        if a.csv:
            report_path = a.csv
            write_csv(a.csv, rows)
    except OSError as exc:
        print(f"ERROR: cannot write {report_path}: {exc.strerror or exc}", file=sys.stderr)
        return 2
    if not a.quiet:
        print(json.dumps(summary, indent=1, allow_nan=False))
        print("wrote %s" % out + (", %s" % a.csv if a.csv else ""))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
