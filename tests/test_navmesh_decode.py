"""Synthetic tests for tools/navmesh_decode.py. No game files are read or needed."""
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import navmesh_decode as nd  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "navmesh_decode.py")
BASE = nd.BASE


def make_rpf(path, streams, data_start=0x40, pad_between=0):
    buf = bytearray(data_start)
    buf[0:4] = b"9FPR"
    struct.pack_into("<I", buf, 4, data_start)
    offsets = []
    for i, raw in enumerate(streams):
        if i and pad_between:
            buf += bytes(pad_between)
        offsets.append(len(buf))
        buf += zlib.compress(raw)
    with open(path, "wb") as f:
        f.write(buf)
    return offsets


def build_0drp(
    *,
    total_size=0x2000,
    split_at=None,
    root_hash=nd.REGION,
    node_count=3,
    region_inumpoints=None,
    node_at_60=True,
    q40_items=((0x101, 32),),
    override_n4e=None,
    overlap_q38=False,
    oob_second_q38=False,
    no_enums=False,
    bad_enum=False,
    bad_bm_streams=None,
    dirty_reserved=False,
    nan_bounds=False,
):
    """Build a synthetic 0DRP (0x7a) cover-point resource, optionally split across two streams."""
    buf = bytearray(total_size)
    nstreams = 2 if split_at is not None else 1

    # Layout from +0x60: blocks, q38, p08, then q28/members, q30/enums, q40.
    cur = 0x60
    blocks = []

    if node_at_60:
        node_off = cur
        node_sz = node_count * nd.NODE_SIZE
        for k in range(node_count):
            o = node_off + k * nd.NODE_SIZE
            struct.pack_into("<QII", buf, o, 0x1122334455667788, 0x1001, k)
            struct.pack_into("<QII", buf, o + 0x10, 0x1122334455667788, 0xFFFFFFFF, 0xFFFFFFFF)
            struct.pack_into("<QII", buf, o + 0x20, 0x99AABBCCDDEEFF00, 0x1002, 1)
        blocks.append((nd.NODE, node_sz, node_off))
        cur += node_sz
    else:
        sub_off = cur
        sub_sz = 0x40
        blocks.append((0xDE5F1345, sub_sz, sub_off))
        cur += sub_sz
        node_off = cur
        node_sz = node_count * nd.NODE_SIZE
        blocks.append((nd.NODE, node_sz, node_off))
        cur += node_sz

    root_off = cur
    root_sz = nd.REGION_SIZE if root_hash == nd.REGION else 48
    if root_hash == nd.REGION:
        pts = node_count if region_inumpoints is None else region_inumpoints
        struct.pack_into("<iiI", buf, root_off, 5, pts, 0xCAFEBABE)
        if nan_bounds:
            struct.pack_into("<3f", buf, root_off + 0x10, float("nan"), float("-inf"), -50.0)
            struct.pack_into("<3f", buf, root_off + 0x20, float("inf"), 384.0, 150.0)
        else:
            struct.pack_into("<3f", buf, root_off + 0x10, -128.0, 256.0, -50.0)
            struct.pack_into("<3f", buf, root_off + 0x20, 0.0, 384.0, 150.0)
        struct.pack_into("<H", buf, root_off + 0xA0, node_count)
    blocks.append((root_hash, root_sz, root_off))
    cur += root_sz

    if overlap_q38:
        blocks.append((0xDEADBEEF, 0x20, 0x60))
    if oob_second_q38:
        blocks.append((0xFEEDFACE, 0x10000, total_size))

    q38_off = (cur + 15) & ~15
    for i, (h, sz, t) in enumerate(blocks):
        struct.pack_into("<IIQ", buf, q38_off + 16 * i, h, sz, BASE + t)
    cur = q38_off + 16 * len(blocks)

    p08_off = (cur + 15) & ~15
    struct.pack_into("<3Q", buf, p08_off, 0, bad_bm_streams if bad_bm_streams is not None else nstreams, 0)
    data_end = max([t + sz for _, sz, t in blocks if t + sz <= len(buf)] + [q38_off + 16 * len(blocks), p08_off + 24])
    cur = p08_off + 24

    # Struct table (q28): NODE and root_hash (and 0xDE5F1345 if present).
    struct_specs = [
        (nd.NODE, nd.NODE_SIZE, [(nd.joaat_cs("Uid"), 0, 0x05, 0, 0, nd.joaat_cs("CCoverNodeId"))]),
        (root_hash, root_sz, [(0x11110001, 0, 0x14, 0, 0, 0)]),
    ]
    if not node_at_60:
        struct_specs.append((0xDE5F1345, 16, [(0x11110002, 0, 0x52, 0, 0, 0)]))
    q28_off = (cur + 7) & ~7
    cur = q28_off + 32 * len(struct_specs)
    for i, (sh, ssz, mems) in enumerate(struct_specs):
        mo = cur
        for k, (mh, moff, mty, msub, mref, mext) in enumerate(mems):
            struct.pack_into("<IIBBHI", buf, mo + 16 * k, mh, moff, mty, msub, mref, mext)
        cur += 16 * len(mems)
        struct.pack_into("<IIQQIHH", buf, q28_off + 32 * i, sh, 0x12345678, 0x300, BASE + mo, ssz, 0, len(mems))

    # Enum table (q30): optional 4-entry enum eCoverHeight.
    if no_enums:
        q30_off = 0
        n_enums = 0
    else:
        q30_off = (cur + 7) & ~7
        ev_off = q30_off + 24
        enum_vals = [0, 1, 2, 3] if not bad_enum else [0, 2, 3, 4]
        for k, val in enumerate(enum_vals):
            struct.pack_into("<qII", buf, ev_off + 16 * k, val, 0x80000000 + k, 0)
        struct.pack_into("<IIQQ", buf, q30_off, nd.joaat_cs("eCoverHeight"), 0x22220001, BASE + ev_off, len(enum_vals))
        cur = ev_off + 16 * len(enum_vals)
        n_enums = 1

    # q40 table: put in continuation when split_at is set and tiny continuation is desired.
    q40_off = split_at if (split_at is not None and split_at >= cur) else ((cur + 7) & ~7)
    need = q40_off + 4 + 8 * len(q40_items) + sum(sz for _, sz in q40_items)
    if need > len(buf):
        buf.extend(bytes(need - len(buf)))
    struct.pack_into("<I", buf, q40_off, len(q40_items))
    for i, (tag, sz) in enumerate(q40_items):
        struct.pack_into("<II", buf, q40_off + 4 + 8 * i, tag, sz)

    # Fixed 0x60 header.
    struct.pack_into("<Q", buf, 0x08, BASE + p08_off)
    buf[0x10:0x14] = b"0DRP"
    struct.pack_into("<I", buf, 0x14, nd.VERSION_7A)
    root_1based = next(i + 1 for i, (h, _, _) in enumerate(blocks) if h == root_hash)
    struct.pack_into("<I", buf, 0x20, root_1based)
    struct.pack_into("<Q", buf, 0x28, BASE + q28_off)
    struct.pack_into("<Q", buf, 0x30, BASE + q30_off if q30_off else 0)
    struct.pack_into("<Q", buf, 0x38, BASE + q38_off)
    struct.pack_into("<Q", buf, 0x40, BASE + q40_off)
    len0 = split_at if split_at is not None else len(buf)
    pg = (data_end + 4095) // 4096
    stream_end = len0 if data_end <= len0 else len(buf)
    n4e = override_n4e if override_n4e is not None else (pg if pg * 4096 < stream_end else 0)
    struct.pack_into("<4H", buf, 0x48, len(struct_specs), n_enums, len(blocks), n4e)
    if dirty_reserved:
        buf[0x55] = 0xAA

    if split_at is not None:
        return [bytes(buf[:split_at]), bytes(buf[split_at:])]
    return [bytes(buf)]


def build_navmesh(
    *,
    flags=3,
    sector=(260 << 9) | 250,
    build_id=0x01020304,
    n_verts=5,
    poly_sizes=(3, 4),
    n_links=2,
    n_link_idx=3,
    n_volumes=2,
    water_poly=False,
    tiny_bm_cont=False,
    corrupt_checksum=False,
    corrupt_vidx=False,
    corrupt_link_idx=False,
    corrupt_null_vol=False,
    corrupt_chain=False,
    qt_cycle=False,
):
    """Build a synthetic 0x00010015 navmesh tile with split arrays (multi-block supported), quadtree, links and volumes."""
    buf = bytearray(0x2000)
    is_dynamic = bool(flags & 4)
    has_links = bool(flags & 2)
    if is_dynamic:
        sector = 0
        build_id = 0
        x0, y0, dx, dy, dz = -5.0, -4.0, 10.0, 8.0, 3.0
    else:
        sx, sy = sector & 0x1FF, sector >> 9
        x0, y0, dx, dy, dz = (sx - 256) * 128.0, (sy - 256) * 128.0, 128.0, 128.0, 16.0
    if not has_links:
        n_links = n_link_idx = 0

    n_polys = len(poly_sizes)
    pool = sum(poly_sizes)

    # Fixed header at +0x00..+0xEF.
    struct.pack_into("<II", buf, 0x10, flags, nd.DATA15)
    struct.pack_into("<16f", buf, 0x20, *nd.IDENTITY)
    struct.pack_into("<3f", buf, 0x60, dx, dy, dz)
    struct.pack_into("<I", buf, 0x90, pool)
    data_size = 6 * n_verts + 14 * pool + 44 * n_polys + 32 * n_links
    if corrupt_checksum:
        data_size += 1
    vol_count = 5 if corrupt_null_vol else n_volumes
    struct.pack_into("<8I", buf, 0xC0, n_verts, n_polys, sector, data_size, 0, n_links, n_link_idx, vol_count)
    struct.pack_into("<I", buf, 0xE8, build_id)

    cur = 0xF0

    def alloc(size, align=16):
        nonlocal cur
        cur = (cur + align - 1) & ~(align - 1)
        off = cur
        cur += size
        if cur > len(buf):
            buf.extend(bytes(cur - len(buf)))
        return off

    # Special links (+0xA8) and index table (+0xB0).
    if has_links:
        la = alloc(32 * n_links)
        lb = alloc(2 * n_link_idx)
        for k in range(n_link_idx):
            val = n_links if (corrupt_link_idx and k == 0) else (k % n_links)
            struct.pack_into("<H", buf, lb + 2 * k, val)
        struct.pack_into("<Q", buf, 0xA8, BASE + la)
        struct.pack_into("<Q", buf, 0xB0, BASE + lb)

    # 96-byte volumes (+0xB8); left NULL when corrupt_null_vol=True to verify null-pointer guard.
    if n_volumes and not corrupt_null_vol:
        vo = alloc(96 * n_volumes)
        struct.pack_into("<Q", buf, 0xB8, BASE + vo)
        for p in range(n_volumes // 2):
            o0 = vo + (2 * p) * 96
            o1 = o0 + 96
            hdr = struct.pack("<Q3f", 0x1122334455667788 + p, x0 + 16.0, y0 + 16.0, 0.3)
            buf[o0:o0 + 20] = hdr
            buf[o1:o1 + 20] = hdr
            buf[o0 + 0x14], buf[o0 + 0x15] = 0, 8
            buf[o1 + 0x14], buf[o1 + 0x15] = 8, 4
            struct.pack_into("<16h", buf, o0 + 0x16, 410, 616, -408, 616, -536, 488, -536, -491,
                             -408, -619, 410, -619, 538, -491, 538, 488)
            struct.pack_into("<8h", buf, o1 + 0x16, -408, 488, -408, -491, 410, -491, 410, 488)
            struct.pack_into("<2H", buf, o0 + 0x56, 2006, 1095)
            struct.pack_into("<2H", buf, o1 + 0x56, 2006, 1095)

    def put_split(hdr_at, stride, data_bytes, total):
        per = nd.SPLIT_BLOCK // stride
        chunks = []
        idx = 0
        while idx < total or not chunks:
            take = min(per, total - idx)
            do = alloc(take * stride)
            buf[do:do + take * stride] = data_bytes[idx * stride:(idx + take) * stride]
            chunks.append((do, take, idx))
            idx += take
        bp = alloc(16 * len(chunks))
        sp = alloc(4 * len(chunks))
        for i, (do, take, start_i) in enumerate(chunks):
            struct.pack_into("<QQ", buf, bp + 16 * i, BASE + do, take)
            struct.pack_into("<I", buf, sp + 4 * i, start_i)
        desc = alloc(48)
        struct.pack_into("<6Q", buf, desc, 0x1000, total, BASE + bp, BASE + sp, len(chunks), 0)
        struct.pack_into("<Q", buf, hdr_at, BASE + desc)

    # Compressed vertices (6 B), vertex indices (2 B), adj polys (12 B), polygons (44 B).
    verts = bytearray(6 * n_verts)
    vidx = bytearray(2 * pool)
    for k in range(pool):
        struct.pack_into("<H", vidx, 2 * k, n_verts if (corrupt_vidx and k == 0) else (k % n_verts))
    adjs = bytearray(12 * pool)
    polys = bytearray(44 * n_polys)
    pos = 0
    for i, nv in enumerate(poly_sizes):
        start_idx = pos + 1 if (corrupt_chain and i == 1) else pos
        struct.pack_into("<II", polys, 44 * i, sector, (start_idx << 4) | nv)
        if water_poly and i == 0:
            struct.pack_into("<I", polys, 44 * i + 0x18, 0x80000)
        pos += nv
    put_split(0x70, 6, verts, n_verts)
    put_split(0x80, 2, vidx, pool)
    put_split(0x88, 12, adjs, pool)
    put_split(0x98, 44, polys, n_polys)

    # 2-level quadtree: root node -> 1 child leaf node -> leaf_data -> u16[n_polys].
    root = alloc(0x60)
    child = alloc(0x60)
    leaf = alloc(0x20)
    lpolys = alloc(2 * n_polys)
    for k in range(n_polys):
        struct.pack_into("<H", buf, lpolys + 2 * k, k)
    struct.pack_into("<3f", buf, root, x0, y0, 0.0)
    struct.pack_into("<3f", buf, root + 0x10, x0 + dx, y0 + dy, dz)
    struct.pack_into("<Q", buf, root + 0x34, BASE + child)
    struct.pack_into("<3f", buf, child, x0, y0, 0.0)
    struct.pack_into("<3f", buf, child + 0x10, x0 + dx, y0 + dy, dz)
    struct.pack_into("<Q", buf, child + 0x2C, BASE + leaf)
    if qt_cycle:
        struct.pack_into("<Q", buf, child + 0x34, BASE + root)
    struct.pack_into("<4Q", buf, leaf, 0, BASE + lpolys, 0, n_polys)
    struct.pack_into("<Q", buf, 0xA0, BASE + root)

    # Block map (32 bytes): either in stream 0 or spilled alone into a tiny 32-byte stream 1.
    if tiny_bm_cont:
        split_at = (cur + 15) & ~15
        buf = buf[:split_at] + bytearray(32)
        struct.pack_into("<Q", buf, 0x08, BASE + split_at)
        struct.pack_into("<4Q", buf, split_at, 0, 2, 0, 0)
        return [bytes(buf[:split_at]), bytes(buf[split_at:])]
    bm = alloc(32)
    struct.pack_into("<Q", buf, 0x08, BASE + bm)
    struct.pack_into("<4Q", buf, bm, 0, 1, 0, 0)
    return [bytes(buf[:cur])]


def build_grid(
    *,
    sector=(260 << 9) | 250,
    groups=8,
    entries=5,
    declared_entries=None,
    extent=128,
    oz=0.0,
    two_streams=False,
    bad_ctrl=False,
    dup_value=False,
    keys=None,
    words=None,
):
    """Build a synthetic Band-3 grid resource (0x60 header + 16-way SwissTable + u32[entries]).

    By default the keys are an octree written by hand (root 1, its children 8.. in order, then the children of
    8 from 64): every key with a child gets 0x0000ff00, every other key a 0x01 leaf word."""
    if keys is None:
        keys = [1] + list(range(8, 8 + min(entries - 1, 8))) + list(range(64, 64 + max(entries - 9, 0)))
        keys = keys[:entries]
    if words is None:
        parents = {k >> 3 for k in keys if k != 1}
        words = [0x0000FF00 if k in parents else 0x00020001 for k in keys]
    table_end = 0x60 + nd.GRID_GROUP * groups
    bm_off = (table_end + 15) & ~15
    payload_off = (bm_off + 32 + 15) & ~15
    total = payload_off + 4 * entries
    buf = bytearray(total)
    sx, sy = sector & 0x1FF, sector >> 9
    ox, oy = (sx - 256) * 128.0, (sy - 256) * 128.0
    decl = entries if declared_entries is None else declared_entries

    struct.pack_into("<Q", buf, 0x08, BASE + bm_off)
    struct.pack_into("<II", buf, 0x10, groups, decl)
    struct.pack_into("<QQ", buf, 0x18, BASE + 0x60, BASE + payload_off)
    struct.pack_into("<3II", buf, 0x28, extent, extent, extent, sector)
    struct.pack_into("<3f", buf, 0x40, ox, oy, oz)
    struct.pack_into("<I", buf, 0x50, decl)
    struct.pack_into("<4Q", buf, bm_off, 0, 2 if two_streams else 1, 0, 0)

    for g in range(groups):
        base = 0x60 + nd.GRID_GROUP * g
        buf[base:base + 16] = bytes([0x80] * 16)
    for k in range(entries):
        g, slot = divmod(k, 16)
        base = 0x60 + nd.GRID_GROUP * g
        buf[base + slot] = (k * 7) & 0x7F
        val = 0 if (dup_value and k == 1) else k
        struct.pack_into("<QQ", buf, base + 16 + 16 * slot, keys[k], val)
    if bad_ctrl:
        buf[0x60 + 15] = 0xFF
    for k in range(entries):
        struct.pack_into("<I", buf, payload_off + 4 * k, words[k])

    if two_streams:
        if total - payload_off < 64:
            buf.extend(bytes(64 - (total - payload_off)))
        return [bytes(buf[:payload_off]), bytes(buf[payload_off:])]
    return [bytes(buf)]


def build_psin(version=256, split_at=None, truncate=False):
    sec1 = b"LSCH" + struct.pack(">I", 64) + bytes(56)
    sec2 = b"CHKS" + struct.pack(">I", 20) + bytes(12)
    table = bytes(0x40)
    first = 16 + len(table)
    hdr = b"PSIN" + struct.pack(">I", first) + struct.pack("<II", version, len(table))
    blob = hdr + table + sec1 + sec2
    if truncate:
        blob = blob[:-4]
    if split_at is not None:
        return [blob[:split_at], blob[split_at:]]
    return [blob]


class NavmeshDecodeTest(unittest.TestCase):
    def test_joaat_cs_known_hashes(self):
        self.assertEqual(nd.joaat_cs("CCoverPointRegion"), 0x2283B2CB)
        self.assertEqual(nd.joaat_cs("CCoverPointNode"), 0x0BCB00E4)
        self.assertEqual(nd.joaat_cs("CCoverNodeId"), 0xA33600FF)
        self.assertEqual(nd.joaat_cs("rage__spdAABB"), 0xF377E8C8)
        self.assertEqual(nd.joaat_cs("eCoverHeight"), 0x5C3B2E01)
        self.assertEqual(nd.joaat_cs("eCoverUsage"), 0x7245BA24)
        self.assertEqual(nd.joaat_cs("eCoverFlags"), 0x42338022)
        self.assertEqual(len(nd.KNOWN), len(nd.NAMES))

    def test_all_classes_deep_and_cli_csv(self):
        sector = (260 << 9) | 250
        streams = []
        # 1. Single-stream CCoverPointRegion 0DRP with CCoverPointNode at +0x60.
        streams += build_0drp(total_size=0x2000, node_count=4)
        # 2. Two-stream 0DRP with a tiny 44-byte continuation (< 64 B) holding q40.
        streams += build_0drp(total_size=0x1000, split_at=0x1000, node_count=2, override_n4e=1)
        # 3. 0DRP with 0xde5f1345 at +0x60, 0-enum dynamic cover 0x4712787b (q30==0, empty q40).
        streams += build_0drp(total_size=0x2000, node_at_60=False, node_count=2, region_inumpoints=5)
        streams += build_0drp(total_size=0x2000, root_hash=nd.DYNAMIC_COVER, node_count=3, q40_items=(), no_enums=True)
        # 4. Static navmesh (d8 < d4), tiny 32-byte block-map cont, linkless flags=1/9, dynamic flags=7, global flags=69.
        streams += build_navmesh(flags=3, sector=sector, n_verts=6, poly_sizes=(3, 4), n_links=3, n_link_idx=2)
        streams += build_navmesh(flags=11, sector=sector + 1, n_verts=5, poly_sizes=(3, 3), water_poly=True, tiny_bm_cont=True)
        streams += build_navmesh(flags=1, sector=sector + 2, n_verts=4, poly_sizes=(4,), n_volumes=0)
        streams += build_navmesh(flags=9, sector=sector + 3, n_verts=4, poly_sizes=(4,), water_poly=True, n_volumes=0)
        streams += build_navmesh(flags=7, n_verts=4, poly_sizes=(4,), n_links=1, n_link_idx=1, n_volumes=0)
        streams += build_navmesh(flags=69, n_verts=4, poly_sizes=(4,), n_volumes=0)
        # 5. Paired Band-3 grids matching all 4 static sectors.
        streams += build_grid(sector=sector, groups=8, entries=5)
        streams += build_grid(sector=sector + 1, groups=8, entries=6, two_streams=True)
        streams += build_grid(sector=sector + 2, groups=8, entries=2)
        streams += build_grid(sector=sector + 3, groups=8, entries=3)
        # 6. Complete two-stream PSIN + one unclassified other stream + one tiny orphan (< 64 B).
        streams += build_psin(version=256, split_at=0x48)
        streams.append(b"OTHR" + bytes(96))
        streams.append(b"TINY" + bytes(16))

        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "synthetic.rpf")
            make_rpf(arc, streams, pad_between=16)
            out_json = os.path.join(tmp, "out.json")
            out_csv = os.path.join(tmp, "out.csv")
            proc = subprocess.run(
                [sys.executable, "-I", TOOL, arc, "--out", out_json, "--csv", out_csv, "--deep", "--quiet"],
                capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out_json, encoding="utf-8") as f:
                doc = json.load(f)
            with open(out_csv, newline="", encoding="utf-8") as f:
                csv_rows = list(csv.DictReader(f))

        self.assertEqual(doc["resources"]["0drp"]["resources"], 4)
        self.assertEqual(doc["resources"]["navmesh"]["resources"], 6)
        self.assertEqual(doc["resources"]["grid"]["resources"], 4)
        self.assertEqual(doc["resources"]["psin"]["resources"], 1)
        self.assertEqual(doc["resources"]["other"]["resources"], 1)
        self.assertEqual(doc["carver_streams"]["tiny_continuations"], 2)
        self.assertEqual(doc["carver_streams"]["tiny_orphans"], 1)
        self.assertEqual(len(csv_rows), 16)
        for cls, table in doc["checks"].items():
            for name, counts in table.items():
                self.assertEqual(counts["fail"], 0, f"{cls}.{name} failed: {counts}")

        self.assertEqual(doc["0drp"]["cover_nodes"], 11)
        self.assertEqual(doc["0drp"]["region_cover_nodes"], 8)
        self.assertEqual(doc["0drp"]["dynamic_cover_nodes"], 3)
        self.assertEqual(doc["0drp"]["region_inumpoints"], {"match": 2, "mismatch": 1})
        self.assertEqual(doc["0drp"]["pages_4k_boundary"], {"nonzero": 1})
        self.assertEqual(doc["0drp"]["first_node"]["id0_index_zero"], 3)
        self.assertEqual(doc["0drp"]["first_node"]["id1_index_ffffffff"], 3)
        self.assertEqual(doc["navmesh"]["static_sectors"], 4)
        self.assertEqual(doc["navmesh"]["quadtree"], {"nodes": 12, "leaves": 6, "max_depth": 1})
        self.assertEqual(doc["navmesh"]["volumes"],
                         {"pairs": 2, "flags": {"0": 2, "8": 2}, "vertex_counts": {"4": 2, "8": 2}})
        self.assertEqual(doc["navmesh"]["water_bit3"], {"b3_0_w_0": 4, "b3_1_w_1": 2})
        self.assertEqual(doc["grid"]["sectors"], 4)
        self.assertEqual(doc["grid"]["sectors_not_in_navmesh"], 0)
        self.assertEqual(doc["grid"]["navmesh_sectors_not_in_grid"], 0)
        # 16 keys: the 4 roots have children, the 12 others are 0x01 leaves one level down.
        self.assertEqual(doc["grid"]["key_depth"], {"1": 12, "0": 4})
        self.assertEqual(doc["grid"]["inner_nodes"], 4)
        self.assertEqual(doc["grid"]["leaf_payload"], {"low_byte_01": 12})
        self.assertEqual(doc["checks"]["grid"]["keys_octree"], {"pass": 4, "fail": 0})
        self.assertEqual(doc["checks"]["grid"]["inner_payload_ff00"], {"pass": 4, "fail": 0})
        self.assertEqual(doc["psin"]["complete"], 1)

    def test_grid_octree_keys_and_payload_words(self):
        # Root 1 -> 8 -> 64 -> 512 (depths 0..3), plus leaves 9, 10, 65.
        keys = [1, 8, 64, 512, 9, 10, 65]
        words = [0xFF00, 0xFF00, 0xFF00, 0x00007803, 0xFF00, 0x00060F01, 0x00000002]
        good = build_grid(entries=7, keys=keys, words=words)
        bad = [
            build_grid(entries=3, keys=[8, 9, 10]),                       # no root
            build_grid(entries=3, keys=[1, 8, 72]),                       # 72 >> 3 == 9 is absent
            build_grid(entries=3, keys=[1, 8, 16]),                       # 16 has its top bit at 4, not 3 or 6
            build_grid(entries=3, keys=[1, 0, 8]),                        # 0 is not a key
            build_grid(entries=3, keys=[1, 8, 8]),                        # two slots share key 8
            build_grid(entries=3, keys=[1, 8, 9], words=[0x01, 0x01, 0x01]),  # inner root without 0xff00
            build_grid(entries=3, keys=[1, 8, 64], words=[0xFF00, 0x01, 0x01]),  # inner 8 without 0xff00
        ]
        # 17 keys over two groups: 8 sits in slot 15 of group 0 and 9 in group 1, both parents of other keys.
        two_groups = build_grid(groups=2, entries=17, keys=[1] + list(range(64, 78)) + [8, 9])
        # Payload pointer past the end: the table is accepted but the words are not read.
        oob = bytearray(build_grid(entries=3)[0])
        struct.pack_into("<Q", oob, 0x20, BASE + len(oob))
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "grid.rpf")
            make_rpf(arc, good + two_groups + [s for g in bad for s in g] + [bytes(oob)])
            doc = nd.scan(arc, deep=True)
            shallow = nd.scan(arc)
        cg = doc["checks"]["grid"]
        self.assertEqual(cg["payload_in_bounds"]["fail"], 1)
        self.assertEqual(cg["values_bijective"], {"pass": 10, "fail": 0})
        self.assertEqual(cg["keys_octree"], {"pass": 4, "fail": 5})
        self.assertEqual(cg["inner_payload_ff00"], {"pass": 2, "fail": 2})
        self.assertEqual(doc["grid"]["key_depth"], {"2": 16, "1": 5, "0": 2, "3": 1})
        self.assertEqual(doc["grid"]["inner_nodes"], 6)
        self.assertEqual(doc["grid"]["leaf_payload"], {"low_byte_01": 15, "ff00": 1, "low_byte_03": 1, "other": 1})
        self.assertNotIn("keys_octree", shallow["checks"]["grid"])
        self.assertEqual(shallow["grid"]["key_depth"], {})

    def test_multiblock_split_array(self):
        # 380 polygons exceeds one 16 KiB block (floor(16384 / 44) == 372), exercising nblocks == 2.
        poly_sizes = (3,) * 380
        streams = build_navmesh(flags=1, n_verts=2735, poly_sizes=poly_sizes, n_volumes=0)
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "multi.rpf")
            make_rpf(arc, streams)
            doc = nd.scan(arc, deep=True)
        for name, counts in doc["checks"]["navmesh"].items():
            self.assertEqual(counts["fail"], 0, f"navmesh.{name} failed: {counts}")
        self.assertEqual(doc["navmesh"]["split_blocks_max"]["0x70"], 2)
        self.assertEqual(doc["navmesh"]["split_blocks_max"]["0x98"], 2)

    def test_malformed_and_adversarial_inputs(self):
        streams = []
        # 1. Corrupt 0DRP: overlapping q38, bad enum, bad n4e, dirty reserved, NaN/Inf bounds.
        streams += build_0drp(total_size=0x2000, overlap_q38=True, bad_enum=True, override_n4e=99,
                              dirty_reserved=True, nan_bounds=True)
        # 2. 0DRP with mid-loop OOB second q38 block + invalid n4e on a 2-stream boundary page.
        streams += build_0drp(total_size=0x1000, split_at=0x1000, oob_second_q38=True, override_n4e=77)
        # 3. Corrupt navmesh: bad checksum, OOB vertex index, OOB link index, null b8 with dc>0, broken chain, qt cycle.
        streams += build_navmesh(corrupt_checksum=True, corrupt_vidx=True, corrupt_link_idx=True,
                                 corrupt_null_vol=True, corrupt_chain=True, qt_cycle=True)
        # 4. Corrupt grid: bad control byte + duplicate value.
        streams += build_grid(bad_ctrl=True, dup_value=True)
        # 5. Adversarial OOM-DoS grid: u32@0x14 == u32@0x50 == 0xFFFFFFFF with 8 empty groups.
        streams += build_grid(groups=8, entries=0, declared_entries=0xFFFFFFFF)
        # 6. Truncated PSIN (placed last so it does not consume later streams).
        streams += build_psin(truncate=True)

        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "bad.rpf")
            make_rpf(arc, streams)
            doc = nd.scan(arc, deep=True)
            json.dumps(doc, allow_nan=False)

            # Separate archives for MAX_STREAMS-capped resources so they cannot swallow other test fixtures.
            arc_cap_0drp = os.path.join(tmp, "cap_0drp.rpf")
            make_rpf(arc_cap_0drp, build_0drp(total_size=0x2000, bad_bm_streams=100) + [bytes(64)] * (nd.MAX_STREAMS + 2))
            doc_cap_0drp = nd.scan(arc_cap_0drp, deep=True)

            arc_cap_psin = os.path.join(tmp, "cap_psin.rpf")
            big_psin = b"PSIN" + struct.pack(">I", 0x2000) + struct.pack("<II", 256, 0x100) + bytes(64)
            make_rpf(arc_cap_psin, [big_psin] + [bytes(64)] * (nd.MAX_STREAMS + 2))
            doc_cap_psin = nd.scan(arc_cap_psin, deep=True)

        c0 = doc["checks"]["0drp"]
        self.assertEqual(c0["blocks_disjoint"]["fail"], 2)
        self.assertEqual(c0["blocks_in_bounds"]["fail"], 1)
        self.assertEqual(c0["enum_values_dense"]["fail"], 1)
        self.assertEqual(c0["pages_4k_rule"]["fail"], 2)
        self.assertEqual(c0["reserved_zero"]["fail"], 1)
        self.assertEqual(doc["0drp"]["pages_4k_boundary"], {})
        self.assertIsNone(doc["0drp"]["region_bounds"])

        self.assertEqual(doc_cap_0drp["resources"]["0drp"]["incomplete"], 1)
        self.assertEqual(doc_cap_0drp["checks"]["0drp"]["block_map"]["fail"], 1)
        self.assertEqual(doc_cap_psin["resources"]["psin"]["incomplete"], 1)
        self.assertEqual(doc_cap_psin["psin"]["complete"], 0)

        cn = doc["checks"]["navmesh"]
        self.assertEqual(cn["data_size_identity"]["fail"], 1)
        self.assertEqual(cn["indices_below_vertices"]["fail"], 1)
        self.assertEqual(cn["links_in_bounds"]["fail"], 1)
        self.assertEqual(cn["volumes_in_bounds"]["fail"], 1)
        self.assertEqual(cn["volumes_valid"]["fail"], 1)
        self.assertEqual(cn["polygons_chain"]["fail"], 1)
        self.assertEqual(cn["quadtree_valid"]["fail"], 1)

        # A malformed second volume pair must not partially increment navmesh.volumes counters.
        tile = bytearray(build_navmesh(n_volumes=4)[0])
        vstart = nd.offset_of(struct.unpack_from("<Q", tile, 0xB8)[0])
        tile[vstart + 3 * 96 + 0x14] = 7
        st = nd.new_stats()["navmesh"]
        _, chk = nd.decode_navmesh(bytes(tile), 1, st, True)
        self.assertFalse(chk["volumes_valid"])
        self.assertEqual(st["volumes"]["pairs"], 0)
        self.assertEqual(dict(st["volumes"]["flags"]), {})
        self.assertEqual(dict(st["volumes"]["vertex_counts"]), {})

        cg = doc["checks"]["grid"]
        self.assertEqual(cg["control_bytes"]["fail"], 1)
        self.assertEqual(cg["occupied_equals_entries"]["fail"], 2)
        self.assertEqual(cg["values_bijective"]["fail"], 2)

        self.assertEqual(doc["resources"]["psin"]["incomplete"], 1)
        self.assertEqual(doc["psin"]["complete"], 0)

    def test_psin_first_section_at_eof_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            with open(os.path.join(carve, "000000000000.bin"), "wb") as f:
                f.write(struct.pack(">4sI", b"PSIN", 64) + bytes(56))
            out = os.path.join(tmp, "out.json")
            proc = subprocess.run([sys.executable, "-I", TOOL, carve, "--quiet", "--out", out],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)

        self.assertEqual(doc["resources"]["psin"]["resources"], 1)
        self.assertEqual(doc["resources"]["psin"]["incomplete"], 1)
        self.assertEqual(doc["psin"]["complete"], 0)

    def test_header_only_psin_does_not_swallow_the_next_resource(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            with open(os.path.join(carve, "000000000000.bin"), "wb") as f:
                f.write(struct.pack(">4sI", b"PSIN", 64) + bytes(56))
            with open(os.path.join(carve, "000000000040.bin"), "wb") as f:
                f.write(struct.pack(">4sI", b"PSIN", 64) + bytes(56) + struct.pack(">4sI", b"SECT", 64) + bytes(56))
            out = os.path.join(tmp, "out.json")
            proc = subprocess.run([sys.executable, "-I", TOOL, carve, "--quiet", "--out", out],
                                  capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)

        self.assertEqual(doc["resources"]["psin"]["resources"], 2)
        self.assertEqual(doc["resources"]["psin"]["incomplete"], 1)
        self.assertEqual(doc["psin"]["complete"], 1)

    def test_carved_directory_mode_and_window_limit(self):
        sector = (260 << 9) | 250
        p0, p1 = build_0drp(total_size=0x1000, split_at=0x1000, node_count=2, override_n4e=0)
        nav = build_navmesh(sector=sector)[0]
        g0, g1 = build_grid(sector=sector, two_streams=True)
        with tempfile.TemporaryDirectory() as tmp:
            empty_dir = os.path.join(tmp, "empty")
            os.mkdir(empty_dir)
            edoc = nd.scan(empty_dir, deep=True)
            self.assertEqual(edoc["resources"], {})

            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            for off, blob in ((0x001000, p0), (0x002000, p1), (0x003000, nav), (0x004000, g0), (0x005000, g1)):
                with open(os.path.join(carve, f"{off:012x}.bin"), "wb") as f:
                    f.write(blob)
            doc = nd.scan(carve, deep=True)
            self.assertEqual(doc["mode"], "directory")
            self.assertEqual(doc["resources"]["0drp"]["resources"], 1)
            self.assertEqual(doc["resources"]["navmesh"]["resources"], 1)
            self.assertEqual(doc["resources"]["grid"]["resources"], 1)
            self.assertEqual(doc["0drp"]["pages_4k_boundary"], {"zero": 1})
            for cls, table in doc["checks"].items():
                for name, counts in table.items():
                    self.assertEqual(counts["fail"], 0, f"{cls}.{name} failed: {counts}")

            # Window --limit stops after a resource whose >=64 B continuation crosses --limit.
            arc = os.path.join(tmp, "win.rpf")
            offs = make_rpf(arc, [g0, g1, nav])
            win = nd.scan(arc, start=offs[0], limit=offs[1] - offs[0], deep=True)
            self.assertEqual(win["resources"]["grid"]["resources"], 1)
            self.assertNotIn("navmesh", win["resources"])
            # The continuation starts at the window's end: decoded as part of the grid, but carved past the
            # window, so the carver counts match navmesh_index.py on the same window (#766).
            self.assertEqual(win["carver_streams"]["streams"], 1)
            self.assertEqual(win["carver_streams"]["past_window"], 1)
            c = win["carver_streams"]
            self.assertEqual(c["stored"] + c["zero_padding"] + c["unread"], win["span"])

    def test_limit_inside_a_stream_counts_nothing_past_the_window(self):
        """--limit ending inside the first of two streams: nothing is carved, and the byte classes add up to
        the span, as navmesh_index.py reports for the same window."""
        import hashlib
        noisy = lambda seed: b"".join(hashlib.sha256(bytes([seed, i])).digest() for i in range(128))
        with tempfile.TemporaryDirectory() as tmp:
            arc = os.path.join(tmp, "w.rpf")
            offs = make_rpf(arc, [noisy(1), noisy(2)])
            doc = nd.scan(arc, start=offs[0], limit=0x100)
            c = doc["carver_streams"]
            self.assertEqual((c["streams"], c["stored"]), (0, 0))
            self.assertEqual(c["stored"] + c["zero_padding"] + c["unread"], doc["span"])
            self.assertEqual(c["past_window"], 1)

    def test_cli_rejects_shared_report_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "nav.rpf")
            make_rpf(archive, build_0drp())
            output = os.path.join(tmp, "nav.navmesh-decode.json")
            result = subprocess.run(
                [sys.executable, "-I", TOOL, archive, "--csv", output, "--quiet"],
                cwd=tmp, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("--out and --csv must use different paths", result.stderr)
            self.assertFalse(os.path.exists(output))

    def test_cli_creates_missing_output_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "nav.rpf")
            make_rpf(archive, build_0drp())
            out_json = os.path.join(tmp, "reports", "deep", "nav.json")
            out_csv = os.path.join(tmp, "tables", "nav.csv")
            result = subprocess.run(
                [sys.executable, "-I", TOOL, archive, "--out", out_json, "--csv", out_csv, "--quiet"],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            with open(out_json, encoding="utf-8") as f:
                self.assertEqual(json.load(f)["resources"]["0drp"]["resources"], 1)
            with open(out_csv, newline="", encoding="utf-8") as f:
                self.assertEqual(len(list(csv.DictReader(f))), 1)

    def test_cli_reports_a_blocked_report_parent(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "nav.rpf")
            make_rpf(archive, build_0drp())
            blocker = os.path.join(tmp, "blocked")
            with open(blocker, "wb") as f:
                f.write(b"keep")
            for flag in ("--out", "--csv"):
                with self.subTest(flag=flag):
                    target = os.path.join(blocker, "nav" + (".json" if flag == "--out" else ".csv"))
                    command = [sys.executable, "-I", TOOL, archive, flag, target, "--quiet"]
                    result = subprocess.run(command, cwd=tmp, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn(target, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)
            with open(blocker, "rb") as f:
                self.assertEqual(f.read(), b"keep")

    def test_cli_refuses_to_overwrite_its_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "nav.rpf")
            make_rpf(archive, build_0drp())
            with open(archive, "rb") as f:
                original = f.read()
            for flag in ("--out", "--csv"):
                result = subprocess.run(
                    [sys.executable, "-I", TOOL, archive, flag, archive, "--quiet"],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("would overwrite the input", result.stderr)
                self.assertNotIn("Traceback", result.stderr)
                with open(archive, "rb") as f:
                    self.assertEqual(f.read(), original, f"{flag} touched the input")

    def test_cli_protects_carved_inputs_from_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            source = os.path.join(carve, "000000.bin")
            original = build_navmesh()[0]
            with open(source, "wb") as f:
                f.write(original)
            alias = os.path.join(tmp, "alias.csv")
            targets = [("--out", source), ("--csv", source)]
            try:
                os.link(source, alias)
            except (OSError, NotImplementedError) as e:
                with self.subTest(flag="--csv", target=alias):
                    self.skipTest(f"hard links unavailable: {e}")
            else:
                targets.append(("--csv", alias))

            out = os.path.join(carve, "report.json")
            csv_path = os.path.join(carve, "report.csv")
            command = [sys.executable, "-I", TOOL, carve, "--quiet"]
            for _ in range(2):
                result = subprocess.run([*command, "--out", out, "--csv", csv_path],
                                        capture_output=True, text=True, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)

            for flag, target in targets:
                with self.subTest(flag=flag, target=target):
                    with open(source, "wb") as f:
                        f.write(original)
                    result = subprocess.run([*command, flag, target], cwd=tmp,
                                            capture_output=True, text=True, check=False)
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("would overwrite the input", result.stderr)
                    with open(source, "rb") as f:
                        self.assertEqual(f.read(), original)

    def test_reports_do_not_replace_the_carve_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            carve = os.path.join(tmp, "carve")
            os.mkdir(carve)
            manifest = os.path.join(carve, "index.json")
            before = b'{"files": []}'
            with open(manifest, "wb") as f:
                f.write(before)
            for tool in ("navmesh_decode.py", "navmesh_index.py"):
                for flag in ("--out", "--csv"):
                    with self.subTest(tool=tool, flag=flag):
                        result = subprocess.run(
                            [sys.executable, "-I", os.path.join(os.path.dirname(TOOL), tool),
                             carve, flag, manifest],
                            cwd=tmp, capture_output=True, text=True,
                        )
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertIn("would overwrite", result.stderr)
                        self.assertIn("index.json", result.stderr)
                        with open(manifest, "rb") as f:
                            self.assertEqual(f.read(), before)

    def test_cli_rejects_bad_arguments_and_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = os.path.join(tmp, "empty.rpf")
            with open(empty, "wb"):
                pass
            tiny = os.path.join(tmp, "tiny.rpf")
            with open(tiny, "wb") as f:
                f.write(b"9FPR123")
            bad_ver = os.path.join(tmp, "badver.rpf")
            with open(bad_ver, "wb") as f:
                f.write(b"6FPR" + bytes(32))
            arc = os.path.join(tmp, "ok.rpf")
            make_rpf(arc, build_0drp())

            for args in (
                [arc, "--limit", "0"],
                [arc, "--limit", "-5"],
                [arc, "--start", "-1"],
                [arc, "--start", "999999"],
                [empty],
                [tiny],
                [bad_ver],
                [tmp, "--limit", "64"],
            ):
                proc = subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)
                self.assertNotEqual(proc.returncode, 0, f"expected failure for {args}")
                self.assertNotIn("Traceback", proc.stderr)


class OutputPathTest(unittest.TestCase):
    """A report path that is already a folder is a usage error, not a PermissionError traceback."""

    def test_out_that_is_a_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = os.path.join(tmp, "carved"), os.path.join(tmp, "reports")
            os.makedirs(source)
            os.makedirs(out)
            r = subprocess.run([sys.executable, "-I", TOOL, source, "--out", out],
                               capture_output=True, text=True, timeout=120)
            self.assertEqual(r.returncode, 2, r.stderr)
            self.assertIn("--out is a directory", r.stderr)
            self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
