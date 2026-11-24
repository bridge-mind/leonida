#!/usr/bin/env python3
"""Unit tests for tools/navmesh_links.py using synthetic navmesh tiles only."""

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import navmesh_decode as nd
from tools import navmesh_links as nl

TOOL = os.path.join(ROOT, "tools", "navmesh_links.py")
BASE = nd.BASE


def make_rpf(path, streams, pad_between=16):
    body = bytearray()
    for s in streams:
        body.extend(zlib.compress(s))
        body.extend(b"\x00" * pad_between)
    hdr = bytearray(0x40)
    hdr[0:4] = b"9FPR"
    struct.pack_into("<I", hdr, 4, len(hdr))
    with open(path, "wb") as fh:
        fh.write(hdr)
        fh.write(body)


def build_tile(
    *,
    flags=3,
    sector=(260 << 9) | 250,
    n_verts=8,
    poly_sizes=(4, 4, 3),
    poly_link_counts=(2, 0, 2),
    poly_hi9_flags=(1, 0, 64),
    edges=None,
    links=None,
    link_indices=None,
    two_streams=False,
    corrupt_edge_oob=False,
    corrupt_edge_sector=False,
    corrupt_pool_vert_sum=False,
    corrupt_link_heading=False,
    corrupt_link_pad=False,
    corrupt_link_sector=False,
    corrupt_link_poly_oob=False,
    corrupt_chain_start=False,
    corrupt_chain_total=False,
    corrupt_b0_endpoint=False,
):
    buf = bytearray(0xF0)
    has_links = bool(flags & 2)
    n_polys = len(poly_sizes)
    pool = sum(poly_sizes)

    if not has_links:
        poly_link_counts = (0,) * n_polys
        links = []
        link_indices = []
    else:
        if links is None:
            links = [
                # link 0: intra-tile poly 0 <-> poly 2
                (100, 200, 300, 400, 500, 600, 0, 2, sector, sector, 0x0200, 0x0050, 63, 0),
                # link 1: cross-tile poly 0 (s0=sector) -> poly 999 (s1=sector+1)
                (110, 210, 310, 410, 510, 610, 0, 999, sector, sector + 1, 0x0400, 0x0040, 127, 0),
                # link 2: cross-tile poly 888 (s0=sector-1) -> poly 2 (s1=sector)
                (120, 220, 320, 420, 520, 620, 888, 2, sector - 1, sector, 0x0000, 0x0030, 254, 0),
            ]
        if link_indices is None:
            # poly 0 gets links [0, 1], poly 1 gets [], poly 2 gets [0, 2]
            link_indices = [0, 1, 0, 2]

    d4 = len(links)
    d8 = len(link_indices)
    sx, sy = sector & 0x1FF, sector >> 9
    x0, y0, dx, dy, dz = (sx - 256) * 128.0, (sy - 256) * 128.0, 128.0, 128.0, 16.0

    struct.pack_into("<II", buf, 0x10, flags, nd.DATA15)
    struct.pack_into("<16f", buf, 0x20, *nd.IDENTITY)
    struct.pack_into("<3f", buf, 0x60, dx, dy, dz)
    struct.pack_into("<I", buf, 0x90, pool)
    data_size = 6 * n_verts + 14 * pool + 44 * n_polys + 32 * d4
    struct.pack_into("<8I", buf, 0xC0, n_verts, n_polys, sector, data_size, 0, d4, d8, 0)
    struct.pack_into("<I", buf, 0xE8, 0x09090516)

    cur = 0xF0

    def alloc(size, align=16):
        nonlocal cur
        cur = (cur + align - 1) & ~(align - 1)
        off = cur
        cur += size
        if cur > len(buf):
            buf.extend(bytes(cur - len(buf)))
        return off

    if has_links:
        la = alloc(32 * d4)
        lb = alloc(2 * d8)
        for i, lk in enumerate(links):
            rec = list(lk)
            if i == d4 - 1:
                if corrupt_link_heading:
                    rec[12] = 256
                if corrupt_link_pad:
                    rec[13] = 1
                if corrupt_link_sector:
                    rec[8] = sector + 5
                    rec[9] = sector + 6
                if corrupt_link_poly_oob:
                    rec[8] = sector
                    rec[6] = n_polys
            struct.pack_into("<6HHHIIHHHH", buf, la + 32 * i, *rec)
        for k, li in enumerate(link_indices):
            val = 1 if (corrupt_b0_endpoint and k == d8 - 1) else li
            struct.pack_into("<H", buf, lb + 2 * k, val)
        struct.pack_into("<Q", buf, 0xA8, BASE + la)
        struct.pack_into("<Q", buf, 0xB0, BASE + lb)

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

    verts = bytearray(6 * n_verts)
    vidx = bytearray(2 * pool)
    for k in range(pool):
        struct.pack_into("<H", vidx, 2 * k, k % n_verts)

    adjs = bytearray(12 * pool)
    if edges is None:
        edges = []
        for k in range(pool):
            if k % 3 == 0:
                edges.append((0xFFFFFFFF, 0xFFFF, 0xFFFF, 0x30))
            else:
                edges.append((sector, (k + 1) % n_polys, 0xFF00, 0x10))
    for k, ed in enumerate(edges):
        asec, apoly, meta16, flags32 = ed
        if k == pool - 1:
            if corrupt_edge_oob:
                asec, apoly = sector, n_polys
            elif corrupt_edge_sector:
                asec, apoly = sector + 1, 0
        struct.pack_into("<IHHI", adjs, 12 * k, asec, apoly, meta16, flags32)

    polys = bytearray(44 * n_polys)
    vpos = 0
    lpos = 0
    for i, nv in enumerate(poly_sizes):
        packed_nv = (nv - 1) if (corrupt_pool_vert_sum and i == n_polys - 1) else nv
        struct.pack_into("<II", polys, 44 * i, sector, (vpos << 4) | packed_nv)
        vpos += nv
        lcnt = poly_link_counts[i]
        lstart = lpos + 1 if (corrupt_chain_start and i == n_polys - 1) else lpos
        if corrupt_chain_total and i == n_polys - 1 and lcnt > 0:
            lcnt -= 1
        hi9 = poly_hi9_flags[i] if i < len(poly_hi9_flags) else 0
        w20 = (hi9 << 23) | ((lstart & 0x1FFFF) << 6) | (lcnt & 0x3F)
        struct.pack_into("<I", polys, 44 * i + 0x20, w20)
        lpos += poly_link_counts[i]

    put_split(0x70, 6, verts, n_verts)
    put_split(0x80, 2, vidx, pool)
    put_split(0x88, 12, adjs, pool)
    put_split(0x98, 44, polys, n_polys)

    root = alloc(0x60)
    struct.pack_into("<3f", buf, root, x0, y0, 0.0)
    struct.pack_into("<3f", buf, root + 0x10, x0 + dx, y0 + dy, dz)
    struct.pack_into("<Q", buf, 0xA0, BASE + root)

    bm = alloc(32)
    struct.pack_into("<Q", buf, 0x08, BASE + bm)
    struct.pack_into("<4Q", buf, bm, 0, 2 if two_streams else 1, 0, 0)
    raw = bytes(buf[:cur])
    if two_streams:
        mid = len(raw) - 64
        return [raw[:mid], raw[mid:]]
    return [raw]


class NavmeshLinksTest(unittest.TestCase):
    def test_reciprocity(self):
        sector = 17
        open_edge = (0xFFFFFFFF, 0xFFFF, 0, 0)
        def local(target):
            return (sector, target, 0, 0)

        # Tile A has three 0->1 and one 1->0. Self edges match themselves.
        a = [local(1)] * 3 + [local(0), local(0), local(1), open_edge,
                               local(2), (sector + 1, 0, 0, 0)]
        # Tile B has the opposite imbalance. Combining tiles first would hide it.
        b = [local(1)] + [local(0)] * 3
        first = build_tile(flags=1, sector=sector, poly_sizes=(4, 5), edges=a)
        second = build_tile(flags=1, sector=sector, poly_sizes=(1, 3), edges=b)
        bad = bytearray(second[0])
        polygons = nd.split_array(bad, 0x98, 44, 2)
        struct.pack_into("<I", bad, polygons[0][0] + 44 + 4, 3)  # overlapping span

        with tempfile.TemporaryDirectory() as tmp:
            archive = os.path.join(tmp, "tiles.rpf")
            make_rpf(archive, first + second + [bad])
            default = nl.scan(archive)
            result = subprocess.run([sys.executable, "-I", TOOL, archive, "--reciprocity"],
                                    check=True, capture_output=True, text=True)
        doc = json.loads(result.stdout)
        self.assertEqual(doc.pop("reciprocity"), {
            "tiles_counted": 2, "same_tile": 10, "excess_directed": 4,
            "invalid_targets": 2, "unlinked": 1,
        })
        self.assertEqual(doc, default)
        self.assertEqual(doc["tiles"], 3)

    def test_endpoint_agreement_cli(self):
        sector = 42
        open_edge = (0xffffffff, 0xffff, 0, 0)
        # Each slot uses a distinct index; equality must use all six coordinate bytes.
        def tile(left, right, local_left=(0,), local_right=(0,)):
            values = left + right
            edges = [((sector, 1, 0, 0) if j in local_left else open_edge)
                     for j in range(len(left))]
            edges.extend((sector, 0, 0, 0) if j in local_right else open_edge
                         for j in range(len(right)))
            buf = bytearray(build_tile(flags=1, sector=sector, n_verts=len(values),
                                       poly_sizes=(len(left), len(right)), edges=edges)[0])
            base = nd.split_array(buf, 0x70, 6, len(values))[0][0]
            for j, z in enumerate(values):
                struct.pack_into("<3H", buf, base + 6 * j, 5, 7, z)
            return buf

        samples = [
            tile([1, 2, 3], [2, 1, 4]),        # outgoing reversed
            tile([1, 2, 3], [1, 2, 4]),        # outgoing same direction
            tile([1, 1, 3], [1, 1, 4]),        # unique zero-length pair
            tile([1, 2, 1, 2], [2, 1, 4], (0, 2)),  # duplicate against one return
            tile([1, 3, 2], [2, 4, 1]),        # incoming reversed across wrap
        ]
        other = build_tile(flags=1, sector=sector, poly_sizes=(3,), edges=[
            (sector, 0, 0, 0), (sector, 1, 0, 0), (sector + 1, 0, 0, 0),
        ])[0]
        samples.append(other)
        bad = bytearray(samples[0])
        offset = nd.split_array(bad, 0x98, 44, 2)[0][0]
        struct.pack_into("<I", bad, offset + 44 + 4, 3)
        bad_index = bytearray(samples[0])
        offset = nd.split_array(bad_index, 0x80, 2, 6)[0][0]
        struct.pack_into("<H", bad_index, offset, 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiles.rpf")
            make_rpf(path, samples + [bad, bad_index])
            base = nl.scan(path, reciprocity=True)
            proc = subprocess.run([sys.executable, "-I", TOOL, path, "--reciprocity",
                                   "--endpoint-agreement"], capture_output=True, check=True)
        report = json.loads(proc.stdout)
        result = report.pop("endpoint_agreement")
        self.assertEqual(report, base)
        self.assertEqual(result["tiles_counted"], 6)
        self.assertEqual(result["self_references"], 1)
        self.assertEqual(result["invalid_targets"], 2)
        self.assertEqual(result["outgoing"], {
            "reversed": 2, "same_direction": 2, "degenerate": 2,
            "repeated": 3, "no_exact_return": 2,
        })
        self.assertEqual(result["incoming"]["reversed"], 2)
        self.assertEqual(result["incoming"]["no_exact_return"], 9)
        self.assertEqual(result["unlinked"], 20)

    def test_relative_winding_cli(self):
        a, b, above, below = (1, 2), (5, 2), (3, 4), (3, 0)
        left = [a, b, above]
        opened = (0xffffffff, 0xffff, 0, 0)

        def pair(right, extent=128):
            edges = [(42, 1, 0, 0), opened, opened, (42, 0, 0, 0)] + [opened] * (len(right) - 1)
            points = left + right
            blob = bytearray(build_tile(flags=1, sector=42, n_verts=len(points),
                                        poly_sizes=(3, len(right)), edges=edges)[0])
            struct.pack_into("<2f", blob, 0x60, extent, 128)
            base = nd.split_array(blob, 0x70, 6, len(points))[0][0]
            for j, (x, y) in enumerate(points):
                struct.pack_into("<3H", blob, base + 6 * j, x, y, 10)
            return blob

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "tiles.rpf")
            make_rpf(path, [pair([a, b, below]), pair([a, b, above]),
                            pair([b, a, below]), pair([b, a, above]),
                            pair([a, b, (3, 2)]), pair([b, a, above], extent=0)])
            baseline = nl.scan(path, endpoint_agreement=True)
            proc = subprocess.run([sys.executable, "-I", TOOL, path, "--relative-winding"],
                                  capture_output=True, check=True)
        report = json.loads(proc.stdout)
        winding = report["endpoint_agreement"].pop("relative_winding")
        self.assertEqual(report, baseline)
        self.assertEqual(winding, {c: {"opposite": 2, "same": 2, "ambiguous": 2}
                                   for c in ("same_direction", "reversed")})

    def test_coplanar_winding_cli(self):
        a, b, c = (0, 2, 2), (0, 4, 2), (0, 4, 6)
        opened = (0xffffffff, 0xffff, 0, 0)
        rings = ([a, b, (0, 3, 4)], [a, b, (0, 3, 0)], [a, b, c, (0, 3, 4)],
                 [a, b, (2, 3, 3)], [a, b, (2, 3, 3), (0, 2, 4)],
                 [a, b, b], [a, b], [a, b, (0, 5, 2)], [a, b, (0, 2, 4), (0, 5, 4)])
        tiles = []
        for right in rings:
            points = [a, b, c] + right
            edges = [(42, 1, 0, 0), opened, opened, (42, 0, 0, 0)] + [opened] * (len(right) - 1)
            tile = bytearray(build_tile(flags=1, sector=42, n_verts=len(points),
                                        poly_sizes=(3, len(right)), edges=edges)[0])
            struct.pack_into("<3f", tile, 0x60, 3, 4, 7)
            start = nd.split_array(tile, 0x70, 6, len(points))[0][0]
            for i, point in enumerate(points):
                struct.pack_into("<3H", tile, start + 6 * i, *point)
            tiles.append(tile)
        # The vertical YZ pair survives a collapsed X axis. Losing Y collapses its shared edge.
        for extents in ((0, 4, 7), (3, 0, 7), (3, 4, float("nan")), (3, 4, -1)):
            tile = bytearray(tiles[0])
            struct.pack_into("<3f", tile, 0x60, *extents)
            tiles.append(tile)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "pairs.rpf")
            make_rpf(path, tiles)
            baseline = nl.scan(path, relative_winding=True)
            proc = subprocess.run([sys.executable, "-I", TOOL, path, "--coplanar-winding"],
                                  capture_output=True, check=True)
        report = json.loads(proc.stdout)
        counts = report["endpoint_agreement"].pop("coplanar_winding")
        self.assertEqual(report, baseline)
        self.assertEqual(counts, {"same": 4, "opposite": 2, "coincident": 2,
                                  "different_planes": 2, "undefined_ring": 16})

    def test_projected_ring_geometry(self):
        concave = [(0, 0), (4, 0), (2, 1), (4, 3), (0, 3)]
        self.assertEqual(nl._xy_winding(concave), 1)
        self.assertEqual(nl._xy_winding(concave[::-1]), -1)
        self.assertEqual(nl._xy_winding([(0, 0), (2, 0), (4, 0), (0, 3)]), 1)
        for ring in ([], [(0, 0), (1, 0)], [(0, 0), (1, 0), (2, 0)],
                     [(0, 0), (4, 0), (2, 0), (0, 3)],          # collinear backtrack
                     [(0, 0), (4, 3), (0, 2), (2, 0)],          # crossed with nonzero area
                     [(0, 0), (4, 0), (4, 4), (2, 0), (0, 4)],  # touch
                     [(0, 0), (3, 0), (2, 1), (3, 0), (0, 4)]):
            with self.subTest(ring=ring):
                self.assertEqual(nl._xy_winding(ring), 0)

    def test_valid_rpf_and_carved_dir(self):
        s1 = build_tile(flags=3)
        s2 = build_tile(flags=1, sector=(260 << 9) | 251, two_streams=True)
        streams = s1 + s2

        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "navmeshes.rpf")
            make_rpf(rpf, streams)
            out_json = os.path.join(tmp, "out.json")
            proc = subprocess.run(
                [sys.executable, "-I", TOOL, rpf, "--out", out_json, "--quiet"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            with open(out_json, encoding="utf-8") as fh:
                doc = json.load(fh)

            carve = os.path.join(tmp, "carved")
            os.mkdir(carve)
            for idx, st in enumerate(streams):
                with open(os.path.join(carve, f"{idx:04d}.bin"), "wb") as fh:
                    fh.write(st)
            doc_carve = nl.scan(carve)

        self.assertEqual(doc, doc_carve)
        self.assertEqual(doc["tiles"], 2)
        for name, counts in doc["checks"].items():
            self.assertEqual(counts, {"pass": 2, "fail": 0}, f"{name} failed: {counts}")
        self.assertEqual(doc["edges"]["total"], 22)
        self.assertEqual(doc["edges"]["same_tile"], 14)
        self.assertEqual(doc["edges"]["unlinked"], 8)
        self.assertEqual(doc["polygons"]["total"], 6)
        self.assertEqual(doc["polygons"]["max_links_per_poly"], 2)
        self.assertEqual(doc["polygons"]["w20_hi9_flags"], {"1": 2, "0": 2, "64": 2})
        self.assertEqual(doc["special_links"]["link_tiles"], 1)
        self.assertEqual(doc["special_links"]["total"], 3)
        self.assertEqual(doc["special_links"]["indices_total"], 4)
        self.assertEqual(doc["special_links"]["intra_tile"], 1)
        self.assertEqual(doc["special_links"]["cross_tile_s0"], 1)
        self.assertEqual(doc["special_links"]["cross_tile_s1"], 1)

    def test_malformed_inputs_and_no_partial_counter_leak(self):
        sector = (260 << 9) | 250
        # Counterexample where polygon 1 lists a cross-tile link whose index-1 endpoint is in sector+1
        # while its local endpoint (poly 2 in sector) does not list it.
        unlisted_local_end = build_tile(
            sector=sector,
            links=[
                (100, 200, 300, 400, 500, 600, 0, 2, sector, sector, 0x0200, 0x0050, 63, 0),
                (110, 210, 310, 410, 510, 610, 1, 2, sector + 1, sector, 0x0400, 0x0040, 127, 0),
            ],
            link_indices=[0, 1, 0],
            poly_link_counts=(1, 1, 1),
        )[0]
        cases = [
            ("edges_valid", build_tile(corrupt_edge_oob=True)[0]),
            ("edges_valid", build_tile(corrupt_edge_sector=True)[0]),
            ("edges_valid", build_tile(corrupt_pool_vert_sum=True)[0]),
            ("special_links_valid", build_tile(corrupt_link_heading=True)[0]),
            ("special_links_valid", build_tile(corrupt_link_pad=True)[0]),
            ("special_links_valid", build_tile(corrupt_link_sector=True)[0]),
            ("special_links_valid", build_tile(corrupt_link_poly_oob=True)[0]),
            ("poly_link_chain", build_tile(corrupt_chain_start=True)[0]),
            ("poly_link_chain", build_tile(corrupt_chain_total=True)[0]),
            ("link_indices_endpoints", build_tile(corrupt_b0_endpoint=True)[0]),
            ("link_indices_endpoints", unlisted_local_end),
        ]
        for failed_check, blob in cases:
            with self.subTest(failed_check=failed_check):
                st = nl.new_stats()
                chk = nl.verify_tile(blob, st)
                self.assertFalse(chk[failed_check])
                if failed_check == "edges_valid":
                    self.assertEqual(st["edges"]["total"], 0)
                if failed_check == "special_links_valid":
                    self.assertEqual(st["special_links"]["total"], 0)
                if failed_check == "poly_link_chain":
                    self.assertEqual(st["polygons"]["total"], 0)

    def test_cli_validation_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            rpf = os.path.join(tmp, "nav.rpf")
            make_rpf(rpf, build_tile())
            rpf_upper = os.path.join(tmp, "NAV.RPF")
            carved_bin = os.path.join(tmp, "0000.bin")
            with open(carved_bin, "wb") as fh:
                fh.write(build_tile()[0])
            missing = os.path.join(tmp, "no_such.rpf")
            for argv in (
                [missing],
                [rpf, "--start", "-1"],
                [rpf, "--limit", "0"],
                [tmp, "--start", "16"],
                [tmp, "--limit", "64"],
                [rpf, "--out", tmp],
                [rpf, "--out", rpf],
                [tmp, "--out", carved_bin],
            ):
                rc = nl.main(argv)
                self.assertEqual(rc, 2, f"expected exit code 2 for {argv}")
            self.assertEqual(os.path.getsize(carved_bin), len(build_tile()[0]))
            if os.name == "nt" or sys.platform == "darwin":
                self.assertEqual(nl.main([rpf, "--out", rpf_upper]), 2)
                with open(rpf, "rb") as fh:
                    self.assertEqual(fh.read(4), b"9FPR")


if __name__ == "__main__":
    unittest.main()
