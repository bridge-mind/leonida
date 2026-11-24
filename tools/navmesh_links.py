#!/usr/bin/env python3
"""Decode and verify navmesh +0x88 adjacent-polygon edges and +0xA8/+0xB0 special links.

Reads a retail RPF9 navmeshes.rpf archive (or a directory of carved .bin streams)
in-place and validates the 12-byte adjacent-polygon edge records (+0x88, count u32@0x90),
the 32-byte special-link records (+0xA8, count u32@0xD4), the u16 special-link index
table (+0xB0, count u32@0xD8), and the polygon +0x20 special-link prefix-sum chain
across all 0x00010015 navmesh tiles. Emits aggregate counts and invariant pass/fail
totals only (no game data).
Usage: `python -I tools/navmesh_links.py --help` lists the options.
"""

import argparse
from collections import Counter
import json
import math
import mmap
import os
import struct
import sys

try:
    from tools import navmesh_decode as nd
    from tools.report_paths import same_report_path
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import navmesh_decode as nd
    from report_paths import same_report_path


CHECK_NAMES = (
    "edges_valid",
    "poly_link_chain",
    "special_links_valid",
    "link_indices_endpoints",
)


def _counter_top(counter, limit=32):
    return {str(k): v for k, v in counter.most_common(limit)}


def _count_reciprocity(b, polygons, edges, sector, polygon_count, totals):
    """Match directed polygon counts within one tile whose pool sizes agree."""
    records = (record for offset, count in edges
               for record in struct.iter_unpack("<IH6x", memoryview(b)[offset:offset + count * 12]))
    directed = Counter()
    source = position = same = invalid = unlinked = 0
    for offset, count in polygons:
        for i in range(count):
            packed = nd._u32(b, offset + 44 * i + 4)
            if packed >> 4 != position:
                return  # Overlapping or missing spans make edge ownership ambiguous.
            for _ in range(packed & 15):
                target_sector, target = next(records)
                if target_sector == 0xFFFFFFFF and target == 0xFFFF:
                    unlinked += 1
                elif target_sector == sector and target < polygon_count:
                    directed[source, target] += 1
                    same += 1
                else:
                    invalid += 1
            position += packed & 15
            source += 1

    excess = sum(max(0, n - directed[target, source])
                 for (source, target), n in directed.items())
    totals["tiles_counted"] += 1
    totals["same_tile"] += same
    totals["excess_directed"] += excess
    totals["invalid_targets"] += invalid
    totals["unlinked"] += unlinked


ENDPOINT_CLASSES = (
    "reversed", "same_direction", "degenerate", "repeated", "no_exact_return",
)


def _turn(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _xy_winding(points):
    """Sign of twice the area of a simple projected ring, or zero if ambiguous."""
    n = len(points)
    if n < 3:
        return 0
    if n == 3:
        area = _turn(*points)
        return (area > 0) - (area < 0)
    if len(set(points)) != n:
        return 0
    area = 0
    for i, a in enumerate(points):
        b = points[(i + 1) % n]
        area += a[0] * b[1] - b[0] * a[1]
        prev = points[i - 1]
        if (_turn(prev, a, b) == 0 and
                (prev[0] - a[0]) * (b[0] - a[0]) + (prev[1] - a[1]) * (b[1] - a[1]) > 0):
            return 0  # Adjacent segments double back over one another.
        for j in range(i + 2, n if i else n - 1):
            c, d = points[j], points[(j + 1) % n]
            if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]) or
                    max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
                continue
            if _turn(a, b, c) * _turn(a, b, d) <= 0 and _turn(c, d, a) * _turn(c, d, b) <= 0:
                return 0
    return (area > 0) - (area < 0)


def _plane_ring(points):
    """Return a canonical plane normal, winding sign and undirected corner cycle."""
    if len(points) < 3 or len(set(points)) != len(points):
        return None
    a, b = points[:2]
    dx, dy, dz = (v - u for u, v in zip(a, b))
    for c in points[2:]:
        ex, ey, ez = (v - u for u, v in zip(a, c))
        normal = (dy * ez - dz * ey, dz * ex - dx * ez, dx * ey - dy * ex)
        if any(normal):
            break
    else:
        return None
    if any(sum(n * (v - u) for n, v, u in zip(normal, point, a)) for point in points):
        return None
    axis = next(i for i, n in enumerate(normal) if n)
    divisor = math.gcd(*normal) * (1 if normal[axis] > 0 else -1)
    plane = tuple(n // divisor for n in normal)
    projected = [(p[(axis + 1) % 3], p[(axis + 2) % 3]) for p in points]
    sign = _xy_winding(projected)
    if not sign:
        return None
    corners = [p for i, p in enumerate(projected)
               if _turn(projected[i - 1], p, projected[(i + 1) % len(points)])]
    boundary = frozenset(tuple(sorted((p, corners[i - 1]))) for i, p in enumerate(corners))
    return plane, sign, boundary


def _plane_relation(left, right):
    if left is None or right is None:
        return "undefined_ring"
    # The exact shared edge supplies a common point, so parallel planes coincide.
    if left[0] != right[0]:
        return "different_planes"
    if left[2] == right[2]:
        return "coincident"
    return "same" if left[1] == right[1] else "opposite"


def _endpoint_counts(b, tables, sector, polygon_count, totals):
    """Compare each slot under both cyclic boundary conventions within this tile."""
    vertices, indices, edges, polygons = tables
    coords = [xyz for offset, count in vertices
              for (xyz,) in struct.iter_unpack("<6s", b[offset:offset + count * 6])]
    ring_indices = [index for offset, count in indices
                    for (index,) in struct.iter_unpack("<H", b[offset:offset + count * 2])]
    targets = (target for offset, count in edges for target in
               struct.iter_unpack("<IH6x", b[offset:offset + count * 12]))
    hypotheses = {"outgoing": Counter(), "incoming": Counter()}
    winding = totals.get("relative_winding")
    if winding is not None:
        xy = [struct.unpack_from("<HH", vertex) for vertex in coords]
        valid_xy = all(math.isfinite(v) and v > 0 for v in nd._f32s(b, 0x60, 2))
        signs = []
    coplanar = totals.get("coplanar_winding")
    if coplanar is not None:
        rings = []
        planes = {}
        extents = nd._f32s(b, 0x60, 3)
        valid_xyz = all(math.isfinite(v) and v >= 0 for v in extents)

        def plane(index):
            if index not in planes:
                points = [tuple(v if extent else 0 for v, extent in zip(struct.unpack("<3H", raw), extents))
                          for raw in rings[index]]
                planes[index] = _plane_ring(points) if valid_xyz else None
            return planes[index]

    position = source = self_refs = unlinked = invalid = 0
    for offset, count in polygons:
        for i in range(count):
            span = nd._u32(b, offset + 44 * i + 4)
            if span >> 4 != position:
                return
            end = position + (span & 15)
            ring = [coords[index] for index in ring_indices[position:end]]
            if coplanar is not None:
                rings.append(ring)
            if winding is not None:
                signs.append(_xy_winding([xy[index] for index in ring_indices[position:end]])
                             if valid_xy else 0)
            for j, vertex in enumerate(ring):
                target_sector, target = next(targets)
                if (target_sector, target) == (0xffffffff, 0xffff):
                    unlinked += 1
                elif target_sector != sector or target >= polygon_count:
                    invalid += 1
                elif target == source:
                    self_refs += 1
                else:
                    hypotheses["outgoing"][source, target, vertex, ring[(j + 1) % len(ring)]] += 1
                    hypotheses["incoming"][source, target, ring[j - 1], vertex] += 1
            position = end
            source += 1

    totals["tiles_counted"] += 1
    totals["self_references"] += self_refs
    totals["unlinked"] += unlinked
    totals["invalid_targets"] += invalid
    for convention, counts in hypotheses.items():
        for (source, target, start, end), n in counts.items():
            reverse = counts[target, source, end, start]
            same = counts[target, source, start, end] if start != end else 0
            if not (same or reverse):
                category = "no_exact_return"
            elif (n > 1 or reverse + same > 1 or
                  (start != end and counts[source, target, end, start])):
                category = "repeated"
            elif start == end:
                category = "degenerate"
            elif reverse:
                category = "reversed"
            else:
                category = "same_direction"
            totals[convention][category] += n
            if winding is not None and convention == "outgoing" and category in winding:
                product = signs[source] * signs[target]
                relation = "ambiguous" if product == 0 else "same" if product > 0 else "opposite"
                winding[category][relation] += n
                if coplanar is not None and category == "same_direction" and relation == "ambiguous":
                    coplanar[_plane_relation(plane(source), plane(target))] += n


def verify_tile(b, stats):
    """Verify +0x88 adjacent edges, polygon +0x20 link chain, and +0xA8/+0xB0 special links."""
    chk = {}
    if len(b) < 0xF0:
        for name in CHECK_NAMES:
            chk[name] = False
        return chk

    flags = nd._u32(b, 0x10)
    c0, c4, c8, _, d4, d8 = (nd._u32(b, o) for o in (0xC0, 0xC4, 0xC8, 0xCC, 0xD4, 0xD8))
    pool = nd._u32(b, 0x90)
    a8, b0 = nd._u64(b, 0xA8), nd._u64(b, 0xB0)

    # Cross-check u32@0xC0 (c0) and u32@0x90 (pool) against +0x70 (6-byte vertices),
    # +0x80 (2-byte vertex indices < c0), +0x88 (12-byte edges), and the sum of polygon vertex counts in +0x98.
    b70 = nd.split_array(b, 0x70, 6, c0)
    b80 = nd.split_array(b, 0x80, 2, pool)
    b88 = nd.split_array(b, 0x88, 12, pool)
    b98 = nd.split_array(b, 0x98, 44, c4)
    poly_vert_sum = 0
    if b98 is not None:
        for po, pcnt in b98:
            for pi in range(pcnt):
                poly_vert_sum += struct.unpack_from("<I", b, po + 44 * pi + 4)[0] & 0xF
    vidx_in_bounds = b80 is not None and all(nd._u16_max(b, o, n) < c0 for o, n in b80)
    pool_confirmed = b70 is not None and vidx_in_bounds and b98 is not None and poly_vert_sum == pool

    # 1. 12-byte adjacent-polygon edges at +0x88 (count = pool = u32@0x90).
    edges_ok = b88 is not None and pool_confirmed
    if edges_ok and "reciprocity" in stats:
        _count_reciprocity(b, b98, b88, c8, c4, stats["reciprocity"])
    if edges_ok and "endpoint_agreement" in stats:
        _endpoint_counts(b, (b70, b80, b88, b98), c8, c4, stats["endpoint_agreement"])
    same_tile = 0
    unlinked = 0
    edge_flags_lo = Counter()
    edge_meta_hi = Counter()
    if edges_ok:
        for bo, cnt in b88:
            for asec, apoly, meta16, flags32 in struct.iter_unpack("<IHHI", b[bo:bo + 12 * cnt]):
                if asec == c8 and apoly < c4:
                    same_tile += 1
                elif asec == 0xFFFFFFFF and apoly == 0xFFFF:
                    unlinked += 1
                else:
                    edges_ok = False
                    break
                edge_flags_lo[flags32 & 0xFF] += 1
                edge_meta_hi[meta16 >> 8] += 1
            if not edges_ok:
                break
    chk["edges_valid"] = edges_ok
    if edges_ok:
        est = stats["edges"]
        est["total"] += pool
        est["same_tile"] += same_tile
        est["unlinked"] += unlinked
        est["flags_low_byte"].update(edge_flags_lo)
        est["meta_high_byte"].update(edge_meta_hi)

    # 2. 32-byte special links at +0xA8 (count = d4 = u32@0xD4) and u16 index table at +0xB0 (count = d8 = u32@0xD8).
    has_links = bool(flags & 2)
    la = nd.offset_of(a8) if a8 else None
    lb = nd.offset_of(b0) if b0 else None
    if not has_links:
        links_bounds_ok = (a8 == 0 and b0 == 0 and d4 == 0 and d8 == 0)
    else:
        links_bounds_ok = (
            d4 > 0
            and d8 > 0
            and la is not None
            and lb is not None
            and la + 32 * d4 <= len(b)
            and lb + 2 * d8 <= len(b)
        )

    links_ok = links_bounds_ok
    intra = cross_s0 = cross_s1 = 0
    f0_hist = Counter()
    f1_hist = Counter()
    if links_ok and d4 > 0:
        for i in range(d4):
            p0, p1, s0, s1, f0, f1, heading, pad = struct.unpack_from("<HHIIHHHH", b, la + 32 * i + 12)
            if heading >= 256 or pad != 0:
                links_ok = False
                break
            if s0 == c8 and s1 == c8:
                if p0 >= c4 or p1 >= c4:
                    links_ok = False
                    break
                intra += 1
            elif s0 == c8:
                if p0 >= c4:
                    links_ok = False
                    break
                cross_s0 += 1
            elif s1 == c8:
                if p1 >= c4:
                    links_ok = False
                    break
                cross_s1 += 1
            else:
                links_ok = False
                break
            f0_hist[f0] += 1
            f1_hist[f1] += 1
    chk["special_links_valid"] = links_ok
    if links_ok and d4 > 0:
        lst = stats["special_links"]
        lst["link_tiles"] += 1
        lst["total"] += d4
        lst["indices_total"] += d8
        lst["intra_tile"] += intra
        lst["cross_tile_s0"] += cross_s0
        lst["cross_tile_s1"] += cross_s1
        lst["flags0"].update(f0_hist)
        lst["flags1"].update(f1_hist)

    # 3. 44-byte polygons at +0x98: u32@0x20 packs (flags_hi9 << 23) | (link_start << 6) | link_count,
    #    chaining across all c4 polygons to d8 and indexing +0xB0 -> +0xA8 endpoints touching that polygon.
    chain_ok = b98 is not None and links_bounds_ok
    endpoints_ok = chain_ok and links_ok
    lidx = struct.unpack_from(f"<{d8}H", b, lb) if (links_bounds_ok and d8 > 0) else ()
    nxt = 0
    gpi = 0
    max_poly_links = 0
    w20_hi9 = Counter()
    by_li = {}
    if b98 is not None:
        for po, pcnt in b98:
            for pi in range(pcnt):
                w20 = struct.unpack_from("<I", b, po + 44 * pi + 0x20)[0]
                cnt = w20 & 0x3F
                start = (w20 >> 6) & 0x1FFFF
                w20_hi9[w20 >> 23] += 1
                if cnt > max_poly_links:
                    max_poly_links = cnt
                if start != nxt or start + cnt > d8:
                    chain_ok = False
                    endpoints_ok = False
                    break
                if cnt > 0 and endpoints_ok:
                    for k in range(start, start + cnt):
                        li = lidx[k]
                        if li >= d4:
                            endpoints_ok = False
                            break
                        p0, p1, s0, s1 = struct.unpack_from("<HHII", b, la + 32 * li + 12)
                        if not ((p0 == gpi or p1 == gpi) and (s0 == c8 or s1 == c8)):
                            endpoints_ok = False
                            break
                        by_li.setdefault(li, set()).add(gpi)
                nxt += cnt
                gpi += 1
            if not chain_ok:
                break
        if nxt != d8:
            chain_ok = False
            endpoints_ok = False

    # Every link referenced in +0xB0 must be listed by all of its local-tile endpoints (p0 when s0 == c8,
    # and p1 when s1 == c8).
    if endpoints_ok and by_li:
        for li, gpis in by_li.items():
            p0, p1, s0, s1 = struct.unpack_from("<HHII", b, la + 32 * li + 12)
            if (s0 == c8 and p0 not in gpis) or (s1 == c8 and p1 not in gpis):
                endpoints_ok = False
                break

    chk["poly_link_chain"] = chain_ok
    chk["link_indices_endpoints"] = endpoints_ok
    if chain_ok:
        pst = stats["polygons"]
        pst["total"] += c4
        if max_poly_links > pst["max_links_per_poly"]:
            pst["max_links_per_poly"] = max_poly_links
        pst["w20_hi9_flags"].update(w20_hi9)
    return chk


def new_stats():
    return {
        "tiles": 0,
        "checks": {name: {"pass": 0, "fail": 0} for name in CHECK_NAMES},
        "edges": {
            "total": 0,
            "same_tile": 0,
            "unlinked": 0,
            "flags_low_byte": Counter(),
            "meta_high_byte": Counter(),
        },
        "polygons": {
            "total": 0,
            "max_links_per_poly": 0,
            "w20_hi9_flags": Counter(),
        },
        "special_links": {
            "link_tiles": 0,
            "total": 0,
            "indices_total": 0,
            "intra_tile": 0,
            "cross_tile_s0": 0,
            "cross_tile_s1": 0,
            "flags0": Counter(),
            "flags1": Counter(),
        },
    }


def _iter_resources(path, start=0, limit=None):
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".bin"))
        idx = 0
        while idx < len(files):
            with open(os.path.join(path, files[idx]), "rb") as fh:
                d0 = fh.read()
            idx += 1
            if len(d0) < 64:
                continue
            cls = nd.classify(d0)
            buf = bytearray(d0)
            n = 1
            while nd.wants_more(cls, buf, n) and idx < len(files):
                with open(os.path.join(path, files[idx]), "rb") as fh:
                    dn = fh.read()
                if len(dn) >= 64 and nd.classify(dn) != "other":
                    break
                buf.extend(dn)
                idx += 1
                n += 1
            if cls == "navmesh":
                yield bytes(buf)
        return

    size = os.path.getsize(path)
    if size < 8:
        return
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        first = struct.unpack_from("<I", m, 4)[0]
        scan_start = max(first, start)
        scan_end = size if limit is None else min(size, scan_start + limit)
        if scan_start >= scan_end:
            return
        gen = nd.walk(m, scan_start, size)
        pending = None
        while True:
            item = pending if pending is not None else next(gen, None)
            pending = None
            if item is None:
                break
            off0, _, d0 = item
            if off0 >= scan_end:
                break
            if len(d0) < 64:
                continue
            cls = nd.classify(d0)
            buf = bytearray(d0)
            n = 1
            while nd.wants_more(cls, buf, n):
                nxt = next(gen, None)
                if nxt is None:
                    break
                if len(nxt[2]) >= 64 and nd.classify(nxt[2]) != "other":
                    pending = nxt
                    break
                buf.extend(nxt[2])
                n += 1
            if cls == "navmesh":
                yield bytes(buf)


def scan(path, start=0, limit=None, *, reciprocity=False, endpoint_agreement=False, relative_winding=False,
         coplanar_winding=False):
    relative_winding = relative_winding or coplanar_winding
    stats = new_stats()
    if endpoint_agreement or relative_winding:
        stats["endpoint_agreement"] = {
            "tiles_counted": 0, "self_references": 0, "unlinked": 0, "invalid_targets": 0,
            **{name: dict.fromkeys(ENDPOINT_CLASSES, 0) for name in ("incoming", "outgoing")},
        }
    if relative_winding:
        stats["endpoint_agreement"]["relative_winding"] = {
            category: dict.fromkeys(("same", "opposite", "ambiguous"), 0)
            for category in ("reversed", "same_direction")
        }
    if coplanar_winding:
        stats["endpoint_agreement"]["coplanar_winding"] = dict.fromkeys(
            ("same", "opposite", "different_planes", "coincident", "undefined_ring"), 0)
    if reciprocity:
        stats["reciprocity"] = dict.fromkeys(
            ("tiles_counted", "same_tile", "excess_directed", "invalid_targets", "unlinked"), 0)
    for buf in _iter_resources(path, start=start, limit=limit):
        stats["tiles"] += 1
        chk = verify_tile(buf, stats)
        for name, ok in chk.items():
            stats["checks"][name]["pass" if ok else "fail"] += 1
    return {
        **({"endpoint_agreement": stats["endpoint_agreement"]} if endpoint_agreement or relative_winding else {}),
        **({"reciprocity": stats["reciprocity"]} if reciprocity else {}),
        "tiles": stats["tiles"],
        "checks": stats["checks"],
        "edges": {
            "total": stats["edges"]["total"],
            "same_tile": stats["edges"]["same_tile"],
            "unlinked": stats["edges"]["unlinked"],
            "flags_low_byte": _counter_top(stats["edges"]["flags_low_byte"]),
            "meta_high_byte": _counter_top(stats["edges"]["meta_high_byte"]),
        },
        "polygons": {
            "total": stats["polygons"]["total"],
            "max_links_per_poly": stats["polygons"]["max_links_per_poly"],
            "w20_hi9_flags": _counter_top(stats["polygons"]["w20_hi9_flags"]),
        },
        "special_links": {
            "link_tiles": stats["special_links"]["link_tiles"],
            "total": stats["special_links"]["total"],
            "indices_total": stats["special_links"]["indices_total"],
            "intra_tile": stats["special_links"]["intra_tile"],
            "cross_tile_s0": stats["special_links"]["cross_tile_s0"],
            "cross_tile_s1": stats["special_links"]["cross_tile_s1"],
            "flags0": _counter_top(stats["special_links"]["flags0"]),
            "flags1": _counter_top(stats["special_links"]["flags1"]),
        },
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Path to navmeshes.rpf or a directory of carved .bin streams")
    ap.add_argument("--start", type=int, default=0, help="Start byte offset inside an RPF9 archive")
    ap.add_argument("--limit", type=int, default=None, help="Maximum byte span to scan")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout summary")
    ap.add_argument("--reciprocity", action="store_true",
                    help="Count same-tile directed edges in excess of their reverse multiplicity")
    ap.add_argument("--endpoint-agreement", action="store_true",
                    help="Compare incoming and outgoing boundary endpoints for same-tile edges")
    ap.add_argument("--coplanar-winding", action="store_true",
                    help="Also classify XY-ambiguous same-direction slots using coplanar 3D rings")
    ap.add_argument("--relative-winding", action="store_true",
                    help="Include endpoint agreement and compare polygon XY winding for unique outgoing matches")
    args = ap.parse_args(argv)

    if args.start < 0:
        sys.stderr.write("error: --start must be >= 0\n")
        return 2
    if args.limit is not None and args.limit < 1:
        sys.stderr.write("error: --limit must be >= 1\n")
        return 2
    if not os.path.exists(args.input):
        sys.stderr.write(f"error: input path not found: {args.input}\n")
        return 2
    if os.path.isdir(args.input) and (args.start != 0 or args.limit is not None):
        sys.stderr.write("error: --start and --limit only apply to RPF9 archive files\n")
        return 2
    if args.out is not None:
        if os.path.isdir(args.out):
            sys.stderr.write(f"error: --out path is a directory: {args.out}\n")
            return 2
        if same_report_path(args.out, args.input):
            sys.stderr.write("error: --out must not overwrite the input path\n")
            return 2
        if os.path.isdir(args.input):
            for fname in os.listdir(args.input):
                if fname.endswith(".bin") and same_report_path(args.out, os.path.join(args.input, fname)):
                    sys.stderr.write("error: --out must not overwrite a carved input stream\n")
                    return 2

    doc = scan(args.input, start=args.start, limit=args.limit, reciprocity=args.reciprocity,
               endpoint_agreement=args.endpoint_agreement, relative_winding=args.relative_winding,
               coplanar_winding=args.coplanar_winding)
    if args.out is not None:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
    if not args.quiet:
        print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
