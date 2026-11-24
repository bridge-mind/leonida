"""Count which navmesh polygon flags lie under the lane-graph nodes.

The lane graph (tools/world_lights_paths.py, levels.rpf) gives every road node a world position. A navmesh tile
(tools/navmesh_decode.py, navmeshes.rpf) gives polygons with two flag words whose bits are mostly unnamed. This
tool puts the two together: for each node it finds the polygon of the node's 128 m tile that contains the node in
plan view and is closest in height, and counts the bits of that polygon's `+0x18` and `+0x28` words. A polygon's
height is the mean of its vertex heights, not the surface height under the node, so on a slope the two differ. Next
to it, it sums the plan area of the polygons that carry each bit, over the tiles that hold at least one node;
surfaces that lie above one another (a bridge over a road, stacked tiles of one sector) each add their area. A bit that
most nodes sit on while it covers a small share of the area is tied to the roads. Two bits of `+0x18` do that,
8 and 13, so the summary also says how they combine (both, only one, neither) for the nodes and for the area; see
docs/navmesh.md, "Polygon flags under the lane graph".

A tile's vertices are three u16 each: position = tile minimum + u16 / 65535 * tile extent (the minimum is the
quadtree root's, the extent the header's `+0x60`). The node heights test that: a wrong scale would leave the
nodes far above or below every polygon.

Only static tiles are used (header flag bit 2 clear); a dynamic tile has no place on the 128 m grid.

Usage:
    python -I navmesh_nodes.py <navmeshes.rpf> <levels.rpf> [--out summary.json] [--jobs N] [--max-dz METRES]

The output is counts only: no position, name or payload is written.
"""
import argparse
import collections
import json
import math
import mmap
import os
import struct
import sys

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling readers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import navmesh_decode as nd  # noqa: E402
import world_lights_paths as wlp  # noqa: E402
from report_paths import same_report_path  # noqa: E402

TILE = 128.0
CELL = 8.0               # bucket size of the per-tile lookup grid, metres
POLYGON = struct.Struct("<II16xI12xI")   # tile id, (first index << 4) | count, +0x18 flags, +0x28 flags
assert POLYGON.size == 44
PAIR = (8, 13)           # the two +0x18 bits most nodes stand on; counted in combination as well


def pair_key(flags18):
    """Which of the PAIR bits a `+0x18` word carries: "both", "only_8", "only_13" or "neither"."""
    low, high = (flags18 >> PAIR[0]) & 1, (flags18 >> PAIR[1]) & 1
    if low and high:
        return "both"
    if low:
        return "only_%d" % PAIR[0]
    return "only_%d" % PAIR[1] if high else "neither"


def sector_of(x, y):
    """The (sx, sy) of the 128 m tile that holds a world position, as stored in a tile's `+0xC8` word."""
    return int(math.floor(x / TILE)) + 256, int(math.floor(y / TILE)) + 256


def tile_geometry(b):
    """Vertices and polygons of one static navmesh tile, or None when it is dynamic or does not read.

    -> {"sector": (sx, sy), "polygons": [(points, flags18, flags28)]} with points as (x, y, z) in world space.
    """
    if len(b) < nd.HEADER_NAV or nd._u32(b, 0x14) != nd.DATA15 or nd._u32(b, 0x10) & 4:
        return None
    n_verts, n_polys, sector = (nd._u32(b, o) for o in (0xC0, 0xC4, 0xC8))
    pool = nd._u32(b, 0x90)
    vertex_blocks = nd.split_array(b, 0x70, 6, n_verts)
    index_blocks = nd.split_array(b, 0x80, 2, pool)
    polygon_blocks = nd.split_array(b, 0x98, 44, n_polys)
    root = nd.offset_of(nd._u64(b, 0xA0))
    if vertex_blocks is None or index_blocks is None or polygon_blocks is None or root is None \
            or root + 0x20 > len(b):
        return None
    lo = nd._f32s(b, root, 3)
    ext = nd._f32s(b, 0x60, 3)
    if not all(math.isfinite(v) for v in (*lo, *ext)):
        return None
    verts = []
    for at, count in vertex_blocks:
        for vx, vy, vz in struct.iter_unpack("<3H", b[at:at + 6 * count]):
            verts.append((lo[0] + vx / 65535.0 * ext[0], lo[1] + vy / 65535.0 * ext[1],
                          lo[2] + vz / 65535.0 * ext[2]))
    index = []
    for at, count in index_blocks:
        index.extend(struct.unpack_from("<%dH" % count, b, at))
    polygons = []
    for at, count in polygon_blocks:
        for _tile, packed, flags18, flags28 in POLYGON.iter_unpack(b[at:at + 44 * count]):
            first, n = packed >> 4, packed & 0xF
            if first + n > len(index) or any(i >= len(verts) for i in index[first:first + n]):
                return None
            polygons.append(([verts[i] for i in index[first:first + n]], flags18, flags28))
    return {"sector": (sector & 0x1FF, sector >> 9), "polygons": polygons}


def plan_area(points):
    """Area of a polygon's projection on the ground plane (shoelace)."""
    total = 0.0
    for k, (x1, y1, _z1) in enumerate(points):
        x2, y2, _z2 = points[(k + 1) % len(points)]
        total += x1 * y2 - x2 * y1
    return abs(total) / 2.0


def contains(points, x, y):
    """Even-odd test of (x, y) against a polygon's ground-plane projection."""
    inside = False
    for k, (x1, y1, _z1) in enumerate(points):
        x2, y2, _z2 = points[(k + 1) % len(points)]
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def polygon_under(polygons, grid, x, y, z):
    """The polygon containing (x, y) whose mean height is closest to z -> (height difference, polygon) or None."""
    best = None
    for i in grid.get((int(math.floor(x / CELL)), int(math.floor(y / CELL))), ()):
        points = polygons[i][0]
        if len(points) >= 3 and contains(points, x, y):
            dz = abs(sum(p[2] for p in points) / len(points) - z)
            if best is None or dz < best[0]:
                best = (dz, polygons[i])
    return best


def new_tally():
    return {
        "tiles_with_nodes": 0, "tiles_unreadable": 0, "polygons": 0, "area": 0.0,
        "area18": collections.Counter(), "area28": collections.Counter(),
        "polygons18": collections.Counter(), "polygons28": collections.Counter(),
        "pair_area": collections.Counter(), "pair_polygons": collections.Counter(),
        "best": {},          # node index -> (height difference, flags18, flags28)
    }


def tally_tile(tally, b, by_sector, nodes):
    """Add one assembled navmesh resource to the tally. Tiles without a node are skipped."""
    if len(b) < nd.HEADER_NAV or nd._u32(b, 0x10) & 4:
        return
    sector = nd._u32(b, 0xC8)
    mine = by_sector.get((sector & 0x1FF, sector >> 9))
    if not mine:
        return
    geometry = tile_geometry(b)
    if geometry is None:
        tally["tiles_unreadable"] += 1
        return
    tally["tiles_with_nodes"] += 1
    polygons = geometry["polygons"]
    grid = collections.defaultdict(list)
    for i, (points, flags18, flags28) in enumerate(polygons):
        if len(points) < 3:
            continue
        area = plan_area(points)
        tally["polygons"] += 1
        tally["area"] += area
        tally["pair_area"][pair_key(flags18)] += area
        tally["pair_polygons"][pair_key(flags18)] += 1
        for word, area_by, count_by in ((flags18, tally["area18"], tally["polygons18"]),
                                        (flags28, tally["area28"], tally["polygons28"])):
            bit = 0
            while word:
                if word & 1:
                    area_by[bit] += area
                    count_by[bit] += 1
                word >>= 1
                bit += 1
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        for gx in range(int(math.floor(min(xs) / CELL)), int(math.floor(max(xs) / CELL)) + 1):
            for gy in range(int(math.floor(min(ys) / CELL)), int(math.floor(max(ys) / CELL)) + 1):
                grid[(gx, gy)].append(i)
    for n in mine:
        x, y, z = nodes[n]
        found = polygon_under(polygons, grid, x, y, z)
        if found is not None and (n not in tally["best"] or found[0] < tally["best"][n][0]):
            tally["best"][n] = (found[0], found[1][1], found[1][2])


def starts_resource(b):
    """True when stream `b` opens a resource of its own (the check navmesh_links.py makes)."""
    return len(b) >= nd.MIN_STREAM and nd.classify(b) != "other"


def navmesh_resources(path):
    """Yield the assembled bytes of every navmesh tile in an RPF archive (the walk of navmesh_decode)."""
    with open(path, "rb") as fh:
        size = os.fstat(fh.fileno()).st_size
        if size < 8:
            raise ValueError("%s is %d bytes, too small for an RPF header" % (path, size))
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
            if m[1:4] != b"FPR" or m[0:1] not in (b"7", b"8", b"9"):
                raise ValueError("%s: not an RPF7/8/9 archive" % path)
            data_start = struct.unpack_from("<I", m, 4)[0]
            if not 8 <= data_start <= size:
                raise ValueError("%s: the data area starts at %d, outside the %d-byte file" % (path, data_start, size))
            stream = nd.walk(m, data_start, size)
            pending = next(stream, None)
            while pending is not None:
                head = pending[2]
                pending = next(stream, None)
                if len(head) < nd.MIN_STREAM:
                    continue
                cls = nd.classify(head)
                buf, parts = bytearray(head), 1
                while pending is not None and nd.wants_more(cls, buf, parts):
                    # A stream that opens a resource of its own is not this one's continuation: a resource whose
                    # next stream is missing stays incomplete instead of swallowing the tile after it.
                    if len(buf) + len(pending[2]) > nd.MAX_INFLATE or starts_resource(pending[2]):
                        break
                    buf.extend(pending[2])
                    parts += 1
                    pending = next(stream, None)
                if cls == "navmesh" and nd.is_complete(cls, buf, parts):
                    yield bytes(buf)


def node_positions(levels, jobs=1):
    """(positions, nodes read): the finite lane-graph node positions of levels.rpf."""
    total = wlp.scan(levels, jobs)
    read = [n["position"] for part in total["nodes"] for n in part]
    return [tuple(p) for p in read if all(math.isfinite(v) for v in p)], len(read)


def cross_check(navmeshes, nodes, max_dz=3.0):
    """Summary of the polygons under `nodes` ((x, y, z) tuples) in the navmesh archive `navmeshes`."""
    by_sector = collections.defaultdict(list)
    for n, (x, y, _z) in enumerate(nodes):
        by_sector[sector_of(x, y)].append(n)
    tally = new_tally()
    for b in navmesh_resources(navmeshes):
        tally_tile(tally, b, by_sector, nodes)
    return summarize(tally, len(nodes), len(by_sector), max_dz)


def summarize(tally, n_nodes, n_sectors, max_dz):
    near = [v for v in tally["best"].values() if v[0] <= max_dz]
    heights = sorted(v[0] for v in tally["best"].values())
    under18, under28 = collections.Counter(), collections.Counter()
    pair_nodes = collections.Counter(pair_key(flags18) for _dz, flags18, _flags28 in near)
    pair_names = ("both", "only_%d" % PAIR[0], "only_%d" % PAIR[1], "neither")
    for _dz, flags18, flags28 in near:
        for word, into in ((flags18, under18), (flags28, under28)):
            bit = 0
            while word:
                if word & 1:
                    into[bit] += 1
                word >>= 1
                bit += 1

    def table(under, area, count):
        return {str(bit): {"nodes": under.get(bit, 0), "polygons": count.get(bit, 0),
                           "area_m2": round(area.get(bit, 0.0))}
                for bit in sorted(set(under) | set(area))}

    def quantile(q):
        return round(heights[min(len(heights) - 1, int(len(heights) * q))], 3) if heights else None

    return {
        "nodes": n_nodes,
        "tiles_named_by_nodes": n_sectors,
        "tiles_with_nodes": tally["tiles_with_nodes"],
        "tiles_unreadable": tally["tiles_unreadable"],
        "nodes_over_a_polygon": len(tally["best"]),
        "max_dz": max_dz,
        "nodes_within_max_dz": len(near),
        "height_difference_m": {"p10": quantile(0.1), "p50": quantile(0.5), "p90": quantile(0.9),
                                "p99": quantile(0.99)},
        "polygons": tally["polygons"],
        "area_m2": round(tally["area"]),
        "flags18": table(under18, tally["area18"], tally["polygons18"]),
        "flags28": table(under28, tally["area28"], tally["polygons28"]),
        "flags18_bits_%d_and_%d" % PAIR: {
            name: {"nodes": pair_nodes.get(name, 0), "polygons": tally["pair_polygons"].get(name, 0),
                   "area_m2": round(tally["pair_area"].get(name, 0.0))}
            for name in pair_names},
    }


def out_problem(path):
    """Why the summary cannot be written at `path`, or None.

    Checked before the archives are read: the levels graph takes minutes to scan, so a destination this run
    cannot write should not cost the whole run and then end in a traceback.
    """
    parent = os.path.dirname(os.path.abspath(path))
    try:
        os.makedirs(parent, exist_ok=True)
    except OSError as e:
        return "cannot create its folder: %s" % e
    if not os.access(parent, os.W_OK):
        return "its folder is not writable: %s" % parent
    if os.path.exists(path) and not os.access(path, os.W_OK):
        return "the existing summary is read-only"
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("navmeshes", help="navmeshes.rpf")
    ap.add_argument("levels", help="levels.rpf (the lane graph)")
    ap.add_argument("--out", help="summary JSON (default: stdout)")
    ap.add_argument("--jobs", type=int, default=1, help="processes for the levels.rpf scan")
    ap.add_argument("--max-dz", type=float, default=3.0,
                    help="a node counts when its polygon is at most this many metres above or below it")
    a = ap.parse_args(argv)
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    if not math.isfinite(a.max_dz) or a.max_dz < 0:
        ap.error("--max-dz must be 0 or more")
    for path in (a.navmeshes, a.levels):
        if not os.path.isfile(path):
            ap.error("%s: no such file" % path)
    if a.out:
        if os.path.isdir(a.out) or any(same_report_path(a.out, p) for p in (a.navmeshes, a.levels)):
            ap.error("--out must be a file other than the two archives")
        problem = out_problem(a.out)
        if problem:
            ap.error("--out: %s" % problem)
    try:
        nodes, read = node_positions(a.levels, a.jobs)
        summary = cross_check(a.navmeshes, nodes, a.max_dz)
    except (OSError, ValueError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 2
    summary["nodes_read"] = read
    text = json.dumps(summary, indent=1, sort_keys=True)
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
