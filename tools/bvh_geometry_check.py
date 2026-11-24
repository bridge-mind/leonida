"""Check whether geometry given to a BLAS rehydration is the geometry the BLAS was built from.

A rehydrated structure that matches `XboxBvhBuilder.dll` byte for byte proves the packing, not the geometry: the
library and `tools/bvh_rehydrate.py` agree on any input. The dehydrated BLAS, though, keeps things that were computed
from the true triangles, and this tool compares a rehydration against them (docs/bvh.md §4a):

1. bounds: every rehydrated leaf lies inside `sceneBounds` mapped through the leaf frame, and the union of the leaves
   equals it (the error is the largest gap over the largest extent; about 1e-7 or less for the right geometry);
2. overrides: each box component stored in the dehydrated groups lies inside the range of the rehydrated slot it
   overrides and is no looser than it, within a few float32 ulps (the offline build rounds some one or two ulps away);
3. pairs: a leaf with two primitives joins triangles that share an edge, so in the right index buffer the two share
   two vertex indices. This needs no positions and no transform, so it also checks one geometry on its own;
4. area: the summed surface area of the internal box slots, a tie-break between candidates (lower is tighter).

Standard library only.

Usage:
    python -I tools/bvh_geometry_check.py <dehydrated.bin> --geom VB[,IB] [...] [--header-offset N] [--stride BYTES]
                                          [--index16] [--transform 12 floats] [--json <report.json>]

The geometry options are those of tools/bvh_rehydrate.py. It prints one line per check; exit status is 0 when bounds,
overrides and pairs are consistent, 1 when one is not, 2 on an input error.
"""
import argparse
import json
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bvh_rehydrate as R  # noqa: E402
from bvh_index import _publish  # noqa: E402
from report_paths import same_report_path  # noqa: E402

BOUNDS_TOLERANCE = 1e-4     # of the largest sceneBounds extent
OUTSIDE_TOLERANCE = 1e-3
OVERRIDE_ULPS = 4


def leaf_offsets(hdr):
    for i in range(hdr.num_elements):
        yield hdr.offset_to_elements + (i // R.LEAVES_PER_CHUNK) * R.CHUNK_STRIDE + (i % R.LEAVES_PER_CHUNK) * 64


def check_bounds(hdr, out):
    """(leaves outside, relative gap between the leaf union and sceneBounds in the leaf frame)."""
    frame = R.LeafFrame(hdr)
    lo, hi = hdr.bounds[:3], hdr.bounds[3:]
    corners = [frame.apply((x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    box = [min(c[a] for c in corners) for a in range(3)] + [max(c[a] for c in corners) for a in range(3)]
    extent = max(max(box[a + 3] - box[a] for a in range(3)), 1e-30)
    union = [math.inf] * 3 + [-math.inf] * 3
    outside = 0
    for off in leaf_offsets(hdr):
        b = R.node_bounds(out, (off >> 6) << 3 | R.TRI0)
        if max([box[a] - b[a] for a in range(3)] + [b[a] - box[a] for a in range(3, 6)]) > OUTSIDE_TOLERANCE * extent:
            outside += 1
        for a in range(3):
            union[a], union[a + 3] = min(union[a], b[a]), max(union[a + 3], b[a + 3])
    gap = max(abs(union[a] - box[a]) for a in range(6)) / extent if hdr.num_elements else 0.0
    return outside, gap


def stored_overrides(src, hdr):
    """Yield (child pointer, component, value) for every override the dehydrated box groups store."""
    if hdr.root & 7 not in (R.BOX16, R.BOX32):
        return
    for b32_start, b32_count, b16_start, b16_count in R.levels(src, hdr):
        for start, count, wide in ((b32_start, b32_count, True), (b16_start, b16_count, False)):
            for g in range((count + R.GROUP - 1) // R.GROUP):
                n = min(R.GROUP, count - g * R.GROUP)
                base = hdr.offset_to_nodes + start + g * ((1 << 13) if wide else (1 << 12))
                words = struct.unpack_from("<%dI" % n, src, base)
                child_at, value_at = base + 4 * n, base + 4 * (n + sum(w >> 24 for w in words))
                for word in words:
                    kids = word >> 24
                    children = struct.unpack_from("<%dI" % kids, src, child_at)
                    child_at += 4 * kids
                    for bit in range(24):
                        if word >> bit & 1:
                            if wide:
                                value = struct.unpack_from("<f", src, value_at)[0]
                                value_at += 4
                            else:
                                value = R.half_to_float(struct.unpack_from("<H", src, value_at)[0])
                                value_at += 2
                            if bit // 6 < kids:
                                yield children[bit // 6], bit % 6, value


def check_overrides(src, hdr, out):
    """(consistent, inconsistent) stored override components."""
    good = bad = 0
    for child, comp, value in stored_overrides(src, hdr):
        b = R.node_bounds(out, child)
        lo, hi = b[comp % 3], b[comp % 3 + 3]
        tol = OVERRIDE_ULPS * 2.0 ** -23 * max(abs(value), abs(b[comp]), 1e-30) if math.isfinite(value) else 0.0
        tighter = value >= b[comp] - tol if comp < 3 else value <= b[comp] + tol
        if tighter and lo - tol <= value <= hi + tol:
            good += 1
        else:
            bad += 1
    return good, bad


def check_pairs(src, hdr, geometries):
    """{geometry index: (pairs sharing an edge, pairs)} from the leaves that hold two primitives."""
    out = {}
    for _, gi, _, prim0, prim1, _ in R.leaf_records(src, hdr):
        if prim1 is None or gi >= len(geometries):
            continue
        g = geometries[gi]
        shared = len(set(g.triangle(prim0)) & set(g.triangle(prim1))) >= 2
        ok, total = out.get(gi, (0, 0))
        out[gi] = (ok + shared, total + 1)
    return out


def internal_area(hdr, out):
    area, stack, seen = 0.0, [hdr.root], set()
    while stack:
        p = stack.pop()
        if p == R.INVALID or p in seen or p & 7 not in (R.BOX16, R.BOX32):
            continue
        seen.add(p)
        for child in struct.unpack_from("<4I", out, (p >> 3) << 6):
            if child == R.INVALID:
                continue
            b = R.node_bounds(out, child)
            e = [b[a + 3] - b[a] if math.isfinite(b[a + 3] - b[a]) and b[a + 3] >= b[a] else 0.0 for a in range(3)]
            area += 2.0 * (e[0] * e[1] + e[1] * e[2] + e[0] * e[2])
            stack.append(child)
    return area


def check(dehydrated, geometries, header_offset=0):
    """Rehydrate and run every check. Returns a report dict; report["consistent"] is the overall verdict."""
    hdr = R.Header(dehydrated, header_offset)
    src = bytes(dehydrated[header_offset:header_offset + hdr.total_bytes])
    out = R.rehydrate(dehydrated, geometries, header_offset)
    outside, gap = check_bounds(hdr, out)
    good, bad = check_overrides(src, hdr, out)
    pairs = check_pairs(src, hdr, list(geometries))
    pairs_ok = all(ok == total for ok, total in pairs.values())
    return {"leaves": hdr.num_elements, "leaves_outside": outside, "bounds_gap": gap,
            "overrides_consistent": good, "overrides_inconsistent": bad,
            "pairs": {str(k): list(v) for k, v in sorted(pairs.items())}, "internal_area": internal_area(hdr, out),
            "consistent": outside == 0 and gap <= BOUNDS_TOLERANCE and bad == 0 and pairs_ok}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Check a BLAS rehydration's geometry against what the BLAS stores.")
    ap.add_argument("dehydrated")
    ap.add_argument("--geom", action="append", default=[], metavar="VB[,IB]")
    ap.add_argument("--header-offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--index16", action="store_true")
    ap.add_argument("--transform")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    geom_paths = []
    for spec in a.geom:
        parts = spec.split(",")
        geom_paths.append((parts[0], parts[1] if len(parts) > 1 else ""))
    if a.json:
        if same_report_path(a.json, a.dehydrated):
            print("error: --json %s must not overwrite the dehydrated input" % a.json, file=sys.stderr)
            return 2
        for vb_path, ib_path in geom_paths:
            for path in (vb_path, ib_path):
                if path and same_report_path(a.json, path):
                    print("error: --json %s must not overwrite a geometry buffer %s" % (a.json, path), file=sys.stderr)
                    return 2
    try:
        transform = R.parse_transform(a.transform) if a.transform else None
        geometries = []
        for vb_path, ib_path in geom_paths:
            with open(vb_path, "rb") as f:
                vb = f.read()
            ib = None
            if ib_path:
                with open(ib_path, "rb") as f:
                    ib = f.read()
            geometries.append(R.Geometry(vb, a.stride, ib, a.index16, transform))
        with open(a.dehydrated, "rb") as f:
            report = check(f.read(), geometries, a.header_offset)
    except (OSError, ValueError, struct.error) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    pairs = report["pairs"]
    print("bounds: %d of %d leaves outside, union gap %.2e" % (report["leaves_outside"], report["leaves"], report["bounds_gap"]))
    print("overrides: %d consistent, %d not" % (report["overrides_consistent"], report["overrides_inconsistent"]))
    print("pairs sharing an edge: %s" % (", ".join("geometry %s %d/%d" % (k, v[0], v[1]) for k, v in pairs.items()) or "no pairs"))
    print("internal box area: %.6g" % report["internal_area"])
    print("consistent" if report["consistent"] else "NOT consistent")
    if a.json:
        try:
            _publish(a.json, lambda f: json.dump(report, f, indent=1), "\n")
        except OSError as e:
            print("error: cannot write %s: %s" % (a.json, e.strerror or e), file=sys.stderr)
            return 2
    return 0 if report["consistent"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
