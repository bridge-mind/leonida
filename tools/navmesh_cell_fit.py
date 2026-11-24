"""Measure the cell grid the local-space navmeshes of ``navmeshes.rpf`` were built on (ASSET-332, docs/navmesh.md).

For every non-grid navmesh tile (``flags & 4``) with ``flags == 7``, the ``u16[3]`` vertices of the split array at
``+0x70`` are dequantised over the tile's ``+0x60`` extents (step ``extent / 65535``). An x or y coordinate is on a
grid of cell size ``c`` when it is within 1.5 steps of a multiple of ``c``. A mesh fits ``c`` when at least
``--min-fraction`` of its x and y coordinates are on that grid. For each fitting mesh, the chance baseline is the
share of all 65,536 quantisation levels of its own step that land on the grid. ``--static-every N`` measures
every Nth static 128 m tile the same way, as a control.

With ``--common`` (a ``tools/rpf9_carve.py`` carve of ``common.rpf``), the ``CVehiclePoints`` documents'
``fNavOverrideRecastCellSize`` values are counted, and every non-grid mesh is searched at every byte offset for the
``joaat`` of each document's ``ModelName`` (as written, lower-cased, and with ``.yft``).

Only aggregates are printed and written: counts, cell sizes, extents and fractions. No model name, no vertex, no
file name.

Usage:
    python -I navmesh_cell_fit.py <navmeshes.rpf> [--common CARVE_DIR] [--static-every N] [--out FILE]
                                  [--min-fraction F]
"""
import argparse
import glob
import json
import os
import re
import struct
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import navmesh_decode as nd  # noqa: E402
from report_paths import same_report_path  # noqa: E402

DOC_CELLS = (0.0625, 0.03125, 0.0325)            # the nonzero fNavOverrideRecastCellSize values, coarsest first
OTHER_CELLS = (0.1, 0.125, 0.15, 0.2, 0.25, 0.5)  # controls no document uses
LEVELS = 65536
TOL_STEPS = 1.5


def on_grid(x, cell, tol):
    r = x % cell
    return r <= tol or cell - r <= tol


def fit_fraction(quantised, extents, cell):
    """Share of the x and y coordinates (axes with a nonzero extent) within TOL_STEPS steps of a multiple of cell."""
    hit = total = 0
    for k in (0, 1):
        if extents[k] <= 0:
            continue
        step = extents[k] / (LEVELS - 1)
        tol = TOL_STEPS * step
        for v in quantised:
            total += 1
            hit += on_grid(v[k] * step, cell, tol)
    return hit / total if total else 0.0


def baseline(extent, cell):
    """Share of all quantisation levels of this extent's step that are on the grid: what the step alone gives."""
    step = extent / (LEVELS - 1)
    tol = TOL_STEPS * step
    return sum(on_grid(q * step, cell, tol) for q in range(LEVELS)) / LEVELS


def vertices(b):
    """The u16[3] vertices of a navmesh tile, or None when its split array does not validate."""
    blocks = nd.split_array(b, 0x70, 6, nd._u32(b, 0xC0))
    if not blocks:
        return None
    out = []
    for o, n in blocks:
        out.extend(struct.unpack_from("<3H", b, o + 6 * i) for i in range(n))
    return out


def vehicle_documents(carve):
    """(cell-size Counter, set of ModelName values) of the CVehiclePoints documents in a carve of common.rpf."""
    cells, names = Counter(), set()
    for p in sorted(glob.glob(os.path.join(carve, "*.bin"))):
        with open(p, "rb") as f:
            t = f.read()
        m = re.search(rb"<(CVehiclePoints\w*)[\s>]", t)
        if not m or m.group(1) != b"CVehiclePoints":
            continue
        v = re.search(rb'<fNavOverrideRecastCellSize value="([^"]+)"', t)
        cells[float(v.group(1)) if v else None] += 1
        names.update(x.decode("ascii", "replace") for x in re.findall(rb"<ModelName>([^<]+)</ModelName>", t))
    return cells, names


def name_hashes(names):
    out = set()
    for n in names:
        for v in {n, n.lower()}:
            out.add(nd.joaat_cs(v))
            out.add(nd.joaat_cs(v + ".yft"))
    return out


def mesh_row(b):
    q = vertices(b)
    if q is None:
        return None
    ext = struct.unpack_from("<3f", b, 0x60)
    return {"n": len(q), "ext": ext, "fit": {c: fit_fraction(q, ext, c) for c in DOC_CELLS + OTHER_CELLS}}


def measure(archive, hashes=None, min_fraction=0.9, static_every=0):
    meshes, static, hash_hits, scanned, n_static = [], [], 0, 0, 0
    orig = nd.decode_navmesh

    def wrapped(b, nstreams, stats, deep):
        nonlocal hash_hits, scanned, n_static
        row, chk = orig(b, nstreams, stats, deep)
        flags = nd._u32(b, 0x10)
        if not flags & 4 and static_every:
            n_static += 1
            if n_static % static_every == 0:
                r = mesh_row(b)
                if r is not None:
                    static.append(r)
        if flags & 4:
            scanned += 1
            if hashes and any(struct.unpack_from("<I", b, o)[0] in hashes for o in range(len(b) - 3)):
                hash_hits += 1
            if flags == 7:
                r = mesh_row(b)
                if r is not None:
                    meshes.append(r)
        return row, chk

    nd.decode_navmesh = wrapped
    try:
        nd.scan(archive, rows=[])
    finally:
        nd.decode_navmesh = orig
    control = summarise_control(static, min_fraction) if static_every else None
    return summarise(meshes, min_fraction), scanned, hash_hits, control


def summarise_control(tiles, min_fraction=0.9):
    """Static tiles: for each cell size, how many fit it and the mean chance baseline of their step."""
    out = {"tiles": len(tiles), "cells": {}}
    for c in DOC_CELLS + OTHER_CELLS:
        bases = [max(baseline(t["ext"][k], c) for k in (0, 1) if t["ext"][k] > 0) for t in tiles[:20]]
        out["cells"][str(c)] = {"fit": sum(1 for t in tiles if t["fit"][c] >= min_fraction),
                                "baseline_mean": round(sum(bases) / len(bases), 4) if bases else None}
    return out


def summarise(meshes, min_fraction=0.9):
    def fits(m, c):
        return m["fit"][c] >= min_fraction
    out = {"meshes": len(meshes), "vertices": sum(m["n"] for m in meshes), "cells": {},
           "fit_none": sum(1 for m in meshes if not any(fits(m, c) for c in DOC_CELLS)),
           "fit_0.03125_and_0.0325": sum(1 for m in meshes if fits(m, 0.03125) and fits(m, 0.0325)),
           "other_cells": {str(c): sum(1 for m in meshes if fits(m, c)) for c in OTHER_CELLS}}
    for c in DOC_CELLS:
        g = [m for m in meshes if next((d for d in DOC_CELLS if fits(m, d)), None) == c]
        if not g:
            out["cells"][str(c)] = {"meshes": 0}
            continue
        bases = [max(baseline(m["ext"][k], c) for k in (0, 1) if m["ext"][k] > 0) for m in g]
        out["cells"][str(c)] = {
            "meshes": len(g),
            "distinct_xy_extents": len({(round(m["ext"][0], 4), round(m["ext"][1], 4)) for m in g}),
            "min_fraction": round(min(m["fit"][c] for m in g), 6),
            "baseline_max": round(max(bases), 4), "baseline_mean": round(sum(bases) / len(bases), 4),
            "extents_whole_cells": sum(1 for m in g if all(abs(m["ext"][k] / c - round(m["ext"][k] / c)) < 1e-3
                                                           for k in (0, 1))),
            "vertex_counts": dict(sorted(Counter(m["n"] for m in g).items())) if len(g) <= 16 else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive", help="navmeshes.rpf")
    ap.add_argument("--common", help="a tools/rpf9_carve.py carve of common.rpf (CVehiclePoints counts, hash search)")
    ap.add_argument("--out", help="write the summary as JSON here")
    ap.add_argument("--min-fraction", type=float, default=0.9)
    ap.add_argument("--static-every", type=int, default=0, help="also measure every Nth static tile as a control")
    a = ap.parse_args()
    if not 0 <= a.min_fraction <= 1:
        ap.error("--min-fraction must be between 0 and 1")
    if a.static_every < 0:
        ap.error("--static-every must be non-negative")
    if a.out and same_report_path(a.out, a.archive):
        ap.error("--out would overwrite the archive")
    if not os.path.exists(a.archive):
        print(f"ERROR: cannot read {a.archive}: no such file or folder", file=sys.stderr)
        return 2
    hashes, docs = None, None
    if a.common:
        cells, names = vehicle_documents(a.common)
        hashes = name_hashes(names)
        docs = {"documents": sum(cells.values()), "models": len(names),
                "cell_sizes": {("absent" if k is None else str(k)): v for k, v in sorted(cells.items(), key=str)}}
    summary, scanned, hits, control = measure(a.archive, hashes, a.min_fraction, a.static_every)
    report = {"local_space_meshes": summary, "non_grid_scanned": scanned}
    if control is not None:
        report["static_control"] = control
    if docs is not None:
        report["vehicle_documents"] = docs
        report["non_grid_with_a_model_hash"] = hits
    text = json.dumps(report, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
