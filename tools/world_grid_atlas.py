"""Place the tile textures of the GridAtlasInfo map grids and check what the grid documents' zInflation box covers.

The four ``GridAtlasInfo`` documents in ``common.rpf`` (docs/gamedata.md, "Grid atlases") describe world-space
texture grids of 512-unit tiles. Their tile textures are in the texture dictionaries that ``make_game.py --stages
assets`` exports, but every tile carries the same texture name (the document's ``tileTextureNameSuffix``) and the
dictionary names are not in the data, so a tile's cell is not written anywhere. This tool recovers it:

- Each grid's global fallback (``globalFallbackTextureName``) is a mosaic of the whole grid at a few pixels per
  tile. Fallback image row 0 is the grid's north edge (largest y); cell = row * columns + column.
- ``content`` grids (water height R16, water fog BC6H) compare each tile, point-sampled at the fallback's
  resolution, with every fallback cell. Grids whose tiles share their dictionaries (the two fog grids) are matched
  jointly. A tile with detail whose runner-up distance is at least twice the best is an anchor.
- ``mask`` grids (shoreline, 8-bit BGRA) are mostly one background colour, which swamps a plain distance. The
  tool compares feature masks (pixels away from the fallback's most common colour) by intersection over union and
  aligns the tiles, in archive order, to cells in increasing text order with a dynamic programme.
- Archive order: the tile dictionaries are stored in the order of their cells sorted as text of the decimal cell
  index (``sorted(cells, key=str)``). The tool counts ascending steps of the anchors under that order and three
  other spellings, then places the tiles between two anchors when their number equals the free cells between them.

``zInflation`` is checked, not measured at runtime: the tool reports each grid's vertical box (origin z +-
zInflation), the Z bounds of the MHIU and PAMH heightmaps in the same carve, and a least-squares fit of the water
height fallback against MHIU terrain heights (values above ``FIT_MIN_VALUE``), which also gives the value-to-metre
scale and confirms the north-up row order.

    python tools/world_grid_atlas.py <work> --carve <common carve> [--json FILE]

``<work>`` is the builder's work folder (``<work>/assets/<archive>/textures/manifest.json``); ``--carve`` is the
``rpf9_carve.py`` output of ``common.rpf``. The tool prints one summary line per grid. ``--json`` also writes the
cell of every placed tile: that is data derived from your own copy of the game, so keep the file local.

Unlike most tools here this one imports numpy and Pillow (Pillow decodes BC6H); it is one of the texture-tool
exceptions in AGENTS.md. The placement, ordering and fitting logic is plain Python, and its tests run on the
standard library alone.
Usage: `python -I tools/world_grid_atlas.py --help` lists the options.
"""
import argparse
import json
import math
import os
import struct
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import report_paths  # noqa: E402
from file_mode import published_mode  # noqa: E402

DX10_HEADER = 148
DXGI_R16_UNORM = 56
MASK_THRESHOLD = 32          # max channel distance from the background colour that still counts as background
MASK_BLOCK_SHARE = 0.25      # share of feature pixels in a tile block for the block to count as feature
CLEAR_MARGIN = 0.2           # a shoreline tile is an anchor when its best overlap beats the runner-up by more
DETAIL_STD = 0.5             # a texture with a smaller standard deviation is flat
ANCHOR_RATIO = 2.0           # runner-up distance / best distance for an anchor
ANCHOR_GAP = 0.5             # and runner-up minus best at least this
FIT_MIN_VALUE = 36000        # water-height values used for the terrain fit (clearly above sea level)
HEIGHT_GRID = "WaterSystemHeightMap"   # the other grids' placed cells are cross-checked against this one
METHODS = {"R16_UNORM": "content", "BC6H_UF16": "content", "B8G8R8A8_UNORM": "mask"}
ORDER_KEYS = {
    "text": lambda cell, cols: str(cell),
    "numeric": lambda cell, cols: cell,
    "row_col_text": lambda cell, cols: f"{cell // cols}_{cell % cols}",
    "col_row_text": lambda cell, cols: f"{cell % cols}_{cell // cols}",
}


# ---- plain Python: geometry, order, alignment, fit ------------------------------------------------------------

def geometry(doc: dict) -> dict:
    """Columns, rows and world box of a grid document; origin is the grid centre (docs/gamedata.md)."""
    tile, width, height = (doc[k] for k in ("tileSize", "width", "height"))
    if any(not float(v).is_integer() for v in (tile, width, height)):
        raise ValueError(f"grid size {width}x{height} or tile {tile} is not a whole number")
    tile, width, height = int(tile), int(width), int(height)
    if tile <= 0 or width % tile or height % tile:
        raise ValueError(f"grid {width}x{height} is not a whole number of {tile}-unit tiles")
    ox, oy, oz = (float(v) for v in doc["origin"])
    z = float(doc.get("zInflation", 0.0))
    return {"columns": width // tile, "rows": height // tile, "tile": tile,
            "x_min": ox - width / 2, "x_max": ox + width / 2,
            "y_min": oy - height / 2, "y_max": oy + height / 2,
            "z_min": oz - z, "z_max": oz + z}


def cell_box(geom: dict, cell: int) -> tuple:
    """World x/y box of a cell; row 0 is the north edge (largest y)."""
    row, col = divmod(cell, geom["columns"])
    if not 0 <= row < geom["rows"]:
        raise ValueError(f"cell {cell} is outside the {geom['columns']}x{geom['rows']} grid")
    t = geom["tile"]
    x0 = geom["x_min"] + col * t
    y1 = geom["y_max"] - row * t
    return (x0, y1 - t, x0 + t, y1)


def text_order(cells: int) -> list:
    """Cells in the order the tile dictionaries are stored: sorted as text of the decimal index."""
    return sorted(range(cells), key=str)


def ascending_steps(cells_in_archive_order: list, columns: int) -> dict:
    """For each spelling in ORDER_KEYS: [ascending steps, steps] of the given cells in archive order."""
    out = {}
    steps = max(len(cells_in_archive_order) - 1, 0)
    for name, key in ORDER_KEYS.items():
        keys = [key(c, columns) for c in cells_in_archive_order]
        out[name] = [sum(1 for a, b in zip(keys, keys[1:]) if b > a), steps]
    return out


def longest_increasing(pairs: list) -> list:
    """Longest subsequence of (tile, position) pairs, in tile order, whose positions strictly increase."""
    import bisect
    tails, tail_at, back = [], [], [-1] * len(pairs)
    for n, (_, p) in enumerate(pairs):
        k = bisect.bisect_left(tails, p)
        back[n] = tail_at[k - 1] if k else -1
        if k == len(tails):
            tails.append(p)
            tail_at.append(n)
        else:
            tails[k] = p
            tail_at[k] = n
    out, n = [], tail_at[-1] if tail_at else -1
    while n >= 0:
        out.append(pairs[n])
        n = back[n]
    return out[::-1]


def fill_by_order(n_tiles: int, anchors: dict, order: list) -> tuple:
    """Place tiles between anchors (tile index -> cell) by archive order.

    ``order`` lists the cells in storage order. The anchors kept are a longest subsequence whose cells strictly
    increase in that order; the others are dropped and counted. Between two kept anchors, k tiles fill the k cells
    between them only when there are exactly k free cells. Returns (placement dict tile -> cell, dropped anchors,
    tiles left unplaced).
    """
    pos = {c: i for i, c in enumerate(order)}
    kept = longest_increasing([(i, pos[anchors[i]]) for i in sorted(anchors)])
    dropped = len(anchors) - len(kept)
    placed = {i: order[p] for i, p in kept}
    left = 0
    bounds = [(-1, -1)] + kept + [(n_tiles, len(order))]
    for (i, a), (j, b) in zip(bounds, bounds[1:]):
        k = j - i - 1
        if k <= 0:
            continue
        if k == b - a - 1:
            for n in range(k):
                placed[i + 1 + n] = order[a + 1 + n]
        else:
            left += k
    return placed, dropped, left


def align(scores: list) -> list:
    """Order-constrained alignment: tile i (archive order) to slot path[i], strictly increasing, maximising the sum.

    ``scores[i][j]`` is tile i's score at slot j. Needs at least as many slots as tiles.
    """
    n = len(scores)
    if not n:
        return []
    m = len(scores[0])
    if m < n:
        raise ValueError(f"{n} tiles do not fit {m} slots in order")
    neg = float("-inf")
    best = list(scores[0])
    back = []
    for i in range(1, n):
        row = scores[i]
        cur = [neg] * m
        arg = [-1] * m
        run, run_at = neg, -1
        for j in range(1, m):
            if best[j - 1] > run:
                run, run_at = best[j - 1], j - 1
            if j >= i and run > neg:
                cur[j] = run + row[j]
                arg[j] = run_at
        back.append(arg)
        best = cur
    j = max(range(m), key=lambda k: best[k])
    path = [j]
    for arg in reversed(back):
        j = arg[j]
        path.append(j)
    return path[::-1]


def linear_fit(xs: list, ys: list) -> dict:
    """Least-squares y = slope * x + intercept, with Pearson r."""
    n = len(xs)
    if n < 2:
        raise ValueError("a fit needs at least two points")
    mx, my = math.fsum(xs) / n, math.fsum(ys) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    syy = math.fsum((y - my) ** 2 for y in ys)
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        raise ValueError("all x values are equal")
    slope = sxy / sxx
    r = sxy / math.sqrt(sxx * syy) if syy else 0.0
    return {"slope": slope, "intercept": my - slope * mx, "r": r, "n": n}


def inside(lo: float, hi: float, box_lo: float, box_hi: float) -> bool:
    return box_lo <= lo and hi <= box_hi


# ---- inputs -----------------------------------------------------------------------------------------------------

def load_manifests(work: str) -> list:
    """Every exported texture: dicts with archive, offset, dir, index, name, width, height, format, path."""
    assets = os.path.join(work, "assets")
    if not os.path.isdir(assets):
        raise ValueError(f"{work} has no assets folder (run make_game.py --stages assets)")
    out = []
    for archive in sorted(os.listdir(assets)):
        root = os.path.join(assets, archive, "textures")
        path = os.path.join(root, "manifest.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8-sig") as f:
            manifest = json.load(f)
        for item in manifest.get("Items", []):
            for t in item.get("Textures", []):
                out.append({"archive": archive, "offset": int(item["Offset"]), "dir": item["Dir"],
                            "index": int(t["Index"]), "name": t["Name"], "width": int(t["Width"]),
                            "height": int(t["Height"]), "format": t["Format"],
                            "path": os.path.join(root, item["Dir"], "%03d_%s.dds" % (t["Index"], t["Name"])),
                            "manifest": path})
    if not out:
        raise ValueError(f"no texture manifest under {assets}")
    return out


def load_documents(carve: str) -> dict:
    """GridAtlasInfo and TextureGridInfo documents of a carve, as plain values."""
    import gamedata_reader as gr
    reader = gr.Reader()
    docs = {"GridAtlasInfo": [], "TextureGridInfo": []}
    for doc in gr.documents([carve]):
        if doc.get("root_hint") not in docs:
            continue
        tag, _, value = reader.read(doc["data"])
        if tag in docs:
            docs[tag].append(value)
    if not docs["GridAtlasInfo"]:
        raise ValueError(f"no GridAtlasInfo document in {carve}")
    return docs


# ---- images (numpy / Pillow) ------------------------------------------------------------------------------------

def read_image(rec: dict):
    """A texture as a float array: (h, w) for R16, (h, w, c) otherwise."""
    import numpy as np
    with open(rec["path"], "rb") as f:
        data = f.read()
    fmt = rec["format"]
    if fmt == "R16_UNORM":
        if data[:4] != b"DDS " or data[84:88] != b"DX10" or struct.unpack_from("<I", data, 128)[0] != DXGI_R16_UNORM:
            raise ValueError(f"{rec['path']} is not a DX10 R16_UNORM DDS")
        count = rec["width"] * rec["height"]
        if len(data) < DX10_HEADER + 2 * count:
            raise ValueError(f"{rec['path']} is truncated")
        return np.frombuffer(data, "<u2", count, DX10_HEADER).reshape(rec["height"], rec["width"]).astype(float)
    if fmt.startswith("BC6H"):
        import io
        from PIL import Image
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("RGB"), dtype=float)
    import dds_png
    w, h, rgba = dds_png.decode_dds(data)
    return np.frombuffer(rgba, np.uint8).reshape(h, w, 4).astype(float)


def split_cells(fallback, geom: dict):
    """Fallback mosaic -> array (cells, p, p[, c]) with cell = row * columns + column, row 0 at the top."""
    rows, cols = geom["rows"], geom["columns"]
    h, w = fallback.shape[:2]
    if w % cols or h != rows * (w // cols):
        raise ValueError(f"fallback {w}x{h} is not a {cols}x{rows} mosaic of square cells")
    p = w // cols
    rest = fallback.shape[2:]
    a = fallback.reshape(rows, p, cols, p, *rest).swapaxes(1, 2)
    return a.reshape(rows * cols, p, p, *rest)


def point_sample(tile, p: int):
    step = tile_step(tile, p)
    return tile[step // 2::step, step // 2::step][:p, :p]


def tile_step(tile, p: int) -> int:
    """Tile pixels per fallback pixel; the tile must be square and a whole multiple of the fallback cell."""
    h, w = tile.shape[:2]
    if h != w or w < p or w % p:
        raise ValueError(f"a {w}x{h} tile does not fit a fallback cell of {p}x{p} pixels")
    return w // p


def match_content(tiles: list, fallbacks: list, geom: dict) -> dict:
    """tiles: per dictionary, a list of images (one per grid); fallbacks: the matching mosaics."""
    import numpy as np
    cells = [split_cells(f, geom) for f in fallbacks]
    if cells[0].shape[0] < 2:
        raise ValueError("a grid needs at least two cells to rank a match")
    p = cells[0].shape[1]
    best, runner, detail = [], [], []
    for images in tiles:
        d = np.zeros(cells[0].shape[0])
        for img, c in zip(images, cells):
            s = point_sample(img, p)
            d += np.abs(c - s).reshape(len(c), -1).mean(1)
        o = np.argsort(d, kind="stable")[:2]
        best.append((int(o[0]), float(d[o[0]])))
        runner.append(float(d[o[1]]))
        detail.append(any(float(img.std()) > DETAIL_STD for img in images))
    anchors = {i: best[i][0] for i in range(len(tiles))
               if detail[i] and runner[i] >= ANCHOR_RATIO * max(best[i][1], 1e-9)
               and runner[i] - best[i][1] >= ANCHOR_GAP}
    return {"anchors": anchors, "detail": detail}


def match_mask(tiles: list, fallback, geom: dict) -> dict:
    """Feature-mask IoU of each tile against each cell, then order-constrained alignment to text order."""
    import numpy as np
    flat = fallback.reshape(-1, fallback.shape[-1])
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colours[int(counts.argmax())]
    cells = split_cells(fallback, geom)
    p = cells.shape[1]
    cm = (np.abs(cells - bg).max(-1) > MASK_THRESHOLD).astype(np.int32)
    tm = []
    for img in tiles:
        b = tile_step(img, p)
        m = (np.abs(img - bg).max(-1) > MASK_THRESHOLD).reshape(p, b, p, b).mean((1, 3)) > MASK_BLOCK_SHARE
        tm.append(m)
    tm = np.array(tm, dtype=np.int32)
    inter = np.einsum("tij,cij->tc", tm, cm)
    union = tm.sum((1, 2))[:, None] + cm.sum((1, 2))[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    top = np.sort(iou, 1)[:, ::-1]
    clear = {i: int(iou[i].argmax()) for i in range(len(tiles)) if top[i, 0] - top[i, 1] > CLEAR_MARGIN}
    order = text_order(len(cells))
    path = align(iou[:, order].tolist())
    placed = {i: order[j] for i, j in enumerate(path) if iou[i, order[j]] > 0}
    scores = sorted(float(iou[i, c]) for i, c in placed.items())
    featureless = tm.sum((1, 2)) == 0
    return {"background": [int(v) for v in bg], "clear": clear, "placed": placed,
            "featureless_tiles": int(featureless.sum()),
            "unplaced_with_features": sum(1 for i in range(len(tiles)) if i not in placed and not featureless[i]),
            "anchors_moved_by_alignment": sum(1 for i, c in clear.items() if placed.get(i) != c),
            "feature_cells": int((cm.sum((1, 2)) > 0).sum()),
            "overlap_median": scores[len(scores) // 2] if scores else None}


def fallback_stats(fallback, geom: dict) -> dict:
    import numpy as np
    cells = split_cells(fallback, geom).reshape(geom["rows"] * geom["columns"], -1)
    const = cells.min(1) == cells.max(1)
    out = {"constant_cells": int(const.sum()), "value_min": float(fallback.min()), "value_max": float(fallback.max())}
    if fallback.ndim == 2:
        vals, counts = np.unique(fallback, return_counts=True)
        out["most_common_value"] = float(vals[counts.argmax()])
        out["most_common_share"] = round(float(counts.max()) / fallback.size, 4)
    return out


def water_fit(fallback, geom: dict, payloads: list) -> tuple:
    """Fit water-height fallback values against each MHIU terrain grid, north-up and (for contrast) south-up.

    Returns (one row per MHIU payload, the fallback's value range in metres under the best north-up fit or None).
    """
    import numpy as np
    rows, cols = fallback.shape
    px = (geom["x_max"] - geom["x_min"]) / cols
    r, c = np.mgrid[0:rows, 0:cols]
    x = geom["x_min"] + (c + 0.5) * px
    out, fits = [], []
    for pl in payloads:
        if pl.get("type") != "MHIU":
            continue
        b = pl["header"]["bounds"]
        w, h = pl["header"]["width"], pl["header"]["height"]
        z = np.frombuffer(pl["_grid"], np.uint8).reshape(h, w).astype(float)
        z = b["min"][2] + z / 255.0 * (b["max"][2] - b["min"][2])
        row = {"heightmap": "MHIU " + pl.get("offset_hex", "")}
        for label, y in (("north_up", geom["y_max"] - (r + 0.5) * px), ("south_up", geom["y_min"] + (r + 0.5) * px)):
            i = np.floor((x - b["min"][0]) / (b["max"][0] - b["min"][0]) * w).astype(int)
            j = np.floor((y - b["min"][1]) / (b["max"][1] - b["min"][1]) * h).astype(int)
            ok = (i >= 0) & (i < w) & (j >= 0) & (j < h) & (fallback > FIT_MIN_VALUE)
            row[label] = None
            if ok.sum() < 2:
                continue
            try:
                fit = linear_fit(fallback[ok].tolist(), z[j[ok], i[ok]].tolist())
            except ValueError:
                continue
            if label == "north_up":
                fits.append(fit)
            row[label] = {"n": fit["n"], "r": round(fit["r"], 4),
                          "units_per_metre": round(1 / fit["slope"], 3) if fit["slope"] else None,
                          "zero_metre_value": round(-fit["intercept"] / fit["slope"], 1) if fit["slope"] else None}
        out.append(row)
    metres = None
    good = [f for f in fits if f["slope"]]
    if good:
        fit = max(good, key=lambda f: f["r"])
        lo, hi = float(fallback.min()), float(fallback.max())
        metres = {"min": round(fit["slope"] * lo + fit["intercept"], 1),
                  "max": round(fit["slope"] * hi + fit["intercept"], 1), "r": round(fit["r"], 4)}
    return out, metres


# ---- driver ----------------------------------------------------------------------------------------------------

def run(textures: list, carve: str) -> dict:
    import gamedata_maps as gm
    by_name = {}
    for t in textures:
        by_name.setdefault(t["name"], []).append(t)
    docs = load_documents(carve)
    grids = sorted(docs["GridAtlasInfo"], key=lambda d: str(d.get("clientSystemName")))
    payloads = gm.read_payloads(carve)

    # group grids whose tiles live in the same dictionaries (matched jointly)
    keys = {}
    for doc in grids:
        recs = by_name.get(doc["tileTextureNameSuffix"], [])
        main = main_archive(recs)
        keys.setdefault(frozenset((t["archive"], t["offset"]) for t in recs if t["archive"] == main), []).append(doc)

    summary = {"grids": [], "texture_grids": [], "heightmaps": [], "placements": {}}
    height_cells = None
    for docset in sorted(keys.values(), key=lambda ds: all(d.get("clientSystemName") != HEIGHT_GRID for d in ds)):
        is_height = any(d.get("clientSystemName") == HEIGHT_GRID for d in docset)
        geom = geometry(docset[0])
        for d in docset[1:]:
            if geometry(d) != geom:
                raise ValueError("grids that share tile dictionaries have different geometry")
        main = main_archive(by_name.get(docset[0]["tileTextureNameSuffix"], []))
        per_doc, duplicates = [], 0
        for doc in docset:
            recs = [t for t in by_name.get(doc["tileTextureNameSuffix"], []) if t["archive"] == main]
            keyed = {(t["offset"], t["dir"]): t for t in recs}
            duplicates += len(recs) - len(keyed)
            per_doc.append(keyed)
        dicts = sorted(set.intersection(*(set(p) for p in per_doc)))
        counts = {"duplicate_records": duplicates,
                  "unshared_dictionaries": sum(len(p) for p in per_doc) - len(per_doc) * len(dicts)}
        if not dicts:
            for doc in docset:
                summary["grids"].append({"name": doc.get("clientSystemName"), "tile_name": doc["tileTextureNameSuffix"],
                                         "dictionaries": 0, **counts})
            continue
        first = per_doc[0][dicts[0]]
        fmt = first["format"]
        method = METHODS.get(fmt)
        if method is None:
            raise ValueError(f"no matching method for tile format {fmt}")
        fallbacks, fallback_recs = [], []
        for doc in docset:
            fb = by_name.get(doc["globalFallbackTextureName"], [])
            if len(fb) != 1:
                raise ValueError(f"expected one {doc['globalFallbackTextureName']}, found {len(fb)}")
            fallbacks.append(read_image(fb[0]))
            fallback_recs.append(fb[0])
        result = {"method": method, "dictionaries": len(dicts), "tile_format": fmt,
                  "tile_size": [first["width"], first["height"]], **counts}
        if method == "content":
            tiles = [[read_image(p[k]) for p in per_doc] for k in dicts]
            m = match_content(tiles, fallbacks, geom)
            anchors = m["anchors"]
            placed, dropped, left = fill_by_order(len(dicts), anchors, text_order(geom["rows"] * geom["columns"]))
            by_order = {i: c for i, c in placed.items() if i not in anchors}
            result.update({"tiles_with_detail": sum(m["detail"]), "anchors": len(anchors),
                           "anchor_order": ascending_steps([anchors[i] for i in sorted(anchors)], geom["columns"]),
                           "anchors_dropped": dropped, "placed_by_order": len(by_order), "unplaced": left,
                           "unplaced_without_detail": sum(1 for i in range(len(dicts))
                                                          if i not in placed and not m["detail"][i])})
            anchor_cells = [c for i, c in anchors.items() if placed.get(i) == c]
        else:
            tiles = [read_image(per_doc[0][k]) for k in dicts]
            m = match_mask(tiles, fallbacks[0], geom)
            placed = m.pop("placed")
            clear = m.pop("clear")
            by_order = {}
            result.update(m)
            result.update({"anchors": len(clear), "anchor_order": ascending_steps(
                [clear[i] for i in sorted(clear)], geom["columns"]), "unplaced": len(dicts) - len(placed)})
            anchor_cells = list(clear.values())
        result["placed"] = len(placed)
        result["distinct_cells"] = len(set(placed.values()))
        if is_height:
            height_cells = set(placed.values())
        elif height_cells is not None:
            result["in_water_height_cells"] = sum(1 for c in set(placed.values()) if c in height_cells)
            result["anchors_in_water_height_cells"] = sum(1 for c in anchor_cells if c in height_cells)
            result["placed_by_order_in_water_height_cells"] = sum(1 for c in by_order.values() if c in height_cells)
        for doc, fb, fb_rec in zip(docset, fallbacks, fallback_recs):
            tile_archives = archive_counts(by_name.get(doc["tileTextureNameSuffix"], []))
            g = {"name": doc.get("clientSystemName"), "tile_name": doc["tileTextureNameSuffix"], **result,
                 "tile_archives": tile_archives,
                 "tiles_in_other_archives": sum(n for a, n in tile_archives.items() if a != main),
                 "fallback": doc["globalFallbackTextureName"], "fallback_size": list(fb.shape[1::-1]),
                 "fallback_format": fb_rec["format"], "fallback_stats": fallback_stats(fb, geom), "geometry": geom,
                 "z_inflation": float(doc.get("zInflation", 0.0)),
                 "sector_fallbacks": {o.get("sectorOverrideGlobalFallbackName"):
                                      archive_counts(by_name.get(o.get("sectorOverrideGlobalFallbackName"), []))
                                      for o in doc.get("sectorOverrides") or []}}
            if is_height and fmt == "R16_UNORM":
                g["water_fit"], g["water_metres"] = water_fit(fb, geom, payloads)
            summary["grids"].append(g)
            summary["placements"][doc["tileTextureNameSuffix"]] = [
                {"archive": main, "offset": dicts[i][0], "dir": dicts[i][1], "cell": c,
                 "row": c // geom["columns"], "column": c % geom["columns"]} for i, c in sorted(placed.items())]

    for doc in docs["TextureGridInfo"]:
        summary["texture_grids"].append({
            "geometry": geometry(doc), "z_inflation": float(doc.get("zInflation", 0.0)),
            "fallbacks": {k: {"name": doc.get(k), "archives": archive_counts(by_name.get(doc.get(k), []))}
                          for k in ("lowDetailFallBack", "highDetailFallBack")}})

    for pl in payloads:
        b = pl["header"]["bounds"]
        summary["heightmaps"].append({"type": pl["type"], "offset": pl.get("offset_hex"),
                                      "z_min": b["min"][2], "z_max": b["max"][2]})
    zboxes = {(g["geometry"]["z_min"], g["geometry"]["z_max"]) for g in summary["grids"] if "geometry" in g}
    summary["z_check"] = [{"z_box": list(z), "heightmaps_inside": sum(
        inside(h["z_min"], h["z_max"], *z) for h in summary["heightmaps"]), "heightmaps": len(summary["heightmaps"])}
        for z in sorted(zboxes)]
    return summary


def main_archive(recs: list):
    """The archive holding most of these records (ties: the later name)."""
    counts = archive_counts(recs)
    return max(counts, key=lambda a: (counts[a], a)) if counts else None


def archive_counts(recs: list) -> dict:
    out = {}
    for t in recs:
        out[t["archive"]] = out.get(t["archive"], 0) + 1
    return out


def _within(path: str, folder: str) -> bool:
    """True when path is folder or lies under it, after resolving links (case-folded where the volume is)."""
    a, b = os.path.realpath(path), os.path.realpath(folder)
    try:
        common = os.path.commonpath([a, b])
    except ValueError:              # different drives on Windows
        return False
    return report_paths.path_key(common) == report_paths.path_key(b)


def check_out_path(out: str, work: str, carve: str) -> None:
    """Refuse a --json path in the exported assets (manifests, textures) or in the carve folder."""
    for folder, what in ((os.path.join(work, "assets"), "the exported assets"), (carve, "the carve folder")):
        if _within(out, folder):
            raise ValueError(f"--json {out} is inside {what} ({folder}), which the tool reads")


def report_lines(summary: dict) -> list:
    out = []
    for g in summary["grids"]:
        if not g["dictionaries"]:
            out.append(f"{g['name']}: no tile dictionaries of {g['tile_name']} found")
            continue
        order = g["anchor_order"]
        text, num = order["text"], order["numeric"]
        line = (f"{g['name']}: {g['dictionaries']} tile dictionaries of {g['tile_name']} "
                f"({g['tile_size'][0]}x{g['tile_size'][1]} {g['tile_format']}), fallback {g['fallback_size'][0]}x"
                f"{g['fallback_size'][1]} {g['fallback_format']}; {g['method']}: {g['anchors']} anchors "
                f"(text order {text[0]}/{text[1]} ascending, numeric {num[0]}/{num[1]}")
        if "anchors_dropped" in g:
            line += f", {g['anchors_dropped']} dropped), {g['placed_by_order']} placed by order"
        else:
            line += f", {g['anchors_moved_by_alignment']} moved by the alignment)"
        line += f"; {g['placed']} placed in {g['distinct_cells']} cells, {g['unplaced']} unplaced"
        if "unplaced_without_detail" in g:
            line += f" ({g['unplaced_without_detail']} without detail)"
        if "featureless_tiles" in g:
            line += f" ({g['featureless_tiles']} without features)"
        out.append(line)
        notes = []
        if g["tiles_in_other_archives"]:
            notes.append(f"{g['tiles_in_other_archives']} tiles in other archives not placed "
                         f"({', '.join(f'{a} {n}' for a, n in sorted(g['tile_archives'].items()))})")
        if g["duplicate_records"]:
            notes.append(f"{g['duplicate_records']} duplicate tile records ignored")
        if g["unshared_dictionaries"]:
            notes.append(f"{g['unshared_dictionaries']} dictionaries hold only some textures of the group, ignored")
        if "in_water_height_cells" in g:
            notes.append(f"cells with a water-height tile: {g['in_water_height_cells']} of {g['placed']} placed, "
                         f"{g['anchors_in_water_height_cells']} of {g['anchors']} anchors, "
                         f"{g['placed_by_order_in_water_height_cells']} of {g.get('placed_by_order', 0)} "
                         f"placed by order")
        if "water_fit" in g:
            fits = "; ".join(
                f"{f['heightmap']}: {f['north_up']['units_per_metre']} units/m, 0 m at "
                f"{f['north_up']['zero_metre_value']}, r {f['north_up']['r']} "
                f"(south-up r {f['south_up']['r'] if f.get('south_up') else None})"
                for f in g["water_fit"] if f.get("north_up"))
            if g["water_metres"]:
                notes.append(f"water height vs terrain (values > {FIT_MIN_VALUE}): {fits}; fallback range "
                             f"{g['water_metres']['min']} to {g['water_metres']['max']} m")
            else:
                notes.append("no MHIU heightmap to fit the water heights against")
        out.extend("  " + n for n in notes)
    for z in summary["z_check"]:
        if not z["heightmaps"]:
            out.append(f"z box {z['z_box'][0]:g} to {z['z_box'][1]:g}: no heightmap in the carve, not checked")
        else:
            out.append(f"z box {z['z_box'][0]:g} to {z['z_box'][1]:g}: {z['heightmaps_inside']} of "
                       f"{z['heightmaps']} heightmap Z ranges inside")
    return out


def write_json(path, data):
    """Publish `data` as JSON at `path` through a unique temporary file in the same folder.

    A fixed `path + ".tmp"` would block every later run while a stale file or folder keeps that name, and two
    runs writing one output would share it. On any failure the temporary file is removed and `path` is untouched.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        try:
            f = os.fdopen(fd, "w", encoding="utf-8", newline="\n")
        except BaseException:
            os.close(fd)
            raise
        with f:
            json.dump(data, f, indent=1)
        os.chmod(tmp, published_mode(path))   # mkstemp creates 0600
        os.replace(tmp, path)
    except BaseException:
        try:
            os.chmod(tmp, 0o600)   # a read-only destination's mode would stop the unlink on Windows
            os.unlink(tmp)
        except OSError:
            pass
        raise


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("work", help="make_game.py work folder (with assets/<archive>/textures/manifest.json)")
    ap.add_argument("--carve", required=True, help="rpf9_carve.py output of common.rpf")
    ap.add_argument("--json", help="write the summary and tile placements here (game-derived: keep it local)")
    a = ap.parse_args(argv)
    try:
        if not os.path.isdir(a.carve):
            raise ValueError(f"--carve {a.carve} is not a folder")
        if a.json:
            if os.path.isdir(a.json):
                raise ValueError(f"--json {a.json} is a folder")
            check_out_path(a.json, a.work, a.carve)
        summary = run(load_manifests(a.work), a.carve)
    except KeyError as e:
        print(f"world_grid_atlas: missing field {e} in an input", file=sys.stderr)
        return 2
    except (OSError, ValueError) as e:
        print(f"world_grid_atlas: {e}", file=sys.stderr)
        return 2
    except ImportError as e:
        need = " (this tool needs numpy and Pillow)" if e.name in ("numpy", "PIL") else ""
        print(f"world_grid_atlas: {e}{need}", file=sys.stderr)
        return 2
    for line in report_lines(summary):
        print(line)
    if a.json:
        try:
            write_json(a.json, summary)
        except OSError as e:
            print(f"world_grid_atlas: cannot write {a.json}: {e}", file=sys.stderr)
            return 2
        print(f"wrote {a.json} (tile placements are game-derived: keep the file local)")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
