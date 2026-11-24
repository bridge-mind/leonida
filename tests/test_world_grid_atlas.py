"""world_grid_atlas.py: grid geometry, archive order, alignment and a forward check on synthetic grids.

    python -I tests/test_world_grid_atlas.py

The order, alignment, fill and fit logic is plain Python and is tested on the standard library alone. The forward
check builds a synthetic work folder and carve: four grids on an 8 x 12 tile map (cells 0..95, so text order and
numeric order differ), tile textures written as DDS files at known cells and stored in text order of the cell, the
fallback mosaics built from those tiles, grid documents as XML and an MHIU terrain grid. It runs the CLI and checks
that every tile comes back at its cell. Those tests need numpy and Pillow and are skipped without them. No game
data is used.
"""
import itertools
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
TOOL = os.path.join(TOOLS, "world_grid_atlas.py")
sys.path.insert(0, TOOLS)

import world_grid_atlas as wga  # noqa: E402

try:
    import numpy as np
    import PIL  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

DOC = {"origin": (0.0, 0.0, 0.0), "width": 4096, "height": 6144, "tileSize": 512, "zInflation": 1000.0}


class PlainLogicTest(unittest.TestCase):
    def test_geometry_of_the_measured_grid(self):
        g = wga.geometry({"origin": (-5120.0, 7680.0, 0.0), "width": 16384, "height": 33792,
                          "tileSize": 512, "zInflation": 1000.0})
        self.assertEqual((g["columns"], g["rows"]), (32, 66))
        self.assertEqual((g["x_min"], g["x_max"], g["y_min"], g["y_max"]), (-13312, 3072, -9216, 24576))
        self.assertEqual((g["z_min"], g["z_max"]), (-1000, 1000))

    def test_geometry_rejects_a_partial_tile(self):
        with self.assertRaises(ValueError):
            wga.geometry({**DOC, "width": 4000})
        with self.assertRaises(ValueError):
            wga.geometry({**DOC, "tileSize": 0})
        with self.assertRaises(ValueError):
            wga.geometry({**DOC, "width": 4096.5})

    def test_row_zero_is_north(self):
        g = wga.geometry(DOC)
        self.assertEqual(wga.cell_box(g, 0), (-2048, 2560, -1536, 3072))
        self.assertEqual(wga.cell_box(g, 95), (1536, -3072, 2048, -2560))
        with self.assertRaises(ValueError):
            wga.cell_box(g, 96)

    def test_text_order(self):
        self.assertEqual(wga.text_order(12), [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_ascending_steps_per_spelling(self):
        cells = [1, 10, 11, 2, 20]                     # text order
        s = wga.ascending_steps(cells, 8)
        self.assertEqual(s["text"], [4, 4])
        self.assertEqual(s["numeric"], [3, 4])
        self.assertEqual(wga.ascending_steps([], 8)["text"], [0, 0])

    def test_fill_by_order_places_exact_gaps_only(self):
        order = list(range(10))
        placed, dropped, left = wga.fill_by_order(6, {0: 1, 3: 4, 5: 9}, order)
        # tiles 1, 2 fill cells 2, 3 exactly; tile 4 has 4 free cells (5..8) for 1 tile: left unplaced
        self.assertEqual(placed, {0: 1, 1: 2, 2: 3, 3: 4, 5: 9})
        self.assertEqual((dropped, left), (0, 1))

    def test_fill_by_order_fills_the_ends(self):
        placed, _, left = wga.fill_by_order(3, {1: 1}, [0, 1, 2])
        self.assertEqual(placed, {0: 0, 1: 1, 2: 2})
        self.assertEqual(left, 0)

    def test_fill_by_order_drops_an_anchor_out_of_order(self):
        placed, dropped, _ = wga.fill_by_order(3, {0: 5, 1: 2, 2: 7}, list(range(10)))
        # cells 5, 2, 7: (5, 7) and (2, 7) are equally long, so either may be kept; one anchor is dropped
        self.assertEqual(dropped, 1)
        self.assertEqual(placed[2], 7)
        cells = [placed[i] for i in sorted(placed)]
        self.assertTrue(all(b > a for a, b in zip(cells, cells[1:])))

    def test_fill_by_order_keeps_the_longest_consistent_anchors(self):
        """One wrong early anchor must not drop the correct ones after it."""
        placed, dropped, left = wga.fill_by_order(5, {0: 9, 1: 1, 2: 2, 3: 3, 4: 4}, list(range(10)))
        self.assertEqual(dropped, 1)
        self.assertEqual({i: placed[i] for i in (1, 2, 3, 4)}, {1: 1, 2: 2, 3: 3, 4: 4})
        self.assertNotEqual(placed.get(0), 9)
        # two anchors on one cell: only one is kept
        placed, dropped, _ = wga.fill_by_order(2, {0: 3, 1: 3}, list(range(5)))
        self.assertEqual(dropped, 1)

    def test_longest_increasing_matches_brute_force(self):
        rnd = random.Random(7)
        for _ in range(300):
            pos = [rnd.randrange(8) for _ in range(rnd.randint(0, 8))]
            pairs = list(enumerate(pos))
            got = wga.longest_increasing(pairs)
            self.assertTrue(all(b[1] > a[1] and b[0] > a[0] for a, b in zip(got, got[1:])))
            best = 0
            for k in range(len(pos), 0, -1):
                if any(all(b > a for a, b in zip(c, c[1:])) for c in itertools.combinations(pos, k)):
                    best = k
                    break
            self.assertEqual(len(got), best)

    def test_main_archive_and_counts(self):
        recs = [{"archive": a} for a in ("levels", "levels", "minigame", "levels")]
        self.assertEqual(wga.archive_counts(recs), {"levels": 3, "minigame": 1})
        self.assertEqual(wga.main_archive(recs), "levels")
        self.assertIsNone(wga.main_archive([]))

    def test_inside_is_inclusive(self):
        self.assertTrue(wga.inside(-1000.0, 1000.0, -1000.0, 1000.0))
        self.assertFalse(wga.inside(-1000.5, 0.0, -1000.0, 1000.0))
        self.assertFalse(wga.inside(0.0, 1000.5, -1000.0, 1000.0))

    def test_align_matches_brute_force(self):
        rnd = random.Random(340)
        for _ in range(300):
            n, m = rnd.randint(1, 4), rnd.randint(4, 7)
            if n > m:
                continue
            s = [[rnd.choice([0.0, 0.0, rnd.random()]) for _ in range(m)] for _ in range(n)]
            path = wga.align(s)
            self.assertEqual(len(path), n)
            self.assertTrue(all(b > a for a, b in zip(path, path[1:])))
            best = max(sum(s[i][j] for i, j in enumerate(c)) for c in itertools.combinations(range(m), n))
            self.assertAlmostEqual(sum(s[i][j] for i, j in enumerate(path)), best)

    def test_align_needs_enough_slots(self):
        self.assertEqual(wga.align([]), [])
        with self.assertRaises(ValueError):
            wga.align([[1.0], [1.0]])

    def test_linear_fit(self):
        xs = [float(x) for x in range(33000, 50000, 500)]
        f = wga.linear_fit(xs, [(x - 32768) / 40.96 for x in xs])
        self.assertAlmostEqual(1 / f["slope"], 40.96, places=6)
        self.assertAlmostEqual(-f["intercept"] / f["slope"], 32768, places=3)
        self.assertAlmostEqual(f["r"], 1.0)
        with self.assertRaises(ValueError):
            wga.linear_fit([1.0, 1.0], [2.0, 3.0])

    def test_load_manifests_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as td:
            tex_dir = os.path.join(td, "assets", "levels", "textures")
            os.makedirs(tex_dir)
            manifest = {
                "Items": [{
                    "Offset": 0x1000,
                    "Dir": "dictionary_at_0000000000001000",
                    "Textures": [{
                        "Index": 0,
                        "Name": "watersystemheightmap",
                        "Width": 64,
                        "Height": 64,
                        "Format": "R16_UNORM",
                    }],
                }]
            }
            with open(os.path.join(tex_dir, "manifest.json"), "wb") as f:
                f.write(b"\xef\xbb\xbf" + json.dumps(manifest).encode("utf-8"))
            recs = wga.load_manifests(td)
            self.assertEqual(len(recs), 1)
            self.assertEqual(recs[0]["name"], "watersystemheightmap")


# ---- synthetic work folder and carve ---------------------------------------------------------------------------

def dds(width, height, dxgi, payload, bpp):
    head = bytearray(148)
    head[:4] = b"DDS "
    struct.pack_into("<7I", head, 4, 124, 0x1 | 0x2 | 0x4 | 0x8 | 0x1000, height, width, width * bpp, 0, 1)
    struct.pack_into("<2I4s", head, 76, 32, 0x4, b"DX10")
    struct.pack_into("<I", head, 108, 0x1000)
    struct.pack_into("<5I", head, 128, dxgi, 3, 0, 1, 0)
    return bytes(head) + payload


def r16_dds(a):
    return dds(a.shape[1], a.shape[0], 56, a.astype("<u2").tobytes(), 2)


def bgra_dds(rgba):
    return dds(rgba.shape[1], rgba.shape[0], 87, rgba[..., [2, 1, 0, 3]].astype(np.uint8).tobytes(), 4)


def grid_xml(name, suffix, fallback):
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n<GridAtlasInfo>\n'
            f'\t<type>GS_TYPE_2D_TEXTURE_ATLAS</type>\n\t<clientSystemName>{name}</clientSystemName>\n'
            f'\t<tileTextureNamePrefix>{name}-</tileTextureNamePrefix>\n'
            f'\t<tileTextureNameSuffix>{suffix}</tileTextureNameSuffix>\n'
            f'\t<hasOwnGlobalFallback value="true"/>\n'
            f'\t<globalFallbackTextureName>{fallback}</globalFallbackTextureName>\n'
            f'\t<useGlobalFallback value="true" />\n\t<origin x="0.0" y="0.0" z="0.0" />\n'
            f'\t<width value="4096" />\n\t<height value="6144" />\n\t<tileSize value="512" />\n'
            f'\t<streamRadius value="512.0" />\n\t<zInflation value="1000.0" />\n'
            f'\t<sectorOverrides>\n\t\t<Item>\n\t\t\t<sectorName>mapswap_test</sectorName>\n'
            f'\t\t\t<sectorOverrideGlobalFallbackName>{fallback}_tblux</sectorOverrideGlobalFallbackName>\n'
            f'\t\t\t<suffix>t-</suffix>\n\t\t</Item>\n\t</sectorOverrides>\n</GridAtlasInfo>\n').encode()


def mhiu(width, height, bounds, heights):
    """Full-width rows, default 23; `heights` is a list of rows of u8 values (row 0 = min y)."""
    table, packed = bytearray(), bytearray()
    for row in heights:
        table += struct.pack("<HHi", 0, width, len(packed))
        packed += bytes(row)
    data = bytes(table) + bytes(packed)
    head = bytearray(44)
    head[:4] = b"MHIU"
    struct.pack_into("<HHHH", head, 4, 1, width, height, 0)
    struct.pack_into("<6f", head, 0x0C, *bounds)
    struct.pack_into("<I", head, 0x24, len(data))
    head[0x28] = 23
    return bytes(head) + data


def terrain(x, y):
    """Synthetic ground height in metres, rising to the north so a flipped row order shows."""
    return 100.0 + 0.04 * (y + 3072) + 20.0 * math.sin(x / 300.0)


class WriteJsonTest(unittest.TestCase):
    """--json publishes through a unique temporary file (stdlib only)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def leftovers(self, name, keep=()):
        return [n for n in os.listdir(self.root) if n.startswith(name + ".") and n not in keep]

    def test_stale_fixed_temp_folder_does_not_block_the_write(self):
        out = os.path.join(self.root, "grid.json")
        os.makedirs(out + ".tmp")
        wga.write_json(out, {"a": 1})
        with open(out, encoding="utf-8") as f:
            self.assertEqual(json.load(f), {"a": 1})
        self.assertTrue(os.path.isdir(out + ".tmp"))
        self.assertEqual(self.leftovers("grid.json", keep=("grid.json.tmp",)), [])

    def test_failed_replace_keeps_the_old_file_and_no_temp(self):
        out = os.path.join(self.root, "grid.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write("old")
        with mock.patch.object(wga.os, "replace", side_effect=OSError("replace denied")):
            with self.assertRaises(OSError):
                wga.write_json(out, {"a": 1})
        with open(out, encoding="utf-8") as f:
            self.assertEqual(f.read(), "old")
        self.assertEqual(self.leftovers("grid.json"), [])

    @unittest.skipIf(os.name == "nt", "POSIX permission bits")
    def test_keeps_the_replaced_file_mode(self):
        out = os.path.join(self.root, "grid.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write("old")
        os.chmod(out, 0o640)
        wga.write_json(out, [1, 2])
        self.assertEqual(os.stat(out).st_mode & 0o777, 0o640)


@unittest.skipUnless(HAVE_NUMPY, "needs numpy and Pillow")
class ForwardCheckTest(unittest.TestCase):
    COLS, ROWS = 8, 12

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        self.work = os.path.join(self.root, "work")
        self.carve = os.path.join(self.root, "carve")
        os.makedirs(self.carve)
        self.rnd = np.random.default_rng(340)
        self.geom = wga.geometry(DOC)
        self.items = {}          # archive -> list of (offset, dir, [texture records])
        self.files = {}          # path -> bytes
        self.rows = []

    def tearDown(self):
        self.tmp.cleanup()

    # -- writers --
    def add_dict(self, archive, textures):
        items = self.items.setdefault(archive, [])
        offset = 0x2010 + 0x10000 * len(items)
        d = f"dictionary_at_{offset:016X}"
        recs = []
        for index, (name, fmt, data, w, h) in enumerate(textures):
            path = os.path.join(self.work, "assets", archive, "textures", d, f"{index:03d}_{name}.dds")
            self.files[path] = data
            recs.append({"Index": index, "Name": name, "Width": w, "Height": h, "Format": fmt,
                         "MipCount": 1, "Hash": "00000000", "Key": None})
        items.append({"Offset": offset, "Dir": d, "Textures": recs})

    def add_stream(self, data):
        off = 0x1000 * (len(self.rows) + 1)
        with open(os.path.join(self.carve, f"{off:012x}.bin"), "wb") as f:
            f.write(data)
        self.rows.append({"off": off, "stored": len(data), "size": len(data), "head": data[:16].hex(), "pre": ""})

    def finish(self):
        for path, data in self.files.items():
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
        for archive, items in self.items.items():
            with open(os.path.join(self.work, "assets", archive, "textures", "manifest.json"), "w") as f:
                json.dump({"src": archive, "Count": len(items), "Scan": {}, "Items": items}, f)
        with open(os.path.join(self.carve, "index.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": {"archive": "synthetic", "files": len(self.rows)}, "files": self.rows}, f)

    def run_tool(self, *extra):
        out = os.path.join(self.root, "out.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.work, "--carve", self.carve, "--json", out, *extra],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        data = None
        if p.returncode == 0:
            with open(out, encoding="utf-8") as f:
                data = json.load(f)
        return p, data

    # -- world --
    def mosaic(self, cells, p, fill, channels=None):
        shape = (self.ROWS * p, self.COLS * p) + ((channels,) if channels else ())
        out = np.empty(shape)
        out[...] = fill
        for cell, block in cells.items():
            r, c = divmod(cell, self.COLS)
            out[r * p:(r + 1) * p, c * p:(c + 1) * p] = block
        return out

    def build(self, order_key=str):
        """Write the four grids; tile dictionaries are stored sorted by order_key(cell)."""
        g = self.geom
        truth = {}
        # water height: 64 px tiles, 4 px per cell in the fallback; cell 10 is a flat tile between 1 and 11
        height_cells = sorted((set(int(c) for c in self.rnd.choice(96, 24, replace=False)) | {1, 11, 12}) - {10})
        tiles, fb = {}, {}
        for cell in height_cells + [10]:
            x0, y0, x1, y1 = wga.cell_box(g, cell)
            if cell == 10:
                t = np.full((64, 64), 20480.0)
            else:
                ys = y1 - (np.arange(64) + 0.5) * 8
                xs = x0 + (np.arange(64) + 0.5) * 8
                z = np.array([[terrain(x, y) for x in xs] for y in ys])
                t = np.round(32768 + 40.96 * z)
            tiles[cell] = t
            fb[cell] = t[8::16, 8::16]
        for cell in sorted(tiles, key=order_key):
            self.add_dict("levels", [("watersystemheightmap", "R16_UNORM", r16_dds(tiles[cell]), 64, 64)])
        truth["watersystemheightmap"] = sorted(tiles, key=order_key)
        self.height_fallback = self.mosaic(fb, 4, 20480.0)
        self.add_dict("levels", [("watersystemheightmap_gfb", "R16_UNORM", r16_dds(self.height_fallback), 32, 48)])
        # fog pair: shared dictionaries, 32 px tiles, 4 px per cell; one tile has detail only in fog2, three flat
        fog_cells = sorted(set(int(c) for c in self.rnd.choice(96, 20, replace=False)))
        flat = set(fog_cells[5:8])
        only_b = fog_cells[2]
        fa, fbb = {}, {}
        for cell in sorted(fog_cells, key=order_key):
            a = np.full((32, 32), 1000.0) if cell in flat or cell == only_b else self.rnd.integers(0, 60000, (32, 32))
            b = np.full((32, 32), 2000.0) if cell in flat else self.rnd.integers(0, 60000, (32, 32))
            fa[cell], fbb[cell] = a[4::8, 4::8], b[4::8, 4::8]
            self.add_dict("textures", [("fog", "R16_UNORM", r16_dds(a), 32, 32),
                                       ("fog2", "R16_UNORM", r16_dds(b), 32, 32)])
        truth["fog"] = sorted(fog_cells, key=order_key)
        self.fog_flat, self.fog_only_b = flat, only_b
        self.add_dict("textures", [("fog_gfb", "R16_UNORM", r16_dds(self.mosaic(fa, 4, 1000.0)), 32, 48),
                                   ("fog_gfb2", "R16_UNORM", r16_dds(self.mosaic(fbb, 4, 2000.0)), 32, 48)])
        # shoreline: 64 px BGRA tiles, 8 px per cell; features are whole 8 px blocks; two identical tiles; one blank
        bg = np.array([0, 128, 0, 128])
        shore_cells = sorted(set(int(c) for c in self.rnd.choice(96, 16, replace=False)))
        twin = self.rnd.random((8, 8)) < 0.3
        st, sf = {}, {}
        for i, cell in enumerate(shore_cells):
            mask = twin if i in (3, 4) else (np.zeros((8, 8), bool) if i == 6 else self.rnd.random((8, 8)) < 0.3)
            small = np.where(mask[..., None], np.array([255, 200, 40, 255]), bg)
            st[cell] = np.kron(small, np.ones((8, 8, 1)))
            sf[cell] = small
        for cell in sorted(shore_cells, key=order_key):
            self.add_dict("levels", [("shore", "B8G8R8A8_UNORM", bgra_dds(st[cell]), 64, 64)])
        truth["shore"] = sorted(shore_cells, key=order_key)
        self.shore_blank = shore_cells[6]
        self.add_dict("levels", [("shore_gfb", "B8G8R8A8_UNORM", bgra_dds(self.mosaic(sf, 8, bg, 4)), 64, 96)])
        # documents and terrain
        for doc in (grid_xml("WaterSystemHeightMap", "watersystemheightmap", "watersystemheightmap_gfb"),
                    grid_xml("fogA", "fog", "fog_gfb"), grid_xml("fogB", "fog2", "fog_gfb2"),
                    grid_xml("shorelinegrid1", "shore", "shore_gfb")):
            self.add_stream(doc)
        w = h = 128
        bounds = (-2048.0, -3072.0, -50.0, 2048.0, 3072.0, 500.0)
        rows = []
        for j in range(h):
            y = bounds[1] + (j + 0.5) * (bounds[4] - bounds[1]) / h
            rows.append([max(0, min(255, round((terrain(bounds[0] + (i + 0.5) * 4096 / w, y) + 50) / 550 * 255)))
                         for i in range(w)])
        self.add_stream(mhiu(w, h, bounds, rows))
        self.finish()
        return truth

    def placements(self, data, name):
        return {(p["offset"]): p["cell"] for p in data["placements"][name]}

    def stored(self, archive, name):
        return [it["Offset"] for it in self.items[archive] if it["Textures"][0]["Name"] == name]

    # -- tests --
    def test_every_tile_returns_to_its_cell(self):
        truth = self.build()
        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stderr)
        grids = {g["name"]: g for g in data["grids"]}

        height = self.placements(data, "watersystemheightmap")
        offs = self.stored("levels", "watersystemheightmap")
        self.assertEqual(height, dict(zip(offs, truth["watersystemheightmap"])))   # flat cell 10 placed by order
        hg = grids["WaterSystemHeightMap"]
        self.assertEqual(hg["anchors"], len(offs) - 1)
        self.assertEqual(hg["anchor_order"]["text"][0], hg["anchor_order"]["text"][1])
        self.assertLess(hg["anchor_order"]["numeric"][0], hg["anchor_order"]["numeric"][1])

        fog = self.placements(data, "fog")
        offs = self.stored("textures", "fog")
        want = dict(zip(offs, truth["fog"]))
        fg = grids["fogA"]
        self.assertEqual(fg["anchors"], len(offs) - 3)       # the fog2-only tile is an anchor, flat ones are not
        for off, cell in fog.items():
            self.assertEqual(cell, want[off])
        self.assertEqual(len(fog) + fg["unplaced"], len(offs))
        self.assertEqual(grids["fogB"]["placed"], fg["placed"])

        shore = self.placements(data, "shore")
        offs = self.stored("levels", "shore")
        want = {o: c for o, c in zip(offs, truth["shore"]) if c != self.shore_blank}
        self.assertEqual(shore, want)                        # the twin tiles resolved by order, blank one unplaced
        sg = grids["shorelinegrid1"]
        self.assertEqual(sg["unplaced"], 1)
        self.assertEqual(sg["featureless_tiles"], 1)
        self.assertEqual(sg["unplaced_with_features"], 0)
        self.assertEqual(sg["anchors_moved_by_alignment"], 0)
        self.assertEqual(sg["background"], [0, 128, 0, 128])
        self.assertEqual(sg["anchors"], len(offs) - 3)      # the twins and the blank tile are not clear
        for g in grids.values():
            self.assertEqual((g["duplicate_records"], g["unshared_dictionaries"], g["tiles_in_other_archives"]),
                             (0, 0, 0))
        self.assertEqual(hg["tile_size"], [64, 64])
        self.assertEqual(hg["fallback_format"], "R16_UNORM")
        self.assertEqual(set(data["placements"]), {"watersystemheightmap", "fog", "fog2", "shore"})

    def test_water_fit_recovers_the_scale_and_north_up(self):
        self.build()
        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stderr)
        hg = next(g for g in data["grids"] if g["name"] == "WaterSystemHeightMap")
        fit = hg["water_fit"][0]
        self.assertAlmostEqual(fit["north_up"]["units_per_metre"], 40.96, delta=0.3)
        self.assertAlmostEqual(fit["north_up"]["zero_metre_value"], 32768, delta=60)
        self.assertGreater(fit["north_up"]["r"], 0.95)
        self.assertLess(fit["south_up"]["r"], 0.5)
        self.assertEqual(data["z_check"], [{"z_box": [-1000.0, 1000.0], "heightmaps_inside": 1, "heightmaps": 1}])
        self.assertIn("WaterSystemHeightMap: ", p.stdout)

    def test_numeric_storage_order_is_detected(self):
        """Negative control: stored by numeric cell, the text-order check must report breaks."""
        self.build(order_key=lambda c: c)
        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stderr)
        hg = next(g for g in data["grids"] if g["name"] == "WaterSystemHeightMap")
        self.assertEqual(hg["anchor_order"]["numeric"][0], hg["anchor_order"]["numeric"][1])
        self.assertLess(hg["anchor_order"]["text"][0], hg["anchor_order"]["text"][1])

    def test_refuses_json_over_a_manifest(self):
        self.build()
        manifest = os.path.join(self.work, "assets", "levels", "textures", "manifest.json")
        with open(manifest, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", TOOL, self.work, "--carve", self.carve, "--json", manifest],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("inside the exported assets", p.stderr)
        with open(manifest, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_refuses_json_over_a_tile_texture(self):
        self.build()
        tile = next(pth for pth in self.files if pth.endswith("_watersystemheightmap.dds"))
        with open(tile, "rb") as f:
            before = f.read()
        p = subprocess.run([sys.executable, "-I", TOOL, self.work, "--carve", self.carve, "--json", tile],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        with open(tile, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_counts_duplicates_and_other_archives(self):
        self.build()
        first = next(it for it in self.items["levels"] if it["Textures"][0]["Name"] == "watersystemheightmap")
        self.items["levels"].append(json.loads(json.dumps(first)))          # the same dictionary listed twice
        other = next(pth for pth in self.files if pth.endswith("_watersystemheightmap.dds"))
        with open(other, "rb") as f:
            data = f.read()
        self.add_dict("minigame", [("watersystemheightmap", "R16_UNORM", data, 64, 64)])
        self.finish()
        p, out = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stderr)
        hg = next(g for g in out["grids"] if g["name"] == "WaterSystemHeightMap")
        self.assertEqual((hg["duplicate_records"], hg["tiles_in_other_archives"]), (1, 1))
        self.assertIn("1 duplicate tile records ignored", p.stdout)
        self.assertIn("1 tiles in other archives not placed", p.stdout)

    def test_says_when_there_is_no_heightmap(self):
        self.build()
        def magic(row):
            with open(os.path.join(self.carve, f"{row['off']:012x}.bin"), "rb") as f:
                return f.read(4)
        self.rows = [r for r in self.rows if magic(r) != b"MHIU"]
        self.finish()
        p, data = self.run_tool()
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("no heightmap in the carve, not checked", p.stdout)
        self.assertIn("no MHIU heightmap", p.stdout)

    def test_refuses_json_inside_the_carve(self):
        self.build()
        out = os.path.join(self.carve, "sub", "grid.json")
        p = subprocess.run([sys.executable, "-I", TOOL, self.work, "--carve", self.carve, "--json", out],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("inside the carve folder", p.stderr)
        self.assertFalse(os.path.exists(out))

    def test_reports_bad_inputs_in_one_line(self):
        self.build()
        blocker = os.path.join(self.root, "file")
        with open(blocker, "w") as f:
            f.write("x")
        cases = [
            [os.path.join(self.root, "nowork"), "--carve", self.carve],
            [self.work, "--carve", os.path.join(self.root, "nocarve")],
            [self.work, "--carve", self.carve, "--json", self.root],
            [self.work, "--carve", self.carve, "--json", os.path.join(blocker, "out.json")],
            [self.carve, "--carve", self.carve],
        ]
        for args in cases:
            p = subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2, args)
            self.assertEqual(len(p.stderr.strip().splitlines()), 1, p.stderr)

    def test_reports_a_fallback_that_is_not_a_mosaic(self):
        self.build()
        path = next(pth for pth in self.files if pth.endswith("watersystemheightmap_gfb.dds"))
        with open(path, "wb") as f:
            f.write(r16_dds(np.full((40, 32), 20480.0)))
        for it in self.items["levels"]:
            for t in it["Textures"]:
                if t["Name"] == "watersystemheightmap_gfb":
                    t["Height"] = 40
        self.finish()
        p, _ = self.run_tool()
        self.assertEqual(p.returncode, 2)
        self.assertIn("mosaic", p.stderr)


if __name__ == "__main__":
    unittest.main()
