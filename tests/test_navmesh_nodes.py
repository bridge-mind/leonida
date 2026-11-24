"""Synthetic tests for tools/navmesh_nodes.py. No game files are read or needed.

    python -I tests/test_navmesh_nodes.py

The navmesh tile comes from the builder in tests/test_navmesh_decode.py with its vertices and polygon flags
filled in here, and the lane graph from the node-list builder in tests/test_world_lights_paths.py.
"""
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import navmesh_decode as nd  # noqa: E402
import navmesh_nodes as nn  # noqa: E402
from test_navmesh_decode import build_0drp, build_navmesh, make_rpf  # noqa: E402
from test_world_lights_paths import GRAPH_NODES, node_resource, write_rpf  # noqa: E402

TOOL = os.path.join(ROOT, "tools", "navmesh_nodes.py")
ROAD, OTHER = 0x2100, 0x1000            # +0x18 words of the two synthetic polygons (bits 8 + 13, and bit 12)
MARK = 1 << 17                          # +0x28 word of the first polygon


def build_tile(squares, sector=(256 << 9) | 256, flags=1):
    """One navmesh tile whose polygons are axis-aligned squares.

    squares: [(x0, y0, x1, y1, z, flags18, flags28)] in metres from the tile's minimum corner (height 0..16).
    """
    tile = bytearray(build_navmesh(flags=flags, sector=sector, n_verts=4 * len(squares),
                                   poly_sizes=(4,) * len(squares), n_links=0, n_link_idx=0, n_volumes=0)[0])
    (vertex_at, _count), = nd.split_array(tile, 0x70, 6, 4 * len(squares))
    (polygon_at, _count), = nd.split_array(tile, 0x98, 44, len(squares))

    def q(value, extent):
        return round(value / extent * 65535)

    for i, (x0, y0, x1, y1, z, flags18, flags28) in enumerate(squares):
        for k, (x, y) in enumerate(((x0, y0), (x1, y0), (x1, y1), (x0, y1))):
            struct.pack_into("<3H", tile, vertex_at + 6 * (4 * i + k), q(x, 128.0), q(y, 128.0), q(z, 16.0))
        struct.pack_into("<I", tile, polygon_at + 44 * i + 0x18, flags18)
        struct.pack_into("<I", tile, polygon_at + 44 * i + 0x28, flags28)
    return bytes(tile)


class GeometryTest(unittest.TestCase):
    def test_sector_of_matches_the_tile_grid(self):
        self.assertEqual(nn.sector_of(0.0, 0.0), (256, 256))
        self.assertEqual(nn.sector_of(127.99, 128.0), (256, 257))
        self.assertEqual(nn.sector_of(-0.01, -128.01), (255, 254))

    def test_vertices_are_minimum_plus_u16_over_65535_times_extent(self):
        sector = (260 << 9) | 250                                   # tile minimum (-768, 512)
        geometry = nn.tile_geometry(build_tile([(16.0, 32.0, 48.0, 64.0, 4.0, ROAD, MARK)], sector=sector))
        self.assertEqual(geometry["sector"], (250, 260))
        (points, flags18, flags28), = geometry["polygons"]
        self.assertEqual((flags18, flags28), (ROAD, MARK))
        for got, want in zip(points, ((-752.0, 544.0), (-720.0, 544.0), (-720.0, 576.0), (-752.0, 576.0))):
            self.assertAlmostEqual(got[0], want[0], delta=0.002)
            self.assertAlmostEqual(got[1], want[1], delta=0.002)
            self.assertAlmostEqual(got[2], 4.0, delta=0.001)
        self.assertAlmostEqual(nn.plan_area(points), 32.0 * 32.0, delta=0.2)
        # The far corner of a tile is exactly minimum + extent: u16 65535, not 65536.
        full = nn.tile_geometry(build_tile([(0.0, 0.0, 128.0, 128.0, 16.0, 0, 0)], sector=sector))
        self.assertEqual(full["polygons"][0][0][2], (-640.0, 640.0, 16.0))

    def test_dynamic_and_unreadable_tiles_give_no_geometry(self):
        self.assertIsNone(nn.tile_geometry(build_tile([(0.0, 0.0, 8.0, 8.0, 0.0, 0, 0)], flags=5)))
        tile = bytearray(build_tile([(0.0, 0.0, 8.0, 8.0, 0.0, 0, 0)]))
        self.assertIsNone(nn.tile_geometry(tile[:0x80]))
        struct.pack_into("<Q", tile, 0x70, 0)                       # no vertex array
        self.assertIsNone(nn.tile_geometry(bytes(tile)))
        self.assertIsNone(nn.tile_geometry(b""))

    def test_contains_and_area(self):
        square = [(0.0, 0.0, 0.0), (4.0, 0.0, 0.0), (4.0, 4.0, 0.0), (0.0, 4.0, 0.0)]
        self.assertTrue(nn.contains(square, 2.0, 2.0))
        self.assertFalse(nn.contains(square, 5.0, 2.0))
        self.assertFalse(nn.contains(square, 2.0, -0.5))
        self.assertEqual(nn.plan_area(square), 16.0)
        self.assertEqual(nn.plan_area(list(reversed(square))), 16.0)


class TallyTest(unittest.TestCase):
    # Two levels over the same ground (a bridge over a road) plus a separate patch.
    SQUARES = [
        (0.0, 0.0, 32.0, 32.0, 2.0, ROAD, MARK),        # lower level
        (0.0, 0.0, 32.0, 32.0, 12.0, OTHER, 0),         # upper level, same plan area
        (64.0, 64.0, 72.0, 72.0, 0.0, OTHER, 0),
    ]
    NODES = [
        (16.0, 16.0, 2.5),      # on the lower level
        (16.0, 16.0, 11.0),     # on the upper level: the closest height wins, not the first polygon
        (68.0, 68.0, 6.0),      # over the patch but 6 m above it
        (100.0, 100.0, 0.0),    # over no polygon
        (500.0, 500.0, 0.0),    # in a tile that is not in the archive
    ]

    def summary(self, max_dz=3.0):
        by_sector = {}
        for n, (x, y, _z) in enumerate(self.NODES):
            by_sector.setdefault(nn.sector_of(x, y), []).append(n)
        tally = nn.new_tally()
        nn.tally_tile(tally, build_tile(self.SQUARES), by_sector, self.NODES)
        nn.tally_tile(tally, build_tile(self.SQUARES, sector=(300 << 9) | 300), by_sector, self.NODES)  # no node
        nn.tally_tile(tally, build_tile(self.SQUARES, flags=5), by_sector, self.NODES)                  # dynamic
        return nn.summarize(tally, len(self.NODES), len(by_sector), max_dz)

    def test_counts(self):
        s = self.summary()
        self.assertEqual(s["nodes"], 5)
        self.assertEqual(s["tiles_named_by_nodes"], 2)
        self.assertEqual(s["tiles_with_nodes"], 1)
        self.assertEqual(s["tiles_unreadable"], 0)
        self.assertEqual(s["nodes_over_a_polygon"], 3)
        self.assertEqual(s["nodes_within_max_dz"], 2)
        self.assertEqual(s["polygons"], 3)
        self.assertEqual(s["area_m2"], 2 * 1024 + 64)
        self.assertEqual(s["flags18"]["13"], {"nodes": 1, "polygons": 1, "area_m2": 1024})
        self.assertEqual(s["flags18"]["8"], {"nodes": 1, "polygons": 1, "area_m2": 1024})
        self.assertEqual(s["flags18"]["12"], {"nodes": 1, "polygons": 2, "area_m2": 1088})
        self.assertEqual(s["flags28"], {"17": {"nodes": 1, "polygons": 1, "area_m2": 1024}})
        self.assertEqual(s["flags18_bits_8_and_13"], {
            "both": {"nodes": 1, "polygons": 1, "area_m2": 1024},
            "only_8": {"nodes": 0, "polygons": 0, "area_m2": 0},
            "only_13": {"nodes": 0, "polygons": 0, "area_m2": 0},
            "neither": {"nodes": 1, "polygons": 2, "area_m2": 1088},
        })
        self.assertAlmostEqual(s["height_difference_m"]["p10"], 0.5, delta=0.01)

    def test_the_two_road_bits_are_counted_in_combination(self):
        self.assertEqual([nn.pair_key(w) for w in (0x2100, 0x100, 0x2000, 0x1000, 0, 0xFFFFFFFF)],
                         ["both", "only_8", "only_13", "neither", "neither", "both"])
        squares = [(0.0, 0.0, 8.0, 8.0, 0.0, 0x100, 0), (8.0, 0.0, 16.0, 8.0, 0.0, 0x2000, 0),
                   (16.0, 0.0, 24.0, 8.0, 0.0, 0x2100, 0), (24.0, 0.0, 32.0, 8.0, 0.0, 0, 0)]
        nodes = [(4.0, 4.0, 0.0), (12.0, 4.0, 0.0), (13.0, 4.0, 0.0), (20.0, 4.0, 0.0), (28.0, 4.0, 0.0)]
        tally = nn.new_tally()
        nn.tally_tile(tally, build_tile(squares), {(256, 256): list(range(len(nodes)))}, nodes)
        pair = nn.summarize(tally, len(nodes), 1, 3.0)["flags18_bits_8_and_13"]
        self.assertEqual({k: v["nodes"] for k, v in pair.items()},
                         {"both": 1, "only_8": 1, "only_13": 2, "neither": 1})
        self.assertEqual({k: v["polygons"] for k, v in pair.items()},
                         {"both": 1, "only_8": 1, "only_13": 1, "neither": 1})
        self.assertEqual(sum(v["area_m2"] for v in pair.values()), 4 * 64)

    def test_max_dz_bounds_which_nodes_count(self):
        self.assertEqual(self.summary(max_dz=0.0)["nodes_within_max_dz"], 0)
        wide = self.summary(max_dz=10.0)
        self.assertEqual(wide["nodes_within_max_dz"], 3)
        self.assertEqual(wide["flags18"]["12"]["nodes"], 2)

    def test_unreadable_tile_is_counted(self):
        tile = bytearray(build_tile(self.SQUARES))
        struct.pack_into("<Q", tile, 0x98, 0)                       # no polygon array
        tally = nn.new_tally()
        nn.tally_tile(tally, bytes(tile), {(256, 256): [0]}, self.NODES)
        self.assertEqual(tally["tiles_unreadable"], 1)
        self.assertEqual(tally["best"], {})

    def test_no_nodes_gives_an_empty_summary(self):
        s = nn.summarize(nn.new_tally(), 0, 0, 3.0)
        self.assertEqual(s["nodes_over_a_polygon"], 0)
        self.assertEqual(s["height_difference_m"], {"p10": None, "p50": None, "p90": None, "p99": None})
        self.assertEqual(s["flags18"], {})


class ArchiveTest(unittest.TestCase):
    """The synthetic lane graph puts node i at (i, 0, 0): i = 0..3."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.levels = os.path.join(self.tmp.name, "levels.rpf")
        write_rpf(self.levels, [node_resource(GRAPH_NODES)])
        self.navmeshes = os.path.join(self.tmp.name, "navmeshes.rpf")
        near = build_tile([(0.0, 0.0, 2.5, 8.0, 0.0, ROAD, MARK), (2.5, 0.0, 6.0, 8.0, 8.0, OTHER, 0)])
        far = build_tile([(0.0, 0.0, 128.0, 128.0, 0.0, OTHER, 0)], sector=(300 << 9) | 300)
        make_rpf(self.navmeshes, [far, near])

    def tearDown(self):
        self.tmp.cleanup()

    def test_node_positions_drop_non_finite_ones(self):
        nodes, read = nn.node_positions(self.levels)
        self.assertEqual((len(nodes), read), (4, 4))
        self.assertEqual(nodes[3], (3.0, 0.0, 0.0))
        self.assertTrue(all(math.isfinite(v) for p in nodes for v in p))

    def test_cross_check(self):
        nodes, _read = nn.node_positions(self.levels)
        s = nn.cross_check(self.navmeshes, nodes)
        self.assertEqual(s["tiles_with_nodes"], 1)
        self.assertEqual(s["nodes_over_a_polygon"], 4)
        self.assertEqual(s["nodes_within_max_dz"], 3)               # node 3 is 8 m below its polygon
        self.assertEqual(s["flags18"]["13"]["nodes"], 3)
        self.assertEqual(s["flags18"]["12"]["nodes"], 0)
        self.assertEqual(s["flags28"]["17"]["nodes"], 3)

    def test_missing_continuation_does_not_swallow_the_next_tile(self):
        """A resource whose second stream is missing is not completed with the tile after it: the tile is read
        on its own, and a navmesh tile cut off the same way is dropped while the tile after it is kept."""
        nodes, _read = nn.node_positions(self.levels)
        near = build_tile([(0.0, 0.0, 2.5, 8.0, 0.0, ROAD, MARK), (2.5, 0.0, 6.0, 8.0, 8.0, OTHER, 0)])
        cover, _rest = build_0drp(total_size=0x1000, split_at=0x1000, node_count=2, override_n4e=0)
        split_tile = bytearray(build_tile([(0.0, 0.0, 128.0, 128.0, 0.0, OTHER, 0)], sector=(300 << 9) | 300))
        struct.pack_into("<Q", split_tile, nd.offset_of(nd._u64(split_tile, 8)) + 8, 2)   # declares two streams
        for name, streams in (("0DRP", [cover, near]), ("navmesh", [bytes(split_tile), near])):
            with self.subTest(cut_off=name):
                make_rpf(self.navmeshes, streams)
                tiles = list(nn.navmesh_resources(self.navmeshes))
                self.assertEqual(len(tiles), 1)
                self.assertTrue(tiles[0] == near, "the tile read is not the one after the cut-off resource")
                s = nn.cross_check(self.navmeshes, nodes)
                self.assertEqual(s["tiles_with_nodes"], 1)
                self.assertEqual(s["nodes_over_a_polygon"], 4)
                self.assertEqual(s["flags18"]["13"]["nodes"], 3)

    def test_cli_writes_counts(self):
        out = os.path.join(self.tmp.name, "reports", "nodes.json")
        cp = subprocess.run([sys.executable, "-I", TOOL, self.navmeshes, self.levels, "--out", out],
                            capture_output=True, text=True)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        with open(out, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r", raw)
        s = json.loads(raw)
        self.assertEqual(s["nodes_read"], 4)
        self.assertEqual(s["nodes_within_max_dz"], 3)
        cp = subprocess.run([sys.executable, "-I", TOOL, self.navmeshes, self.levels, "--max-dz", "10"],
                            capture_output=True, text=True)
        self.assertEqual(json.loads(cp.stdout)["nodes_within_max_dz"], 4)

    def test_cli_refuses_bad_arguments(self):
        missing = os.path.join(self.tmp.name, "nope.rpf")
        cases = (
            [missing, self.levels],
            [self.navmeshes, missing],
            [self.navmeshes, self.levels, "--jobs", "0"],
            [self.navmeshes, self.levels, "--max-dz", "-1"],
            [self.navmeshes, self.levels, "--max-dz", "nan"],
            [self.navmeshes, self.levels, "--out", self.levels],
            [self.navmeshes, self.levels, "--out", self.navmeshes],
            [self.navmeshes, self.levels, "--out", os.path.join(self.tmp.name, ".", "levels.rpf")],
            [self.navmeshes, self.levels, "--out", self.tmp.name],
        )
        before = (os.path.getsize(self.navmeshes), os.path.getsize(self.levels))
        for args in cases:
            with self.subTest(args=args[2:] or args):
                cp = subprocess.run([sys.executable, "-I", TOOL, *args], capture_output=True, text=True)
                self.assertEqual(cp.returncode, 2, cp.stderr)
                self.assertNotIn("Traceback", cp.stderr)
        self.assertEqual((os.path.getsize(self.navmeshes), os.path.getsize(self.levels)), before)

    def test_cli_refuses_a_data_start_outside_the_archive(self):
        # An 8-byte archive whose data area would start at byte 64 was scanned as an empty archive (exit 0).
        for start in (64, 4):
            with self.subTest(start=start):
                with open(self.navmeshes, "wb") as f:
                    f.write(b"9FPR" + struct.pack("<I", start))
                cp = subprocess.run([sys.executable, "-I", TOOL, self.navmeshes, self.levels],
                                    capture_output=True, text=True)
                self.assertEqual(cp.returncode, 2, cp.stdout)
                self.assertIn("data area starts at %d" % start, cp.stderr)
                self.assertNotIn("Traceback", cp.stderr)
                self.assertEqual(cp.stdout, "")

    def test_cli_refuses_an_unwritable_out_before_the_archives_are_read(self):
        # Both archives are read before the summary is written, so a --out that cannot be written used to cost
        # the whole run and then end in a traceback.
        out = os.path.join(self.tmp.name, "read-only.json")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("{}\n")
        os.chmod(out, 0o444)
        try:
            junk = os.path.join(self.tmp.name, "junk.rpf")
            with open(junk, "wb") as f:
                f.write(b"not an archive")
            cp = subprocess.run([sys.executable, "-I", TOOL, junk, self.levels, "--out", out],
                                capture_output=True, text=True)
            self.assertEqual(cp.returncode, 2, cp.stderr)
            self.assertIn("read-only", cp.stderr)
            self.assertNotIn("Traceback", cp.stderr)
            self.assertNotIn("not an RPF9 archive", cp.stderr)   # refused before either archive is read
            with open(out, "rb") as f:
                self.assertEqual(f.read(), b"{}\n")
        finally:
            os.chmod(out, 0o666)

    def test_cli_reports_a_file_that_is_not_an_archive(self):
        junk = os.path.join(self.tmp.name, "junk.rpf")
        with open(junk, "wb") as f:
            f.write(b"not an archive")
        cp = subprocess.run([sys.executable, "-I", TOOL, junk, self.levels], capture_output=True, text=True)
        self.assertEqual(cp.returncode, 2)
        self.assertNotIn("Traceback", cp.stderr)


if __name__ == "__main__":
    unittest.main()
