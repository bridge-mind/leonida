"""Tests for tools/world_archetypes.py, built from synthetic 0x7a resources (no game data).

    python -I tests/test_world_archetypes.py
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import zlib
from contextlib import redirect_stdout

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_tool():
    path = os.path.join(ROOT, "tools", "world_archetypes.py")
    spec = importlib.util.spec_from_file_location("world_archetypes_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


WA = load_tool()
WI = WA.WI
J = WA.joaat_cs
SYS = 0x50000000
BASE, MLO = J("CBaseArchetypeDef"), J("CMloArchetypeDef")
S = math.sqrt(0.5)


def build(structs, blocks):
    """A valid 0x7a resource. structs: [(hash, size, [(member hash, offset, type)])]; blocks: [(hash, bytes)]."""
    st = 0x60
    mt = st + 32 * len(structs)
    member_at, p = [], mt
    for _, _, members in structs:
        member_at.append(p)
        p += 16 * len(members)
    bt = (p + 15) & ~15
    p = bt + 16 * len(blocks)
    data_at = []
    for _, payload in blocks:
        p = (p + 15) & ~15
        data_at.append(p)
        p += len(payload)
    b = bytearray(p + 16)
    b[0x10:0x15] = b"0DRPz"
    struct.pack_into("<Q", b, 0x08, SYS)
    struct.pack_into("<I", b, 0x20, 1)
    struct.pack_into("<Q", b, 0x28, SYS | st)
    struct.pack_into("<Q", b, 0x38, SYS | bt)
    struct.pack_into("<HHH", b, 0x48, len(structs), 0, len(blocks))
    for i, (h, size, members) in enumerate(structs):
        o = st + 32 * i
        struct.pack_into("<I", b, o, h)
        struct.pack_into("<Q", b, o + 0x10, SYS | member_at[i])
        struct.pack_into("<I", b, o + 0x18, size)
        struct.pack_into("<H", b, o + 0x1E, len(members))
        for k, (mh, off, ty) in enumerate(members):
            struct.pack_into("<IIBBHI", b, member_at[i] + 16 * k, mh, off, ty, 0, 0, 0)
    for i, (h, payload) in enumerate(blocks):
        struct.pack_into("<IIQ", b, bt + 16 * i, h, len(payload), SYS | data_at[i])
        b[data_at[i]:data_at[i] + len(payload)] = payload
    return bytes(b)


# Two archetype layouts with the same members at different offsets: fields must be found by hash.
LAYOUT_A = {"lodDist": 0x38, "bbMin": 0x40, "bbMax": 0x4C, "bsCentre": 0x58, "bsRadius": 0x64, "flags": 0x68,
            "name": 0x78, "assetType": 0x84, "assetName": 0x88}
LAYOUT_B = {"assetName": 0x00, "assetType": 0x04, "name": 0x08, "flags": 0x0C, "lodDist": 0x10, "bsRadius": 0x14,
            "bbMin": 0x20, "bbMax": 0x30, "bsCentre": 0x40}
TYPES = {"lodDist": 0x21, "bsRadius": 0x21, "bbMin": 0x32, "bbMax": 0x32, "bsCentre": 0x32, "flags": 0x15,
         "name": 0x4A, "assetName": 0x4A, "assetType": 0x62}
SIZE = 0x90


def archetype(layout, name, asset_type, lo, hi, lod=150.0, flags=7):
    b = bytearray(SIZE)
    o = layout
    struct.pack_into("<f", b, o["lodDist"], lod)
    struct.pack_into("<3f", b, o["bbMin"], *lo)
    struct.pack_into("<3f", b, o["bbMax"], *hi)
    struct.pack_into("<3f", b, o["bsCentre"], *[(lo[i] + hi[i]) / 2 for i in range(3)])
    struct.pack_into("<f", b, o["bsRadius"], 2.0)
    struct.pack_into("<I", b, o["flags"], flags)
    struct.pack_into("<I", b, o["name"], name)
    struct.pack_into("<I", b, o["assetType"], asset_type)
    struct.pack_into("<I", b, o["assetName"], name)
    return bytes(b)


def maptypes(defs, layout=LAYOUT_A, kind=BASE):
    members = [(J(n), off, TYPES[n]) for n, off in layout.items()]
    payload = b"".join(archetype(layout, *d) for d in defs)
    return build([(WA.MAPTYPES, 0x50, []), (kind, SIZE, members)],
                 [(WA.MAPTYPES, bytes(0x50)), (kind, payload)])


ENT = 0x90
ENT_MEMBERS = [(WI.FIELD["archetypeName"], 0x44, 0x4A), (WI.FIELD["position"], 0x38, 0x32),
               (WI.FIELD["rotation"], 0x10, 0x34), (WI.FIELD["scaleXY"], 0x54, 0x21),
               (WI.FIELD["scaleZ"], 0x58, 0x21), (WI.FIELD["lodLevel"], 0x7E, 0x60)]
MAP_MEMBERS = [(WI.FIELD["name"], 0x08, 0x4A), (WI.FIELD["entitiesExtentsMin"], 0x20, 0x33),
               (WI.FIELD["entitiesExtentsMax"], 0x30, 0x33)]


def entity(arch, pos, rot=(0.0, 0.0, 0.0, 1.0), sxy=1.0, sz=1.0):
    b = bytearray(ENT)
    struct.pack_into("<4f", b, 0x10, *rot)
    struct.pack_into("<3f", b, 0x38, *pos)
    struct.pack_into("<I", b, 0x44, arch)
    struct.pack_into("<2f", b, 0x54, sxy, sz)
    return bytes(b)


def mapfile(entities, ex_lo, ex_hi):
    root = bytearray(0x80)
    struct.pack_into("<I", root, 0x08, 0x1234)
    struct.pack_into("<3f", root, 0x20, *ex_lo)
    struct.pack_into("<3f", root, 0x30, *ex_hi)
    return build([(WI.ENTITY, ENT, ENT_MEMBERS), (WI.MAPDATA, 0x80, MAP_MEMBERS)],
                 [(WI.ENTITY, b"".join(entities)), (WI.MAPDATA, bytes(root))])


ROCK, CRATE = 0x11111111, 0x22222222
ROCK_BOX = ((0.0, -2.0, 0.0), (3.0, 1.0, 3.0))  # asymmetric, so +90 and -90 about Z give different boxes
CRATE_BOX = ((-0.5, -0.5, 0.0), (0.5, 0.5, 1.0))


def scene():
    """One rotated rock (+90 degrees about Z stored) and one crate scaled x2; extents as the measured rule makes
    them: conjugated rotation, scale ignored."""
    defs = maptypes([(ROCK, 2, *ROCK_BOX), (CRATE, 1, *CRATE_BOX)])
    rock = {"position": [100.0, 50.0, 10.0], "rotation": [0.0, 0.0, S, S]}
    crate = {"position": [0.0, 0.0, 0.0], "scale_xy": 2.0, "scale_z": 2.0}
    lo1, hi1 = WA.world_box(rock, *ROCK_BOX, conjugate=True, use_scale=False)
    lo2, hi2 = WA.world_box(crate, *CRATE_BOX, conjugate=True, use_scale=False)
    ex_lo = [min(lo1[i], lo2[i]) for i in range(3)]
    ex_hi = [max(hi1[i], hi2[i]) for i in range(3)]
    m = mapfile([entity(ROCK, rock["position"], rock["rotation"]), entity(CRATE, crate["position"], sxy=2.0, sz=2.0)],
                ex_lo, ex_hi)
    return defs, m


def rpf(parts, start=0x40):
    out = bytearray(start)
    out[:4] = b"9FPR"
    struct.pack_into("<I", out, 4, start)
    for part in parts:
        out.extend(b"\0" * (((len(out) + 15) & ~15) - len(out)))
        out.extend(zlib.compress(part))
    return bytes(out)


class HashTest(unittest.TestCase):
    def test_case_sensitive_joaat_matches_world_index(self):
        self.assertEqual(J("CEntityDef"), WI.ENTITY)
        self.assertEqual(J("CMapData"), WI.MAPDATA)
        for name in ("archetypeName", "position", "rotation", "lodDist", "flags", "name"):
            self.assertEqual(J(name), WI.FIELD[name], name)


class ParseTest(unittest.TestCase):
    def test_fields_found_by_hash_not_offset(self):
        rows = [WA.parse_archetypes(maptypes([(ROCK, 2, *ROCK_BOX)], layout))[0] for layout in (LAYOUT_A, LAYOUT_B)]
        self.assertEqual(rows[0], rows[1])
        r = rows[0]
        self.assertEqual((r["name"], r["asset_name"], r["asset_type"], r["kind"]), (ROCK, ROCK, 2, "base"))
        self.assertEqual((r["bb_min"], r["bb_max"]), ([0, -2, 0], [3, 1, 3]))
        self.assertEqual((r["lod_dist"], r["bs_radius"], r["flags"]), (150.0, 2.0, 7))
        self.assertNotIn("time_flags", r)  # absent members are omitted, not invented

    def test_kinds_and_counts(self):
        rows = WA.parse_archetypes(maptypes([(ROCK, 0, *ROCK_BOX)], kind=MLO))
        self.assertEqual([r["kind"] for r in rows], ["mlo"])
        self.assertEqual(len(WA.parse_archetypes(maptypes([(ROCK, 2, *ROCK_BOX), (CRATE, 1, *CRATE_BOX)]))), 2)

    def test_map_file_and_invalid_input_give_nothing(self):
        _, m = scene()
        self.assertEqual(WA.parse_archetypes(m), [])
        self.assertEqual(WA.parse_archetypes(b"\0" * 0x100), [])

    def test_block_not_a_multiple_of_the_struct_size_is_skipped(self):
        members = [(J(n), off, TYPES[n]) for n, off in LAYOUT_A.items()]
        b = build([(WA.MAPTYPES, 0x50, []), (BASE, SIZE, members)],
                  [(WA.MAPTYPES, bytes(0x50)), (BASE, archetype(LAYOUT_A, ROCK, 2, *ROCK_BOX) + b"\0" * 8)])
        self.assertEqual(WA.parse_archetypes(b), [])

    def test_member_table_sorted_by_offset(self):
        table = WA.member_table(maptypes([(ROCK, 2, *ROCK_BOX)], LAYOUT_B), BASE)
        self.assertEqual([off for _, off, _ in table], sorted(LAYOUT_B.values()))


class TransformTest(unittest.TestCase):
    def test_conjugate_and_stored_rotate_opposite_ways(self):
        e = {"position": [0, 0, 0], "rotation": [0.0, 0.0, S, S]}
        box = ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))  # a point at +X
        lo, _ = WA.world_box(e, *box, conjugate=False, use_scale=False)
        self.assertAlmostEqual(lo[1], 1.0, places=6)   # stored: +X -> +Y
        lo, _ = WA.world_box(e, *box, conjugate=True, use_scale=False)
        self.assertAlmostEqual(lo[1], -1.0, places=6)  # conjugate: +X -> -Y

    def test_scale_xy_and_z(self):
        e = {"position": [0, 0, 0], "scale_xy": 2.0, "scale_z": 3.0}
        lo, hi = WA.world_box(e, (-1.0, -1.0, 0.0), (1.0, 1.0, 1.0), conjugate=True, use_scale=True)
        self.assertEqual((lo, hi), ([-2.0, -2.0, 0.0], [2.0, 2.0, 3.0]))
        lo, hi = WA.world_box(e, (-1.0, -1.0, 0.0), (1.0, 1.0, 1.0), conjugate=True, use_scale=False)
        self.assertEqual((lo, hi), ([-1.0, -1.0, 0.0], [1.0, 1.0, 1.0]))

    def test_zero_scales_collapse_only_their_axes(self):
        box = ((-1.0, -2.0, -3.0), (2.0, 4.0, 4.0))
        for zero in (0.0, -0.0):
            cases = [
                (zero, 3.0, ([10.0, 20.0, 21.0], [10.0, 20.0, 42.0])),
                (2.0, zero, ([8.0, 16.0, 30.0], [14.0, 28.0, 30.0])),
                (zero, zero, ([10.0, 20.0, 30.0], [10.0, 20.0, 30.0])),
            ]
            for sxy, sz, expected in cases:
                with self.subTest(scale_xy=sxy, scale_z=sz):
                    e = {"position": [10.0, 20.0, 30.0], "scale_xy": sxy, "scale_z": sz}
                    self.assertEqual(WA.world_box(e, *box, conjugate=True, use_scale=True), expected)
                    self.assertEqual(WA.world_box(e, *box, conjugate=True, use_scale=False),
                                     ([9.0, 18.0, 27.0], [12.0, 24.0, 34.0]))

    def test_missing_and_none_scales_still_default_to_one(self):
        for scales in ({}, {"scale_xy": None}, {"scale_z": None}, {"scale_xy": None, "scale_z": None}):
            with self.subTest(scales=scales):
                e = {"position": [10.0, 20.0, 30.0], **scales}
                self.assertEqual(WA.world_box(e, (-1.0, -2.0, -3.0), (2.0, 4.0, 4.0),
                                              conjugate=True, use_scale=True),
                                 ([9.0, 18.0, 27.0], [12.0, 24.0, 34.0]))


class ExtentsTest(unittest.TestCase):
    def test_conjugate_matches_and_scale_is_ignored(self):
        defs_b, m = scene()
        defs = {d["name"]: d for d in WA.parse_archetypes(defs_b)}
        ents = []
        row = WI.parse_resource(m, collect=ents)
        res = WA.check_extents([(row, ents)], defs)
        self.assertEqual(res["maps_fully_defined"], 1)
        self.assertEqual((res["rotation"]["conjugate_matches"], res["rotation"]["stored_matches"]), (1, 0))
        self.assertLess(res["rotation"]["conjugate_median_error_m"], 1e-4)
        self.assertEqual((res["scale"]["unscaled_matches"], res["scale"]["scaled_matches"]), (1, 0))
        self.assertEqual(res["scale"]["scaled_exceeds_extents"], 1)

    def test_map_with_an_undefined_archetype_is_not_judged(self):
        defs_b, m = scene()
        defs = {d["name"]: d for d in WA.parse_archetypes(defs_b) if d["name"] == ROCK}
        ents = []
        row = WI.parse_resource(m, collect=ents)
        self.assertEqual(WA.check_extents([(row, ents)], defs)["maps_fully_defined"], 0)

    def test_zero_scale_maps_are_included_in_scale_comparisons(self):
        defs = {d["name"]: d for d in WA.parse_archetypes(maptypes([(CRATE, 1, *CRATE_BOX)]))}
        for zero in (0.0, -0.0):
            for sxy, sz in ((zero, 1.0), (1.0, zero), (zero, zero)):
                with self.subTest(scale_xy=sxy, scale_z=sz):
                    m = mapfile([entity(CRATE, (0.0, 0.0, 0.0), sxy=sxy, sz=sz)], *CRATE_BOX)
                    ents = []
                    row = WI.parse_resource(m, collect=ents)
                    res = WA.check_extents([(row, ents)], defs)
                    self.assertEqual(res["maps_fully_defined"], 1)
                    self.assertEqual(res["scale"], {"maps": 1, "scaled_matches": 0, "unscaled_matches": 1,
                                                   "scaled_exceeds_extents": 0, "unscaled_exceeds_extents": 0})

    def test_missing_and_none_scales_are_not_counted_as_scaled(self):
        defs = {CRATE: {"bb_min": CRATE_BOX[0], "bb_max": CRATE_BOX[1]}}
        row = {"entities_min": CRATE_BOX[0], "entities_max": CRATE_BOX[1]}
        for scales in ({}, {"scale_xy": None}, {"scale_z": None}, {"scale_xy": None, "scale_z": None}):
            with self.subTest(scales=scales):
                res = WA.check_extents([(row, [{"archetype": CRATE, **scales}])], defs)
                self.assertEqual(res["maps_fully_defined"], 1)
                self.assertEqual(res["scale"]["maps"], 0)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_cli(self, path):
        out = os.path.join(self.tmp.name, "out", "archetypes.json")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(WA.main([path, "--out", out, "--check-extents"]), 0)
        with open(out, "rb") as f:
            raw = f.read()
        self.assertNotIn(b"\r\n", raw)
        return json.loads(raw)

    def check(self, doc):
        s = doc["summary"]
        self.assertEqual((s["maptypes"], s["archetypes"], s["names"]), (1, 2, 2))
        self.assertEqual((s["kinds"], s["asset_types"]), ({"base": 2}, {"1": 1, "2": 1}))
        self.assertEqual(doc["schemas"]["base"]["size"], SIZE)
        self.assertEqual(doc["extents"]["rotation"]["conjugate_matches"], 1)

    def test_folder_of_carved_files(self):
        defs, m = scene()
        folder = os.path.join(self.tmp.name, "carved")
        os.makedirs(folder)
        for name, data in (("a.bin", defs), ("b.bin", m)):
            with open(os.path.join(folder, name), "wb") as f:
                f.write(data)
        self.check(self.run_cli(folder))

    def test_rpf9_window(self):
        defs, m = scene()
        path = os.path.join(self.tmp.name, "levels.rpf")
        with open(path, "wb") as f:
            f.write(rpf([defs, m]))
        self.check(self.run_cli(path))

    def cli_error(self, argv):
        err = io.StringIO()
        with redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            WA.main(argv)
        self.assertEqual(cm.exception.code, 2)
        self.assertNotIn("Traceback", err.getvalue())
        return err.getvalue()

    def test_out_equal_to_the_input_is_refused_and_leaves_it_alone(self):
        path = os.path.join(self.tmp.name, "carved.bin")
        with open(path, "wb") as f:
            f.write(b"\x00" * 3000)
        self.assertIn("overwrite the input", self.cli_error([path, "--out", path]))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), b"\x00" * 3000)

    def test_out_under_a_regular_file_or_naming_a_folder_is_one_error(self):
        path = os.path.join(self.tmp.name, "carved.bin")
        with open(path, "wb") as f:
            f.write(b"\x00" * 64)
        self.assertIn("--out", self.cli_error([path, "--out", os.path.join(path, "sub", "r.json")]))
        self.assertIn("folder", self.cli_error([path, "--out", self.tmp.name]))

    def test_missing_path_is_an_error(self):
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            sys.stderr, saved = io.StringIO(), sys.stderr
            try:
                WA.main([os.path.join(self.tmp.name, "nope")])
            finally:
                sys.stderr = saved


if __name__ == "__main__":
    unittest.main()
