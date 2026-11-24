"""Tests for placing models at their map placements (no game data).

Covers world_index.py --entities-out and world_build.py --placements: the entity row format, the transform
(the conjugate of the stored quaternion, non-uniform scale, normals, mirroring, RAGE Z-up to glTF Y-up), the
name/key match against rpf9dump manifests, and the placement counts in world.json. Every model, entity and
resource is synthetic.

    python -I tests/test_world_place.py
"""
import contextlib
import io
import json
import math
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(ROOT, "tools"), TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import test_world_index as twi  # noqa: E402  (synthetic 0x7a resource builder)
import world_build  # noqa: E402
import world_names  # noqa: E402

WI = twi.WI
S2 = math.sqrt(0.5)


def stored_rotation_about_z(degrees):
    """The quaternion a map file stores for a world rotation of `degrees` about +Z (stored = inverse)."""
    h = math.radians(degrees) / 2
    return [0.0, 0.0, -math.sin(h), math.cos(h)]


def apply(m, v):
    return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))


def close(a, b, tol=1e-9):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


UNIT_BOX_OBJ = """# unit box 0..1 in RAGE axes (Z up)
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 0 0 1
v 1 0 1
v 1 1 1
v 0 1 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


def write_assets(root, name="test_box"):
    models = os.path.join(root, "assets", "levels", "models")
    os.makedirs(models)
    with open(os.path.join(models, "box.obj"), "w", encoding="utf-8") as f:
        f.write(UNIT_BOX_OBJ)
    with open(os.path.join(models, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"Items": [{"File": "box.obj", "Name": name + ".#dr", "Min": [0, 0, 0], "Max": [1, 1, 1]}]}, f)
    return os.path.join(root, "assets")


def write_entities(path, rows, extra_lines=()):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        for line in extra_lines:
            f.write(line + "\n")


def entity(name, position, rotation=(0.0, 0.0, 0.0, 1.0), sxy=1.0, sz=1.0, lod=0, key=True):
    row = {"archetype": f"{world_names.joaat(name):08x}", "position": list(position), "rotation": list(rotation),
           "scale_xy": sxy, "scale_z": sz, "lod_level": lod}
    if key is True:
        row["key"] = f"{world_names.hash64(name):016x}"
    elif key:
        row["key"] = key
    return row


class EntitiesOutTest(unittest.TestCase):
    def test_one_row_per_entity(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "map.bin")
            with open(src, "wb") as f:
                f.write(twi.make_resource())
            ents, arch = os.path.join(tmp, "e.jsonl"), os.path.join(tmp, "a.json")
            with contextlib.redirect_stdout(io.StringIO()):
                rc = WI.main([src, "--out", os.path.join(tmp, "o.json"), "--entities-out", ents, "--archetypes-out", arch])
            self.assertEqual(rc, 0)
            with open(ents, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual((r["archetype"], r["key"]), ("deadbeef", "1122334455667788"))
            self.assertEqual(r["position"], [1.5, 2.5, 3.5])
            self.assertEqual(r["rotation"], [0.0, 0.0, 0.0, 1.0])
            self.assertEqual((r["scale_xy"], r["scale_z"], r["lod_level"]), (1.0, 1.0, 1))
            with open(arch, encoding="utf-8") as f:   # the aggregate file is still written alongside
                self.assertEqual(json.load(f)["archetypes"], {"deadbeef": 1})

    def test_meta_resource_writes_no_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "meta.bin")
            with open(src, "wb") as f:
                f.write(twi.make_resource(map_file=False))
            ents = os.path.join(tmp, "e.jsonl")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(WI.main([src, "--out", os.path.join(tmp, "o.json"), "--entities-out", ents]), 0)
            with open(ents, encoding="utf-8") as f:
                self.assertEqual(f.read(), "")

    def test_error_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            ents = os.path.join(tmp, "e.jsonl")
            with contextlib.redirect_stderr(io.StringIO()):
                rc = WI.main([os.path.join(tmp, "missing"), "--out", os.path.join(tmp, "o.json"), "--entities-out", ents])
            self.assertEqual(rc, 2)
            self.assertFalse(os.path.exists(ents))


class TransformTest(unittest.TestCase):
    def test_identity(self):
        m, n, mirror = world_build.entity_transform([0, 0, 0, 1], 1.0, 1.0)
        self.assertTrue(close(apply(m, (1, 2, 3)), (1, 2, 3)))
        self.assertFalse(mirror)

    def test_stored_quaternion_is_the_inverse(self):
        # A world rotation of +90 degrees about Z takes +X to +Y; the file stores its inverse.
        m, _, _ = world_build.entity_transform(stored_rotation_about_z(90), 1.0, 1.0)
        self.assertTrue(close(apply(m, (1, 0, 0)), (0, 1, 0)))
        self.assertTrue(close(apply(m, (0, 1, 0)), (-1, 0, 0)))

    def test_scale_is_applied_before_rotation(self):
        m, _, _ = world_build.entity_transform(stored_rotation_about_z(90), 2.0, 3.0)
        self.assertTrue(close(apply(m, (1, 0, 1)), (0, 2, 3)))

    def test_unnormalised_quaternion(self):
        q = [v * 5 for v in stored_rotation_about_z(90)]
        m, _, _ = world_build.entity_transform(q, 1.0, 1.0)
        self.assertTrue(close(apply(m, (1, 0, 0)), (0, 1, 0)))

    def test_normals_use_the_inverse_transpose(self):
        # Squash Z by 2: a 45-degree face normal tilts towards Z.
        _, n, _ = world_build.entity_transform([0, 0, 0, 1], 1.0, 0.5)
        v = apply(n, (S2, 0, S2))
        self.assertGreater(v[2], v[0])

    def test_place_converts_to_gltf_y_up(self):
        prims = [{"positions": [(1.0, 0.0, 0.0)], "normals": [(0.0, 0.0, 1.0)], "indices": [0, 0, 0], "uvs": []}]
        out = world_build.place_primitives(prims, [0, 0, 0, 1], 1.0, 1.0, (100.0, 200.0, 10.0))
        self.assertTrue(close(out[0]["positions"][0], (101.0, 10.0, -200.0)))
        self.assertTrue(close(out[0]["normals"][0], (0.0, 1.0, 0.0)))
        self.assertEqual(prims[0]["positions"], [(1.0, 0.0, 0.0)])   # the cached model is not modified

    def test_negative_scale_flips_winding(self):
        prims = [{"positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "normals": [], "indices": [0, 1, 2], "uvs": []}]
        self.assertEqual(world_build.place_primitives(prims, [0, 0, 0, 1], 1.0, -1.0, (0, 0, 0))[0]["indices"], [0, 2, 1])
        self.assertEqual(world_build.place_primitives(prims, [0, 0, 0, 1], 1.0, 1.0, (0, 0, 0))[0]["indices"], [0, 1, 2])


class PlacementLoadTest(unittest.TestCase):
    def load(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "entities.jsonl")
            write_entities(path, rows)
            return world_build.load_placements(path, **kwargs)

    def test_malformed_values_preserve_valid_neighbors(self):
        before = entity("test_box", (1, 2, 3), sxy=2.0, sz=3.0)
        after = entity("test_box", (9, 8, 7), sxy=-1.0, sz=0.5)
        patches = (
            {"lod_level": []}, {"lod_level": {}},
            {"position": [10**400, 0, 0]}, {"rotation": [0, 0, 0, -(10**400)]},
        )
        for patch in patches:
            with self.subTest(field=next(iter(patch))):
                bad = dict(entity("test_box", (4, 5, 6)), **patch)
                rows, counts = self.load([before, bad, after])
                self.assertEqual([row["_pos"] for row in rows], [[1.0, 2.0, 3.0], [9.0, 8.0, 7.0]])
                self.assertEqual([(row["_sxy"], row["_sz"]) for row in rows], [(2.0, 3.0), (-1.0, 0.5)])
                self.assertEqual(counts, {"rows": 3, "malformed": 1, "other_lod_level": 0,
                                          "outside_region": 0, "selected": 2})

    def test_invalid_optional_scales_keep_unit_fallback(self):
        for field in ("scale_xy", "scale_z"):
            for value in (None, "2.0", [], {}, float("nan"), float("inf"), -float("inf"), 10**400, -(10**400)):
                with self.subTest(field=field, value=value):
                    middle = entity("test_box", (4, 5, 6), sxy=2.0, sz=3.0)
                    middle[field] = value
                    rows, counts = self.load([entity("test_box", (0, 0, 0)), middle, entity("test_box", (9, 8, 7))])
                    self.assertEqual(counts["selected"], 3)
                    self.assertEqual(counts["malformed"], 0)
                    self.assertEqual((rows[1]["_sxy"], rows[1]["_sz"]),
                                     (1.0, 3.0) if field == "scale_xy" else (2.0, 1.0))

    def test_finite_scales_and_filter_counts_unchanged(self):
        missing_lod = entity("test_box", (1, 1, 0))
        del missing_lod["lod_level"]
        rows, counts = self.load([
            entity("test_box", (1, 1, 0), sxy=0, sz=-2.5),
            entity("test_box", (2, 2, 0), sxy=-3, sz=0, lod=5),
            entity("test_box", (1, 1, 0), lod=1), missing_lod,
            entity("test_box", (10, 10, 0)),
        ], region=(0, 0, 5, 5))
        self.assertEqual([(row["_sxy"], row["_sz"]) for row in rows], [(0.0, -2.5), (-3.0, 0.0)])
        self.assertEqual(counts, {"rows": 5, "malformed": 0, "other_lod_level": 2,
                                  "outside_region": 1, "selected": 2})


class PlacedBuildTest(unittest.TestCase):
    def build(self, rows, extra_lines=(), **kw):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        assets = write_assets(tmp)
        ents = os.path.join(tmp, "entities.jsonl")
        write_entities(ents, rows, extra_lines)
        doc = world_build.build_world(assets, os.path.join(tmp, "world"), textures=False, placements=ents, **kw)
        return doc

    def test_counts_and_bounds(self):
        rows = [
            entity("test_box", (100, 200, 10), stored_rotation_about_z(90), sxy=2.0, sz=1.0),      # placed
            entity("test_box", (1000, 1000, 0), lod=5, key=False),                                 # unverified
            entity("test_box", (0, 0, 0), key="0000000000000001"),                                 # key mismatch
            entity("other_box", (0, 0, 0)),                                                        # unresolved
            entity("test_box", (0, 0, 0), lod=1),                                                  # LOD level
        ]
        doc = self.build(rows, extra_lines=["{not json", json.dumps({"archetype": "1", "position": [0, 0]})])
        pl = doc["stats"]["placements"]
        self.assertEqual(pl["rows"], 7)
        self.assertEqual(pl["malformed"], 2)
        self.assertEqual(pl["other_lod_level"], 1)
        self.assertEqual(pl["selected"], 4)
        self.assertEqual((pl["placed"], pl["placed_unverified"], pl["key_mismatch"], pl["unresolved_archetype"]),
                         (1, 1, 1, 1))
        self.assertEqual(pl["distinct_models"], 1)
        self.assertEqual(doc["layout"], "placements")
        self.assertEqual(doc["stats"]["models"], 2)
        by_id = {c["id"]: c for c in doc["chunks"]}
        # Box 0..1 scaled (2, 2, 1), turned +90 degrees about Z, moved to (100, 200, 10):
        # RAGE x 98..100, y 200..202, z 10..11 -> glTF X 98..100, Y 10..11, Z -202..-200.
        self.assertTrue(close(by_id["c_0_-1"]["bounds"], [98, 10, -202, 100, 11, -200], 1e-4))
        self.assertTrue(close(by_id["c_3_-4"]["bounds"], [1000, 0, -1001, 1001, 1, -1000], 1e-4))
        self.assertEqual(by_id["c_0_-1"]["triangles"], 12)

    def test_region_and_lod_levels(self):
        rows = [entity("test_box", (10, 10, 0)), entity("test_box", (500, 10, 0)), entity("test_box", (10, 10, 0), lod=1)]
        pl = self.build(rows, region=(0, 0, 100, 100), lod_levels=(0, 1))["stats"]["placements"]
        self.assertEqual((pl["selected"], pl["outside_region"], pl["placed"]), (2, 1, 2))

    def test_build_continues_after_malformed_placement_rows(self):
        rows = [entity("test_box", (10, 10, 0)),
                entity("test_box", (0, 0, 0), lod=[]),
                entity("test_box", (10**400, 0, 0)),
                entity("test_box", (20, 20, 0), sxy=10**400, sz=-(10**400)),
                entity("test_box", (30, 30, 0))]
        doc = self.build(rows)
        counts = doc["stats"]["placements"]
        self.assertEqual((counts["rows"], counts["malformed"], counts["selected"], counts["placed"]), (5, 2, 3, 3))
        # Overflowing optional scales act like the existing missing-scale defaults, including geometry bounds.
        control = self.build([rows[0], entity("test_box", (20, 20, 0)), rows[-1]])
        self.assertEqual(doc["stats"]["triangles"], control["stats"]["triangles"])
        self.assertEqual([chunk["bounds"] for chunk in doc["chunks"]], [chunk["bounds"] for chunk in control["chunks"]])

    def test_nothing_placeable_leaves_output_unchanged(self):
        with self.assertRaisesRegex(ValueError, "could be placed"):
            self.build([entity("other_box", (0, 0, 0))])

    def test_option_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = write_assets(tmp)
            ents = os.path.join(tmp, "e.jsonl")
            write_entities(ents, [entity("test_box", (0, 0, 0))])
            with self.assertRaisesRegex(ValueError, "grid-layout"):
                world_build.build_world(assets, os.path.join(tmp, "w"), placements=ents, grid_layout=True)
            with self.assertRaisesRegex(ValueError, "--up z"):
                world_build.build_world(assets, os.path.join(tmp, "w"), placements=ents, up_axis="y")
            with self.assertRaises(FileNotFoundError):
                world_build.build_world(assets, os.path.join(tmp, "w"), placements=os.path.join(tmp, "none"))

    def test_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            assets = write_assets(tmp)
            ents = os.path.join(tmp, "e.jsonl")
            write_entities(ents, [entity("test_box", (10, 10, 0)), entity("test_box", (900, 10, 0), lod=5)])
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = world_build.main([assets, "--out", os.path.join(tmp, "w"), "--no-textures", "--placements", ents,
                                       "--lod-levels", "0", "--region", "0,0,100,100"])
            self.assertEqual(rc, 0)
            self.assertIn("placements:", out.getvalue())
            with open(os.path.join(tmp, "w", "world.json"), encoding="utf-8") as f:
                pl = json.load(f)["stats"]["placements"]
            self.assertEqual((pl["placed"], pl["other_lod_level"]), (1, 1))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    world_build.main([assets, "--out", os.path.join(tmp, "w"), "--region", "5,5,1,1"])


if __name__ == "__main__":
    unittest.main()
