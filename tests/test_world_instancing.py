"""Tests for the instanced placement build (no game data).

Covers glb_writer.pack_primitive / build_instanced_glb and world_build.py --placements with --instancing
gpu (EXT_mesh_gpu_instancing), nodes and bake: the per-placement TRS against the baked transform, mirrored
placements, one stored mesh per model and chunk, chunk bounds and triangle counts against the baked build,
process workers, textures and the outcome counts. Every model, entity and texture is synthetic.

    python -I tests/test_world_instancing.py
"""
import json
import math
import os
import random
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(ROOT, "tools"), TESTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import glb_writer  # noqa: E402
import test_world_build as twb  # noqa: E402  (independent GLB reader, synthetic DDS fixtures)
import test_world_place as twp  # noqa: E402  (synthetic box model and entity rows)
import world_build  # noqa: E402

EXT = "EXT_mesh_gpu_instancing"


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def trs_matrix(q, s):
    r = glb_writer._quat_matrix(q)
    return [[r[i][j] * s[j] for j in range(3)] for i in range(3)]


# RAGE (x, y, z) -> glTF (x, z, -y)
C = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
CT = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
MIRROR_Y = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]


def read_chunk(world_dir, chunk_id):
    with open(os.path.join(world_dir, "chunks", f"{chunk_id}.glb"), "rb") as f:
        return twb.parse_and_validate_glb(f.read())


def read_indices(gltf, bin_data, acc_idx):
    acc = gltf["accessors"][acc_idx]
    bv = gltf["bufferViews"][acc["bufferView"]]
    fmt = "<%d%s" % (acc["count"], "H" if acc["componentType"] == 5123 else "I")
    return list(struct.unpack_from(fmt, bin_data, bv.get("byteOffset", 0) + acc.get("byteOffset", 0)))


class PlacementTrsTest(unittest.TestCase):
    def test_trs_equals_the_baked_transform(self):
        rnd = random.Random(7)
        for _ in range(500):
            q = [rnd.uniform(-1, 1) for _ in range(4)]
            sxy = rnd.choice([1.0, 2.5, 0.3, -1.0, -2.0])
            sz = rnd.choice([1.0, 0.5, 3.0, -1.0, -0.7])
            pos = [rnd.uniform(-5000, 5000) for _ in range(3)]
            t, qi, si, mirrored = world_build.placement_trs(q, sxy, sz, pos)
            self.assertTrue(all(v > 0 for v in si))
            self.assertEqual(mirrored, sz < 0)
            m_rage, _, mirror = world_build.entity_transform(q, sxy, sz)
            self.assertEqual(mirror, mirrored)
            want = matmul(matmul(C, m_rage), CT)            # the baked transform, in glTF axes
            got = trs_matrix(qi, si)
            if mirrored:
                got = matmul(got, MIRROR_Y)                 # a mirrored placement draws the Y-mirrored mesh
            for i in range(3):
                for j in range(3):
                    self.assertAlmostEqual(got[i][j], want[i][j], places=9)
            self.assertTrue(twp.close(t, (pos[0], pos[2], -pos[1])))
            self.assertAlmostEqual(sum(v * v for v in qi), 1.0, places=12)

    def test_instance_bounds_of_a_turned_box(self):
        # Unit box, scale 2, turned 90 degrees about +Y, moved by (10, 0, 0): x 10..12, z -2..0 after the turn.
        h = math.sqrt(0.5)
        b = glb_writer.instance_bounds([0, 0, 0], [1, 1, 1], [10, 0, 0], [0, h, 0, h], [2, 2, 2])
        self.assertTrue(twp.close(b, [10, 0, -2, 12, 2, 0], 1e-12))


def tri_prim():
    return glb_writer.pack_primitive({"positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)], "normals": [],
                                      "indices": [0, 1, 2], "uvs": []})


class InstancedGlbTest(unittest.TestCase):
    def test_gpu_instance_bounds_follow_the_stored_translation_and_scale(self) -> None:
        meshes = [{
            "primitives": [tri_prim()],
            "instances": [((16_777_219, 0, 0), (0, 0, 0, 1), (1, 1.00000003, 1))],
        }]
        data, stats = glb_writer.build_instanced_glb(meshes)
        gltf, binary = twb.parse_and_validate_glb(data)
        attrs = gltf["nodes"][0]["extensions"][EXT]["attributes"]
        (translation,) = twb.read_accessor_floats(gltf, binary, attrs["TRANSLATION"])
        (scale,) = twb.read_accessor_floats(gltf, binary, attrs["SCALE"])
        self.assertEqual(translation, (16_777_220, 0, 0))
        self.assertEqual(scale, (1, 1, 1))
        self.assertEqual(stats["bounds"], [16_777_220, 0, 0, 16_777_221, 1, 0])

    def test_gpu_mode(self):
        meshes = [{"name": "a", "primitives": [tri_prim()],
                   "instances": [((0, 0, 0), (0, 0, 0, 1), (1, 1, 1)), ((5, 0, 0), (0, 0, 0, 1), (2, 2, 2))]},
                  {"name": "b", "primitives": [tri_prim()], "instances": [((0, 9, 0), (0, 0, 0, 1), (1, 1, 1))]}]
        data, st = glb_writer.build_instanced_glb(meshes)
        gltf, bin_data = twb.parse_and_validate_glb(data)
        glb_writer.read_glb_summary(data)
        self.assertEqual(gltf["extensionsUsed"], [EXT])
        self.assertEqual(gltf["extensionsRequired"], [EXT])
        self.assertEqual(len(gltf["meshes"]), 2)
        self.assertEqual(len(gltf["nodes"]), 2)
        self.assertEqual(gltf["scenes"][0]["nodes"], [0, 1])
        attrs = gltf["nodes"][0]["extensions"][EXT]["attributes"]
        self.assertEqual(twb.read_accessor_floats(gltf, bin_data, attrs["TRANSLATION"]), [(0, 0, 0), (5, 0, 0)])
        self.assertEqual(twb.read_accessor_floats(gltf, bin_data, attrs["SCALE"]), [(1, 1, 1), (2, 2, 2)])
        self.assertEqual(gltf["accessors"][attrs["ROTATION"]]["type"], "VEC4")
        self.assertNotIn("target", gltf["bufferViews"][gltf["accessors"][attrs["ROTATION"]]["bufferView"]])
        self.assertEqual((st["triangles"], st["stored_triangles"], st["instances"], st["meshes"]), (3, 2, 3, 2))
        self.assertTrue(twp.close(st["bounds"], [0, 0, 0, 7, 10, 0]))

    def test_nodes_mode_writes_one_node_per_instance(self):
        h = math.sqrt(0.5)
        meshes = [{"primitives": [tri_prim()],
                   "instances": [((0, 0, 0), (0, 0, 0, 1), (1, 1, 1)), ((1, 2, 3), (0, 0, h, h), (1, 1, 4))]}]
        data, st = glb_writer.build_instanced_glb(meshes, mode="nodes")
        gltf, _ = twb.parse_and_validate_glb(data)
        self.assertNotIn("extensionsUsed", gltf)
        self.assertEqual(gltf["nodes"][0], {"mesh": 0})
        self.assertEqual(gltf["nodes"][1]["translation"], [1, 2, 3])
        self.assertEqual(gltf["nodes"][1]["scale"], [1, 1, 4])
        self.assertTrue(twp.close(gltf["nodes"][1]["rotation"], [0, 0, h, h], 1e-12))
        self.assertEqual(len(gltf["meshes"]), 1)
        self.assertEqual(st["instances"], 2)

    def test_indexed_material_is_preserved_in_both_instancing_modes(self) -> None:
        primitive = {
            "positions": [(0, 0, 0), (1, 0, 0), (0, 1, 0)],
            "indices": [0, 1, 2],
            "material_index": 2,
        }
        materials = {2: {"name": "paint", "base_color_factor": [0.25, 0.5, 0.75, 1.0]}}
        meshes = [{"primitives": [glb_writer.pack_primitive(primitive)],
                   "instances": [((0, 0, 0), (0, 0, 0, 1), (1, 1, 1))]}]
        for mode in ("gpu", "nodes"):
            with self.subTest(mode=mode):
                data, _ = glb_writer.build_instanced_glb(meshes, materials, mode=mode)
                gltf, _ = twb.parse_and_validate_glb(data)
                material_index = gltf["meshes"][0]["primitives"][0]["material"]
                material = gltf["materials"][material_index]
                self.assertEqual(material["name"], "paint")
                self.assertEqual(material["pbrMetallicRoughness"]["baseColorFactor"],
                                 [0.25, 0.5, 0.75, 1.0])

    def test_rejects_mirroring_scale_and_bad_input(self):
        bad = [{"primitives": [tri_prim()], "instances": [((0, 0, 0), (0, 0, 0, 1), (1, -1, 1))]}]
        with self.assertRaisesRegex(glb_writer.GlbError, "determinant"):
            glb_writer.build_instanced_glb(bad)
        with self.assertRaisesRegex(glb_writer.GlbError, "no mesh"):
            glb_writer.build_instanced_glb([{"primitives": [tri_prim()], "instances": []}])
        with self.assertRaisesRegex(glb_writer.GlbError, "mode"):
            glb_writer.build_instanced_glb(bad, mode="other")

    def test_mirror_packed(self):
        p = glb_writer.pack_primitive({"positions": [(0, 1, 0), (1, 2, 0), (0, 3, 1)], "normals": [],
                                       "indices": [0, 1, 2], "uvs": []})
        m = world_build._mirror_packed(p)
        pos = struct.unpack("<9f", m["positions"])
        self.assertEqual(pos[1::3], (-1.0, -2.0, -3.0))
        self.assertEqual(struct.unpack("<3H", m["indices"]), (0, 2, 1))
        self.assertEqual((m["min"][1], m["max"][1]), (-3.0, -1.0))
        nrm = struct.unpack("<9f", m["normals"])
        orig = struct.unpack("<9f", p["normals"])
        self.assertEqual(nrm[1::3], tuple(-v for v in orig[1::3]))

    def test_pack_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "m.lpk")
            p = tri_prim()
            p["material"] = "tex_x"
            world_build._write_pack(path, [p, tri_prim()])
            back = world_build._read_pack(path)
        self.assertEqual(len(back), 2)
        for k in ("positions", "normals", "uvs", "indices", "index_type", "vertices", "index_count", "min", "max"):
            self.assertEqual(back[0][k], p[k])
        self.assertEqual(back[0]["material"], "tex_x")


# A placement set that exercises turns, non-uniform scale, both negative scales and two chunks.
def mixed_rows():
    S2 = math.sqrt(0.5)
    return [
        twp.entity("test_box", (100, 200, 10), twp.stored_rotation_about_z(90), sxy=2.0, sz=1.0),
        twp.entity("test_box", (110, 205, 12), [0.1, 0.2, 0.3, 0.9], sxy=1.5, sz=0.5),
        twp.entity("test_box", (120, 210, 0), [0.0, S2, 0.0, S2], sxy=1.0, sz=-1.0),       # mirrored
        twp.entity("test_box", (130, 220, 0), [0.0, 0.0, 0.0, 1.0], sxy=-2.0, sz=1.0),     # XY turn, not a mirror
        twp.entity("test_box", (1000, 1000, 0), lod=5, key=False),                          # second chunk
        twp.entity("test_box", (5, 5, 5), sxy=0.0),                                         # degenerate
        twp.entity("other_box", (0, 0, 0)),                                                 # unresolved
    ]


class PlacementArchetypeParityTest(unittest.TestCase):
    def test_integer_and_hex_string_parse_to_the_same_archetype(self):
        rows = [
            {"archetype": 10, "position": [0, 0, 0], "rotation": [0, 0, 0, 1], "lod_level": 0},
            {"archetype": "a", "position": [0, 0, 0], "rotation": [0, 0, 0, 1], "lod_level": 0},   # 0xa == 10
            {"archetype": True, "position": [0, 0, 0], "rotation": [0, 0, 0, 1], "lod_level": 0},  # bool stays invalid
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "entities.jsonl")
            twp.write_entities(path, rows)
            loaded, load_counts = world_build.load_placements(path)
            iter_counts = world_build._placement_counts()
            itered = list(world_build.iter_placements(path, iter_counts))
        self.assertEqual([row["_arch"] for row in loaded], [10, 10])
        self.assertEqual([row["_arch"] for row in itered], [10, 10])
        self.assertEqual(load_counts, iter_counts)
        self.assertEqual((load_counts["rows"], load_counts["malformed"], load_counts["selected"]), (3, 1, 2))


class InstancedBuildTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(self.tmp, ignore_errors=True))
        self.assets = twp.write_assets(self.tmp)

    def build(self, rows, name="world", **kw):
        ents = os.path.join(self.tmp, f"{name}.jsonl")
        twp.write_entities(ents, rows)
        kw.setdefault("textures", False)
        out = os.path.join(self.tmp, name)
        return world_build.build_world(self.assets, out, placements=ents, **kw), out

    def test_shared_geometry_per_chunk(self):
        rows = [twp.entity("test_box", (10 + i, 10, 0)) for i in range(3)]
        doc, out = self.build(rows)
        self.assertEqual(doc["instancing"], "gpu")
        (chunk,) = doc["chunks"]
        self.assertEqual((chunk["models"], chunk["distinct_models"], chunk["triangles"], chunk["stored_triangles"]),
                         (3, 1, 36, 12))
        gltf, bin_data = read_chunk(out, chunk["id"])
        self.assertEqual((len(gltf["meshes"]), len(gltf["nodes"])), (1, 1))
        attrs = gltf["nodes"][0]["extensions"][EXT]["attributes"]
        got = twb.read_accessor_floats(gltf, bin_data, attrs["TRANSLATION"])
        self.assertEqual(sorted(got), [(10, 0, -10), (11, 0, -10), (12, 0, -10)])
        self.assertEqual(doc["stats"]["stored_triangles"], 12)
        self.assertEqual(doc["stats"]["max_chunk_bytes"], doc["stats"]["chunk_bytes"])

    def test_same_chunks_bounds_and_counts_as_the_baked_build(self):
        rows = [r for r in mixed_rows() if r["scale_xy"] != 0]      # bake also places a flat (zero-scale) copy
        gpu, _ = self.build(rows, "gpu")
        bake, _ = self.build(rows, "bake", instancing="bake")
        self.assertEqual([c["id"] for c in gpu["chunks"]], [c["id"] for c in bake["chunks"]])
        for a, b in zip(gpu["chunks"], bake["chunks"]):
            self.assertTrue(twp.close(a["bounds"], b["bounds"], 1e-4), (a["bounds"], b["bounds"]))
            self.assertEqual((a["models"], a["triangles"]), (b["models"], b["triangles"]))
        pg, pb = gpu["stats"]["placements"], bake["stats"]["placements"]
        for k in ("rows", "selected", "placed", "placed_unverified", "unresolved_archetype", "key_mismatch"):
            self.assertEqual(pg[k], pb[k], k)
        self.assertEqual((pg["degenerate_scale"], pg["mirrored"]), (0, 1))

    def test_zero_scale_is_counted_not_placed(self):
        doc, _ = self.build(mixed_rows())
        pl = doc["stats"]["placements"]
        self.assertEqual((pl["degenerate_scale"], pl["placed"], pl["placed_unverified"]), (1, 4, 1))

    def test_mirrored_placement_draws_a_mirrored_mesh(self):
        doc, out = self.build([twp.entity("test_box", (10, 10, 0), sz=-1.0), twp.entity("test_box", (12, 10, 0))])
        gltf, bin_data = read_chunk(out, doc["chunks"][0]["id"])
        names = sorted(m.get("name", "") for m in gltf["meshes"])
        self.assertEqual(len(names), 2)
        self.assertTrue(names[1].endswith("_mirror"))
        by_name = {m["name"]: m for m in gltf["meshes"]}
        plain, mirror = by_name[names[0]]["primitives"][0], by_name[names[1]]["primitives"][0]
        a = read_indices(gltf, bin_data, plain["indices"])
        b = read_indices(gltf, bin_data, mirror["indices"])
        self.assertEqual(b, [x for t in zip(a[0::3], a[2::3], a[1::3]) for x in t])
        pa = twb.read_accessor_floats(gltf, bin_data, plain["attributes"]["POSITION"])
        pb = twb.read_accessor_floats(gltf, bin_data, mirror["attributes"]["POSITION"])
        self.assertEqual(pb, [(x, -y, z) for x, y, z in pa])
        # Mirrored about the model's Z (glTF Y): the box spans y 0..1 below the placement instead of above.
        self.assertTrue(twp.close(doc["chunks"][0]["bounds"][1:5:3], [-1, 1], 1e-6))

    def test_nodes_mode(self):
        doc, out = self.build([twp.entity("test_box", (10 + i, 10, 0)) for i in range(4)], instancing="nodes")
        gltf, _ = read_chunk(out, doc["chunks"][0]["id"])
        self.assertEqual((len(gltf["meshes"]), len(gltf["nodes"])), (1, 4))
        self.assertNotIn("extensionsRequired", gltf)
        self.assertEqual(doc["instancing"], "nodes")

    def test_process_workers_give_the_same_world(self):
        one, out1 = self.build(mixed_rows(), "one", jobs=1)
        two, out2 = self.build(mixed_rows(), "two", jobs=2)
        one.pop("spawn"), two.pop("spawn")
        for c in one["chunks"] + two["chunks"]:
            c.pop("url")
        self.assertEqual(one, two)
        for c in one["chunks"]:
            with open(os.path.join(out1, "chunks", c["id"] + ".glb"), "rb") as f1, \
                    open(os.path.join(out2, "chunks", c["id"] + ".glb"), "rb") as f2:
                self.assertEqual(f1.read(), f2.read())

    def test_missing_model_file_is_counted(self):
        os.remove(os.path.join(self.assets, "levels", "models", "box.obj"))
        with self.assertRaisesRegex(ValueError, "could be placed"):
            self.build([twp.entity("test_box", (10, 10, 0))])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "world", "world.json")))

    def test_pack_cache_is_removed(self):
        _, out = self.build([twp.entity("test_box", (10, 10, 0))])
        self.assertEqual(sorted(os.listdir(out)), ["chunks", "world.json"])

    def test_textured_instances_embed_the_texture_once(self):
        assets = os.path.join(self.tmp, "tex_assets")
        twb._write_param_fixture(assets, twb.PARAM_CASES["albedo-first"])
        ents = os.path.join(self.tmp, "tex.jsonl")
        twp.write_entities(ents, [twp.entity("tri", (10 + i, 10, 0)) for i in range(5)])
        out = os.path.join(self.tmp, "tex_world")
        doc = world_build.build_world(assets, out, placements=ents)
        self.assertEqual(doc["stats"]["textures_embedded"], 1)
        glb = os.path.join(out, "chunks", doc["chunks"][0]["id"] + ".glb")
        _, rgba = twb._glb_first_base_color_pixel(glb)
        self.assertEqual(rgba, tuple(twb.ALBEDO_RGBA))
        gltf, _ = read_chunk(out, doc["chunks"][0]["id"])
        self.assertEqual((len(gltf["images"]), len(gltf["meshes"])), (1, 1))

    def test_texture_cache_respects_each_build_size(self) -> None:
        assets = os.path.join(self.tmp, "tex_assets")
        twb._write_param_fixture(assets, twb.PARAM_CASES["albedo-first"])
        ents = os.path.join(self.tmp, "tex.jsonl")
        twp.write_entities(ents, [twp.entity("tri", (10, 10, 0))])

        for limit, expected in ((1, 1), (0, 4), (2, 2)):
            with self.subTest(limit=limit):
                out = os.path.join(self.tmp, f"world_{limit}")
                doc = world_build.build_world(assets, out, placements=ents, jobs=1, max_tex_size=limit)
                gltf, bin_data = read_chunk(out, doc["chunks"][0]["id"])
                (image,) = gltf["images"]
                view = gltf["bufferViews"][image["bufferView"]]
                start = view.get("byteOffset", 0)
                width, height, rgba = twb.parse_and_validate_png(bin_data[start:start + view["byteLength"]])
                self.assertEqual((width, height), (expected, expected))
                self.assertEqual(tuple(rgba[:4]), tuple(twb.ALBEDO_RGBA))

    def test_rebuild_embeds_changed_texture_at_the_same_path(self) -> None:
        assets = os.path.join(self.tmp, "tex_assets")
        twb._write_param_fixture(assets, twb.PARAM_CASES["albedo-first"])
        ents = os.path.join(self.tmp, "tex.jsonl")
        twp.write_entities(ents, [twp.entity("tri", (10, 10, 0))])
        out = os.path.join(self.tmp, "tex_world")
        dds = os.path.join(assets, "models", "tri_tex", "001_albedo.dds")

        for color in (tuple(twb.ALBEDO_RGBA), (25, 90, 175, 255)):
            with self.subTest(color=color):
                with open(dds, "wb") as texture:
                    texture.write(twb._uniform_bgra_dds(color))
                doc = world_build.build_world(assets, out, placements=ents, jobs=1)
                glb = os.path.join(out, "chunks", doc["chunks"][0]["id"] + ".glb")
                _, rgba = twb._glb_first_base_color_pixel(glb)
                self.assertEqual(rgba, color)

    def test_cli_instancing_option(self):
        ents = os.path.join(self.tmp, "cli.jsonl")
        twp.write_entities(ents, [twp.entity("test_box", (10, 10, 0))])
        out = os.path.join(self.tmp, "cli")
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = world_build.main([self.assets, "--out", out, "--no-textures", "--placements", ents,
                                   "--instancing", "nodes"])
        self.assertEqual(rc, 0)
        self.assertIn("placements", buf.getvalue())
        with open(os.path.join(out, "world.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["instancing"], "nodes")


class LodSetsTest(unittest.TestCase):
    def test_bands_follow_the_parent_child_distances(self):
        child = {1: [120, 130, 140], 2: [450], 3: [1000], 4: [1976], 6: [6000]}
        own = {0: [124], 1: [450], 2: [1000], 3: [2000], 4: [4000], 5: [900, 1100], 6: [20000]}
        bands, measured = world_build.lod_bands(child, own, [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(bands, {"hd": [0.0, 130.0], "lod": [130.0, 450.0], "slod1": [450.0, 1000.0],
                                 "slod2": [1000.0, 1976.0], "slod3": [1976.0, 6000.0], "slod4": [6000.0, 20000.0],
                                 "orphan_hd": [0.0, 1000.0]})
        self.assertEqual(measured["lod"]["child_lod_dist_median"], 130.0)

    def test_missing_tiers_still_give_contiguous_bands(self):
        bands, _ = world_build.lod_bands({2: [450]}, {2: [1000]}, [0, 2])
        self.assertEqual(bands, {"hd": [0.0, 450.0], "slod1": [450.0, 1000.0]})

    def build(self, rows, **kw):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: shutil.rmtree(tmp, ignore_errors=True))
        assets = twp.write_assets(tmp)
        ents = os.path.join(tmp, "e.jsonl")
        twp.write_entities(ents, rows)
        out = os.path.join(tmp, "w")
        return world_build.build_world(assets, out, placements=ents, textures=False, **kw), out

    def lod_row(self, pos, lod, child=None, own=None):
        r = twp.entity("test_box", pos, lod=lod)
        r["child_lod_dist"] = child
        r["lod_dist"] = own
        return r

    def test_one_chunk_per_tier_and_cell(self):
        rows = [self.lod_row((10, 10, 0), 0, own=-1.0), self.lod_row((12, 10, 0), 0),
                self.lod_row((10, 10, 0), 1, child=130.0, own=450.0), self.lod_row((10, 10, 0), 2, child=450.0, own=1000.0),
                self.lod_row((300, 10, 0), 5, own=800.0)]
        doc, out = self.build(rows, lod_sets=True, lod_levels=world_build.ALL_LOD_LEVELS)
        ids = sorted(c["id"] for c in doc["chunks"])
        self.assertEqual(ids, ["c_hd_0_-1", "c_lod_0_-1", "c_orphan_hd_1_-1", "c_slod1_0_-1"])
        by = {c["id"]: c for c in doc["chunks"]}
        self.assertEqual(by["c_hd_0_-1"]["cell"], [0.0, -256.0, 256.0, 0.0])
        self.assertEqual(by["c_hd_0_-1"]["lod"], "hd")
        self.assertEqual(by["c_hd_0_-1"]["models"], 2)
        self.assertEqual(doc["lod_bands"], {"hd": [0.0, 130.0], "lod": [130.0, 450.0], "slod1": [450.0, 1000.0],
                                            "orphan_hd": [0.0, 800.0]})
        self.assertEqual(doc["stats"]["lod_tiers"]["hd"]["placements"], 2)
        with open(os.path.join(out, "world.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["lod_bands"]["lod"], [130.0, 450.0])

    def test_export_uses_finer_lod_distance_when_child_distance_is_missing(self) -> None:
        rows = [self.lod_row((10, 10, 0), 0, own=120.0),
                self.lod_row((10, 10, 0), 1, own=450.0)]
        _, out = self.build(rows, lod_sets=True, lod_levels=(0, 1))
        with open(os.path.join(out, "world.json"), encoding="utf-8") as handle:
            world = json.load(handle)
        self.assertEqual(sorted(chunk["lod"] for chunk in world["chunks"]), ["hd", "lod"])
        for chunk in world["chunks"]:
            self.assertTrue(os.path.isfile(os.path.join(out, "chunks", f'{chunk["id"]}.glb')))
        self.assertEqual(world["lod_bands"], {"hd": [0.0, 120.0], "lod": [120.0, 450.0]})

    def test_option_rules(self):
        with self.assertRaisesRegex(ValueError, "lod-sets needs --instancing"):
            self.build([self.lod_row((0, 0, 0), 0)], lod_sets=True, instancing="bake")
        with tempfile.TemporaryDirectory() as tmp:
            assets = twp.write_assets(tmp)
            with self.assertRaisesRegex(ValueError, "lod-sets needs --placements"):
                world_build.build_world(assets, os.path.join(tmp, "w"), lod_sets=True)

    def test_cli_rejects_unnamed_lod_levels_with_lod_sets(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            assets = twp.write_assets(tmp)
            ents = os.path.join(tmp, "e.jsonl")
            twp.write_entities(ents, [self.lod_row((10, 10, 0), level) for level in (7, 255)])
            out = os.path.join(tmp, "w")
            for level in (7, 255):
                with self.subTest(level=level), contextlib.redirect_stderr(io.StringIO()) as stderr:
                    result = world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                               "--lod-sets", "--lod-levels", f"0,{level}"])
                    self.assertEqual(result, 1)
                    self.assertIn(f"--lod-sets does not recognize lodLevel {level}", stderr.getvalue())
                    self.assertFalse(os.path.exists(out))

    def test_unnamed_lod_levels_work_without_lod_sets(self):
        rows = [self.lod_row((10, 10, 0), level) for level in (7, 255)]
        doc, _ = self.build(rows, lod_levels=(7, 255))
        self.assertEqual(doc["stats"]["placements"]["placed"], 2)
        self.assertNotIn("lod_bands", doc)

    def test_cli_places_every_level_with_lod_sets(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            assets = twp.write_assets(tmp)
            ents = os.path.join(tmp, "e.jsonl")
            twp.write_entities(ents, [self.lod_row((10, 10, 0), 0), self.lod_row((10, 10, 0), 3, child=1000.0)])
            out = os.path.join(tmp, "w")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                                   "--lod-sets"]), 0)
                with open(os.path.join(out, "world.json"), encoding="utf-8") as f:
                    self.assertEqual(sorted(json.load(f)["lod_bands"]), ["hd", "slod2"])
                self.assertEqual(world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                                   "--lod-sets", "--lod-levels", "0"]), 0)
                with open(os.path.join(out, "world.json"), encoding="utf-8") as f:
                    self.assertEqual(sorted(json.load(f)["lod_bands"]), ["hd"])

    def test_prune_stale_chunks_removes_obsolete_lod_set_chunks(self):
        import contextlib
        import io
        with tempfile.TemporaryDirectory() as tmp:
            chunks_dir = os.path.join(tmp, "direct_chunks")
            os.makedirs(chunks_dir)
            for name in ("c_hd_0_-1.glb", "c_lod_0_-1.glb", "c_slod2_0_-1.glb",
                         "c_orphan_hd_1_-1.glb", "c_hd_0_-1.glb.tmp", "c_0_-1.glb", "unrelated.glb"):
                with open(os.path.join(chunks_dir, name), "wb") as f:
                    f.write(b"x")
            world_build._prune_stale_chunks(chunks_dir, {"c_hd_0_-1.glb"})
            self.assertEqual(sorted(os.listdir(chunks_dir)), ["c_hd_0_-1.glb", "unrelated.glb"])

            assets = twp.write_assets(tmp)
            ents = os.path.join(tmp, "e.jsonl")
            twp.write_entities(ents, [self.lod_row((10, 10, 0), 0), self.lod_row((10, 10, 0), 3, child=1000.0)])
            out = os.path.join(tmp, "w")
            out_chunks = os.path.join(out, "chunks")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                                   "--lod-sets"]), 0)
                self.assertEqual(sorted(os.listdir(out_chunks)), ["c_hd_0_-1.glb", "c_slod2_0_-1.glb"])
                # Narrowing --lod-levels must prune the obsolete slod2 chunk.
                self.assertEqual(world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                                   "--lod-sets", "--lod-levels", "0"]), 0)
                self.assertEqual(sorted(os.listdir(out_chunks)), ["c_hd_0_-1.glb"])
                # Switching to a non-LOD build must prune the obsolete c_hd_* chunk.
                self.assertEqual(world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                                   "--lod-levels", "0"]), 0)
                self.assertEqual(sorted(os.listdir(out_chunks)), ["c_0_-1.glb"])
                # Rebuilding with --lod-sets and then --limit 0 must clear all LOD chunks.
                self.assertEqual(world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                                   "--lod-sets"]), 0)
                self.assertEqual(world_build.main([assets, "--out", out, "--no-textures", "--placements", ents,
                                                   "--lod-sets", "--limit", "0"]), 0)
                self.assertEqual(os.listdir(out_chunks), [])


class IterPlacementsTest(unittest.TestCase):
    def test_same_rows_and_counts_as_load_placements(self):
        good = [twp.entity("test_box", (1, 2, 3), sxy=2.0, sz=3.0), twp.entity("test_box", (9, 8, 7), lod=5),
                twp.entity("test_box", (5, 5, 5), lod=1), twp.entity("test_box", (500, 5, 5))]
        odd = [dict(twp.entity("test_box", (4, 5, 6)), lod_level=[]),
               dict(twp.entity("test_box", (4, 5, 6)), position=[10 ** 400, 0, 0]),
               dict(twp.entity("test_box", (4, 5, 6)), scale_xy="2.0", scale_z=10 ** 400)]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "e.jsonl")
            twp.write_entities(path, good + odd, extra_lines=["{not json"])
            for kw in ({}, {"lod_levels": (0, 1)}, {"region": (0, 0, 100, 100)}):
                rows, counts = world_build.load_placements(path, **kw)
                c2 = world_build._placement_counts()
                rows2 = list(world_build.iter_placements(path, c2, **kw))
                self.assertEqual(rows2, rows, kw)
                self.assertEqual(c2, counts, kw)


if __name__ == "__main__":
    unittest.main()
