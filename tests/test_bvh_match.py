"""Synthetic checks for tools/bvh_match.py: the manifest entry the BLAS was built from is found and verified, a
wrong geometry, a wrong shader rule and a missing transform are not, and geometries are matched inside one model.
No game data is read.

    python -I tests/test_bvh_match.py
"""
import contextlib
import io
import itertools
import json
import os
import struct
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import bvh_match as M  # noqa: E402
import bvh_rehydrate as R  # noqa: E402
from test_bvh_rehydrate import build  # noqa: E402

# Two triangles over x,y 0..1 sharing the edge (1, 2); the leaf holds both primitives.
QUAD = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
QUAD_FACES = [(0, 1, 2), (2, 1, 3)]
LEAVES = [(0, 0x81, 0, 1, 0x09)]
BOUNDS = (0.0, 0.0, 0.0, 1.0, 1.0, 0.0)
SHIFTED = ((1.0, 0.0, 0.0, 0.5), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0))


def group(triangles):
    """A geometry of `triangles` triangles over the quad's four vertices."""
    return QUAD, [QUAD_FACES[i % 2] for i in range(triangles)]


def write_obj(path, groups):
    """groups: [(positions, faces)] -> one `o` group each, in the shape CreateObj writes."""
    lines, first = [], 1
    for index, (positions, faces) in enumerate(groups):
        lines.append("o model_0_geometry_%d" % index)
        lines += ["v %.9g %.9g %.9g" % p for p in positions]
        lines += ["f %d//%d %d//%d %d//%d" % (first + a, first + a, first + b, first + b, first + c, first + c)
                  for a, b, c in faces]
        first += len(positions)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def item(offset, file, triangles, hashes=None, buckets=None):
    """One manifest row, as tools/rpf9dump writes it: per geometry triangles and shader index, and with `buckets`
    one render bucket per material (None leaves the field out, as an older export does)."""
    materials = [{"ShaderIndex": i, "Shader": "%08X" % h} for i, h in enumerate(hashes or [])]
    for material, bucket in zip(materials, buckets or []):
        if bucket is not None:
            material["RenderBucket"] = bucket
    return {"Offset": offset, "File": file, "Name": "drawable_at_%X" % offset, "Meshes": len(triangles),
            "Verts": 0, "Tris": sum(triangles), "MeshTris": list(triangles),
            "MeshModels": [0] * len(triangles), "MeshGeoms": list(range(len(triangles))),
            "MeshShaders": list(range(len(triangles))) if hashes else [None] * len(triangles),
            "Materials": materials}


class TestMatching(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = directory.name

    def guide(self, blas, items, obj_groups, argv=()):
        """Write the export this run reads, call the matcher, return (status, stdout, stderr)."""
        manifest = os.path.join(self.dir, "manifest.json")
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({"Count": len(items), "Items": items}, f)
        for name, groups in obj_groups.items():
            write_obj(os.path.join(self.dir, name), groups)
        path = os.path.join(self.dir, "blas.bin")
        with open(path, "wb") as f:
            f.write(blas)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            status = M.main([path, manifest, *argv])
        return status, out.getvalue(), err.getvalue()

    def test_a_manifest_that_is_not_an_object_is_one_error_line(self):
        path = os.path.join(self.dir, "blas.bin")
        with open(path, "wb") as f:
            f.write(build(LEAVES, bounds=BOUNDS)[0])
        for doc in ([1], [], "x", 3, None):
            manifest = os.path.join(self.dir, "manifest.json")
            with open(manifest, "w", encoding="utf-8") as f:
                json.dump(doc, f)
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                status = M.main([path, manifest])
            self.assertEqual(status, 2, doc)
            lines = err.getvalue().splitlines()
            self.assertEqual(len(lines), 1, err.getvalue())
            self.assertTrue(lines[0].startswith("error: "), lines[0])
            self.assertNotIn("Traceback", err.getvalue())

    def test_the_right_geometry_is_found_and_consistent(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [5, 2, 3])],
                                    {"a.obj": [group(5), group(2), group(3)]})
        self.assertEqual(status, 0, out)
        self.assertIn("triangles per geometry index: [2]", out)
        self.assertIn("1 of 1 scored candidates are consistent", out)
        self.assertIn("model 0 geometry [1] (bucket rule not applied: a geometry has no render bucket) | outside 0 of 1",
                      out)

    def test_a_geometry_with_other_triangle_counts_is_not_a_candidate(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [5, 3])], {"a.obj": [group(5), group(3)]})
        self.assertEqual(status, 1, out)
        self.assertIn("0 candidates", out)
        self.assertIn("0 of 0 scored candidates are consistent", out)
        # An empty run says why: the drawable is outside the exported window, or the rule does not fit.
        self.assertIn("outside the exported window", out)

    def test_a_wrong_geometry_is_rejected(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        # The same two triangles, half a unit away from what the BLAS stores: the leaves fall outside sceneBounds.
        moved = [(x + 0.5, y, z) for x, y, z in QUAD]
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [(moved, QUAD_FACES)]})
        self.assertEqual(status, 1, out)
        self.assertIn("rejected", out)
        self.assertIn("outside 1 of 1", out)

    def test_the_shader_rule_leaves_a_geometry_out(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        hashes = [0xAAAA1111, 0xBBBB2222]
        files = {"a.obj": [group(4), group(2)]}
        # Geometry 0 has the excluded shader, so the two triangles belong to geometry 1.
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [4, 2], hashes)], files,
                                    ["--skip-shaders", "%08X" % hashes[0]])
        self.assertEqual(status, 0, out)
        self.assertIn("model 0 geometry [1]", out)
        # Excluding the other shader instead leaves a four-triangle geometry, which the signature does not fit.
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [4, 2], hashes)], files,
                                    ["--skip-shaders", "%08X" % hashes[1]])
        self.assertEqual(status, 1, out)

    def test_the_bucket_rule_leaves_buckets_1_and_2_out(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        hashes = [0x11110000, 0x22220000, 0x33330000, 0x44440000]
        # Buckets 1 and 2 are left out, so geometry 3 (bucket 3) is index 1 and geometry 0 (bucket 0) index 0; the
        # BLAS's one leaf uses index 0 with two triangles, which only the bucket-0 geometry has once 1 and 2 are out.
        files = {"a.obj": [group(2), group(2), group(2), group(9)]}
        rows = [item(0x1000, "a.obj", [2, 2, 2, 9], hashes, [0, 1, 2, 3])]
        self.assertEqual([mesh[5] for mesh in M.model_geometries(rows[0])], [0, 1, 2, 3])
        candidates = M.match_candidates(rows, [2, 9], frozenset(), 8)
        self.assertEqual([[c["kept"][k][0] for k in c["used"]] for c in candidates], [[0, 3]])
        # Without the rule, [2, 9] also embeds through geometries 1 and 2.
        loose = M.match_candidates(rows, [2, 9], frozenset(), 8, buckets=False)
        self.assertEqual(sorted([c["kept"][k][0] for k in c["used"]] for c in loose), [[0, 3], [1, 3], [2, 3]])
        status, out, _ = self.guide(src, rows, files)
        self.assertEqual(status, 0, out)
        self.assertIn("1 candidates", out)
        self.assertIn("model 0 geometry [0] | outside 0 of 1", out)

    def test_the_bucket_rule_finds_what_the_signature_alone_cannot(self):
        # Signature [2, 5]: geometries 0 and 2 (bucket 0 and 3); geometry 1 (bucket 2, 5 triangles) is a decal the
        # BVH leaves out. Without the rule the ordered match would also take geometry 1 for the 5.
        rows = [item(0x1000, "a.obj", [2, 5, 5], [1, 2, 3], [0, 2, 3])]
        self.assertEqual([[c["kept"][k][0] for k in c["used"]] for c in M.match_candidates(rows, [2, 5], frozenset(), 8)],
                         [[0, 2]])
        self.assertEqual(len(M.match_candidates(rows, [2, 5], frozenset(), 8, buckets=False)), 2)
        # A model all of whose geometries are left out has no candidate.
        self.assertEqual(M.match_candidates([item(0x2000, "b.obj", [2], [1], [1])], [2], frozenset(), 8), [])

    def test_the_bucket_rule_ignores_bit_7(self):
        # 0x83 is bucket 3 with bit 7 set (kept); 0x82 is bucket 2 with bit 7 set (left out); 6 is left out.
        rows = [item(0x1000, "a.obj", [4, 3, 3, 8], [1, 2, 3, 4], [0x83, 0x82, 6, 0x80])]
        self.assertEqual([[c["kept"][k][0] for k in c["used"]] for c in M.match_candidates(rows, [4, 8], frozenset(), 8)],
                         [[0, 3]])
        self.assertEqual(M.match_candidates(rows, [4, 3], frozenset(), 8), [])

    def test_the_bucket_rule_is_not_applied_without_every_bucket(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        # Geometry 1 has no bucket: the rule cannot be applied, so the bucket-1 geometry stays a candidate.
        rows = [item(0x1000, "a.obj", [2, 7], [1, 2], [1, None])]
        status, out, _ = self.guide(src, rows, {"a.obj": [group(2), group(7)]})
        self.assertEqual(status, 0, out)
        self.assertIn("model 0 geometry [0] (bucket rule not applied", out)

    def test_no_bucket_rule_keeps_every_geometry(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        rows = [item(0x1000, "a.obj", [2], [1], [2])]
        status, out, _ = self.guide(src, rows, {"a.obj": [group(2)]})
        self.assertEqual(status, 1, out)
        self.assertIn("0 candidates", out)
        status, out, _ = self.guide(src, rows, {"a.obj": [group(2)]}, ["--no-bucket-rule"])
        self.assertEqual(status, 0, out)
        self.assertIn("model 0 geometry [0] | outside 0 of 1", out)

    def test_bucket_kept(self):
        self.assertEqual([M.bucket_kept(b) for b in (0, 1, 2, 3, 4, 5, 6, None)],
                         [True, False, False, True, False, False, False, None])
        # Bit 7 does not change the choice.
        self.assertEqual([M.bucket_kept(b) for b in (0x80, 0x81, 0x82, 0x83)], [True, False, False, True])

    def test_skip_shaders_still_applies_on_top_of_the_bucket_rule(self):
        rows = [item(0x1000, "a.obj", [2, 2], [0xAAAA1111, 0xBBBB2222], [0, 0])]
        candidates = M.match_candidates(rows, [2], frozenset([0xAAAA1111]), 8)
        self.assertEqual([[c["kept"][k][0] for k in c["used"]] for c in candidates], [[1]])

    def test_the_instance_transform_decides(self):
        src, _, _ = build(LEAVES, bounds=(0.5, 0.0, 0.0, 1.5, 1.0, 0.0))
        descriptor = os.path.join(self.dir, "instance208.bin")
        with open(descriptor, "wb") as f:
            f.write(bytes(0x50) + struct.pack("<12f", *[v for row in SHIFTED for v in row]) + bytes(60))
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--descriptor", descriptor])
        self.assertEqual(status, 0, out)
        status, out, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]})
        self.assertEqual(status, 1, out)
        self.assertIn("rejected", out)
        self.assertIn("without the instance transform", err)

    def test_geometries_are_matched_inside_one_model(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        row = item(0x1000, "a.obj", [5, 2])
        row["MeshModels"] = [0, 1]               # two models in the primary LOD, one geometry each
        row["MeshGeoms"] = [0, 0]
        status, out, _ = self.guide(src, [row], {"a.obj": [group(5), group(2)]})
        self.assertEqual(status, 0, out)
        self.assertIn("model 1 geometry [0]", out)

    def test_a_manifest_without_per_geometry_counts_is_an_input_error(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        row = item(0x1000, "a.obj", [2])
        del row["MeshTris"]
        status, _, err = self.guide(src, [row], {"a.obj": [group(2)]})
        self.assertEqual(status, 2)
        self.assertIn("MeshTris", err)

    def test_a_manifest_that_lists_more_geometries_than_its_obj_holds_is_an_error_not_a_traceback(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        row = item(0x1000, "a.obj", [5, 2])          # the signature fits the second geometry only
        row["MeshTris"] = [5, 2]
        status, out, _ = self.guide(src, [row], {"a.obj": [group(5)]})   # ... but the OBJ has one group
        self.assertEqual(status, 1, out)
        self.assertIn("the manifest lists", out)
        self.assertIn("scored candidates are consistent", out)

    def test_a_manifest_with_no_models_says_what_is_wrong(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        status, _, err = self.guide(src, [], {"a.obj": [group(2)]})
        self.assertEqual(status, 2)
        self.assertIn("found no model in its window", err)

    def test_json_over_the_dehydrated_input_is_refused_before_it_is_read(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        blas = os.path.join(self.dir, "blas.bin")
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--json", blas])
        self.assertEqual(status, 2)
        self.assertIn("--json would overwrite", err)
        with open(blas, "rb") as f:
            self.assertEqual(f.read(), src)

    def test_json_over_the_manifest_is_refused(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        manifest = os.path.join(self.dir, "manifest.json")
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--json", manifest])
        self.assertEqual(status, 2)
        self.assertIn("--json would overwrite", err)
        with open(manifest, encoding="utf-8") as f:
            self.assertEqual(len(json.load(f)["Items"]), 1)

    def test_json_over_the_descriptor_is_refused(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        descriptor = os.path.join(self.dir, "instance208.bin")
        data = bytes(0x50) + struct.pack("<12f", *[v for row in SHIFTED for v in row]) + bytes(60)
        with open(descriptor, "wb") as f:
            f.write(data)
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--descriptor", descriptor, "--json", descriptor])
        self.assertEqual(status, 2)
        self.assertIn("--json would overwrite", err)
        with open(descriptor, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_json_over_the_write_export_is_refused(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        out_dir = os.path.join(self.dir, "out")
        os.makedirs(out_dir)
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--write", out_dir, "--json", os.path.join(out_dir, "export.json")])
        self.assertEqual(status, 2)
        self.assertIn("--write's export.json", err)

    def test_json_over_a_write_vertex_buffer_is_refused(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        out_dir = os.path.join(self.dir, "out")
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--write", out_dir, "--json", os.path.join(out_dir, "g0_f3.bin")])
        self.assertEqual(status, 2)
        self.assertIn("--write's g0_f3.bin", err)
        # Refused before the export runs: no buffer and no export.json were written.
        self.assertFalse(os.path.exists(os.path.join(out_dir, "export.json")))

    def test_json_over_a_write_index_buffer_is_refused(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        out_dir = os.path.join(self.dir, "out")
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--write", out_dir, "--json", os.path.join(out_dir, "g0_tri.bin")])
        self.assertEqual(status, 2)
        self.assertIn("--write's g0_tri.bin", err)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "export.json")))

    def test_json_over_a_later_geometry_buffer_is_refused(self):
        # Two matched geometries: the report must not land on the second one's buffers either.
        src, _, _ = build([(0, 0x81, 0, 1, 0x09), (1, 0x81, 0, 1, 0x09)], bounds=BOUNDS)
        out_dir = os.path.join(self.dir, "out")
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2, 2])], {"a.obj": [group(2), group(2)]},
                                    ["--write", out_dir, "--json", os.path.join(out_dir, "g1_tri.bin")])
        self.assertEqual(status, 2)
        self.assertIn("--write's g1_tri.bin", err)
        self.assertFalse(os.path.exists(os.path.join(out_dir, "export.json")))

    def test_json_beside_the_write_buffers_still_works(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        out_dir = os.path.join(self.dir, "out")
        report = os.path.join(self.dir, "report.json")
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--write", out_dir, "--json", report])
        self.assertEqual(status, 0, out)
        with open(os.path.join(out_dir, "g0_f3.bin"), "rb") as f:
            self.assertEqual(len(f.read()), 4 * 12)
        with open(report, encoding="utf-8") as f:
            self.assertTrue(json.load(f)["consistent"])

    def test_json_over_an_unwritable_target_is_one_error_line(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        nested = os.path.join(self.dir, "missing_dir", "report.json")
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [5, 2, 3])],
                                    {"a.obj": [group(5), group(2), group(3)]}, ["--json", nested])
        self.assertEqual(status, 2)
        self.assertIn("cannot write", err)
        self.assertNotIn("Traceback", err)

    def test_json_to_its_own_path_still_writes_the_report(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        report = os.path.join(self.dir, "report.json")
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [5, 2, 3])],
                                    {"a.obj": [group(5), group(2), group(3)]}, ["--json", report])
        self.assertEqual(status, 0, out)
        with open(report, encoding="utf-8") as f:
            body = json.load(f)
        self.assertTrue(body["consistent"])
        self.assertEqual(body["matched"]["geometry"], [1])

    def test_write_produces_the_buffers_bvh_rehydrate_reads(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        out = os.path.join(self.dir, "written")
        status, text, _ = self.guide(src, [item(0x1000, "a.obj", [5, 2, 3])],
                                    {"a.obj": [group(5), group(2), group(3)]}, ["--write", out])
        self.assertEqual(status, 0, text)
        with open(os.path.join(out, "export.json"), encoding="utf-8") as f:
            rows = json.load(f)
        self.assertEqual(len(rows), 1)                        # only the geometry the BLAS uses is written
        row = rows[0]
        self.assertEqual((row["drawable_geometry"], row["vertices"], row["triangles"]), (1, 4, 2))
        self.assertIsNone(row["transform"])
        with open(os.path.join(out, row["vb"]), "rb") as f:
            vb = f.read()
        with open(os.path.join(out, row["ib"]), "rb") as f:
            ib = f.read()
        self.assertEqual((len(vb), len(ib)), (4 * 12, 2 * 12))
        geometry = R.Geometry(vb, row["stride"], ib, False)
        self.assertEqual([geometry.triangle(0), geometry.triangle(1)], list(QUAD_FACES))

    def test_write_stores_nothing_when_no_candidate_is_consistent(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        moved = [(x + 0.5, y, z) for x, y, z in QUAD]
        out = os.path.join(self.dir, "nothing")
        status, text, _ = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [(moved, QUAD_FACES)]},
                                     ["--write", out])
        self.assertEqual(status, 1, text)
        self.assertIn("nothing written", text)
        self.assertFalse(os.path.exists(out))

    def test_the_shader_rule_is_not_applied_without_every_hash(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        status, out, _ = self.guide(src, [item(0x1000, "a.obj", [2, 7])], {"a.obj": [group(2), group(7)]},
                                    ["--skip-shaders", "12345678"])
        self.assertEqual(status, 0, out)
        self.assertIn("shader rule not applied", out)

    def test_json_in_a_missing_directory_is_one_error_line(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        nested = os.path.join(self.dir, "nodir", "r.json")
        status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                    ["--json", nested])
        self.assertEqual(status, 2, err)
        self.assertIn("cannot write", err)
        self.assertNotIn("Traceback", err)
        self.assertFalse(os.path.exists(nested))

    def test_readonly_json_is_one_error_line(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        ro = os.path.join(self.dir, "ro.json")
        with open(ro, "w", encoding="utf-8") as f:
            f.write("{}")
        os.chmod(ro, 0o444)
        try:
            try:
                with open(ro, "a", encoding="utf-8"):
                    pass
                self.skipTest("the file is still writable (running as root, or Windows)")
            except OSError:
                pass
            status, _, err = self.guide(src, [item(0x1000, "a.obj", [2])], {"a.obj": [group(2)]},
                                        ["--json", ro])
            self.assertEqual(status, 2, err)
            self.assertIn("cannot write", err)
            self.assertNotIn("Traceback", err)
        finally:
            os.chmod(ro, 0o644)


class TestPieces(unittest.TestCase):
    def test_leaf_signature_counts_a_paired_leaf_twice(self):
        self.assertEqual(M.leaf_signature(build([(0, 0x81, 0, 1, 0x09)], bounds=BOUNDS)[0]), [2])
        self.assertEqual(M.leaf_signature(build([(0, 0x01, 0, None, 0x09)], bounds=BOUNDS)[0]), [1])
        self.assertEqual(M.leaf_signature(build([(0, 0x01, 0, None, 0x09), (1, 0x01, 0, None, 0x09)], bounds=BOUNDS)[0]),
                         [1, 1])

    def test_leaf_signature_applies_the_header_offset(self):
        src, _, _ = build(LEAVES, bounds=BOUNDS)
        self.assertEqual(M.leaf_signature(bytes(0x100) + src, 0x100), [2])

    def test_embeddings_are_ordered_by_default(self):
        self.assertEqual(M.embeddings([2, 3], [2, 3, 3], 8), [[0, 1], [0, 2]])
        self.assertEqual(M.embeddings([3, 2], [2, 3, 3], 8), [])
        self.assertEqual(M.embeddings([3, 2], [2, 3, 3], 8, ordered=False), [[1, 0], [2, 0]])
        self.assertEqual(len(M.embeddings([1], [1] * 20, 5)), 5)          # the cap holds

    def test_embeddings_prune_impossible_suffixes(self):
        # Count triangle-count comparisons so the regression doesn't depend on machine speed.
        class Count(int):
            __hash__ = int.__hash__

            def __eq__(self, other):
                nonlocal comparisons
                comparisons += 1
                return int.__eq__(self, other)

            def __ne__(self, other):
                return not self == other

        cases = [
            (True, [1] * 8 + [2], [2] + [1] * 16, []),  # the only 2 comes too early
            (False, [1] * 6 + [2, 2], [2] + [1] * 8, []),  # two distinct 2s are needed
            (True, [1] * 8 + [2], [1] * 8 + [2] + [1] * 8, [list(range(9))]),
        ]
        for ordered, signature, counts, expected in cases:
            with self.subTest(ordered=ordered, counts=counts):
                comparisons = 0
                self.assertEqual(M.embeddings(signature, list(map(Count, counts)), 2, ordered), expected)
                self.assertLess(comparisons, 4096)

    def test_embeddings_keep_first_matches_in_index_order(self):
        counts = [7] * 8 + [1, 2, 1, 2, 1]
        signature = [1, 2, 1]
        for ordered in (True, False):
            choices = itertools.combinations if ordered else itertools.permutations
            expected = [list(indices) for indices in choices(range(len(counts)), len(signature))
                        if [counts[i] for i in indices] == signature]
            for cap in (0, 1, 3, len(expected) + 1):
                with self.subTest(ordered=ordered, cap=cap):
                    self.assertEqual(M.embeddings(signature, counts, cap, ordered), expected[:cap])
            self.assertEqual(M.embeddings([], counts, 1, ordered), [[]])
            self.assertEqual(M.embeddings([], counts, 0, ordered), [])

    def test_read_groups_rebases_faces_per_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "a.obj")
            write_obj(path, [(QUAD, QUAD_FACES), (QUAD, [(0, 1, 2)])])
            groups = M.read_groups(path)
            self.assertEqual([len(positions) for positions, _ in groups], [4, 4])
            self.assertEqual(groups[0][1], QUAD_FACES)
            self.assertEqual(groups[1][1], [(0, 1, 2)])      # second group's faces start at its own first vertex
            self.assertEqual(groups[1][0], QUAD)

    def test_read_transform_reads_the_matrix_at_0x50(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "instance208.bin")
            with open(path, "wb") as f:
                f.write(bytes(0x50) + struct.pack("<12f", *[v for row in SHIFTED for v in row]) + bytes(60))
            self.assertEqual(M.read_transform(path), SHIFTED)

    def test_shader_hashes_parse_from_a_list(self):
        self.assertEqual(M.parse_shader_hashes("0xAA, bb"), frozenset({0xAA, 0xBB}))
        self.assertEqual(M.parse_shader_hashes(""), frozenset())
        self.assertEqual(M.parse_shader_hashes(None), frozenset())

    def test_json_directory_is_refused_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                rc = M.main([os.path.join(directory, "missing.bvh"),
                             os.path.join(directory, "missing.json"), "--json", directory])
            self.assertEqual(rc, 2)
            self.assertIn("is a directory", err.getvalue())


if __name__ == "__main__":
    unittest.main()
