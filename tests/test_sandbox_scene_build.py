"""Real sandbox scene builds keep searching until enough usable props are placed.

    python -I tests/test_sandbox_scene_build.py
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import sandbox_scene  # noqa: E402
import world_names  # noqa: E402


TRIANGLE = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"


def entity(name, x, y=0, **extra):
    return dict(archetype=world_names.joaat(name), position=[x, y, 0],
                rotation=[0, 0, 0, 1], lod_level=0, scale_xy=1, scale_z=1, **extra)


class SceneBuildTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.models = self.root / "models"
        self.models.mkdir()
        self.items = []
        self.add_model("synthetic_prop")

    def add_model(self, name, obj=TRIANGLE):
        filename = name + ".obj"
        (self.models / filename).write_text(obj, encoding="utf-8", newline="\n")
        self.items.append({"Name": name, "File": filename, "MeshShaders": [0]})

    def build_scene(self, rows, max_props, radius=20, center=(0, 0)):
        (self.models / "manifest.json").write_text(json.dumps({"Items": self.items}), encoding="utf-8", newline="\n")
        placements = self.root / "entities.jsonl"
        placements.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8", newline="\n")
        scene = self.root / "scene.bin"
        result = subprocess.run(
            [sys.executable, "-I", sandbox_scene.__file__, "--entities", str(placements),
             "--models", str(self.models), *(["--center", *map(str, center)] if center is not None else []),
             "--radius", str(radius),
             "--max-props", str(max_props), "--out", str(scene)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stderr, "")
        data = scene.read_bytes()
        self.assertEqual(data[:4], b"LSB1")
        version, textures, meshes, draws = struct.unpack_from("<4I", data, 4)
        self.assertEqual((version, textures), (1, 0))
        offset = 32  # LSB1 header, including the scene center.
        for _ in range(meshes):
            vertices, indices = struct.unpack_from("<II", data, offset)
            self.assertEqual((vertices, indices), (3, 3))
            offset += 8 + vertices * 32 + indices * 4
        positions = []
        for _ in range(draws):
            mesh, texture, *transform = struct.unpack_from("<Ii12f", data, offset)
            self.assertLess(mesh, meshes)
            self.assertEqual(texture, -1)
            positions.append(tuple(transform[-3:]))
            offset += 56
        self.assertEqual(offset, len(data))
        return meshes, positions, result.stdout

    def test_nearer_unexported_models_do_not_hide_available_prop(self):
        rows = [entity(f"synthetic_missing_{i}", i + 1) for i in range(4)]
        rows.append(entity("synthetic_prop", 5))
        meshes, positions, report = self.build_scene(rows, max_props=1)
        self.assertEqual((meshes, positions), (1, [(5.0, 0.0, 0.0)]))
        self.assertIn("placed=1 unresolved=4", report)

    def test_name_key_mismatches_do_not_consume_placement_limit(self):
        wrong_key = f"{world_names.hash64('synthetic_other_prop'):016x}"
        rows = [entity("synthetic_prop", i + 1, key=wrong_key) for i in range(4)]
        rows.append(entity("synthetic_prop", 5, key=f"{world_names.hash64('synthetic_prop'):016x}"))
        meshes, positions, report = self.build_scene(rows, max_props=1)
        self.assertEqual((meshes, positions), (1, [(5.0, 0.0, 0.0)]))
        self.assertIn("placed=1 unresolved=4", report)

    def test_auto_center_ignores_cells_with_only_mismatched_keys(self):
        wrong_key = f"{world_names.hash64('synthetic_other_prop'):016x}"
        rows = [entity("synthetic_prop", 25, 25, key=wrong_key) for _ in range(2)]
        rows.append(entity("synthetic_prop", 225, 25))
        meshes, positions, _ = self.build_scene(rows, max_props=1, center=None)
        self.assertEqual((meshes, positions), (1, [(225.0, 25.0, 0.0)]))

    def test_models_without_usable_geometry_do_not_consume_limit(self):
        self.add_model("synthetic_empty", "o empty\n")
        rows = [entity("synthetic_empty", i + 1) for i in range(4)]
        rows.append(entity("synthetic_prop", 5))
        meshes, positions, report = self.build_scene(rows, max_props=1)
        self.assertEqual((meshes, positions), (1, [(5.0, 0.0, 0.0)]))
        self.assertIn("placed=1 unresolved=4", report)

    def test_nearest_order_radius_and_successful_limit_are_preserved(self):
        rows = [entity("synthetic_prop", 0, 21), entity("synthetic_prop", 9),
                entity("synthetic_prop", 3), entity("synthetic_prop", 0, 2),
                entity("synthetic_prop", 0, 3), entity("synthetic_prop", 0, 20)]
        expected = [(0.0, 2.0, 0.0), (3.0, 0.0, 0.0), (0.0, 3.0, 0.0),
                    (9.0, 0.0, 0.0), (0.0, 20.0, 0.0)]
        for limit in (2, 10):
            with self.subTest(limit=limit):
                meshes, positions, _ = self.build_scene(rows, max_props=limit)
                self.assertEqual(meshes, 1)  # Repeated props share the mesh.
                self.assertEqual(positions, expected[:limit])

    def test_no_available_model_inside_radius_still_writes_empty_scene(self):
        rows = [entity("synthetic_missing", 1), entity("synthetic_prop", 21)]
        meshes, positions, report = self.build_scene(rows, max_props=1)
        self.assertEqual((meshes, positions), (0, []))
        self.assertIn("placed=0 unresolved=1", report)

    def test_zero_limit_still_places_nothing(self):
        meshes, positions, report = self.build_scene([entity("synthetic_prop", 1)], max_props=0)
        self.assertEqual((meshes, positions), (0, []))
        self.assertIn("entities=0 placed=0", report)


if __name__ == "__main__":
    unittest.main()
