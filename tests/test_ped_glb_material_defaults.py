"""GLB material defaults through the real writer, importer, and assembly CLI.

Synthetic fixtures only. Literal expectations follow the pinned Khronos schema:
https://github.com/KhronosGroup/glTF/blob/5b00b913b74c8f0260c64578b98a09ef2dde0b41/specification/2.0/schema/material.pbrMetallicRoughness.schema.json
The same revision's Specification.adoc#default-material applies these defaults
when a primitive does not reference a material.
"""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ped_assemble as pa

DEFAULTS = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 1.0, "roughnessFactor": 1.0}
CLAY = {"baseColorFactor": [0.8, 0.8, 0.8, 1.0], "metallicFactor": 0.0, "roughnessFactor": 0.85}
OMISSIONS = ("baseColorFactor", "metallicFactor", "roughnessFactor", "pbrMetallicRoughness",
             "empty_pbr", "empty_material", "no_reference", "no_materials")


def split_glb(data):
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + length])
    offset = 20 + length
    length, kind = struct.unpack_from("<II", data, offset)
    assert kind == pa.CHUNK_TYPE_BIN
    return doc, data[offset + 8:offset + 8 + length]


def container(doc, blob):
    data = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    data += b" " * (-len(data) % 4)
    return b"".join((
        struct.pack("<4sII", pa.GLB_MAGIC, 2, 28 + len(data) + len(blob)),
        struct.pack("<II", len(data), pa.CHUNK_TYPE_JSON), data,
        struct.pack("<II", len(blob), pa.CHUNK_TYPE_BIN), blob,
    ))


def fixture():
    part = pa.MeshPart(
        "surface", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)],
        coord_space="y-up", material=pa.MaterialSpec(
            name="surface_mat", base_color_factor=(1.0, 1.0, 1.0, 1.0),
            metallic_factor=1.0, roughness_factor=1.0),
    )
    data, _ = pa.build_assembled_glb([part], apply_standing=False)
    return split_glb(data)


def omit(doc, field):
    changed = copy.deepcopy(doc)
    material = changed["materials"][0]
    if field in DEFAULTS:
        del material["pbrMetallicRoughness"][field]
    elif field == "pbrMetallicRoughness":
        del material[field]
    elif field == "empty_pbr":
        material["pbrMetallicRoughness"] = {}
    elif field == "empty_material":
        changed["materials"][0] = {}
    else:
        del changed["meshes"][0]["primitives"][0]["material"]
        if field == "no_materials":
            del changed["materials"]
        else:
            material["pbrMetallicRoughness"] = copy.deepcopy(CLAY)
    return changed


class GlbMaterialDefaultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def read(self, data):
        path = self.root / "source.glb"
        path.write_bytes(data)
        parts = pa.read_glb_parts(str(path))
        self.assertEqual(path.read_bytes(), data)
        return parts

    def run_cli(self, path):
        before = path.read_bytes()
        output = self.root / "out.glb"
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "ped_assemble.py"),
             "assemble", str(path), "-o", str(output), "--no-cull", "--no-standing-offset"],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(path.read_bytes(), before)
        return output.read_bytes()

    def assert_factors(self, material, expected):
        self.assertIsNotNone(material)
        self.assertEqual(list(material.base_color_factor), expected["baseColorFactor"])
        self.assertEqual(material.metallic_factor, expected["metallicFactor"])
        self.assertEqual(material.roughness_factor, expected["roughnessFactor"])

    def test_omitted_fields_use_literal_gltf_defaults(self):
        doc, blob = fixture()
        for field in OMISSIONS:
            with self.subTest(field=field):
                part, = self.read(container(omit(doc, field), blob))
                self.assert_factors(part.material, DEFAULTS)
                self.assertEqual(part.positions, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
                self.assertEqual(part.triangles, [(0, 1, 2)])

    def test_cli_omissions_match_explicit_one_control(self):
        doc, blob = fixture()
        path = self.root / "source.glb"
        path.write_bytes(container(doc, blob))
        control = self.run_cli(path)
        self.assertEqual(split_glb(control)[0]["materials"][0]["pbrMetallicRoughness"], DEFAULTS)
        for field in OMISSIONS:
            with self.subTest(field=field):
                path.write_bytes(container(omit(doc, field), blob))
                self.assertEqual(self.run_cli(path), control)

    def test_authored_factors_and_other_fields_are_preserved(self):
        doc, blob = fixture()
        authored = {"baseColorFactor": [0.25, 0.5, 0.75, 0.625], "metallicFactor": 0.0, "roughnessFactor": 0.0}
        material = doc["materials"][0]
        material.update(alphaMode="MASK", alphaCutoff=0.375, doubleSided=True)
        material["pbrMetallicRoughness"] = authored
        for field in (None, *DEFAULTS):
            with self.subTest(field=field):
                changed = doc if field is None else omit(doc, field)
                expected = dict(authored)
                if field is not None:
                    expected[field] = DEFAULTS[field]
                path = self.root / "source.glb"
                path.write_bytes(container(changed, blob))
                result = split_glb(self.run_cli(path))[0]["materials"][0]
                self.assertEqual(result["pbrMetallicRoughness"], expected)
                self.assertEqual((result["alphaMode"], result["alphaCutoff"], result["doubleSided"]),
                                 ("MASK", 0.375, True))

    def test_omitted_materials_keep_per_primitive_names_and_order(self):
        doc, blob = fixture()
        primitive = doc["meshes"][0]["primitives"][0]
        del primitive["material"]
        doc["meshes"][0]["primitives"].append(copy.deepcopy(primitive))
        parts = self.read(container(doc, blob))
        self.assertEqual([p.name for p in parts], ["surface_p0", "surface_p1"])
        for part in parts:
            self.assert_factors(part.material, DEFAULTS)
            self.assertEqual(part.material.name, part.name + "_mat")
        result = split_glb(self.run_cli(self.root / "source.glb"))[0]
        self.assertEqual([m["name"] for m in result["materials"]], ["surface_p0_mat", "surface_p1_mat"])
        self.assertEqual([m["pbrMetallicRoughness"] for m in result["materials"]], [DEFAULTS, DEFAULTS])

    def test_json_obj_and_materialspec_defaults_remain_clay(self):
        self.assert_factors(pa.MaterialSpec(), CLAY)
        raw = {"name": "surface", "coord_space": "y-up", "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
               "triangles": [[0, 1, 2]]}
        for material in (None, {}):
            with self.subTest(material=material):
                raw["material"] = material
                path = self.root / "source.json"
                path.write_text(json.dumps(raw), encoding="utf-8", newline="\n")
                result = split_glb(self.run_cli(path))[0]
                self.assertEqual(result["materials"][0]["pbrMetallicRoughness"], CLAY)
        path = self.root / "source.obj"
        path.write_text("o surface\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8", newline="\n")
        result = split_glb(self.run_cli(path))[0]
        self.assertEqual(result["materials"][0]["pbrMetallicRoughness"], CLAY)

    def test_existing_out_of_range_reference_fallback_is_unchanged(self):
        doc, blob = fixture()
        for index in (-1, len(doc["materials"])):
            with self.subTest(index=index):
                doc["meshes"][0]["primitives"][0]["material"] = index
                part, = self.read(container(doc, blob))
                self.assertIsNone(part.material)


if __name__ == "__main__":
    unittest.main()
