"""Sparse mesh attribute and index accessors, using hand-built glTF 2.0 triangles."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

TESTS = Path(__file__).resolve().parent
TOOLS = TESTS.parent / "tools"
sys.path[:0] = [str(TESTS), str(TOOLS)]
from test_ped_assemble_malformed_glb import container, document
import glb_skinning
import ped_assemble


def triangle(index_type=5121, dense=True):
    views, data = [], bytearray()

    def view(raw, **fields):
        data.extend(bytes(-len(data) % 4))
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=len(raw), **fields))
        data.extend(raw)
        return len(views) - 1

    # The first index/element in each view is padding, skipped by byteOffset.
    fmt = {5121: "B", 5123: "H", 5125: "I"}[index_type]
    indices = view(struct.pack("<3" + fmt, 0, 1, 2))
    sparse_indices = dict(bufferView=indices, byteOffset=struct.calcsize("<" + fmt), componentType=index_type)

    def sparse_accessor(raw, ctype, shape, offset=0):
        replacements = view(raw)
        return dict(componentType=ctype, type=shape, count=3, sparse=dict(
            count=2, indices=sparse_indices.copy(), values=dict(bufferView=replacements, byteOffset=offset)))

    positions = sparse_accessor(struct.pack("<7f", 77, 1, 0, 0, 0, 2, 0), 5126, "VEC3", 4)
    positions.update(min=[0, 0, 0], max=[1, 2, 0])
    if dense:
        base = struct.pack("<12f", 77, -1, 0, 0, 77, 18, 0, 0, 77, 0, 19, 0)
        positions.update(bufferView=view(base, byteStride=16), byteOffset=4, min=[-1, 0, 0])
    colors = sparse_accessor(bytes([255, 0, 128, 255, 0, 255, 0, 255]), 5121, "VEC4")
    colors["normalized"] = True
    triangles = sparse_accessor(struct.pack("<2H", 2, 1), 5123, "SCALAR")
    doc = dict(asset=dict(version="2.0"), scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0)],
               meshes=[dict(primitives=[dict(attributes={"POSITION": 0, "COLOR_0": 1}, indices=2)])],
               buffers=[dict(byteLength=len(data))], bufferViews=views, accessors=[positions, colors, triangles])
    return doc, data


class SparseGlbTests(unittest.TestCase):
    def test_assemble_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            src, dst = Path(directory) / "in.glb", Path(directory) / "out.glb"
            for index_type in (5121, 5123, 5125):
                for dense in (False, True):
                    with self.subTest(index_type=index_type, dense=dense):
                        doc, data = triangle(index_type, dense)
                        src.write_bytes(container(json.dumps(doc).encode(), bytes(data)))
                        run = subprocess.run([sys.executable, "-I", str(TOOLS / "ped_assemble.py"), "assemble",
                                              str(src), "-o", str(dst), "--no-standing-offset"],
                                             capture_output=True, text=True, timeout=30)
                        self.assertEqual(run.returncode, 0, run.stderr)
                        # Inspect the emitted dense accessors with a separate reader.
                        out, binary, _ = glb_skinning.parse_glb(dst.read_bytes())
                        reader = glb_skinning._Reader(out, binary)
                        prim = out["meshes"][0]["primitives"][0]
                        pos, _ = reader.read(prim["attributes"]["POSITION"], "positions")
                        self.assertEqual(pos, [(-1 if dense else 0, 0, 0), (1, 0, 0), (0, 2, 0)])
                        indices, _ = reader.read(prim["indices"], "indices")
                        self.assertEqual(indices, [(0,), (2,), (1,)])
                        colors, _ = reader.read(prim["attributes"]["COLOR_0"], "colors")
                        self.assertEqual(colors[0], (0, 0, 0, 0))
                        self.assertEqual(colors[2], (0, 1, 0, 1))
                        self.assertEqual((colors[1][0], colors[1][1], colors[1][3]), (1, 0, 1))
                        self.assertAlmostEqual(colors[1][2], 128 / 255)

    def test_normalized_skin_weights_use_the_same_conversion_for_base_and_replacements(self):
        doc, data = triangle()
        views = doc["bufferViews"]
        weights = doc["accessors"][1]
        value_view = views[weights["sparse"]["values"]["bufferView"]]
        at = value_view["byteOffset"]
        data[at:at + 8] = bytes((128, 127, 0, 0, 0, 255, 0, 0))
        data.extend(bytes(-len(data) % 4))
        weights["bufferView"] = len(views)
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=12))
        data.extend(bytes((255, 0, 0, 0)) * 3)
        joints = dict(bufferView=len(views), componentType=5121, type="VEC4", count=3)
        views.append(dict(buffer=0, byteOffset=len(data), byteLength=12))
        data.extend(bytes((0, 1, 0, 0)) * 3)
        doc["accessors"].append(joints)
        doc["buffers"][0]["byteLength"] = len(data)
        attrs = doc["meshes"][0]["primitives"][0]["attributes"]
        attrs["WEIGHTS_0"] = attrs.pop("COLOR_0")
        attrs["JOINTS_0"] = 3
        doc["nodes"][0]["skin"] = 0
        doc["nodes"].extend([dict(name="Root", children=[2]), dict(name="Bone")])
        doc["scenes"][0]["nodes"].append(1)
        doc["skins"] = [dict(joints=[1, 2])]
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "skin.glb"
            src.write_bytes(container(json.dumps(doc).encode(), bytes(data)))
            part, = ped_assemble.read_glb_parts(src)
        self.assertEqual(part.weights_0, [(1, 0, 0, 0), (128 / 255, 127 / 255, 0, 0), (0, 1, 0, 0)])
        self.assertEqual(part.joint_names, ["Root", "Bone"])

    def test_implicit_rows_count_each_attribute_and_mesh_instance(self):
        doc, data = triangle(dense=True)
        del doc["accessors"][1]["sparse"]  # COLOR_0 has neither a base view nor overrides.
        doc["nodes"].append(dict(mesh=0))
        doc["scenes"][0]["nodes"].append(1)
        doc["meshes"].append(doc["meshes"][0])  # No node imports this mesh.
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "two_parts.glb"
            src.write_bytes(container(json.dumps(doc).encode(), bytes(data)))
            # Colors and indices each expand 3 rows per instance. Positions have a dense base.
            with patch.object(ped_assemble, "MAX_IMPLICIT_ACCESSOR_ROWS", 12):
                parts = ped_assemble.read_glb_parts(src)
                self.assertEqual(len(parts), 2)
                for part in parts:
                    self.assertEqual(part.colors, [(0, 0, 0, 0)] * 3)
            with patch.object(ped_assemble, "MAX_IMPLICIT_ACCESSOR_ROWS", 11):
                with self.assertRaisesRegex(ped_assemble.PedAssembleError, "per-file limit"):
                    ped_assemble.read_glb_parts(src)
            src.write_bytes(container(json.dumps(document()).encode()))
            with patch.object(ped_assemble, "MAX_IMPLICIT_ACCESSOR_ROWS", 0):
                self.assertEqual(len(ped_assemble.read_glb_parts(src)), 1)

    def test_sparse_integer_fields_accept_whole_json_numbers(self):
        doc, data = triangle()
        for acc in doc["accessors"]:
            sparse = acc["sparse"]
            sparse["count"] = float(sparse["count"])
            sparse["indices"]["componentType"] = float(sparse["indices"]["componentType"])
            for section in (sparse["indices"], sparse["values"]):
                section["bufferView"] = float(section["bufferView"])
                section["byteOffset"] = float(section["byteOffset"])
        for view in doc["bufferViews"]:
            view["buffer"] = 0.0
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "whole.glb"
            src.write_bytes(container(json.dumps(doc).encode(), bytes(data)))
            part, = ped_assemble.read_glb_parts(src)
        self.assertEqual(part.positions[-1], (0, 2, 0))
        self.assertEqual(part.triangles, [(0, 2, 1)])

    def test_required_draco_geometry_cannot_be_read_as_zero_rows(self):
        doc, data = triangle()
        extension = "KHR_draco_mesh_compression"
        doc["extensionsUsed"] = [extension]
        prim = doc["meshes"][0]["primitives"][0]
        prim["extensions"] = {extension: {"bufferView": 0, "attributes": {"POSITION": 0}}}
        del doc["accessors"][1]["sparse"]  # Uncompressed fallback colors are zero.
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "draco.glb"
            src.write_bytes(container(json.dumps(doc).encode(), bytes(data)))
            part, = ped_assemble.read_glb_parts(src)
            self.assertEqual(part.positions[-1], (0, 2, 0))
            self.assertEqual(part.colors, [(0, 0, 0, 0)] * 3)

            doc["extensionsRequired"] = [extension]
            for key in ("bufferView", "byteOffset", "sparse"):
                del doc["accessors"][0][key]
            del doc["accessors"][2]["sparse"]
            src.write_bytes(container(json.dumps(doc).encode(), bytes(data)))
            with self.assertRaisesRegex(ped_assemble.PedAssembleError, extension):
                ped_assemble.read_glb_parts(src)

    def assert_bad_position(self, src, doc, data):
        src.write_bytes(container(json.dumps(doc).encode(), bytes(data)))
        with self.assertRaisesRegex(ped_assemble.PedAssembleError, r"accessors\[0\]"):
            ped_assemble.read_glb_parts(src)

    def test_malformed_sparse_layouts(self):
        doc, data = triangle()
        acc = doc["accessors"][0]
        sparse = acc["sparse"]
        base_view = doc["bufferViews"][acc["bufferView"]]
        indices_view = doc["bufferViews"][sparse["indices"]["bufferView"]]
        values_view = doc["bufferViews"][sparse["values"]["bufferView"]]
        cases = [
            (sparse, "count", 4), (sparse, "count", 0), (sparse, "count", 1.5), (sparse, "count", True),
            (sparse["indices"], "componentType", 5120), (sparse["indices"], "componentType", 5121.5),
            (sparse["indices"], "bufferView", False), (sparse["values"], "bufferView", 1.5),
            (sparse["values"], "byteOffset", 1), (indices_view, "byteLength", 2),
            (values_view, "byteLength", 27), (base_view, "byteLength", 47), (values_view, "buffer", 1),
            (indices_view, "byteStride", 4), (values_view, "target", 34962), (values_view, "buffer", False),
        ]
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "bad.glb"
            for section, field, value in cases:
                with self.subTest(field=field, value=value):
                    original = dict(section)
                    section[field] = value
                    try:
                        self.assert_bad_position(src, doc, data)
                    finally:
                        section.clear()
                        section.update(original)

    def test_sparse_indices_must_increase_and_fit(self):
        doc, data = triangle()
        indices = doc["accessors"][0]["sparse"]["indices"]
        at = doc["bufferViews"][indices["bufferView"]]["byteOffset"] + indices["byteOffset"]
        with tempfile.TemporaryDirectory() as directory:
            src = Path(directory) / "bad.glb"
            for values in ((1, 1), (2, 1), (1, 3)):
                with self.subTest(values=values):
                    data[at:at + 2] = bytes(values)
                    self.assert_bad_position(src, doc, data)


if __name__ == "__main__":
    unittest.main()
