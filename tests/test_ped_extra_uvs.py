"""Preserve authored UV sets through the actual skinned producer and ped CLIs.

All geometry and skin streams are synthetic. glTF defines TEXCOORD_n as VEC2
float or normalized unsigned byte/short attributes with consecutive set indices:
https://github.com/KhronosGroup/glTF/blob/5b00b913b74c8f0260c64578b98a09ef2dde0b41/specification/2.0/Specification.adoc#meshes-attributes

    python -I tests/test_ped_extra_uvs.py
"""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tests"))

import glb_writer
import ped_assemble as pa
import skinned_glb
import test_skinned_glb as skin_fixture


POSITIONS = [(0, 0, 0), (1, 0, 0), (0, 1, 0),
             (2, 0, 0), (3, 0, 0), (2, 1, 0)]
UV0 = [(0, 0), (1, 0), (0, 1), (0.25, 0), (0.75, 0), (0.25, 1)]
EXTRA = {
    1: [(i / 8, 0.5 + i / 16) for i in range(6)],
    2: [((10 + 30 * i) / 255, (240 - 20 * i) / 255) for i in range(6)],
    3: [((1000 + 4000 * i) / 65535, (62000 - 5000 * i) / 65535) for i in range(6)],
}


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8", newline="\n")


def encode_glb(doc, blob):
    raw = json.dumps(doc, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 4)
    blob = bytes(blob) + b"\0" * (-len(blob) % 4)
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(raw) + len(blob))
            + struct.pack("<II", len(raw), 0x4E4F534A) + raw
            + struct.pack("<II", len(blob), 0x004E4942) + blob)


def write_geometry(path, extra=True):
    path.write_bytes(glb_writer.build_glb([{
        "positions": POSITIONS, "normals": [(0, 0, 1)] * 6,
        "uvs": UV0, "indices": list(range(6)),
    }]))
    if not extra:
        return
    doc, blob = skinned_glb.read_glb(str(path))
    blob = bytearray(blob)
    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    # Padding, accessor offsets and normalized integer storage exercise the real
    # accessor path rather than assuming tightly packed float attributes.
    for channel, ctype, stride, offset, rows in (
        (1, 5126, 16, 4, [struct.pack("<I2fI", 0x12345678, *uv, 0x87654321) for uv in EXTRA[1]]),
        (2, 5121, 4, 0, [bytes((10 + 30 * i, 240 - 20 * i, 91, 92)) for i in range(6)]),
        (3, 5123, 8, 4, [struct.pack("<I2H", 777, 1000 + 4000 * i, 62000 - 5000 * i)
                         for i in range(6)]),
    ):
        blob.extend(b"\0" * (-len(blob) % 4))
        payload = b"".join(rows)
        view = len(doc["bufferViews"])
        doc["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(payload),
                                   "byteStride": stride, "target": 34962})
        blob.extend(payload)
        index = len(doc["accessors"])
        doc["accessors"].append({"bufferView": view, "byteOffset": offset, "componentType": ctype,
                                 "count": 6, "type": "VEC2", **({"normalized": True} if ctype != 5126 else {})})
        attrs[f"TEXCOORD_{channel}"] = index
    doc["buffers"][0]["byteLength"] = len(blob)
    path.write_bytes(encode_glb(doc, blob))


def attributes(path):
    """Decode actual output bytes independently of ped_assemble's accessor reader."""
    doc, blob = skinned_glb.read_glb(str(path))
    prim = doc["meshes"][0]["primitives"][0]
    values = {}
    for name, index in {**prim["attributes"], "indices": prim["indices"]}.items():
        acc = doc["accessors"][index]
        view = doc["bufferViews"][acc["bufferView"]]
        fmt, scale = {5121: ("B", 255), 5123: ("H", 65535),
                      5125: ("I", None), 5126: ("f", None)}[acc["componentType"]]
        components = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
        stride = view.get("byteStride", struct.calcsize("<" + fmt * components))
        base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        rows = [struct.unpack_from("<" + fmt * components, blob, base + i * stride)
                for i in range(acc["count"])]
        if acc.get("normalized"):
            rows = [tuple(value / scale for value in row) for row in rows]
        values[name] = rows
    return doc, values


def f32_rows(rows):
    return [tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in row) for row in rows]


class ExtraUvsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def cli(self, script, *args):
        result = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / script), *map(str, args)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_json_and_writer_preserve_all_seven_extra_sets(self):
        expected = {channel: [(channel / 8, i / 8) for i in range(6)] for channel in range(1, 8)}
        part = pa.MeshPart("all_sets", POSITIONS, [(0, 1, 2), (3, 4, 5)],
                           coord_space="y-up", uvs=UV0, uvs_extra=expected)
        encoded = pa.part_to_dict(part)
        self.assertEqual(encoded.get("uvs_extra"),
                         {str(key): [list(row) for row in rows] for key, rows in expected.items()})
        source = self.root / "part.json"
        write_json(source, encoded)
        loaded, _ = pa.load_parts_from_path(str(source))
        self.assertEqual(loaded[0].uvs_extra, expected)
        output = self.root / "direct.glb"
        pa.write_assembled_glb(str(output), [part], apply_standing=False)
        _, data = attributes(output)
        self.assertEqual({key for key in data if key.startswith("TEXCOORD_")},
                         {f"TEXCOORD_{key}" for key in range(8)})
        for channel, rows in expected.items():
            self.assertEqual(data[f"TEXCOORD_{channel}"], rows)
        self.assertEqual(data["TEXCOORD_0"], UV0)

    def test_import_reuses_interleaved_and_normalized_accessor_path(self):
        source = self.root / "packed.glb"
        write_geometry(source)
        original = source.read_bytes()
        loaded = pa.read_glb_parts(str(source))[0]
        self.assertEqual(loaded.uvs, UV0)
        self.assertEqual(loaded.uvs_extra, EXTRA)
        output = self.root / "packed-roundtrip.glb"
        self.cli("ped_assemble.py", "assemble", source, "-o", output,
                 "--source-space", "y-up", "--no-cull", "--no-standing-offset")
        _, data = attributes(output)
        for channel, rows in EXTRA.items():
            self.assertEqual(data[f"TEXCOORD_{channel}"], f32_rows(rows))
        self.assertEqual(source.read_bytes(), original)

    def test_skinned_producer_cull_json_and_assemble_keep_remapped_uvs(self):
        streams = [value * 2 for value in skin_fixture._streams()]
        buffer = skin_fixture._skin_buffer(*streams, offsets={6: 0x100, 7: 0x120, 8: 0x140, 9: 0x160},
                                           size=0x180)
        component = Path(skin_fixture._component(str(self.root), buffer=buffer))
        skeleton = Path(skin_fixture._skeleton_dir(str(self.root)))
        model_path = component / "model.json"
        model = json.loads(model_path.read_text(encoding="utf-8"))
        model["vertex_count"] = model["meshes"][0]["vertex_count"] = 6
        write_json(model_path, model)
        write_geometry(component / "model.glb")
        originals = {path: hashlib.sha256(path.read_bytes()).digest()
                     for path in self.root.rglob("*") if path.is_file()}
        produced = self.root / "skinned.glb"
        self.cli("skinned_glb.py", produced, "--skeleton", skeleton / "model.json",
                 "--component", model_path)
        producer_doc, before = attributes(produced)
        for channel, rows in EXTRA.items():
            self.assertEqual(before[f"TEXCOORD_{channel}"], rows)
        reference = self.root / "reference.json"
        write_json(reference, {"name": "reference", "coord_space": "y-up",
                              "positions": POSITIONS[:3], "triangles": [[0, 1, 2]]})
        culled = self.root / "culled.json"
        options = ("--target", produced, "--ref", reference, "--source-space", "y-up",
                   "--no-standing-offset", "--compact-vertices")
        result = self.cli("ped_assemble.py", "cull-overlap", *options, "-o", culled)
        self.assertIn("2 -> 1 tris", result.stdout)
        part = json.loads(culled.read_text(encoding="utf-8"))
        self.assertEqual(part["positions"], [list(row) for row in POSITIONS[3:]])
        self.assertEqual(part["triangles"], [[0, 1, 2]])
        self.assertEqual(part.get("uvs_extra"),
                         {str(key): [list(row) for row in rows[3:]] for key, rows in EXTRA.items()})
        # Supply the source skin's joint order so the normal ped skin-remapping
        # path is exercised as well. This assertion concerns vertex data only.
        joint_names = [producer_doc["nodes"][index]["name"]
                       for index in producer_doc["skins"][0]["joints"]]
        manifest = self.root / "manifest.json"
        write_json(manifest, {"parts": [part], "skeleton": {
            "coord_space": "y-up", "joints": [{"name": name} for name in joint_names],
        }})
        assembled = self.root / "assembled.glb"
        self.cli("ped_assemble.py", "assemble", "--manifest", manifest, "-o", assembled,
                 "--source-space", "y-up", "--no-cull", "--no-standing-offset")
        out_doc, after = attributes(assembled)
        self.assertEqual(len(out_doc["skins"][0]["joints"]), len(joint_names))
        for name in ("POSITION", "NORMAL", "TEXCOORD_0", "TEXCOORD_1", "TEXCOORD_2", "TEXCOORD_3",
                     "JOINTS_0", "JOINTS_1", "WEIGHTS_0", "WEIGHTS_1"):
            self.assertEqual(after[name], f32_rows(before[name][3:]), name)
        self.assertEqual(after["indices"], [(0,), (1,), (2,)])
        direct = self.root / "culled.glb"
        self.cli("ped_assemble.py", "cull-overlap", *options, "-o", direct)
        _, direct_data = attributes(direct)
        self.assertEqual(direct_data, after)
        self.assertTrue(all(hashlib.sha256(path.read_bytes()).digest() == digest
                            for path, digest in originals.items()))

    def test_no_extra_sets_keep_existing_json_and_attributes(self):
        source = self.root / "primary.glb"
        write_geometry(source, extra=False)
        loaded = pa.read_glb_parts(str(source))[0]
        self.assertEqual(loaded.uvs_extra, {})
        self.assertNotIn("uvs_extra", pa.part_to_dict(loaded))
        output = self.root / "primary-out.glb"
        self.cli("ped_assemble.py", "assemble", source, "-o", output,
                 "--source-space", "y-up", "--no-cull", "--no-standing-offset")
        _, before = attributes(source)
        _, after = attributes(output)
        self.assertEqual(before, after)

    def test_sparse_extra_sets_use_the_same_precheck_and_overrides(self):
        source = self.root / "sparse.glb"
        write_geometry(source)
        doc, blob = skinned_glb.read_glb(str(source))
        blob = bytearray(blob)
        attrs = doc["meshes"][0]["primitives"][0]["attributes"]

        def add_view(data):
            blob.extend(b"\0" * (-len(blob) % 4))
            index = len(doc["bufferViews"])
            doc["bufferViews"].append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)})
            blob.extend(data)
            return index

        first = doc["accessors"][attrs["TEXCOORD_1"]]
        first.pop("bufferView")
        first.pop("byteOffset")
        first["sparse"] = {"count": 2,
                           "indices": {"bufferView": add_view(bytes((1, 4))), "componentType": 5121},
                           "values": {"bufferView": add_view(struct.pack("<4f", .25, .75, .125, .875))}}
        doc["accessors"][attrs["TEXCOORD_2"]]["sparse"] = {
            "count": 2, "indices": {"bufferView": add_view(bytes((0, 5))), "componentType": 5121},
            "values": {"bufferView": add_view(bytes((77, 33, 199, 22)))},
        }
        doc["buffers"][0]["byteLength"] = len(blob)
        source.write_bytes(encode_glb(doc, blob))
        original = source.read_bytes()
        output = self.root / "sparse-out.glb"
        self.cli("ped_assemble.py", "assemble", source, "-o", output,
                 "--source-space", "y-up", "--no-cull", "--no-standing-offset")
        _, data = attributes(output)
        expected1 = [(0, 0), (.25, .75), (0, 0), (0, 0), (.125, .875), (0, 0)]
        expected2 = EXTRA[2].copy()
        expected2[0], expected2[5] = (77 / 255, 33 / 255), (199 / 255, 22 / 255)
        for channel, rows in ((1, expected1), (2, expected2), (3, EXTRA[3])):
            self.assertIn(f"TEXCOORD_{channel}", data)
            self.assertEqual(data[f"TEXCOORD_{channel}"], f32_rows(rows))
        self.assertEqual(source.read_bytes(), original)

        first["type"] = "VEC3"
        malformed = self.root / "wrong-shape.glb"
        malformed.write_bytes(encode_glb(doc, blob))
        rejected_output = self.root / "must-not-exist.glb"
        result = subprocess.run([sys.executable, "-I", str(ROOT / "tools/ped_assemble.py"),
                                 "assemble", str(malformed), "-o", str(rejected_output)],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("expected VEC2", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertFalse(rejected_output.exists())

    def test_extra_implicit_rows_share_the_import_budget_across_instances(self):
        source = self.root / "zero.glb"
        write_geometry(source)
        doc, blob = skinned_glb.read_glb(str(source))
        attrs = doc["meshes"][0]["primitives"][0]["attributes"]
        for channel in (1, 2, 3):
            doc["accessors"][attrs[f"TEXCOORD_{channel}"]] = {
                "componentType": 5126, "count": 6, "type": "VEC2",
            }
        for instances in (1, 2):
            with self.subTest(instances=instances):
                doc["nodes"] = [{"name": f"instance_{i}", "mesh": 0} for i in range(instances)]
                doc["scenes"] = [{"nodes": list(range(instances))}]
                source.write_bytes(encode_glb(doc, blob))
                # A small test-only cap exercises the same per-file boundary without
                # allocating a million Python rows. Every fixture is valid glTF.
                rows = 18 * instances
                with mock.patch.object(pa, "MAX_IMPLICIT_ACCESSOR_ROWS", rows):
                    parts = pa.read_glb_parts(str(source))
                self.assertEqual(len(parts), instances)
                for part in parts:
                    self.assertEqual(part.uvs_extra, {key: [(0.0, 0.0)] * 6 for key in (1, 2, 3)})
                with mock.patch.object(pa, "MAX_IMPLICIT_ACCESSOR_ROWS", rows - 1):
                    with self.assertRaisesRegex(pa.PedAssembleError, "implicit accessor rows"):
                        pa.read_glb_parts(str(source))

    def test_extra_indices_cannot_replace_primary_or_leave_gaps(self):
        for extra in ({0: UV0}, {2: UV0}, {1.0: UV0}, {True: UV0}, {1: UV0[:-1]}):
            with self.subTest(extra=list(extra)):
                part = pa.MeshPart("invalid", POSITIONS, [(0, 1, 2)], coord_space="y-up",
                                   uvs=UV0, uvs_extra=extra)
                with self.assertRaises(pa.PedAssembleError):
                    pa.build_assembled_glb([part], apply_standing=False)


if __name__ == "__main__":
    unittest.main()
