"""Tests for tools/glb_inspect.py.

Every GLB here is built by hand from JSON and BIN chunks; no game data, exporter
or third-party library is involved.

    python -I tests/test_glb_inspect.py
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "glb_inspect.py")
sys.path.insert(0, os.path.dirname(TOOL))
import glb_inspect  # noqa: E402

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
EXT_CHUNK = 0x54534554


def json_bytes(gltf):
    raw = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return raw + b" " * ((-len(raw)) % 4)


def raw_glb(chunks):
    """A GLB from (type, payload) chunks; the caller pads each payload to 4 bytes."""
    body = b"".join(struct.pack("<II", len(payload), chunk_type) + payload
                    for chunk_type, payload in chunks)
    return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body


def container(json_payload, bin_payload=None, *, magic=b"glTF", version=2, length_delta=0):
    """A GLB with an optional BIN chunk; `length_delta` corrupts the header length."""
    body = struct.pack("<II", len(json_payload), JSON_CHUNK) + json_payload
    if bin_payload is not None:
        bin_payload += b"\x00" * ((-len(bin_payload)) % 4)
        body += struct.pack("<II", len(bin_payload), BIN_CHUNK) + bin_payload
    size = 12 + len(body) + length_delta
    return struct.pack("<4sII", magic, version, size) + body


def glb(gltf, bin_payload=None, **kwargs):
    return container(json_bytes(gltf), bin_payload, **kwargs)


def triangle_gltf():
    return {
        "asset": {"version": "2.0", "generator": "synthetic glb_inspect test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [{"name": "Triangle", "primitives": [
            {"attributes": {"POSITION": 0}, "indices": 1, "material": 0},
        ]}],
        "materials": [{
            "name": "Red",
            "pbrMetallicRoughness": {
                "baseColorFactor": [1, 0, 0, 1],
                "baseColorTexture": {"index": 0},
            },
            "alphaMode": "MASK",
            "doubleSided": True,
        }],
        "textures": [{"name": "tex-red", "source": 0}],
        "images": [{"name": "red.png", "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3",
             "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 36},
            {"buffer": 0, "byteOffset": 36, "byteLength": 6},
        ],
        "buffers": [{"byteLength": 42}],
    }


def triangle_bin():
    positions = struct.pack("<9f", 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    indices = struct.pack("<3H", 0, 1, 2)
    return positions + indices


def triangle_glb():
    return glb(triangle_gltf(), triangle_bin())


def skinned_gltf():
    return {
        "asset": {"version": "2.0", "generator": "synthetic skinned test"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [
            {"name": "Armature", "children": [1, 2]},
            {"name": "JointA"},
            {"name": "JointB"},
            {"name": "Body", "mesh": 0, "skin": 0},
        ],
        "meshes": [{"name": "Body", "primitives": [{
            "attributes": {"POSITION": 0, "JOINTS_0": 1, "WEIGHTS_0": 2},
            "indices": 3,
            "material": 0,
            "mode": 4,
        }]}],
        "materials": [{"name": "Skin", "pbrMetallicRoughness": {
            "baseColorFactor": [0.8, 0.6, 0.5, 1],
        }}],
        "skins": [{"name": "Rig", "joints": [1, 2]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3",
             "min": [0, 0, 0], "max": [1, 1, 0]},
            {"bufferView": 1, "componentType": 5121, "count": 4, "type": "VEC4"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC4"},
            {"bufferView": 3, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": 48},
            {"buffer": 0, "byteOffset": 48, "byteLength": 16},
            {"buffer": 0, "byteOffset": 64, "byteLength": 64},
            {"buffer": 0, "byteOffset": 128, "byteLength": 12},
        ],
        "buffers": [{"byteLength": 140}],
    }


def skinned_bin():
    positions = struct.pack("<12f", 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0)
    joints = bytes((0, 1, 0, 0)) * 4
    weights = struct.pack("<16f", *([1.0, 0.0, 0.0, 0.0] * 4))
    indices = struct.pack("<6H", 0, 1, 2, 2, 1, 3)
    return positions + joints + weights + indices


def bbox_gltf(with_bounds):
    accessor = {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"}
    if with_bounds:
        accessor["min"] = [-1.5, -2, -3]
        accessor["max"] = [4, 5, 6]
    return {
        "asset": {"version": "2.0"},
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
        "accessors": [accessor],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 36}],
        "buffers": [{"byteLength": 36}],
    }


def animated_gltf():
    gltf = triangle_gltf()
    gltf["nodes"][0]["name"] = "Triangle"
    gltf["animations"] = [{
        "name": "Bounce",
        "samplers": [{"input": 2, "output": 3, "interpolation": "LINEAR"}],
        "channels": [{"sampler": 0, "target": {"node": 0, "path": "translation"}}],
    }]
    gltf["accessors"].extend([
        {"componentType": 5126, "count": 2, "type": "SCALAR", "min": [0.0], "max": [1.0]},
        {"componentType": 5126, "count": 2, "type": "VEC3"},
    ])
    return gltf



class GlbInspectTests(unittest.TestCase):
    def run_tool(self, *args):
        return subprocess.run([sys.executable, "-I", TOOL, *args],
                              capture_output=True, text=True, timeout=60, errors="replace")

    def run_bytes(self, data):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "input.glb")
            with open(path, "wb") as handle:
                handle.write(data)
            return self.run_tool(path)

    def assert_error(self, data, message):
        result = self.run_bytes(data)
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn(message.lower(), result.stderr.lower())
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual(len(result.stderr.strip().splitlines()), 1, result.stderr)

    def test_extension_only_base_color_image(self):
        for extension, uri in (("KHR_texture_basisu", "tile.ktx2"), ("EXT_texture_webp", "tile.webp")):
            with self.subTest(extension=extension):
                doc = triangle_gltf()
                doc["extensionsUsed"] = doc["extensionsRequired"] = [extension]
                doc["images"] = [{"uri": uri}]
                doc["textures"] = [{"name": "sampler", "extensions": {extension: {"source": 0}}}]
                result = self.run_bytes(glb(doc, triangle_bin()))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'baseColorTexture="{uri}"', result.stdout)

                # Keep the core report unchanged when the extension is optional.
                del doc["extensionsRequired"]
                doc["images"].append({"uri": "tile.png"})
                doc["textures"][0]["source"] = 1
                result = self.run_bytes(glb(doc, triangle_bin()))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('baseColorTexture="tile.png"', result.stdout)

    def test_minimal_triangle_mesh_and_material(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "triangle.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(triangle_glb())

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('asset: version="2.0" generator="synthetic glb_inspect test"', result.stdout)
            self.assertIn("counts: scenes=1 nodes=1 meshes=1 materials=1 images=1 skins=0 animations=0",
                          result.stdout)
            self.assertIn('name="Triangle" primitives=1 vertices=3 triangles=1 materials=["Red"]',
                          result.stdout)
            self.assertIn('baseColorFactor=[1, 0, 0, 1] baseColorTexture="red.png"', result.stdout)
            self.assertIn('alphaMode="MASK" doubleSided=true', result.stdout)
            self.assertIn("POSITION bbox: min [0, 0, 0] max [1, 1, 0]", result.stdout)

            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["asset"], {"version": "2.0", "generator": "synthetic glb_inspect test"})
            self.assertEqual(report["counts"], {"scenes": 1, "nodes": 1, "meshes": 1, "materials": 1,
                                                "images": 1, "skins": 0, "animations": 0})
            self.assertEqual([chunk["type"] for chunk in report["chunks"]], ["JSON", "BIN"])
            self.assertEqual(report["chunks"][1]["padding"], 2)
            self.assertEqual(report["meshes"][0]["vertices"], 3)
            self.assertEqual(report["meshes"][0]["triangles"], 1)
            self.assertEqual(report["meshes"][0]["materials"], ["Red"])
            self.assertEqual(report["meshes"][0]["primitives"][0]["mode_name"], "TRIANGLES")
            self.assertIsNone(report["meshes"][0]["primitives"][0]["mode"])
            self.assertTrue(report["meshes"][0]["primitives"][0]["mode_defaulted"])
            self.assertEqual(report["materials"][0]["base_color_texture"], "red.png")
            self.assertEqual(report["materials"][0]["base_color_texture_index"], 0)
            self.assertEqual(report["position_bbox"], {"min": [0, 0, 0], "max": [1, 1, 0]})
            self.assertTrue(report["position_bbox_complete"])

    def test_skinned_mesh_and_joint_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "skinned.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(skinned_gltf(), skinned_bin()))

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('name="Body" primitives=1 vertices=4 triangles=2 materials=["Skin"]', result.stdout)
            self.assertIn('skins:', result.stdout)
            self.assertIn('name="Rig" joints=2', result.stdout)

            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["counts"]["nodes"], 4)
            self.assertEqual(report["counts"]["skins"], 1)
            self.assertEqual(report["skins"], [{"index": 0, "name": "Rig", "joint_count": 2}])
            primitive = report["meshes"][0]["primitives"][0]
            self.assertEqual(primitive["vertices"], 4)
            self.assertEqual(primitive["triangles"], 2)
            self.assertEqual(primitive["material"], "Skin")
            self.assertEqual(primitive["mode"], 4)
            self.assertFalse(primitive["mode_defaulted"])

    def test_bad_magic_version_and_length_are_one_line_errors(self):
        cases = (
            ("bad magic", glb(triangle_gltf(), triangle_bin(), magic=b"GLBB"), "bad GLB magic"),
            ("bad version", glb(triangle_gltf(), triangle_bin(), version=1), "unsupported GLB version 1"),
            ("bad length", glb(triangle_gltf(), triangle_bin(), length_delta=4), "does not match file size"),
        )
        for label, data, message in cases:
            with self.subTest(label):
                self.assert_error(data, message)

    def test_missing_json_malformed_json_and_truncated_chunk(self):
        missing_json = struct.pack("<4sII", b"glTF", 2, 24) + struct.pack("<II", 4, BIN_CHUNK) + b"junk"
        self.assert_error(missing_json, "missing JSON chunk")

        malformed = container(b"{ nope }")
        self.assert_error(malformed, "malformed JSON chunk")

        truncated = struct.pack("<4sII", b"glTF", 2, 24) + struct.pack("<II", 100, JSON_CHUNK) + b"    "
        self.assert_error(truncated, "truncated JSON chunk")

    def test_extension_chunks_are_ignored_but_bin_stays_second(self):
        payload = json_bytes({"asset": {"version": "2.0"}})
        accepted = raw_glb([(JSON_CHUNK, payload), (EXT_CHUNK, b"TEST")])
        result = self.run_bytes(accepted)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"chunks: JSON {len(payload)} bytes", result.stdout)
        self.assertIn("0x54534554 4 bytes", result.stdout)

        with_bin = raw_glb([(JSON_CHUNK, payload), (BIN_CHUNK, b"\x00" * 4),
                             (EXT_CHUNK, b"TEST")])
        self.assertEqual(self.run_bytes(with_bin).returncode, 0)

        misplaced = raw_glb([(JSON_CHUNK, payload), (EXT_CHUNK, b"TEST"),
                             (BIN_CHUNK, b"\x00" * 4)])
        self.assert_error(misplaced, "BIN chunk must immediately follow")

        duplicate = raw_glb([(JSON_CHUNK, payload), (JSON_CHUNK, payload)])
        self.assert_error(duplicate, "duplicate JSON chunk")

    def test_parse_only_copies_the_json_payload(self):
        metadata = json_bytes({"asset": {"version": "2.0"}, "buffers": [{"byteLength": 3}]})
        json_end = glb_inspect.GLB_HEADER_SIZE + glb_inspect.CHUNK_HEADER_SIZE + len(metadata)

        class MetadataOnlyBytes(bytes):
            def __getitem__(self, key):
                if isinstance(key, slice) and (key.stop is None or key.stop > json_end):
                    raise AssertionError("copied payload outside the JSON chunk")
                return super().__getitem__(key)

        data = MetadataOnlyBytes(raw_glb([
            (JSON_CHUNK, metadata), (BIN_CHUNK, b"abc\x00"), (EXT_CHUNK, b"TEST"),
        ]))
        chunks = glb_inspect.parse_glb(data)["chunks"]
        self.assertEqual(chunks[1]["padding"], 1)
        self.assertEqual(chunks[2]["byte_length"], 4)

    def test_chunk_and_header_lengths_must_be_4_byte_aligned(self):
        payload = b'{"asset":{"version":"2.0"}}'
        self.assertEqual(len(payload) % 4, 3)

        header = struct.pack("<4sII", b"glTF", 2, 48)
        body = struct.pack("<II", len(payload), JSON_CHUNK) + payload + b" "
        self.assert_error(header + body, "not a multiple of 4")

        header = struct.pack("<4sII", b"glTF", 2, 47)
        body = struct.pack("<II", len(payload), JSON_CHUNK) + payload
        self.assert_error(header + body, "not 4-byte aligned")

    def test_nul_padding_is_accepted_and_counted(self):
        payload = b'{"asset":{"version":"2.0"}}' + b"\x00"
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "nul_padding.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(raw_glb([(JSON_CHUNK, payload)]))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("padding 1", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["chunks"][0]["padding"], 1)
            self.assertEqual(report["chunks"][0]["nul_padding"], 1)

    def test_negative_indices_are_treated_as_missing(self):
        gltf = {
            "asset": {"version": "2.0"},
            "meshes": [{"primitives": [{"attributes": {"POSITION": -1}, "material": -2}]}],
            "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": -4}}}],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "negative.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            primitive = report["meshes"][0]["primitives"][0]
            self.assertIsNone(primitive["position_accessor"])
            self.assertIsNone(primitive["material_index"])
            self.assertIsNone(primitive["material"])
            self.assertIsNone(report["materials"][0]["base_color_texture_index"])
            self.assertIsNone(report["position_bbox"])
            self.assertFalse(report["position_bbox_complete"])

    def test_non_object_entries_are_counted_as_invalid(self):
        gltf = {
            "asset": {"version": "2.0"},
            "meshes": [None, {"primitives": []}],
            "materials": [None],
            "skins": [None],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "invalid.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("invalid entries: meshes=1 materials=1 skins=1", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["counts"]["meshes"], 2)
            self.assertEqual(report["counts"]["materials"], 1)
            self.assertEqual(report["counts"]["skins"], 1)
            self.assertEqual(len(report["meshes"]), 1)
            self.assertEqual(len(report["materials"]), 0)
            self.assertEqual(len(report["skins"]), 0)
            self.assertEqual(report["invalid_counts"], {"meshes": 1, "materials": 1, "skins": 1})

    def test_json_write_failure_does_not_print_a_partial_report(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.glb")
            out = os.path.join(folder, "missing", "report.json")
            with open(path, "wb") as handle:
                handle.write(triangle_glb())
            result = self.run_tool(path, "--json", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot write", result.stderr)
            self.assertEqual(result.stdout.strip(), "")

    def test_json_serialization_failure_does_not_create_a_partial_report(self):
        payload = (
            b'{"asset":{"version":"2.0"},"materials":[{"name":"Huge",'
            b'"pbrMetallicRoughness":{"baseColorFactor":[1e400,0,0,1]}}]}'
        )
        payload += b" " * ((-len(payload)) % 4)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(container(payload))
            result = self.run_tool(path, "--json", out)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("cannot write", result.stderr)
            self.assertIn("Out of range float values", result.stderr)
            self.assertEqual(result.stdout.strip(), "")
            self.assertFalse(os.path.exists(out), "a serialization failure left a partial report")

    def test_json_round_trip_and_input_alias_refusal(self):
        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "model.glb")
            first = os.path.join(folder, "first.json")
            second = os.path.join(folder, "second.json")
            data = triangle_glb()
            with open(source, "wb") as handle:
                handle.write(data)

            result = self.run_tool(source, "--json", first)
            self.assertEqual(result.returncode, 0, result.stderr)
            again = self.run_tool(source, "--json", second)
            self.assertEqual(again.returncode, 0, again.stderr)
            with open(first, "rb") as handle:
                first_bytes = handle.read()
            with open(second, "rb") as handle:
                second_bytes = handle.read()
            self.assertEqual(first_bytes, second_bytes, "the JSON report is not deterministic")
            self.assertNotIn(b"\r\n", first_bytes, "--json must use LF on every platform")
            self.assertTrue(first_bytes.endswith(b"\n"))
            self.assertEqual(json.loads(first_bytes.decode("utf-8"))["meshes"][0]["name"], "Triangle")

            alias = self.run_tool(source, "--json", source)
            self.assertNotEqual(alias.returncode, 0)
            self.assertIn("must not overwrite the input", alias.stderr)
            with open(source, "rb") as handle:
                self.assertEqual(handle.read(), data, "the report path overwrote the input")

            dotted = self.run_tool(source, "--json", os.path.join(folder, ".", "model.glb"))
            self.assertNotEqual(dotted.returncode, 0)
            self.assertIn("must not overwrite the input", dotted.stderr)
            with open(source, "rb") as handle:
                self.assertEqual(handle.read(), data)

    def test_json_only_glb_is_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "empty.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb({"asset": {"version": "2.0"}}))

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("chunks: JSON", result.stdout)
            self.assertNotIn("BIN", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual([chunk["type"] for chunk in report["chunks"]], ["JSON"])
            self.assertEqual(report["counts"], {"scenes": 0, "nodes": 0, "meshes": 0, "materials": 0,
                                                "images": 0, "skins": 0, "animations": 0})
            self.assertIsNone(report["position_bbox"])

    def test_position_bbox_from_accessor_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "bounds.glb")
            out = os.path.join(folder, "report.json")
            payload = struct.pack("<9f", -1.5, -2.0, -3.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
            with open(path, "wb") as handle:
                handle.write(glb(bbox_gltf(with_bounds=True), payload))

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("POSITION bbox: min [-1.5, -2, -3] max [4, 5, 6]", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["position_bbox"], {"min": [-1.5, -2, -3], "max": [4, 5, 6]})
            self.assertTrue(report["position_bbox_complete"])
            self.assertEqual(report["meshes"][0]["primitives"][0]["bbox"], report["position_bbox"])

    def test_missing_bounds_and_material_fields_are_reported_not_guessed(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "missing.glb")
            out = os.path.join(folder, "report.json")
            gltf = bbox_gltf(with_bounds=False)
            gltf["materials"] = [{}]
            with open(path, "wb") as handle:
                handle.write(glb(gltf, bytes(36)))

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("POSITION bbox: (missing)", result.stdout)
            self.assertIn("baseColorFactor=(missing)", result.stdout)
            self.assertIn("alphaMode=(missing)", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertIsNone(report["position_bbox"])
            self.assertFalse(report["position_bbox_complete"])
            self.assertIsNone(report["meshes"][0]["name"])
            self.assertIsNone(report["meshes"][0]["primitives"][0]["bbox"])
            self.assertEqual(report["materials"][0], {
                "index": 0,
                "name": None,
                "base_color_factor": None,
                "base_color_texture": None,
                "base_color_texture_index": None,
                "alpha_mode": None,
                "double_sided": None,
            })

    def test_animation_channels_samplers_and_duration(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "animated.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(animated_gltf(), triangle_bin()))

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("animations=1", result.stdout)
            self.assertIn('name="Bounce" channels=1 samplers=1 duration=1.0 targets=1', result.stdout)
            self.assertIn("paths: translation=1; interpolation: LINEAR", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            animation = report["animations"][0]
            self.assertEqual(animation["channel_count"], 1)
            self.assertEqual(animation["sampler_count"], 1)
            self.assertEqual(animation["duration"], 1.0)
            self.assertEqual(animation["target_node_count"], 1)
            self.assertEqual(animation["paths"], {"translation": 1})
            self.assertEqual(animation["channels"][0]["node_name"], "Triangle")
            self.assertEqual(animation["channels"][0]["path"], "translation")
            self.assertEqual(animation["samplers"][0]["interpolation"], "LINEAR")
            self.assertEqual(animation["issues"], [])

    def test_animation_reference_issues_are_reported(self):
        gltf = animated_gltf()
        gltf["animations"][0]["channels"] = [
            {"sampler": 9, "target": {"node": 99, "path": "translation"}},
            {"sampler": 0, "target": {"node": 0, "path": "bogus"}},
        ]
        gltf["animations"][0]["samplers"][0]["output"] = 4
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "issues.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("issue:", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            issues = report["animations"][0]["issues"]
            self.assertTrue(any("no valid sampler" in issue for issue in issues))
            self.assertTrue(any("missing node 99" in issue for issue in issues))
            self.assertTrue(any("unknown target path 'bogus'" in issue for issue in issues))
            self.assertTrue(any("output accessor is missing" in issue for issue in issues))
            self.assertEqual(report["animations"][0]["target_node_count"], 1)

    def test_animation_type_mismatches_are_reported(self):
        gltf = animated_gltf()
        gltf["accessors"][2]["type"] = "VEC3"
        gltf["accessors"][3]["type"] = "VEC4"
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "types.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))

            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            issues = report["animations"][0]["issues"]
            self.assertTrue(any("input accessor type is VEC3, expected SCALAR" in issue for issue in issues))
            self.assertTrue(any("path 'translation' expects VEC3 output, got VEC4" in issue for issue in issues))

    def test_non_object_sampler_is_an_issue_not_a_crash(self):
        gltf = animated_gltf()
        gltf["animations"][0]["samplers"] = [7]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "sampler.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("issue: sampler[0] is not an object", result.stdout)
            with open(out, encoding="utf-8") as handle:
                animation = json.load(handle)["animations"][0]
            self.assertEqual(animation["issues"], ["sampler[0] is not an object"])
            self.assertIsNone(animation["samplers"][0]["output_type"])

    def test_oversized_integer_max_reports_no_duration_instead_of_crashing(self):
        gltf = animated_gltf()
        gltf["accessors"][2]["max"] = [10 ** 400]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "huge.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn("duration=(missing)", result.stdout)
            with open(out, encoding="utf-8") as handle:
                animation = json.load(handle)["animations"][0]
            self.assertIsNone(animation["duration"])
            self.assertIsNone(animation["samplers"][0]["duration"])

    def test_zero_keyframes_has_no_duration(self):
        gltf = animated_gltf()
        gltf["accessors"][2]["count"] = 0
        gltf["accessors"][3]["count"] = 0
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "empty_keys.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("duration=(missing)", result.stdout)
            with open(out, encoding="utf-8") as handle:
                animation = json.load(handle)["animations"][0]
            self.assertIsNone(animation["duration"])
            self.assertEqual(animation["issues"], ["sampler[0] input accessor has no keyframes"])

    def test_non_string_interpolation_is_unknown_not_linear(self):
        gltf = animated_gltf()
        gltf["animations"][0]["samplers"][0]["interpolation"] = 5
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "interpolation.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("interpolation: none", result.stdout)
            with open(out, encoding="utf-8") as handle:
                animation = json.load(handle)["animations"][0]
            self.assertEqual(animation["interpolations"], [])
            self.assertIsNone(animation["samplers"][0]["interpolation"])
            self.assertEqual(animation["issues"], ["sampler[0] has unknown interpolation 5"])

    def test_wrong_typed_arrays_and_channels_are_reported(self):
        gltf = {
            "asset": {"version": "2.0"},
            "scenes": {}, "nodes": 5, "meshes": "x", "materials": {},
            "images": 3, "skins": {}, "animations": {}, "accessors": {},
        }
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "types.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("invalid types: scenes=dict nodes=int meshes=str", result.stdout)
            with open(out, encoding="utf-8") as handle:
                report = json.load(handle)
            self.assertEqual(report["invalid_types"], {
                "scenes": "dict", "nodes": "int", "meshes": "str", "materials": "dict",
                "images": "int", "skins": "dict", "animations": "dict", "accessors": "dict",
            })

            animated = animated_gltf()
            animated["animations"][0]["channels"] = {}
            with open(path, "wb") as handle:
                handle.write(glb(animated, triangle_bin()))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out, encoding="utf-8") as handle:
                animation = json.load(handle)["animations"][0]
            self.assertEqual(animation["channel_count"], 0)
            self.assertIn("animation channels is not an array", animation["issues"])

    def test_cubic_spline_output_count_is_validated(self):
        gltf = animated_gltf()
        gltf["animations"][0]["samplers"][0]["interpolation"] = "CUBICSPLINE"
        gltf["accessors"][3]["count"] = 6
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "cubic.glb")
            out = os.path.join(folder, "report.json")
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out, encoding="utf-8") as handle:
                animation = json.load(handle)["animations"][0]
            self.assertEqual(animation["interpolations"], ["CUBICSPLINE"])
            self.assertEqual(animation["samplers"][0]["output_count"], 6)
            self.assertEqual(animation["issues"], [])

            gltf["accessors"][3]["count"] = 5
            with open(path, "wb") as handle:
                handle.write(glb(gltf, triangle_bin()))
            result = self.run_tool(path, "--json", out)
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out, encoding="utf-8") as handle:
                issues = json.load(handle)["animations"][0]["issues"]
            self.assertTrue(any("has 5 output values for 2 keys with CUBICSPLINE interpolation" in issue
                                for issue in issues))


if __name__ == "__main__":
    unittest.main()
