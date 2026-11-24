"""tools/ped_assemble.py on a malformed GLB: one `ERROR:` line and exit 2, not a traceback.

    python -I tests/test_ped_assemble_malformed_glb.py

Every GLB here is built by hand in the test (one triangle, one material); no game files and no GPU.
"""
from __future__ import annotations

import copy
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import ped_assemble as pa  # noqa: E402

DELETE = object()
POSITIONS = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
UVS = struct.pack("<6f", 0, 0, 1, 0, 0, 1)
INDICES = struct.pack("<3H", 0, 1, 2) + b"\0\0"
PAYLOAD = POSITIONS + UVS + INDICES


def document() -> dict:
    """A valid one-triangle document over PAYLOAD: POSITION, TEXCOORD_0, 16-bit indices and one material."""
    return {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": len(PAYLOAD)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(POSITIONS)},
            {"buffer": 0, "byteOffset": len(POSITIONS), "byteLength": len(UVS)},
            {"buffer": 0, "byteOffset": len(POSITIONS) + len(UVS), "byteLength": 6},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 3, "type": "SCALAR"},
        ],
        "materials": [{"name": "m", "alphaCutoff": 0.5,
                       "pbrMetallicRoughness": {"baseColorFactor": [1, 0.5, 0.25, 1], "metallicFactor": 0.1}}],
        "meshes": [{"name": "tri", "primitives": [
            {"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "nodes": [{"name": "tri", "mesh": 0, "translation": [1, 2, 3]}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }


def container(json_chunk: bytes, payload: bytes = PAYLOAD) -> bytes:
    json_chunk += b" " * (-len(json_chunk) % 4)
    body = struct.pack("<II", len(json_chunk), pa.CHUNK_TYPE_JSON) + json_chunk
    if payload:
        payload += b"\0" * (-len(payload) % 4)
        body += struct.pack("<II", len(payload), pa.CHUNK_TYPE_BIN) + payload
    return pa.GLB_MAGIC + struct.pack("<II", pa.GLB_VERSION, 12 + len(body)) + body


def edited(path: tuple, value: object) -> dict:
    """The valid document with the value at `path` replaced (or removed, for DELETE)."""
    doc = copy.deepcopy(document())
    cursor = doc
    for key in path[:-1]:
        cursor = cursor[key]
    if value is DELETE:
        del cursor[path[-1]]
    else:
        cursor[path[-1]] = value
    return doc


PRIM = ("meshes", 0, "primitives", 0)
MALFORMED = (
    # (label, path into the document, replacement, text the message must contain)
    ("document is a list", (), [], "JSON chunk must be an object"),
    ("accessors is an object", ("accessors",), {}, "accessors must be a list"),
    ("meshes is a number", ("meshes",), 3, "meshes must be a list"),
    ("mesh is a string", ("meshes", 0), "tri", "meshes[0] must be an object"),
    ("primitives is an object", ("meshes", 0, "primitives"), {}, "primitives must be a list"),
    ("primitive is a number", PRIM, 7, "primitives[0] must be an object"),
    ("attributes is a list", PRIM + ("attributes",), ["POSITION"], "attributes must be an object"),
    ("accessor index out of range", PRIM + ("attributes", "POSITION"), 9, "names accessor 9"),
    ("accessor index is a string", PRIM + ("attributes", "POSITION"), "0", "names accessor '0'"),
    ("accessor index is a bool", PRIM + ("indices",), True, "names accessor True"),
    ("accessor index is negative", PRIM + ("indices",), -1, "names accessor -1"),
    ("accessors table missing", ("accessors",), DELETE, "names accessor 0"),
    ("accessor is a list", ("accessors", 0), [], "accessors[0] must be an object"),
    ("accessor offset without bufferView", ("accessors", 0),
     {"componentType": 5126, "count": 3, "type": "VEC3", "byteOffset": 0}, "cannot specify byteOffset"),
    ("bufferView index out of range", ("accessors", 0, "bufferView"), 5, "names bufferView 5"),
    ("bufferView is a number", ("bufferViews", 0), 0, "bufferViews[0] must be an object"),
    ("unknown componentType", ("accessors", 0, "componentType"), 5130, "unsupported componentType 5130"),
    ("componentType is a list", ("accessors", 0, "componentType"), [5126], "unsupported componentType"),
    ("unknown accessor type", ("accessors", 0, "type"), "VEC5", "unsupported type 'VEC5'"),
    ("POSITION is a scalar accessor", ("accessors", 0, "type"), "SCALAR", "expected VEC3"),
    ("TEXCOORD_0 is a VEC3 accessor", ("accessors", 1, "type"), "VEC3", "expected VEC2"),
    ("indices is a VEC2 accessor", ("accessors", 2, "type"), "VEC2", "expected SCALAR"),
    ("count is a string", ("accessors", 0, "count"), "3", "invalid count '3'"),
    ("count is negative", ("accessors", 0, "count"), -3, "invalid count -3"),
    ("count missing", ("accessors", 0, "count"), DELETE, "invalid count None"),
    ("count past the BIN chunk", ("accessors", 0, "count"), 1_000_000, "runs past the BIN chunk"),
    ("view offset past the BIN chunk", ("bufferViews", 2, "byteOffset"), 1 << 40, "runs past the BIN chunk"),
    ("accessor offset is a float", ("accessors", 0, "byteOffset"), 1.5, "invalid byteOffset or byteStride"),
    ("byteStride is a string", ("bufferViews", 0, "byteStride"), "12", "invalid byteOffset or byteStride"),
    ("byteStride walks past the end", ("bufferViews", 0, "byteStride"), 64, "runs past the BIN chunk"),
    ("material index is a string", PRIM + ("material",), "m", "material must be an index"),
    ("material is a number", ("materials", 0), 3, "materials[0] must be an object"),
    ("pbrMetallicRoughness is a list", ("materials", 0, "pbrMetallicRoughness"), [], "must be an object"),
    ("baseColorFactor is short", ("materials", 0, "pbrMetallicRoughness", "baseColorFactor"), [1, 1, 1],
     "baseColorFactor must be a list of 4 numbers"),
    ("baseColorFactor holds a list", ("materials", 0, "pbrMetallicRoughness", "baseColorFactor"),
     [1, 1, [1], 1], "baseColorFactor must be a number"),
    ("metallicFactor is an object", ("materials", 0, "pbrMetallicRoughness", "metallicFactor"), {},
     "metallicFactor must be a number"),
    ("alphaCutoff is a string", ("materials", 0, "alphaCutoff"), "half", "alphaCutoff must be a number"),
    ("node is a number", ("nodes", 0), 0, "nodes[0] must be an object"),
    ("node mesh out of range", ("nodes", 0, "mesh"), 4, "names mesh 4"),
    ("node mesh is a string", ("nodes", 0, "mesh"), "tri", "names mesh 'tri'"),
    ("count is a fraction", ("accessors", 0, "count"), 2.5, "invalid count 2.5"),
    ("count is a bool", ("accessors", 0, "count"), True, "invalid count True"),
    ("node name is a number", ("nodes", 0, "name"), 5, "nodes[0].name must be a string, got 5"),
    ("node name is a fraction", ("nodes", 0, "name"), 3.5, "nodes[0].name must be a string"),
    ("node name is a bool", ("nodes", 0, "name"), True, "nodes[0].name must be a string"),
    ("node name is a list", ("nodes", 0, "name"), ["tri"], "nodes[0].name must be a string"),
    ("translation has two values", ("nodes", 0, "translation"), [1, 2], "translation must be a list of 3"),
    ("translation holds a string", ("nodes", 0, "translation"), [1, "x", 3], "translation must be a number"),
)


class MalformedGlb(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = self.enterContext(tempfile.TemporaryDirectory())

    def write(self, name: str, data: bytes) -> str:
        path = os.path.join(self.folder, name)
        with open(path, "wb") as fh:
            fh.write(data)
        return path

    def write_doc(self, name: str, doc: object, payload: bytes = PAYLOAD) -> str:
        return self.write(name, container(json.dumps(doc).encode("utf-8"), payload))

    def test_the_valid_document_still_reads(self) -> None:
        (part,) = pa.read_glb_parts(self.write_doc("ok.glb", document()))
        self.assertEqual(part.positions, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])
        self.assertEqual(part.triangles, [(0, 1, 2)])
        self.assertEqual(part.uvs, [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
        self.assertEqual(part.translation, (1.0, 2.0, 3.0))
        self.assertEqual(part.material.base_color_factor, (1.0, 0.5, 0.25, 1.0))
        self.assertEqual(part.material.alpha_cutoff, 0.5)

    def test_what_the_reader_tolerated_is_still_tolerated(self) -> None:
        """Not errors before, not errors now: no nodes, a material index past the table, a part with no POSITION,
        an accessor nothing reads, and numbers written as strings that float() takes."""
        doc = edited(("nodes",), DELETE)
        self.assertEqual(len(pa.read_glb_parts(self.write_doc("no_nodes.glb", doc))), 1)
        (part,) = pa.read_glb_parts(self.write_doc("mat.glb", edited(PRIM + ("material",), 5)))
        self.assertIsNone(part.material)
        doc = document()
        doc["meshes"][0]["primitives"].append({"attributes": {"NORMAL": 99}, "indices": "junk", "material": []})
        doc["accessors"].append("never read")
        self.assertEqual(len(pa.read_glb_parts(self.write_doc("unread.glb", doc))), 1)
        doc = edited(("materials", 0, "pbrMetallicRoughness", "metallicFactor"), "0.75")
        (part,) = pa.read_glb_parts(self.write_doc("string_number.glb", doc))
        self.assertEqual(part.material.metallic_factor, 0.75)

    def test_whole_floats_read_as_the_reader_always_took_them(self) -> None:
        """The reader takes count, byteOffset and byteStride through int(), so 3.0 was 3 before and still is."""
        for label, path, value in (
            ("count", ("accessors", 0, "count"), 3.0),
            ("view byteOffset", ("bufferViews", 0, "byteOffset"), 0.0),
            ("accessor byteOffset", ("accessors", 0, "byteOffset"), 0.0),
            ("byteStride", ("bufferViews", 0, "byteStride"), 12.0),
            ("byteStride zero", ("bufferViews", 0, "byteStride"), 0.0),
        ):
            with self.subTest(label):
                (part,) = pa.read_glb_parts(self.write_doc("whole.glb", edited(path, value)))
                self.assertEqual(part.positions, [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)])

    def test_a_name_that_is_not_text_is_an_error_only_where_it_names_the_part(self) -> None:
        """The part is named after the node, else the mesh: a non-text name is refused where the reader would use
        it, and left alone where it would not."""
        mesh_named_5 = edited(("nodes", 0, "name"), DELETE)
        mesh_named_5["meshes"][0]["name"] = 5
        no_nodes = edited(("nodes",), DELETE)
        no_nodes["meshes"][0]["name"] = 5
        for label, doc in (("node without a name", mesh_named_5), ("no nodes", no_nodes)):
            with self.subTest(label):
                with self.assertRaises(pa.PedAssembleError) as caught:
                    pa.read_glb_parts(self.write_doc("name.glb", doc))
                self.assertIn("meshes[0].name must be a string, got 5", str(caught.exception))

        unused = document()
        unused["meshes"][0]["name"] = 5                      # the node has a text name, so this one is never used
        (part,) = pa.read_glb_parts(self.write_doc("unused_name.glb", unused))
        self.assertEqual(part.name, "tri")
        for label, value in (("empty list", []), ("zero", 0), ("empty text", "")):
            with self.subTest(label):                        # falsy: the reader falls back to the mesh's name
                (part,) = pa.read_glb_parts(self.write_doc("falsy_name.glb", edited(("nodes", 0, "name"), value)))
                self.assertEqual(part.name, "tri")

    def test_each_malformed_document_is_one_error(self) -> None:
        for index, (label, path, value, expected) in enumerate(MALFORMED):
            with self.subTest(label):
                doc = value if not path else edited(path, value)
                glb = self.write_doc(f"case{index}.glb", doc)
                with self.assertRaises(pa.PedAssembleError) as caught:
                    pa.read_glb_parts(glb)
                message = str(caught.exception)
                self.assertTrue(message.startswith(f"{glb}: malformed GLB: "), message)
                self.assertIn(expected, message)

    def test_a_short_bin_chunk_is_an_error(self) -> None:
        for label, payload in (("no BIN chunk", b""), ("BIN chunk cut inside the indices", PAYLOAD[:60])):
            with self.subTest(label):
                glb = self.write_doc("short.glb", document(), payload)
                with self.assertRaises(pa.PedAssembleError) as caught:
                    pa.read_glb_parts(glb)
                self.assertIn("runs past the BIN chunk", str(caught.exception))

    def test_a_json_chunk_that_does_not_parse_is_an_error(self) -> None:
        for label, chunk in (("not JSON", b"{not json"), ("not UTF-8", b'{"asset": "\xff\xfe"}')):
            with self.subTest(label):
                glb = self.write(f"{label.replace(' ', '_')}.glb", container(chunk))
                with self.assertRaises(pa.PedAssembleError) as caught:
                    pa.read_glb_parts(glb)
                self.assertIn("the JSON chunk does not parse", str(caught.exception))

    def test_cli_prints_one_error_line_and_exits_2(self) -> None:
        script = os.path.join(TOOLS, "ped_assemble.py")
        out = os.path.join(self.folder, "out.glb")
        cases = (
            ("material is not an object", edited(("materials", 0), 3), PAYLOAD),
            ("pbrMetallicRoughness is not an object", edited(("materials", 0, "pbrMetallicRoughness"), []), PAYLOAD),
            ("BIN chunk shorter than the accessors", document(), PAYLOAD[:60]),
        )
        for index, (label, doc, payload) in enumerate(cases):
            with self.subTest(label):
                glb = self.write_doc(f"cli{index}.glb", doc, payload)
                proc = subprocess.run(
                    [sys.executable, "-I", script, "assemble", glb, "-o", out, "--source-space", "y-up"],
                    capture_output=True, text=True, timeout=60)
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertNotIn("Traceback", proc.stderr)
                self.assertEqual(len(proc.stderr.strip().splitlines()), 1, proc.stderr)
                self.assertTrue(proc.stderr.startswith("ERROR: "), proc.stderr)
                self.assertFalse(os.path.exists(out))


if __name__ == "__main__":
    unittest.main()
