"""Signed-normalized glTF endpoints through the real ped importer and CLI.

The glTF accessor-data-types contract maps -128/-127 and -32768/-32767 to -1.
KHR_mesh_quantization permits these POSITION/TEXCOORD component types:
https://github.com/KhronosGroup/glTF/blob/5b00b913b74c8f0260c64578b98a09ef2dde0b41/extensions/2.0/Khronos/KHR_mesh_quantization/README.md
"""
from __future__ import annotations

import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glb_skinning
import ped_assemble as pa


def make_glb(component_type, values, one, *, normalized, interleaved=False, sparse=None):
    """Repack a real writer triangle; every vertex row/offset is four-byte aligned."""
    part = pa.MeshPart(name="quantized", coord_space="y-up",
                       positions=[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
                       triangles=[(0, 1, 2)], uvs=[(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
    original, _ = pa.build_assembled_glb([part], apply_standing=False)
    doc, binary, _ = glb_skinning.parse_glb(original)
    data = bytearray(binary)
    fmt = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5126: "f"}[component_type]

    def view(raw, **fields):
        data.extend(bytes(-len(data) % 4))
        index = len(doc["bufferViews"])
        doc["bufferViews"].append(dict(buffer=0, byteOffset=len(data), byteLength=len(raw), **fields))
        data.extend(raw)
        return index

    attrs = doc["meshes"][0]["primitives"][0]["attributes"]
    for name, width in (("POSITION", 3), ("TEXCOORD_0", 2)):
        accessor = doc["accessors"][attrs[name]]
        rows = [(v, one if i == 1 else 0, *([0] if width == 3 else [])) for i, v in enumerate(values)]
        if sparse is not None:
            rows = [(0, 0, *([0] if width == 3 else [])),
                    (0, one, *([0] if width == 3 else [])),
                    (one, 0, *([0] if width == 3 else []))]
        size = struct.calcsize("<" + fmt * width)
        stride = (size + 3) // 4 * 4 + (4 if interleaved else 0)
        offset = 4 if interleaved else 0
        raw = bytearray(b"PAD!" if offset else b"")
        for row in rows:
            raw.extend(struct.pack("<" + fmt * width, *row))
            raw.extend(bytes(stride - size))
        accessor.update(bufferView=view(bytes(raw), byteStride=stride, target=34962), byteOffset=offset,
                        componentType=component_type, normalized=normalized)
        if sparse is not None:
            if sparse == "implicit":
                del accessor["bufferView"]
                del accessor["byteOffset"]
                rows = [(0,) * width] * 3
            replacement = (values[0],) + (0,) * (width - 1)
            accessor["sparse"] = dict(count=1, indices=dict(bufferView=view(b"\0"), componentType=5121),
                                      values=dict(bufferView=view(struct.pack("<" + fmt * width, *replacement))))
            rows[0] = replacement
        if name == "POSITION":
            accessor["min"] = [min(row[c] for row in rows) for c in range(3)]
            accessor["max"] = [max(row[c] for row in rows) for c in range(3)]
    doc["buffers"][0]["byteLength"] = len(data)
    doc["extensionsUsed"] = ["KHR_mesh_quantization"]
    doc["extensionsRequired"] = ["KHR_mesh_quantization"]
    encoded = json.dumps(doc, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 4)
    data.extend(bytes(-len(data) % 4))
    return (struct.pack("<4sII", b"glTF", 2, 28 + len(encoded) + len(data))
            + struct.pack("<II", len(encoded), 0x4E4F534A) + encoded
            + struct.pack("<II", len(data), 0x004E4942) + data)


def output_rows(path, semantic):
    """Read emitted FLOAT rows directly, independently of the importer under test."""
    doc, data, _ = glb_skinning.parse_glb(path.read_bytes())
    index = doc["meshes"][0]["primitives"][0]["attributes"][semantic]
    accessor = doc["accessors"][index]
    assert accessor["componentType"] == 5126 and not accessor.get("normalized", False)
    view = doc["bufferViews"][accessor["bufferView"]]
    width = {"VEC2": 2, "VEC3": 3}[accessor["type"]]
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", width * 4)
    return [struct.unpack_from("<" + "f" * width, data, offset + i * stride)
            for i in range(accessor["count"])]


class SignedNormalizedPedTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.source = self.root / "input.glb"
        self.output = self.root / "output.glb"

    def read(self, data):
        self.source.write_bytes(data)
        part, = pa.read_glb_parts(str(self.source))
        self.assertEqual(self.source.read_bytes(), data)
        return part

    def assemble(self, data):
        self.source.write_bytes(data)
        run = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "ped_assemble.py"),
                              "assemble", str(self.source), "-o", str(self.output), "--no-standing-offset"],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.source.read_bytes(), data)
        return output_rows(self.output, "POSITION"), output_rows(self.output, "TEXCOORD_0")

    def test_dense_signed_endpoints_with_packed_and_interleaved_rows(self):
        for ctype, minimum, maximum in ((5120, -128, 127), (5122, -32768, 32767)):
            for interleaved in (False, True):
                with self.subTest(component_type=ctype, interleaved=interleaved):
                    part = self.read(make_glb(ctype, [minimum, minimum + 1, maximum], maximum,
                                             normalized=True, interleaved=interleaved))
                    self.assertEqual(part.positions, [(-1.0, 0.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 0.0, 0.0)])
                    self.assertEqual(part.uvs, [(-1.0, 0.0), (-1.0, 1.0), (1.0, 0.0)])

    def test_sparse_signed_endpoints_with_dense_and_implicit_bases(self):
        for ctype, minimum, maximum in ((5120, -128, 127), (5122, -32768, 32767)):
            for sparse in ("dense", "implicit"):
                with self.subTest(component_type=ctype, sparse=sparse):
                    part = self.read(make_glb(ctype, [minimum, 0, maximum], maximum,
                                             normalized=True, sparse=sparse))
                    tail = [(0.0, 1.0), (1.0, 0.0)] if sparse == "dense" else [(0.0, 0.0)] * 2
                    self.assertEqual(part.uvs, [(-1.0, 0.0), *tail])
                    self.assertEqual(part.positions, [(u, v, 0.0) for u, v in [(-1.0, 0.0), *tail]])

    def test_unsigned_raw_signed_and_float_controls(self):
        cases = (
            (5120, False, [-128, -127, 127], 1, [-128.0, -127.0, 127.0]),
            (5122, False, [-32768, -32767, 32767], 1, [-32768.0, -32767.0, 32767.0]),
            (5121, True, [0, 1, 255], 255, [0.0, 1 / 255, 1.0]),
            (5123, True, [0, 1, 65535], 65535, [0.0, 1 / 65535, 1.0]),
            (5126, False, [-1.25, 0.5, 2.0], 1.0, [-1.25, 0.5, 2.0]),
        )
        for ctype, normalized, raw, one, expected in cases:
            with self.subTest(component_type=ctype, normalized=normalized):
                part = self.read(make_glb(ctype, raw, one, normalized=normalized, interleaved=True))
                self.assertEqual([row[0] for row in part.positions], expected)
                self.assertEqual([row[0] for row in part.uvs], expected)
                self.assertEqual([row[1] for row in part.positions], [0.0, 1.0, 0.0])

    def test_cli_equivalent_signed_minimum_encodings(self):
        expected = ([(-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 0.0)],
                    [(-1.0, 0.0), (0.0, 1.0), (1.0, 0.0)])
        for ctype, minimum, maximum in ((5120, -128, 127), (5122, -32768, 32767)):
            for endpoint in (minimum, minimum + 1):
                with self.subTest(component_type=ctype, endpoint=endpoint):
                    result = self.assemble(make_glb(ctype, [endpoint, 0, maximum], maximum, normalized=True))
                    self.assertEqual(result, expected)

    def test_cli_sparse_signed_endpoints(self):
        for ctype, minimum, maximum in ((5120, -128, 127), (5122, -32768, 32767)):
            with self.subTest(component_type=ctype):
                result = self.assemble(make_glb(ctype, [minimum, 0, maximum], maximum,
                                               normalized=True, sparse="implicit"))
                self.assertEqual(result, ([(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)],
                                          [(-1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]))


if __name__ == "__main__":
    unittest.main()
