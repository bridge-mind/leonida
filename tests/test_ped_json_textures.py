"""Synthetic PNGs survive part JSON and the actual cull/assemble CLI pipeline."""
import base64
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import ped_assemble as pa

SLOTS = ("base_color_png", "normal_png", "metallic_roughness_png")


def png(r, g, b):
    def chunk(kind, payload):
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))
    return b"".join((
        b"\x89PNG\r\n\x1a\n", chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(bytes([0, r, g, b, 255]))), chunk(b"IEND", b""),
    ))


IMAGES = (png(30, 80, 150), png(128, 128, 255), png(0, 220, 10))


def part(material=None):
    return pa.MeshPart(
        "surface", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(0, 1, 2)],
        coord_space="y-up", material=material,
    )


def split_glb(data):
    length = struct.unpack_from("<I", data, 12)[0]
    doc = json.loads(data[20:20 + length])
    offset = 20 + length
    length, kind = struct.unpack_from("<II", data, offset)
    assert kind == pa.CHUNK_TYPE_BIN
    return doc, data[offset + 8:offset + 8 + length]


class PedJsonTextureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def roundtrip(self, source):
        path = self.root / "serialized.json"
        path.write_text(json.dumps(pa.part_to_dict(source)), encoding="utf-8", newline="\n")
        loaded, skeleton = pa.load_parts_from_path(str(path))
        self.assertIsNone(skeleton)
        self.assertEqual(len(loaded), 1)
        return loaded[0]

    def cli(self, *args):
        result = subprocess.run(
            [sys.executable, "-I", str(ROOT / "tools" / "ped_assemble.py"), *map(str, args)],
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_part_json_roundtrip_preserves_payloads_and_metadata(self):
        source = part(pa.MaterialSpec(
            name="paint", base_color_factor=(0.25, 0.5, 0.75, 0.625), metallic_factor=0.125,
            roughness_factor=0.375, alpha_mode="MASK", alpha_cutoff=0.25, double_sided=True,
            **dict(zip(SLOTS, IMAGES)),
        ))
        source.normals = [(0.0, 0.0, 1.0)] * 3
        source.uvs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
        source.colors = [(0.25, 0.5, 0.75, 1.0)] * 3
        source.translation = (2.0, 3.0, 4.0)
        before = copy.deepcopy(source)
        self.assertEqual(self.roundtrip(source), source)
        self.assertEqual(source, before)

    def test_each_slot_survives_without_creating_other_slots(self):
        for slot, payload in zip(SLOTS, IMAGES):
            with self.subTest(slot=slot):
                source = part(pa.MaterialSpec(**{slot: payload}))
                result = self.roundtrip(source)
                self.assertEqual(getattr(result.material, slot), payload)
                for other in set(SLOTS) - {slot}:
                    self.assertIsNone(getattr(result.material, other))

    def test_culled_json_cli_matches_direct_glb_for_inline_and_file_textures(self):
        for relative_files in (False, True):
            for shared in (False, True):
                with self.subTest(relative_files=relative_files, shared=shared):
                    root = self.root / f"{relative_files}-{shared}"
                    inputs = root / "inputs"
                    inputs.mkdir(parents=True)
                    payloads = (IMAGES[0],) * 3 if shared else IMAGES
                    textures = {}
                    for slot, payload in zip(SLOTS, payloads):
                        if relative_files:
                            (inputs / (slot + ".png")).write_bytes(payload)
                            textures[slot] = slot + ".png"
                        else:
                            textures[slot] = "base64:" + base64.b64encode(payload).decode("ascii")
                    target = {
                        "name": "surface", "coord_space": "y-up",
                        "positions": [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
                        "triangles": [[0, 1, 2], [3, 4, 5]],
                        "normals": [[0, 0, 1]] * 6,
                        "uvs": [[0, 0], [1, 0], [0, 1], [0.25, 0.25], [0.5, 0.25], [0.25, 0.5]],
                        "material": {"name": "paint", "base_color_factor": [0.25, 0.5, 0.75, 0.625],
                                     "metallic_factor": 0.125, "roughness_factor": 0.375,
                                     "alpha_mode": "MASK", "alpha_cutoff": 0.25, "double_sided": True,
                                     **textures},
                    }
                    reference = {"name": "reference", "coord_space": "y-up",
                                 "positions": target["positions"][:3], "triangles": [[0, 1, 2]]}
                    for name, value in (("target", target), ("reference", reference)):
                        (inputs / (name + ".json")).write_text(json.dumps(value), encoding="utf-8", newline="\n")
                    originals = {path: path.read_bytes() for path in inputs.iterdir()}
                    args = ("cull-overlap", "--target", inputs / "target.json", "--ref", inputs / "reference.json",
                            "--source-space", "y-up", "--no-standing-offset", "--compact-vertices")
                    direct, serialized, rebuilt = root / "direct.glb", root / "culled.json", root / "rebuilt.glb"
                    self.cli(*args, "-o", direct)
                    result = self.cli(*args, "-o", serialized)
                    self.assertIn("2 -> 1 tris", result.stdout)
                    self.cli("assemble", serialized, "-o", rebuilt, "--no-cull", "--no-standing-offset",
                             "--source-space", "y-up")
                    self.assertEqual(rebuilt.read_bytes(), direct.read_bytes())

                    doc, blob = split_glb(rebuilt.read_bytes())
                    self.assertEqual(len(doc["images"]), 1 if shared else 3)
                    embedded = []
                    for image in doc["images"]:
                        view = doc["bufferViews"][image["bufferView"]]
                        start = view.get("byteOffset", 0)
                        embedded.append(blob[start:start + view["byteLength"]])
                    self.assertEqual(set(embedded), set(payloads))
                    primitive = doc["meshes"][0]["primitives"][0]
                    self.assertEqual(doc["accessors"][primitive["indices"]]["count"], 3)
                    self.assertEqual(doc["accessors"][primitive["attributes"]["POSITION"]]["count"], 3)
                    restored, _ = pa.load_parts_from_path(str(serialized))
                    self.assertEqual(restored[0].positions, [(2.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 1.0, 0.0)])
                    self.assertEqual(restored[0].uvs, [(0.25, 0.25), (0.5, 0.25), (0.25, 0.5)])
                    self.assertEqual({path: path.read_bytes() for path in originals}, originals)

    def test_no_texture_serialization_is_unchanged(self):
        source = part()
        expected = {"name": "surface", "role": "other", "coord_space": "y-up", "translation": [0.0, 0.0, 0.0],
                    "positions": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], "triangles": [[0, 1, 2]]}
        self.assertEqual(pa.part_to_dict(source), expected)
        source.material = pa.MaterialSpec()
        expected["material"] = {"name": "default", "base_color_factor": [0.8, 0.8, 0.8, 1.0],
                                "metallic_factor": 0.0, "roughness_factor": 0.85,
                                "alpha_mode": "OPAQUE", "double_sided": False}
        self.assertEqual(pa.part_to_dict(source), expected)


if __name__ == "__main__":
    unittest.main()
