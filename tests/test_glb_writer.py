"""Unit tests for tools/glb_writer.py covering glTF 2.0 binary (.glb) container
header generation, chunk padding, buffer views, accessors, materials, textures,
OBJ parsing, and container validation.

All geometry, textures, and primitives are synthetic (zero game data).

Usage:
    python -I tests/test_glb_writer.py
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
import zlib
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS = os.path.join(ROOT, "tools")
if TOOLS not in sys.path:
    sys.path.insert(0, TOOLS)

import glb_writer


def _make_synthetic_png(
    width: int = 2,
    height: int = 2,
    rgba_bytes: bytes | None = None,
) -> bytes:
    """Generate a minimal valid RGBA PNG entirely via standard library struct/zlib."""
    if rgba_bytes is None:
        rgba_bytes = bytes([255, 0, 0, 255] * (width * height))
    assert len(rgba_bytes) == width * height * 4

    sig = b"\x89PNG\r\n\x1a\n"
    # IHDR: width, height, bit_depth=8, color_type=6 (RGBA), comp=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr = struct.pack(">I4s", len(ihdr_data), b"IHDR") + ihdr_data + struct.pack(">I", ihdr_crc)

    # Raw scanlines: 1 filter byte (0 = None) per row + width * 4 RGBA bytes
    raw_lines = bytearray()
    row_bytes = width * 4
    for y in range(height):
        raw_lines.append(0)
        raw_lines.extend(rgba_bytes[y * row_bytes:(y + 1) * row_bytes])

    idat_data = zlib.compress(bytes(raw_lines))
    idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
    idat = struct.pack(">I4s", len(idat_data), b"IDAT") + idat_data + struct.pack(">I", idat_crc)

    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend = struct.pack(">I4s", 0, b"IEND") + struct.pack(">I", iend_crc)
    return sig + ihdr + idat + iend


def _simple_triangle_primitive(
    *,
    has_normals: bool = True,
    has_uvs: bool = True,
    offset: float = 0.0,
) -> dict:
    """Return a single synthetic triangle primitive."""
    positions = [
        (0.0 + offset, 0.0, 0.0),
        (1.0 + offset, 0.0, 0.0),
        (0.0 + offset, 1.0, 0.0),
    ]
    indices = [0, 1, 2]
    prim: dict = {
        "positions": positions,
        "indices": indices,
    }
    if has_normals:
        prim["normals"] = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
    if has_uvs:
        prim["uvs"] = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)]
    return prim


class TestGlbContainerHeaderAndChunkPadding(unittest.TestCase):
    """Tests glTF 2.0 binary (.glb) 12-byte header and 8-byte chunk headers with padding."""

    def test_glb_header_12_bytes_structure(self):
        prim = _simple_triangle_primitive()
        glb_bytes = glb_writer.build_glb([prim])

        self.assertGreaterEqual(len(glb_bytes), 20)
        magic, version, total_length = struct.unpack_from("<4sII", glb_bytes, 0)
        self.assertEqual(magic, glb_writer.GLB_MAGIC)
        self.assertEqual(magic, b"glTF")
        self.assertEqual(version, glb_writer.GLB_VERSION)
        self.assertEqual(version, 2)
        self.assertEqual(total_length, len(glb_bytes))
        self.assertEqual(total_length % 4, 0)

    def test_json_chunk_header_and_space_padding(self):
        prim = _simple_triangle_primitive()
        glb_bytes = glb_writer.build_glb([prim])

        json_len, json_type = struct.unpack_from("<II", glb_bytes, 12)
        self.assertEqual(json_type, glb_writer.CHUNK_TYPE_JSON)
        self.assertEqual(json_type, 0x4E4F534A)
        self.assertEqual(json_len % 4, 0)

        json_raw = glb_bytes[20:20 + json_len]
        # Valid JSON when decoded as UTF-8
        gltf = json.loads(json_raw.decode("utf-8"))
        self.assertIsInstance(gltf, dict)
        self.assertEqual(gltf["asset"]["version"], "2.0")

        # Per glTF 2.0 spec, trailing padding bytes for JSON chunk MUST be ASCII spaces (0x20)
        stripped = json_raw.rstrip(b" ")
        pad_len = len(json_raw) - len(stripped)
        self.assertLess(pad_len, 4)
        if pad_len > 0:
            self.assertEqual(json_raw[-pad_len:], b" " * pad_len)

    def test_bin_chunk_header_and_null_padding(self):
        prim = _simple_triangle_primitive()
        glb_bytes = glb_writer.build_glb([prim])

        json_len, _ = struct.unpack_from("<II", glb_bytes, 12)
        bin_offset = 20 + json_len
        self.assertLessEqual(bin_offset + 8, len(glb_bytes))

        bin_len, bin_type = struct.unpack_from("<II", glb_bytes, bin_offset)
        self.assertEqual(bin_type, glb_writer.CHUNK_TYPE_BIN)
        self.assertEqual(bin_type, 0x004E4942)
        self.assertEqual(bin_len % 4, 0)

        bin_data = glb_bytes[bin_offset + 8:bin_offset + 8 + bin_len]
        self.assertEqual(len(bin_data), bin_len)
        self.assertEqual(bin_offset + 8 + bin_len, len(glb_bytes))

    def test_chunk_budget_exact_sum(self):
        prim = _simple_triangle_primitive()
        glb_bytes = glb_writer.build_glb([prim])

        _, _, total_length = struct.unpack_from("<4sII", glb_bytes, 0)
        json_len, _ = struct.unpack_from("<II", glb_bytes, 12)
        bin_offset = 20 + json_len
        bin_len, _ = struct.unpack_from("<II", glb_bytes, bin_offset)

        expected_total = 12 + 8 + json_len + 8 + bin_len
        self.assertEqual(total_length, expected_total)
        self.assertEqual(len(glb_bytes), expected_total)

    def test_custom_generator_metadata(self):
        prim = _simple_triangle_primitive()
        custom_gen = "CustomTestGen/1.0"
        glb_bytes = glb_writer.build_glb([prim], generator=custom_gen)
        summary = glb_writer.read_glb_summary(glb_bytes)
        self.assertEqual(summary["json"]["asset"]["generator"], custom_gen)


class TestGlbBufferViewsAndAccessors(unittest.TestCase):
    """Tests glTF bufferViews and accessors for positions, normals, UVs, and indices."""

    def test_buffer_views_alignment_and_targets(self):
        prim = _simple_triangle_primitive(has_normals=True, has_uvs=True)
        glb_bytes = glb_writer.build_glb([prim])
        summary = glb_writer.read_glb_summary(glb_bytes)
        gltf = summary["json"]
        bin_bytes = summary["bin_bytes"]

        buffer_views = gltf["bufferViews"]
        self.assertGreaterEqual(len(buffer_views), 4)  # pos, nrm, uv, idx

        for i, bv in enumerate(buffer_views):
            self.assertEqual(bv.get("buffer"), 0, f"bufferView[{i}] buffer must be 0")
            offset = bv.get("byteOffset", 0)
            length = bv["byteLength"]
            self.assertEqual(offset % 4, 0, f"bufferView[{i}] offset {offset} must be 4-byte aligned")
            self.assertLessEqual(offset + length, bin_bytes, f"bufferView[{i}] exceeds BIN length")

        mesh_prim = gltf["meshes"][0]["primitives"][0]
        pos_bv = buffer_views[gltf["accessors"][mesh_prim["attributes"]["POSITION"]]["bufferView"]]
        nrm_bv = buffer_views[gltf["accessors"][mesh_prim["attributes"]["NORMAL"]]["bufferView"]]
        uv_bv = buffer_views[gltf["accessors"][mesh_prim["attributes"]["TEXCOORD_0"]]["bufferView"]]
        idx_bv = buffer_views[gltf["accessors"][mesh_prim["indices"]]["bufferView"]]

        self.assertEqual(pos_bv.get("target"), glb_writer.TARGET_ARRAY_BUFFER)
        self.assertEqual(nrm_bv.get("target"), glb_writer.TARGET_ARRAY_BUFFER)
        self.assertEqual(uv_bv.get("target"), glb_writer.TARGET_ARRAY_BUFFER)
        self.assertEqual(idx_bv.get("target"), glb_writer.TARGET_ELEMENT_ARRAY_BUFFER)

    def test_position_accessor_min_max_and_float32(self):
        positions = [
            (-10.5, 2.0, 5.0),
            (20.0, -4.5, 0.0),
            (0.0, 15.25, -8.0),
        ]
        prim = {
            "positions": positions,
            "indices": [0, 1, 2],
        }
        glb_bytes = glb_writer.build_glb([prim])
        summary = glb_writer.read_glb_summary(glb_bytes)
        gltf = summary["json"]

        pos_acc_idx = gltf["meshes"][0]["primitives"][0]["attributes"]["POSITION"]
        pos_acc = gltf["accessors"][pos_acc_idx]

        self.assertEqual(pos_acc["componentType"], glb_writer.COMPONENT_FLOAT)
        self.assertEqual(pos_acc["componentType"], 5126)
        self.assertEqual(pos_acc["type"], "VEC3")
        self.assertEqual(pos_acc["count"], 3)
        self.assertEqual(pos_acc["min"], [-10.5, -4.5, -8.0])
        self.assertEqual(pos_acc["max"], [20.0, 15.25, 5.0])

        # Verify decoded floats from BIN chunk
        json_len, _ = struct.unpack_from("<II", glb_bytes, 12)
        bin_payload = glb_bytes[20 + json_len + 8:]
        bv = gltf["bufferViews"][pos_acc["bufferView"]]
        raw_floats = struct.unpack_from("<9f", bin_payload, bv.get("byteOffset", 0))
        for got, exp in zip(raw_floats, [-10.5, 2.0, 5.0, 20.0, -4.5, 0.0, 0.0, 15.25, -8.0]):
            self.assertAlmostEqual(got, exp, places=5)

    def test_normal_accessor_automatic_computation(self):
        # XY triangle with vertices in counter-clockwise order: normal should point in +Z (0, 0, 1)
        prim = {
            "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "indices": [0, 1, 2],
        }
        glb_bytes = glb_writer.build_glb([prim])
        summary = glb_writer.read_glb_summary(glb_bytes)
        gltf = summary["json"]

        nrm_acc_idx = gltf["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        nrm_acc = gltf["accessors"][nrm_acc_idx]
        self.assertEqual(nrm_acc["componentType"], glb_writer.COMPONENT_FLOAT)
        self.assertEqual(nrm_acc["type"], "VEC3")
        self.assertEqual(nrm_acc["count"], 3)

        json_len, _ = struct.unpack_from("<II", glb_bytes, 12)
        bin_payload = glb_bytes[20 + json_len + 8:]
        bv = gltf["bufferViews"][nrm_acc["bufferView"]]
        raw_normals = struct.unpack_from("<9f", bin_payload, bv.get("byteOffset", 0))
        for i in range(3):
            nx, ny, nz = raw_normals[i * 3:(i + 1) * 3]
            self.assertAlmostEqual(nx, 0.0, places=5)
            self.assertAlmostEqual(ny, 0.0, places=5)
            self.assertAlmostEqual(nz, 1.0, places=5)

    def test_indices_uint16_vs_uint32_threshold(self):
        # Small primitive: <= 65535 vertices -> COMPONENT_UNSIGNED_SHORT (5123)
        small_prim = _simple_triangle_primitive()
        glb_bytes = glb_writer.build_glb([small_prim])
        summary = glb_writer.read_glb_summary(glb_bytes)
        idx_acc = summary["json"]["accessors"][summary["json"]["meshes"][0]["primitives"][0]["indices"]]
        self.assertEqual(idx_acc["componentType"], glb_writer.COMPONENT_UNSIGNED_SHORT)
        self.assertEqual(idx_acc["componentType"], 5123)
        self.assertEqual(idx_acc["type"], "SCALAR")
        self.assertEqual(idx_acc["count"], 3)

        # Large primitive: 65536 vertices -> COMPONENT_UNSIGNED_INT (5125)
        vert_count = 65536
        large_positions = [(float(i % 10), float(i // 10), 0.0) for i in range(vert_count)]
        # Triangle referencing index 0, 1, and 65535
        large_indices = [0, 1, 65535]
        large_prim = {
            "positions": large_positions,
            "indices": large_indices,
        }
        glb_bytes_large = glb_writer.build_glb([large_prim])
        summary_large = glb_writer.read_glb_summary(glb_bytes_large)
        idx_acc_large = summary_large["json"]["accessors"][
            summary_large["json"]["meshes"][0]["primitives"][0]["indices"]
        ]
        self.assertEqual(idx_acc_large["componentType"], glb_writer.COMPONENT_UNSIGNED_INT)
        self.assertEqual(idx_acc_large["componentType"], 5125)
        self.assertEqual(idx_acc_large["count"], 3)

        # Unpack uint32 from BIN payload
        json_len, _ = struct.unpack_from("<II", glb_bytes_large, 12)
        bin_payload = glb_bytes_large[20 + json_len + 8:]
        bv = summary_large["json"]["bufferViews"][idx_acc_large["bufferView"]]
        unpacked_idx = struct.unpack_from("<3I", bin_payload, bv.get("byteOffset", 0))
        self.assertEqual(list(unpacked_idx), [0, 1, 65535])

    def test_append_buffer_view_alignment_logic(self):
        bin_buf = bytearray(b"\x01\x02\x03")  # len = 3, unaligned
        buffer_views: list[dict] = []
        payload = b"ABC"  # len = 3
        idx = glb_writer._append_buffer_view(bin_buf, buffer_views, payload, target=123)

        self.assertEqual(idx, 0)
        self.assertEqual(len(buffer_views), 1)
        bv = buffer_views[0]
        self.assertEqual(bv["byteOffset"] % 4, 0)
        self.assertEqual(bv["byteLength"], 3)
        self.assertEqual(bv["target"], 123)
        self.assertEqual(len(bin_buf) % 4, 0)


class TestGlbMaterialsAndTextures(unittest.TestCase):
    """Tests glTF material generation, embedded PNG textures, and texture deduplication."""

    def test_untextured_default_clay_material(self):
        prim = _simple_triangle_primitive(has_uvs=False)
        glb_bytes = glb_writer.build_glb([prim], materials=None)
        summary = glb_writer.read_glb_summary(glb_bytes)
        gltf = summary["json"]

        self.assertEqual(len(gltf["materials"]), 1)
        mat = gltf["materials"][0]
        self.assertEqual(mat["name"], "default_clay")
        pbr = mat["pbrMetallicRoughness"]
        self.assertEqual(pbr["baseColorFactor"], glb_writer.DEFAULT_CLAY_COLOR)
        self.assertEqual(pbr["metallicFactor"], glb_writer.DEFAULT_METALLIC)
        self.assertEqual(pbr["roughnessFactor"], glb_writer.DEFAULT_ROUGHNESS)
        self.assertNotIn("baseColorTexture", pbr)
        self.assertNotIn("textures", gltf)
        self.assertNotIn("images", gltf)
        self.assertNotIn("samplers", gltf)

    def test_texture_deduplication_by_sha256(self):
        png_data = _make_synthetic_png(2, 2)
        prim1 = dict(_simple_triangle_primitive(offset=0.0), material="mat_one")
        prim2 = dict(_simple_triangle_primitive(offset=5.0), material="mat_two")

        materials = {
            "mat_one": {"name": "mat_one", "png_bytes": png_data},
            "mat_two": {"name": "mat_two", "png_bytes": png_data},  # identical PNG
        }

        glb_bytes = glb_writer.build_glb([prim1, prim2], materials=materials)
        summary = glb_writer.read_glb_summary(glb_bytes)
        gltf = summary["json"]

        # Deduplication must result in exactly 1 image, 1 texture, and 1 sampler
        self.assertEqual(len(gltf["images"]), 1)
        self.assertEqual(len(gltf["textures"]), 1)
        self.assertEqual(len(gltf["samplers"]), 1)
        self.assertEqual(len(gltf["materials"]), 2)
        self.assertEqual(gltf["images"][0]["mimeType"], "image/png")

        # Image bufferView must not have a target and must be 4-byte aligned
        img_bv_idx = gltf["images"][0]["bufferView"]
        img_bv = gltf["bufferViews"][img_bv_idx]
        self.assertNotIn("target", img_bv)
        self.assertEqual(img_bv.get("byteOffset", 0) % 4, 0)
        self.assertEqual(img_bv["byteLength"], len(png_data))

        # Both materials reference texture index 0
        self.assertEqual(gltf["materials"][0]["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0)
        self.assertEqual(gltf["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"]["index"], 0)

    def test_distinct_textures_create_multiple_images(self):
        png1 = _make_synthetic_png(1, 1, bytes([255, 0, 0, 255]))
        png2 = _make_synthetic_png(1, 1, bytes([0, 255, 0, 255]))
        prim1 = dict(_simple_triangle_primitive(offset=0.0), material="m1")
        prim2 = dict(_simple_triangle_primitive(offset=5.0), material="m2")

        materials = {
            "m1": {"png_bytes": png1},
            "m2": {"png_bytes": png2},
        }

        glb_bytes = glb_writer.build_glb([prim1, prim2], materials=materials)
        summary = glb_writer.read_glb_summary(glb_bytes)
        gltf = summary["json"]

        self.assertEqual(len(gltf["images"]), 2)
        self.assertEqual(len(gltf["textures"]), 2)

    def test_material_without_uvs_omits_texture(self):
        png = _make_synthetic_png(1, 1)
        # Primitive without UV coordinates
        prim = dict(_simple_triangle_primitive(has_uvs=False), material="m1")
        materials = {"m1": {"name": "m1", "png_bytes": png}}

        glb_bytes = glb_writer.build_glb([prim], materials=materials)
        summary = glb_writer.read_glb_summary(glb_bytes)
        gltf = summary["json"]

        self.assertNotIn("textures", gltf)
        self.assertNotIn("images", gltf)
        self.assertNotIn("baseColorTexture", gltf["materials"][0]["pbrMetallicRoughness"])

    def test_material_double_sided_and_color_factor(self):
        prim = dict(_simple_triangle_primitive(has_uvs=False), material="m1")
        materials = {
            "m1": {
                "base_color_factor": (0.2, 0.4, 0.6),  # 3 elements -> alpha 1.0
                "double_sided": True,
            }
        }
        glb_bytes = glb_writer.build_glb([prim], materials=materials)
        summary = glb_writer.read_glb_summary(glb_bytes)
        mat = summary["json"]["materials"][0]
        self.assertTrue(mat.get("doubleSided"))
        self.assertEqual(mat["pbrMetallicRoughness"]["baseColorFactor"], [0.2, 0.4, 0.6, 1.0])

    def _alpha_materials(self, materials: dict, *, has_uvs: bool = False) -> list[dict]:
        prims = [
            dict(_simple_triangle_primitive(has_uvs=has_uvs, offset=5.0 * i), material=key)
            for i, key in enumerate(materials)
        ]
        return glb_writer.read_glb_summary(glb_writer.build_glb(prims, materials=materials))["json"]["materials"]

    def test_alpha_mode_inferred_and_explicit(self):
        red = (1.0, 0.0, 0.0)
        mats = self._alpha_materials({
            "plain": {"base_color_factor": red},
            "half": {"base_color_factor": red + (0.5,)},
            "opaque": {"base_color_factor": red + (0.5,), "alpha_mode": "OPAQUE"},
            "blend": {"base_color_factor": red, "alpha_mode": "BLEND"},
            "mask_default": {"alpha_mode": "MASK"},
            "mask_low": {"base_color_factor": red + (0.4,), "alpha_mode": "MASK", "alpha_cutoff": 0.5},
        })
        modes = {m["name"]: (m.get("alphaMode"), m.get("alphaCutoff")) for m in mats}
        self.assertEqual(modes, {
            "plain": (None, None),
            "half": ("BLEND", None),
            "opaque": (None, None),
            "blend": ("BLEND", None),
            "mask_default": ("MASK", None),
            "mask_low": ("MASK", 0.5),
        })
        alphas = [m["pbrMetallicRoughness"]["baseColorFactor"][3] for m in mats]
        self.assertEqual(alphas, [1.0, 0.5, 0.5, 1.0, 1.0, 0.4])

    def test_opaque_materials_stay_byte_identical(self):
        prim = dict(_simple_triangle_primitive(), material="m1")
        plain = glb_writer.build_glb([prim], {"m1": {"base_color_factor": (1.0, 0.0, 0.0)}})
        explicit = glb_writer.build_glb(
            [prim], {"m1": {"base_color_factor": (1.0, 0.0, 0.0, 1.0), "alpha_mode": "OPAQUE"}},
        )
        self.assertEqual(plain, explicit)
        self.assertNotIn(b"alphaMode", plain)
        self.assertNotIn(b"alphaMode", glb_writer.build_glb([prim]))

    def test_alpha_mode_on_deduplicated_textured_material(self):
        png = _make_synthetic_png(1, 1, bytes([0, 255, 0, 0]))
        materials = {
            "leaf": {"png_bytes": png, "alpha_mode": "MASK", "alpha_cutoff": 0.25},
            "no_mode": {"png_bytes": png},  # transparent texel alone never selects a mode
        }
        prims = [
            dict(_simple_triangle_primitive(offset=o), material=key)
            for o, key in ((0.0, "leaf"), (5.0, "leaf"), (10.0, "no_mode"))
        ]
        gltf = glb_writer.read_glb_summary(glb_writer.build_glb(prims, materials=materials))["json"]
        leaf, no_mode = gltf["materials"]
        self.assertEqual((leaf["alphaMode"], leaf["alphaCutoff"]), ("MASK", 0.25))
        self.assertNotIn("alphaMode", no_mode)
        self.assertEqual([p["material"] for p in gltf["meshes"][0]["primitives"]], [0, 0, 1])
        self.assertEqual(len(gltf["textures"]), 1)

    def test_invalid_alpha_specs_raise_error(self):
        prim = dict(_simple_triangle_primitive(has_uvs=False), material="m1")
        for spec in (
            {"alpha_mode": "blend"},
            {"alpha_mode": "TRANSPARENT"},
            {"alpha_mode": 1},
            {"alpha_cutoff": 0.5},
            {"alpha_mode": "BLEND", "alpha_cutoff": 0.5},
            {"alpha_mode": "MASK", "alpha_cutoff": float("nan")},
            {"alpha_mode": "MASK", "alpha_cutoff": -0.1},
            {"alpha_mode": "MASK", "alpha_cutoff": True},
            {"alpha_mode": "MASK", "alpha_cutoff": "0.5"},
        ):
            with self.subTest(spec=spec), self.assertRaises(glb_writer.GlbError):
                glb_writer.build_glb([prim], {"m1": spec})


class TestGlbWriterValidationAndErrors(unittest.TestCase):
    """Tests error cases and validation in build_glb and helper functions."""

    def test_each_position_has_three_components_before_normals_are_generated(self):
        # The total is nine floats, but flattening these rows moves 9 into the next vertex's X.
        positions = [(0.0, 0.0, 0.0, 9.0), (1.0, 0.0), (0.0, 1.0, 0.0)]
        for code in (glb_writer._FLOAT32_CODE, None):
            for has_normals in (False, True):
                with self.subTest(packing=code, has_normals=has_normals):
                    prim = _simple_triangle_primitive(has_normals=has_normals)
                    prim["positions"] = positions
                    with mock.patch.object(glb_writer, "_FLOAT32_CODE", code):
                        with self.assertRaisesRegex(glb_writer.GlbError, "expected 3 floats per vertex"):
                            glb_writer.build_glb([prim])

    def test_invalid_indices_without_normals_raise_glb_error(self):
        for invalid in (3, -4):
            with self.subTest(index=invalid):
                prim = _simple_triangle_primitive(has_normals=False)
                prim["indices"] = [0, 1, invalid]
                with self.assertRaisesRegex(glb_writer.GlbError, "index out of range"):
                    glb_writer.build_glb([prim])

    def test_valid_triangle_packing_with_generated_and_explicit_normals(self):
        positions = struct.pack("<9f", 0, 0, 0, 1, 0, 0, 0, 1, 0)
        normals = struct.pack("<9f", 0, 0, 1, 0, 0, 1, 0, 0, 1)
        indices = struct.pack("<3H", 0, 1, 2)
        for code in (glb_writer._FLOAT32_CODE, None):
            for has_normals in (False, True):
                with self.subTest(packing=code, has_normals=has_normals):
                    prim = _simple_triangle_primitive(has_normals=has_normals)
                    with mock.patch.object(glb_writer, "_FLOAT32_CODE", code):
                        packed = glb_writer.pack_primitive(prim)
                    self.assertEqual(packed["positions"], positions)
                    self.assertEqual(packed["normals"], normals)
                    self.assertEqual(packed["indices"], indices)
                    self.assertEqual(packed["min"], [0.0, 0.0, 0.0])
                    self.assertEqual(packed["max"], [1.0, 1.0, 0.0])

    def test_malformed_positions_do_not_replace_an_existing_glb(self):
        prim = _simple_triangle_primitive()
        prim["positions"] = [(0.0, 0.0, 0.0, 9.0), (1.0, 0.0), (0.0, 1.0, 0.0)]
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "model.glb")
            previous = glb_writer.build_glb([_simple_triangle_primitive()])
            with open(path, "wb") as f:
                f.write(previous)
            with self.assertRaisesRegex(glb_writer.GlbError, "expected 3 floats per vertex"):
                glb_writer.write_glb(path, [prim])
            with open(path, "rb") as f:
                self.assertEqual(f.read(), previous)
            self.assertEqual(os.listdir(folder), ["model.glb"])

    def test_empty_primitives_raises_error(self):
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([])

        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([{"positions": [], "indices": []}])

    def test_non_multiple_of_3_indices_raises_error(self):
        prim = {
            "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
            "indices": [0, 1],  # 2 indices
        }
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([prim])

    def test_out_of_bounds_indices_raise_error(self):
        prim_negative = {
            "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "normals": [(0.0, 0.0, 1.0)] * 3,
            "indices": [-1, 0, 1],
        }
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([prim_negative])

        prim_exceed = {
            "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "normals": [(0.0, 0.0, 1.0)] * 3,
            "indices": [0, 1, 3],  # 3 >= len(positions)
        }
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([prim_exceed])

    def test_mismatched_attribute_lengths_raise_error(self):
        prim_bad_normals = {
            "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "normals": [(0.0, 1.0, 0.0)],  # 1 != 3
            "indices": [0, 1, 2],
        }
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([prim_bad_normals])

        prim_bad_uvs = {
            "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "uvs": [(0.0, 0.0)],  # 1 != 3
            "indices": [0, 1, 2],
        }
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([prim_bad_uvs])

    def test_non_finite_floats_raise_error(self):
        prim_nan = {
            "positions": [(float("nan"), 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "indices": [0, 1, 2],
        }
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([prim_nan])

        prim_inf = {
            "positions": [(float("inf"), 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
            "indices": [0, 1, 2],
        }
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.build_glb([prim_inf])

    def test_invalid_base_color_factors_raise_error(self):
        prim = _simple_triangle_primitive()
        bad_factors = [
            "red",
            [1.0, 0.0],           # only 2 elements
            [1.0, 0.0, 0.0, 0.0, 1.0],  # 5 elements
            [-0.1, 0.5, 0.5, 1.0],      # negative
            [1.1, 0.5, 0.5, 1.0],       # > 1.0
            [float("nan"), 0.5, 0.5, 1.0],
        ]
        for bad in bad_factors:
            with self.subTest(bad=bad):
                with self.assertRaises(glb_writer.GlbError):
                    glb_writer.build_glb([prim], materials={"": {"base_color_factor": bad}})


class TestReadGlbSummaryValidation(unittest.TestCase):
    """Tests container framing and validation in read_glb_summary."""

    @staticmethod
    def _container(doc, *chunks):
        json_bytes = json.dumps(doc).encode("utf-8")
        json_bytes += b" " * ((-len(json_bytes)) % 4)
        body = b"".join(
            struct.pack("<I4s", len(payload), kind) + payload
            for kind, payload in ((b"JSON", json_bytes), *chunks)
        )
        return struct.pack("<4sII", b"glTF", 2, 12 + len(body)) + body

    def test_non_bytes_input_rejected(self):
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary("not bytes")  # type: ignore

    def test_truncated_header_rejected(self):
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(b"glTF\x02\x00\x00\x00")

    def test_corrupt_magic_rejected(self):
        raw = b"BADM" + struct.pack("<II", 2, 20) + struct.pack("<II", 0, glb_writer.CHUNK_TYPE_JSON)
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_invalid_version_rejected(self):
        raw = b"glTF" + struct.pack("<II", 1, 20) + struct.pack("<II", 0, glb_writer.CHUNK_TYPE_JSON)
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_length_mismatch_rejected(self):
        raw = b"glTF" + struct.pack("<II", 2, 100) + struct.pack("<II", 0, glb_writer.CHUNK_TYPE_JSON)
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_unaligned_total_length_rejected(self):
        raw = b"glTF" + struct.pack("<II", 2, 21) + struct.pack("<II", 0, glb_writer.CHUNK_TYPE_JSON) + b"\x00"
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_first_chunk_not_json_rejected(self):
        raw = b"glTF" + struct.pack("<II", 2, 20) + struct.pack("<II", 0, glb_writer.CHUNK_TYPE_BIN)
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_unaligned_json_chunk_length_rejected(self):
        raw = b"glTF" + struct.pack("<II", 2, 24) + struct.pack("<II", 1, glb_writer.CHUNK_TYPE_JSON) + b"{\x00\x00\x00"
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_malformed_json_syntax_rejected(self):
        raw = b"glTF" + struct.pack("<II", 2, 24) + struct.pack("<II", 4, glb_writer.CHUNK_TYPE_JSON) + b"BAD!"
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_json_root_not_object_rejected(self):
        jb = b"[1, 2, 3] "
        total = 12 + 8 + len(jb)
        raw = b"glTF" + struct.pack("<II", 2, total) + struct.pack("<II", len(jb), glb_writer.CHUNK_TYPE_JSON) + jb
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)

    def test_extension_chunk_is_ignored_with_or_without_bin(self):
        for embedded in (False, True):
            with self.subTest(embedded=embedded):
                buffer = {"byteLength": 4}
                chunks = [(b"BIN\0", b"\x00" * 4)] if embedded else []
                if not embedded:
                    buffer["uri"] = "sample.bin"
                doc = {"asset": {"version": "2.0"}, "buffers": [buffer]}
                raw = self._container(doc, *chunks, (b"EXT!", b"data"))
                summary = glb_writer.read_glb_summary(raw)
                self.assertEqual(summary["json"], doc)
                self.assertEqual(summary["bin_bytes"], 4 if embedded else 0)

    def test_known_chunk_order_and_cardinality(self):
        doc = {"asset": {"version": "2.0"}}
        binary, extension = (b"BIN\0", b"\x00" * 4), (b"EXT!", b"data")
        for label, chunks, message in (
            ("late BIN", (extension, binary), "BIN"),
            ("duplicate BIN", (binary, binary), "BIN chunk must be second"),
            ("duplicate JSON", (binary, (b"JSON", b"{}  ")), "duplicate GLB JSON chunk"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(glb_writer.GlbError, message):
                    glb_writer.read_glb_summary(self._container(doc, *chunks))

    def test_extension_chunk_cannot_overrun_container(self):
        doc = {"asset": {"version": "2.0"}}
        raw = bytearray(self._container(doc, (b"BIN\0", b"\x00" * 4), (b"EXT!", b"data")))
        struct.pack_into("<I", raw, len(raw) - 12, 8)
        with self.assertRaisesRegex(glb_writer.GlbError, "exceeds.*GLB"):
            glb_writer.read_glb_summary(raw)

    def test_buffer_view_exceeding_bin_rejected(self):
        doc = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 4}],
            "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 8}],  # 8 > 4
        }
        jb = json.dumps(doc).encode("utf-8")
        jb += b" " * ((-len(jb)) % 4)
        bb = b"\x00\x00\x00\x00"
        total = 12 + 8 + len(jb) + 8 + len(bb)
        raw = (
            b"glTF"
            + struct.pack("<II", 2, total)
            + struct.pack("<II", len(jb), glb_writer.CHUNK_TYPE_JSON)
            + jb
            + struct.pack("<II", len(bb), glb_writer.CHUNK_TYPE_BIN)
            + bb
        )
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.read_glb_summary(raw)


class TestWriteGlbFilesystem(unittest.TestCase):
    """Tests writing GLB files to disk, nested directory creation, and return stats."""

    def test_write_glb_nested_directory_creation_and_stats(self):
        prim = _simple_triangle_primitive()
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_file = os.path.join(tmp_dir, "nested", "sub", "output.glb")
            stats = glb_writer.write_glb(out_file, [prim])

            self.assertTrue(os.path.isfile(out_file))
            self.assertEqual(stats["triangles"], 1)
            self.assertEqual(stats["vertices"], 3)
            self.assertEqual(stats["primitives"], 1)
            self.assertEqual(stats["textures"], 0)
            self.assertEqual(stats["bounds"], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0])

            with open(out_file, "rb") as f:
                disk_bytes = f.read()
            self.assertEqual(stats["bytes"], len(disk_bytes))

            # Round-trip summary verification
            summary = glb_writer.read_glb_summary(disk_bytes)
            self.assertEqual(summary["version"], 2)


class TestObjParserAndBounds(unittest.TestCase):
    """Tests Wavefront OBJ parsing, coordinate transforms, UV flipping, and bounds."""

    def test_utf8_bom_preserves_first_obj_record(self) -> None:
        text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        expected = glb_writer.parse_obj(text, is_path=False)
        self.assertEqual(expected[0]["indices"], [0, 1, 2])

        with tempfile.TemporaryDirectory() as folder:
            source = os.path.join(folder, "triangle.obj")
            with open(source, "w", encoding="utf-8-sig", newline="\n") as f:
                f.write(text)
            with self.subTest(source="file"):
                self.assertEqual(glb_writer.parse_obj(source), expected)

        with self.subTest(source="in-memory text"):
            self.assertEqual(glb_writer.parse_obj("\ufeff" + text, is_path=False), expected)

    def test_parse_obj_zup_to_yup(self):
        # OBJ coordinate (1, 2, 3) in Z-up becomes (1, 3, -2) in glTF Y-up
        obj_text = "\n".join([
            "v 1 2 3",
            "v 2 2 3",
            "v 1 3 3",
            "f 1 2 3",
        ])
        prims = glb_writer.parse_obj(obj_text, is_path=False, up_axis="z")
        self.assertEqual(len(prims), 1)
        positions = prims[0]["positions"]
        self.assertIn((1.0, 3.0, -2.0), positions)
        self.assertIn((2.0, 3.0, -2.0), positions)
        self.assertIn((1.0, 3.0, -3.0), positions)

    def test_parse_obj_yup_identity(self):
        obj_text = "\n".join([
            "v 1 2 3",
            "v 2 2 3",
            "v 1 3 3",
            "f 1 2 3",
        ])
        prims = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(len(prims), 1)
        positions = prims[0]["positions"]
        self.assertEqual(positions, [(1.0, 2.0, 3.0), (2.0, 2.0, 3.0), (1.0, 3.0, 3.0)])

    def test_parse_obj_uv_v_flip(self):
        # Wavefront UV (u, v) -> glTF UV (u, 1.0 - v)
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "vt 0.25 0.75",
            "vt 0.5 0.2",
            "vt 0.0 1.0",
            "f 1/1 2/2 3/3",
        ])
        prims = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        uvs = prims[0]["uvs"]
        self.assertAlmostEqual(uvs[0][0], 0.25)
        self.assertAlmostEqual(uvs[0][1], 0.25)  # 1.0 - 0.75
        self.assertAlmostEqual(uvs[1][0], 0.5)
        self.assertAlmostEqual(uvs[1][1], 0.8)   # 1.0 - 0.2
        self.assertAlmostEqual(uvs[2][0], 0.0)
        self.assertAlmostEqual(uvs[2][1], 0.0)   # 1.0 - 1.0

    def test_glb_generated_normals_split_at_explicit_obj_smoothing_boundaries(self) -> None:
        vertices = ["v 0 0 0", "v 1 0 0", "v 0 1 0", "v 0 0 1"]
        for directives in (("s off", "s off"), ("s 0", "s 0"), ("s 1", "s 2")):
            with self.subTest(directives=directives):
                obj = "\n".join([*vertices, directives[0], "f 1 2 3", directives[1], "f 2 1 4"])
                prims = glb_writer.parse_obj(obj, is_path=False, up_axis="y")
                data = glb_writer.build_glb(prims)
                gltf = glb_writer.read_glb_summary(data)["json"]
                mesh = gltf["meshes"][0]["primitives"][0]
                accessor = gltf["accessors"][mesh["attributes"]["NORMAL"]]
                view = gltf["bufferViews"][accessor["bufferView"]]
                json_length = struct.unpack_from("<I", data, 12)[0]
                start = 28 + json_length + view["byteOffset"]
                normals = list(struct.iter_unpack("<3f", data[start:start + view["byteLength"]]))
                self.assertEqual(normals, [(0.0, 0.0, 1.0)] * 3 + [(0.0, 1.0, 0.0)] * 3)

    def test_glb_preserves_finite_obj_normal_directions(self) -> None:
        obj = """\
v 0 0 0
v 1 0 0
v 0 1 0
vn 1e-200 0 0
vn 1.7e308 0 1.7e308
vn 5e-324 0 5e-324
f 1//1 2//2 3//3
"""
        primitives = glb_writer.parse_obj(obj, is_path=False, up_axis="y")
        data = glb_writer.build_glb(primitives)
        gltf = glb_writer.read_glb_summary(data)["json"]
        normal_idx = gltf["meshes"][0]["primitives"][0]["attributes"]["NORMAL"]
        normals = gltf["accessors"][normal_idx]
        view = gltf["bufferViews"][normals["bufferView"]]
        json_len, = struct.unpack_from("<I", data, 12)
        offset = 28 + json_len + view.get("byteOffset", 0) + normals.get("byteOffset", 0)
        values = struct.unpack_from("<9f", data, offset)

        h = math.sqrt(0.5)
        for i, expected in enumerate((1.0, 0.0, 0.0, h, 0.0, h, h, 0.0, h)):
            with self.subTest(component=i):
                self.assertAlmostEqual(values[i], expected, places=7)

    def test_parse_obj_generated_normals_cross_uv_seams(self):
        # The second triangle has twice the area of the first and a different UV
        # on each shared corner. They contribute +2Y and +Z to the shared normals.
        expected = (0.0, 2 / math.sqrt(5), 1 / math.sqrt(5))
        for suffix in ("", "/1"):
            with self.subTest(explicit_normal=bool(suffix)):
                obj_text = "\n".join([
                    "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 0 0 2",
                    "vt 0 0", "vt 1 0", "vt 0 1", "vt .75 0", "vt .25 0", "vt .25 1",
                    "vn 1 0 0",
                    "s 1",
                    "f 1/1 2/2 3/3",
                    f"f 2/4 1/5{suffix} 4/6",
                ])
                prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
                self.assertEqual(len(prim["positions"]), 6)
                normals = {
                    (position, uv): normal
                    for position, uv, normal in zip(prim["positions"], prim["uvs"], prim["normals"])
                }
                shared_edge = [
                    ((0.0, 0.0, 0.0), (0.0, 1.0), expected),
                    ((0.0, 0.0, 0.0), (0.25, 1.0), (1.0, 0.0, 0.0) if suffix else expected),
                    ((1.0, 0.0, 0.0), (1.0, 1.0), expected),
                    ((1.0, 0.0, 0.0), (0.75, 1.0), expected),
                ]
                for position, uv, expected_normal in shared_edge:
                    for actual, target in zip(normals[position, uv], expected_normal):
                        self.assertAlmostEqual(actual, target)

    def test_parse_obj_explicit_faces_preserve_generated_hard_edge(self):
        # A top face without normals shares two positions with a fully normaled side.
        vertices = ["v 0 1 0", "v 1 1 0", "v 1 1 1", "v 0 1 1", "v 0 0 1", "v 1 0 1"]
        top = "f 1 4 3 2"
        side = "f 5//1 6//1 3//1 4//1"
        for faces in ((top, side), (side, top)):
            with self.subTest(first_face=faces[0]):
                obj_text = "\n".join([*vertices, "vn 0 0 1", *faces])
                prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
                normals_by_face = [
                    {(0.0, 1.0, 0.0)} if face == top else {(0.0, 0.0, 1.0)}
                    for face in faces
                ]
                self.assertEqual(len(prim["positions"]), 8)
                self.assertEqual(len(prim["indices"]), 12)
                for start, expected in zip((0, 6), normals_by_face):
                    actual = {prim["normals"][i] for i in prim["indices"][start:start + 6]}
                    self.assertEqual(actual, expected)

    def test_parse_obj_quad_fan_triangulation(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 1 1 0",
            "v 0 1 0",
            "f 1 2 3 4",
        ])
        prims = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(len(prims), 1)
        # 1 quad triangulates into 2 triangles (6 indices)
        self.assertEqual(len(prims[0]["indices"]), 6)

    def test_parse_obj_negative_indices(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f -3 -2 -1",
        ])
        prims = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(len(prims), 1)
        self.assertEqual(len(prims[0]["indices"]), 3)

    def test_parse_obj_single_line_face_remains_unchanged(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 2 3",
        ])
        prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(prim["indices"], [0, 1, 2])

    def test_parse_obj_joins_face_split_across_two_lines(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 2 \\  ",
            "3",
        ])
        prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(prim["indices"], [0, 1, 2])

    def test_parse_obj_joins_face_split_across_three_lines(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 \\",
            "2 \\",
            "3",
        ])
        prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(prim["indices"], [0, 1, 2])

    def test_parse_obj_joins_face_split_immediately_after_keyword(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f \\",
            "1 2 3",
        ])
        prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(prim["indices"], [0, 1, 2])

    def test_parse_obj_continuation_skips_blank_and_comment_lines(self):
        obj_text = "\r\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 2 \\",
            "   # split after two corners",
            "",
            "3",
        ]) + "\r\n"
        prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(prim["indices"], [0, 1, 2])

    def test_parse_obj_backslash_inside_line_stays_an_invalid_token(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 2\\ 3",
        ])
        self.assertEqual(glb_writer.parse_obj(obj_text, is_path=False, up_axis="y"), [])

    def test_parse_obj_inline_comment_backslash_does_not_continue(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "v 1 1 0",
            "f 1 2 3 # keep this comment \\",
            "f 1 2 4",
        ])
        prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(prim["indices"], [0, 1, 2, 0, 1, 3])

    def test_parse_obj_continuation_before_inline_comment(self):
        obj_text = "\n".join([
            "v 0 0 0",
            "v 1 0 0",
            "v 0 1 0",
            "f 1 2 \\ # split before the comment",
            "3",
        ])
        prim, = glb_writer.parse_obj(obj_text, is_path=False, up_axis="y")
        self.assertEqual(prim["indices"], [0, 1, 2])

    def test_compute_bounds_calculation(self):
        prims = [
            {"positions": [(-5.0, 1.0, 2.0), (10.0, -3.0, 8.0)]},
            {"positions": [(0.0, 15.0, -4.0)]},
        ]
        bounds = glb_writer.compute_bounds(prims)
        self.assertEqual(bounds, [-5.0, -3.0, -4.0, 10.0, 15.0, 8.0])

        # Empty primitives return None
        self.assertIsNone(glb_writer.compute_bounds([]))
        self.assertIsNone(glb_writer.compute_bounds([{"positions": []}]))

    def test_parse_obj_invalid_options(self):
        with self.assertRaises(glb_writer.GlbError):
            glb_writer.parse_obj("", is_path=False, up_axis="x")

        with self.assertRaises(glb_writer.GlbError):
            glb_writer.parse_obj("", is_path=False, offset=(1.0, 2.0))  # only 2 components

        with self.assertRaises(glb_writer.GlbError):
            glb_writer.parse_obj("", is_path=False, scale=float("nan"))


class TestConcaveObjFaces(unittest.TestCase):
    def test_collinear_corner_keeps_uv_in_exported_surface(self) -> None:
        text = (
            "v 0 0 0\nv 1 0 0\nv 2 0 0\nv 2 1 0\nv 0 1 0\n"
            "vt 0 0.25\nvt 0.75 0.25\nf 1/1 2/2 3/1 4/1 5/1\n"
        )
        primitive, = glb_writer.parse_obj(text, is_path=False, up_axis="y")
        packed = glb_writer.pack_primitive(primitive)
        positions = list(struct.iter_unpack("<3f", packed["positions"]))
        uvs = list(struct.iter_unpack("<2f", packed["uvs"]))
        self.assertEqual(uvs[positions.index((1, 0, 0))], (0.75, 0.75))

        areas = []
        for triangle in struct.iter_unpack("<3H", packed["indices"]):
            a, b, c = (positions[i] for i in triangle)
            areas.append((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        self.assertEqual(len(areas), 3)
        self.assertTrue(all(area > 0 for area in areas), areas)
        self.assertEqual(sum(areas), 4.0)

    def test_notched_face_preserves_its_surface_and_corner_attributes(self):
        # A square with a 2 x 3 notch, including a collinear point on its base.
        # Embed the outline in YZ so the down-axis projection matters.
        outline = [(0, 0), (2, 0), (4, 0), (4, 4), (3, 4), (3, 1), (1, 1), (1, 4), (0, 4)]
        vertices = [f"v 10 {20 + u} {30 + v}" for u, v in outline]
        uvs = [f"vt {i / 16} 0.25" for i in range(len(outline))]
        for reverse in (False, True):
            for scale in (1, -1):
                with self.subTest(reverse=reverse, scale=scale):
                    order = list(range(len(outline)))
                    if reverse:
                        order.reverse()
                    text = "\n".join(vertices + uvs + ["f " + " ".join(f"{i + 1}/{i + 1}" for i in order)])
                    prim, = glb_writer.parse_obj(text, is_path=False, up_axis="y", scale=scale)
                    positions = prim["positions"]
                    expected_uvs = {
                        (10 * scale, (20 + u) * scale, (30 + v) * scale): (i / 16, 0.75)
                        for i, (u, v) in enumerate(outline)
                    }
                    self.assertEqual(dict(zip(positions, prim["uvs"])), expected_uvs)
                    sign = (-1 if reverse else 1) * scale
                    self.assertEqual(set(prim["normals"]), {(sign, 0, 0)})
                    idx = prim["indices"]
                    self.assertEqual(len(idx), (len(outline) - 2) * 3)
                    total = 0
                    for i in range(0, len(idx), 3):
                        a, b, c = (positions[j] for j in idx[i:i + 3])
                        area2 = (b[1] - a[1]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[1] - a[1])
                        self.assertGreater(area2 * sign, 0)
                        total += abs(area2)
                    self.assertEqual(total, 2 * (4 * 4 - 2 * 3))

    def test_valid_first_corner_fan_keeps_its_index_order(self):
        for shape in ([(0, 0), (2, 0), (3, 2), (1, 3), (-1, 2)],
                      [(2, 1), (4, 0), (2, 4), (0, 0)]):
            with self.subTest(shape=shape):
                lines = [f"v {x} {y} 0" for x, y in shape]
                lines.append("f " + " ".join(str(i + 1) for i in range(len(shape))))
                prim, = glb_writer.parse_obj("\n".join(lines), is_path=False, up_axis="y")
                self.assertEqual(prim["positions"], [(x, y, 0) for x, y in shape])
                self.assertEqual(prim["indices"], [j for i in range(1, len(shape) - 1) for j in (0, i, i + 1)])


class TestGlbWriterMainCli(unittest.TestCase):
    """Tests the command-line entrypoint of tools/glb_writer.py."""

    def test_main_cli_execution(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            obj_path = os.path.join(tmp_dir, "input.obj")
            glb_path = os.path.join(tmp_dir, "output.glb")
            with open(obj_path, "w", encoding="utf-8") as f:
                f.write("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n")

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ret = glb_writer.main([obj_path, glb_path, "--up", "y"])
            self.assertEqual(ret, 0)
            self.assertTrue(os.path.isfile(glb_path))

    def test_main_cli_error_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bad_obj = os.path.join(tmp_dir, "non_existent.obj")
            glb_path = os.path.join(tmp_dir, "output.glb")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                ret = glb_writer.main([bad_obj, glb_path])
            self.assertEqual(ret, 1)

    def test_main_cli_refuses_to_overwrite_the_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            obj_path = os.path.join(tmp_dir, "input.obj")
            text = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
            with open(obj_path, "w", encoding="utf-8", newline="") as f:
                f.write(text)
            links = [obj_path]
            try:
                link = os.path.join(tmp_dir, "alias.obj")
                os.link(obj_path, link)
                links.append(link)
            except (OSError, NotImplementedError):
                pass
            for out in links:
                err = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                    ret = glb_writer.main([obj_path, out])
                self.assertEqual(ret, 1)
                self.assertIn("overwrite the input", err.getvalue())
                self.assertNotIn("Traceback", err.getvalue())
                with open(obj_path, encoding="utf-8", newline="") as f:
                    self.assertEqual(f.read(), text)


if __name__ == "__main__":
    unittest.main()
