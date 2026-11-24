"""Synthetic tests for tools/skinned_glb.py.

    python -I tests/test_skinned_glb.py

Every model, skeleton, skin stream and GLB here is built by the test. No game
file, value or name is used. The fixture is a three-bone ped skeleton, a
two-entry drawable palette and a three-vertex 8-influence skin buffer.
"""
import hashlib
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import glb_writer  # noqa: E402
import skinned_glb  # noqa: E402


def _matrix(translation=(0.0, 0.0, 0.0)):
    return [1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            float(translation[0]), float(translation[1]), float(translation[2]), 1.0]


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def _write_glb(path):
    primitive = {
        "positions": [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        "normals": [(0.0, 0.0, 1.0)] * 3,
        "uvs": [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)],
        "indices": [0, 1, 2],
    }
    with open(path, "wb") as handle:
        handle.write(glb_writer.build_glb([primitive]))


def _read(path):
    with open(path, "rb") as handle:
        return handle.read()


def _section_word(offset, fmt):
    return ((offset // 2) << 8) | fmt


def _skin_buffer(j0, j1, w0, w1, *, omit_slot=None, formats=None, offsets=None, size=0x160):
    """Graphics buffer with the measured 23-word section header and per-slot rows."""
    formats = {6: 0x26, 7: 0x26, 8: 0x25, 9: 0x25, **(formats or {})}
    offsets = {6: 0x100, 7: 0x110, 8: 0x120, 9: 0x130, **(offsets or {})}
    data = {6: w0, 7: w1, 8: j0, 9: j1}
    blob = bytearray(size)
    for slot in (6, 7, 8, 9):
        if slot == omit_slot:
            continue
        struct.pack_into("<I", blob, slot * 4, _section_word(offsets[slot], formats[slot]))
        end = offsets[slot] + len(data[slot])
        if end > len(blob):
            raise ValueError("test skin stream does not fit its buffer")
        blob[offsets[slot]:end] = data[slot]
    return bytes(blob)


def _packed_indices(rows):
    """Pack four u8 joint indices per vertex into the 0x8C four-u32 layout.

    Bits 12-31 get a recognisable pattern, including bits 12-15, so a decoder
    that masks 16 bits instead of 12 fails instead of passing.
    """
    return b"".join(
        struct.pack("<4I", *(0xA5CFF000 | index for index in row)) for row in rows
    )


def _packed_buffer(j0_rows, j1_rows):
    _j0, _j1, w0, w1 = _streams()
    return _skin_buffer(
        _packed_indices(j0_rows), _packed_indices(j1_rows), w0, w1,
        formats={8: 0x8C, 9: 0x8C},
        offsets={8: 0x120, 9: 0x160},
        size=0x1A0,
    )


def _skeleton():
    """A three-bone RAGE Z-up bind pose: root, spine, left upper arm."""
    return [
        {"index": 0, "name": "SKEL_ROOT", "parent": None, "bind_matrix": _matrix()},
        {"index": 1, "name": "SKEL_Spine", "parent": 0,
         "bind_matrix": _matrix((0.0, 0.0, 1.0))},
        {"index": 2, "name": "SKEL_L_UpperArm", "parent": 1,
         "bind_matrix": _matrix((1.0, 0.0, 1.0))},
    ]


def _palette():
    return [
        skinned_glb.joaat32("SKEL_Spine"),
        skinned_glb.joaat32("GM_TEST_L_UpperArm01_O"),
    ]


def _streams():
    j0 = bytes([0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0])
    j1 = bytes(12)
    w0 = bytes([64, 191, 0, 0, 128, 127, 0, 0, 255, 0, 0, 0])
    w1 = bytes(12)
    return j0, j1, w0, w1


def _component(root, *, own_skeleton=None, omit_slot=None, buffer=None, mesh_blob=None):
    directory = os.path.join(root, "component")
    os.makedirs(os.path.join(directory, "buffers"))
    if buffer is None:
        j0, j1, w0, w1 = _streams()
        buffer = _skin_buffer(j0, j1, w0, w1, omit_slot=omit_slot)
    with open(os.path.join(directory, "buffers", "01_0.bin"), "wb") as handle:
        handle.write(buffer)
    _write_glb(os.path.join(directory, "model.glb"))
    model = {
        "buffer_files": ["buffers/01_0.bin"],
        "glb": "model.glb",
        "vertex_count": 3,
        "drawable": {
            "name": "synthetic",
            "buffers": [{"bytes": len(buffer), "gfx_off": 0}],
            "geometries": [{"vertex_buffer": 0, "index_buffer": 0}],
            "models": [{"bone_palette": _palette()}],
            "strings": ["GM_TEST_L_UpperArm01_O"],
        },
        "meshes": [{
            "geometry": 0,
            "model": 0,
            "name": "synthetic_mesh",
            "vertex_count": 3,
            "skinned": True,
            "skin_influences": 8,
        }],
    }
    if own_skeleton is not None:
        model["drawable"]["skeleton"] = own_skeleton
    if mesh_blob is not None:
        model["drawable"]["buffers"][0]["mesh_blob"] = mesh_blob
    _write_json(os.path.join(directory, "model.json"), model)
    return directory


def _read_model(directory):
    with open(os.path.join(directory, "model.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _skeleton_dir(root):
    directory = os.path.join(root, "skeleton")
    os.makedirs(directory)
    _write_json(os.path.join(directory, "model.json"), {
        "drawable": {"skeleton": _skeleton()},
    })
    return directory


def _accessor(gltf, binary, index):
    payload, accessor = skinned_glb.accessor_bytes(gltf, binary, index)
    return payload, accessor


def _world_matrices(gltf):
    nodes = gltf["nodes"]
    parent = {}
    for index, node in enumerate(nodes):
        for child in node.get("children", []):
            parent[child] = index
    cache = {}

    def world(index):
        if index in cache:
            return cache[index]
        matrix = nodes[index].get(
            "matrix", [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                       0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        if index in parent:
            matrix = skinned_glb.mat_mul(world(parent[index]), matrix)
        cache[index] = matrix
        return matrix

    return world


def _transform(matrix, point):
    return [
        sum(matrix[column * 4 + row] * point[column] for column in range(3)) + matrix[12 + row]
        for row in range(3)
    ]


class InterleavedAccessorTest(unittest.TestCase):
    def test_last_attribute_can_end_at_the_last_bin_byte(self) -> None:
        gltf = {
            "bufferViews": [{"buffer": 0, "byteStride": 24, "byteLength": 48}],
            "accessors": [{"bufferView": 0, "byteOffset": 12, "componentType": 5126,
                           "count": 2, "type": "VEC3"}],
        }
        vertices = struct.pack("<12f", 1, 2, 3, 0, 1, 0, 4, 5, 6, 0, 0, 1)
        normals, _ = skinned_glb.accessor_bytes(gltf, vertices, 0)
        self.assertEqual(normals, struct.pack("<6f", 0, 1, 0, 0, 0, 1))
        with self.assertRaisesRegex(skinned_glb.SkinError, "outside the BIN chunk"):
            skinned_glb.accessor_bytes(gltf, vertices[:-1], 0)


class SkinnedGlbTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = self.temp.name

    def tearDown(self):
        self.temp.cleanup()

    def _build(self):
        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        payload, stats = skinned_glb.build_skinned_glb(skeleton, [component])
        gltf, binary = skinned_glb.read_glb(self._write(payload))
        return gltf, binary, stats

    def _write(self, payload):
        path = os.path.join(self.root, "out.glb")
        with open(path, "wb") as handle:
            handle.write(payload)
        return path

    def test_joint_count_and_palette_resolution(self):
        _gltf, _binary, stats = self._build()
        self.assertEqual(stats["ped_joints"], 3)
        self.assertEqual(stats["component_joints"], 1)
        self.assertEqual(stats["joints"], 4)
        self.assertEqual(stats["vertices"], 3)
        self.assertEqual(stats["triangles"], 1)
        self.assertEqual(stats["source_weight_sum_min"], 255)
        self.assertEqual(stats["source_weight_sum_max"], 255)

    def test_local_child_listed_before_its_parent_keeps_the_own_skeleton_link(self):
        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        model = _read_model(component)
        model["drawable"]["skeleton"] = [
            {"name": "HelperParent", "parent_name": "SKEL_Spine",
             "bind_matrix": _matrix((0.0, 0.0, 2.0))},
            {"name": "HelperChild", "parent_name": "HelperParent",
             "bind_matrix": _matrix((0.0, 0.0, 3.0))},
        ]
        model["drawable"]["models"][0]["bone_palette"] = [
            skinned_glb.joaat32("HelperChild"),
            skinned_glb.joaat32("HelperParent"),
        ]
        _write_json(os.path.join(component, "model.json"), model)

        payload, stats = skinned_glb.build_skinned_glb(skeleton, [component])
        gltf, _binary = skinned_glb.read_glb(self._write(payload))
        self.assertEqual(stats["component_joints"], 2)
        by_name = {node.get("name"): index for index, node in enumerate(gltf["nodes"])}
        parent, child = by_name["HelperParent"], by_name["HelperChild"]
        self.assertIn(child, gltf["nodes"][parent].get("children", []))
        self.assertIn(parent, gltf["nodes"][by_name["SKEL_Spine"]].get("children", []))

    def test_local_child_under_a_parent_the_palette_never_lists(self):
        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        model = _read_model(component)
        model["drawable"]["skeleton"] = [
            {"name": "HelperParent", "parent_name": "SKEL_Spine",
             "bind_matrix": _matrix((0.0, 0.0, 2.0))},
            {"name": "HelperMiddle", "parent": 0, "bind_matrix": _matrix((0.0, 0.0, 3.0))},
            {"name": "HelperChild", "parent": 1, "bind_matrix": _matrix((0.0, 0.0, 4.0))},
        ]
        model["drawable"]["models"][0]["bone_palette"] = [
            skinned_glb.joaat32("HelperChild"),
            skinned_glb.joaat32("GM_TEST_L_UpperArm01_O"),
        ]
        _write_json(os.path.join(component, "model.json"), model)

        payload, stats = skinned_glb.build_skinned_glb(skeleton, [component])
        gltf, _binary = skinned_glb.read_glb(self._write(payload))
        self.assertEqual(stats["component_joints"], 4)
        by_name = {node.get("name"): index for index, node in enumerate(gltf["nodes"])}
        parent, middle, child = (by_name["HelperParent"], by_name["HelperMiddle"],
                                 by_name["HelperChild"])
        self.assertIn(middle, gltf["nodes"][parent].get("children", []))
        self.assertIn(child, gltf["nodes"][middle].get("children", []))
        self.assertIn(parent, gltf["nodes"][by_name["SKEL_Spine"]].get("children", []))

    def test_normalized_byte_uvs_have_aligned_output_rows(self) -> None:
        from anim_glb import pack_glb

        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        source_path = os.path.join(component, "model.glb")
        source, source_bin = skinned_glb.read_glb(source_path)
        uv_index = source["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]
        uv_accessor = source["accessors"][uv_index]
        uv_view = source["bufferViews"][uv_accessor["bufferView"]]
        uv_accessor.update(componentType=5121, normalized=True)
        uv_view.update(byteStride=4, byteLength=12)
        offset = uv_view["byteOffset"]
        rows = bytes((0, 255, 0, 0, 64, 128, 0, 0, 255, 0, 0, 0))
        source_bin = source_bin[:offset] + rows + source_bin[offset + len(rows):]
        with open(source_path, "wb") as handle:
            handle.write(pack_glb(source, source_bin))

        payload, _ = skinned_glb.build_skinned_glb(skeleton, [component])
        gltf, binary = skinned_glb.read_glb(self._write(payload))
        index = gltf["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"]
        accessor = gltf["accessors"][index]
        view = gltf["bufferViews"][accessor["bufferView"]]
        self.assertEqual((accessor["componentType"], accessor["normalized"]), (5121, True))
        self.assertEqual((view.get("byteStride"), view["byteLength"]), (4, 12))
        start = view["byteOffset"]
        self.assertEqual(binary[start:start + view["byteLength"]], rows)
        self.assertEqual(_accessor(gltf, binary, index)[0], bytes((0, 255, 64, 128, 255, 0)))

    def test_weight_sums_are_one(self):
        gltf, binary, _stats = self._build()
        primitive = gltf["meshes"][0]["primitives"][0]
        w0, _ = _accessor(gltf, binary, primitive["attributes"]["WEIGHTS_0"])
        w1, _ = _accessor(gltf, binary, primitive["attributes"]["WEIGHTS_1"])
        for vertex in range(3):
            values = struct.unpack_from("<4f", w0, 16 * vertex) + \
                struct.unpack_from("<4f", w1, 16 * vertex)
            self.assertAlmostEqual(sum(values), 1.0, places=6)
            self.assertGreater(values[0], 0.0)

    def test_joint_indices_are_in_skin_range(self):
        gltf, binary, _stats = self._build()
        joint_count = len(gltf["skins"][0]["joints"])
        self.assertEqual(joint_count, 4)
        primitive = gltf["meshes"][0]["primitives"][0]
        j0, _ = _accessor(gltf, binary, primitive["attributes"]["JOINTS_0"])
        j1, _ = _accessor(gltf, binary, primitive["attributes"]["JOINTS_1"])
        for vertex in range(3):
            indices = struct.unpack_from("<4H", j0, 8 * vertex) + \
                struct.unpack_from("<4H", j1, 8 * vertex)
            for index in indices:
                self.assertLess(index, joint_count)
        # The second palette entry is the added helper joint (index 3).
        self.assertEqual(struct.unpack_from("<4H", j0, 0), (1, 3, 0, 0))

    def test_inverse_bind_round_trip(self):
        gltf, binary, _stats = self._build()
        skin = gltf["skins"][0]
        ibm_payload, ibm_accessor = _accessor(gltf, binary, skin["inverseBindMatrices"])
        self.assertEqual(ibm_accessor["count"], len(skin["joints"]))
        inverse_binds = [struct.unpack_from("<16f", ibm_payload, 64 * index)
                         for index in range(len(skin["joints"]))]
        world = _world_matrices(gltf)
        for joint, inverse_bind in zip(skin["joints"], inverse_binds):
            product = skinned_glb.mat_mul(world(joint), inverse_bind)
            expected = [1.0 if index in (0, 5, 10, 15) else 0.0 for index in range(16)]
            for actual, wanted in zip(product, expected):
                self.assertAlmostEqual(actual, wanted, places=5)

        primitive = gltf["meshes"][0]["primitives"][0]
        positions, _ = _accessor(gltf, binary, primitive["attributes"]["POSITION"])
        j0, _ = _accessor(gltf, binary, primitive["attributes"]["JOINTS_0"])
        j1, _ = _accessor(gltf, binary, primitive["attributes"]["JOINTS_1"])
        w0, _ = _accessor(gltf, binary, primitive["attributes"]["WEIGHTS_0"])
        w1, _ = _accessor(gltf, binary, primitive["attributes"]["WEIGHTS_1"])
        for vertex in range(3):
            point = struct.unpack_from("<3f", positions, 12 * vertex)
            indices = struct.unpack_from("<4H", j0, 8 * vertex) + \
                struct.unpack_from("<4H", j1, 8 * vertex)
            weights = struct.unpack_from("<4f", w0, 16 * vertex) + \
                struct.unpack_from("<4f", w1, 16 * vertex)
            result = [0.0, 0.0, 0.0]
            for index, weight in zip(indices, weights):
                matrix = skinned_glb.mat_mul(world(skin["joints"][index]),
                                             inverse_binds[index])
                moved = _transform(matrix, point)
                for axis in range(3):
                    result[axis] += weight * moved[axis]
            for actual, wanted in zip(result, point):
                self.assertAlmostEqual(actual, wanted, places=5)

    def test_numpy_and_stdlib_weight_paths_agree(self):
        if skinned_glb._np is None:
            self.skipTest("numpy is not importable, so the optional path cannot be compared")
        j0, j1, w0, w1 = _streams()
        j0 = bytearray(j0)
        j0[2] = 200  # zero-weight component outside the two-entry palette
        j0 = bytes(j0)
        with_numpy = skinned_glb.normalise_weights(j0, j1, w0, w1, 3, 2, [1, 3], 4)
        saved = skinned_glb._np
        try:
            skinned_glb._np = None
            without_numpy = skinned_glb.normalise_weights(j0, j1, w0, w1, 3, 2, [1, 3], 4)
        finally:
            skinned_glb._np = saved
        for key in ("j0", "j1", "w0", "w1"):
            self.assertEqual(with_numpy[key], without_numpy[key], key)
        self.assertEqual(with_numpy["sum_min"], without_numpy["sum_min"])
        self.assertEqual(with_numpy["sum_max"], without_numpy["sum_max"])
        self.assertAlmostEqual(struct.unpack_from("<4f", with_numpy["w0"], 0)[0], 64 / 255, delta=1e-6)

    def test_only_weighted_joint_indices_must_fit_unsigned_short(self):
        backends = [None] if skinned_glb._np is None else [None, skinned_glb._np]
        palette = [65536, 65535]
        for backend in backends:
            with self.subTest(backend=backend), mock.patch.object(skinned_glb, "_np", backend):
                for second_set in (False, True):
                    w0, w1 = bytes([255, 0, 0, 0]), bytes(4)
                    if second_set:
                        w0, w1 = w1, w0
                    with self.subTest(second_set=second_set):
                        with self.assertRaisesRegex(skinned_glb.SkinError, "UNSIGNED_SHORT"):
                            skinned_glb.normalise_weights(bytes(4), bytes(4), w0, w1, 1, 2, palette, 65537)
                        joints = bytes([1, 0, 0, 0])
                        out = skinned_glb.normalise_weights(joints, joints, w0, w1, 1, 2, palette, 65537)
                        weighted, unused = ("j1", "j0") if second_set else ("j0", "j1")
                        self.assertEqual(struct.unpack("<4H", out[weighted]), (65535, 0, 0, 0))
                        self.assertEqual(out[unused], bytes(8))

    def test_full_export_without_numpy(self):
        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        saved = skinned_glb._np
        try:
            skinned_glb._np = None
            payload, stats = skinned_glb.build_skinned_glb(skeleton, [component])
        finally:
            skinned_glb._np = saved
        gltf, binary = skinned_glb.read_glb(self._write(payload))
        self.assertEqual(stats["joints"], 4)
        primitive = gltf["meshes"][0]["primitives"][0]
        w0, _ = _accessor(gltf, binary, primitive["attributes"]["WEIGHTS_0"])
        for vertex in range(3):
            self.assertAlmostEqual(sum(struct.unpack_from("<4f", w0, 16 * vertex)), 1.0, places=6)

    def test_packed_joint_stream_decodes_to_four_u8_indices(self):
        j0_rows = [(1, 2, 0, 0), (1, 2, 0, 0), (2, 1, 0, 0)]
        j1_rows = [(3, 0, 0, 0)] * 3
        component = _component(self.root, buffer=_packed_buffer(j0_rows, j1_rows))
        model = _read_model(component)
        streams = skinned_glb.read_skin_streams(component, model, model["meshes"][0])
        self.assertEqual(streams["j0"], bytes([1, 2, 0, 0, 1, 2, 0, 0, 2, 1, 0, 0]))
        self.assertEqual(streams["j1"], bytes([3, 0, 0, 0] * 3))

    def test_packed_joint_index_above_u8_is_rejected(self):
        j0_rows = [(1, 2, 0, 0), (1, 2, 0, 0), (256, 0, 0, 0)]
        j1_rows = [(3, 0, 0, 0)] * 3
        component = _component(self.root, buffer=_packed_buffer(j0_rows, j1_rows))
        model = _read_model(component)
        with self.assertRaisesRegex(skinned_glb.SkinError, "does not fit u8"):
            skinned_glb.read_skin_streams(component, model, model["meshes"][0])

    def test_short_packed_joint_stream_is_rejected(self):
        j0 = _packed_indices([(1, 2, 0, 0)] * 3)[:47]
        j1 = _packed_indices([(3, 0, 0, 0)] * 3)
        _j0, _j1, w0, w1 = _streams()
        buffer = _skin_buffer(
            j0, j1, w0, w1,
            formats={8: 0x8C, 9: 0x8C},
            offsets={8: 0x150, 9: 0x120},
            size=0x17F,
        )
        component = _component(self.root, buffer=buffer)
        model = _read_model(component)
        with self.assertRaisesRegex(skinned_glb.SkinError, "outside its graphics buffer"):
            skinned_glb.read_skin_streams(component, model, model["meshes"][0])

    def test_missing_joint_stream_is_rejected(self):
        component = _component(self.root, omit_slot=8)
        skeleton = _skeleton_dir(self.root)
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.build_skinned_glb(skeleton, [component])

    def test_mesh_blob_sections_supply_the_stream_offsets(self):
        j0, j1, w0, w1 = _streams()
        buffer = bytearray(_skin_buffer(j0, j1, w0, w1))
        for slot in (6, 7, 8, 9):
            struct.pack_into("<I", buffer, slot * 4, 0)  # no usable header words
        sections = [[6, 0x100, 0x26], [7, 0x110, 0x26], [8, 0x120, 0x25], [9, 0x130, 0x25]]
        component = _component(self.root, buffer=bytes(buffer),
                               mesh_blob={"sections": sections})
        model = _read_model(component)
        streams = skinned_glb.read_skin_streams(component, model, model["meshes"][0])
        self.assertEqual(streams["sections"], {6: (0x100, 0x26), 7: (0x110, 0x26),
                                               8: (0x120, 0x25), 9: (0x130, 0x25)})
        self.assertEqual(streams["j0"], j0)
        self.assertEqual(streams["j1"], j1)

    def test_empty_component_list_is_rejected(self):
        skeleton = _skeleton_dir(self.root)
        with self.assertRaisesRegex(skinned_glb.SkinError, "at least one --component"):
            skinned_glb.build_skinned_glb(skeleton, [])

    def test_zero_weight_vertex_is_rejected(self):
        j0, j1, w0, w1 = _streams()
        broken = bytearray(w0)
        broken[0:4] = b"\0\0\0\0"
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.normalise_weights(j0, j1, bytes(broken), w1, 3, 2, [1, 3], 4)

    def test_out_of_range_weighted_joint_is_rejected(self):
        j0, j1, w0, w1 = _streams()
        broken = bytearray(j0)
        broken[0] = 2  # local palette has two entries; this weighted index is invalid
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.normalise_weights(bytes(broken), j1, w0, w1, 3, 2, [1, 3], 4)

    def test_zero_weight_padding_joint_index_outside_local_palette_is_zeroed(self):
        j0, j1, w0, w1 = _streams()
        padded_j1 = bytearray(j1)
        padded_j1[0] = 255  # w1[0] is 0, so this unused slot must be ignored and zeroed in both paths
        got_default = skinned_glb.normalise_weights(j0, bytes(padded_j1), w0, w1, 3, 2, [1, 3], 4)
        self.assertEqual(struct.unpack_from("<4H", got_default["j1"], 0)[0], 0)

        class _FakeArray:
            def __init__(self, rows):
                self.rows = [list(r) for r in rows]

            def astype(self, _dtype):
                return _FakeArray(self.rows)

            def sum(self, axis=1):
                assert axis == 1
                return _FakeVec([sum(r) for r in self.rows])

            def __gt__(self, other):
                return _FakeMask([[v > other for v in r] for r in self.rows])

            def __eq__(self, other):
                return _FakeMask([[v == other for v in r] for r in self.rows])

            def __ge__(self, other):
                return _FakeMask([[v >= other for v in r] for r in self.rows])

            def __itruediv__(self, col):
                for r, d in zip(self.rows, col.vals):
                    for i in range(len(r)):
                        r[i] = r[i] / d
                return self

            def __getitem__(self, key):
                if isinstance(key, _FakeMask):
                    return _FakeVec([v for r, mr in zip(self.rows, key.rows) for v, m in zip(r, mr) if m])
                if isinstance(key, tuple) and len(key) == 2 and key[0] == slice(None):
                    return _FakeArray([r[key[1]] for r in self.rows])
                raise TypeError(key)

            def __setitem__(self, key, value):
                assert isinstance(key, _FakeMask)
                for r, mr in zip(self.rows, key.rows):
                    for i, m in enumerate(mr):
                        if m:
                            r[i] = value

            def tobytes(self):
                if self.rows and isinstance(self.rows[0][0], float):
                    return b"".join(struct.pack("<f", v) for r in self.rows for v in r)
                return b"".join(struct.pack("<H", int(v)) for r in self.rows for v in r)

        class _FakeMask:
            def __init__(self, rows):
                self.rows = rows

            def any(self):
                return any(any(r) for r in self.rows)

        class _FakeVec:
            def __init__(self, vals):
                self.vals = list(vals)

            def __le__(self, other):
                return _FakeMask([[v <= other for v in self.vals]])

            def __getitem__(self, key):
                if isinstance(key, _FakeArray):
                    return _FakeArray([[self.vals[int(idx)] for idx in r] for r in key.rows])
                return self

            def max(self):
                return max(self.vals)

            def min(self):
                return min(self.vals)

        class _FakeNp:
            uint8 = "u1"
            uint16 = "u2"
            int64 = "i8"
            float64 = "f8"

            @staticmethod
            def frombuffer(buf, dtype):
                class _Buf:
                    def reshape(self_inner, n, k):
                        return _FakeArray([list(buf[i * k:(i + 1) * k]) for i in range(n)])
                return _Buf()

            @staticmethod
            def concatenate(parts, axis=1):
                assert axis == 1
                return _FakeArray([r0 + r1 for r0, r1 in zip(parts[0].rows, parts[1].rows)])

            @staticmethod
            def asarray(seq, dtype):
                return _FakeVec(seq)

        saved = skinned_glb._np
        try:
            skinned_glb._np = _FakeNp
            got_np = skinned_glb.normalise_weights(j0, bytes(padded_j1), w0, w1, 3, 2, [1, 3], 4)
        finally:
            skinned_glb._np = saved
        self.assertEqual(got_np["j0"], got_default["j0"])
        self.assertEqual(got_np["j1"], got_default["j1"])

    def test_infer_ped_parent_matches_suffixed_ped_bone_via_prefix_fallback(self):
        ped = {
            "SKEL_ROOT": {"name": "SKEL_ROOT", "index": 0},
            "SKEL_L_UpperArm01": {"name": "SKEL_L_UpperArm01", "index": 1},
            "SKEL_R_UpperArm01": {"name": "SKEL_R_UpperArm01", "index": 2},
            "SKEL_SPINE10": {"name": "SKEL_SPINE10", "index": 3},
        }
        self.assertEqual(skinned_glb.infer_ped_parent("GM_TEST_L_UpperArm_O", ped), 1)
        self.assertEqual(skinned_glb.infer_ped_parent("GM_TEST_R_UpperArm_O", ped), 2)
        # Numbered aliases (e.g. SPINE1) must not prefix-match a different numbered bone (SPINE10).
        self.assertEqual(skinned_glb.infer_ped_parent("GM_C_SPINE1", ped), 0)
        # Duplicate earlier ped bone names must not shift the inferred joint index onto another bone.
        ident = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        dup_skel = [
            {"name": "SKEL_ROOT", "parent": None, "bind_matrix": ident},
            {"name": "AUX_pin", "parent": 0, "bind_matrix": ident},
            {"name": "AUX_pin", "parent": 0, "bind_matrix": ident},
            {"name": "SKEL_R_UpperArm01", "parent": 0, "bind_matrix": ident},
            {"name": "SKEL_L_UpperArm01", "parent": 0, "bind_matrix": ident},
        ]
        jset = skinned_glb.JointSet(dup_skel)
        self.assertEqual(
            skinned_glb.infer_ped_parent("GM_TEST_L_UpperArm_O", jset.ped_by_name, jset._by_name),
            4,
        )

    def test_write_uses_unique_temp_when_fixed_name_is_a_directory(self):
        source_root = os.path.join(self.root, "stale")
        skeleton = _skeleton_dir(source_root)
        component = _component(source_root)
        out = os.path.join(self.root, "nested", "out.glb")
        stale = out + ".tmp"
        os.makedirs(stale)
        saved = skinned_glb._np
        try:
            skinned_glb._np = None
            stats = skinned_glb.write_skinned_glb(out, skeleton, [component])
        finally:
            skinned_glb._np = saved
        self.assertEqual(stats["joints"], 4)
        self.assertTrue(os.path.isfile(out))
        with open(out, "rb") as handle:
            self.assertEqual(handle.read(4), b"glTF")
        self.assertTrue(os.path.isdir(stale))
        self.assertEqual([n for n in os.listdir(os.path.dirname(out)) if n.startswith("out.glb.")],
                         ["out.glb.tmp"])

    def test_failed_replace_leaves_no_readonly_temp(self):
        source_root = os.path.join(self.root, "locked")
        skeleton = _skeleton_dir(source_root)
        component = _component(source_root)
        out = os.path.join(self.root, "locked.glb")
        with open(out, "wb") as handle:
            handle.write(b"old")
        os.chmod(out, 0o444)
        saved = skinned_glb._np
        try:
            skinned_glb._np = None
            with mock.patch.object(skinned_glb.os, "replace", side_effect=OSError("replace denied")):
                with self.assertRaises(OSError):
                    skinned_glb.write_skinned_glb(out, skeleton, [component])
        finally:
            skinned_glb._np = saved
            os.chmod(out, 0o644)
        leftovers = [n for n in os.listdir(self.root)
                     if n.startswith("locked.glb.") and n.endswith(".tmp")]
        self.assertEqual(leftovers, [])
        with open(out, "rb") as handle:
            self.assertEqual(handle.read(), b"old")

    def test_cli_returns_nonzero_for_bad_input(self):
        result = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "skinned_glb.py"),
             os.path.join(self.root, "out.glb"), "--skeleton", self.root,
             "--component", self.root],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("error:", result.stderr)

    # BUILD-1031: an interrupted export leaves an empty, cut-off or non-UTF-8 model.json.
    BAD_JSON = {
        "empty": b"",
        "truncated": b'{"a":',
        "not_utf8": b'{"name": "\xff\xfe"}',
    }

    def _run_cli(self, skeleton, component):
        return subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "skinned_glb.py"),
             os.path.join(self.root, "out.glb"), "--skeleton", skeleton, "--component", component],
            capture_output=True, text=True, check=False,
        )

    def _assert_one_line_error(self, result, path):
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        lines = result.stderr.strip().splitlines()
        self.assertEqual(len(lines), 1, result.stderr)
        self.assertTrue(lines[0].startswith("error: "), result.stderr)
        self.assertIn(path, lines[0])
        self.assertIn("invalid JSON", lines[0])

    def test_bad_model_json_is_a_skin_error(self):
        for name, data in self.BAD_JSON.items():
            with self.subTest(name):
                path = os.path.join(self.root, name + ".json")
                with open(path, "wb") as handle:
                    handle.write(data)
                with self.assertRaises(skinned_glb.SkinError) as caught:
                    skinned_glb._load_json(path)
                self.assertIn(path, str(caught.exception))

    def _case_root(self, name):
        """A fresh folder per case: the fixture builders make fixed subfolder names."""
        root = os.path.join(self.root, name)
        os.makedirs(root)
        return root

    def test_cli_bad_skeleton_json_is_one_error_line(self):
        for name, data in self.BAD_JSON.items():
            with self.subTest(name):
                root = self._case_root("skeleton_" + name)
                component = _component(root)
                skeleton = _skeleton_dir(root)
                path = os.path.join(skeleton, "model.json")
                with open(path, "wb") as handle:
                    handle.write(data)
                self._assert_one_line_error(self._run_cli(skeleton, component), path)

    def test_cli_bad_component_json_is_one_error_line(self):
        for name, data in self.BAD_JSON.items():
            with self.subTest(name):
                root = self._case_root("component_" + name)
                skeleton = _skeleton_dir(root)
                component = _component(root)
                path = os.path.join(component, "model.json")
                with open(path, "wb") as handle:
                    handle.write(data)
                self._assert_one_line_error(self._run_cli(skeleton, component), path)

    def test_output_must_not_overwrite_a_source(self):
        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        victims = {
            "component model.json": os.path.join(component, "model.json"),
            "skeleton model.json (file)": os.path.join(skeleton, "model.json"),
            "skeleton model.json (dir)": os.path.join(skeleton, "model.json"),
            "component model.glb": os.path.join(component, "model.glb"),
            "component buffer": os.path.join(component, "buffers", "01_0.bin"),
        }
        originals = {key: _read(path) for key, path in victims.items()}
        for argv, key in (
                ([victims["component model.json"], "--skeleton", skeleton,
                  "--component", component], "component model.json"),
                ([victims["skeleton model.json (file)"], "--skeleton", victims["skeleton model.json (file)"],
                  "--component", component], "skeleton model.json (file)"),
                ([victims["skeleton model.json (dir)"], "--skeleton", skeleton,
                  "--component", component], "skeleton model.json (dir)"),
                ([victims["component model.glb"], "--skeleton", skeleton, "--component", component],
                 "component model.glb"),
                ([victims["component buffer"], "--skeleton", skeleton, "--component", component],
                 "component buffer")):
            with self.subTest(victim=key):
                result = subprocess.run(
                    [sys.executable, "-I", os.path.join(ROOT, "tools", "skinned_glb.py")] + argv,
                    capture_output=True, text=True, check=False,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("must not overwrite the source", result.stderr)
                with open(victims[key], "rb") as handle:
                    self.assertEqual(handle.read(), originals[key], key)
        alt = os.path.join(component, "MODEL.JSON")
        if os.path.exists(alt):     # case-insensitive volume: MODEL.JSON is model.json, so it must refuse
            result = subprocess.run(
                [sys.executable, "-I", os.path.join(ROOT, "tools", "skinned_glb.py"), alt,
                 "--skeleton", skeleton, "--component", component],
                capture_output=True, text=True, check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overwrite the source", result.stderr)
            with open(victims["component model.json"], "rb") as handle:
                self.assertEqual(handle.read(), originals["component model.json"])

    def test_distinct_output_still_writes_the_glb(self):
        # The guard must not break the normal path: sources stay byte-identical and the GLB is written.
        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        component_json = os.path.join(component, "model.json")
        skeleton_json = os.path.join(skeleton, "model.json")
        originals = {path: _read(path)
                     for path in (component_json, skeleton_json,
                                  os.path.join(component, "model.glb"),
                                  os.path.join(component, "buffers", "01_0.bin"))}
        out = os.path.join(self.root, "out.glb")
        result = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "skinned_glb.py"), out,
             "--skeleton", skeleton, "--component", component],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("wrote", result.stdout)
        with open(out, "rb") as handle:
            self.assertTrue(handle.read().startswith(b"glTF"))
        for path, before in originals.items():
            with open(path, "rb") as handle:
                self.assertEqual(handle.read(), before, path)

    def test_output_temp_file_does_not_clobber_named_buffer(self):
        # A source buffer named <output>.tmp must stay intact after exporting <output>.
        skeleton = _skeleton_dir(self.root)
        component = _component(self.root)
        old_buf = os.path.join(component, "buffers", "01_0.bin")
        new_buf_rel = "buffers/export.glb.tmp"
        new_buf = os.path.join(component, "buffers", "export.glb.tmp")
        os.replace(old_buf, new_buf)
        model_json_path = os.path.join(component, "model.json")
        with open(model_json_path, "r", encoding="utf-8") as handle:
            model = json.load(handle)
        model["buffer_files"] = [new_buf_rel]
        _write_json(model_json_path, model)
        before_hash = hashlib.sha256(_read(new_buf)).hexdigest()

        out = os.path.join(component, "buffers", "export.glb")
        result = subprocess.run(
            [sys.executable, "-I", os.path.join(ROOT, "tools", "skinned_glb.py"), out,
             "--skeleton", skeleton, "--component", component],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(_read(out).startswith(b"glTF"))
        self.assertTrue(os.path.exists(new_buf))
        self.assertEqual(hashlib.sha256(_read(new_buf)).hexdigest(), before_hash)


if __name__ == "__main__":
    unittest.main()
