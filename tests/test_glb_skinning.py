"""Synthetic GLB fixtures for tools/glb_skinning.py. No game data is read.

    python -I tests/test_glb_skinning.py -v

Every fixture is built here from JSON and hand-packed binary accessors. The
tests cover a valid two-joint skinned mesh, bad weight sums, an out-of-range
joint index, a joint/inverseBindMatrices count mismatch, a static mesh next to a
skinned one, malformed containers, normalized byte/short weights, non-finite
values, the JSON report, and the input-alias guard.
"""
from __future__ import annotations

import contextlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import glb_skinning  # noqa: E402

TOOL = TOOLS / "glb_skinning.py"
IDENTITY = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)


def padded(data, fill=b"\0"):
    return data + fill * ((-len(data)) % 4)


def build_glb(gltf, bin_data):
    """Serialize a minimal JSON + BIN container with the required padding."""
    json_bytes = padded(json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8"), b" ")
    bin_bytes = padded(bytes(bin_data))
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    return b"".join((
        struct.pack("<4sII", b"glTF", 2, total_length),
        struct.pack("<II", len(json_bytes), glb_skinning.CHUNK_TYPE_JSON),
        json_bytes,
        struct.pack("<II", len(bin_bytes), glb_skinning.CHUNK_TYPE_BIN),
        bin_bytes,
    ))


class Builder:
    """Collect accessors and their bufferViews while packing one BIN payload."""

    def __init__(self):
        self.data = bytearray()
        self.buffer_views = []
        self.accessors = []

    def add_view(self, payload):
        offset = len(self.data)
        self.data.extend(payload)
        self.data.extend(b"\0" * ((-len(payload)) % 4))
        self.buffer_views.append({
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        })
        return len(self.buffer_views) - 1

    def add_accessor(self, payload, component_type, type_name, count, *,
                     normalized=False, minimum=None, maximum=None):
        view = self.add_view(payload)
        accessor = {
            "bufferView": view,
            "componentType": component_type,
            "count": count,
            "type": type_name,
        }
        if normalized:
            accessor["normalized"] = True
        if minimum is not None:
            accessor["min"] = minimum
        if maximum is not None:
            accessor["max"] = maximum
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def document(self, meshes, nodes, skins, scene_nodes):
        return {
            "asset": {"version": "2.0", "generator": "test_glb_skinning"},
            "scene": 0,
            "scenes": [{"nodes": scene_nodes}],
            "nodes": nodes,
            "meshes": meshes,
            "skins": skins,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self.data)}],
        }


def make_glb(weights=None, joints=None, *, ibm_count=2, add_static=False,
             include_skin=True, bind_skin=True, weight_component="float",
             skin_name="Rig", two_sets=False):
    """A three-vertex, two-joint skinned triangle, optionally plus a static mesh."""
    builder = Builder()
    positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    position_payload = b"".join(struct.pack("<3f", *point) for point in positions)
    position_accessor = builder.add_accessor(
        position_payload, 5126, "VEC3", len(positions),
        minimum=[0.0, 0.0, 0.0], maximum=[1.0, 1.0, 0.0],
    )
    if weights is None:
        if two_sets:
            weights = ((0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0),
                       (0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0),
                       (0.25, 0.25, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0))
        else:
            weights = ((1.0, 0.0, 0.0, 0.0),
                       (0.5, 0.5, 0.0, 0.0),
                       (0.25, 0.75, 0.0, 0.0))
    if joints is None:
        if two_sets:
            joints = ((0, 1, 0, 0, 0, 0, 0, 0),
                      (0, 1, 0, 1, 0, 0, 0, 0),
                      (0, 1, 0, 0, 1, 0, 0, 0))
        else:
            joints = ((0, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0))
    width = 8 if two_sets else 4
    if any(len(row) != width for row in weights):
        raise ValueError(f"test weights must have {width} components per vertex")
    if any(len(row) != width for row in joints):
        raise ValueError(f"test joints must have {width} components per vertex")
    weight_sets = (
        [list(row[:4]) for row in weights], [list(row[4:]) for row in weights]
    ) if two_sets else [weights]
    joint_sets = (
        [list(row[:4]) for row in joints], [list(row[4:]) for row in joints]
    ) if two_sets else [joints]

    joint_accessors = []
    for rows in joint_sets:
        if two_sets:
            payload = b"".join(struct.pack("<4H", *row) for row in rows)
            component_type = 5123
        else:
            payload = b"".join(struct.pack("<4B", *row) for row in rows)
            component_type = 5121
        joint_accessors.append(
            builder.add_accessor(payload, component_type, "VEC4", len(rows))
        )

    if weight_component == "float":
        pack_format, weight_type, normalized = "<4f", 5126, False
    elif weight_component == "u8":
        pack_format, weight_type, normalized = "<4B", 5121, True
    elif weight_component == "u16":
        pack_format, weight_type, normalized = "<4H", 5123, True
    else:
        raise ValueError(f"unsupported test weight component {weight_component!r}")
    weight_accessors = []
    for rows in weight_sets:
        payload = b"".join(struct.pack(pack_format, *row) for row in rows)
        weight_accessors.append(
            builder.add_accessor(
                payload, weight_type, "VEC4", len(rows), normalized=normalized,
            )
        )

    inverse_payload = b"".join(struct.pack("<16f", *IDENTITY) for _ in range(ibm_count))
    inverse_accessor = builder.add_accessor(
        inverse_payload, 5126, "MAT4", ibm_count,
    )

    attributes = {"POSITION": position_accessor}
    for set_number, (joint_accessor, weight_accessor) in enumerate(
            zip(joint_accessors, weight_accessors)):
        attributes[f"JOINTS_{set_number}"] = joint_accessor
        attributes[f"WEIGHTS_{set_number}"] = weight_accessor
    mesh = {
        "name": "Body",
        "primitives": [{"attributes": attributes, "mode": 4}],
    }
    meshes = [mesh]
    nodes = [
        {"name": "Bone0", "children": [1]},
        {"name": "Bone1"},
        {"name": "BodyNode", "mesh": 0},
    ]
    scene_nodes = [0, 2]
    if bind_skin:
        nodes[2]["skin"] = 0
    if include_skin:
        skin = {"joints": [0, 1], "inverseBindMatrices": inverse_accessor}
        if skin_name is not None:
            skin["name"] = skin_name
        skins = [skin]
    else:
        skins = []
    if add_static:
        static_accessor = builder.add_accessor(
            position_payload, 5126, "VEC3", len(positions),
            minimum=[0.0, 0.0, 0.0], maximum=[1.0, 1.0, 0.0],
        )
        meshes.append({
            "name": "Prop",
            "primitives": [{"attributes": {"POSITION": static_accessor}, "mode": 4}],
        })
        nodes.append({"name": "PropNode", "mesh": 1})
        scene_nodes.append(3)
    return build_glb(builder.document(meshes, nodes, skins, scene_nodes), builder.data)


class GlbSkinningTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, name, data):
        path = self.dir / name
        path.write_bytes(data)
        return path

    def run_tool(self, *args):
        return subprocess.run(
            [sys.executable, "-I", str(TOOL), *(str(arg) for arg in args)],
            capture_output=True, text=True, errors="replace", timeout=60,
        )

    def test_inverse_bind_fourth_row_is_affine_not_the_translation_column(self):
        doc, binary, _ = glb_skinning.parse_glb(make_glb())
        accessor = doc["accessors"][doc["skins"][0]["inverseBindMatrices"]]
        second = doc["bufferViews"][accessor["bufferView"]]["byteOffset"] + 64
        data = bytearray(binary)
        struct.pack_into("<3f", data, second + 12 * 4, 2.0, -3.0, 4.0)
        self.assertTrue(glb_skinning.analyze_glb(build_glb(doc, data))["ok"])
        for component in (3, 7, 11, 15):
            with self.subTest(component=component):
                invalid = bytearray(data)
                struct.pack_into("<f", invalid, second + component * 4, 0.5)
                proc = self.run_tool(self.write("projective.glb", build_glb(doc, invalid)))
                self.assertNotEqual(proc.returncode, 0)
                self.assertIn("inverseBindMatrices[1] fourth row", proc.stderr)

    def test_valid_two_joint_skinned_mesh(self):
        report = glb_skinning.analyze_glb(make_glb())
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(1, report["summary"]["skin_count"])
        self.assertEqual(["Rig"], report["summary"]["skin_names"])
        skin = report["skins"][0]
        self.assertEqual(2, skin["joint_count"])
        self.assertEqual([0, 1], skin["joints"])
        self.assertEqual(2, skin["inverse_bind_matrices"]["count"])
        self.assertEqual("MAT4", skin["inverse_bind_matrices"]["type"])

        primitive = report["meshes"][0]["primitives"][0]
        self.assertTrue(primitive["skinned"])
        self.assertEqual([0], primitive["skins"])
        self.assertEqual(3, primitive["vertex_count"])
        self.assertEqual(2, primitive["max_influences"])
        self.assertAlmostEqual(1.0, primitive["weight_sum"]["min"])
        self.assertAlmostEqual(1.0, primitive["weight_sum"]["max"])
        self.assertEqual(0, primitive["weight_sum"]["bad_count"])
        self.assertEqual(0, primitive["joint_sets"][0]["out_of_range"])
        self.assertEqual(0, report["summary"]["out_of_range_joint_count"])
        self.assertEqual([0.0, 0.0, 0.0], primitive["position"]["bbox_min"])
        self.assertEqual([1.0, 1.0, 0.0], primitive["position"]["bbox_max"])
        self.assertEqual(0, report["summary"]["non_finite_value_count"])
        self.assertEqual(1, report["summary"]["skinned_primitive_count"])
        self.assertEqual(0, report["summary"]["static_primitive_count"])

        path = self.write("valid.glb", make_glb())
        proc = self.run_tool(path)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("skinned", proc.stdout)

    def test_duplicate_skin_joints_are_rejected(self):
        doc, binary, _ = glb_skinning.parse_glb(make_glb(joints=((0, 1, 0, 0),) * 3))
        doc["skins"][0]["joints"] = [0, 0]
        data = build_glb(doc, binary)
        report = glb_skinning.analyze_glb(data)
        self.assertFalse(report["ok"])
        self.assertIn("duplicate joint node 0", "\n".join(report["errors"]))
        path = self.write("duplicate.glb", data)
        proc = self.run_tool(path)
        self.assertEqual(proc.returncode, 1)
        self.assertNotIn("Traceback", proc.stderr)
        self.assertEqual(Path(path).read_bytes(), data)

    def test_skin_sets_must_start_at_zero_and_have_no_gaps(self):
        for two_sets, old, new in ((False, 0, 1), (True, 1, 2), (False, 0, 999999999)):
            with self.subTest(two_sets=two_sets, new=new):
                doc, binary, _ = glb_skinning.parse_glb(make_glb(two_sets=two_sets))
                attrs = doc["meshes"][0]["primitives"][0]["attributes"]
                for semantic in ("JOINTS", "WEIGHTS"):
                    attrs[f"{semantic}_{new}"] = attrs.pop(f"{semantic}_{old}")
                report = glb_skinning.analyze_glb(build_glb(doc, binary))
                self.assertFalse(report["ok"])
                self.assertIn("consecutive", "\n".join(report["errors"]))

    def test_quantized_weight_sums_are_exact(self):
        for component, maximum in (("u8", 255), ("u16", 65535)):
            for delta in (-1, 0, 1):
                with self.subTest(component=component, delta=delta):
                    weights = ((maximum - 10, 10 + delta, 0, 0),) * 3
                    report = glb_skinning.analyze_glb(make_glb(weights=weights, weight_component=component, joints=((0, 1, 0, 0),) * 3))
                    self.assertEqual(report["ok"], delta == 0, report["errors"])
                    self.assertEqual(report["summary"]["bad_weight_sum_count"], 0 if delta == 0 else 3)
                    self.assertEqual(report["meshes"][0]["primitives"][0]["weight_sum"]["tolerance"], 0)

    def test_mixed_quantized_sets_use_common_integer_scale(self):
        # 128/255 + 32639/65535 == 1 exactly; +/- one u16 unit must fail.
        for delta in (-1, 0, 1):
            with self.subTest(delta=delta):
                doc, binary, _ = glb_skinning.parse_glb(make_glb(two_sets=True, joints=((0, 0, 0, 0, 1, 0, 0, 0),) * 3))
                binary = bytearray(binary)
                attrs = doc["meshes"][0]["primitives"][0]["attributes"]
                for set_no, component, value, fmt in ((0, 5121, 128, "<4B"), (1, 5123, 32639 + delta, "<4H")):
                    acc = doc["accessors"][attrs[f"WEIGHTS_{set_no}"]]
                    acc.update(componentType=component, normalized=True)
                    view = doc["bufferViews"][acc["bufferView"]]
                    view["byteStride"] = 16  # original float VEC4 allocation
                    for vertex in range(3):
                        struct.pack_into(fmt, binary, view["byteOffset"] + vertex * 16, value, 0, 0, 0)
                report = glb_skinning.analyze_glb(build_glb(doc, binary))
                self.assertEqual(report["ok"], delta == 0, report["errors"])

    def test_mixed_float_and_integer_sets_keep_float_tolerance(self):
        doc, binary, _ = glb_skinning.parse_glb(make_glb(two_sets=True, joints=((0, 0, 0, 0, 1, 0, 0, 0),) * 3))
        binary = bytearray(binary)
        attrs = doc["meshes"][0]["primitives"][0]["attributes"]
        for set_no in (0, 1):
            acc = doc["accessors"][attrs[f"WEIGHTS_{set_no}"]]
            view = doc["bufferViews"][acc["bufferView"]]
            if set_no == 0:
                acc.update(componentType=5121, normalized=True)
                view["byteStride"] = 16
            for vertex in range(3):
                value = 128 if set_no == 0 else 1 - 128 / 255 + 1e-5
                struct.pack_into("<4B" if set_no == 0 else "<4f", binary,
                                 view["byteOffset"] + vertex * 16, value, 0, 0, 0)
        report = glb_skinning.analyze_glb(build_glb(doc, binary))
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["meshes"][0]["primitives"][0]["weight_sum"]["tolerance"],
                         glb_skinning.WEIGHT_SUM_TOLERANCE)

    def test_float_weight_tolerance_is_preserved(self):
        weights = ((0.5, 0.50001, 0, 0),) * 3
        report = glb_skinning.analyze_glb(make_glb(weights=weights, joints=((0, 1, 0, 0),) * 3))
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(report["meshes"][0]["primitives"][0]["weight_sum"]["tolerance"],
                         glb_skinning.WEIGHT_SUM_TOLERANCE)

    def test_weight_sum_not_one_identifies_primitive_and_vertex(self):
        weights = ((1.0, 0.0, 0.0, 0.0),
                   (0.5, 0.25, 0.0, 0.0),
                   (0.25, 0.75, 0.0, 0.0))
        report = glb_skinning.analyze_glb(make_glb(weights=weights))
        self.assertFalse(report["ok"])
        self.assertEqual(1, report["summary"]["bad_weight_sum_count"])
        self.assertAlmostEqual(0.75, report["summary"]["weight_sum_min"])
        messages = "\n".join(report["errors"])
        self.assertIn("primitive[0]", messages)
        self.assertIn("vertex 1", messages)

        path = self.write("bad_weight.glb", make_glb(weights=weights))
        proc = self.run_tool(path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("primitive[0]", proc.stderr)
        self.assertIn("vertex 1", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_joint_index_out_of_range(self):
        joints = ((0, 0, 0, 0), (0, 1, 0, 0), (2, 1, 0, 0))
        report = glb_skinning.analyze_glb(make_glb(joints=joints))
        self.assertFalse(report["ok"])
        self.assertEqual(1, report["summary"]["out_of_range_joint_count"])
        self.assertEqual(2, report["summary"]["joint_index_max"])
        messages = "\n".join(report["errors"])
        self.assertIn("vertex 2", messages)
        self.assertIn("joint index 2", messages)

        path = self.write("bad_joint.glb", make_glb(joints=joints))
        proc = self.run_tool(path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("joint index 2", proc.stderr)

    def test_oversized_declared_max_is_a_one_line_error(self):
        builder = Builder()
        positions = ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        position_payload = b"".join(struct.pack("<3f", *point) for point in positions)
        position = builder.add_accessor(
            position_payload, 5126, "VEC3", len(positions),
            minimum=[0.0, 0.0, 0.0], maximum=[10 ** 400, 1.0, 0.0],
        )
        joints = builder.add_accessor(bytes((0, 0, 0, 0)) * 3, 5121, "VEC4", 3)
        weights = builder.add_accessor(
            struct.pack("<12f", *([1.0, 0.0, 0.0, 0.0] * 3)), 5126, "VEC4", 3,
        )
        inverse = builder.add_accessor(struct.pack("<16f", *IDENTITY), 5126, "MAT4", 1)
        mesh = {"name": "Body", "primitives": [{"attributes": {
            "POSITION": position, "JOINTS_0": joints, "WEIGHTS_0": weights,
        }}]}
        nodes = [{"name": "Bone"}, {"name": "Body", "mesh": 0, "skin": 0}]
        skins = [{"joints": [0], "inverseBindMatrices": inverse}]
        data = build_glb(builder.document([mesh], nodes, skins, [0, 1]), builder.data)

        with self.assertRaises(glb_skinning.GlbSkinError) as caught:
            glb_skinning.analyze_glb(data)
        self.assertIn("accessors[0].max must be 3 finite number(s)", str(caught.exception))

        path = self.write("huge_max.glb", data)
        proc = self.run_tool(path)
        self.assertNotEqual(0, proc.returncode)
        lines = [line for line in proc.stderr.splitlines() if line.strip()]
        self.assertEqual(1, len(lines), proc.stderr)
        self.assertIn("accessors[0].max must be 3 finite number(s)", lines[0])
        self.assertNotIn("Traceback", proc.stderr)

    def test_skin_joint_must_reference_a_node_even_when_nodes_is_empty(self):
        builder = Builder()
        data = build_glb(
            builder.document([], [], [{"joints": [5]}], []),
            builder.data,
        )
        report = glb_skinning.analyze_glb(data)
        self.assertFalse(report["ok"])
        errors = "\n".join(report["errors"])
        self.assertIn("joint 0 references node 5", errors)
        self.assertIn("the GLB has 0 nodes", errors)

        path = self.write("missing_joint_node.glb", data)
        proc = self.run_tool(path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("references node 5", proc.stderr)
        self.assertNotIn("Traceback", proc.stderr)

    def test_inverse_bind_matrices_count_mismatch(self):
        report = glb_skinning.analyze_glb(make_glb(ibm_count=1))
        self.assertFalse(report["ok"])
        self.assertIn(
            "inverseBindMatrices count 1 != joint count 2",
            "\n".join(report["errors"]),
        )
        self.assertEqual(1, report["skins"][0]["inverse_bind_matrices"]["count"])

        path = self.write("bad_ibm.glb", make_glb(ibm_count=1))
        proc = self.run_tool(path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("inverseBindMatrices count 1 != joint count 2", proc.stderr)

    def test_static_mesh_alongside_skinned_mesh_is_reported_not_failed(self):
        report = glb_skinning.analyze_glb(make_glb(add_static=True))
        self.assertTrue(report["ok"], report["errors"])
        self.assertEqual(1, report["summary"]["skinned_primitive_count"])
        self.assertEqual(1, report["summary"]["static_primitive_count"])
        static_primitive = report["meshes"][1]["primitives"][0]
        self.assertFalse(static_primitive["skinned"])
        self.assertEqual(0, static_primitive["max_influences"])
        self.assertIsNone(static_primitive["weight_sum"]["min"])

        path = self.write("static.glb", make_glb(add_static=True))
        proc = self.run_tool(path)
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("static", proc.stdout)

    def test_unbound_skinning_attributes_use_the_file_skin(self):
        report = glb_skinning.analyze_glb(make_glb(bind_skin=False))
        self.assertTrue(report["ok"], report["errors"])
        primitive = report["meshes"][0]["primitives"][0]
        self.assertTrue(primitive["skinned"])
        self.assertEqual([0], primitive["skins"])
        self.assertEqual("file", primitive["skin_source"])
        self.assertIn("no node binds", "\n".join(report["warnings"]))
        self.assertEqual(0, report["summary"]["out_of_range_joint_count"])

        joints = ((0, 0, 0, 0), (0, 1, 0, 0), (2, 1, 0, 0))
        bad = glb_skinning.analyze_glb(make_glb(joints=joints, bind_skin=False))
        self.assertFalse(bad["ok"])
        self.assertEqual(1, bad["summary"]["out_of_range_joint_count"])

    def test_missing_skin_is_one_line_error(self):
        report = glb_skinning.analyze_glb(make_glb(include_skin=False))
        self.assertFalse(report["ok"])
        self.assertEqual(["no skin found in GLB"], report["errors"])

        path = self.write("no_skin.glb", make_glb(include_skin=False))
        proc = self.run_tool(path)
        self.assertNotEqual(0, proc.returncode)
        lines = [line for line in proc.stderr.splitlines() if line.strip()]
        self.assertEqual(1, len(lines), proc.stderr)
        self.assertTrue(lines[0].startswith("ERROR: "))
        self.assertIn("no skin", lines[0])

    def test_malformed_containers_are_one_line_errors(self):
        valid = make_glb()
        variants = {
            "bad_magic": b"XXXX" + valid[4:],
            "truncated": valid[:-4],
            "wrong_first_chunk": valid[:16] + struct.pack("<I", 1) + valid[20:],
        }
        for name, data in variants.items():
            with self.subTest(container=name):
                path = self.write(f"{name}.glb", data)
                proc = self.run_tool(path)
                self.assertNotEqual(0, proc.returncode)
                lines = [line for line in proc.stderr.splitlines() if line.strip()]
                self.assertEqual(1, len(lines), proc.stderr)
                self.assertTrue(lines[0].startswith("ERROR: "), lines[0])
                self.assertNotIn("Traceback", proc.stderr)

    def test_json_decoder_recursion_is_a_container_error(self):
        data = make_glb()
        # Decoder depth limits differ between Python versions.
        with mock.patch.object(glb_skinning.json, "loads", side_effect=RecursionError):
            with self.assertRaisesRegex(glb_skinning.GlbSkinError, "nesting is too deep"):
                glb_skinning.parse_glb(data)

    def test_json_decoder_recursion_preserves_existing_report(self):
        data = make_glb()
        path = self.write("deep.glb", data)
        previous = b"existing report\n"
        out = self.write("report.json", previous)
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(glb_skinning.json, "loads", side_effect=RecursionError):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                status = glb_skinning.main([str(path), "--json", str(out)])
        self.assertEqual(1, status)
        self.assertEqual("", stdout.getvalue())
        self.assertEqual(
            "ERROR: invalid GLB JSON chunk: nesting is too deep\n", stderr.getvalue()
        )
        self.assertEqual(previous, out.read_bytes())
        self.assertEqual(data, path.read_bytes())

    def test_normalized_byte_and_short_weights(self):
        cases = {
            "u8": ((255, 0, 0, 0), (128, 127, 0, 0), (85, 85, 85, 0)),
            "u16": ((65535, 0, 0, 0), (32768, 32767, 0, 0), (21845, 21845, 21845, 0)),
        }
        for component, weights in cases.items():
            with self.subTest(component=component):
                report = glb_skinning.analyze_glb(
                    make_glb(weights=weights, weight_component=component)
                )
                self.assertTrue(report["ok"], report["errors"])
                primitive = report["meshes"][0]["primitives"][0]
                self.assertAlmostEqual(1.0, primitive["weight_sum"]["min"], places=6)
                self.assertAlmostEqual(1.0, primitive["weight_sum"]["max"], places=6)
                self.assertEqual(0, primitive["weight_sum"]["bad_count"])
                self.assertTrue(primitive["weight_sets"][0]["normalized"])
                self.assertEqual(3, primitive["max_influences"])

    def test_two_influence_sets_are_summed_per_vertex(self):
        report = glb_skinning.analyze_glb(make_glb(two_sets=True))
        self.assertTrue(report["ok"], report["errors"])
        primitive = report["meshes"][0]["primitives"][0]
        self.assertEqual(2, len(primitive["joint_sets"]))
        self.assertEqual(2, len(primitive["weight_sets"]))
        self.assertEqual(4, primitive["max_influences"])
        self.assertAlmostEqual(1.0, primitive["weight_sum"]["min"])
        self.assertAlmostEqual(1.0, primitive["weight_sum"]["max"])
        self.assertEqual(0, primitive["weight_sum"]["bad_count"])
        self.assertEqual(0, report["summary"]["out_of_range_joint_count"])
        self.assertEqual([0, 1], report["skins"][0]["joints"])

    def test_non_finite_weight_is_reported_and_serializable(self):
        weights = ((1.0, 0.0, 0.0, 0.0),
                   (float("nan"), 0.5, 0.0, 0.0),
                   (0.25, 0.75, 0.0, 0.0))
        report = glb_skinning.analyze_glb(make_glb(weights=weights))
        self.assertFalse(report["ok"])
        self.assertEqual(1, report["summary"]["non_finite_value_count"])
        self.assertIn("non-finite weight", "\n".join(report["errors"]))
        out = self.dir / "nan_report.json"
        glb_skinning.write_json(out, report)
        loaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(1, loaded["summary"]["non_finite_value_count"])
        self.assertEqual(0, loaded["summary"]["bad_weight_sum_count"])

        path = self.write("nan.glb", make_glb(weights=weights))
        proc = self.run_tool(path)
        self.assertNotEqual(0, proc.returncode)
        self.assertIn("non-finite", proc.stderr)

    def test_json_round_trip_is_lf_only(self):
        data = make_glb(add_static=True)
        path = self.write("roundtrip.glb", data)
        out = self.dir / "report.json"
        proc = self.run_tool(path, "--json", out)
        self.assertEqual(0, proc.returncode, proc.stderr)
        raw = out.read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.endswith(b"\n"))
        loaded = json.loads(raw.decode("utf-8"))
        self.assertEqual(glb_skinning.analyze_glb(data), loaded)
        self.assertTrue(loaded["ok"])
        self.assertEqual(1, loaded["summary"]["skinned_primitive_count"])
        self.assertEqual(1, loaded["summary"]["static_primitive_count"])

    def test_json_refuses_input_alias(self):
        data = make_glb()
        path = self.write("alias.glb", data)
        before = path.read_bytes()
        nested = self.dir / "nested"
        nested.mkdir()
        aliases = [path, nested / ".." / path.name]
        for alias in aliases:
            with self.subTest(alias=str(alias)):
                proc = self.run_tool(path, "--json", alias)
                self.assertNotEqual(0, proc.returncode)
                self.assertIn("must not overwrite the input", proc.stderr)
                self.assertEqual(before, path.read_bytes())

    def test_accessor_byte_stride_and_offset(self):
        joints = ((0, 0, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0))
        weights = ((1.0, 0.0, 0.0, 0.0),
                   (0.5, 0.5, 0.0, 0.0),
                   (0.25, 0.75, 0.0, 0.0))
        interleaved = b"".join(
            struct.pack("<4B", *joint_row) + b"\0" * 4 + struct.pack("<4f", *weight_row)
            for joint_row, weight_row in zip(joints, weights)
        )
        document = {
            "buffers": [{"byteLength": len(interleaved)}],
            "bufferViews": [{
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": len(interleaved),
                "byteStride": 24,
            }],
            "accessors": [
                {"bufferView": 0, "byteOffset": 0, "componentType": 5121,
                 "count": 3, "type": "VEC4"},
                {"bufferView": 0, "byteOffset": 8, "componentType": 5126,
                 "count": 3, "type": "VEC4"},
            ],
        }
        reader = glb_skinning._Reader(document, interleaved)
        joint_values, joint_meta = reader.read(0, "JOINTS_0")
        weight_values, weight_meta = reader.read(1, "WEIGHTS_0")
        self.assertEqual(list(joints), joint_values)
        self.assertEqual(list(weights), weight_values)
        self.assertEqual(0, joint_meta["buffer_view"])
        self.assertEqual(0, weight_meta["buffer_view"])
        self.assertEqual(0, joint_meta["accessor"])
        self.assertEqual(1, weight_meta["accessor"])

    def test_help(self):
        proc = self.run_tool("--help")
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertIn("input.glb", proc.stdout)


if __name__ == "__main__":
    unittest.main()
