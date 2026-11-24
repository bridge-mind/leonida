"""Validate the skinning data in a binary glTF 2.0 (.glb) file.

Usage:
    python -I tools/glb_skinning.py <in.glb> [--json report.json]

The tool parses the GLB container itself (12-byte header, JSON chunk, optional
BIN chunk, 4-byte padding), then checks every skin and every mesh primitive:

* skin names, unique joint nodes, joint counts and inverseBindMatrices counts,
* the inverseBindMatrices accessor type and finite values,
* which mesh primitives carry skinning data and which are static,
* consecutive JOINTS_n/WEIGHTS_n pairs starting at 0, accessor types and vertex counts,
* per-vertex weight sums, handled as FLOAT, normalized UNSIGNED_BYTE or
  normalized UNSIGNED_SHORT (exact integer sums for all-quantized sets),
* joint-index ranges and out-of-range counts,
* non-finite POSITION, weight and inverseBindMatrices values,
* the POSITION bounding box, computed from the accessor data.

Missing optional fields are reported as null; they are never filled in with a
guess. A malformed container, a file with no skin, or a skin whose
inverseBindMatrices count differs from its joint count exits nonzero. Structural
problems also exit nonzero with a one-line ``ERROR:`` message.

Each joint's inverse bind matrix must have the affine fourth row (0, 0, 0, 1).

Only the standard library is used, and the input is read, never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import struct
import sys

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from report_paths import same_report_path  # noqa: E402

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
CHUNK_TYPE_JSON = 0x4E4F534A
CHUNK_TYPE_BIN = 0x004E4942

WEIGHT_SUM_TOLERANCE = 1e-4
WEIGHT_RANGE_TOLERANCE = 1e-6

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

COMPONENT_TYPES = {
    5120: ("b", 1, "BYTE"),
    5121: ("B", 1, "UNSIGNED_BYTE"),
    5122: ("h", 2, "SHORT"),
    5123: ("H", 2, "UNSIGNED_SHORT"),
    5125: ("I", 4, "UNSIGNED_INT"),
    5126: ("f", 4, "FLOAT"),
}

MODE_NAMES = {
    0: "POINTS",
    1: "LINES",
    2: "LINE_LOOP",
    3: "LINE_STRIP",
    4: "TRIANGLES",
    5: "TRIANGLE_STRIP",
    6: "TRIANGLE_FAN",
}

ATTRIBUTE_RE = re.compile(r"^(JOINTS|WEIGHTS)_(\d+)$")
_MISSING = object()


class GlbSkinError(ValueError):
    """Raised when the GLB container or a required skin field is malformed."""


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _int_field(obj, key, path, default=_MISSING, minimum=0):
    value = obj.get(key, default)
    if value is _MISSING:
        raise GlbSkinError(f"{path}.{key} is missing")
    if not _is_int(value):
        raise GlbSkinError(f"{path}.{key} must be an integer")
    if minimum is not None and value < minimum:
        raise GlbSkinError(f"{path}.{key} must be >= {minimum}")
    return value


def _list_field(obj, key, path):
    value = obj.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise GlbSkinError(f"{path}.{key} must be an array")
    return value


def _name_field(obj, path, warnings):
    value = obj.get("name")
    if value is None:
        return None
    if not isinstance(value, str):
        warnings.append(f"{path}.name is not a string")
        return None
    return value


def _number(value):
    return _is_int(value) or isinstance(value, float)


def _finite_number(value):
    """True for a finite JSON number; an integer too large for float is not one."""
    if not _number(value):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _reject_json_constant(token):
    raise ValueError(f"non-finite JSON number {token}")


def parse_glb(data):
    """Return ``(gltf, bin_chunk, container)`` for a binary glTF 2.0 container."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise GlbSkinError("GLB data must be bytes-like")
    raw = bytes(data)
    if len(raw) < 12:
        raise GlbSkinError(f"GLB too short for a header: {len(raw)} bytes")

    magic, version, total_length = struct.unpack_from("<4sII", raw, 0)
    if magic != GLB_MAGIC:
        raise GlbSkinError(f"not a GLB: magic is {magic!r}, expected {GLB_MAGIC!r}")
    if version != GLB_VERSION:
        raise GlbSkinError(f"unsupported GLB version {version}, expected {GLB_VERSION}")
    if total_length != len(raw):
        raise GlbSkinError(f"GLB length field says {total_length} bytes, file has {len(raw)}")
    if total_length % 4:
        raise GlbSkinError(f"GLB total length {total_length} is not 4-byte aligned")

    offset = 12
    json_chunk = None
    bin_chunk = None
    trailing_chunks = 0
    chunk_index = 0
    while offset < total_length:
        if offset + 8 > total_length:
            raise GlbSkinError("truncated GLB chunk header")
        chunk_length, chunk_type = struct.unpack_from("<II", raw, offset)
        if chunk_length % 4:
            raise GlbSkinError(f"GLB chunk length {chunk_length} is not 4-byte aligned")
        start = offset + 8
        end = start + chunk_length
        if end > total_length:
            raise GlbSkinError(
                f"GLB chunk at {offset} declares {chunk_length} bytes, past the end of the file"
            )
        payload = raw[start:end]
        if chunk_index == 0:
            if chunk_type != CHUNK_TYPE_JSON:
                raise GlbSkinError(
                    f"first GLB chunk must be JSON (0x{CHUNK_TYPE_JSON:08X}), "
                    f"got 0x{chunk_type:08X}"
                )
            json_chunk = payload
        elif chunk_index == 1 and chunk_type == CHUNK_TYPE_BIN:
            bin_chunk = payload
        else:
            if chunk_type == CHUNK_TYPE_JSON:
                raise GlbSkinError("repeated JSON chunk in GLB")
            if chunk_type == CHUNK_TYPE_BIN:
                raise GlbSkinError("repeated BIN chunk in GLB")
            trailing_chunks += 1
        offset = end
        chunk_index += 1

    if json_chunk is None:
        raise GlbSkinError("GLB has no JSON chunk")
    try:
        text = json_chunk.decode("utf-8")
        gltf = json.loads(text, parse_constant=_reject_json_constant)
    except RecursionError as exc:
        raise GlbSkinError("invalid GLB JSON chunk: nesting is too deep") from exc
    except (UnicodeDecodeError, ValueError) as exc:
        raise GlbSkinError(f"invalid GLB JSON chunk: {exc}") from exc
    if not isinstance(gltf, dict):
        raise GlbSkinError("GLB JSON root must be an object")

    asset = gltf.get("asset")
    asset_version = asset.get("version") if isinstance(asset, dict) else None
    if isinstance(asset_version, str):
        if not asset_version.startswith("2."):
            raise GlbSkinError(f"unsupported glTF version {asset_version!r}")
    else:
        asset_version = None

    container = {
        "version": version,
        "asset_version": asset_version,
        "bytes": total_length,
        "json_bytes": len(json_chunk),
        "bin_bytes": len(bin_chunk) if bin_chunk is not None else 0,
        "bin_present": bin_chunk is not None,
        "trailing_chunks": trailing_chunks,
    }
    return gltf, bin_chunk, container


class _Reader:
    """Read accessor rows from the single embedded buffer of a GLB."""

    def __init__(self, gltf, bin_chunk):
        self.gltf = gltf
        self.bin_chunk = bin_chunk
        self.buffers = _list_field(gltf, "buffers", "glTF")
        self.buffer_views = _list_field(gltf, "bufferViews", "glTF")
        self.accessors = _list_field(gltf, "accessors", "glTF")
        self.buffer_limit = self._buffer_limit()

    def _buffer_limit(self):
        if not self.buffers:
            return 0
        buffer0 = self.buffers[0]
        if not isinstance(buffer0, dict):
            raise GlbSkinError("buffers[0] is not an object")
        if "uri" in buffer0:
            raise GlbSkinError("external buffer URIs are not supported; expected an embedded GLB")
        declared = _int_field(buffer0, "byteLength", "buffers[0]")
        if self.bin_chunk is None:
            raise GlbSkinError("buffer 0 is embedded but the GLB has no BIN chunk")
        if declared > len(self.bin_chunk):
            raise GlbSkinError(
                f"buffers[0].byteLength {declared} exceeds the BIN chunk ({len(self.bin_chunk)} bytes)"
            )
        return declared

    def accessor(self, index, path):
        if not _is_int(index):
            raise GlbSkinError(f"{path} must be an accessor index")
        if index < 0 or index >= len(self.accessors):
            raise GlbSkinError(
                f"{path} references accessor {index}, but the GLB has {len(self.accessors)} accessors"
            )
        accessor = self.accessors[index]
        if not isinstance(accessor, dict):
            raise GlbSkinError(f"accessors[{index}] is not an object")
        return accessor

    def read(self, index, path):
        accessor = self.accessor(index, path)
        component_type = _int_field(accessor, "componentType", f"accessors[{index}]")
        type_name = accessor.get("type")
        if not isinstance(type_name, str):
            raise GlbSkinError(f"accessors[{index}].type is missing")
        if type_name not in TYPE_COMPONENTS:
            raise GlbSkinError(f"accessors[{index}].type {type_name!r} is not a glTF accessor type")
        count = _int_field(accessor, "count", f"accessors[{index}]")
        normalized = accessor.get("normalized", False)
        if not isinstance(normalized, bool):
            raise GlbSkinError(f"accessors[{index}].normalized must be a boolean")
        if "sparse" in accessor:
            raise GlbSkinError(f"{path}: sparse accessor {index} is not supported")
        if component_type not in COMPONENT_TYPES:
            raise GlbSkinError(f"accessors[{index}].componentType {component_type} is not supported")

        fmt_char, component_size, component_name = COMPONENT_TYPES[component_type]
        components = TYPE_COMPONENTS[type_name]
        element_size = component_size * components
        meta = {
            "accessor": index,
            "count": count,
            "type": type_name,
            "component_type": component_type,
            "component_type_name": component_name,
            "components": components,
            "normalized": normalized,
            "buffer_view": None,
            "declared_min": None,
            "declared_max": None,
        }
        for key, label in (("min", "declared_min"), ("max", "declared_max")):
            if key not in accessor:
                continue
            value = accessor[key]
            if (
                not isinstance(value, list)
                or len(value) != components
                or not all(_finite_number(v) for v in value)
            ):
                raise GlbSkinError(
                    f"accessors[{index}].{key} must be {components} finite number(s)"
                )
            meta[label] = [float(v) for v in value]

        if "bufferView" not in accessor:
            zero = 0.0 if component_type == 5126 else 0
            values = [tuple(zero for _ in range(components)) for _ in range(count)]
            return values, meta

        view_index = _int_field(accessor, "bufferView", f"accessors[{index}]")
        if view_index >= len(self.buffer_views):
            raise GlbSkinError(
                f"accessors[{index}].bufferView {view_index} is out of range "
                f"({len(self.buffer_views)} bufferViews)"
            )
        view = self.buffer_views[view_index]
        if not isinstance(view, dict):
            raise GlbSkinError(f"bufferViews[{view_index}] is not an object")
        buffer_index = _int_field(view, "buffer", f"bufferViews[{view_index}]", default=0)
        if buffer_index != 0:
            raise GlbSkinError(
                f"bufferViews[{view_index}].buffer {buffer_index} is not supported; "
                "only embedded buffer 0 is read"
            )
        view_offset = _int_field(view, "byteOffset", f"bufferViews[{view_index}]", default=0)
        view_length = _int_field(view, "byteLength", f"bufferViews[{view_index}]")
        if view_offset + view_length > self.buffer_limit:
            raise GlbSkinError(
                f"bufferViews[{view_index}] exceeds buffer 0 "
                f"({view_offset}+{view_length} > {self.buffer_limit})"
            )
        accessor_offset = _int_field(accessor, "byteOffset", f"accessors[{index}]", default=0)
        stride = view.get("byteStride")
        if stride is None:
            stride = element_size
        else:
            if not _is_int(stride) or stride < element_size:
                raise GlbSkinError(
                    f"bufferViews[{view_index}].byteStride must be an integer >= {element_size}"
                )
            if stride % component_size:
                raise GlbSkinError(
                    f"bufferViews[{view_index}].byteStride {stride} is not a multiple "
                    f"of the component size {component_size}"
                )
        if count:
            last_end = accessor_offset + (count - 1) * stride + element_size
            if last_end > view_length:
                raise GlbSkinError(
                    f"accessors[{index}] needs {last_end} bytes of bufferViews[{view_index}], "
                    f"which has {view_length}"
                )

        if self.buffer_limit == 0:
            raise GlbSkinError(f"accessors[{index}] needs buffer 0, but the GLB has no embedded buffer")
        data = self.bin_chunk[: self.buffer_limit]
        unpacker = struct.Struct("<" + fmt_char * components)
        base = view_offset + accessor_offset
        values = []
        for i in range(count):
            values.append(unpacker.unpack_from(data, base + i * stride))
        meta["buffer_view"] = view_index
        return values, meta


def _normalized(value, component_type):
    if component_type == 5121:
        return value / 255.0
    if component_type == 5123:
        return value / 65535.0
    if component_type == 5120:
        return max(value / 127.0, -1.0)
    if component_type == 5122:
        return max(value / 32767.0, -1.0)
    return float(value)


def _mesh_label(mesh_index, mesh_name):
    suffix = f" {json.dumps(mesh_name)}" if mesh_name is not None else ""
    return f"mesh[{mesh_index}]{suffix}"


def _primitive_label(mesh_index, mesh_name, primitive_index):
    return f"{_mesh_label(mesh_index, mesh_name)} primitive[{primitive_index}]"


def _skin_label(skin_index, skin_name):
    suffix = f" {json.dumps(skin_name)}" if skin_name is not None else ""
    return f"skin[{skin_index}]{suffix}"


def _bbox(values):
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    non_finite = 0
    finite = 0
    for row in values:
        for axis in range(3):
            value = row[axis]
            if not math.isfinite(value):
                non_finite += 1
                continue
            finite += 1
            if value < minimum[axis]:
                minimum[axis] = value
            if value > maximum[axis]:
                maximum[axis] = value
    if not finite:
        return None, None, non_finite
    return minimum, maximum, non_finite


def _validate_skins(gltf, reader, warnings, errors):
    nodes = _list_field(gltf, "nodes", "glTF")
    skins = _list_field(gltf, "skins", "glTF")
    reports = []
    if not skins:
        errors.append("no skin found in GLB")
        return reports

    for skin_index, skin in enumerate(skins):
        path = f"skins[{skin_index}]"
        if not isinstance(skin, dict):
            raise GlbSkinError(f"{path} is not an object")
        name = _name_field(skin, path, warnings)
        joints_present = "joints" in skin
        joints = _list_field(skin, "joints", path)
        if not joints:
            errors.append(
                f"{_skin_label(skin_index, name)} has "
                f"{'no joints' if joints_present else 'no joints field'}"
            )
        seen_joints = set()
        for joint_offset, joint in enumerate(joints):
            if not _is_int(joint):
                raise GlbSkinError(f"{path}.joints[{joint_offset}] must be an integer")
            if joint in seen_joints:
                errors.append(f"{path}.joints[{joint_offset}]: duplicate joint node {joint}")
            seen_joints.add(joint)
            if joint < 0 or joint >= len(nodes):
                errors.append(
                    f"{_skin_label(skin_index, name)} joint {joint_offset} references "
                    f"node {joint}, but the GLB has {len(nodes)} nodes"
                )

        skeleton = skin.get("skeleton")
        if skeleton is not None:
            if not _is_int(skeleton):
                raise GlbSkinError(f"{path}.skeleton must be an integer")
            if skeleton < 0 or skeleton >= len(nodes):
                errors.append(
                    f"{_skin_label(skin_index, name)} skeleton references node {skeleton}, "
                    f"but the GLB has {len(nodes)} nodes"
                )

        inverse = {
            "present": False,
            "accessor": None,
            "count": None,
            "type": None,
            "component_type": None,
            "normalized": None,
            "non_finite_count": 0,
        }
        inverse_index = skin.get("inverseBindMatrices")
        if inverse_index is None:
            warnings.append(
                f"{_skin_label(skin_index, name)} has no inverseBindMatrices; "
                "the glTF default is identity and was not assumed here"
            )
        else:
            if not _is_int(inverse_index):
                raise GlbSkinError(f"{path}.inverseBindMatrices must be an accessor index")
            values, meta = reader.read(inverse_index, f"skin[{skin_index}].inverseBindMatrices")
            inverse.update({
                "present": True,
                "accessor": meta["accessor"],
                "count": meta["count"],
                "type": meta["type"],
                "component_type": meta["component_type"],
                "normalized": meta["normalized"],
            })
            if meta["count"] != len(joints):
                errors.append(
                    f"{_skin_label(skin_index, name)}: inverseBindMatrices count "
                    f"{meta['count']} != joint count {len(joints)}"
                )
            if meta["type"] != "MAT4":
                errors.append(
                    f"{_skin_label(skin_index, name)}: inverseBindMatrices accessor type "
                    f"{meta['type']!r} is not MAT4"
                )
            if meta["component_type"] != 5126:
                errors.append(
                    f"{_skin_label(skin_index, name)}: inverseBindMatrices component type "
                    f"{meta['component_type_name']} is not FLOAT"
                )
            bad = 0
            first_non_affine = None
            for index, row in enumerate(values):
                bad += sum(1 for value in row if not math.isfinite(value))
                # glTF stores MAT4 columns consecutively; translation is in the last column.
                if (first_non_affine is None and index < len(joints) and meta["type"] == "MAT4"
                        and row[3::4] != (0.0, 0.0, 0.0, 1.0)):
                    first_non_affine = index
            if first_non_affine is not None:
                errors.append(
                    f"{_skin_label(skin_index, name)}: inverseBindMatrices[{first_non_affine}] "
                    "fourth row must be [0, 0, 0, 1]"
                )
            inverse["non_finite_count"] = bad
            if bad:
                errors.append(
                    f"{_skin_label(skin_index, name)}: inverseBindMatrices has "
                    f"{bad} non-finite value(s)"
                )

        reports.append({
            "index": skin_index,
            "name": name,
            "joints_present": joints_present,
            "joint_count": len(joints),
            "joints": [int(joint) if _is_int(joint) else joint for joint in joints],
            "skeleton": skeleton,
            "inverse_bind_matrices": inverse,
        })
    return reports


def _node_bindings(gltf, skin_count, mesh_count, warnings, errors):
    nodes = _list_field(gltf, "nodes", "glTF")
    bindings = {}
    for node_index, node in enumerate(nodes):
        path = f"nodes[{node_index}]"
        if not isinstance(node, dict):
            raise GlbSkinError(f"{path} is not an object")
        mesh_index = node.get("mesh")
        skin_index = node.get("skin")
        if mesh_index is not None:
            if not _is_int(mesh_index):
                raise GlbSkinError(f"{path}.mesh must be an integer")
            if mesh_index < 0 or mesh_index >= mesh_count:
                errors.append(
                    f"{path} references mesh {mesh_index}, but the GLB has {mesh_count} meshes"
                )
                mesh_index = None
        if skin_index is not None:
            if not _is_int(skin_index):
                raise GlbSkinError(f"{path}.skin must be an integer")
            if skin_index < 0 or skin_index >= skin_count:
                if skin_count:
                    errors.append(
                        f"{path} references skin {skin_index}, "
                        f"but the GLB has {skin_count} skins"
                    )
                # With no skins at all the single "no skin found" error is enough.
                skin_index = None
        if mesh_index is None:
            if skin_index is not None:
                warnings.append(f"{path} has a skin but no mesh")
            continue
        bindings.setdefault(mesh_index, []).append({
            "node": node_index,
            "skin": skin_index,
        })
    return bindings


def _read_weight_set(reader, accessor_index, primitive_path, set_index, errors):
    values, meta = reader.read(accessor_index, f"{primitive_path}.WEIGHTS_{set_index}")
    if meta["type"] != "VEC4":
        raise GlbSkinError(
            f"{primitive_path}.WEIGHTS_{set_index} accessor type is {meta['type']!r}, expected VEC4"
        )
    component_type = meta["component_type"]
    if component_type not in (5126, 5121, 5123):
        errors.append(
            f"{primitive_path}.WEIGHTS_{set_index}: component type "
            f"{meta['component_type_name']} is not FLOAT, UNSIGNED_BYTE or UNSIGNED_SHORT"
        )
    if component_type in (5121, 5123) and not meta["normalized"]:
        errors.append(
            f"{primitive_path}.WEIGHTS_{set_index}: integer weights are not normalized"
        )
    if component_type == 5126 and meta["normalized"]:
        errors.append(
            f"{primitive_path}.WEIGHTS_{set_index}: FLOAT weights must not be normalized"
        )
    rows = []
    for row in values:
        if meta["normalized"] and component_type in (5121, 5123):
            rows.append([_normalized(value, component_type) for value in row])
        else:
            rows.append([float(value) for value in row])
    finite = [value for row in rows for value in row if math.isfinite(value)]
    non_finite = sum(1 for row in rows for value in row if not math.isfinite(value))
    report = {
        "set": set_index,
        "accessor": meta["accessor"],
        "count": meta["count"],
        "component_type": meta["component_type"],
        "component_type_name": meta["component_type_name"],
        "normalized": meta["normalized"],
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "non_finite_count": non_finite,
    }
    # Preserve integer sums before dequantization: u8 * 257 and u16 share denominator 65535.
    units = None
    if meta["normalized"] and component_type in (5121, 5123):
        factor = 257 if component_type == 5121 else 1
        units = [sum(row) * factor for row in values]
    return rows, meta, report, units


def _read_joint_set(reader, accessor_index, primitive_path, set_index, errors):
    values, meta = reader.read(accessor_index, f"{primitive_path}.JOINTS_{set_index}")
    if meta["type"] != "VEC4":
        raise GlbSkinError(
            f"{primitive_path}.JOINTS_{set_index} accessor type is {meta['type']!r}, expected VEC4"
        )
    if meta["component_type"] not in (5121, 5123):
        raise GlbSkinError(
            f"{primitive_path}.JOINTS_{set_index}: component type "
            f"{meta['component_type_name']} is not UNSIGNED_BYTE or UNSIGNED_SHORT"
        )
    if meta["normalized"]:
        errors.append(f"{primitive_path}.JOINTS_{set_index}: joint indices must not be normalized")
    indices = [[int(value) for value in row] for row in values]
    flat = [value for row in indices for value in row]
    report = {
        "set": set_index,
        "accessor": meta["accessor"],
        "count": meta["count"],
        "component_type": meta["component_type"],
        "component_type_name": meta["component_type_name"],
        "min": min(flat) if flat else None,
        "max": max(flat) if flat else None,
        "out_of_range": None,
        "bindings": [],
    }
    return indices, meta, report


def _validate_primitive(reader, primitive, mesh_index, mesh_name, primitive_index,
                        bindings, skins, warnings, errors):
    path = _primitive_label(mesh_index, mesh_name, primitive_index)
    if not isinstance(primitive, dict):
        raise GlbSkinError(f"{path} is not an object")
    attributes = primitive.get("attributes")
    if not isinstance(attributes, dict):
        warnings.append(f"{path} has no attributes object")
        attributes = {}

    mode = primitive.get("mode")
    if mode is not None and not _is_int(mode):
        raise GlbSkinError(f"{path}.mode must be an integer")
    mode_name = MODE_NAMES.get(mode) if mode is not None else None

    position_report = {
        "accessor": None,
        "count": None,
        "type": None,
        "component_type": None,
        "bbox_min": None,
        "bbox_max": None,
        "declared_min": None,
        "declared_max": None,
        "non_finite_count": 0,
    }
    position_count = None
    non_finite_positions = 0
    if "POSITION" not in attributes:
        warnings.append(f"{path} has no POSITION accessor; its bounding box is unknown")
    else:
        values, meta = reader.read(attributes["POSITION"], f"{path}.POSITION")
        if meta["type"] != "VEC3":
            raise GlbSkinError(
                f"{path}.POSITION accessor type is {meta['type']!r}, expected VEC3"
            )
        minimum, maximum, non_finite_positions = _bbox(values)
        position_count = meta["count"]
        position_report.update({
            "accessor": meta["accessor"],
            "count": meta["count"],
            "type": meta["type"],
            "component_type": meta["component_type_name"],
            "bbox_min": minimum,
            "bbox_max": maximum,
            "declared_min": meta["declared_min"],
            "declared_max": meta["declared_max"],
            "non_finite_count": non_finite_positions,
        })
        if non_finite_positions:
            errors.append(
                f"{path}.POSITION has {non_finite_positions} non-finite component(s)"
            )

    set_numbers = sorted({
        int(match.group(2))
        for key in attributes
        for match in [ATTRIBUTE_RE.match(key)]
        if match
    })
    if any(number != expected for expected, number in enumerate(set_numbers)):
        errors.append(f"{path}: JOINTS/WEIGHTS sets must be consecutive starting at 0")
    joint_reports = []
    weight_reports = []
    weight_rows_by_set = {}
    quantized_units_by_set = {}
    joint_rows_by_set = {}
    first_set_count = None
    for set_number in set_numbers:
        joint_key = f"JOINTS_{set_number}"
        weight_key = f"WEIGHTS_{set_number}"
        if joint_key not in attributes:
            errors.append(f"{path} has {weight_key} but no {joint_key}")
        if weight_key not in attributes:
            errors.append(f"{path} has {joint_key} but no {weight_key}")
        if joint_key in attributes:
            joint_rows, meta, joint_report = _read_joint_set(
                reader, attributes[joint_key], path, set_number, errors
            )
            joint_rows_by_set[set_number] = joint_rows
            first_set_count = meta["count"] if first_set_count is None else first_set_count
            joint_reports.append(joint_report)
        if weight_key in attributes:
            weight_rows, meta, weight_report, units = _read_weight_set(
                reader, attributes[weight_key], path, set_number, errors
            )
            weight_rows_by_set[set_number] = weight_rows
            if units is not None:
                quantized_units_by_set[set_number] = units
            first_set_count = meta["count"] if first_set_count is None else first_set_count
            weight_reports.append(weight_report)

    if position_count is None:
        vertex_count = first_set_count
    else:
        vertex_count = position_count
    for set_number, rows in weight_rows_by_set.items():
        if vertex_count is not None and len(rows) != vertex_count:
            errors.append(
                f"{path}.WEIGHTS_{set_number} has {len(rows)} vertices, "
                f"expected {vertex_count}"
            )
    for set_number, rows in joint_rows_by_set.items():
        if vertex_count is not None and len(rows) != vertex_count:
            errors.append(
                f"{path}.JOINTS_{set_number} has {len(rows)} vertices, "
                f"expected {vertex_count}"
            )

    node_skins = sorted({binding["skin"] for binding in bindings
                         if binding["skin"] is not None})
    has_skin_attributes = bool(set_numbers)
    skinned = has_skin_attributes or bool(node_skins)
    if skinned and not has_skin_attributes:
        errors.append(f"{path} is bound to a skin but has no JOINTS_0/WEIGHTS_0")
    if has_skin_attributes and not node_skins and skins:
        warnings.append(
            f"{path} has JOINTS_0/WEIGHTS_0 but no node binds it to a skin; "
            f"checked against the {len(skins)} skin(s) in the file"
        )
    if node_skins:
        checked_skins = node_skins
        skin_source = "node"
    elif has_skin_attributes:
        checked_skins = list(range(len(skins)))
        skin_source = "file" if skins else None
    else:
        checked_skins = []
        skin_source = None

    binding_limits = []
    for skin_index in checked_skins:
        skin = skins[skin_index]
        binding_limits.append({
            "skin": skin_index,
            "name": skin["name"],
            "joint_count": skin["joint_count"],
        })

    joint_min = None
    joint_max = None
    out_of_range_total = 0
    for joint_report in joint_reports:
        rows = joint_rows_by_set[joint_report["set"]]
        flat = [value for row in rows for value in row]
        if flat:
            report_min = min(flat)
            report_max = max(flat)
            joint_min = report_min if joint_min is None else min(joint_min, report_min)
            joint_max = report_max if joint_max is None else max(joint_max, report_max)
            joint_report["min"] = report_min
            joint_report["max"] = report_max
        for binding in binding_limits:
            count = binding["joint_count"]
            bad_count = 0
            first_bad = None
            for vertex, row in enumerate(rows):
                for component, value in enumerate(row):
                    if value < 0 or value >= count:
                        bad_count += 1
                        if first_bad is None:
                            first_bad = (vertex, component, value)
            binding_report = {
                "skin": binding["skin"],
                "name": binding["name"],
                "joint_count": count,
                "out_of_range": bad_count,
            }
            joint_report["bindings"].append(binding_report)
            if first_bad is not None:
                vertex, component, value = first_bad
                errors.append(
                    f"{path} vertex {vertex} joint component {component}: joint index "
                    f"{value} is outside {_skin_label(binding['skin'], binding['name'])} "
                    f"({count} joints)"
                )
        if binding_limits:
            tightest = min(binding["joint_count"] for binding in binding_limits)
            joint_report["out_of_range"] = sum(
                1 for row in rows for value in row if value < 0 or value >= tightest
            )
            out_of_range_total += joint_report["out_of_range"]
        else:
            joint_report["out_of_range"] = None

    weight_min = None
    weight_max = None
    bad_weight_sums = 0
    first_bad_sum = None
    non_finite_weights = 0
    out_of_range_weights = 0
    first_bad_weight = None
    max_influences = 0
    for set_number in sorted(weight_rows_by_set):
        for vertex, row in enumerate(weight_rows_by_set[set_number]):
            for value in row:
                if not math.isfinite(value):
                    non_finite_weights += 1
                elif value < -WEIGHT_RANGE_TOLERANCE or value > 1.0 + WEIGHT_RANGE_TOLERANCE:
                    out_of_range_weights += 1
                    if first_bad_weight is None:
                        first_bad_weight = (set_number, vertex, value)
    common_weight_count = min(
        (len(rows) for rows in weight_rows_by_set.values()), default=0
    )
    all_quantized = bool(weight_rows_by_set) and len(quantized_units_by_set) == len(weight_rows_by_set)
    sum_tolerance = 0.0 if all_quantized else WEIGHT_SUM_TOLERANCE
    for vertex in range(common_weight_count):
        values = []
        for set_number in sorted(weight_rows_by_set):
            values.extend(weight_rows_by_set[set_number][vertex])
        influences = sum(1 for value in values if math.isfinite(value) and value != 0.0)
        max_influences = max(max_influences, influences)
        if values and all(math.isfinite(value) for value in values):
            total = sum(values)
            weight_min = total if weight_min is None else min(weight_min, total)
            weight_max = total if weight_max is None else max(weight_max, total)
            bad_sum = (sum(units[vertex] for units in quantized_units_by_set.values()) != 65535
                       if all_quantized else abs(total - 1.0) > sum_tolerance)
            if bad_sum:
                bad_weight_sums += 1
                if first_bad_sum is None:
                    first_bad_sum = (vertex, total)
    if first_bad_sum is not None:
        vertex, total = first_bad_sum
        suffix = f"; {bad_weight_sums} vertices bad" if bad_weight_sums > 1 else ""
        errors.append(
            f"{path} vertex {vertex}: weight sum {total:.7g} != 1 "
            f"(tolerance {sum_tolerance:g}){suffix}"
        )
    if non_finite_weights:
        errors.append(f"{path} has {non_finite_weights} non-finite weight value(s)")
    if out_of_range_weights:
        set_number, vertex, value = first_bad_weight
        errors.append(
            f"{path} WEIGHTS_{set_number} vertex {vertex}: weight {value:.7g} is outside [0, 1] "
            f"({out_of_range_weights} value(s) bad)"
        )
    return {
        "index": primitive_index,
        "mode": mode,
        "mode_name": mode_name,
        "skinned": skinned,
        "skins": checked_skins,
        "skin_source": skin_source,
        "vertex_count": vertex_count,
        "position": position_report,
        "joint_sets": joint_reports,
        "weight_sets": weight_reports,
        "max_influences": max_influences,
        "weight_sum": {
            "min": weight_min,
            "max": weight_max,
            "bad_count": bad_weight_sums,
            "tolerance": sum_tolerance,
        },
        "non_finite": {
            "positions": non_finite_positions,
            "weights": non_finite_weights,
            "total": non_finite_positions + non_finite_weights,
        },
        "_joint_min": joint_min,
        "_joint_max": joint_max,
        "_out_of_range_joints": out_of_range_total,
    }


def analyze_glb(data):
    """Validate a GLB and return a JSON-serializable report."""
    gltf, bin_chunk, container = parse_glb(data)
    reader = _Reader(gltf, bin_chunk)
    warnings = []
    errors = []
    if container["asset_version"] is None:
        warnings.append("asset.version is missing; the GLB header still says version 2")
    skins = _validate_skins(gltf, reader, warnings, errors)
    meshes = _list_field(gltf, "meshes", "glTF")
    mesh_bindings = _node_bindings(gltf, len(skins), len(meshes), warnings, errors)
    if not meshes:
        warnings.append("GLB has no meshes")
    mesh_reports = []
    summary = {
        "skin_count": len(skins),
        "skin_names": [skin["name"] for skin in skins],
        "mesh_count": len(meshes),
        "primitive_count": 0,
        "skinned_primitive_count": 0,
        "static_primitive_count": 0,
        "max_influences": None,
        "weight_sum_min": None,
        "weight_sum_max": None,
        "bad_weight_sum_count": 0,
        "joint_index_min": None,
        "joint_index_max": None,
        "out_of_range_joint_count": 0,
        "non_finite_value_count": sum(
            skin["inverse_bind_matrices"]["non_finite_count"] for skin in skins
        ),
    }
    for mesh_index, mesh in enumerate(meshes):
        path = f"meshes[{mesh_index}]"
        if not isinstance(mesh, dict):
            raise GlbSkinError(f"{path} is not an object")
        name = _name_field(mesh, path, warnings)
        primitives = _list_field(mesh, "primitives", path)
        if not primitives:
            warnings.append(f"{_mesh_label(mesh_index, name)} has no primitives")
        bindings = mesh_bindings.get(mesh_index, [])
        if not bindings:
            warnings.append(f"{_mesh_label(mesh_index, name)} is not referenced by any node")
        primitive_reports = []
        for primitive_index, primitive in enumerate(primitives):
            report = _validate_primitive(
                reader, primitive, mesh_index, name, primitive_index,
                bindings, skins, warnings, errors,
            )
            primitive_reports.append(report)
            summary["primitive_count"] += 1
            if report["skinned"]:
                summary["skinned_primitive_count"] += 1
            else:
                summary["static_primitive_count"] += 1
            if summary["max_influences"] is None:
                summary["max_influences"] = report["max_influences"]
            else:
                summary["max_influences"] = max(
                    summary["max_influences"], report["max_influences"]
                )
            weight_sum = report["weight_sum"]
            if weight_sum["min"] is not None:
                if summary["weight_sum_min"] is None:
                    summary["weight_sum_min"] = weight_sum["min"]
                    summary["weight_sum_max"] = weight_sum["max"]
                else:
                    summary["weight_sum_min"] = min(summary["weight_sum_min"], weight_sum["min"])
                    summary["weight_sum_max"] = max(summary["weight_sum_max"], weight_sum["max"])
            summary["bad_weight_sum_count"] += weight_sum["bad_count"]
            joint_min = report["_joint_min"]
            joint_max = report["_joint_max"]
            if joint_min is not None:
                summary["joint_index_min"] = (
                    joint_min if summary["joint_index_min"] is None
                    else min(summary["joint_index_min"], joint_min)
                )
                summary["joint_index_max"] = (
                    joint_max if summary["joint_index_max"] is None
                    else max(summary["joint_index_max"], joint_max)
                )
            summary["out_of_range_joint_count"] += report["_out_of_range_joints"]
            summary["non_finite_value_count"] += report["non_finite"]["total"]
            for key in ("_joint_min", "_joint_max", "_out_of_range_joints"):
                report.pop(key, None)
        mesh_reports.append({
            "index": mesh_index,
            "name": name,
            "primitives_present": "primitives" in mesh,
            "primitive_count": len(primitives),
            "node_references": bindings,
            "primitives": primitive_reports,
        })

    return {
        "ok": not errors,
        "container": container,
        "skins": skins,
        "meshes": mesh_reports,
        "summary": summary,
        "warnings": warnings,
        "errors": errors,
    }


def _fmt(value):
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.7g}"
    return str(value)


def _fmt_list(values):
    if values is None:
        return "null"
    return "[" + ", ".join(_fmt(value) for value in values) + "]"


def format_report(report):
    """Render the report as short human-readable lines."""
    container = report["container"]
    lines = [
        f"GLB: {container['bytes']} bytes, glTF {container['version']}, "
        f"JSON {container['json_bytes']} bytes, BIN {container['bin_bytes']} bytes"
    ]
    summary = report["summary"]
    lines.append(f"skins: {summary['skin_count']}")
    for skin in report["skins"]:
        name = json.dumps(skin["name"]) if skin["name"] is not None else "(unnamed)"
        inverse = skin["inverse_bind_matrices"]
        inverse_text = "missing" if not inverse["present"] else _fmt(inverse["count"])
        lines.append(
            f"  skin[{skin['index']}] {name}: {skin['joint_count']} joint(s), "
            f"inverseBindMatrices {inverse_text}"
        )
    lines.append(f"meshes: {summary['mesh_count']}")
    for mesh in report["meshes"]:
        name = json.dumps(mesh["name"]) if mesh["name"] is not None else "(unnamed)"
        lines.append(f"  mesh[{mesh['index']}] {name}: {mesh['primitive_count']} primitive(s)")
        for primitive in mesh["primitives"]:
            kind = "skinned" if primitive["skinned"] else "static"
            if primitive["skins"]:
                kind += " via skin " + ", ".join(
                    str(index) for index in primitive["skins"]
                )
            lines.append(f"    primitive[{primitive['index']}]: {kind}, "
                         f"{_fmt(primitive['vertex_count'])} vertices, "
                         f"max {primitive['max_influences']} influence(s)")
            weight_sum = primitive["weight_sum"]
            lines.append(
                f"      weight sums: {_fmt(weight_sum['min'])}..{_fmt(weight_sum['max'])}, "
                f"bad {weight_sum['bad_count']}"
            )
            if primitive["joint_sets"]:
                joint_min = min(
                    (joint_set["min"] for joint_set in primitive["joint_sets"]
                     if joint_set["min"] is not None),
                    default=None,
                )
                joint_max = max(
                    (joint_set["max"] for joint_set in primitive["joint_sets"]
                     if joint_set["max"] is not None),
                    default=None,
                )
                out_of_range = sum(
                    joint_set["out_of_range"] or 0 for joint_set in primitive["joint_sets"]
                )
                lines.append(
                    f"      joint indices: {_fmt(joint_min)}..{_fmt(joint_max)}, "
                    f"out of range {out_of_range}"
                )
            lines.append(
                f"      POSITION bbox: {_fmt_list(primitive['position']['bbox_min'])} .. "
                f"{_fmt_list(primitive['position']['bbox_max'])}, "
                f"non-finite {primitive['non_finite']['total']}"
            )
    lines.append(
        "summary: "
        f"{summary['skinned_primitive_count']} skinned, "
        f"{summary['static_primitive_count']} static primitive(s); "
        f"max influences {_fmt(summary['max_influences'])}; "
        f"weight sums {_fmt(summary['weight_sum_min'])}..{_fmt(summary['weight_sum_max'])} "
        f"({summary['bad_weight_sum_count']} bad); "
        f"joints {_fmt(summary['joint_index_min'])}..{_fmt(summary['joint_index_max'])} "
        f"({summary['out_of_range_joint_count']} out of range); "
        f"non-finite {summary['non_finite_value_count']}"
    )
    return "\n".join(lines)


def write_json(path, report):
    """Write LF-only JSON without ever guessing values that JSON cannot hold."""
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with open(path, "wb") as handle:
        handle.write(payload.encode("utf-8"))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Validate the skinning data in a binary glTF 2.0 (.glb) file."
    )
    parser.add_argument("glb", metavar="input.glb", help="input binary glTF file")
    parser.add_argument("--json", metavar="FILE", help="write the report as LF-only JSON")
    args = parser.parse_args(argv)

    if args.json and same_report_path(args.glb, args.json):
        print("ERROR: --json must not overwrite the input GLB", file=sys.stderr)
        return 2
    try:
        with open(args.glb, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        detail = exc.strerror or str(exc)
        print(f"ERROR: cannot read {args.glb}: {detail}", file=sys.stderr)
        return 1
    try:
        report = analyze_glb(data)
    except (ValueError, TypeError, struct.error) as exc:
        # Malformed JSON types and bad offsets must not escape as a traceback.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(format_report(report))
    if args.json:
        try:
            write_json(args.json, report)
        except (OSError, ValueError) as exc:
            print(f"ERROR: cannot write report {args.json}: {exc}", file=sys.stderr)
            return 2
    for message in report["errors"]:
        print(f"ERROR: {message}", file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
