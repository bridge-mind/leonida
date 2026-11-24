"""Inspect a glTF 2.0 binary (.glb): meshes, materials, skins, animations and POSITION bounds.

Usage:
    python -I tools/glb_inspect.py <model.glb> [--json FILE]

Reads the GLB container (12-byte header, JSON chunk, optional BIN chunk and
chunk padding), then reports what the JSON actually contains. Missing fields are
reported as missing; nothing is inferred from the binary buffers.
For base-color textures without a core source, the inspector checks the declared
KHR_texture_basisu and EXT_texture_webp image sources without loading either format.

All geometry, names and paths come from the inspected file. No game data is
needed or bundled.
"""
from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
import struct
import sys
import tempfile

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from file_mode import published_mode  # noqa: E402
from report_paths import same_report_path  # noqa: E402

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
GLB_HEADER_SIZE = 12
CHUNK_HEADER_SIZE = 8
CHUNK_TYPE_JSON = 0x4E4F534A  # b"JSON"
CHUNK_TYPE_BIN = 0x004E4942   # b"BIN\x00"

MODE_NAMES = {
    0: "POINTS",
    1: "LINES",
    2: "LINE_LOOP",
    3: "LINE_STRIP",
    4: "TRIANGLES",
    5: "TRIANGLE_STRIP",
    6: "TRIANGLE_FAN",
}


class GlbError(ValueError):
    """A malformed GLB container or JSON document."""


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _index(value: object) -> int | None:
    """A non-negative integer index, or None."""
    return value if _is_int(value) and value >= 0 else None


def _chunk_type_name(chunk_type: int) -> str:
    if chunk_type == CHUNK_TYPE_JSON:
        return "JSON"
    if chunk_type == CHUNK_TYPE_BIN:
        return "BIN"
    return f"0x{chunk_type:08X}"


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _array(value: object) -> list:
    return value if isinstance(value, list) else []


def _count(gltf: dict, key: str) -> int:
    return len(_array(gltf.get(key)))


ARRAY_FIELDS = ("scenes", "nodes", "meshes", "materials", "images", "skins",
                "animations", "accessors")


def _invalid_types(gltf: dict) -> dict[str, str]:
    """Known array fields that are present but hold a non-array value."""
    return {key: type(gltf[key]).__name__ for key in ARRAY_FIELDS
            if key in gltf and not isinstance(gltf[key], list)}


def _entry(gltf: dict, key: str, index: object) -> dict | None:
    """The object at ``gltf[key][index]``, or None for a missing/invalid index."""
    if not _is_int(index) or index < 0:
        return None
    entries = gltf.get(key)
    if not isinstance(entries, list) or index >= len(entries):
        return None
    entry = entries[index]
    return entry if isinstance(entry, dict) else None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _reject_json_constant(value: str) -> None:
    raise GlbError(f"malformed JSON chunk: {value} is not a JSON number")


def parse_glb(data: bytes) -> dict:
    """Validate the GLB container and return the parsed JSON and chunk table."""
    size = len(data)
    if size < GLB_HEADER_SIZE:
        raise GlbError(f"truncated GLB header: {size} bytes, need at least {GLB_HEADER_SIZE}")

    magic, version, declared_length = struct.unpack_from("<4sII", data, 0)
    if magic != GLB_MAGIC:
        raise GlbError(f"bad GLB magic {magic!r}, expected {GLB_MAGIC!r}")
    if version != GLB_VERSION:
        raise GlbError(f"unsupported GLB version {version}, expected {GLB_VERSION}")
    if declared_length != size:
        detail = "file is truncated" if declared_length > size else "file has trailing bytes"
        raise GlbError(f"declared GLB length {declared_length} does not match file size {size} ({detail})")
    if declared_length % 4:
        raise GlbError(f"declared GLB length {declared_length} is not 4-byte aligned")
    if size < GLB_HEADER_SIZE + CHUNK_HEADER_SIZE:
        raise GlbError("missing JSON chunk")

    offset = GLB_HEADER_SIZE
    chunks: list[dict] = []
    while offset < size:
        if offset + CHUNK_HEADER_SIZE > size:
            raise GlbError(f"truncated chunk header at offset {offset}")
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        remaining = size - offset - CHUNK_HEADER_SIZE
        label = _chunk_type_name(chunk_type)
        if chunk_length > remaining:
            raise GlbError(
                f"truncated {label} chunk at offset {offset}: "
                f"declared {chunk_length} bytes, {remaining} available"
            )
        if chunk_length % 4:
            raise GlbError(f"{label} chunk at offset {offset} has length {chunk_length}, not a multiple of 4")
        chunks.append({"type": chunk_type, "length": chunk_length})
        offset += CHUNK_HEADER_SIZE + chunk_length

    if not chunks or chunks[0]["type"] != CHUNK_TYPE_JSON:
        kind = _chunk_type_name(chunks[0]["type"]) if chunks else "none"
        raise GlbError(f"missing JSON chunk: first chunk is {kind}")

    json_offset = GLB_HEADER_SIZE + CHUNK_HEADER_SIZE
    json_payload = data[json_offset:json_offset + chunks[0]["length"]]
    pad_start = len(json_payload.rstrip(b" \x00"))
    padding = len(json_payload) - pad_start
    nul_padding = json_payload[pad_start:].count(b"\x00")
    # Some writers pad with NUL although the spec requires spaces; accept it for
    # parsing, while still reporting exactly how many bytes were padding.
    try:
        json_text = json_payload[:pad_start].decode("utf-8", errors="strict")
        gltf = json.loads(json_text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise GlbError(f"malformed JSON chunk: {exc.msg} at line {exc.lineno} column {exc.colno}") from exc
    except UnicodeDecodeError as exc:
        raise GlbError(f"malformed JSON chunk: not UTF-8 ({exc.reason} at byte {exc.start})") from exc
    except RecursionError as exc:
        raise GlbError("malformed JSON chunk: nesting is too deep") from exc
    if not isinstance(gltf, dict):
        raise GlbError("malformed JSON chunk: root is not a JSON object")

    bin_padding = None
    seen_bin = False
    for chunk_index, chunk in enumerate(chunks[1:], 1):
        if chunk["type"] == CHUNK_TYPE_JSON:
            raise GlbError(f"duplicate JSON chunk at chunk index {chunk_index}")
        if chunk["type"] != CHUNK_TYPE_BIN:
            continue  # extension chunks are allowed and ignored, per glTF 2.0 section 4.4
        if seen_bin:
            raise GlbError("duplicate BIN chunk")
        if chunk_index != 1:
            raise GlbError("BIN chunk must immediately follow the JSON chunk")
        seen_bin = True
        bin_length = chunk["length"]
        buffers = _array(gltf.get("buffers"))
        buffer0 = _object(buffers[0]) if buffers else {}
        declared = buffer0.get("byteLength")
        if "uri" not in buffer0 and _is_int(declared) and 0 <= declared <= bin_length:
            bin_padding = bin_length - declared

    chunk_table = []
    for chunk_index, chunk in enumerate(chunks):
        entry = {"type": _chunk_type_name(chunk["type"]), "byte_length": chunk["length"]}
        if chunk_index == 0:
            entry["padding"] = padding
            if nul_padding:
                entry["nul_padding"] = nul_padding
        elif chunk["type"] == CHUNK_TYPE_BIN:
            entry["padding"] = bin_padding
        chunk_table.append(entry)
    return {
        "version": version,
        "json": gltf,
        "chunks": chunk_table,
    }


def _accessor(gltf: dict, index: object) -> dict | None:
    return _entry(gltf, "accessors", index)


def _accessor_count(gltf: dict, index: object) -> int | None:
    accessor = _accessor(gltf, index)
    if accessor is None:
        return None
    count = accessor.get("count")
    return count if _is_int(count) and count >= 0 else None


def _accessor_max(accessor: dict | None) -> float | None:
    """An accessor's declared max[0], or None when it is missing or not a number."""
    if accessor is None:
        return None
    upper = accessor.get("max")
    if (isinstance(upper, list) and upper
            and isinstance(upper[0], (int, float)) and not isinstance(upper[0], bool)):
        try:
            return float(upper[0])
        except (OverflowError, ValueError):
            return None  # an integer too large to be a float has no usable bound
    return None


def _accessor_type(gltf: dict, index: object) -> str | None:
    accessor = _accessor(gltf, index)
    return _text(accessor.get("type")) if accessor is not None else None


def _animation_report(gltf: dict, index: int, animation: dict) -> dict:
    """Summarize an animation and flag references that cannot be resolved."""
    issues: list[str] = []
    raw_samplers_value = animation.get("samplers")
    if raw_samplers_value is not None and not isinstance(raw_samplers_value, list):
        issues.append("animation samplers is not an array")
    raw_samplers = _array(raw_samplers_value)
    raw_channels_value = animation.get("channels")
    if raw_channels_value is not None and not isinstance(raw_channels_value, list):
        issues.append("animation channels is not an array")
    raw_channels = _array(raw_channels_value)
    samplers: list[dict] = []
    for sampler_index, sampler in enumerate(raw_samplers):
        if not isinstance(sampler, dict):
            issues.append(f"sampler[{sampler_index}] is not an object")
            samplers.append({"index": sampler_index, "input_accessor": None, "input_count": None,
                             "output_accessor": None, "output_count": None,
                             "output_type": None, "interpolation": None, "duration": None})
            continue
        input_index = _index(sampler.get("input"))
        output_index = _index(sampler.get("output"))
        input_accessor = _accessor(gltf, input_index)
        output_accessor = _accessor(gltf, output_index)
        input_count = _accessor_count(gltf, input_index)
        output_count = _accessor_count(gltf, output_index)
        input_type = _accessor_type(gltf, input_index)
        output_type = _accessor_type(gltf, output_index)
        raw_interpolation = sampler.get("interpolation")
        interpolation = _text(raw_interpolation)
        if raw_interpolation is None:
            interpolation = "LINEAR"
        elif interpolation is None:
            issues.append(f"sampler[{sampler_index}] has unknown interpolation {raw_interpolation!r}")
        duration = _accessor_max(input_accessor) if input_count else None
        if input_accessor is None:
            issues.append(f"sampler[{sampler_index}] input accessor is missing")
        elif input_count is None or input_count == 0:
            issues.append(f"sampler[{sampler_index}] input accessor has no keyframes")
        if input_type is not None and input_type != "SCALAR":
            issues.append(f"sampler[{sampler_index}] input accessor type is {input_type}, expected SCALAR")
        if output_accessor is None:
            issues.append(f"sampler[{sampler_index}] output accessor is missing")
        if input_count is not None and output_count is not None:
            expected = input_count * 3 if interpolation == "CUBICSPLINE" else input_count
            if output_count != expected:
                issues.append(
                    f"sampler[{sampler_index}] has {output_count} output values for {input_count} "
                    f"keys with {interpolation} interpolation"
                )
        if interpolation is not None and interpolation not in ("LINEAR", "STEP", "CUBICSPLINE"):
            issues.append(f"sampler[{sampler_index}] has unknown interpolation {interpolation!r}")
        samplers.append({
            "index": sampler_index,
            "input_accessor": input_index,
            "input_count": input_count,
            "input_type": input_type,
            "output_accessor": output_index,
            "output_count": output_count,
            "output_type": output_type,
            "interpolation": interpolation,
            "duration": duration,
        })

    channels: list[dict] = []
    paths: dict[str, int] = {}
    target_nodes: set[int] = set()
    for channel_index, channel in enumerate(raw_channels):
        if not isinstance(channel, dict):
            issues.append(f"channel[{channel_index}] is not an object")
            channels.append({"index": channel_index, "sampler": None, "node": None,
                             "node_name": None, "path": None})
            continue
        sampler_index = _index(channel.get("sampler"))
        target = _object(channel.get("target"))
        node_index = _index(target.get("node"))
        path = _text(target.get("path"))
        node = _entry(gltf, "nodes", node_index)
        if sampler_index is None or sampler_index >= len(raw_samplers):
            issues.append(f"channel[{channel_index}] has no valid sampler")
        if node_index is None:
            issues.append(f"channel[{channel_index}] has no target node")
        elif node is None:
            issues.append(f"channel[{channel_index}] targets missing node {node_index}")
        else:
            target_nodes.add(node_index)
        if path is None:
            issues.append(f"channel[{channel_index}] has no target path")
        elif path not in ("translation", "rotation", "scale", "weights"):
            issues.append(f"channel[{channel_index}] has unknown target path {path!r}")
        else:
            paths[path] = paths.get(path, 0) + 1
            expected_type = {"translation": "VEC3", "rotation": "VEC4",
                             "scale": "VEC3", "weights": "SCALAR"}[path]
            sampler = samplers[sampler_index] if sampler_index is not None and sampler_index < len(samplers) else None
            output_type = sampler["output_type"] if sampler is not None else None
            if output_type is not None and output_type != expected_type:
                issues.append(
                    f"channel[{channel_index}] path {path!r} expects {expected_type} output, "
                    f"got {output_type}"
                )
        channels.append({
            "index": channel_index,
            "sampler": sampler_index,
            "node": node_index,
            "node_name": _text(node.get("name")) if node is not None else None,
            "path": path,
        })

    known_durations = [sampler["duration"] for sampler in samplers if sampler["duration"] is not None]
    return {
        "index": index,
        "name": _text(animation.get("name")),
        "channel_count": len(raw_channels),
        "sampler_count": len(raw_samplers),
        "duration": max(known_durations) if known_durations else None,
        "target_node_count": len(target_nodes),
        "paths": dict(sorted(paths.items())),
        "interpolations": sorted({sampler["interpolation"] for sampler in samplers
                                  if sampler["interpolation"] is not None}),
        "channels": channels,
        "samplers": samplers,
        "issues": issues,
    }



def _bbox(accessor: dict | None) -> dict | None:
    """An accessor's declared POSITION min/max, or None when either is missing."""
    if accessor is None:
        return None
    lower = accessor.get("min")
    upper = accessor.get("max")
    if not (isinstance(lower, list) and isinstance(upper, list) and len(lower) >= 3 and len(upper) >= 3):
        return None
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in lower[:3] + upper[:3]):
        return None
    return {"min": lower[:3], "max": upper[:3]}


def _merge_bbox(boxes: list[dict]) -> dict | None:
    if not boxes:
        return None
    return {
        "min": [min(box["min"][axis] for box in boxes) for axis in range(3)],
        "max": [max(box["max"][axis] for box in boxes) for axis in range(3)],
    }


def _material_name(gltf: dict, index: object) -> str | None:
    material = _entry(gltf, "materials", index)
    return _text(material.get("name")) if material is not None else None


def _texture_name(gltf: dict, texture_index: object) -> str | None:
    """The baseColor texture's image name, falling back to its URI then the texture name."""
    texture = _entry(gltf, "textures", texture_index)
    if texture is None:
        return None
    image = _entry(gltf, "images", texture.get("source"))
    if "source" not in texture:
        extensions = _object(texture.get("extensions"))
        for extension in ("KHR_texture_basisu", "EXT_texture_webp"):
            source = _object(extensions.get(extension)).get("source")
            image = _entry(gltf, "images", source)
            if image is not None:
                break
    if image is not None:
        name = _text(image.get("name"))
        if name is not None:
            return name
        uri = _text(image.get("uri"))
        if uri is not None:
            return uri
    return _text(texture.get("name"))


def _material_report(gltf: dict, index: int, material: dict) -> dict:
    pbr = _object(material.get("pbrMetallicRoughness"))
    base_color_texture = _object(pbr.get("baseColorTexture"))
    texture_index = _index(base_color_texture.get("index"))
    return {
        "index": index,
        "name": _text(material.get("name")),
        "base_color_factor": pbr.get("baseColorFactor"),
        "base_color_texture": _texture_name(gltf, texture_index),
        "base_color_texture_index": texture_index,
        "alpha_mode": material.get("alphaMode"),
        "double_sided": material.get("doubleSided"),
    }


def _primitive_mode(primitive: dict) -> tuple[int | None, str | None, bool]:
    """The effective mode for counts, its name, and whether the spec default filled it in."""
    if "mode" not in primitive:
        return 4, MODE_NAMES[4], True
    value = primitive["mode"]
    if not _is_int(value):
        return None, None, False
    return value, MODE_NAMES.get(value, f"MODE_{value}"), False


def _triangle_count(mode: int | None, count: int | None) -> int | None:
    if mode is None or count is None:
        return None
    if mode == 4:
        return count // 3
    if mode in (5, 6):
        return max(0, count - 2)
    if 0 <= mode <= 3:
        return 0
    return None


def _primitive_report(gltf: dict, primitive_index: int, primitive: dict) -> dict:
    attributes = _object(primitive.get("attributes"))
    position_index = attributes.get("POSITION")
    position_accessor = _accessor(gltf, position_index)
    vertices = _accessor_count(gltf, position_index)
    mode, mode_name, mode_defaulted = _primitive_mode(primitive)
    indices_index = primitive.get("indices")
    element_count = _accessor_count(gltf, indices_index) if indices_index is not None else vertices
    material_index = _index(primitive.get("material"))
    return {
        "index": primitive_index,
        "mode": primitive.get("mode") if _is_int(primitive.get("mode")) else None,
        "mode_name": mode_name,
        "mode_defaulted": mode_defaulted,
        "vertices": vertices,
        "triangles": _triangle_count(mode, element_count),
        "material_index": material_index,
        "material": _material_name(gltf, material_index),
        "position_accessor": _index(position_index),
        "bbox": _bbox(position_accessor),
    }


def _mesh_report(gltf: dict, index: int, mesh: dict) -> dict:
    primitives = [_primitive_report(gltf, i, primitive)
                  for i, primitive in enumerate(_array(mesh.get("primitives")))
                  if isinstance(primitive, dict)]
    material_names: list[str | None] = []
    for primitive in primitives:
        if primitive["material_index"] is not None and primitive["material"] not in material_names:
            material_names.append(primitive["material"])
    vertices = [primitive["vertices"] for primitive in primitives]
    triangles = [primitive["triangles"] for primitive in primitives]
    return {
        "index": index,
        "name": _text(mesh.get("name")),
        "primitive_count": len(primitives),
        "vertices": sum(vertices) if vertices and all(value is not None for value in vertices) else None,
        "triangles": sum(triangles) if triangles and all(value is not None for value in triangles) else None,
        "materials": material_names,
        "primitives": primitives,
    }


def build_report(glb: dict, file_bytes: int) -> dict:
    """Turn the parsed GLB JSON into the inspection report."""
    gltf = glb["json"]
    asset = _object(gltf.get("asset"))
    meshes = [_mesh_report(gltf, i, mesh)
              for i, mesh in enumerate(_array(gltf.get("meshes")))
              if isinstance(mesh, dict)]
    materials = [_material_report(gltf, i, material)
                  for i, material in enumerate(_array(gltf.get("materials")))
                  if isinstance(material, dict)]
    skins = []
    for i, skin in enumerate(_array(gltf.get("skins"))):
        if not isinstance(skin, dict):
            continue
        joints = skin.get("joints")
        skins.append({
            "index": i,
            "name": _text(skin.get("name")),
            "joint_count": len(joints) if isinstance(joints, list) else None,
        })
    animations = [_animation_report(gltf, i, animation)
                  for i, animation in enumerate(_array(gltf.get("animations")))
                  if isinstance(animation, dict)]

    boxes = [primitive["bbox"] for mesh in meshes for primitive in mesh["primitives"]
             if primitive["bbox"] is not None]
    position_primitives = sum(1 for mesh in meshes for primitive in mesh["primitives"]
                              if primitive["position_accessor"] is not None)
    invalid_counts = {}
    for key in ("meshes", "materials", "skins", "animations"):
        count = sum(1 for entry in _array(gltf.get(key)) if not isinstance(entry, dict))
        if count:
            invalid_counts[key] = count
    invalid_types = _invalid_types(gltf)
    report = {
        "bytes": file_bytes,
        "asset": {
            "version": asset.get("version"),
            "generator": asset.get("generator"),
        },
        "counts": {
            "scenes": _count(gltf, "scenes"),
            "nodes": _count(gltf, "nodes"),
            "meshes": _count(gltf, "meshes"),
            "materials": _count(gltf, "materials"),
            "images": _count(gltf, "images"),
            "skins": _count(gltf, "skins"),
            "animations": _count(gltf, "animations"),
        },
        "chunks": glb["chunks"],
        "position_bbox": _merge_bbox(boxes),
        "position_bbox_complete": bool(boxes) and len(boxes) == position_primitives,
        "meshes": meshes,
        "materials": materials,
        "skins": skins,
        "animations": animations,
    }
    if any(invalid_counts.values()):
        report["invalid_counts"] = invalid_counts
    if invalid_types:
        report["invalid_types"] = invalid_types
    return report


def _show(value: object) -> str:
    if value is None:
        return "(missing)"
    if isinstance(value, list):
        return "[" + ", ".join(_show(item) for item in value) + "]"
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _bbox_text(box: dict | None) -> str:
    if box is None:
        return "(missing)"
    return f"min {_show(box['min'])} max {_show(box['max'])}"


def _padding_text(value: object) -> str:
    return "n/a" if value is None else _show(value)


def format_report(report: dict) -> str:
    """Render the human-readable report."""
    lines = [f"bytes: {report['bytes']:,}"]
    asset = report["asset"]
    lines.append(f"asset: version={_show(asset['version'])} generator={_show(asset['generator'])}")
    counts = report["counts"]
    lines.append("counts: " + " ".join(f"{key}={counts[key]}" for key in
                                       ("scenes", "nodes", "meshes", "materials", "images", "skins", "animations")))
    invalid = report.get("invalid_counts")
    if invalid:
        lines.append("invalid entries: " + " ".join(f"{key}={value}" for key, value in invalid.items() if value))
    invalid_types = report.get("invalid_types")
    if invalid_types:
        lines.append("invalid types: " + " ".join(f"{key}={value}" for key, value in invalid_types.items()))
    lines.append("chunks: " + ", ".join(
        f"{chunk['type']} {chunk['byte_length']} bytes (padding {_padding_text(chunk.get('padding'))})"
        for chunk in report["chunks"]))
    bbox = report["position_bbox"]
    suffix = (" (partial: some POSITION accessors have no min/max)"
              if bbox is not None and not report["position_bbox_complete"] else "")
    lines.append(f"POSITION bbox: {_bbox_text(bbox)}{suffix}")

    lines.append("meshes:")
    if not report["meshes"]:
        lines.append("  (none)")
    for mesh in report["meshes"]:
        names = "[" + ", ".join(_show(name) for name in mesh["materials"]) + "]" if mesh["materials"] else "none"
        lines.append(f"  [{mesh['index']}] name={_show(mesh['name'])} primitives={mesh['primitive_count']} "
                     f"vertices={_show(mesh['vertices'])} triangles={_show(mesh['triangles'])} "
                     f"materials={names}")
        for primitive in mesh["primitives"]:
            material = ("none" if primitive["material_index"] is None else
                        f"{_show(primitive['material'])} (index {primitive['material_index']})")
            mode = _show(primitive["mode_name"])
            if primitive["mode_defaulted"]:
                mode += " (default)"
            lines.append(
                f"    primitive {primitive['index']}: mode={mode} "
                f"vertices={_show(primitive['vertices'])} triangles={_show(primitive['triangles'])} "
                f"material={material} bbox {_bbox_text(primitive['bbox'])}")

    lines.append("materials:")
    if not report["materials"]:
        lines.append("  (none)")
    for material in report["materials"]:
        lines.append(
            f"  [{material['index']}] name={_show(material['name'])} "
            f"baseColorFactor={_show(material['base_color_factor'])} "
            f"baseColorTexture={_show(material['base_color_texture'])} "
            f"(index {_show(material['base_color_texture_index'])}) "
            f"alphaMode={_show(material['alpha_mode'])} doubleSided={_show(material['double_sided'])}")

    lines.append("skins:")
    if not report["skins"]:
        lines.append("  (none)")
    for skin in report["skins"]:
        lines.append(f"  [{skin['index']}] name={_show(skin['name'])} joints={_show(skin['joint_count'])}")

    lines.append("animations:")
    if not report["animations"]:
        lines.append("  (none)")
    for animation in report["animations"]:
        paths = ", ".join(f"{name}={count}" for name, count in animation["paths"].items()) or "none"
        interpolation = ", ".join(animation["interpolations"]) or "none"
        lines.append(
            f"  [{animation['index']}] name={_show(animation['name'])} "
            f"channels={animation['channel_count']} samplers={animation['sampler_count']} "
            f"duration={_show(animation['duration'])} targets={animation['target_node_count']}")
        lines.append(f"    paths: {paths}; interpolation: {interpolation}")
        for issue in animation["issues"]:
            lines.append(f"    issue: {issue}")
    return "\n".join(lines)


def inspect_file(path: str) -> dict:
    with open(path, "rb") as handle:
        data = handle.read()
    return build_report(parse_glb(data), len(data))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="binary glTF 2.0 (.glb) file")
    parser.add_argument("--json", dest="json_path", metavar="FILE", help="also write the report as JSON to FILE")
    args = parser.parse_args(argv)

    if args.json_path:
        if os.path.isdir(args.json_path):
            print(f"error: --json {args.json_path} is a directory", file=sys.stderr)
            return 2
        if same_report_path(args.json_path, args.input):
            print(f"error: --json {args.json_path} must not overwrite the input {args.input}", file=sys.stderr)
            return 2

    try:
        report = inspect_file(args.input)
    except GlbError as exc:
        print(f"error: {args.input}: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: cannot read {args.input}: {exc.strerror or exc}", file=sys.stderr)
        return 2

    if args.json_path:
        try:
            text = json.dumps(report, indent=2, allow_nan=False) + "\n"
        except (TypeError, ValueError) as exc:
            print(f"error: cannot write {args.json_path}: {exc}", file=sys.stderr)
            return 2
        staged = None
        try:
            directory = os.path.dirname(os.path.abspath(args.json_path))
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n", dir=directory,
                                             prefix=".glb_inspect-", delete=False) as handle:
                staged = handle.name
                handle.write(text)
            os.chmod(staged, published_mode(args.json_path))
            os.replace(staged, args.json_path)
            staged = None
        except OSError as exc:
            print(f"error: cannot write {args.json_path}: {getattr(exc, 'strerror', None) or exc}", file=sys.stderr)
            return 2
        finally:
            if staged is not None:
                with suppress(OSError):
                    os.chmod(staged, 0o600)
                    os.unlink(staged)
    print(format_report(report))
    if args.json_path:
        print(f"wrote: {args.json_path}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
