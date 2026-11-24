"""Census decoded character models: counts, materials, textures, bones and skin layout.

Usage::

    python -I tools/character_census.py <models-dir-or-manifest> [--json FILE] [--csv FILE]

The input is either a directory tree of decoded pack models (every nested ``model.json`` is read) or a
manifest. The decoded ``model.json`` layout written by the local pack exporter is::

    type                      "dr" drawable, "ft" fragment, ...
    vertex_count              total over all meshes
    triangle_count            total over all meshes
    materials[]               shader_name_hash, textures[] (hash64/hash32)
    drawable.skeleton[]       fragment skeleton
    drawable.models[]         bone_palette[] (local bone count)
    meshes[]                  vertex_count, triangle_count, skinned, skin_influences, bones

The local exporter does not write an index count, so the index column and total normally stay
unknown. An input that supplies ``index_count`` (or ``indices``/``IndexCount``) is counted; the
field is optional, not part of the decoded layout above.
When a model total is absent, every mesh must supply its contribution for that total to be known.
A mesh without a triangle count can contribute its own index count divided by three.

An ``rpf9dump models`` manifest (``Items[]`` with ``Verts``, ``Tris``, ``IsFragment``,
``HasBoneBindings`` and ``Materials[]``) and a simple ``models[]`` list of model paths or inline
model objects are also accepted. An ``rpf9dump models`` manifest has only ``HasBoneBindings``, not
a bone count or influence width, so those cells are reported as unknown rather than guessed.

Every model is reported independently. A missing field, a wrong type or an unreadable JSON file
adds an issue to that row and never stops the run. Only model names, hashes, counts and issue text
are emitted; no vertex, texture or package bytes are copied.

The report paths are refused when they alias an input (the manifest, a nested ``model.json`` or a
model referenced by the manifest), so the census cannot overwrite its own source.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:  # -I does not put the script's directory on sys.path
    sys.path.insert(0, HERE)
from report_paths import same_report_path  # noqa: E402

MODEL_JSON = "model.json"
SHADER_DIGITS = 8
TEXTURE_DIGITS = 16

KINDS = {
    "dr": "drawable",
    "ydr": "drawable",
    "drawable": "drawable",
    "d": "drawable",
    "ft": "fragment",
    "yft": "fragment",
    "fragment": "fragment",
    "f": "fragment",
    "dd": "dictionary",
    "ydd": "dictionary",
    "dictionary": "dictionary",
    "component": "component",
    "cmp": "component",
}

VERTEX_KEYS = ("vertex_count", "vertices", "Verts", "Vertices", "VertexCount")
TRIANGLE_KEYS = ("triangle_count", "triangles", "Tris", "Triangles", "TriangleCount")
INDEX_KEYS = ("index_count", "indices", "Indices", "IndexCount")
BONE_KEYS = ("bone_count", "BoneCount", "Bones", "bones")
SHADER_KEYS = ("shader_name_hash", "shader_hash", "Shader", "name_hash", "hash")
TEXTURE_KEYS = ("hash64", "Id64", "Key", "asset_key", "hash", "hash32", "Name")

CSV_FIELDS = ["source", "name", "kind", "vertices", "triangles", "indices", "bones", "skin_8",
              "shaders", "textures", "issues"]


# --------------------------------------------------------------------- hashes
def _hash(value, digits: int):
    """Canonical ``0x`` + uppercase hex hash, or None when the value is not hex."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value < 0 or value >= 1 << (4 * digits):
            return None
        return f"0x{value:0{digits}X}"
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text[:2].lower() == "0x":
        text = text[2:]
    if not text or len(text) > digits:
        return None
    if any(ch not in "0123456789abcdefABCDEF" for ch in text):
        return None
    try:
        number = int(text, 16)
    except ValueError:
        return None
    return f"0x{number:0{digits}X}"


def shader_hash(value):
    """An 8-digit material shader hash; zero and malformed values are not hashes."""
    result = _hash(value, SHADER_DIGITS)
    return None if result in (None, "0x00000000") else result


def texture_hash(value):
    """A texture key: 16 digits for 64-bit values, 8 for 32-bit values."""
    if isinstance(value, str):
        text = value.strip()
        digits = TEXTURE_DIGITS if len(text.removeprefix("0x").removeprefix("0X")) > 8 else 8
    elif isinstance(value, int) and not isinstance(value, bool):
        digits = TEXTURE_DIGITS if value > 0xFFFFFFFF else 8
    else:
        digits = 8
    result = _hash(value, digits)
    return None if result in (None, "0x" + "0" * digits) else result


def count(value):
    """A non-negative integer, or None for bools, floats, negatives and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


# -------------------------------------------------------------------- reading
def _walk_error(exc: OSError):
    # os.walk skips a directory it cannot list unless onerror raises; a census must not report an
    # incomplete scan as complete. main() turns the OSError into one ERROR line and exit 2.
    raise exc


def model_json_files(directory: str):
    """Every nested model.json, in a stable order. An unlistable sub-folder raises OSError."""
    for base, dirs, files in os.walk(directory, onerror=_walk_error):
        dirs.sort()
        for name in sorted(files):
            if name.lower() == MODEL_JSON:
                yield os.path.join(base, name)


def _read_json(path: str):
    # utf-8-sig: a model.json or manifest saved by a Windows editor starts with a BOM, which json.load refuses.
    with open(path, encoding="utf-8-sig") as handle:
        return json.load(handle)


def _manifest_entries(data):
    """Return (entries, label) for a manifest object/list, or (None, None) for a model."""
    if isinstance(data, list):
        return data, "models"
    if not isinstance(data, dict):
        return None, None
    for key in ("Items", "items"):
        if isinstance(data.get(key), list):
            return data[key], key
    if isinstance(data.get("models"), list):
        return data["models"], "models"
    return None, None


def _looks_like_model(data) -> bool:
    if not isinstance(data, dict):
        return False
    return any(key in data for key in ("drawable", "meshes", "vertex_count", "triangle_count",
                                       "materials", "buffer_files", "glb"))


def _entry_path(entry):
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in ("model_json", "model_path", "path", "json"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def _resolve_path(path: str, base: str) -> str:
    path = path if os.path.isabs(path) else os.path.join(base, path)
    if os.path.isdir(path):
        path = os.path.join(path, MODEL_JSON)
    return os.path.abspath(path)


def input_files(target: str):
    """Every file the census may read, so a report path cannot overwrite one."""
    if os.path.isdir(target):
        yield from model_json_files(target)
        return
    if not os.path.isfile(target):
        return
    yield target
    try:
        data = _read_json(target)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return
    entries, _ = _manifest_entries(data)
    if entries is None:
        return
    base = os.path.dirname(os.path.abspath(target))
    for entry in entries:
        path = _entry_path(entry)
        if path:
            try:
                yield _resolve_path(path, base)
            except (OSError, ValueError):
                continue


def _fallback_name(source: str) -> str:
    source = os.path.normpath(source)
    if os.path.basename(source).lower() == MODEL_JSON:
        return os.path.basename(os.path.dirname(source)) or MODEL_JSON
    return os.path.splitext(os.path.basename(source))[0] or "model"


def _blank_row(source: str, name: str) -> dict:
    return {
        "source": source,
        "name": name,
        "kind": "unknown",
        "vertices": None,
        "triangles": None,
        "indices": None,
        "bones": None,
        "skin_8": None,
        "shaders": [],
        "textures": [],
        "issues": [],
    }


def _text_field(obj, keys, label: str, issues: list) -> str | None:
    for key in keys:
        if key not in obj:
            continue
        value = obj[key]
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None:
            issues.append(f"{label} is not a non-empty string")
    return None


def _count_field(obj, keys, label: str, issues: list, *, list_len: bool = False) -> int | None:
    for key in keys:
        if key not in obj:
            continue
        value = obj[key]
        if value is None:
            return None
        if list_len and isinstance(value, list):
            return len(value)
        result = count(value)
        if result is None:
            issues.append(f"{label} is not a non-negative integer")
        return result
    return None


def _mesh_count(meshes: list, keys, label: str, issues: list, *, list_len: bool = False):
    total = 0
    complete = True
    for index, mesh in enumerate(meshes):
        if not isinstance(mesh, dict):
            issues.append(f"mesh {index} is not an object")
            complete = False
            continue
        value = _count_field(mesh, keys, f"mesh {index} {label}", issues, list_len=list_len)
        if value is None and keys == TRIANGLE_KEYS:
            indices = _count_field(mesh, INDEX_KEYS, f"mesh {index} index count", issues, list_len=True)
            if indices is not None:
                value = indices // 3
        if value is None:
            complete = False
        else:
            total += value
    return total if complete else None


def _kind(model: dict, row: dict, issues: list) -> str:
    if "type" in model:
        raw = model.get("type")
        if not isinstance(raw, str) or not raw.strip():
            issues.append("type is not a non-empty string")
        else:
            text = raw.strip().lower()
            return KINDS.get(text, text)
    elif "kind" in model:
        raw = model.get("kind")
        if not isinstance(raw, str) or not raw.strip():
            issues.append("kind is not a non-empty string")
        else:
            text = raw.strip().lower()
            return KINDS.get(text, text)
    if row.get("IsFragment") is True:
        return "fragment"
    if row.get("IsFragment") is False:
        return "drawable"
    if "drawable" in model:
        return "drawable"
    if "meshes" in model:
        return "component"
    return "unknown"


def _collect_shaders(materials, drawable, issues: list) -> list:
    found = set()
    material_rows = materials if isinstance(materials, list) else []
    for index, material in enumerate(material_rows):
        if not isinstance(material, dict):
            issues.append(f"material {index} is not an object")
            continue
        for key in SHADER_KEYS:
            if key not in material or material[key] is None:
                continue
            value = shader_hash(material[key])
            if value is None:
                issues.append(f"material {index} {key} is not an 8-digit shader hash")
            else:
                found.add(value)
    shader_rows = drawable.get("shaders") if isinstance(drawable, dict) else None
    for index, shader in enumerate(shader_rows if isinstance(shader_rows, list) else []):
        if not isinstance(shader, dict):
            issues.append(f"drawable shader {index} is not an object")
            continue
        for key in SHADER_KEYS:
            if key not in shader or shader[key] is None:
                continue
            value = shader_hash(shader[key])
            if value is None:
                issues.append(f"drawable shader {index} {key} is not an 8-digit shader hash")
            else:
                found.add(value)
    if material_rows and not found:
        issues.append("no shader hash on any material")
    return sorted(found)


def _texture_rows(textures, where: str, issues: list):
    if isinstance(textures, list):
        return textures
    if isinstance(textures, dict):
        return list(textures.values())
    if textures is not None:
        issues.append(f"{where} is not a list or object")
    return []


def _collect_textures(materials, drawable, issues: list) -> list:
    found = set()

    def add(value, where: str):
        if value is None:
            return
        key = texture_hash(value)
        if key is None:
            issues.append(f"{where} is not a hexadecimal texture key")
        else:
            found.add(key)

    def add_rows(rows, where: str):
        for index, texture in enumerate(rows):
            if isinstance(texture, dict):
                for key in TEXTURE_KEYS:
                    if key in texture and texture[key] is not None:
                        add(texture[key], f"{where} {index} {key}")
                        break
            else:
                add(texture, f"{where} {index}")

    material_rows = materials if isinstance(materials, list) else []
    for index, material in enumerate(material_rows):
        if isinstance(material, dict):
            for key in ("textures", "Textures"):
                if key in material:
                    add_rows(_texture_rows(material[key], f"material {index} textures", issues),
                             f"material {index} texture")
                    break

    shader_rows = drawable.get("shaders") if isinstance(drawable, dict) else None
    for index, shader in enumerate(shader_rows if isinstance(shader_rows, list) else []):
        if isinstance(shader, dict) and "textures" in shader:
            add_rows(_texture_rows(shader["textures"], f"drawable shader {index} textures", issues),
                     f"drawable shader {index} texture")
    return sorted(found)


def _bones(model, drawable, meshes, row, issues: list):
    for source, where in ((row, "manifest"), (model, "model")):
        if not isinstance(source, dict):
            continue
        for key in BONE_KEYS:
            if key not in source:
                continue
            value = source[key]
            if isinstance(value, list):
                if value:
                    return len(value)
                continue
            result = count(value)
            if result is not None:
                return result
            if value is not None:
                issues.append(f"{where} {key} is not a non-negative integer or a list")
    if isinstance(drawable, dict):
        skeleton = drawable.get("skeleton")
        if isinstance(skeleton, list) and skeleton:
            return len(skeleton)
        models = drawable.get("models")
        palettes = [len(m["bone_palette"]) for m in models if isinstance(m, dict)
                    and isinstance(m.get("bone_palette"), list)] if isinstance(models, list) else []
        if any(palettes):
            return max(palettes)
    mesh_bones = []
    for mesh in meshes:
        if not isinstance(mesh, dict):
            continue
        value = mesh.get("bones")
        if isinstance(value, list) and value:
            mesh_bones.append(len(value))
        elif count(value) is not None:
            mesh_bones.append(count(value))
    if mesh_bones:
        return max(mesh_bones)
    if row.get("HasBoneBindings") is False:
        return 0
    if any(isinstance(mesh, dict) and mesh.get("skinned") is True for mesh in meshes):
        issues.append("no bone count for a skinned model")
        return None
    if row.get("HasBoneBindings") is True:
        return None
    return 0


def _eight_influence_sections(drawable) -> bool:
    """True when a vertex buffer's measured section table has all four skin slots.

    The decoded layout writes ``mesh_blob.sections`` as ``[semantic, offset, format]``; 6/7 are
    the WEIGHTS_0/1 streams and 8/9 the JOINTS_0/1 streams. This is a fallback for a model.json
    that omits ``meshes[].skin_influences``.
    """
    if not isinstance(drawable, dict):
        return False
    buffers = drawable.get("buffers")
    if not isinstance(buffers, list):
        return False
    for buffer in buffers:
        if not isinstance(buffer, dict):
            continue
        blob = buffer.get("mesh_blob")
        if not isinstance(blob, dict) or not isinstance(blob.get("sections"), list):
            continue
        semantics = set()
        for section in blob["sections"]:
            if isinstance(section, (list, tuple)) and section:
                value = count(section[0])
                if value is not None:
                    semantics.add(value)
        if {6, 7, 8, 9} <= semantics:
            return True
    return False


def _skin_8(meshes, row, issues: list, drawable=None):
    for key in ("skin_8", "Skin8", "skin_influences_8", "SkinInfluences8"):
        if key in row:
            value = row[key]
            if isinstance(value, bool):
                return value
            if value is not None:
                issues.append(f"{key} is not a boolean")
    for key in ("skin_influences", "SkinInfluences", "Influences"):
        if key not in row or row[key] is None:
            continue
        influences = count(row[key])
        if influences is None:
            issues.append(f"{key} is not a non-negative integer")
        else:
            return influences >= 8
    states = []
    for index, mesh in enumerate(meshes):
        if not isinstance(mesh, dict):
            continue
        attributes = mesh.get("attributes")
        if isinstance(attributes, dict):
            names = {str(key) for key in attributes}
        elif isinstance(attributes, list):
            names = {str(value) for value in attributes}
        else:
            names = set()
        if {"JOINTS_0", "JOINTS_1", "WEIGHTS_0", "WEIGHTS_1"} <= names:
            states.append(True)
            continue
        value = mesh.get("skin_influences")
        if value is not None:
            influences = count(value)
            if influences is None:
                issues.append(f"mesh {index} skin_influences is not a non-negative integer")
                states.append(None)
            else:
                states.append(influences >= 8)
        elif mesh.get("skinned") is True:
            states.append(None)
    if True in states:
        return True
    if _eight_influence_sections(drawable):
        return True
    if None in states:
        return None
    if states:
        return False
    if row.get("HasBoneBindings") is True:
        return None
    if row.get("HasBoneBindings") is False:
        return False
    return False


def model_row(model: dict, source: str, issues: list | None = None) -> dict:
    """Build one census row from a decoded model.json object."""
    issues = [] if issues is None else issues
    row = _blank_row(source, "")
    row["issues"] = issues

    drawable = model.get("drawable")
    if drawable is not None and not isinstance(drawable, dict):
        issues.append("drawable is not an object")
        drawable = None
    meshes = model.get("meshes")
    if meshes is not None and not isinstance(meshes, list):
        issues.append("meshes is not a list")
        meshes = []
    meshes = meshes or []
    materials = model.get("materials")
    if materials is not None and not isinstance(materials, list):
        issues.append("materials is not a list")
        materials = []

    name = _text_field(model, ("name", "Name"), "name", issues)
    if name is None and isinstance(drawable, dict):
        name = _text_field(drawable, ("name", "Name"), "drawable name", issues)
    if name is None:
        name = _fallback_name(source)
        issues.append("missing name")
    row["name"] = name
    row["kind"] = _kind(model, row, issues)

    vertices = _count_field(model, VERTEX_KEYS, "vertex count", issues)
    if vertices is None and meshes:
        vertices = _mesh_count(meshes, VERTEX_KEYS, "vertex count", issues)
    triangles = _count_field(model, TRIANGLE_KEYS, "triangle count", issues)
    if triangles is None and meshes:
        triangles = _mesh_count(meshes, TRIANGLE_KEYS, "triangle count", issues)
    indices = _count_field(model, INDEX_KEYS, "index count", issues, list_len=True)
    if indices is None and meshes:
        indices = _mesh_count(meshes, INDEX_KEYS, "index count", issues, list_len=True)
    if vertices is None:
        issues.append("missing vertex count")
    if triangles is None:
        issues.append("missing triangle count")
    row["vertices"] = vertices
    row["triangles"] = triangles
    row["indices"] = indices

    row["shaders"] = _collect_shaders(materials, drawable, issues)
    row["textures"] = _collect_textures(materials, drawable, issues)
    row["bones"] = _bones(model, drawable, meshes, row, issues)
    row["skin_8"] = _skin_8(meshes, row, issues, drawable)
    return row


def manifest_row(entry: dict, source: str) -> dict:
    """Build one census row from an rpf9dump models manifest item."""
    issues = []
    row = _blank_row(source, "")
    name = _text_field(entry, ("Name", "name"), "name", issues)
    if name is None:
        file_name = entry.get("File")
        name = (os.path.splitext(os.path.basename(file_name))[0]
                if isinstance(file_name, str) and file_name else source)
        issues.append("missing name")
    row["name"] = name
    row["kind"] = _kind(entry, entry, issues)
    if "IsFragment" in entry and not isinstance(entry["IsFragment"], bool):
        issues.append("IsFragment is not a boolean")
    row["vertices"] = _count_field(entry, VERTEX_KEYS, "vertex count", issues)
    row["triangles"] = _count_field(entry, TRIANGLE_KEYS, "triangle count", issues)
    row["indices"] = _count_field(entry, INDEX_KEYS, "index count", issues, list_len=True)
    if row["vertices"] is None:
        issues.append("missing vertex count")
    if row["triangles"] is None:
        issues.append("missing triangle count")

    materials = entry.get("Materials")
    if materials is not None and not isinstance(materials, list):
        issues.append("Materials is not a list")
        materials = []
    row["shaders"] = _collect_shaders(materials, None, issues)
    row["textures"] = _collect_textures(materials, None, issues)
    row["bones"] = _bones(entry, None, [], entry, issues)
    row["skin_8"] = _skin_8([], entry, issues)
    if "HasBoneBindings" not in entry:
        explicit_bones = any(key in entry for key in BONE_KEYS)
        explicit_skin = any(key in entry for key in
                            ("skin_8", "Skin8", "skin_influences", "SkinInfluences", "Influences"))
        if not explicit_bones:
            row["bones"] = None
        if not explicit_skin:
            row["skin_8"] = None
        issues.append("HasBoneBindings is missing")
    elif not isinstance(entry["HasBoneBindings"], bool):
        issues.append("HasBoneBindings is not a boolean")
        row["bones"] = None
        row["skin_8"] = None
    row["issues"] = issues
    return row


def _read_model(path: str, source: str) -> dict:
    try:
        data = _read_json(path)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        row = _blank_row(source, _fallback_name(path))
        detail = getattr(exc, "strerror", None) or str(exc)
        row["issues"].append(f"cannot read model.json: {detail}")
        return row
    if not isinstance(data, dict):
        row = _blank_row(source, _fallback_name(path))
        row["issues"].append("model.json root is not an object")
        return row
    return model_row(data, source)


def load_rows(target: str):
    """Yield one row per model. Per-model failures become rows with an issue."""
    if os.path.isdir(target):
        root = os.path.abspath(target)
        for path in model_json_files(target):
            source = os.path.relpath(path, root)
            yield _read_model(path, source)
        return

    source = os.path.basename(target)
    try:
        data = _read_json(target)
    except (OSError, UnicodeDecodeError, ValueError, RecursionError) as exc:
        row = _blank_row(source, _fallback_name(target))
        detail = getattr(exc, "strerror", None) or str(exc)
        row["issues"].append(f"cannot read manifest: {detail}")
        yield row
        return

    if _looks_like_model(data):
        yield model_row(data, source)
        return

    entries, label = _manifest_entries(data)
    if entries is None:
        row = _blank_row(source, _fallback_name(target))
        row["issues"].append("manifest has no Items[] or models[] list")
        yield row
        return

    base = os.path.dirname(os.path.abspath(target))
    for index, entry in enumerate(entries):
        where = f"{source}#{label}[{index}]"
        path = _entry_path(entry)
        if path:
            try:
                resolved = _resolve_path(path, base)
            except (OSError, ValueError) as exc:
                row = _blank_row(where, _fallback_name(where))
                row["issues"].append(f"cannot resolve model path: {exc}")
                yield row
                continue
            try:
                relative = os.path.relpath(resolved, base)
            except ValueError:  # another drive on Windows
                relative = resolved
            yield _read_model(resolved, f"{source}#{relative}")
        elif isinstance(entry, dict) and _looks_like_model(entry):
            yield model_row(entry, where)
        elif isinstance(entry, dict):
            source_name = entry.get("File") or entry.get("Name") or where
            yield manifest_row(entry, str(source_name))
        else:
            row = _blank_row(where, _fallback_name(where))
            row["issues"].append("manifest entry is neither a model object nor a path")
            yield row


# ----------------------------------------------------------------- aggregates
def _known_total(rows, key):
    values = [row[key] for row in rows if row[key] is not None]
    return sum(values), len(rows) - len(values)


def summarize(rows: list) -> dict:
    vertices, unknown_vertices = _known_total(rows, "vertices")
    triangles, unknown_triangles = _known_total(rows, "triangles")
    indices, unknown_indices = _known_total(rows, "indices")
    summary = {
        "models": len(rows),
        "vertices": vertices,
        "triangles": triangles,
        "indices": indices,
        "unknown_vertices": unknown_vertices,
        "unknown_triangles": unknown_triangles,
        "unknown_indices": unknown_indices,
        "skin_8": sum(1 for row in rows if row["skin_8"] is True),
        "skin_unknown": sum(1 for row in rows if row["skin_8"] is None),
        "malformed": sum(1 for row in rows if row["issues"]),
        "shader_hashes": len({value for row in rows for value in row["shaders"]}),
        "texture_keys": len({value for row in rows for value in row["textures"]}),
        "by_kind": {},
        "by_shader": {},
    }
    kinds, shaders = summary["by_kind"], summary["by_shader"]
    for row in rows:
        group = kinds.setdefault(row["kind"], {
            "models": 0, "vertices": 0, "triangles": 0, "indices": 0,
            "skin_8": 0, "skin_unknown": 0, "malformed": 0,
            "unknown_vertices": 0, "unknown_triangles": 0, "unknown_indices": 0,
        })
        group["models"] += 1
        for key in ("vertices", "triangles", "indices"):
            if row[key] is None:
                group[f"unknown_{key}"] += 1
            else:
                group[key] += row[key]
        if row["skin_8"] is True:
            group["skin_8"] += 1
        elif row["skin_8"] is None:
            group["skin_unknown"] += 1
        if row["issues"]:
            group["malformed"] += 1
        for value in row["shaders"]:
            shader = shaders.setdefault(value, {"models": 0, "vertices": 0, "triangles": 0})
            shader["models"] += 1
            if row["vertices"] is not None:
                shader["vertices"] += row["vertices"]
            if row["triangles"] is not None:
                shader["triangles"] += row["triangles"]
    summary["by_kind"] = dict(sorted(kinds.items()))
    summary["by_shader"] = dict(sorted(shaders.items()))
    return summary


# --------------------------------------------------------------------- output
def _fmt(value, *, missing: str = "?") -> str:
    if value is None:
        return missing
    return f"{value:,}" if isinstance(value, int) else str(value)


def _fmt_flag(value) -> str:
    if value is None:
        return "?"
    return "yes" if value else "no"


def _table(headers, rows, right=()) -> list:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = []
    for cells in [headers] + rows:
        parts = []
        for index, cell in enumerate(cells):
            parts.append(cell.rjust(widths[index]) if index in right else cell.ljust(widths[index]))
        lines.append("  ".join(parts).rstrip())
    return lines


def _plural(number: int, singular: str, plural: str | None = None) -> str:
    word = singular if number == 1 else (plural or singular + "s")
    return f"{number:,} {word}"


def _total_text(total: int, unknown: int, models: int, noun: str) -> str:
    if models and unknown == models:
        return f"? {noun}"
    text = f"{_fmt(total)} {noun}"
    if unknown:
        text += f" ({unknown} without)"
    return text


def print_report(summary: dict, rows: list) -> None:
    verts = _total_text(summary["vertices"], summary["unknown_vertices"],
                        summary["models"], "vertices")
    tris = _total_text(summary["triangles"], summary["unknown_triangles"],
                       summary["models"], "triangles")
    indices = _total_text(summary["indices"], summary["unknown_indices"],
                          summary["models"], "indices")
    print(f'character census: {_plural(summary["models"], "model")}, {verts}, {tris}, {indices}')
    print(f'  8-influence skin: {summary["skin_8"]} yes, '
          f'{summary["models"] - summary["skin_8"] - summary["skin_unknown"]} no, '
          f'{summary["skin_unknown"]} unknown; '
          f'{_plural(summary["shader_hashes"], "shader hash", "shader hashes")}, '
          f'{_plural(summary["texture_keys"], "texture key")}')
    print(f'  {_plural(summary["malformed"], "model")} with issues')

    print("by kind:")
    kind_rows = []
    for kind, group in summary["by_kind"].items():
        kind_rows.append([kind, _fmt(group["models"]), _fmt(group["vertices"]),
                          _fmt(group["triangles"]), _fmt(group["skin_8"]), _fmt(group["malformed"])])
    for line in _table(["KIND", "MODELS", "VERTICES", "TRIANGLES", "8-INFLUENCE", "ISSUES"],
                       kind_rows, right={1, 2, 3, 4, 5}):
        print("  " + line)

    print("by shader (a model with several materials counts under each):")
    shader_rows = []
    for value, group in summary["by_shader"].items():
        shader_rows.append([value, _fmt(group["models"]), _fmt(group["vertices"]),
                            _fmt(group["triangles"])])
    for line in _table(["SHADER", "MODELS", "VERTICES", "TRIANGLES"], shader_rows, right={1, 2, 3}):
        print("  " + line)

    print("models:")
    model_rows = []
    for row in rows:
        model_rows.append([
            row["name"], row["kind"], _fmt(row["vertices"]), _fmt(row["triangles"]),
            _fmt(row["indices"], missing="-"), _fmt(row["bones"]), _fmt_flag(row["skin_8"]),
            ",".join(row["shaders"]) or "-", ",".join(row["textures"]) or "-",
        ])
    for line in _table(["NAME", "KIND", "VERTS", "TRIS", "INDICES", "BONES", "8-INFL",
                        "SHADERS", "TEXTURES"], model_rows, right={2, 3, 4, 5}):
        print("  " + line)

    problems = [(row["source"], issue) for row in rows for issue in row["issues"]]
    if problems:
        print("problems:")
        for source, issue in problems:
            print(f"  {source}: {issue}")


def _make_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def write_csv(path: str, rows: list) -> None:
    _make_parent(path)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            out = {key: row[key] for key in CSV_FIELDS}
            out["shaders"] = ";".join(row["shaders"])
            out["textures"] = ";".join(row["textures"])
            out["issues"] = ";".join(row["issues"])
            out["skin_8"] = "" if row["skin_8"] is None else str(row["skin_8"]).lower()
            for key in ("vertices", "triangles", "indices", "bones"):
                out[key] = "" if row[key] is None else row[key]
            writer.writerow(out)


def write_json(path: str, summary: dict, rows: list) -> None:
    _make_parent(path)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump({"summary": summary, "models": rows}, handle, indent=1, ensure_ascii=False)
        handle.write("\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="directory of decoded model.json files or a manifest path")
    parser.add_argument("--json", metavar="FILE", help="write the summary and per-model rows as JSON")
    parser.add_argument("--csv", metavar="FILE", help="write one row per model as CSV")
    args = parser.parse_args(argv)

    if args.csv and args.json and same_report_path(args.csv, args.json):
        parser.error("--csv and --json must use different paths")
    try:
        sources = list(input_files(args.target))
    except (OSError, ValueError) as exc:
        print(f"ERROR: cannot read {getattr(exc, 'filename', None) or args.target}: "
              f"{getattr(exc, 'strerror', None) or exc}",
              file=sys.stderr)
        return 2
    for flag, path in (("--json", args.json), ("--csv", args.csv)):
        if not path:
            continue
        if os.path.isdir(path):
            parser.error(f"{flag} is a directory: {path}")
        for source in sources:
            if same_report_path(path, source):
                parser.error(f"{flag} would overwrite the input {source}")

    if not os.path.exists(args.target):
        print(f"not found: {args.target}", file=sys.stderr)
        return 2
    try:
        rows = list(load_rows(args.target))
    except OSError as exc:
        print(f"ERROR: cannot read {getattr(exc, 'filename', None) or args.target}: "
              f"{getattr(exc, 'strerror', None) or exc}",
              file=sys.stderr)
        return 2
    rows.sort(key=lambda row: (row["name"].lower(), row["source"].lower()))

    summary = summarize(rows)
    reports = []
    for path, writer, values in ((args.csv, write_csv, (rows,)),
                                 (args.json, write_json, (summary, rows))):
        if path:
            try:
                writer(path, *values)
            except OSError as exc:
                print(f"ERROR: cannot write {path}: {exc.strerror or exc}", file=sys.stderr)
                return 2
            reports.append(path)
    print_report(summary, rows)
    for path in reports:
        print("wrote", path)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
