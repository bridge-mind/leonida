"""Export a skinned GLB from a decoded RAGE pack model and its 8-influence skin streams.

Usage::

    python -I tools/skinned_glb.py out.glb \\
        --skeleton ambient_femalebody/model.json \\
        --component head/model.json \\
        --component jacket/model.json

Inputs are read-only decoded pack directories: ``model.json``, its ``model.glb``
geometry, and the graphics buffer named by ``buffer_files``. Nothing is written
except the output GLB; no source bytes are copied into the repository.

Source layout (measured on the 1.0.2.1 package)
------------------------------------------------

The vertex skin streams are 4-byte elements at the offsets below, relative to the
start of the graphics buffer (``buffers/01_0.bin`` for both measured components).
Each section word in the 23-word mesh-block header encodes
``offset = (word >> 8) * 2`` and ``format = word & 0xFF``; the same triples are
also present in ``drawable.buffers[vertex_buffer].mesh_blob.sections`` as
``[semantic, offset, format]``.

=====  =====  ==============================  ========  ==========  =====
Mesh   Slot   Meaning                         Offset    Format      Bytes
=====  =====  ==============================  ========  ==========  =====
head   8      JOINTS_0, 4 x u8                215,720   0x25        35,932
head   9      JOINTS_1, 4 x u8                251,652   0x25        35,932
head   6      WEIGHTS_0, 4 x u8 UNORM         143,856   0x26        35,932
head   7      WEIGHTS_1, 4 x u8 UNORM         179,788   0x26        35,932
jacket 8      JOINTS_0                        275,032   0x25        39,272
jacket 9      JOINTS_1                        314,304   0x25        39,272
jacket 6      WEIGHTS_0                       196,488   0x26        39,272
jacket 7      WEIGHTS_1                       235,760   0x26        39,272
=====  =====  ==============================  ========  ==========  =====

Slots 8/9 may also use the packed layout, format ``0x8C``: 16 bytes per vertex,
four little-endian u32 words whose low 12 bits are the joint index. The reader
converts each vertex to the same 4 x u8 row as ``0x25`` and rejects an index
above 255. Only ``0x8C`` is accepted: no ``0x8B`` joint layout has been measured,
so ``0x8B`` is rejected rather than guessed.

The joint index is local to the drawable model's own palette: a sorted ``u32``
joaat32 array at ``grmModel + 0x20`` with its length at ``grmModel + 0x0C``. The
ped skeleton bind pose is ``drawable.skeleton[].bind_matrix`` from a decoded
skeleton resource (bone -> model, column-major, RAGE Z-up). Palette hashes are
resolved against the drawable's own skeleton/string table and then the ped
skeleton; a hash that resolves nowhere is emitted as ``bone_0x........``.

Output has one skin. Its joints are the full ped skeleton plus any drawable-local
helper joints named by the palette. Every vertex keeps all eight influences as
``JOINTS_0``/``JOINTS_1`` (UNSIGNED_SHORT VEC4) and
``WEIGHTS_0``/``WEIGHTS_1`` (FLOAT VEC4). Source weights are u8 values that sum
to 255; they are normalised to 1.0. Bind matrices and inverse bind matrices use
the same RAGE Z-up -> glTF Y-up conversion as positions, ``(x, y, z) -> (x, z, -y)``.

Only skinned primitives are exported, with clay materials. Textures, UV atlas
data and materials of the source package are not copied. This is a static export
path: it does not assemble a body or any other unrequested component.

numpy is optional: when present it accelerates the weight pass; without it the
standard-library path produces the same bytes and is covered by the tests.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402

try:  # optional: the stdlib path below is the fallback and is covered by the tests
    import numpy as _np
except ImportError:  # pragma: no cover - depends on the interpreter
    _np = None

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling module.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from report_paths import same_report_path  # noqa: E402

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
MODE_TRIANGLES = 4

COMPONENT_BYTE = 5121
COMPONENT_UNSIGNED_SHORT = 5123
COMPONENT_UNSIGNED_INT = 5125
COMPONENT_FLOAT = 5126

JOINT_SLOTS = (8, 9)
WEIGHT_SLOTS = (6, 7)
INDEX_FORMAT = 0x25   # R8G8B8A8_UINT, one 4-byte element per vertex
PACKED_INDEX_FORMAT = 0x8C  # R32G32B32A32_UINT, 16-byte element, index in bits 0..11
WEIGHT_FORMAT = 0x26  # R8G8B8A8_UNORM
COMPONENT_SIZES = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}
TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

# RAGE Z-up -> glTF Y-up: (x, y, z) -> (x, z, -y).
_YUP = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))

# Attribute names copied from the decoded model.glb. Skin attributes are rebuilt
# from the source streams; custom private attributes are not needed for display.
GEOMETRY_ATTRIBUTES = (
    "POSITION", "NORMAL", "TANGENT",
    "TEXCOORD_0", "TEXCOORD_1", "TEXCOORD_2", "TEXCOORD_3",
    "TEXCOORD_4", "TEXCOORD_5", "TEXCOORD_6", "TEXCOORD_7",
    "COLOR_0",
)

_ORIENTATION = {"O", "I", "F", "B", "CF", "CB", "LF", "LB", "RF", "RB"}
_SIDES = {"L", "R", "C"}


class SkinError(ValueError):
    """A decoded model or skin stream is missing, inconsistent or out of range."""


# --------------------------------------------------------------------------- math
def joaat32(name: str) -> int:
    """RAGE joaat32 over the lowercased Latin-1 bytes of a bone name."""
    value = 0
    for byte in name.lower().encode("latin-1", "replace"):
        value = (value + byte) & 0xFFFFFFFF
        value = (value + (value << 10)) & 0xFFFFFFFF
        value ^= value >> 6
    value = (value + (value << 3)) & 0xFFFFFFFF
    value ^= value >> 11
    return (value + (value << 15)) & 0xFFFFFFFF


def mat_mul(a, b):
    """Column-major 4x4 product ``a * b``."""
    out = [0.0] * 16
    for column in range(4):
        for row in range(4):
            out[column * 4 + row] = sum(a[k * 4 + row] * b[column * 4 + k] for k in range(4))
    return out


def invert_affine(m):
    """Inverse of a column-major affine 4x4 (bottom row 0, 0, 0, 1)."""
    a = lambda row, column: m[column * 4 + row]
    det = (a(0, 0) * (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1))
           - a(0, 1) * (a(1, 0) * a(2, 2) - a(1, 2) * a(2, 0))
           + a(0, 2) * (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)))
    if not math.isfinite(det) or abs(det) < 1e-12:
        raise SkinError("singular or non-finite bind matrix")
    k = 1.0 / det
    cofactors = (
        (a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1)) * k,
        (a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2)) * k,
        (a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1)) * k,
        (a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2)) * k,
        (a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0)) * k,
        (a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2)) * k,
        (a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0)) * k,
        (a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1)) * k,
        (a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0)) * k,
    )
    tx, ty, tz = a(0, 3), a(1, 3), a(2, 3)
    out = [0.0] * 16
    for row in range(3):
        for column in range(3):
            out[column * 4 + row] = cofactors[row * 3 + column]
        out[12 + row] = -(cofactors[row * 3] * tx
                          + cofactors[row * 3 + 1] * ty
                          + cofactors[row * 3 + 2] * tz)
    out[15] = 1.0
    return out


def bind_to_gltf(m):
    """Convert a column-major RAGE Z-up matrix to glTF Y-up: ``C * M * C^T``."""
    a = lambda row, column: m[column * 4 + row]
    out = [0.0] * 16
    for row in range(3):
        for column in range(3):
            out[column * 4 + row] = sum(
                _YUP[row][i] * a(i, j) * _YUP[column][j]
                for i in range(3) for j in range(3)
            )
        out[12 + row] = sum(_YUP[row][i] * a(i, 3) for i in range(3))
    out[15] = 1.0
    return out


def _identity() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]


def _is_identity(m) -> bool:
    return all(abs(v - (1.0 if i in (0, 5, 10, 15) else 0.0)) <= 1e-9 for i, v in enumerate(m))


def _finite_matrix(m) -> bool:
    return len(m) == 16 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in m)


# ------------------------------------------------------------------------ files
def _load_json(path: str) -> dict:
    # An interrupted export leaves an empty or cut-off model.json: report it like any other bad input (BUILD-1031).
    try:
        with open(path, encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkinError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SkinError(f"{path}: model.json root is not an object")
    return value


def model_json_path(path: str) -> tuple[str, str]:
    """Return ``(directory, model.json path)`` for a decoded pack directory or file."""
    if os.path.isdir(path):
        return path, os.path.join(path, "model.json")
    if os.path.isfile(path) and os.path.basename(path).lower().endswith(".json"):
        return os.path.dirname(os.path.abspath(path)), path
    raise SkinError(f"{path}: expected a decoded model directory or its model.json")


def _stream_bytes(blob: bytes, base: int, offset: int, size: int, slot: int) -> bytes:
    """Slice a section at ``base + offset``, rejecting a short graphics buffer."""
    start = base + offset
    end = start + size
    if start < base or end > len(blob):
        raise SkinError(f"skin stream slot {slot} is outside its graphics buffer")
    return blob[start:end]


def _unpack_packed_joints(data: bytes, slot: int) -> bytes:
    """Convert four packed u32 joint words per vertex to four u8 indices."""
    out = bytearray()
    for at in range(0, len(data), 16):
        for word in struct.unpack_from("<4I", data, at):
            index = word & 0xFFF
            if index > 0xFF:
                raise SkinError(f"skin stream slot {slot} joint index {index} does not fit u8")
            out.append(index)
    return bytes(out)


def source_input_paths(source: str) -> list[str]:
    """Files one --skeleton/--component source may read or reference: its model.json,
    the model.glb beside it and the buffer files named in model.json. A source that cannot
    be resolved or read yet contributes only the paths known so far; the normal input checks
    report it."""
    try:
        model_dir, json_path = model_json_path(source)
    except SkinError:
        return []
    paths = [json_path]
    try:
        model = _load_json(json_path)
    except (OSError, ValueError, SkinError):
        return paths
    glb_name = model.get("glb", "model.glb")
    if isinstance(glb_name, str) and glb_name:
        paths.append(os.path.join(model_dir, glb_name))
    buffer_files = model.get("buffer_files", [])
    if isinstance(buffer_files, list):
        paths += [os.path.join(model_dir, name) for name in buffer_files
                  if isinstance(name, str) and name]
    return paths


def read_skin_streams(model_dir: str, model: dict, mesh: dict) -> dict:
    """Read one mesh's JOINTS_0/1 and WEIGHTS_0/1 streams.

    Joint slots accept the 4 x u8 layout (``0x25``) and the packed 4 x u32 layout
    (``0x8C``); both become 4 x u8 per vertex. Returns the four raw byte rows plus the
    vertex count, local palette length and measured section table. The caller still
    maps local indices to GLB joints.
    """
    drawable = model.get("drawable")
    if not isinstance(drawable, dict):
        raise SkinError("model.json has no drawable object")
    meshes = model.get("meshes")
    if not isinstance(meshes, list):
        raise SkinError("model.json has no meshes list")
    try:
        geometry = drawable["geometries"][mesh["geometry"]]
        vertex_buffer_index = geometry["vertex_buffer"]
        vertex_buffer = drawable["buffers"][vertex_buffer_index]
        model_record = drawable["models"][mesh["model"]]
    except (KeyError, IndexError, TypeError) as exc:
        raise SkinError(f"mesh links to a missing drawable record: {exc}") from exc

    palette = model_record.get("bone_palette")
    if not isinstance(palette, list) or not palette:
        raise SkinError("model has no bone palette")
    if not all(isinstance(v, int) and 0 <= v <= 0xFFFFFFFF for v in palette):
        raise SkinError("bone palette must be u32 hashes")

    buffer_name = model.get("buffer_files", [])
    try:
        buffer_file = buffer_name[vertex_buffer_index]
    except (IndexError, TypeError) as exc:
        raise SkinError("vertex buffer has no buffer_files entry") from exc
    buffer_path = os.path.join(model_dir, buffer_file)
    with open(buffer_path, "rb") as handle:
        blob = handle.read()

    base = int(vertex_buffer.get("gfx_off", 0) or 0)
    if base < 0 or base > len(blob):
        raise SkinError("vertex buffer offset is outside its buffer")

    sections = {}
    mesh_blob = vertex_buffer.get("mesh_blob")
    raw_sections = mesh_blob.get("sections") if isinstance(mesh_blob, dict) else None
    if raw_sections:
        for row in raw_sections:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise SkinError("mesh_blob.sections entries must be [semantic, offset, format]")
            semantic, offset, fmt = (int(v) for v in row)
            sections[semantic] = (offset, fmt)
    else:
        for semantic in range(23):
            word = struct.unpack_from("<I", blob, base + semantic * 4)[0]
            if word:
                sections[semantic] = ((word >> 8) * 2, word & 0xFF)

    for slot in (6, 8):
        if slot not in sections:
            raise SkinError(f"missing skin stream slot {slot}")
    if (9 in sections) != (7 in sections):
        raise SkinError("JOINTS_1 and WEIGHTS_1 must be present together")

    vertex_count = mesh.get("vertex_count", model.get("vertex_count"))
    if not isinstance(vertex_count, int) or vertex_count <= 0:
        raise SkinError("mesh has no positive vertex_count")

    rows = {}
    for slot in (6, 7, 8, 9):
        if slot not in sections:
            continue
        offset, fmt = sections[slot]
        if slot in WEIGHT_SLOTS:
            if fmt != WEIGHT_FORMAT:
                raise SkinError(
                    f"skin stream slot {slot} has format 0x{fmt:02x}; "
                    f"expected 0x{WEIGHT_FORMAT:02x}"
                )
            rows[slot] = _stream_bytes(blob, base, offset, 4 * vertex_count, slot)
        elif fmt == INDEX_FORMAT:
            rows[slot] = _stream_bytes(blob, base, offset, 4 * vertex_count, slot)
        elif fmt == PACKED_INDEX_FORMAT:
            packed = _stream_bytes(blob, base, offset, 16 * vertex_count, slot)
            rows[slot] = _unpack_packed_joints(packed, slot)
        else:
            raise SkinError(
                f"skin stream slot {slot} has format 0x{fmt:02x}; "
                f"expected 0x{INDEX_FORMAT:02x} or 0x{PACKED_INDEX_FORMAT:02x}"
            )

    return {
        "j0": rows[8],
        "j1": rows.get(9, bytes(4 * vertex_count)),
        "w0": rows[6],
        "w1": rows.get(7, bytes(4 * vertex_count)),
        "count": vertex_count,
        "local_bones": len(palette),
        "sections": sections,
        "palette": palette,
    }


# ---------------------------------------------------------------------- skeleton
def _bone_name_map(skeleton) -> tuple[dict[str, dict], dict[int, dict]]:
    if not isinstance(skeleton, list) or not skeleton:
        raise SkinError("skeleton has no bones")
    by_name, by_hash = {}, {}
    for bone in skeleton:
        name = bone.get("name")
        if not isinstance(name, str) or not name:
            raise SkinError("skeleton bone has no name")
        if not _finite_matrix(bone.get("bind_matrix")):
            raise SkinError(f"skeleton bone {name!r} has no finite bind matrix")
        by_name.setdefault(name, bone)
        by_hash.setdefault(joaat32(name), bone)
    return by_name, by_hash


class JointSet:
    """Ped joints followed by drawable-local helper joints, with bind-pose worlds."""

    def __init__(self, ped_skeleton):
        self.ped_by_name, _ = _bone_name_map(ped_skeleton)
        self.ped_count = len(ped_skeleton)
        self.names: list[str] = []
        self.binds: list[list[float]] = []
        self.parents: list[int | None] = []
        self._by_name: dict[str, int] = {}

        roots = [i for i, bone in enumerate(ped_skeleton) if bone.get("parent") is None]
        if not roots:
            raise SkinError("ped skeleton has no root bone")
        self.root = roots[0]
        for index, bone in enumerate(ped_skeleton):
            bind = bind_to_gltf(bone["bind_matrix"])
            parent = bone.get("parent")
            if not isinstance(parent, int) or not 0 <= parent < len(ped_skeleton):
                parent = None if index == self.root else self.root
            self.names.append(bone["name"])
            self.binds.append(bind)
            self.parents.append(parent)
            self._by_name.setdefault(bone["name"], index)

    @property
    def count(self) -> int:
        return len(self.names)

    def find(self, name: str) -> int | None:
        return self._by_name.get(name)

    def add(self, name: str, bind: list[float], parent: int) -> int:
        if not _finite_matrix(bind):
            raise SkinError(f"joint {name!r} has a non-finite bind matrix")
        if not 0 <= parent < self.count:
            raise SkinError(f"joint {name!r} has an invalid parent")
        existing = self._by_name.get(name)
        if existing is not None:
            return existing
        index = len(self.names)
        self.names.append(name)
        self.binds.append(bind)
        self.parents.append(parent)
        self._by_name[name] = index
        return index

    def local_matrix(self, index: int) -> list[float]:
        parent = self.parents[index]
        if parent is None:
            return list(self.binds[index])
        return mat_mul(invert_affine(self.binds[parent]), self.binds[index])

    def inverse_bind(self, index: int) -> list[float]:
        return invert_affine(self.binds[index])


def _normalise_tokens(name: str) -> tuple[list[str], str | None]:
    """Name tokens and side, with the drawable prefix and orientation suffixes removed."""
    upper = name.upper()
    for prefix in ("GM_U01_", "EGM_U01_", "SKEL_", "BM_", "MH_", "RB_", "HH_", "PH_", "EO_", "IK_", "OH_"):
        if upper.startswith(prefix):
            upper = upper[len(prefix):]
            break
    tokens = [part for part in upper.split("_") if part]
    while tokens and tokens[-1] in _ORIENTATION:
        tokens.pop()
    side = next((part for part in tokens if part in _SIDES), None)
    return tokens, side


def _ped_tokens(name: str) -> tuple[list[str], str | None]:
    return _normalise_tokens(name)


def _alias_tokens(tokens: list[str], side: str | None) -> set[str]:
    aliases = set()
    for token in tokens:
        base = token.rstrip("0123456789")
        digits = token[len(base):]
        if "UPPERARM" in base:
            aliases.add("UPPERARM")
        elif "FOREARM" in base:
            aliases.add("FOREARM")
        elif "ELBOW" in base:
            aliases.add("FOREARM")
        elif "DELT" in base or "ARMPIT" in base:
            aliases.add("UPPERARM")
        elif "THIGH" in base:
            aliases.add("PELVIS" if side == "C" else "THIGH")
        elif "TRAP" in base:
            aliases.add("CLAVICLE" if side in ("L", "R") else "SPINE6")
        elif base == "PELVIS":
            aliases.add("PELVIS")
        elif base == "CLAVICLE":
            aliases.add("CLAVICLE")
        elif base == "NECK" and digits.isdigit():
            aliases.add(token)  # Neck0 .. Neck2 are distinct deform bones
        elif base == "SPINE" and digits.isdigit():
            aliases.add(token)  # Spine0 .. Spine6 are distinct deform bones
    return aliases


def infer_ped_parent(
    name: str,
    ped_by_name: dict[str, dict],
    ped_indices: dict[str, int] | None = None,
) -> int | None:
    """Pick the ped bone a drawable-local helper covers, by name tokens.

    Garment helpers such as ``GM_*_L_UpperArm01_CF`` carry no hierarchy of their
    own in the measured resources. The helper is attached to the deform bone with
    the matching side and body-part token; anatomy aliases cover common helper
    spellings (delt/armpit -> upper arm, elbow -> forearm, side trap -> clavicle,
    centre trap -> spine6). The fallback is the first deform bone.
    """
    tokens, side = _ped_tokens(name)
    aliases = _alias_tokens(tokens, side)
    best: tuple[int, int, int, int] | None = None
    for ordinal, bone in enumerate(ped_by_name.values()):
        ped_name = bone["name"]
        if not ped_name.upper().startswith("SKEL_"):
            continue
        index = ped_indices[ped_name] if ped_indices is not None and ped_name in ped_indices else ordinal
        ped_tokens, ped_side = _ped_tokens(ped_name)
        if side in ("L", "R") and ped_side in ("L", "R") and side != ped_side:
            continue
        if side == "C" and ped_side in ("L", "R"):
            continue
        ped_set = set(ped_tokens)
        score = 0
        for alias in aliases:
            if alias in ped_set:
                score += 10
            elif not alias[-1].isdigit() and any(token.startswith(alias) for token in ped_set):
                score += 5
        if score:
            candidate = (score, -len(ped_name), -index, index)
            if best is None or candidate > best:
                best = candidate
    if best is not None:
        return best[3]
    for ordinal, bone in enumerate(ped_by_name.values()):
        ped_name = bone["name"]
        if ped_name.upper().startswith("SKEL_"):
            return ped_indices[ped_name] if ped_indices is not None and ped_name in ped_indices else ordinal
    return None


class PaletteResolver:
    """joaat32 palette -> bone name, using the drawable tables before the ped table."""

    def __init__(self, component: dict, ped_by_name: dict[str, dict]):
        self.ped_by_name = ped_by_name
        drawable = component.get("drawable") or {}
        own = drawable.get("skeleton") or []
        self.own_skeleton = own
        self.own_by_name = {}
        self.by_hash: dict[int, tuple[str, str]] = {}
        for bone in own:
            name = bone.get("name")
            if isinstance(name, str) and name:
                self.own_by_name.setdefault(name, bone)
                self.by_hash.setdefault(joaat32(name), (name, "own"))
        for text in drawable.get("strings") or []:
            if not isinstance(text, str) or not text:
                continue
            self._add_string(text, "string")
            # The measured string table has one entry with a stray leading
            # character (EGM_... for the GM_... palette hash). Index that alias
            # only when the unstripped spelling did not already claim the hash.
            if len(text) > 2 and text[0].isalpha() and text[1].isupper():
                self._add_string(text[1:], "alias", only_if_absent=True)
        for name, bone in ped_by_name.items():
            self.by_hash.setdefault(joaat32(name), (name, "ped"))

    def _add_string(self, text: str, source: str, only_if_absent: bool = False) -> None:
        key = joaat32(text)
        if only_if_absent and key in self.by_hash:
            return
        self.by_hash.setdefault(key, (text, source))

    def resolve(self, value: int) -> tuple[str, str]:
        found = self.by_hash.get(value)
        if found is None:
            return f"bone_0x{value:08x}", "fallback"
        return found


def own_parent_name(record, own_skeleton) -> str | None:
    """The name of a drawable-local bone's own_skeleton parent, by name or by index."""
    if not isinstance(record, dict):
        return None
    parent_name = record.get("parent_name")
    if parent_name is None:
        parent = record.get("parent")
        if isinstance(parent, int) and 0 <= parent < len(own_skeleton):
            parent_name = own_skeleton[parent].get("name")
    return parent_name if isinstance(parent_name, str) and parent_name else None


# --------------------------------------------------------------------- weights
def normalise_weights(j0: bytes, j1: bytes, w0: bytes, w1: bytes, count: int,
                      local_bones: int, local_to_joint: list[int],
                      joint_count: int | None = None) -> dict:
    """Map local joint bytes to GLB joints and normalise every 8-weight row to 1.0."""
    if count <= 0 or local_bones <= 0:
        raise SkinError("skin stream has no vertices or no local bones")
    if len(local_to_joint) != local_bones:
        raise SkinError("joint palette map has the wrong length")
    if any(not isinstance(v, int) or v < 0 or
           (joint_count is not None and v >= joint_count) for v in local_to_joint):
        raise SkinError("joint palette map contains an invalid joint index")
    expected = 4 * count
    for row in (j0, j1, w0, w1):
        if len(row) != expected:
            raise SkinError("skin stream is shorter than the vertex count")

    oversized = {slot for slot, joint in enumerate(local_to_joint) if joint > 0xFFFF}
    if oversized:
        for indices, weights in ((j0, w0), (j1, w1)):
            if any(weight and index in oversized for index, weight in zip(indices, weights)):
                raise SkinError("a weighted joint index exceeds the glTF UNSIGNED_SHORT limit (65535)")
        # NumPy converts the entire palette, including entries no weighted slot uses.
        local_to_joint = [0 if slot in oversized else joint for slot, joint in enumerate(local_to_joint)]

    if _np is not None:
        indices = _np.concatenate(
            [_np.frombuffer(j0, dtype=_np.uint8).reshape(count, 4),
             _np.frombuffer(j1, dtype=_np.uint8).reshape(count, 4)], axis=1,
        ).astype(_np.int64)
        weights = _np.concatenate(
            [_np.frombuffer(w0, dtype=_np.uint8).reshape(count, 4),
             _np.frombuffer(w1, dtype=_np.uint8).reshape(count, 4)], axis=1,
        ).astype(_np.float64)
        totals = weights.sum(axis=1)
        if bool((totals <= 0).any()):
            raise SkinError("a vertex has no skin weight")
        if bool((weights > 0).any()):
            weighted = indices[weights > 0]
            if int(weighted.max()) >= local_bones:
                raise SkinError("a weighted joint index is outside the local palette")
        weights /= totals[:, None]
        # A zero-weight component may carry any index; clear it before the palette lookup
        # so an out-of-palette value cannot raise where the stdlib path returns zeros.
        zero = weights == 0
        indices[zero] = 0
        mapping = _np.asarray(local_to_joint, dtype=_np.uint16)
        indices[weights == 0] = 0
        indices = mapping[indices]
        indices[zero] = 0
        if joint_count is not None and bool((indices >= joint_count).any()):
            raise SkinError("remapped joint index is invalid")
        return {
            "j0": indices[:, :4].astype("<u2").tobytes(),
            "j1": indices[:, 4:].astype("<u2").tobytes(),
            "w0": weights[:, :4].astype("<f4").tobytes(),
            "w1": weights[:, 4:].astype("<f4").tobytes(),
            "sum_min": float(totals.min()) / 255.0,
            "sum_max": float(totals.max()) / 255.0,
        }

    j0_out, j1_out, w0_out, w1_out = bytearray(), bytearray(), bytearray(), bytearray()
    sum_min, sum_max = math.inf, -math.inf
    for vertex in range(count):
        local = list(j0[4 * vertex:4 * vertex + 4]) + list(j1[4 * vertex:4 * vertex + 4])
        raw = list(w0[4 * vertex:4 * vertex + 4]) + list(w1[4 * vertex:4 * vertex + 4])
        total = sum(raw)
        if total <= 0:
            raise SkinError("a vertex has no skin weight")
        sum_min, sum_max = min(sum_min, total), max(sum_max, total)
        for slot, weight in enumerate(raw):
            if weight and local[slot] >= local_bones:
                raise SkinError("a weighted joint index is outside the local palette")
        remapped = [local_to_joint[index] if raw[slot] else 0 for slot, index in enumerate(local)]
        floats = [weight / total for weight in raw]
        j0_out.extend(struct.pack("<4H", *remapped[:4]))
        j1_out.extend(struct.pack("<4H", *remapped[4:]))
        w0_out.extend(struct.pack("<4f", *floats[:4]))
        w1_out.extend(struct.pack("<4f", *floats[4:]))
    return {
        "j0": bytes(j0_out), "j1": bytes(j1_out), "w0": bytes(w0_out), "w1": bytes(w1_out),
        "sum_min": sum_min / 255.0, "sum_max": sum_max / 255.0,
    }


# ------------------------------------------------------------------------ glb io
def read_glb(path: str) -> tuple[dict, bytes]:
    """Parse a GLB container into its JSON object and BIN chunk."""
    with open(path, "rb") as handle:
        data = handle.read()
    if len(data) < 20 or data[:4] != GLB_MAGIC:
        raise SkinError(f"{path}: not a binary glTF file")
    json_length = struct.unpack_from("<I", data, 12)[0]
    try:
        gltf = json.loads(data[20:20 + json_length].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SkinError(f"{path}: invalid JSON chunk: {exc}") from exc
    offset = 20 + json_length
    bin_data = b""
    if offset + 8 <= len(data):
        bin_length = struct.unpack_from("<I", data, offset)[0]
        bin_data = data[offset + 8:offset + 8 + bin_length]
    if not isinstance(gltf, dict):
        raise SkinError(f"{path}: JSON root is not an object")
    return gltf, bin_data


def accessor_bytes(gltf: dict, bin_data: bytes, index: int) -> tuple[bytes, dict]:
    """Return tightly packed accessor bytes and a copy of its glTF metadata."""
    try:
        accessor = gltf["accessors"][index]
        view = gltf["bufferViews"][accessor["bufferView"]]
    except (KeyError, IndexError, TypeError) as exc:
        raise SkinError(f"invalid accessor {index}: {exc}") from exc
    if accessor.get("sparse") is not None:
        raise SkinError("sparse accessors are not supported")
    component_type = accessor.get("componentType")
    if component_type not in COMPONENT_SIZES:
        raise SkinError(f"unsupported accessor component type {component_type!r}")
    element = COMPONENT_SIZES[component_type] * TYPE_COMPONENTS.get(accessor.get("type"), 0)
    count = accessor.get("count")
    if element <= 0 or not isinstance(count, int) or count < 0:
        raise SkinError(f"invalid accessor {index} shape")
    stride = view.get("byteStride", element)
    if not isinstance(stride, int) or stride < element:
        raise SkinError(f"accessor {index} has an invalid byteStride")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    end = start + ((count - 1) * stride + element if count else 0)
    if start < 0 or end > len(bin_data):
        raise SkinError(f"accessor {index} is outside the BIN chunk")
    if stride == element:
        packed = bin_data[start:start + element * count]
    else:
        packed = b"".join(bin_data[start + i * stride:start + i * stride + element]
                         for i in range(count))
    return packed, dict(accessor)


# -------------------------------------------------------------------- exporter
class _GlbBuilder:
    def __init__(self) -> None:
        self.bin = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, payload: bytes, target: int | None = None, *, stride: int | None = None) -> int:
        self.bin.extend(b"\x00" * ((-len(self.bin)) % 4))
        offset = len(self.bin)
        self.bin.extend(payload)
        entry = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
        if target is not None:
            entry["target"] = target
        if stride is not None:
            entry["byteStride"] = stride
        self.views.append(entry)
        return len(self.views) - 1

    def accessor(self, payload: bytes, component_type: int, kind: str, count: int,
                 target: int | None, *, normalized: bool = False,
                 minimum: list[float] | None = None, maximum: list[float] | None = None) -> int:
        stride = None
        if target == ARRAY_BUFFER:
            element_size = COMPONENT_SIZES[component_type] * TYPE_COMPONENTS[kind]
            padding = bytes((-element_size) % 4)
            if padding:
                # Accessors are read as tightly packed bytes, but glTF vertex rows must align to four bytes.
                payload = b"".join(payload[i:i + element_size] + padding
                                   for i in range(0, len(payload), element_size))
                stride = element_size + len(padding)
        entry = {
            "bufferView": self.view(payload, target, stride=stride),
            "componentType": component_type,
            "count": count,
            "type": kind,
        }
        if normalized:
            entry["normalized"] = True
        if minimum is not None:
            entry["min"] = minimum
        if maximum is not None:
            entry["max"] = maximum
        self.accessors.append(entry)
        return len(self.accessors) - 1


def _component_geometry(gltf: dict, bin_data: bytes, primitive: dict) -> tuple[dict, int]:
    attributes = primitive.get("attributes")
    if not isinstance(attributes, dict):
        raise SkinError("source primitive has no attributes")
    out = {}
    for name in GEOMETRY_ATTRIBUTES:
        if name not in attributes:
            continue
        payload, accessor = accessor_bytes(gltf, bin_data, attributes[name])
        out[name] = (payload, accessor)
    if "POSITION" not in out:
        raise SkinError("source primitive has no POSITION accessor")
    if "indices" not in primitive:
        raise SkinError("source primitive has no index accessor")
    index_payload, index_accessor = accessor_bytes(gltf, bin_data, primitive["indices"])
    return out, (index_payload, index_accessor)


def _source_meshes(model: dict) -> list[dict]:
    meshes = model.get("meshes")
    if not isinstance(meshes, list) or not meshes:
        raise SkinError("model.json has no mesh records")
    return meshes


def _component_primitives(gltf: dict, wanted: int) -> list[dict]:
    primitives = []
    for mesh in gltf.get("meshes") or []:
        primitives.extend(mesh.get("primitives") or [])
    if len(primitives) != wanted:
        raise SkinError(
            f"model.glb has {len(primitives)} primitives for {wanted} mesh records"
        )
    return primitives


def build_skinned_glb(skeleton_path: str, component_paths: list[str], *,
                      generator: str = "Leonida skinned_glb") -> tuple[bytes, dict]:
    """Build one skinned GLB from a ped skeleton and one or more component models."""
    if not component_paths:
        raise SkinError("at least one --component is required")
    _skeleton_dir, skeleton_json = model_json_path(skeleton_path)
    ped_model = _load_json(skeleton_json)
    ped_skeleton = (ped_model.get("drawable") or {}).get("skeleton")
    joints = JointSet(ped_skeleton)

    builder = _GlbBuilder()
    primitives_out = []
    records = []
    weight_min, weight_max = math.inf, -math.inf

    for component_path in component_paths:
        model_dir, json_path = model_json_path(component_path)
        model = _load_json(json_path)
        source_meshes = _source_meshes(model)
        glb_name = model.get("glb", "model.glb")
        if not isinstance(glb_name, str) or not glb_name:
            raise SkinError(f"{model_dir}: model.json has no model.glb name")
        gltf, glb_bin = read_glb(os.path.join(model_dir, glb_name))
        source_primitives = _component_primitives(gltf, len(source_meshes))
        pairs = [(mesh, primitive) for mesh, primitive in zip(source_meshes, source_primitives)
                 if mesh.get("skinned") and mesh.get("skin_influences")]
        if not pairs:
            raise SkinError(f"{model_dir}: model has no skinned mesh with skin_influences")
        resolver = PaletteResolver(model, joints.ped_by_name)

        def attach(name: str, chain: tuple = ()) -> int:
            """Add a drawable-local bone after the own_skeleton ancestors it names.

            A palette can list a child before its parent, and that parent can be an
            unweighted own_skeleton bone no palette entry names; attaching the ancestors
            first keeps the explicit link instead of falling back to infer_ped_parent.
            """
            existing = joints.find(name)
            if existing is not None:
                return existing
            record = resolver.own_by_name.get(name)
            parent_name = own_parent_name(record, resolver.own_skeleton)
            parent_index = joints.find(parent_name) if parent_name else None
            if parent_index is None and parent_name and parent_name not in chain:
                if parent_name in resolver.own_by_name:
                    parent_index = attach(parent_name, chain + (name,))
            if parent_index is None:
                parent_index = infer_ped_parent(name, joints.ped_by_name, joints._by_name)
                if parent_index is None:
                    parent_index = joints.root
            bind = None
            if isinstance(record, dict) and _finite_matrix(record.get("bind_matrix")):
                bind = bind_to_gltf(record["bind_matrix"])
            if bind is None:
                bind = list(joints.binds[parent_index])
            return joints.add(name, bind, parent_index)

        component_joint_start = joints.count
        component_record_start = len(records)
        for mesh, source_primitive in pairs:
            streams = read_skin_streams(model_dir, model, mesh)
            palette = streams["palette"]
            if len(palette) != streams["local_bones"]:
                raise SkinError("palette length changed while reading the skin stream")
            local_to_joint = []
            for value in palette:
                name, _source = resolver.resolve(value)
                local_to_joint.append(attach(name))

            attributes, index_pair = _component_geometry(gltf, glb_bin, source_primitive)
            position = attributes["POSITION"][1]
            if position.get("count") != streams["count"]:
                raise SkinError(
                    f"{mesh.get('name', 'mesh')}: model.glb position count "
                    f"{position.get('count')} != skin stream count {streams['count']}"
                )

            normalised = normalise_weights(
                streams["j0"], streams["j1"], streams["w0"], streams["w1"],
                streams["count"], streams["local_bones"], local_to_joint, joints.count,
            )
            weight_min = min(weight_min, normalised["sum_min"])
            weight_max = max(weight_max, normalised["sum_max"])

            gltf_attributes = {}
            for name, (payload, accessor) in attributes.items():
                minimum = accessor.get("min") if name == "POSITION" else None
                maximum = accessor.get("max") if name == "POSITION" else None
                gltf_attributes[name] = builder.accessor(
                    payload, accessor["componentType"], accessor["type"], accessor["count"],
                    ARRAY_BUFFER, normalized=bool(accessor.get("normalized")),
                    minimum=minimum, maximum=maximum,
                )
            index_payload, index_accessor = index_pair
            index_count = index_accessor["count"]
            if index_count % 3:
                raise SkinError("source index count is not a multiple of 3")
            gltf_attributes["JOINTS_0"] = builder.accessor(
                normalised["j0"], COMPONENT_UNSIGNED_SHORT, "VEC4", streams["count"], ARRAY_BUFFER)
            gltf_attributes["JOINTS_1"] = builder.accessor(
                normalised["j1"], COMPONENT_UNSIGNED_SHORT, "VEC4", streams["count"], ARRAY_BUFFER)
            gltf_attributes["WEIGHTS_0"] = builder.accessor(
                normalised["w0"], COMPONENT_FLOAT, "VEC4", streams["count"], ARRAY_BUFFER)
            gltf_attributes["WEIGHTS_1"] = builder.accessor(
                normalised["w1"], COMPONENT_FLOAT, "VEC4", streams["count"], ARRAY_BUFFER)
            index_accessor_index = builder.accessor(
                index_payload, index_accessor["componentType"], index_accessor["type"],
                index_count, ELEMENT_ARRAY_BUFFER,
            )
            primitives_out.append({
                "attributes": gltf_attributes,
                "indices": index_accessor_index,
                "material": 0,
                "mode": MODE_TRIANGLES,
            })
            records.append({
                "component": os.path.basename(os.path.dirname(json_path)) or "component",
                "mesh": mesh.get("name", "mesh"),
                "vertices": streams["count"],
                "triangles": index_count // 3,
                "source_weight_sum_min": round(normalised["sum_min"] * 255.0, 6),
                "source_weight_sum_max": round(normalised["sum_max"] * 255.0, 6),
                "local_bones": streams["local_bones"],
            })
        component_joints = joints.count - component_joint_start
        for record in records[component_record_start:]:
            record["component_local_joints"] = component_joints

    if joints.count > 0x10000:
        raise SkinError("a skin may address at most 65,536 joints with UNSIGNED_SHORT indices")
    materials = [{
        "name": "skinned_clay",
        "doubleSided": True,
        "pbrMetallicRoughness": {
            "baseColorFactor": [0.68, 0.62, 0.58, 1.0],
            "metallicFactor": 0.0,
            "roughnessFactor": 0.8,
        },
    }]

    ibm_payload = b"".join(struct.pack("<16f", *joints.inverse_bind(i))
                           for i in range(joints.count))
    ibm_accessor = builder.accessor(ibm_payload, COMPONENT_FLOAT, "MAT4", joints.count, None)
    mesh_node = len(joints.names)
    nodes = []
    for index, name in enumerate(joints.names):
        node = {"name": name}
        local = joints.local_matrix(index)
        if not _is_identity(local):
            node["matrix"] = [round(v, 9) for v in local]
        nodes.append(node)
    nodes.append({"name": "skinned_export", "mesh": 0, "skin": 0})
    for index, parent in enumerate(joints.parents):
        if parent is not None:
            nodes[parent].setdefault("children", []).append(index)

    scene_nodes = [joints.root, mesh_node]
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"name": "skinned_export", "nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": [{"name": "skinned_export", "primitives": primitives_out}],
        "materials": materials,
        "accessors": builder.accessors,
        "bufferViews": builder.views,
        "buffers": [{"byteLength": 0}],
        "skins": [{
            "name": "ped_skeleton",
            "joints": list(range(joints.count)),
            "skeleton": joints.root,
            "inverseBindMatrices": ibm_accessor,
        }],
    }
    builder.bin.extend(b"\x00" * ((-len(builder.bin)) % 4))
    gltf["buffers"][0]["byteLength"] = len(builder.bin)

    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False,
                            allow_nan=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(builder.bin)
    payload = b"".join((
        struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total),
        struct.pack("<II", len(json_bytes), CHUNK_JSON), json_bytes,
        struct.pack("<II", len(builder.bin), CHUNK_BIN), bytes(builder.bin),
    ))
    stats = {
        "bytes": len(payload),
        "vertices": sum(record["vertices"] for record in records),
        "triangles": sum(record["triangles"] for record in records),
        "joints": joints.count,
        "ped_joints": joints.ped_count,
        "component_joints": joints.count - joints.ped_count,
        "source_weight_sum_min": round(weight_min * 255.0, 6),
        "source_weight_sum_max": round(weight_max * 255.0, 6),
        "components": records,
    }
    return payload, stats


def write_skinned_glb(path: str, skeleton_path: str, component_paths: list[str], *,
                      generator: str = "Leonida skinned_glb") -> dict:
    """Build and atomically write a skinned GLB; returns its stats."""
    payload, stats = build_skinned_glb(skeleton_path, component_paths, generator=generator)
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    # A fixed `path + ".tmp"` blocks every later export when a stale directory keeps that
    # name, and two writers to one output race on the same temporary file. Use a unique name
    # in the output directory so os.replace() still publishes atomically.
    fd, temporary = tempfile.mkstemp(
        prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        os.chmod(temporary, published_mode(path))  # mkstemp creates 0600
        os.replace(temporary, path)
    except BaseException:
        try:
            # The temp was chmodded to the destination's mode before replace. On Windows a
            # read-only destination leaves a read-only temp that cannot be unlinked; restore
            # mkstemp's private writable mode first so a failed publish cleans up after itself.
            os.chmod(temporary, 0o600)
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="output .glb path")
    parser.add_argument("--skeleton", required=True,
                        help="decoded ped skeleton directory or its model.json")
    parser.add_argument("--component", action="append", required=True, metavar="DIR",
                        help="decoded component directory or its model.json (repeatable)")
    parser.add_argument("--generator", default="Leonida skinned_glb")
    args = parser.parse_args(argv)

    # The output must never replace a file the export reads (a source's model.json, its model.glb or a
    # buffer file). same_report_path also catches a case-only spelling on Windows and macOS's default
    # volume; sources it cannot resolve yet are left to the normal input checks below.
    for source in [args.skeleton] + args.component:
        for source_path in source_input_paths(source):
            if same_report_path(args.output, source_path):
                print(f"error: {args.output} must not overwrite the source {source_path}",
                      file=sys.stderr)
                return 2

    try:
        stats = write_skinned_glb(args.output, args.skeleton, args.component,
                                  generator=args.generator)
    except (SkinError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}: {stats['vertices']} vertices, {stats['triangles']} triangles, "
          f"{stats['joints']} joints ({stats['ped_joints']} ped + "
          f"{stats['component_joints']} component)")
    for component in stats["components"]:
        print(f"  {component['mesh']}: {component['vertices']} vertices, "
              f"{component['local_bones']} local bones, source weight sum "
              f"{component['source_weight_sum_min']:g}..{component['source_weight_sum_max']:g}")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
