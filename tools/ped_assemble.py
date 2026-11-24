"""Modular ped body assembler, standing-space aligner, and coplanar seam stitcher.

Usage:
    python -I tools/ped_assemble.py assemble <parts...> -o <out.glb> [options]
    python -I tools/ped_assemble.py cull-overlap --target <part> --ref <part> [options]
    python -I tools/ped_assemble.py inspect-seams <parts...> [options]

Pure standard-library Python (`python -I`, no numpy or Pillow). Implements the
three geometric assembly steps required to combine RAGE 9 modular ped parts
(`head`, `eyes`, `teeth`, `hand_*_nude_cull`, `feet_*_nude_cull`,
`bodynipplesgen_*`, `bodygenitalsgen_*`, and garments) into a watertight
binary glTF 2.0 (`.glb`):

1. **Standing-space alignment:** Human modular drawables store root-relative
   bind positions anchored at ``SKEL_ROOT = (0, 0, +1.0 m)`` in RAGE Z-up
   (``(0, +1.0 m, 0)`` in glTF Y-up). Applying the standing-space transform
   places feet on the ground plane and aligns all slots without per-part
   offset tables.
2. **Coplanar triangle overlap culling (``_nude_cull`` stitching):**
   ``hand_*_nude_cull`` (upper body) and ``feet_*_nude_cull`` (lower body)
   share an intentional ~13.9 cm coplanar waist overlap band (258 triangles),
   and ``hand_fr1_000_nude_cull`` / ``feet_fr1_000_nude_cull`` carry coplanar
   backing triangles behind ``bodynipplesgen_fr1_*`` (26 triangles) and
   ``bodygenitalsgen_fr1_*`` (64 triangles). Culling coplanar overlapping
   triangles via quantized vertex/triangle key matching plus spatial-hashed
   centroid/plane/normal checks eliminates Z-fighting while keeping the
   assembled seam watertight.
3. **Multi-part glTF 2.0 (`.glb`) export:** Merges decoded parts (positions,
   normals, UVs, optional vertex colors, optional 4- or 8-influence skinning
   ``JOINTS_0``/``WEIGHTS_0``/``JOINTS_1``/``WEIGHTS_1``, optional skeleton
   hierarchy with inverse bind matrices, and PBR material/texture slots) into
   a strict binary glTF 2.0 container.

Supports JSON mesh/manifest specs, Wavefront OBJ (via ``tools/glb_writer.py``),
and binary glTF 2.0 (``.glb``) inputs, including sparse mesh attributes and indices.
A GLB may expand up to 1,000,000 implicit zero-initialized accessor rows across its
imported mesh instances. For larger parts, supply a dense base bufferView.
Standard library only; no game files.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

import glb_writer  # noqa: E402

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
MAX_IMPLICIT_ACCESSOR_ROWS = 1_000_000
CHUNK_TYPE_JSON = 0x4E4F534A
CHUNK_TYPE_BIN = 0x004E4942
TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963

COMPONENT_BYTE = 5120
COMPONENT_UNSIGNED_BYTE = 5121
COMPONENT_SHORT = 5122
COMPONENT_UNSIGNED_SHORT = 5123
COMPONENT_UNSIGNED_INT = 5125
COMPONENT_FLOAT = 5126

COMPONENT_STRUCT = {
    COMPONENT_BYTE: ("b", 1),
    COMPONENT_UNSIGNED_BYTE: ("B", 1),
    COMPONENT_SHORT: ("h", 2),
    COMPONENT_UNSIGNED_SHORT: ("H", 2),
    COMPONENT_UNSIGNED_INT: ("I", 4),
    COMPONENT_FLOAT: ("f", 4),
}

TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}

# Standing-space offset between RAGE bind-space (SKEL_ROOT at origin, feet ~ -1.0 m)
# and standing-space (feet at ground plane, SKEL_ROOT at +1.0 m).
DEFAULT_STANDING_OFFSET_M = 1.0

# Measured coplanar culling tolerances (meters) for RAGE 9 modular ped parts:
# - hand_*_nude_cull vs feet_*_nude_cull waist overlap: 258 triangles (<1.5 mm separation)
# - hand_fr1_000_nude_cull vs bodynipplesgen_fr1_*: 26 triangles
# - feet_fr1_000_nude_cull vs bodygenitalsgen_fr1_*: 64 triangles
DEFAULT_WAIST_CENTROID_TOL = 0.004
DEFAULT_NIPPLE_CENTROID_TOL = 0.0025
DEFAULT_GENITAL_CENTROID_TOL = 0.0016
DEFAULT_VERTEX_QUANT_TOL = 1e-4
DEFAULT_NORMAL_COS_MIN = 0.85
DEFAULT_WELD_TOL = 1e-4

ROLE_UPPER_BODY = "upper_body"
ROLE_LOWER_BODY = "lower_body"
ROLE_NIPPLES = "nipples"
ROLE_GENITALS_FEMALE = "genitals_female"
ROLE_GENITALS_MALE = "genitals_male"
ROLE_HEAD = "head"
ROLE_EYES = "eyes"
ROLE_EYECAP = "eyecap"
ROLE_TEETH = "teeth"
ROLE_CLOTHING = "clothing"
ROLE_OTHER = "other"

_RE_UPPER_NUDE = re.compile(r"(?:^|[/\\])(?:hand_[a-z0-9]+_000_nude_cull|uppr(?:_|$))", re.I)
_RE_LOWER_NUDE = re.compile(r"(?:^|[/\\])(?:feet_[a-z0-9]+_000_nude_cull|lowr(?:_|$))", re.I)
_RE_NIPPLES = re.compile(r"bodynipplesgen_", re.I)
_RE_GENITALS_FR = re.compile(r"bodygenitalsgen_f[a-z0-9]+_", re.I)
_RE_GENITALS_MR = re.compile(r"bodygenitalsgen_m[a-z0-9]+_", re.I)
_RE_EYECAP = re.compile(r"(?:^|_)eyecap(?:$|_)", re.I)
_RE_EYES = re.compile(r"(?:^|_)eyes(?:$|_)", re.I)
_RE_TEETH = re.compile(r"(?:^|_)teeth", re.I)
_RE_HEAD = re.compile(r"(?:^|_)head(?:$|_)", re.I)


class PedAssembleError(ValueError):
    """Raised when a modular ped input, seam spec, or GLB container is invalid."""


@dataclass
class MaterialSpec:
    """PBR material definition for an assembled ped part."""

    name: str = "default"
    base_color_factor: tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)
    metallic_factor: float = 0.0
    roughness_factor: float = 0.85
    alpha_mode: str = "OPAQUE"
    alpha_cutoff: float | None = None
    double_sided: bool = False
    base_color_png: bytes | None = None
    normal_png: bytes | None = None
    metallic_roughness_png: bytes | None = None


@dataclass
class MeshPart:
    """Decoded modular body part geometry in a single coordinate space."""

    name: str
    positions: list[tuple[float, float, float]]
    triangles: list[tuple[int, int, int]]
    role: str = ""
    coord_space: str = "z-up"
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    normals: list[tuple[float, float, float]] | None = None
    uvs: list[tuple[float, float]] | None = None
    uvs_extra: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    colors: list[tuple[float, float, float, float]] | None = None
    joints_0: list[tuple[int, int, int, int]] | None = None
    weights_0: list[tuple[float, float, float, float]] | None = None
    joints_1: list[tuple[int, int, int, int]] | None = None
    weights_1: list[tuple[float, float, float, float]] | None = None
    joint_names: list[str] | None = None
    material: MaterialSpec | None = None

    def __post_init__(self) -> None:
        if not self.role:
            self.role = classify_part_role(self.name)
        if self.coord_space not in ("z-up", "y-up"):
            raise PedAssembleError(f"{self.name}: coord_space must be 'z-up' or 'y-up'")


@dataclass
class SkeletonJoint:
    """Single joint in an optional ped skeleton hierarchy."""

    name: str
    parent: int = -1
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    # 16 column-major floats in glTF Y-up (or RAGE Z-up if skeleton.coord_space == "z-up")
    inverse_bind_matrix: tuple[float, ...] | None = None


@dataclass
class SkeletonSpec:
    """Optional skeleton definition attached to an assembled GLB."""

    name: str
    joints: list[SkeletonJoint]
    coord_space: str = "z-up"
    standing_offset: float | None = None


# --------------------------------------------------------------------------- math & coords

def rage_to_yup_vec3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert RAGE Z-up ``(x, y, z)`` to glTF Y-up ``(x, z, -y)``."""
    return (float(v[0]), float(v[2]), float(-v[1]))


def yup_to_rage_vec3(v: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert glTF Y-up ``(x, y, z)`` to RAGE Z-up ``(x, -z, y)``."""
    return (float(v[0]), float(-v[2]), float(v[1]))


def rage_to_yup_quat(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Convert a RAGE Z-up quaternion ``(x, y, z, w)`` to glTF Y-up."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    nx, ny, nz, nw = x, z, -y, w
    norm = math.sqrt(nx * nx + ny * ny + nz * nz + nw * nw)
    if norm <= 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (nx / norm, ny / norm, nz / norm, nw / norm)


def rage_to_yup_mat4_col(col: Iterable[float]) -> tuple[float, ...]:
    """Convert a 4x4 column-major matrix from RAGE Z-up to glTF Y-up via ``C * M * C^T``."""
    vals = [float(x) for x in col]
    if len(vals) != 16:
        raise PedAssembleError(f"Expected 16 floats for MAT4, got {len(vals)}")
    # Convert column-major to row-major 4x4
    m = [[vals[c * 4 + r] for c in range(4)] for r in range(4)]
    # Basis change C: (x, y, z, 1) -> (x, z, -y, 1)
    c_mat = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
    # cm = C * M
    cm = [
        [sum(c_mat[r][k] * m[k][col_i] for k in range(4)) for col_i in range(4)]
        for r in range(4)
    ]
    # out = (C * M) * C^T
    out = [
        [sum(cm[r][k] * c_mat[col_i][k] for k in range(4)) for col_i in range(4)]
        for r in range(4)
    ]
    return tuple(out[r][col_i] for col_i in range(4) for r in range(4))


def _vec_sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _vec_add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vec_dot(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _vec_cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _vec_len(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _vec_norm(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    length = _vec_len(v)
    if length <= 1e-12:
        return None
    inv = 1.0 / length
    return (v[0] * inv, v[1] * inv, v[2] * inv)


def _identity_mat4_col() -> tuple[float, ...]:
    return (
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    )


def classify_part_role(name: str) -> str:
    """Infer the modular ped slot role from a drawable or part name."""
    base = os.path.splitext(os.path.basename(name))[0].strip().lower()
    if _RE_UPPER_NUDE.search(base):
        return ROLE_UPPER_BODY
    if _RE_LOWER_NUDE.search(base):
        return ROLE_LOWER_BODY
    if _RE_NIPPLES.search(base):
        return ROLE_NIPPLES
    if _RE_GENITALS_FR.search(base):
        return ROLE_GENITALS_FEMALE
    if _RE_GENITALS_MR.search(base):
        return ROLE_GENITALS_MALE
    if _RE_EYECAP.search(base):
        return ROLE_EYECAP
    if _RE_EYES.search(base):
        return ROLE_EYES
    if _RE_TEETH.search(base):
        return ROLE_TEETH
    if _RE_HEAD.search(base):
        return ROLE_HEAD
    if any(tok in base for tok in ("hand_", "torso", "upper", "bodyupper")):
        return ROLE_UPPER_BODY
    if any(tok in base for tok in ("feet_", "pelvis", "lower", "bodylower")):
        return ROLE_LOWER_BODY
    if any(tok in base for tok in ("nipple",)):
        return ROLE_NIPPLES
    if any(tok in base for tok in ("genital",)):
        return ROLE_GENITALS_FEMALE if ("fr1" in base or "female" in base) else ROLE_GENITALS_MALE
    if any(tok in base for tok in ("shirt", "jacket", "pants", "shorts", "shoes", "hoodie", "bra", "undies", "dress")):
        return ROLE_CLOTHING
    return ROLE_OTHER


# --------------------------------------------------------------------------- standing space

def standing_offset_vector(offset_m: float = DEFAULT_STANDING_OFFSET_M, coord_space: str = "z-up") -> tuple[float, float, float]:
    """Return the vertical ``SKEL_ROOT`` standing-space offset vector in the requested space."""
    if coord_space == "z-up":
        return (0.0, 0.0, float(offset_m))
    if coord_space == "y-up":
        return (0.0, float(offset_m), 0.0)
    raise PedAssembleError(f"Unsupported coord_space {coord_space!r}; expected 'z-up' or 'y-up'")


def is_bind_space_biped(
    positions: Iterable[tuple[float, float, float]],
    *,
    coord_space: str = "z-up",
    threshold: float = -0.4,
) -> bool:
    """Return True when the lowest vertical coordinate sits below ``threshold`` (bind space, feet ~ -1.0 m)."""
    axis = 2 if coord_space == "z-up" else 1
    min_v = None
    for p in positions:
        v = p[axis]
        if min_v is None or v < min_v:
            min_v = v
    return min_v is not None and min_v < threshold


def apply_standing_offset(
    positions: list[tuple[float, float, float]],
    *,
    offset_m: float = DEFAULT_STANDING_OFFSET_M,
    coord_space: str = "z-up",
) -> list[tuple[float, float, float]]:
    """Translate ``positions`` by the vertical ``SKEL_ROOT`` standing offset."""
    ox, oy, oz = standing_offset_vector(offset_m, coord_space)
    return [(p[0] + ox, p[1] + oy, p[2] + oz) for p in positions]


def align_part_to_standing_space(
    part: MeshPart,
    *,
    standing_offset: float = DEFAULT_STANDING_OFFSET_M,
    target_space: str = "y-up",
    apply_offset: bool = True,
) -> MeshPart:
    """Return a copy of ``part`` aligned into standing space and converted to ``target_space``.

    Applies any part-local ``translation`` first, then applies ``+standing_offset``
    along the part's vertical axis when ``apply_offset`` is True, and finally
    converts coordinates and normals from ``part.coord_space`` to ``target_space``.
    """
    if target_space not in ("z-up", "y-up"):
        raise PedAssembleError(f"Unsupported target_space {target_space!r}")
    tx, ty, tz = part.translation
    ox, oy, oz = standing_offset_vector(standing_offset, part.coord_space) if apply_offset else (0.0, 0.0, 0.0)
    shifted = [(p[0] + tx + ox, p[1] + ty + oy, p[2] + tz + oz) for p in part.positions]
    normals = list(part.normals) if part.normals is not None else None

    if part.coord_space != target_space:
        convert = rage_to_yup_vec3 if target_space == "y-up" else yup_to_rage_vec3
        shifted = [convert(p) for p in shifted]
        if normals is not None:
            normals = [convert(n) for n in normals]

    return MeshPart(
        name=part.name,
        positions=shifted,
        triangles=list(part.triangles),
        role=part.role,
        coord_space=target_space,
        translation=(0.0, 0.0, 0.0),
        normals=normals,
        uvs=list(part.uvs) if part.uvs is not None else None,
        uvs_extra={k: list(v) for k, v in part.uvs_extra.items()},
        colors=list(part.colors) if part.colors is not None else None,
        joints_0=list(part.joints_0) if part.joints_0 is not None else None,
        weights_0=list(part.weights_0) if part.weights_0 is not None else None,
        joints_1=list(part.joints_1) if part.joints_1 is not None else None,
        weights_1=list(part.weights_1) if part.weights_1 is not None else None,
        joint_names=list(part.joint_names) if part.joint_names is not None else None,
        material=part.material,
    )


# --------------------------------------------------------------------------- overlap culling & seams

def _quantize_point(p: tuple[float, float, float], tol: float) -> tuple[int, int, int]:
    inv = 1.0 / max(tol, 1e-12)
    return (int(round(p[0] * inv)), int(round(p[1] * inv)), int(round(p[2] * inv)))


def _canonical_tri_key(
    p0: tuple[float, float, float],
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    tol: float,
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    q0 = _quantize_point(p0, tol)
    q1 = _quantize_point(p1, tol)
    q2 = _quantize_point(p2, tol)
    a, b, c = sorted((q0, q1, q2))
    return (a, b, c)


def _triangle_geometry(
    positions: list[tuple[float, float, float]],
    tri: tuple[int, int, int],
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float] | None,
]:
    ox, oy, oz = offset
    a = positions[tri[0]]
    b = positions[tri[1]]
    c = positions[tri[2]]
    p0 = (a[0] + ox, a[1] + oy, a[2] + oz)
    p1 = (b[0] + ox, b[1] + oy, b[2] + oz)
    p2 = (c[0] + ox, c[1] + oy, c[2] + oz)
    centroid = ((p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0, (p0[2] + p1[2] + p2[2]) / 3.0)
    normal = _vec_norm(_vec_cross(_vec_sub(p1, p0), _vec_sub(p2, p0)))
    return p0, p1, p2, centroid, normal


def _cell_of(p: tuple[float, float, float], cell_size: float) -> tuple[int, int, int]:
    inv = 1.0 / max(cell_size, 1e-9)
    return (math.floor(p[0] * inv), math.floor(p[1] * inv), math.floor(p[2] * inv))


def find_coplanar_overlap_indices(
    target: MeshPart,
    ref_parts: list[MeshPart],
    *,
    centroid_tol: float = DEFAULT_WAIST_CENTROID_TOL,
    plane_tol: float | None = None,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
    vertex_tol: float = DEFAULT_VERTEX_QUANT_TOL,
) -> tuple[set[int], dict]:
    """Identify triangle indices in ``target`` that coplanarly overlap any triangle in ``ref_parts``.

    Combines two checks:
    1. Canonical quantized vertex-set triangle key matching (``vertex_tol``),
       catching exact 1:1 shared triangles regardless of vertex index order.
    2. Spatial-hashed centroid + plane distance + normal alignment checks
       (``centroid_tol``, ``plane_tol``, ``normal_cos_min``), catching slightly
       offset coplanar overlap strips (e.g. ``hand_*_nude_cull`` vs
       ``feet_*_nude_cull`` / ``bodynipplesgen_*`` / ``bodygenitalsgen_fr1_*``)
       while preserving nearby non-coplanar geometry.
    """
    if not target.triangles or not ref_parts:
        return set(), {
            "target": target.name,
            "target_triangles_before": len(target.triangles),
            "culled_exact_key": 0,
            "culled_coplanar_proximity": 0,
            "culled_total": 0,
            "target_triangles_after": len(target.triangles),
        }

    eff_plane_tol = float(centroid_tol if plane_tol is None else plane_tol)
    cell_size = max(float(centroid_tol), 1e-4)
    ctol_sq = float(centroid_tol) * float(centroid_tol)

    ref_exact_keys: dict[tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]], list[tuple[float, float, float] | None]] = defaultdict(list)
    grid: dict[tuple[int, int, int], list[tuple[tuple[float, float, float], tuple[float, float, float] | None]]] = defaultdict(list)

    for ref in ref_parts:
        if ref.coord_space != target.coord_space:
            raise PedAssembleError(
                f"Coordinate space mismatch: target {target.name!r} is {target.coord_space} "
                f"but ref {ref.name!r} is {ref.coord_space}"
            )
        for tri in ref.triangles:
            p0, p1, p2, cent, norm = _triangle_geometry(ref.positions, tri, ref.translation)
            key = _canonical_tri_key(p0, p1, p2, vertex_tol)
            ref_exact_keys[key].append(norm)
            grid[_cell_of(cent, cell_size)].append((cent, norm))

    culled: set[int] = set()
    exact_hits = 0
    proximity_hits = 0

    for ti, tri in enumerate(target.triangles):
        p0, p1, p2, cent, norm = _triangle_geometry(target.positions, tri, target.translation)

        # Tier 1: exact canonical quantized vertex key match
        key = _canonical_tri_key(p0, p1, p2, vertex_tol)
        if key in ref_exact_keys:
            matched_exact = False
            if normal_cos_min <= 0.0 or norm is None:
                matched_exact = True
            else:
                for rn in ref_exact_keys[key]:
                    if rn is None or abs(_vec_dot(norm, rn)) >= normal_cos_min:
                        matched_exact = True
                        break
            if matched_exact:
                culled.add(ti)
                exact_hits += 1
                continue

        # Tier 2: 3D spatial hash neighbor search for coplanar centroid/plane/normal overlap
        cx, cy, cz = _cell_of(cent, cell_size)
        hit = False
        for dx in (-1, 0, 1):
            if hit:
                break
            for dy in (-1, 0, 1):
                if hit:
                    break
                for dz in (-1, 0, 1):
                    bucket = grid.get((cx + dx, cy + dy, cz + dz))
                    if not bucket:
                        continue
                    for rcent, rnorm in bucket:
                        ddx = cent[0] - rcent[0]
                        ddy = cent[1] - rcent[1]
                        ddz = cent[2] - rcent[2]
                        if ddx * ddx + ddy * ddy + ddz * ddz > ctol_sq:
                            continue
                        if normal_cos_min > 0.0 and norm is not None and rnorm is not None:
                            if abs(_vec_dot(norm, rnorm)) < normal_cos_min:
                                continue
                        if rnorm is not None:
                            # Both centroid and all 3 vertices must lie near the reference triangle plane
                            # to reject skewed non-coplanar faces whose centroid happens to cross the plane.
                            plane_d = abs(ddx * rnorm[0] + ddy * rnorm[1] + ddz * rnorm[2])
                            if plane_d > eff_plane_tol:
                                continue
                            v_plane_limit = max(eff_plane_tol * 1.5, eff_plane_tol + 1e-4)
                            if (
                                abs(_vec_dot(_vec_sub(p0, rcent), rnorm)) > v_plane_limit
                                or abs(_vec_dot(_vec_sub(p1, rcent), rnorm)) > v_plane_limit
                                or abs(_vec_dot(_vec_sub(p2, rcent), rnorm)) > v_plane_limit
                            ):
                                continue
                        hit = True
                        break
                    if hit:
                        break
        if hit:
            culled.add(ti)
            proximity_hits += 1

    stats = {
        "target": target.name,
        "target_triangles_before": len(target.triangles),
        "culled_exact_key": exact_hits,
        "culled_coplanar_proximity": proximity_hits,
        "culled_total": len(culled),
        "target_triangles_after": len(target.triangles) - len(culled),
    }
    return culled, stats


def cull_part_overlap(
    target: MeshPart,
    ref_parts: list[MeshPart],
    *,
    centroid_tol: float = DEFAULT_WAIST_CENTROID_TOL,
    plane_tol: float | None = None,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
    vertex_tol: float = DEFAULT_VERTEX_QUANT_TOL,
    compact_vertices: bool = False,
) -> tuple[MeshPart, dict]:
    """Return a copy of ``target`` with coplanar triangles overlapping ``ref_parts`` removed."""
    culled_idx, stats = find_coplanar_overlap_indices(
        target,
        ref_parts,
        centroid_tol=centroid_tol,
        plane_tol=plane_tol,
        normal_cos_min=normal_cos_min,
        vertex_tol=vertex_tol,
    )
    kept_tris = [t for i, t in enumerate(target.triangles) if i not in culled_idx]
    out = MeshPart(
        name=target.name,
        positions=list(target.positions),
        triangles=kept_tris,
        role=target.role,
        coord_space=target.coord_space,
        translation=target.translation,
        normals=list(target.normals) if target.normals is not None else None,
        uvs=list(target.uvs) if target.uvs is not None else None,
        uvs_extra={k: list(v) for k, v in target.uvs_extra.items()},
        colors=list(target.colors) if target.colors is not None else None,
        joints_0=list(target.joints_0) if target.joints_0 is not None else None,
        weights_0=list(target.weights_0) if target.weights_0 is not None else None,
        joints_1=list(target.joints_1) if target.joints_1 is not None else None,
        weights_1=list(target.weights_1) if target.weights_1 is not None else None,
        joint_names=list(target.joint_names) if target.joint_names is not None else None,
        material=target.material,
    )
    if compact_vertices:
        out = compact_part_vertices(out)
    return out, stats


def compact_part_vertices(part: MeshPart) -> MeshPart:
    """Drop vertices no longer referenced by any triangle and remap triangle indices."""
    used: dict[int, int] = {}
    order: list[int] = []
    new_tris: list[tuple[int, int, int]] = []
    for a, b, c in part.triangles:
        remapped = []
        for idx in (a, b, c):
            ni = used.get(idx)
            if ni is None:
                ni = len(order)
                used[idx] = ni
                order.append(idx)
            remapped.append(ni)
        new_tris.append((remapped[0], remapped[1], remapped[2]))

    def _remap(seq):
        return [seq[i] for i in order] if seq is not None else None

    return MeshPart(
        name=part.name,
        positions=[part.positions[i] for i in order],
        triangles=new_tris,
        role=part.role,
        coord_space=part.coord_space,
        translation=part.translation,
        normals=_remap(part.normals),
        uvs=_remap(part.uvs),
        uvs_extra={k: [v[i] for i in order] for k, v in part.uvs_extra.items()},
        colors=_remap(part.colors),
        joints_0=_remap(part.joints_0),
        weights_0=_remap(part.weights_0),
        joints_1=_remap(part.joints_1),
        weights_1=_remap(part.weights_1),
        joint_names=list(part.joint_names) if part.joint_names is not None else None,
        material=part.material,
    )


def apply_standard_ped_culling(
    parts: list[MeshPart],
    *,
    waist_tol: float = DEFAULT_WAIST_CENTROID_TOL,
    nipple_tol: float = DEFAULT_NIPPLE_CENTROID_TOL,
    genital_tol: float = DEFAULT_GENITAL_CENTROID_TOL,
    plane_tol: float | None = None,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
    vertex_tol: float = DEFAULT_VERTEX_QUANT_TOL,
    compact_vertices: bool = False,
) -> tuple[list[MeshPart], list[dict]]:
    """Apply the RAGE 9 modular ped coplanar overlap culling passes across ``parts``.

    Passes match the measured ``_nude_cull`` overlap hierarchy:
    1. Upper body (``hand_*_nude_cull`` / ``uppr``) culled against lower body
       (``feet_*_nude_cull`` / ``lowr``) using ``waist_tol`` (default 4.0 mm).
    2. Upper body culled against female nipple patches (``bodynipplesgen_*``)
       using ``nipple_tol`` (default 2.5 mm).
    3. Lower body culled against female genital patches (``bodygenitalsgen_fr1_*``)
       using ``genital_tol`` (default 1.6 mm).
    """
    lower_parts = [p for p in parts if p.role == ROLE_LOWER_BODY]
    nipple_parts = [p for p in parts if p.role == ROLE_NIPPLES]
    fr_genital_parts = [p for p in parts if p.role == ROLE_GENITALS_FEMALE]

    out_parts: list[MeshPart] = []
    reports: list[dict] = []

    for part in parts:
        cur = part
        if part.role == ROLE_UPPER_BODY:
            if lower_parts:
                cur, st = cull_part_overlap(
                    cur,
                    lower_parts,
                    centroid_tol=waist_tol,
                    plane_tol=plane_tol,
                    normal_cos_min=normal_cos_min,
                    vertex_tol=vertex_tol,
                )
                st["pass"] = "upper_vs_lower_waist"
                reports.append(st)
            if nipple_parts:
                cur, st = cull_part_overlap(
                    cur,
                    nipple_parts,
                    centroid_tol=nipple_tol,
                    plane_tol=plane_tol,
                    normal_cos_min=normal_cos_min,
                    vertex_tol=vertex_tol,
                )
                st["pass"] = "upper_vs_nipples"
                reports.append(st)
            if compact_vertices and cur is not part:
                cur = compact_part_vertices(cur)
        elif part.role == ROLE_LOWER_BODY:
            if fr_genital_parts:
                cur, st = cull_part_overlap(
                    cur,
                    fr_genital_parts,
                    centroid_tol=genital_tol,
                    plane_tol=plane_tol,
                    normal_cos_min=normal_cos_min,
                    vertex_tol=vertex_tol,
                )
                st["pass"] = "lower_vs_female_genitals"
                reports.append(st)
                if compact_vertices:
                    cur = compact_part_vertices(cur)
        out_parts.append(cur)

    return out_parts, reports


def count_mesh_islands(
    positions: list[tuple[float, float, float]],
    triangles: list[tuple[int, int, int]],
    *,
    by_position: bool = False,
    quant_tol: float = DEFAULT_WELD_TOL,
) -> dict:
    """Count disconnected triangle islands (connected components) in a mesh.

    With ``by_position=False`` (default), connectivity follows shared vertex
    indices, matching RAGE 9 physical ``_nude_cull`` island splits (e.g.
    449 / 290 / 257 / 264 islands). With ``by_position=True``, vertices within
    ``quant_tol`` are welded before computing components.
    """
    n_tris = len(triangles)
    if n_tris == 0:
        return {"island_count": 0, "largest_island_triangles": 0, "smallest_island_triangles": 0}

    parent = list(range(n_tris))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    vert_owner: dict[object, int] = {}
    for ti, tri in enumerate(triangles):
        for vi in tri:
            key: object = _quantize_point(positions[vi], quant_tol) if by_position else vi
            first = vert_owner.get(key)
            if first is None:
                vert_owner[key] = ti
            else:
                _union(first, ti)

    sizes: dict[int, int] = defaultdict(int)
    for ti in range(n_tris):
        sizes[_find(ti)] += 1
    counts = sorted(sizes.values(), reverse=True)
    return {
        "island_count": len(counts),
        "largest_island_triangles": counts[0],
        "smallest_island_triangles": counts[-1],
    }


def analyze_edge_topology(
    parts: list[MeshPart],
    *,
    weld_tol: float = DEFAULT_WELD_TOL,
) -> dict:
    """Analyze welded edge manifoldness across one or more parts.

    Returns counts of total unique welded edges, boundary edges (incident to
    1 triangle), manifold interior edges (incident to 2 triangles), and
    non-manifold edges (incident to >2 triangles — typically caused by
    double-drawn coplanar overlap bands). Triangles with fewer than three
    distinct welded corners are counted separately; they no longer form faces.
    """
    edge_counts: dict[tuple[tuple[int, int, int], tuple[int, int, int]], int] = defaultdict(int)
    collapsed = 0
    for part in parts:
        ox, oy, oz = part.translation
        qpts = [
            _quantize_point((p[0] + ox, p[1] + oy, p[2] + oz), weld_tol)
            for p in part.positions
        ]
        for a, b, c in part.triangles:
            qa, qb, qc = qpts[a], qpts[b], qpts[c]
            if qa == qb or qb == qc or qc == qa:
                collapsed += 1
                continue
            for u, v in ((qa, qb), (qb, qc), (qc, qa)):
                edge = (u, v) if u < v else (v, u)
                edge_counts[edge] += 1

    boundary = 0
    manifold = 0
    non_manifold = 0
    for cnt in edge_counts.values():
        if cnt == 1:
            boundary += 1
        elif cnt == 2:
            manifold += 1
        else:
            non_manifold += 1

    return {
        "unique_edges": len(edge_counts),
        "boundary_edges": boundary,
        "manifold_edges": manifold,
        "non_manifold_edges": non_manifold,
        "collapsed_triangles": collapsed,
    }


def inspect_seams(
    parts: list[MeshPart],
    *,
    weld_tol: float = DEFAULT_WELD_TOL,
    overlap_tol: float = DEFAULT_WAIST_CENTROID_TOL,
    plane_tol: float | None = None,
    normal_cos_min: float = DEFAULT_NORMAL_COS_MIN,
) -> dict:
    """Inspect islands, bounding boxes, welded edge topology, and pairwise overlaps."""
    part_summaries = []
    for p in parts:
        ox, oy, oz = p.translation
        if p.positions:
            xs = [v[0] + ox for v in p.positions]
            ys = [v[1] + oy for v in p.positions]
            zs = [v[2] + oz for v in p.positions]
            bounds = [[min(xs), min(ys), min(zs)], [max(xs), max(ys), max(zs)]]
        else:
            bounds = None
        islands = count_mesh_islands(p.positions, p.triangles, by_position=False)
        welded_islands = count_mesh_islands(p.positions, p.triangles, by_position=True, quant_tol=weld_tol)
        part_summaries.append({
            "name": p.name,
            "role": p.role,
            "coord_space": p.coord_space,
            "vertices": len(p.positions),
            "triangles": len(p.triangles),
            "physical_islands": islands["island_count"],
            "welded_islands": welded_islands["island_count"],
            "bounds": bounds,
        })

    pairwise_overlaps = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            culled_ij, st = find_coplanar_overlap_indices(
                parts[i],
                [parts[j]],
                centroid_tol=overlap_tol,
                plane_tol=plane_tol,
                normal_cos_min=normal_cos_min,
            )
            if culled_ij:
                pairwise_overlaps.append({
                    "part_a": parts[i].name,
                    "part_b": parts[j].name,
                    "overlapping_triangles_in_a": len(culled_ij),
                    "exact_key_hits": st["culled_exact_key"],
                    "proximity_hits": st["culled_coplanar_proximity"],
                })

    topology = analyze_edge_topology(parts, weld_tol=weld_tol)
    return {
        "part_count": len(parts),
        "parts": part_summaries,
        "combined_topology": topology,
        "pairwise_overlaps": pairwise_overlaps,
    }


# --------------------------------------------------------------------------- I/O (JSON, OBJ, GLB)

def _read_png_spec(val: object, base_dir: str) -> bytes | None:
    if val is None:
        return None
    if isinstance(val, (bytes, bytearray)):
        return bytes(val)
    if isinstance(val, str):
        if val.startswith("base64:"):
            return base64.b64decode(val[len("base64:"):])
        path = val if os.path.isabs(val) else os.path.join(base_dir, val)
        with open(path, "rb") as f:
            return f.read()
    raise PedAssembleError(f"Unsupported PNG texture spec: {type(val).__name__}")


def _parse_material_dict(raw: dict | None, base_dir: str) -> MaterialSpec | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise PedAssembleError("material must be a JSON object")
    bcf = raw.get("base_color_factor", raw.get("baseColorFactor", (0.8, 0.8, 0.8, 1.0)))
    if len(bcf) == 3:
        bcf = (float(bcf[0]), float(bcf[1]), float(bcf[2]), 1.0)
    elif len(bcf) == 4:
        bcf = (float(bcf[0]), float(bcf[1]), float(bcf[2]), float(bcf[3]))
    else:
        raise PedAssembleError("base_color_factor must have 3 or 4 floats")
    return MaterialSpec(
        name=str(raw.get("name", "material")),
        base_color_factor=bcf,
        metallic_factor=float(raw.get("metallic_factor", raw.get("metallicFactor", 0.0))),
        roughness_factor=float(raw.get("roughness_factor", raw.get("roughnessFactor", 0.85))),
        alpha_mode=str(raw.get("alpha_mode", raw.get("alphaMode", "OPAQUE"))).upper(),
        alpha_cutoff=float(raw["alpha_cutoff"]) if "alpha_cutoff" in raw and raw["alpha_cutoff"] is not None else None,
        double_sided=bool(raw.get("double_sided", raw.get("doubleSided", False))),
        base_color_png=_read_png_spec(raw.get("base_color_png", raw.get("baseColorPng")), base_dir),
        normal_png=_read_png_spec(raw.get("normal_png", raw.get("normalPng")), base_dir),
        metallic_roughness_png=_read_png_spec(raw.get("metallic_roughness_png", raw.get("metallicRoughnessPng")), base_dir),
    )


def _normalize_triangles(raw_indices: list) -> list[tuple[int, int, int]]:
    if not raw_indices:
        return []
    first = raw_indices[0]
    if isinstance(first, (list, tuple)):
        return [(int(t[0]), int(t[1]), int(t[2])) for t in raw_indices]
    if len(raw_indices) % 3 != 0:
        raise PedAssembleError(f"Flat index array length {len(raw_indices)} is not a multiple of 3")
    return [
        (int(raw_indices[i]), int(raw_indices[i + 1]), int(raw_indices[i + 2]))
        for i in range(0, len(raw_indices), 3)
    ]


def _parse_joint_names(raw: object, *, part_name: str) -> list[str] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise PedAssembleError(f"{part_name}: joint_names must be a list of strings")
    out: list[str] = []
    for idx, item in enumerate(raw):
        if item is None:
            out.append("")
        elif isinstance(item, str):
            out.append(item)
        else:
            raise PedAssembleError(
                f"{part_name}: joint_names[{idx}] must be a string or null, got {type(item).__name__}"
            )
    return out


def _part_from_dict(raw: dict, *, default_name: str, default_space: str, base_dir: str) -> MeshPart:
    name = str(raw.get("name", default_name))
    positions = [(float(p[0]), float(p[1]), float(p[2])) for p in raw.get("positions", [])]
    triangles = _normalize_triangles(raw.get("triangles", raw.get("indices", [])))
    normals = [(float(n[0]), float(n[1]), float(n[2])) for n in raw["normals"]] if raw.get("normals") else None
    uvs = [(float(u[0]), float(u[1])) for u in raw["uvs"]] if raw.get("uvs") else None
    raw_extra = raw.get("uvs_extra", {})
    if not isinstance(raw_extra, dict) or any(
        re.fullmatch(r"[1-9][0-9]{0,8}", str(key)) is None for key in raw_extra
    ):
        raise PedAssembleError(f"{name}: uvs_extra keys must be positive texture coordinate set indices")
    uvs_extra = {
        int(key): [(float(u[0]), float(u[1])) for u in values]
        for key, values in raw_extra.items()
    }
    colors = None
    if raw.get("colors"):
        colors = [
            (float(c[0]), float(c[1]), float(c[2]), float(c[3]) if len(c) > 3 else 1.0)
            for c in raw["colors"]
        ]
    joints_0 = [tuple(int(x) for x in j) for j in raw["joints_0"]] if raw.get("joints_0") else None
    weights_0 = [tuple(float(x) for x in w) for w in raw["weights_0"]] if raw.get("weights_0") else None
    joints_1 = [tuple(int(x) for x in j) for j in raw["joints_1"]] if raw.get("joints_1") else None
    weights_1 = [tuple(float(x) for x in w) for w in raw["weights_1"]] if raw.get("weights_1") else None
    joint_names = _parse_joint_names(raw.get("joint_names", raw.get("jointNames")), part_name=name)
    trans = raw.get("translation", (0.0, 0.0, 0.0))
    return MeshPart(
        name=name,
        positions=positions,
        triangles=triangles,
        role=str(raw.get("role", "")),
        coord_space=str(raw.get("coord_space", default_space)),
        translation=(float(trans[0]), float(trans[1]), float(trans[2])),
        normals=normals,
        uvs=uvs,
        uvs_extra=uvs_extra,
        colors=colors,
        joints_0=joints_0,
        weights_0=weights_0,
        joints_1=joints_1,
        weights_1=weights_1,
        joint_names=joint_names,
        material=_parse_material_dict(raw.get("material"), base_dir),
    )


def _skeleton_from_dict(raw: dict | None, default_space: str) -> SkeletonSpec | None:
    if raw is None:
        return None
    joints = []
    for j in raw.get("joints", []):
        t = j.get("translation", (0.0, 0.0, 0.0))
        r = j.get("rotation", (0.0, 0.0, 0.0, 1.0))
        s = j.get("scale", (1.0, 1.0, 1.0))
        ibm = j.get("inverse_bind_matrix", j.get("inverseBindMatrix"))
        joints.append(SkeletonJoint(
            name=str(j["name"]),
            parent=int(j.get("parent", -1)),
            translation=(float(t[0]), float(t[1]), float(t[2])),
            rotation=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
            scale=(float(s[0]), float(s[1]), float(s[2])),
            inverse_bind_matrix=tuple(float(x) for x in ibm) if ibm is not None else None,
        ))
    return SkeletonSpec(
        name=str(raw.get("name", "ped_skeleton")),
        joints=joints,
        coord_space=str(raw.get("coord_space", default_space)),
        standing_offset=float(raw["standing_offset"]) if "standing_offset" in raw and raw["standing_offset"] is not None else None,
    )


def part_to_dict(part: MeshPart) -> dict:
    """Serialize a ``MeshPart`` to a JSON-compatible dict."""
    out: dict = {
        "name": part.name,
        "role": part.role,
        "coord_space": part.coord_space,
        "translation": list(part.translation),
        "positions": [list(p) for p in part.positions],
        "triangles": [list(t) for t in part.triangles],
    }
    if part.normals is not None:
        out["normals"] = [list(n) for n in part.normals]
    if part.uvs is not None:
        out["uvs"] = [list(u) for u in part.uvs]
    if part.uvs_extra:
        out["uvs_extra"] = {str(key): [list(u) for u in values]
                            for key, values in sorted(part.uvs_extra.items())}
    if part.colors is not None:
        out["colors"] = [list(c) for c in part.colors]
    if part.joints_0 is not None:
        out["joints_0"] = [list(j) for j in part.joints_0]
    if part.weights_0 is not None:
        out["weights_0"] = [list(w) for w in part.weights_0]
    if part.joints_1 is not None:
        out["joints_1"] = [list(j) for j in part.joints_1]
    if part.weights_1 is not None:
        out["weights_1"] = [list(w) for w in part.weights_1]
    if part.joint_names is not None:
        out["joint_names"] = list(part.joint_names)
    if part.material is not None:
        mat: dict = {
            "name": part.material.name,
            "base_color_factor": list(part.material.base_color_factor),
            "metallic_factor": part.material.metallic_factor,
            "roughness_factor": part.material.roughness_factor,
            "alpha_mode": part.material.alpha_mode,
            "double_sided": part.material.double_sided,
        }
        if part.material.alpha_cutoff is not None:
            mat["alpha_cutoff"] = part.material.alpha_cutoff
        for slot in ("base_color_png", "normal_png", "metallic_roughness_png"):
            payload = getattr(part.material, slot)
            if payload is not None:
                mat[slot] = "base64:" + base64.b64encode(payload).decode("ascii")
        out["material"] = mat
    return out


_GLB_ATTRIBUTE_TYPES = {
    "POSITION": ("VEC3",),
    "NORMAL": ("VEC3",),
    "TEXCOORD_0": ("VEC2",),
    "COLOR_0": ("VEC3", "VEC4"),
    "JOINTS_0": ("VEC4",),
    "WEIGHTS_0": ("VEC4",),
    "JOINTS_1": ("VEC4",),
    "WEIGHTS_1": ("VEC4",),
}


def _check_glb_document(doc: object, bin_chunk: bytes, path: str) -> None:
    """Refuse a GLB document `read_glb_parts` cannot walk, with one message instead of a traceback.

    Checks only what the reader dereferences: the tables it indexes, the accessors its primitives name (their
    type, and that their bytes are inside the BIN chunk) and the material fields it converts.
    """
    def bad(detail: str) -> PedAssembleError:
        return PedAssembleError(f"{path}: malformed GLB: {detail}")

    def is_index(value: object) -> bool:
        return type(value) is int and value >= 0

    def is_whole(value: object) -> bool:
        """A count, offset or stride: the reader takes these through int(), so 3.0 reads as 3."""
        return is_index(value) or (type(value) is float and value.is_integer() and value >= 0)

    def check_name(value: object, what: str) -> None:
        """The part name is the node's name, else the mesh's; a value that is not text cannot be classified."""
        if value and not isinstance(value, str):
            raise bad(f"{what} must be a string, got {value!r}")

    def number(value: object, what: str) -> None:
        try:
            float(value)
        except (TypeError, ValueError):
            raise bad(f"{what} must be a number, got {value!r}") from None

    if not isinstance(doc, dict):
        raise bad("the JSON chunk must be an object")
    tables = {}
    for name in ("nodes", "meshes", "materials", "skins", "accessors", "bufferViews"):
        table = doc.get(name, [])
        if not isinstance(table, list):
            raise bad(f"{name} must be a list")
        tables[name] = table
    nodes, meshes, materials = tables["nodes"], tables["meshes"], tables["materials"]
    accessors, views = tables["accessors"], tables["bufferViews"]

    def check_accessor(index: object, what: str, allowed: tuple[str, ...]) -> None:
        if not is_index(index) or index >= len(accessors):
            raise bad(f"{what} names accessor {index!r} (GLB has {len(accessors)})")
        acc = accessors[index]
        if not isinstance(acc, dict):
            raise bad(f"accessors[{index}] must be an object")
        ctype, atype, count = acc.get("componentType"), acc.get("type"), acc.get("count")
        if type(ctype) is not int or ctype not in COMPONENT_STRUCT:
            raise bad(f"accessors[{index}] has unsupported componentType {ctype!r}")
        if not isinstance(atype, str) or atype not in TYPE_COMPONENTS:
            raise bad(f"accessors[{index}] has unsupported type {atype!r}")
        if atype not in allowed:
            raise bad(f"{what} is accessor {index} of type {atype}, expected {' or '.join(allowed)}")
        if not is_whole(count):
            raise bad(f"accessors[{index}] has invalid count {count!r}")
        count = int(count)
        component_size = COMPONENT_STRUCT[ctype][1]
        elem_bytes = component_size * TYPE_COMPONENTS[atype]
        label = f"accessors[{index}]"
        has_sparse = "sparse" in acc

        def check_region(section, name, rows, size, alignment, *, packed=False):
            """Validate the rows before unpacking them; sparse indices/values always use tight packing."""
            view_index = section.get("bufferView")
            if packed and is_whole(view_index):
                view_index = int(view_index)
            if not is_index(view_index) or view_index >= len(views):
                raise bad(f"{name} names bufferView {view_index!r} (GLB has {len(views)})")
            view = views[view_index]
            if not isinstance(view, dict):
                raise bad(f"bufferViews[{view_index}] must be an object")
            offsets = (view.get("byteOffset", 0), section.get("byteOffset", 0))
            stride = view.get("byteStride")
            if not all(is_whole(v) for v in offsets) or not (stride is None or is_whole(stride)):
                raise bad(f"{name} has an invalid byteOffset or byteStride")
            start, offset = map(int, offsets)
            step = int(stride or size)
            end = offset + ((rows - 1) * step + size if rows else 0)
            if rows and start + end > len(bin_chunk):
                raise bad(f"{name} ({what}) runs past the BIN chunk ({len(bin_chunk)} bytes)")
            if has_sparse:
                if packed and ("byteStride" in view or "target" in view):
                    raise bad(f"{name} must use a bufferView without byteStride or target")
                if offset % alignment or (start + offset) % alignment or step < size or step % alignment:
                    raise bad(f"{name} has an invalid offset alignment or stride")
                buffers = doc.get("buffers")
                if (not is_whole(view.get("buffer")) or view["buffer"] != 0
                        or not isinstance(buffers, list) or not buffers or not isinstance(buffers[0], dict)
                        or "uri" in buffers[0]):
                    raise bad(f"{name} must use the embedded GLB buffer")
                length = view.get("byteLength")
                limit = buffers[0].get("byteLength")
                if (not is_whole(length) or not is_whole(limit) or end > length
                        or start + length > limit or limit > len(bin_chunk)):
                    raise bad(f"{name} exceeds its bufferView or embedded buffer")
            return start + offset

        if "bufferView" in acc:
            check_region(acc, label, count, elem_bytes, component_size)
        elif "byteOffset" in acc:
            raise bad(f"{label} cannot specify byteOffset without a base bufferView")
        elif not has_sparse:
            required = doc.get("extensionsRequired", [])
            if not isinstance(required, list):
                raise bad("extensionsRequired must be a list")
            if "KHR_draco_mesh_compression" in required:
                raise PedAssembleError(f"{path}: required KHR_draco_mesh_compression is unsupported")
        if not has_sparse:
            return

        sparse = acc["sparse"]
        if not isinstance(sparse, dict):
            raise bad(f"{label}.sparse must be an object")
        n = sparse.get("count")
        if not is_whole(n) or not 1 <= n <= count:
            raise bad(f"{label}.sparse.count must be between 1 and {count}")
        n = int(n)
        indices, values = sparse.get("indices"), sparse.get("values")
        if not isinstance(indices, dict) or not isinstance(values, dict):
            raise bad(f"{label}.sparse needs indices and values objects")
        index_type = indices.get("componentType")
        if not is_whole(index_type) or index_type not in (COMPONENT_UNSIGNED_BYTE, COMPONENT_UNSIGNED_SHORT,
                                                           COMPONENT_UNSIGNED_INT):
            raise bad(f"{label}.sparse.indices must use unsigned byte, short or int components")
        fmt, width = COMPONENT_STRUCT[int(index_type)]
        start = check_region(indices, f"{label}.sparse.indices", n, width, width, packed=True)
        check_region(values, f"{label}.sparse.values", n, elem_bytes, component_size, packed=True)
        previous = -1
        for (dest,) in struct.iter_unpack("<" + fmt, memoryview(bin_chunk)[start:start + n * width]):
            if not previous < dest < count:
                raise bad(f"{label}.sparse.indices must be strictly increasing and below {count}")
            previous = dest

    for i, mesh in enumerate(meshes):
        if not isinstance(mesh, dict):
            raise bad(f"meshes[{i}] must be an object")
        primitives = mesh.get("primitives", [])
        if not isinstance(primitives, list):
            raise bad(f"meshes[{i}].primitives must be a list")
        for pi, prim in enumerate(primitives):
            where = f"meshes[{i}].primitives[{pi}]"
            if not isinstance(prim, dict):
                raise bad(f"{where} must be an object")
            attrs = prim.get("attributes", {})
            if not isinstance(attrs, dict):
                raise bad(f"{where}.attributes must be an object")
            if "POSITION" not in attrs:
                continue
            for name, allowed in _GLB_ATTRIBUTE_TYPES.items():
                if name in attrs:
                    check_accessor(attrs[name], f"{where}.{name}", allowed)
            for name, index in attrs.items():
                if re.fullmatch(r"TEXCOORD_[1-9][0-9]{0,8}", name):
                    check_accessor(index, f"{where}.{name}", ("VEC2",))
            if "indices" in prim:
                check_accessor(prim["indices"], f"{where}.indices", ("SCALAR",))
            if "material" not in prim:
                continue
            material_index = prim["material"]
            if type(material_index) is not int:
                raise bad(f"{where}.material must be an index, got {material_index!r}")
            if not 0 <= material_index < len(materials):
                continue                                    # the reader leaves such a part without a material
            material = materials[material_index]
            if not isinstance(material, dict):
                raise bad(f"materials[{material_index}] must be an object")
            pbr = material.get("pbrMetallicRoughness", {})
            if not isinstance(pbr, dict):
                raise bad(f"materials[{material_index}].pbrMetallicRoughness must be an object")
            factor = pbr.get("baseColorFactor", [0.8, 0.8, 0.8, 1.0])
            if not isinstance(factor, list) or len(factor) < 4:
                raise bad(f"materials[{material_index}] baseColorFactor must be a list of 4 numbers")
            for value in factor[:4]:
                number(value, f"materials[{material_index}] baseColorFactor")
            for owner, key in ((pbr, "metallicFactor"), (pbr, "roughnessFactor"), (material, "alphaCutoff")):
                if key in owner:
                    number(owner[key], f"materials[{material_index}] {key}")

    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            raise bad(f"nodes[{i}] must be an object")
    if not any("mesh" in node for node in nodes):
        for i, mesh in enumerate(meshes):                   # the reader then makes one part per mesh
            check_name(mesh.get("name"), f"meshes[{i}].name")
    for i, node in enumerate(nodes):
        if "mesh" not in node:
            continue
        if not is_index(node["mesh"]) or node["mesh"] >= len(meshes):
            raise bad(f"nodes[{i}] names mesh {node['mesh']!r} (GLB has {len(meshes)})")
        check_name(node.get("name"), f"nodes[{i}].name")
        if not node.get("name"):
            check_name(meshes[node["mesh"]].get("name"), f"meshes[{node['mesh']}].name")
        translation = node.get("translation", (0.0, 0.0, 0.0))
        if not isinstance(translation, (list, tuple)) or len(translation) != 3:
            raise bad(f"nodes[{i}].translation must be a list of 3 numbers")
        for value in translation:
            number(value, f"nodes[{i}].translation")


def _read_glb_accessor(doc: dict, bin_chunk: bytes, acc_idx: int) -> list:
    acc = doc["accessors"][acc_idx]
    ctype = acc["componentType"]
    atype = acc["type"]
    count = int(acc["count"])
    fmt_char, comp_size = COMPONENT_STRUCT[ctype]
    n_comp = TYPE_COMPONENTS[atype]
    elem_bytes = comp_size * n_comp
    normalized = bool(acc.get("normalized", False))
    denom = None
    if normalized:
        if ctype == COMPONENT_UNSIGNED_BYTE:
            denom = 255.0
        elif ctype == COMPONENT_UNSIGNED_SHORT:
            denom = 65535.0
        elif ctype == COMPONENT_BYTE:
            denom = 127.0
        elif ctype == COMPONENT_SHORT:
            denom = 32767.0

    unpack = struct.Struct(f"<{n_comp}{fmt_char}").unpack_from

    def read_row(offset):
        vals = unpack(bin_chunk, offset)
        if denom is not None:
            if ctype in (COMPONENT_BYTE, COMPONENT_SHORT):
                # Both signed minimum encodings represent -1 in glTF's normalized range.
                vals = tuple(max(v, -denom) / denom for v in vals)
            else:
                vals = tuple(v / denom for v in vals)
        return vals[0] if n_comp == 1 else vals

    def start_of(section):
        view = doc["bufferViews"][int(section["bufferView"])]
        return int(view.get("byteOffset", 0)) + int(section.get("byteOffset", 0))

    if "bufferView" in acc:
        base = start_of(acc)
        stride = int(doc["bufferViews"][acc["bufferView"]].get("byteStride") or elem_bytes)
        out = [read_row(base + i * stride) for i in range(count)]
    else:
        zero = 0 if n_comp == 1 else (0,) * n_comp
        out = [zero] * count

    if "sparse" in acc:
        sparse = acc["sparse"]
        indices = sparse["indices"]
        fmt, width = COMPONENT_STRUCT[int(indices["componentType"])]
        start = start_of(indices)
        packed_indices = memoryview(bin_chunk)[start:start + int(sparse["count"]) * width]
        values_start = start_of(sparse["values"])
        for i, (dest,) in enumerate(struct.iter_unpack("<" + fmt, packed_indices)):
            out[dest] = read_row(values_start + i * elem_bytes)
    return out


def _extract_glb_skin_joint_names(
    doc: dict,
    skin_idx: object,
    *,
    path: str,
    part_name: str,
) -> list[str]:
    skins = doc.get("skins", [])
    nodes = doc.get("nodes", [])
    if not isinstance(skin_idx, int) or isinstance(skin_idx, bool) or not (0 <= skin_idx < len(skins)):
        raise PedAssembleError(
            f"{path}: part {part_name!r} references invalid skin index {skin_idx!r} "
            f"(GLB has {len(skins)} skin(s))"
        )
    skin = skins[skin_idx]
    if not isinstance(skin, dict):
        raise PedAssembleError(f"{path}: skins[{skin_idx}] must be a JSON object")
    raw_joints = skin.get("joints")
    if not isinstance(raw_joints, list):
        raise PedAssembleError(f"{path}: skins[{skin_idx}].joints must be a list")
    joint_names: list[str] = []
    for slot, j_node_idx in enumerate(raw_joints):
        if (
            not isinstance(j_node_idx, int)
            or isinstance(j_node_idx, bool)
            or not (0 <= j_node_idx < len(nodes))
        ):
            raise PedAssembleError(
                f"{path}: skins[{skin_idx}].joints[{slot}] references invalid node index {j_node_idx!r} "
                f"(GLB has {len(nodes)} node(s))"
            )
        jnode = nodes[j_node_idx]
        if not isinstance(jnode, dict):
            raise PedAssembleError(f"{path}: nodes[{j_node_idx}] must be a JSON object")
        raw_name = jnode.get("name")
        if raw_name is None:
            joint_names.append("")
        elif isinstance(raw_name, str):
            joint_names.append(raw_name)
        else:
            raise PedAssembleError(
                f"{path}: skins[{skin_idx}].joints[{slot}] (node {j_node_idx}) "
                f"has non-string name {raw_name!r}"
            )
    return joint_names


def read_glb_parts(path: str) -> list[MeshPart]:
    """Read mesh parts from a binary glTF 2.0 (``.glb``) file (always Y-up)."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20 or data[:4] != GLB_MAGIC:
        raise PedAssembleError(f"{path}: not a valid GLB container")
    version, total_len = struct.unpack_from("<II", data, 4)
    if version != GLB_VERSION or total_len > len(data):
        raise PedAssembleError(f"{path}: unsupported GLB version or truncated length")

    pos = 12
    doc = None
    bin_chunk = b""
    while pos + 8 <= total_len:
        chunk_len, chunk_type = struct.unpack_from("<II", data, pos)
        pos += 8
        payload = data[pos:pos + chunk_len]
        pos += chunk_len
        if chunk_type == CHUNK_TYPE_JSON:
            try:
                doc = json.loads(payload.decode("utf-8"))
            except ValueError as exc:                       # UnicodeDecodeError and JSONDecodeError
                raise PedAssembleError(f"{path}: malformed GLB: the JSON chunk does not parse ({exc})") from None
        elif chunk_type == CHUNK_TYPE_BIN:
            bin_chunk = payload
    if doc is None:
        raise PedAssembleError(f"{path}: GLB has no JSON chunk")
    _check_glb_document(doc, bin_chunk, path)

    stem = os.path.splitext(os.path.basename(path))[0]
    nodes = doc.get("nodes", [])
    meshes = doc.get("meshes", [])
    materials = doc.get("materials", [])
    skins = doc.get("skins", [])
    parts: list[MeshPart] = []

    # Unlike dense rows, a zero base has no corresponding bytes in the file to bound the
    # Python lists built for each visible attribute and each instance of the mesh.
    remaining_implicit_rows = MAX_IMPLICIT_ACCESSOR_ROWS

    def read_accessor(index):
        nonlocal remaining_implicit_rows
        acc = doc["accessors"][index]
        if "bufferView" not in acc:
            rows = int(acc["count"])
            if rows > remaining_implicit_rows:
                raise PedAssembleError(f"{path}: accessors[{index}] would exceed the per-file limit of "
                                       f"{MAX_IMPLICIT_ACCESSOR_ROWS:,} implicit accessor rows")
            remaining_implicit_rows -= rows
        return _read_glb_accessor(doc, bin_chunk, index)

    mesh_nodes = [(n, n["mesh"]) for n in nodes if "mesh" in n]
    if not mesh_nodes:
        mesh_nodes = [({"name": m.get("name", f"{stem}_{i}")}, i) for i, m in enumerate(meshes)]

    for node, mesh_idx in mesh_nodes:
        mesh = meshes[mesh_idx]
        node_name = node.get("name") or mesh.get("name") or stem
        trans = tuple(float(v) for v in node.get("translation", (0.0, 0.0, 0.0)))
        has_node_skin = "skin" in node and node["skin"] is not None
        node_joint_names = (
            _extract_glb_skin_joint_names(doc, node["skin"], path=path, part_name=node_name)
            if has_node_skin
            else None
        )
        for pi, prim in enumerate(mesh.get("primitives", [])):
            attrs = prim.get("attributes", {})
            if "POSITION" not in attrs:
                continue
            positions = [tuple(float(x) for x in p) for p in read_accessor(attrs["POSITION"])]
            if "indices" in prim:
                raw_idx = [int(i) for i in read_accessor(prim["indices"])]
            else:
                raw_idx = list(range(len(positions)))
            triangles = _normalize_triangles(raw_idx)
            normals = (
                [tuple(float(x) for x in n) for n in read_accessor(attrs["NORMAL"])]
                if "NORMAL" in attrs else None
            )
            uvs = (
                [tuple(float(x) for x in u) for u in read_accessor(attrs["TEXCOORD_0"])]
                if "TEXCOORD_0" in attrs else None
            )
            uvs_extra = {
                int(semantic[9:]): [tuple(float(x) for x in u)
                                   for u in read_accessor(accessor)]
                for semantic, accessor in attrs.items()
                if re.fullmatch(r"TEXCOORD_[1-9][0-9]{0,8}", semantic)
            }
            colors = None
            if "COLOR_0" in attrs:
                raw_c = read_accessor(attrs["COLOR_0"])
                colors = [
                    (float(c[0]), float(c[1]), float(c[2]), float(c[3]) if len(c) > 3 else 1.0)
                    for c in raw_c
                ]
            joints_0 = (
                [tuple(int(x) for x in j) for j in read_accessor(attrs["JOINTS_0"])]
                if "JOINTS_0" in attrs else None
            )
            weights_0 = (
                [tuple(float(x) for x in w) for w in read_accessor(attrs["WEIGHTS_0"])]
                if "WEIGHTS_0" in attrs else None
            )
            joints_1 = (
                [tuple(int(x) for x in j) for j in read_accessor(attrs["JOINTS_1"])]
                if "JOINTS_1" in attrs else None
            )
            weights_1 = (
                [tuple(float(x) for x in w) for w in read_accessor(attrs["WEIGHTS_1"])]
                if "WEIGHTS_1" in attrs else None
            )
            pname = node_name if len(mesh.get("primitives", [])) == 1 else f"{node_name}_p{pi}"
            has_skin_attrs = joints_0 is not None or joints_1 is not None
            if has_node_skin:
                part_joint_names = list(node_joint_names) if (has_skin_attrs and node_joint_names is not None) else None
            elif has_skin_attrs:
                if len(skins) == 1:
                    part_joint_names = _extract_glb_skin_joint_names(
                        doc,
                        0,
                        path=path,
                        part_name=pname,
                    )
                elif len(skins) > 1:
                    raise PedAssembleError(
                        f"{path}: skinned part {pname!r} has no node 'skin' index "
                        f"and GLB defines {len(skins)} skins (ambiguous skin binding)"
                    )
                else:
                    part_joint_names = []
            else:
                part_joint_names = None

            mat_spec = None
            if "material" not in prim or 0 <= prim["material"] < len(materials):
                gm = materials[prim["material"]] if "material" in prim else {"name": f"{pname}_mat"}
                pbr = gm.get("pbrMetallicRoughness", {})
                # glTF defaults also apply to primitives with no material; other input formats keep their clay defaults.
                bcf = pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])
                mat_spec = MaterialSpec(
                    name=str(gm.get("name", f"{node_name}_mat")),
                    base_color_factor=(float(bcf[0]), float(bcf[1]), float(bcf[2]), float(bcf[3])),
                    metallic_factor=float(pbr.get("metallicFactor", 1.0)),
                    roughness_factor=float(pbr.get("roughnessFactor", 1.0)),
                    alpha_mode=str(gm.get("alphaMode", "OPAQUE")),
                    alpha_cutoff=float(gm["alphaCutoff"]) if "alphaCutoff" in gm else None,
                    double_sided=bool(gm.get("doubleSided", False)),
                )
            parts.append(MeshPart(
                name=pname,
                positions=positions,
                triangles=triangles,
                coord_space="y-up",
                translation=trans,
                normals=normals,
                uvs=uvs,
                uvs_extra=uvs_extra,
                colors=colors,
                joints_0=joints_0,
                weights_0=weights_0,
                joints_1=joints_1,
                weights_1=weights_1,
                joint_names=part_joint_names,
                material=mat_spec,
            ))
    return parts


def load_parts_from_path(
    path: str,
    *,
    source_space: str = "z-up",
) -> tuple[list[MeshPart], SkeletonSpec | None]:
    """Load one or more ``MeshPart`` objects from a ``.json``, ``.obj``, or ``.glb`` file."""
    ext = os.path.splitext(path)[1].lower()
    stem = os.path.splitext(os.path.basename(path))[0]
    base_dir = os.path.dirname(os.path.abspath(path))

    if ext == ".glb":
        return read_glb_parts(path), None

    if ext == ".obj":
        # Parse OBJ keeping raw coordinates in the declared source_space
        prims = glb_writer.parse_obj(path, up_axis="y")
        parts: list[MeshPart] = []
        for idx, prim in enumerate(prims):
            pname = prim.get("name") or (stem if len(prims) == 1 else f"{stem}_{idx}")
            parts.append(MeshPart(
                name=pname,
                positions=[tuple(p) for p in prim["positions"]],
                triangles=_normalize_triangles(prim["indices"]),
                coord_space=source_space,
                normals=[tuple(n) for n in prim["normals"]] if prim.get("normals") else None,
                uvs=[tuple(u) for u in prim["uvs"]] if prim.get("uvs") else None,
            ))
        return parts, None

    if ext == ".json":
        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)
        if isinstance(data, dict) and "parts" in data:
            skel = _skeleton_from_dict(data.get("skeleton"), default_space=data.get("coord_space", source_space))
            default_sp = str(data.get("coord_space", source_space))
            parts = []
            for idx, entry in enumerate(data["parts"]):
                if isinstance(entry, str):
                    sub_path = entry if os.path.isabs(entry) else os.path.join(base_dir, entry)
                    sub_parts, _ = load_parts_from_path(sub_path, source_space=default_sp)
                    parts.extend(sub_parts)
                elif isinstance(entry, dict) and "path" in entry and "positions" not in entry:
                    sub_path = entry["path"] if os.path.isabs(entry["path"]) else os.path.join(base_dir, entry["path"])
                    sub_parts, _ = load_parts_from_path(sub_path, source_space=entry.get("coord_space", default_sp))
                    for sp in sub_parts:
                        if entry.get("name"):
                            sp.name = str(entry["name"])
                        if entry.get("role"):
                            sp.role = str(entry["role"])
                        if "joint_names" in entry or "jointNames" in entry:
                            raw_jn = entry.get("joint_names", entry.get("jointNames"))
                            sp.joint_names = _parse_joint_names(raw_jn, part_name=sp.name)
                        parts.append(sp)
                elif isinstance(entry, dict):
                    parts.append(_part_from_dict(entry, default_name=f"{stem}_{idx}", default_space=default_sp, base_dir=base_dir))
                else:
                    raise PedAssembleError(f"{path}: unsupported part entry at index {idx}")
            return parts, skel
        if isinstance(data, dict):
            return [_part_from_dict(data, default_name=stem, default_space=source_space, base_dir=base_dir)], None
        raise PedAssembleError(f"{path}: JSON root must be an object")

    raise PedAssembleError(f"{path}: unsupported input extension {ext!r} (expected .json, .obj, or .glb)")


# --------------------------------------------------------------------------- GLB assembly writer

def _pad4(buf: bytearray, pad_byte: int = 0) -> None:
    rem = len(buf) & 3
    if rem:
        buf.extend(bytes([pad_byte]) * (4 - rem))


def _append_bv(
    bin_buf: bytearray,
    buffer_views: list[dict],
    payload: bytes,
    target: int | None = None,
) -> int:
    _pad4(bin_buf, 0)
    offset = len(bin_buf)
    bin_buf.extend(payload)
    bv: dict = {"buffer": 0, "byteOffset": offset, "byteLength": len(payload)}
    if target is not None:
        bv["target"] = target
    buffer_views.append(bv)
    return len(buffer_views) - 1


def _append_accessor(
    accessors: list[dict],
    buffer_view: int,
    component_type: int,
    count: int,
    acc_type: str,
    *,
    min_vals: list[float] | None = None,
    max_vals: list[float] | None = None,
    normalized: bool = False,
) -> int:
    acc: dict = {
        "bufferView": buffer_view,
        "componentType": component_type,
        "count": count,
        "type": acc_type,
    }
    if normalized:
        acc["normalized"] = True
    if min_vals is not None:
        acc["min"] = min_vals
    if max_vals is not None:
        acc["max"] = max_vals
    accessors.append(acc)
    return len(accessors) - 1


def _pack_u16_vec4(rows: list[tuple[int, int, int, int]]) -> bytes:
    buf = bytearray(len(rows) * 8)
    for i, (a, b, c, d) in enumerate(rows):
        struct.pack_into("<4H", buf, i * 8, a, b, c, d)
    return bytes(buf)


def _pack_f32_vec(rows: list[tuple[float, ...]], comps: int) -> bytes:
    fmt = f"<{comps}f"
    stride = comps * 4
    buf = bytearray(len(rows) * stride)
    for i, row in enumerate(rows):
        struct.pack_into(fmt, buf, i * stride, *row[:comps])
    return bytes(buf)


def remap_part_joints(
    part: MeshPart,
    skeleton: SkeletonSpec,
) -> tuple[list[tuple[int, int, int, int]] | None, list[tuple[int, int, int, int]] | None]:
    """Validate and remap ``part``'s skinning joint indices into ``skeleton.joints`` slot order.

    When ``part.joint_names`` is present (for example from a skinned ``.glb`` input
    or a manifest specifying source joint names), every non-zero-weighted influence
    slot is resolved through ``part.joint_names[src_idx]`` and mapped to the unique
    matching joint name in ``skeleton.joints``. Missing, unnamed, out-of-range, or
    ambiguous (duplicate-name) correspondences raise :class:`PedAssembleError`
    (a :class:`ValueError` subclass). Zero-weight padding slots are remapped when
    unambiguous and otherwise reset to ``0``.

    When ``part.joint_names`` is ``None`` (raw inputs authored directly in the
    target skeleton's index space), every non-zero-weighted joint index is validated
    to lie within ``[0, len(skeleton.joints))``.
    """
    if part.joints_0 is None and part.weights_0 is None and part.joints_1 is None and part.weights_1 is None:
        return None, None
    if (part.joints_0 is None) != (part.weights_0 is None):
        raise PedAssembleError(f"{part.name}: joints_0 and weights_0 must both be provided")
    if (part.joints_1 is None) != (part.weights_1 is None):
        raise PedAssembleError(f"{part.name}: joints_1 and weights_1 must both be provided")
    if part.joints_1 is not None and part.joints_0 is None:
        raise PedAssembleError(f"{part.name}: joints_1/weights_1 provided without joints_0/weights_0")

    n_verts = len(part.positions)
    if part.joints_0 is not None and (len(part.joints_0) != n_verts or len(part.weights_0) != n_verts):
        raise PedAssembleError(
            f"{part.name}: joints_0 ({len(part.joints_0)}) / weights_0 ({len(part.weights_0)}) "
            f"must match vertex count ({n_verts})"
        )
    if part.joints_1 is not None and (len(part.joints_1) != n_verts or len(part.weights_1) != n_verts):
        raise PedAssembleError(
            f"{part.name}: joints_1 ({len(part.joints_1)}) / weights_1 ({len(part.weights_1)}) "
            f"must match vertex count ({n_verts})"
        )

    n_dst = len(skeleton.joints)
    if n_dst == 0:
        raise PedAssembleError(
            f"{part.name}: target skeleton {skeleton.name!r} has no joints for skinned mesh"
        )
    if n_dst > 65536:
        raise PedAssembleError(
            f"{part.name}: target skeleton {skeleton.name!r} has {n_dst} joints (exceeds u16 JOINTS_n limit)"
        )

    slot_map: list[tuple[bool, int | str]] | None = None
    if part.joint_names is not None:
        src_by_name: dict[str, list[int]] = defaultdict(list)
        for src_idx, sname in enumerate(part.joint_names):
            if sname and sname.strip():
                src_by_name[sname].append(src_idx)

        dst_by_name: dict[str, list[int]] = defaultdict(list)
        for dst_idx, joint in enumerate(skeleton.joints):
            if joint.name and joint.name.strip():
                dst_by_name[joint.name].append(dst_idx)

        slot_map = []
        for src_idx, sname in enumerate(part.joint_names):
            if not sname or not sname.strip():
                slot_map.append((
                    False,
                    f"source joint slot {src_idx} is unnamed and cannot be mapped to skeleton {skeleton.name!r}",
                ))
                continue
            src_slots = src_by_name[sname]
            if len(src_slots) > 1:
                slot_map.append((
                    False,
                    f"source joint name {sname!r} is ambiguous in source skin (slots {src_slots})",
                ))
                continue
            dst_slots = dst_by_name.get(sname)
            if not dst_slots:
                slot_map.append((
                    False,
                    f"source joint {sname!r} (slot {src_idx}) not found in target skeleton {skeleton.name!r}",
                ))
                continue
            if len(dst_slots) > 1:
                slot_map.append((
                    False,
                    f"target joint name {sname!r} is ambiguous in skeleton {skeleton.name!r} (slots {dst_slots})",
                ))
                continue
            slot_map.append((True, dst_slots[0]))

    def _remap_attr(
        joints: list[tuple[int, int, int, int]] | None,
        weights: list[tuple[float, float, float, float]] | None,
        attr_name: str,
    ) -> list[tuple[int, int, int, int]] | None:
        if joints is None or weights is None:
            return None
        remapped: list[tuple[int, int, int, int]] = []
        for vi, (j_row, w_row) in enumerate(zip(joints, weights)):
            if len(j_row) != 4 or len(w_row) != 4:
                raise PedAssembleError(
                    f"{part.name}: {attr_name} vertex {vi} must have 4 joint/weight components"
                )
            out_row: list[int] = []
            for comp, (src_idx, weight) in enumerate(zip(j_row, w_row)):
                active = weight != 0.0
                if slot_map is not None:
                    if active:
                        if not (0 <= src_idx < len(slot_map)):
                            if len(slot_map) == 0:
                                raise PedAssembleError(
                                    f"{part.name}: {attr_name} vertex {vi}[{comp}] has weight {weight} "
                                    f"on slot {src_idx}, but source part has no skin joints to map to "
                                    f"skeleton {skeleton.name!r}"
                                )
                            raise PedAssembleError(
                                f"{part.name}: {attr_name} vertex {vi}[{comp}] references out-of-range "
                                f"source joint slot {src_idx} (source skin has {len(slot_map)} joint(s))"
                            )
                        ok, payload = slot_map[src_idx]
                        if not ok:
                            raise PedAssembleError(
                                f"{part.name}: {attr_name} vertex {vi}[{comp}] (weight={weight}): {payload}"
                            )
                        out_row.append(int(payload))
                    else:
                        if 0 <= src_idx < len(slot_map) and slot_map[src_idx][0]:
                            out_row.append(int(slot_map[src_idx][1]))
                        else:
                            out_row.append(0)
                else:
                    if active:
                        if not (0 <= src_idx < n_dst):
                            raise PedAssembleError(
                                f"{part.name}: {attr_name} vertex {vi}[{comp}] references out-of-range "
                                f"joint index {src_idx} for skeleton {skeleton.name!r} ({n_dst} joint(s))"
                            )
                        out_row.append(src_idx)
                    else:
                        out_row.append(src_idx if 0 <= src_idx < n_dst else 0)
            remapped.append((out_row[0], out_row[1], out_row[2], out_row[3]))
        return remapped

    new_j0 = _remap_attr(part.joints_0, part.weights_0, "JOINTS_0")
    new_j1 = _remap_attr(part.joints_1, part.weights_1, "JOINTS_1")
    return new_j0, new_j1


def build_assembled_glb(
    parts: list[MeshPart],
    *,
    skeleton: SkeletonSpec | None = None,
    standing_offset: float = DEFAULT_STANDING_OFFSET_M,
    apply_standing: bool = True,
    generator: str = "Leonida ped_assemble.py",
) -> tuple[bytes, dict]:
    """Serialize assembled modular ped ``parts`` (and optional ``skeleton``) to a glTF 2.0 GLB.

    All mesh geometry is converted into glTF Y-up standing space. When
    ``skeleton`` is provided, joint nodes, inverse bind matrices, and a
    standing-space root wrapper node are emitted and bound to every skinned
    mesh node. Skinned parts carrying ``joint_names`` have their ``JOINTS_0``
    and ``JOINTS_1`` slots remapped by joint name into ``skeleton.joints`` order.
    """
    non_empty = [p for p in parts if p.positions and p.triangles]
    if not non_empty:
        raise PedAssembleError("No non-empty mesh parts to export")

    bin_buf = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    images: list[dict] = []
    textures: list[dict] = []
    samplers: list[dict] = []
    materials: list[dict] = []
    meshes: list[dict] = []
    nodes: list[dict] = []
    scene_nodes: list[int] = []

    png_cache: dict[bytes, int] = {}

    def _add_png_texture(png_bytes: bytes) -> int:
        if png_bytes in png_cache:
            return png_cache[png_bytes]
        if not samplers:
            samplers.append({"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497})
        bv = _append_bv(bin_buf, buffer_views, png_bytes, target=None)
        img_idx = len(images)
        images.append({"bufferView": bv, "mimeType": "image/png"})
        tex_idx = len(textures)
        textures.append({"sampler": 0, "source": img_idx})
        png_cache[png_bytes] = tex_idx
        return tex_idx

    def _add_material(spec: MaterialSpec | None, fallback_name: str) -> int:
        ms = spec or MaterialSpec(name=fallback_name)
        pbr: dict = {
            "baseColorFactor": [float(v) for v in ms.base_color_factor],
            "metallicFactor": float(ms.metallic_factor),
            "roughnessFactor": float(ms.roughness_factor),
        }
        if ms.base_color_png is not None:
            pbr["baseColorTexture"] = {"index": _add_png_texture(ms.base_color_png)}
        if ms.metallic_roughness_png is not None:
            pbr["metallicRoughnessTexture"] = {"index": _add_png_texture(ms.metallic_roughness_png)}
        mat_obj: dict = {"name": ms.name, "pbrMetallicRoughness": pbr}
        if ms.normal_png is not None:
            mat_obj["normalTexture"] = {"index": _add_png_texture(ms.normal_png)}
        if ms.alpha_mode and ms.alpha_mode != "OPAQUE":
            mat_obj["alphaMode"] = ms.alpha_mode
            if ms.alpha_mode == "MASK" and ms.alpha_cutoff is not None:
                mat_obj["alphaCutoff"] = float(ms.alpha_cutoff)
        if ms.double_sided:
            mat_obj["doubleSided"] = True
        materials.append(mat_obj)
        return len(materials) - 1

    skinned_node_indices: list[int] = []
    total_verts = 0
    total_tris = 0

    for part in non_empty:
        is_skinned_part = part.joints_0 is not None and part.weights_0 is not None and skeleton is not None
        # For skinned parts with a skeleton, standing offset lives on the skeleton root node
        # (so bind-space vertices match IBMs cleanly); for static parts, bake standing offset into vertices.
        aligned = align_part_to_standing_space(
            part,
            standing_offset=standing_offset,
            target_space="y-up",
            apply_offset=(apply_standing and not is_skinned_part),
        )
        n_verts = len(aligned.positions)
        flat_indices = [idx for tri in aligned.triangles for idx in tri]
        if aligned.normals is not None and len(aligned.normals) == n_verts:
            normals = aligned.normals
        else:
            normals = glb_writer._compute_smooth_normals(aligned.positions, flat_indices)

        pos_bytes = _pack_f32_vec(aligned.positions, 3)
        pos_bv = _append_bv(bin_buf, buffer_views, pos_bytes, target=TARGET_ARRAY_BUFFER)
        xs = [p[0] for p in aligned.positions]
        ys = [p[1] for p in aligned.positions]
        zs = [p[2] for p in aligned.positions]
        pos_acc = _append_accessor(
            accessors, pos_bv, COMPONENT_FLOAT, n_verts, "VEC3",
            min_vals=[min(xs), min(ys), min(zs)],
            max_vals=[max(xs), max(ys), max(zs)],
        )

        nrm_bytes = _pack_f32_vec(normals, 3)
        nrm_bv = _append_bv(bin_buf, buffer_views, nrm_bytes, target=TARGET_ARRAY_BUFFER)
        nrm_acc = _append_accessor(accessors, nrm_bv, COMPONENT_FLOAT, n_verts, "VEC3")

        attrs: dict[str, int] = {"POSITION": pos_acc, "NORMAL": nrm_acc}

        if aligned.uvs is not None and len(aligned.uvs) == n_verts:
            uv_bytes = _pack_f32_vec(aligned.uvs, 2)
            uv_bv = _append_bv(bin_buf, buffer_views, uv_bytes, target=TARGET_ARRAY_BUFFER)
            attrs["TEXCOORD_0"] = _append_accessor(accessors, uv_bv, COMPONENT_FLOAT, n_verts, "VEC2")

        if any(type(key) is not int or not 1 <= key <= 999999999 for key in aligned.uvs_extra):
            raise PedAssembleError(f"{part.name}: uvs_extra keys must be positive texture coordinate set indices")
        for key, values in sorted(aligned.uvs_extra.items()):
            if f"TEXCOORD_{key - 1}" not in attrs:
                raise PedAssembleError(f"{part.name}: texture coordinate sets must start at 0 without gaps")
            if len(values) != n_verts or any(len(u) != 2 for u in values):
                raise PedAssembleError(f"{part.name}: TEXCOORD_{key} must contain one VEC2 per vertex")
            uv_bv = _append_bv(bin_buf, buffer_views, _pack_f32_vec(values, 2), target=TARGET_ARRAY_BUFFER)
            attrs[f"TEXCOORD_{key}"] = _append_accessor(accessors, uv_bv, COMPONENT_FLOAT, n_verts, "VEC2")

        if aligned.colors is not None and len(aligned.colors) == n_verts:
            col_bytes = _pack_f32_vec(aligned.colors, 4)
            col_bv = _append_bv(bin_buf, buffer_views, col_bytes, target=TARGET_ARRAY_BUFFER)
            attrs["COLOR_0"] = _append_accessor(accessors, col_bv, COMPONENT_FLOAT, n_verts, "VEC4")

        joints_0_out = aligned.joints_0
        joints_1_out = aligned.joints_1
        has_any_skin = (
            aligned.joints_0 is not None
            or aligned.weights_0 is not None
            or aligned.joints_1 is not None
            or aligned.weights_1 is not None
        )
        if has_any_skin and skeleton is not None:
            joints_0_out, joints_1_out = remap_part_joints(aligned, skeleton)

        if joints_0_out is not None and aligned.weights_0 is not None:
            j0_bv = _append_bv(bin_buf, buffer_views, _pack_u16_vec4(joints_0_out), target=TARGET_ARRAY_BUFFER)
            w0_bv = _append_bv(bin_buf, buffer_views, _pack_f32_vec(aligned.weights_0, 4), target=TARGET_ARRAY_BUFFER)
            attrs["JOINTS_0"] = _append_accessor(accessors, j0_bv, COMPONENT_UNSIGNED_SHORT, n_verts, "VEC4")
            attrs["WEIGHTS_0"] = _append_accessor(accessors, w0_bv, COMPONENT_FLOAT, n_verts, "VEC4")

        if joints_1_out is not None and aligned.weights_1 is not None:
            j1_bv = _append_bv(bin_buf, buffer_views, _pack_u16_vec4(joints_1_out), target=TARGET_ARRAY_BUFFER)
            w1_bv = _append_bv(bin_buf, buffer_views, _pack_f32_vec(aligned.weights_1, 4), target=TARGET_ARRAY_BUFFER)
            attrs["JOINTS_1"] = _append_accessor(accessors, j1_bv, COMPONENT_UNSIGNED_SHORT, n_verts, "VEC4")
            attrs["WEIGHTS_1"] = _append_accessor(accessors, w1_bv, COMPONENT_FLOAT, n_verts, "VEC4")

        idx_bytes, idx_ctype = glb_writer._pack_indices(flat_indices, n_verts)
        idx_bv = _append_bv(bin_buf, buffer_views, idx_bytes, target=TARGET_ELEMENT_ARRAY_BUFFER)
        idx_acc = _append_accessor(accessors, idx_bv, idx_ctype, len(flat_indices), "SCALAR")

        mat_idx = _add_material(aligned.material, f"{aligned.name}_mat")
        mesh_idx = len(meshes)
        meshes.append({
            "name": aligned.name,
            "primitives": [{
                "attributes": attrs,
                "indices": idx_acc,
                "material": mat_idx,
                "mode": 4,
            }],
        })

        node_idx = len(nodes)
        node_obj: dict = {"name": aligned.name, "mesh": mesh_idx}
        nodes.append(node_obj)
        scene_nodes.append(node_idx)
        if is_skinned_part:
            skinned_node_indices.append(node_idx)

        total_verts += n_verts
        total_tris += len(aligned.triangles)

    skins: list[dict] = []
    if skeleton is not None and skeleton.joints:
        joint_node_indices: list[int] = []
        root_joint_nodes: list[int] = []
        ibm_floats: list[float] = []
        base_idx = len(nodes)
        skel_zup = skeleton.coord_space == "z-up"

        for ji, j in enumerate(skeleton.joints):
            t = rage_to_yup_vec3(j.translation) if skel_zup else j.translation
            r = rage_to_yup_quat(j.rotation) if skel_zup else j.rotation
            s = (j.scale[0], j.scale[2], j.scale[1]) if skel_zup else j.scale
            jnode: dict = {"name": j.name}
            if any(abs(v) > 1e-7 for v in t):
                jnode["translation"] = [round(v, 7) for v in t]
            if any(abs(v - d0) > 1e-7 for v, d0 in zip(r, (0.0, 0.0, 0.0, 1.0))):
                jnode["rotation"] = [round(v, 7) for v in r]
            if any(abs(v - 1.0) > 1e-6 for v in s):
                jnode["scale"] = [round(v, 7) for v in s]
            nodes.append(jnode)
            joint_node_indices.append(base_idx + ji)

            if j.inverse_bind_matrix is not None:
                ibm = rage_to_yup_mat4_col(j.inverse_bind_matrix) if skel_zup else j.inverse_bind_matrix
            else:
                ibm = _identity_mat4_col()
            ibm_floats.extend(ibm)

        for ji, j in enumerate(skeleton.joints):
            nidx = joint_node_indices[ji]
            if 0 <= j.parent < len(skeleton.joints):
                nodes[joint_node_indices[j.parent]].setdefault("children", []).append(nidx)
            else:
                root_joint_nodes.append(nidx)

        skel_off = standing_offset if skeleton.standing_offset is None else skeleton.standing_offset
        rig_root_node: dict = {"name": f"{skeleton.name}_root", "children": root_joint_nodes}
        if apply_standing and abs(skel_off) > 1e-7:
            rig_root_node["translation"] = [0.0, float(skel_off), 0.0]
        rig_root_idx = len(nodes)
        nodes.append(rig_root_node)
        scene_nodes.append(rig_root_idx)

        ibm_bytes = struct.pack(f"<{len(ibm_floats)}f", *ibm_floats)
        ibm_bv = _append_bv(bin_buf, buffer_views, ibm_bytes, target=None)
        ibm_acc = _append_accessor(accessors, ibm_bv, COMPONENT_FLOAT, len(skeleton.joints), "MAT4")

        skins.append({
            "name": skeleton.name,
            "joints": joint_node_indices,
            "skeleton": rig_root_idx,
            "inverseBindMatrices": ibm_acc,
        })
        for sni in skinned_node_indices:
            nodes[sni]["skin"] = 0

    _pad4(bin_buf, 0)
    gltf: dict = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_buf)}],
    }
    if images:
        gltf["images"] = images
        gltf["textures"] = textures
        gltf["samplers"] = samplers
    if skins:
        gltf["skins"] = skins

    json_bytes = bytearray(json.dumps(gltf, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    _pad4(json_bytes, 0x20)

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    header = struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total_length)
    json_hdr = struct.pack("<II", len(json_bytes), CHUNK_TYPE_JSON)
    bin_hdr = struct.pack("<II", len(bin_buf), CHUNK_TYPE_BIN)
    glb_bytes = header + json_hdr + bytes(json_bytes) + bin_hdr + bytes(bin_buf)

    stats = {
        "parts": len(non_empty),
        "vertices": total_verts,
        "triangles": total_tris,
        "materials": len(materials),
        "textures": len(textures),
        "skins": len(skins),
        "joints": len(skeleton.joints) if skeleton else 0,
        "bytes": len(glb_bytes),
    }
    return glb_bytes, stats


def write_assembled_glb(
    path: str,
    parts: list[MeshPart],
    *,
    skeleton: SkeletonSpec | None = None,
    standing_offset: float = DEFAULT_STANDING_OFFSET_M,
    apply_standing: bool = True,
) -> dict:
    """Build and atomically write an assembled modular ped GLB to ``path``."""
    glb_bytes, stats = build_assembled_glb(
        parts,
        skeleton=skeleton,
        standing_offset=standing_offset,
        apply_standing=apply_standing,
    )
    glb_writer._write_bytes_atomic(path, glb_bytes)
    return stats


# --------------------------------------------------------------------------- CLI

def _load_cli_parts(
    inputs: list[str],
    manifest: str | None,
    source_space: str,
) -> tuple[list[MeshPart], SkeletonSpec | None]:
    parts: list[MeshPart] = []
    skeleton: SkeletonSpec | None = None
    paths = ([manifest] if manifest else []) + list(inputs)
    if not paths:
        raise PedAssembleError("Provide at least one input part or --manifest")
    for p in paths:
        loaded, skel = load_parts_from_path(p, source_space=source_space)
        parts.extend(loaded)
        if skel is not None and skeleton is None:
            skeleton = skel
    return parts, skeleton


def _write_json_report(path: str, data: dict) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def _cmd_assemble(args: argparse.Namespace) -> int:
    parts, skeleton = _load_cli_parts(args.inputs, args.manifest, args.source_space)
    # Normalize all parts into a common coordinate space before overlap culling
    common_space = args.source_space
    aligned_for_cull = [
        align_part_to_standing_space(
            p,
            standing_offset=0.0,
            target_space=common_space,
            apply_offset=False,
        )
        for p in parts
    ]
    before_seams = inspect_seams(
        aligned_for_cull,
        weld_tol=args.weld_tol,
        overlap_tol=args.waist_tol,
        plane_tol=args.plane_tol,
        normal_cos_min=args.normal_cos,
    )

    cull_reports: list[dict] = []
    assembled_parts = aligned_for_cull
    if not args.no_cull:
        assembled_parts, cull_reports = apply_standard_ped_culling(
            aligned_for_cull,
            waist_tol=args.waist_tol,
            nipple_tol=args.nipple_tol,
            genital_tol=args.genital_tol,
            plane_tol=args.plane_tol,
            normal_cos_min=args.normal_cos,
            vertex_tol=args.vertex_tol,
            compact_vertices=args.compact_vertices,
        )

    after_seams = inspect_seams(
        assembled_parts,
        weld_tol=args.weld_tol,
        overlap_tol=args.waist_tol,
        plane_tol=args.plane_tol,
        normal_cos_min=args.normal_cos,
    )

    glb_stats = write_assembled_glb(
        args.out,
        assembled_parts,
        skeleton=skeleton,
        standing_offset=args.standing_offset,
        apply_standing=not args.no_standing_offset,
    )

    report = {
        "output": args.out,
        "standing_offset_m": 0.0 if args.no_standing_offset else args.standing_offset,
        "cull_passes": cull_reports,
        "before_seams": before_seams,
        "after_seams": after_seams,
        "glb": glb_stats,
    }
    if args.json:
        _write_json_report(args.json, report)

    total_culled = sum(r["culled_total"] for r in cull_reports)
    print(
        f"assembled {glb_stats['parts']} part(s) -> {args.out} "
        f"({glb_stats['vertices']} verts, {glb_stats['triangles']} tris, "
        f"culled {total_culled} overlap tris, "
        f"non_manifold {before_seams['combined_topology']['non_manifold_edges']} -> "
        f"{after_seams['combined_topology']['non_manifold_edges']})"
    )
    return 0


def _cmd_cull_overlap(args: argparse.Namespace) -> int:
    target_parts, _ = load_parts_from_path(args.target, source_space=args.source_space)
    if len(target_parts) != 1:
        raise PedAssembleError(f"--target must contain exactly 1 mesh part, got {len(target_parts)}")
    target = align_part_to_standing_space(
        target_parts[0], standing_offset=0.0, target_space=args.source_space, apply_offset=False
    )

    refs: list[MeshPart] = []
    for ref_path in args.ref:
        loaded, _ = load_parts_from_path(ref_path, source_space=args.source_space)
        for rp in loaded:
            refs.append(align_part_to_standing_space(
                rp, standing_offset=0.0, target_space=args.source_space, apply_offset=False
            ))

    culled_part, stats = cull_part_overlap(
        target,
        refs,
        centroid_tol=args.tol,
        plane_tol=args.plane_tol,
        normal_cos_min=args.normal_cos,
        vertex_tol=args.vertex_tol,
        compact_vertices=args.compact_vertices,
    )

    if args.out:
        out_ext = os.path.splitext(args.out)[1].lower()
        if out_ext == ".glb":
            write_assembled_glb(
                args.out,
                [culled_part],
                standing_offset=args.standing_offset,
                apply_standing=not args.no_standing_offset,
            )
        elif out_ext == ".json":
            _write_json_report(args.out, part_to_dict(culled_part))
        else:
            raise PedAssembleError(f"Unsupported output extension {out_ext!r}; use .glb or .json")

    if args.json:
        _write_json_report(args.json, stats)

    print(
        f"cull-overlap {target.name}: {stats['target_triangles_before']} -> "
        f"{stats['target_triangles_after']} tris "
        f"(culled {stats['culled_total']}: {stats['culled_exact_key']} exact, "
        f"{stats['culled_coplanar_proximity']} proximity)"
    )
    return 0


def _cmd_inspect_seams(args: argparse.Namespace) -> int:
    parts, _ = _load_cli_parts(args.inputs, args.manifest, args.source_space)
    normalized = [
        align_part_to_standing_space(p, standing_offset=0.0, target_space=args.source_space, apply_offset=False)
        for p in parts
    ]
    report = inspect_seams(
        normalized,
        weld_tol=args.weld_tol,
        overlap_tol=args.overlap_tol,
        plane_tol=args.plane_tol,
        normal_cos_min=args.normal_cos,
    )
    if args.json:
        _write_json_report(args.json, report)

    topo = report["combined_topology"]
    print(
        f"inspect-seams: {report['part_count']} part(s), "
        f"edges={topo['unique_edges']} (manifold={topo['manifold_edges']}, "
        f"boundary={topo['boundary_edges']}, non_manifold={topo['non_manifold_edges']}), "
        f"collapsed_triangles={topo['collapsed_triangles']}, "
        f"overlapping_pairs={len(report['pairwise_overlaps'])}"
    )
    for p in report["parts"]:
        print(
            f"  {p['name']} [{p['role']}]: {p['vertices']} v / {p['triangles']} t, "
            f"islands={p['physical_islands']} (welded={p['welded_islands']})"
        )
    for ov in report["pairwise_overlaps"]:
        print(
            f"  overlap {ov['part_a']} vs {ov['part_b']}: "
            f"{ov['overlapping_triangles_in_a']} tris "
            f"({ov['exact_key_hits']} exact, {ov['proximity_hits']} proximity)"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ped_assemble.py",
        description="Assemble modular RAGE 9 ped parts with standing-space alignment and coplanar overlap culling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_asm = sub.add_parser("assemble", help="Align, cull coplanar overlaps, and export a multi-part GLB.")
    p_asm.add_argument("inputs", nargs="*", help="Input part files (.json, .obj, or .glb).")
    p_asm.add_argument("--manifest", help="Optional JSON manifest listing parts and optional skeleton.")
    p_asm.add_argument("-o", "--out", required=True, help="Output binary glTF 2.0 (.glb) path.")
    p_asm.add_argument("--source-space", choices=("z-up", "y-up"), default="z-up",
                       help="Coordinate space of JSON/OBJ inputs (default: z-up).")
    p_asm.add_argument("--standing-offset", type=float, default=DEFAULT_STANDING_OFFSET_M,
                       help="Vertical SKEL_ROOT standing offset in meters (default: 1.0).")
    p_asm.add_argument("--no-standing-offset", action="store_true",
                       help="Do not add the +1.0 m standing-space offset.")
    p_asm.add_argument("--no-cull", action="store_true",
                       help="Skip coplanar overlap culling passes.")
    p_asm.add_argument("--waist-tol", type=float, default=DEFAULT_WAIST_CENTROID_TOL,
                       help="Centroid tolerance (m) for upper/lower waist culling (default: 0.004).")
    p_asm.add_argument("--nipple-tol", type=float, default=DEFAULT_NIPPLE_CENTROID_TOL,
                       help="Centroid tolerance (m) for upper/nipple culling (default: 0.0025).")
    p_asm.add_argument("--genital-tol", type=float, default=DEFAULT_GENITAL_CENTROID_TOL,
                       help="Centroid tolerance (m) for lower/female genital culling (default: 0.0016).")
    p_asm.add_argument("--plane-tol", type=float, default=None,
                       help="Optional explicit point-to-plane tolerance (m) for coplanarity check.")
    p_asm.add_argument("--normal-cos", type=float, default=DEFAULT_NORMAL_COS_MIN,
                       help="Minimum |dot(n_t, n_r)| for coplanar match (default: 0.85).")
    p_asm.add_argument("--vertex-tol", type=float, default=DEFAULT_VERTEX_QUANT_TOL,
                       help="Quantization tolerance (m) for exact vertex-key triangle match (default: 1e-4).")
    p_asm.add_argument("--weld-tol", type=float, default=DEFAULT_WELD_TOL,
                       help="Vertex weld tolerance (m) for seam topology inspection (default: 1e-4).")
    p_asm.add_argument("--compact-vertices", action="store_true",
                       help="Drop unreferenced vertices after triangle culling.")
    p_asm.add_argument("--json", help="Optional JSON report output path.")
    p_asm.set_defaults(func=_cmd_assemble)

    p_cull = sub.add_parser("cull-overlap", help="Cull coplanar triangles in a target part against reference parts.")
    p_cull.add_argument("--target", required=True, help="Target part file (.json, .obj, or .glb).")
    p_cull.add_argument("--ref", action="append", required=True, help="Reference part file (repeatable).")
    p_cull.add_argument("-o", "--out", help="Optional output path (.glb or .json).")
    p_cull.add_argument("--source-space", choices=("z-up", "y-up"), default="z-up",
                        help="Coordinate space of JSON/OBJ inputs (default: z-up).")
    p_cull.add_argument("--tol", type=float, default=DEFAULT_WAIST_CENTROID_TOL,
                        help="Centroid distance tolerance in meters (default: 0.004).")
    p_cull.add_argument("--plane-tol", type=float, default=None,
                        help="Optional explicit point-to-plane tolerance in meters.")
    p_cull.add_argument("--normal-cos", type=float, default=DEFAULT_NORMAL_COS_MIN,
                        help="Minimum |dot(n_t, n_r)| for coplanar match (default: 0.85).")
    p_cull.add_argument("--vertex-tol", type=float, default=DEFAULT_VERTEX_QUANT_TOL,
                        help="Quantization tolerance (m) for exact triangle match (default: 1e-4).")
    p_cull.add_argument("--standing-offset", type=float, default=DEFAULT_STANDING_OFFSET_M,
                        help="Standing offset (m) when writing .glb output (default: 1.0).")
    p_cull.add_argument("--no-standing-offset", action="store_true",
                        help="Do not add standing offset when writing .glb output.")
    p_cull.add_argument("--compact-vertices", action="store_true",
                        help="Drop unreferenced vertices after triangle culling.")
    p_cull.add_argument("--json", help="Optional JSON stats output path.")
    p_cull.set_defaults(func=_cmd_cull_overlap)

    p_seam = sub.add_parser("inspect-seams", help="Inspect island counts, boundary edges, and coplanar overlaps.")
    p_seam.add_argument("inputs", nargs="*", help="Input part files (.json, .obj, or .glb).")
    p_seam.add_argument("--manifest", help="Optional JSON manifest listing parts.")
    p_seam.add_argument("--source-space", choices=("z-up", "y-up"), default="z-up",
                        help="Coordinate space of JSON/OBJ inputs (default: z-up).")
    p_seam.add_argument("--weld-tol", type=float, default=DEFAULT_WELD_TOL,
                        help="Vertex weld tolerance in meters (default: 1e-4).")
    p_seam.add_argument("--overlap-tol", type=float, default=DEFAULT_WAIST_CENTROID_TOL,
                        help="Centroid tolerance (m) for overlap detection (default: 0.004).")
    p_seam.add_argument("--plane-tol", type=float, default=None,
                        help="Optional explicit point-to-plane tolerance in meters.")
    p_seam.add_argument("--normal-cos", type=float, default=DEFAULT_NORMAL_COS_MIN,
                        help="Minimum |dot(n_t, n_r)| for coplanar match (default: 0.85).")
    p_seam.add_argument("--json", help="Optional JSON report output path.")
    p_seam.set_defaults(func=_cmd_inspect_seams)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except PedAssembleError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
