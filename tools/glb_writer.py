"""Pure standard-library Wavefront OBJ parser and binary glTF 2.0 (.glb) writer.

Usage:
    python -I tools/glb_writer.py <in.obj> <out.glb> [--up z|y]

Reads a Wavefront OBJ file (positions, UVs, normals, faces, groups, objects, materials),
converts coordinates from RAGE Z-up to glTF Y-up (or keeps Y-up), computes smooth
area-weighted vertex normals when explicit normals are absent, and serializes a strict
glTF 2.0 binary (.glb) container with optional deduplicated embedded PNG textures.
Simple planar polygon faces may be convex or concave.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import itertools
import json
import math
import os
import struct
import sys
import tempfile
import time
from collections.abc import Iterable, Iterator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from file_mode import published_mode  # noqa: E402

GLB_MAGIC = b"glTF"
GLB_VERSION = 2
CHUNK_TYPE_JSON = 0x4E4F534A  # b"JSON"
CHUNK_TYPE_BIN = 0x004E4942   # b"BIN\x00"

TARGET_ARRAY_BUFFER = 34962
TARGET_ELEMENT_ARRAY_BUFFER = 34963

COMPONENT_UNSIGNED_SHORT = 5123
COMPONENT_UNSIGNED_INT = 5125
COMPONENT_FLOAT = 5126

MODE_TRIANGLES = 4

DEFAULT_CLAY_COLOR = [0.75, 0.75, 0.75, 1.0]
DEFAULT_TEXTURED_COLOR = [1.0, 1.0, 1.0, 1.0]
DEFAULT_METALLIC = 0.0
DEFAULT_ROUGHNESS = 0.85

ALPHA_MODES = ("OPAQUE", "MASK", "BLEND")

DEFAULT_SAMPLER = {
    "magFilter": 9729,  # LINEAR
    "minFilter": 9987,  # LINEAR_MIPMAP_LINEAR
    "wrapS": 10497,     # REPEAT
    "wrapT": 10497,     # REPEAT
}

_NEEDS_BYTESWAP = sys.byteorder != "little"
_UINT16_CODE = "H" if array.array("H").itemsize == 2 else None
_UINT32_CODE = "I" if array.array("I").itemsize == 4 else ("L" if array.array("L").itemsize == 4 else None)
_FLOAT32_CODE = "f" if array.array("f").itemsize == 4 else None


class GlbError(ValueError):
    """Raised when OBJ/primitive data or GLB binary validation fails."""


def _clean_zero(v: float) -> float:
    return 0.0 if v == 0.0 else v


def _normalize3(nx: float, ny: float, nz: float) -> tuple[float, float, float]:
    if not (math.isfinite(nx) and math.isfinite(ny) and math.isfinite(nz)):
        return (0.0, 1.0, 0.0)
    length = math.hypot(nx, ny, nz)
    if length == 0.0:
        return (0.0, 1.0, 0.0)
    if length < sys.float_info.min or not math.isfinite(length):
        # Scaling first keeps an unrepresentable or subnormal length from changing the direction.
        scale = max(abs(nx), abs(ny), abs(nz))
        nx, ny, nz = nx / scale, ny / scale, nz / scale
        length = math.hypot(nx, ny, nz)
    return (_clean_zero(nx / length), _clean_zero(ny / length), _clean_zero(nz / length))


def _compute_smooth_normals(
    positions: list[tuple[float, float, float]],
    indices: Iterable[int],
) -> list[tuple[float, float, float]]:
    """Compute area-weighted smooth vertex normals with (0, 1, 0) fallback."""
    n = len(positions)
    ax = [0.0] * n
    ay = [0.0] * n
    az = [0.0] * n

    index_iter = iter(indices)
    for i0, i1, i2 in zip(index_iter, index_iter, index_iter):
        p0 = positions[i0]
        p1 = positions[i1]
        p2 = positions[i2]
        e1x = p1[0] - p0[0]
        e1y = p1[1] - p0[1]
        e1z = p1[2] - p0[2]
        e2x = p2[0] - p0[0]
        e2y = p2[1] - p0[1]
        e2z = p2[2] - p0[2]
        # Cross product magnitude equals 2 * triangle area -> exact area weighting.
        cx = e1y * e2z - e1z * e2y
        cy = e1z * e2x - e1x * e2z
        cz = e1x * e2y - e1y * e2x
        if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(cz)):
            continue
        ax[i0] += cx
        ay[i0] += cy
        az[i0] += cz
        ax[i1] += cx
        ay[i1] += cy
        az[i1] += cz
        ax[i2] += cx
        ay[i2] += cy
        az[i2] += cz

    normals: list[tuple[float, float, float]] = []
    append = normals.append
    for i in range(n):
        append(_normalize3(ax[i], ay[i], az[i]))
    return normals


def _resolve_obj_index(token: str, count: int) -> int | None:
    if not token:
        return None
    try:
        idx = int(token)
    except ValueError:
        return None
    if idx > 0:
        resolved = idx - 1
    elif idx < 0:
        resolved = count + idx
    else:
        return None
    if 0 <= resolved < count:
        return resolved
    return None


def _triangulate_face(points: list[tuple[float, float, float]]) -> list[tuple[int, int, int]] | None:
    """Return corner triples when the first-corner fan folds over or emits a zero-area triangle.

    None keeps an already valid fan or falls back to it if ear clipping cannot complete.
    """
    # Project onto the plane with the largest signed area. Work relative to one
    # corner so translating the mesh far from the origin does not affect the sum.
    origin = points[0]
    rel = [(x - origin[0], y - origin[1], z - origin[2]) for x, y, z in points]
    edges = list(zip(rel, rel[1:] + rel[:1]))
    planes = ((1, 2), (2, 0), (0, 1))
    areas = [sum(a[u] * b[v] - a[v] * b[u] for a, b in edges) for u, v in planes]
    axis = max(range(3), key=lambda i: abs(areas[i]))
    area = areas[axis]
    if not area or not math.isfinite(area):
        return None
    u, v = planes[axis]
    winding = 1 if area > 0 else -1
    xy = [(p[u], p[v]) for p in rel]

    def turn(a, b, c):
        ax, ay = xy[a]
        bx, by = xy[b]
        cx, cy = xy[c]
        return winding * ((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))

    if all(turn(0, i, i + 1) > 0 for i in range(1, len(points) - 1)):
        return None

    # Clip ears in boundary order. A point on the ear's edge blocks it too:
    # that corner may carry its own UV or normal and still needs a triangle.
    ring = list(range(len(points)))
    triangles = []
    while len(ring) > 3:
        for i, b in enumerate(ring):
            a, c = ring[i - 1], ring[(i + 1) % len(ring)]
            if turn(a, b, c) <= 0:
                continue
            if any(turn(a, b, p) >= 0 and turn(b, c, p) >= 0 and turn(c, a, p) >= 0
                   for p in ring if p not in (a, b, c)):
                continue
            triangles.append((a, b, c))
            del ring[i]
            break
        else:
            return None
    triangles.append(tuple(ring))
    return triangles


def _logical_obj_lines(text: str) -> Iterator[str]:
    """Yield OBJ logical lines, joining trailing-backslash continuations with a space.

    A '#' starts an inline comment. A backslash inside that comment does not continue the line;
    a backslash in the code before the comment does.
    """
    pending = ""
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        code, _, _comment = line.partition("#")
        code = code.rstrip()
        if code.endswith("\\"):
            pending += code[:-1].rstrip() + " "
            continue
        yield pending + code
        pending = ""
    if pending:
        yield pending


def parse_obj(
    text_or_path: str,
    *,
    is_path: bool = True,
    up_axis: str = "z",
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
) -> list[dict]:
    """Parse Wavefront OBJ geometry into glTF-ready primitives.

    Groups primitives in order of appearance by (group_index, material_name),
    converts coordinates to glTF Y-up, flips V for top-left UV origin,
    deduplicates vertices per primitive, and synthesizes smooth area-weighted
    normals within the primitive and OBJ smoothing group when explicit normals are missing.
    """
    up = up_axis.lower()
    if up not in ("z", "y"):
        raise GlbError(f"unsupported up_axis: {up_axis!r} (expected 'z' or 'y')")
    if len(offset) != 3:
        raise GlbError(f"offset must have 3 components, got {len(offset)}")
    ox, oy, oz = float(offset[0]), float(offset[1]), float(offset[2])
    sc = float(scale)
    if not (math.isfinite(sc) and math.isfinite(ox) and math.isfinite(oy) and math.isfinite(oz)):
        raise GlbError("scale and offset must be finite numbers")

    if is_path:
        with open(text_or_path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
    else:
        text = text_or_path

    z_up = up == "z"
    mirrored = sc < 0
    raw_v: list[tuple[float, float, float] | None] = []
    raw_vt: list[tuple[float, float]] = []
    raw_vn: list[tuple[float, float, float]] = []

    raw_v_append = raw_v.append
    raw_vt_append = raw_vt.append
    raw_vn_append = raw_vn.append

    group_index = 0
    group_name = ""
    saw_g = False
    group_has_faces = False
    material_name = ""
    smoothing_group: int | None = None
    flat_face_id = 0

    # Preserve order of appearance for (group_index, material_name) primitives.
    builders: dict[tuple[int, str], dict] = {}

    for raw_line in _logical_obj_lines(text.removeprefix("\ufeff")):
        line = raw_line.strip()
        if not line or line[0] == "#":
            continue

        # Fast dispatch on the leading keyword.
        parts = line.split()
        cmd = parts[0]

        if cmd == "v":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 4:
                raw_v_append(None)
                continue
            try:
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
            except ValueError:
                raw_v_append(None)
                continue
            if z_up:
                px = x * sc + ox
                py = z * sc + oy
                pz = (-y) * sc + oz
            else:
                px = x * sc + ox
                py = y * sc + oy
                pz = z * sc + oz
            if math.isfinite(px) and math.isfinite(py) and math.isfinite(pz):
                raw_v_append((_clean_zero(px), _clean_zero(py), _clean_zero(pz)))
            else:
                raw_v_append(None)

        elif cmd == "vt":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 2:
                raw_vt_append((0.0, 0.0))
                continue
            try:
                u = float(parts[1])
                v = float(parts[2]) if len(parts) >= 3 else 0.0
            except ValueError:
                raw_vt_append((0.0, 0.0))
                continue
            v_gltf = 1.0 - v
            if not math.isfinite(u):
                u = 0.0
            if not math.isfinite(v_gltf):
                v_gltf = 0.0
            raw_vt_append((_clean_zero(u), _clean_zero(v_gltf)))

        elif cmd == "vn":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 4:
                raw_vn_append((0.0, 1.0, 0.0))
                continue
            try:
                nx = float(parts[1])
                ny = float(parts[2])
                nz = float(parts[3])
            except ValueError:
                raw_vn_append((0.0, 1.0, 0.0))
                continue
            if z_up:
                nx, ny, nz = nx, nz, -ny
            if mirrored:
                nx, ny, nz = -nx, -ny, -nz
            raw_vn_append(_normalize3(nx, ny, nz))

        elif cmd == "g":
            if saw_g or group_has_faces:
                group_index += 1
            saw_g = True
            group_has_faces = False
            group_name = line[len(cmd):].split("#", 1)[0].strip()

        elif cmd == "o":
            if group_has_faces:
                group_index += 1
            saw_g = False
            group_has_faces = False
            group_name = line[len(cmd):].split("#", 1)[0].strip()

        elif cmd == "usemtl":
            material_name = line[len(cmd):].split("#", 1)[0].strip()

        elif cmd == "s" and len(parts) > 1:
            value = parts[1].split("#", 1)[0]
            try:
                group = 0 if value == "off" else int(value)
            except ValueError:
                continue
            if group >= 0:
                smoothing_group = group

        elif cmd == "f":
            if "#" in line:
                parts = line.split("#", 1)[0].split()
            if len(parts) < 4:
                continue

            nv = len(raw_v)
            nvt = len(raw_vt)
            nvn = len(raw_vn)
            corners: list[tuple[int, int | None, int | None]] = []
            valid_face = True

            for tok in parts[1:]:
                slashes = tok.split("/")
                ns = len(slashes)
                if ns > 3:
                    valid_face = False
                    break

                vi = _resolve_obj_index(slashes[0], nv)
                if vi is None or raw_v[vi] is None:
                    valid_face = False
                    break

                vti: int | None = None
                if ns >= 2 and slashes[1]:
                    vti = _resolve_obj_index(slashes[1], nvt)
                    if vti is None:
                        valid_face = False
                        break

                vni: int | None = None
                if ns == 3 and slashes[2]:
                    vni = _resolve_obj_index(slashes[2], nvn)
                    if vni is None:
                        valid_face = False
                        break

                corners.append((vi, vti, vni))

            if not valid_face or len(corners) < 3:
                continue

            indices = ((0, 1, 2),)
            if len(corners) > 3:
                indices = None
                if len({c[0] for c in corners}) == len(corners):
                    indices = _triangulate_face([raw_v[c[0]] for c in corners])
                if indices is None:
                    indices = ((0, i, i + 1) for i in range(1, len(corners) - 1))
            tris = []
            for a, b, c in indices:
                c0, c1, c2 = corners[a], corners[b], corners[c]
                if c0[0] == c1[0] or c1[0] == c2[0] or c0[0] == c2[0]:
                    continue
                tris.append((c0, c2, c1) if mirrored else (c0, c1, c2))

            if not tris:
                continue

            group_has_faces = True
            key = (group_index, material_name)
            builder = builders.get(key)
            if builder is None:
                builder = {
                    "name": group_name,
                    "material": material_name,
                    "group_index": group_index,
                    "vert_map": {},
                    "positions": [],
                    "normals": [],
                    "uvs": [],
                    "indices": [],
                    "has_any_uv": False,
                    "has_missing_normal": False,
                }
                builders[key] = builder

            normal_group = None
            if smoothing_group is not None and any(vni is None for _, _, vni in corners):
                normal_group = smoothing_group
                if smoothing_group == 0:
                    flat_face_id -= 1
                    normal_group = flat_face_id

            vert_map: dict[tuple[int, int | None, int | None, int | None], int] = builder["vert_map"]
            b_pos: list[tuple[float, float, float]] = builder["positions"]
            b_nrm: list[tuple[float, float, float] | None] = builder["normals"]
            b_uv: list[tuple[float, float] | None] = builder["uvs"]
            b_idx: list[int] = builder["indices"]

            for tri in tris:
                for corner in tri:
                    vertex_key = (*corner, normal_group)
                    local_idx = vert_map.get(vertex_key)
                    if local_idx is None:
                        local_idx = len(b_pos)
                        vert_map[vertex_key] = local_idx
                        vi, vti, vni = corner
                        pos = raw_v[vi]
                        assert pos is not None
                        b_pos.append(pos)
                        if vti is not None:
                            builder["has_any_uv"] = True
                            b_uv.append(raw_vt[vti])
                        else:
                            b_uv.append(None)
                        if vni is not None:
                            b_nrm.append(raw_vn[vni])
                        else:
                            builder["has_missing_normal"] = True
                            b_nrm.append(None)
                    b_idx.append(local_idx)

    primitives: list[dict] = []
    for builder in builders.values():
        indices = builder["indices"]
        positions = builder["positions"]
        if not indices or not positions:
            continue

        if builder["has_missing_normal"]:
            # UVs and explicit normals split render vertices. Generated normals
            # share an accumulator for each position within the same smoothing group.
            first_vertex: dict[tuple[int, int | None], int] = {}
            normal_vertices = [
                first_vertex.setdefault((vi, normal_group), vertex)
                for (vi, _, _, normal_group), vertex in builder["vert_map"].items()
            ]
            raw_normals = builder["normals"]
            index_iter = iter(indices)
            # Fully explicit triangles should not smooth neighboring generated normals.
            normal_indices = (
                normal_vertices[i]
                for a, b, c in zip(index_iter, index_iter, index_iter)
                if None in (raw_normals[a], raw_normals[b], raw_normals[c])
                for i in (a, b, c)
            )
            computed = _compute_smooth_normals(positions, normal_indices)
            normals = [
                normal if normal is not None else computed[normal_vertex]
                for normal, normal_vertex in zip(raw_normals, normal_vertices)
            ]
        else:
            normals = builder["normals"]

        if builder["has_any_uv"]:
            uvs = [uv if uv is not None else (0.0, 0.0) for uv in builder["uvs"]]
        else:
            uvs = []

        primitives.append({
            "name": builder["name"],
            "material": builder["material"],
            "group_index": builder["group_index"],
            "positions": positions,
            "normals": normals,
            "uvs": uvs,
            "indices": indices,
        })

    return primitives


def compute_bounds(primitives: list[dict]) -> list[float] | None:
    """Return [minx, miny, minz, maxx, maxy, maxz] over all finite positions, or None."""
    minx = math.inf
    miny = math.inf
    minz = math.inf
    maxx = -math.inf
    maxy = -math.inf
    maxz = -math.inf
    found = False

    for prim in primitives:
        for pos in prim.get("positions") or ():
            if len(pos) < 3:
                continue
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
                continue
            found = True
            if x < minx:
                minx = x
            if y < miny:
                miny = y
            if z < minz:
                minz = z
            if x > maxx:
                maxx = x
            if y > maxy:
                maxy = y
            if z > maxz:
                maxz = z

    if not found:
        return None
    return [
        _clean_zero(minx),
        _clean_zero(miny),
        _clean_zero(minz),
        _clean_zero(maxx),
        _clean_zero(maxy),
        _clean_zero(maxz),
    ]


def _pack_f32(values: list[tuple[float, ...]], components: int) -> tuple[bytes, array.array]:
    if any(len(item) != components for item in values):
        raise GlbError(f"expected {components} floats per vertex")
    if _FLOAT32_CODE is not None:
        arr = array.array(_FLOAT32_CODE, itertools.chain.from_iterable(values))
        if not all(map(math.isfinite, arr)):
            raise GlbError("non-finite float32 vertex attribute")
        raw = arr
        if _NEEDS_BYTESWAP:
            swapped = array.array(_FLOAT32_CODE, arr)
            swapped.byteswap()
            return swapped.tobytes(), raw
        return raw.tobytes(), raw

    flat: list[float] = []
    for item in values:
        for c in item:
            v = float(c)
            if not math.isfinite(v):
                raise GlbError("non-finite float32 vertex attribute")
            flat.append(v)
    packed = struct.pack(f"<{len(flat)}f", *flat)
    unpacked = struct.unpack(f"<{len(flat)}f", packed)
    if not all(map(math.isfinite, unpacked)):
        raise GlbError("non-finite float32 vertex attribute")
    return packed, array.array("d", unpacked)


def _pack_indices(indices: list[int], vert_count: int) -> tuple[bytes, int]:
    if len(indices) % 3 != 0:
        raise GlbError(f"triangle index count must be a multiple of 3, got {len(indices)}")
    lo = min(indices)
    hi = max(indices)
    if lo < 0 or hi >= vert_count:
        raise GlbError(f"index out of range [0, {vert_count - 1}]: min={lo}, max={hi}")

    if vert_count <= 65535:
        comp_type = COMPONENT_UNSIGNED_SHORT
        if _UINT16_CODE is not None:
            arr = array.array(_UINT16_CODE, indices)
            if _NEEDS_BYTESWAP:
                arr.byteswap()
            return arr.tobytes(), comp_type
        return struct.pack(f"<{len(indices)}H", *indices), comp_type

    comp_type = COMPONENT_UNSIGNED_INT
    if _UINT32_CODE is not None:
        arr = array.array(_UINT32_CODE, indices)
        if _NEEDS_BYTESWAP:
            arr.byteswap()
        return arr.tobytes(), comp_type
    return struct.pack(f"<{len(indices)}I", *indices), comp_type


def _append_buffer_view(
    bin_buf: bytearray,
    buffer_views: list[dict],
    payload: bytes,
    *,
    target: int | None = None,
) -> int:
    pad = (-len(bin_buf)) % 4
    if pad:
        bin_buf.extend(b"\x00" * pad)
    offset = len(bin_buf)
    bin_buf.extend(payload)
    end_pad = (-len(bin_buf)) % 4
    if end_pad:
        bin_buf.extend(b"\x00" * end_pad)

    bv: dict = {
        "buffer": 0,
        "byteOffset": offset,
        "byteLength": len(payload),
    }
    if target is not None:
        bv["target"] = target
    bv_idx = len(buffer_views)
    buffer_views.append(bv)
    return bv_idx


def pack_primitive(prim: dict) -> dict:
    """Packed little-endian glTF buffers for one primitive from ``parse_obj``.

    Returns ``positions``/``normals``/``uvs``/``indices`` as bytes (``uvs`` is b"" without UVs), the index
    component type, the vertex and index counts, the POSITION ``min``/``max``, and the primitive's
    ``material`` key and optional ``material_index``. The same packed primitive can be written into
    a GLB once and drawn by many instances.
    """
    positions = list(prim["positions"])
    indices = list(prim["indices"])
    vert_count = len(positions)
    pos_bytes, pos_f32 = _pack_f32(positions, 3)
    idx_bytes, idx_comp_type = _pack_indices(indices, vert_count)
    normals_in = prim.get("normals")
    if normals_in is None or len(normals_in) == 0:
        normals = _compute_smooth_normals(positions, indices)
    elif len(normals_in) == vert_count:
        normals = [
            _normalize3(float(n[0]), float(n[1]), float(n[2]))
            for n in normals_in
        ]
    else:
        raise GlbError(
            f"normals length ({len(normals_in)}) does not match positions length ({vert_count})"
        )
    xs = pos_f32[0::3]
    ys = pos_f32[1::3]
    zs = pos_f32[2::3]
    nrm_bytes, _ = _pack_f32(normals, 3)
    uv_bytes = b""
    uvs_in = prim.get("uvs")
    if uvs_in is not None and len(uvs_in) > 0:
        if len(uvs_in) != vert_count:
            raise GlbError(
                f"uvs length ({len(uvs_in)}) does not match positions length ({vert_count})"
            )
        clean_uvs = [
            (
                _clean_zero(float(uv[0])) if math.isfinite(float(uv[0])) else 0.0,
                _clean_zero(float(uv[1])) if math.isfinite(float(uv[1])) else 0.0,
            )
            for uv in uvs_in
        ]
        uv_bytes, _ = _pack_f32(clean_uvs, 2)
    material = prim.get("material", "")
    return {
        "positions": pos_bytes,
        "normals": nrm_bytes,
        "uvs": uv_bytes,
        "indices": idx_bytes,
        "index_type": idx_comp_type,
        "vertices": vert_count,
        "index_count": len(indices),
        "min": [_clean_zero(min(xs)), _clean_zero(min(ys)), _clean_zero(min(zs))],
        "max": [_clean_zero(max(xs)), _clean_zero(max(ys)), _clean_zero(max(zs))],
        "material": material if isinstance(material, (str, int)) and not isinstance(material, bool) else "",
        "material_index": prim.get("material_index"),
    }


def _append_packed(bin_buf: bytearray, buffer_views: list[dict], accessors: list[dict], packed: dict) -> tuple[dict, int]:
    """Write one packed primitive's views and accessors; returns (attributes, indices accessor)."""
    vert_count = packed["vertices"]
    pos_bv = _append_buffer_view(bin_buf, buffer_views, packed["positions"], target=TARGET_ARRAY_BUFFER)
    pos_acc = len(accessors)
    accessors.append({
        "bufferView": pos_bv,
        "componentType": COMPONENT_FLOAT,
        "count": vert_count,
        "type": "VEC3",
        "min": list(packed["min"]),
        "max": list(packed["max"]),
    })
    nrm_bv = _append_buffer_view(bin_buf, buffer_views, packed["normals"], target=TARGET_ARRAY_BUFFER)
    nrm_acc = len(accessors)
    accessors.append({
        "bufferView": nrm_bv,
        "componentType": COMPONENT_FLOAT,
        "count": vert_count,
        "type": "VEC3",
    })
    attributes: dict[str, int] = {"POSITION": pos_acc, "NORMAL": nrm_acc}
    if packed["uvs"]:
        uv_bv = _append_buffer_view(bin_buf, buffer_views, packed["uvs"], target=TARGET_ARRAY_BUFFER)
        uv_acc = len(accessors)
        accessors.append({
            "bufferView": uv_bv,
            "componentType": COMPONENT_FLOAT,
            "count": vert_count,
            "type": "VEC2",
        })
        attributes["TEXCOORD_0"] = uv_acc
    idx_bv = _append_buffer_view(bin_buf, buffer_views, packed["indices"], target=TARGET_ELEMENT_ARRAY_BUFFER)
    idx_acc = len(accessors)
    accessors.append({
        "bufferView": idx_bv,
        "componentType": packed["index_type"],
        "count": packed["index_count"],
        "type": "SCALAR",
    })
    return attributes, idx_acc


def _parse_base_color_factor(value: object, default: list[float]) -> list[float]:
    if value is None:
        return list(default)
    if not isinstance(value, (list, tuple)) or len(value) not in (3, 4):
        raise GlbError("base_color_factor must be a 3- or 4-element float sequence")
    out = [float(c) for c in value]
    if len(out) == 3:
        out.append(1.0)
    if not all(math.isfinite(c) and 0.0 <= c <= 1.0 for c in out):
        raise GlbError("base_color_factor values must be finite numbers in [0.0, 1.0]")
    return [_clean_zero(c) for c in out]


def _parse_alpha_mode(mode: object, cutoff: object, factor_alpha: float) -> dict:
    """glTF alphaMode/alphaCutoff fields for a material spec; OPAQUE is the glTF default and is omitted."""
    if mode is None:
        mode = "BLEND" if factor_alpha < 1.0 else "OPAQUE"
    if not isinstance(mode, str) or mode not in ALPHA_MODES:
        raise GlbError("alpha_mode must be one of 'OPAQUE', 'MASK', 'BLEND'")
    if cutoff is not None and mode != "MASK":
        raise GlbError("alpha_cutoff is only valid with alpha_mode 'MASK'")
    if mode == "OPAQUE":
        return {}
    out: dict = {"alphaMode": mode}
    if cutoff is not None:
        if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or cutoff < 0.0:
            raise GlbError("alpha_cutoff must be a finite number >= 0.0")
        out["alphaCutoff"] = _clean_zero(float(cutoff))
    return out


class _MaterialTable:
    """glTF materials, images and textures for one GLB, shared by every primitive that uses them."""

    def __init__(self, materials, bin_buf: bytearray, buffer_views: list[dict]) -> None:
        self.materials = materials
        self.bin_buf = bin_buf
        self.buffer_views = buffer_views
        self.gltf_materials: list[dict] = []
        self.images: list[dict] = []
        self.textures: list[dict] = []
        self.mat_index_by_key: dict[tuple[str | int, bool], int] = {}
        self.default_mat_index: int | None = None
        self.tex_index_by_sha256: dict[bytes, int] = {}

    def get_texture_index(self, png_bytes: bytes) -> int:
        digest = hashlib.sha256(png_bytes).digest()
        existing = self.tex_index_by_sha256.get(digest)
        if existing is not None:
            return existing
        bv_idx = _append_buffer_view(self.bin_buf, self.buffer_views, png_bytes, target=None)
        img_idx = len(self.images)
        self.images.append({
            "bufferView": bv_idx,
            "mimeType": "image/png",
        })
        tex_idx = len(self.textures)
        self.textures.append({
            "sampler": 0,
            "source": img_idx,
        })
        self.tex_index_by_sha256[digest] = tex_idx
        return tex_idx

    def _lookup_material_spec(self, prim: dict) -> tuple[str | int, dict] | None:
        materials = self.materials
        if materials is None:
            return None
        mat_key = prim.get("material", "")
        if mat_key is None:
            mat_key = ""
        if isinstance(materials, dict):
            if mat_key in materials:
                return mat_key, materials[mat_key] or {}
            if "material_index" in prim:
                midx = prim["material_index"]
                if isinstance(midx, int) and not isinstance(midx, bool) and midx in materials:
                    return midx, materials[midx] or {}
        elif isinstance(materials, (list, tuple)):
            midx = prim.get("material_index")
            if isinstance(midx, int) and not isinstance(midx, bool) and 0 <= midx < len(materials):
                return midx, materials[midx] or {}
        return None

    def get_material_index(self, prim: dict, *, has_uv: bool) -> int:
        found = self._lookup_material_spec(prim)
        if found is not None:
            resolved_key, spec = found
            raw_png = spec.get("png_bytes")
            has_png = has_uv and isinstance(raw_png, (bytes, bytearray, memoryview)) and len(raw_png) > 0
            cache_key = (resolved_key, has_png)
            existing = self.mat_index_by_key.get(cache_key)
            if existing is not None:
                return existing
            default_color = DEFAULT_TEXTURED_COLOR if has_png else DEFAULT_CLAY_COLOR
            base_color = _parse_base_color_factor(spec.get("base_color_factor"), default_color)
            alpha_fields = _parse_alpha_mode(spec.get("alpha_mode"), spec.get("alpha_cutoff"), base_color[3])

            pbr: dict = {
                "baseColorFactor": base_color,
                "metallicFactor": DEFAULT_METALLIC,
                "roughnessFactor": DEFAULT_ROUGHNESS,
            }
            if has_png:
                tex_idx = self.get_texture_index(bytes(raw_png))
                pbr["baseColorTexture"] = {"index": tex_idx}

            mat_obj: dict = {"pbrMetallicRoughness": pbr}
            mat_name = spec.get("name")
            if mat_name is None and isinstance(resolved_key, str):
                mat_name = resolved_key
            if isinstance(mat_name, str) and mat_name:
                mat_obj["name"] = mat_name
            if spec.get("double_sided"):
                mat_obj["doubleSided"] = True
            mat_obj.update(alpha_fields)

            mat_idx = len(self.gltf_materials)
            self.gltf_materials.append(mat_obj)
            self.mat_index_by_key[cache_key] = mat_idx
            return mat_idx

        if self.default_mat_index is None:
            self.default_mat_index = len(self.gltf_materials)
            self.gltf_materials.append({
                "name": "default_clay",
                "pbrMetallicRoughness": {
                    "baseColorFactor": list(DEFAULT_CLAY_COLOR),
                    "metallicFactor": DEFAULT_METALLIC,
                    "roughnessFactor": DEFAULT_ROUGHNESS,
                },
            })
        return self.default_mat_index

    def apply(self, gltf: dict) -> None:
        gltf["materials"] = self.gltf_materials
        if self.textures:
            gltf["samplers"] = [dict(DEFAULT_SAMPLER)]
            gltf["images"] = self.images
            gltf["textures"] = self.textures


def _build_glb_with_stats(
    primitives: list[dict],
    materials: dict[str, dict] | None = None,
    *,
    generator: str = "Leonida glb_writer",
) -> tuple[bytes, dict]:
    non_empty: list[dict] = []
    for prim in primitives:
        positions = prim.get("positions") or ()
        indices = prim.get("indices") or ()
        if len(positions) > 0 and len(indices) > 0:
            non_empty.append(prim)

    if not non_empty:
        raise GlbError("cannot build GLB: no non-empty primitives")

    bin_buf = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    table = _MaterialTable(materials, bin_buf, buffer_views)
    get_material_index = table.get_material_index

    gltf_primitives: list[dict] = []
    total_verts = 0
    total_tris = 0

    for prim in non_empty:
        packed = pack_primitive(prim)
        total_verts += packed["vertices"]
        total_tris += packed["index_count"] // 3
        attributes, idx_acc = _append_packed(bin_buf, buffer_views, accessors, packed)
        mat_idx = get_material_index(prim, has_uv="TEXCOORD_0" in attributes)

        gltf_primitives.append({
            "attributes": attributes,
            "indices": idx_acc,
            "material": mat_idx,
            "mode": MODE_TRIANGLES,
        })

    # Ensure BIN buffer is 4-byte aligned.
    bin_pad = (-len(bin_buf)) % 4
    if bin_pad:
        bin_buf.extend(b"\x00" * bin_pad)

    mesh_obj: dict = {"primitives": gltf_primitives}
    if len(non_empty) == 1:
        sole_name = non_empty[0].get("name")
        if isinstance(sole_name, str) and sole_name:
            mesh_obj["name"] = sole_name

    gltf: dict = {
        "asset": {
            "version": "2.0",
            "generator": generator,
        },
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "meshes": [mesh_obj],
        "materials": table.gltf_materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_buf)}],
    }

    if table.textures:
        gltf["samplers"] = [dict(DEFAULT_SAMPLER)]
        gltf["images"] = table.images
        gltf["textures"] = table.textures

    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    json_pad = (-len(json_bytes)) % 4
    if json_pad:
        json_bytes += b" " * json_pad

    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    header = struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total_length)
    json_hdr = struct.pack("<II", len(json_bytes), CHUNK_TYPE_JSON)
    bin_hdr = struct.pack("<II", len(bin_buf), CHUNK_TYPE_BIN)
    glb_bytes = b"".join((header, json_hdr, json_bytes, bin_hdr, bytes(bin_buf)))

    bounds = compute_bounds(non_empty)
    assert bounds is not None
    stats = {
        "bytes": len(glb_bytes),
        "triangles": total_tris,
        "vertices": total_verts,
        "primitives": len(non_empty),
        "textures": len(table.textures),
        "bounds": bounds,
    }
    return glb_bytes, stats


def build_glb(
    primitives: list[dict],
    materials: dict[str, dict] | None = None,
    *,
    generator: str = "Leonida glb_writer",
) -> bytes:
    """Serialize glTF-ready primitives and optional PNG materials into binary .glb bytes.

    ``materials`` maps a primitive's ``material`` name (or ``material_index``) to a
    spec dict. Each spec becomes one glTF material shared by every primitive that
    resolves to it; primitives without a matching spec share an opaque
    ``default_clay`` material. Recognised spec keys:

    - ``name``: material name (defaults to the string key).
    - ``png_bytes``: PNG embedded as ``baseColorTexture`` for primitives with UVs.
    - ``base_color_factor``: 3 or 4 finite floats in [0, 1]; alpha defaults to 1.
    - ``double_sided``: emit ``doubleSided: true`` when truthy.
    - ``alpha_mode``: ``"OPAQUE"``, ``"MASK"`` or ``"BLEND"``. When omitted, a
      ``base_color_factor`` alpha below 1 gives ``BLEND``; otherwise no ``alphaMode``
      is written (glTF default ``OPAQUE``). Texture alpha is never inspected.
    - ``alpha_cutoff``: finite number >= 0, only with ``MASK`` (glTF default 0.5).

    Invalid specs raise GlbError.
    """
    glb_bytes, _ = _build_glb_with_stats(primitives, materials, generator=generator)
    return glb_bytes


# On Windows, MoveFileEx fails with a sharing violation (PermissionError) while
# another writer is replacing the same destination or a reader holds it open
# without FILE_SHARE_DELETE. Those holds last milliseconds, so the replace is
# retried with a short backoff (about 1.3 s in total) before the error is raised.
_REPLACE_RETRY = os.name == "nt"
_REPLACE_DELAYS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.1, 0.2, 0.3, 0.5)


def _replace(source: str, destination: str) -> None:
    for delay in _REPLACE_DELAYS if _REPLACE_RETRY else ():
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            time.sleep(delay)
    os.replace(source, destination)


def write_glb(
    path: str,
    primitives: list[dict],
    materials: dict[str, dict] | None = None,
) -> dict:
    """Atomically publish a GLB using a private staging file in its destination directory.

    Overlapping writers publish complete files (the last replacement wins), and
    existing sibling files are never used as staging files. On Windows a replace
    that hits a sharing violation is retried briefly (see ``_replace``); one held
    longer than that still raises PermissionError and leaves the old file in place.
    ``materials`` takes the spec keys documented on ``build_glb``.
    """
    glb_bytes, stats = _build_glb_with_stats(primitives, materials)
    _write_bytes_atomic(path, glb_bytes)
    return stats


def _write_bytes_atomic(path: str, glb_bytes: bytes) -> None:
    """Publish ``glb_bytes`` at ``path`` through a private staging file in the same directory."""
    out_path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=parent, prefix=".glb-", suffix=".tmp", delete=False,
        ) as f:
            tmp_path = f.name
            f.write(glb_bytes)
        # NamedTemporaryFile creates 0600; give the GLB the mode of the file it replaces, or a new file's.
        os.chmod(tmp_path, published_mode(out_path))
        # Close before replacement: Windows cannot rename an open staging file.
        _replace(tmp_path, out_path)
    except BaseException:
        try:
            if tmp_path is not None:
                os.remove(tmp_path)
        except OSError:
            pass
        raise


INSTANCING_EXTENSION = "EXT_mesh_gpu_instancing"
INSTANCING_MODES = ("gpu", "nodes")


def _quat_matrix(q) -> list[list[float]]:
    x, y, z, w = (float(v) for v in q)
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def instance_bounds(box_min, box_max, translation, rotation, scale) -> list[float]:
    """Exact axis-aligned bounds of the box [box_min, box_max] after scale, rotation (unit x, y, z, w) and
    translation, the order glTF applies a node's TRS in."""
    r = _quat_matrix(rotation)
    m = [[r[i][j] * float(scale[j]) for j in range(3)] for i in range(3)]
    c = [(float(box_min[j]) + float(box_max[j])) * 0.5 for j in range(3)]
    h = [(float(box_max[j]) - float(box_min[j])) * 0.5 for j in range(3)]
    out_min, out_max = [], []
    for i in range(3):
        ci = m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2] + float(translation[i])
        hi = abs(m[i][0]) * h[0] + abs(m[i][1]) * h[1] + abs(m[i][2]) * h[2]
        out_min.append(ci - hi)
        out_max.append(ci + hi)
    return out_min + out_max


def build_instanced_glb(
    meshes: list[dict],
    materials: dict[str | int, dict] | None = None,
    *,
    mode: str = "gpu",
    generator: str = "Leonida glb_writer",
) -> tuple[bytes, dict]:
    """One GLB in which each mesh is stored once and drawn by many instances.

    ``meshes`` is a list of ``{"name": str or None, "primitives": [pack_primitive(...)], "instances":
    [(translation xyz, rotation xyzw, scale xyz), ...]}``. Every instance transform must have a positive
    determinant (no mirroring scale); give mirrored instances their own mirrored mesh instead, so the winding is
    right in every loader. ``mode`` "gpu" writes one node per mesh with ``EXT_mesh_gpu_instancing`` (listed in
    extensionsUsed and extensionsRequired); "nodes" writes one plain glTF node per instance, which every loader
    reads but which costs a draw call per instance. ``materials`` uses the same lookup and spec keys as
    ``build_glb``. Returns (GLB bytes, stats), including rendered triangles (each mesh's triangles times its
    instances) and the bounds of every instance.
    """
    if mode not in INSTANCING_MODES:
        raise GlbError(f"instancing mode must be one of {INSTANCING_MODES}, got {mode!r}")
    bin_buf = bytearray()
    buffer_views: list[dict] = []
    accessors: list[dict] = []
    table = _MaterialTable(materials, bin_buf, buffer_views)
    gltf_meshes: list[dict] = []
    nodes: list[dict] = []
    stored_verts = stored_tris = drawn_tris = primitives = instances_total = 0
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for mesh in meshes:
        prims = [p for p in mesh.get("primitives") or () if p.get("vertices") and p.get("index_count")]
        insts = list(mesh.get("instances") or ())
        if not prims or not insts:
            continue
        gltf_prims = []
        box_min = [math.inf] * 3
        box_max = [-math.inf] * 3
        mesh_tris = 0
        for packed in prims:
            attributes, idx_acc = _append_packed(bin_buf, buffer_views, accessors, packed)
            gltf_prims.append({
                "attributes": attributes,
                "indices": idx_acc,
                "material": table.get_material_index(packed, has_uv="TEXCOORD_0" in attributes),
                "mode": MODE_TRIANGLES,
            })
            stored_verts += packed["vertices"]
            mesh_tris += packed["index_count"] // 3
            for k in range(3):
                box_min[k] = min(box_min[k], packed["min"][k])
                box_max[k] = max(box_max[k], packed["max"][k])
        primitives += len(prims)
        stored_tris += mesh_tris
        mesh_obj: dict = {"primitives": gltf_prims}
        if isinstance(mesh.get("name"), str) and mesh["name"]:
            mesh_obj["name"] = mesh["name"]
        mesh_idx = len(gltf_meshes)
        gltf_meshes.append(mesh_obj)
        ts, rs, ss = [], [], []
        for t, q, sc in insts:
            t = [float(v) for v in t]
            sc = [float(v) for v in sc]
            q = [float(v) for v in q]
            n = math.sqrt(sum(v * v for v in q))
            if not n or not math.isfinite(n):
                raise GlbError("instance rotation must be a finite non-zero quaternion")
            q = [v / n for v in q]
            if sc[0] * sc[1] * sc[2] <= 0:
                raise GlbError("instance scale must have a positive determinant; mirror the mesh instead")
            if not all(math.isfinite(v) for v in t + sc):
                raise GlbError("instance translation and scale must be finite")
            ts.append(t)
            rs.append(q)
            ss.append(sc)
        instances_total += len(insts)
        drawn_tris += mesh_tris * len(insts)
        bounds_transforms = zip(ts, rs, ss)
        if mode == "gpu":
            attrs = {}
            stored_transforms: list[Iterable[tuple[float, ...]]] = []
            for key, values, kind in (("TRANSLATION", ts, "VEC3"), ("ROTATION", rs, "VEC4"), ("SCALE", ss, "VEC3")):
                components = 4 if kind == "VEC4" else 3
                data, _ = _pack_f32(values, components)
                stored_transforms.append(struct.iter_unpack(f"<{components}f", data))
                bv = _append_buffer_view(bin_buf, buffer_views, data, target=None)
                attrs[key] = len(accessors)
                accessors.append({"bufferView": bv, "componentType": COMPONENT_FLOAT, "count": len(values), "type": kind})
            nodes.append({"mesh": mesh_idx, "extensions": {INSTANCING_EXTENSION: {"attributes": attrs}}})
            bounds_transforms = zip(*stored_transforms)
        else:
            for t, q, sc in zip(ts, rs, ss):
                node: dict = {"mesh": mesh_idx}
                if any(t):
                    node["translation"] = [_clean_zero(v) for v in t]
                if q != [0.0, 0.0, 0.0, 1.0]:
                    node["rotation"] = [_clean_zero(v) for v in q]
                if sc != [1.0, 1.0, 1.0]:
                    node["scale"] = [_clean_zero(v) for v in sc]
                nodes.append(node)
        for t, q, sc in bounds_transforms:
            b = instance_bounds(box_min, box_max, t, q, sc)
            for k in range(3):
                lo[k] = min(lo[k], b[k])
                hi[k] = max(hi[k], b[k + 3])
    if not nodes:
        raise GlbError("cannot build GLB: no mesh with primitives and instances")
    bin_pad = (-len(bin_buf)) % 4
    if bin_pad:
        bin_buf.extend(b"\x00" * bin_pad)
    gltf: dict = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_buf)}],
    }
    table.apply(gltf)
    if mode == "gpu":
        gltf["extensionsUsed"] = [INSTANCING_EXTENSION]
        gltf["extensionsRequired"] = [INSTANCING_EXTENSION]
    json_bytes = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    json_bytes += b" " * ((-len(json_bytes)) % 4)
    total_length = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    glb_bytes = b"".join((
        struct.pack("<4sII", GLB_MAGIC, GLB_VERSION, total_length),
        struct.pack("<II", len(json_bytes), CHUNK_TYPE_JSON), json_bytes,
        struct.pack("<II", len(bin_buf), CHUNK_TYPE_BIN), bytes(bin_buf),
    ))
    return glb_bytes, {
        "bytes": len(glb_bytes),
        "triangles": drawn_tris,
        "stored_triangles": stored_tris,
        "vertices": stored_verts,
        "primitives": primitives,
        "meshes": len(gltf_meshes),
        "instances": instances_total,
        "textures": len(table.textures),
        "bounds": lo + hi,
    }


def write_instanced_glb(path: str, meshes: list[dict], materials: dict[str, dict] | None = None, *,
                        mode: str = "gpu") -> dict:
    """``build_instanced_glb`` published atomically at ``path`` (same staging rules as ``write_glb``)."""
    glb_bytes, stats = build_instanced_glb(meshes, materials, mode=mode)
    _write_bytes_atomic(path, glb_bytes)
    return stats


def read_glb_summary(data: bytes) -> dict:
    """Validate binary glTF 2.0 container headers/chunks and return parsed summary."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise GlbError("GLB data must be bytes-like")
    buf = memoryview(data)
    actual_len = len(buf)
    if actual_len < 20:
        raise GlbError(f"GLB data too short: {actual_len} bytes")

    magic, version, total_length = struct.unpack_from("<4sII", buf, 0)
    if magic != GLB_MAGIC:
        raise GlbError(f"invalid GLB magic: {magic!r}")
    if version != GLB_VERSION:
        raise GlbError(f"unsupported GLB version: {version}")
    if total_length != actual_len:
        raise GlbError(f"GLB total length mismatch: header={total_length}, actual={actual_len}")
    if total_length % 4 != 0:
        raise GlbError(f"GLB total length not 4-byte aligned: {total_length}")

    json_len, json_type = struct.unpack_from("<II", buf, 12)
    if json_type != CHUNK_TYPE_JSON:
        raise GlbError(f"first GLB chunk must be JSON (0x{CHUNK_TYPE_JSON:08X}), got 0x{json_type:08X}")
    if json_len % 4 != 0:
        raise GlbError(f"JSON chunk length not 4-byte aligned: {json_len}")
    if 20 + json_len > total_length:
        raise GlbError(f"JSON chunk length {json_len} exceeds total length {total_length}")

    json_raw = bytes(buf[20:20 + json_len])
    try:
        gltf = json.loads(json_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GlbError(f"invalid JSON chunk: {exc}") from exc
    if not isinstance(gltf, dict):
        raise GlbError("GLB JSON root must be a JSON object")

    after_json = 20 + json_len
    offset = after_json
    bin_bytes = 0
    while offset < total_length:
        if offset + 8 > total_length:
            raise GlbError("truncated GLB chunk header")
        chunk_len, chunk_type = struct.unpack_from("<II", buf, offset)
        if chunk_len % 4 != 0:
            raise GlbError(f"GLB chunk length not 4-byte aligned: {chunk_len}")
        chunk_end = offset + 8 + chunk_len
        if chunk_end > total_length:
            raise GlbError(f"GLB chunk length {chunk_len} exceeds GLB total length {total_length}")
        if chunk_type == CHUNK_TYPE_JSON:
            raise GlbError("duplicate GLB JSON chunk")
        if chunk_type == CHUNK_TYPE_BIN:
            if offset != after_json:
                raise GlbError("GLB BIN chunk must be second")
            bin_bytes = chunk_len
        # Extension chunks are ignored after checking their framing.
        offset = chunk_end

    buf0_limit = bin_bytes
    buffers = gltf.get("buffers")
    if isinstance(buffers, list) and buffers:
        buf0 = buffers[0]
        if isinstance(buf0, dict):
            embedded = "uri" not in buf0
            declared = buf0.get("byteLength", 0 if embedded else None)
            invalid_length = isinstance(declared, bool) or not isinstance(declared, int) or declared < 0
            if embedded and (invalid_length or declared > bin_bytes):
                raise GlbError(f"buffers[0].byteLength ({declared}) exceeds BIN chunk ({bin_bytes})")
            if invalid_length:
                raise GlbError("invalid buffers[0].byteLength")
            buf0_limit = declared

    buffer_views = gltf.get("bufferViews")
    if isinstance(buffer_views, list):
        for i, bv in enumerate(buffer_views):
            if not isinstance(bv, dict):
                raise GlbError(f"bufferViews[{i}] is not an object")
            if bv.get("buffer", 0) == 0:
                bv_off = bv.get("byteOffset", 0)
                bv_len = bv.get("byteLength", -1)
                if (
                    isinstance(bv_off, bool)
                    or isinstance(bv_len, bool)
                    or not isinstance(bv_off, int)
                    or not isinstance(bv_len, int)
                    or bv_off < 0
                    or bv_len < 0
                ):
                    raise GlbError(f"invalid byteOffset/byteLength in bufferViews[{i}]")
                if bv_off + bv_len > buf0_limit:
                    raise GlbError(
                        f"bufferViews[{i}] exceeds buffer 0 ({bv_off}+{bv_len} > {buf0_limit})"
                    )

    return {
        "version": version,
        "total_bytes": total_length,
        "json": gltf,
        "bin_bytes": bin_bytes,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a Wavefront OBJ file to binary glTF 2.0 (.glb)."
    )
    parser.add_argument("in_obj", help="Input Wavefront .obj file path")
    parser.add_argument("out_glb", help="Output binary .glb file path")
    parser.add_argument(
        "--up",
        choices=("z", "y"),
        default="z",
        help="Source up axis: 'z' converts RAGE Z-up to glTF Y-up (default), 'y' keeps Y-up",
    )
    args = parser.parse_args(argv)

    try:
        # write_glb replaces its target, so naming the OBJ as the output would silently destroy the model.
        if os.path.exists(args.out_glb) and os.path.exists(args.in_obj) and os.path.samefile(
                args.in_obj, args.out_glb):
            raise GlbError(f"output must not overwrite the input OBJ: {args.in_obj}")
        primitives = parse_obj(args.in_obj, is_path=True, up_axis=args.up)
        stats = write_glb(args.out_glb, primitives)
    except (GlbError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(
        f"wrote {args.out_glb}: {stats['bytes']} bytes, "
        f"{stats['primitives']} primitives, {stats['vertices']} verts, "
        f"{stats['triangles']} tris"
    )
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    raise SystemExit(main())
