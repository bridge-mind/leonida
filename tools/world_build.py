"""Build streamed spatial world.json and .glb chunks from rpf9dump asset exports.

Consumes rpf9dump model manifests (models/manifest.json + .obj + <stem>_tex/*.dds) and
optional TTPF texture manifests (manifest.json + *.dds), groups geometry into spatial
or grid-laid-out XZ chunks, embeds decoded PNG textures into binary glTF 2.0 (.glb)
chunks, and writes world.json atomically for viewer/serve.py + viewer/index.html.
Chunks are built and verified in a staging directory first. If the process stops during
publication, the next run recovers the last committed world before reading its new inputs.

With --placements, models are instanced at their map placements instead: each entity row written by
``world_index.py --entities-out`` is matched to an exported model by the joaat of its name (and, when the row
carries it, the 64-bit Murmur3 archetype key, as in world_names.py), then scaled, rotated by the conjugate of
the stored quaternion and translated into world space.

Each placed model is parsed once and stored once per chunk; its placements in that chunk are instance
transforms (EXT_mesh_gpu_instancing by default, or plain glTF nodes with --instancing nodes), and every chunk is
written as soon as it is built, so memory stays bounded by the largest chunk rather than the size of the map.

Usage:
    python -I tools/world_build.py <assets_dir> --out <world_dir> [options]
    python -I tools/world_build.py <assets_dir> --out <world_dir> --placements <entities.jsonl>
        [--lod-levels 0,5] [--region MINX,MINY,MAXX,MAXY] [--instancing gpu|nodes|bake]
"""
from __future__ import annotations

import argparse
import array
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import functools
import hashlib
import json
import math
import os
import re
import shutil
import stat
import statistics
import sys
import tempfile

try:
    import fcntl
except ImportError:                       # Windows
    fcntl = None
try:
    import msvcrt
except ImportError:                       # POSIX
    msvcrt = None
import threading

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from file_mode import published_mode  # noqa: E402
from pool_workers import process_workers  # noqa: E402

try:
    import dds_png
except ImportError:  # pragma: no cover - sibling module may not be present during --help
    dds_png = None

try:
    import glb_writer
except ImportError:  # pragma: no cover - sibling module may not be present during --help
    glb_writer = None


GRID_GAP = 4.0
# lodLevel values (docs/world.md): 0 HD, 1 LOD, 2 SLOD1, 3 SLOD2, 4 SLOD3, 5 ORPHAN_HD, 6 SLOD4.
DEFAULT_LOD_LEVELS = (0, 5)


def _get_world_names():
    import world_names  # sibling tool; only needed for --placements
    return world_names


def _placement_scale(value) -> float:
    """Optional scales that cannot be represented as finite floats use the existing unit default."""
    if not isinstance(value, (int, float)):
        return 1.0
    try:
        scale = float(value)
    except OverflowError:
        return 1.0
    return scale if math.isfinite(scale) else 1.0


def _parse_archetype(value) -> int:
    """Parse one placement's archetype: a real JSON integer is used as-is, a string is base-16.

    bool is an int subclass but is never a valid archetype. Any other type (float, list, ...) is invalid and
    raises, so the callers count the row as malformed. Shared by ``load_placements`` and ``iter_placements``
    so the two readers stay in step.
    """
    if isinstance(value, bool):
        raise ValueError("bool is not an archetype")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value, 16)
    raise TypeError("archetype must be an integer or a hex string")


def load_placements(path: str, lod_levels=DEFAULT_LOD_LEVELS, region=None) -> tuple[list[dict], dict]:
    """Read ``world_index.py --entities-out`` rows. Returns (selected rows, counts).

    A row is kept when its lod_level is in lod_levels and, with region = (minx, miny, maxx, maxy) in RAGE
    world metres, its position lies inside it. Rows without a finite position, rotation or archetype are
    counted as malformed and skipped, as are unhashable LOD values. Optional scales that are missing,
    non-numeric or not representable as finite floats default to 1.0.
    """
    levels = set(lod_levels)
    counts = {"rows": 0, "malformed": 0, "other_lod_level": 0, "outside_region": 0, "selected": 0}
    rows: list[dict] = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            counts["rows"] += 1
            try:
                row = json.loads(line)
                arch = _parse_archetype(row["archetype"])
                pos = [float(v) for v in row["position"]]
                rot = [float(v) for v in row["rotation"]]
                selected_lod = row.get("lod_level") in levels
            except (ValueError, KeyError, TypeError, OverflowError):
                counts["malformed"] += 1
                continue
            if len(pos) != 3 or len(rot) != 4 or not all(math.isfinite(v) for v in pos + rot):
                counts["malformed"] += 1
                continue
            if not selected_lod:
                counts["other_lod_level"] += 1
                continue
            if region is not None and not (region[0] <= pos[0] <= region[2] and region[1] <= pos[1] <= region[3]):
                counts["outside_region"] += 1
                continue
            sxy, sz = row.get("scale_xy"), row.get("scale_z")
            row["_arch"] = arch & 0xFFFFFFFF
            row["_pos"], row["_rot"] = pos, rot
            row["_sxy"] = _placement_scale(sxy)
            row["_sz"] = _placement_scale(sz)
            rows.append(row)
            counts["selected"] += 1
    return rows, counts


def _placement_counts() -> dict:
    return {"rows": 0, "malformed": 0, "other_lod_level": 0, "outside_region": 0, "selected": 0}


def iter_placements(path: str, counts: dict, lod_levels=DEFAULT_LOD_LEVELS, region=None):
    """The rows ``load_placements`` selects, one at a time, adding to ``counts`` (a whole city need not be held).

    The row checks are the same as ``load_placements``'; keep the two in step.
    """
    levels = set(lod_levels)
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            counts["rows"] += 1
            try:
                row = json.loads(line)
                arch = _parse_archetype(row["archetype"])
                pos = [float(v) for v in row["position"]]
                rot = [float(v) for v in row["rotation"]]
                selected_lod = row.get("lod_level") in levels
            except (ValueError, KeyError, TypeError, OverflowError):
                counts["malformed"] += 1
                continue
            if len(pos) != 3 or len(rot) != 4 or not all(math.isfinite(v) for v in pos + rot):
                counts["malformed"] += 1
                continue
            if not selected_lod:
                counts["other_lod_level"] += 1
                continue
            if region is not None and not (region[0] <= pos[0] <= region[2] and region[1] <= pos[1] <= region[3]):
                counts["outside_region"] += 1
                continue
            sxy, sz = row.get("scale_xy"), row.get("scale_z")
            row["_arch"] = arch & 0xFFFFFFFF
            row["_pos"], row["_rot"] = pos, rot
            row["_sxy"] = _placement_scale(sxy)
            row["_sz"] = _placement_scale(sz)
            counts["selected"] += 1
            yield row


def entity_transform(rotation, scale_xy: float, scale_z: float) -> tuple[list[list[float]], list[list[float]], bool]:
    """3x3 position and normal matrices (RAGE Z-up) for one entity, and whether it mirrors.

    The stored quaternion (x, y, z, w) is the inverse of the entity's world rotation, so the world rotation is
    its conjugate (-x, -y, -z, w). Positions use R * diag(sxy, sxy, sz); normals use R * diag(1/sxy, 1/sxy, 1/sz).
    """
    x, y, z, w = (float(v) for v in rotation)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0 or not math.isfinite(n):
        x, y, z, w = 0.0, 0.0, 0.0, 1.0
    else:
        x, y, z, w = -x / n, -y / n, -z / n, w / n
    r = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    sc = (scale_xy, scale_xy, scale_z)
    pos_m = [[r[i][j] * sc[j] for j in range(3)] for i in range(3)]
    inv = tuple((1.0 / v) if v else 0.0 for v in sc)
    nrm_m = [[r[i][j] * inv[j] for j in range(3)] for i in range(3)]
    return pos_m, nrm_m, scale_xy * scale_xy * scale_z < 0


def place_primitives(prims: list[dict], rotation, scale_xy: float, scale_z: float, position) -> list[dict]:
    """New primitives for one placed instance: model-space RAGE (Z-up) -> world -> glTF Y-up (x, z, -y).

    ``prims`` must come from ``glb_writer.parse_obj(..., up_axis="y")``, i.e. unconverted RAGE coordinates.
    """
    m, nm, mirror = entity_transform(rotation, scale_xy, scale_z)
    tx, ty, tz = (float(v) for v in position)
    out = []
    for p in prims:
        positions = []
        for vx, vy, vz in p.get("positions") or ():
            wx = m[0][0] * vx + m[0][1] * vy + m[0][2] * vz + tx
            wy = m[1][0] * vx + m[1][1] * vy + m[1][2] * vz + ty
            wz = m[2][0] * vx + m[2][1] * vy + m[2][2] * vz + tz
            positions.append((wx, wz, -wy))
        normals = []
        for nx, ny, nz in p.get("normals") or ():
            ax = nm[0][0] * nx + nm[0][1] * ny + nm[0][2] * nz
            ay = nm[1][0] * nx + nm[1][1] * ny + nm[1][2] * nz
            az = nm[2][0] * nx + nm[2][1] * ny + nm[2][2] * nz
            ln = math.sqrt(ax * ax + ay * ay + az * az) or 1.0
            normals.append((ax / ln, az / ln, -ay / ln))
        indices = list(p.get("indices") or ())
        if mirror:
            for i in range(0, len(indices) - 2, 3):
                indices[i + 1], indices[i + 2] = indices[i + 2], indices[i + 1]
        q = dict(p)
        q.update(positions=positions, normals=normals, indices=indices, uvs=list(p.get("uvs") or ()))
        out.append(q)
    return out


def model_name_index(entries: list[tuple[str, dict]], world_names) -> dict[int, list[tuple[str, dict, int]]]:
    """32-bit joaat of each manifest item's model name -> [(models_dir, item, Murmur3-64 of the name)]."""
    index: dict[int, list[tuple[str, dict, int]]] = {}
    for models_dir, item in entries:
        name = item.get("Name")
        if not isinstance(name, str) or not name.strip():
            continue
        stem = name.split(".#", 1)[0].strip()
        if not stem:
            continue
        index.setdefault(world_names.joaat(stem), []).append((models_dir, item, world_names.hash64(stem)))
    return index


def resolve_placement(index, arch: int, key_hex) -> tuple[tuple[str, dict] | None, str]:
    """Pick the exported model for one entity. Returns (model or None, outcome)."""
    cands = index.get(arch)
    if not cands:
        return None, "unresolved_archetype"
    key = None
    if isinstance(key_hex, str):
        try:
            key = int(key_hex, 16)
        except ValueError:
            key = None
    if key is not None:
        good = [c for c in cands if c[2] == key]
        if not good:
            return None, "key_mismatch"
        return (good[0][0], good[0][1]), "placed"
    return (cands[0][0], cands[0][1]), "placed_unverified"


INSTANCING_MODES = ("gpu", "nodes", "bake")
# lodLevel -> chunk-set name for --lod-sets, and the parent chain HD -> LOD -> SLOD1 -> SLOD2 -> SLOD3 -> SLOD4.
LOD_TIER_NAMES = {0: "hd", 1: "lod", 2: "slod1", 3: "slod2", 4: "slod3", 5: "orphan_hd", 6: "slod4"}
LOD_CHAIN = (0, 1, 2, 3, 4, 6)
ALL_LOD_LEVELS = tuple(sorted(LOD_TIER_NAMES))


def _median(values) -> float | None:
    vals = sorted(values)
    if not vals:
        return None
    mid = len(vals) // 2
    return float(vals[mid]) if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def lod_bands(child_lod: dict, lod_dist: dict, present) -> tuple[dict, dict]:
    """Camera distance band [start, end) per tier, from the selected entities' medians.

    A parent's childLodDist is the distance inside which its children are drawn instead of it, so a tier starts
    at its own median childLodDist (the finest starts at 0) and ends where the next coarser tier starts.
    A missing childLodDist uses the previous finer tier's median lodDist. The coarsest tier ends at its
    median lodDist. ORPHAN_HD has no parent and is drawn from 0 to its median lodDist. Returns
    (bands by tier name, the measured medians by tier name)."""
    measured = {}
    for lvl in present:
        measured[LOD_TIER_NAMES[lvl]] = {"child_lod_dist_median": _median(child_lod.get(lvl, ())),
                                         "lod_dist_median": _median(lod_dist.get(lvl, ()))}
    bands = {}
    chain = [lvl for lvl in LOD_CHAIN if lvl in present]
    starts = []
    for i, lvl in enumerate(chain):
        if i == 0:
            starts.append(0.0)
            continue
        start = measured[LOD_TIER_NAMES[lvl]]["child_lod_dist_median"]
        if start is None:
            previous_tier = LOD_TIER_NAMES[chain[i - 1]]
            start = measured[previous_tier]["lod_dist_median"]
        starts.append(max(start or 0.0, starts[-1]))
    for i, lvl in enumerate(chain):
        if i + 1 < len(chain):
            end = starts[i + 1]
        else:
            end = measured[LOD_TIER_NAMES[lvl]]["lod_dist_median"] or max(4.0 * starts[i], 1000.0)
        bands[LOD_TIER_NAMES[lvl]] = [starts[i], max(end, starts[i])]
    if 5 in present:
        end = measured["orphan_hd"]["lod_dist_median"]
        if end is None:
            end = bands["hd"][1] if "hd" in bands else 1000.0
        bands["orphan_hd"] = [0.0, float(end)]
    return bands, measured
_PACK_MAGIC = b"LPK1"


def placement_trs(rotation, scale_xy: float, scale_z: float, position):
    """glTF (Y-up) instance transform for one entity: (translation, rotation xyzw, scale, mirrored).

    The same transform as ``entity_transform`` + the Z-up to Y-up conversion, split into a node TRS with a
    positive determinant. A negative XY scale is a turn by 180 degrees about the model's Z axis, so it is folded
    into the rotation. A negative Z scale mirrors, which no TRS with positive scale can express: ``mirrored`` is
    then True and the instance must draw the model mirrored along its Z axis (glTF Y), with reversed winding.
    """
    x, y, z, w = (float(v) for v in rotation)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0 or not math.isfinite(n):
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 1.0
    else:
        qx, qy, qz, qw = -x / n, -y / n, -z / n, w / n      # the world rotation is the stored quaternion's conjugate
    sxy, sz = float(scale_xy), float(scale_z)
    if sxy < 0:
        qx, qy, qz, qw = qy, -qx, qw, -qz                    # q * (0, 0, 1, 0): turn 180 degrees about model Z first
        sxy = -sxy
    mirrored = sz < 0
    if mirrored:
        sz = -sz
    px, py, pz = (float(v) for v in position)
    # RAGE (x, y, z) -> glTF (x, z, -y) for the translation, the rotation axis and the scale axes.
    return (px, pz, -py), (qx, qz, -qy, qw), (sxy, sz, sxy), mirrored


def _write_pack(path: str, packed: list[dict]) -> int:
    """Store packed primitives in one file: magic, u32 header length, JSON header, then the buffers."""
    header = []
    blobs = []
    for prim in packed:
        meta = {k: prim[k] for k in ("index_type", "vertices", "index_count", "min", "max", "material")}
        meta["lengths"] = [len(prim[k]) for k in ("positions", "normals", "uvs", "indices")]
        header.append(meta)
        blobs.extend(prim[k] for k in ("positions", "normals", "uvs", "indices"))
    head = json.dumps(header, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    with open(path, "wb") as f:
        f.write(_PACK_MAGIC + len(head).to_bytes(4, "little") + head)
        for blob in blobs:
            f.write(blob)
        return f.tell()


def _read_pack(path: str) -> list[dict]:
    with open(path, "rb") as f:
        data = f.read()
    if data[:4] != _PACK_MAGIC:
        raise ValueError(f"{path} is not a packed model")
    n = int.from_bytes(data[4:8], "little")
    header = json.loads(data[8:8 + n].decode("ascii"))
    at = 8 + n
    out = []
    for meta in header:
        prim = {k: meta[k] for k in ("index_type", "vertices", "index_count", "min", "max", "material")}
        for key, length in zip(("positions", "normals", "uvs", "indices"), meta["lengths"]):
            prim[key] = data[at:at + length]
            at += length
        out.append(prim)
    return out


def _le_array(code: str, raw: bytes) -> array.array:
    arr = array.array(code)
    arr.frombytes(raw)
    if sys.byteorder == "big":
        arr.byteswap()
    return arr


def _le_bytes(arr: array.array) -> bytes:
    if sys.byteorder == "big":
        arr = array.array(arr.typecode, arr)
        arr.byteswap()
    return arr.tobytes()


def _mirror_packed(prim: dict) -> dict:
    """The primitive mirrored along glTF Y (the model's RAGE Z): y of positions and normals negated, winding
    reversed, so a positive-determinant instance of it draws a mirrored placement correctly."""
    pos = _le_array("f", prim["positions"])
    pos[1::3] = array.array("f", [-v for v in pos[1::3]])
    nrm = _le_array("f", prim["normals"])
    nrm[1::3] = array.array("f", [-v for v in nrm[1::3]])
    code = "H" if prim["index_type"] == 5123 else "I"
    if array.array(code).itemsize != (2 if code == "H" else 4):
        code = "L" if code == "I" else code
    idx = _le_array(code, prim["indices"])
    a, b = idx[1::3], idx[2::3]
    idx[1::3], idx[2::3] = b, a
    out = dict(prim)
    out.update(positions=_le_bytes(pos), normals=_le_bytes(nrm), indices=_le_bytes(idx),
               min=[prim["min"][0], -prim["max"][1], prim["min"][2]],
               max=[prim["max"][0], -prim["min"][1], prim["max"][2]])
    return out


def _pack_model_job(job: tuple) -> tuple[int, dict | None]:
    """Parse one model's OBJ once (glTF Y-up model space), pack it to its cache file and return its summary."""
    model_id, obj_path, mesh_shaders, tex_map, textures, cache_path = job
    glb_mod = _get_glb_writer()
    try:
        prims = glb_mod.parse_obj(obj_path, is_path=True, up_axis="z")
    except (OSError, glb_mod.GlbError, ValueError):
        return model_id, None
    mat_to_dds = _assign_materials(prims, mesh_shaders, tex_map, textures)
    try:
        packed = [glb_mod.pack_primitive(p) for p in prims if p.get("positions") and p.get("indices")]
    except glb_mod.GlbError:
        return model_id, None
    if not packed:
        return model_id, None
    lo = [min(p["min"][k] for p in packed) for k in range(3)]
    hi = [max(p["max"][k] for p in packed) for k in range(3)]
    if not all(math.isfinite(v) for v in lo + hi):
        return model_id, None
    size = _write_pack(cache_path, packed)
    used = {p["material"] for p in packed}
    return model_id, {
        "box": lo + hi,
        "triangles": sum(p["index_count"] // 3 for p in packed),
        "mat_to_dds": {k: v for k, v in mat_to_dds.items() if k in used},
        "pack_bytes": size,
    }


_PNG_CACHE: dict[tuple[str, str, int], bytes | None] = {}
_PNG_CACHE_BUDGET = 256 * 1024 * 1024


def _png_for(staging_dir: str, dds_path: str, max_tex_size: int) -> bytes | None:
    """Decode once per worker per build, keeping at most about 256 MB of PNGs (oldest dropped first)."""
    key = (staging_dir, os.path.normcase(os.path.abspath(dds_path)), max_tex_size)
    if key in _PNG_CACHE:
        return _PNG_CACHE[key]
    dds_mod = _get_dds_png()
    try:
        png = dds_mod.dds_file_to_png_bytes(dds_path, max_size=max_tex_size)
    except (dds_mod.DdsError, OSError):
        png = None
    _PNG_CACHE[key] = png
    while sum(len(v or b"") for v in _PNG_CACHE.values()) > _PNG_CACHE_BUDGET and len(_PNG_CACHE) > 1:
        del _PNG_CACHE[next(iter(_PNG_CACHE))]
    return png


def _build_chunk_job(job: tuple) -> tuple[dict, int, int, int]:
    """Write one chunk: every distinct (model, mirrored) once, drawn by its instances. Returns
    (chunk record, textures embedded, textures skipped, bytes), after checking the written file."""
    (chunk_id, staging_dir, mode, insts, packs, mat_to_dds, textures, max_tex_size) = job
    glb_mod = _get_glb_writer()
    groups: dict[tuple[int, bool], list] = {}
    for model_id, mirrored, t, q, sc in insts:
        groups.setdefault((model_id, mirrored), []).append((t, q, sc))
    loaded: dict[int, list[dict]] = {}
    meshes = []
    for (model_id, mirrored), instances in sorted(groups.items()):
        if model_id not in loaded:
            loaded[model_id] = _read_pack(packs[model_id])
        prims = loaded[model_id]
        if mirrored:
            prims = [_mirror_packed(p) for p in prims]
        meshes.append({"name": f"m{model_id}{'_mirror' if mirrored else ''}", "primitives": prims,
                       "instances": instances})
    materials: dict[str, dict] = {}
    skipped = 0
    if textures:
        keys = sorted({p["material"] for m in meshes for p in m["primitives"] if p["material"]})
        for key in keys:
            dds = mat_to_dds.get(key)
            png = _png_for(staging_dir, dds, max_tex_size) if dds else None
            if png is None:
                skipped += 1
                continue
            materials[key] = {"name": os.path.splitext(os.path.basename(dds))[0], "png_bytes": png,
                              "base_color_factor": [1.0, 1.0, 1.0, 1.0], "double_sided": True}
    path = os.path.join(staging_dir, f"{chunk_id}.glb")
    info = glb_mod.write_instanced_glb(path, meshes, materials or None, mode=mode)
    _verify_staged_chunk(path, int(info["bytes"]), glb_mod)
    rec = {
        "id": chunk_id,
        "url": f"world/chunks/{chunk_id}.glb",
        "bounds": info["bounds"],
        "models": info["instances"],
        "triangles": info["triangles"],
        "distinct_models": info["meshes"],
        "stored_triangles": info["stored_triangles"],
    }
    return rec, int(info.get("textures", 0)), skipped, int(info["bytes"])


def _run_jobs(fn, jobs, workers: int):
    """Yield fn(job) for every job, in order. With workers > 1 the jobs run in one process pool (fn and jobs
    must be picklable), and at most 4 * workers jobs are taken from ``jobs`` ahead of the results, so a lazy
    ``jobs`` iterator is never materialised."""
    if workers <= 1:
        for job in jobs:
            yield fn(job)
        return
    import collections
    from concurrent.futures import ProcessPoolExecutor
    workers = process_workers(workers)
    window = 4 * workers
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = collections.deque()
        try:
            for job in jobs:
                pending.append(pool.submit(fn, job))
                if len(pending) >= window:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
        except BaseException:
            for f in pending:
                f.cancel()
            raise


def _build_instanced_world(entries, ttpf_index, *, placements, lod_levels, region, out_dir, chunk_size,
                           textures, max_tex_size, base_color_param, workers, mode, glb_mod, limit,
                           lod_sets=False) -> dict:
    """``build_world --placements`` with shared geometry: memory holds about 100 bytes per placement plus the
    models of the chunk being written, not a copy of every placed model."""
    counts = _placement_counts()
    for outcome in ("placed", "placed_unverified", "unresolved_archetype", "key_mismatch", "model_not_loadable",
                    "degenerate_scale", "mirrored"):
        counts[outcome] = 0
    index = model_name_index(entries, _get_world_names())
    model_ids: dict[tuple[str, str], int] = {}
    model_src: list[tuple[str, dict]] = []
    inst_model = array.array("l")
    inst_flags = array.array("b")          # bit 0 mirrored, bit 1 unverified
    inst_trs = array.array("d")            # 10 per instance: translation xyz, rotation xyzw, scale xyz
    inst_tier = array.array("b")           # lodLevel, used with lod_sets
    child_lod: dict[int, array.array] = {}
    lod_dist: dict[int, array.array] = {}
    for row in iter_placements(placements, counts, lod_levels, region):
        lvl = row.get("lod_level")
        if lod_sets:
            for src, dst in (("child_lod_dist", child_lod), ("lod_dist", lod_dist)):
                v = row.get(src)
                if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
                    dst.setdefault(lvl, array.array("d")).append(float(v))
        found, outcome = resolve_placement(index, row["_arch"], row.get("key"))
        if found is None:
            counts[outcome] += 1
            continue
        t, q, sc, mirrored = placement_trs(row["_rot"], row["_sxy"], row["_sz"], row["_pos"])
        if sc[0] == 0 or sc[1] == 0:
            counts["degenerate_scale"] += 1
            continue
        models_dir, item = found
        ck = (models_dir, str(item.get("File")))
        mid = model_ids.get(ck)
        if mid is None:
            mid = model_ids[ck] = len(model_src)
            model_src.append(found)
        inst_model.append(mid)
        inst_flags.append((1 if mirrored else 0) | (2 if outcome == "placed_unverified" else 0))
        inst_trs.extend(t + q + sc)
        inst_tier.append(lvl if lod_sets else 0)

    abs_out = os.path.abspath(out_dir)
    chunks_dir = os.path.join(abs_out, "chunks")
    created_dirs = [d for d in (abs_out, chunks_dir) if not os.path.isdir(d)]
    os.makedirs(abs_out, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=CHUNK_STAGING_PREFIX, dir=abs_out)
    pack_dir = os.path.join(staging_dir, "packs")
    os.makedirs(pack_dir)
    try:
        jobs = []
        for mid, (models_dir, item) in enumerate(model_src):
            rel_file = item.get("File")
            obj_path = os.path.join(models_dir, rel_file) if isinstance(rel_file, str) and rel_file.strip() else ""
            tex_map = resolve_model_textures(models_dir, item, ttpf_index, base_color_param) if textures else {}
            jobs.append((mid, obj_path, _list_field(item, "MeshShaders"), tex_map, textures,
                         os.path.join(pack_dir, f"{mid}.lpk")))
        info: list[dict | None] = [None] * len(model_src)
        for mid, summary in _run_jobs(_pack_model_job, jobs, workers):
            info[mid] = summary
        del jobs

        cells: dict[tuple[int, int], array.array] = {}
        for i in range(len(inst_model)):
            summary = info[inst_model[i]]
            if summary is None:
                counts["model_not_loadable"] += 1
                continue
            counts["placed_unverified" if inst_flags[i] & 2 else "placed"] += 1
            counts["mirrored"] += inst_flags[i] & 1
            box = summary["box"]
            if inst_flags[i] & 1:
                box = [box[0], -box[4], box[2], box[3], -box[1], box[5]]
            v = inst_trs[i * 10:i * 10 + 10]
            b = glb_mod.instance_bounds(box[:3], box[3:], v[0:3], v[3:7], v[7:10])
            key = (inst_tier[i], math.floor(0.5 * (b[0] + b[3]) / chunk_size),
                   math.floor(0.5 * (b[2] + b[5]) / chunk_size))
            cells.setdefault(key, array.array("l")).append(i)
        used = {inst_model[i] for arr in cells.values() for i in arr}
        counts["distinct_models"] = len(used)
        if not cells and limit != 0:
            raise ValueError(f"No entity in {placements} could be placed ({counts}); output left unchanged")

        packs = {mid: os.path.join(pack_dir, f"{mid}.lpk") for mid in used}

        def chunk_id(tier, ix, iz):
            return f"c_{LOD_TIER_NAMES[tier]}_{ix}_{iz}" if lod_sets else f"c_{ix}_{iz}"

        def chunk_jobs():
            for (tier, ix, iz), arr in sorted(cells.items()):
                insts = []
                mids = set()
                for i in arr:
                    v = inst_trs[i * 10:i * 10 + 10]
                    mid = inst_model[i]
                    mids.add(mid)
                    insts.append((mid, bool(inst_flags[i] & 1), tuple(v[0:3]), tuple(v[3:7]), tuple(v[7:10])))
                dds = {}
                for mid in mids:
                    dds.update(info[mid]["mat_to_dds"])
                yield (chunk_id(tier, ix, iz), staging_dir, mode, insts, {m: packs[m] for m in mids}, dds,
                       textures, max_tex_size)

        chunks = []
        total_triangles = stored_triangles = textures_embedded = textures_skipped = 0
        total_bytes = max_bytes = 0
        cell_keys = iter(sorted(cells))
        for rec, embedded, skipped, size in _run_jobs(_build_chunk_job, chunk_jobs(), workers):
            tier, ix, iz = next(cell_keys)
            if lod_sets:
                rec["lod"] = LOD_TIER_NAMES[tier]
                rec["cell"] = [ix * chunk_size, iz * chunk_size, (ix + 1) * chunk_size, (iz + 1) * chunk_size]
            chunks.append(rec)
            total_triangles += rec["triangles"]
            stored_triangles += rec["stored_triangles"]
            textures_embedded += embedded
            textures_skipped += skipped
            total_bytes += size
            max_bytes = max(max_bytes, size)
        shutil.rmtree(pack_dir, ignore_errors=True)

        bands = measured = None
        if lod_sets:
            present = sorted({k[0] for k in cells})
            bands, measured = lod_bands(child_lod, lod_dist, present)
            for name, m in measured.items():
                m["chunks"] = sum(1 for c in chunks if c["lod"] == name)
                m["placements"] = sum(c["models"] for c in chunks if c["lod"] == name)
        world_doc = {
            "spawn": _compute_spawn([c for c in chunks if c.get("lod", "hd") in ("hd", "orphan_hd")] or chunks),
            "chunk_size": float(chunk_size),
            "layout": "placements",
            "instancing": mode,
            "chunks": chunks,
            "stats": {
                "models": counts["placed"] + counts["placed_unverified"],
                "chunks": len(chunks),
                "triangles": total_triangles,
                "stored_triangles": stored_triangles,
                "chunk_bytes": total_bytes,
                "max_chunk_bytes": max_bytes,
                "textures_embedded": textures_embedded,
                "textures_skipped": textures_skipped,
                "placements": dict(counts),
            },
        }
        if lod_sets:
            world_doc["lod_bands"] = bands
            world_doc["stats"]["lod_tiers"] = measured
        _publish_world(staging_dir, chunks_dir, [f"{c['id']}.glb" for c in chunks],
                       os.path.join(abs_out, "world.json"), world_doc, glb_mod)
    except BaseException:
        if not _holds_recovery_data(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        for d in reversed(created_dirs):
            try:
                os.rmdir(d)
            except OSError:
                pass
        raise
    _discard_publication_staging(staging_dir)
    _prune_stale_chunks(chunks_dir, {f"{c['id']}.glb" for c in world_doc["chunks"]})
    return world_doc


SPAWN_SPAN_CAP = 1200.0


def _get_dds_png():
    global dds_png
    if dds_png is None:
        import dds_png as _dds_png
        dds_png = _dds_png
    return dds_png


def _get_glb_writer():
    global glb_writer
    if glb_writer is None:
        import glb_writer as _glb_writer
        glb_writer = _glb_writer
    return glb_writer


def _load_json_dict(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _list_field(record: dict, key: str) -> list:
    """Read an optional JSON array, treating other types as absent."""
    value = record.get(key)
    return value if isinstance(value, list) else []


def _is_model_manifest(path: str) -> bool:
    data = _load_json_dict(path)
    if data is None:
        return False
    items = data.get("Items")
    if not isinstance(items, list):
        return False
    # rpf9dump textures mode writes textures/manifest.json with Items containing
    # {"Dir": ..., "Textures": [...]} and no "File" field. Exclude those.
    parent_name = os.path.basename(os.path.dirname(os.path.abspath(path))).lower()
    if parent_name == "textures":
        return False
    if items:
        has_file = any(isinstance(it, dict) and "File" in it for it in items)
        all_tex_dicts = all(
            isinstance(it, dict) and "Dir" in it and "File" not in it
            for it in items
        )
        if not has_file and all_tex_dicts:
            return False
    return True


def discover_model_manifests(assets_dir: str) -> list[str]:
    """Find all rpf9dump model manifest.json files under assets_dir.

    Handles:
      - a single models/ directory containing manifest.json
      - an archive directory containing models/manifest.json
      - a whole <work>/assets tree containing */models/manifest.json
    Returns sorted unique absolute paths.
    """
    abs_root = os.path.abspath(assets_dir)
    if not os.path.exists(abs_root):
        return []

    found: dict[str, str] = {}

    if os.path.isfile(abs_root):
        if _is_model_manifest(abs_root):
            found[os.path.normcase(abs_root)] = abs_root
        return sorted(found.values())

    # 1. Direct models/ directory (manifest.json directly inside assets_dir).
    direct = os.path.join(abs_root, "manifest.json")
    if os.path.isfile(direct) and _is_model_manifest(direct):
        abs_direct = os.path.abspath(direct)
        found[os.path.normcase(abs_direct)] = abs_direct

    # 2. Archive directory (models/manifest.json) or whole assets tree.
    for dirpath, dirnames, filenames in os.walk(abs_root):
        dirnames.sort()
        if "manifest.json" in filenames and os.path.basename(dirpath).lower() == "models":
            cand = os.path.abspath(os.path.join(dirpath, "manifest.json"))
            if _is_model_manifest(cand):
                found[os.path.normcase(cand)] = cand

    return sorted(found.values())


def _is_nonzero_key(key: str) -> bool:
    """Return True for a non-empty texture key/hash that is not an all-zero sentinel."""
    return bool(key) and bool(key.strip("0"))


def index_ttpf_textures(assets_dir: str) -> dict[str, str]:
    """Index TTPF textures under assets_dir: uppercase 16-hex AssetKey -> absolute .dds path."""
    abs_root = os.path.abspath(assets_dir)
    index: dict[str, str] = {}
    if not os.path.isdir(abs_root):
        return index

    for dirpath, dirnames, filenames in os.walk(abs_root):
        dirnames.sort()
        if "manifest.json" not in filenames:
            continue
        manifest_path = os.path.join(dirpath, "manifest.json")
        data = _load_json_dict(manifest_path)
        if data is None:
            continue
        textures = data.get("Textures")
        if not isinstance(textures, list):
            continue
        for entry in textures:
            if not isinstance(entry, dict):
                continue
            raw_key = entry.get("AssetKey")
            raw_file = entry.get("File")
            if isinstance(raw_key, int) and not isinstance(raw_key, bool):
                key = f"{raw_key:016X}"
            elif isinstance(raw_key, str):
                key = raw_key.strip().upper()
            else:
                continue
            if not _is_nonzero_key(key) or not isinstance(raw_file, str):
                continue
            rel_file = raw_file.strip()
            if not rel_file:
                continue
            dds_path = os.path.abspath(os.path.join(dirpath, rel_file))
            if os.path.isfile(dds_path):
                index.setdefault(key, dds_path)

    return index


def _scan_embedded_dir(tex_dir: str) -> tuple[list[str], dict[int, str]]:
    """Return sorted .dds filenames in tex_dir and prefix-index -> absolute path map."""
    if not tex_dir or not os.path.isdir(tex_dir):
        return [], {}
    try:
        entries = sorted(os.listdir(tex_dir))
    except OSError:
        return [], {}
    dds_files: list[str] = []
    disk_by_index: dict[int, str] = {}
    for fname in entries:
        if not fname.lower().endswith(".dds"):
            continue
        full = os.path.abspath(os.path.join(tex_dir, fname))
        if not os.path.isfile(full):
            continue
        dds_files.append(fname)
        head, sep, _ = fname.partition("_")
        if sep and head.isdigit():
            disk_by_index.setdefault(int(head), full)
    return dds_files, disk_by_index


def _find_embedded_dds(tex_dir: str, dds_files: list[str], idx: int) -> str | None:
    prefix = f"{idx:03d}_"
    for fname in dds_files:
        if fname.startswith(prefix):
            return os.path.abspath(os.path.join(tex_dir, fname))
    return None


_PARAM_HASH_RE = re.compile(r"(?:0[xX])?([0-9A-Fa-f]{8})")


def _parse_param_hash(value: object) -> str:
    """Normalize a 32-bit shader parameter hash ("C8E8C282" or "0xc8e8c282") to uppercase hex."""
    if isinstance(value, str):
        match = _PARAM_HASH_RE.fullmatch(value.strip())
        if match:
            return match.group(1).upper()
    raise ValueError(f"base_color_param must be eight hex digits with optional 0x, got {value!r}")


def _ref_param_hash(ref: dict) -> str | None:
    """Return a texture reference's Param as uppercase 8-hex, or None if absent/malformed."""
    raw = ref.get("Param")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return f"{raw:08X}" if 0 <= raw <= 0xFFFFFFFF else None
    if isinstance(raw, str):
        match = _PARAM_HASH_RE.fullmatch(raw.strip())
        return match.group(1).upper() if match else None
    return None


def resolve_model_textures(
    models_dir: str,
    item: dict,
    ttpf_index: dict[str, str],
    base_color_param: str | None = None,
) -> dict[int, str]:
    """Resolve shader_index -> existing absolute .dds path for a model manifest item.

    With base_color_param (normalized uppercase 8-hex), only texture references whose Param
    matches are considered and the empty-Materials fallback is disabled; unmatched shaders
    are left out so they render as clay.
    """
    result: dict[int, str] = {}
    file_name = item.get("File")
    stem = os.path.splitext(file_name)[0] if isinstance(file_name, str) and file_name else ""
    tex_dir = os.path.join(models_dir, f"{stem}_tex") if stem else ""
    dds_files, disk_by_index = _scan_embedded_dir(tex_dir)

    emb_by_index: dict[int, str] = {}
    emb_by_key: dict[str, str] = {}
    emb_by_hash: dict[str, str] = {}

    for emb in _list_field(item, "Embedded"):
        if not isinstance(emb, dict):
            continue
        raw_idx = emb.get("Index")
        if raw_idx is None or isinstance(raw_idx, bool):
            continue
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        dds_path = _find_embedded_dds(tex_dir, dds_files, idx) if dds_files else None
        if not dds_path:
            continue
        emb_by_index[idx] = dds_path
        raw_key = emb.get("Key")
        if isinstance(raw_key, str):
            norm_key = raw_key.strip().upper()
            if _is_nonzero_key(norm_key):
                emb_by_key.setdefault(norm_key, dds_path)
        raw_hash = emb.get("Hash")
        if isinstance(raw_hash, str):
            norm_hash = raw_hash.strip().upper()
            if _is_nonzero_key(norm_hash):
                emb_by_hash.setdefault(norm_hash, dds_path)

    materials = _list_field(item, "Materials")
    if materials:
        for mat in materials:
            if not isinstance(mat, dict):
                continue
            raw_sidx = mat.get("ShaderIndex")
            if raw_sidx is None or isinstance(raw_sidx, bool):
                continue
            try:
                sidx = int(raw_sidx)
            except (TypeError, ValueError):
                continue
            if sidx in result:
                continue

            chosen: str | None = None
            for ref in _list_field(mat, "Textures"):
                if not isinstance(ref, dict):
                    continue
                if base_color_param is not None and _ref_param_hash(ref) != base_color_param:
                    continue
                # 1. Embedded by TextureIndex
                tidx = ref.get("TextureIndex")
                if isinstance(tidx, int) and not isinstance(tidx, bool) and tidx >= 0:
                    cand = emb_by_index.get(tidx) or disk_by_index.get(tidx)
                    if cand and os.path.isfile(cand):
                        chosen = cand
                        break
                # 2. Embedded by Id64 -> Key
                raw_id64 = ref.get("Id64")
                id64 = raw_id64.strip().upper() if isinstance(raw_id64, str) else ""
                if _is_nonzero_key(id64) and id64 in emb_by_key:
                    cand = emb_by_key[id64]
                    if os.path.isfile(cand):
                        chosen = cand
                        break
                # 3. Embedded by Name -> Hash
                raw_name = ref.get("Name")
                name_hash = raw_name.strip().upper() if isinstance(raw_name, str) else ""
                if _is_nonzero_key(name_hash) and name_hash in emb_by_hash:
                    cand = emb_by_hash[name_hash]
                    if os.path.isfile(cand):
                        chosen = cand
                        break
                # 4. External TTPF by uppercase Id64
                if _is_nonzero_key(id64) and id64 in ttpf_index:
                    cand = ttpf_index[id64]
                    if isinstance(cand, str) and os.path.isfile(cand):
                        chosen = os.path.abspath(cand)
                        break

            if chosen is not None:
                result[sidx] = chosen
    elif dds_files and base_color_param is None:
        # Fallback when Materials is empty/missing and no parameter is selected: map embedded indices and 0.
        for idx, path in emb_by_index.items():
            result[idx] = path
        for idx, path in disk_by_index.items():
            result.setdefault(idx, path)
        result.setdefault(0, os.path.abspath(os.path.join(tex_dir, dds_files[0])))

    return result


def _assign_materials(prims: list[dict], mesh_shaders: list, tex_map: dict, textures: bool) -> dict[str, str]:
    """Set each primitive's ``material`` key from its shader's resolved DDS; returns key -> DDS path."""
    mat_to_dds: dict[str, str] = {}
    for p in prims:
        gidx = p.get("group_index", 0)
        if isinstance(gidx, int) and not isinstance(gidx, bool) and 0 <= gidx < len(mesh_shaders):
            sidx = mesh_shaders[gidx]
        else:
            sidx = gidx
        has_uv = bool(p.get("uvs"))
        if (
            textures
            and has_uv
            and isinstance(sidx, int)
            and not isinstance(sidx, bool)
            and sidx in tex_map
        ):
            dds_path = os.path.abspath(tex_map[sidx])
            mat_key = f"tex_{os.path.normcase(dds_path)}"
            p["material"] = mat_key
            mat_to_dds[mat_key] = dds_path
        else:
            p["material"] = ""
    return mat_to_dds


def _translate_primitives(prims: list[dict], dx: float, dy: float, dz: float) -> None:
    """Translate primitive vertex positions in-place by (dx, dy, dz)."""
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        return
    for p in prims:
        pos = p.get("positions")
        if pos is None:
            continue
        if isinstance(pos, (bytes, bytearray, memoryview)):
            n_floats = len(pos) // 4
            if n_floats >= 3 and n_floats % 3 == 0:
                vals = array.array("f")
                vals.frombytes(pos)
                if sys.byteorder != "little":
                    vals.byteswap()
                for i in range(0, len(vals), 3):
                    vals[i] += dx
                    vals[i + 1] += dy
                    vals[i + 2] += dz
                if sys.byteorder != "little":
                    vals.byteswap()
                raw = vals.tobytes()
                p["positions"] = bytearray(raw) if isinstance(pos, bytearray) else raw
        elif isinstance(pos, array.array):
            for i in range(0, len(pos) - 2, 3):
                pos[i] += dx
                pos[i + 1] += dy
                pos[i + 2] += dz
        elif isinstance(pos, list):
            if not pos:
                continue
            first = pos[0]
            if isinstance(first, (int, float)):
                for i in range(0, len(pos) - 2, 3):
                    pos[i] += dx
                    pos[i + 1] += dy
                    pos[i + 2] += dz
            elif isinstance(first, tuple):
                p["positions"] = [(v[0] + dx, v[1] + dy, v[2] + dz) for v in pos]
            elif isinstance(first, list):
                for v in pos:
                    v[0] += dx
                    v[1] += dy
                    v[2] += dz
        elif isinstance(pos, tuple):
            if not pos:
                continue
            first = pos[0]
            if isinstance(first, (int, float)):
                out = list(pos)
                for i in range(0, len(out) - 2, 3):
                    out[i] += dx
                    out[i + 1] += dy
                    out[i + 2] += dz
                p["positions"] = tuple(out)
            else:
                p["positions"] = tuple((v[0] + dx, v[1] + dy, v[2] + dz) for v in pos)

        # Keep optional cached bounds consistent if the primitive carries them.
        if isinstance(p.get("min"), list) and len(p["min"]) == 3:
            p["min"] = [p["min"][0] + dx, p["min"][1] + dy, p["min"][2] + dz]
        if isinstance(p.get("max"), list) and len(p["max"]) == 3:
            p["max"] = [p["max"][0] + dx, p["max"][1] + dy, p["max"][2] + dz]
        if isinstance(p.get("bounds"), list) and len(p["bounds"]) == 6:
            b = p["bounds"]
            p["bounds"] = [
                b[0] + dx, b[1] + dy, b[2] + dz,
                b[3] + dx, b[4] + dy, b[5] + dz,
            ]


def _apply_grid_layout(models: list[dict], glb_mod) -> None:
    """Arrange models in non-overlapping XZ rows (4.0 m gap, min Y at 0.0)."""
    if not models:
        return
    cols = max(1, math.ceil(math.sqrt(len(models))))
    cur_x = 0.0
    cur_z = 0.0
    row_depth = 0.0
    col = 0

    for m in models:
        b = m["bounds"]
        sx = b[3] - b[0]
        sz = b[5] - b[2]
        dx = cur_x - b[0]
        dy = -b[1]
        dz = cur_z - b[2]
        _translate_primitives(m["prims"], dx, dy, dz)
        new_bounds = glb_mod.compute_bounds(m["prims"])
        if new_bounds is not None and len(new_bounds) == 6:
            m["bounds"] = new_bounds
        else:
            m["bounds"] = [
                b[0] + dx, b[1] + dy, b[2] + dz,
                b[3] + dx, b[4] + dy, b[5] + dz,
            ]

        row_depth = max(row_depth, sz)
        col += 1
        if col >= cols:
            col = 0
            cur_x = 0.0
            cur_z += row_depth + GRID_GAP
            row_depth = 0.0
        else:
            cur_x += sx + GRID_GAP


def _compute_spawn(chunks: list[dict]) -> list[float]:
    if not chunks:
        return [0.0, 20.0, 40.0]

    centers = [
        (0.5 * (c["bounds"][0] + c["bounds"][3]), 0.5 * (c["bounds"][2] + c["bounds"][5]))
        for c in chunks
    ]
    med_x = statistics.median(cx for cx, _ in centers)
    med_z = statistics.median(cz for _, cz in centers)

    best_idx = 0
    best_d2 = float("inf")
    for i, (cx, cz) in enumerate(centers):
        d2 = (cx - med_x) ** 2 + (cz - med_z) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i

    b = chunks[best_idx]["bounds"]
    cx = 0.5 * (b[0] + b[3])
    max_y = b[4]
    max_z = b[5]
    # The viewer loads chunks within 1400 m. Limiting the offsets to 300 m up and 720 m back
    # keeps the camera within 780 m of the chosen chunk's bounds, even for very large models.
    span = min(max(b[3] - b[0], b[4] - b[1], b[5] - b[2]), SPAWN_SPAN_CAP)
    return [
        float(cx),
        float(max_y + max(8.0, 0.25 * span)),
        float(max_z + max(15.0, 0.6 * span)),
    ]


def _write_json_atomic(path: str, data: dict, replace=None) -> None:
    """Publish ``data`` as JSON at ``path`` through a private staging file in the same folder.

    As for the GLB export (#425): an existing file named ``<path>.tmp`` is never used or removed, overlapping
    writers each stage their own file (the last replacement wins), and the published file keeps the mode of the
    one it replaces. ``replace`` defaults to ``os.replace``.
    """
    replace = replace or os.replace
    parent = os.path.dirname(os.path.abspath(path))
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=parent, prefix=".world-", suffix=".tmp", delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(data, f, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        # NamedTemporaryFile creates 0600; give world.json the mode of the file it replaces, or a new file's.
        os.chmod(tmp_path, published_mode(path))
        # Closed first: Windows cannot rename an open file.
        replace(tmp_path, path)
    except BaseException:
        try:
            if tmp_path is not None:
                os.remove(tmp_path)
        except OSError:
            pass
        raise


# Cell coordinates are written as decimal integers, without leading zeros; --lod-sets prefixes the tier name.
_LOD_TIER_PATTERN = "|".join(re.escape(name) for name in LOD_TIER_NAMES.values())
_CHUNK_FILE_RE = re.compile(
    rf"c_(?:(?:{_LOD_TIER_PATTERN})_)?(?:0|-?[1-9][0-9]*)_(?:0|-?[1-9][0-9]*)\.glb(?:\.tmp)?"
)


def _prune_stale_chunks(chunks_dir: str, keep_filenames: set[str]) -> None:
    """Remove stale chunk files and partial writes left by earlier builds."""
    try:
        entries = os.listdir(chunks_dir)
    except OSError:
        return
    for name in entries:
        if name in keep_filenames:
            continue
        if not _CHUNK_FILE_RE.fullmatch(name):
            continue
        full = os.path.join(chunks_dir, name)
        if os.path.isfile(full):
            try:
                os.remove(full)
            except OSError:
                pass


# A build stages its chunks in <out>/<prefix><random>/ (the same volume as <out>/chunks, so
# publishing is a rename) and keeps the chunks it replaces in <staging>/previous/ until
# world.json is committed.
CHUNK_STAGING_PREFIX = ".chunks-staging-"
_STAGING_BACKUP = "previous"
_STAGING_JOURNAL = "publication.json"
_STAGING_WORLD = "world.json"
_WORLD_BUILD_LOCK = "leonida-world-build-"


@contextmanager
def _exclusive_output(out_dir: str):
    """Hold one exclusive build lock for ``out_dir``.

    Staging cleanup, publication and stale-chunk pruning all work on the shared output
    directory, so two builds must not run there at once.  A killed process releases the
    OS file lock automatically.
    """
    # Keep the lock outside the output: a failed build must still leave no output directory,
    # and a successful one must contain only chunks/ and world.json.  Key it on the resolved
    # path so every spelling of one folder (8.3 names, junctions, symlinks) shares a lock; for
    # a folder that does not exist yet, realpath resolves the part that does and appends the rest.
    key = hashlib.sha256(
        os.path.normcase(os.path.realpath(out_dir)).encode("utf-8", "surrogatepass")
    ).hexdigest()[:20]
    lock_path = os.path.join(tempfile.gettempdir(), _WORLD_BUILD_LOCK + key)
    with open(lock_path, "a+b") as fh:
        fd = fh.fileno()
        locked = False
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
            elif msvcrt is not None:
                if os.fstat(fd).st_size == 0:
                    fh.write(b"\0")
                    fh.flush()
                fh.seek(0)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                locked = True
            else:
                raise RuntimeError("no file-lock implementation available")
        except OSError as exc:
            raise RuntimeError(f"another world_build is using {out_dir}") from exc
        try:
            yield
        finally:
            if locked:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    fh.seek(0)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _with_output_lock(fn):
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        out_dir = kwargs.get("out_dir")
        if out_dir is None and len(args) > 1:
            out_dir = args[1]
        if out_dir is None:
            raise TypeError("build_world requires an out_dir")
        abs_out = os.path.abspath(out_dir)
        with _exclusive_output(abs_out):
            _remove_stale_staging(abs_out)
            return fn(*args, **kwargs)
    return wrapped


def _holds_unrestored_chunks(staging_dir: str) -> bool:
    backup_dir = os.path.join(staging_dir, _STAGING_BACKUP)
    try:
        return bool(os.listdir(backup_dir))
    except FileNotFoundError:
        return False
    except OSError:
        return True  # cannot tell: keep it rather than risk deleting a recovery copy


def _holds_recovery_data(staging_dir: str) -> bool:
    return os.path.lexists(os.path.join(staging_dir, _STAGING_JOURNAL)) or _holds_unrestored_chunks(staging_dir)


def _discard_publication_staging(staging_dir: str) -> None:
    # Keep the journal until the backups are gone, including if the process dies during cleanup.
    shutil.rmtree(os.path.join(staging_dir, _STAGING_BACKUP), ignore_errors=True)
    if not _holds_unrestored_chunks(staging_dir):
        shutil.rmtree(staging_dir, ignore_errors=True)


def _restore_publication(staging_dir: str, chunks_dir: str, replace) -> None:
    """Undo chunk renames while the staged manifest has yet to be committed."""
    pending_world = os.path.join(staging_dir, _STAGING_WORLD)
    try:
        os.lstat(pending_world)
    except FileNotFoundError:
        return  # the manifest rename is the commit point
    journal = os.path.join(staging_dir, _STAGING_JOURNAL)
    backup_dir = os.path.join(staging_dir, _STAGING_BACKUP)
    try:
        with open(journal, encoding="utf-8") as f:
            doc = json.load(f)
        added = doc.get("added") if isinstance(doc, dict) else None
        if not isinstance(added, list):
            raise ValueError("publication journal has no list of added chunks")
        backups = os.listdir(backup_dir)
        if not all(
            isinstance(name, str) and name.endswith(".glb") and _CHUNK_FILE_RE.fullmatch(name)
            for name in [*added, *backups]
        ):
            raise ValueError("invalid chunk names in publication journal or backups")
        if not set(added).isdisjoint(backups):
            raise ValueError("a chunk cannot be both new and backed up")
        for name in added:
            try:
                os.unlink(os.path.join(chunks_dir, name))
            except FileNotFoundError:
                pass
        for name in backups:
            live = os.path.join(chunks_dir, name)
            if os.name == "nt":
                try:
                    mode = os.lstat(live).st_mode
                except FileNotFoundError:
                    pass
                else:
                    if stat.S_ISREG(mode) and not mode & stat.S_IWRITE:
                        # The uncommitted replacement inherited the original's mode.
                        # Windows cannot overwrite it until it is writable.
                        os.chmod(live, stat.S_IMODE(mode) | stat.S_IWRITE)
            replace(os.path.join(backup_dir, name), live)
        os.unlink(journal)
    except (OSError, ValueError) as exc:
        raise OSError(f"could not recover the previous world from {staging_dir}: {exc}") from exc


def _remove_stale_staging(out_dir: str) -> None:
    """Recover a stopped publication before any later build can touch the live chunks."""
    try:
        entries = os.listdir(out_dir)
    except FileNotFoundError:
        return
    for name in entries:
        path = os.path.join(out_dir, name)
        if not name.startswith(CHUNK_STAGING_PREFIX) or not os.path.isdir(path) or os.path.islink(path):
            continue
        if os.path.lexists(os.path.join(path, _STAGING_JOURNAL)):
            _restore_publication(path, os.path.join(out_dir, "chunks"), _get_glb_writer()._replace)
            _discard_publication_staging(path)
        elif not _holds_unrestored_chunks(path):
            # Older builds did not record whether they committed. Preserve their backups for manual recovery.
            shutil.rmtree(path, ignore_errors=True)


def _verify_staged_chunk(path: str, expected_size: int, glb_mod) -> None:
    """Raise GlbError unless the staged file is the complete GLB that write_glb reported."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) != expected_size:
        raise glb_mod.GlbError(f"staged chunk {path} is {len(data)} bytes, expected {expected_size}")
    glb_mod.read_glb_summary(data)


def _publish_world(
    staging_dir: str,
    chunks_dir: str,
    chunk_names: list[str],
    world_path: str,
    world_doc: dict,
    glb_mod,
) -> None:
    """Move the verified staged chunks into chunks_dir, then commit world.json.

    Record which chunks had no predecessor before moving any files. The staged manifest
    stays in place until the final rename, so recovery can distinguish an incomplete
    publish from an interruption after the new world was committed.
    """
    backup_dir = os.path.join(staging_dir, _STAGING_BACKUP)
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    pending_world = os.path.join(staging_dir, _STAGING_WORLD)
    _write_json_atomic(pending_world, world_doc)
    os.chmod(pending_world, published_mode(world_path))
    added = {name for name in chunk_names if not os.path.lexists(os.path.join(chunks_dir, name))}
    _write_json_atomic(os.path.join(staging_dir, _STAGING_JOURNAL), {"added": sorted(added)})
    try:
        for name in chunk_names:
            live = os.path.join(chunks_dir, name)
            staged = os.path.join(staging_dir, name)
            os.chmod(staged, published_mode(live))
            if name not in added:
                glb_mod._replace(live, os.path.join(backup_dir, name))
            glb_mod._replace(staged, live)
        glb_mod._replace(pending_world, world_path)
    except BaseException:
        _restore_publication(staging_dir, chunks_dir, glb_mod._replace)
        raise


@_with_output_lock
def build_world(
    assets_dir: str,
    out_dir: str,
    *,
    chunk_size: float = 256.0,
    grid_layout: bool = False,
    up_axis: str = "z",
    max_tex_size: int = 512,
    textures: bool = True,
    limit: int | None = None,
    jobs: int = 1,
    base_color_param: str | None = None,
    placements: str | None = None,
    lod_levels=DEFAULT_LOD_LEVELS,
    region=None,
    instancing: str = "gpu",
    lod_sets: bool = False,
) -> dict:
    """Build world.json and chunk .glb files from rpf9dump asset exports.

    With ``placements`` (a ``world_index.py --entities-out`` file), every selected entity whose archetype
    resolves to an exported model becomes one placed instance; models are not placed at their own coordinates.
    world.json's stats then carry a ``placements`` block with the counts of each outcome.

    Unless limit=0 explicitly requests an empty world, raise ValueError before changing
    the output if no readable model manifests or no usable selected models are found.
    Individual unusable models are still skipped when other selected models can load.

    Chunks are written to a staging directory inside out_dir and verified before any live
    chunk is replaced. Publishing renames one chunk at a time and then replaces world.json,
    so a reader during those renames can see old and new chunks mixed. A publication that
    raises before the manifest rename is rolled back. After an abrupt process exit, the
    next build rolls back an unfinished publication before reading its own inputs.
    One build at a time may use an output directory: a second build refuses with a clear error
    instead of removing the first build's staging.
    """
    if chunk_size <= 0 or not math.isfinite(chunk_size):
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    if up_axis not in ("z", "y"):
        raise ValueError(f"up_axis must be 'z' or 'y', got {up_axis!r}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit!r}")
    if max_tex_size < 0:
        raise ValueError(f"max_tex_size must be non-negative, got {max_tex_size!r}")
    if base_color_param is not None:
        base_color_param = _parse_param_hash(base_color_param)
    if placements is not None:
        if grid_layout:
            raise ValueError("--placements and --grid-layout cannot be combined")
        if up_axis != "z":
            raise ValueError("--placements needs the rpf9dump OBJ axes (--up z)")
        if not os.path.isfile(placements):
            raise FileNotFoundError(f"placements file does not exist: {placements}")
        if region is not None and (len(region) != 4 or not (region[0] < region[2] and region[1] < region[3])):
            raise ValueError(f"region must be MINX,MINY,MAXX,MAXY with min < max, got {region!r}")
        if instancing not in INSTANCING_MODES:
            raise ValueError(f"instancing must be one of {INSTANCING_MODES}, got {instancing!r}")
        if lod_sets:
            if instancing == "bake":
                raise ValueError("--lod-sets needs --instancing gpu or nodes")
            lod_levels = tuple(lod_levels)
            for level in lod_levels:
                if level not in LOD_TIER_NAMES:
                    raise ValueError(f"--lod-sets does not recognize lodLevel {level!r}; use levels 0..6")
    elif lod_sets:
        raise ValueError("--lod-sets needs --placements")
    workers = max(1, int(jobs))

    if not os.path.exists(assets_dir):
        raise FileNotFoundError(f"assets path does not exist: {assets_dir}")

    glb_mod = _get_glb_writer()
    dds_mod = _get_dds_png() if textures else None

    manifests = discover_model_manifests(assets_dir)
    if not manifests and limit != 0:
        raise ValueError(f"No readable model manifests found in {assets_dir}; output left unchanged")
    entries: list[tuple[str, dict]] = []
    for mpath in manifests:
        data = _load_json_dict(mpath)
        if data is None:
            continue
        items = data.get("Items")
        if not isinstance(items, list):
            continue
        models_dir = os.path.dirname(mpath)
        for item in items:
            if isinstance(item, dict):
                entries.append((models_dir, item))

    if limit is not None:
        entries = entries[:limit]

    ttpf_index = index_ttpf_textures(assets_dir) if textures else {}

    if placements is not None and instancing != "bake":
        return _build_instanced_world(
            entries, ttpf_index, placements=placements, lod_levels=lod_levels, region=region, out_dir=out_dir,
            chunk_size=chunk_size, textures=textures, max_tex_size=max_tex_size,
            base_color_param=base_color_param, workers=workers, mode=instancing, glb_mod=glb_mod, limit=limit,
            lod_sets=lod_sets,
        )

    def load_model(models_dir: str, item: dict, axis: str) -> dict | None:
        rel_file = item.get("File")
        if not isinstance(rel_file, str) or not rel_file.strip():
            return None
        obj_path = os.path.join(models_dir, rel_file)
        if not os.path.isfile(obj_path):
            return None
        try:
            prims = glb_mod.parse_obj(obj_path, is_path=True, up_axis=axis)
        except (OSError, glb_mod.GlbError, ValueError):
            return None
        if not prims:
            return None
        bounds = glb_mod.compute_bounds(prims)
        if bounds is None or len(bounds) != 6 or not all(math.isfinite(v) for v in bounds):
            return None

        tex_map = (
            resolve_model_textures(models_dir, item, ttpf_index, base_color_param)
            if textures
            else {}
        )
        mat_to_dds = _assign_materials(prims, _list_field(item, "MeshShaders"), tex_map, textures)
        return {"prims": prims, "bounds": bounds, "mat_to_dds": mat_to_dds}

    loaded_models: list[dict] = []
    placement_stats: dict | None = None
    if placements is None:
        for models_dir, item in entries:
            model = load_model(models_dir, item, up_axis)
            if model is not None:
                loaded_models.append(model)
    else:
        rows, placement_stats = load_placements(placements, lod_levels, region)
        index = model_name_index(entries, _get_world_names())
        cache: dict[tuple[str, str], dict | None] = {}
        used: set[tuple[str, str]] = set()
        for outcome in ("placed", "placed_unverified", "unresolved_archetype", "key_mismatch", "model_not_loadable"):
            placement_stats[outcome] = 0
        for row in rows:
            found, outcome = resolve_placement(index, row["_arch"], row.get("key"))
            if found is None:
                placement_stats[outcome] += 1
                continue
            models_dir, item = found
            ck = (models_dir, str(item.get("File")))
            if ck not in cache:
                cache[ck] = load_model(models_dir, item, "y")   # RAGE axes; converted after the transform
            model = cache[ck]
            if model is None:
                placement_stats["model_not_loadable"] += 1
                continue
            prims = place_primitives(model["prims"], row["_rot"], row["_sxy"], row["_sz"], row["_pos"])
            bounds = glb_mod.compute_bounds(prims)
            if bounds is None or len(bounds) != 6 or not all(math.isfinite(v) for v in bounds):
                placement_stats["model_not_loadable"] += 1
                continue
            loaded_models.append({"prims": prims, "bounds": bounds, "mat_to_dds": model["mat_to_dds"]})
            placement_stats[outcome] += 1
            used.add(ck)
        placement_stats["distinct_models"] = len(used)

    if not loaded_models and limit != 0 and placements is not None:
        raise ValueError(
            f"No entity in {placements} could be placed ({placement_stats}); output left unchanged"
        )
    if not loaded_models and limit != 0:
        raise ValueError(
            f"No usable models in {assets_dir} ({len(entries)} selected manifest items); "
            "check model File fields and OBJ geometry; output left unchanged"
        )

    if grid_layout:
        _apply_grid_layout(loaded_models, glb_mod)

    cells: dict[tuple[int, int], list[dict]] = {}
    for m in loaded_models:
        b = m["bounds"]
        cx = 0.5 * (b[0] + b[3])
        cz = 0.5 * (b[2] + b[5])
        cell_key = (math.floor(cx / chunk_size), math.floor(cz / chunk_size))
        cells.setdefault(cell_key, []).append(m)

    abs_out = os.path.abspath(out_dir)
    chunks_dir = os.path.join(abs_out, "chunks")
    created_dirs = [d for d in (abs_out, chunks_dir) if not os.path.isdir(d)]
    os.makedirs(abs_out, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=CHUNK_STAGING_PREFIX, dir=abs_out)
    try:
        world_doc = _build_staged_world(
            staging_dir, chunks_dir, os.path.join(abs_out, "world.json"), cells, loaded_models,
            glb_mod=glb_mod, dds_mod=dds_mod, textures=textures, max_tex_size=max_tex_size,
            workers=workers, chunk_size=chunk_size, grid_layout=grid_layout,
            placement_stats=placement_stats,
        )
    except BaseException:
        if not _holds_recovery_data(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)
        for d in reversed(created_dirs):
            try:
                os.rmdir(d)  # only if this build created it and left it empty
            except OSError:
                pass
        raise
    _discard_publication_staging(staging_dir)
    _prune_stale_chunks(chunks_dir, {f"{c['id']}.glb" for c in world_doc["chunks"]})
    return world_doc


def _build_staged_world(
    staging_dir: str,
    chunks_dir: str,
    world_path: str,
    cells: dict[tuple[int, int], list[dict]],
    loaded_models: list[dict],
    *,
    glb_mod,
    dds_mod,
    textures: bool,
    max_tex_size: int,
    workers: int,
    chunk_size: float,
    grid_layout: bool,
    placement_stats: dict | None = None,
) -> dict:
    """Write every chunk into staging_dir, verify them all, then publish them with world.json."""
    # Thread-safe single-flight PNG decode cache keyed by normalized DDS path.
    tex_lock = threading.Lock()
    tex_cache: dict[str, bytes | None] = {}
    tex_inflight: dict[str, threading.Event] = {}

    def decode_dds_cached(dds_path: str) -> bytes | None:
        assert dds_mod is not None
        norm = os.path.normcase(os.path.abspath(dds_path))
        with tex_lock:
            if norm in tex_cache:
                return tex_cache[norm]
            ev = tex_inflight.get(norm)
            if ev is None:
                ev = threading.Event()
                tex_inflight[norm] = ev
                owner = True
            else:
                owner = False
        if not owner:
            ev.wait()
            with tex_lock:
                return tex_cache.get(norm)

        png_bytes: bytes | None = None
        try:
            png_bytes = dds_mod.dds_file_to_png_bytes(dds_path, max_size=max_tex_size)
        except (dds_mod.DdsError, OSError):
            png_bytes = None
        finally:
            with tex_lock:
                tex_cache[norm] = png_bytes
                tex_inflight.pop(norm, None)
            ev.set()
        return png_bytes

    def build_cell(cell_key: tuple[int, int], cell_models: list[dict]) -> tuple[dict, int, int, int]:
        ix, iz = cell_key
        cell_prims: list[dict] = []
        cell_dds: dict[str, str] = {}
        for m in cell_models:
            cell_prims.extend(m["prims"])
            cell_dds.update(m["mat_to_dds"])

        materials: dict[str, dict] = {}
        skipped_keys: set[str] = set()
        cell_skipped = 0

        if textures and cell_dds:
            for mat_key in sorted(cell_dds):
                dds_path = cell_dds[mat_key]
                png_bytes = decode_dds_cached(dds_path)
                if png_bytes is None:
                    cell_skipped += 1
                    skipped_keys.add(mat_key)
                else:
                    materials[mat_key] = {
                        "name": os.path.splitext(os.path.basename(dds_path))[0],
                        "png_bytes": png_bytes,
                        "base_color_factor": [1.0, 1.0, 1.0, 1.0],
                        "double_sided": True,
                    }

        if skipped_keys:
            for p in cell_prims:
                if p.get("material") in skipped_keys:
                    p["material"] = ""

        chunk_id = f"c_{ix}_{iz}"
        glb_path = os.path.join(staging_dir, f"{chunk_id}.glb")
        info = glb_mod.write_glb(glb_path, cell_prims, materials=materials if materials else None)
        chunk_rec = {
            "id": chunk_id,
            "url": f"world/chunks/{chunk_id}.glb",
            "bounds": info["bounds"],
            "models": len(cell_models),
            "triangles": info["triangles"],
        }
        return chunk_rec, int(info.get("textures", 0)), cell_skipped, int(info["bytes"])

    sorted_cells = sorted(cells.items(), key=lambda kv: kv[0])
    chunks: list[dict] = []
    total_triangles = 0
    textures_embedded = 0
    textures_skipped = 0

    if workers == 1 or len(sorted_cells) <= 1:
        cell_results = [build_cell(k, ms) for k, ms in sorted_cells]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(build_cell, k, ms) for k, ms in sorted_cells]
            try:
                cell_results = [f.result() for f in futures]
            except BaseException:
                for f in futures:
                    f.cancel()  # stop cells that have not started; running ones finish before cleanup
                raise

    for chunk_rec, _, _, size in cell_results:
        _verify_staged_chunk(os.path.join(staging_dir, f"{chunk_rec['id']}.glb"), size, glb_mod)

    for chunk_rec, embedded_count, skipped_count, _ in cell_results:
        chunks.append(chunk_rec)
        total_triangles += int(chunk_rec["triangles"])
        textures_embedded += embedded_count
        textures_skipped += skipped_count

    spawn = _compute_spawn(chunks)
    world_doc = {
        "spawn": spawn,
        "chunk_size": float(chunk_size),
        "layout": "grid" if grid_layout else ("placements" if placement_stats is not None else "spatial"),
        "chunks": chunks,
        "stats": {
            "models": len(loaded_models),
            "chunks": len(chunks),
            "triangles": total_triangles,
            "textures_embedded": textures_embedded,
            "textures_skipped": textures_skipped,
        },
    }
    if placement_stats is not None:
        world_doc["stats"]["placements"] = dict(placement_stats)

    _publish_world(staging_dir, chunks_dir, [f"{c['id']}.glb" for c in chunks], world_path, world_doc, glb_mod)
    return world_doc


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _non_negative_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid int value: {text!r}") from None
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _positive_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid float value: {text!r}") from None
    if value <= 0 or not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return value


def _param_hash(text: str) -> str:
    try:
        return _parse_param_hash(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid 32-bit hex value: {text!r}") from None


def _lod_levels(text: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(v, 0) for v in text.split(",") if v.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a comma-separated list of lodLevel numbers: {text!r}") from None
    if not levels or any(v < 0 or v > 255 for v in levels):
        raise argparse.ArgumentTypeError(f"lodLevel values must be 0..255, got {text!r}")
    return levels


def _region(text: str) -> tuple[float, float, float, float]:
    try:
        vals = tuple(float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"region must be MINX,MINY,MAXX,MAXY, got {text!r}") from None
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals) or not (vals[0] < vals[2] and vals[1] < vals[3]):
        raise argparse.ArgumentTypeError(f"region must be MINX,MINY,MAXX,MAXY with min < max, got {text!r}")
    return vals


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build spatial world.json and .glb chunks from rpf9dump asset exports."
    )
    ap.add_argument(
        "assets_dir",
        help="Path to a models/ dir, an archive dir, or a full <work>\\assets tree.",
    )
    ap.add_argument(
        "--out",
        required=True,
        metavar="WORLD_DIR",
        help="Output world directory for world.json and chunks/*.glb.",
    )
    ap.add_argument(
        "--chunk-size",
        type=_positive_float,
        default=256.0,
        metavar="METRES",
        help="Spatial chunk edge length in metres (default: %(default)s).",
    )
    ap.add_argument(
        "--grid-layout",
        action="store_true",
        help="Arrange models in non-overlapping XZ rows instead of world-space positions.",
    )
    ap.add_argument(
        "--up",
        choices=("z", "y"),
        default="z",
        help="Source OBJ up axis to convert to glTF Y-up (default: %(default)s).",
    )
    ap.add_argument(
        "--max-tex-size",
        type=_non_negative_int,
        default=512,
        metavar="PX",
        help="Maximum texture width/height in pixels; 0 disables downscaling (default: %(default)s).",
    )
    ap.add_argument(
        "--no-textures",
        action="store_true",
        help="Skip texture resolution and DDS decoding (output clay geometry only).",
    )
    ap.add_argument(
        "--limit",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help="Process at most N manifest items; 0 explicitly writes an empty world and removes old chunks.",
    )
    ap.add_argument(
        "--jobs",
        type=_positive_int,
        default=1,
        metavar="N",
        help="Number of parallel workers: threads for chunk building, or processes for packing models and "
        "writing chunks with --placements (default: %(default)s).",
    )
    ap.add_argument(
        "--base-color-param",
        type=_param_hash,
        default=None,
        metavar="HEX",
        help="Use only texture references with this 32-bit shader parameter hash as base colour "
        "(e.g. C8E8C282 for character materials); materials without a resolvable match stay clay. "
        "Default: first resolvable reference.",
    )
    ap.add_argument(
        "--placements",
        default=None,
        metavar="ENTITIES",
        help="Instance models at their map placements from a world_index.py --entities-out file "
        "instead of at their own coordinates.",
    )
    ap.add_argument(
        "--lod-levels",
        type=_lod_levels,
        default=DEFAULT_LOD_LEVELS,
        metavar="N,N",
        help="With --placements: entity lodLevel values to place (default: 0,5 = HD and ORPHAN_HD).",
    )
    ap.add_argument(
        "--region",
        type=_region,
        default=None,
        metavar="MINX,MINY,MAXX,MAXY",
        help="With --placements: only entities whose RAGE world X/Y position lies in this box (metres).",
    )
    ap.add_argument(
        "--instancing",
        choices=INSTANCING_MODES,
        default="gpu",
        help="With --placements: 'gpu' stores each model once per chunk and draws its placements with "
        "EXT_mesh_gpu_instancing (default); 'nodes' writes one plain glTF node per placement for loaders "
        "without that extension; 'bake' copies the geometry into every placement (memory grows with the "
        "number of placements).",
    )
    ap.add_argument(
        "--lod-sets",
        action="store_true",
        help="With --placements: one chunk set per LOD tier (HD, LOD, SLOD1-4, ORPHAN_HD). "
        "Use --lod-levels to select a subset of levels 0..6. The camera distance bands in world.json "
        "come from the entities' childLodDist/lodDist so the viewer shows one tier per grid cell.",
    )
    args = ap.parse_args(argv)
    lod_levels = args.lod_levels
    if args.lod_sets and not any(a == "--lod-levels" or a.startswith("--lod-levels=") for a in (argv or sys.argv[1:])):
        lod_levels = ALL_LOD_LEVELS

    try:
        result = build_world(
            args.assets_dir,
            args.out,
            chunk_size=args.chunk_size,
            grid_layout=args.grid_layout,
            up_axis=args.up,
            max_tex_size=args.max_tex_size,
            textures=not args.no_textures,
            limit=args.limit,
            jobs=args.jobs,
            base_color_param=args.base_color_param,
            placements=args.placements,
            lod_levels=lod_levels,
            region=args.region,
            instancing=args.instancing,
            lod_sets=args.lod_sets,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    st = result["stats"]
    print(
        f"world built: {st['models']} {'placements' if 'placements' in st else 'models'} -> {st['chunks']} chunks, "
        f"{st['triangles']} tris, {st['textures_embedded']} textures "
        f"({st['textures_skipped']} skipped) -> {os.path.abspath(args.out)}",
        flush=True,
    )
    if "placements" in st:
        pl = st["placements"]
        print("placements: " + ", ".join(f"{k} {pl[k]}" for k in sorted(pl)), flush=True)
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
