"""Export a skinned character part with its skeleton and decoded clips as an animated binary glTF (.glb).

Three readers meet here: the skeleton that a character fragment carries at drawable + 0x18, the skin streams of
a part drawable (docs/character_skeleton.md, ASSET-12) and the clip dictionaries of anim.rpf (anim_clips.py). Each
part mesh is bound to the skeleton through its model's bone list, and each clip track to a skeleton joint through
its bone hash. Nothing is invented: a part whose weighted bones are not in the skeleton is refused, and clip tracks
that name no joint are counted and left out.

Skeleton (measured on the 596-bone female and 560-bone male humanoid fragments of packs.rpf):

    header = drawable + 0x18      +0x28 bone records, +0x30 inverse bind matrices (64 bytes each),
                                  +0x40 u16 parent indices (0xFFFF = root), +0x6C/+0x6E u16 bone count
    bone record (0x50 bytes)      +0x00 rotation quaternion (x, y, z, w), +0x10 translation (x, y, z),
                                  +0x1C u32 joaat32(lowercase bone name), +0x20 scale (x, y, z),
                                  +0x30 name pointer, +0x46 u16 parent, +0x48 u16 own index

The default rotation, translation and scale are local to the parent. Composed down the hierarchy they give each
joint's model-space transform, and its inverse is the stored inverse bind matrix (the check is in `read_skeleton`).
The matrices are row-vector (D3D) matrices; stored row by row, they are glTF's column-major layout as is.

Part meshes: grmModel + 0x14 u16 bone count, grmModel + 0x28 pointer to that many u32 bone hashes (joaat32 of the
bone names). A vertex's bone indices (FVF semantics 20/21) index this list; its weights are the UNORM bytes of
semantics 16/17 (they sum to 255). Positions and the clustered index buffer are read as rpf9-explorer reads them.

Clips: a `.clip` lists animations, each played over its own [start, end] window at its rate. A track is
{joaat32(bone name), track type, format}: `bonerotation` (quaternion), `bonetranslation` (vec3) and `bonescale`
(vec3) drive the joint with that hash, and bone hash 0 is the mover (root motion), which drives an extra node
above the skeleton. Other track types (`genericcontrol`, facial, camera) are counted, not exported.

The skeleton, meshes and clips stay in RAGE's Z-up space; one root node turns it into glTF's Y-up.

Usage:
    python -I anim_glb.py <archive.rpf> <out.glb> --skeleton <offset> --part <offset>[:<child>] [--part ...]
                          [--anim <anim.rpf> --dictionary <offset> [--clip <name or 0xhash> ...]]
                          [--lod N] [--max-influences 4|8] [--no-root-motion] [--report <out.json>]
    python -I anim_glb.py --coverage <archive.rpf> --skeleton <offset> --anim <anim.rpf> --report <out.json>
                          [--jobs N]

Offsets are the resources' zlib stream offsets in the archive (the `0x...` offsets rpf9dump and skin_streams
report). `--part` takes a drawable, a fragment or a drawable dictionary; `:<child>` picks a dictionary child.
`--coverage` writes counts only: how many clip-dictionary bone tracks name a joint of the skeleton.
Usage: `python -I tools/anim_glb.py --help` lists the options.
"""
import argparse
import collections
import json
import math
import mmap
import os
import struct
import sys
import time
import zlib

# -I keeps the script directory off sys.path, so add it explicitly before importing the sibling readers.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_clips  # noqa: E402
import anim_index  # noqa: E402
import glb_writer  # noqa: E402
from pool_workers import process_workers  # noqa: E402
import skin_streams  # noqa: E402

SYSTEM = 0x50000000
GRAPHICS = 0x60000000
MAX_BONES = 4096
BONE_STRIDE = 0x50
WEIGHT_SLOTS = (16, 17)
INDEX_SLOTS = (20, 21)
# (element bytes, format byte) of the bone-index slots, as skin_streams measures them
INDEX_LAYOUTS = {(4, 30): "u8x4", (16, 3): "packed", (8, 12): "u16x4", (1, 62): "rigid-u8", (2, 57): "rigid-u16"}
TRACK_PATHS = {anim_index.joaat32("bonerotation"): "rotation", anim_index.joaat32("bonetranslation"): "translation",
               anim_index.joaat32("bonescale"): "scale"}
PATH_FORMATS = {"rotation": 0, "translation": 2, "scale": 2}
Z_UP_TO_Y_UP = [-math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]   # -90 degrees about X: (x, y, z) -> (x, z, -y)
COMPONENT_UNSIGNED_SHORT = 5123
COMPONENT_FLOAT = 5126


class ExportError(ValueError):
    """The input is not what the exporter reads; the message says which check failed."""


def require(condition, message):
    if not condition:
        raise ExportError(message)


def u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def sys_at(b, field, size, what):
    """A system pointer field resolved to an offset in the system pages, with `size` bytes in range."""
    p = u64(b, field)
    require(SYSTEM <= p < GRAPHICS, f"{what}: not a system pointer")
    off = p - SYSTEM
    require(off + size <= len(b), f"{what}: points past the system pages")
    return off


def gfx_at(sys, gfx, field, size, what):
    p = u64(sys, field)
    require(GRAPHICS <= p < GRAPHICS + 0x10000000, f"{what}: not a graphics pointer")
    off = p - GRAPHICS
    require(off + size <= len(gfx), f"{what}: points past the graphics pages")
    return off


def c_string(b, field):
    p = u64(b, field)
    if not SYSTEM <= p < GRAPHICS or p - SYSTEM >= len(b):
        return None
    at = p - SYSTEM
    end = b.find(b"\0", at, at + 256)
    if end < 0:
        return None
    return b[at:end].decode("latin-1")


# ---------------------------------------------------------------- resources


def load_resource(m, at):
    """(system pages, graphics pages) of the resource whose first zlib stream starts at `at`."""
    size = len(m)
    require(0 <= at < size, f"offset 0x{at:x} is outside the archive")
    data, used = skin_streams.inflate_at(m, at, size)
    require(data is not None and len(data) >= 192, f"no resource stream at 0x{at:x}")
    directory = sys_at(data, 8, 16, "page directory")
    system_pages, graphics_pages = data[directory + 8], data[directory + 9]
    require(0 < system_pages <= 32 and graphics_pages <= 32, "page directory counts")
    parts, fp = [data], at + used
    for _ in range(system_pages - 1 + graphics_pages):
        nxt = skin_streams.next_stream(m, fp, size)
        require(nxt is not None, "a page stream of the resource is missing")
        parts.append(nxt[0])
        fp = nxt[1] + nxt[2]
    return b"".join(parts[:system_pages]), b"".join(parts[system_pages:])


def drawable_at(sys, child):
    """System offset of the drawable to read: a dictionary child, a fragment's drawable or the drawable itself."""
    kind = skin_streams.resource_kind(sys)
    require(kind is not None, "not a drawable, fragment or drawable dictionary")
    found = list(skin_streams.drawable_offsets(sys, kind))
    if child is None:
        require(len(found) == 1, f"{kind} with {len(found)} drawables: pick one with :<child>")
        child = 0
    require(0 <= child < len(found), f"child {child} is not in 0..{len(found) - 1}")
    return found[child]


# ---------------------------------------------------------------- skeleton


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def quat_rotate(q, v):
    x, y, z, w = q
    vx, vy, vz = v
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + y * tz - z * ty, vy + w * ty + z * tx - x * tz, vz + w * tz + x * ty - y * tx)


def read_skeleton(sys, drawable):
    """-> list of joints {hash, name, parent, rotation, translation, scale, inverse_bind (16 floats)}.

    Also returns the largest distance between a joint's model-space position composed from the default
    transforms and the one its inverse bind matrix gives; `export` refuses a skeleton where they disagree."""
    header = sys_at(sys, drawable + 0x18, 0x70, "skeleton")
    count = u16(sys, header + 0x6C)
    require(0 < count <= MAX_BONES and count == u16(sys, header + 0x6E), "skeleton bone count")
    records = sys_at(sys, header + 0x28, count * BONE_STRIDE, "bone records")
    inverse = sys_at(sys, header + 0x30, count * 64, "inverse bind matrices")
    parents = sys_at(sys, header + 0x40, count * 2, "parent indices")
    joints = []
    for i in range(count):
        r = records + i * BONE_STRIDE
        parent = u16(sys, parents + 2 * i)
        require(u16(sys, r + 0x48) == i and u16(sys, r + 0x46) == parent, f"bone record {i} index/parent")
        require(parent == 0xFFFF or parent < i, f"bone {i} parent {parent} is not an earlier bone")
        m = struct.unpack_from("<16f", sys, inverse + 64 * i)
        require(all(math.isfinite(v) for v in m) and m[3] == m[7] == m[11] == 0, f"inverse bind matrix {i}")
        # RageSkeleton.cs:109 also refuses a degenerate matrix. The stored last column is (0, 0, 0, 1),
        # so the 4x4 determinant is the determinant of the upper-left 3x3 linear part. As in the C#,
        # abs() keeps mirrored binds; the epsilon rejects clearly non-invertible ones, not just float32 zero.
        det = (m[0] * (m[5] * m[10] - m[6] * m[9])
               - m[1] * (m[4] * m[10] - m[6] * m[8])
               + m[2] * (m[4] * m[9] - m[5] * m[8]))
        require(math.isfinite(det) and abs(det) >= 1e-12, f"inverse bind matrix {i} determinant")
        joints.append({
            "hash": u32(sys, r + 0x1C), "name": c_string(sys, r + 0x30), "parent": None if parent == 0xFFFF else parent,
            "rotation": struct.unpack_from("<4f", sys, r), "translation": struct.unpack_from("<3f", sys, r + 0x10),
            "scale": struct.unpack_from("<3f", sys, r + 0x20), "inverse_bind": list(m[:15]) + [1.0],
        })
    # Compose the default transforms; compare each joint's origin with the one its inverse bind matrix gives.
    world, worst = [], 0.0
    for j in joints:
        q, t = j["rotation"], j["translation"]
        if j["parent"] is None:
            world.append((q, t))
        else:
            pq, pt = world[j["parent"]]
            world.append((quat_mul(pq, q), tuple(a + b for a, b in zip(pt, quat_rotate(pq, t)))))
        m = j["inverse_bind"]
        rows, t_inv = (m[0:3], m[4:7], m[8:11]), m[12:15]
        origin = tuple(-sum(t_inv[k] * rows[c][k] for k in range(3)) for c in range(3))
        worst = max(worst, max(abs(a - b) for a, b in zip(origin, world[-1][1])))
    return joints, worst


# ---------------------------------------------------------------- part meshes


def read_positions(sys, gfx, vb, fvf, vertices):
    offset, size, fmt = u32(sys, fvf), sys[fvf + 0xD0], sys[fvf + 0x104]
    length = u32(sys, vb + 8) * u16(sys, vb + 12)
    require(length > 0 and 96 <= offset <= length and size * vertices <= length - offset, "position stream bounds")
    at = gfx_at(sys, gfx, vb + 24, length, "vertex buffer")
    data = gfx[at:at + length]
    if (size, fmt) == (12, 6):
        require(u32(data, 0) & 127 == 0x63, "float position descriptor")
        return [struct.unpack_from("<3f", data, offset + 12 * i) for i in range(vertices)], data
    if (size, fmt) == (8, 16):
        require(offset >= 128 and u32(data, 0) & 127 == 0x42, "quantized position descriptor")
        bias, scale = struct.unpack_from("<3f", data, 96), struct.unpack_from("<3f", data, 112)
        axis = u32(data, 108)
        require(axis <= 2 and min(scale) >= 0, "quantized position bias/scale")
        out = []
        for i in range(vertices):
            q = u64(data, offset + 8 * i)
            p = (bias[0] + scale[0] * ((q >> 8) & 0xFFFFFF), bias[1] + scale[1] * ((((q >> 32) & 0xFFFF) << 8) | (q & 0xFF)),
                 bias[2] + scale[2] * (q >> 48))
            out.append((p[2], p[1], p[0]) if axis == 0 else (p[1], p[2], p[0]) if axis == 1 else p)
        return out, data
    raise ExportError(f"position encoding {size}/{fmt} is not read")


def read_triangles(sys, gfx, geometry, vertices):
    """The clustered index buffer: cluster records, 7-bit corner triplets, then the vertex reference list."""
    descriptor = sys_at(sys, geometry + 64, 16, "index descriptor")
    ib = sys_at(sys, descriptor, 64, "index buffer")
    require(u16(sys, ib + 12) == 4, "index buffer stride")
    length = u32(sys, ib + 8) * 4
    at = gfx_at(sys, gfx, ib + 24, length, "index data")
    data = gfx[at:at + length]
    count_flags = u32(data, 0)
    palette = count_flags & 0x40000000
    clusters = count_flags & (0xFFFF if palette else 0x7FFFFFFF)
    triangles, refs = u32(data, 4), u32(data, 8)
    require(0 < clusters <= 2048 and 0 < triangles <= 65535 and 0 < refs <= 65535, "index buffer counts")
    ref_size = 2 if count_flags & 0x80000000 else 4
    tri_at = 12 + clusters * 16
    refs_at = tri_at + triangles * 4
    require(refs_at + refs * ref_size <= length, "index buffer length")
    fmt = "<%d%s" % (refs, "H" if ref_size == 2 else "I")
    references = struct.unpack_from(fmt, data, refs_at)
    require(max(references) < vertices, "vertex reference past the vertex count")
    indices, seen_refs, seen_tris = [], 0, 0
    for c in range(clusters):
        packed, offsets = u32(data, 12 + 16 * c), u32(data, 16 + 16 * c)
        n_vertices, n_tris = (packed & 127) + 1, ((packed >> 7) & 255) + 1
        v_start, t_start = offsets & 0xFFFF, offsets >> 16
        require(v_start == seen_refs and t_start == seen_tris, "cluster starts (meshes over 65,535 are not read)")
        for t in range(t_start, t_start + n_tris):
            word = u32(data, tri_at + 4 * t)
            require(word >> 21 == c, "triangle cluster tag")
            for corner in range(3):
                local = (word >> (7 * corner)) & 127
                require(local < n_vertices, "triangle corner past its cluster")
                indices.append(references[v_start + local])
        seen_refs += n_vertices
        seen_tris += n_tris
    require(seen_refs == refs and seen_tris == triangles, "clusters do not cover the index buffer")
    constant = count_flags & 0x20000000
    hint = (count_flags >> 16) & 0x1FFF
    return indices, (hint // 2 if palette and constant else None)


def read_skin(sys, fvf, data, vertices, constant_bone):
    """Per vertex: up to 8 (model bone index, weight 0..255) pairs with a non-zero weight."""
    slots = {}
    for i in WEIGHT_SLOTS + INDEX_SLOTS:
        size = sys[fvf + 0xD0 + i]
        if size:
            offset = u32(sys, fvf + 4 * i)
            require(offset + size * vertices <= len(data), f"skin slot {i} bounds")
            slots[i] = (size, sys[fvf + 0x104 + i], offset)
    index_slots = [i for i in INDEX_SLOTS if i in slots]
    if not index_slots:
        require(constant_bone is not None, "mesh has no bone-index stream and no constant binding")
        return [[(constant_bone, 255)] for _ in range(vertices)], "constant"
    layout = INDEX_LAYOUTS.get(slots[index_slots[0]][:2])
    require(layout is not None and all(slots[i][:2] == slots[index_slots[0]][:2] for i in index_slots),
            "bone-index stream format")
    weight_slots = [i for i in WEIGHT_SLOTS if i in slots]
    require(all(slots[i][:2] == (4, 28) for i in weight_slots), "weight stream format")
    if layout.startswith("rigid"):
        require(not weight_slots, "rigid index stream with weights")
        size, _fmt, offset = slots[index_slots[0]]
        code = "B" if size == 1 else "H"
        return [[(v, 255)] for v in struct.unpack_from(f"<{vertices}{code}", data, offset)], layout
    require(len(weight_slots) == len(index_slots), "weight and index stream counts differ")
    out = []
    for v in range(vertices):
        pairs = []
        for wi, ii in zip(weight_slots, index_slots):
            weights = data[slots[wi][2] + 4 * v:slots[wi][2] + 4 * v + 4]
            at = slots[ii][2]
            if layout == "u8x4":
                bones = data[at + 4 * v:at + 4 * v + 4]
            elif layout == "u16x4":
                bones = struct.unpack_from("<4H", data, at + 8 * v)
            else:   # packed: bits 0..11 of each u32
                bones = [w & 0xFFF for w in struct.unpack_from("<4I", data, at + 16 * v)]
            pairs.extend((b, w) for b, w in zip(bones, weights) if w)
        out.append(pairs)
    return out, layout


def read_part(sys, gfx, drawable, lod=0):
    """-> list of meshes {model, geometry, positions, indices, skin [(model bone, weight)], palette [hash], layout}."""
    lod_field = drawable + 80 + 8 * lod
    require(0 <= lod < 4 and u64(sys, lod_field), f"LOD {lod} is empty")
    lod_at = sys_at(sys, lod_field, 16, "LOD")
    models = u16(sys, lod_at + 8)
    require(0 < models <= 256 and models == u16(sys, lod_at + 10), "LOD model count")
    model_list = sys_at(sys, lod_at, models * 8, "model list")
    meshes = []
    for mi in range(models):
        model = sys_at(sys, model_list + 8 * mi, 56, "model")
        count, bones = u16(sys, model + 16), u16(sys, model + 20)
        require(0 < count <= 256 and count == u16(sys, model + 18) == u16(sys, model + 54), "geometry count")
        require(0 < bones <= MAX_BONES, f"model {mi} has no bone list (not skinned)")
        palette_at = sys_at(sys, model + 0x28, bones * 4, "model bone list")
        palette = list(struct.unpack_from(f"<{bones}I", sys, palette_at))
        geometries = sys_at(sys, model + 8, count * 8, "geometry list")
        for gi in range(count):
            geometry = sys_at(sys, geometries + 8 * gi, 160, "geometry")
            vb = sys_at(sys, geometry + 8, 80, "vertex buffer header")
            fvf = sys_at(sys, vb + 64, 0x13C, "FVF table")
            flags = u32(sys, fvf + 0x138)
            vertices = flags >> 10
            require(flags & 1 and 0 < vertices <= 1_000_000, "vertex count")
            positions, data = read_positions(sys, gfx, vb, fvf, vertices)
            indices, constant_bone = read_triangles(sys, gfx, geometry, vertices)
            skin, layout = read_skin(sys, fvf, data, vertices, constant_bone)
            for pairs in skin:
                for b, _w in pairs:
                    require(b < bones, "weighted bone index past the model's bone list")
            meshes.append({"model": mi, "geometry": gi, "positions": positions, "indices": indices, "skin": skin,
                           "palette": palette, "layout": layout})
    return meshes


def bind_part(meshes, joints, max_influences=8):
    """Model bone index -> skeleton joint index through the bone hashes; -> per mesh JOINTS/WEIGHTS sets.

    With `max_influences` 4, a vertex keeps its four largest weights, renormalised, in one set (three.js and
    other viewers read only JOINTS_0/WEIGHTS_0)."""
    by_hash = {}
    for i, j in enumerate(joints):
        by_hash[j["hash"]] = i if j["hash"] not in by_hash else -1
    for mesh in meshes:
        mapping = [by_hash.get(h) for h in mesh["palette"]]
        missing = {mesh["palette"][b] for pairs in mesh["skin"] for b, _w in pairs if mapping[b] is None or mapping[b] < 0}
        require(not missing, "weighted bones not in the skeleton: " + ", ".join(f"0x{h:08x}" for h in sorted(missing)))
        sets = 2 if max_influences > 4 and max((len(p) for p in mesh["skin"]), default=0) > 4 else 1
        joint_rows, weight_rows = [], []
        for pairs in mesh["skin"]:
            pairs = sorted(pairs, key=lambda p: -p[1])[:4 * sets]
            total = float(sum(w for _b, w in pairs))
            require(total > 0, "vertex with no weight")
            row_j = [mapping[b] for b, _w in pairs] + [0] * (4 * sets - len(pairs))
            row_w = [w / total for _b, w in pairs] + [0.0] * (4 * sets - len(pairs))
            joint_rows.append(row_j)
            weight_rows.append(row_w)
        mesh["joints"], mesh["weights"], mesh["sets"] = joint_rows, weight_rows, sets
    return meshes


# ---------------------------------------------------------------- clips


def contiguous_rows(m, offset):
    end = min(len(m), offset + (64 << 20))
    while 0 <= offset < end and anim_index.ZHDR.match(m, offset):
        data, used = skin_streams.inflate_at(m, offset, end)
        if data is None or len(data) < 64:
            break
        yield anim_index.classify(data, offset, used)
        offset = (offset + used + 15) & ~15


def load_dictionary(path, offset):
    """Parse the clip dictionary whose root stream starts at `offset` -> (System, dictionary)."""
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        streams = contiguous_rows(m, offset)
        first = next(streams, None)
        rows = [first] if first is not None else []
        groups, _unresolved, _gaps = anim_clips.group(m, rows, first_row_only=True, continuation_rows=streams)
    require(groups and rows[groups[0][0]]["offset"] == offset, f"no clip dictionary starts at 0x{offset:x}")
    data = b""
    with open(path, "rb") as fh:
        for k in groups[0][1]:
            off, stored, raw = anim_clips.segment(rows, k)
            fh.seek(off)
            data += fh.read(stored) if raw else zlib.decompressobj().decompress(fh.read(stored))
    s = anim_clips.System(data)
    return s, anim_clips.parse_dictionary(s)


def clip_label(clip):
    name = clip.get("name") or f"0x{clip['hash']:08x}"
    if name.startswith("pack:/"):
        name = name[6:]
    return name[:-5] if name.endswith(".clip") else name


def select_clips(dictionary, wanted):
    clips = [c for c in dictionary["clips"] if c["kind"] == "clip"]
    if not wanted:
        return clips
    out = []
    for w in wanted:
        wanted_hash = None
        if w.lower().startswith("0x"):
            try:
                wanted_hash = int(w, 16)
            except ValueError:
                wanted_hash = None
        hits = [c for c in clips if clip_label(c) == w or (wanted_hash is not None and c["hash"] == wanted_hash)]
        require(hits, f"clip {w!r} is not in the dictionary")
        out.extend(hits)
    return out


def clip_channels(s, dictionary, clip, joint_by_hash, root_motion, tally):
    """-> {(target, path): [(time, value)]} for one clip. target is a joint index or 'mover'."""
    animations = {a["offset"]: a for p in dictionary["packs"] for a in p["animations"]}
    channels = {}
    for entry in clip["animations"]:
        anim = animations.get(entry["animation"])
        if anim is None:
            tally["animations_not_local"] += 1
            continue
        tracks, frames = anim_clips.decode_animation(s, anim)
        rate = entry["rate"] if entry["rate"] and math.isfinite(entry["rate"]) and entry["rate"] > 0 else 1.0
        step = anim["duration"] / (frames - 1) if frames > 1 and anim["duration"] > 0 else 1.0 / 30
        start = entry["start"] if math.isfinite(entry["start"]) else 0.0
        end = entry["end"] if math.isfinite(entry["end"]) and entry["end"] >= start else anim["duration"]
        if end == start:
            keep = [min(range(frames), key=lambda k: abs(k * step - start))] if frames else []
            single_pose = True
        else:
            keep = [k for k in range(frames) if start - 1e-4 <= k * step <= end + 1e-4]
            single_pose = not keep
            if single_pose:
                target_time = 0.5 * (start + end)
                keep = [min(range(frames), key=lambda k: abs(k * step - target_time))] if frames else []
        if not keep:
            continue
        tally["animations"] += 1
        for t, (bone, ttype, fmt) in enumerate(anim["tracks"]):
            path = TRACK_PATHS.get(ttype)
            if path is None:
                tally["tracks_other_type"] += 1
                continue
            if fmt != PATH_FORMATS[path] or t not in tracks:
                tally["tracks_unexpected_format"] += 1
                continue
            if bone == 0:
                if not root_motion:
                    tally["tracks_mover_skipped"] += 1
                    continue
                target = "mover"
            else:
                target = joint_by_hash.get(bone)
                if target is None:
                    tally["tracks_bone_not_in_skeleton"] += 1
                    continue
            tally["tracks_exported"] += 1
            if (target, path) in channels:
                tally["tracks_overridden"] += 1
            # a frame kept by the 1e-4 tolerance before `start` would get a slightly negative time
            if single_pose:
                channels[(target, path)] = [(0.0, tuple(tracks[t][k])) for k in keep]
            else:
                channels[(target, path)] = [(max(0.0, (k * step - start) / rate), tuple(tracks[t][k])) for k in keep]
    return channels


# ---------------------------------------------------------------- glTF


def _view(buf, views, payload, target=None):
    return glb_writer._append_buffer_view(buf, views, payload, target=target)


def _accessor(accessors, view, component, count, kind, **extra):
    accessors.append(dict(bufferView=view, componentType=component, count=count, type=kind, **extra))
    return len(accessors) - 1


def _floats(values):
    return struct.pack(f"<{len(values)}f", *values)


def build(joints, meshes, clips, *, root_motion=True, generator="Leonida anim_glb"):
    """glTF JSON + binary chunk for the skeleton, the bound meshes and the clip channels -> GLB bytes."""
    buf, views, accessors = bytearray(), [], []
    nodes = [{"name": "rage_z_up", "rotation": Z_UP_TO_Y_UP, "children": []}]
    holder = nodes[0]
    mover = None
    if root_motion:
        mover = len(nodes)
        nodes.append({"name": "mover", "children": []})
        holder["children"].append(mover)
        holder = nodes[mover]
    first_joint = len(nodes)
    for i, j in enumerate(joints):
        node = {"name": j["name"] or f"bone_0x{j['hash']:08x}"}
        node["rotation"] = [round(v, 7) for v in j["rotation"]]
        node["translation"] = [round(v, 7) for v in j["translation"]]
        if any(abs(s - 1.0) > 1e-6 for s in j["scale"]):
            node["scale"] = [round(v, 7) for v in j["scale"]]
        nodes.append(node)
    for i, j in enumerate(joints):
        if j["parent"] is None:
            holder["children"].append(first_joint + i)
        else:
            nodes[first_joint + j["parent"]].setdefault("children", []).append(first_joint + i)
    ibm_view = _view(buf, views, _floats([v for j in joints for v in j["inverse_bind"]]))
    ibm = _accessor(accessors, ibm_view, COMPONENT_FLOAT, len(joints), "MAT4")
    skin = {"joints": list(range(first_joint, first_joint + len(joints))), "inverseBindMatrices": ibm}
    roots = [first_joint + i for i, j in enumerate(joints) if j["parent"] is None]
    if len(roots) == 1:
        skin["skeleton"] = roots[0]
    gltf_meshes, scene_nodes = [], [0]
    for mesh in meshes:
        packed = glb_writer.pack_primitive({"positions": mesh["positions"], "indices": mesh["indices"]})
        attributes, index_accessor = glb_writer._append_packed(buf, views, accessors, packed)
        for k in range(mesh["sets"]):
            jv = _view(buf, views, struct.pack(f"<{4 * len(mesh['joints'])}H",
                                               *[x for row in mesh["joints"] for x in row[4 * k:4 * k + 4]]),
                       target=glb_writer.TARGET_ARRAY_BUFFER)
            attributes[f"JOINTS_{k}"] = _accessor(accessors, jv, COMPONENT_UNSIGNED_SHORT, len(mesh["joints"]), "VEC4")
            wv = _view(buf, views, _floats([x for row in mesh["weights"] for x in row[4 * k:4 * k + 4]]),
                       target=glb_writer.TARGET_ARRAY_BUFFER)
            attributes[f"WEIGHTS_{k}"] = _accessor(accessors, wv, COMPONENT_FLOAT, len(mesh["weights"]), "VEC4")
        gltf_meshes.append({"name": mesh["name"], "primitives": [{"attributes": attributes, "indices": index_accessor,
                                                                  "mode": glb_writer.MODE_TRIANGLES}]})
        scene_nodes.append(len(nodes))
        # A skinned mesh's own transform is ignored (glTF 2.0, 3.7.3), so its node sits at the scene root.
        nodes.append({"name": mesh["name"], "mesh": len(gltf_meshes) - 1, "skin": 0})
    animations = []
    for name, channels in clips:
        samplers, out_channels = [], []
        for (target, path), keys in sorted(channels.items(), key=lambda kv: (str(kv[0][0]), kv[0][1])):
            time_bytes = _floats([t for t, _v in keys])
            times = struct.unpack(f"<{len(keys)}f", time_bytes)
            require(all(b > a for a, b in zip(times, times[1:])), "key times are not increasing")
            tv = _view(buf, views, time_bytes)
            ta = _accessor(accessors, tv, COMPONENT_FLOAT, len(times), "SCALAR", min=[min(times)], max=[max(times)])
            values = [x for _t, v in keys for x in v]
            vv = _view(buf, views, _floats(values))
            va = _accessor(accessors, vv, COMPONENT_FLOAT, len(keys), "VEC4" if path == "rotation" else "VEC3")
            samplers.append({"input": ta, "output": va, "interpolation": "LINEAR"})
            node = mover if target == "mover" else first_joint + target
            out_channels.append({"sampler": len(samplers) - 1, "target": {"node": node, "path": path}})
        if out_channels:
            animations.append({"name": name, "samplers": samplers, "channels": out_channels})
    gltf = {
        "asset": {"version": "2.0", "generator": generator},
        "scene": 0, "scenes": [{"nodes": scene_nodes}], "nodes": nodes, "skins": [skin], "meshes": gltf_meshes,
        "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(buf)}],
    }
    if animations:
        gltf["animations"] = animations
    return pack_glb(gltf, buf), gltf


def pack_glb(gltf, buf):
    buf = bytes(buf) + b"\0" * ((-len(buf)) % 4)
    gltf["buffers"][0]["byteLength"] = len(buf)
    text = json.dumps(gltf, separators=(",", ":"), allow_nan=False).encode("utf-8")
    text += b" " * ((-len(text)) % 4)
    total = 12 + 8 + len(text) + 8 + len(buf)
    return b"".join((struct.pack("<4sII", glb_writer.GLB_MAGIC, glb_writer.GLB_VERSION, total),
                     struct.pack("<II", len(text), glb_writer.CHUNK_TYPE_JSON), text,
                     struct.pack("<II", len(buf), glb_writer.CHUNK_TYPE_BIN), buf))


# ---------------------------------------------------------------- commands


def parse_offset(text):
    value, _, child = text.partition(":")
    try:
        return int(value, 0), (int(child, 0) if child else None)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an offset: {text!r}") from None


def same_path(first, second):
    return anim_clips.same_output_path(first, second)


def export(args):
    started = time.monotonic()
    report = {"archive": os.path.basename(args.archive), "parts": [], "clips": [], "counts": collections.Counter()}
    with open(args.archive, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        sys_pages, _gfx = load_resource(m, args.skeleton[0])
        joints, worst = read_skeleton(sys_pages, drawable_at(sys_pages, args.skeleton[1]))
        require(worst < 1e-3, f"default transforms disagree with the inverse bind matrices by {worst:.4g}")
        report["skeleton"] = {"offset": f"0x{args.skeleton[0]:x}", "joints": len(joints), "bind_error": worst}
        meshes = []
        for offset, child in args.part:
            sys_pages, gfx = load_resource(m, offset)
            drawable = drawable_at(sys_pages, child)
            name = c_string(sys_pages, drawable + 168) or f"part_0x{offset:x}"
            if name.endswith(".#dr"):
                name = name[:-4]
            part = bind_part(read_part(sys_pages, gfx, drawable, args.lod), joints, args.max_influences)
            for mesh in part:
                mesh["name"] = f"{name}_m{mesh['model']}_g{mesh['geometry']}"
            meshes.extend(part)
            report["parts"].append({"offset": f"0x{offset:x}", "child": child, "meshes": len(part),
                                    "vertices": sum(len(x["positions"]) for x in part),
                                    "triangles": sum(len(x["indices"]) // 3 for x in part),
                                    "layouts": sorted({x["layout"] for x in part})})
    require(meshes, "no part meshes")
    clips = []
    if args.anim:
        s, dictionary = load_dictionary(args.anim, args.dictionary)
        joint_by_hash = {}
        for i, j in enumerate(joints):
            joint_by_hash.setdefault(j["hash"], i)
        for clip in select_clips(dictionary, args.clip):
            tally = collections.Counter()
            channels = clip_channels(s, dictionary, clip, joint_by_hash, not args.no_root_motion, tally)
            clips.append((clip_label(clip), channels))
            report["clips"].append(dict(hash=f"0x{clip['hash']:08x}", channels=len(channels), **tally))
    data, gltf = build(joints, meshes, clips, root_motion=not args.no_root_motion)
    glb_writer._write_bytes_atomic(args.out, data)
    report["counts"].update(meshes=len(meshes), animations=len(gltf.get("animations", [])), bytes=len(data))
    report["seconds"] = round(time.monotonic() - started, 2)
    return report


def coverage_worker(job):
    """Count the bone tracks of the clip dictionaries in one slice of anim.rpf that name a skeleton joint."""
    path, segments_list, hashes = job
    counts = collections.Counter()
    hashes = set(hashes)
    for segments in segments_list:
        with open(path, "rb") as fh:
            data = b""
            for off, stored, raw in segments:
                fh.seek(off)
                data += fh.read(stored) if raw else zlib.decompressobj().decompress(fh.read(stored))
        try:
            d = anim_clips.parse_dictionary(anim_clips.System(data))
        except (anim_clips.DecodeError, struct.error):
            counts["dictionaries_not_read"] += 1
            continue
        counts["dictionaries"] += 1
        bones, hit = set(), set()
        for pack in d["packs"]:
            for anim in pack["animations"]:
                for bone, ttype, _fmt in anim["tracks"]:
                    if ttype not in TRACK_PATHS or bone == 0:
                        continue
                    counts["bone_tracks"] += 1
                    bones.add(bone)
                    if bone in hashes:
                        counts["bone_tracks_in_skeleton"] += 1
                        hit.add(bone)
        if bones:
            counts["dictionaries_with_bone_tracks"] += 1
            share = len(hit) / len(bones)
            counts["dictionaries_all_bones_in_skeleton"] += share == 1
            counts["dictionaries_90pct_bones_in_skeleton"] += share >= 0.9
            counts["dictionaries_no_bone_in_skeleton"] += not hit
    return counts


def coverage(args):
    started = time.monotonic()
    with open(args.archive, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        sys_pages, _gfx = load_resource(m, args.skeleton[0])
        joints, worst = read_skeleton(sys_pages, drawable_at(sys_pages, args.skeleton[1]))
    rows, _walk = anim_index.scan_rpf(args.anim)
    with open(args.anim, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        groups, _unresolved, _gaps = anim_clips.group(m, rows)
    segments = [[anim_clips.segment(rows, k) for k in parts] for _root, parts in groups]
    hashes = sorted({j["hash"] for j in joints})
    jobs = process_workers(max(1, args.jobs))   # a pool takes at most 61 workers on Windows
    slices = [(args.anim, segments[i::jobs], hashes) for i in range(jobs)]
    total = collections.Counter()
    if jobs == 1:
        total.update(coverage_worker(slices[0]))
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(jobs) as ex:
            for part in ex.map(coverage_worker, slices):
                total.update(part)
    return {"skeleton": {"offset": f"0x{args.skeleton[0]:x}", "joints": len(joints), "bind_error": worst},
            "counts": dict(sorted(total.items())), "seconds": round(time.monotonic() - started, 1)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive", help="archive holding the skeleton and the parts (packs.rpf)")
    ap.add_argument("out", nargs="?", help="output .glb (export mode)")
    ap.add_argument("--skeleton", type=parse_offset, required=True, help="resource with the skeleton, <offset>[:<child>]")
    ap.add_argument("--part", type=parse_offset, action="append", default=[], help="part drawable, <offset>[:<child>]")
    ap.add_argument("--anim", help="anim.rpf")
    ap.add_argument("--dictionary", type=lambda t: parse_offset(t)[0], help="export: clip dictionary offset in --anim")
    ap.add_argument("--clip", action="append", default=[],
                    help="export: name (without pack:/ and .clip) or 0xhash in the selected --anim dictionary")
    ap.add_argument("--lod", type=int, default=0, help="LOD level of the parts (0 = highest)")
    ap.add_argument("--max-influences", type=int, choices=(4, 8), default=8,
                    help="bone influences per vertex; 4 keeps the largest four, renormalised (three.js reads four)")
    ap.add_argument("--no-root-motion", action="store_true", help="leave out the mover (bone hash 0) tracks")
    ap.add_argument("--coverage", action="store_true", help="count clip bone tracks that name a skeleton joint")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for --coverage (at least 1)")
    ap.add_argument("--report", help="write a JSON report (counts only)")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    inputs = [args.archive] + ([args.anim] if args.anim else [])
    outputs = [p for p in (args.out, args.report) if p]
    for o in outputs:
        for i in inputs:
            if os.path.exists(o) and same_path(i, o):
                ap.error(f"{o} is an input; refusing to overwrite it")
    if args.report and os.path.isdir(args.report):
        ap.error(f"--report is a directory: {args.report}")
    if args.out and args.report and same_path(args.out, args.report):
        ap.error("the GLB and the report are the same path")
    if args.clip or args.dictionary is not None:
        if args.coverage:
            ap.error("--clip and --dictionary select clips for export, not --coverage")
        if not args.anim:
            ap.error("--clip and --dictionary require --anim")
    if args.coverage:
        if not args.anim or not args.report:
            ap.error("--coverage needs --anim and --report")
        if args.out:
            ap.error("--coverage writes no GLB")
    else:
        if not args.out or not args.part:
            ap.error("export needs <out.glb> and at least one --part")
        if args.anim and args.dictionary is None:
            ap.error("--anim needs --dictionary")
    try:
        report = coverage(args) if args.coverage else export(args)
    except OSError as e:
        print(f"ERROR: {e.filename or args.archive}: {e.strerror or e}", file=sys.stderr)
        return 2
    except (ExportError, anim_clips.DecodeError, struct.error, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.report:
        try:
            with open(args.report, "w", encoding="utf-8", newline="\n") as f:
                json.dump(report, f, indent=1, sort_keys=True)
                f.write("\n")
        except OSError as e:
            print(f"ERROR: cannot write report {args.report}: {e.strerror or e}", file=sys.stderr)
            return 2
    print(json.dumps({k: v for k, v in report.items() if k in ("counts", "skeleton", "seconds")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
