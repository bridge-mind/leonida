"""Index the `63 30 00 00` mesh pages (vertex blocks plus meshlet groups) in carved streams or an RPF9 archive.

`docs/story.md` lists 45 inflated streams in `shared_minigame000.rpf` that start with `63 30 00 00`. They are
geometry pages: a run of 16-byte-aligned chunks, each either a vertex block or a meshlet group.

Vertex block (header 0x60 bytes, first u32 is 0x00003063):
    +0x00..+0x58  23 slot words. A non-zero word is tag (low byte) | offset/2 (high 24 bits); the offset is
                  from the block start. Slot 0 is always tag 0x63 at offset 0x60.
    +0x5C         u32, 0x200 in every block measured.
    slot 0  tag 0x63  12 bytes/vertex  float3 position
    slot 1  tag 0x26   4 bytes/vertex  RGBA8 (absent in some blocks)
    slot 2  tag 0x63  12 bytes/vertex  float3 normal (absent in some blocks)
    slot 3  tag 0x42   8 bytes/vertex  float2
    slot 4  tag 0x42   8 bytes/vertex  float2 (absent in some blocks)
    a slot past 4 (seen: slot 12, tag 0x26) takes its tag's stride (the tag rule below)
    The streams are stored in slot order 0, 2, 1, 3, 4. Only slot 0 is required. The vertex count is the span
    from slot 0 to the next stream in offset order, divided by 12; the block ends 8 * count bytes after its last
    slot.

Meshlet group (first u32 has bit 31 set):
    +0   u32  0x80000000 | meshlet count K
    +4   u32  triangle count T
    +8   u32  vertex reference count V
    +12  K * 16-byte meshlet records:
         u32 w0: bits 0-6 = vertices - 1, bits 7-13 = triangles - 1, bits 14-15 zero, bits 16-31 = the
                 meshlet's bounding-sphere radius around the w2/w3 centre, as an f16 (measured on `42 40` geometry)
         u32 w1: low u16 = first vertex reference, high u16 = first triangle (running sums)
         u32 w2, w3: bits 0-11 and 12-23 of w2 and bits 0-11 of w3 are the meshlet's centre in x, y and z as
                     12-bit steps around 2047, (field - 2047) * step (measured on `42 40` geometry; the default
                     0x007FF7FF / 0x7FF is the origin). The step is 2^e / 16 m with e = w2 bits 28-31 (0 to 4
                     seen; larger meshlets further out), and w2 bits 24-27 are 0. w3 bits 14-30 are the normal cone: axis
                     octahedral u = bits 14-19, v = bits 20-25 (x / 31.5 - 1), sin(half-angle) = (31 - bits 26-30)
                     / 31; all zero = no cone.
    then T u32 triangles: three 7-bit meshlet-local indices (bits 0-6, 7-13, 14-20), meshlet index in bits 21-31
    then V u16 vertex references into a vertex block

`42 40 00 00` vertex block (header 0x80 bytes, first u32 is 0x00004042; ASSET-146):
    +0x00..+0x58  23 slot words, encoded as above. Slot 0 is always tag 0x42 at offset 0x80.
    +0x5C         u32, 0x200 (as in the `63 30` block)
    +0x60         f32 x3 bias        +0x6C  u32 axis order, 0, 1 or 2
    +0x70         f32 x3 scale       +0x7C  f32 UV scale for type-8 slots (4.0f in every block measured)
    A slot tag byte is (stride / 4) << 5 | type (tag_stride, SLOT_TYPES): 0x42 = 8 bytes, 0x21, 0x25, 0x26, 0x27 and
    0x28 = 4 bytes, and 0x63 in the `63 30` block = 12 bytes. The vertex count is the span
    from slot 0 to the next stream divided by 8; every stream is stride * count bytes and the block ends there.
    Measured over every `42 40` block of `shared_minigame000.rpf`: 14,671 fit this exactly, and each one is
    followed by a `42 40` block, a meshlet group, a `63 30` block or the end of the stream.
    slot 0  tag 0x42  8 bytes/vertex  packed position, read as one u64 q:
        a = bits 8-31 (24 bits), b = bits 32-47 << 8 | bits 0-7 (24 bits), c = bits 48-63 (16 bits)
        (a, b, c) = bias + scale * (a, b, c); axis 0 stores (z, y, x), axis 1 (y, z, x), axis 2 (x, y, z)
    This is the rule `third_party/rpf9-explorer/Core/RageModelDecoder.cs` uses for `(8, 16)` positions;
    `docs/story.md` has the independent check over the minigame archive.
    slot 2  the vertex normal, in world order whatever the axis word: tag 0x21 (type 1) is octahedral, u = bits
        0-9 and v = bits 10-19 as unorm (x / 511.5 - 1), bits 20-31 not decoded; tag 0x27 (type 7) is x, y, z in
        bits 0-9, 10-19, 20-29 as the same unorm.
    slots 11 and 18  tag 0x27 (type 7): a tangent, x, y, z as in slot 2's type 7, perpendicular to the normal; bits
        30-31 are 3 or 0, read here as the bitangent sign +1 or -1. Other 4-byte streams are not decoded.

A page starts with a `63 30` block, a `42 40` block or a group (continuation pages). The walk stops, without an
error, at a `42 40` block whose layout doesn't fit the rule above, at a group that continues past the end of
the stream, or where other data starts on a 256-byte boundary behind zero padding after the mesh chunks. Only
counts and layout checks are written; no vertex, index or float value leaves this tool.

Usage:
    python -I tools/meshlet_page.py <file|dir|archive.rpf> [...] [--json out.json] [--quiet]
        [--faces] [--centres] [--normals] [--cones]
    python -I tools/meshlet_page.py <graphics-segment> --draw-system <system-segment> --geometry-offset <offset> [--json out.json]
    python -I tools/meshlet_page.py <graphics-segment> --draw-system <system-segment> --drawable-offset <offset>
"""
import argparse
import json
import math
import mmap
import os
import re
import struct
import sys
import zlib

BLOCK_MAGIC = 0x3063
OTHER_BLOCK_MAGIC = 0x4042
HEADER_SIZE = 0x60
SLOT_COUNT = 23
SLOT_STRIDE = {0: 12, 1: 4, 2: 12, 3: 8, 4: 8}
SLOT_TAG = {0: 0x63, 1: 0x26, 2: 0x63, 3: 0x42, 4: 0x42}
OTHER_HEADER_SIZE = 0x80
# A slot tag byte is (stride / 4) << 5 | type, as rpf9-explorer's RageModelDecoder reads vertex slots. Types seen in
# shared_minigame000.rpf: 1, 2, 3, 6, 7, 8. Skinned blocks also use type 5 for joint indices.
SLOT_TYPES = {1: "octahedral normal, 10 + 10 bits unorm", 2: "2 x f32 or a packed position", 3: "3 x f32",
              5: "joint indices in skin slots", 6: "RGBA8",
              7: "3 x 10 bits unorm (the normal in slot 2, a tangent in slots 11 and 18)",
              8: "2 x snorm16 UV, scaled by the 42 40 header's +0x7C"}


def tag_stride(tag):
    """Bytes per vertex for a slot tag (bits 5-7 times 4), or None for a type not seen or a zero stride."""
    stride = ((tag >> 5) & 7) * 4
    return stride if stride and (tag & 15) in SLOT_TYPES else None
ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")


class PageError(ValueError):
    pass


def _u32(data, at):
    if at + 4 > len(data):
        raise PageError(f"truncated at +0x{at:x}")
    return struct.unpack_from("<I", data, at)[0]


def parse_block(data, at):
    """Parse a vertex block at `at`. Returns (record, end offset)."""
    if at + HEADER_SIZE > len(data):
        raise PageError(f"block header truncated at +0x{at:x}")
    words = struct.unpack_from("<24I", data, at)
    if words[0] != BLOCK_MAGIC:
        raise PageError(f"no block magic at +0x{at:x}")
    slots = {i: (w & 0xFF, (w >> 8) * 2) for i, w in enumerate(words[:SLOT_COUNT]) if w}
    if 0 not in slots:
        raise PageError(f"block at +0x{at:x} has no slot 0")
    if slots[0][1] != HEADER_SIZE:
        raise PageError(f"block at +0x{at:x}: slot 0 is not at +0x60")
    order = sorted(slots, key=lambda i: slots[i][1])
    if len(order) < 2 or order[0] != 0:
        raise PageError(f"block at +0x{at:x}: slot 0 is not followed by another stream")
    span = slots[order[1]][1] - slots[0][1]
    if span <= 0 or span % SLOT_STRIDE[0]:
        raise PageError(f"block at +0x{at:x}: slot {order[1]} offset is not 12 * count past slot 0")
    count = span // SLOT_STRIDE[0]
    for here, nxt in zip(order, order[1:]):
        stride = SLOT_STRIDE.get(here, tag_stride(slots[here][0]))   # a slot past 4: its tag's stride
        if stride is None or slots[nxt][1] - slots[here][1] != stride * count:
            raise PageError(f"block at +0x{at:x}: slot {here} size is not stride * count")
    last = order[-1]
    if last not in (3, 4):
        raise PageError(f"block at +0x{at:x}: last slot is {last}, expected a float2 slot")
    end = at + slots[last][1] + 8 * count
    if end > len(data):
        raise PageError(f"block at +0x{at:x} runs past the page end")
    record = {"kind": "block", "offset": at, "vertices": count, "flags": words[23],
              "slots": [[i, slots[i][0], slots[i][1]] for i in sorted(slots)],
              "tags_expected": all(SLOT_TAG.get(i) == t for i, (t, _) in slots.items())}
    return record, end


def parse_other_block(data, at):
    """Size a `42 40 00 00` vertex block at `at` from its slot words. Returns (record, end offset); the record holds
    counts and tags only."""
    if at + OTHER_HEADER_SIZE > len(data):
        raise PageError(f"42 40 block header truncated at +0x{at:x}")
    words = struct.unpack_from("<32I", data, at)
    if words[0] != OTHER_BLOCK_MAGIC:
        raise PageError(f"no 42 40 block magic at +0x{at:x}")
    slots = {i: (w & 0xFF, (w >> 8) * 2) for i, w in enumerate(words[:SLOT_COUNT]) if w}
    if any(tag_stride(tag) is None for tag, _ in slots.values()):
        raise PageError(f"42 40 block at +0x{at:x} has a slot tag with no known stride")
    order = sorted(slots, key=lambda i: slots[i][1])
    if len(order) < 2 or order[0] != 0 or slots[0][1] != OTHER_HEADER_SIZE:
        raise PageError(f"42 40 block at +0x{at:x}: slot 0 is not at +0x80 or has no stream after it")
    span = slots[order[1]][1] - OTHER_HEADER_SIZE
    if span <= 0 or span % tag_stride(slots[0][0]):
        raise PageError(f"42 40 block at +0x{at:x}: slot 0 span is not a whole number of vertices")
    count = span // tag_stride(slots[0][0])
    for here, nxt in zip(order, order[1:]):
        if slots[nxt][1] - slots[here][1] != tag_stride(slots[here][0]) * count:
            raise PageError(f"42 40 block at +0x{at:x}: slot {here} size is not stride * count")
    last = order[-1]
    end = at + slots[last][1] + tag_stride(slots[last][0]) * count
    if end > len(data):
        raise PageError(f"42 40 block at +0x{at:x} runs past the page end")
    record = {"kind": "block4240", "offset": at, "vertices": count, "flags": words[23], "axis": words[27],
              "slots": [[i, slots[i][0], slots[i][1]] for i in sorted(slots)]}
    return record, end


def positions_4240(data, block):
    """Decode a `42 40` block's slot-0 positions (the packed 8-byte form; see the module docstring). Returns a list of
    (x, y, z) floats. Raises PageError for an axis word other than 0, 1 or 2 or a slot 0 that isn't tag 0x42."""
    at, n = block["offset"], block["vertices"]
    slots = {i: (tag, off) for i, tag, off in block["slots"]}
    if slots.get(0, (None,))[0] != 0x42:
        raise PageError(f"42 40 block at +0x{at:x}: slot 0 is not a packed position (tag 0x42)")
    bias = struct.unpack_from("<3f", data, at + 0x60)
    axis = _u32(data, at + 0x6C)
    scale = struct.unpack_from("<3f", data, at + 0x70)
    if axis > 2:
        raise PageError(f"42 40 block at +0x{at:x}: axis word {axis} is not 0, 1 or 2")
    out = []
    for (q,) in struct.iter_unpack("<Q", data[at + slots[0][1]:at + slots[0][1] + 8 * n]):
        a = bias[0] + scale[0] * ((q >> 8) & 0xFFFFFF)
        b = bias[1] + scale[1] * (((q >> 32) & 0xFFFF) << 8 | (q & 0xFF))
        c = bias[2] + scale[2] * (q >> 48)
        out.append((c, b, a) if axis == 0 else (b, c, a) if axis == 1 else (a, b, c))
    return out


def group_triangles(data, group):
    """Yield each triangle of a parsed group as three vertex references (indices into its vertex block)."""
    at, k = group["offset"], group["meshlets"]
    rec_at, tri_at = at + 12, at + 12 + 16 * k
    refs = struct.unpack_from(f"<{group['vertex_refs']}H", data, tri_at + 4 * group["triangles"])
    for t in struct.unpack_from(f"<{group['triangles']}I", data, tri_at):
        vbase = _u32(data, rec_at + 16 * (t >> 21) + 4) & 0xFFFF
        yield refs[vbase + (t & 0x7F)], refs[vbase + ((t >> 7) & 0x7F)], refs[vbase + ((t >> 14) & 0x7F)]


def face_agreement(positions, triangles):
    """Count edges shared by exactly two triangles, and how many of those pairs have face normals within ~25 degrees
    (dot > 0.9) or opposed (dot < -0.9). A wrong position decode scatters the normals; a right one leaves most
    neighbours agreeing and almost none opposed."""
    normals = []
    for a, b, c in triangles:
        if max(a, b, c) >= len(positions):
            raise PageError("triangle references a vertex past its block")
        p, q, r = positions[a], positions[b], positions[c]
        u = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
        v = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        normals.append((n[0] / length, n[1] / length, n[2] / length) if length and math.isfinite(length) else None)
    faces = {}
    for i, (a, b, c) in enumerate(triangles):
        for e in ((a, b), (b, c), (c, a)):
            faces.setdefault((min(e), max(e)), []).append(i)
    out = {"shared_edges": 0, "agree": 0, "opposed": 0}
    for pair in faces.values():
        if len(pair) != 2 or normals[pair[0]] is None or normals[pair[1]] is None:
            continue
        d = sum(x * y for x, y in zip(normals[pair[0]], normals[pair[1]]))
        out["shared_edges"] += 1
        out["agree"] += d > 0.9
        out["opposed"] += d < -0.9
    return out


def centre_step(w2):
    """Metres per step of a meshlet record's centre fields: 2^e / 16 with the exponent e in w2 bits 28-31."""
    return 2.0 ** ((w2 >> 28) & 15) / 16


def meshlet_centre(w2, w3):
    """The centre stored in a meshlet record's words +8 and +12, in metres (see the module docstring)."""
    step = centre_step(w2)
    return ((w2 & 0xFFF) - 2047) * step, (((w2 >> 12) & 0xFFF) - 2047) * step, ((w3 & 0xFFF) - 2047) * step


def meshlet_radius(w0):
    """The bounding-sphere radius in a meshlet record's word 0 (bits 16-31, an f16), in metres."""
    return struct.unpack("<e", struct.pack("<H", w0 >> 16))[0]


def meshlet_cone(w3):
    """The normal cone in a meshlet record's word +12: ((x, y, z) axis, cos of the half-angle), or None when bits
    12-31 are all zero (no cone)."""
    if not w3 >> 12:
        return None
    axis = _octahedral(((w3 >> 14) & 63) / 31.5 - 1, ((w3 >> 20) & 63) / 31.5 - 1)
    s = (31 - ((w3 >> 26) & 31)) / 31
    return axis, math.sqrt(max(0.0, 1 - s * s))


def cone_agreement(data, group, positions):
    """For each meshlet with a cone: does the axis lie within ~25 degrees of the area-weighted mean face normal, and
    does every face normal lie inside the cone (dot with the axis at least cos(half-angle)), exactly or within 0.05?"""
    at, k = group["offset"], group["meshlets"]
    rec_at, tri_at = at + 12, at + 12 + 16 * k
    refs = struct.unpack_from(f"<{group['vertex_refs']}H", data, tri_at + 4 * group["triangles"])
    out = {"meshlets": k, "cones": 0, "axis_within_25_degrees": 0, "faces_inside": 0, "faces_inside_within_0_05": 0}
    for m in range(k):
        w0, w1, _, w3 = struct.unpack_from("<4I", data, rec_at + 16 * m)
        cone = meshlet_cone(w3)
        if cone is None:
            continue
        axis, cos_half = cone
        vb, tb, tc = w1 & 0xFFFF, w1 >> 16, ((w0 >> 7) & 0x7F) + 1
        normals, total = [], [0.0, 0.0, 0.0]
        for t in struct.unpack_from(f"<{tc}I", data, tri_at + 4 * tb):
            idx = [refs[vb + ((t >> s) & 0x7F)] for s in (0, 7, 14)]
            if max(idx) >= len(positions):
                raise PageError("meshlet references a vertex past its block")
            p, q, r = (positions[i] for i in idx)
            u = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
            v = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
            n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
            length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
            if length and math.isfinite(length):
                normals.append((n[0] / length, n[1] / length, n[2] / length))
                total = [a + b for a, b in zip(total, n)]
        length = math.sqrt(sum(c * c for c in total))
        if not normals or not length:
            continue
        out["cones"] += 1
        out["axis_within_25_degrees"] += sum(a * c / length for a, c in zip(axis, total)) > 0.9
        smallest = min(sum(a * c for a, c in zip(axis, n)) for n in normals)
        out["faces_inside"] += smallest >= cos_half
        out["faces_inside_within_0_05"] += smallest >= cos_half - 0.05
    return out


def centre_agreement(data, group, positions):
    """Compare each meshlet's stored centre with the centre of its vertices' box (axes whose difference rounds to 0
    or to at most one step of 1/16 or 1/8 m), and its stored radius with the farthest vertex from the stored centre
    (covers it, covers it once the centre's rounding of at most sqrt(3)/2 steps is allowed for, and is at most 1.25
    times it plus that rounding)."""
    at, k = group["offset"], group["meshlets"]
    rec_at = at + 12
    refs_at = rec_at + 16 * k + 4 * group["triangles"]
    refs = struct.unpack_from(f"<{group['vertex_refs']}H", data, refs_at)
    out = {"meshlets": k, "default_words": 0, "coarse_steps": 0, "axes": 3 * k, "axes_exact": 0,
           "axes_within_step": 0,
           "radius_covers": 0, "radius_covers_within_rounding": 0, "radius_tight": 0}
    for m in range(k):
        w0, w1, w2, w3 = struct.unpack_from("<4I", data, rec_at + 16 * m)
        out["default_words"] += w2 == 0x007FF7FF and w3 == 0x7FF
        vb, vc = w1 & 0xFFFF, (w0 & 0x7F) + 1
        pts = [positions[r] for r in refs[vb:vb + vc] if r < len(positions)]
        if len(pts) != vc:
            raise PageError("meshlet references a vertex past its block")
        stored, size = meshlet_centre(w2, w3), centre_step(w2)
        out["coarse_steps"] += (w2 >> 28) != 0
        for i in range(3):
            step = round(((min(p[i] for p in pts) + max(p[i] for p in pts)) / 2 - stored[i]) / size)
            out["axes_exact"] += step == 0
            out["axes_within_step"] += abs(step) <= 1
        radius, far = meshlet_radius(w0), max(math.dist(p, stored) for p in pts)
        slack = math.sqrt(3) / 2 * size
        out["radius_covers"] += radius >= far * (1 - 2 ** -11)
        out["radius_covers_within_rounding"] += radius + slack >= far
        out["radius_tight"] += radius <= 1.25 * far + slack
    return out


def _unorm10(x):
    return x / 511.5 - 1


def _octahedral(u, v):
    z = 1 - abs(u) - abs(v)
    x, y = u, v
    if z < 0:
        x, y = (1 - abs(v)) * math.copysign(1, u), (1 - abs(u)) * math.copysign(1, v)
    length = math.sqrt(x * x + y * y + z * z)
    return x / length, y / length, z / length


def normals_4240(data, block):
    """Decode a `42 40` block's slot-2 normals (see the module docstring). Returns a list of unit (x, y, z), or None
    when slot 2 is absent or of a type other than 1 or 7."""
    slots = {i: (tag, off) for i, tag, off in block["slots"]}
    if 2 not in slots or (slots[2][0] & 15) not in (1, 7) or tag_stride(slots[2][0]) != 4:
        return None
    tag, off = slots[2]
    at, n = block["offset"] + off, block["vertices"]
    out = []
    for (w,) in struct.iter_unpack("<I", data[at:at + 4 * n]):
        if tag & 15 == 1:
            out.append(_octahedral(_unorm10(w & 1023), _unorm10((w >> 10) & 1023)))
        else:
            v = [_unorm10((w >> s) & 1023) for s in (0, 10, 20)]
            length = math.sqrt(sum(c * c for c in v)) or 1.0
            out.append(tuple(c / length for c in v))
    return out


def tangents_4240(data, block):
    """Decode a `42 40` block's type-7 tangent slots (11 and 18). Returns {slot: [(x, y, z, sign), ...]}; sign is +1
    for bits 30-31 = 3, -1 for 0, and 0 for anything else."""
    out = {}
    for slot, tag, off in block["slots"]:
        if slot == 2 or tag & 15 != 7 or tag_stride(tag) != 4:
            continue
        at, n = block["offset"] + off, block["vertices"]
        rows = []
        for (w,) in struct.iter_unpack("<I", data[at:at + 4 * n]):
            v = [_unorm10((w >> s) & 1023) for s in (0, 10, 20)]
            length = math.sqrt(sum(c * c for c in v)) or 1.0
            rows.append((v[0] / length, v[1] / length, v[2] / length, {3: 1, 0: -1}.get(w >> 30, 0)))
        out[slot] = rows
    return out


def tangent_agreement(normals, tangents):
    """Counts for tangents against the same vertex's stored normal: |dot| < 0.05 (perpendicular) and the sign bits."""
    out = {"vertices": 0, "perpendicular": 0, "sign_plus": 0, "sign_minus": 0}
    for rows in tangents.values():
        for nrm, (x, y, z, sign) in zip(normals, rows):
            out["vertices"] += 1
            out["perpendicular"] += abs(nrm[0] * x + nrm[1] * y + nrm[2] * z) < 0.05
            out["sign_plus"] += sign == 1
            out["sign_minus"] += sign == -1
    return out


def normal_agreement(positions, normals, triangles):
    """For each vertex the triangles use, compare its stored normal with the sum of its faces' (area-weighted)
    normals. Counts vertices, dot > 0.99, dot > 0.9 and dot < 0."""
    acc = {}
    for a, b, c in triangles:
        if max(a, b, c) >= min(len(positions), len(normals)):
            raise PageError("triangle references a vertex past its block")
        p, q, r = positions[a], positions[b], positions[c]
        u = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
        v = (r[0] - p[0], r[1] - p[1], r[2] - p[2])
        n = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
        for k in (a, b, c):
            s = acc.setdefault(k, [0.0, 0.0, 0.0])
            s[0] += n[0]
            s[1] += n[1]
            s[2] += n[2]
    out = {"vertices": 0, "within_8_degrees": 0, "within_25_degrees": 0, "opposed": 0}
    for k, s in acc.items():
        length = math.sqrt(s[0] * s[0] + s[1] * s[1] + s[2] * s[2])
        if not length or not math.isfinite(length):
            continue
        d = sum(x * y / length for x, y in zip(normals[k], s))
        out["vertices"] += 1
        out["within_8_degrees"] += d > 0.99
        out["within_25_degrees"] += d > 0.9
        out["opposed"] += d < 0
    return out


def centre_distance(data, group, positions):
    """Sum over a group's meshlets of the distance (per axis, in metres) between the stored centre and the centre of
    the meshlet's vertex box, with these positions. The `42 40` block a group was built against gives the smallest
    sum; a vertex reference past the block, or a sum that is not finite (NaN or infinite positions), gives None."""
    at, k = group["offset"], group["meshlets"]
    rec_at = at + 12
    refs = struct.unpack_from(f"<{group['vertex_refs']}H", data, rec_at + 16 * k + 4 * group["triangles"])
    total = 0.0
    for m in range(k):
        w0, w1, w2, w3 = struct.unpack_from("<4I", data, rec_at + 16 * m)
        vb, vc = w1 & 0xFFFF, (w0 & 0x7F) + 1
        pts = [positions[r] for r in refs[vb:vb + vc] if r < len(positions)]
        if len(pts) != vc:
            return None
        stored = meshlet_centre(w2, w3)
        total += sum(abs((min(p[i] for p in pts) + max(p[i] for p in pts)) / 2 - stored[i]) for i in range(3))
    return total if math.isfinite(total) else None


def block4240_checks(data, block):
    """Count-only plausibility checks for one `42 40` block's decoded positions. Every position is bias + scale * q
    with q between 0 and 2^24 - 1 (2^16 - 1 for the third field), so when the header's box is finite and inside
    300 m on every axis the counts follow from the header alone; only other blocks are decoded vertex by vertex."""
    at, n = block["offset"], block["vertices"]
    bias, scale = struct.unpack_from("<3f", data, at + 0x60), struct.unpack_from("<3f", data, at + 0x70)
    slot0 = {i: tag for i, tag, _ in block["slots"]}.get(0)
    if slot0 == 0x42 and _u32(data, at + 0x6C) <= 2:
        ends = [(b, b + s * q) for b, s, q in zip(bias, scale, (0xFFFFFF, 0xFFFFFF, 0xFFFF))]
        if all(math.isfinite(e) for pair in ends for e in pair) and all(abs(e) <= 300 for pair in ends for e in pair):
            return {"positions_4240": n, "positions_4240_finite": n, "positions_4240_within_300m": n}

    slots = {i: (tag, off) for i, tag, off in block["slots"]}
    if slots.get(0, (None,))[0] != 0x42:
        raise PageError(f"42 40 block at +0x{at:x}: slot 0 is not a packed position (tag 0x42)")
    axis = _u32(data, at + 0x6C)
    if axis > 2:
        raise PageError(f"42 40 block at +0x{at:x}: axis word {axis} is not 0, 1 or 2")

    b0, b1, b2 = bias
    s0, s1, s2 = scale
    total = 0
    finite_count = 0
    within_300m_count = 0

    slot0_off = slots[0][1]
    for (q,) in struct.iter_unpack("<Q", memoryview(data)[at + slot0_off:at + slot0_off + 8 * n]):
        total += 1
        a = b0 + s0 * ((q >> 8) & 0xFFFFFF)
        b = b1 + s1 * (((q >> 32) & 0xFFFF) << 8 | (q & 0xFF))
        c = b2 + s2 * (q >> 48)
        if math.isfinite(a) and math.isfinite(b) and math.isfinite(c):
            finite_count += 1
            if abs(a) <= 300 and abs(b) <= 300 and abs(c) <= 300:
                within_300m_count += 1

    return {"positions_4240": total, "positions_4240_finite": finite_count,
            "positions_4240_within_300m": within_300m_count}


def parse_group(data, at):
    """Parse and validate a meshlet group at `at`. Returns (record, end offset)."""
    head = _u32(data, at)
    if not head & 0x80000000:
        raise PageError(f"no group header at +0x{at:x}")
    meshlets = head & 0x7FFFFFFF
    triangles, refs = _u32(data, at + 4), _u32(data, at + 8)
    rec_at = at + 12
    tri_at = rec_at + 16 * meshlets
    ref_at = tri_at + 4 * triangles
    end = ref_at + 2 * refs
    if end > len(data):
        raise PageError(f"group at +0x{at:x} runs past the page end")
    vbase = tbase = 0
    full = 0
    for m in range(meshlets):
        w0, w1, _, _ = struct.unpack_from("<4I", data, rec_at + 16 * m)
        vcount, tcount = (w0 & 0x7F) + 1, ((w0 >> 7) & 0x7F) + 1
        if (w1 & 0xFFFF) != (vbase & 0xFFFF) or (w1 >> 16) != (tbase & 0xFFFF):
            raise PageError(f"group at +0x{at:x}: meshlet {m} offsets are not running sums")
        if tbase + tcount > triangles or vbase + vcount > refs:
            raise PageError(f"group at +0x{at:x}: meshlet {m} runs past the group")
        for t in struct.unpack_from(f"<{tcount}I", data, tri_at + 4 * tbase):
            if (t >> 21) != m or max(t & 0x7F, (t >> 7) & 0x7F, (t >> 14) & 0x7F) >= vcount:
                raise PageError(f"group at +0x{at:x}: bad triangle in meshlet {m}")
        full += vcount == 128
        vbase += vcount
        tbase += tcount
    if vbase != refs or tbase != triangles:
        raise PageError(f"group at +0x{at:x}: meshlet sums do not match the header")
    vref = struct.unpack_from(f"<{refs}H", data, ref_at) if refs else ()
    span = (max(vref) + 1) if vref else 0
    record = {"kind": "group", "offset": at, "meshlets": meshlets, "triangles": triangles,
              "vertex_refs": refs, "vertex_span": span, "unique_refs": len(set(vref)),
              "full_meshlets": full}
    return record, end


def is_mesh_page(data):
    """True for a page that starts with a vertex block, or with a meshlet group that validates."""
    if len(data) < 16:
        return False
    word = struct.unpack_from("<I", data, 0)[0]
    if word == BLOCK_MAGIC:
        return True
    if word == OTHER_BLOCK_MAGIC:
        try:
            parse_other_block(data, 0)
        except PageError:
            return False
        return True
    if not word & 0x80000000:
        return False
    try:
        parse_group(data, 0)
    except PageError:
        return False
    return True


def _other_data(data, pos, chunks):
    """True where a page's mesh chunks end and other data begins: after at least one chunk, on a 256-byte boundary,
    behind at least 16 zero bytes. story000.rpf has 256 KiB pages laid out that way (mesh chunks, zero padding, then
    texture-like blocks); anything else that doesn't parse stays an error."""
    return bool(chunks) and pos % 256 == 0 and pos >= 16 and not any(data[pos - 16:pos])


def parse_page(data):
    """Walk a whole page. Returns a list of chunk records; raises PageError on anything unexpected."""
    if len(data) < 4 or not is_mesh_page(data):
        raise PageError("page starts with neither a vertex block nor a meshlet group")
    chunks = []
    pos = 0
    while True:
        while pos < len(data) and data[pos] == 0:
            pos += 1
        if pos >= len(data):
            break
        pos &= ~15
        while _u32(data, pos) == 0:
            pos += 4
        if pos % 16:
            raise PageError(f"chunk at +0x{pos:x} is not 16-byte aligned")
        word = _u32(data, pos)
        if word == BLOCK_MAGIC:
            record, pos = parse_block(data, pos)
        elif word == OTHER_BLOCK_MAGIC:
            try:
                record, pos = parse_other_block(data, pos)
            except PageError:
                # a `42 40 00 00` block whose layout doesn't fit the measured strides: stop instead of guessing
                chunks.append({"kind": "stop", "offset": pos, "reason": "42 40 00 00 block"})
                break
        elif word & 0x80000000:
            try:
                record, pos = parse_group(data, pos)
            except PageError as exc:
                if "past the page end" in str(exc):
                    chunks.append({"kind": "stop", "offset": pos, "reason": "group continues past the stream"})
                    break
                if not _other_data(data, pos, chunks):
                    raise
                chunks.append({"kind": "stop", "offset": pos, "reason": "non-mesh data after the chunks"})
                break
        elif _other_data(data, pos, chunks):
            chunks.append({"kind": "stop", "offset": pos, "reason": "non-mesh data after the chunks"})
            break
        else:
            raise PageError(f"unknown chunk at +0x{pos:x}")
        chunks.append(record)
    return chunks


def block_checks(data, block):
    """Count-only plausibility checks for one block's streams. Returns a dict of counts."""
    n = block["vertices"]
    at = block["offset"]
    slots = {i: off for i, _, off in block["slots"]}
    out = {"vertices": n, "positions_finite": 0, "positions_within_300m": 0, "normals_unit": 0,
           "colour_alpha_255": 0, "colour_vertices": 0, "uv_finite": 0, "uv_vertices": 0,
           "position_low_byte_zero": 0}
    pos = struct.unpack_from(f"<{3 * n}f", data, at + slots[0])
    for v in range(n):
        p = pos[3 * v:3 * v + 3]
        if all(math.isfinite(c) for c in p):
            out["positions_finite"] += 1
            out["positions_within_300m"] += all(abs(c) <= 300 for c in p)
    if 2 in slots:
        nrm = struct.unpack_from(f"<{3 * n}f", data, at + slots[2])
        for v in range(n):
            q = nrm[3 * v:3 * v + 3]
            if all(math.isfinite(c) for c in q) and abs(math.sqrt(sum(c * c for c in q)) - 1) < 0.02:
                out["normals_unit"] += 1
    raw = data[at + slots[0]:at + slots[0] + 12 * n]
    out["position_low_byte_zero"] = sum(1 for i in range(0, len(raw), 4) if raw[i] == 0)
    if 1 in slots:
        out["colour_vertices"] = n
        out["colour_alpha_255"] = sum(1 for i in range(3, 4 * n, 4) if data[at + slots[1] + i] == 255)
    for s in (3, 4):
        if s in slots:
            uv = struct.unpack_from(f"<{2 * n}f", data, at + slots[s])
            out["uv_vertices"] += n
            out["uv_finite"] += sum(1 for v in range(n) if math.isfinite(uv[2 * v]) and math.isfinite(uv[2 * v + 1]))
    return out


def summarise_page(data, name, faces=False, centres=False, normals=False, cones=False):
    """Counts for one page. faces=True also runs face_agreement, centres=True centre_agreement, normals=True
    normal_agreement and cones=True cone_agreement on every group whose vertex span matches a `42 40` block in the
    page. With one such block, that block. With several, nothing
    in the page orders them (a group of `Q0 Q1 G0 G1` belongs to Q0 in some pages and to Q1 in others), so the
    block whose vertex boxes are closest to the group's stored centres (`centre_distance`); ties, and groups no
    block fits, keep the nearest block before the group, else the first after it."""
    try:
        chunks = parse_page(data)
    except PageError as exc:
        return {"name": name, "size": len(data), "error": str(exc)}
    blocks = [c for c in chunks if c["kind"] == "block"]
    others = [c for c in chunks if c["kind"] == "block4240"]
    groups = [c for c in chunks if c["kind"] == "group"]
    checks = {}
    try:
        for b in blocks:
            for k, v in block_checks(data, b).items():
                checks[k] = checks.get(k, 0) + v
        for b in others:
            for k, v in block4240_checks(data, b).items():
                checks[k] = checks.get(k, 0) + v
            if normals:
                nrm = normals_4240(data, b)
                if nrm is not None:
                    for k, v in tangent_agreement(nrm, tangents_4240(data, b)).items():
                        checks["tangents_4240_" + k] = checks.get("tangents_4240_" + k, 0) + v
        if faces or centres or normals or cones:
            decoded = {}
            for g in groups:
                same = [b for b in others if b["vertices"] == g["vertex_span"]]
                if not same:
                    continue
                before = [b for b in same if b["offset"] < g["offset"]]
                b = before[-1] if before else same[0]
                if len(same) > 1:
                    checks["groups_4240_several_blocks"] = checks.get("groups_4240_several_blocks", 0) + 1
                    scored = []
                    for c in same:
                        if c["offset"] not in decoded:
                            decoded[c["offset"]] = positions_4240(data, c)
                        cost = centre_distance(data, g, decoded[c["offset"]])
                        if cost is not None:
                            scored.append((cost, c is not b, c["offset"], c))
                    best = min(scored, key=lambda x: x[:3])[3] if scored else b
                    if best is not b:
                        b = best
                        checks["groups_4240_block_by_centres"] = checks.get("groups_4240_block_by_centres", 0) + 1
                if b["offset"] not in decoded:
                    decoded[b["offset"]] = positions_4240(data, b)
                if faces:
                    for k, v in face_agreement(decoded[b["offset"]], list(group_triangles(data, g))).items():
                        checks["faces_4240_" + k] = checks.get("faces_4240_" + k, 0) + v
                if centres:
                    for k, v in centre_agreement(data, g, decoded[b["offset"]]).items():
                        checks["centres_4240_" + k] = checks.get("centres_4240_" + k, 0) + v
                if cones:
                    for k, v in cone_agreement(data, g, decoded[b["offset"]]).items():
                        checks["cones_4240_" + k] = checks.get("cones_4240_" + k, 0) + v
                if normals:
                    nrm = normals_4240(data, b)
                    if nrm is not None:
                        tris = list(group_triangles(data, g))
                        for k, v in normal_agreement(decoded[b["offset"]], nrm, tris).items():
                            checks["normals_4240_" + k] = checks.get("normals_4240_" + k, 0) + v
    except PageError as exc:
        return {"name": name, "size": len(data), "error": str(exc)}
    counts = {b["vertices"] for b in blocks}
    counts_4240 = {b["vertices"] for b in others}
    stop = [c for c in chunks if c["kind"] == "stop"]
    return {"name": name, "size": len(data), "starts_with": chunks[0]["kind"],
            "sequence": "".join({"block": "B", "block4240": "Q", "group": "G", "stop": "."}[c["kind"]] for c in chunks),
            "stop": stop[0]["reason"] if stop else None,
            "blocks": len(blocks), "groups": len(groups),
            "blocks_with_colour": sum(1 for b in blocks if any(s[0] == 1 for s in b["slots"])),
            "blocks_with_uv1": sum(1 for b in blocks if any(s[0] == 4 for s in b["slots"])),
            "block_flags": sorted({b["flags"] for b in blocks}),
            "tags_expected": all(b["tags_expected"] for b in blocks),
            "meshlets": sum(g["meshlets"] for g in groups),
            "full_meshlets": sum(g["full_meshlets"] for g in groups),
            "triangles": sum(g["triangles"] for g in groups),
            "vertex_refs": sum(g["vertex_refs"] for g in groups),
            "groups_dense": sum(1 for g in groups if g["unique_refs"] == g["vertex_span"]),
            "groups_matching_block_here": sum(1 for g in groups if g["vertex_span"] in counts),
            "blocks_4240": len(others), "vertices_4240": sum(b["vertices"] for b in others),
            "blocks_4240_flags": sorted({b["flags"] for b in others}),
            "blocks_4240_axes": sorted({b["axis"] for b in others}),
            "groups_matching_4240_block_here": sum(1 for g in groups if g["vertex_span"] in counts_4240),
            "group_spans": [g["vertex_span"] for g in groups],
            "block_counts": [b["vertices"] for b in blocks],
            "checks": checks}


def iter_archive(path):
    """Yield (name, inflated bytes) for every zlib stream in an RPF9 data area that is_mesh_page accepts."""
    with open(path, "rb") as fh:
        if not os.fstat(fh.fileno()).st_size:
            return
        m = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            size = len(m)
            pos = struct.unpack_from("<I", m, 4)[0] if size >= 8 else size
            while pos < size:
                match = ZHDR.search(m, pos)
                if not match:
                    break
                s = match.start()
                d = zlib.decompressobj()
                chunks, p = [], s
                try:
                    while not d.eof and p < size:
                        block = m[p:p + (1 << 20)]
                        chunks.append(d.decompress(block))
                        p += len(block)
                        if d.unused_data or d.eof:
                            break
                    out = b"".join(chunks) + d.flush()
                except zlib.error:
                    pos = s + 1
                    continue
                if not d.eof or len(out) < 64:
                    pos = s + 1
                    continue
                if is_mesh_page(out):
                    yield f"0x{s:x}", out
                pos = p - len(d.unused_data)
        finally:
            m.close()


def iter_inputs(paths, existing_report=None):
    for path in paths:
        if os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    with open(full, "rb") as fh:
                        head = fh.read(4)
                        if len(head) == 4 and (head in (b"\x63\x30\x00\x00", b"\x42\x40\x00\x00") or head[3] & 0x80):
                            data = head + fh.read()
                            if is_mesh_page(data):
                                if existing_report and same_output_path(existing_report, full):
                                    raise PageError(f"--json would overwrite the input {full}")
                                yield os.path.relpath(full, path), data
        elif path.lower().endswith(".rpf"):
            yield from iter_archive(path)
        else:
            with open(path, "rb") as fh:
                yield os.path.basename(path), fh.read()


def totals(pages):
    keys = ("blocks", "groups", "blocks_with_colour", "blocks_with_uv1", "meshlets", "full_meshlets",
            "triangles", "vertex_refs", "groups_dense", "groups_matching_block_here", "blocks_4240",
            "vertices_4240", "groups_matching_4240_block_here")
    out = {"pages": len(pages), "pages_parsed": sum(1 for p in pages if "error" not in p),
           "block_led_pages": sum(1 for p in pages if p.get("starts_with") == "block"),
           "group_led_pages": sum(1 for p in pages if p.get("starts_with") == "group"),
           "block_4240_led_pages": sum(1 for p in pages if p.get("starts_with") == "block4240"),
           "stopped_at_4240_block": sum(1 for p in pages if p.get("stop") == "42 40 00 00 block"),
           "stopped_in_group": sum(1 for p in pages if p.get("stop") == "group continues past the stream"),
           "stopped_at_other_data": sum(1 for p in pages if p.get("stop") == "non-mesh data after the chunks"),
           "bytes": sum(p["size"] for p in pages)}
    good = [p for p in pages if "error" not in p]
    for k in keys:
        out[k] = sum(p[k] for p in good)
    out["vertices"] = sum(sum(p["block_counts"]) for p in good)
    checks = {}
    for p in good:
        for k, v in p["checks"].items():
            checks[k] = checks.get(k, 0) + v
    out["checks"] = checks
    out["block_flags"] = sorted({f for p in good for f in p["block_flags"]})
    out["blocks_4240_flags"] = sorted({f for p in good for f in p.get("blocks_4240_flags", [])})
    out["blocks_4240_axes"] = sorted({m for p in good for m in p.get("blocks_4240_axes", [])})
    # A group's vertex span that matches a block in the same page or a neighbouring page (input order).
    near = 0
    for i, p in enumerate(pages):
        if "error" in p:
            continue
        pool = set(p["block_counts"])
        for j in (i - 1, i + 1):
            if 0 <= j < len(pages) and "error" not in pages[j]:
                pool.update(pages[j]["block_counts"])
        near += sum(1 for s in p["group_spans"] if s in pool)
    out["groups_matching_block_near"] = near
    return out


def _resource_pointer(system, at, base, segment, size):
    if not 0 <= at <= len(system) - 8:
        raise PageError("pointer field outside the system segment")
    address = struct.unpack_from("<Q", system, at)[0] - base
    if not 0 <= address <= len(segment) - size:
        raise PageError("buffer or metadata pointer outside its segment")
    return address


def drawable_geometries(system, drawable, lod_index=0):
    """Read one LOD geometry list at a caller-identified drawable."""
    def pointer(at, size):
        return _resource_pointer(system, at, 0x50000000, system, size)

    def count(at):
        used, capacity = struct.unpack_from("<HH", system, at)
        if not 1 <= used <= 256 or used != capacity:
            raise PageError("invalid drawable array count")
        return used

    if not 0 <= drawable <= len(system) - 176:
        raise PageError("drawable outside the system segment")
    if lod_index not in range(4):
        raise PageError("drawable LOD index must be between 0 and 3")
    lod = pointer(drawable + 80 + 8 * lod_index, 16)
    models = count(lod + 8)
    table = pointer(lod, 8 * models)
    for i in range(models):
        model = pointer(table + 8 * i, 56)
        geometries = count(model + 16)
        if struct.unpack_from("<H", system, model + 54)[0] != geometries:
            raise PageError("inconsistent model geometry count")
        entries = pointer(model + 8, 8 * geometries)
        for j in range(geometries):
            yield pointer(entries + 8 * j, 160)


def draw_binding(system, graphics, geometry):
    """Resolve a known geometry record in paired, inflated resource segments."""

    def system_pointer(at, size):
        return _resource_pointer(system, at, 0x50000000, system, size)

    def buffer(at):
        count, stride = struct.unpack_from("<IH", system, at + 8)
        if not count or not 1 <= stride <= 32:
            raise PageError("invalid buffer extent")
        length = count * stride
        return _resource_pointer(system, at + 24, 0x60000000, graphics, length), length, stride

    if not 0 <= geometry <= len(system) - 160:
        raise PageError("geometry outside the system segment")
    vb = system_pointer(geometry + 8, 80)
    layout = system_pointer(vb + 64, 320)
    descriptor = system_pointer(geometry + 64, 16)
    ib = system_pointer(descriptor, 64)
    vertex, vertex_bytes, _ = buffer(vb)
    index, index_bytes, stride = buffer(ib)
    flags = _u32(system, layout + 0x138)
    count = flags >> 10
    if (not flags & 1 or not count or _u32(system, layout) != 128
            or system[layout + 0xD0] != 8 or system[layout + 0x104] != 16
            or vertex_bytes < 128 + count * 8 or _u32(graphics, vertex) != OTHER_BLOCK_MAGIC):
        raise PageError("unsupported vertex layout or invalid position extent")
    meshlets, triangles, binding = struct.unpack_from("<HHI", system, descriptor + 8)
    refs = (binding >> 8) & 0x7FFFFF
    if (stride != 4 or index_bytes < 12 or binding >> 31 or not (meshlets and triangles and refs)
            or struct.unpack_from("<III", graphics, index) != (0x80000000 | meshlets, triangles, refs)):
        raise PageError("unsupported or inconsistent meshlet descriptor")
    data = memoryview(graphics)[index:index + index_bytes]
    group, end = parse_group(data, 0)
    if (group["vertex_span"] > count or ((end + 15) & ~15) != index_bytes or any(data[end:])):
        raise PageError("group references or padded extent exceed the declared layout")
    return {"vertex_block": vertex, "meshlet_group": index, "vertices": count}


def same_output_path(first, second):
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def is_scanned_input(path, input_path):
    """True when a report path is one of the files the scan reads: writing it would destroy the input."""
    if same_output_path(path, input_path):
        return True
    if not os.path.isdir(input_path):
        return False
    # Reserved even before the carve writes it: other tools read <input>/index.json as the carve manifest.
    if same_output_path(path, os.path.join(input_path, "index.json")):
        return True
    if not os.path.isfile(path):
        return False
    root = os.path.normcase(os.path.abspath(input_path))
    target = os.path.normcase(os.path.abspath(path))
    if os.path.dirname(target) != root and not target.startswith(root + os.sep):
        return False
    base = os.path.basename(target)
    return base.endswith(".bin") or base.endswith(".rpf")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="carved page files, folders of them, or .rpf archives")
    ap.add_argument("--json", help="write per-page records and totals here")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--faces", action="store_true",
                    help="also count how often neighbouring faces of 42 40 geometry agree (slower)")
    ap.add_argument("--normals", action="store_true",
                    help="also compare 42 40 slot-2 normals with their vertices' face normals, and tangents with "
                         "the normals (slower)")
    ap.add_argument("--cones", action="store_true",
                    help="also compare meshlet records' normal cones with their 42 40 face normals (slower)")
    ap.add_argument("--centres", action="store_true",
                    help="also compare meshlet records' stored centres with their 42 40 vertices (slower)")
    ap.add_argument("--draw-system", help="inflated system segment paired with one graphics segment")
    select = ap.add_mutually_exclusive_group()
    select.add_argument("--geometry-offset", action="append", type=lambda s: int(s, 0),
                        help="known geometry record offset in the system segment (repeatable)")
    select.add_argument("--drawable-offset", type=lambda s: int(s, 0),
                        help="walk geometry records from this system-segment drawable")
    ap.add_argument("--lod", type=int, choices=range(4), help="drawable LOD index (default: 0)")
    args = ap.parse_args(argv)
    if args.lod is not None and args.drawable_offset is None:
        ap.error("--lod requires --drawable-offset")
    if args.json:
        if os.path.isdir(args.json):
            print(f"error: --json {args.json} is a directory", file=sys.stderr)
            return 2
        candidates = list(args.inputs)
        if args.draw_system:
            candidates.append(args.draw_system)
        for inp in candidates:
            if is_scanned_input(args.json, inp):
                print(f"error: --json would overwrite the input {inp}", file=sys.stderr)
                return 2
    if args.draw_system is not None or args.geometry_offset is not None or args.drawable_offset is not None:
        if not args.draw_system or (not args.geometry_offset and args.drawable_offset is None) or len(args.inputs) != 1:
            ap.error("draw binding needs one graphics segment, --draw-system and a geometry or drawable offset")
        try:
            with open(args.draw_system, "rb") as source:
                system = source.read()
            with open(args.inputs[0], "rb") as source:
                graphics = source.read()
            rows = []
            offsets = args.geometry_offset or drawable_geometries(system, args.drawable_offset, args.lod or 0)
            try:
                for offset in offsets:
                    try:
                        rows.append({"geometry": offset, **draw_binding(system, graphics, offset)})
                    except PageError as exc:
                        rows.append({"geometry": offset, "unresolved": str(exc)})
            except PageError as exc:
                rows.append({"drawable": args.drawable_offset, "unresolved": str(exc)})
            report = json.dumps({"bindings": rows}, indent=2)
            if args.json:
                try:
                    with open(args.json, "w", encoding="utf-8", newline="\n") as out:
                        out.write(report + "\n")
                except OSError as exc:
                    print(f"error: cannot write {args.json}: {exc.strerror or exc}", file=sys.stderr)
                    return 2
            if not args.quiet:
                print(report)
            return int(any("unresolved" in row for row in rows))
        except OSError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    for path in args.inputs:
        if not os.path.exists(path):
            print(f"error: {path} does not exist", file=sys.stderr)
            return 2
    existing_report = args.json if args.json and os.path.isfile(args.json) else None
    try:
        pages = [summarise_page(data, name, args.faces, args.centres, args.normals, args.cones)
                 for name, data in iter_inputs(args.inputs, existing_report)]
    except PageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    total = totals(pages)
    if not args.quiet:
        for p in pages:
            if "error" in p:
                print(f"{p['name']}: {p['size']} bytes: ERROR {p['error']}")
            else:
                print(f"{p['name']}: {p['size']} bytes {p['sequence']} blocks={p['blocks']} groups={p['groups']} "
                      f"meshlets={p['meshlets']} triangles={p['triangles']}")
        print(json.dumps(total, indent=1))
    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8", newline="\n") as fh:
                json.dump({"totals": total, "pages": pages}, fh, indent=1)
        except OSError as exc:
            print(f"error: cannot write {args.json}: {exc.strerror or exc}", file=sys.stderr)
            return 2
    return 1 if total["pages"] != total["pages_parsed"] else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
