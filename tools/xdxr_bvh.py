#!/usr/bin/env python3
"""Clean-room reference encoder, validator and traversal for the Xbox XDXR BVH4 runtime layout.

The game's ray-tracing compute shaders walk acceleration structures themselves (docs/d3d12_vulkan.md section 4.8),
so a builder on the Vulkan layer has to write the console's node records, not a VkAccelerationStructureKHR. This
module is the CPU reference for that builder:

* ``build_blas`` turns triangles into a bottom-level BVH4 (header, parent table, leaves, BOX32 or BOX16 nodes);
* ``build_tlas`` turns instances into a top-level BVH4 whose leaves are 128-byte INSTANCE records;
* ``intersect_node`` is one traversal step with the semantics of the hardware instruction
  (``image_bvh_intersect_ray``) as the public XDXR software path describes it;
* ``trace`` walks a TLAS or BLAS with ``intersect_node`` and returns the closest hit;
* ``validate`` checks a buffer's header, pointers, bounds and parent table.

The record layouts, flag values and step semantics come from docs/bvh.md section 3.1 and the public
``XDXRStandaloneTraverseShortStackCPU.hlsli`` (DagorEngine commit 9639942, quoted there): header (L2531), Node16
(L2285), Node32 (L2302), Leaf (L2342), Instance (L2436), node types (L2165), the triangle vertex selection per
node type (Read3Verts), the barycentric remap byte at leaf +60, the parent table (XboxBvh4Node_GetParentTableOffset)
and the box/triangle tests (TestXboxBvh4Node, TestXboxBvh4Leaf). No code is taken from that file; this is an
independent implementation of the documented layout. Choices the public source does not fix are marked
ASSUMPTION: the order of the regions inside a buffer, the leaf packing and the build heuristic. A traversal shader
only reaches the regions through the header, so those choices do not change what it reads.

Standard library only. Nothing here reads game files; ``validate`` reads a buffer the user dumped locally.
Usage: `python -I tools/xdxr_bvh.py --help` lists the options.
"""
import argparse
import math
import random
import struct
import sys

# ---- layout constants (docs/bvh.md 3.1) ----
UNIT = 64
HEADER_SIZE = 64
NODE16_SIZE = 64
NODE32_SIZE = 128
LEAF_SIZE = 64
INSTANCE_SIZE = 128
AS_ALIGNMENT = 256          # instance records and descriptors hold the BVH address >> 8

TRI0, TRI1, TRI2, TRI3, BOX16, BOX32, INSTANCE, AABB = range(8)
INVALID = 0xFFFFFFFF

FLAG_BOTTOM_LEVEL = 0x80000000
FLAG_COMPRESSED = 0x00010000
FLAG_OFFLINE = 0x00020000
FLAG_UPDATEABLE = 0x00040000
FLAG_DEHYDRATED = 0x01000000
VERSION_2403 = 0x0006       # the version in every shipped BVH (docs/bvh.md: flags 0x81020006)

# instance InternalFlags_RootNodePtr: two flag bits above the BLAS root node pointer
INST_IDENTITY_ROTATION = 0x80000000
INST_SCALED_LOCAL_SPACE = 0x40000000
INST_FLAGS_MASK = INST_IDENTITY_ROTATION | INST_SCALED_LOCAL_SPACE

# leaf barycentric remap byte: bits [1:0] pick the edge function returned as the first barycentric (0 = U, 1 = V,
# 2 = W), bits [3:2] the second. U weighs the triangle's first vertex, so 1 | 2 << 2 gives D3D's (b1, b2).
REMAP_IDENTITY = 1 | (2 << 2)

# vertex slots per triangle node type (Read3Verts); slot 4 overlaps the leaf metadata, so TRI2/TRI3 are not emitted
TRI_VERTS = {TRI0: (0, 1, 2), TRI1: (1, 3, 2), TRI2: (2, 3, 4), TRI3: (2, 4, 0)}

FLT_MAX = struct.unpack("<f", struct.pack("<I", 0x7F7FFFFF))[0]
HALF_MAX = 65504.0
HALF_MIN_NORMAL = 2.0 ** -14
INF = math.inf


def f32(x):
    """Round a Python float to the nearest float32 value."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def ptr(offset, kind):
    """Node pointer for a record at byte ``offset`` from the BVH start: 64-byte unit index << 3 | type."""
    if offset % UNIT:
        raise ValueError(f"node offset {offset:#x} is not 64-byte aligned")
    p = (offset // UNIT) << 3 | kind
    if p >= INVALID:
        raise ValueError("BVH too large for 32-bit node pointers")
    return p


def ptr_offset(p):
    return (p >> 3) * UNIT


def ptr_type(p):
    return p & 7


# ---- half-float bounds (Node16) ----
def _half_bits(x):
    return struct.unpack("<H", struct.pack("<e", x))[0]


def _half_value(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def _half_step(bits, up):
    """The next half value above (up) or below the one ``bits`` encodes."""
    if bits & 0x7FFF == 0:                      # +-0
        return 0x0001 if up else 0x8001
    neg = bits & 0x8000
    return bits + 1 if (up != bool(neg)) else bits - 1


def half_round(x, up):
    """Round a float to a half that is >= x (up) or <= x, so a quantised box still contains the exact one. Values
    beyond the half range become +-inf on their outward side (the decoder clamps them to +-FLT_MAX: still
    conservative) and +-HALF_MAX on their inward side, so an empty box stays empty. No denormal half
    is produced: a value below the smallest normal half (2^-14) in magnitude becomes 0 or +-2^-14, whichever is on its
    side, because a decoder that flushes half denormals to zero would otherwise shrink the box. The GPU builder
    (runtime/src/xdxr_build.comp half_directed) computes the same bits."""
    if x > HALF_MAX:
        return 0x7C00 if up else _half_bits(HALF_MAX)
    if x < -HALF_MAX:
        return _half_bits(-HALF_MAX) if up else 0xFC00
    if abs(x) < HALF_MIN_NORMAL:
        if x == 0:
            return _half_bits(x)                    # keeps the sign of a zero
        if x > 0:
            return _half_bits(HALF_MIN_NORMAL) if up else 0x0000
        return 0x8000 if up else _half_bits(-HALF_MIN_NORMAL)
    b = _half_bits(x)
    v = _half_value(b)
    if (up and v < x) or (not up and v > x):
        b = _half_step(b, up)
    return b


def decode_box16(words):
    w0, w1, w2 = words
    h = [w0 & 0xFFFF, w0 >> 16, w1 & 0xFFFF, w1 >> 16, w2 & 0xFFFF, w2 >> 16]
    v = [min(max(_half_value(x), -FLT_MAX), FLT_MAX) for x in h]
    return v[0:3], v[3:6]


def encode_box16(lo, hi):
    h = [half_round(c, False) for c in lo] + [half_round(c, True) for c in hi]
    return (h[0] | h[1] << 16, h[2] | h[3] << 16, h[4] | h[5] << 16)


# ---- geometry helpers ----
def _bounds(points):
    lo = [min(p[c] for p in points) for c in range(3)]
    hi = [max(p[c] for p in points) for c in range(3)]
    return lo, hi


def _union(boxes):
    lo = [min(b[0][c] for b in boxes) for c in range(3)]
    hi = [max(b[1][c] for b in boxes) for c in range(3)]
    return lo, hi


def _transform_point(m, p):
    return [m[r][0] * p[0] + m[r][1] * p[1] + m[r][2] * p[2] + m[r][3] for r in range(3)]


def _transform_dir(m, d):
    return [m[r][0] * d[0] + m[r][1] * d[1] + m[r][2] * d[2] for r in range(3)]


def invert_affine(m):
    """Inverse of a 3x4 row-major affine transform."""
    a = [row[:3] for row in m]
    det = (a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
           - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
           + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
    if det == 0:
        raise ValueError("singular instance transform")
    inv = [[0.0] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            r1, r2 = (c + 1) % 3, (c + 2) % 3
            c1, c2 = (r + 1) % 3, (r + 2) % 3
            inv[r][c] = (a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]) / det
    t = [row[3] for row in m]
    return [inv[r] + [-(inv[r][0] * t[0] + inv[r][1] * t[1] + inv[r][2] * t[2])] for r in range(3)]


# ---- tree construction (ASSUMPTION: the heuristic is ours; any valid BVH4 traverses the same) ----
def _split4(items, centre):
    """Split items into up to four groups by two median cuts along the widest centroid axes."""
    def cut(group):
        if len(group) < 2:
            return [group]
        lo, hi = _bounds([centre(i) for i in group])
        axis = max(range(3), key=lambda c: hi[c] - lo[c])
        group = sorted(group, key=lambda i: centre(i)[axis])
        mid = len(group) // 2
        return [group[:mid], group[mid:]]
    out = []
    for half in cut(items):
        out.extend(g for g in cut(half) if g)
    return out


class _Inner:
    __slots__ = ("children", "box")

    def __init__(self, children, box):
        self.children = children
        self.box = box


def _build_tree(leaves, centre, box_of):
    """leaves: list of leaf handles; returns a tree of _Inner nodes whose children are _Inner or leaf handles."""
    if len(leaves) <= 4:
        return _Inner(list(leaves), _union([box_of(x) for x in leaves]))
    kids = [_build_tree(g, centre, box_of) if len(g) > 1 else g[0] for g in _split4(leaves, centre)]
    return _Inner(kids, _union([k.box if isinstance(k, _Inner) else box_of(k) for k in kids]))


def _inner_nodes(root):
    order = []
    stack = [root]
    while stack:
        n = stack.pop()
        order.append(n)
        stack.extend(c for c in n.children if isinstance(c, _Inner))
    return order


def _layout(n_elements, element_size, inner, box16):
    """Byte offsets of the regions. ASSUMPTION: header | parent table | elements | nodes; the shaders find each
    region through the header, and XboxBvh4Node_GetParentTableOffset only needs offsetToParents to be a multiple
    of 4 bytes."""
    node_size = NODE16_SIZE if box16 else NODE32_SIZE
    body_units = (n_elements * element_size + len(inner) * node_size) // UNIT
    parents = HEADER_SIZE
    parent_bytes = -(-4 * body_units // UNIT) * UNIT
    elements = parents + parent_bytes
    nodes = elements + n_elements * element_size
    end = nodes + len(inner) * node_size
    return parents, elements, nodes, end, node_size


def _write_nodes(buf, inner, node_offsets, child_ptr, child_box, box16, parent_of):
    for n in inner:
        off = node_offsets[id(n)]
        me = ptr(off, BOX16 if box16 else BOX32)
        kids = n.children
        ptrs = [child_ptr(k) for k in kids] + [INVALID] * (4 - len(kids))
        struct.pack_into("<4I", buf, off, *ptrs)
        for i in range(4):
            if i < len(kids):
                lo, hi = child_box(kids[i])
                parent_of[ptrs[i]] = me
            else:
                lo, hi = [INF] * 3, [-INF] * 3          # min > max: the box test misses it
            if box16:
                struct.pack_into("<3I", buf, off + 16 + 12 * i, *encode_box16(lo, hi))
            else:
                struct.pack_into("<6f", buf, off + 16 + 24 * i, *lo, *hi)


def _write_parents(buf, parents_off, elements_off, end, parent_of, root_ptr):
    """One dword per 64-byte unit from the elements region on: the parent node pointer of the record that starts
    there, INVALID for the root and for units that start no record."""
    units = (end - elements_off) // UNIT
    for u in range(units):
        struct.pack_into("<I", buf, parents_off + 4 * u, INVALID)
    for child, parent in parent_of.items():
        u = (ptr_offset(child) - elements_off) // UNIT
        struct.pack_into("<I", buf, parents_off + 4 * u, parent)


def _write_header(buf, flags, end, n_elements, bounds, nodes_off, root, elements_off, parents_off):
    lo, hi = bounds
    struct.pack_into("<4I6fIiIIII", buf, 0, flags, end // UNIT - 1, 0, n_elements, *lo, *hi,
                     nodes_off, 0, root, elements_off, parents_off, 0)


def _remap_for(tri, order):
    """Remap byte that returns D3D's barycentrics (the weights of the triangle's own vertices 1 and 2) when the
    node reads the vertices in ``order``: the edge function of slot s weighs order[s]."""
    order = [tuple(p) for p in order]
    return order.index(tuple(tri[1])) | order.index(tuple(tri[2])) << 2


def _pair_triangles(tris):
    """Greedy pairing of triangles that share an edge into one leaf (TRI0 = v0 v1 v2, TRI1 = v1 v3 v2). Returns a
    list of leaves, each a list of (slots, remap, prim) entries plus the five vertex slots."""
    leaves = []
    used = [False] * len(tris)
    edges = {}
    for i, t in enumerate(tris):
        for k in range(3):
            a, b = tuple(t[k]), tuple(t[(k + 1) % 3])
            edges.setdefault((b, a), []).append((i, k))     # an opposite-winding neighbour has edge (b, a)
    for i, t in enumerate(tris):
        if used[i]:
            continue
        used[i] = True
        mate = None
        for k in range(3):
            a, b = tuple(t[k]), tuple(t[(k + 1) % 3])
            for j, kj in edges.get((a, b), []):
                if not used[j]:
                    mate = (k, j, kj)
                    break
            if mate:
                break
        if not mate:
            v = [list(t[0]), list(t[1]), list(t[2]), list(t[2])]
            leaves.append((v, [(TRI0, REMAP_IDENTITY, i)]))
            continue
        k, j, kj = mate
        used[j] = True
        # rotate tri i so the shared edge is v1 -> v2: v0 = its opposite vertex
        p0, p1, p2 = t[(k + 2) % 3], t[k], t[(k + 1) % 3]
        q = tris[j]
        p3 = q[(kj + 2) % 3]                                  # the mate's vertex off the shared edge
        v = [list(p0), list(p1), list(p2), list(p3)]
        leaves.append((v, [(TRI0, _remap_for(t, (p0, p1, p2)), i), (TRI1, _remap_for(q, (p1, p3, p2)), j)]))
    return leaves


def build_blas(triangles, geometry_index=0, flags=0, box16=False, pair=False, opaque=True):
    """Bottom-level BVH4 from triangles (each three xyz points). Returns (bytes, info). Primitive indices are the
    triangles' positions in the list; ``flags`` is the geometry's D3D12_RAYTRACING_GEOMETRY_FLAGS (bit 0 opaque).
    With ``pair`` two triangles that share an edge go into one leaf as TRI0 and TRI1 (ASSUMPTION: greedy pairing;
    the console builder's packing is not public)."""
    tris = [[[f32(c) for c in p] for p in t] for t in triangles]
    if not tris:
        raise ValueError("a BLAS needs at least one triangle")
    gflags = (flags | (1 if opaque else 0)) & 0xFF
    if pair:
        leaves = _pair_triangles(tris)
    else:
        leaves = [([list(t[0]), list(t[1]), list(t[2]), list(t[2])], [(TRI0, REMAP_IDENTITY, i)])
                  for i, t in enumerate(tris)]
    # the tree's leaves are triangle references (leaf record, node type): a paired record is reached through two
    # pointers, each with its own triangle's box
    refs = [(h, kind, prim) for h, (_, entries) in enumerate(leaves) for kind, _, prim in entries]
    ref_box = {r: _bounds(tris[r[2]]) for r in refs}
    centre = lambda r: [(ref_box[r][0][c] + ref_box[r][1][c]) * 0.5 for c in range(3)]
    root = _build_tree(refs, centre, lambda r: ref_box[r])
    inner = _inner_nodes(root)
    parents_off, elements_off, nodes_off, end, node_size = _layout(len(leaves), LEAF_SIZE, inner, box16)
    buf = bytearray(end)
    for h, (verts, entries) in enumerate(leaves):
        off = elements_off + h * LEAF_SIZE
        for slot, v in enumerate(verts):
            struct.pack_into("<3f", buf, off + 12 * slot, *v)
        prims = {kind: prim for kind, _, prim in entries}
        remap = 0
        for kind, r, _ in entries:
            remap |= r << (8 * kind)
        struct.pack_into("<4I", buf, off + 48, (geometry_index << 8 | gflags) & 0xFFFFFFFF, prims[TRI0],
                         prims.get(TRI1, INVALID), remap)
    node_offsets = {id(n): nodes_off + i * node_size for i, n in enumerate(inner)}
    kind = BOX16 if box16 else BOX32
    child_ptr = lambda k: (ptr(node_offsets[id(k)], kind) if isinstance(k, _Inner)
                           else ptr(elements_off + k[0] * LEAF_SIZE, k[1]))
    child_box = lambda k: k.box if isinstance(k, _Inner) else ref_box[k]
    parent_of = {}
    _write_nodes(buf, inner, node_offsets, child_ptr, child_box, box16, parent_of)
    root_ptr = ptr(node_offsets[id(root)], kind)
    _write_parents(buf, parents_off, elements_off, end, parent_of, root_ptr)
    _write_header(buf, FLAG_BOTTOM_LEVEL | VERSION_2403, end, len(refs), root.box, nodes_off, root_ptr,
                  elements_off, parents_off)
    info = {"size": end, "root": root_ptr, "leaves": len(leaves), "triangles": len(refs), "nodes": len(inner),
            "bounds": root.box, "elements": elements_off, "parents": parents_off, "node_offset": nodes_off}
    return bytes(buf), info


class Instance:
    """One D3D12_RAYTRACING_INSTANCE_DESC plus what the encoder needs from its BLAS."""

    def __init__(self, transform, blas_va, blas_root, blas_bounds, instance_id=0, mask=0xFF, contribution=0,
                 flags=0):
        # Derive the inverse and bounds from the float32 matrix stored in the instance record.
        self.transform = [[f32(x) for x in row] for row in transform]
        self.blas_va = blas_va
        self.blas_root = blas_root
        self.blas_bounds = blas_bounds
        self.instance_id = instance_id
        self.mask = mask
        self.contribution = contribution
        self.flags = flags


def _world_box(inst):
    lo, hi = inst.blas_bounds
    corners = [[(lo, hi)[(i >> c) & 1][c] for c in range(3)] for i in range(8)]
    pts = [_transform_point(inst.transform, p) for p in corners]
    return _bounds(pts)


def build_tlas(instances, box16=False):
    """Top-level BVH4 whose leaves are INSTANCE records (128 bytes, two units). Returns (bytes, info)."""
    if not instances:
        raise ValueError("a TLAS needs at least one instance")
    for inst in instances:
        if inst.blas_va % AS_ALIGNMENT:
            raise ValueError("BLAS addresses must be 256-byte aligned")
    boxes = {i: _world_box(inst) for i, inst in enumerate(instances)}
    handles = list(range(len(instances)))
    centre = lambda h: [(boxes[h][0][c] + boxes[h][1][c]) * 0.5 for c in range(3)]
    root = _build_tree(handles, centre, lambda h: boxes[h])
    inner = _inner_nodes(root)
    parents_off, elements_off, nodes_off, end, node_size = _layout(len(instances), INSTANCE_SIZE, inner, box16)
    buf = bytearray(end)
    inst_ptrs = {}
    for h, inst in enumerate(instances):
        off = elements_off + h * INSTANCE_SIZE
        inv = invert_affine(inst.transform)
        identity_rot = all(inst.transform[r][c] == (1.0 if r == c else 0.0) for r in range(3) for c in range(3))
        internal = (inst.blas_root & ~INST_FLAGS_MASK) | (INST_IDENTITY_ROTATION if identity_rot else 0)
        struct.pack_into("<12f", buf, off, *[x for row in inst.transform for x in row])
        struct.pack_into("<4I", buf, off + 48, (inst.instance_id & 0xFFFFFF) | (inst.mask & 0xFF) << 24,
                         (inst.contribution & 0xFFFFFF) | (inst.flags & 0xFF) << 24, internal,
                         inst.blas_va >> 8)
        struct.pack_into("<12f", buf, off + 64, *[x for row in inv for x in row])
        struct.pack_into("<4I", buf, off + 112, h, INVALID, 0, 0)
        inst_ptrs[h] = ptr(off, INSTANCE)
    node_offsets = {id(n): nodes_off + i * node_size for i, n in enumerate(inner)}
    kind = BOX16 if box16 else BOX32
    child_ptr = lambda k: ptr(node_offsets[id(k)], kind) if isinstance(k, _Inner) else inst_ptrs[k]
    child_box = lambda k: k.box if isinstance(k, _Inner) else boxes[k]
    parent_of = {}
    _write_nodes(buf, inner, node_offsets, child_ptr, child_box, box16, parent_of)
    for child, parent in parent_of.items():
        if ptr_type(child) == INSTANCE:
            struct.pack_into("<I", buf, ptr_offset(child) + 116, parent)
    root_ptr = ptr(node_offsets[id(root)], kind)
    _write_parents(buf, parents_off, elements_off, end, parent_of, root_ptr)
    _write_header(buf, VERSION_2403, end, len(instances), root.box, nodes_off, root_ptr, elements_off, parents_off)
    info = {"size": end, "root": root_ptr, "instances": len(instances), "nodes": len(inner), "bounds": root.box}
    return bytes(buf), info


# ---- memory and one traversal step ----
class Memory:
    """A sparse GPU address space: buffers placed at virtual addresses."""

    def __init__(self):
        self.regions = []

    def place(self, va, data):
        self.regions.append((va, bytes(data)))
        return va

    def read(self, va, n):
        for base, data in self.regions:
            if base <= va and va + n <= base + len(data):
                return data[va - base:va - base + n]
        raise IndexError(f"read of {n} bytes at {va:#x} outside every buffer")

    def u32(self, va, n=1):
        return struct.unpack(f"<{n}I", self.read(va, 4 * n))

    def f32(self, va, n=1):
        return struct.unpack(f"<{n}f", self.read(va, 4 * n))


def _box_test(extent, org, d, inv, lo, hi):
    """Entry distance of the ray into the box, or inf on a miss (TestXboxBvh4Node / RayBoxTest)."""
    if any(lo[c] > hi[c] for c in range(3)):
        return INF
    tnear, tfar = 0.0, extent
    for c in range(3):
        if d[c] == 0.0:
            if org[c] < lo[c] or org[c] > hi[c]:
                return INF
        else:
            t1 = (lo[c] - org[c]) * inv[c]
            t2 = (hi[c] - org[c]) * inv[c]
            tnear = max(tnear, min(t1, t2))
            tfar = min(tfar, max(t1, t2))
            if tfar < tnear:
                return INF
    return tnear


def _tie_miss(ay, by, cy, winding):
    if by == 0.0 and cy == 0.0:
        return ay > 0.0
    below = by < 0.0 or (by == 0.0 and cy > 0.0)
    return below != winding


def triangle_test(v, org, d, remap=REMAP_IDENTITY):
    """(t_num, t_denom, i_num, j_num) of a ray against a triangle, or (inf, 1, 0, 0) on a miss; t = t_num / t_denom
    and the barycentrics are i_num / t_denom, j_num / t_denom (TestXboxBvh4Leaf)."""
    dd = list(d)
    mag = [abs(c) for c in dd]
    m = max(mag)
    if mag[0] == m:
        perm = (1, 2, 0)
    elif mag[1] == m:
        perm = (2, 0, 1)
    else:
        perm = (0, 1, 2)
    sw = lambda p: [p[perm[0]], p[perm[1]], p[perm[2]]]
    rel = [sw([p[c] - org[c] for c in range(3)]) for p in v]
    dd = sw(dd)
    A, B, C = ([p[0] * dd[2] - dd[0] * p[2], p[1] * dd[2] - dd[1] * p[2], p[2]] for p in rel)
    U = C[0] * B[1] - C[1] * B[0]
    V = A[0] * C[1] - A[1] * C[0]
    W = B[0] * A[1] - B[1] * A[0]
    t_num = U * A[2] + V * B[2] + W * C[2]
    t_den = (U + V + W) * dd[2]
    missed = (U < 0 or V < 0 or W < 0) and (U > 0 or V > 0 or W > 0)
    missed |= t_den == 0.0
    missed |= t_num * math.copysign(1.0, t_den) <= 0.0
    missed |= math.isnan(t_num)
    winding = t_den > 0.0
    if U == 0.0:
        missed |= _tie_miss(A[1], B[1], C[1], winding)
    if V == 0.0:
        missed |= _tie_miss(B[1], C[1], A[1], winding)
    if W == 0.0:
        missed |= _tie_miss(C[1], A[1], B[1], winding)
    if missed:
        return (INF, 1.0, 0.0, 0.0)
    e = (U, V, W)
    # t_den carries dd[2], so scale i_num/j_num by dd[2] too: i_num / t_den = e_i / (U + V + W),
    # which is both ray-scale invariant and non-negative even when i_num == 0 and dd[2] < 0.
    i_num = (e[remap & 3] if remap & 3 < 3 else U) * dd[2]   # selector 3 falls back to U, as the public path does
    j_num = (e[(remap >> 2) & 3] if (remap >> 2) & 3 < 3 else U) * dd[2]
    return (t_num, t_den, i_num, j_num)


def intersect_node(mem, bvh_va, node_ptr, org, d, inv, extent, sort=True):
    """One traversal step on the node ``node_ptr`` of the BVH at ``bvh_va``: four child pointers for a box node
    (misses INVALID, sorted by entry distance when ``sort``), the triangle result for TRI0..TRI3, None otherwise."""
    va = bvh_va + ptr_offset(node_ptr)
    kind = ptr_type(node_ptr)
    if kind <= TRI3:
        slots = TRI_VERTS[kind]
        verts = [mem.f32(va + 12 * s, 3) for s in slots]
        remap = (mem.u32(va + 60)[0] >> (8 * kind)) & 0xFF
        return triangle_test(verts, org, d, remap)
    if kind not in (BOX16, BOX32):
        return None
    kids = list(mem.u32(va, 4))
    dist = []
    for i in range(4):
        if kind == BOX16:
            lo, hi = decode_box16(mem.u32(va + 16 + 12 * i, 3))
        else:
            b = mem.f32(va + 16 + 24 * i, 6)
            lo, hi = list(b[:3]), list(b[3:])
        dist.append(_box_test(extent, org, d, inv, lo, hi))
    # a miss is +inf; it must stay a miss for an unbounded ray too (the public path compares with <= the extent,
    # which a finite TMax keeps apart from the miss value)
    kids = [k if dist[i] != INF and dist[i] <= extent else INVALID for i, k in enumerate(kids)]
    if sort:
        for a, b in ((0, 2), (1, 3), (0, 1), (2, 3), (1, 2)):
            if dist[a] > dist[b]:
                dist[a], dist[b] = dist[b], dist[a]
                kids[a], kids[b] = kids[b], kids[a]
    return kids


def _inverse_dir(d):
    return [1.0 / c if c != 0.0 else math.copysign(FLT_MAX, c) for c in d]


def _trace_blas(mem, bvh_va, org, d, tmax, hit, inst, stats=None):
    inv = _inverse_dir(d)
    stack = [mem.u32(bvh_va + 48)[0]]
    while stack:
        p = stack.pop()
        if p == INVALID:
            continue
        if stats is not None:
            stats["steps"] += 1
        r = intersect_node(mem, bvh_va, p, org, d, inv, hit["t"] if hit else tmax)
        if r is None:
            continue
        if ptr_type(p) <= TRI3:
            t_num, t_den, i_num, j_num = r
            if t_num == INF:
                continue
            t = t_num / t_den
            if t <= (hit["t"] if hit else tmax):
                leaf = bvh_va + ptr_offset(p)
                gif, base, prim2 = mem.u32(leaf + 48, 3)
                prim = prim2 if ptr_type(p) == TRI1 else base
                hit = {"t": t, "instance": inst, "geometry": gif >> 8, "primitive": prim, "leaf": p,
                       "bary": (i_num / t_den, j_num / t_den), "front": t_den > 0}
        else:
            stack.extend(reversed([k for k in r if k != INVALID]))
    return hit


def trace(mem, bvh_va, org, d, tmax=INF, mask=0xFF, bias=0, stats=None):
    """Closest hit of a ray in a TLAS or BLAS at ``bvh_va``: a dict with t, instance (None for a BLAS), geometry,
    primitive, bary and front, or None. ``bias`` is added to the BLAS addresses instance records hold (application
    address to the address in ``mem``). With ``stats`` (a dict), its "steps" counts the traversal steps: one per node
    or triangle tested, the work of one image_bvh_intersect_ray."""
    flags = mem.u32(bvh_va)[0]
    if flags & FLAG_BOTTOM_LEVEL:
        return _trace_blas(mem, bvh_va, org, d, tmax, None, None, stats)
    hit = None
    inv = _inverse_dir(d)
    stack = [mem.u32(bvh_va + 48)[0]]
    while stack:
        p = stack.pop()
        if p == INVALID:
            continue
        if ptr_type(p) == INSTANCE:
            rec = bvh_va + ptr_offset(p)
            id_mask, _, internal, blas = mem.u32(rec + 48, 4)
            if not (id_mask >> 24) & mask or blas == 0:         # masked out, or an instance with no BLAS
                continue
            w2o = mem.f32(rec + 64, 12)
            m = [list(w2o[0:4]), list(w2o[4:8]), list(w2o[8:12])]
            lo, ld = _transform_point(m, org), _transform_dir(m, d)
            blas_va = (blas << 8) + bias
            root = internal & ~INST_FLAGS_MASK
            if root != mem.u32(blas_va + 48)[0]:
                raise ValueError(f"instance root {root:#x} is not the BLAS header's root")
            index = mem.u32(rec + 112)[0]
            # the object-space ray keeps the world t: the direction is transformed, not normalised
            hit = _trace_blas(mem, blas_va, lo, ld, tmax, hit, index, stats)
            continue
        if stats is not None:
            stats["steps"] += 1
        r = intersect_node(mem, bvh_va, p, org, d, inv, hit["t"] if hit else tmax)
        if r:
            stack.extend(reversed([k for k in r if k != INVALID]))
    return hit


def brute_force(scene, org, d, tmax=INF, mask=0xFF):
    """Closest hit over every triangle of every instance, with the same triangle test: what a BVH must reproduce.
    ``scene`` is a list of (transform or None, mask, triangles); the result names the scene index as instance."""
    best = None
    for idx, (m, imask, tris) in enumerate(scene):
        if m is not None and not imask & mask:
            continue
        if m is None:
            lo, ld = org, d
        else:
            inv = invert_affine(m)
            lo, ld = _transform_point(inv, org), _transform_dir(inv, d)
        for prim, t in enumerate(tris):
            v = [[f32(c) for c in p] for p in t]
            t_num, t_den, i_num, j_num = triangle_test(v, lo, ld)
            if t_num == INF:
                continue
            tt = t_num / t_den
            if tt <= (best["t"] if best else tmax):
                best = {"t": tt, "instance": None if m is None else idx, "primitive": prim,
                        "bary": (i_num / t_den, j_num / t_den)}
    return best


# ---- validation ----
def validate(data, expect_bottom=None):
    """Structural checks of one BVH buffer placed at address 0. Returns a list of problems (empty = valid)."""
    errs = []
    if len(data) < HEADER_SIZE:
        return ["buffer shorter than the 64-byte header"]
    # offsetToParents is read signed: a console structure keeps it negative (the entries of its low units, the leaves,
    # fall before the header and are not stored), a runtime build keeps it at 0x40.
    (flags, units, _crc, n_el, *rest) = struct.unpack_from("<4I6fIiIIiI", data, 0)
    bounds = rest[0:6]
    nodes_off, _remap, root, elements_off, parents_off, _cp = rest[6:]
    bottom = bool(flags & FLAG_BOTTOM_LEVEL)
    if expect_bottom is not None and bottom != expect_bottom:
        errs.append("bottom-level flag does not match the expected level")
    if flags & (FLAG_COMPRESSED | FLAG_DEHYDRATED):
        errs.append("compressed or dehydrated BVH: not a runtime tree (docs/bvh.md 3.1)")
        return errs
    if (units + 1) * UNIT != len(data):   # the field does not count the 64-byte header (docs/bvh.md 3.1)
        errs.append(f"totalSizeIn64BNodes {units} does not cover the {len(data)}-byte buffer (64 * (field + 1) bytes)")
    for name, off in (("offsetToNodes", nodes_off), ("offsetToElements", elements_off)):
        if off % UNIT or off > len(data):
            errs.append(f"{name} {off:#x} is unaligned or past the end")
    if errs:
        return errs
    seen_leaves = 0
    seen = set()
    stack = [(root, None, None)]
    while stack:
        p, box, parent = stack.pop()
        if p == INVALID:
            continue
        kind, off = ptr_type(p), ptr_offset(p)
        size = NODE32_SIZE if kind == BOX32 else INSTANCE_SIZE if kind == INSTANCE else UNIT
        if off + size > len(data) or off < HEADER_SIZE:
            errs.append(f"node {p:#x} outside the buffer")
            continue
        if p in seen:
            errs.append(f"{'leaf' if kind <= TRI3 or kind == INSTANCE else 'node'} {p:#x} reached twice")
            continue
        seen.add(p)
        if kind > TRI3 and parents_off:
            u = (off - elements_off) // UNIT
            entry = parents_off + 4 * u
            # Only the entries that lie between the header and offsetToElements are stored in the structure; the
            # others (the low units) are not, so there is nothing to compare (docs/bvh.md 3.1).
            if entry + 4 > len(data):
                errs.append(f"parent table entry of {p:#x} is outside the buffer")
            elif HEADER_SIZE <= entry and entry + 4 <= elements_off:
                stored = struct.unpack_from("<I", data, entry)[0]
                expected = INVALID if parent is None else parent
                if stored != expected:
                    errs.append(f"parent table entry of {p:#x} is {stored:#x}, expected {expected:#x}")
        if kind <= TRI3:
            if not bottom:
                errs.append(f"triangle node {p:#x} in a top-level BVH")
                continue
            if kind >= TRI2:
                errs.append(f"node {p:#x}: TRI2/TRI3 overlap the leaf metadata and are not emitted")
            pts = [struct.unpack_from("<3f", data, off + 12 * s) for s in TRI_VERTS[kind]]
            inactive = any(math.isnan(c) for q in pts for c in q)      # D3D12: never hit; its box is empty
            if box and not inactive and not all(box[0][c] <= q[c] <= box[1][c] for q in pts for c in range(3)):
                errs.append(f"triangle {p:#x} is not inside its parent's box")
            seen_leaves += 1
        elif kind == INSTANCE:
            if bottom:
                errs.append(f"instance node {p:#x} in a bottom-level BVH")
            # D3D12 allows an instance with no BLAS (it is inactive); it must not be reachable through a real box
            if struct.unpack_from("<I", data, off + 60)[0] == 0 and box and \
                    all(box[0][c] <= box[1][c] for c in range(3)):
                errs.append(f"instance {p:#x} has no BLAS address but a non-empty box")
            seen_leaves += 1
        elif kind in (BOX16, BOX32):
            kids = struct.unpack_from("<4I", data, off)
            for i, k in enumerate(kids):
                if kind == BOX16:
                    lo, hi = decode_box16(struct.unpack_from("<3I", data, off + 16 + 12 * i))
                else:
                    b = struct.unpack_from("<6f", data, off + 16 + 24 * i)
                    lo, hi = b[:3], b[3:]
                if k == INVALID:
                    continue
                if box and not all(box[0][c] <= lo[c] and hi[c] <= box[1][c] for c in range(3)):
                    errs.append(f"child box {i} of {p:#x} is not inside its parent's box")
                stack.append((k, (lo, hi), p))
        else:
            errs.append(f"node {p:#x} has type AABB, not produced by this encoder")
    if seen_leaves != n_el:
        errs.append(f"numElements {n_el} but {seen_leaves} leaves are reachable")
    return errs


def box16_rounding(data):
    """Checks that every BOX16 child box of a bottom-level BVH holds exactly encode_box16 of the exact box of the
    triangles under that child: the bits, not just containment. The exact box comes from the leaves' own vertices, so
    this needs nothing but the buffer. A triangle with a NaN vertex has an empty box, as in the builder. Returns a list
    of problems (empty = every BOX16 word matches)."""
    errs = []
    root = struct.unpack_from("<I", data, 48)[0]
    empty = ([INF] * 3, [-INF] * 3)
    memo = {}

    def exact(p):
        if p in memo:
            return memo[p]
        kind, off = ptr_type(p), ptr_offset(p)
        if kind <= TRI3:
            pts = [struct.unpack_from("<3f", data, off + 12 * s) for s in TRI_VERTS[kind]]
            box = empty if any(math.isnan(c) for q in pts for c in q) else _bounds(pts)
        else:
            kids = struct.unpack_from("<4I", data, off)
            boxes = [exact(k) for k in kids if k != INVALID]
            boxes = [b for b in boxes if all(b[0][c] <= b[1][c] for c in range(3))]
            box = _union(boxes) if boxes else empty
            if kind == BOX16:
                for i, k in enumerate(kids):
                    want = encode_box16(*(exact(k) if k != INVALID else empty))
                    got = struct.unpack_from("<3I", data, off + 16 + 12 * i)
                    if tuple(got) != tuple(want):
                        errs.append(f"BOX16 {p:#x} child {i}: {[hex(w) for w in got]}, expected "
                                    f"{[hex(w) for w in want]}")
        memo[p] = box
        return box

    if root != INVALID:
        exact(root)
    return errs


def node_census(data):
    """Reachable pointers of a valid BVH by node type: {type: count}, the root included."""
    count = {}
    stack = [struct.unpack_from("<I", data, 48)[0]]
    while stack:
        p = stack.pop()
        if p == INVALID:
            continue
        kind = ptr_type(p)
        count[kind] = count.get(kind, 0) + 1
        if kind in (BOX16, BOX32):
            stack.extend(struct.unpack_from("<4I", data, ptr_offset(p)))
    return count


def pair_rotations(data):
    """The (TRI0 remap byte, TRI1 remap byte) of every paired leaf reachable in a valid bottom-level BVH: which of
    the three rotations each triangle of a pair was stored in."""
    seen = set()
    stack = [struct.unpack_from("<I", data, 48)[0]]
    while stack:
        p = stack.pop()
        if p == INVALID:
            continue
        kind = ptr_type(p)
        if kind == TRI1:
            remap = struct.unpack_from("<I", data, ptr_offset(p) + 60)[0]
            seen.add((remap & 0xFF, (remap >> 8) & 0xFF))
        elif kind in (BOX16, BOX32):
            stack.extend(struct.unpack_from("<4I", data, ptr_offset(p)))
    return seen


def count_pairs(data):
    """TRI1 pointers reachable from the root of a valid bottom-level BVH: the triangles paired into another's leaf."""
    return node_census(data).get(TRI1, 0)


# ---- checking a GPU build (runtime/test/vk_xdxr_build_test.cpp) ----
def _case_names(directory):
    """Find cases using a literal directory and the usual JSON filename pattern."""
    import glob
    import os
    pattern = os.path.join(glob.escape(os.fspath(directory)), "*.json")
    return sorted(os.path.splitext(os.path.basename(path))[0] for path in glob.glob(pattern))


class CaseError(Exception):
    """A <case>.json / <case>.bin pair that is missing, unreadable or not a case description."""


def _load_case(directory, name):
    """desc, data for one <name>.json and <name>.bin pair. CaseError names the .json at fault: any *.json in the
    directory counts as a case (check-build is run on a build folder that may hold other reports), so a stray or
    half-written file there must say which one it is, not end in a traceback."""
    import json
    js = f"{directory}/{name}.json"
    fix = lambda x: math.nan if x == "nan" else x                                   # noqa: E731
    try:
        with open(js, encoding="utf-8") as fh:
            desc = json.load(fh)
        with open(f"{directory}/{name}.bin", "rb") as fh:
            data = fh.read()
        kind = desc["kind"]
        if kind == "blas":
            desc["triangles"] = [[[fix(c) for c in v] for v in t[:3]] + t[3:] for t in desc["triangles"]]
        elif kind != "tlas":
            raise CaseError(f"{js}: kind {kind!r} is neither 'blas' nor 'tlas'")
        # Check the keys check_build needs while the filename is still in scope: a KeyError there would name only the
        # directory and the missing key, and in a folder of several cases the user cannot tell which file to fix.
        for key in ("address",) + (("blas", "instances") if kind == "tlas" else ()):
            if key not in desc:
                raise CaseError(f"{js}: the case has no {key!r} key")
        if kind == "tlas":
            if not isinstance(desc["blas"], list) or not all(isinstance(b, str) and b for b in desc["blas"]):
                raise CaseError(f"{js}: 'blas' is not a list of case names")
            if not isinstance(desc["instances"], list):
                raise CaseError(f"{js}: 'instances' is not a list")
            for i, inst in enumerate(desc["instances"]):
                if not isinstance(inst, dict):
                    raise CaseError(f"{js}: instance {i} is not an object")
                for key in ("transform", "blas", "mask"):
                    if key not in inst:
                        raise CaseError(f"{js}: instance {i} has no {key!r} key")
    except KeyError as error:
        raise CaseError(f"{js}: the case has no {error} key") from None
    except (OSError, ValueError, TypeError, IndexError) as error:
        raise CaseError(f"{js}: {error}") from None
    return desc, data


def check_build(directory, rays=300, seed=1):
    """Validate every <case>.bin of a GPU build and trace rays against brute force. Returns a list of problems."""
    names = _case_names(directory)
    if not names:
        return [f"no cases in {directory}"]
    problems = []
    cases = {n: _load_case(directory, n) for n in names}
    for n, (desc, _) in cases.items():
        if desc["kind"] != "tlas":
            continue
        js = f"{directory}/{n}.json"
        for b in desc["blas"]:
            if b not in cases:
                raise CaseError(f"{js}: BLAS {b!r} is not a case in this folder")
            if cases[b][0]["kind"] != "blas":
                raise CaseError(f"{js}: {b!r} is a {cases[b][0]['kind']}, not a blas")
        for i, inst in enumerate(desc["instances"]):
            index = inst["blas"]
            # -1 is an instance with no BLAS (vk_xdxr_build_test's "no BLAS" case), which the model skips
            if not isinstance(index, int) or not -1 <= index < len(desc["blas"]):
                raise CaseError(f"{js}: instance {i} names BLAS index {index!r}, not -1..{len(desc['blas']) - 1}")
    rng = random.Random(seed)
    for n, (desc, data) in cases.items():
        # An in-place update leaves two dumps at one address (before and after): the case being checked comes first,
        # so its own bytes answer reads there.
        mem = Memory()
        mem.place(desc["address"], data)
        for other, (odesc, odata) in cases.items():
            if other != n:
                mem.place(odesc["address"], odata)
        errs = validate(data, expect_bottom=desc["kind"] == "blas")
        if desc["kind"] == "blas" and not errs:
            errs += box16_rounding(data)
        pairs = count_pairs(data) if desc["kind"] == "blas" and not errs else 0
        if desc.get("pairs") is True and not pairs:
            errs.append("built with pairing, but no TRI1 node is reachable")
        if desc.get("pairs") is False and pairs:
            errs.append(f"{pairs} TRI1 nodes, but none of its triangles may pair")
        if desc.get("all_rotations") and not errs:   # every (first, sibling) rotation of a pair is exercised
            rotations = pair_rotations(data)
            if len(rotations) < 9:
                errs.append(f"paired leaves use {len(rotations)} of the 9 (TRI0, TRI1) rotations: "
                            f"{sorted(rotations)}")
        if "box16" in desc and not errs:          # the node type the build was asked for, so no check runs vacuously
            census = node_census(data)
            want, other = (BOX16, BOX32) if desc["box16"] else (BOX32, BOX16)
            if census.get(other) or not census.get(want):
                errs.append(f"inner nodes {census.get(BOX16, 0)} BOX16 and {census.get(BOX32, 0)} BOX32, expected "
                            f"only {'BOX16' if desc['box16'] else 'BOX32'}")
        problems += [f"{n}: {e}" for e in errs]
        if desc["kind"] == "blas":
            tris = desc["triangles"]
            scene = [(None, 0xFF, [t[:3] for t in tris])]
            ids = [(t[3], t[4]) for t in tris]
        else:
            blas = [cases[b][0] for b in desc["blas"]]
            scene, ids, inst_ids = [], None, []
            for inst in desc["instances"]:
                m = inst["transform"]
                rows = [m[0:4], m[4:8], m[8:12]]
                src = blas[inst["blas"]]["triangles"] if inst["blas"] >= 0 else []
                scene.append((rows, inst["mask"], [t[:3] for t in src]))
                inst_ids.append([(t[3], t[4]) for t in src])      # list position -> (geometry, primitive)
        stats = {"steps": 0}
        bad = hits = 0
        case_rays = min(rays, desc.get("rays", rays))      # a large case may ask for fewer (brute force is slow)
        for _ in range(case_rays):
            o = [rng.uniform(-12, 12) for _ in range(3)]
            target = [rng.uniform(-6, 6) for _ in range(3)]
            d = [target[c] - o[c] for c in range(3)]
            got = trace(mem, desc["address"], o, d, bias=desc.get("bias", 0), stats=stats)
            want = brute_force(scene, o, d)
            hits += got is not None
            if (got is None) != (want is None):
                bad += 1
                continue
            if got is None:
                continue
            if ids is not None:
                same = (got["geometry"], got["primitive"]) == ids[want["primitive"]]
                same = same and math.isclose(got["t"], want["t"], rel_tol=1e-6, abs_tol=1e-9)
                # D3D12's barycentrics: a paired (TRI1) or rotated TRI0 triangle must still return them through its
                # remap byte, not its slot order
                same = same and all(math.isclose(g, w, rel_tol=1e-5, abs_tol=1e-6)
                                    for g, w in zip(got["bary"], want["bary"]))
            else:
                # the instance's world-to-object matrix is a float32 inverse; allow for its rounding in t
                want_id = inst_ids[want["instance"]][want["primitive"]]
                same = (got["instance"], (got["geometry"], got["primitive"])) == (want["instance"], want_id)
                same = same and math.isclose(got["t"], want["t"], rel_tol=1e-4)
            bad += not same
        paired = f", {pairs} TRI1" if pairs else ""
        print(f"{n}: {len(data)} bytes{paired}, {len(errs)} validation errors, {bad}/{case_rays} rays differ "
              f"({hits} hits), {stats['steps'] / max(case_rays, 1):.1f} steps per ray")
        if bad:
            problems.append(f"{n}: {bad} of {case_rays} rays differ from brute force")
        if desc["kind"] == "blas" and len(desc["triangles"]) > 4 and hits == 0:
            problems.append(f"{n}: no ray hit anything; the check proves nothing")
    return problems


# ---- self test and CLI ----
def random_scene(rng, n_tris=64, spread=4.0):
    tris = []
    for _ in range(n_tris):
        c = [rng.uniform(-spread, spread) for _ in range(3)]
        tris.append([[c[k] + rng.uniform(-0.6, 0.6) for k in range(3)] for _ in range(3)])
    return tris


def grid_mesh(n=6, size=4.0):
    """A regular grid of quads, two triangles each: a mesh where pairing finds shared edges."""
    pts = [[(i / n - 0.5) * size, (j / n - 0.5) * size, 0.1 * ((i * 7 + j * 3) % 5)] for j in range(n + 1)
           for i in range(n + 1)]
    at = lambda i, j: pts[j * (n + 1) + i]
    tris = []
    for j in range(n):
        for i in range(n):
            tris.append([at(i, j), at(i + 1, j), at(i + 1, j + 1)])
            tris.append([at(i, j), at(i + 1, j + 1), at(i, j + 1)])
    return tris


def selftest(seed=1, rays=400, verbose=False):
    rng = random.Random(seed)
    failures = 0
    cases = [("random box32", random_scene(rng), False, False), ("random box16", random_scene(rng), True, False),
             ("grid paired", grid_mesh(), False, True), ("grid paired box16", grid_mesh(), True, True)]
    for name, tris, b16, pair in cases:
        data, info = build_blas(tris, box16=b16, pair=pair)
        errs = validate(data, expect_bottom=True)
        errs += box16_rounding(data)
        mem = Memory()
        mem.place(0x10000, data)
        bad = 0
        for _ in range(rays):
            o = [rng.uniform(-6, 6) for _ in range(3)]
            target = [rng.uniform(-3, 3) for _ in range(3)]
            d = [target[c] - o[c] for c in range(3)]
            a = trace(mem, 0x10000, o, d)
            b = brute_force([(None, 0xFF, tris)], o, d)
            if (a is None) != (b is None) or (a and (a["primitive"] != b["primitive"] or
                                                     abs(a["bary"][0] - b["bary"][0]) > 1e-9 or
                                                     abs(a["bary"][1] - b["bary"][1]) > 1e-9)):
                bad += 1
        failures += bad + len(errs)
        print(f"{name}: {info['triangles']} triangles in {info['leaves']} leaves, {info['nodes']} nodes, "
              f"{info['size']} bytes; {len(errs)} validation errors; {bad}/{rays} rays differ")
        if verbose:
            for e in errs:
                print("  " + e)
    return failures


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest", help="build synthetic BVHs and compare traversal with brute force")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--rays", type=int, default=400)
    v = sub.add_parser("validate", help="structural check of one runtime BVH buffer dumped to a file")
    v.add_argument("file")
    c = sub.add_parser("check-build", help="check the cases vk_xdxr_build_test wrote to a directory")
    c.add_argument("dir")
    c.add_argument("--rays", type=int, default=300)
    a = ap.parse_args(argv)
    if a.cmd == "check-build":
        if a.rays < 1:
            ap.error("--rays must be at least 1")
        if not _case_names(a.dir):
            print(f"no cases in {a.dir}: skipped (the build step skipped or did not run)")
            return 77                                   # ctest SKIP_RETURN_CODE
        try:
            problems = check_build(a.dir, a.rays)
        except CaseError as error:
            print(f"ERROR: {error}", file=sys.stderr)
            return 2
        except (KeyError, ValueError, IndexError, TypeError) as error:
            print(f"ERROR: a case in {a.dir} does not describe a build: {error}", file=sys.stderr)
            return 2
        for e in problems:
            print(e)
        print("all cases match" if not problems else f"{len(problems)} problems")
        return 1 if problems else 0
    if a.cmd == "selftest":
        if a.rays < 1:
            ap.error("--rays must be at least 1")
        return 1 if selftest(a.seed, a.rays, verbose=True) else 0
    try:
        with open(a.file, "rb") as fh:
            data = fh.read()
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    errs = validate(data)
    for e in errs:
        print(e)
    print("valid" if not errs else f"{len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
