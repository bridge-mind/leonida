"""Rehydrate a dehydrated XDXR BVH4 bottom-level acceleration structure on the CPU.

The `*_bvh4.rpf` packs hold offline-built BLASes in the builder's dehydrated form: the header and the box
hierarchy are there, but the leaves name primitives instead of holding vertices, and most box bounds are left
to be recomputed. `XboxBvhBuilder.dll` turns such a buffer back into the runtime XDXR layout (docs/bvh.md §3.1)
with its geometry. This tool does the same on the CPU, following the packing recorded in docs/bvh.md §4a, so a
rehydrated structure can be produced without the DLL and the builder's own rehydration programs can be checked
against it on a GPU. Standard library only; nothing is read from the package.

Input layout. The buffer starts at the 64-byte header (pass --header-offset 256 for a whole `bvh4` stream, whose
header sits at payload 0x100). Every offset in the header counts from the header. The output is the rehydrated
structure only, `64 * (totalSizeIn64BNodes + 1)` bytes from the header on.

What is done, in order (docs/bvh.md §4a):

* Header. The header is copied together with the bytes after it that the builder keeps: up to `offsetToParents`
  when the word at +0x2c is non-zero, else `16 * (9 * k * k + 4)` bytes with `k = flags >> 20 & 15`, rounded up to
  16. The dehydrated flag (0x01000000) is cleared and the root's parent entry is set to 0xffffffff.
* Leaves. `numElements` leaves sit at `offsetToElements` in chunks of 2,048 every 128 KiB. A chunk of `c` leaves is
  five arrays: geometry index (3 bytes each), flags (1), base primitive (3), second primitive (3; 0xffffff = none)
  and a vertex-order byte (1). Each leaf becomes a 64-byte `Leaf`: the base triangle's three vertices in the slots
  the vertex-order byte selects, a fourth vertex for a second primitive (NaN when there is none), then
  `geometryIndex << 8 | flags`, the two primitive indices and the remap word.
* Boxes. The level table at the end of the input (last word: level count, before it one (BOX32 end, BOX16 end)
  pair per level, offsets from `offsetToNodes`) lists the levels bottom-up. In each level the BOX32 nodes come
  first, in groups of up to 64 per 8 KiB, then the BOX16 nodes, 64 per 4 KiB. A group starts with one word per node
  (`children << 24 | 24-bit override mask`), then all the child pointers, then one stored value per mask bit
  (float for BOX32, half for BOX16). A slot's box is the bounds of the already rehydrated child; the stored values
  replace the masked components (bit `6 * slot + c`, min xyz then max xyz). BOX16 rounds min down and max up and
  overflows to +-inf. Each node writes itself as the parent of its box children (and of TRI0 leaves when the
  structure is updatable, flag 0x00040000).
* Leaf frame. Vertices are mapped `(R * v - t) * s` per `compressionParams` (+0x3c): bit 31 turns on the scale
  `2^(e - 127)` with `e` the bytes 0, 1, 2; bit 30 the translation, the centre of `sceneBounds` on an axis whose
  bounds do not straddle zero; bits 24-29 pick one of 64 rotations in steps of pi/8 about x, y and z. The value
  0x007f7f7f means none of it. Only rotation index 0 occurs in the measured samples.

Float arithmetic is rounded to float32 after every operation, as the GPU does, so the output is meant to be
byte-exact. Geometry is float3 vertices (any stride) with uint32, uint16 or no indices, plus an optional 3x4
geometry transform (the instance descriptor's matrix at +0x50, docs/bvh.md §3.2), applied as `x' = M * x`.

Usage:
    python -I tools/bvh_rehydrate.py <dehydrated.bin> --geom VB[,IB] [--geom VB[,IB] ...] --out <rehydrated.bin>
                                     [--header-offset N] [--stride BYTES] [--index16] [--transform 12 floats]
                                     [--compare <reference.bin>]

One --geom per geometry index, in order: VB is a file of vertices (three float32 at the start of each --stride
record, default 12), IB a file of uint32 (or, with --index16, uint16) triangle indices; without IB the vertices are
a triangle list. --transform takes the 3x4 row-major matrix as 12 comma-separated floats and applies to every
geometry. --compare prints how many bytes differ from a reference structure. Exit status 0 on success, 2 on an
input error.
"""
import argparse
import contextlib
import math
import os
import struct
import sys
import tempfile

FLAG_DEHYDRATED = 0x01000000
FLAG_UPDATABLE = 0x00040000
NO_TRANSFORM = 0x007F7F7F
INVALID = 0xFFFFFFFF
TRI0, TRI1, BOX16, BOX32 = 0, 1, 4, 5
FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
HALF_POS_INF, HALF_NEG_INF = 0x7C00, 0xFC00
HALF_MIN_NORMAL = 2.0 ** -14
NAN32 = struct.unpack("<f", struct.pack("<I", 0x7FC00000))[0]
LEAVES_PER_CHUNK, CHUNK_STRIDE = 2048, 1 << 17
GROUP = 64
SLOT_OF_ORDER = (2, 0, 1)          # vertex-order value -> leaf vertex slot
M32 = 0xFFFFFFFF


# --- float32 / half helpers -------------------------------------------------------------------------------------

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def bits32(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits32(b):
    return struct.unpack("<f", struct.pack("<I", b & M32))[0]


def half_to_float(h):
    return struct.unpack("<e", struct.pack("<H", h & 0xFFFF))[0]


def float_to_half(x):
    try:
        return struct.unpack("<H", struct.pack("<e", x))[0]
    except OverflowError:
        return HALF_NEG_INF if x < 0 else HALF_POS_INF


def min_num(a, b):
    """IEEE minNum: a NaN operand loses."""
    if a != a:
        return b
    if b != b:
        return a
    return a if a <= b else b


def max_num(a, b):
    if a != a:
        return b
    if b != b:
        return a
    return a if a >= b else b


def half_down(x):
    """Largest half not above x (a box16 minimum)."""
    if x != x:
        return float_to_half(x)
    b = bits32(x)
    if x < 0.0:
        y = min_num(from_bits32((b + 0x1FFF) & ~0x1FFF), -HALF_MIN_NORMAL)
    else:
        y = from_bits32(b & ~0x1FFF)
        y = 0.0 if y < HALF_MIN_NORMAL else y
    return float_to_half(y)


def half_up(x):
    """Smallest half not below x (a box16 maximum)."""
    if x != x:
        return float_to_half(x)
    b = bits32(x)
    if x > 0.0:
        y = max_num(from_bits32((b + 0x1FFF) & ~0x1FFF), HALF_MIN_NORMAL)
    else:
        y = from_bits32(b & ~0x1FFF)
        y = 0.0 if y > -HALF_MIN_NORMAL else y
    return float_to_half(y)


# --- header and frames ------------------------------------------------------------------------------------------

class Header:
    """The XboxBvh4Header fields this tool needs (docs/bvh.md §3.1); offsets are from the header."""

    def __init__(self, buf, at=0):
        if at < 0 or at + 64 > len(buf):
            raise ValueError("no 64-byte header at offset %d" % at)
        (self.flags, self.size_units, self.crc32, self.num_elements) = struct.unpack_from("<4I", buf, at)
        self.bounds = struct.unpack_from("<6f", buf, at + 16)
        (self.offset_to_nodes, self.prim_remap_word, self.root, self.offset_to_elements, self.offset_to_parents,
         self.compression) = struct.unpack_from("<6I", buf, at + 40)

    @property
    def total_bytes(self):
        return (self.size_units + 1) * 64

    def kept_header_bytes(self):
        if self.prim_remap_word:
            n = self.offset_to_parents
        else:
            k = (self.flags >> 20) & 15
            n = 16 * (9 * k * k + 4)
        return (n + 15) & ~15

    def parent_entry(self, node_offset):
        """Address of the parent-table entry for the node at node_offset (all arithmetic is 32-bit)."""
        unit = ((node_offset - self.offset_to_elements) & M32) >> 6
        return (self.offset_to_parents + 4 * unit) & M32

    def stored_parent_entry(self, node_offset, size):
        """The entry's address when the structure stores it, else None: only the entries from the header end to
        offsetToElements are kept (docs/bvh.md §3.1), so a console structure's low units have none."""
        entry = self.parent_entry(node_offset)
        if 64 <= entry and entry + 4 <= min(self.offset_to_elements, size):
            return entry
        return None


def rotation(index):
    """Rotation index -> 3x3 matrix Rz * Ry * Rx, each angle a multiple of pi/8, products rounded to float32."""
    c = (1.0, math.cos(math.pi / 8), math.cos(math.pi / 4), math.cos(3 * math.pi / 8), 0.0)
    ix, iy, iz = index & 3, (index >> 2) & 3, (index >> 4) & 3
    cx, sx, cy, sy, cz, sz = (f32(c[ix]), f32(c[4 - ix]), f32(c[iy]), f32(c[4 - iy]), f32(c[iz]), f32(c[4 - iz]))
    rx = ((1.0, 0.0, 0.0), (0.0, cx, sx), (0.0, -sx, cx))
    ry = ((cy, 0.0, -sy), (0.0, 1.0, 0.0), (sy, 0.0, cy))
    rz = ((cz, sz, 0.0), (-sz, cz, 0.0), (0.0, 0.0, 1.0))

    def mul(a, b):
        out = []
        for r in range(3):
            row = []
            for col in range(3):
                acc = 0.0
                for m in range(3):
                    acc = f32(acc + f32(a[r][m] * b[m][col]))
                row.append(acc)
            out.append(tuple(row))
        return tuple(out)

    return mul(rz, mul(ry, rx))


class LeafFrame:
    """The compressionParams mapping (R * v - t) * s, or the identity."""

    def __init__(self, hdr):
        word = hdr.compression
        self.on = word != NO_TRANSFORM
        self.r = rotation((word >> 24) & 0x3F)
        self.t = [0.0, 0.0, 0.0]
        self.s = [1.0, 1.0, 1.0]
        if word >> 30 & 1:
            for a in range(3):
                lo, hi = hdr.bounds[a], hdr.bounds[a + 3]
                if lo > 0.0 or hi < 0.0:
                    self.t[a] = f32(f32(lo + hi) * 0.5)
        if word >> 31 & 1:
            self.s = [from_bits32(((word >> (8 * a)) & 0xFF) << 23) for a in range(3)]

    def apply(self, v):
        if not self.on:
            return v
        out = []
        for row in range(3):
            acc = f32(self.r[row][0] * v[0])
            acc = f32(self.r[row][1] * v[1] + acc)
            acc = f32(self.r[row][2] * v[2] + acc)
            out.append(f32(f32(acc - self.t[row]) * self.s[row]))
        return tuple(out)


# --- geometry ---------------------------------------------------------------------------------------------------

class Geometry:
    """float3 vertices (the first 12 bytes of each `stride` record) with optional uint32/uint16 triangle indices and
    an optional 3x4 row-major transform applied as x' = M * x."""

    def __init__(self, vertices, stride=12, indices=None, index16=False, transform=None):
        if stride < 12:
            raise ValueError("vertex stride %d is below 12" % stride)
        self.vertices, self.stride, self.indices, self.index16 = bytes(vertices), stride, indices, index16
        self.transform = transform

    def triangle(self, prim):
        if self.indices is None:
            return 3 * prim, 3 * prim + 1, 3 * prim + 2
        if self.index16:
            return struct.unpack_from("<3H", self.indices, 6 * prim)
        return struct.unpack_from("<3I", self.indices, 12 * prim)

    def vertex(self, i):
        v = struct.unpack_from("<3f", self.vertices, i * self.stride)
        m = self.transform
        if m is None:
            return v
        out = []
        for row in range(3):
            acc = f32(m[row][0] * v[0])
            acc = f32(m[row][1] * v[1] + acc)
            acc = f32(m[row][2] * v[2] + acc)
            out.append(f32(acc + m[row][3]))
        return tuple(out)


# --- rehydration ------------------------------------------------------------------------------------------------

def read_u24(buf, at):
    return buf[at] | buf[at + 1] << 8 | buf[at + 2] << 16


def leaf_records(src, hdr):
    """Yield (leaf output offset, geometry, flags, prim0, prim1 or None, vertex order) for every leaf."""
    remaining, chunk = hdr.num_elements, 0
    while remaining > 0:
        c = min(LEAVES_PER_CHUNK, remaining)
        base = hdr.offset_to_elements + chunk * CHUNK_STRIDE
        if base + 11 * c > len(src):
            raise ValueError("leaf chunk %d runs past the input" % chunk)
        for i in range(c):
            prim1 = read_u24(src, base + 7 * c + 3 * i)
            yield (base + 64 * i, read_u24(src, base + 3 * i), src[base + 3 * c + i], read_u24(src, base + 4 * c + 3 * i),
                   None if prim1 == 0xFFFFFF else prim1, src[base + 10 * c + i])
        remaining -= c
        chunk += 1


def write_leaves(src, dst, hdr, geometries, frame):
    for out, gi, flags, prim0, prim1, order in leaf_records(src, hdr):
        if gi >= len(geometries):
            raise ValueError("a leaf uses geometry index %d but %d geometries were given" % (gi, len(geometries)))
        g = geometries[gi]
        tri = [frame.apply(g.vertex(i)) for i in g.triangle(prim0)]
        if prim1 is None:
            fourth = (NAN32, NAN32, NAN32)
        else:
            j = g.triangle(prim1)
            fourth = frame.apply(g.vertex(j[0] if order & 0x20 else j[1] if order & 0x80 else j[2]))
        a, b = order & 3, (order >> 2) & 3
        slots = [None, None, None, fourth]
        for which, value in zip((a, b, (a ^ 3) - b), tri):
            slots[SLOT_OF_ORDER[which]] = value
        struct.pack_into("<12f4I", dst, out, *slots[0], *slots[1], *slots[2], *slots[3],
                         (gi << 8 | flags) & M32, prim0, INVALID if prim1 is None else prim1,
                         (order & 0xF) | ((order << 4) & 0xF00))


def node_bounds(dst, ptr):
    """(min xyz, max xyz) of the rehydrated node a child pointer names."""
    kind, at = ptr & 7, (ptr >> 3) << 6
    if kind == TRI0:
        v = struct.unpack_from("<12f", dst, at)
        pts = [v[0:3], v[3:6], v[6:9]] + ([v[9:12]] if struct.unpack_from("<I", dst, at + 48)[0] & 0x80 else [])
    elif kind == TRI1:
        v = struct.unpack_from("<9f", dst, at + 12)
        pts = [v[0:3], v[3:6], v[6:9]]
    elif kind == BOX32:
        f = struct.unpack_from("<24f", dst, at + 16)
        pts = None
        lo = [f[a] for a in range(3)]
        hi = [f[3 + a] for a in range(3)]
        for s in range(1, 4):
            for a in range(3):
                lo[a] = min_num(lo[a], f[6 * s + a])
                hi[a] = max_num(hi[a], f[6 * s + 3 + a])
        return tuple(lo) + tuple(hi)
    elif kind == BOX16:
        h = struct.unpack_from("<24H", dst, at + 16)
        out = []
        for c in range(6):
            best = h[c]
            for s in range(1, 4):
                cand = h[6 * s + c]
                x, y = half_to_float(best), half_to_float(cand)
                if x != x or (y == y and ((y < x) if c < 3 else (y > x))):
                    best = cand
            out.append(half_to_float(best))
        return tuple(out)
    else:
        return (FLT_MAX,) * 3 + (-FLT_MAX,) * 3
    lo, hi = list(pts[0]), list(pts[0])
    for p in pts[1:]:
        for a in range(3):
            lo[a] = min_num(lo[a], p[a])
            hi[a] = max_num(hi[a], p[a])
    return tuple(lo) + tuple(hi)


def levels(src, hdr):
    """[(box32 start, box32 count, box16 start, box16 count)] bottom-up, offsets from offsetToNodes."""
    end = hdr.total_bytes
    count = struct.unpack_from("<I", src, end - 4)[0]
    table = end - 4 - 8 * count
    if table < 64:
        raise ValueError("level table with %d levels does not fit" % count)
    out, prev = [], 0
    for lv in range(count):
        b32, b16 = struct.unpack_from("<2I", src, table + 8 * lv)
        if b32 < prev or b16 < b32 or (b32 - prev) % 128 or (b16 - b32) % 64:
            raise ValueError("level %d of the level table is out of order" % lv)
        out.append((prev, (b32 - prev) // 128, b32, (b16 - b32) // 64))
        prev = b16
    return out


def write_box_groups(src, dst, hdr, start, count, wide):
    """Rehydrate `count` BOX32 (wide) or BOX16 nodes stored from offsetToNodes + start."""
    node_size, group_bytes = (128, 1 << 13) if wide else (64, 1 << 12)
    for g in range((count + GROUP - 1) // GROUP):
        n = min(GROUP, count - g * GROUP)
        base = hdr.offset_to_nodes + start + g * group_bytes
        words = struct.unpack_from("<%dI" % n, src, base)
        total_children = sum(w >> 24 for w in words)
        child_at = base + 4 * n
        value_at = base + 4 * (n + total_children)
        for t, word in enumerate(words):
            kids, mask = word >> 24, word & 0xFFFFFF
            children = list(struct.unpack_from("<%dI" % kids, src, child_at)) + [INVALID] * (4 - kids)
            child_at += 4 * kids
            comps = []
            for slot in range(4):
                if slot < kids:
                    b = node_bounds(dst, children[slot])
                    comps += list(b) if wide else [half_down(b[0]), half_down(b[1]), half_down(b[2]),
                                                   half_up(b[3]), half_up(b[4]), half_up(b[5])]
                else:
                    comps += [FLT_MAX] * 3 + [-FLT_MAX] * 3 if wide else [HALF_POS_INF] * 3 + [HALF_NEG_INF] * 3
            for bit in range(24):
                if mask >> bit & 1:
                    if wide:
                        comps[bit] = struct.unpack_from("<f", src, value_at)[0]
                        value_at += 4
                    else:
                        comps[bit] = struct.unpack_from("<H", src, value_at)[0]
                        value_at += 2
            at = base + t * node_size
            if wide:
                struct.pack_into("<4I24f4I", dst, at, *children, *comps, 0, 0, 0, 0)
            else:
                struct.pack_into("<4I24H", dst, at, *children, *comps)
            me = (at >> 6) << 3 | (BOX32 if wide else BOX16)
            for child in children:
                if (child & 6) == 4 or ((child & 7) == TRI0 and hdr.flags & FLAG_UPDATABLE):
                    entry = hdr.stored_parent_entry(((child >> 3) << 6) & M32, len(dst))
                    if entry is not None:
                        struct.pack_into("<I", dst, entry, me)


def rehydrate(dehydrated, geometries, header_offset=0):
    """Return the rehydrated structure (bytearray) for a dehydrated BLAS and one Geometry per geometry index."""
    hdr = Header(dehydrated, header_offset)
    if not hdr.flags & FLAG_DEHYDRATED:
        raise ValueError("the dehydrated flag (0x01000000) is not set")
    src = bytes(dehydrated[header_offset:header_offset + hdr.total_bytes])
    if len(src) < hdr.total_bytes:
        raise ValueError("the input is shorter than the %d bytes the header declares" % hdr.total_bytes)
    dst = bytearray(hdr.total_bytes)
    kept = hdr.kept_header_bytes()
    if kept > len(src):
        raise ValueError("the kept header bytes (%d) run past the input" % kept)
    dst[:kept] = src[:kept]
    struct.pack_into("<I", dst, 0, hdr.flags & ~FLAG_DEHYDRATED & M32)
    root_entry = hdr.stored_parent_entry(((hdr.root >> 3) << 6) & M32, len(dst))
    if root_entry is not None:
        struct.pack_into("<I", dst, root_entry, INVALID)
    write_leaves(src, dst, hdr, list(geometries), LeafFrame(hdr))
    if hdr.root & 7 in (BOX16, BOX32):
        for b32_start, b32_count, b16_start, b16_count in levels(src, hdr):
            write_box_groups(src, dst, hdr, b32_start, b32_count, True)
            write_box_groups(src, dst, hdr, b16_start, b16_count, False)
    return dst


# --- command line -----------------------------------------------------------------------------------------------

def parse_transform(text):
    vals = [float(x) for x in text.split(",")]
    if len(vals) != 12:
        raise ValueError("--transform needs 12 comma-separated floats, got %d" % len(vals))
    try:
        rounded = [f32(v) for v in vals]
    except (OverflowError, struct.error) as exc:
        raise ValueError("--transform values must fit in finite float32") from exc
    if not all(math.isfinite(v) for v in rounded):
        raise ValueError("--transform values must fit in finite float32")
    return tuple(tuple(rounded[4 * row:4 * row + 4]) for row in range(3))




def _make_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _publish(dest, data):
    """Write output binary next to dest and atomically swap it in, so an interrupted or failing write
    leaves the previous file intact instead of truncated."""
    _make_parent(dest)
    directory = os.path.dirname(os.path.abspath(dest))
    temp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=directory, prefix=".bvh_rehydrate-", suffix=".tmp", delete=False) as f:
            temp = f.name
            f.write(data)
        os.replace(temp, dest)
    except BaseException:
        if temp is not None:
            with contextlib.suppress(OSError):
                os.unlink(temp)
        raise

def main(argv=None):
    ap = argparse.ArgumentParser(description="Rehydrate a dehydrated XDXR BVH4 BLAS on the CPU (docs/bvh.md §4a).")
    ap.add_argument("dehydrated")
    ap.add_argument("--geom", action="append", default=[], metavar="VB[,IB]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--header-offset", type=int, default=0)
    ap.add_argument("--stride", type=int, default=12)
    ap.add_argument("--index16", action="store_true")
    ap.add_argument("--transform")
    ap.add_argument("--compare")
    a = ap.parse_args(argv)
    if os.path.isdir(a.out):
        print("error: --out is a folder, not a file: %s" % a.out, file=sys.stderr)
        return 2
    # The output must never replace a file the tool was asked to read. samefile also catches a
    # case-only difference on Windows and macOS's default volume (tools/hashdict.py).
    if os.path.exists(a.out):
        inputs = [a.dehydrated]
        inputs += [p for spec in a.geom for p in spec.split(",") if p]
        if a.compare:
            inputs.append(a.compare)
        for path in inputs:
            if os.path.isfile(path) and os.path.samefile(a.out, path):
                print("error: --out must not overwrite the input %s" % path, file=sys.stderr)
                return 2
    try:
        transform = parse_transform(a.transform) if a.transform else None
        geometries = []
        for spec in a.geom:
            parts = spec.split(",")
            with open(parts[0], "rb") as f:
                vb = f.read()
            ib = None
            if len(parts) > 1 and parts[1]:
                with open(parts[1], "rb") as f:
                    ib = f.read()
            geometries.append(Geometry(vb, a.stride, ib, a.index16, transform))
        with open(a.dehydrated, "rb") as f:
            data = f.read()
        out = rehydrate(data, geometries, a.header_offset)
    except (OSError, ValueError, struct.error) as e:
        print("error: %s" % e, file=sys.stderr)
        return 2
    try:
        _publish(a.out, out)
    except OSError as e:
        print("error: cannot write %s: %s" % (a.out, e), file=sys.stderr)
        return 2
    print("rehydrated %d bytes, %d leaves, %d geometries -> %s" % (len(out), Header(data, a.header_offset).num_elements,
                                                                  len(geometries), a.out))
    if a.compare:
        try:
            with open(a.compare, "rb") as f:
                ref = f.read()
        except OSError as e:
            print("error: cannot read %s: %s" % (a.compare, e), file=sys.stderr)
            return 2
        diff = sum(x != y for x, y in zip(out, ref)) + abs(len(out) - len(ref))
        print("differs from %s in %d bytes" % (a.compare, diff))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
