"""Synthetic tests for tools/anim_clips.py.

    python -I tests/test_anim_clips.py

Every byte here is built by the test: a fake RPF9 data area holding clip-dictionary resources made by the
encoder below (dictionary root, clip, animation pack, animation, sequences and every decoded channel type,
including a type-10 channel). No game file or game asset name is used.
"""
import collections
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOL = os.path.join(ROOT, "tools", "anim_clips.py")
sys.path.insert(0, os.path.join(ROOT, "tools"))

import anim_clips  # noqa: E402
from anim_index import joaat32  # noqa: E402

SYSTEM = 0x50000000
BONEROTATION = joaat32("bonerotation")
BONETRANSLATION = joaat32("bonetranslation")
GENERICCONTROL = joaat32("genericcontrol")


# ---------------------------------------------------------------- encoders


class Bits:
    """LSB-first bit writer."""

    def __init__(self):
        self.value, self.n = 0, 0

    def put(self, x, k):
        assert 0 <= x < (1 << k) or k == 0 and x == 0
        self.value |= x << self.n
        self.n += k

    def bytes(self, length=None):
        length = (self.n + 7) // 8 if length is None else length
        return self.value.to_bytes(length, "little")


def zz(v):
    return 2 * v if v >= 0 else -2 * v - 1


def encode_delta2(ints, quantum, offset):
    """Type-10 channel for non-negative integers `ints` (value = int * quantum + offset)."""
    nf = len(ints)
    nc = (nf + 63) // 64
    starts = [ints[64 * c] for c in range(nc)]
    raw = []
    for c in range(nc):
        v = 0
        for j in range(64):
            f = 64 * c + j + 1
            if f >= nf:
                break
            step = ints[f] - ints[f - 1]
            raw.append(zz(step - v))
            v = step
    assert len(raw) == nf - 1
    raw += [0] * ((-len(raw)) % 8)
    widths = [max(x.bit_length() for x in raw[g:g + 8]) for g in range(0, len(raw), 8)]
    total = sum(widths)
    c3 = (total + 31) // 32
    positions, acc = [], 0
    for c in range(nc):
        positions.append(acc)
        acc += sum(widths[8 * c:8 * c + 8])
    c1 = max(1, max(positions).bit_length())
    c2 = max(1, max(starts).bit_length())
    front = Bits()
    for p in positions:
        front.put(p, c1)
    for s in starts:
        front.put(s, c2)
    for k in widths:
        front.put(0, k)
        front.put(1, 1)
    lanes = [Bits() for _ in range(8)]
    for g, k in enumerate(widths):
        for j in range(8):
            lanes[j].put(raw[8 * g + j], k)
    front_bytes = front.bytes(((front.n + 31) // 32) * 4)
    words = []
    lane_words = [lanes[j].value.to_bytes(4 * c3, "little") for j in range(8)]
    for w in range(c3):
        for j in range(8):
            words.append(lane_words[j][4 * w:4 * w + 4])
    blob = front_bytes + b"".join(words)
    head = struct.pack("<IIff", (16 + len(blob)) // 4, c1 | c2 << 8 | c3 << 16, quantum, offset)
    return head + blob


def encode_sequence(nf, channels):
    """channels: list of (type, track, comp, spec). spec per type:
    0/1: (x, y, z); 2: f; 3: [f32 per frame]; 4: (bits, quantum, offset, [ints]);
    5: (frame_bits, value_bits, [table ints], quantum, offset, [indices]); 7/8: None; 10: (quantum, offset, [ints]).
    """
    by_type = {t: [] for t in range(11)}
    for ch in channels:
        by_type[ch[0]].append(ch)
    defs = bytearray()
    frame_fields = []
    for t in range(11):
        for _t, _track, _comp, spec in by_type[t]:
            if t in (0, 1):
                defs += struct.pack("<3f", *spec)
            elif t == 2:
                defs += struct.pack("<f", spec)
            elif t == 3:
                frame_fields.append((32, [struct.unpack("<I", struct.pack("<f", v))[0] for v in spec]))
            elif t == 4:
                bits, quantum, offset, ints = spec
                defs += struct.pack("<Iff", bits, quantum, offset)
                frame_fields.append((bits, ints))
            elif t == 5:
                fbits, vbits, table, quantum, offset, idx = spec
                tb = Bits()
                for x in table:
                    tb.put(x, vbits)
                words = (tb.n + 31) // 32
                defs += struct.pack("<IIIff", fbits, vbits, words, quantum, offset) + tb.bytes(4 * words)
                frame_fields.append((fbits, idx))
            elif t == 10:
                defs += encode_delta2(spec[2], spec[0], spec[1])
    # type-4 bits are summed in the header; frame fields go type 3, then 4, then 5 (the order above)
    qbits = sum(spec[0] for t, _tr, _c, spec in channels if t == 4)
    need = sum(b for b, _ in frame_fields)
    fl = ((need + 31) // 32) * 4
    frames = bytearray()
    for f in range(nf):
        row = Bits()
        for bits, vals in frame_fields:
            row.put(vals[f], bits)
        frames += row.bytes(fl)
    counts = [len(by_type[t]) for t in range(11)]
    lists = bytearray()
    for t in range(11):
        entries = [track << 2 | comp for _t, track, comp, _s in by_type[t]]
        entries += [0] * ((-len(entries)) % 4)
        lists += struct.pack(f"<{len(entries)}H", *entries)
    data = bytes(defs) + bytes(frames) + struct.pack("<11H", *counts) + bytes(lists)
    header = struct.pack("<5I5H2B", 0x1234, len(data), 0, len(defs), 0x20 + len(data), 2, nf, fl, 0, qbits, 11, 0)
    return header + data


class Page:
    """System-page builder: offsets are page offsets, pointers are 0x50000000 + offset."""

    def __init__(self):
        self.b = bytearray(0x80)

    def alloc(self, data, align=16):
        self.b += bytes((-len(self.b)) % align)
        at = len(self.b)
        self.b += data
        return at

    def ptr(self, at, target):
        struct.pack_into("<Q", self.b, at, 0 if target is None else SYSTEM + target)

    def string(self, at, text):
        p = self.alloc(text.encode() + b"\x00")
        self.ptr(at, p)
        struct.pack_into("<HH", self.b, at + 8, len(text), len(text) + 1)

    def atmap(self, at, items):
        """items: [(key, value offset)] spread over 11 buckets."""
        buckets = self.alloc(bytes(8 * 11))
        heads = [None] * 11
        for key, value in items:
            e = self.alloc(bytes(0x18))
            struct.pack_into("<Q", self.b, e, key)
            self.ptr(e + 8, value)
            self.ptr(e + 16, heads[key % 11])
            heads[key % 11] = e
        for i, h in enumerate(heads):
            self.ptr(buckets + 8 * i, h)
        self.ptr(at, buckets)
        struct.pack_into("<I", self.b, at + 8, 11)
        struct.pack_into("<I", self.b, at + 0x10, len(items))


def animation(page, frames, tracks, sequences):
    a = page.alloc(bytes(0x70))
    struct.pack_into("<H", page.b, a + 0x24, frames)
    struct.pack_into("<f", page.b, a + 0x28, (frames - 1) / 30.0)
    seq_ptrs = page.alloc(bytes(8 * len(sequences)))
    for i, s in enumerate(sequences):
        page.ptr(seq_ptrs + 8 * i, page.alloc(s))
    page.ptr(a + 0x50, seq_ptrs)
    struct.pack_into("<HH", page.b, a + 0x58, len(sequences), len(sequences))
    tr = page.alloc(b"".join(struct.pack("<3I", *t) for t in tracks))
    page.ptr(a + 0x60, tr)
    struct.pack_into("<HH", page.b, a + 0x68, len(tracks), len(tracks))
    return a


def table(p, field, groups, slot, entries, control=lambda data: 0x11):
    """An open-addressing table at `field`: {u32 groups; u32 entries; ptr}, each group 16 control bytes (0x80 empty)
    then 16 slots of `slot` bytes. `entries` are (group, slot index, slot bytes)."""
    block = bytearray(bytes([0x80]) * 16 + bytes(16 * slot)) * groups
    for g, i, data in entries:
        base = g * (16 + 16 * slot)
        block[base + i] = control(data)
        block[base + 16 + slot * i:base + 16 + slot * i + len(data)] = data
    struct.pack_into("<II", p.b, field, groups, len(entries))
    p.ptr(field + 8, p.alloc(bytes(block)))


def matcher_array(p, field, data, count):
    """A `.motionmatcher` array: pointer at `field`, u32 count at `field - 4`, u32 capacity at `field - 8`."""
    struct.pack_into("<II", p.b, field - 8, count, count)
    p.ptr(field, p.alloc(data))


def pose_value(sample, column):
    """The fixture's pose matrix: exact in f16."""
    return sample + column / 16


BONES = [("root", -1, 0, (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),
         ("spine", 0, anim_clips.bone_id("spine"), (0.0, 0.0, 0.6, 0.8), (0.0, 1.0, 0.0)),
         ("head", 1, anim_clips.bone_id("head"), (0.0, 0.0, 0.0, 1.0), (0.0, 0.5, 0.0))]


def skeleton(p):
    """A three-bone chain: 80-byte bones, local and inverse bind matrices, s16 parents, and a tag table that maps
    each bone's joaat32(name) and its u16 id to its index."""
    a = p.alloc(bytes(0x80))
    n = len(BONES)
    struct.pack_into("<HH", p.b, a + 0x6C, n, n)
    bones, local, inverse, world = b"", b"", b"", []
    for i, (name, parent, bone_id, q, t) in enumerate(BONES):
        b = bytearray(80)
        struct.pack_into("<4f3f4x4f", b, 0, *q, *t, 1, 1, 1, 1)
        struct.pack_into("<Q", b, 0x30, SYSTEM + p.alloc(name.encode() + bytes(1)))
        struct.pack_into("<hHHH", b, 0x46, parent, i, bone_id, i)
        bones += bytes(b)
        r = anim_clips.quaternion_rows(*q)
        m = [r[0] + [0.0], r[1] + [0.0], r[2] + [0.0], list(t) + [1.0]]
        local += struct.pack("<16f", *(x for row in m for x in row))
        g = m if parent < 0 else [[sum(m[x][k] * world[parent][k][y] for k in range(4)) for y in range(4)]
                                  for x in range(4)]
        world.append(g)
        rt = [[g[y][x] for y in range(3)] for x in range(3)]                      # rotation transposed
        tt = [-sum(g[3][k] * rt[k][y] for k in range(3)) for y in range(3)]
        inverse += struct.pack("<16f", *rt[0], 0, *rt[1], 0, *rt[2], 0, *tt, 1)
    p.ptr(a + 0x28, p.alloc(bones))
    p.ptr(a + 0x30, p.alloc(inverse))
    p.ptr(a + 0x38, p.alloc(local))
    p.ptr(a + 0x40, p.alloc(struct.pack(f"<{n}h", *(b[1] for b in BONES))))
    slots = [(0xFFFFFFFF, 0)] * 8
    for i, (name, _parent, bone_id, _q, _t) in enumerate(BONES):
        slots[2 * i], slots[2 * i + 1] = (joaat32(name), i), (bone_id, i)
    struct.pack_into("<II", p.b, a + 0x18, 8, 2 * n)
    p.ptr(a + 0x20, p.alloc(b"".join(struct.pack("<II", *x) for x in slots)))
    # the name tree: "head", "root", then "sp" with the child "ine"; values point into the bone array at +46
    tree = struct.pack("<HH", n, 46) + struct.pack("<HHh", 11, 0, 42) + b"head\0" + struct.pack(
        "<HHh", 11, 0, 33) + b"root\0" + struct.pack("<HHh", 0, 9, -26) + b"sp\0" + struct.pack(
        "<HHh", 0, 0, 15) + b"ine\0" + b"\0" + struct.pack("<3H", 2, 0, 1)
    p.ptr(a + 0x48, p.alloc(tree))
    return a


COLUMN_WEIGHTS = list(struct.unpack("<16f", struct.pack("<16f", *[0.2] * 3 + [0.4 / 6] * 6 + [0.0] * 7)))  # sum 1


def features(p, rows=True):
    """A `.motionmatcher` feature object: a vec3 feature at column 0 and a two-vec3 one at column 3, a name table
    that maps another hash of each name to its index, and a 9-sample x 9-column pose matrix, row-major when the
    `+0x30` object is there (`rows`), else in blocks of 8 samples."""
    obj = p.alloc(bytes(0x140))
    struct.pack_into("<I", p.b, obj + 0x18, 1)
    p.ptr(obj + 0x28, skeleton(p))
    samples, cols = 9, 9
    if rows:
        b = p.alloc(bytes(0x70))                      # form 2: two weighted bones, a copy of the first feature
        struct.pack_into("<I", p.b, b + 8, 2)
        struct.pack_into("<IIff", p.b, b + 0x10, joaat32("spine"), joaat32("head"), 0.25, 0.75)
        struct.pack_into("<H", p.b, b + 0x3E, 1)
        struct.pack_into("<II", p.b, b + 0x48, joaat32("spine"), 1)
        struct.pack_into("<IH", p.b, b + 0x68, 3, 6)
        p.ptr(obj + 0x30, b)
        stride = 16
        h = [pose_value(r, c) if c < cols else 0.0 for r in range(samples) for c in range(stride)]
        struct.pack_into("<IIII", p.b, obj + 0xF8, samples, cols, 1, 1)
        transforms = b"".join(struct.pack("<12f", 0, 0, 0, 1, r, 0, 0, 0, 1, 1, 1, 0) for r in range(samples))
        matcher_array(p, obj + 0xA8, transforms, samples)
    else:
        stride = cols + 1
        h = [0.0] * (2 * 8 * stride)
        for r in range(16):
            for c in range(stride):
                h[(r // 8) * stride * 8 + c * 8 + r % 8] = pose_value(min(r, samples - 1), c) if c < cols else 0.0
        struct.pack_into("<II", p.b, obj + 0xF8, cols, cols)
        struct.pack_into("<I", p.b, obj + 0x128, samples)
    matcher_array(p, obj + 0xF0, struct.pack(f"<{len(h)}e", *h), len(h) // 16)
    matcher_array(p, obj + 0x98, struct.pack("<9I", *([1] * 4 + [6] * 5)), samples)  # key bits, one per sample
    matcher_array(p, obj + 0xB8, struct.pack("<16f", *COLUMN_WEIGHTS), 2)       # one per column, chunks of 8
    struct.pack_into("<ffI", p.b, obj + 0xC0, 25.0, 1.0, 1 if rows else 3)
    items = []
    for column, width, name, kind in ((0, 3, joaat32("spine"), 1), (3, 6, joaat32("head"), 3)):
        f = p.alloc(bytes(0x30))
        struct.pack_into("<IHH", p.b, f + 8, column, width, 1)
        struct.pack_into("<II", p.b, f + 0x18, name, kind)
        items.append(f)
    matcher_array(p, obj + 0x40, struct.pack("<2Q", *(SYSTEM + f for f in items)), 2)
    table(p, obj + 0x48, 2, 8, [(0, 4, struct.pack("<II", 0x21, 1)), (1, 0, struct.pack("<II", 0xA2, 0))],
          lambda data: data[0] & 0x7F)                  # placed by the key: group (h >> 7) & 1, control h & 0x7F
    return obj


def motion_matcher(p, name, refs=True, pose_rows=True):
    """A `.motionmatcher` with two records: record 0 is the dictionary's own `.clip` (hash 1), record 1 a clip that
    is not in the archive. Its own key is (3, joaat32(name))."""
    m = p.alloc(bytes(0x190))
    p.ptr(m + 0x20, features(p, pose_rows))           # the feature object
    p.b[m + 0x28:m + 0x38] = bytes(range(0x80, 0x90))   # 16-byte id; +0x30 is not a pointer
    p.string(m + 0x38, "pack:/one.motionmatcher")
    records = struct.pack("<fffIQ", 0.0, 30.0, -1.0, 0, 0) + struct.pack("<fffIQ", 0.5, 15.0, -1.0, 1, 0)
    matcher_array(p, m + 0x58, records, 2)
    matcher_array(p, m + 0x68, struct.pack("<3I", 0, 4, 9), 3)
    matcher_array(p, m + 0x78, struct.pack("<3H", 0, 1, 2), 3)
    matcher_array(p, m + 0xB8, bytes([0, 1]), 2)
    clips = [(1, joaat32(name)), (0xC1105, 0xD1C7)]
    if refs:
        matcher_array(p, m + 0x98, b"".join(struct.pack("<QII", 0, ch, dh) for ch, dh in clips), 2)
    table(p, m + 0xA0, 1, 16, [(0, 3, struct.pack("<IIII", clips[0][0], 0x77, 0, 0)),
                               (0, 9, struct.pack("<IIII", clips[1][0], 0x78, 1, 0))],
          lambda data: anim_clips.fmix64(int.from_bytes(data[:8], "little")) & 0x7F)  # one group: control only
    table(p, m + 0x170, 2, 8, [(0, 0, struct.pack("<II", 0x5A, 2)), (1, 5, struct.pack("<II", 1, 0)),
                               (1, 6, struct.pack("<II", 0x5B, 1))])
    struct.pack_into("<II", p.b, m + 0x188, 3, joaat32(name))
    struct.pack_into("<I", p.b, m + 0x44, 1)          # count only: +0x40 is the name's length and capacity
    p.ptr(m + 0x48, p.alloc(struct.pack("<10I", *([0] * 8 + [1, 0]))))
    matcher_array(p, m + 0x118, struct.pack("<I", 9), 1)                       # the pose sample count
    matcher_array(p, m + 0x128, struct.pack("<H", 0), 1)
    return m


def dictionary(name="synthetic@dict", sequences=None, frames=None, tracks=None, external=0, children=(),
               external_hashes=(), parameters=1, times=None, axis=None, node=None, sample_lists=None,
               node_axis_count=None, node_pairs=None, matcher_refs=True, pose_rows=True):
    """One clip dictionary page: one `.clip`, one `.motion`, one `.motionmatcher`, one local pack with one animation."""
    p = Page()
    pm = p.alloc(bytes(16))
    p.b[pm + 8], p.b[pm + 9] = 1, 0
    p.ptr(8, pm)
    p.string(0x38, name)
    struct.pack_into("<I", p.b, 0x68, joaat32(name))
    anim = animation(p, frames, tracks, sequences)
    pack = p.alloc(bytes(0x70))
    p.string(pack + 0x38, name + "-0")
    struct.pack_into("<I", p.b, pack + 0x68, joaat32(name + "-0"))
    p.atmap(pack + 0x48, [(joaat32("one"), anim)])
    clip = p.alloc(bytes(0x80))
    struct.pack_into("<I", p.b, clip + 0x20, 2)
    p.string(clip + 0x28, "pack:/one.clip")
    struct.pack_into("<f", p.b, clip + 0x78, (frames - 1) / 30.0)
    entries = p.alloc(bytes(0x38))
    p.ptr(entries, anim)
    struct.pack_into("<I", p.b, entries + 8, joaat32("one"))
    struct.pack_into("<3f", p.b, entries + 0x28, 0.0, (frames - 1) / 30.0, 1.0)
    lst = p.alloc(bytes(0x18))
    p.ptr(lst, entries)
    struct.pack_into("<HH", p.b, lst + 8, 1, 1)
    p.ptr(clip + 0x68, lst)
    motion = p.alloc(bytes(0x150))
    struct.pack_into("<I", p.b, motion + 0x20, 0x5000)
    p.string(motion + 0x30, "pack:/one.motion")
    struct.pack_into("<I", p.b, motion + 0x58, parameters)          # 1 or 2
    if children:                                                     # {u64 0; u32 clip hash; u32 dictionary hash}
        kids = p.alloc(b"".join(struct.pack("<QII", 0, ch, dh) for ch, dh in children))
        p.ptr(motion + 0x60, kids)
        struct.pack_into("<HH", p.b, motion + 0x68, len(children), len(children))
    if times is not None:      # +0x70: per child {f32 duration, f32 blend duration, (child time, blend time) curve}
        table = p.alloc(bytes(0x28 * len(times)))
        for i, (duration, blend, points) in enumerate(times):
            e = table + 0x28 * i
            struct.pack_into("<ff", p.b, e, duration, blend)
            p.ptr(e + 8, p.alloc(b"".join(struct.pack("<ff", x, y) for x, y in points)))
            struct.pack_into("<HH", p.b, e + 16, len(points), len(points))
        p.ptr(motion + 0x70, table)
        struct.pack_into("<HH", p.b, motion + 0x78, len(times), len(times))
    if axis is not None:       # a type-3 node: {u64 0, u32 3, u32 0, axes, (index, value) samples, f32 min, f32 max}
        names, samples, lo, hi = axis
        node_at = p.alloc(bytes(0x40))
        struct.pack_into("<I", p.b, node_at + 8, 3)
        p.ptr(node_at + 0x10, p.alloc(b"".join(struct.pack("<IHH", h, 0, 0) for h in names)))
        struct.pack_into("<HH", p.b, node_at + 0x18, len(names), len(names))
        p.ptr(node_at + 0x20, p.alloc(b"".join(struct.pack("<If", i, v) for i, v in samples)))
        struct.pack_into("<HH", p.b, node_at + 0x28, len(samples), len(samples))
        struct.pack_into("<ff", p.b, node_at + 0x30, lo, hi)
    elif node is not None:     # (type, [(name hash, u16, u16)], sample count): type 1 or a multi-axis node
        kind, axes, count = node
        node_at = p.alloc(bytes(0xB0))
        struct.pack_into("<I", p.b, node_at + 8, kind)
        if kind == 1:
            struct.pack_into("<I", p.b, node_at + 0x20, count)
        else:
            p.ptr(node_at + 0x10, p.alloc(b"".join(struct.pack("<IHH", *a) for a in axes)))
            n_axes = len(axes) if node_axis_count is None else node_axis_count   # override: an axis array overrun
            struct.pack_into("<HH", p.b, node_at + 0x18, n_axes, n_axes)
            struct.pack_into("<I", p.b, node_at + anim_clips.NODE_SAMPLE_COUNT_AT[kind], count)
            if node_pairs is not None:
                p.ptr(node_at + 0x20, p.alloc(b"".join(struct.pack("<IIII", x, 0, y, 0) for x, y in node_pairs)))
                struct.pack_into("<HH", p.b, node_at + 0x28, len(node_pairs), len(node_pairs))
    if axis is not None or node is not None:
        if parameters == 1:
            p.ptr(motion + 0x138, node_at)
        else:                  # +0x120 points at a block whose +0x18 is the node
            block = p.alloc(bytes(0x20))
            p.ptr(block + 0x18, node_at)
            p.ptr(motion + 0x120, block)
            struct.pack_into("<HH", p.b, motion + 0x128, 1, 1)
    if sample_lists is not None:   # +0x90: per sample {ptr, u16 count} of {u32 child, f32, f32}
        table = p.alloc(bytes(16 * len(sample_lists)))
        for i, items in enumerate(sample_lists):
            p.ptr(table + 16 * i, p.alloc(b"".join(struct.pack("<Iff", *it) for it in items)))
            struct.pack_into("<HH", p.b, table + 16 * i + 8, len(items), len(items))
        p.ptr(motion + 0x90, table)
        struct.pack_into("<HH", p.b, motion + 0x98, len(sample_lists), len(sample_lists))
    matcher = motion_matcher(p, name, refs=matcher_refs, pose_rows=pose_rows)
    p.atmap(0x48, [(joaat32(name) << 32 | 1, clip), (joaat32(name) << 32 | 2, motion),
                   (joaat32(name) << 32 | 3, matcher)])
    hashes = [0x1000 + i for i in range(external)] + list(external_hashes)
    refs = p.alloc(bytes(16 * (1 + len(hashes))))
    p.ptr(refs, pack)
    struct.pack_into("<I", p.b, refs + 8, joaat32(name + "-0"))
    for i, h in enumerate(hashes):
        struct.pack_into("<I", p.b, refs + 16 * (i + 1) + 8, h)
    p.ptr(0x70, refs)
    struct.pack_into("<HH", p.b, 0x78, 1 + len(hashes), 1 + len(hashes))
    p.b += bytes((-len(p.b)) % 16)
    return bytes(p.b)


# Track 0: rotation, rebuilt (type 7 at comp 3) from x/y/z: x quantized, y static float, z type 10.
# Track 1: rotation, type 8 with four stored components. Track 2: translation, static vec3.
# Track 3: control, indirect (type 5). Track 4: control, raw f32 (type 3).
NF = 70
Z_INTS = [(f * f) // 50 % 97 + 3 for f in range(NF)]
X_INTS = [(5 * f) % 64 for f in range(NF)]
TRACKS = [(joaat32("skel_a"), BONEROTATION, 0), (joaat32("skel_b"), BONEROTATION, 0),
          (joaat32("skel_c"), BONETRANSLATION, 2), (0x11, GENERICCONTROL, 4), (0x12, GENERICCONTROL, 4)]
TABLE = [0, 5, 9, 15]


def sequence(lo, hi):
    n = hi - lo
    q = 0.002
    return encode_sequence(n, [
        (4, 0, 0, (6, q, -0.05, X_INTS[lo:hi])),
        (2, 0, 1, 0.25),
        (10, 0, 2, (q, -0.1, Z_INTS[lo:hi])),
        (7, 0, 3, None),
        (2, 1, 0, 0.5), (2, 1, 1, 0.5), (2, 1, 2, 0.5), (2, 1, 3, 0.5), (8, 1, 0, None),
        (1, 2, 0, (1.0, 2.0, 3.0)),
        (5, 3, 0, (2, 4, TABLE, 0.1, 1.0, [f % 4 for f in range(lo, hi)])),
        (3, 4, 0, [float(f) * 1.5 for f in range(lo, hi)]),
    ])


def two_sequence_dictionary(name="synthetic@dict", **kw):
    # two sequences share frame 40
    return dictionary(name, [sequence(0, 41), sequence(40, NF)], NF, TRACKS, **kw)


def write_rpf(path, streams, gap_before=()):
    """streams: list of inflated payloads; each is zlib-compressed and 16-byte aligned, or preceded by a 4 KiB
    gap when its index is in `gap_before`."""
    header = bytearray(0x100)
    header[:4] = b"9FPR"
    struct.pack_into("<I", header, 4, len(header))
    body, offsets = bytearray(), []
    for i, payload in enumerate(streams):
        if i in gap_before:
            body += b"\x55" * 4096
        offsets.append(len(header) + len(body))
        body += zlib.compress(payload)
        body += bytes((-len(body)) % 16)
    body += bytes(64)
    with open(path, "wb") as f:
        f.write(header + body)
    return offsets


def split(page, at, move_map=True):
    """Two-page system: set the page count to 2 and, with `move_map`, put the page map on the second page."""
    b = bytearray(page)
    if move_map:
        pm = len(b)
        b += bytes(16)
        struct.pack_into("<Q", b, 8, SYSTEM + pm)
    else:
        pm = struct.unpack_from("<Q", b, 8)[0] - SYSTEM
    b[pm + 8], b[pm + 9] = 2, 0
    return bytes(b[:at]), bytes(b[at:])


# ---------------------------------------------------------------- tests


class DecodeTest(unittest.TestCase):
    def setUp(self):
        self.s = anim_clips.System(two_sequence_dictionary())
        self.d = anim_clips.parse_dictionary(self.s)

    def test_motionmatcher(self):
        c = [c for c in self.d["clips"] if c["kind"] == "motionmatcher"][0]
        m = c["matcher"]
        self.assertEqual(m["records"], [(0.0, 30.0, -1.0, 0), (0.5, 15.0, -1.0, 1)])
        self.assertEqual(m["offsets"], [0, 4, 9])
        self.assertEqual(m["refs"], [(1, joaat32("synthetic@dict")), (0xC1105, 0xD1C7)])
        self.assertEqual(m["clips"], [(1, 0x77, 0), (0xC1105, 0x78, 1)])            # in slot order
        self.assertEqual(m["keys"], [(0x5A, 2), (1, 0), (0x5B, 1)])
        self.assertEqual(m["self"], (c["hash"], self.d["hash"]))
        self.assertEqual((m["u16_78"], m["u8_b8"]), ([0, 1, 2], [0, 1]))
        self.assertEqual(m["features"], [(0, 3, joaat32("spine")), (3, 6, joaat32("head"))])
        self.assertEqual(m["feature_kinds"], [(1, 1), (1, 3)])
        sk = m["skeleton"]
        self.assertEqual([b[:3] for b in sk["bones"]], [b[:3] for b in BONES])
        self.assertEqual((sk["parents"], len(sk["tags"]), len(sk["local"])), ([-1, 0, 1], 6, 3))
        self.assertEqual(anim_clips.skeleton_checks(sk), (True, True))
        self.assertEqual(sk["name_tree"], [("head", 2), ("root", 0), ("spine", 1)])
        self.assertEqual(anim_clips.bone_id("a"), 0x41 + 0x170)                      # upper-cased: "A"
        self.assertEqual(anim_clips.bone_id("ABCDEFGHI"), 0xDC6D)            # ELF 0x0789EEA9: the top nibble folds
        self.assertEqual(m["sample_u32"], [1] * 4 + [6] * 5)
        self.assertEqual(m["feature_names"], [(0x21, 1), (0xA2, 0)])
        self.assertEqual(m["tables_placed_by_key"], (True, False, True))
        self.assertTrue(m["feature_second"])
        self.assertEqual(m["feature_b"], {"form": 2, "bones": [joaat32("spine"), joaat32("head")],
                                          "weights": [0.25, 0.75], "feature": (1, joaat32("spine")), "kind": 1,
                                          "width": 3, "flag": 6})
        self.assertEqual((m["column_weights"], m["c0"]), (COLUMN_WEIGHTS, (25.0, 1.0, 1)))
        self.assertEqual((m["items_48"], m["u32_118"], m["u16_128"]), ([(0,) * 8 + (1, 0)], [9], [0]))
        self.assertEqual((m["poses"]["layout"], m["poses"]["samples"], m["poses"]["columns"]), ("rows", 9, 9))
        for r in range(9):
            self.assertEqual(anim_clips.matcher_pose(self.s, m["poses"], r), [pose_value(r, c) for c in range(9)])
        with self.assertRaises(IndexError):
            anim_clips.matcher_pose(self.s, m["poses"], 9)
        self.assertEqual(len(m["pose_transforms"]), 9)
        stats = anim_clips.new_stats()
        anim_clips.count_matcher(c, self.d["hash"], stats)
        for k in ("matcher_record_index_is_position", "matcher_clip_table_indices_0_to_n",
                  "matcher_record_third_is_minus_one",
                  "matcher_offsets_one_more_than_records", "matcher_offsets_from_0_nondecreasing",
                  "matcher_clip_table_one_per_record", "matcher_refs_one_per_clip",
                  "matcher_refs_name_the_table_clip", "matcher_key_table_indices_0_to_n", "matcher_self_is_own_key",
                  "matcher_u16_78_from_0_to_records", "matcher_u16_78_one_more_than_clips",
                  "matcher_record_clip_in_its_u16_78_range", "matcher_clip_table_placed_by_key",
                  "matcher_sample_u32_bits_are_key_indices", "matcher_column_weights_chunks_of_8",
                  "matcher_column_weights_positive_sum_1_then_0", "matcher_column_weights_same_within_each_vec3",
                  "matcher_object_b_form_2_copies_the_first_feature", "matcher_object_b_form_2_two_distinct_bones",
                  "matcher_object_b_form_2_flag_6", "matcher_u8_b8_one_per_clip", "matcher_u8_b8_0_or_1",
                  "matcher_u8_b8_set", "matcher_feature_columns_chain_from_0", "matcher_feature_widths_multiple_of_3",
                  "matcher_feature_names_distinct", "matcher_feature_name_table_indices_0_to_f",
                  "matcher_feature_object_second_pointer", "matcher_pose_layout_rows",
                  "matcher_pose_samples_are_offsets_end", "matcher_pose_values_finite",
                  "matcher_pose_padding_as_expected", "matcher_pose_transforms_one_per_sample",
                  "matcher_pose_transforms_unit_quaternion_scale_1", "matcher_skeleton_one_root_parents_first",
                  "matcher_skeleton_tags_are_name_hash_and_id", "matcher_skeleton_local_is_bone_transform",
                  "matcher_skeleton_name_tree_maps_each_name_to_its_bone",
                  "matcher_skeleton_bone_ids_from_names_root_0",
                  "matcher_skeleton_inverse_bind_undoes_chain", "matcher_sample_u32_one_per_sample",
                  "matcher_sample_u32_same_within_a_record", "matcher_feature_name_table_placed_by_key",
                  "matcher_item_48_one_zero_item_with_u32_1_at_0x20", "matcher_u32_118_is_the_sample_count",
                  "matcher_u16_128_is_one_0",
                  "matcher_c8_1_with_rows_3_with_blocks", "matcher_object_b_form_2",
                  "matcher_object_b_form_2_width_from_kind_bits", "matcher_object_b_form_2_kind_is_a_feature_kind",
                  "matcher_object_b_weights_positive_sum_1"):
            self.assertEqual(stats[k], 1, k)
        self.assertEqual((stats["matcher_records"], stats["matcher_clip_table_entries"], stats["matcher_refs"],
                          stats["matcher_key_table_entries"], stats["matcher_features"],
                          stats["matcher_feature_columns"], stats["matcher_skeleton_bones"],
                          stats["matcher_features_with_0e_1"], stats["matcher_features_0e_1_name_a_bone"],
                          stats["matcher_features_0e_1_width_from_kind_bits"]), (2, 2, 2, 3, 2, 9, 3, 2, 2, 2))
        sk["inverse_bind"][2][3] = (0.0, 0.0, 0.0, 1.0)                              # a bind that doesn't undo
        self.assertEqual(anim_clips.skeleton_checks(sk), (True, False))
        sk["local"][1][0] = (1.0, 0.0, 0.0, 0.0)                                     # not the bone's rotation
        self.assertEqual(anim_clips.skeleton_checks(sk)[0], False)
        sk["bones"][1] = ("spine", 2) + sk["bones"][1][2:]                           # a parent after its child
        self.assertEqual(anim_clips.skeleton_checks(sk), (False, False))
        sk["bones"] = []                                                             # an empty skeleton
        self.assertEqual(anim_clips.skeleton_checks(sk), (True, True))

    def test_motionmatcher_poses_in_blocks_of_8(self):
        s = anim_clips.System(bytearray(two_sequence_dictionary(pose_rows=False)))
        d = anim_clips.parse_dictionary(s)
        c = [c for c in d["clips"] if c["kind"] == "motionmatcher"][0]
        m = c["matcher"]
        self.assertEqual((m["poses"]["layout"], m["poses"]["samples"], m["poses"]["stride"]), ("blocks of 8", 9, 10))
        for r in range(9):
            self.assertEqual(anim_clips.matcher_pose(s, m["poses"], r), [pose_value(r, c) for c in range(9)])
        self.assertEqual((m["feature_second"], m["pose_transforms"], m["feature_b"]), (False, [], None))
        stats = anim_clips.new_stats()
        anim_clips.count_matcher(c, d["hash"], stats)
        for k in ("matcher_pose_layout_blocks_of_8", "matcher_pose_samples_are_offsets_end",
                  "matcher_c8_1_with_rows_3_with_blocks",
                  "matcher_pose_values_finite", "matcher_pose_padding_as_expected"):
            self.assertEqual(stats[k], 1, k)
        self.assertEqual(stats.get("matcher_pose_transforms_one_per_sample", 0), 0)
        at = [v for k, v in s.atmap(0x48) if k & 0xFFFFFFFF == 3][0]
        obj = s.ptr(at + 0x20)
        last = s.ptr(obj + 0xF0) + 2 * (1 * 10 * 8 + 0 * 8 + 7)                    # column 0, last lane
        struct.pack_into("<e", s.d, last, 99.0)                                     # padding that doesn't copy
        self.assertFalse(anim_clips.parse_matcher(s, at)["poses"]["padding"])
        struct.pack_into("<e", s.d, last, pose_value(8, 0))                         # restored
        spare = s.ptr(obj + 0xF0) + 2 * (9 * 8 + 3)                                 # the spare 10th column
        struct.pack_into("<e", s.d, spare, 1.0)
        self.assertFalse(anim_clips.parse_matcher(s, at)["poses"]["padding"])
        struct.pack_into("<e", s.d, spare, 0.0)
        self.assertTrue(anim_clips.parse_matcher(s, at)["poses"]["padding"])
        for field, value in ((obj + 0x128, 17),                                     # 17 samples need 3 blocks
                             (obj + 0x128, 0),                                      # no samples
                             (obj + 0xFC, 8),                                       # not the features' columns
                             (obj + 0xF0, 0x50000000 + len(s.d) - 64)):            # past the pages
            fmt = "<Q" if field == obj + 0xF0 else "<I"
            kept = struct.unpack_from(fmt, s.d, field)[0]
            struct.pack_into(fmt, s.d, field, value)
            with self.assertRaises(anim_clips.DecodeError, msg=hex(field - obj)):
                anim_clips.parse_matcher(s, at)
            struct.pack_into(fmt, s.d, field, kept)
        anim_clips.parse_matcher(s, at)

    def test_motionmatcher_pose_rows_that_do_not_fit(self):
        s = anim_clips.System(bytearray(two_sequence_dictionary()))
        at = [v for k, v in s.atmap(0x48) if k & 0xFFFFFFFF == 3][0]
        obj = s.ptr(at + 0x20)
        struct.pack_into("<e", s.d, s.ptr(obj + 0xF0) + 2 * 12, 1.0)                # a padding column not 0
        self.assertFalse(anim_clips.parse_matcher(s, at)["poses"]["padding"])
        struct.pack_into("<e", s.d, s.ptr(obj + 0xF0) + 2 * 12, 0.0)
        for writes in ([(obj + 0xFC, "<I", 8)],                                     # not the features' columns
                       [(obj + 0xF8, "<I", 10)],                                    # chunks don't fit 10 samples
                       [(obj + 0xF8, "<I", 0), (obj + 0xEC, "<I", 0)],             # no samples
                       [(obj + 0x100, "<I", 2), (obj + 0xEC, "<I", 18)],           # wider rows than needed
                       [(obj + 0xF0, "<Q", 0)],                                     # no data
                       [(obj + 0xF0, "<Q", 0x50000000 + len(s.d) - 64)],           # matrix past the pages
                       [(obj + 0xA4, "<I", len(s.d))]):                             # transforms past the pages
            kept = [(f, fmt, struct.unpack_from(fmt, s.d, f)[0]) for f, fmt, _v in writes]
            for f, fmt, v in writes:
                struct.pack_into(fmt, s.d, f, v)
            with self.assertRaises(anim_clips.DecodeError, msg=repr(writes)):
                anim_clips.parse_matcher(s, at)
            for f, fmt, v in kept:
                struct.pack_into(fmt, s.d, f, v)
        anim_clips.parse_matcher(s, at)

    def test_motionmatcher_object_b_form_1_and_arrays_that_do_not_fit(self):
        s = anim_clips.System(bytearray(two_sequence_dictionary()))
        at = [v for k, v in s.atmap(0x48) if k & 0xFFFFFFFF == 3][0]
        obj = s.ptr(at + 0x20)
        b = s.ptr(obj + 0x30)
        values = s.ptr(at + 0x48)                     # 40 spare bytes: 2 bone hashes, then 2 weights
        struct.pack_into("<2I2f", s.d, values, joaat32("root"), joaat32("head"), 0.5, 0.5)
        struct.pack_into("<I", s.d, b + 8, 1)
        struct.pack_into("<4f", s.d, b + 0x10, 0.0, 0.0, 0.6, 0.8)                   # the reference rotation
        struct.pack_into("<I", s.d, b + 0x60, 7)
        for field in (b + 0x40, b + 0x50):
            struct.pack_into("<II", s.d, field, 2, 2)
        struct.pack_into("<Q", s.d, b + 0x48, 0x50000000 + values)
        struct.pack_into("<Q", s.d, b + 0x58, 0x50000000 + values + 8)
        self.assertEqual(anim_clips.matcher_object_b(s, b),
                         {"form": 1, "rotation": (0.0, 0.0, struct.unpack("<f", struct.pack("<f", 0.6))[0],
                                                  struct.unpack("<f", struct.pack("<f", 0.8))[0]), "flag": 7,
                          "bones": [joaat32("root"), joaat32("head")], "weights": [0.5, 0.5]})
        stats = anim_clips.new_stats()
        anim_clips.count_matcher({"matcher": anim_clips.parse_matcher(s, at), "hash": 0}, 0, stats)
        for k in ("matcher_object_b_form_1", "matcher_object_b_weights_positive_sum_1",
                  "matcher_object_b_form_1_bones_one_per_weight", "matcher_object_b_form_1_bones_name_skeleton_bones",
                  "matcher_object_b_form_1_rotation_unit", "matcher_object_b_form_1_flag_7"):
            self.assertEqual(stats[k], 1, k)
        struct.pack_into("<I", s.d, values + 4, 0x1234)                              # not a skeleton bone
        stats = anim_clips.new_stats()
        anim_clips.count_matcher({"matcher": anim_clips.parse_matcher(s, at), "hash": 0}, 0, stats)
        self.assertEqual(stats["matcher_object_b_form_1_bones_name_skeleton_bones"], 0)
        struct.pack_into("<I", s.d, b + 8, 7)                                         # another form: not read
        self.assertEqual(anim_clips.matcher_object_b(s, b), {"form": 7})
        struct.pack_into("<I", s.d, b + 8, 1)
        for field, fmt, value in ((obj + 0x30, "<Q", 0x50000000 + len(s.d) - 0x40),  # the object past the pages
                                  (b + 0x50, "<I", len(s.d)),                       # weights past the pages
                                  (b + 0x58, "<Q", 0),                              # weights without data
                                  (at + 0x114, "<I", len(s.d)),                     # +0x118 past the pages
                                  (obj + 0xB4, "<I", len(s.d)),                     # +0xB8 past the pages
                                  (obj + 0xB4, "<I", (len(s.d) - s.ptr(obj + 0xB8)) // 32 + 1)):  # one chunk past
            kept = struct.unpack_from(fmt, s.d, field)[0]
            struct.pack_into(fmt, s.d, field, value)
            with self.assertRaises(anim_clips.DecodeError, msg=hex(field)):
                anim_clips.parse_matcher(s, at)
            struct.pack_into(fmt, s.d, field, kept)
        anim_clips.parse_matcher(s, at)

    def test_motionmatcher_skeleton_that_does_not_fit(self):
        s = anim_clips.System(bytearray(two_sequence_dictionary()))
        at = [v for k, v in s.atmap(0x48) if k & 0xFFFFFFFF == 3][0]
        obj = s.ptr(at + 0x20)
        sk = s.ptr(obj + 0x28)
        name = s.ptr(s.ptr(sk + 0x28) + 0x30)
        tree = s.ptr(sk + 0x48)
        for writes in ([(sk + 0x1C, "<I", 5)],                                      # 6 full slots for 5 entries
                       [(sk + 0x20, "<Q", 0)],                                      # no tag table
                       [(sk + 0x18, "<I", len(s.d))],                               # tag table past the pages
                       [(sk + 0x38, "<Q", 0)],                                      # no local matrices
                       [(sk + 0x6C, "<H", 0x7FFF)],                                 # bones past the pages
                       [(name + 8 * i, "<Q", 0x6161616161616161) for i in range(40)],  # a name without its end
                       [(tree + 4, "<H", 200)],                                     # a sibling outside the tree
                       [(tree + 2, "<H", 10)],                                      # a node text without its end
                       [(sk + 0x48, "<Q", 0x50000000 + len(s.d) - 2)],              # tree header past the pages
                       [(obj + 0x94, "<I", len(s.d))]):                             # per-sample words past the pages
            kept = [(f, fmt, struct.unpack_from(fmt, s.d, f)[0]) for f, fmt, _v in writes]
            for f, fmt, v in writes:
                struct.pack_into(fmt, s.d, f, v)
            with self.assertRaises(anim_clips.DecodeError, msg=repr(writes[0])):
                anim_clips.parse_matcher(s, at)
            for f, fmt, v in kept:
                struct.pack_into(fmt, s.d, f, v)
        anim_clips.parse_matcher(s, at)

    def test_table_placement_with_linear_probing(self):
        """An entry whose home group is full sits in the next one; with room at home, the same layout is not
        placed by the key. The clip table is placed by `clip_slot_hash`, not by its first u32."""
        s = anim_clips.System(bytearray(two_sequence_dictionary()))
        at = [v for k, v in s.atmap(0x48) if k & 0xFFFFFFFF == 3][0]
        size = 16 + 16 * 8

        def build(home_full):
            block = bytearray(bytes([0x80]) * 16 + bytes(16 * 8)) * 2
            keys = list(range(16 if home_full else 15)) + [16]                   # all home 0; 16 lands in group 1
            for n, h in enumerate(keys):
                g, i = (0, n) if n < len(keys) - 1 else (1, 0)
                block[g * size + i] = h & 0x7F
                struct.pack_into("<II", block, g * size + 16 + 8 * i, h, n)
            s.d += bytes((-len(s.d)) % 16)
            struct.pack_into("<IIQ", s.d, at + 0x170, 2, len(keys), 0x50000000 + len(s.d))
            s.d += block
        build(True)
        self.assertTrue(anim_clips.table_placed_by_key(s, at + 0x170, 8))
        build(False)
        self.assertFalse(anim_clips.table_placed_by_key(s, at + 0x170, 8))
        # wrapping: 16 entries home in the last group (1), and one more home there that lands in group 0
        block = bytearray(bytes([0x80]) * 16 + bytes(16 * 8)) * 2
        for n, h in enumerate([0x80 + i for i in range(17)]):
            g, i = (1, n) if n < 16 else (0, 0)
            block[g * size + i] = h & 0x7F
            struct.pack_into("<II", block, g * size + 16 + 8 * i, h, n)
        s.d += bytes((-len(s.d)) % 16)
        struct.pack_into("<IIQ", s.d, at + 0x170, 2, 17, 0x50000000 + len(s.d))
        s.d += block
        self.assertTrue(anim_clips.table_placed_by_key(s, at + 0x170, 8))
        self.assertTrue(anim_clips.table_placed_by_key(s, at + 0xA0, 16, anim_clips.clip_slot_hash))
        self.assertFalse(anim_clips.table_placed_by_key(s, at + 0xA0, 16))
        self.assertEqual(anim_clips.fmix64(0), 0)
        self.assertEqual(anim_clips.fmix64(1), 0xB456BCFC34C2CB2C)                  # MurmurHash3 fmix64(1)

    def test_motionmatcher_without_refs_and_with_a_bad_table(self):
        d = anim_clips.parse_dictionary(anim_clips.System(two_sequence_dictionary(matcher_refs=False)))
        c = [c for c in d["clips"] if c["kind"] == "motionmatcher"][0]
        self.assertEqual(c["matcher"]["refs"], [])
        self.assertEqual(len(c["matcher"]["clips"]), 2)
        s = anim_clips.System(bytearray(two_sequence_dictionary()))
        at = [v for k, v in s.atmap(0x48) if k & 0xFFFFFFFF == 3][0]                # the matcher's key
        s.d[s.ptr(at + 0xA8) + 5] = 0x81                                            # neither empty nor full
        with self.assertRaises(anim_clips.DecodeError):
            anim_clips.parse_matcher(s, at)
        s.d[s.ptr(at + 0xA8) + 5] = 0x22                                            # a third full slot for 2 entries
        with self.assertRaises(anim_clips.DecodeError):
            anim_clips.parse_matcher(s, at)
        s.d[s.ptr(at + 0xA8) + 5] = 0x80
        anim_clips.parse_matcher(s, at)                                             # fixed again
        first = s.ptr(s.ptr(at + 0x20) + 0x40)
        kept = s.u64(first)
        struct.pack_into("<Q", s.d, first, 0)                                       # a null feature pointer
        with self.assertRaises(anim_clips.DecodeError):
            anim_clips.parse_matcher(s, at)
        struct.pack_into("<Q", s.d, first, kept)
        for field, value in ((at + 0x20, 0),                                        # no feature object
                             (s.ptr(at + 0x20) + 0x28, 0),                         # no skeleton
                             (first, 0x50000000 + len(s.d) - 8),                   # a feature past the pages
                             (s.ptr(at + 0x20) + 0x48, 3)):                        # feature table: 3 groups
            kept = s.u64(field)
            struct.pack_into("<Q", s.d, field, value)
            with self.assertRaises((anim_clips.DecodeError, struct.error), msg=hex(field)):
                anim_clips.parse_matcher(s, at)
            struct.pack_into("<Q", s.d, field, kept)
        for count_at in (at + 0x74, at + 0xB4):                                     # counts past the pages
            kept = s.u32(count_at)
            struct.pack_into("<I", s.d, count_at, len(s.d))
            with self.assertRaises(struct.error, msg=hex(count_at)):
                anim_clips.parse_matcher(s, at)
            struct.pack_into("<I", s.d, count_at, kept)
        anim_clips.parse_matcher(s, at)                                             # all restored
        struct.pack_into("<I", s.d, at + 0xA0, 3)                                   # 3 groups: not a power of two
        with self.assertRaises(anim_clips.DecodeError):
            anim_clips.parse_matcher(s, at)
        # through parse_dictionary a broken matcher keeps its name, and the stats count it as unreadable
        d = anim_clips.parse_dictionary(s)
        c = [c for c in d["clips"] if c["kind"] == "motionmatcher"][0]
        self.assertEqual((c["name"], c["matcher"]), ("pack:/one.motionmatcher", None))
        stats = anim_clips.new_stats()
        anim_clips.count_matcher(c, d["hash"], stats)
        self.assertEqual(stats["matcher_unreadable"], 1)

    def test_structure(self):
        d = self.d
        self.assertEqual(d["name"], "synthetic@dict")
        self.assertEqual(d["hash"], joaat32("synthetic@dict"))
        self.assertEqual(sorted((c["kind"], c["name"]) for c in d["clips"]),
                         [("clip", "pack:/one.clip"), ("motion", "pack:/one.motion"),
                          ("motionmatcher", "pack:/one.motionmatcher")])
        clip = [c for c in d["clips"] if c["kind"] == "clip"][0]
        self.assertEqual(clip["name"], "pack:/one.clip")
        self.assertEqual(len(clip["animations"]), 1)
        self.assertEqual(clip["animations"][0]["hash"], joaat32("one"))
        self.assertEqual(len(d["packs"]), 1)
        anim = d["packs"][0]["animations"][0]
        self.assertEqual(anim["frames"], NF)
        self.assertEqual(len(anim["sequences"]), 2)
        self.assertEqual(anim["tracks"], TRACKS)

    def test_motion_times_and_axis(self):
        f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
        times = [(1.5, 1.5, [(0.0, 0.0), (1.5, 1.5)]), (2.0, 1.5, [(0.0, 0.0), (1.0, 0.5), (2.0, 1.5)])]
        samples = [(1, -1.5), (0, 1.5)]
        for parameters in (1, 2):
            d = anim_clips.parse_dictionary(anim_clips.System(two_sequence_dictionary(
                children=[(1, 2), (3, 4)], parameters=parameters, times=times,
                axis=([joaat32("speed")], samples, -1.5, 1.5))))
            motion = [c for c in d["clips"] if c["kind"] == "motion"][0]
            self.assertEqual(motion["parameters"], parameters)
            self.assertEqual(motion["times"], [(f32(a), f32(b), [(f32(x), f32(y)) for x, y in pts])
                                               for a, b, pts in times])
            kind, axis = motion["node"]
            self.assertEqual((kind, axis["names"], axis["samples"], axis["min"], axis["max"]),
                             (3, [joaat32("speed")], samples, -1.5, 1.5))
            stats = anim_clips.new_stats()
            anim_clips.count_motion_blend(motion, stats)
            self.assertEqual((stats["motion_child_times_identity"], stats["motion_child_times_warped"],
                              stats["motion_axes"], stats["motion_axis_sorted"],
                              stats["motion_axis_range_is_first_last"], stats["motion_axis_name:speed"]),
                             (1, 1, 1, 1, 1, 1))

    def test_motion_without_blend_data(self):
        motion = [c for c in self.d["clips"] if c["kind"] == "motion"][0]
        self.assertEqual((motion["times"], motion["node"]), ([], None))      # null +0x70, null +0x138
        self.assertEqual(motion["samples"], [])                              # null +0x90

    def motion(self, **kw):
        d = anim_clips.parse_dictionary(anim_clips.System(two_sequence_dictionary(**kw)))
        return [c for c in d["clips"] if c["kind"] == "motion"][0]

    def test_type3_samples_map_to_children(self):
        # three samples over two children: sample 0 and 2 play child 1, sample 1 plays child 0
        lists = [[(1, 1.0, 1.0)], [(0, 1.0, 0.5)], [(1, 1.0, 1.0)]]
        motion = self.motion(children=[(1, 2), (3, 4)], axis=([joaat32("speed")], [(0, 0.0), (2, 0.5), (1, 1.0)],
                                                              0.0, 1.0), sample_lists=lists)
        self.assertEqual(motion["samples"], lists)
        kind, axis = motion["node"]
        self.assertEqual((kind, axis["slots"], axis["sample_count"]), (3, [(0, 0)], 3))
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        pre = "motion_samples:3:"
        self.assertEqual({k[len(pre):]: v for k, v in stats.items() if k.startswith(pre)},
                         {"motions": 1, "lists_equal_node_samples": 1, "lists_more_than_children": 1,
                          "every_list_one_item": 1, "children_in_range": 1, "every_child_sampled": 1,
                          "sample_k_is_child_k": 0, "items": 3, "items_first_float_1": 3,
                          "items_second_float_1": 2})

    def test_identity_samples(self):
        motion = self.motion(children=[(1, 2), (3, 4)], axis=([joaat32("speed")], [(0, 0.0), (1, 1.0)], 0.0, 1.0),
                             sample_lists=[[(0, 1.0, 1.0)], [(1, 1.0, 1.0)]])
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        self.assertEqual(stats["motion_samples:3:sample_k_is_child_k"], 1)

    def test_several_items_in_a_list_are_not_identity(self):
        # list 0 holds children 0 and 1: its first child is 0, but that is not sample k -> child k
        motion = self.motion(children=[(1, 2), (3, 4)], axis=([joaat32("speed")], [(0, 0.0), (1, 1.0)], 0.0, 1.0),
                             sample_lists=[[(0, 1.0, 1.0), (1, 1.0, 1.0)], [(1, 1.0, 1.0)]])
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        self.assertEqual((stats["motion_samples:3:sample_k_is_child_k"], stats["motion_samples:3:every_list_one_item"],
                          stats["motion_samples:3:every_child_sampled"]), (0, 0, 1))

    def test_malformed_node_keeps_the_time_table(self):
        times = [(1.5, 1.5, [(0.0, 0.0), (1.5, 1.5)])]
        motion = self.motion(children=[(1, 2)], times=times, node=(8, [(1, 0, 0)], 1), node_axis_count=0xFFFF)
        self.assertIsNone(motion["node"])                       # the axis array runs past the page
        self.assertEqual(len(motion["times"]), 1)

    def test_child_out_of_range_is_counted(self):
        motion = self.motion(children=[(1, 2)], axis=([joaat32("speed")], [(0, 0.0)], 0.0, 0.0),
                             sample_lists=[[(5, 1.0, 1.0)]])
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        self.assertEqual((stats["motion_samples:3:children_in_range"], stats["motion_samples:3:every_child_sampled"]),
                         (0, 0))

    def test_multi_axis_nodes(self):
        for kind, n_axes in ((4, 2), (5, 3), (6, 4), (8, 5)):
            axes = [(joaat32("speed") if i == 0 else 0xABC0 + i, i, 1) for i in range(n_axes)]
            lists = [[(i % 2, 1.0, 1.0)] for i in range(4)]
            motion = self.motion(children=[(1, 2), (3, 4)], node=(kind, axes, 4), sample_lists=lists)
            k, axis = motion["node"]
            self.assertEqual((k, axis["names"], axis["slots"], axis["sample_count"]),
                             (kind, [a[0] for a in axes], [(i, 1) for i in range(n_axes)], 4))
            self.assertNotIn("samples", axis)
            stats = anim_clips.new_stats()
            anim_clips.count_motion_blend(motion, stats)
            self.assertEqual((stats[f"motion_node_axes:{kind}:{n_axes}"],
                              stats[f"motion_node_axis_name:{kind}:speed"],
                              stats[f"motion_node_axis_name:{kind}:unresolved"],
                              stats[f"motion_samples:{kind}:lists_equal_node_samples"],
                              stats[f"motion_samples:{kind}:every_child_sampled"], stats["motion_axes"]),
                             (1, 1, n_axes - 1, 1, 1, 0))

    def test_multi_axis_pairs(self):
        motion = self.motion(children=[(1, 2)], node=(6, [(1, 0, 0)] * 4, 1), node_pairs=[(0xAA, 0xBB), (0xCC, 0xDD)])
        self.assertEqual(motion["node"][1]["pairs"], [(0xAA, 0xBB), (0xCC, 0xDD)])
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        self.assertEqual(stats["motion_node_pairs:6:2"], 1)
        motion = self.motion(children=[(1, 2)], node=(4, [(1, 0, 0)] * 2, 1))      # null +0x20
        self.assertIsNone(motion["node"][1]["pairs"])
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        self.assertEqual(stats["motion_node_pairs:4:null"], 1)

    def test_type1_node(self):
        motion = self.motion(children=[(1, 2)], node=(1, [], 1), sample_lists=[[(0, 1.0, 1.0)]])
        self.assertEqual(motion["node"], (1, {"names": [], "slots": [], "sample_count": 1}))
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        self.assertEqual((stats["motion_node_type:1"], stats["motion_node_axes:1:0"],
                          stats["motion_samples:1:sample_k_is_child_k"]), (1, 1, 1))

    def test_sample_lists_of_a_two_node_motion_are_not_counted(self):
        motion = self.motion(children=[(1, 2)], parameters=2, node=(1, [], 1), sample_lists=[[(0, 1.0, 1.0)]] * 3)
        self.assertEqual(len(motion["samples"]), 3)
        stats = anim_clips.new_stats()
        anim_clips.count_motion_blend(motion, stats)
        self.assertFalse([k for k in stats if k.startswith("motion_samples:")])

    def test_delta2_round_trip(self):
        for n in (1, 2, 8, 9, 63, 64, 65, 129, 200):
            ints = [(7 * f * f) % 1000 + 20 * (f % 3) for f in range(n)]
            blob = encode_delta2(ints, 0.5, -3.0) + bytes(8)
            stats = anim_clips.new_stats()
            values, size = anim_clips.decode_delta2(blob, 0, n, stats)
            self.assertEqual(size, len(blob) - 8)
            self.assertEqual(values, [x * 0.5 - 3.0 for x in ints], n)
            self.assertEqual(stats["delta2_boundaries"], (n - 1) // 64)
            self.assertEqual(stats["delta2_boundaries_closed"], stats["delta2_boundaries"])

    def test_tracks(self):
        anim = self.d["packs"][0]["animations"][0]
        stats = anim_clips.new_stats()
        tracks, total = anim_clips.decode_animation(self.s, anim, stats)
        self.assertEqual(total, NF)
        self.assertEqual(stats["frames_sum_matches"], 1)
        for f in (0, 39, 40, 41, NF - 1):
            x, y, z = X_INTS[f] * 0.002 - 0.05, 0.25, Z_INTS[f] * 0.002 - 0.1
            w = math.sqrt(1 - x * x - y * y - z * z)
            for got, want in zip(tracks[0][f], (x, y, z, w)):
                self.assertAlmostEqual(got, want, places=5)
            self.assertEqual(len(tracks[1][f]), 4)
            for got in tracks[1][f]:
                self.assertAlmostEqual(got, 0.5, places=6)
            self.assertEqual(tracks[2][f], (1.0, 2.0, 3.0))
            self.assertAlmostEqual(tracks[3][f][0], TABLE[f % 4] * 0.1 + 1.0, places=5)
            self.assertEqual(tracks[4][f][0], f * 1.5)
        self.assertEqual(stats["boundary_quaternions"], stats["boundary_quaternions_same_rotation"])
        self.assertEqual(stats["boundary_values"], stats["boundary_values_within_1e-3"])
        self.assertEqual(stats["channel:10_delta2_float"], 2)
        self.assertEqual(stats["channel:03_raw_float"], 2)

    def test_rebuilt_component_goes_to_the_type7_position(self):
        seq = encode_sequence(1, [(2, 0, 0, 0.1), (2, 0, 1, 0.2), (2, 0, 2, 0.3), (7, 0, 1, None)])
        s = anim_clips.System(dictionary("p", [seq], 1, TRACKS[:1]))
        anim = anim_clips.parse_dictionary(s)["packs"][0]["animations"][0]
        tracks, _ = anim_clips.decode_animation(s, anim)
        w = math.sqrt(1 - 0.01 - 0.04 - 0.09)
        for got, want in zip(tracks[0][0], (0.1, w, 0.2, 0.3)):
            self.assertAlmostEqual(got, want, places=6)

    def test_definitions_must_end_at_the_frame_data(self):
        seq = bytearray(encode_sequence(3, [(2, 0, 0, 1.0), (2, 0, 1, 2.0)]))
        struct.pack_into("<H", seq, 0x20 + 8 + 2 * 2, 1)   # count says one static float, 8 definition bytes
        s = anim_clips.System(dictionary("p", [bytes(seq)], 3, TRACKS[3:4]))
        anim = anim_clips.parse_dictionary(s)["packs"][0]["animations"][0]
        with self.assertRaisesRegex(anim_clips.DecodeError, "channel definitions end"):
            anim_clips.decode_animation(s, anim)

    def test_pointer_outside_the_pages(self):
        b = bytearray(two_sequence_dictionary())
        struct.pack_into("<Q", b, 0x48, SYSTEM + len(b) + 0x100)      # clip map buckets
        with self.assertRaisesRegex(anim_clips.DecodeError, "outside the pages"):
            anim_clips.parse_dictionary(anim_clips.System(bytes(b)))

    def test_inline_name(self):
        b = bytearray(two_sequence_dictionary("short@name"))
        b[0x38:0x48] = b"short@name".ljust(15, b"\x00") + bytes([0x80 | 10])
        d = anim_clips.parse_dictionary(anim_clips.System(bytes(b)))
        self.assertEqual(d["name"], "short@name")
        self.assertEqual(d["hash"], joaat32("short@name"))

    def test_motion_with_an_inline_name(self):
        b = bytearray(two_sequence_dictionary())
        motion = {k & 0xFFFFFFFF: v for k, v in self.s.atmap(0x48)}[2]
        b[motion + 0x30:motion + 0x40] = b"walk.motion".ljust(15, b"\x00") + bytes([0x80 | 11])
        d = anim_clips.parse_dictionary(anim_clips.System(bytes(b)))
        self.assertEqual(sorted((c["kind"], c["name"]) for c in d["clips"]),
                         [("clip", "pack:/one.clip"), ("motion", "walk.motion"),
                          ("motionmatcher", "pack:/one.motionmatcher")])

    def test_rig_root_is_not_a_clip_dictionary(self):
        b = bytearray(two_sequence_dictionary())
        b[0x70:0x78] = b"RIG\x02\x10\x00\x00\x00"
        with self.assertRaisesRegex(anim_clips.NotClipDictionary, "rig"):
            anim_clips.parse_dictionary(anim_clips.System(bytes(b)))


def write_raw(path, parts):
    """parts: [(bytes, compress?)] written 16-byte aligned after a minimal RPF9 header."""
    body = bytearray(0x100)
    body[:4] = b"9FPR"
    struct.pack_into("<I", body, 4, 0x100)
    offsets = []
    for data, compress in parts:
        offsets.append(len(body))
        body += zlib.compress(data) if compress else data
        body += bytes((-len(body)) % 16)
    body += bytes(64)
    with open(path, "wb") as f:
        f.write(body)
    return offsets


class GapPageTest(unittest.TestCase):
    """A dictionary's last system page that anim_index does not list: a zlib stream inflating to fewer than 64
    bytes, or a 16-byte raw page. Both are joined only when the pages then parse."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rpf = os.path.join(self.tmp.name, "gap.rpf")

    def tearDown(self):
        self.tmp.cleanup()

    def scan(self, parts):
        write_raw(self.rpf, parts)
        return anim_clips.scan(self.rpf)

    def test_small_zlib_last_page(self):
        page = two_sequence_dictionary("small@dict")
        a, b = split(page, len(page) - 32, move_map=False)
        rows, summary = self.scan([(a, True), (b, True), (two_sequence_dictionary("next@dict"), True)])
        self.assertEqual(summary["streams"], 2)                 # anim_index skips the 32-byte page
        self.assertEqual(summary["gap_pages"], {"small zlib page": 1})
        self.assertEqual(summary["unresolved"], {})
        self.assertEqual(summary["counts"]["dictionaries"], 2)
        self.assertEqual(summary["counts"]["animations_decoded"], 2)
        self.assertEqual(summary["pages"], {"1": 1, "2": 1})

    def test_raw_16_byte_last_page(self):
        page = two_sequence_dictionary("raw@dict")
        a, b = split(page, len(page) - 16, move_map=False)
        rows, summary = self.scan([(a, True), (b, False), (two_sequence_dictionary("next@dict"), True)])
        self.assertEqual(summary["gap_pages"], {"raw 16-byte page": 1})
        self.assertEqual(summary["counts"]["dictionaries"], 2)
        self.assertEqual(summary["counts"]["animations_decoded"], 2)

    def test_page_map_in_an_unlisted_last_page_stays_unresolved(self):
        # The tail is tried only after the page map has been read from listed streams (see docs/animation.md).
        for raw, size in ((False, 16), (True, 32)):
            page = two_sequence_dictionary("map@dict")
            a, b = split(page, len(page) + 16 - size)          # move_map: the page map is the last 16 bytes
            rows, summary = self.scan([(a, True), (b, raw), (two_sequence_dictionary("next@dict"), True)])
            self.assertEqual(summary["unresolved"], {"page map not in the next stream": 1})
            self.assertEqual(summary["counts"]["dictionaries"], 1)

    def test_gap_that_does_not_parse_stays_unresolved(self):
        page = two_sequence_dictionary("cut@dict")
        a, _b = split(page, 0x400, move_map=False)              # the rest of the page is missing
        rows, summary = self.scan([(a, True), (b"\x55" * 64, False), (two_sequence_dictionary("next@dict"), True)])
        self.assertEqual(summary["gap_pages"], {})
        self.assertEqual(summary["unresolved"], {"system page not the next stream": 1})
        self.assertEqual(summary["counts"]["dictionaries"], 1)


class ArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.rpf = os.path.join(self.tmp.name, "synthetic.rpf")
        second, nowhere = joaat32("second@dict"), joaat32("nowhere@dict")
        clip_duration = (NF - 1) / 30.0                    # the `.clip` (hash 1) in every synthetic dictionary
        one = two_sequence_dictionary("first@dict", children=[(1, second), (7, second), (1, nowhere)],
                                      external_hashes=[second, nowhere],
                                      times=[(clip_duration, clip_duration, [(0.0, 0.0), (clip_duration,) * 2]),
                                             (2.0, 2.0, [(0.0, 0.0), (2.0, 2.0)]),
                                             (2.0, 3.0, [(0.0, 0.0), (1.0, 2.0), (2.0, 3.0)])],
                                      axis=([joaat32("yawangle")], [(2, -1.0), (0, 0.0), (1, 2.0)], -1.0, 2.0))
        a, b = split(two_sequence_dictionary("second@dict"), 0x400)
        c, d = split(two_sequence_dictionary("third@dict"), 0x400, move_map=False)
        other = bytearray(0x60)
        struct.pack_into("<Q", other, 8, 0x50000020)
        other[16:20] = b"0DRP"
        struct.pack_into("<I", other, 0x14, 122)
        # stream 5 (third dictionary's second page) sits after a gap, so it is not the next stream
        self.offsets = write_rpf(self.rpf, [one, a, b, bytes(other), c, d], gap_before={5})

    def tearDown(self):
        self.tmp.cleanup()

    def test_scan_counts(self):
        rows, summary = anim_clips.scan(self.rpf)
        self.assertEqual(summary["streams"], 6)
        self.assertEqual(summary["dictionary_roots"], 3)
        self.assertEqual(summary["dictionaries_grouped"], 2)
        self.assertEqual(summary["unresolved"], {"system page not the next stream": 1})
        self.assertEqual(summary["pages"], {"1": 1, "2": 1})
        c = summary["counts"]
        self.assertEqual((c["dictionaries"], c["clips"], c["clips_motion"], c["clips_motionmatcher"],
                          c["animations"], c["animations_decoded"], c["sequences"]), (2, 6, 2, 2, 2, 2, 4))
        self.assertEqual(c["dictionary_hash_matches_name"], 2)
        self.assertEqual(c["frames_sum_matches"], 2)
        self.assertEqual(c["boundary_quaternions"], c["boundary_quaternions_same_rotation"])
        self.assertEqual(c["track_type:bonerotation"], 4)
        # `.motion` children are {clip hash, dictionary hash}; root refs without a pointer name dictionaries
        self.assertEqual((c["motion_children"], c["motion_children_in_external_refs"],
                          c["motion_children_dictionary_here"], c["motion_children_resolved_to_clip"]), (3, 3, 2, 1))
        # `.motionmatcher`: each has one clip in this archive (its dictionary's own) and one that is not
        self.assertEqual((c["matcher_records"], c["matcher_refs"], c["matcher_table_clips_found_here"],
                          c["matcher_refs_resolved_to_clip"], c["matcher_self_is_own_key"],
                          c["matcher_key_table_hashes_that_are_clips"], c.get("matcher_unreadable", 0)),
                         (4, 4, 2, 2, 2, 2, 0))
        self.assertEqual((c["external_refs_to_a_dictionary_here"], c["external_hashes_distinct"],
                          c["external_hashes_distinct_found_here"]), (1, 2, 1))
        self.assertEqual(c["motion_parameters:1"], 2)
        # +0x70 child times: one per child; the resolved child's duration equals its `.clip`'s
        self.assertEqual((c["motion_times_one_per_child"], c["motion_child_times"], c["motion_child_times_identity"],
                          c["motion_child_times_warped"], c["motion_child_duration_equals_clip"]), (2, 3, 2, 1, 1))
        self.assertEqual((c["motion_node_type:3"], c["motion_axes"], c["motion_axis_samples"],
                          c["motion_axis_indices_are_0_to_n"], c["motion_axis_name:yawangle"]), (1, 1, 3, 1, 1))
        self.assertEqual(c["motion_node_unreadable"], 1)                    # the second dictionary has no node
        self.assertNotIn("_children", rows[0])
        self.assertNotIn("_child_durations", rows[0])
        self.assertEqual(summary["gap_pages"], {})
        self.assertEqual([r["offset"] for r in rows], [self.offsets[0], self.offsets[1]])
        self.assertEqual([r["streams"] for r in rows], [1, 2])

    def test_jobs_give_the_same_counts(self):
        rows1, one = anim_clips.scan(self.rpf, jobs=1)
        rows2, two = anim_clips.scan(self.rpf, jobs=2)
        for summary in (one, two):
            summary.pop("slowest_dictionary_seconds")
        self.assertEqual(one, two)
        self.assertEqual([r["offset"] for r in rows1], [r["offset"] for r in rows2])

    def test_budget_skips_instead_of_blocking(self):
        _rows, summary = anim_clips.scan(self.rpf, budget=0)
        c = summary["counts"]
        self.assertEqual(c["animations_skipped_over_budget"], 2)
        self.assertEqual(c["dictionaries_over_budget"], 2)

    def test_cli_writes_counts_and_no_values(self):
        out = os.path.join(self.tmp.name, "out.json")
        r = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"], capture_output=True, text=True, errors="replace")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["counts"]["animations_decoded"], 2)
        for row in doc["dictionaries"]:      # per-dictionary rows hold counts, never values
            self.assertTrue(all(not isinstance(v, (list, dict)) for v in row.values()), row)

    def test_decode_rejects_negative_frames(self):
        result = subprocess.run(
            [sys.executable, "-I", TOOL, self.rpf, "--decode", str(self.offsets[1]), "--frames", "-1"],
            capture_output=True, text=True, errors="replace",
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("--frames must be 0 or more", result.stderr)
        self.assertEqual(result.stdout, "")

    def test_decode_prints_values(self):
        d = anim_clips.decode_one(self.rpf, self.offsets[1], 2)
        anim = d["packs"][0]["animations"][0]
        self.assertEqual(anim["decoded_frames"], NF)
        self.assertEqual(anim["decoded"]["2"], [[1.0, 2.0, 3.0]] * 2)
        self.assertEqual(anim["tracks"][0]["type"], "bonerotation")
        with self.assertRaisesRegex(ValueError, "no clip dictionary"):
            anim_clips.decode_one(self.rpf, self.offsets[3], 2)

    def test_decode_prints_track_values_from_a_six_page_dictionary(self) -> None:
        data = bytearray(two_sequence_dictionary("six-pages@dict"))
        page_map = struct.unpack_from("<Q", data, 8)[0] - SYSTEM
        data[page_map + 8] = 6
        pages = [data[i:i + 256] for i in range(0, 5 * 256, 256)] + [data[5 * 256:]]
        self.assertTrue(all(len(page) >= 64 for page in pages))
        path = os.path.join(self.tmp.name, "six-pages.rpf")
        root, *_ = write_rpf(path, pages)
        _, summary = anim_clips.scan(path)
        self.assertEqual(summary["pages"], {"6": 1})
        self.assertEqual(summary["counts"]["animations_decoded"], 1)

        result = subprocess.run(
            [sys.executable, "-I", TOOL, path, "--decode", str(root), "--frames", "3"],
            capture_output=True, text=True, errors="replace", check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        animation = json.loads(result.stdout)["packs"][0]["animations"][0]
        self.assertEqual(animation["decoded_frames"], NF)
        self.assertEqual(animation["decoded"]["2"], [[1.0, 2.0, 3.0]] * 3)
        self.assertEqual(animation["decoded"]["4"], [[0.0], [1.5], [3.0]])

    def test_decode_page_counts_on_or_across_sixth_page(self):
        data = bytearray(two_sequence_dictionary())
        pm = len(data)
        data.extend(bytes(96))
        struct.pack_into("<Q", data, 8, SYSTEM + pm)
        data[pm + 8] = 6
        for last_page in (pm, pm + 9):
            cuts = [0, 256, 512, 768, 1024, last_page, len(data)]
            pages = [data[a:b] for a, b in zip(cuts, cuts[1:])]
            root, *_ = write_rpf(self.rpf, pages)
            result = subprocess.run([sys.executable, "-I", TOOL, self.rpf, "--decode", str(root)],
                                    capture_output=True, text=True, errors="replace")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["packs"][0]["animations"][0]["decoded_frames"], NF)

    def test_invalid_map_cannot_borrow_counts_from_next_root(self):
        data = bytearray(two_sequence_dictionary())
        for pm in (len(data) + 64, 0xfffffff):
            struct.pack_into("<Q", data, 8, SYSTEM + pm)
            following = bytearray(two_sequence_dictionary())
            following[72] = 2
            write_rpf(self.rpf, [data, following])
            _, summary = anim_clips.scan(self.rpf)
            self.assertEqual(summary["unresolved"], {"page map not in the next stream": 1})
            self.assertEqual(summary["pages"], {"1": 1})

    def test_budget_rejects_nan_and_negative(self):
        for value in ("nan", "-1"):
            r = subprocess.run(
                [sys.executable, "-I", TOOL, self.rpf, os.path.join(self.tmp.name, "o.json"),
                 "--budget", value],
                capture_output=True, text=True, errors="replace",
            )
            self.assertEqual(r.returncode, 2, value)
            self.assertIn("--budget", r.stderr)
            self.assertEqual(r.stdout, "")

    def test_bad_arguments(self):
        for args in (["--jobs", "0", "x.json"], [], ["x.json", "--start", "-1"]):
            r = subprocess.run([sys.executable, "-I", TOOL, self.rpf, *args], capture_output=True, text=True, errors="replace")
            self.assertEqual(r.returncode, 2, args)
        missing = os.path.join(self.tmp.name, "missing.rpf")
        r = subprocess.run([sys.executable, "-I", TOOL, missing, os.path.join(self.tmp.name, "o.json")],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(r.returncode, 1)
        self.assertIn("missing.rpf", r.stderr)
        self.assertEqual(len(r.stderr.strip().splitlines()), 1)
        empty = os.path.join(self.tmp.name, "empty.rpf")
        open(empty, "wb").close()
        for source in (empty, self.tmp.name):
            r = subprocess.run([sys.executable, "-I", TOOL, source, os.path.join(self.tmp.name, "o.json")],
                               capture_output=True, text=True, errors="replace")
            self.assertNotEqual(r.returncode, 0, source)
            self.assertNotIn("Traceback", r.stderr)


    def test_output_cannot_overwrite_source_archive(self):
        with open(self.rpf, "rb") as f:
            data = f.read()
        other = os.path.join(os.path.dirname(self.rpf), ".", os.path.basename(self.rpf))
        r = subprocess.run([sys.executable, "-I", TOOL, self.rpf, other, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("output must not overwrite the input archive", r.stderr)
        self.assertNotIn("Traceback", r.stderr)
        with open(self.rpf, "rb") as f:
            self.assertEqual(f.read(), data)

    def test_output_that_is_a_directory_is_refused_before_the_scan(self):
        r = subprocess.run([sys.executable, "-I", TOOL, self.rpf, self.tmp.name, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("is a directory", r.stderr)
        self.assertNotIn("Traceback", r.stderr)

    def test_output_in_a_missing_folder_is_created(self):
        out = os.path.join(self.tmp.name, "reports", "sub", "out.json")
        r = subprocess.run([sys.executable, "-I", TOOL, self.rpf, out, "--quiet"],
                           capture_output=True, text=True, errors="replace")
        self.assertEqual(r.returncode, 0, r.stderr)
        with open(out, encoding="utf-8") as f:
            doc = json.load(f)
        self.assertEqual(doc["summary"]["counts"]["animations_decoded"], 2)


class ClipWindowTest(unittest.TestCase):
    """docs/animation.md: (end - start) / rate is the clip's +0x78 duration."""

    def test_window_matches_duration(self):
        f = anim_clips.window_matches_duration
        self.assertTrue(f(0.0, 2.0, 1.0, 2.0))
        self.assertTrue(f(1.0, 3.0, 0.5, 4.0))           # half speed: twice as long
        self.assertTrue(f(0.0, 1.0 / 3.0, 1.0, 0.33333334))  # float32 rounding of the duration
        self.assertFalse(f(0.0, 2.0, 1.0, 2.1))
        self.assertFalse(f(0.0, 2.0, 0.0, 2.0))          # rate 0
        self.assertFalse(f(0.0, math.inf, 1.0, math.inf))
        self.assertFalse(f(0.0, math.nan, 1.0, 1.0))

    def test_counts_only_ordinary_clips(self):
        clips = [
            {"kind": "clip", "duration": 2.0, "animations": [
                {"start": 0.0, "end": 2.0, "rate": 1.0}, {"start": 0.0, "end": 1.0, "rate": 0.5},
                {"start": 0.0, "end": 1.0, "rate": 1.0}, {"start": 0.0, "end": 1.0, "rate": 0.0}]},
            {"kind": "motion", "duration": 9.0, "animations": [{"start": 0.0, "end": 1.0, "rate": 1.0}]},
        ]
        stats = collections.Counter()
        anim_clips.count_clip_windows(clips, stats)
        self.assertEqual(stats, {"clip_windows": 4, "clip_windows_match_duration": 2, "clip_windows_rate_zero": 1})


if __name__ == "__main__":
    unittest.main()
