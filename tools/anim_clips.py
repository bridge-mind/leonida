"""Decode the clip dictionaries in anim.rpf: clips, animations and their keyframe channels.

The table of contents is encrypted, so this starts from the same stream walk as anim_index.py: every zlib stream
that inflates cleanly to its end. A clip dictionary is a RAGE resource whose system pages are one or two of those
streams. Its root stream starts with 8 zero bytes, a 0x50xxxxxx page-map pointer at +8 and zeros up to +0x38. The
page map's byte at +8 gives the number of system pages, and the dictionary takes that many back-to-back streams.
A last page that anim_index does not list (a zlib stream inflating to fewer than 64 bytes, or a 16-byte raw page
right after the stream) is joined when the pages then parse. Pointers are then offsets into the joined pages
(pointer - 0x50000000). See docs/animation.md, "Keyframe decoding".

Hash-only references are joined across the archive: a root pack reference with a null pointer is another
dictionary's joaat32(name), and each `.motion` child is {clip hash, dictionary hash}; the counts say how many of
them are found in the same archive.

Every animation is decoded down to the per-frame values. The default output is counts only: the number of
dictionaries, clips, animations, sequences, channels by type and track types. The checks are also counted: channel
definitions ending at the frame data, type-10 chunk closure, sequence boundaries and quaternion norms. The local
JSON adds one row per dictionary (offset, hash, counts, and its name, read from your own copy). No values or
payloads are written.

Usage:
    python -I anim_clips.py <anim.rpf> <out.json> [--start N] [--limit BYTES] [--jobs N] [--quiet]
    python -I anim_clips.py <anim.rpf> --decode <dictionary offset> [--frames N]

`--decode` prints one dictionary's clips and the decoded track values (first N frames, default 4) as JSON on
stdout, for local inspection. It writes no file.
Usage: `python -I tools/anim_clips.py --help` lists the options.
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

# -I keeps the script directory off sys.path, so add it explicitly before importing the stream walk.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anim_index  # noqa: E402
from pool_workers import process_workers  # noqa: E402

SYSTEM = 0x50000000
CHANNEL_NAMES = ("static_quat", "static_vec3", "static_float", "raw_float", "quantize_float",
                 "indirect_quantize_float", "linear_float", "cached_quat", "quat_normalize", "type9",
                 "delta2_float")
# Track-type hashes are joaat32 of these names.
TRACK_TYPE_NAMES = ("bonerotation", "bonetranslation", "bonescale", "genericcontrol", "generictranslation",
                    "facialtranslation", "cameratranslation", "camerarotation")
TRACK_TYPES = {anim_index.joaat32(n): n for n in TRACK_TYPE_NAMES}
TRACK_FORMATS = {0: "quaternion", 2: "vector3", 4: "float"}

# `.motion` axis names: joaat32 of the lower-case name, recovered by brute force over a word list
AXIS_NAMES = {anim_index.joaat32(n): n for n in ("yawangle", "rotationz", "facingdirection", "speed", "surfaceincline",
                                                "roll", "verticalheight", "pitch", "pitchangle", "height")}


class DecodeError(ValueError):
    pass


class NotClipDictionary(DecodeError):
    """A root with the dictionary header whose +0x70 is not a pack array."""


class System:
    """Joined system pages of one resource. Non-null pointers must land inside the pages."""

    def __init__(self, data):
        self.d = data

    def u16(self, at):
        return struct.unpack_from("<H", self.d, at)[0]

    def u32(self, at):
        return struct.unpack_from("<I", self.d, at)[0]

    def u64(self, at):
        return struct.unpack_from("<Q", self.d, at)[0]

    def f32(self, at):
        return struct.unpack_from("<f", self.d, at)[0]

    def ptr(self, at):
        if at < 0 or at + 8 > len(self.d):
            raise DecodeError(f"field at 0x{at:x} is outside the pages")
        q = self.u64(at)
        if q == 0:
            return None
        if not SYSTEM <= q < SYSTEM + len(self.d):
            raise DecodeError(f"pointer 0x{q:x} at 0x{at:x} is outside the pages")
        return q - SYSTEM

    def string(self, at):
        """16-byte atString: {char *ptr; u16 length; u16 capacity}, or a short string stored inline, in which
        case byte 15 is 0x80 | length and the characters start at byte 0."""
        if at < 0 or at + 16 > len(self.d):
            raise DecodeError(f"string at 0x{at:x} is outside the pages")
        tag = self.d[at + 15]
        if tag & 0x80:
            n = tag & 0x7F
            if n > 15:
                raise DecodeError(f"inline string at 0x{at:x} is {n} bytes")
            return self.d[at:at + n].decode("latin-1")
        p = self.ptr(at)
        if p is None:
            return None
        n = self.u16(at + 8)
        if p + n > len(self.d):
            raise DecodeError("string runs past the pages")
        return self.d[p:p + n].decode("latin-1")

    def array(self, at):
        """{T *ptr; u16 count; u16 capacity} -> (offset or None, count)"""
        p = self.ptr(at)
        return (p, self.u16(at + 8)) if p is not None else (None, 0)

    def atmap(self, at):
        """atMap {Entry **buckets; u32 bucket count; ...}, Entry {u64 key; T *value; Entry *next}"""
        buckets, n = self.ptr(at), self.u32(at + 8)
        if buckets is None:
            return []
        if n > 4096:
            raise DecodeError(f"atMap at 0x{at:x} has {n} buckets")
        out, seen = [], set()
        for i in range(n):
            e = self.ptr(buckets + 8 * i)
            while e is not None:
                if e in seen:
                    raise DecodeError("atMap chain loops")
                seen.add(e)
                out.append((self.u64(e), self.ptr(e + 8)))
                e = self.ptr(e + 16)
        return out


# ---------------------------------------------------------------- structure


def parse_animation(s, at):
    seqs, nseq = s.array(at + 0x50)
    tracks, ntr = s.array(at + 0x60)
    return {
        "offset": at, "frames": s.u16(at + 0x24), "duration": s.f32(at + 0x28),
        "sequences": [s.ptr(seqs + 8 * i) for i in range(nseq)],
        "tracks": [struct.unpack_from("<3I", s.d, tracks + 12 * i) for i in range(ntr)],
    }


def motion_times(s, at):
    """`.motion` +0x70: one 0x28-byte entry per child, {f32 child duration, f32 duration in the blend,
    {ptr, u16 count} curve of (child time, blend time) f32 pairs, 16 zero bytes}. -> [(duration, blend, points)]"""
    p, n = s.array(at + 0x70)
    out = []
    for i in range(n if p is not None else 0):
        e = p + 0x28 * i
        cp, cn = s.array(e + 8)
        points = [(s.f32(cp + 8 * j), s.f32(cp + 8 * j + 4)) for j in range(cn)] if cp is not None else []
        out.append((s.f32(e), s.f32(e + 4), points))
    return out


# Where a multi-axis node keeps its sample count (the number of +0x90 sample lists of a one-node `.motion`)
NODE_SAMPLE_COUNT_AT = {4: 0x90, 5: 0x90, 6: 0x50, 8: 0xA0}


def parameter_node(s, at, parameters):
    """The node that places a `.motion`'s children in parameter space: the pointer at +0x138 when +0x58 is 1;
    when +0x58 is 2, +0x120 points at a block whose +0x18 is the node. A node starts {u64 0, u32 type, u32 0}.
    Types 3, 4, 5, 6 and 8 have 1, 2, 3, 4 and 5 axes: +0x10 is an array of 8-byte {u32 name hash, u16, u16}.
    Type 3 is one sampled axis: then an array of (u32 sample index, f32 value) sorted by value, f32 min, f32 max.
    Types 4, 5, 6 and 8 keep their sample count at NODE_SAMPLE_COUNT_AT, and +0x20 is a {ptr, count} array of
    16-byte items ("pairs": words 0 and 2; {u32, u32 0, u32, u32 0} on types 6 and 8, null on type 4, not decoded);
    the rest of them is not decoded.
    Type 1 has no axes and its sample count (1) at +0x20.
    -> (type, axis or None) or None; axis = {names, slots, sample_count, and samples, min, max for type 3 or
    pairs for types 4-8}"""
    if parameters == 1:
        node = s.ptr(at + 0x138)
    elif parameters == 2:
        block = s.ptr(at + 0x120)
        node = s.ptr(block + 0x18) if block is not None else None
    else:
        return None
    if node is None or s.u64(node) != 0 or s.u32(node + 12) != 0:
        return None
    kind, axis = s.u32(node + 8), None
    if kind == 1:
        axis = {"names": [], "slots": [], "sample_count": s.u32(node + 0x20)}
    elif kind == 3 or kind in NODE_SAMPLE_COUNT_AT:
        names, nn = s.array(node + 0x10)
        if names is not None:
            axis = {"names": [s.u32(names + 8 * i) for i in range(nn)],
                    "slots": [(s.u16(names + 8 * i + 4), s.u16(names + 8 * i + 6)) for i in range(nn)]}
            if kind == 3:
                samples, ns = s.array(node + 0x20)
                if samples is None:
                    return kind, None
                axis.update(samples=[(s.u32(samples + 8 * i), s.f32(samples + 8 * i + 4)) for i in range(ns)],
                            min=s.f32(node + 0x30), max=s.f32(node + 0x34), sample_count=ns)
            else:
                axis["sample_count"] = s.u32(node + NODE_SAMPLE_COUNT_AT[kind])
                pairs, npairs = s.array(node + 0x20)     # null on type 4 and most type 5
                axis["pairs"] = None if pairs is None else [(s.u32(pairs + 16 * i), s.u32(pairs + 16 * i + 8))
                                                            for i in range(npairs)]
    return kind, axis


def matcher_array(s, field):
    """A `.motionmatcher` array: the pointer at `field`, its u32 count at `field - 4`. -> (offset or None, count)"""
    p, n = s.ptr(field), s.u32(field - 4)
    if p is None and n:
        raise DecodeError(f"array at 0x{field:x} has {n} items and no data")
    return p, n


def matcher_values(s, field, fmt):
    """The items of the `matcher_array` at `field`, each unpacked with the struct format `fmt` (one value per item
    for a one-letter format). -> list"""
    p, n = matcher_array(s, field)
    size = struct.calcsize("<" + fmt)
    if n and p + size * n > len(s.d):
        raise DecodeError(f"array at 0x{field:x} runs past the pages")
    items = [struct.unpack_from("<" + fmt, s.d, p + size * i) for i in range(n)]
    return [x[0] for x in items] if len(fmt) == 1 else items


def hash_table(s, field, slot):
    """An open-addressing table: u32 group count at `field`, u32 entry count at +4, pointer at +8. Each group is 16
    control bytes (0x80 empty, below 0x80 full) and then 16 slots of `slot` bytes. -> offsets of the full slots"""
    groups, count, p = s.u32(field), s.u32(field + 4), s.ptr(field + 8)
    if p is None:
        if groups or count:
            raise DecodeError(f"table at 0x{field:x} has no data")
        return []
    if groups & (groups - 1) or p + groups * (16 + 16 * slot) > len(s.d):
        raise DecodeError(f"table at 0x{field:x} has {groups} groups")
    full = []
    for g in range(groups):
        base = p + g * (16 + 16 * slot)
        for i, b in enumerate(s.d[base:base + 16]):
            if b < 0x80:
                full.append(base + 16 + slot * i)
            elif b != 0x80:
                raise DecodeError(f"table at 0x{field:x}: control byte 0x{b:02x}")
    if len(full) != count:
        raise DecodeError(f"table at 0x{field:x}: {len(full)} full slots for {count} entries")
    return full


def fmix64(k):
    """MurmurHash3's 64-bit finaliser."""
    k = (k ^ k >> 33) * 0xFF51AFD7ED558CCD & 0xFFFFFFFFFFFFFFFF
    k = (k ^ k >> 33) * 0xC4CEB9FE1A85EC53 & 0xFFFFFFFFFFFFFFFF
    return k ^ k >> 33


def clip_slot_hash(s, o):
    """The clip table's hash of a slot: `fmix64` of its first two u32 as one u64 (clip hash low, second word high)."""
    return fmix64(s.u32(o) | s.u32(o + 4) << 32)


def table_placed_by_key(s, field, slot, key=None):
    """True when every full slot of the `hash_table` at `field` sits where its hash h puts it: control byte
    h & 0x7F, in group (h >> 7) & (groups - 1) or, when that group is full, in a later one with every group
    between full (linear probing, wrapping). h is `key(s, slot offset)`, by default the slot's first u32."""
    groups, p = s.u32(field), s.ptr(field + 8)
    size = 16 + 16 * slot
    full = lambda g: all(b < 0x80 for b in s.d[p + g * size:p + g * size + 16])  # noqa: E731
    for o in hash_table(s, field, slot):
        g, i = divmod(o - p, size)
        h = key(s, o) if key else s.u32(o)
        home = (h >> 7) & (groups - 1)
        if s.d[p + g * size + (i - 16) // slot] != h & 0x7F or not all(
                full((home + d) % groups) for d in range((g - home) % groups)):
            return False
    return True


def parse_matcher(s, at):
    """The measured part of a `.motionmatcher` after its name (docs/animation.md, "`.motion` and `.motionmatcher`
    clips"). Arrays keep their u32 count just before the pointer. -> dict, values only for local inspection"""
    rec, n = matcher_array(s, at + 0x58)
    records = [(s.f32(rec + 24 * i), s.f32(rec + 24 * i + 4), s.f32(rec + 24 * i + 8), s.u32(rec + 24 * i + 12))
               for i in range(n)]
    off, no = matcher_array(s, at + 0x68)
    a78, ng = matcher_array(s, at + 0x78)
    ab8, nf = matcher_array(s, at + 0xB8)
    refs_at, nr = matcher_array(s, at + 0x98)
    refs = []
    for i in range(nr):
        if s.u64(refs_at + 16 * i) != 0:
            raise DecodeError("motionmatcher clip reference does not start with 8 zero bytes")
        refs.append((s.u32(refs_at + 16 * i + 8), s.u32(refs_at + 16 * i + 12)))
    return {
        "records": records,                                              # {f32; f32; f32; u32 clip; u64 0}
        "offsets": [s.u32(off + 4 * i) for i in range(no)],              # +0x68: u32, one more than the records
        "refs": refs,                                                    # +0x98: per clip {u64 0; clip; dictionary}
        "clips": [(s.u32(o), s.u32(o + 4), s.u32(o + 8)) for o in hash_table(s, at + 0xA0, 16)],  # (clip, ?, index)
        "keys": [(s.u32(o), s.u32(o + 4)) for o in hash_table(s, at + 0x170, 8)],                 # (hash, index)
        "self": (s.u32(at + 0x188), s.u32(at + 0x18C)),                  # own clip hash, joaat32(dictionary)
        "u16_78": [s.u16(a78 + 2 * i) for i in range(ng)],            # +0x78: first record of each clip, then n
        "u8_b8": list(struct.unpack_from(f"<{nf}B", s.d, ab8)) if nf else [],  # +0xB8: 0 or 1, one per clip
        "items_48": matcher_values(s, at + 0x48, "10I"),                 # 40-byte items
        "u32_118": matcher_values(s, at + 0x118, "I"),                   # the pose sample count
        "u16_128": matcher_values(s, at + 0x128, "H"),
        **matcher_features(s, at),                                       # first: it refuses a null +0x20
        "tables_placed_by_key": (table_placed_by_key(s, at + 0xA0, 16, clip_slot_hash),
                                 table_placed_by_key(s, at + 0x170, 8),
                                 table_placed_by_key(s, s.ptr(at + 0x20) + 0x48, 8)),  # clips, keys, feature names
    }


def matcher_features(s, at):
    """The `.motionmatcher`'s `+0x20` object: the features a pose is compared on. `+0x40` is an array of pointers to
    0x30-byte features {u32 column at +0x08; u16 width at +0x0C; u16 at +0x0E; u32 name hash at +0x18; u32 kind
    bits at +0x1C}, and `+0x48` a hash table of (name hash, feature index) with 8-byte slots, hashed differently
    from `+0x18`. With `+0x0E` = 1 the name hash is joaat32 of a skeleton bone's name and the width is 3 columns
    for kind bit 0, 3 for bit 1 and 6 for bit 3. `+0x28` is the skeleton (`matcher_skeleton`); `+0x98` holds one
    u32 per sample, a bit mask over the matcher's key table indices; `+0xB8` one f32 weight per pose column, in
    chunks of 8 (count at `+0xB4`), zero after the last column."""
    obj = s.ptr(at + 0x20)
    if obj is None:
        raise DecodeError("motionmatcher without a feature object")
    arr, n = matcher_array(s, obj + 0x40)
    features = []
    for i in range(n):
        f = s.ptr(arr + 8 * i)
        if f is None:
            raise DecodeError("motionmatcher feature pointer is null")
        features.append((s.u32(f + 0x08), s.u16(f + 0x0C), s.u32(f + 0x18)))
    kinds = [(s.u16(s.ptr(arr + 8 * i) + 0x0E), s.u32(s.ptr(arr + 8 * i) + 0x1C)) for i in range(n)]
    names = [(s.u32(o), s.u32(o + 4)) for o in hash_table(s, obj + 0x48, 8)]
    skel = s.ptr(obj + 0x28)
    if skel is None:
        raise DecodeError("motionmatcher feature object without its skeleton")
    second = s.ptr(obj + 0x30) is not None
    columns = features[-1][0] + features[-1][1] if features else 0
    poses = matcher_poses(s, obj, columns, second)
    tags, nt = matcher_array(s, obj + 0x98)
    if nt and tags + 4 * nt > len(s.d):
        raise DecodeError("motionmatcher per-sample words run past the pages")
    sample_u32 = list(struct.unpack_from(f"<{nt}I", s.d, tags)) if nt else []
    wp, nw = matcher_array(s, obj + 0xB8)
    if nw and wp + 32 * nw > len(s.d):
        raise DecodeError("motionmatcher column weights run past the pages")
    weights = list(struct.unpack_from(f"<{8 * nw}f", s.d, wp)) if nw else []
    return {"features": features, "feature_names": names,                  # (column, width, name hash); (hash, index)
            "feature_kinds": kinds,                                         # (u16 +0x0E, u32 +0x1C) per feature
            "feature_second": second,                                       # +0x30 object present
            "feature_b": matcher_object_b(s, s.ptr(obj + 0x30)) if second else None,
            "column_weights": weights,                                      # +0xB8: chunks of 8 f32, one per column
            "c0": struct.unpack_from("<ffI", s.d, obj + 0xC0),               # f32; f32 1.0; u32 1 rows / 3 blocks
            "skeleton": matcher_skeleton(s, skel), "sample_u32": sample_u32, **poses}


def matcher_object_b(s, at):
    """The feature object's `+0x30` object (112 bytes): the weighted bones a pose's reference frame is built from,
    in two forms told apart by its u32 at `+0x08`. Form 2: two bones (u32 joaat32 of their names at `+0x10` and
    `+0x14`) with f32 weights at `+0x18`/`+0x1C`, a copy of a 0x30-byte feature at `+0x30` (its u16 `+0x0E`, name
    hash `+0x18` and kind `+0x1C` at `+0x3E`, `+0x48` and `+0x4C`), its width at `+0x68` and a u16 flag at
    `+0x6C`. Form 1: a unit quaternion at `+0x10`, two arrays of {u32 count; u32 capacity; pointer} at `+0x40`
    and `+0x50` (n u32 bone name hashes, n f32 weights) and a u32 flag at `+0x60`. -> dict"""
    if at + 0x70 > len(s.d):
        raise DecodeError("motionmatcher +0x30 object runs past the pages")
    form = s.u32(at + 8)
    if form == 2:
        return {"form": 2, "bones": [s.u32(at + 0x10), s.u32(at + 0x14)],
                "weights": list(struct.unpack_from("<2f", s.d, at + 0x18)),
                "feature": (s.u16(at + 0x3E), s.u32(at + 0x48)), "kind": s.u32(at + 0x4C),
                "width": s.u32(at + 0x68), "flag": s.u16(at + 0x6C)}
    if form != 1:
        return {"form": form}
    out = {"form": 1, "rotation": struct.unpack_from("<4f", s.d, at + 0x10), "flag": s.u32(at + 0x60)}
    for name, field, fmt in (("bones", at + 0x40, "I"), ("weights", at + 0x50, "f")):
        n, p = s.u32(field), s.ptr(field + 8)
        if n and (p is None or p + 4 * n > len(s.d)):
            raise DecodeError("motionmatcher +0x30 object array runs past the pages")
        out[name] = list(struct.unpack_from(f"<{n}{fmt}", s.d, p)) if n else []
    return out


def matcher_skeleton(s, at):
    """The feature object's `+0x28`: the skeleton the features are measured on. `+0x6C` u16 bone count; `+0x28` the
    80-byte bones {f32 rotation quaternion[4]; f32 translation[3]; f32 ?; f32 1.0[4]; char *name at +0x30; s16
    parent at +0x46; u16 index at +0x48 and +0x4C; u16 bone id at +0x4A}; `+0x38` each bone's 4x4 local matrix and
    `+0x30` its inverse bind matrix (row-vector convention, translation in row 3); `+0x40` the parents as s16; and a
    tag table at `+0x18` (u32 slots, u32 entries, pointer) of 8-byte {u32 tag; u32 bone} slots, empty tag 0xFFFFFFFF.
    `+0x48` is a tree of the bone names (`skeleton_name_tree`).
    -> {"bones": [(name, parent, id, rotation, translation)], "tags", "local", "inverse_bind", "name_tree"}"""
    n, bones = s.u16(at + 0x6C), s.ptr(at + 0x28)
    mats = [s.ptr(at + 0x38), s.ptr(at + 0x30), s.ptr(at + 0x40)]
    if n and (bones is None or None in mats):
        raise DecodeError("motionmatcher skeleton without its arrays")
    if n and (bones + 80 * n > len(s.d) or max(mats[0], mats[1]) + 64 * n > len(s.d) or mats[2] + 2 * n > len(s.d)):
        raise DecodeError("motionmatcher skeleton runs past the pages")
    out = []
    for i in range(n):
        b = bones + 80 * i
        name_at = s.ptr(b + 0x30)
        end = s.d.find(b"\0", name_at, name_at + 256) if name_at is not None else -1
        if end < 0:
            raise DecodeError("motionmatcher bone without a name")
        out.append((s.d[name_at:end].decode("latin-1"), struct.unpack_from("<h", s.d, b + 0x46)[0], s.u16(b + 0x4A),
                    struct.unpack_from("<4f", s.d, b), struct.unpack_from("<3f", s.d, b + 0x10)))
    slots, entries, tt = s.u32(at + 0x18), s.u32(at + 0x1C), s.ptr(at + 0x20)
    if tt is None and (slots or entries) or tt is not None and tt + 8 * slots > len(s.d):
        raise DecodeError("motionmatcher skeleton tag table runs past the pages")
    tags = [struct.unpack_from("<II", s.d, tt + 8 * i) for i in range(slots)] if tt is not None else []
    tags = [t for t in tags if t[0] != 0xFFFFFFFF]
    if len(tags) != entries:
        raise DecodeError("motionmatcher skeleton tag table count does not fit")
    matrix = lambda p, i: [struct.unpack_from("<4f", s.d, p + 64 * i + 16 * r) for r in range(4)]  # noqa: E731
    return {"bones": out, "tags": tags, "parents": list(struct.unpack_from(f"<{n}h", s.d, mats[2])) if n else [],
            "local": [matrix(mats[0], i) for i in range(n)], "inverse_bind": [matrix(mats[1], i) for i in range(n)],
            "name_tree": skeleton_name_tree(s, s.ptr(at + 0x48))}


def skeleton_name_tree(s, at):
    """The skeleton's `+0x48` block: a prefix tree of the bone names. `{u16 n; u16 size}`, then nodes from `+4`, each
    `{u16 next sibling (relative, 0 for the last); u16 first child (relative, 0 for none); s16 value; char text[]}`.
    A node's name is its parents' texts and its own; where node + value lands in the n u16 at `+size`, that u16 is
    the bone the name belongs to. -> [(name, bone index)] in tree order, or [] for a null pointer"""
    if at is None:
        return []
    if at + 4 > len(s.d):
        raise DecodeError("motionmatcher bone name tree runs past the pages")
    n, size = s.u16(at), s.u16(at + 2)
    if at + size + 2 * n > len(s.d):
        raise DecodeError("motionmatcher bone name tree runs past the pages")
    out, stack, steps = [], [(4, "")] if size > 4 else [], 0
    while stack:
        node, prefix = stack.pop()
        steps += 1
        end = s.d.find(b"\0", at + node + 6, at + size) if 4 <= node and node + 6 < size else -1
        if end < 0 or steps > size:
            raise DecodeError("motionmatcher bone name tree node outside its block")
        sibling, child, value = struct.unpack_from("<HHh", s.d, at + node)
        name = prefix + s.d[at + node + 6:end].decode("latin-1")
        target = node + value
        if size <= target < size + 2 * n and (target - size) % 2 == 0:
            out.append((name, s.u16(at + target)))
        if sibling:
            stack.append((node + sibling, prefix))
        if child:                                                           # children before the next sibling
            stack.append((node + child, name))
    return out


def matcher_poses(s, obj, columns, rows_layout):
    """The pose matrix of a `.motionmatcher`: one row of f16 feature values per sample, at the feature object's
    `+0xF0` (pointer) with its count of 32-byte chunks at `+0xEC`. Two layouts, told apart by the `+0x30` object:
    with it, row-major, `+0xF8` samples, `+0xFC` columns, `+0x100` chunks per row (the fewest that hold the columns;
    the rest of a row is zero); without it, blocks of 8 samples, column-major inside a block (column c of sample r at
    `(r // 8) * D' * 8 + c * 8 + r % 8`, D' = columns rounded up to even), the samples at `+0x128`, and the lanes
    after the last sample copy it (and an odd column count's spare column is zero). With rows, `+0xA8` holds one
    48-byte `{f32 quaternion[4]; f32 vec3; f32 0; f32 1, 1, 1; f32 0}` per sample (count at `+0xA4`).
    -> {"poses": {...}, "pose_transforms": [...]}"""
    chunks, data = s.u32(obj + 0xEC), s.ptr(obj + 0xF0)
    if data is None:
        raise DecodeError("motionmatcher without a pose matrix")
    if rows_layout:
        samples, cols, per_row = s.u32(obj + 0xF8), s.u32(obj + 0xFC), s.u32(obj + 0x100)
        stride, layout = 16 * per_row, "rows"
        if not samples or cols != columns or per_row != -(-cols // 16) or chunks != samples * per_row:
            raise DecodeError("motionmatcher pose matrix size does not fit its rows")
    else:
        samples, cols = s.u32(obj + 0x128), s.u32(obj + 0xFC)
        stride, layout = cols + (cols & 1), "blocks of 8"
        if not samples or cols != columns or 16 * chunks != -(-samples // 8) * 8 * stride:
            raise DecodeError("motionmatcher pose matrix size does not fit its blocks")
    if data + 32 * chunks > len(s.d):
        raise DecodeError("motionmatcher pose matrix runs past the pages")
    h = struct.unpack_from(f"<{16 * chunks}e", s.d, data)
    if layout == "rows":
        padding = all(h[r * stride + c] == 0 for r in range(samples) for c in range(cols, stride))
    else:
        last, base = samples - 1, (samples - 1) // 8 * stride * 8
        padding = all(h[base + c * 8 + lane] == h[base + c * 8 + last % 8]
                      for c in range(cols) for lane in range(last % 8 + 1, 8)) and all(
            h[b * stride * 8 + c * 8 + lane] == 0 for b in range(-(-samples // 8)) for c in range(cols, stride)
            for lane in range(8))
    transforms = []
    if rows_layout:
        tr, nt = matcher_array(s, obj + 0xA8)
        if nt and tr + 48 * nt > len(s.d):
            raise DecodeError("motionmatcher pose transforms run past the pages")
        transforms = [struct.unpack_from("<12f", s.d, tr + 48 * i) for i in range(nt)]
    poses = {"layout": layout, "samples": samples, "columns": cols, "stride": stride, "at": data,
             "finite": all(map(math.isfinite, h)), "padding": padding}
    return {"poses": poses, "pose_transforms": transforms}


def matcher_pose(s, poses, sample):
    """One row of the pose matrix (see `matcher_poses`): the f16 feature values of `sample`, as floats."""
    if not 0 <= sample < poses["samples"]:
        raise IndexError(sample)
    cols, stride, at = poses["columns"], poses["stride"], poses["at"]
    if poses["layout"] == "rows":
        return list(struct.unpack_from(f"<{cols}e", s.d, at + 2 * sample * stride))
    base = at + 2 * (sample // 8) * stride * 8
    return [struct.unpack_from("<e", s.d, base + 2 * (c * 8 + sample % 8))[0] for c in range(cols)]


def bone_id(name):
    """A non-root bone's u16 id: the ELF hash of its upper-cased name, mod 0xFE8F, plus 0x170."""
    h = 0
    for c in name.upper().encode("latin-1"):
        h = ((h << 4) + c) & 0xFFFFFFFF
        top = h & 0xF0000000
        if top:
            h ^= top >> 24
        h &= ~top & 0xFFFFFFFF
    return h % 0xFE8F + 0x170


def quaternion_rows(x, y, z, w):
    """The rotation of a unit quaternion as a 3x3 matrix for row vectors (v' = v M)."""
    return [[1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)],
            [2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)],
            [2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)]]


def skeleton_checks(skel):
    """-> (local matrices are the bones' rotation and translation, inverse binds undo the parent chain), each
    to 2e-3, for the row-vector convention. A parent that isn't an earlier bone makes the chain check False."""
    local_ok, inverse_ok, world = True, True, []
    for i, (_name, parent, _id, q, t) in enumerate(skel["bones"]):
        m = skel["local"][i]
        r = quaternion_rows(*q)
        local_ok &= all(abs(m[a][b] - r[a][b]) < 2e-3 for a in range(3) for b in range(3)) and all(
            abs(m[3][a] - t[a]) < 2e-3 for a in range(3))
        lm = [list(m[a][:3]) + [0.0] for a in range(3)] + [list(t) + [1.0]]
        if parent >= i:
            inverse_ok, g = False, lm
        else:
            g = lm if parent < 0 else [[sum(lm[a][k] * world[parent][k][b] for k in range(4)) for b in range(4)]
                                       for a in range(4)]
        world.append(g)
        ib = skel["inverse_bind"][i]
        inv = [list(row[:3]) + [0.0] for row in ib[:3]] + [list(ib[3][:3]) + [1.0]]
        inverse_ok &= all(abs(sum(g[a][k] * inv[k][b] for k in range(4)) - (a == b)) < 2e-3
                          for a in range(4) for b in range(4))
    return local_ok, inverse_ok


def count_matcher(c, dictionary_hash, stats):
    """Counts for a `.motionmatcher`'s decoded fields (no values)."""
    m = c.get("matcher")
    if m is None:
        stats["matcher_unreadable"] += 1
        return
    n = len(m["records"])
    stats["matcher_records"] += n
    stats["matcher_record_index_is_position"] += [r[3] for r in m["records"]] == list(range(n))
    stats["matcher_record_third_is_minus_one"] += all(r[2] == -1.0 for r in m["records"])
    o = m["offsets"]
    stats["matcher_offsets_one_more_than_records"] += len(o) == n + 1
    stats["matcher_offsets_from_0_nondecreasing"] += bool(o) and o[0] == 0 and all(a <= b for a, b in zip(o, o[1:]))
    stats["matcher_clip_table_entries"] += len(m["clips"])
    stats["matcher_clip_table_indices_0_to_n"] += sorted(i for _c, _k, i in m["clips"]) == list(range(len(m["clips"])))
    stats["matcher_clip_table_one_per_record"] += sorted(i for _c, _k, i in m["clips"]) == list(range(n))
    nc = len(m["clips"])
    stats["matcher_with_refs"] += bool(m["refs"])
    stats["matcher_refs"] += len(m["refs"])
    if m["refs"]:
        stats["matcher_refs_one_per_clip"] += len(m["refs"]) == nc
        stats["matcher_refs_name_the_table_clip"] += all(
            i < len(m["refs"]) and m["refs"][i][0] == ch for ch, _k, i in m["clips"])
    stats["matcher_key_table_entries"] += len(m["keys"])
    stats["matcher_key_table_indices_0_to_n"] += sorted(i for _h, i in m["keys"]) == list(range(len(m["keys"])))
    stats["matcher_self_is_own_key"] += m["self"] == (c["hash"], dictionary_hash)
    g = m["u16_78"]
    stats["matcher_u16_78_from_0_to_records"] += bool(g) and g[0] == 0 and g[-1] == n and all(
        a <= b for a, b in zip(g, g[1:]))
    stats["matcher_u16_78_one_more_than_clips"] += len(g) == nc + 1
    stats["matcher_record_clip_in_its_u16_78_range"] += len(g) == nc + 1 and all(
        r[3] < nc and g[r[3]] <= i < g[r[3] + 1] for i, r in enumerate(m["records"]))
    stats["matcher_u8_b8_one_per_clip"] += len(m["u8_b8"]) == nc
    stats["matcher_u8_b8_set"] += sum(m["u8_b8"])
    stats["matcher_u8_b8_0_or_1"] += bool(m["u8_b8"]) and all(f in (0, 1) for f in m["u8_b8"])
    f = m["features"]
    stats["matcher_features"] += len(f)
    stats["matcher_feature_columns"] += f[-1][0] + f[-1][1] if f else 0
    stats["matcher_feature_columns_chain_from_0"] += bool(f) and f[0][0] == 0 and all(
        a[0] + a[1] == b[0] for a, b in zip(f, f[1:]))
    stats["matcher_feature_widths_multiple_of_3"] += bool(f) and all(w % 3 == 0 for _c, w, _h in f)
    stats["matcher_feature_names_distinct"] += bool(f) and len({h for _c, _w, h in f}) == len(f)
    stats["matcher_feature_name_table_indices_0_to_f"] += bool(f) and sorted(
        i for _h, i in m["feature_names"]) == list(range(len(f)))
    stats["matcher_feature_object_second_pointer"] += m["feature_second"]
    stats["matcher_item_48_one_zero_item_with_u32_1_at_0x20"] += len(m["items_48"]) == 1 and m["items_48"][0] == (
        0,) * 8 + (1, 0)
    stats["matcher_u32_118_is_the_sample_count"] += m["u32_118"] == [m["poses"]["samples"]]
    stats["matcher_u16_128_is_one_0"] += m["u16_128"] == [0]
    cols, w = m["poses"]["columns"], m["column_weights"]
    stats["matcher_column_weights_chunks_of_8"] += len(w) == 8 * -(-cols // 8)
    stats["matcher_column_weights_positive_sum_1_then_0"] += len(w) >= cols and all(x > 0 for x in w[:cols]) and abs(
        sum(w[:cols]) - 1) < 1e-4 and not any(w[cols:])
    stats["matcher_column_weights_same_within_each_vec3"] += len(w) >= cols and all(
        abs(w[i] - w[i + 1]) <= 1e-6 * w[i] and abs(w[i] - w[i + 2]) <= 1e-6 * w[i] for i in range(0, cols - 2, 3))
    stats["matcher_c8_1_with_rows_3_with_blocks"] += m["c0"][1:] == (1.0, 1 if m["poses"]["layout"] == "rows" else 3)
    names = {anim_index.joaat32(x[0]) for x in m["skeleton"]["bones"]}
    b = m["feature_b"]
    if b is not None:
        stats[f"matcher_object_b_form_{b['form']}"] += 1
        if b["form"] == 2:
            k = b["kind"]
            stats["matcher_object_b_form_2_width_from_kind_bits"] += b["width"] == 3 * (k & 1) + 3 * (
                k >> 1 & 1) + 6 * (k >> 3 & 1)
            stats["matcher_object_b_form_2_kind_is_a_feature_kind"] += k in {kk for _e, kk in m["feature_kinds"]}
            stats["matcher_object_b_form_2_copies_the_first_feature"] += bool(f) and b["feature"] == (
                m["feature_kinds"][0][0], f[0][2]) and k == m["feature_kinds"][0][1]
            stats["matcher_object_b_form_2_two_distinct_bones"] += len(set(b["bones"])) == 2 and set(
                b["bones"]) <= names
        if b["form"] == 1:
            q = b["rotation"]
            stats["matcher_object_b_form_1_rotation_unit"] += abs(sum(x * x for x in q) - 1) < 1e-4
            stats["matcher_object_b_form_1_bones_one_per_weight"] += len(b["bones"]) == len(b["weights"])
            stats["matcher_object_b_form_1_bones_name_skeleton_bones"] += bool(b["bones"]) and set(
                b["bones"]) <= names
        if b["form"] in (1, 2):
            wb = b["weights"]
            stats["matcher_object_b_weights_positive_sum_1"] += bool(wb) and all(x > 0 for x in wb) and abs(
                sum(wb) - 1) < 1e-4
            stats[f"matcher_object_b_form_{b['form']}_flag_{b['flag']}"] += 1
    for name, placed in zip(("clip", "key", "feature_name"), m["tables_placed_by_key"]):
        stats[f"matcher_{name}_table_placed_by_key"] += placed
    sk = m["skeleton"]
    bones = sk["bones"]
    stats["matcher_skeleton_bones"] += len(bones)
    parents = sk["parents"]
    stats["matcher_skeleton_one_root_parents_first"] += bool(bones) and parents == [b[1] for b in bones] and sum(
        p < 0 for p in parents) == 1 and all(0 <= p < i for i, p in enumerate(parents) if p >= 0)
    stats["matcher_skeleton_tags_are_name_hash_and_id"] += bool(bones) and sorted(sk["tags"]) == sorted(
        [(anim_index.joaat32(b[0]), i) for i, b in enumerate(bones)] + [(b[2], i) for i, b in enumerate(bones)])
    stats["matcher_skeleton_bone_ids_from_names_root_0"] += bool(bones) and all(
        b[2] == (0 if b[1] < 0 else bone_id(b[0])) for b in bones)
    stats["matcher_skeleton_name_tree_maps_each_name_to_its_bone"] += bool(bones) and sorted(
        sk["name_tree"], key=lambda x: x[1]) == [(b[0], i) for i, b in enumerate(bones)]
    local_ok, inverse_ok = skeleton_checks(sk)
    stats["matcher_skeleton_local_is_bone_transform"] += bool(bones) and local_ok
    stats["matcher_skeleton_inverse_bind_undoes_chain"] += bool(bones) and inverse_ok
    simple = [(fw, k, h) for (_c, fw, h), (e, k) in zip(f, m["feature_kinds"]) if e == 1]
    stats["matcher_features_with_0e_1"] += len(simple)
    stats["matcher_features_0e_1_name_a_bone"] += sum(h in names for _w, _k, h in simple)
    stats["matcher_features_0e_1_width_from_kind_bits"] += sum(
        fw == 3 * (k & 1) + 3 * (k >> 1 & 1) + 6 * (k >> 3 & 1) for fw, k, _h in simple)
    for (_c, fw, h), (e, k) in zip(f, m["feature_kinds"]):
        if e == 5:                                  # not a bone: one of +0x18 and +0x1C is 0
            stats["matcher_features_0e_5"] += 1
            stats["matcher_features_0e_5_name_a_bone"] += h in names
            stats["matcher_features_0e_5_with_1c_0" if k == 0 else "matcher_features_0e_5_with_18_0" if h == 0 else
                  "matcher_features_0e_5_with_neither_0"] += 1
    u = m["sample_u32"]
    if u:
        stats["matcher_sample_u32_one_per_sample"] += len(u) == m["poses"]["samples"]
        stats["matcher_sample_u32_same_within_a_record"] += all(
            len(set(u[a:b])) <= 1 for a, b in zip(m["offsets"], m["offsets"][1:]))
        used = 0
        for x in u:
            used |= x
        stats["matcher_sample_u32_bits_are_key_indices"] += used == (1 << len(m["keys"])) - 1
    p = m["poses"]
    stats["matcher_pose_layout_rows" if p["layout"] == "rows" else "matcher_pose_layout_blocks_of_8"] += 1
    stats["matcher_pose_samples"] += p["samples"]
    stats["matcher_pose_samples_are_offsets_end"] += bool(m["offsets"]) and p["samples"] == m["offsets"][-1]
    stats["matcher_pose_values_finite"] += p["finite"]
    stats["matcher_pose_padding_as_expected"] += p["padding"]
    t = m["pose_transforms"]
    if t:
        stats["matcher_pose_transforms_one_per_sample"] += len(t) == p["samples"]
        stats["matcher_pose_transforms_unit_quaternion_scale_1"] += all(
            abs(sum(x * x for x in q[:4]) - 1) < 1e-3 and q[7] == 0 and q[8:11] == (1.0, 1.0, 1.0) and q[11] == 0
            for q in t)


def motion_samples(s, at):
    """`.motion` +0x90: one {ptr, u16 count} list per sample of the parameter node, each item a 12-byte
    {u32 child index, f32, f32}. A one-node `.motion` has as many lists as its node has samples, so type-3 sample
    index k plays child list[k][0]. -> [[(child, f32, f32)]] ([] when +0x90 is null)"""
    p, n = s.array(at + 0x90)
    out = []
    for i in range(n if p is not None else 0):
        items, count = s.array(p + 16 * i)
        out.append([(s.u32(items + 12 * j), s.f32(items + 12 * j + 4), s.f32(items + 12 * j + 8))
                    for j in range(count)] if items is not None else [])
    return out


def parse_clip(s, at):
    kind = s.u32(at + 0x20)
    if kind >= 16:  # +0x20 is a pointer: a parametric clip that refers to packs outside the dictionary
        name_at, tag = s.u64(at + 0x30), s.d[at + 0x3F]
        # `.motion`: name at +0x30, a pointer or inline (+0x3F 0x80 | length). For a `.motionmatcher` +0x3F is
        # byte 7 of its name at +0x38: a pointer's zero high byte or an inline character.
        if SYSTEM <= name_at < SYSTEM + len(s.d) or tag & 0x80 and tag & 0x7F <= 15:
            motion = {"kind": "motion", "name": s.string(at + 0x30), "animations": [], "children": None,
                      "parameters": None}
            try:
                # +0x58 u32 parameter count; +0x60 array of {u64 0; u32 clip hash; u32 dictionary hash}
                children, n = s.array(at + 0x60)
                if children is None:
                    motion["children"], motion["parameters"] = [], s.u32(at + 0x58)
                elif children + 16 * n <= len(s.d) and all(s.u64(children + 16 * i) == 0 for i in range(n)):
                    motion["children"] = [(s.u32(children + 16 * i + 8), s.u32(children + 16 * i + 12))
                                          for i in range(n)]
                    motion["parameters"] = s.u32(at + 0x58)
            except (DecodeError, struct.error):
                pass
            try:  # read apart: a malformed node must not mark a readable time table unreadable
                motion["times"] = motion_times(s, at)
            except (DecodeError, struct.error):
                motion["times"] = None
            try:
                motion["node"] = parameter_node(s, at, motion["parameters"])
            except (DecodeError, struct.error):
                motion["node"] = None
            try:
                motion["samples"] = motion_samples(s, at)
            except (DecodeError, struct.error):
                motion["samples"] = None
            return motion
        # `.motionmatcher`: a 16-byte id at +0x28, name at +0x38
        matcher = {"kind": "motionmatcher", "name": s.string(at + 0x38), "animations": [], "matcher": None}
        try:
            matcher["matcher"] = parse_matcher(s, at)
        except (DecodeError, struct.error):
            pass
        return matcher
    clip = {"kind": "clip", "name": s.string(at + 0x28), "duration": s.f32(at + 0x78), "animations": []}
    lst = s.ptr(at + 0x68)
    if lst is not None:
        entries, n = s.array(lst)
        for i in range(n):
            e = entries + 0x38 * i
            clip["animations"].append({"animation": s.ptr(e), "hash": s.u32(e + 8), "start": s.f32(e + 0x28),
                                       "end": s.f32(e + 0x2C), "rate": s.f32(e + 0x30)})
    return clip


def parse_dictionary(s):
    """Root object at page offset 0: +0x38 name, +0x48 clip map, +0x68 joaat32(name), +0x70 pack references."""
    refs_word = s.u64(0x70)
    if refs_word and not SYSTEM <= refs_word < SYSTEM + len(s.d):
        # Same root layout, other content: "RIG" + version at +0x70 is a rig dictionary.
        raise NotClipDictionary("rig" if s.d[0x70:0x73] == b"RIG" else "other")
    d = {"name": s.string(0x38), "hash": s.u32(0x68), "clips": [], "packs": [], "external_packs": 0,
         "external_hashes": []}
    for key, value in s.atmap(0x48):
        if value is not None:
            clip = parse_clip(s, value)
            clip["hash"] = key & 0xFFFFFFFF          # the key is joaat32(dictionary) << 32 | clip hash
            d["clips"].append(clip)
    refs, n = s.array(0x70)
    for i in range(n):
        pack = s.ptr(refs + 16 * i)
        if pack is None:
            d["external_packs"] += 1        # only a hash: another clip dictionary's joaat32(name)
            d["external_hashes"].append(s.u32(refs + 16 * i + 8))
            continue
        anims = [parse_animation(s, v) for _k, v in s.atmap(pack + 0x48) if v is not None]
        d["packs"].append({"name": s.string(pack + 0x38), "hash": s.u32(pack + 0x68), "animations": anims})
    return d


# ---------------------------------------------------------------- channels


def zigzag(u):
    return (u >> 1) if not u & 1 else -((u + 1) >> 1)


def _bits(blob, pos, k):
    if not k:
        return 0
    o = pos >> 3
    v = int.from_bytes(blob[o:o + ((pos & 7) + k + 7 >> 3)], "little")
    return (v >> (pos & 7)) & ((1 << k) - 1)


def decode_delta2(data, at, nf, stats=None):
    """Channel type 10, second-order delta coded. Returns (values, byte length).

    u32 word count | u8 c1, c2, c3, 0 | f32 quantum | f32 offset | LSB-first bits:
    nc = ceil(nf / 64) chunk payload positions (c1 bits each), nc chunk start values (c2 bits),
    ceil((nf - 1) / 8) unary widths, one per 8 deltas (k zeros then a one), and at the end 8 lanes x c3 u32 words,
    interleaved (word w * 8 + j is lane j). Lane j holds delta j of every 8-delta group, k bits each.
    In a chunk: x = start, v = 0; for each frame v += zigzag(raw); x += v. Value = x * quantum + offset.
    """
    words, packed, quantum, offset = struct.unpack_from("<IIff", data, at)
    c1, c2, c3 = packed & 255, (packed >> 8) & 255, (packed >> 16) & 255
    size = 4 * words
    if size < 16 or at + size > len(data):
        raise DecodeError("type-10 channel runs past the sequence")
    blob = bytes(data[at + 16:at + size])
    nbits = 8 * len(blob)
    nc, ng = (nf + 63) // 64, (nf + 6) // 8      # nf - 1 deltas, 8 per width code
    starts = [_bits(blob, nc * c1 + i * c2, c2) for i in range(nc)]
    pos = nc * (c1 + c2)
    widths = []
    for _ in range(ng):                     # unary: k zero bits, then a one
        o, k = pos >> 3, 0
        byte = blob[o] >> (pos & 7) if o < len(blob) else 0
        if byte:
            k = (byte & -byte).bit_length() - 1
        else:
            k = 8 - (pos & 7)
            o += 1
            while o < len(blob) and not blob[o]:
                k += 8
                o += 1
            if o >= len(blob):
                raise DecodeError("type-10 width code runs past the channel")
            k += (blob[o] & -blob[o]).bit_length() - 1
        pos += k + 1
        widths.append(k)
    lane_at = nbits - 256 * c3
    if lane_at < pos:
        raise DecodeError("type-10 lanes overlap the width codes")
    if sum(widths) > 32 * c3:
        raise DecodeError("type-10 widths need more lane bits than the channel has")
    lw = struct.unpack_from(f"<{8 * c3}I", blob, lane_at // 8)
    lanes = [b"".join(struct.pack("<I", lw[w * 8 + j]) for w in range(c3)) for j in range(8)]
    raw, lp = [], 0
    for k in widths:
        for j in range(8):
            raw.append(_bits(lanes[j], lp, k))
        lp += k
    values = [0.0] * nf
    for ci in range(nc):
        x, v, f0 = starts[ci], 0, ci * 64
        values[f0] = x * quantum + offset
        for j in range(64):
            f = f0 + j + 1
            if f >= nf:
                break
            v += zigzag(raw[f0 + j])
            x += v
            if j == 63:
                if stats is not None:
                    stats["delta2_boundaries"] += 1
                    stats["delta2_boundaries_closed"] += x == starts[ci + 1]
                break
            values[f] = x * quantum + offset
    return values, size


def sequence_header(data, at):
    h = struct.unpack_from("<5I5H2B", data, at)
    return {"length": h[1], "frame_offset": h[3], "frames": h[6], "frame_bytes": h[7], "quantize_bits": h[9]}


def decode_sequence(s, at, stats=None):
    """One sequence: 0x20-byte header, then channel definitions, frame bits, u16 counts[11] and the
    (track << 2 | component) lists. Returns (frames, channels) with channels = [type, track, comp, values]."""
    h = sequence_header(s.d, at)
    data = memoryview(s.d)[at + 0x20:at + 0x20 + h["length"]]
    nf, fo, fl = h["frames"], h["frame_offset"], h["frame_bytes"]
    lists_at = fo + fl * nf
    if lists_at + 22 > len(data):
        raise DecodeError("sequence counts run past its data")
    counts = struct.unpack_from("<11H", data, lists_at)
    lists, p = [], lists_at + 22
    for c in counts:
        lists.append([(x >> 2, x & 3) for x in struct.unpack_from(f"<{c}H", data, p)])
        p += 2 * ((c + 3) // 4 * 4)
    p, channels, frame_fields = 0, [], []
    for t, c in enumerate(counts):
        if c and t in (6, 9):
            raise DecodeError(f"channel type {t} ({CHANNEL_NAMES[t]}) is not decoded")
        for i in range(c):
            track, comp = lists[t][i]
            value = None
            if t == 0 or t == 1:
                value = struct.unpack_from("<3f", data, p)
                p += 12
            elif t == 2:
                value = struct.unpack_from("<f", data, p)[0]
                p += 4
            elif t == 3:                      # raw f32 per frame, no definition bytes
                frame_fields.append((len(channels), 32, "raw", None))
            elif t == 4:
                bits, quantum, offset = struct.unpack_from("<Iff", data, p)
                p += 12
                frame_fields.append((len(channels), bits, quantum, offset))
            elif t == 5:
                fbits, vbits, words, quantum, offset = struct.unpack_from("<IIIff", data, p)
                table_bytes = bytes(data[p + 20:p + 20 + 4 * words])
                p += 20 + 4 * words
                n = (32 * words) // vbits if vbits else 1
                table = [(_bits(table_bytes, i * vbits, vbits) if vbits else 0) * quantum + offset for i in range(n)]
                frame_fields.append((len(channels), fbits, "table", table))
            elif t == 10:
                value, size = decode_delta2(data, p, nf, stats)
                p += size
            channels.append([t, track, comp, value])
    if p != fo:
        raise DecodeError(f"channel definitions end at 0x{p:x}, frame data starts at 0x{fo:x}")
    if stats is not None:
        stats["quantize_bits_match"] += sum(b for _i, b, q, _o in frame_fields if q not in ("raw", "table")) \
            == h["quantize_bits"]
    if frame_fields:
        need = sum(b for _i, b, _q, _o in frame_fields)
        if need > 8 * fl:
            raise DecodeError(f"frame fields need {need} bits, frames have {8 * fl}")
        columns = {i: [] for i, *_ in frame_fields}
        for f in range(nf):
            row = int.from_bytes(data[fo + f * fl:fo + (f + 1) * fl], "little")
            for i, bits, quantum, offset in frame_fields:
                x = row & ((1 << bits) - 1)
                row >>= bits
                if quantum == "raw":
                    columns[i].append(struct.unpack("<f", struct.pack("<I", x))[0])
                elif quantum == "table":
                    if x >= len(offset):
                        raise DecodeError("indirect index past its value table")
                    columns[i].append(offset[x])
                else:
                    columns[i].append(x * quantum + offset)
        for i, col in columns.items():
            channels[i][3] = col
    return nf, channels


def assemble_tracks(nf, channels, ntracks, stats=None):
    """Channels -> {track: [per-frame tuple]}. Quaternions are (x, y, z, w).

    Type 7 on a track with three stored components (comp 0..2): rebuild the missing component, placed at the
    type-7 channel's comp position, as +sqrt(1 - sum of squares); the stored three fill the other positions in
    order. Type 8 on a track with all four components stored: use them and normalise (nothing is rebuilt)."""
    by = collections.defaultdict(list)
    for t, track, comp, v in channels:
        if track >= ntracks:
            raise DecodeError("channel names a track past the track list")
        by[track].append((t, comp, v))
    out = {}
    for track, chs in by.items():
        whole = [c for c in chs if c[0] in (0, 1)]
        if whole:
            t, _comp, v = whole[0]
            if t == 0:
                x, y, z = v
                r = 1.0 - x * x - y * y - z * z
                if stats is not None:
                    stats["quaternion_static"] += 1
                    stats["quaternion_static_xyz_over_unit"] += r < -1e-3
                v = (x, y, z, math.sqrt(max(0.0, r)))
            out[track] = [tuple(v)] * nf
            continue
        comps, rebuild, normalise = {}, None, False
        for t, comp, v in chs:
            if t == 7:
                rebuild = comp
            elif t == 8:
                normalise = True
            else:
                comps[comp] = v if isinstance(v, list) else [v] * nf
        if rebuild is not None and sorted(comps) != [0, 1, 2]:
            raise DecodeError("type-7 channel on a track without exactly three stored components")
        if normalise and sorted(comps) != [0, 1, 2, 3]:
            raise DecodeError("type-8 channel on a track without four stored components")
        n = 3 if rebuild is not None else (max(comps) + 1 if comps else 0)
        frames, worst = [], 0.0
        for f in range(nf):
            vec = [comps[c][f] if c in comps else 0.0 for c in range(n)]
            if rebuild is not None:
                r = 1.0 - sum(a * a for a in vec)
                worst = max(worst, -r)
                vec.insert(rebuild, math.sqrt(max(0.0, r)))
            elif n == 4:
                worst = max(worst, abs(math.sqrt(sum(a * a for a in vec)) - 1.0))
            if rebuild is not None or normalise:
                norm = math.sqrt(sum(a * a for a in vec)) or 1.0
                vec = [a / norm for a in vec]      # quantisation leaves |q| slightly off 1
            frames.append(tuple(vec))
        if stats is not None and (rebuild is not None or n == 4):
            kind = "rebuilt" if rebuild is not None else ("type8_four_stored" if normalise else "four_stored")
            stats[f"quaternion_{kind}"] += 1
            stats[f"quaternion_{kind}_worst_1e6"] = max(stats[f"quaternion_{kind}_worst_1e6"], round(worst * 1e6))
        out[track] = frames
    return out


def decode_animation(s, anim, stats=None):
    """Concatenate the sequences. Consecutive sequences share one frame, so sum(frames) == total + n - 1 and the
    first frame of every sequence after the first is dropped (and compared with the frame it repeats)."""
    tracks, total, frames_sum = {}, 0, 0
    for si, at in enumerate(anim["sequences"]):
        nf, channels = decode_sequence(s, at, stats)
        frames_sum += nf
        if stats is not None:
            for t, *_ in channels:
                stats[f"channel:{t:02d}_{CHANNEL_NAMES[t]}"] += 1
        part = assemble_tracks(nf, channels, len(anim["tracks"]), stats)
        for k, frames in part.items():
            if si and stats is not None and k in tracks and frames:
                a, b = tracks[k][-1], frames[0]
                if anim["tracks"][k][2] == 0 and len(a) == 4 == len(b):
                    stats["boundary_quaternions"] += 1
                    stats["boundary_quaternions_same_rotation"] += abs(sum(x * y for x, y in zip(a, b))) > 0.9999
                else:
                    stats["boundary_values"] += 1
                    stats["boundary_values_within_1e-3"] += max(abs(x - y) for x, y in zip(a, b)) < 1e-3
            have = tracks.setdefault(k, [frames[0]] * total if frames else [])
            have.extend(frames[1:] if si else frames)
        total += nf - (1 if si else 0)
        for k, have in tracks.items():
            if len(have) < total:
                have.extend([have[-1]] * (total - len(have)))
    if stats is not None:
        stats["frames_sum_matches"] += frames_sum == anim["frames"] + len(anim["sequences"]) - 1
    return tracks, total


# ---------------------------------------------------------------- resources


def is_root(head):
    """A clip-dictionary root page: 8 zero bytes, a 0x50xxxxxx pointer at +8, zeros from +0x10 to +0x38."""
    return (len(head) >= 0x40 and head[:8] == bytes(8) and head[16:0x38] == bytes(0x28)
            and struct.unpack_from("<Q", head, 8)[0] >> 28 == 5)


def _named_root_hash(page):
    """Return the verified page-local name hash shared by dictionary and pack headers."""
    if len(page) < 0x6C or not is_root(page):
        return None
    s = System(page)
    try:
        name = s.string(0x38)
    except DecodeError:
        return None
    stored = s.u32(0x68)
    return stored if name and anim_index.joaat32(name) == stored else None


def _local_packs(data):
    """Referenced (system pointer, name hash) pairs, or None until the table is available."""
    if len(data) < 0x7a:
        return None
    refs, count = struct.unpack_from("<QH", data, 0x70)
    if not count or refs < SYSTEM:
        return set()
    start = refs - SYSTEM
    end = start + count * 16
    if end > len(data):
        return None
    return {(ptr, name_hash) for ptr, name_hash in struct.iter_unpack("<QI4x", memoryview(data)[start:end])
            if ptr}


def _inflate(m, row):
    return zlib.decompressobj().decompress(m[row["offset"]:row["offset"] + row["stored"]])


def _next_to(prev, row):
    return row["offset"] == (prev["offset"] + prev["stored"] + 15) & ~15


def _gap_page(m, row, first):
    """The page right after `row` when it is not a listed stream: a zlib stream that inflates to fewer than the
    64 bytes anim_index lists, or a 16-byte raw tail (as rpf9-explorer reads). It is used only when the joined
    pages parse as a clip dictionary. -> segment dict or None."""
    d = zlib.decompressobj()
    d.decompress(m[row["offset"]:row["offset"] + row["stored"]])
    end = row["offset"] + row["stored"] - len(d.unused_data)
    at = (end + 15) & ~15
    if at + 16 > len(m) or m[end:at] != bytes(at - end):
        return None
    if anim_index.ZHDR.match(m, at) and m[at:at + 3] != b"KB2":
        z, chunk = zlib.decompressobj(), m[at:at + 4096]
        try:
            page = z.decompress(chunk)
        except zlib.error:
            return None
        if not z.eof:
            return None
        seg = {"offset": at, "stored": len(chunk) - len(z.unused_data), "raw": False, "kind": "small zlib page"}
    else:
        page = bytes(m[at:at + 16])
        seg = {"offset": at, "stored": 16, "raw": True, "kind": "raw 16-byte page"}
    try:
        parse_dictionary(System(first + page))
    except (DecodeError, struct.error, IndexError):
        return None
    return seg


def segment(rows, k):
    """(offset, stored, raw) for a stream index or a gap page."""
    if isinstance(k, dict):
        return k["offset"], k["stored"], k["raw"]
    return rows[k]["offset"], rows[k]["stored"], False


# A continuation can start with pack:/ paths or bone names inside the dictionary.
_MAP_BOUNDARIES = anim_index.STOP_SIGNATURES - {"name-table", "skeleton-names"}


def group(m, rows, *, first_row_only: bool = False, continuation_rows=None):
    """-> ([(root index, [stream indices or gap pages])], unresolved count by reason, gap pages by kind). A
    dictionary takes the page map's system page count of back-to-back streams. When the last page is not the next
    listed stream, the bytes right after the stream are tried (`_gap_page`); otherwise the root is reported.
    With first_row_only, later rows supply continuation pages but cannot start other dictionaries.
    A selected-root caller can supply continuation_rows to discover rows on demand."""
    def has_next(k):
        if k + 1 < len(rows):
            return True
        row = next(continuation_rows, None) if continuation_rows is not None else None
        if row is None:
            return False
        rows.append(row)
        return True

    out, unresolved, taken, gaps = [], collections.Counter(), set(), collections.Counter()
    for i, row in enumerate(rows[:1] if first_row_only else rows):
        if i in taken or row["signature"] != "zero-prefix":
            continue
        data = _inflate(m, row)
        if not is_root(data):
            continue
        pm = struct.unpack_from("<Q", data, 8)[0] - SYSTEM
        parts, k = [i], i
        def separate_root(page, offset):
            name_hash = _named_root_hash(page)
            if name_hash is None:
                return None  # Unnamed: use the existing page-map discovery rules.
            # Only pages already accepted for this owner may establish membership.
            # The candidate and its successors could belong to a different dictionary.
            packs = _local_packs(data)
            if packs is None and len(parts) > 1:
                prefix = data + b"".join(_inflate(m, rows[j]) for j in parts[1:])
                packs = _local_packs(prefix)
            return packs is None or (SYSTEM + offset, name_hash) not in packs

        # Only the two count bytes are needed. Locate them by inflated size,
        # including when they straddle a listed stream boundary.
        count_at = pm + 8
        counts = data[count_at:count_at + 2]
        end = len(data)
        while len(counts) < 2 and len(parts) < 255 and has_next(k):
            following = rows[k + 1]
            if not _next_to(rows[k], following) or following["signature"] in _MAP_BOUNDARIES:
                break
            start, end = end, end + following["size"]
            page = None
            if following["signature"] == "zero-prefix":
                page = _inflate(m, following)
                # Short continuation pages can resemble an unnamed root. Within
                # the original four-page window, check the root name or local map.
                if is_root(page) and (len(parts) >= 4 or any(page[0x38:0x40])
                                      or struct.unpack_from("<Q", page, 8)[0] - SYSTEM + 10 <= len(page)):
                    if separate_root(page, start) is not False:
                        break
            k += 1
            parts.append(k)
            if end > count_at:
                if page is None:
                    page = _inflate(m, following)
                counts += page[max(0, count_at - start):count_at + 2 - start]
        if len(counts) != 2:
            unresolved["page map not in the next stream"] += 1
            continue
        pages, graphics = counts
        if graphics:
            unresolved["graphics pages"] += 1
            continue
        while len(parts) < pages and has_next(k) and _next_to(rows[k], rows[k + 1]):
            following = rows[k + 1]
            if following["signature"] == "zero-prefix" and separate_root(_inflate(m, following), end):
                break
            end += following["size"]
            k += 1
            parts.append(k)
        # A listed adjacent root cannot also be an unlisted gap page.
        if len(parts) == pages - 1 and (k + 1 == len(rows) or not _next_to(rows[k], rows[k + 1])):
            joined = b"".join(_inflate(m, rows[j]) for j in parts)
            gap = _gap_page(m, rows[k], joined)
            if gap is not None:
                parts.append(gap)
        if len(parts) != pages:
            unresolved["system page not the next stream"] += 1
            continue
        if isinstance(parts[-1], dict):
            gaps[parts[-1]["kind"]] += 1
        taken.update(p for p in parts if not isinstance(p, dict))
        out.append((i, parts))
    return out, unresolved, gaps


def child_times(c):
    """A `.motion`'s +0x70 entries lined up with its children (None where they don't line up)."""
    times = c.get("times")
    return times if times is not None and len(times) == len(c["children"]) else [None] * len(c["children"])


def count_motion_blend(c, stats):
    """Counts for a `.motion`'s +0x70 child times and its parameter node (no values)."""
    times = c.get("times")
    if times is None:
        stats["motion_times_unreadable"] += 1
    else:
        stats["motion_times_one_per_child"] += len(times) == len(c["children"])
        for duration, blend, points in times:
            stats["motion_child_times"] += 1
            if len(points) == 2 and points == [(0.0, 0.0), (duration, blend)] and duration == blend:
                stats["motion_child_times_identity"] += 1
            elif (len(points) > 2 and points[0] == (0.0, 0.0) and points[-1] == (duration, blend) and
                  all(a[0] <= b[0] and a[1] <= b[1] for a, b in zip(points, points[1:]))):
                stats["motion_child_times_warped"] += 1
            else:
                stats["motion_child_times_other"] += 1
    node = c.get("node")
    if node is None:
        stats["motion_node_unreadable"] += 1
        return
    kind, axis = node
    stats[f"motion_node_type:{kind}"] += 1
    if axis is None:
        return
    count_motion_samples(c, kind, axis, stats)
    if kind != 3:
        stats[f"motion_node_axes:{kind}:{len(axis['names'])}"] += 1
        if kind != 1:
            pairs = axis["pairs"]
            stats[f"motion_node_pairs:{kind}:" + ("null" if pairs is None else str(len(pairs)))] += 1
        for h in axis["names"]:
            stats[f"motion_node_axis_name:{kind}:" + AXIS_NAMES.get(h, "unresolved")] += 1
        return
    values = [v for _i, v in axis["samples"]]
    stats["motion_axes"] += 1
    stats["motion_axis_samples"] += len(values)
    stats["motion_axis_one_name"] += len(axis["names"]) == 1
    stats["motion_axis_sorted"] += all(a <= b for a, b in zip(values, values[1:]))
    stats["motion_axis_range_is_first_last"] += bool(values) and (axis["min"], axis["max"]) == (values[0], values[-1])
    stats["motion_axis_indices_are_0_to_n"] += sorted(i for i, _v in axis["samples"]) == list(range(len(values)))
    stats["motion_axis_samples_equal_children"] += len(values) == len(c["children"] or [])
    stats[f"motion_axis_sample_count:{len(values)}"] += 1
    for h in axis["names"]:
        stats["motion_axis_name:" + AXIS_NAMES.get(h, f"0x{h:08x}")] += 1


def window_matches_duration(start, end, rate, duration):
    """True when an animation entry's clip window covers the clip exactly: (end - start) / rate equals the clip's
    +0x78 duration, to float32 precision (relative 1e-5, absolute 1e-6). False for a zero or non-finite rate."""
    if not rate or not all(math.isfinite(x) for x in (start, end, rate, duration)):
        return False
    return math.isclose((end - start) / rate, duration, rel_tol=1e-5, abs_tol=1e-6)


def count_clip_windows(clips, stats):
    """docs/animation.md: a clip plays each animation over [start, end] at its rate, and (end - start) / rate is the
    clip's duration. Count the animation entries of ordinary clips that hold to it, and the ones with rate 0."""
    for c in clips:
        if c["kind"] != "clip":
            continue
        for e in c["animations"]:
            stats["clip_windows"] += 1
            stats["clip_windows_rate_zero"] += e["rate"] == 0
            stats["clip_windows_match_duration"] += window_matches_duration(e["start"], e["end"], e["rate"],
                                                                            c["duration"])


def count_motion_samples(c, kind, axis, stats):
    """Counts for the +0x90 sample lists of a one-node `.motion` (+0x58 == 1) against its node (no values)."""
    lists = c.get("samples")
    if c["parameters"] != 1 or lists is None:
        return
    children = len(c["children"])
    mapped = [items[0][0] for items in lists if items]
    pre = f"motion_samples:{kind}:"
    stats[pre + "motions"] += 1
    stats[pre + "lists_equal_node_samples"] += len(lists) == axis["sample_count"]
    stats[pre + "lists_more_than_children"] += len(lists) > children
    stats[pre + "every_list_one_item"] += all(len(items) == 1 for items in lists)
    stats[pre + "children_in_range"] += all(ch < children for items in lists for ch, _a, _b in items)
    stats[pre + "every_child_sampled"] += {ch for items in lists for ch, _a, _b in items} == set(range(children))
    stats[pre + "sample_k_is_child_k"] += (len(lists) == children and all(len(items) == 1 for items in lists) and
                                           mapped == list(range(children)))
    for items in lists:
        for _ch, a, b in items:
            stats[pre + "items"] += 1
            stats[pre + "items_first_float_1"] += a == 1.0
            stats[pre + "items_second_float_1"] += b == 1.0


def new_stats():
    return collections.Counter()


def merge(total, part):
    """Add counters; `*_worst_1e6` entries keep the maximum."""
    for k, v in part.items():
        total[k] = max(total[k], v) if k.endswith("_worst_1e6") else total[k] + v


def decode_resource(path, segments, budget=None):
    """Worker: inflate and decode one dictionary. -> (row, stats). The row carries no values.
    With `budget` (seconds), animations left when it runs out are counted as skipped and the row says so."""
    started = time.monotonic()
    stats = new_stats()
    with open(path, "rb") as fh:
        data = b""
        for off, stored, raw in segments:
            fh.seek(off)
            data += fh.read(stored) if raw else zlib.decompressobj().decompress(fh.read(stored))
    s = System(data)
    row = {"offset": segments[0][0], "streams": len(segments), "system_bytes": len(data)}
    try:
        d = parse_dictionary(s)
    except NotClipDictionary as e:
        stats["roots_not_clip_dictionary:" + str(e)] += 1
        row["kind"] = str(e)
        return row, stats
    except (DecodeError, struct.error) as e:
        stats["dictionaries_unparsed"] += 1
        stats["dictionary_error:" + str(e).split(" at 0x")[0].split("pointer 0x")[0][:60]] += 1
        row["error"] = str(e)[:120]
        return row, stats
    stats["dictionaries"] += 1
    stats["dictionary_hash_matches_name"] += d["name"] is not None and anim_index.joaat32(d["name"]) == d["hash"]
    stats["clips"] += len(d["clips"])
    stats["clips_motion"] += sum(c["kind"] == "motion" for c in d["clips"])
    stats["clips_motionmatcher"] += sum(c["kind"] == "motionmatcher" for c in d["clips"])
    for c in d["clips"]:
        if c["kind"] == "motionmatcher":
            count_matcher(c, d["hash"], stats)
    external = set(d["external_hashes"])
    for c in d["clips"]:
        if c["kind"] != "motion":
            continue
        if c["children"] is None:
            stats["motion_children_unreadable"] += 1
            continue
        stats[f"motion_parameters:{c['parameters']}"] += 1
        stats["motion_children"] += len(c["children"])
        stats["motion_children_in_external_refs"] += sum(dh in external for _ch, dh in c["children"])
        count_motion_blend(c, stats)
    stats["clip_animation_refs"] += sum(len(c["animations"]) for c in d["clips"])
    count_clip_windows(d["clips"], stats)
    stats["packs_local"] += len(d["packs"])
    stats["packs_external"] += d["external_packs"]
    stats["dictionaries_without_local_pack"] += not d["packs"]
    ok = failed = skipped = 0
    for pack in d["packs"]:
        for anim in pack["animations"]:
            stats["animations"] += 1
            if budget is not None and time.monotonic() - started >= budget:  # >=: a coarse clock (Windows) reads 0.0 for short spans
                skipped += 1
                continue
            stats["sequences"] += len(anim["sequences"])
            for bone, ttype, fmt in anim["tracks"]:
                stats["track_type:" + TRACK_TYPES.get(ttype, f"0x{ttype:08x}")] += 1
                stats["track_format:" + TRACK_FORMATS.get(fmt, str(fmt))] += 1
            try:
                local = new_stats()
                decode_animation(s, anim, local)
                merge(stats, local)
                ok += 1
            except (DecodeError, struct.error, IndexError) as e:
                failed += 1
                stats["animation_error:" + str(e).split(" at 0x")[0][:60]] += 1
    stats["animations_decoded"] += ok
    stats["animations_failed"] += failed
    if skipped:
        stats["animations_skipped_over_budget"] += skipped
        stats["dictionaries_over_budget"] += 1
        row["over_budget"] = True
    # for the archive-wide join in scan(); scan() removes these lists from the rows it returns
    row["_clip_hashes"] = [c["hash"] for c in d["clips"] if c["kind"] == "clip"]
    row["_clip_durations"] = [c["duration"] for c in d["clips"] if c["kind"] == "clip"]
    row["_children"] = [ch for c in d["clips"] if c["kind"] == "motion" and c["children"] for ch in c["children"]]
    row["_child_durations"] = [t[0] if t else None for c in d["clips"] if c["kind"] == "motion" and c["children"]
                               for t in child_times(c)]
    row["_external"] = d["external_hashes"]
    matchers = [c["matcher"] for c in d["clips"] if c["kind"] == "motionmatcher" and c.get("matcher")]
    row["_matcher_clips"] = [ch for m in matchers for ch, _k, _i in m["clips"]]
    row["_matcher_refs"] = [r for m in matchers for r in m["refs"]]
    row["_matcher_keys"] = [h for m in matchers for h, _i in m["keys"]]
    row.update(hash=f"0x{d['hash']:08x}", name=d["name"], clips=len(d["clips"]), packs=len(d["packs"]),
               external_packs=d["external_packs"], animations=ok + failed + skipped, decoded=ok,
               seconds=round(time.monotonic() - started, 3))
    return row, stats


def join_references(rows, stats):
    """Resolve hash-only references across the archive: a root `+0x70` item with a null pointer names another
    dictionary (its joaat32 name hash), and a `.motion` child is {clip hash, dictionary hash}."""
    clips_by_dict = collections.defaultdict(set)
    duration = {}
    for r in rows:
        if "_clip_hashes" in r:
            clips_by_dict[int(r["hash"], 16)].update(r["_clip_hashes"])
            for h, t in zip(r["_clip_hashes"], r.get("_clip_durations", [])):
                duration[(int(r["hash"], 16), h)] = t
    all_clips = set().union(*clips_by_dict.values()) if clips_by_dict else set()
    ext_seen = set()
    for r in rows:
        for h in r.pop("_external", []):
            stats["external_refs_to_a_dictionary_here"] += h in clips_by_dict
            if h not in ext_seen:
                ext_seen.add(h)
                stats["external_hashes_distinct"] += 1
                stats["external_hashes_distinct_found_here"] += h in clips_by_dict
        kids = r.pop("_children", [])
        for (ch, dh), t in zip(kids, r.pop("_child_durations", [None] * len(kids))):
            if dh in clips_by_dict:
                stats["motion_children_dictionary_here"] += 1
                stats["motion_children_resolved_to_clip"] += ch in clips_by_dict[dh]
                if ch in clips_by_dict[dh] and t is not None:
                    stats["motion_child_duration_equals_clip"] += duration.get((dh, ch)) == t
        for ch in r.pop("_matcher_clips", []):
            stats["matcher_table_clips_found_here"] += ch in all_clips
        for h in r.pop("_matcher_keys", []):
            stats["matcher_key_table_hashes_that_are_clips"] += h in all_clips
        for ch, dh in r.pop("_matcher_refs", []):
            stats["matcher_refs_resolved_to_clip"] += ch in clips_by_dict.get(dh, ())
        r.pop("_clip_hashes", None)
        r.pop("_clip_durations", None)


def _decode_star(args):
    return decode_resource(*args)


def scan(path, start=None, limit=None, jobs=1, progress=None, budget=None):
    rows, walk = anim_index.scan_rpf(path, start, limit)
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        groups, unresolved, gaps = group(m, rows)
    work = [(path, [segment(rows, k) for k in parts], budget) for _i, parts in groups]
    used = [k for _i, parts in groups for k in parts if not isinstance(k, dict)]
    stats, out = new_stats(), []
    if jobs > 1 and len(work) > 1:
        import multiprocessing
        with multiprocessing.Pool(process_workers(jobs)) as pool:
            for n, (row, st) in enumerate(pool.imap_unordered(_decode_star, work, chunksize=4), 1):
                out.append(row)
                merge(stats, st)
                if progress and n % 5000 == 0:
                    progress(n, len(work))
    else:
        for n, w in enumerate(work, 1):
            row, st = decode_resource(*w)
            out.append(row)
            merge(stats, st)
            if progress and n % 5000 == 0:
                progress(n, len(work))
    out.sort(key=lambda r: r["offset"])
    join_references(out, stats)
    summary = {
        "streams": walk["files"],
        "dictionary_roots": len(groups) + sum(unresolved.values()),
        "dictionaries_grouped": len(groups),
        "unresolved": dict(unresolved),
        "gap_pages": dict(gaps),
        "pages": {str(k): v for k, v in sorted(collections.Counter(len(p) for _i, p in groups).items())},
        "dictionary_streams": len(used),
        "dictionary_stored_bytes": sum(rows[k]["stored"] for k in used),
        "dictionary_inflated_bytes": sum(rows[k]["size"] for k in used),
        "dictionary_stream_signatures": dict(collections.Counter(
            rows[k]["signature"].split(":")[0] for k in used).most_common()),
        "slowest_dictionary_seconds": max([r.get("seconds", 0) for r in out] or [0]),
        "counts": dict(sorted(stats.items())),
    }
    return out, summary


def decode_one(path, offset, frames):
    rows, _walk = anim_index.scan_rpf(path, offset, 64 << 20)
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        groups, _unresolved, _gaps = group(m, rows, first_row_only=True)
    if not groups or rows[groups[0][0]]["offset"] != offset:
        raise ValueError(f"no clip dictionary starts at 0x{offset:x}")
    with open(path, "rb") as fh:
        data = b""
        for k in groups[0][1]:
            off, stored, raw = segment(rows, k)
            fh.seek(off)
            data += fh.read(stored) if raw else zlib.decompressobj().decompress(fh.read(stored))
    s = System(data)
    d = parse_dictionary(s)
    for pack in d["packs"]:
        for anim in pack["animations"]:
            try:
                tracks, total = decode_animation(s, anim)
                anim["decoded"] = {str(k): [list(v) for v in vals[:frames]] for k, vals in tracks.items()}
                anim["decoded_frames"] = total
            except (DecodeError, struct.error, IndexError) as e:
                anim["error"] = str(e)
            anim["tracks"] = [{"bone": f"0x{b:08x}", "type": TRACK_TYPES.get(t, f"0x{t:08x}"),
                               "format": TRACK_FORMATS.get(f, f)} for b, t, f in anim["tracks"]]
            anim.pop("sequences")
    for clip in d["clips"]:
        for a in clip["animations"]:
            a.pop("animation")
    return d


def same_output_path(first, second):
    """True when writing ``second`` would truncate ``first`` (same file, path or hard link)."""
    if os.path.normcase(os.path.realpath(first)) == os.path.normcase(os.path.realpath(second)):
        return True
    try:
        return os.path.samefile(first, second)
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("archive", help="anim.rpf")
    ap.add_argument("out", nargs="?", help="output .json (counts and one row per dictionary)")
    ap.add_argument("--start", type=anim_index.offset, help="first byte to scan (default: the RPF data start)")
    ap.add_argument("--limit", type=anim_index.byte_count, help="number of bytes to scan")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes for decoding (default 1)")
    ap.add_argument("--budget", type=float, default=600.0,
                    help="seconds per dictionary; animations left after it are counted as skipped (default 600)")
    ap.add_argument("--decode", type=anim_index.offset, metavar="OFFSET",
                    help="print one dictionary (root stream offset) with decoded values to stdout")
    ap.add_argument("--frames", type=int, default=4, help="frames per track printed by --decode (default 4)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    if a.frames < 0:
        ap.error("--frames must be 0 or more")
    if math.isnan(a.budget) or a.budget < 0:
        ap.error("--budget must be 0 or a finite positive value (or infinite)")
    if a.decode is None and not a.out:
        ap.error("an output .json is required unless --decode is given")
    if a.out:
        if os.path.isdir(a.out):
            ap.error(f"{a.out} is a directory, not a .json file")
        if same_output_path(a.archive, a.out):
            ap.error("output must not overwrite the input archive")
    try:
        if a.decode is not None:
            json.dump(decode_one(a.archive, a.decode, a.frames), sys.stdout, indent=1)
            print()
            return 0

        def progress(n, total):
            if not a.quiet:
                print(f"[{n / max(1, total):6.1%}] {n} dictionaries", flush=True)
        rows, summary = scan(a.archive, a.start, a.limit, a.jobs, progress, a.budget)
    except ValueError as e:
        sys.exit(f"{a.archive}: {e}")
    except OSError as e:
        sys.exit(f"{e.filename or a.archive}: {e.strerror or e}")
    try:
        parent = os.path.dirname(os.path.abspath(a.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"summary": summary, "dictionaries": rows}, f, indent=1)
    except OSError as e:
        sys.exit(f"{a.out}: cannot write: {e.strerror or e}")
    c = summary["counts"]
    print(json.dumps({k: summary[k] for k in ("streams", "dictionary_roots", "dictionaries_grouped", "unresolved")}))
    print(json.dumps({k: c.get(k, 0) for k in ("dictionaries", "clips", "animations", "animations_decoded",
                                                "animations_failed", "sequences")}))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
