"""Index world metadata and map-placement resources in an RPF9 archive or a folder of carved files.

The table of contents of an RPF9 archive is encrypted, but entry payloads are plain zlib streams. A world
metadata resource is a stream whose first system page starts with 8 zero bytes, a ``0x5...`` pointer at +0x08,
and ``0DRP`` + the version byte ``0x7a`` at offset 0x10. Map-placement resources (the GTA V ``.ymap``
equivalent) are a subset of those: their schema contains the ``CMapData`` root (``d3593fa6``) and
``CEntityDef`` (``ce501483``) structs.

The tool only reads headers, the resource's own schema, and the scalar placement fields named in that schema.
It does not decrypt the archive table and does not extract models or textures.

Usage:
    python -I world_index.py <archive.rpf|folder> [--out FILE] [--csv FILE]
                             [--start N] [--limit BYTES] [--max-inflate BYTES]
                             [--archetypes-out FILE] [--entities-out FILE]

``--archetypes-out`` also writes the aggregate archetype file that ``world_names.py resolve --archetypes``
reads: entity count per ``archetypeName`` hash and the entity's 64-bit archetype key. Counts and hashes only.

``--entities-out`` writes one JSON line per complete ``CEntityDef`` (archetype hash and 64-bit key, position,
stored rotation, scales and LOD fields), the placement input of ``world_build.py --placements``. The rows are
derived from your own copy of the game: keep the file in your local work folder. The placement report is
replaced after the scan and other report writes succeed.

For a 76 GB archive, bound the run with --start/--limit. Coverage and the scanned window are written to the
JSON summary. Place reports outside a directory input if you want to overwrite them on a later run.
"""
import argparse
import csv
import json
import math
import mmap
import os
import re
import stat
import struct
import sys
import tempfile
import unicodedata
import zlib
from contextlib import contextmanager

SIGNATURE = b"0DRP"
VERSION = 0x7A
ENTITY = 0xCE501483
MAPDATA = 0xD3593FA6
CAP = 64 << 20
CHUNK = 1 << 20
# Field-name hashes are case-sensitive joaat of the names below, confirmed by matching the schema member
# table of every sampled map file.
FIELD = {
    "archetypeName": 0xA023A02C,
    "position": 0x01166164,
    "rotation": 0x77DD4010,
    "scaleXY": 0x9CA32637,
    "scaleZ": 0x10FB7C42,
    "flags": 0x67D26872,
    "guid": 0x9A7B6E6D,
    "lodLevel": 0x6C8F1715,
    "parentIndex": 0xD89231BD,
    "parentInstance": 0xE74C3C23,
    "numChildren": 0xA687AC89,
    "lodDist": 0xBF74DFA7,
    "childLodDist": 0xCA974BCD,
    "priorityLevel": 0x2691F019,
    "name": 0xE6E17F14,
    "name64": 0x8BFB6095,
    "parent": 0xA99C4651,
    "streamingExtentsMin": 0xDD22821F,
    "streamingExtentsMax": 0xA22EA335,
    "entitiesExtentsMin": 0x1C75BCF1,
    "entitiesExtentsMax": 0x6D074437,
    "flags_map": 0x67D26872,
    "contentFlags": 0x6A675035,
}
# Per-entity 64-bit archetype key (Murmur3-64 of the archetype name, docs/world_names.md); the engine name of
# this member is not recovered.
ARCHETYPE_KEY = 0xC8A7B90A
# CMapData.entities and the measured terrain map-reference array and entity selector (docs/world.md).
MAP_ENTITIES = 0xCCAB9307
PARENT_MAPS = 0x75D16ADF
PARENT_MAP_SLOT = 0x84799847
LOD_NAMES = {0: "HD", 1: "LOD", 2: "SLOD1", 3: "SLOD2", 4: "SLOD3", 5: "ORPHAN_HD", 6: "SLOD4"}
# Byte widths of supported member types in CEntityDef and CMapData schemas.
MEMBER_WIDTH = {
    0x10: 1, 0x11: 1, 0x60: 1,
    0x12: 2, 0x13: 2,
    0x14: 4, 0x15: 4, 0x21: 4, 0x4A: 4,
    0x08: 8, 0x4F: 8,
    0x32: 12, 0x33: 12,
    0x34: 16,
}
# Expected member types for known fields in CEntityDef and CMapData schemas.
# A resource whose schema declares a known field with a mismatched type byte is rejected.
ENTITY_MEMBER_TYPES = {
    FIELD["archetypeName"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x4A),
    ARCHETYPE_KEY: (0x08, 0x4F),
    FIELD["position"]: (0x32, 0x33),
    FIELD["rotation"]: (0x34,),
    FIELD["scaleXY"]: (0x21,),
    FIELD["scaleZ"]: (0x21,),
    FIELD["flags"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x4A),
    FIELD["guid"]: (0x08, 0x4F),
    FIELD["lodLevel"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x60),
    FIELD["parentIndex"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15),
    FIELD["parentInstance"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15),
    FIELD["numChildren"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15),
    FIELD["lodDist"]: (0x21,),
    FIELD["childLodDist"]: (0x21,),
    FIELD["priorityLevel"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x60),
    PARENT_MAP_SLOT: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15),
}
MAPDATA_MEMBER_TYPES = {
    FIELD["name"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x4A),
    FIELD["name64"]: (0x08, 0x4F),
    FIELD["parent"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x4A),
    FIELD["streamingExtentsMin"]: (0x32, 0x33),
    FIELD["streamingExtentsMax"]: (0x32, 0x33),
    FIELD["entitiesExtentsMin"]: (0x32, 0x33),
    FIELD["entitiesExtentsMax"]: (0x32, 0x33),
    FIELD["flags_map"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x4A),
    FIELD["contentFlags"]: (0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x4A),
}


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0] if 0 <= o and o + 2 <= len(b) else 0


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0] if 0 <= o and o + 4 <= len(b) else 0


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0] if 0 <= o and o + 8 <= len(b) else 0


def _f32(b, o):
    return struct.unpack_from("<f", b, o)[0] if 0 <= o and o + 4 <= len(b) else 0.0


def _ptr(value, size):
    """Resolve a 0x5xxxxxxx pointer, the only segment a single carved system stream contains."""
    if value >> 28 != 5:
        return None
    off = value & 0x0FFFFFFF
    return off if off < size else None


def _members(b, table, count, size):
    if count > 256 or (count and table is None) or (table is not None and table + 16 * count > size):
        return None
    out = {}
    if not count:
        return out
    for k in range(count):
        o = table + 16 * k
        out[_u32(b, o)] = (_u32(b, o + 4), b[o + 8], b[o + 9], _u16(b, o + 10), _u32(b, o + 12))
    return out


def _value(b, base, member):
    if member is None:
        return None
    off, ty = member[0], member[1]
    p = base + off
    if ty == 0x10:
        return struct.unpack_from("<b", b, p)[0] if p < len(b) else None
    if ty == 0x11:
        return b[p] if p < len(b) else None
    if ty == 0x12:
        return struct.unpack_from("<h", b, p)[0] if p + 2 <= len(b) else None
    if ty == 0x13:
        return _u16(b, p)
    if ty == 0x14:
        return struct.unpack_from("<i", b, p)[0] if p + 4 <= len(b) else None
    if ty in (0x15, 0x4A):
        return _u32(b, p)
    if ty == 0x21:
        return _f32(b, p)
    if ty in (0x32, 0x33):
        return [_f32(b, p + 4 * i) for i in range(3)]
    if ty == 0x34:
        return [_f32(b, p + 4 * i) for i in range(4)]
    if ty in (0x08, 0x4F):
        return _u64(b, p)
    if ty == 0x60:
        return b[p] if p < len(b) else None
    return None


def _r4(values):
    if not values:
        return None
    res = [round(v, 4) if (isinstance(v, (int, float)) and math.isfinite(v)) else None for v in values]
    return res if any(v is not None for v in res) else None


def _typed_member(schema, key, ty, width):
    size, members = schema
    m = members.get(key)
    return m if m is not None and m[1] == ty and m[0] + width <= size else None


def _placement_tables(b, blocks, map_schema, entity_schema):
    """Index encoded entity references by their original slot in the declared map root."""
    root = _u32(b, 0x20)
    if not 1 <= root <= len(blocks):
        return None, {}, ()
    tag, size, at = blocks[root - 1]
    name = _typed_member(map_schema, FIELD["name"], 0x4A, 4)
    if (tag != MAPDATA or at is None or name is None or map_schema[0] > size
            or at + map_schema[0] > len(b)):
        return None, {}, ()

    def array(key, block_type, fmt):
        member = _typed_member(map_schema, key, 0x52, 12)
        if member is None:
            return ()
        ref, count, capacity = struct.unpack_from("<QHH", b, at + member[0])
        number, offset = ref & 0xFFFFF, ref >> 20
        if not count or count > capacity or not 1 <= number <= len(blocks):
            return ()
        kind, length, base = blocks[number - 1]
        end = offset + count * struct.calcsize("<" + fmt)
        if kind != block_type or base is None or end > length or base + end > len(b):
            return ()
        return struct.unpack_from(f"<{count}{fmt}", b, base + offset)

    ordinals = {}
    for i, ref in enumerate(array(MAP_ENTITIES, 7, "Q")):
        # A repeated reference has no unique ordinal. Other classes still occupy their authored slots.
        ordinals[ref] = None if ref in ordinals else i
    parents = ()
    if (ordinals and _typed_member(entity_schema, PARENT_MAP_SLOT, 0x10, 1)
            and _typed_member(entity_schema, FIELD["parentIndex"], 0x12, 2)
            and _typed_member(entity_schema, FIELD["parentInstance"], 0x12, 2)):
        parents = array(PARENT_MAPS, 0x4A, "I")
    return f"{_u32(b, at + name[0]):08x}", ordinals, parents


def parse_resource(b, collect=None):
    """Return a row dict for a 0x7a resource, or None if the header/schema does not validate.

    When ``collect`` is given (a list, or an object with ``append`` such as ``ArchetypeCounts``), one dict per
    CEntityDef is appended to it. ``complete`` is False for an entity whose record is not entirely inside ``b``,
    including every entity of a block whose start lies past the end of ``b``. The CLI uses collectors for
    ``--archetypes-out`` counts and ``--entities-out`` placement rows.
    """
    if len(b) < 0x50 or b[0x10:0x14] != SIGNATURE or b[0x14] != VERSION:
        return None
    nst, nen, nbl = _u16(b, 0x48), _u16(b, 0x4A), _u16(b, 0x4C)
    if not (1 <= nst <= 4096 and nbl <= 4096 and nen <= 4096) or nbl == 0:
        return None
    st, it = _ptr(_u64(b, 0x28), len(b)), _ptr(_u64(b, 0x38), len(b))
    if st is None or it is None or st + 32 * nst > len(b) or it + 16 * nbl > len(b):
        return None

    structs = {}
    for i in range(nst):
        o = st + 32 * i
        name, size, count = _u32(b, o), _u32(b, o + 0x18), _u16(b, o + 0x1E)
        table = _ptr(_u64(b, o + 0x10), len(b))
        if size > (1 << 20) or count > 256:
            return None
        members = _members(b, table, count, len(b))
        if members is None:
            return None
        structs.setdefault(name, (size, members))

    if ENTITY in structs:
        esize, em = structs[ENTITY]
        for k, allowed in ENTITY_MEMBER_TYPES.items():
            if k in em:
                m = em[k]
                w = MEMBER_WIDTH.get(m[1], 0)
                if m[1] not in allowed or (w and m[0] + w > esize):
                    return None
    if MAPDATA in structs:
        msize, mm = structs[MAPDATA]
        for k, allowed in MAPDATA_MEMBER_TYPES.items():
            if k in mm:
                m = mm[k]
                w = MEMBER_WIDTH.get(m[1], 0)
                if m[1] not in allowed or (w and m[0] + w > msize):
                    return None

    blocks = []
    for i in range(nbl):
        o = it + 16 * i
        name, size, p = _u32(b, o), _u32(b, o + 4), _u64(b, o + 8)
        if size > (1 << 24):
            return None
        off = _ptr(p, len(b))
        blocks.append((name, size, off))
    if not any(name in structs and off is not None for name, _, off in blocks):
        return None

    entity_size = structs.get(ENTITY, (0, {}))[0]
    map_size = structs.get(MAPDATA, (0, {}))[0]
    row = {
        "signature": "0DRPz",
        "version": VERSION,
        "kind": "meta",
        "structs": nst,
        "enums": nen,
        "blocks": nbl,
        "root_objects": _u32(b, 0x20),
    }
    if not (entity_size and map_size):
        return row

    em = structs[ENTITY][1]
    mm = structs[MAPDATA][1]
    map_name, ordinals, parents = None, {}, ()
    if getattr(collect, "placement_ids", False):
        map_name, ordinals, parents = _placement_tables(b, blocks, structs[MAPDATA], structs[ENTITY])
    lod = {}
    archetypes = set()
    positions = []
    lod_dists = []
    for block_number, (name, size, off) in enumerate(blocks, 1):
        if name != ENTITY or entity_size == 0 or size % entity_size:
            continue
        if off is None:
            # The block starts past the bytes read (e.g. beyond --max-inflate): report its entities as incomplete.
            if collect is not None:
                for _ in range(size // entity_size):
                    collect.append({"complete": False, "archetype": None})
            continue
        for k in range(size // entity_size):
            o = off + k * entity_size
            a = _value(b, o, em.get(FIELD["archetypeName"]))
            if a is not None and isinstance(a, int):
                archetypes.add(a)
            pos = _value(b, o, em.get(FIELD["position"]))
            if isinstance(pos, list) and len(pos) == 3:
                positions.append(pos)
            level = _value(b, o, em.get(FIELD["lodLevel"]))
            if isinstance(level, int) and level in LOD_NAMES:
                lod[LOD_NAMES[level]] = lod.get(LOD_NAMES[level], 0) + 1
            dist = _value(b, o, em.get(FIELD["lodDist"]))
            if isinstance(dist, (int, float)) and math.isfinite(dist) and dist >= 0:
                lod_dists.append(dist)
            if collect is not None:
                rot = _value(b, o, em.get(FIELD["rotation"]))
                sxy = _value(b, o, em.get(FIELD["scaleXY"]))
                sz = _value(b, o, em.get(FIELD["scaleZ"]))
                cld = _value(b, o, em.get(FIELD["childLodDist"]))
                entity = {
                    "complete": o + entity_size <= len(b),
                    "archetype": a,
                    "archetype_key": _value(b, o, em.get(ARCHETYPE_KEY)),
                    "position": [v if (isinstance(v, (int, float)) and math.isfinite(v)) else None for v in pos] if pos else None,
                    "rotation": [v if (isinstance(v, (int, float)) and math.isfinite(v)) else None for v in rot] if rot else None,
                    "scale_xy": sxy if (isinstance(sxy, (int, float)) and math.isfinite(sxy)) else None,
                    "scale_z": sz if (isinstance(sz, (int, float)) and math.isfinite(sz)) else None,
                    "flags": _value(b, o, em.get(FIELD["flags"])),
                    "guid": _value(b, o, em.get(FIELD["guid"])),
                    "lod_level": level,
                    "parent_index": _value(b, o, em.get(FIELD["parentIndex"])),
                    "parent_instance": _value(b, o, em.get(FIELD["parentInstance"])),
                    "num_children": _value(b, o, em.get(FIELD["numChildren"])),
                    "lod_dist": dist if (isinstance(dist, (int, float)) and math.isfinite(dist)) else None,
                    "child_lod_dist": cld if (isinstance(cld, (int, float)) and math.isfinite(cld)) else None,
                }
                # Match a typed record start exactly, without renumbering or reordering the legacy rows.
                ordinal = ordinals.get(((k * entity_size) << 20) | block_number) if ordinals else None
                if ordinal is not None and entity["complete"]:
                    entity.update(map_name=map_name, entity_index=ordinal)
                    if parents and entity["parent_instance"] == -1 and entity["parent_index"] >= 0:
                        slot = _value(b, o, em[PARENT_MAP_SLOT])
                        if slot is not None and 0 <= slot < len(parents):
                            entity["parent_map_name"] = f"{parents[slot]:08x}"
                collect.append(entity)
    row.update({
        "kind": "map",
        "entity_size": entity_size,
        "entities": sum(size // entity_size for n, size, off in blocks
                        if n == ENTITY and off is not None and size % entity_size == 0),
        "archetypes": len(archetypes),
        "lod": lod,
        "lod_dist_min": min(lod_dists) if lod_dists else None,
        "lod_dist_max": max(lod_dists) if lod_dists else None,
    })
    if positions:
        pos_min = []
        pos_max = []
        for i in range(3):
            axis_vals = [p[i] for p in positions if isinstance(p[i], (int, float)) and math.isfinite(p[i])]
            pos_min.append(round(min(axis_vals), 4) if axis_vals else None)
            pos_max.append(round(max(axis_vals), 4) if axis_vals else None)
        if any(v is not None for v in pos_min):
            row["position_min"] = pos_min
            row["position_max"] = pos_max

    for name, size, off in blocks:
        if name != MAPDATA or off is None or map_size == 0:
            continue
        row["map_name"] = _u32(b, off + mm.get(FIELD["name"], (8,))[0])
        row["parent"] = _u32(b, off + mm.get(FIELD["parent"], (0x68,))[0])
        if FIELD["name64"] in mm:
            row["map_name64"] = _u64(b, off + mm[FIELD["name64"]][0])
        for key, field in (("extent_min", "streamingExtentsMin"), ("extent_max", "streamingExtentsMax"),
                           ("entities_min", "entitiesExtentsMin"), ("entities_max", "entitiesExtentsMax")):
            v = _value(b, off, mm.get(FIELD[field]))
            if v:
                r = _r4(v)
                if r is not None:
                    row[key] = r
        row["map_flags"] = _value(b, off, mm.get(FIELD["flags_map"]))
        row["content_flags"] = _value(b, off, mm.get(FIELD["contentFlags"]))
        break
    return row


ZHDR = re.compile(rb"\x78[\x01\x5e\x9c\xda]")


def _needs_continuation(b):
    """True when a 0x7a header's instance table or a system block extends beyond this stream."""
    if len(b) < 0x50 or b[0x10:0x14] != SIGNATURE or b[0x14] != VERSION:
        return False
    nbl = _u16(b, 0x4C)
    if nbl == 0 or nbl > 4096:
        return False
    it = _ptr(_u64(b, 0x38), len(b))
    if it is None or it + 16 * nbl > len(b):
        return True
    for i in range(nbl):
        o = it + 16 * i
        size, p = _u32(b, o + 4), _u64(b, o + 8)
        if size > (1 << 24):
            return False
        if p >> 28 == 5 and (p & 0x0FFFFFFF) + size > len(b):
            return True
    return False


def _inflate_stream(m, s, end, max_inflate, keep=False):
    """Inflate one zlib stream within m[s:end]. Returns (data, used, eof, is_meta); data is kept for 0x7a or keep=True."""
    d = zlib.decompressobj()
    chunks = []
    kept = 0
    head = bytearray()
    is_meta = False
    p = s
    while not d.eof and p < end:
        piece = m[p:min(p + CHUNK, end)]
        if not piece:
            break
        produced = d.decompress(piece)
        p += len(piece)
        if len(head) < 0x15:
            head.extend(produced[:0x15 - len(head)])
            if len(head) >= 0x15:
                is_meta = head[0x10:0x15] == b"0DRPz"
        if (is_meta or keep) and kept < max_inflate:
            chunks.append(produced)
            kept += len(produced)
        if d.unused_data or d.eof:
            break
    if not d.eof:
        return b"", 0, False, False
    return b"".join(chunks), p - s - len(d.unused_data), True, is_meta


def scan_rpf(path, start=None, limit=None, max_inflate=CAP, collect=None):
    """Yield rows for every 0x7a resource in an RPF9 data area, plus scan statistics.

    ``collect`` is passed to ``parse_resource`` once per accepted resource (after any continuation join)."""
    rows = []
    scanned = streams = 0
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        size = len(m)
        data_start = _u32(m[:8], 4)
        if start is not None and not data_start <= start < size:
            raise ValueError(f"--start 0x{start:x} is outside the data area 0x{data_start:x}..0x{size:x} of {path}")
        pos = start if start is not None else data_start
        end = min(size, pos + limit) if limit is not None else size     # --limit 0 is not "no limit"
        while pos < end:
            found = ZHDR.search(m, pos, end)
            if not found:
                break
            s = found.start()
            if s % 16:
                pos = s + 1
                continue
            try:
                data, used, eof, is_meta = _inflate_stream(m, s, end, max_inflate)
            except zlib.error:
                pos = s + 1
                continue
            if not eof:
                pos = s + 1
                continue
            consumed = 1
            tail = s + used  # end of the last joined stream; `used` leaves out the padding between streams
            if is_meta:
                # Large resources put the system page in one stream and a small continuation in the next.
                # Append contiguous zlib streams until the resource's own schema resolves.
                for _ in range(2):
                    if parse_resource(data) is not None and not _needs_continuation(data):
                        break
                    nxt = (tail + 15) & ~15
                    if nxt >= end:
                        break
                    mt = ZHDR.match(m, nxt, end)
                    if not mt:
                        break
                    try:
                        more, mused, meof, more_is_meta = _inflate_stream(m, nxt, end, max_inflate, keep=True)
                    except zlib.error:
                        break
                    if not meof or not more or more_is_meta:
                        break
                    data += more
                    used += mused
                    tail = nxt + mused
                    consumed += 1
                row = parse_resource(data, collect)
                if row is not None:
                    row["offset"] = s
                    row["size"] = len(data)
                    rows.append(row)
            streams += consumed
            scanned += used
            pos = tail
            if pos <= s:
                pos = s + 1
        return rows, {"path": os.path.abspath(path), "bytes": size,
                      "scan_start": start if start is not None else data_start,
                      "scan_end": end, "scanned_bytes": scanned, "streams": streams}


def _directory_files(path):
    def scan_failed(error: OSError) -> None:
        raise error

    # A partial walk must fail before the caller starts writing reports.
    for root, _, files in os.walk(path, onerror=scan_failed):
        for name in sorted(files):
            yield os.path.join(root, name)


def _carve_successors(folder):
    """{file: next file} for streams that were contiguous in the archive, from rpf9_carve.py's index.json in
    `folder` (each file is <offset>.bin; the next stream starts at the 16-byte boundary after `stored`). Empty
    when there is no readable index, so files are then parsed one by one."""
    try:
        with open(os.path.join(folder, "index.json"), encoding="utf-8") as f:
            entries = json.load(f).get("files", [])
        by_off = {int(e["off"]): int(e["stored"]) for e in entries}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}
    return {f"{off:012x}.bin": f"{(off + stored + 15) & ~15:012x}.bin"
            for off, stored in by_off.items() if (off + stored + 15) & ~15 in by_off}


def _read_carved(path, max_inflate):
    """Read a carved file within the cap, or return None if it is too large."""
    with open(path, "rb") as f:
        if os.fstat(f.fileno()).st_size > max_inflate:
            return None
        data = f.read(max_inflate + 1)  # detect growth after fstat
    return data if len(data) <= max_inflate else None


def scan_path(path, start=None, limit=None, max_inflate=CAP, collect=None, *, directory_files=None):
    """Scan an RPF9 archive, or every carved file in a directory."""
    if directory_files is not None or os.path.isdir(path):
        rows, stats = [], {"path": os.path.abspath(path), "files": 0, "scanned_bytes": 0}
        if directory_files is None:
            directory_files = _directory_files(path)
        directory_files = list(directory_files)
        listed = set(directory_files)
        successors = {}   # folder -> _carve_successors(folder)
        consumed = set()
        for p in directory_files:
            if p in consumed:
                continue
            with open(p, "rb") as f:
                head = f.read(4)
            if head == b"9FPR":
                more, sub = scan_rpf(p, start, limit, max_inflate, collect)
                rows.extend(more)
                stats["files"] += 1
                stats["scanned_bytes"] += sub["scanned_bytes"]
                continue
            b = _read_carved(p, max_inflate)
            if b is None:
                continue
            stats["files"] += 1
            stats["scanned_bytes"] += len(b)
            # As in scan_rpf: a 0DRPz resource that needs a continuation takes the next stream, when the
            # carve's index.json shows that stream started right after this one.
            root = os.path.dirname(p)
            if root not in successors:
                successors[root] = _carve_successors(root)
            nxt = successors[root]
            current = os.path.basename(p)
            for _ in range(2):
                if b[0x10:0x15] != b"0DRPz" or (parse_resource(b) is not None and not _needs_continuation(b)):
                    break
                follow = nxt.get(current)
                q = os.path.join(root, follow) if follow is not None else None
                if q is None or q not in listed:
                    break
                if len(b) + os.path.getsize(q) > max_inflate:
                    break
                with open(q, "rb") as f:
                    more = f.read()
                if more[0x10:0x15] == b"0DRPz":
                    break
                b += more
                consumed.add(q)
                stats["files"] += 1
                stats["scanned_bytes"] += len(more)
                current = follow
            row = parse_resource(b, collect)   # collect once, on the joined resource
            if row is not None:
                row["offset"] = 0
                row["size"] = len(b)
                row["file"] = os.path.relpath(p, path).replace("\\", "/")
                rows.append(row)
        return rows, stats
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"9FPR":
        return scan_rpf(path, start, limit, max_inflate, collect)
    b = _read_carved(path, max_inflate)
    if b is None:
        return [], {"path": os.path.abspath(path), "files": 0, "scanned_bytes": 0}
    row = parse_resource(b, collect)
    return ([dict(row, offset=0, size=len(b), file=os.path.basename(path))] if row else [],
            {"path": os.path.abspath(path), "files": 1, "scanned_bytes": len(b)})


class ArchetypeCounts:
    """``collect`` sink for parse_resource: entities per archetype hash and the archetype's 64-bit key.

    Entities whose record extends past the inflated bytes are counted as ``incomplete`` and not aggregated. A hash
    seen with two different 64-bit keys gets no key in the output (it can then resolve but not verify) and is
    listed under ``conflicting_keys``."""

    def __init__(self):
        self.counts = {}
        self.keys = {}
        self.incomplete = 0

    def append(self, entity):
        if not entity.get("complete", True):
            self.incomplete += 1
            return
        a = entity.get("archetype")
        if a is None:
            return
        self.counts[a] = self.counts.get(a, 0) + 1
        k = entity.get("archetype_key")
        if k:
            self.keys.setdefault(a, set()).add(k)

    def to_json(self):
        single = {a: next(iter(k)) for a, k in self.keys.items() if len(k) == 1}
        conflicts = {f"{a:08x}": sorted(f"{x:016x}" for x in k) for a, k in sorted(self.keys.items()) if len(k) > 1}
        return {"archetypes": {f"{a:08x}": n for a, n in sorted(self.counts.items())},
                "keys": {f"{a:08x}": f"{k:016x}" for a, k in sorted(single.items())},
                "conflicting_keys": conflicts,
                "summary": {"entities": sum(self.counts.values()), "archetypes": len(self.counts),
                            "with_key": len(single), "key_conflicts": len(conflicts),
                            "incomplete_entities": self.incomplete}}


class EntityRows:
    """A ``parse_resource`` collector that writes one JSON line per complete entity (``--entities-out``)."""

    FIELDS = ("position", "rotation", "scale_xy", "scale_z", "lod_level", "parent_index", "num_children",
              "lod_dist", "child_lod_dist")
    placement_ids = True

    def __init__(self, handle):
        self.handle = handle
        self.written = 0
        self.incomplete = 0

    def append(self, entity):
        if not entity.get("complete", True) or entity.get("archetype") is None:
            self.incomplete += 1
            return
        row = {"archetype": f"{entity['archetype']:08x}"}
        if entity.get("archetype_key"):
            row["key"] = f"{entity['archetype_key']:016x}"
        for k in self.FIELDS:
            row[k] = entity.get(k)
        for k in ("map_name", "entity_index", "parent_map_name"):
            if k in entity:
                row[k] = entity[k]
        self.handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
        self.written += 1


class _Both:
    """Feed one entity stream to two collectors."""

    def __init__(self, *sinks):
        self.sinks = [x for x in sinks if x is not None]
        self.placement_ids = any(getattr(x, "placement_ids", False) for x in self.sinks)

    def append(self, entity):
        for x in self.sinks:
            x.append(entity)


def _count(minimum):
    """An argparse type for a whole number (decimal or 0x hex) of at least `minimum`."""
    def parse(text):
        try:
            value = int(text, 0)
        except ValueError:
            raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}")
        return value
    return parse


def _file_identity(path):
    try:
        stat = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    return stat.st_dev, stat.st_ino


def _resolve_report_path(path):
    if os.name != "nt":
        return os.path.realpath(path)
    if not path.startswith("\\\\?\\"):
        # GetFullPathName normalizes ordinary Windows paths the same way opening them does.
        return os.path.realpath(os.path.abspath(path))

    # realpath can trim literal trailing dots and spaces from missing verbatim components.
    # Resolve the existing ancestor and append the missing components as written.
    missing = []
    while not os.path.lexists(path):
        parent, name = os.path.split(path)
        if not name:
            break
        missing.append(name)
        path = parent
    return os.path.join(os.path.realpath(path), *reversed(missing))


_PC_CASE_SENSITIVE = 11  # <sys/unistd.h> on macOS; os.pathconf_names doesn't list it


def _case_insensitive_volume(path):
    """True when the volume that will hold `path` ignores letter case, as macOS's default APFS volume does.
    os.path.normcase folds case only on Windows, so on macOS `out/R.json` and `out/r.json` would pass as two
    reports and the second write would replace the first."""
    if sys.platform != "darwin":
        return False
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    try:
        return os.pathconf(path, _PC_CASE_SENSITIVE) == 0
    except (OSError, ValueError):
        return True  # the macOS default; folding can only refuse a pair, never overwrite one


def _pending_path_key(path):
    path = os.path.normcase(_resolve_report_path(path))
    if os.name != "nt":
        if _case_insensitive_volume(path):
            # APFS and HFS+ also treat composed and decomposed accents as one name.
            path = unicodedata.normalize("NFD", path.casefold())
        return path

    # Drive and UNC paths can also be spelled using the extended-length prefix.
    extended_prefix = "\\\\?\\"
    unc_prefix = extended_prefix + "unc\\"
    if path.startswith(unc_prefix):
        return "\\\\" + path[len(unc_prefix):]
    drive, _ = os.path.splitdrive(path)
    if drive.startswith(extended_prefix) and len(drive) == len(extended_prefix) + 2 and drive.endswith(":"):
        return path[len(extended_prefix):]
    return path


def _check_report_paths(parser, reports, source, directory_files):
    existing, pending = {}, {}
    for flag, path in reports:
        if not path:
            continue
        key = _file_identity(path)
        if key is None:
            key = _pending_path_key(path)
            destinations = pending
        else:
            destinations = existing
        if other := destinations.get(key):
            parser.error(f"{other} and {flag} must use different paths")
        destinations[key] = flag

    source_id = _file_identity(source)
    if source_id is None:
        flag = pending.get(_pending_path_key(source))
    else:
        flag = existing.get(source_id)
    if flag:
        parser.error(f"{flag} would overwrite the input {source}")

    if existing and directory_files is not None:
        for path in directory_files:
            if flag := existing.get(_file_identity(path)):
                parser.error(f"{flag} would overwrite the input {path}")


@contextmanager
def _staged_entity_report(path):
    if not path:
        yield None
        return

    destination = _resolve_report_path(path)
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    try:
        mode = os.stat(destination).st_mode
    except FileNotFoundError:
        mode = None
    if mode is not None and not stat.S_ISREG(mode):
        raise OSError(f"placement report destination is not a regular file: {path}")

    # mkdtemp may normalize the returned parent on Windows. Keep the spelling used to create it.
    directory = os.path.join(parent, os.path.basename(tempfile.mkdtemp(dir=parent, prefix=".world-index-")))
    staged = os.path.join(directory, "entities.jsonl")
    try:
        with open(staged, "w", encoding="utf-8", newline="\n") as handle:
            yield EntityRows(handle)
        if mode is not None:
            os.chmod(staged, mode & 0o777)
        os.replace(staged, destination)
    finally:
        if os.path.exists(staged):
            if os.name == "nt":
                os.chmod(staged, stat.S_IWRITE)
            os.unlink(staged)
        os.rmdir(directory)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="RPF9 archive, one carved file, or a directory of carved files")
    ap.add_argument("--out", default=os.path.join("reports", "world_index.json"),
                    help="JSON output (default: reports/world_index.json)")
    ap.add_argument("--csv", help="optional CSV output")
    ap.add_argument("--start", type=_count(0), help="first byte of an RPF9 data-area window")
    ap.add_argument("--limit", type=_count(1), help="window size in bytes (default: to end of archive)")
    ap.add_argument("--max-inflate", type=_count(1), default=CAP,
                    help="stop keeping bytes above this per resource, default 64 MiB")
    ap.add_argument("--archetypes-out",
                    help="also write entity counts and 64-bit keys per archetype hash (world_names resolve input)")
    ap.add_argument("--entities-out",
                    help="also write one JSON line per complete entity (world_build --placements input; local only)")
    a = ap.parse_args(argv)
    reports = (("--out", a.out), ("--csv", a.csv), ("--archetypes-out", a.archetypes_out),
               ("--entities-out", a.entities_out))
    agg = ArchetypeCounts() if a.archetypes_out else None
    try:
        # Check the same inputs the directory scan will read, before the streaming report exists.
        directory_files = list(_directory_files(a.path)) if os.path.isdir(a.path) else None
        _check_report_paths(ap, reports, a.path, directory_files)
        with _staged_entity_report(a.entities_out) as ent:
            sink = _Both(agg, ent) if agg is not None and ent is not None else (agg if agg is not None else ent)
            rows, stats = scan_path(a.path, a.start, a.limit, a.max_inflate, sink, directory_files=directory_files)
            summary = _write_reports(a, rows, stats, agg)
            if ent is not None:
                summary["entity_rows"] = {"written": ent.written, "incomplete": ent.incomplete}
    except (OSError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=1, allow_nan=False))
    return 0


def _write_reports(a, rows, stats, agg):
    summary = dict(stats)
    summary["resources"] = len(rows)
    summary["maps"] = sum(1 for r in rows if r["kind"] == "map")
    summary["meta"] = sum(1 for r in rows if r["kind"] == "meta")
    summary["entities"] = sum(r.get("entities") or 0 for r in rows)
    summary["versions"] = sorted({r["version"] for r in rows})
    doc = {"summary": summary, "resources": rows}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, indent=1, sort_keys=True, allow_nan=False)
    if a.csv:
        cols = ["source", "offset", "size", "signature", "version", "kind", "structs", "enums", "blocks",
                "root_objects", "entities", "entity_size", "archetypes", "map_name", "map_name64", "parent",
                "map_flags",
                "content_flags", "extent_min", "extent_max", "entities_min", "entities_max", "position_min",
                "position_max", "lod", "lod_dist_min", "lod_dist_max"]
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
            w.writeheader()
            for r in rows:
                r = dict(r)
                r["source"] = r.get("file") or r.get("path") or os.path.basename(a.path)
                for k, v in list(r.items()):
                    if isinstance(v, (list, dict)):
                        r[k] = json.dumps(v, sort_keys=True, allow_nan=False)
                w.writerow(r)
    if agg is not None:
        os.makedirs(os.path.dirname(os.path.abspath(a.archetypes_out)), exist_ok=True)
        with open(a.archetypes_out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(agg.to_json(), f, indent=1, sort_keys=True, allow_nan=False)
        summary["archetype_file"] = agg.to_json()["summary"]
    return summary


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
