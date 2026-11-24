"""Resolve the interior keys of the location-name table to interior archetypes and their placements.

``GameShape__Data__LocationNameLoopkup`` (``common.rpf``, see docs/world.md) lists ``InteriorNames/Item key=...``
with one or more inner items, each a display-name key and a world position. This tool tests what a key names:

- every ``CMloArchetypeDef`` (10506455) in ``levels.rpf`` is read with its ``name`` hash, its 64-bit keys (the
  ``0x4f`` members), its box and its rooms (``CMloRoomDef`` 0b1811f1);
- every ``CMloInstanceDef`` (09cc28be), the record that places an interior in a map, is read with its
  ``archetypeName``, position, rotation and scale; ``world_index.py`` reads only ``CEntityDef``, so interiors
  never appear in its entity rows;
- each key is hashed (lower-cased joaat) and looked up among the interior names; a match counts as confirmed
  only when the lower-cased key's MurmurHash3 x64 low 64 equals one of that interior's 64-bit keys;
- each position is compared with the placements of its own key's interior: how many placements there are, and
  how far the position lies from the placed interior box (0 = inside), with the stored and the conjugated
  quaternion. The interior and room boxes are scaled by each placement's ``scaleXY`` (X, Y) and ``scaleZ`` (Z)
  first, so distances stay in metres;
- the rooms are also counted for two checks: how many interiors have ``limbo`` as their first room, and how many
  keys equal the hash member ``a261e0a9`` of some room.

Every field is found by the case-sensitive joaat of its name in the resource's member table.

Usage:
    python -I world_interior_names.py <levels.rpf|carved folder> --names <LocationNameLoopkup JSON>
                                      [--start N] [--limit BYTES] [--jobs N] [--out FILE]

``--names`` is the JSON file ``gamedata_reader.py --root GameShape__Data__LocationNameLoopkup --out DIR`` writes
for ``common.rpf``. The printed summary holds counts only. ``--out`` also lists every key's hash, match and
distances: game data, keep it local.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import struct
import sys
from concurrent.futures import ProcessPoolExecutor

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import world_archetypes as WA  # noqa: E402
import world_index as WI  # noqa: E402
from pool_workers import process_workers  # noqa: E402
from report_paths import same_report_path  # noqa: E402
from world_names import hash64, joaat  # noqa: E402

J = WA.joaat_cs
MLO = J("CMloArchetypeDef")
ROOM = J("CMloRoomDef")
INSTANCE = J("CMloInstanceDef")
NAME, BBMIN, BBMAX = J("name"), J("bbMin"), J("bbMax")
U64, ARRAY = 0x4F, 0x52
LIMBO = joaat("limbo")
ROOM_KEY = 0xA261E0A9  # a room's hash member that holds an interior key's joaat
OVERLAP = 64 << 20  # a worker reads this far past its window so a resource that straddles the edge is whole
THRESHOLDS = (0, 1, 2, 5, 10, 25, 100)
# Metres. A point on an interior or room face stays inside despite the rounding of the inverse rotation (about
# 1e-17 m with a float32 quaternion). Positions are float32, which near the 13 km map edge resolve to about 1 mm,
# so any tolerance far below that decides exactly as exact arithmetic would.
EPS = 1e-6


def _vec3(b, base, member):
    if member is None:
        return None
    p = base + member[0]
    return list(struct.unpack_from("<3f", b, p)) if p + 12 <= len(b) else None


def _instances(b, structs, blocks, h):
    if h not in structs:
        return
    ssize, members = structs[h]
    for name, size, off in blocks:
        if name != h or off is None or not ssize or size % ssize or off + size > len(b):
            continue
        for k in range(size // ssize):
            yield off + k * ssize, members


def _array(b, blocks, base, member):
    """(block hash, first byte, count) of an array member: a u64 (low 12 bits: 1-based block index, bits 20+:
    byte offset in that block) and a u16 count at +8, as docs/world.md describes for instance lists."""
    p = base + member[0]
    if p + 10 > len(b):
        return None
    ref, count = struct.unpack_from("<Q", b, p)[0], struct.unpack_from("<H", b, p + 8)[0]
    index = ref & 0xFFF
    if not count or not 1 <= index <= len(blocks):
        return None
    h, size, off = blocks[index - 1]
    start = ref >> 20
    if off is None or start >= size:
        return None
    return h, off + start, count


def _rooms(b, structs, blocks, base, members):
    """[[name hash, bbMin, bbMax, a261e0a9 hash]] of the CMloRoomDef array an interior points at."""
    if ROOM not in structs:
        return []
    rsize, rm = structs[ROOM]
    for mm in members.values():
        if mm[1] != ARRAY:
            continue
        found = _array(b, blocks, base, mm)
        if found and found[0] == ROOM and rsize:
            h, first, count = found
            return [[WI._value(b, o, rm.get(NAME)), _vec3(b, o, rm.get(BBMIN)), _vec3(b, o, rm.get(BBMAX)),
                     WI._value(b, o, rm.get(ROOM_KEY))]
                    for o in range(first, first + rsize * count, rsize) if o + rsize <= len(b)]
    return []


def parse(b):
    """(interiors, placements) of one validated 0x7a resource."""
    if WI.parse_resource(b) is None:
        return [], []
    structs, blocks = WA.schema(b)
    interiors = []
    for base, m in _instances(b, structs, blocks, MLO):
        keys = {"%08x" % mh: struct.unpack_from("<Q", b, base + mm[0])[0]
                for mh, mm in m.items() if mm[1] == U64 and base + mm[0] + 8 <= len(b)}
        interiors.append({"name": WI._value(b, base, m.get(NAME)), "keys": keys,
                          "bb_min": _vec3(b, base, m.get(BBMIN)), "bb_max": _vec3(b, base, m.get(BBMAX)),
                          "rooms": _rooms(b, structs, blocks, base, m)})
    placements = []
    for base, m in _instances(b, structs, blocks, INSTANCE):
        placements.append({"archetype": WI._value(b, base, m.get(WI.FIELD["archetypeName"])),
                           "position": WI._value(b, base, m.get(WI.FIELD["position"])),
                           "rotation": WI._value(b, base, m.get(WI.FIELD["rotation"])),
                           "scale_xy": WI._value(b, base, m.get(WI.FIELD["scaleXY"])),
                           "scale_z": WI._value(b, base, m.get(WI.FIELD["scaleZ"]))})
    return interiors, placements


def scan_window(job):
    """Resources whose stream starts in [start, end) of an RPF9 archive (read up to end + OVERLAP)."""
    path, start, end, size = job
    out = []
    limit = min(size, end + OVERLAP) - start
    for row, data in WA.iter_resources(path, start, limit):
        if row["offset"] < end:
            out.append((row["offset"], parse(data)))
    return out


def start_error(path, start):
    """Why --start cannot be used on ``path``, or None. Checked before scanning, so --jobs > 1 also gives one line
    instead of a traceback from a worker; a carved file or folder has no data area and ignores --start."""
    if start is None or not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        magic = f.read(8)
    if magic[:4] != b"9FPR" or len(magic) < 8:
        return None
    data_start, size = WI._u32(magic, 4), os.path.getsize(path)
    if not data_start <= start < size:
        return f"--start 0x{start:x} is outside the data area 0x{data_start:x}..0x{size:x} of {path}"
    return None


def scan(path, start=None, limit=None, jobs=1):
    """{offset or file: (interiors, placements)} for an RPF9 archive window or a carved file/folder."""
    magic = b""
    if os.path.isfile(path):
        with open(path, "rb") as f:
            magic = f.read(8)
    if magic[:4] != b"9FPR":
        return {row.get("file", i): parse(data) for i, (row, data) in enumerate(WA.iter_resources(path, start, limit))}
    size = os.path.getsize(path)
    lo = start if start is not None else WI._u32(magic, 4)
    hi = min(size, lo + limit) if limit is not None else size
    n = process_workers(max(1, jobs))  # one window per worker: a window past the cap only re-reads OVERLAP
    step = max(1, -(-(hi - lo) // n))
    windows = [(path, s, min(hi, s + step), size) for s in range(lo, hi, step)]
    found = {}
    if n == 1:
        results = [scan_window(w) for w in windows]
    else:
        with ProcessPoolExecutor(max_workers=n) as ex:
            results = list(ex.map(scan_window, windows))
    for rows in results:
        for off, parsed in rows:
            found[off] = parsed
    return found


def _pairs(pairs):
    out = {}
    for k, v in pairs:
        if k in out:  # a repeated key keeps all its items and is counted, not silently replaced
            out.setdefault("\0duplicates", []).append(k)
            prev = out[k] if isinstance(out[k], list) else [out[k]]
            out[k] = prev + (v if isinstance(v, list) else [v])
        else:
            out[k] = v
    return out


def _position(pos):
    """``pos`` as three floats, or None when it is not a list of three finite numbers."""
    if not isinstance(pos, list) or len(pos) != 3:
        return None
    if not all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in pos):
        return None
    try:
        out = [float(c) for c in pos]  # an integer past float range (over 308 digits) raises OverflowError
    except OverflowError:
        return None
    return out if all(math.isfinite(c) for c in out) else None


def _json_type(v):
    if v is None:
        return "null"
    if isinstance(v, bool):  # before the number test: bool is an int in Python
        return "a boolean"
    return ("a list" if isinstance(v, list) else "a string" if isinstance(v, str)
            else "a number" if isinstance(v, (int, float)) else "an object")


def load_names(path):
    """({key: [position]}, repeated keys in the InteriorNames section, positions that are not three numbers).

    Raises ValueError (with OSError and json.JSONDecodeError, which is one) when the file is not such a table."""
    with open(path, encoding="utf-8") as f:
        doc = json.load(f, object_pairs_hook=_pairs)
    if not isinstance(doc, dict):
        raise ValueError(f"expected a JSON object, found {_json_type(doc)}")
    section = doc.get("InteriorNames", doc)
    if not isinstance(section, dict):
        raise ValueError(f"InteriorNames is {_json_type(section)}, not an object")
    duplicates = len(section.pop("\0duplicates", []))
    out, malformed = {}, 0
    for key, items in section.items():
        items = items if isinstance(items, list) else [items]
        out[key] = []
        for it in items:
            if not isinstance(it, dict) or "Pos" not in it:
                continue
            pos = _position(it["Pos"])
            if pos is None:
                malformed += 1
            else:
                out[key].append(pos)
    return out, duplicates, malformed


def _local(q, p, r):
    d = [q[i] - p[i] for i in range(3)]
    return [sum(r[j][i] * d[j] for j in range(3)) for i in range(3)]  # R^T (q - p)


def _outside(v, lo, hi):
    return math.sqrt(sum(max(lo[i] - v[i], 0.0, v[i] - hi[i]) ** 2 for i in range(3)))


def _scaled(lo, hi, p):
    """Box [lo, hi] scaled by placement p (scaleXY on X and Y, scaleZ on Z; a missing or zero scale is 1)."""
    sxy, sz = p.get("scale_xy") or 1.0, p.get("scale_z") or 1.0
    s = (sxy, sxy, sz)
    a, b = [lo[i] * s[i] for i in range(3)], [hi[i] * s[i] for i in range(3)]
    return [min(a[i], b[i]) for i in range(3)], [max(a[i], b[i]) for i in range(3)]


def _median(values):
    """Median of an already sorted list."""
    n = len(values)
    if not n:
        return None
    m = n // 2
    return values[m] if n % 2 else (values[m - 1] + values[m]) / 2


def _buckets(values):
    values = sorted(values)
    median = _median(values)
    return {"n": len(values), "median_m": round(median, 3) if median is not None else None,
            "within_m": {str(t): bisect.bisect_right(values, t) for t in THRESHOLDS}}


def resolve(names, found, duplicate_keys=0, malformed_positions=0):
    interiors, placements = {}, {}
    n_interiors = n_placements = duplicate_interiors = 0
    for parsed_interiors, parsed_placements in found.values():
        for d in parsed_interiors:
            n_interiors += 1
            if d["name"] in interiors:  # the first definition is used; the others are counted
                duplicate_interiors += 1
            interiors.setdefault(d["name"], d)
        for p in parsed_placements:
            n_placements += 1
            placements.setdefault(p["archetype"], []).append(p)
    room_keys = {r[3] for d in interiors.values() for r in d["rooms"] if r[3]}
    summary = {"interior_definitions": n_interiors, "interior_names": len(interiors),
               "duplicate_interior_definitions": duplicate_interiors,
               "interiors_with_rooms": sum(1 for d in interiors.values() if d["rooms"]),
               "interiors_first_room_limbo": sum(1 for d in interiors.values()
                                                 if d["rooms"] and d["rooms"][0][0] == LIMBO),
               "placements": n_placements,
               "placed_interior_names": sum(1 for k in placements if k in interiors),
               "placed_unknown_names": sum(1 for k in placements if k not in interiors),
               "placements_of_unknown_interior": sum(len(v) for k, v in placements.items() if k not in interiors),
               "keys": len(names), "duplicate_keys": duplicate_keys, "positions": sum(map(len, names.values())),
               "malformed_positions": malformed_positions, "keys_without_a_placement_position": 0,
               "keys_in_a_room_key_member": sum(1 for k in names if joaat(k) in room_keys),
               "keys_naming_an_interior": 0, "confirmed_by_64bit_key": {}, "keys_not_confirmed": 0,
               "keys_with_placements": 0, "positions_vs_placements": {"equal": 0, "more_positions": 0,
                                                                      "fewer_positions": 0},
               "positions_in_a_room_box": {"stored": 0, "conjugate": 0}}
    rows, dist = {}, {"stored": [], "conjugate": []}
    for key, positions in names.items():
        h = joaat(key)
        row = rows[key] = {"hash": "%08x" % h, "positions": len(positions)}
        d = interiors.get(h)
        if d is None:
            continue
        summary["keys_naming_an_interior"] += 1
        h64 = hash64(key.lower())
        member = next((mh for mh, v in d["keys"].items() if v == h64), None)
        row["confirmed_by"] = member
        if member is None:
            summary["keys_not_confirmed"] += 1
        else:
            c = summary["confirmed_by_64bit_key"]
            c[member] = c.get(member, 0) + 1
        pl = placements.get(h, [])
        row["placements"] = len(pl)
        if not pl:
            continue
        summary["keys_with_placements"] += 1
        pv = summary["positions_vs_placements"]
        pv["equal" if len(positions) == len(pl) else "more_positions" if len(positions) > len(pl)
           else "fewer_positions"] += 1
        positioned = [p for p in pl if p["position"] is not None and len(p["position"]) == 3]
        if not positioned:  # no placement to measure from; counted instead of failing the whole run
            summary["keys_without_a_placement_position"] += 1
            continue
        lo, hi = d["bb_min"], d["bb_max"]
        if lo is None or hi is None:
            continue
        for tag, conj in (("stored", False), ("conjugate", True)):
            placed = [(p, WA._rotation(p["rotation"], conj)) for p in positioned]
            out = []
            for q in positions:
                locs = [(_local(q, p["position"], r), p) for p, r in placed]
                gap = min(_outside(v, *_scaled(lo, hi, p)) for v, p in locs)
                out.append(gap if gap > EPS else 0.0)
                if any(rn != LIMBO and rlo and rhi and _outside(v, *_scaled(rlo, rhi, p)) <= EPS
                       for v, p in locs for rn, rlo, rhi, _ in d["rooms"]):
                    summary["positions_in_a_room_box"][tag] += 1
            row["outside_box_m_" + tag] = [round(x, 3) for x in out]
            dist[tag].extend(out)
    summary["distance_outside_own_interior_box"] = {k: _buckets(v) for k, v in dist.items()}
    return summary, rows


def _inside(folder, path):
    """True when ``folder`` is a folder and ``path`` lies in it (a report there would be read as a carved file)."""
    if not os.path.isdir(folder):
        return False
    root = os.path.normcase(os.path.realpath(folder))
    try:
        return os.path.commonpath([root, os.path.normcase(os.path.realpath(path))]) == root
    except ValueError:  # different drives on Windows
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="levels.rpf, one carved 0x7a file, or a folder of carved files")
    ap.add_argument("--names", required=True, help="GameShape__Data__LocationNameLoopkup JSON (gamedata_reader.py)")
    ap.add_argument("--start", type=int, help="first byte of an RPF9 data-area window")
    ap.add_argument("--limit", type=int, help="window size in bytes (default: to the end of the archive)")
    ap.add_argument("--jobs", type=int, default=1, help="parallel windows over an RPF9 archive (default 1)")
    ap.add_argument("--out", default=os.path.join("reports", "world_interior_names.json"))
    a = ap.parse_args(argv)
    for p in (a.path, a.names):
        if not os.path.exists(p):
            ap.error("no such file or folder: %s" % p)
    if a.jobs < 1:
        ap.error("--jobs must be at least 1")
    if a.start is not None and a.start < 0:
        ap.error("--start must not be negative")
    if a.limit is not None and a.limit < 1:
        ap.error("--limit must be at least 1 byte")  # 0 would scan nothing and report every count as 0
    out_abs = os.path.abspath(a.out)
    if same_report_path(a.out, a.path) or same_report_path(a.out, a.names) or _inside(a.path, out_abs):
        print("world_interior_names.py: --out would overwrite an input: %s" % a.out, file=sys.stderr)
        return 2
    if os.path.isdir(a.out):
        print("world_interior_names.py: --out is a directory: %s" % a.out, file=sys.stderr)
        return 2
    try:
        names, duplicate_keys, malformed = load_names(a.names)
    except (OSError, ValueError) as exc:  # json.JSONDecodeError is a ValueError
        why = exc.strerror if isinstance(exc, OSError) and exc.strerror else exc
        print("world_interior_names.py: --names %s: %s" % (a.names, why), file=sys.stderr)
        return 2
    why = start_error(a.path, a.start)
    if why:
        print("world_interior_names.py: %s" % why, file=sys.stderr)
        return 2
    summary, rows = resolve(names, scan(a.path, a.start, a.limit, a.jobs), duplicate_keys, malformed)
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"summary": summary, "keys": rows}, f, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
