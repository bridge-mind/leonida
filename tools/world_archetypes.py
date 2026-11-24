"""Read archetype definitions (CMapTypes / CBaseArchetypeDef) from 0x7a resources, and check map extents.

A ``CMapTypes`` resource (the ``.ytyp`` role) is a 0x7a metadata resource like a map file. Its archetype structs
(``CBaseArchetypeDef`` 82d6fc83, ``CTimeArchetypeDef`` 76b0c56c, ``CMloArchetypeDef`` 10506455) are stored in
blocks of fixed-size instances, the way ``world_index.py`` reads ``CEntityDef``. Every field is found by the
case-sensitive joaat of its name in the resource's own member table; no offset is assumed.

``--check-extents`` also re-reads the window's map files and, for every map whose entities all have a definition
in the window, compares the union of the archetype boxes transformed by each entity with the map's stored
``entitiesExtentsMin``/``Max``: once with the stored quaternion and once with its conjugate, and once with and
once without the entity scale. See docs/world.md, "Archetype definitions".

Usage:
    python -I world_archetypes.py <archive.rpf|carved file|folder> [--start N] [--limit BYTES]
                                  [--out FILE] [--check-extents]

Only hashes, bounds and scalar fields are written; no model, texture or payload.
"""
from __future__ import annotations

import argparse
import json
import math
import mmap
import os
import statistics
import struct
import sys

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

import world_census as WC  # noqa: E402
from report_paths import same_report_path  # noqa: E402
import world_index as WI  # noqa: E402


def joaat_cs(s):
    """Case-sensitive joaat (schema struct and member names; world_index.FIELD uses the same function)."""
    h = 0
    for c in s.encode("latin-1"):
        h = (h + c) & 0xFFFFFFFF
        h = (h + (h << 10)) & 0xFFFFFFFF
        h ^= h >> 6
    h = (h + (h << 3)) & 0xFFFFFFFF
    h ^= h >> 11
    return (h + (h << 15)) & 0xFFFFFFFF


MAPTYPES = joaat_cs("CMapTypes")
KINDS = {joaat_cs("CBaseArchetypeDef"): "base", joaat_cs("CTimeArchetypeDef"): "time",
         joaat_cs("CMloArchetypeDef"): "mlo"}
FIELDS = {key: joaat_cs(name) for key, name in (
    ("name", "name"), ("asset_name", "assetName"), ("asset_type", "assetType"), ("lod_dist", "lodDist"),
    ("hd_texture_dist", "hdTextureDist"), ("special_attribute", "specialAttribute"), ("flags", "flags"),
    ("bb_min", "bbMin"), ("bb_max", "bbMax"), ("bs_centre", "bsCentre"), ("bs_radius", "bsRadius"),
    ("physics_dictionary", "physicsDictionary"), ("clip_dictionary", "clipDictionary"),
    ("time_flags", "timeFlags"))}
ENUM = 0x62  # enum member (u12 = enum-name hash); 4 bytes in every measured schema
TOLERANCE = 0.05


def schema(b):
    """Return ({struct hash: (size, members)}, [(block struct hash, size, system offset or None)])."""
    n = len(b)
    st, it = WI._ptr(WI._u64(b, 0x28), n), WI._ptr(WI._u64(b, 0x38), n)
    structs, blocks = {}, []
    if st is None or it is None:
        return structs, blocks
    for i in range(WI._u16(b, 0x48)):
        o = st + 32 * i
        members = WI._members(b, WI._ptr(WI._u64(b, o + 0x10), n), WI._u16(b, o + 0x1E), n) or {}
        structs.setdefault(WI._u32(b, o), (WI._u32(b, o + 0x18), members))
    for i in range(WI._u16(b, 0x4C)):
        o = it + 16 * i
        blocks.append((WI._u32(b, o), WI._u32(b, o + 4), WI._ptr(WI._u64(b, o + 8), n)))
    return structs, blocks


def _value(b, base, member):
    if member is not None and member[1] == ENUM:
        p = base + member[0]
        return struct.unpack_from("<I", b, p)[0] if p + 4 <= len(b) else None
    return WI._value(b, base, member)


def parse_archetypes(b):
    """One dict per archetype instance in a validated 0x7a resource (empty when it has none)."""
    if WI.parse_resource(b) is None:
        return []
    structs, blocks = schema(b)
    out = []
    for name, size, off in blocks:
        kind = KINDS.get(name)
        if kind is None or off is None or name not in structs:
            continue
        ssize, members = structs[name]
        if not ssize or size % ssize or off + size > len(b):
            continue
        for k in range(size // ssize):
            base = off + k * ssize
            row = {key: _value(b, base, members.get(h)) for key, h in FIELDS.items()}
            row = {key: v for key, v in row.items() if v is not None}
            row["kind"] = kind
            out.append(row)
    return out


def member_table(b, struct_hash):
    """[(member hash, offset, type)] of one struct, sorted by offset (for the docs table)."""
    structs, _ = schema(b)
    if struct_hash not in structs:
        return []
    return sorted(((h, m[0], m[1]) for h, m in structs[struct_hash][1].items()), key=lambda t: (t[1], t[0]))


def iter_resources(path, start=None, limit=None):
    """Yield (row, data) for every validated 0x7a resource: an RPF9 window, a carved file or a folder."""
    magic = b""
    if os.path.isfile(path):
        with open(path, "rb") as f:
            magic = f.read(4)
    if magic == b"9FPR":
        rows, _ = WI.scan_rpf(path, start, limit)
        with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            for r in rows:
                res = WC.assemble(m, r["offset"], len(m))
                if res and res.get("complete"):
                    yield r, res["data"]
        return
    rows, _ = WI.scan_path(path, start, limit)
    for r in rows:
        p = os.path.join(path, r["file"]) if os.path.isdir(path) else path
        with open(p, "rb") as f:
            yield r, f.read(WI.CAP + 1)


# ---- transforms (source axes: RAGE world, Z up, metres)

def _rotation(q, conjugate):
    x, y, z, w = q or (0.0, 0.0, 0.0, 1.0)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    if conjugate:
        x, y, z = -x, -y, -z
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))


def _scale(entity, key):
    """Missing scales use one; an explicit zero collapses the corresponding axes."""
    value = entity.get(key)
    return 1.0 if value is None else value


def world_box(entity, lo, hi, conjugate, use_scale):
    """World AABB of archetype box [lo, hi] placed by ``entity`` (T * R * S, scale XY on X/Y and Z on Z)."""
    r = _rotation(entity.get("rotation"), conjugate)
    sxy = _scale(entity, "scale_xy") if use_scale else 1.0
    sz = _scale(entity, "scale_z") if use_scale else 1.0
    s = (sxy, sxy, sz)
    c = [(lo[i] + hi[i]) * 0.5 * s[i] for i in range(3)]
    e = [(hi[i] - lo[i]) * 0.5 * abs(s[i]) for i in range(3)]
    p = entity.get("position") or (0.0, 0.0, 0.0)
    out_lo, out_hi = [], []
    for i in range(3):
        wc = p[i] + sum(r[i][j] * c[j] for j in range(3))
        we = sum(abs(r[i][j]) * e[j] for j in range(3))
        out_lo.append(wc - we)
        out_hi.append(wc + we)
    return out_lo, out_hi


def _union_error(entities, defs, ref_lo, ref_hi, conjugate, use_scale):
    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for e in entities:
        d = defs[e["archetype"]]
        a, b = world_box(e, d["bb_min"], d["bb_max"], conjugate, use_scale)
        for i in range(3):
            lo[i], hi[i] = min(lo[i], a[i]), max(hi[i], b[i])
    err = max(max(abs(lo[i] - ref_lo[i]), abs(hi[i] - ref_hi[i])) for i in range(3))
    inside = all(ref_lo[i] - TOLERANCE <= lo[i] and hi[i] <= ref_hi[i] + TOLERANCE for i in range(3))
    return err, inside


def check_extents(maps, defs):
    """Compare each fully-defined map's stored entitiesExtents with the union of its transformed archetype boxes.

    ``maps`` = [(map row, [entity dicts from world_index.parse_resource(collect=...)])]; ``defs`` = {name hash:
    first definition}. A map matches a variant when every bound is within TOLERANCE metres.
    """
    out = {"tolerance_m": TOLERANCE, "maps": len(maps), "maps_fully_defined": 0,
           "rotation": {"maps": 0, "stored_matches": 0, "conjugate_matches": 0,
                        "stored_median_error_m": None, "conjugate_median_error_m": None,
                        "union_inside_extents_conjugate": 0, "neither_matches": 0},
           "scale": {"maps": 0, "scaled_matches": 0, "unscaled_matches": 0,
                     "scaled_exceeds_extents": 0, "unscaled_exceeds_extents": 0}}
    errs = {False: [], True: []}
    for row, ents in maps:
        ref_lo, ref_hi = row.get("entities_min"), row.get("entities_max")
        if not ref_lo or not ref_hi or not ents:
            continue
        if any(e.get("archetype") not in defs or not e.get("complete", True) for e in ents):
            continue
        if any("bb_min" not in defs[e["archetype"]] or "bb_max" not in defs[e["archetype"]] for e in ents):
            continue
        out["maps_fully_defined"] += 1
        rotated = any(e.get("rotation") and [round(c, 6) for c in e["rotation"]] != [0, 0, 0, 1] for e in ents)
        scaled = any(abs(_scale(e, "scale_xy") - 1) > 1e-4 or abs(_scale(e, "scale_z") - 1) > 1e-4
                     for e in ents)
        if rotated:
            rot = out["rotation"]
            rot["maps"] += 1
            res = {conj: _union_error(ents, defs, ref_lo, ref_hi, conj, False) for conj in (False, True)}
            for conj, (err, _) in res.items():
                errs[conj].append(err)
            rot["stored_matches"] += res[False][0] <= TOLERANCE
            rot["conjugate_matches"] += res[True][0] <= TOLERANCE
            rot["union_inside_extents_conjugate"] += res[True][1]
            rot["neither_matches"] += min(res[False][0], res[True][0]) > TOLERANCE
        if scaled:
            sc = out["scale"]
            sc["maps"] += 1
            for use_scale, key in ((True, "scaled"), (False, "unscaled")):
                err, inside = _union_error(ents, defs, ref_lo, ref_hi, True, use_scale)
                sc[key + "_matches"] += err <= TOLERANCE
                sc[key + "_exceeds_extents"] += not inside
    for conj, key in ((False, "stored_median_error_m"), (True, "conjugate_median_error_m")):
        if errs[conj]:
            out["rotation"][key] = round(statistics.median(errs[conj]), 6)
    return out


def run(path, start=None, limit=None, extents=False):
    defs, maps, resources = {}, [], []
    stats = {"resources": 0, "maptypes": 0, "archetype_resources": 0, "archetypes": 0, "kinds": {},
             "asset_types": {}, "names": 0, "names_with_differing_boxes": 0}
    schemas = {}
    for row, data in iter_resources(path, start, limit):
        stats["resources"] += 1
        if row.get("kind") == "map":
            if extents:
                ents = []
                WI.parse_resource(data, collect=ents)
                maps.append((row, ents))
            continue
        structs, _ = schema(data)
        if MAPTYPES in structs:
            stats["maptypes"] += 1
        found = parse_archetypes(data)
        if not found:
            continue
        stats["archetype_resources"] += 1
        for h, kind in KINDS.items():
            if h in structs and kind not in schemas:
                schemas[kind] = {"size": structs[h][0], "members": [
                    {"hash": "%08x" % mh, "offset": off, "type": "0x%02x" % ty}
                    for mh, off, ty in member_table(data, h)]}
        for d in found:
            d["source"] = row.get("file") or row["offset"]
            resources.append(d)
            stats["archetypes"] += 1
            stats["kinds"][d["kind"]] = stats["kinds"].get(d["kind"], 0) + 1
            t = str(d.get("asset_type"))
            stats["asset_types"][t] = stats["asset_types"].get(t, 0) + 1
            prev = defs.get(d.get("name"))
            if prev is None:
                defs[d.get("name")] = d
            elif (prev.get("bb_min"), prev.get("bb_max")) != (d.get("bb_min"), d.get("bb_max")):
                prev["_differs"] = True
    stats["names"] = len(defs)
    stats["names_with_differing_boxes"] = sum(1 for d in defs.values() if d.pop("_differs", False))
    doc = {"summary": stats, "schemas": schemas, "archetypes": resources}
    if extents:
        doc["extents"] = check_extents(maps, defs)
    return doc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", help="RPF9 archive, one carved 0x7a file, or a folder of carved files")
    ap.add_argument("--start", type=int, help="first byte of an RPF9 data-area window")
    ap.add_argument("--limit", type=int, help="window size in bytes (default: to the end of the archive)")
    ap.add_argument("--out", default=os.path.join("reports", "world_archetypes.json"))
    ap.add_argument("--check-extents", action="store_true",
                    help="also compare each map's entitiesExtents with its transformed archetype boxes")
    a = ap.parse_args(argv)
    if not os.path.exists(a.path):
        ap.error("no such file or folder: %s" % a.path)
    # Check the report path before the scan: --out equal to the archive used to truncate the input after the run, and
    # an --out under a regular file or naming a folder raised a traceback once all the work was done.
    if same_report_path(a.out, a.path):
        ap.error("--out would overwrite the input %s" % a.path)
    if os.path.isdir(a.out):
        ap.error("--out %s is a folder, not a file" % a.out)
    out_dir = os.path.dirname(os.path.abspath(a.out))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        ap.error("--out %s: cannot create %s: %s" % (a.out, out_dir, e.strerror or e))
    doc = run(a.path, a.start, a.limit, a.check_extents)
    doc["summary"].update({"path": os.path.abspath(a.path), "start": a.start, "limit": a.limit})
    try:
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1, sort_keys=True)
    except OSError as e:
        ap.error("--out %s: %s" % (a.out, e.strerror or e))
    print(json.dumps({k: doc[k] for k in ("summary", "extents") if k in doc}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
