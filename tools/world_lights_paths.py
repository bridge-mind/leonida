"""Measure the map lights and the vehicle lane graph in the 0x7a world resources of an RPF9 archive.

Both live in the same self-describing ``0x7a`` resources as the map placements (``docs/world.md``), so this reuses
``world_census.assemble`` to inflate and join each resource and ``world_index`` to read its header. Every field is
read through the resource's own member table by its name hash. The struct and member hashes used here are the
case-sensitive joaat of the names below, matched against the schemas in the archive.

Lights. A ``CExtensionDefLightEffect`` (``27922c43``) holds two lists of ``CLightAttrDef`` (``f54b227b``, 672 bytes
in this build). It hangs off the ``extensions`` list (``f17e7f28``) of a ``CBaseArchetypeDef`` (``82d6fc83``), a
``CMloArchetypeDef`` (``10506455``) or a ``CEntityDef`` (``ce501483``). Archetype lights are in model space and
appear once per placed entity of that archetype; entity lights belong to one entity. With ``--entities`` (the
``world_index.py --entities-out`` rows), archetype lights are counted per placement and, with ``--lights-out``,
written in world space: ``position + R(conjugate(rotation)) * (scale * (offsetPosition + posn))``.

Lane graph. ``CPathNode`` (``a2a107f9``) records are embedded in a node-list resource (``0c69a100``, 60 bytes: the
node, a node address and the index of the node's first link). The links (``0f2da4c2``, 20 bytes) are in a
separate resource, and a node's links are ``links[first .. next node's first]``. Junctions are ``eba8aebb``
records (name not recovered) with ``entrances`` (``CJunctionEntrance``) and traffic-light ``phases``. A link names
its target by node address; a junction and its entrances name nodes by a second address field of the node.

Member names: the ones listed as joaat-verified in ``docs/world_lights_paths.md`` are the case-sensitive joaat of
the engine names; the others in ``M`` are descriptive labels for hashes whose names are not recovered.

Usage:
    python -I world_lights_paths.py <levels.rpf> [--out FILE] [--jobs N] [--start N] [--limit BYTES]
                                    [--entities FILE] [--lights-out FILE]

The summary has counts, value ranges and hashes only. ``--lights-out`` writes one JSON line per placed light
(position, direction, colour, type, ...): it is derived from your own copy of the game, so keep it in your local
work folder.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import mmap
import os
import statistics
import struct
import sys
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

_TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if _TOOLS_DIR not in sys.path:
    sys.path.insert(0, _TOOLS_DIR)

from pool_workers import process_workers  # noqa: E402
from report_paths import same_report_path  # noqa: E402
import world_census as WC  # noqa: E402
import world_index as WI  # noqa: E402

# Struct hashes (case-sensitive joaat of the engine struct name where recovered).
ARCHETYPE = 0x82D6FC83        # CBaseArchetypeDef
MLO_ARCHETYPE = 0x10506455    # CMloArchetypeDef
ENTITY = WI.ENTITY            # CEntityDef
MAPDATA = WI.MAPDATA          # CMapData
ENTITY_GROUP = 0x35ABC248     # root of the entity-group resources (docs/world_census.md)
LIGHT_EXT = 0x27922C43        # CExtensionDefLightEffect
LIGHT = 0xF54B227B            # CLightAttrDef
NODE_LIST = 0x0C69A100        # node-list element: CPathNode + node address + first link
NODE = 0xA2A107F9             # CPathNode
LINK = 0x0F2DA4C2             # link record
JUNCTION = 0xEBA8AEBB         # junction (name not recovered)
ENTRANCE = 0xABC0CC55         # CJunctionEntrance
TURN = 0xBADD6DEF             # turn path through a junction (fPathLength, start, end)
PHASE = 0xAEF55528            # traffic-light phase (fDuration, light boxes)
BOX = 0xC47CB7B3              # light box: u64 hash, boxId

# Member hashes.
M = {
    "name": 0xE6E17F14, "flags": 0x67D26872, "bbMin": 0xD9EF8236, "bbMax": 0xE78AA618, "extensions": 0xF17E7F28, "position": 0x01166164,
    "rotation": 0x77DD4010, "scaleXY": 0x9CA32637, "scaleZ": 0x10FB7C42, "archetypeName": 0xA023A02C,
    # CExtensionDefLightEffect
    "offsetPosition": 0xD8950703, "instances": 0x10579DF2, "instances2": 0xC4E96A00,
    # CLightAttrDef
    "posn": 0x39A625E4, "position120": 0x30ECA388, "colour": 0x5BDC13E3, "volOuterColour": 0x8822FBCE,
    "intensity": 0xEFCD993D, "timeFlags": 0x8609D52C, "falloff": 0xCC6806E8, "falloffExponent": 0x2BB342BA,
    "volIntensity": 0x291D3720, "coronaSize": 0x65A03C8B, "coronaIntensity": 0x88A2B1FB,
    "direction": 0x34DF9828, "tangent": 0x8E6F0BA9, "coneInnerAngle": 0x455C3938, "coneOuterAngle": 0xF8D9E344,
    "extents": 0x2D3F79C0, "lightType": 0x1CBBBE30, "groupId": 0x951BDD14, "flashiness": 0xE4447B12,
    "lightFadeDistance": 0x4DF55F03, "shadowFadeDistance": 0x73E32C64,
    # node list / CPathNode (labels; names not recovered)
    "node": 0x14DDBC4D, "list_address": 0x77487F26, "first_link": 0x7C2BFB66,
    "node_position": 0x0C928991, "node_flags": 0x4B621302, "address": 0x65F1B6A3, "key": 0x3374D50E,
    "address3": 0x2522B86A, "streetNameHash": 0xD2CFA3BC, "speed": 0x3B407343, "kind": 0xBD3CA0C8,
    "road_class": 0xBDA79497, "num_links": 0x12F96053, "density": 0x3D3377EC,
    "area": 0x9E70E58E, "index": 0x3DBCF2F4,
    # link record
    "target": 0x888306FE, "length": 0x75B2CEA8, "lanes_to": 0x5E323133, "lanes_from": 0xD85153A7,
    "link_flags": 0x4B621302,
    # junctions
    "junction_node": 0x59A31DA2, "entrances": 0x78FF580A, "phases": 0x7BB18D3C, "entrance_node": 0xD56341FE,
    "turns": 0x244A2817, "bIsStop": 0x58DBE783, "bIsYield": 0xFF6FDBA9, "fPathLength": 0x86C88D74,
    "turn_start": 0x07FA0E56, "turn_end": 0x1484276A, "fDuration": 0xDB064419, "lightBoxes": 0xAC83E8CA,
    "boxId": 0xD9DE139B,
}
LIGHT_TYPES = {1: "point", 2: "spot", 4: "capsule"}
LINK_FLAG_BITS = 11
NODE_FLAG_BITS = 51
CHUNK = 1 << 20
BBOX_SLACK = 0.25  # metres: a light on a model's surface may sit just outside its bounding box


class Meta:
    """A parsed 0x7a resource: structs with their member tables, blocks, and pointer resolution."""

    def __init__(self, b):
        if len(b) < 0x50 or b[0x10:0x14] != WI.SIGNATURE or b[0x14] != WI.VERSION:
            raise ValueError("not a 0x7a resource")
        self.b = b
        self.system_shaped = 0
        nst, nbl = WI._u16(b, 0x48), WI._u16(b, 0x4C)
        st, it = WI._ptr(WI._u64(b, 0x28), len(b)), WI._ptr(WI._u64(b, 0x38), len(b))
        if st is None or it is None or st + 32 * nst > len(b) or it + 16 * nbl > len(b):
            raise ValueError("struct or block table out of range")
        self.structs = {}
        for i in range(nst):
            o = st + 32 * i
            name, size, count = WI._u32(b, o), WI._u32(b, o + 0x18), WI._u16(b, o + 0x1E)
            members = WI._members(b, WI._ptr(WI._u64(b, o + 0x10), len(b)), count, len(b))
            if members is None:
                raise ValueError("member table out of range")
            self.structs.setdefault(name, (size, members))
        self.blocks = []
        for i in range(nbl):
            o = it + 16 * i
            self.blocks.append((WI._u32(b, o), WI._u32(b, o + 4), WI._ptr(WI._u64(b, o + 8), len(b))))

    def has(self, name):
        return name in self.structs

    def off(self, struct_name, member):
        m = self.structs.get(struct_name, (0, {}))[1].get(M[member])
        return None if m is None else m[0]

    def ref(self, v):
        """Resolve a pointer: a pool reference (1-based block number in bits 0..11, byte offset from bit 20),
        or a 0x5xxxxxxx system pointer. Returns (block struct hash or None, offset) or None."""
        bi, off = v & 0xFFF, v >> 20
        if (v >> 12) & 0xFF == 0 and 1 <= bi <= len(self.blocks):
            name, size, base = self.blocks[bi - 1]
            if base is not None and off < size:
                # A pool offset of 0x500..0x5ff makes the value look like a 0x5xxxxxxx pointer as well.
                self.system_shaped += v >> 28 == 5 and v >> 32 == 0
                return name, base + off
            return None
        p = WI._ptr(v, len(self.b))
        return None if p is None else (None, p)

    def objects(self, name):
        size = self.structs.get(name, (0, {}))[0]
        out = []
        if not size:
            return out
        for n, bsize, base in self.blocks:
            if n == name and base is not None:
                out.extend(base + k * size for k in range(bsize // size) if base + (k + 1) * size <= len(self.b))
        return out

    def list52(self, o):
        """A {u64 pointer, u16 count, u16 capacity} list: (target offset, count)."""
        if o + 12 > len(self.b):
            return None, 0
        n = WI._u16(self.b, o + 8)
        t = self.ref(WI._u64(self.b, o)) if n else None
        return (t[1], n) if t else (None, 0)

    def list5c(self, o):
        """A {u32 count, u32 capacity, u64 pointer} list: (target offset, count)."""
        if o + 16 > len(self.b):
            return None, 0
        n = WI._u32(self.b, o)
        t = self.ref(WI._u64(self.b, o + 8)) if n else None
        return (t[1], n) if t else (None, 0)

    def f32(self, o, k=1):
        return struct.unpack_from("<%df" % k, self.b, o) if k > 1 else WI._f32(self.b, o)


# ---------------------------------------------------------------------------------------------------- lights

def read_light(meta, o):
    """One CLightAttrDef as a dict (only the members the schema has)."""
    b, out = meta.b, {}
    for key, k in (("posn", 3), ("position120", 3), ("direction", 3), ("tangent", 3), ("extents", 3)):
        at = meta.off(LIGHT, key)
        if at is not None:
            out[key] = list(meta.f32(o + at, 3))
    for key in ("intensity", "falloff", "falloffExponent", "volIntensity", "coronaSize", "coronaIntensity",
                "coneInnerAngle", "coneOuterAngle"):
        at = meta.off(LIGHT, key)
        if at is not None:
            out[key] = meta.f32(o + at)
    for key in ("flags", "timeFlags"):
        at = meta.off(LIGHT, key)
        if at is not None:
            out[key] = WI._u32(b, o + at)
    for key in ("lightType", "groupId", "flashiness", "lightFadeDistance", "shadowFadeDistance"):
        at = meta.off(LIGHT, key)
        if at is not None:
            out[key] = b[o + at]
    for key in ("colour", "volOuterColour"):
        at = meta.off(LIGHT, key)
        if at is not None:
            out[key] = bytes(b[o + at:o + at + 3]).hex()
    return out


def light_effects(meta):
    """Every CExtensionDefLightEffect reachable from an ``extensions`` list, with its owner.

    Returns a list of (owner kind, owner offset, effect offset); owner kind is "archetype", "mlo_archetype",
    "entity" or "other". Effects in the blocks that no extensions list reaches are returned as "unreferenced".
    """
    found, seen = [], set()
    meta.effects_system_shaped = 0
    kinds = {ARCHETYPE: "archetype", MLO_ARCHETYPE: "mlo_archetype", ENTITY: "entity"}
    for sname, (size, members) in meta.structs.items():
        ext = members.get(M["extensions"])
        if ext is None or ext[1] != 0x52:
            continue
        for o in meta.objects(sname):
            p, n = meta.list52(o + ext[0])
            if p is None:
                continue
            for i in range(min(n, 256)):
                if p + 8 * i + 8 > len(meta.b):
                    break
                before = meta.system_shaped
                t = meta.ref(WI._u64(meta.b, p + 8 * i))
                if t and t[0] == LIGHT_EXT and t[1] not in seen:
                    seen.add(t[1])
                    found.append((kinds.get(sname, "other"), sname, o, t[1]))
                    meta.effects_system_shaped += meta.system_shaped > before
    for o in meta.objects(LIGHT_EXT):
        if o not in seen:
            found.append(("unreferenced", None, None, o))
    return found


def effect_lights(meta, eo):
    out = []
    for key in ("instances", "instances2"):
        at = meta.off(LIGHT_EXT, key)
        if at is None:
            continue
        p, n = meta.list52(eo + at)
        size = meta.structs.get(LIGHT, (0, {}))[0]
        if p is None or not size:
            continue
        for k in range(min(n, 10000)):
            o = p + k * size
            if o + size > len(meta.b):
                break
            light = read_light(meta, o)
            light["list"] = key
            out.append(light)
    return out


def _local_light_position(light: dict, offset: list[float]) -> list[float]:
    position = light.get("position120", light.get("posn", (0.0, 0.0, 0.0)))
    return [position[i] + offset[i] for i in range(3)]


def quat_rotate(q, v):
    """Rotate v by the unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    tx, ty, tz = 2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])
    return [v[0] + w * tx + (y * tz - z * ty), v[1] + w * ty + (z * tx - x * tz), v[2] + w * tz + (x * ty - y * tx)]


def place(
    local: Sequence[float],
    direction: Sequence[float] | None,
    position: Sequence[float],
    rotation: Sequence[float],
    scale_xy: float | None,
    scale_z: float | None,
) -> tuple[list[float], list[float] | None]:
    """World position and direction of a model-space light under an entity's stored transform.

    Normalize and conjugate the stored quaternion to match the rotation used for placed geometry.
    """
    x, y, z, w = rotation
    length = math.hypot(x, y, z, w)
    if length and math.isfinite(length):
        q = (-x / length, -y / length, -z / length, w / length)
    else:
        q = (0.0, 0.0, 0.0, 1.0)
    xy = 1.0 if scale_xy is None else scale_xy
    z_scale = 1.0 if scale_z is None else scale_z
    scaled = [local[0] * xy, local[1] * xy, local[2] * z_scale]
    p = quat_rotate(q, scaled)
    return [position[0] + p[0], position[1] + p[1], position[2] + p[2]], (quat_rotate(q, direction) if direction else None)


def _entity_transform(meta, o):
    em = meta.structs[ENTITY][1]
    vals = {}
    for key in ("position", "rotation", "scaleXY", "scaleZ", "archetypeName"):
        vals[key] = WI._value(meta.b, o, em.get(M[key]))
    return vals


# ------------------------------------------------------------------------------------------------- lane graph

def _addr(meta, o):
    return (WI._u16(meta.b, o), WI._u16(meta.b, o + 2))


def read_nodes(meta):
    """CPathNode records of a node-list resource, in order."""
    nodes = []
    size = meta.structs[NODE_LIST][0]
    node_at = meta.off(NODE_LIST, "node") or 0
    first_at = meta.off(NODE_LIST, "first_link")
    for o in meta.objects(NODE_LIST):
        n = o + node_at
        rec = {
            "position": list(meta.f32(n + meta.off(NODE, "node_position"), 3)),
            "flags": WI._u64(meta.b, n + meta.off(NODE, "node_flags")),
            "address": _addr(meta, n + meta.off(NODE, "address")),
            "key": _addr(meta, n + meta.off(NODE, "key")),
            "street": WI._u32(meta.b, n + meta.off(NODE, "streetNameHash")),
            "speed": struct.unpack_from("<e", meta.b, n + meta.off(NODE, "speed"))[0],
            "kind": meta.b[n + meta.off(NODE, "kind")],
            "road_class": meta.b[n + meta.off(NODE, "road_class")],
            "num_links": meta.b[n + meta.off(NODE, "num_links")],
            "density": meta.b[n + meta.off(NODE, "density")],
            "first_link": WI._u32(meta.b, o + first_at) if first_at is not None else None,
        }
        nodes.append(rec)
    del size
    return nodes


def read_links(meta):
    links = []
    for o in meta.objects(LINK):
        links.append({
            "target": _addr(meta, o + meta.off(LINK, "target")),
            "length": WI._u16(meta.b, o + meta.off(LINK, "length")),
            "lanes_to": meta.b[o + meta.off(LINK, "lanes_to")],
            "lanes_from": meta.b[o + meta.off(LINK, "lanes_from")],
            "flags": WI._u32(meta.b, o + meta.off(LINK, "link_flags")),
        })
    return links


def read_junctions(meta):
    out = []
    ent_size = meta.structs.get(ENTRANCE, (0, {}))[0]
    turn_size = meta.structs.get(TURN, (0, {}))[0]
    phase_size = meta.structs.get(PHASE, (0, {}))[0]
    box_size = meta.structs.get(BOX, (0, {}))[0]
    for o in meta.objects(JUNCTION):
        j = {"node": _addr(meta, o + meta.off(JUNCTION, "junction_node")), "entrances": [], "phases": []}
        p, n = meta.list5c(o + meta.off(JUNCTION, "entrances"))
        for k in range(n if p is not None else 0):
            e = p + k * ent_size
            tp, tn = meta.list5c(e + meta.off(ENTRANCE, "turns"))
            turns = []
            for t in range(tn if tp is not None else 0):
                q = tp + t * turn_size
                turns.append({"to": _addr(meta, q + meta.off(TURN, "entrance_node")),
                              "length": meta.f32(q + meta.off(TURN, "fPathLength"))})
            j["entrances"].append({"node": _addr(meta, e + meta.off(ENTRANCE, "entrance_node")),
                                   "stop": bool(meta.b[e + meta.off(ENTRANCE, "bIsStop")]),
                                   "yield": bool(meta.b[e + meta.off(ENTRANCE, "bIsYield")]), "turns": turns})
        p, n = meta.list5c(o + meta.off(JUNCTION, "phases"))
        for k in range(n if p is not None else 0):
            ph = p + k * phase_size
            bp, bn = meta.list5c(ph + meta.off(PHASE, "lightBoxes"))
            j["phases"].append({"duration": meta.f32(ph + meta.off(PHASE, "fDuration")),
                                "boxes": bn if bp is not None else 0})
        out.append(j)
    del box_size
    return out


def graph_stats(nodes, links, junctions):
    """Counts for the lane graph: link ownership, resolution, reciprocity, components, lanes, speeds, flags."""
    s = collections.Counter()
    s["nodes"], s["links"], s["junctions"] = len(nodes), len(links), len(junctions)
    by_addr = {}
    for i, n in enumerate(nodes):
        by_addr.setdefault(n["address"], i)
    s["node_addresses_unique"] = len(by_addr)
    by_key = {n["key"]: i for i, n in enumerate(nodes)}
    s["node_keys_unique"] = len(by_key)
    edges = []
    firsts = [n["first_link"] for n in nodes]
    ranges_ok = (all(v is not None for v in firsts)
                 and all(firsts[i] <= firsts[i + 1] for i in range(len(firsts) - 1)))
    s["first_link_monotonic"] = int(ranges_ok)
    for i, n in enumerate(nodes):
        a = n["first_link"]
        e = nodes[i + 1]["first_link"] if i + 1 < len(nodes) else len(links)
        if a is None or e is None or e < a:
            continue
        s["num_links_matches_range"] += (e - a) == n["num_links"]
        for li in range(a, min(e, len(links))):
            to = by_addr.get(links[li]["target"])
            if to is None:
                s["links_unresolved"] += 1
            else:
                edges.append((i, to, li))
    s["links_owned"] = len(edges) + s["links_unresolved"]
    pairs = {(a, b) for a, b, _ in edges}
    s["links_with_reverse"] = sum(1 for a, b, _ in edges if (b, a) in pairs)
    s["self_links"] = sum(1 for a, b, _ in edges if a == b)
    parent = list(range(len(nodes)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b, _ in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    comp = collections.Counter(find(i) for i in range(len(nodes)))
    s["components"] = len(comp)
    s["largest_component_nodes"] = max(comp.values()) if comp else 0
    s["isolated_nodes"] = sum(1 for i in range(len(nodes)) if nodes[i]["num_links"] == 0)
    hist = {
        "lanes_to": collections.Counter(l["lanes_to"] for l in links),
        "lanes_from": collections.Counter(l["lanes_from"] for l in links),
        "speed": collections.Counter(round(n["speed"], 2) for n in nodes),
        "road_class": collections.Counter(n["road_class"] for n in nodes),
        "kind": collections.Counter(n["kind"] for n in nodes),
        "density": collections.Counter(n["density"] for n in nodes),
        "areas": len({n["address"][0] for n in nodes}),
        "node_flag_bits": {b: sum(1 for n in nodes if n["flags"] >> b & 1) for b in range(NODE_FLAG_BITS)},
        "link_flag_bits": {b: sum(1 for l in links if l["flags"] >> b & 1) for b in range(LINK_FLAG_BITS)},
        "node_flag_bits_above_51": sum(1 for n in nodes if n["flags"] >> NODE_FLAG_BITS),
        "link_flag_bits_above_11": sum(1 for l in links if l["flags"] >> LINK_FLAG_BITS),
    }
    if nodes:
        hist["bbox_min"] = [round(min(n["position"][k] for n in nodes), 1) for k in range(3)]
        hist["bbox_max"] = [round(max(n["position"][k] for n in nodes), 1) for k in range(3)]
    lengths = [l["length"] for l in links]
    if lengths:
        hist["link_length_m"] = {"min": min(lengths), "median": statistics.median(lengths), "max": max(lengths)}
    s["street_named_nodes"] = sum(1 for n in nodes if n["street"])
    s["junction_nodes_resolved"] = sum(1 for j in junctions if j["node"] in by_key)
    s["junction_entrances"] = sum(len(j["entrances"]) for j in junctions)
    s["junction_entrances_resolved"] = sum(1 for j in junctions for e in j["entrances"] if e["node"] in by_key)
    s["junction_turns"] = sum(len(e["turns"]) for j in junctions for e in j["entrances"])
    s["junction_turns_resolved"] = sum(1 for j in junctions for e in j["entrances"] for t in e["turns"]
                                       if t["to"] in by_key)
    s["entrances_stop"] = sum(e["stop"] for j in junctions for e in j["entrances"])
    s["entrances_yield"] = sum(e["yield"] for j in junctions for e in j["entrances"])
    s["signalised_junctions"] = sum(1 for j in junctions if j["phases"])
    s["phases"] = sum(len(j["phases"]) for j in junctions)
    durations = [p["duration"] for j in junctions for p in j["phases"]]
    if durations:
        hist["phase_duration_s"] = {"min": round(min(durations), 2), "median": round(statistics.median(durations), 2),
                                    "max": round(max(durations), 2)}
    s["junction_nodes_flagged_junction"] = sum(1 for j in junctions if j["node"] in by_key
                                               and nodes[by_key[j["node"]]]["flags"] >> 4 & 1)
    s["nodes_flagged_junction"] = sum(1 for n in nodes if n["flags"] >> 4 & 1)
    return dict(s), hist


# ------------------------------------------------------------------------------------------------- scanning

def _ranges(stats, key, values):
    values = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if values:
        r = stats.setdefault("ranges", {}).setdefault(key, [math.inf, -math.inf])
        r[0], r[1] = min(r[0], min(values)), max(r[1], max(values))


def inspect_resource(data, lights_out=False):
    """Light and lane-graph content of one assembled 0x7a resource.

    Returns a dict with ``lights`` stats, ``archetype_lights`` {archetype hash: [local lights]} (always, so the
    caller can place them), ``world_lights`` (entity-owned lights in map resources, placed, when lights_out),
    and the graph parts (``nodes``/``links``/``junctions``) when present.
    """
    meta = Meta(data)
    out = {}
    if meta.has(LIGHT_EXT):
        kind = ("map" if meta.has(MAPDATA) else "mlo" if meta.has(MLO_ARCHETYPE)
                else "group" if meta.has(ENTITY_GROUP) else "archetypes")
        st = collections.Counter()
        arch, world = {}, []
        effects = light_effects(meta)
        st["effects_reached_by_0x5_shaped_pool_reference"] += meta.effects_system_shaped
        for owner, sname, oo, eo in effects:
            lights = effect_lights(meta, eo)
            st["effects"] += 1
            st["effects_" + owner] += 1
            st["lights"] += len(lights)
            st["lights_" + owner] += len(lights)
            st["lights_in_%s_resources" % kind] += len(lights)
            st["lights_%s_in_%s_resources" % (owner, kind)] += len(lights)
            off = list(meta.f32(eo + meta.off(LIGHT_EXT, "offsetPosition"), 3)) if meta.off(LIGHT_EXT, "offsetPosition") is not None else [0.0, 0.0, 0.0]
            st["effects_nonzero_offset"] += any(abs(v) > 1e-6 for v in off)
            for L in lights:
                st["type_%s" % LIGHT_TYPES.get(L.get("lightType"), L.get("lightType"))] += 1
                st["list_" + L["list"]] += 1
                if "posn" in L and "position120" in L:
                    st["posn_equals_position120"] += L["posn"] == L["position120"]
                if "direction" in L:
                    d = math.sqrt(sum(v * v for v in L["direction"]))
                    st["direction_unit"] += abs(d - 1) < 1e-3
                tf = L.get("timeFlags")
                if tf is not None:
                    h = tf & 0xFFFFFF
                    st["time_always"] += h == 0xFFFFFF
                    st["time_never_daytime"] += h != 0 and h & 0x0FFF80 == 0
                    st["time_zero"] += h == 0
                    st["time_flag_high_bits"] += tf >> 24 != 0
                if L.get("lightType") == 2:
                    st["spot_inner_le_outer"] += L.get("coneInnerAngle", 0) <= L.get("coneOuterAngle", 0)
                    _ranges(st, "spot_coneOuterAngle", [L.get("coneOuterAngle")])
                    _ranges(st, "spot_coneInnerAngle", [L.get("coneInnerAngle")])
                for key in ("intensity", "falloff", "falloffExponent", "coronaSize", "volIntensity"):
                    _ranges(st, key, [L.get(key)])
            st["effects_empty"] += not lights
            if len(lights) >= 2:
                st["effects_with_2_or_more_lights"] += 1
                for key in ("position120", "posn"):
                    pts = [tuple(L[key]) for L in lights if key in L]
                    st["effects_%s_all_distinct" % key] += len(set(pts)) == len(pts)
            if owner in ("archetype", "mlo_archetype"):
                lo_at, hi_at = meta.off(sname, "bbMin"), meta.off(sname, "bbMax")
                if lo_at is not None and hi_at is not None:
                    lo, hi = meta.f32(oo + lo_at, 3), meta.f32(oo + hi_at, 3)
                    for L in lights:
                        for key in ("position120", "posn"):
                            if key in L:
                                v = [off[k] + L[key][k] for k in range(3)]
                                st["%s_inside_bbox" % key] += all(lo[k] - BBOX_SLACK <= v[k] <= hi[k] + BBOX_SLACK
                                                                  for k in range(3))
                    st["archetype_lights_bbox_checked"] += len(lights)
                name = WI._u32(meta.b, oo + meta.off(sname, "name")) if meta.off(sname, "name") is not None else None
                if name is not None:
                    arch.setdefault(name, []).extend(dict(L, offset=off) for L in lights)
            elif owner == "entity" and kind == "map":
                t = _entity_transform(meta, oo)
                if lights_out and t["position"] and t["rotation"]:
                    for L in lights:
                        local = _local_light_position(L, off)
                        p, d = place(local, L.get("direction"), t["position"], t["rotation"],
                                     t["scaleXY"], t["scaleZ"])
                        world.append(_light_row(p, d, L, "entity", t["archetypeName"]))
                st["lights_entity_in_map_placed"] += len(lights)
        out["lights"] = st
        out["archetype_lights"] = arch
        out["world_lights"] = world
    if meta.has(NODE_LIST) and meta.objects(NODE_LIST):
        out["nodes"] = read_nodes(meta)
    if meta.has(LINK) and meta.objects(LINK):
        out["links"] = read_links(meta)
    if meta.has(JUNCTION) and meta.objects(JUNCTION):
        out["junctions"] = read_junctions(meta)
    return out


def _light_row(p, d, L, owner, archetype):
    archetype = "%08x" % archetype if isinstance(archetype, int) else archetype
    return {"position": [round(v, 3) for v in p], "direction": [round(v, 4) for v in d] if d else None,
            "type": L.get("lightType"), "colour": L.get("colour"), "intensity": L.get("intensity"),
            "range": L.get("falloff"), "cone": [L.get("coneInnerAngle"), L.get("coneOuterAngle")],
            "time_flags": L.get("timeFlags"), "owner": owner, "archetype": archetype}


def scan_window(job):
    """Walk [start, end) of the archive and inspect every complete 0x7a resource that starts there."""
    path, start, end, data_start, lights_out = job
    res = {"resources": 0, "incomplete": 0, "lights": collections.Counter(), "ranges": {},
           "archetype_lights": {}, "world_lights": [], "nodes": [], "links": [], "junctions": [],
           "graph_resources": collections.Counter(), "light_resources": 0}
    with open(path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
        size = len(m)
        # Start a little early, as world_census does: a stream that begins before `start` belongs to the previous
        # window and is only walked over, so this window's first stream is found on a real boundary.
        pos = max(data_start, start - WC.WARMUP)
        while pos < end:
            found = WC.ZHDR.search(m, pos, min(size, end + 16))
            if not found or found.start() >= end:
                break
            s = found.start()
            if s % 16:
                pos = s + 1
                continue
            try:
                r = WC.assemble(m, s, size)
            except (zlib.error, struct.error, IndexError):
                pos = s + 1
                continue
            if r is None:
                pos = s + 1
                continue
            pos = max(r["end"], s + 1)
            if s < start or not r["is_meta"]:
                continue
            if not r.get("complete"):
                res["incomplete"] += 1
                continue
            res["resources"] += 1
            try:
                got = inspect_resource(r["data"], lights_out)
            except (ValueError, struct.error, IndexError, TypeError):
                res["incomplete"] += 1
                continue
            if "lights" in got:
                res["light_resources"] += 1
                st = got["lights"]
                for key, (lo, hi) in st.pop("ranges", {}).items():
                    r0 = res["ranges"].setdefault(key, [math.inf, -math.inf])
                    r0[0], r0[1] = min(r0[0], lo), max(r0[1], hi)
                res["lights"].update(st)
                for h, ls in got["archetype_lights"].items():
                    res["archetype_lights"].setdefault(h, []).append(ls)
                res["world_lights"].extend(got["world_lights"])
            for part in ("nodes", "links", "junctions"):
                if part in got:
                    res["graph_resources"][part] += 1
                    res[part].append(got[part])
    return res


def scan(path, jobs=1, start=None, limit=None, lights_out=False):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
    if head[:4] == b"9FPR":
        if len(head) < 8:
            # unpack_from on the data-start word would raise struct.error; a truncated header is a usage error.
            raise ValueError(f"the RPF9 header is {len(head)} bytes, the format needs 8")
        data_start = struct.unpack_from("<I", head, 4)[0]
    else:
        data_start = 0
    lo = max(data_start, start or 0)
    hi = size if limit is None else min(size, lo + limit)
    # Each window walks its own byte range; a resource that starts inside it is read to its end even past the
    # window, and the next window walks over it in its warm-up.
    step = 64 << 20
    cuts = list(range(lo, hi, step)) + [hi]
    windows = [(path, cuts[i], cuts[i + 1], data_start, lights_out) for i in range(len(cuts) - 1)]
    total = {"resources": 0, "incomplete": 0, "lights": collections.Counter(), "ranges": {},
             "archetype_lights": {}, "world_lights": [], "nodes": [], "links": [], "junctions": [],
             "graph_resources": collections.Counter(), "light_resources": 0}

    def merge(r):
        for k in ("resources", "incomplete", "light_resources"):
            total[k] += r[k]
        total["lights"].update(r["lights"])
        total["graph_resources"].update(r["graph_resources"])
        for key, (a, b) in r["ranges"].items():
            t = total["ranges"].setdefault(key, [math.inf, -math.inf])
            t[0], t[1] = min(t[0], a), max(t[1], b)
        for h, sets in r["archetype_lights"].items():
            total["archetype_lights"].setdefault(h, []).extend(sets)
        total["world_lights"].extend(r["world_lights"])
        for part in ("nodes", "links", "junctions"):
            total[part].extend(r[part])
    if jobs <= 1:
        for w in windows:
            merge(scan_window(w))
    else:
        with ProcessPoolExecutor(process_workers(jobs)) as ex:
            for r in ex.map(scan_window, windows):
                merge(r)
    return total


def read_entities(path):
    """Rows of a world_index.py ``--entities-out`` file, shape-checked: archetype hash, 3-float position,
    4-float rotation, optional float scale (``None`` keeps a field unset, as the producer writes for
    incomplete rows).

    Raises ValueError naming file and line for any row the tool cannot use, so malformed input is refused
    before the scan instead of crashing the summary after it. UnicodeDecodeError (binary garbage) is
    re-raised as ValueError with the path.
    """
    rows = []
    try:
        with open(path, encoding="utf-8") as f:
            for n, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except ValueError as exc:
                    raise ValueError("%s:%d: not valid JSON (%s)" % (path, n, exc)) from None
                if not isinstance(e, dict):
                    raise ValueError("%s:%d: not a JSON object" % (path, n))
                a = e.get("archetype")
                try:
                    a = int(a, 16) if isinstance(a, str) else a    # world_index writes the hash as hex
                    hash(a)
                except ValueError:
                    raise ValueError("%s:%d: archetype is not a hex hash: %r"
                                     % (path, n, e.get("archetype"))) from None
                except TypeError:
                    raise ValueError("%s:%d: archetype is not a hash: %r"
                                     % (path, n, e.get("archetype"))) from None
                row = {"archetype": a}
                for key, width in (("position", 3), ("rotation", 4)):
                    v = e.get(key)
                    if v is None:
                        row[key] = None
                    elif isinstance(v, list) and len(v) == width and all(
                            isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                        row[key] = list(v)
                    else:
                        raise ValueError("%s:%d: %s is not %d numbers: %r" % (path, n, key, width, v))
                for key in ("scale_xy", "scale_z"):
                    v = e.get(key)
                    if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float))):
                        raise ValueError("%s:%d: %s is not a number: %r" % (path, n, key, v))
                    row[key] = v
                rows.append(row)
    except UnicodeDecodeError:
        raise ValueError("%s: not UTF-8 text" % path) from None
    return rows


def summarize(total, entities=None, lights_out=None):
    """Build the counts-only summary, and place archetype lights through the entity rows if given."""
    lights = dict(total["lights"])
    arch = total["archetype_lights"]
    summary = {
        "resources_0x7a": total["resources"], "resources_incomplete": total["incomplete"],
        "light_resources": total["light_resources"], "lights": lights,
        "light_value_ranges": {k: [round(v[0], 4), round(v[1], 4)] for k, v in total["ranges"].items()},
        "archetypes_with_lights": len(arch),
        "archetypes_defined_more_than_once": sum(1 for sets in arch.values() if len(sets) > 1),
        "archetype_light_definitions_differ": sum(1 for sets in arch.values() if len(sets) > 1
                                                  and any(sets[i] != sets[0] for i in range(1, len(sets)))),
    }
    placed = collections.Counter()
    out = None
    staged = None
    try:
        if lights_out:
            # Stage the report: a bad --entities row below must not truncate a report from an earlier run.
            staged = lights_out + ".partial"
            out = open(staged, "w", encoding="utf-8", newline="\n")
            for row in total["world_lights"]:
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
        placed["entity_lights_in_maps"] = len(total["world_lights"]) if lights_out else lights.get("lights_entity_in_map_placed", 0)
        if entities:
            for row in read_entities(entities):
                a = row["archetype"]
                sets = arch.get(a)
                placed["entities"] += 1
                if not sets:
                    continue
                ls = sets[0]
                placed["entities_with_lights"] += 1
                placed["archetype_lights_placed"] += len(ls)
                if out is not None and row["position"] and row["rotation"]:
                    for L in ls:
                        local = _local_light_position(L, L["offset"])
                        p, d = place(local, L.get("direction"), row["position"], row["rotation"],
                                     row["scale_xy"], row["scale_z"])
                        out.write(json.dumps(_light_row(p, d, L, "archetype", a),
                                             separators=(",", ":")) + "\n")
    except BaseException:
        if out is not None:
            try:
                out.close()
            except OSError:
                pass
            out = None
        if staged is not None and os.path.exists(staged):
            os.remove(staged)
        raise
    finally:
        if out is not None:
            out.close()
    if staged is not None:
        try:
            os.replace(staged, lights_out)
        except BaseException:
            # A destination that is a directory (or read-only) must not leave the staged copy behind.
            if os.path.exists(staged):
                os.remove(staged)
            raise
    summary["placement"] = dict(placed)
    nodes = [n for part in total["nodes"] for n in part]
    links = [l for part in total["links"] for l in part]
    junctions = [j for part in total["junctions"] for j in part]
    summary["graph_resources"] = dict(total["graph_resources"])
    if len(total["nodes"]) == 1 and len(total["links"]) == 1:
        s, hist = graph_stats(nodes, links, junctions)
        summary["lane_graph"] = s
        summary["lane_graph_histograms"] = {
            k: ({str(a): b for a, b in sorted(v.items())} if isinstance(v, dict) else v) for k, v in hist.items()}
    else:
        summary["lane_graph"] = {"nodes": len(nodes), "links": len(links), "junctions": len(junctions),
                                 "note": "graph statistics need exactly one node list and one link list"}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("archive")
    ap.add_argument("--out", help="summary JSON (default: stdout)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--start", type=lambda t: int(t, 0))
    ap.add_argument("--limit", type=lambda t: int(t, 0))
    ap.add_argument("--entities", help="world_index.py --entities-out rows, to place archetype lights")
    ap.add_argument("--lights-out", help="write placed lights as JSON lines (local work folder only)")
    a = ap.parse_args(argv)
    if a.jobs < 1:
        print("ERROR: --jobs must be at least 1", file=sys.stderr)
        return 2
    if a.start is not None and a.start < 0:
        print("ERROR: --start must be non-negative", file=sys.stderr)
        return 2
    if a.limit is not None and a.limit < 1:
        print("ERROR: --limit must be at least 1", file=sys.stderr)
        return 2
    if not os.path.isfile(a.archive):
        print(f"ERROR: {a.archive}: no such file", file=sys.stderr)
        return 2
    if a.entities and not os.path.isfile(a.entities):
        print(f"ERROR: {a.entities}: no such file", file=sys.stderr)
        return 2
    if a.entities:
        try:
            read_entities(a.entities)    # refuse malformed rows before the scan, not after it
        except (ValueError, OSError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    if a.out and a.lights_out and same_report_path(a.out, a.lights_out):
        ap.error("--out and --lights-out must use different paths")
    inputs = [("archive", a.archive)]
    if a.entities:
        inputs.append(("--entities", a.entities))
    for rname, rpath in (("--out", a.out), ("--lights-out", a.lights_out)):
        if not rpath:
            continue
        for iname, ipath in inputs:
            if same_report_path(rpath, ipath):
                ap.error("%s must use a different path than %s" % (rname, iname))
    try:
        total = scan(a.archive, a.jobs, a.start, a.limit, bool(a.lights_out))
    except ValueError as e:  # a truncated RPF9 header: one error line, not a struct.error traceback
        print(f"ERROR: {a.archive}: {e}", file=sys.stderr)
        return 2
    try:
        summary = summarize(total, a.entities, a.lights_out)
    except ValueError as e:            # a bad --entities row, or an archetype that is not a number
        print(f"ERROR: {a.entities}: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    text = json.dumps(summary, indent=1, sort_keys=True)
    if a.out:
        try:
            with open(a.out, "w", encoding="utf-8", newline="\n") as f:
                f.write(text + "\n")
        except OSError as e:
            print(f"ERROR: cannot write {a.out}: {e.strerror or e}", file=sys.stderr)
            return 2
    else:
        sys.stdout.write(text + "\n")
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
