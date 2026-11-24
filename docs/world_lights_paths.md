# Map lights and the vehicle lane graph (`levels.rpf`)

Both live in the same self-describing `0x7a` resources as the map placements ([world.md](world.md)), and
`tools/world_lights_paths.py` reads them through each resource's own schema: every field below is found by its
member hash in the resource's member table, not by a fixed offset. Counts only; no game data.

```
python -I tools/world_lights_paths.py <levels.rpf> --jobs 24 --out reports/world_lights_paths.json \
    [--entities <world_index.py --entities-out rows>] [--lights-out <local file>]
```

`--lights-out` writes one JSON line per placed light (position, direction, colour, type, ...). It is derived
from your own copy of the game, so it stays in your local work folder.

`--limit` caps where a scanned resource may start (a resource that starts inside it is still read to its end). It
must be at least 1 (decimal or `0x` hex); a smaller value stops with a one-line error and exit code 2 before any
scan or report is written.

## Names

Struct and member hashes are the case-sensitive joaat of the GTA V names where those are recovered:

| Hash | Name |
|---|---|
| `27922c43` | `CExtensionDefLightEffect` (80 bytes) |
| `f54b227b` | `CLightAttrDef` (672 bytes) |
| `82d6fc83`, `10506455`, `ce501483` | `CBaseArchetypeDef`, `CMloArchetypeDef`, `CEntityDef` |
| `a2a107f9` | `CPathNode` (52 bytes) |
| `abc0cc55` | `CJunctionEntrance` (80 bytes) |
| `0c69a100`, `0f2da4c2`, `eba8aebb`, `badd6def`, `aef55528`, `c47cb7b3` | node-list element (60), link (20), junction (88), turn (176), traffic-light phase (24), light box (16): names not recovered |

Joaat-verified member names: `extensions`, `offsetPosition`, `instances`, `posn`, `colour`, `volOuterColour`,
`intensity`, `timeFlags`, `falloff`, `falloffExponent`, `volIntensity`, `coronaSize`, `coronaIntensity`,
`direction`, `tangent`, `coneInnerAngle`, `coneOuterAngle`, `extents`, `lightType`, `groupId`, `flashiness`,
`lightFadeDistance`, `shadowFadeDistance`, `entrances`, `phases`, `bIsStop`, `bIsYield`, `fPathLength`,
`fDuration`, `boxId`, plus the placement members of [world.md](world.md). The tool's other member keys
(`instances2`, `position120`, and every lane-graph field except those above) are descriptive labels for hashes
whose names are not recovered.

## Lights

A `CExtensionDefLightEffect` hangs off the `extensions` list of a `CBaseArchetypeDef`, a `CMloArchetypeDef` or a
`CEntityDef`. It has two `{pointer, u16 count, u16 capacity}` lists of `CLightAttrDef` at `+0x20` (`instances`)
and `+0x30` (`instances2`). Archetype lights are in model space and appear once per placed entity of that
archetype; entity lights belong to one entity.

`CLightAttrDef` members used (offsets in this build): `position120` `+0x120` (float3), `posn` `+0x12c` (float3),
`intensity` `+0x138`, `flags` `+0x140`, `timeFlags` `+0x144` (24 bits, one per hour), `falloff` `+0x148` (range),
`falloffExponent` `+0x14c`, `volIntensity` `+0x154`, `coronaSize` `+0x168`, `coronaIntensity` `+0x174`,
`direction` `+0x1a0`, `tangent` `+0x1ac`, `coneInnerAngle` `+0x1b8`, `coneOuterAngle` `+0x1bc`, `extents`
`+0x1dc`, `volOuterColour` `+0x282` and `colour` `+0x288` (RGB bytes), `flashiness` `+0x28b`,
`lightFadeDistance` `+0x28d`, `shadowFadeDistance` `+0x28e`, `lightType` `+0x291` (1 point, 2 spot, 4 capsule, 32 seen but not named),
`groupId` `+0x292`. Its member table has 100 entries.

Pointers are pool references (1-based block number in bits 0..11, byte offset from bit 20) or `0x5xxxxxxx`
system pointers. A pool reference whose offset is `0x500`..`0x5ff` also has
the shape of a `0x5xxxxxxx` pointer, so the reader tries the pool form first and only then a system pointer.

| Measure | Count |
|---|---:|
| Resources with light effects | 8,576 |
| Light effects (`CExtensionDefLightEffect`) / with no lights | 146,733 / 7,442 |
| Lights (`CLightAttrDef`) | **280,105** |
| on archetypes / on entities | 137,294 / 142,811 |
| in archetype / map / interior (MLO) resources | 88,467 / 100,396 / 91,242 |
| in `instances` / `instances2` | 272,166 / 7,939 |
| spot / capsule / point / type 32 | 152,773 / 66,383 / 55,115 / 5,834 |
| lit every hour / never in daytime | 254,380 / 20,605 |
| unit `direction` | 280,105 |
| archetypes with lights | 3,766 |
| effects reached through a pool reference with a `0x5xx` offset | 10,038 |

## Placing archetype lights

With `--entities`, every map entity whose archetype carries lights places them in world space:

    world = position + R(conjugate(rotation)) * (scale * (offsetPosition + posn))

with `scale = (scaleXY, scaleXY, scaleZ)`, the same transform `world_build.py --placements` uses
([world.md](world.md)). Directions are rotated without the translation. Entity lights are placed through
their own entity.

When an archetype is defined more than once, placement uses its first light list. The summary counts that
archetype once in `archetype_light_definitions_differ` if any other list differs in length, order or decoded
light fields, including the effect offset.

| Measure | Count |
|---|---:|
| Map entities read (`world_index.py --entities-out`) | 1,670,435 |
| entities whose archetype has lights | 190,319 |
| archetype lights placed | 349,799 |
| entity lights in maps | 100,396 |
| **lights in world space** | **450,195** |

Interior (MLO) entities are not placed: their lights stay in interior space until the interior instance is
known.

## Lane graph

One node-list resource holds the 60-byte elements: the embedded `CPathNode`, its address at `+0x34` and the
index of its first link at `+0x38`. A separate resource holds the 20-byte links, and a node's links are
`links[first .. next node's first]`. A link names its target by node address, with the lane counts in each
direction at `+0x06`/`+0x07` and its flags at `+0x0c`. One junction resource holds the junctions: each names its
node, and lists entrances (`CJunctionEntrance`, with `bIsStop`/`bIsYield` and the turn paths, each with
`fPathLength` and start/end points) and traffic-light phases (`fDuration` and light boxes, a 64-bit hash plus
`boxId`). A junction with phases is signalised.

| Measure | Count |
|---|---:|
| Nodes / links | 63,808 / 145,518 |
| links with a reverse link | 145,510 |
| connected components / largest | 27 / 58,239 nodes |
| isolated nodes | 0 |
| junctions / signalised | 1,546 / 369 |
| traffic-light phases | 1,033 (2–120 s, median 10 s) |
| junction entrances / turn paths (all resolve to nodes) | 6,790 / 11,604 |
| lanes per direction (links) | 0: 30,400 · 1: 108,396 · 2: 5,742 · 3: 879 · 4: 94 · 5: 7 |
| link length | median 12 m, max 1,483 m |

## Cross-check with SixNG

[SixNG](https://github.com/holdmysocks/SixNG) decodes the same records independently, and its output for the
same package was compared resource by resource.

- **Lane graph:** identical on every count above (nodes, links, junctions, signalised junctions, phases,
  entrances, turns).
- **Lights:** 280,105 here against SixNG's 274,820. All 5,285 extra lights are on this side, and SixNG has no
  light this tool lacks. SixNG resolves a pointer as a `0x5xxxxxxx` system pointer (or, in two-page resources, a
  `0x6xxxxxxx` graphics pointer) before trying the pool form, so pool references with an offset of
  `0x500`..`0x6ff` that also land inside the resource are misread and their effects are dropped. Rereading every light with SixNG's order gives equal counts on 8,434 of 8,576
  light resources. Of the 5,285, 208 lights are in 9 map resources for which SixNG writes no light set at all.
- **Placed lights:** SixNG places 774,847, including archetype lights inside interiors (MLO instances), which
  this tool leaves in interior space.
